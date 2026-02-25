# src/train/train_cae.py
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def _get(d: Dict[str, Any], *keys: str, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def set_seed(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NpzMixDataset(Dataset):
    """
    NPZ keys:
      X: (N,224,224,3) uint8
      y_dirty: (N,) int {0,1}  (0=clean, 1=dirty)
    Returns:
      x_u8: (224,224,3) uint8
      y_dirty: int64
      idx: int64
    """

    def __init__(self, npz_path: Path, only_clean: bool = False, limit: Optional[int] = None):
        d = np.load(npz_path, allow_pickle=True)
        X = d["X"]
        y = d["y_dirty"].astype(np.int64)

        assert X.ndim == 4 and X.shape[-1] == 3, f"Bad X shape: {X.shape}"
        assert len(X) == len(y), "X and y_dirty length mismatch"

        if only_clean:
            keep = np.where(y == 0)[0]
            X = X[keep]
            y = y[keep]

        if limit is not None:
            limit = int(limit)
            X = X[:limit]
            y = y[:limit]

        self.X = X
        self.y_dirty = y

    def __len__(self) -> int:
        return len(self.y_dirty)

    def __getitem__(self, i: int) -> Tuple[np.ndarray, np.int64, np.int64]:
        return self.X[i], np.int64(self.y_dirty[i]), np.int64(i)


def _to_tensor_u8_bhwc(x_u8: np.ndarray, device: torch.device, img_size: int) -> torch.Tensor:
    """
    (B,224,224,3) uint8 -> (B,3,img_size,img_size) float in [0,1]
    """
    x = torch.from_numpy(x_u8).permute(0, 3, 1, 2).float() / 255.0
    if img_size != x.shape[-1]:
        x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return x.to(device, non_blocking=True)


@torch.no_grad()
def _encode_latents(model, ds: NpzMixDataset, ids: np.ndarray, device: torch.device, img_size: int, batch: int = 256):
    """
    Return latents H (len(ids), D) on device, detached.
    """
    model.eval()
    H = []
    for s in range(0, len(ids), batch):
        chunk = ids[s:s + batch]
        x = _to_tensor_u8_bhwc(ds.X[chunk], device=device, img_size=img_size)
        _, z = model(x)
        H.append(z.detach())
    return torch.cat(H, dim=0)  # (M, D)


def _latent_nn_ref_loss(z: torch.Tensor, Href: torch.Tensor) -> torch.Tensor:
    """
    Paper-style:
      h_near = argmin_{h_r in Href} ||h - h_r||^2
      Lref = mean_i ||h_i - h_near_i||^2   (sum over dim, then mean over batch)
    z:    (B,D)
    Href: (M,D)
    """
    # dist^2 = |z|^2 + |r|^2 - 2 z r^T
    z2 = (z ** 2).sum(dim=1, keepdim=True)              # (B,1)
    r2 = (Href ** 2).sum(dim=1, keepdim=True).T         # (1,M)
    dist2 = z2 + r2 - 2.0 * (z @ Href.T)                # (B,M)
    nn = dist2.argmin(dim=1)                            # (B,)
    z_near = Href[nn]                                   # (B,D)
    return ((z - z_near) ** 2).sum(dim=1).mean()


def _rec_loss_l2sum_mean(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Match paper epsilon = ||x - x'||^2 (sum), then averaged over batch.
    """
    diff = (x_hat - x).reshape(x.shape[0], -1)
    return (diff ** 2).sum(dim=1).mean()


def train_cae(cfg: Dict[str, Any]) -> str:
    """
    Called by scripts/run_train_cae.py
    Keys expected (your yaml):
      data.npz_path, data.img_size, data.only_clean, data.limit
      train.device, train.seed, train.epochs, train.batch_size, train.lr
      train.lambda_refer, train.refer_m, train.num_workers, train.save_every
      train.ref_clean_only (optional, debug only; default False)
      output.out_dir
    """

    # ---- make repo importable (so `from src...` works no matter cwd) ----
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    exp_name = _get(cfg, "exp_name", default="run")

    npz_path = _get(cfg, "data", "npz_path", default=None)
    img_size = int(_get(cfg, "data", "img_size", default=64))
    only_clean = bool(_get(cfg, "data", "only_clean", default=False))
    limit = _get(cfg, "data", "limit", default=None)

    device_str = str(_get(cfg, "train", "device", default="auto")).lower()
    seed = int(_get(cfg, "train", "seed", default=0))
    epochs = int(_get(cfg, "train", "epochs", default=100))
    batch_size = int(_get(cfg, "train", "batch_size", default=256))
    lr = float(_get(cfg, "train", "lr", default=5e-4))
    lambda_refer = float(_get(cfg, "train", "lambda_refer", default=1.0))
    refer_m = int(_get(cfg, "train", "refer_m", default=2048))
    num_workers = int(_get(cfg, "train", "num_workers", default=0))
    save_every = int(_get(cfg, "train", "save_every", default=10))

    # Debug option ONLY (oracle-style)
    ref_clean_only = bool(_get(cfg, "train", "ref_clean_only", default=False))

    out_dir = _get(cfg, "output", "out_dir", default=f"outputs/cae/{exp_name}")

    set_seed(seed)

    if npz_path is None:
        raise ValueError("cfg.data.npz_path is required")
    npz_path = (repo_root / npz_path).resolve()

    ds = NpzMixDataset(npz_path, only_clean=only_clean, limit=limit)
    y_dirty = ds.y_dirty
    N = len(ds)

    # Windows: avoid multiprocessing dataloader issues
    if os.name == "nt" and num_workers != 0:
        print("[WARN] Windows: forcing num_workers=0.")
        num_workers = 0

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # device
    if device_str == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_str == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    from src.models.cae import CAE
    model = CAE(latent_dim=256).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # ---- sample Drefer (fixed for all epochs, paper-style) ----
    rng = np.random.default_rng(seed)
    if ref_clean_only:
        pool = np.where(y_dirty == 0)[0]
        ref_tag = "CLEAN_ONLY(debug)"
    else:
        pool = np.arange(N)
        ref_tag = "MIXED(paper)"

    if len(pool) < refer_m:
        raise ValueError(f"ref pool size={len(pool)} < refer_m={refer_m}")

    ref_indices = rng.choice(pool, size=refer_m, replace=False).astype(np.int64)

    print(
        f"[REF] REF={ref_tag} | M={refer_m} dirty_ratio={float(y_dirty[ref_indices].mean()):.4f} "
        f"(dirty={int(y_dirty[ref_indices].sum())}/{len(ref_indices)})"
    )

    # ---- outputs/log ----
    out_dir = (repo_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text("epoch,loss,rec,ref,lambda_refer,lambda_refer_times_ref\n", encoding="utf-8")

    # ---- training ----
    t0 = time.time()
    for ep in range(1, epochs + 1):
        # build Href ONCE per epoch using current encoder (paper wording)
        Href = _encode_latents(model, ds, ref_indices, device=device, img_size=img_size, batch=256)

        model.train()
        pbar = tqdm(dl, desc=f"epoch {ep}/{epochs}", ncols=110)

        loss_sum = rec_sum = ref_sum = 0.0
        steps = 0

        for x_u8, _, _ in pbar:
            x_u8_np = x_u8.numpy() if isinstance(x_u8, torch.Tensor) else np.asarray(x_u8)
            x = _to_tensor_u8_bhwc(x_u8_np, device=device, img_size=img_size)

            x_hat, z = model(x)

            rec = _rec_loss_l2sum_mean(x_hat, x)
            ref = _latent_nn_ref_loss(z, Href)

            loss = rec + lambda_refer * ref

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            steps += 1
            loss_sum += float(loss.item())
            rec_sum += float(rec.item())
            ref_sum += float(ref.item())

            pbar.set_postfix({
                "loss": f"{loss_sum/steps:.4f}",
                "rec":  f"{rec_sum/steps:.4f}",
                "ref":  f"{ref_sum/steps:.4f}",
            })

        # epoch log
        loss_m = loss_sum / max(steps, 1)
        rec_m = rec_sum / max(steps, 1)
        ref_m = ref_sum / max(steps, 1)
        lam_ref_m = lambda_refer * ref_m

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{ep},{loss_m},{rec_m},{ref_m},{lambda_refer},{lam_ref_m}\n")

        # save ckpt
        if (ep % save_every == 0) or (ep == epochs):
            ckpt_path = out_dir / f"ckpt_ep{ep:03d}.pt"
            torch.save(
                {
                    "epoch": ep,
                    "cfg": cfg,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "ref_indices": ref_indices,
                },
                ckpt_path,
            )

    print(f"[OK] CAE training finished in {(time.time()-t0)/60:.1f} min. outputs in: {out_dir.as_posix()}")
    return out_dir.as_posix()