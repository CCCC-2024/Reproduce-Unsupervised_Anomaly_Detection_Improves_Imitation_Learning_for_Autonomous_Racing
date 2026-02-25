# src/train/train_cae.py
from __future__ import annotations

import os
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
    Expect NPZ keys:
      X: (N,224,224,3) uint8
      y_dirty: (N,) int {0,1}  (0=clean, 1=dirty)
    Return:
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
    x_u8: (B,224,224,3) uint8 -> (B,3,img_size,img_size) float in [0,1]
    """
    x = torch.from_numpy(x_u8).permute(0, 3, 1, 2).float() / 255.0
    if img_size != x.shape[-1]:
        x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return x.to(device, non_blocking=True)


@torch.no_grad()
def build_hrefer(
    model,
    ds: NpzMixDataset,
    ref_indices: np.ndarray,
    device: torch.device,
    img_size: int,
    batch: int = 256,
) -> torch.Tensor:
    """
    Build normalized Href (M,D) once per epoch.
    Return: (M,D) on GPU, each row L2-normalized.
    """
    model.eval()
    H = []
    for s in range(0, len(ref_indices), batch):
        ids = ref_indices[s : s + batch]
        x = _to_tensor_u8_bhwc(ds.X[ids], device=device, img_size=img_size)
        _, z = model(x)
        z_norm = F.normalize(z, dim=1)  # normalize direction
        H.append(z_norm.detach())
    return torch.cat(H, dim=0)  # (M,D) normalized


def cosine_dist(a_norm: torch.Tensor, b_norm: torch.Tensor) -> torch.Tensor:
    """
    a_norm, b_norm: (B,D) normalized -> cosine distance per sample
    """
    return 1.0 - (a_norm * b_norm).sum(dim=1)


def latent_ref_loss_nn(z_norm: torch.Tensor, Href_norm: torch.Tensor) -> torch.Tensor:
    """
    NN by cosine similarity, loss by cosine distance.
    z_norm:   (B,D) normalized
    Href_norm:(M,D) normalized
    """
    sim = z_norm @ Href_norm.T           # (B,M)
    nn = sim.argmax(dim=1)               # (B,)
    hnear = Href_norm[nn]                # (B,D)
    return cosine_dist(z_norm, hnear).mean()


def latent_ref_loss_random(z_norm: torch.Tensor, Href_norm: torch.Tensor) -> torch.Tensor:
    """
    Random pairing in normalized latent space, loss by cosine distance.
    """
    rid = torch.randint(0, Href_norm.shape[0], (z_norm.shape[0],), device=z_norm.device)
    zref = Href_norm[rid].detach()
    return cosine_dist(z_norm, zref).mean()


def train_cae(cfg: Dict[str, Any]) -> str:
    """
    Called by scripts/run_train_cae.py
    """

    exp_name = _get(cfg, "exp_name", default="run")
    refer_mode = str(_get(cfg, "train", "refer_mode", default="nn")).lower()  # "nn" or "random"

    npz_path = _get(cfg, "data", "npz_path", default=None)
    img_size = int(_get(cfg, "data", "img_size", default=64))
    only_clean = bool(_get(cfg, "data", "only_clean", default=False))
    limit = _get(cfg, "data", "limit", default=None)

    device_str = str(_get(cfg, "train", "device", default="auto")).lower()
    seed = int(_get(cfg, "train", "seed", default=0))
    epochs = int(_get(cfg, "train", "epochs", default=100))
    batch_size = int(_get(cfg, "train", "batch_size", default=256))
    lr = float(_get(cfg, "train", "lr", default=1e-3))
    lambda_refer = float(_get(cfg, "train", "lambda_refer", default=1.0))
    refer_m = int(_get(cfg, "train", "refer_m", default=2048))
    num_workers = int(_get(cfg, "train", "num_workers", default=0))
    save_every = int(_get(cfg, "train", "save_every", default=10))

    # Prevent the "hide info in ||z||" loophole
    beta_norm = float(_get(cfg, "train", "beta_norm", default=5.0))

    ref_clean_only = bool(_get(cfg, "train", "ref_clean_only", default=False))
    out_dir = _get(cfg, "output", "out_dir", default=f"outputs/cae/{exp_name}")

    # ---------------- setup ----------------
    set_seed(seed)
    repo_root = Path.cwd()

    if npz_path is None:
        raise ValueError("cfg.data.npz_path is required")

    npz_path = (repo_root / npz_path).resolve()
    ds = NpzMixDataset(npz_path, only_clean=only_clean, limit=limit)

    y_dirty = ds.y_dirty
    N = len(ds)

    # Windows: avoid multiprocessing dataloader issues
    if os.name == "nt" and num_workers != 0:
        print("[WARN] Windows: forcing num_workers=0 to avoid dataloader multiprocessing issues.")
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

    # ---------------- build ref_indices (once) ----------------
    rng = np.random.default_rng(seed)

    if ref_clean_only:
        pool = np.where(y_dirty == 0)[0]
        tag = "CLEAN_ONLY"
    else:
        pool = np.arange(N)
        tag = "MIXED"

    if len(pool) < refer_m:
        raise ValueError(f"ref pool size={len(pool)} < refer_m={refer_m}")

    ref_indices = rng.choice(pool, size=refer_m, replace=False).astype(np.int64)

    print(
        f"[REF] mode={tag} size={len(ref_indices)} dirty_ratio={float(y_dirty[ref_indices].mean()):.4f} "
        f"(dirty={int(y_dirty[ref_indices].sum())}/{len(ref_indices)}) "
        f"min={int(ref_indices.min())} max={int(ref_indices.max())} unique={len(np.unique(ref_indices))}"
    )
    print(f"[CFG] refer_mode={refer_mode} lambda_refer={lambda_refer} beta_norm={beta_norm} epochs={epochs} out_dir={out_dir}")

    # ---------------- outputs/log ----------------
    out_dir = (repo_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text(
            "epoch,loss,rec,ref,norm_reg,lambda_ref,beta_norm,lambda_ref_times_ref,beta_norm_times_norm\n",
            encoding="utf-8",
        )

    # ---------------- train ----------------
    t0 = time.time()
    for ep in range(1, epochs + 1):
        # Step1: build Href once per epoch (paper-style)
        Href = build_hrefer(model, ds, ref_indices, device=device, img_size=img_size, batch=256)

        model.train()
        pbar = tqdm(dl, desc=f"epoch {ep}/{epochs}", ncols=130)

        loss_sum = rec_sum = ref_sum = norm_sum = 0.0
        steps = 0

        for x_u8, _, _ in pbar:
            x_u8_np = x_u8.numpy() if isinstance(x_u8, torch.Tensor) else np.asarray(x_u8)
            x = _to_tensor_u8_bhwc(x_u8_np, device=device, img_size=img_size)

            x_hat, z = model(x)
            rec = F.mse_loss(x_hat, x)

            # refer works on direction only
            z_norm = F.normalize(z, dim=1)

            # stop model from encoding info into ||z||
            norm_reg = ((z.norm(dim=1) - 1.0) ** 2).mean()

            if refer_mode == "random":
                ref = latent_ref_loss_random(z_norm, Href)
            elif refer_mode == "nn":
                ref = latent_ref_loss_nn(z_norm, Href)
            else:
                raise ValueError(f"Unknown refer_mode={refer_mode} (expected 'nn' or 'random')")

            loss = rec + lambda_refer * ref + beta_norm * norm_reg

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            steps += 1
            loss_sum += float(loss.item())
            rec_sum += float(rec.item())
            ref_sum += float(ref.item())
            norm_sum += float(norm_reg.item())

            pbar.set_postfix({
                "loss": f"{loss_sum/steps:.4f}",
                "rec":  f"{rec_sum/steps:.4f}",
                "ref":  f"{ref_sum/steps:.6f}",
                "lam*ref": f"{(lambda_refer*(ref_sum/steps)):.4f}",
                "norm": f"{norm_sum/steps:.6f}",
                "b*norm": f"{(beta_norm*(norm_sum/steps)):.4f}",
            })

        # epoch log
        loss_m = loss_sum / max(steps, 1)
        rec_m = rec_sum / max(steps, 1)
        ref_m = ref_sum / max(steps, 1)
        norm_m = norm_sum / max(steps, 1)
        lam_ref_m = lambda_refer * ref_m
        b_norm_m = beta_norm * norm_m

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{ep},{loss_m},{rec_m},{ref_m},{norm_m},{lambda_refer},{beta_norm},{lam_ref_m},{b_norm_m}\n")

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