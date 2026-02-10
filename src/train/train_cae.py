# src/train/train_cae.py
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

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
      y: int64
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

    def __getitem__(self, i: int):
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
def _encode_latent(model, x: torch.Tensor) -> torch.Tensor:
    # model forward: x_hat, z = model(x)
    _, z = model(x)
    return z


def _latent_ref_loss(z: torch.Tensor, z_ref: torch.Tensor) -> torch.Tensor:
    """
    NN latent ref loss:
      min_j ||z_i - zref_j||^2, average over i
    z:     (B,D)
    z_ref: (M,D)
    """
    z2 = (z ** 2).sum(dim=1, keepdim=True)          # (B,1)
    r2 = (z_ref ** 2).sum(dim=1, keepdim=True).T    # (1,M)
    dist2 = z2 + r2 - 2.0 * (z @ z_ref.T)           # (B,M)
    return dist2.min(dim=1).values.mean()


def train_cae(cfg: Dict[str, Any]) -> str:
    """
    Called by scripts/run_train_cae.py
    Compatible with your configs/cae/raindrop.yaml fields.
    """

    # ---------------- cfg parse (match your yaml) ----------------
    exp_name = _get(cfg, "exp_name", default="run")

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

    out_dir = _get(cfg, "output", "out_dir", default=f"outputs/cae/{exp_name}")

    # IMPORTANT FIX: ref_indices must be sampled from CLEAN only
    ref_clean_only = True  # <-- 强制 clean-only（你现在的核心问题就在这里）

    # ---------------- setup ----------------
    set_seed(seed)
    repo_root = Path.cwd()

    if npz_path is None:
        raise ValueError("cfg.data.npz_path is required")

    npz_path = (repo_root / npz_path).resolve()
    ds = NpzMixDataset(npz_path, only_clean=only_clean, limit=limit)

    y_dirty = ds.y_dirty
    N = len(ds)

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

    # ---------------- build ref_indices (CLEAN ONLY) ----------------
    rng = np.random.default_rng(seed)

    if ref_clean_only:
        clean_pool = np.where(y_dirty == 0)[0]
        if len(clean_pool) < refer_m:
            raise ValueError(f"clean_pool={len(clean_pool)} < refer_m={refer_m}")
        ref_indices = rng.choice(clean_pool, size=refer_m, replace=False)
    else:
        ref_indices = rng.choice(np.arange(N), size=refer_m, replace=False)

    ref_indices = np.array(ref_indices, dtype=np.int64)

    print(f"[REF] size={len(ref_indices)} dirty_ratio={float(y_dirty[ref_indices].mean()):.4f} "
          f"(dirty={int(y_dirty[ref_indices].sum())}/{len(ref_indices)}) "
          f"min={int(ref_indices.min())} max={int(ref_indices.max())} unique={len(np.unique(ref_indices))}")

    # ---------------- train ----------------
    out_dir = (repo_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()

        pbar = tqdm(dl, desc=f"epoch {ep}/{epochs}", ncols=110)
        loss_sum = rec_sum = ref_sum = 0.0
        steps = 0

        for x_u8, _, _ in pbar:
            x_u8_np = x_u8.numpy() if isinstance(x_u8, torch.Tensor) else np.asarray(x_u8)
            x = _to_tensor_u8_bhwc(x_u8_np, device=device, img_size=img_size)

            x_hat, z = model(x)
            rec = F.mse_loss(x_hat, x)

            # sample ref batch from CLEAN ref_indices
            rid = rng.choice(ref_indices, size=min(batch_size, len(ref_indices)), replace=False)
            xref = _to_tensor_u8_bhwc(ds.X[rid], device=device, img_size=img_size)
            zref = _encode_latent(model, xref)

            ref = _latent_ref_loss(z, zref)

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

        if (ep % save_every == 0) or (ep == epochs):
            ckpt_path = out_dir / f"ckpt_ep{ep:03d}.pt"
            torch.save(
                {
                    "epoch": ep,
                    "cfg": cfg,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "ref_indices": ref_indices,  # <-- 关键：写入 clean-only ref
                },
                ckpt_path,
            )

    print(f"[OK] CAE training finished in {(time.time()-t0)/60:.1f} min. outputs in: {out_dir.as_posix()}")
    return out_dir.as_posix()
