# scripts/check_dirty_nn_in_ref.py
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# ====== change here if needed ======
NPZ_PATH = "data/processed/raindrop_mix.npz"
CKPT_PATH = "outputs/cae/raindrop/ckpt_ep100.pt"
SAMPLE_DIRTY = 400
BATCH = 256
# ===================================


def main():
    # ---- make repo root importable (so `import src...` works) ----
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))

    d = np.load(repo / NPZ_PATH, allow_pickle=True)
    X = d["X"]  # (N,224,224,3) uint8
    y_dirty = d["y_dirty"].astype(np.int64)
    N = len(y_dirty)

    ckpt = torch.load(repo / CKPT_PATH, map_location="cpu")
    ref_idx = np.array(ckpt.get("ref_indices", []), dtype=np.int64)
    if len(ref_idx) == 0:
        print("[ERR] ckpt has no ref_indices")
        return

    print(f"[REF] size={len(ref_idx)} dirty_ratio={y_dirty[ref_idx].mean():.4f} "
          f"(dirty={int(y_dirty[ref_idx].sum())}/{len(ref_idx)})")
    print(f"[REF] min={int(ref_idx.min())} max={int(ref_idx.max())} unique={len(np.unique(ref_idx))}")

    # ---- load model ----
    from src.models.cae import CAE
    model = CAE(latent_dim=256)
    model.load_state_dict(ckpt["model"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    @torch.no_grad()
    def latent_from_u8(img_u8: np.ndarray) -> torch.Tensor:
        """
        img_u8: (B,224,224,3) uint8
        return: (B, latent_dim) on CPU
        """
        x = torch.from_numpy(img_u8).permute(0, 3, 1, 2).float() / 255.0  # (B,3,224,224)
        x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
        x = x.to(device)
        x_hat, z = model(x)  # forward returns (recon, latent)
        return z.detach().cpu()

    # ---- encode reference latents ----
    Href = []
    for s in range(0, len(ref_idx), BATCH):
        ids = ref_idx[s:s + BATCH]
        Href.append(latent_from_u8(X[ids]))
    Href = torch.cat(Href, dim=0)  # (M, D)

    # ---- sample dirty queries ----
    dirty_ids = np.where(y_dirty == 1)[0]
    if len(dirty_ids) == 0:
        print("[ERR] no dirty samples in dataset")
        return

    q_ids = np.random.choice(dirty_ids, size=min(SAMPLE_DIRTY, len(dirty_ids)), replace=False)

    Hq = []
    for s in range(0, len(q_ids), BATCH):
        ids = q_ids[s:s + BATCH]
        Hq.append(latent_from_u8(X[ids]))
    Hq = torch.cat(Hq, dim=0)  # (Q, D)

    # ---- nearest neighbor in ref for each dirty query ----
    # dist^2 = |q|^2 + |r|^2 - 2 q r^T
    q2 = (Hq ** 2).sum(dim=1, keepdim=True)          # (Q,1)
    r2 = (Href ** 2).sum(dim=1, keepdim=True).T      # (1,M)
    dist2 = q2 + r2 - 2.0 * (Hq @ Href.T)            # (Q,M)

    nn_pos = dist2.argmin(dim=1).numpy()             # index in [0..M-1]
    nn_dataset_ids = ref_idx[nn_pos]                 # map to original dataset index
    nn_dirty_ratio = float(y_dirty[nn_dataset_ids].mean())

    print(f"[NN] among dirty queries, NN-in-ref dirty_ratio = {nn_dirty_ratio:.4f} "
          f"(dirty={int(y_dirty[nn_dataset_ids].sum())}/{len(nn_dataset_ids)})")


if __name__ == "__main__":
    main()
