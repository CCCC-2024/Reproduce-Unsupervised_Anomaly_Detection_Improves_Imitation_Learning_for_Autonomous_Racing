# scripts/check_dirty_nn_in_ref.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to ckpt .pt (relative to repo root)")
    ap.add_argument("--npz", default="data/processed/raindrop_mix.npz", help="npz path for X/y_dirty")
    ap.add_argument("--sample_dirty", type=int, default=400)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--img_size", type=int, default=64)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))

    ckpt_path = (repo / args.ckpt).resolve()
    npz_path = (repo / args.npz).resolve()

    d = np.load(npz_path, allow_pickle=True)
    X = d["X"]  # (N,224,224,3) uint8
    y_dirty = d["y_dirty"].astype(np.int64)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ref_idx = np.array(ckpt.get("ref_indices", []), dtype=np.int64)
    if len(ref_idx) == 0:
        print("[ERR] ckpt has no ref_indices")
        return

    print(f"[CKPT] {ckpt_path}")
    print(f"[REF] size={len(ref_idx)} dirty_ratio={float(y_dirty[ref_idx].mean()):.4f} "
          f"(dirty={int(y_dirty[ref_idx].sum())}/{len(ref_idx)})")
    print(f"[REF] min={int(ref_idx.min())} max={int(ref_idx.max())} unique={len(np.unique(ref_idx))}")

    from src.models.cae import CAE
    model = CAE(latent_dim=256)
    model.load_state_dict(ckpt["model"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    @torch.no_grad()
    def latent_from_u8(img_u8: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(img_u8).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(args.img_size, args.img_size), mode="bilinear", align_corners=False)
        x = x.to(device)
        _, z = model(x)
        return z.detach().cpu()

    # encode ref latents
    Href = []
    for s in range(0, len(ref_idx), args.batch):
        ids = ref_idx[s:s + args.batch]
        Href.append(latent_from_u8(X[ids]))
    Href = torch.cat(Href, dim=0)  # (M,D)

    # sample dirty queries
    dirty_ids = np.where(y_dirty == 1)[0]
    qn = min(args.sample_dirty, len(dirty_ids))
    q_ids = np.random.choice(dirty_ids, size=qn, replace=False)

    Hq = []
    for s in range(0, len(q_ids), args.batch):
        ids = q_ids[s:s + args.batch]
        Hq.append(latent_from_u8(X[ids]))
    Hq = torch.cat(Hq, dim=0)  # (Q,D)

    # NN in ref
    q2 = (Hq ** 2).sum(dim=1, keepdim=True)          # (Q,1)
    r2 = (Href ** 2).sum(dim=1, keepdim=True).T      # (1,M)
    dist2 = q2 + r2 - 2.0 * (Hq @ Href.T)            # (Q,M)

    nn_pos = dist2.argmin(dim=1).numpy()
    nn_dataset_ids = ref_idx[nn_pos]
    nn_dirty_ratio = float(y_dirty[nn_dataset_ids].mean())

    print(f"[NN] among dirty queries, NN-in-ref dirty_ratio = {nn_dirty_ratio:.4f} "
          f"(dirty={int(y_dirty[nn_dataset_ids].sum())}/{len(nn_dataset_ids)})")


if __name__ == "__main__":
    main()