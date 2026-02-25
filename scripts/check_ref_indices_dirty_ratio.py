from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to ckpt .pt")
    ap.add_argument("--npz", default="data/processed/raindrop_mix.npz", help="npz path for y_dirty")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    ckpt_path = (repo / args.ckpt).resolve()
    npz_path = (repo / args.npz).resolve()

    d = np.load(npz_path, allow_pickle=True)
    y_dirty = d["y_dirty"].astype(np.int64)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ref = ckpt.get("ref_indices", None)
    if ref is None:
        print("[ERR] no ref_indices in ckpt")
        return
    ref = np.array(ref, dtype=np.int64)

    print(f"[CKPT] {ckpt_path}")
    print(f"[REF] size={len(ref)} dirty_ratio={float(y_dirty[ref].mean()):.4f} (dirty={int(y_dirty[ref].sum())} / {len(ref)})")
    print(f"[REF] min={int(ref.min())} max={int(ref.max())} unique={len(np.unique(ref))}")

if __name__ == "__main__":
    main()