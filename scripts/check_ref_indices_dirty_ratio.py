# scripts/check_ref_indices_dirty_ratio.py
from pathlib import Path
import numpy as np
import torch

NPZ_PATH  = "data/processed/raindrop_mix.npz"
CKPT_PATH = "outputs/cae/raindrop/ckpt_ep100.pt"

def main():
    repo = Path(__file__).resolve().parents[1]
    d = np.load(repo / NPZ_PATH, allow_pickle=True)
    y = d["y_dirty"].astype(int)

    ckpt = torch.load(repo / CKPT_PATH, map_location="cpu")
    ref = ckpt.get("ref_indices", None)
    if ref is None:
        print("[ERR] no ref_indices in ckpt")
        return
    ref = np.array(ref, dtype=int)

    ratio = y[ref].mean()
    print(f"[REF] size={len(ref)} dirty_ratio={ratio:.4f} (dirty={y[ref].sum()} / {len(ref)})")
    print(f"[REF] min={ref.min()} max={ref.max()} unique={len(np.unique(ref))}")

if __name__ == "__main__":
    main()
