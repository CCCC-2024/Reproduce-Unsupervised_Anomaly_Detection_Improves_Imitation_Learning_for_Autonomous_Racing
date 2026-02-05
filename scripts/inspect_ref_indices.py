# scripts/inspect_ref_indices.py
from pathlib import Path
import torch
import numpy as np

CKPT = "outputs/cae/raindrop/ckpt_ep100.pt"

def main():
    repo = Path(__file__).resolve().parents[1]
    ckpt = torch.load(repo / CKPT, map_location="cpu")
    idx = ckpt.get("ref_indices", None)
    if idx is None:
        print("[ERR] no ref_indices in ckpt")
        return
    idx = np.array(idx, dtype=np.int64)
    print("[REF] len(ref_indices) =", len(idx))
    print("[REF] unique =", len(np.unique(idx)))
    print("[REF] min/max =", int(idx.min()), int(idx.max()))

if __name__ == "__main__":
    main()
