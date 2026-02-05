# scripts/viz_npz_samples.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

NPZ_PATH = "data/processed/raindrop_mix.npz"
OUT_FIG = "outputs/figs/npz_samples_clean_vs_dirty.png"
K = 8          # 每类展示多少张
SEED = 0

def main():
    repo = Path(__file__).resolve().parents[1]
    d = np.load(repo / NPZ_PATH, allow_pickle=True)
    X = d["X"]                 # (N,224,224,3) uint8
    y = d["y_dirty"].astype(int)

    idx_clean = np.where(y == 0)[0]
    idx_dirty = np.where(y == 1)[0]

    print(f"[INFO] N={len(y)} clean={len(idx_clean)} dirty={len(idx_dirty)}")
    if len(idx_dirty) == 0:
        print("[ERR] dirty count is 0, check your dataset build step.")
        return

    rng = np.random.default_rng(SEED)
    pick_clean = rng.choice(idx_clean, size=min(K, len(idx_clean)), replace=False)
    pick_dirty = rng.choice(idx_dirty, size=min(K, len(idx_dirty)), replace=False)

    ncols = max(len(pick_clean), len(pick_dirty))
    fig, axes = plt.subplots(2, ncols, figsize=(2.2*ncols, 4.6))

    for j in range(ncols):
        ax = axes[0, j]
        ax.axis("off")
        if j < len(pick_clean):
            ax.imshow(X[pick_clean[j]])
            ax.set_title(f"clean idx={pick_clean[j]}", fontsize=9)

    for j in range(ncols):
        ax = axes[1, j]
        ax.axis("off")
        if j < len(pick_dirty):
            ax.imshow(X[pick_dirty[j]])
            ax.set_title(f"dirty idx={pick_dirty[j]}", fontsize=9)

    out = repo / OUT_FIG
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print("[OK] saved:", out.resolve())

if __name__ == "__main__":
    main()
