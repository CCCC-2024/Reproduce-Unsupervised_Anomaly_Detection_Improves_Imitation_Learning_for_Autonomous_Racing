# scripts/build_mix_dataset.py
from __future__ import annotations

import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

# =========================
# User editable (no terminal needed)
# =========================
PKL_PATH = "data/raw/16000clean7dirty.pkl"

DIRTY_KEY = "raindrop"      # foggy, greenmarker, plastic, raindrop, hitwall, debris, dirtytrain
CLEAN_N = 16000
DIRTY_N = 1600              # 10:1 default
SEED = 42
SHUFFLE = True

# Saving options
COMPRESS = False            # True -> smaller file, slower; False -> faster, larger
COPY_BATCH = 512            # batch size for progress-visible copying
# =========================


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_pkl(pkl_path: Path) -> dict:
    print(f"[1/5] Loading PKL: {pkl_path.resolve()}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in pkl, got {type(data)}")
    return data


def chunk_copy(dst: np.ndarray, src: np.ndarray, desc: str, batch: int) -> None:
    n = src.shape[0]
    for i in tqdm(range(0, n, batch), desc=desc, unit="img"):
        j = min(i + batch, n)
        dst[i:j] = src[i:j]


def main() -> None:
    pkl_path = Path(PKL_PATH)
    if not pkl_path.exists():
        raise FileNotFoundError(f"PKL not found: {pkl_path.resolve()}")

    out_npz = Path(f"data/processed/{DIRTY_KEY}_mix.npz")
    out_idx = Path(f"data/splits/{DIRTY_KEY}_mix_idx.npz")
    ensure_dir(out_npz.parent)
    ensure_dir(out_idx.parent)

    rng = np.random.default_rng(SEED)

    data = load_pkl(pkl_path)

    if "clean" not in data:
        raise KeyError("Key 'clean' not found in pkl.")
    if DIRTY_KEY not in data:
        raise KeyError(f"Key '{DIRTY_KEY}' not found. Available keys: {list(data.keys())}")

    clean = data["clean"]
    dirty = data[DIRTY_KEY]

    if not (isinstance(clean, np.ndarray) and isinstance(dirty, np.ndarray)):
        raise TypeError("clean/dirty must be numpy arrays")

    if clean.ndim != 4 or dirty.ndim != 4:
        raise ValueError(f"Expect 4D arrays (N,H,W,C). Got clean {clean.shape}, dirty {dirty.shape}")
    if clean.shape[1:] != dirty.shape[1:]:
        raise ValueError(f"Image shape mismatch: clean {clean.shape[1:]}, dirty {dirty.shape[1:]}")

    n_clean_total = clean.shape[0]
    n_dirty_total = dirty.shape[0]
    n_clean = min(CLEAN_N, n_clean_total)
    n_dirty = min(DIRTY_N, n_dirty_total)

    print(f"[2/5] Sampling indices (seed={SEED})")
    idx_clean = rng.choice(n_clean_total, size=n_clean, replace=False)
    idx_dirty = rng.choice(n_dirty_total, size=n_dirty, replace=False)

    print(f"[3/5] Building mixed array X (clean={n_clean}, dirty={n_dirty})")
    N = n_clean + n_dirty
    H, W, C = clean.shape[1:]
    X = np.empty((N, H, W, C), dtype=np.uint8)

    # copy with progress (chunked)
    clean_sel = clean[idx_clean].astype(np.uint8, copy=False)
    dirty_sel = dirty[idx_dirty].astype(np.uint8, copy=False)
    chunk_copy(X[:n_clean], clean_sel, desc="Copy clean", batch=COPY_BATCH)
    chunk_copy(X[n_clean:], dirty_sel, desc=f"Copy {DIRTY_KEY}", batch=COPY_BATCH)

    y_dirty = np.zeros((N,), dtype=np.uint8)
    y_dirty[n_clean:] = 1

    print(f"[4/5] Shuffling: {SHUFFLE}")
    perm = np.arange(N)
    if SHUFFLE:
        rng.shuffle(perm)
        X = X[perm]
        y_dirty = y_dirty[perm]

    print(f"[5/5] Saving NPZ (compress={COMPRESS})")
    save_fn = np.savez_compressed if COMPRESS else np.savez

    save_fn(
        out_npz,
        X=X,
        y_dirty=y_dirty,
        dirty_key=np.array(DIRTY_KEY),
        seed=np.array(SEED, dtype=np.int64),
        perm=perm.astype(np.int64),
    )

    np.savez_compressed(
        out_idx,
        idx_clean=idx_clean.astype(np.int64),
        idx_dirty=idx_dirty.astype(np.int64),
        clean_total=np.array(n_clean_total, dtype=np.int64),
        dirty_total=np.array(n_dirty_total, dtype=np.int64),
        n_clean=np.array(n_clean, dtype=np.int64),
        n_dirty=np.array(n_dirty, dtype=np.int64),
        dirty_key=np.array(DIRTY_KEY),
        seed=np.array(SEED, dtype=np.int64),
        shuffle=np.array(bool(SHUFFLE)),
    )

    print("\n=== DONE ===")
    print(f"dirty_key: {DIRTY_KEY}")
    print(f"X: shape={X.shape}, dtype={X.dtype}")
    print(f"Saved mix: {out_npz.resolve()}")
    print(f"Saved idx: {out_idx.resolve()}")


if __name__ == "__main__":
    main()
