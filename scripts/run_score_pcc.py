# scripts/run_score_pcc.py
from __future__ import annotations

import sys
from pathlib import Path
from xml.parsers.expat import model

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import yaml


# ====== change here if needed, then click "Run" ======
NPZ_PATH = "data/processed/raindrop_mix.npz"
# CKPT_PATH = "outputs/cae/raindrop_ep10_cleanref_lam200/ckpt_ep010.pt"
# CKPT_PATH = "outputs/cae/raindrop_ep10_cleanref_lam200_random/ckpt_ep010.pt"
THRESH_CFG = "configs/postprocess/threshold.yaml"


# OUT_CSV = "outputs/scores/raindrop_ep010_scores_224pcc.csv"
# OUT_FIG = "outputs/figs/raindrop_ep010_pcc_224pcc.png"

CKPT_PATH = "outputs/cae/raindrop__ep10__refrandom__cleanref1__lam200__cos__normreg5/ckpt_ep010.pt"
OUT_CSV  = "outputs/scores/raindrop__ep10__refrandom__cleanref1__lam200__scores_224pcc_normreg5.csv"
OUT_FIG  = "outputs/figs/raindrop__ep10__refrandom__cleanref1__lam200__pcc_224pcc_normreg5.png"
# =====================================================

def pcc_batch(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Pearson correlation coefficient per sample.
    x,y: (B,3,H,W) float in [0,1]
    return: (B,)
    """
    B = x.shape[0]
    x = x.reshape(B, -1)
    y = y.reshape(B, -1)

    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)

    num = (x * y).mean(dim=1)
    den = x.std(dim=1) * y.std(dim=1) + eps
    return num / den


def median_filter_1d(arr: np.ndarray, win: int) -> np.ndarray:
    """
    Simple 1D median filter.
    Note: N~17600 and win=100 is OK for a one-off run.
    """
    n = len(arr)
    r = win // 2
    out = np.empty_like(arr)
    for i in range(n):
        a = max(0, i - r)
        b = min(n, i + r + 1)
        out[i] = np.median(arr[a:b])
    return out


def main():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    # ---- load threshold config ----
    tcfg = yaml.safe_load((repo_root / THRESH_CFG).read_text(encoding="utf-8"))
    win = int(tcfg.get("median_window", 100))
    offset = float(tcfg.get("delta_offset", 0.05))
    reorder_dirty_last = bool(tcfg.get("reorder_dirty_last", True))

    # ---- load npz ----
    d = np.load(repo_root / NPZ_PATH, allow_pickle=True)
    X = d["X"]  # (N,224,224,3) uint8
    y_dirty = d["y_dirty"].astype(np.int64)  # (N,)
    N = len(y_dirty)

    # ---- order index (paper-style plotting: clean first, dirty last) ----
    orig_idx = np.arange(N, dtype=np.int64)
    if reorder_dirty_last:
        order_idx = np.concatenate([orig_idx[y_dirty == 0], orig_idx[y_dirty == 1]])
    else:
        order_idx = orig_idx.copy()

    # ---- load ckpt/model ----
    ckpt_path_abs = repo_root / CKPT_PATH
    print("[DEBUG] CKPT_PATH =", ckpt_path_abs)
    ckpt = torch.load(ckpt_path_abs, map_location="cpu", weights_only=False)
    print("[DEBUG] ckpt keys:", list(ckpt.keys()))

    from src.models.cae import CAE
    model = CAE(latent_dim=256)
    model.load_state_dict(ckpt["model"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # ---- scoring loop (order space) ----
    batch = 128
    pcc_order = np.empty((N,), dtype=np.float32)

    with torch.no_grad():
        for s in range(0, N, batch):
            ids = order_idx[s:s + batch]
            x224 = torch.from_numpy(X[ids]).permute(0, 3, 1, 2).float() / 255.0   # (B,3,224,224)
            x64  = F.interpolate(x224, size=(64, 64), mode="bilinear", align_corners=False)

            x64  = x64.to(device)
            x224 = x224.to(device)

            x_hat64, _ = model(x64)
            x_hat64 = x_hat64.clamp(0.0, 1.0)

            x_hat224 = F.interpolate(x_hat64, size=(224, 224), mode="bilinear", align_corners=False)

            p = pcc_batch(x224, x_hat224).detach().cpu().numpy().astype(np.float32)
            pcc_order[s:s + len(p)] = p


    # ---- smoothing + threshold (order space) ----
    pcc_smooth_order = median_filter_1d(pcc_order, win=win).astype(np.float32)
    delta = float(np.median(pcc_smooth_order) - offset)
    pred_anom_order = (pcc_smooth_order < delta).astype(np.int64)  # 1=dirty/anomaly

    # ---- map back to original index space (useful for cleaning later) ----
    pcc_smooth_orig = np.empty((N,), dtype=np.float32)
    pred_anom_orig = np.empty((N,), dtype=np.int64)
    pcc_smooth_orig[order_idx] = pcc_smooth_order
    pred_anom_orig[order_idx] = pred_anom_order

    # ---- save csv ----
    out_csv = repo_root / OUT_CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        # order-space rows
        "order_i": np.arange(N, dtype=np.int64),
        "orig_i": order_idx.astype(np.int64),

        # labels in the same order as rows
        "y_dirty": y_dirty[order_idx].astype(np.int64),

        # scores in order-space
        "pcc": pcc_order.astype(np.float32),
        "pcc_smooth": pcc_smooth_order.astype(np.float32),

        # threshold + prediction in order-space
        "delta": np.full((N,), delta, dtype=np.float32),
        "pred_anom": pred_anom_order.astype(np.int64),
    })
    df.to_csv(out_csv, index=False)
    print(f"[OK] saved scores csv: {out_csv.resolve()}")

    # ---- plot (order space) ----
    out_fig = repo_root / OUT_FIG
    out_fig.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))
    plt.plot(pcc_smooth_order, linewidth=1)
    plt.axhline(delta, linestyle="--")
    plt.title(f"PCC smooth (win={win}) | delta = median - {offset:.2f} = {delta:.4f}")
    plt.xlabel("ordered index (clean first, dirty last)" if reorder_dirty_last else "index")
    plt.ylabel("PCC")
    plt.tight_layout()
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"[OK] saved figure: {out_fig.resolve()}")

    # ---- quick PR (Table II style) ----
    y = y_dirty[order_idx]
    p = pred_anom_order
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    print(f"[PR] tp={tp} fp={fp} fn={fn} | precision={precision:.4f} recall={recall:.4f}")

    # ---- extra debug: how many predicted anomalies in total ----
    print(f"[DEBUG] pred_anom_order sum = {int(pred_anom_order.sum())} / {N}")
    print(f"[DEBUG] pred_anom_orig  sum = {int(pred_anom_orig.sum())} / {N}")


if __name__ == "__main__":
    main()
