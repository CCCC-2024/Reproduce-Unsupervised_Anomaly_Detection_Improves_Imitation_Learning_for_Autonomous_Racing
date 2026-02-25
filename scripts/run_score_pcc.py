# scripts/run_score_pcc.py
from __future__ import annotations

import sys
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import yaml

# ===================== YOU ONLY CHANGE THESE =====================
NPZ_PATH = "data/processed/raindrop_mix.npz"

# 具体 ckpt 文件路径
CKPT_PATH = "outputs/cae/raindrop_22503/ckpt_ep010.pt"

THRESH_CFG = "configs/postprocess/threshold.yaml"

TAG = "raindrop_225_ep010"  # 文件名前缀
OUT_CSV = f"outputs/scores/{TAG}_scores03.csv"
OUT_FIG = f"outputs/figs/{TAG}_pcc03.png"
# ================================================================


def pcc_batch(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    PCC per sample (paper Eq.8 style).
    x,y: (B,3,H,W) float in [0,1]
    return: (B,)
    """
    B = x.shape[0]
    x = x.reshape(B, -1)
    y = y.reshape(B, -1)

    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)

    num = (x * y).sum(dim=1)
    den = torch.sqrt((x * x).sum(dim=1) + eps) * torch.sqrt((y * y).sum(dim=1) + eps)
    return num / den


def median_filter_1d(arr: np.ndarray, win: int) -> np.ndarray:
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

    # threshold config
    tcfg = yaml.safe_load((repo_root / THRESH_CFG).read_text(encoding="utf-8"))
    win = int(tcfg.get("median_window", 100))
    offset = float(tcfg.get("delta_offset", 0.05))
    reorder_dirty_last = bool(tcfg.get("reorder_dirty_last", True))

    # load npz
    d = np.load(repo_root / NPZ_PATH, allow_pickle=True)
    X = d["X"]  # (N,224,224,3) uint8
    y_dirty = d["y_dirty"].astype(np.int64)
    N = len(y_dirty)

    clean_ids = np.where(y_dirty == 0)[0]
    dirty_ids = np.where(y_dirty == 1)[0]
    if reorder_dirty_last:
        order_idx = np.concatenate([clean_ids, dirty_ids])
    else:
        order_idx = np.arange(N)

    # load ckpt
    ckpt_path_abs = (repo_root / CKPT_PATH).resolve()
    print("[DEBUG] NPZ  =", (repo_root / NPZ_PATH).resolve())
    print("[DEBUG] CKPT =", ckpt_path_abs)

    ckpt = torch.load(ckpt_path_abs, map_location="cpu", weights_only=False)
    print("[DEBUG] ckpt keys:", list(ckpt.keys()))

    # read img_size from ckpt cfg if exists (keep consistent with training)
    img_size = 64
    try:
        img_size = int(ckpt.get("cfg", {}).get("data", {}).get("img_size", img_size))
    except Exception:
        pass

    from src.models.cae import CAE
    model = CAE(latent_dim=256)
    model.load_state_dict(ckpt["model"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # scoring
    batch = 256
    pcc_all = np.empty((N,), dtype=np.float32)

    with torch.no_grad():
        for s in range(0, N, batch):
            ids = order_idx[s:s + batch]
            x = torch.from_numpy(X[ids]).permute(0, 3, 1, 2).float() / 255.0
            if img_size != x.shape[-1]:
                x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
            x = x.to(device)

            x_hat, _ = model(x)
            p = pcc_batch(x, x_hat).detach().cpu().numpy().astype(np.float32)
            pcc_all[s:s + len(p)] = p

    # smooth + threshold
    pcc_smooth = median_filter_1d(pcc_all, win=win).astype(np.float32)
    delta = float(np.median(pcc_smooth) - offset)
    pred_anom = (pcc_smooth < delta).astype(np.int64)  # 1=dirty predicted

    # save csv
    out_csv = repo_root / OUT_CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "order_i": np.arange(N, dtype=np.int64),
        "orig_i": order_idx.astype(np.int64),
        "y_dirty": y_dirty[order_idx].astype(np.int64),
        "pcc": pcc_all,
        "pcc_smooth": pcc_smooth,
        "delta": np.full((N,), delta, dtype=np.float32),
        "pred_anom": pred_anom.astype(np.int64),
    })
    df.to_csv(out_csv, index=False)
    print(f"[OK] saved scores csv: {out_csv.resolve()}")

    # plot
    out_fig = repo_root / OUT_FIG
    out_fig.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))
    plt.plot(df["pcc_smooth"].values, linewidth=1)
    plt.axhline(delta, linestyle="--")
    if reorder_dirty_last:
        plt.axvline(len(clean_ids) - 0.5, color="gray", linestyle="--", linewidth=1)
    plt.title(f"PCC smooth (win={win}) | delta = median - {offset:.2f} = {delta:.4f}")
    plt.xlabel("ordered index (clean first, dirty last)" if reorder_dirty_last else "index")
    plt.ylabel("PCC")
    plt.tight_layout()
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"[OK] saved figure: {out_fig.resolve()}")

    # PR
    y = df["y_dirty"].values
    p = df["pred_anom"].values
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    print(f"[PR] tp={tp} fp={fp} fn={fn} | precision={precision:.4f} recall={recall:.4f}")

    print(f"[DEBUG] pred_anom sum = {int(p.sum())} / {len(p)}")
    print(f"[DEBUG] mean(pcc_smooth) clean={float(df.loc[y==0,'pcc_smooth'].mean()):.4f} "
          f"dirty={float(df.loc[y==1,'pcc_smooth'].mean()):.4f} (expect dirty < clean)")


if __name__ == "__main__":
    main()