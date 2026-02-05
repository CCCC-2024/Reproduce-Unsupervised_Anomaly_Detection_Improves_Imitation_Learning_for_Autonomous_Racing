# scripts/debug_pcc_stats.py
from pathlib import Path
import numpy as np
import pandas as pd

CSV_PATH = "outputs/scores/raindrop_ep100_scores.csv"

def main():
    repo = Path(__file__).resolve().parents[1]
    df = pd.read_csv(repo / CSV_PATH)

    p = df["pcc_smooth"].to_numpy()
    y = df["y_dirty"].to_numpy()

    delta = float(df["delta"].iloc[0])
    print("[STAT] N =", len(df))
    print("[STAT] delta =", delta)
    print("[STAT] pcc_smooth: min/median/max =",
          float(p.min()), float(np.median(p)), float(p.max()))
    print("[STAT] pcc_smooth percentiles (1,5,10,50,90,95,99) =",
          np.percentile(p, [1,5,10,50,90,95,99]).round(6))

    pc = p[y == 0]
    pd_ = p[y == 1]
    print("[STAT] clean count =", len(pc), "dirty count =", len(pd_))
    print("[STAT] clean min/median/max =",
          float(pc.min()), float(np.median(pc)), float(pc.max()))
    print("[STAT] dirty min/median/max =",
          float(pd_.min()), float(np.median(pd_)), float(pd_.max()))

    print("[CHECK] any below delta? ", int((p < delta).sum()))

if __name__ == "__main__":
    main()
