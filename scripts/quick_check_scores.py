import pandas as pd
import numpy as np

CSV = "outputs/scores/raindrop_raindrop_scores.csv"

df = pd.read_csv(CSV)
y = df["y_dirty"].values
p = df["pred_anom"].values

def stat(arr):
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }

print("=== label counts ===")
print("clean:", int((y==0).sum()), "dirty:", int((y==1).sum()))

print("\n=== pcc_smooth stats ===")
print("clean:", stat(df.loc[y==0, "pcc_smooth"].values))
print("dirty:", stat(df.loc[y==1, "pcc_smooth"].values))

print("\n=== threshold ===")
print("delta:", float(df["delta"].iloc[0]))

print("\n=== pred counts ===")
print("pred_anom=1:", int((p==1).sum()), "pred_anom=0:", int((p==0).sum()))

tp = int(((p==1)&(y==1)).sum())
fp = int(((p==1)&(y==0)).sum())
fn = int(((p==0)&(y==1)).sum())
precision = tp/(tp+fp+1e-12)
recall = tp/(tp+fn+1e-12)
print("\n=== PR ===")
print("tp fp fn:", tp, fp, fn)
print("precision:", precision, "recall:", recall)
