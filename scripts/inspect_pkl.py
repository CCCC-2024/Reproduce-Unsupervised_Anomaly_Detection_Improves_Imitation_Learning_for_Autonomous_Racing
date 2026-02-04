# inspect_pkl.py
import os, pickle, argparse
import numpy as np

def brief(x):
    t = type(x).__name__
    if isinstance(x, np.ndarray):
        mb = x.nbytes / (1024**2)
        return f"ndarray shape={x.shape} dtype={x.dtype} mem={mb:.1f}MB min={x.min():.3f} max={x.max():.3f}"
    if hasattr(x, "shape") and not isinstance(x, (list, tuple, dict, str, bytes)):
        try:
            return f"{t} shape={getattr(x,'shape',None)}"
        except:
            return t
    if isinstance(x, (list, tuple)):
        return f"{t} len={len(x)}"
    if isinstance(x, dict):
        return f"dict keys={list(x.keys())[:20]} (total {len(x)})"
    if isinstance(x, (str, bytes)):
        return f"{t} len={len(x)}"
    return t

def summarize(obj, name="root", depth=0, max_depth=3):
    indent = "  " * depth
    print(f"{indent}- {name}: {brief(obj)}")
    if depth >= max_depth:
        return
    if isinstance(obj, dict):
        # 优先展示常见字段
        keys = list(obj.keys())
        common = [k for k in keys if str(k).lower() in ["images","imgs","obs","frames","x","actions","act","y","steering","throttle","label","labels","is_dirty","dirty","clean","idx_clean","idx_dirty"]]
        ordered = common + [k for k in keys if k not in common]
        for k in ordered[:30]:
            summarize(obj[k], name=str(k), depth=depth+1, max_depth=max_depth)
    elif isinstance(obj, (list, tuple)) and len(obj) > 0:
        summarize(obj[0], name="(first_item)", depth=depth+1, max_depth=max_depth)

def load_pickle(path):
    # 兼容部分旧 pickle
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            return pickle.load(f, encoding="latin1")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="16000clean7dirty.pkl", help="path to .pkl")
    args = ap.parse_args()

    obj = load_pickle(args.pkl)
    print("=== PKL SUMMARY ===")
    summarize(obj, max_depth=4)

    # 尝试估计样本数
    N_guess = None
    if isinstance(obj, dict):
        for k in ["images","imgs","obs","frames","x"]:
            if k in obj and hasattr(obj[k], "__len__"):
                N_guess = len(obj[k]); break
    elif isinstance(obj, (list, tuple)):
        N_guess = len(obj)
    print("\n=== N GUESS ===", N_guess)
