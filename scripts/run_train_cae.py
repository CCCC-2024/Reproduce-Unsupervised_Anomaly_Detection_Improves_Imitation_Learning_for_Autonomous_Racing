from __future__ import annotations
import sys
from pathlib import Path
import yaml

# change here, then click "Run"
# CFG_PATH = "configs/cae/debug_small.yaml"
CFG_PATH = "configs/cae/raindrop.yaml"

def main():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    cfg = yaml.safe_load((repo_root / CFG_PATH).read_text(encoding="utf-8"))
    from src.train.train_cae import train_cae
    out_dir = train_cae(cfg)
    print(f"[OK] CAE training finished. outputs in: {out_dir}")

if __name__ == "__main__":
    main()
