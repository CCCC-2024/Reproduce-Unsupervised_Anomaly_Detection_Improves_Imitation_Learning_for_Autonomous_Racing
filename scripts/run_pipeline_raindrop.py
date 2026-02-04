# scripts/run_pipeline_raindrop.py
from __future__ import annotations

import sys
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    py = sys.executable

    # 1) build mixed dataset 
    mix_npz = repo / "data" / "processed" / "raindrop_mix.npz"
    mix_idx = repo / "data" / "splits" / "raindrop_mix_idx.npz"
    if not (mix_npz.exists() and mix_idx.exists()):
        run([py, str(repo / "scripts" / "build_mix_dataset.py")])
    else:
        print("[OK] mix dataset exists:", mix_npz)

    # 2) train CAE (raindrop config) - edit CFG_PATH inside run_train_cae.py beforehand
    run([py, str(repo / "scripts" / "run_train_cae.py")])

    # 3) score PCC - edit CKPT_PATH inside run_score_pcc.py beforehand
    run([py, str(repo / "scripts" / "run_score_pcc.py")])

    # 4) quick check CSV stats
    run([py, str(repo / "scripts" / "quick_check_scores.py")])


if __name__ == "__main__":
    main()
