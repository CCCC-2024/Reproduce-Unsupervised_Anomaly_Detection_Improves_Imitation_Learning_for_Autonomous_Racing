# scripts/print_env_report.py
from __future__ import annotations

import os
import platform
import sys
import textwrap
from datetime import datetime
from importlib import import_module

try:
    # Python 3.8+
    from importlib.metadata import version as pkg_version, PackageNotFoundError
except Exception:  # pragma: no cover
    pkg_version = None
    PackageNotFoundError = Exception


KEY_PACKAGES = [
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "opencv-python",
    "Pillow",
    "matplotlib",
    "pyyaml",
    "tqdm",
]


def get_pkg_ver(dist_name: str) -> str:
    """
    Return installed distribution version (preferred) or module __version__ as fallback.
    dist_name should be the pip distribution name.
    """
    # 1) Try importlib.metadata (distribution name)
    if pkg_version is not None:
        try:
            return pkg_version(dist_name)
        except PackageNotFoundError:
            pass
        except Exception:
            pass

    # 2) Fallback: try module import + __version__
    # Map a few dist names to import module names
    module_map = {
        "scikit-learn": "sklearn",
        "opencv-python": "cv2",
        "Pillow": "PIL",
        "pyyaml": "yaml",
    }
    mod = module_map.get(dist_name, dist_name)
    try:
        m = import_module(mod)
        v = getattr(m, "__version__", None)
        return str(v) if v is not None else "installed (version unknown)"
    except Exception:
        return "not installed"


def torch_details() -> str:
    try:
        import torch  # type: ignore

        lines = []
        lines.append(f"torch: {get_pkg_ver('torch')}")
        lines.append(f"torchvision: {get_pkg_ver('torchvision')}")
        lines.append(f"cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"cuda version (torch): {torch.version.cuda}")
            try:
                n = torch.cuda.device_count()
                lines.append(f"gpu count: {n}")
                if n > 0:
                    lines.append(f"gpu[0]: {torch.cuda.get_device_name(0)}")
            except Exception:
                pass
        return "\n".join(lines)
    except Exception:
        return "torch not available"


def build_report() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"# Environment Report\n\nGenerated: {now}\n\n"

    sys_info = textwrap.dedent(
        f"""\
        ## System

        - OS: {platform.system()} {platform.release()} ({platform.version()})
        - Machine: {platform.machine()}
        - Processor: {platform.processor() or "unknown"}
        - Python: {sys.version.split()[0]}
        - Python executable: {sys.executable}
        """
    )

    # Key packages
    pkg_lines = []
    for p in KEY_PACKAGES:
        pkg_lines.append(f"- {p}: {get_pkg_ver(p)}")
    pkgs = "## Key Packages\n\n" + "\n".join(pkg_lines) + "\n"

    torch_info = "## PyTorch Details\n\n" + "```\n" + torch_details() + "\n```\n"

    hint = textwrap.dedent(
        """\
        ## Notes

        - `requirements.txt` is a runnable dependency list for this repo (the paper does not provide one).
        - For strict reproducibility, you can additionally create a lock file:
          - `pip freeze > requirements.lock.txt`
        """
    )

    return header + sys_info + "\n" + pkgs + "\n" + torch_info + "\n" + hint


def find_repo_root(start: str) -> str:
    """
    Find repo root by walking upward until we find a marker folder.
    Markers: 'configs', 'data', 'src', 'scripts'
    """
    cur = os.path.abspath(start)
    while True:
        markers = ["configs", "data", "src", "scripts"]
        if all(os.path.isdir(os.path.join(cur, m)) for m in markers):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            # fallback: current working dir
            return os.getcwd()
        cur = parent


def main() -> None:
    # Determine repo root (so the output always lands at project root)
    repo_root = find_repo_root(os.getcwd())
    out_path = os.path.join(repo_root, "ENVIRONMENT.md")

    report = build_report()

    # Print to console
    print(report)

    # Write to file
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n[OK] Saved environment report to: {out_path}")


if __name__ == "__main__":
    main()
