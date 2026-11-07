"""Shared figure style and I/O helpers for the chapter scripts.

Author: Abyn Scaria. MIT License.
"""
import json
import os
import warnings
from pathlib import Path

# Small numpy problems: one BLAS thread is faster and keeps timings comparable across samplers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
# numpy 2.0 + Apple Accelerate can emit spurious matmul warnings on finite inputs.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

# Okabe-Ito colour-blind-safe palette.
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
     "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "black": "#000000",
     "grey": "#7F7F7F"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 9,
    "axes.titlesize": 10, "legend.fontsize": 8, "legend.frameon": False,
})


def savefig(fig, name):
    path = FIG_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure  -> figures/{path.name}")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def save_json(obj, name):
    path = RES_DIR / f"{name}.json"
    path.write_text(json.dumps(_clean(obj), indent=2))
    print(f"  results -> results/{path.name}")


def save_csv(rows, name):
    """rows: list of dicts with identical keys."""
    path = RES_DIR / f"{name}.csv"
    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for r in rows:
        vals = []
        for k in keys:
            v = r[k]
            vals.append(f"{v:.6g}" if isinstance(v, (float, np.floating)) else str(v))
        lines.append(",".join(vals))
    path.write_text("\n".join(lines) + "\n")
    print(f"  results -> results/{path.name}")


def print_table(rows, cols=None, fmt="{:.4g}"):
    """Print rows (list of dicts) as a markdown table, so README tables can be pasted from logs."""
    cols = cols or list(rows[0].keys())
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            cells.append(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
        print("| " + " | ".join(cells) + " |")
