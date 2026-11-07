"""Shared helpers for the causal-inference examples.

Author: Abyn Scaria. MIT License.

Everything here is plain numpy / scikit-learn / matplotlib so the examples run
without any causal-inference library installed.
"""
import json
import os
import warnings
from pathlib import Path

# Small models on small data: one thread is far faster than fighting over 14 cores.
# Must be set before numpy / sklearn import their thread pools.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

# numpy 2.0 + Apple Accelerate emits spurious "divide by zero in matmul" warnings
# on valid, finite inputs (a known upstream issue). The results are correct; hide the noise.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

# A small, colour-blind-friendly palette (Okabe-Ito) used across all figures.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


def savefig(fig, name):
    """Save a matplotlib figure to figures/<name>.png and close it."""
    path = FIG_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figure -> figures/{path.name}")


def save_json(obj, name):
    """Dump a dict of results to results/<name>.json (floats rounded to 4 dp)."""

    def _round(x):
        if isinstance(x, dict):
            return {k: _round(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_round(v) for v in x]
        if isinstance(x, (float, np.floating)):
            return round(float(x), 4)
        if isinstance(x, (np.integer,)):
            return int(x)
        return x

    path = RES_DIR / f"{name}.json"
    path.write_text(json.dumps(_round(obj), indent=2))
    print(f"  saved results -> results/{path.name}")


def ols(y, X, add_const=True):
    """Ordinary least squares with classical (homoskedastic) standard errors.

    Returns (beta, se). If add_const, a column of ones is prepended and
    beta[0] is the intercept.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if add_const:
        X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    sigma2 = resid @ resid / (n - k)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    return beta, np.sqrt(np.diag(cov))


def ols_robust(y, X, add_const=True):
    """OLS with HC1 heteroskedasticity-robust standard errors."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if add_const:
        X = np.column_stack([np.ones(len(X)), X])
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    n, k = X.shape
    meat = (X * resid[:, None] ** 2).T @ X
    cov = XtX_inv @ meat @ XtX_inv * n / (n - k)
    return beta, np.sqrt(np.diag(cov))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def summarize(estimates, truth):
    """Bias, standard deviation and RMSE of a vector of simulation estimates."""
    e = np.asarray(estimates, dtype=float)
    return {
        "mean": float(e.mean()),
        "bias": float(e.mean() - truth),
        "sd": float(e.std(ddof=1)) if len(e) > 1 else 0.0,
        "rmse": float(np.sqrt(np.mean((e - truth) ** 2))),
    }


def print_table(rows, truth=None, title=None):
    """Pretty-print {name: summary-dict} as a fixed-width table."""
    if title:
        print(f"\n{title}")
    if truth is not None:
        print(f"  true effect = {truth:.3f}")
    print(f"  {'estimator':<34}{'mean':>9}{'bias':>9}{'sd':>9}{'rmse':>9}")
    for name, s in rows.items():
        print(f"  {name:<34}{s['mean']:>9.3f}{s['bias']:>9.3f}{s['sd']:>9.3f}{s['rmse']:>9.3f}")


def d_separated(G, x, y, z):
    """networkx renamed d_separated -> is_d_separator in 3.3 and removed the old name in 3.5."""
    import networkx as nx
    fn = getattr(nx, "is_d_separator", None) or getattr(nx, "d_separated")
    return fn(G, set(x), set(y), set(z))
