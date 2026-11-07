"""Example 5: sharp regression discontinuity.

Author: Abyn Scaria. MIT License.

Running variable X ~ Uniform(-1, 1), cutoff c = 0, treatment D = 1[X >= 0].
    Y = 0.5 + 0.8*X + 0.9*X^2 - 0.6*X^3 + 0.4*D + N(0, 0.3^2)
The true jump at the cutoff is tau = 0.4.

We compare:
  * naive difference of means above vs below the cutoff (confounded by X)
  * global polynomial fits of order 1..4 on each side (sensitive to the order)
  * local linear regression with a triangular kernel over a grid of bandwidths
    (the modern default, cf. rdrobust), plus a simple cross-validated bandwidth
  * a placebo cutoff at X = -0.5 where nothing should happen
"""
from common import PALETTE, plt, save_json, savefig  # noqa: I001

import numpy as np

TAU = 0.4


def simulate(n, rng):
    X = rng.uniform(-1, 1, n)
    D = (X >= 0).astype(float)
    Y = 0.5 + 0.8 * X + 0.9 * X ** 2 - 0.6 * X ** 3 + TAU * D + rng.normal(0, 0.3, n)
    return X, D, Y


def wls(y, X, w):
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    resid = y - X @ beta
    # HC0 sandwich variance for weighted least squares
    bread = np.linalg.pinv((X * w[:, None]).T @ X)
    meat = (X * (w * resid)[:, None] ** 2).T @ X
    cov = bread @ meat @ bread
    return beta, np.sqrt(np.diag(cov))


def local_linear(X, Y, h, c=0.0):
    """Local linear RD estimate: separate slopes on each side, triangular kernel."""
    u = (X - c) / h
    w = np.clip(1 - np.abs(u), 0, None)
    keep = w > 0
    x, y, w = X[keep] - c, Y[keep], w[keep]
    d = (x >= 0).astype(float)
    Z = np.column_stack([np.ones_like(x), d, x, d * x])
    beta, se = wls(y, Z, w)
    return beta[1], se[1], int(keep.sum())


def global_poly(X, Y, order, c=0.0):
    x = X - c
    d = (x >= 0).astype(float)
    cols = [np.ones_like(x), d]
    for p in range(1, order + 1):
        cols += [x ** p, d * x ** p]
    beta, *_ = np.linalg.lstsq(np.column_stack(cols), Y, rcond=None)
    return beta[1]


def cv_bandwidth(X, Y, grid, c=0.0):
    """Ludwig-Miller style CV: predict each point near the cutoff from a one-sided
    local linear fit using only points farther from the cutoff on the same side."""
    scores = []
    near = np.abs(X - c) < 0.5
    idx = np.where(near)[0]
    sub = idx[:: max(1, len(idx) // 400)]  # subsample to keep it fast
    for h in grid:
        err = []
        for i in sub:
            xi = X[i]
            side = (X >= c) if xi >= c else (X < c)
            if xi >= c:
                mask = side & (X > xi) & (X <= xi + h)
            else:
                mask = side & (X < xi) & (X >= xi - h)
            if mask.sum() < 10:
                continue
            Z = np.column_stack([np.ones(mask.sum()), X[mask] - xi])
            b, *_ = np.linalg.lstsq(Z, Y[mask], rcond=None)
            err.append((Y[i] - b[0]) ** 2)
        scores.append(np.mean(err))
    return grid[int(np.argmin(scores))], scores


def main(n=4000):
    rng = np.random.default_rng(5)
    X, D, Y = simulate(n, rng)
    naive = Y[D == 1].mean() - Y[D == 0].mean()
    polys = {p: global_poly(X, Y, p) for p in (1, 2, 3, 4)}
    hs = np.round(np.linspace(0.05, 1.0, 20), 3)
    ll = [local_linear(X, Y, h) for h in hs]
    h_cv, _ = cv_bandwidth(X, Y, hs[hs >= 0.1])
    est_cv, se_cv, n_cv = local_linear(X, Y, h_cv)
    placebo, placebo_se, _ = local_linear(X[X < 0], Y[X < 0], 0.3, c=-0.5)

    # Monte Carlo of the CV-bandwidth local-linear estimator and the linear global fit
    mc_ll, mc_lin, cover = [], [], 0
    rng2 = np.random.default_rng(55)
    for _ in range(300):
        Xs, _, Ys = simulate(n, rng2)
        e, s, _ = local_linear(Xs, Ys, h_cv)
        mc_ll.append(e)
        cover += abs(e - TAU) < 1.96 * s
        mc_lin.append(global_poly(Xs, Ys, 1))

    res = {"true_tau": TAU, "n": n, "naive_diff": naive,
           "global_poly": {str(k): v for k, v in polys.items()},
           "local_linear_by_h": {str(h): {"est": e, "se": s, "n_eff": m} for h, (e, s, m) in zip(hs, ll)},
           "cv_bandwidth": float(h_cv), "local_linear_cv": {"est": est_cv, "se": se_cv, "n_used": n_cv},
           "placebo_cutoff_-0.5": {"est": placebo, "se": placebo_se},
           "mc_300": {"local_linear_mean": float(np.mean(mc_ll)), "local_linear_sd": float(np.std(mc_ll)),
                      "ci95_coverage": cover / 300,
                      "global_linear_mean": float(np.mean(mc_lin)), "global_linear_sd": float(np.std(mc_lin))}}
    print(f"naive {naive:.3f} | global poly {polys} | CV h={h_cv} LL={est_cv:.3f} (se {se_cv:.3f})")
    print(f"placebo at -0.5: {placebo:.3f} (se {placebo_se:.3f}) | MC: {res['mc_300']}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    bins = np.linspace(-1, 1, 41)
    centers = 0.5 * (bins[1:] + bins[:-1])
    means = [Y[(X >= a) & (X < b)].mean() for a, b in zip(bins[:-1], bins[1:])]
    axes[0].scatter(X, Y, s=2, alpha=0.15, color="grey")
    axes[0].scatter(centers, means, color=PALETTE[0], s=18, zorder=3, label="bin means")
    for side in (-1, 1):
        m = (X < 0) if side < 0 else (X >= 0)
        msk = m & (np.abs(X) <= h_cv)
        b = np.polyfit(X[msk], Y[msk], 1)
        xs = np.linspace(0, side * h_cv, 50)
        axes[0].plot(xs, np.polyval(b, xs), color=PALETTE[3], lw=2.5,
                     label="local linear fit (CV bandwidth)" if side > 0 else None)
    axes[0].axvline(0, color="k", ls="--", lw=1)
    axes[0].set_xlabel("running variable X (cutoff at 0)")
    axes[0].set_ylabel("Y")
    axes[0].set_title(f"Sharp RDD: estimated jump {est_cv:.3f} vs true {TAU}")
    axes[0].legend(fontsize=8)

    e = np.array([r[0] for r in ll])
    s = np.array([r[1] for r in ll])
    axes[1].fill_between(hs, e - 1.96 * s, e + 1.96 * s, alpha=0.25, color=PALETTE[0])
    axes[1].plot(hs, e, "o-", color=PALETTE[0], label="local linear, 95% CI")
    for p, v in polys.items():
        axes[1].scatter([1.05], [v], marker="s", color=PALETTE[p], zorder=3, label=f"global poly order {p}")
    axes[1].axhline(TAU, color="k", ls="--", label="truth")
    axes[1].axvline(h_cv, color=PALETTE[3], ls=":", label=f"CV bandwidth {h_cv}")
    axes[1].set_xlabel("bandwidth h")
    axes[1].set_ylabel("estimated jump")
    axes[1].set_title("Bandwidth sensitivity")
    axes[1].legend(fontsize=7, ncol=2)
    savefig(fig, "ex05_rdd")
    save_json(res, "ex05_rdd")


if __name__ == "__main__":
    main()
