"""Example 10: double / debiased machine learning in the partially linear model.

Author: Abyn Scaria. MIT License.

Partially linear model (Robinson 1988; Chernozhukov et al. 2018), theta = 0.5:
    X ~ N(0, Sigma), Sigma_jk = 0.5^|j-k|, 10 covariates
    D = m0(X) + V,  V ~ N(0, 1);   Y = theta*D + g0(X) + U,  U ~ N(0, 1)

Two nuisance designs:
  "smooth":      m0 = sin(X1) + 0.5*X2**2
                 g0 = cos(X1)**2 + 0.5*exp(X2/2) + 0.5*X2**2
  "interaction": the same plus 0.5*X3*X4 in m0 and X3*X4 in g0. Products of
                 correlated features are harder for tree learners at n = 2000,
                 so nuisance errors shrink slowly and DML's rate condition is strained.

Both nuisances depend on the same features, so ignoring them confounds D.
Estimators (gradient boosting as the ML learner unless stated):
  1. OLS of Y on D and linear X                 (misspecified nuisance)
  2. Naive "plug-in" ML: estimate g0 by ML, regress Y - g_hat on D
     (non-orthogonal score: first-order sensitive to ML errors in g)
  3. Orthogonalised (partialling-out) but in-sample, no cross-fitting, with an
     overfitting random forest (overfitted nuisances leak into the residuals)
  4. DML: partialling-out + 5-fold cross-fitting  (Chernozhukov et al. 2018)
We also report the coverage of the DML 95% CI built from the orthogonal score.
"""
from common import PALETTE, plt, save_json, savefig  # noqa: I001

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold

THETA = 0.5
P = 10


def simulate(n, rng, design="smooth"):
    cov = 0.5 ** np.abs(np.subtract.outer(np.arange(P), np.arange(P)))
    X = rng.multivariate_normal(np.zeros(P), cov, size=n)
    m0 = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2
    g0 = np.cos(X[:, 0]) ** 2 + 0.5 * np.exp(X[:, 1] / 2) + 0.5 * X[:, 1] ** 2
    if design == "interaction":
        m0 = m0 + 0.5 * X[:, 2] * X[:, 3]
        g0 = g0 + X[:, 2] * X[:, 3]
    D = m0 + rng.normal(size=n)
    Y = THETA * D + g0 + rng.normal(size=n)
    return X, D, Y


def ml(seed):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=20, random_state=seed)


def rf_overfit(seed):
    # deliberately overfits: fully grown trees nearly reproduce the training targets
    return RandomForestRegressor(n_estimators=100, min_samples_leaf=1, max_features=0.5,
                                 random_state=seed, n_jobs=1)


def est_ols(X, D, Y):
    Z = np.column_stack([np.ones(len(D)), D, X])
    return np.linalg.lstsq(Z, Y, rcond=None)[0][1]


def est_plugin(X, D, Y, seed):
    """Non-orthogonal: alternate between fitting g on Y - theta*D and updating theta.
    Uses a 50/50 split (as in the DML paper's illustration) so overfitting is not the
    issue; regularisation bias in g_hat is."""
    idx = np.random.default_rng(seed).permutation(len(Y))
    a, b = idx[: len(Y) // 2], idx[len(Y) // 2:]
    theta = 0.0
    for _ in range(4):
        g = ml(seed).fit(X[a], Y[a] - theta * D[a])
        r = Y[b] - g.predict(X[b])
        theta = (D[b] @ r) / (D[b] @ D[b])
    return theta


def est_insample(X, D, Y, seed):
    """Partialling-out with nuisances fit and evaluated on the same data."""
    ry = Y - rf_overfit(seed).fit(X, Y).predict(X)
    rd = D - rf_overfit(seed).fit(X, D).predict(X)
    return (rd @ ry) / (rd @ rd)


def est_dml(X, D, Y, seed, k=5):
    ry, rd = np.zeros(len(Y)), np.zeros(len(Y))
    for tr, te in KFold(k, shuffle=True, random_state=seed).split(X):
        ry[te] = Y[te] - ml(seed).fit(X[tr], Y[tr]).predict(X[te])
        rd[te] = D[te] - ml(seed).fit(X[tr], D[tr]).predict(X[te])
    theta = (rd @ ry) / (rd @ rd)
    # variance from the orthogonal score psi = (ry - theta*rd)*rd
    psi = (ry - theta * rd) * rd
    se = np.sqrt(np.mean(psi ** 2) / np.mean(rd ** 2) ** 2 / len(Y))
    return theta, se


def one_rep(s, n, design):
    import warnings
    warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
    rng = np.random.default_rng(1000 + s)
    X, D, Y = simulate(n, rng, design)
    th, se = est_dml(X, D, Y, s)
    return {"OLS (linear X)": est_ols(X, D, Y),
            "Plug-in ML (non-orthogonal)": est_plugin(X, D, Y, s),
            "Orthogonal, no cross-fit": est_insample(X, D, Y, s),
            "DML (orthogonal + cross-fit)": th,
            "_se": se}


def run_design(design, n_sims, n):
    reps = Parallel(n_jobs=10)(delayed(one_rep)(s, n, design) for s in range(n_sims))
    names = [k for k in reps[0] if not k.startswith("_")]
    draws = {k: np.array([r[k] for r in reps]) for k in names}
    ses = np.array([r["_se"] for r in reps])
    cover = float(np.mean(np.abs(draws["DML (orthogonal + cross-fit)"] - THETA) < 1.96 * ses))
    summary = {k: {"mean": float(v.mean()), "bias": float(v.mean() - THETA), "sd": float(v.std()),
                   "rmse": float(np.sqrt(np.mean((v - THETA) ** 2)))} for k, v in draws.items()}
    print(f"\n[{design}] true theta = {THETA}  ({n_sims} sims, n={n})")
    for k, v in summary.items():
        print(f"  {k:<32} mean {v['mean']:.3f}  bias {v['bias']:+.3f}  sd {v['sd']:.3f}  rmse {v['rmse']:.3f}")
    print(f"  DML 95% CI coverage: {cover:.3f}  (mean SE {ses.mean():.3f})")
    return names, draws, summary, cover, float(ses.mean())


def main(n_sims=100, n=2000):
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, design in zip(axes, ["smooth", "interaction"]):
        names, draws, summary, cover, mse = run_design(design, n_sims, n)
        out[design] = {"estimators": summary, "dml_ci95_coverage": cover, "dml_mean_se": mse}
        lo = min(v.min() for v in draws.values()) - 0.02
        hi = max(v.max() for v in draws.values()) + 0.02
        bins = np.linspace(lo, hi, 60)
        for i, k in enumerate(names):
            ax.hist(draws[k], bins=bins, alpha=0.55, color=PALETTE[[3, 1, 4, 0][i]],
                    label=f"{k} (bias {summary[k]['bias']:+.3f})")
        ax.axvline(THETA, color="k", ls="--", label="true theta")
        ax.set_xlabel("estimate of theta")
        ax.set_title(f"{design} nuisances: DML CI coverage {cover:.2f}")
        ax.legend(fontsize=7)
    fig.suptitle(f"Double ML in the partially linear model ({n_sims} sims, n={n})")
    savefig(fig, "ex10_dml")
    save_json({"true_theta": THETA, "n": n, "n_sims": n_sims, "designs": out}, "ex10_dml")


if __name__ == "__main__":
    main()
