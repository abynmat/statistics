"""Example 1: confounding bias and four ways to adjust for it.

Author: Abyn Scaria. MIT License.

Data-generating process (true ATE = 2.0):
    X1, X2, X3 ~ N(0, 1)                               observed confounders
    P(T=1 | X) = sigmoid(-0.5 + 0.9*X1 + 0.9*X2 - 0.4*X3)
    tau(X)     = 2 + 1.5*X2                            heterogeneous effect, mean 2
    Y = tau(X)*T + 1.0*X1 + 1.5*exp(0.7*X2) + 0.5*X3 + N(0, 1)

The propensity model is correctly specified as a logistic regression on X.
The outcome is non-linear in X2 and the effect varies with X2, so a single
regression of Y on T and linear X is misspecified (it also weights units by
e(X)(1-e(X)) instead of equally, so it does not target the ATE). That lets us show the difference between estimators that lean on
the outcome model (OLS), the treatment model (IPW), or both (AIPW).

Estimators compared over 200 simulated datasets of n = 2000:
  * naive difference in means        (ignores confounding)
  * OLS of Y on T and linear X       (outcome model, misspecified)
  * IPW with logistic propensity     (treatment model, correct)
  * AIPW, linear outcome + logistic  (doubly robust: one model right is enough)
  * AIPW, boosted outcome + logistic, cross-fitted
"""
from common import PALETTE, plt, print_table, save_json, savefig, sigmoid, summarize, ols  # noqa: I001 (sets threads first)

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold

TRUE_ATE = 2.0


def simulate(n, rng):
    X = rng.normal(size=(n, 3))
    ps = sigmoid(-0.5 + 0.9 * X[:, 0] + 0.9 * X[:, 1] - 0.4 * X[:, 2])
    T = rng.binomial(1, ps)
    tau = TRUE_ATE + 1.5 * X[:, 1]
    Y = tau * T + 1.0 * X[:, 0] + 1.5 * np.exp(0.7 * X[:, 1]) + 0.5 * X[:, 2] + rng.normal(size=n)
    return X, T, Y, ps


def naive(X, T, Y):
    return Y[T == 1].mean() - Y[T == 0].mean()


def ols_adjust(X, T, Y):
    beta, _ = ols(Y, np.column_stack([T, X]))
    return beta[1]


def fit_propensity(X, T):
    # Large C -> essentially unpenalised logistic regression.
    m = LogisticRegression(C=1e6, max_iter=1000).fit(X, T)
    return np.clip(m.predict_proba(X)[:, 1], 0.01, 0.99)


def ipw(X, T, Y):
    e = fit_propensity(X, T)
    # Normalised (Hajek) IPW: weights sum to one within each arm, lower variance.
    w1, w0 = T / e, (1 - T) / (1 - e)
    return np.sum(w1 * Y) / np.sum(w1) - np.sum(w0 * Y) / np.sum(w0)


def aipw_from_nuisances(T, Y, e, mu0, mu1):
    """Augmented IPW: outcome-model prediction plus an IPW correction of its residual."""
    psi = mu1 - mu0 + T * (Y - mu1) / e - (1 - T) * (Y - mu0) / (1 - e)
    return psi.mean(), psi.std(ddof=1) / np.sqrt(len(psi))


def aipw_linear(X, T, Y):
    e = fit_propensity(X, T)
    m1 = LinearRegression().fit(X[T == 1], Y[T == 1])
    m0 = LinearRegression().fit(X[T == 0], Y[T == 0])
    return aipw_from_nuisances(T, Y, e, m0.predict(X), m1.predict(X))[0]


def aipw_boosted_crossfit(X, T, Y, seed=0, k=2):
    """Cross-fitting: nuisances for fold k are trained on the other folds only."""
    n = len(Y)
    e, mu0, mu1 = np.zeros(n), np.zeros(n), np.zeros(n)
    for train, test in KFold(k, shuffle=True, random_state=seed).split(X):
        Xt, Tt, Yt = X[train], T[train], Y[train]
        ps = LogisticRegression(C=1e6, max_iter=1000).fit(Xt, Tt)
        e[test] = np.clip(ps.predict_proba(X[test])[:, 1], 0.01, 0.99)
        g1 = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, random_state=seed)
        g0 = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, random_state=seed)
        mu1[test] = g1.fit(Xt[Tt == 1], Yt[Tt == 1]).predict(X[test])
        mu0[test] = g0.fit(Xt[Tt == 0], Yt[Tt == 0]).predict(X[test])
    return aipw_from_nuisances(T, Y, e, mu0, mu1)[0]


def main(n_sims=200, n=2000):
    rng = np.random.default_rng(42)
    estimators = {
        "Naive difference in means": naive,
        "OLS, linear in X": ols_adjust,
        "IPW (logistic propensity)": ipw,
        "AIPW (linear outcome)": aipw_linear,
        "AIPW (boosted, cross-fit)": aipw_boosted_crossfit,
    }
    draws = {k: [] for k in estimators}
    for s in range(n_sims):
        X, T, Y, _ = simulate(n, rng)
        for name, f in estimators.items():
            draws[name].append(f(X, T, Y))
    summary = {k: summarize(v, TRUE_ATE) for k, v in draws.items()}
    print_table(summary, TRUE_ATE, f"Example 1: confounding ({n_sims} sims, n={n})")

    # Population ATE / ATT / ATC from one very large draw (effects vary with X2).
    Xb, Tb, _, _ = simulate(2_000_000, np.random.default_rng(0))
    tau_b = TRUE_ATE + 1.5 * Xb[:, 1]
    estimands = {"ATE": float(tau_b.mean()), "ATT": float(tau_b[Tb == 1].mean()),
                 "ATC": float(tau_b[Tb == 0].mean())}
    print(f"  population estimands (n=2e6): {estimands}")

    # Overlap diagnostic on one dataset.
    X, T, Y, ps_true = simulate(5000, np.random.default_rng(7))
    e_hat = fit_propensity(X, T)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = list(summary)
    biases = [summary[k]["bias"] for k in names]
    sds = [summary[k]["sd"] for k in names]
    axes[0].barh(names, biases, xerr=sds, color=PALETTE[: len(names)], capsize=4)
    axes[0].axvline(0, color="k", lw=1)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("bias = mean estimate - true ATE  (error bar = 1 sd across sims)")
    axes[0].set_title(f"Estimator bias, true ATE = {TRUE_ATE}")
    bins = np.linspace(0, 1, 41)
    axes[1].hist(e_hat[T == 1], bins=bins, alpha=0.6, label="treated", color=PALETTE[0], density=True)
    axes[1].hist(e_hat[T == 0], bins=bins, alpha=0.6, label="control", color=PALETTE[1], density=True)
    axes[1].set_xlabel("estimated propensity score e(X)")
    axes[1].set_title("Overlap check")
    axes[1].legend()
    savefig(fig, "ex01_confounding_bias")

    save_json({"true_ate": TRUE_ATE, "population_estimands": estimands, "n": n, "n_sims": n_sims,
               "estimators": summary,
               "propensity_range": [float(e_hat.min()), float(e_hat.max())]}, "ex01_confounding")


if __name__ == "__main__":
    main()
