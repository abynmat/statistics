"""Example 9: heterogeneous treatment effects. Meta-learners vs an honest causal forest.

Author: Abyn Scaria. MIT License.

Observational data with confounding and a CATE that has a step and a slope:
    X ~ N(0, I_6)
    e(X)   = sigmoid(0.7*X1 - 0.5*X2 + 0.3*X3)
    mu0(X) = 1 + X1 + 0.8*X2**2 + sin(2*X3) + 0.5*X4
    tau(X) = 1 + 2*1[X1 > 0] + 0.75*X2
    Y = mu0(X) + tau(X)*T + N(0, 1)

Learners, all on scikit-learn HistGradientBoosting base models:
  S-learner  one model of Y on (X, T), tau = f(x,1) - f(x,0)
  T-learner  separate outcome models per arm
  X-learner  Kunzel et al. (2019): impute individual effects, regress, blend with e(x)
  DR-learner regress the cross-fitted AIPW pseudo-outcome on X
  R-learner  Nie & Wager (2021): minimise the residual-on-residual (R-) loss
  Honest causal forest (from scratch): trees grown on the R-loss with local
             centring (as in GRF), leaves re-estimated on a held-out half (honesty),
             averaged over subsampled trees.

Metric: PEHE = sqrt(mean (tau_hat(x) - tau(x))^2) on a fresh test set,
averaged over 5 replications. Also an uplift (targeting) curve.
"""
from common import PALETTE, plt, save_json, savefig, sigmoid  # noqa: I001

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

D = 6


def simulate(n, rng):
    X = rng.normal(size=(n, D))
    e = sigmoid(0.7 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2])
    T = rng.binomial(1, e)
    mu0 = 1 + X[:, 0] + 0.8 * X[:, 1] ** 2 + np.sin(2 * X[:, 2]) + 0.5 * X[:, 3]
    tau = 1 + 2 * (X[:, 0] > 0) + 0.75 * X[:, 1]
    Y = mu0 + tau * T + rng.normal(size=n)
    return X, T, Y, tau, e


def reg(seed=0):
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=20, random_state=seed)


def clf(seed=0):
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=20, random_state=seed)


def s_learner(X, T, Y, Xt):
    m = reg().fit(np.column_stack([X, T]), Y)
    return m.predict(np.column_stack([Xt, np.ones(len(Xt))])) - m.predict(np.column_stack([Xt, np.zeros(len(Xt))]))


def t_learner(X, T, Y, Xt):
    m1 = reg().fit(X[T == 1], Y[T == 1])
    m0 = reg().fit(X[T == 0], Y[T == 0])
    return m1.predict(Xt) - m0.predict(Xt)


def x_learner(X, T, Y, Xt):
    m1 = reg().fit(X[T == 1], Y[T == 1])
    m0 = reg().fit(X[T == 0], Y[T == 0])
    d1 = Y[T == 1] - m0.predict(X[T == 1])   # imputed effect for treated
    d0 = m1.predict(X[T == 0]) - Y[T == 0]   # imputed effect for controls
    t1 = reg().fit(X[T == 1], d1)
    t0 = reg().fit(X[T == 0], d0)
    g = np.clip(clf().fit(X, T).predict_proba(Xt)[:, 1], 0.02, 0.98)
    return g * t0.predict(Xt) + (1 - g) * t1.predict(Xt)


def crossfit_nuisances(X, T, Y, k=5, seed=0):
    """Out-of-fold m(x)=E[Y|X], e(x)=P(T=1|X), mu0(x), mu1(x)."""
    n = len(Y)
    m, e, mu0, mu1 = (np.zeros(n) for _ in range(4))
    for tr, te in KFold(k, shuffle=True, random_state=seed).split(X):
        m[te] = reg().fit(X[tr], Y[tr]).predict(X[te])
        e[te] = clf().fit(X[tr], T[tr]).predict_proba(X[te])[:, 1]
        Tt, Yt = T[tr], Y[tr]
        mu1[te] = reg().fit(X[tr][Tt == 1], Yt[Tt == 1]).predict(X[te])
        mu0[te] = reg().fit(X[tr][Tt == 0], Yt[Tt == 0]).predict(X[te])
    return m, np.clip(e, 0.02, 0.98), mu0, mu1


def dr_learner(X, T, Y, Xt, nuis):
    _, e, mu0, mu1 = nuis
    psi = mu1 - mu0 + T * (Y - mu1) / e - (1 - T) * (Y - mu0) / (1 - e)
    return reg().fit(X, psi).predict(Xt)


def r_learner(X, T, Y, Xt, nuis):
    m, e, _, _ = nuis
    yr, tr = Y - m, T - e
    # min sum (yr - tau(x) tr)^2  ==  weighted regression of yr/tr on X with weights tr^2
    return reg().fit(X, yr / tr, sample_weight=tr ** 2).predict(Xt)


class HonestCausalForest:
    """A small, readable stand-in for a GRF causal forest.

    For each tree: draw a 50% subsample, split it in two halves. On half A, grow a
    regression tree that minimises the R-loss (target yr/tr, weights tr^2), which
    picks splits that separate units with different treatment effects. On half B,
    recompute each leaf's effect as sum(tr*yr)/sum(tr^2) (honest estimation).
    Predictions average the leaf effects over trees.
    """

    def __init__(self, n_trees=300, min_leaf=30, max_features=0.7, seed=0):
        self.n_trees, self.min_leaf, self.max_features, self.seed = n_trees, min_leaf, max_features, seed

    def fit(self, X, yr, tr):
        rng = np.random.default_rng(self.seed)
        n = len(yr)
        self.trees = []
        for b in range(self.n_trees):
            idx = rng.choice(n, n // 2, replace=False)
            A, B = idx[: len(idx) // 2], idx[len(idx) // 2:]
            tree = DecisionTreeRegressor(min_samples_leaf=self.min_leaf, max_features=self.max_features,
                                         random_state=int(rng.integers(1e9)))
            tree.fit(X[A], yr[A] / tr[A], sample_weight=tr[A] ** 2)
            leaves = tree.apply(X[B])
            num = np.bincount(leaves, weights=tr[B] * yr[B], minlength=tree.tree_.node_count)
            den = np.bincount(leaves, weights=tr[B] ** 2, minlength=tree.tree_.node_count)
            fallback = tree.tree_.value[:, 0, 0]  # half-A estimate if a leaf got no half-B units
            leaf_tau = np.where(den > 1e-8, num / np.maximum(den, 1e-12), fallback)
            self.trees.append((tree, leaf_tau))
        return self

    def predict(self, X):
        return np.mean([lt[t.apply(X)] for t, lt in self.trees], axis=0)


def main(n_train=4000, n_test=2000, reps=5):
    rng = np.random.default_rng(9)
    names = ["S-learner", "T-learner", "X-learner", "DR-learner", "R-learner", "Honest causal forest"]
    pehe = {k: [] for k in names}
    ate = {k: [] for k in names}
    keep = None
    for r in range(reps):
        X, T, Y, tau, _ = simulate(n_train, rng)
        Xt, _, _, tau_t, _ = simulate(n_test, rng)
        nuis = crossfit_nuisances(X, T, Y, seed=r)
        m, e, _, _ = nuis
        preds = {
            "S-learner": s_learner(X, T, Y, Xt),
            "T-learner": t_learner(X, T, Y, Xt),
            "X-learner": x_learner(X, T, Y, Xt),
            "DR-learner": dr_learner(X, T, Y, Xt, nuis),
            "R-learner": r_learner(X, T, Y, Xt, nuis),
            "Honest causal forest": HonestCausalForest(seed=r).fit(X, Y - m, T - e).predict(Xt),
        }
        for k, p in preds.items():
            pehe[k].append(float(np.sqrt(np.mean((p - tau_t) ** 2))))
            ate[k].append(float(p.mean() - tau_t.mean()))
        if keep is None:
            keep = (Xt, tau_t, preds)
        print(f"rep {r}: " + "  ".join(f"{k}={pehe[k][-1]:.3f}" for k in names))
    summary = {k: {"pehe_mean": float(np.mean(pehe[k])), "pehe_sd": float(np.std(pehe[k])),
                   "ate_bias_mean": float(np.mean(ate[k]))} for k in names}
    print("\nPEHE (lower is better), mean over reps:")
    for k, v in summary.items():
        print(f"  {k:<22} {v['pehe_mean']:.3f} +- {v['pehe_sd']:.3f}   ATE bias {v['ate_bias_mean']:+.3f}")

    Xt, tau_t, preds = keep
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), sharex=True, sharey=True)
    for ax, (k, p) in zip(axes.ravel(), preds.items()):
        ax.scatter(tau_t, p, s=3, alpha=0.3, color=PALETTE[names.index(k) % len(PALETTE)])
        lim = [tau_t.min() - 0.5, tau_t.max() + 0.5]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_title(f"{k}\nPEHE {np.sqrt(np.mean((p - tau_t) ** 2)):.3f}", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("true tau(x)")
    for ax in axes[:, 0]:
        ax.set_ylabel("estimated tau(x)")
    fig.suptitle("CATE estimates on a held-out test set (one replication)")
    savefig(fig, "ex09_cate_scatter")

    # Uplift / targeting curve: treat the top-q fraction ranked by predicted tau
    fig, ax = plt.subplots(figsize=(7, 4.2))
    qs = np.linspace(0, 1, 101)
    order_oracle = np.argsort(-tau_t)

    def gain(order):
        cs = np.concatenate([[0], np.cumsum(tau_t[order])]) / len(tau_t)
        return cs[(qs * len(tau_t)).astype(int)]

    auuc = {}
    ax.plot(qs, gain(order_oracle), "k-", lw=2, label="oracle ranking")
    for k in ["S-learner", "T-learner", "R-learner", "Honest causal forest"]:
        gk = gain(np.argsort(-preds[k]))
        auuc[k] = float(np.trapz(gk, qs))
        ax.plot(qs, gk, lw=1.5, color=PALETTE[names.index(k)], label=f"{k} (area {auuc[k]:.3f})")
    ax.plot(qs, qs * tau_t.mean(), "k:", label="random targeting")
    auuc["oracle"] = float(np.trapz(gain(order_oracle), qs))
    auuc["random"] = float(np.trapz(qs * tau_t.mean(), qs))
    ax.set_xlabel("fraction of population treated (ranked by predicted effect)")
    ax.set_ylabel("total true effect gained / n")
    ax.set_title("Uplift curve: how well does each CATE model rank units?")
    ax.legend(fontsize=8)
    savefig(fig, "ex09_uplift_curve")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(names, [summary[k]["pehe_mean"] for k in names], xerr=[summary[k]["pehe_sd"] for k in names],
            color=PALETTE[: len(names)], capsize=4)
    ax.invert_yaxis()
    ax.set_xlabel(f"PEHE on test set (mean +- sd over {reps} replications)")
    ax.set_title("CATE accuracy by learner")
    savefig(fig, "ex09_pehe_bar")
    save_json({"n_train": n_train, "n_test": n_test, "reps": reps, "summary": summary,
               "auuc_one_rep": auuc}, "ex09_hte_metalearners")


if __name__ == "__main__":
    main()
