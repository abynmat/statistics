"""Chapter 6: Bayesian linear and logistic regression with HMC, checked three ways.

Author: Abyn Scaria. MIT License.

Parts
  1. Linear regression with a normal-inverse-gamma prior: HMC on (beta, log sigma) vs the
     exact conjugate posterior (multivariate Student-t for beta, inverse-gamma for sigma^2).
  2. Posterior predictive checks: the same model on well-specified data and on data with
     heavy-tailed (Student-t, 2 dof) noise. The check uses the kurtosis of OLS residuals.
  3. Logistic regression on a small sample: calibration of the Bayesian posterior predictive
     vs the maximum-likelihood plug-in on held-out data, and interval coverage when the true
     coefficients are drawn from the prior (a light simulation-based calibration).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from scipy import optimize, stats  # noqa: E402
from scipy.special import expit, log_expit  # noqa: E402

from bayeslab import diagnostics as dg  # noqa: E402
from bayeslab import samplers as sp  # noqa: E402
from bayeslab.plotting import C, plt, print_table, save_csv, save_json, savefig  # noqa: E402

SEED = 606
A0, B0, V0 = 2.0, 2.0, 10.0 ** 2          # sigma^2 ~ IG(A0, B0);  beta | sigma^2 ~ N(0, sigma^2 V0 I)


# ---------------------------------------------------------------- linear regression
def linear_logp(X, y):
    n, d = X.shape

    def lpg(q):
        b, ls = q[:d], q[d]
        s2 = np.exp(2 * ls)
        r = y - X @ b
        quad = r @ r + b @ b / V0
        # log IG(s2) + log Jacobian(2 s2) + log N(beta | 0, s2 V0) + log-lik
        lp = -(A0 + 1) * 2 * ls - B0 / s2 + 2 * ls - (d + n) * ls - 0.5 * quad / s2
        g_b = (X.T @ r - b / V0) / s2
        g_ls = -2 * (A0 + 1) + 2 * B0 / s2 + 2 - (d + n) + quad / s2
        return lp, np.append(g_b, g_ls)
    return lpg


def linear_exact(X, y):
    n, d = X.shape
    Vn = np.linalg.inv(np.eye(d) / V0 + X.T @ X)
    mn = Vn @ X.T @ y
    an = A0 + n / 2
    bn = B0 + 0.5 * (y @ y - mn @ np.linalg.solve(Vn, mn))
    nu = 2 * an
    beta_sd = np.sqrt(bn / an * np.diag(Vn) * nu / (nu - 2))
    ig = stats.invgamma(an, scale=bn)
    return {"mn": mn, "Vn": Vn, "an": an, "bn": bn, "beta_sd": beta_sd,
            "s2_mean": ig.mean(), "s2_sd": ig.std()}


def ols_resid_kurtosis(X, Y):
    """Excess kurtosis of OLS residuals; Y can be (n,) or (n, R) replicates."""
    H = X @ np.linalg.pinv(X)
    R = Y - H @ Y
    return stats.kurtosis(R, axis=0)


def ppc(X, y, seed):
    lpg = linear_logp(X, y)
    d = X.shape[1]
    dr, rs = sp.run_chains(sp.hmc, 4, lambda r: np.r_[r.standard_normal(d) * 0.1, 0.0], seed,
                           logp_grad=lpg, n_draws=1000, n_warmup=500, path_length=1.5, adapt_mass=True)
    flat = dr.reshape(-1, d + 1)
    rng = np.random.default_rng(seed)
    idx = rng.choice(flat.shape[0], 2000, replace=False)
    b, sig = flat[idx, :d], np.exp(flat[idx, d])
    yrep = (X @ b.T) + rng.standard_normal((X.shape[0], idx.size)) * sig
    t_obs = float(ols_resid_kurtosis(X, y))
    t_rep = ols_resid_kurtosis(X, yrep)
    # coverage of 90% posterior predictive intervals for the observed points
    lo, hi = np.quantile(yrep, [0.05, 0.95], axis=1)
    return {"t_obs": t_obs, "t_rep": t_rep, "ppp": float(np.mean(t_rep >= t_obs)),
            "pi90_cover": float(np.mean((y >= lo) & (y <= hi))), "draws": dr, "res": rs}


# ---------------------------------------------------------------- logistic regression
def logistic_logp(X, y, prior_sd):
    def lpg(b):
        z = X @ b
        lp = np.sum(y * log_expit(z) + (1 - y) * log_expit(-z)) - 0.5 * b @ b / prior_sd ** 2
        g = X.T @ (y - expit(z)) - b / prior_sd ** 2
        return lp, g
    return lpg


def logistic_mle(X, y):
    def nll(b):
        z = X @ b
        return -np.sum(y * log_expit(z) + (1 - y) * log_expit(-z)), -X.T @ (y - expit(z))
    return optimize.minimize(nll, np.zeros(X.shape[1]), jac=True, method="BFGS").x


def calibration(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    conf, acc, cnt = [], [], []
    for b in range(bins):
        m = idx == b
        if m.sum() > 0:
            conf.append(p[m].mean())
            acc.append(y[m].mean())
            cnt.append(m.sum())
    conf, acc, cnt = map(np.array, (conf, acc, cnt))
    return conf, acc, float(np.sum(cnt * np.abs(conf - acc)) / cnt.sum())


def scores(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return {"log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "brier": float(np.mean((p - y) ** 2)), "ece": calibration(p, y)[2]}


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)

    # ------------------------------------------------------------ 1. linear: HMC vs exact
    n, d = 100, 4
    X = np.column_stack([np.ones(n), rng.standard_normal((n, d - 1))])
    beta_true = np.array([1.0, 2.0, -1.0, 0.5])
    y = X @ beta_true + rng.normal(0, 1.5, n)
    ex = linear_exact(X, y)
    dr, rs = sp.run_chains(sp.hmc, 4, lambda r: np.r_[r.standard_normal(d) * 0.1, 0.0], SEED,
                           logp_grad=linear_logp(X, y), n_draws=2000, n_warmup=1000, path_length=1.5,
                           adapt_mass=True)
    flat = dr.reshape(-1, d + 1)
    s2 = np.exp(2 * flat[:, d])
    lin_rows = []
    for j in range(d):
        x = dr[:, :, j]
        lin_rows.append({"param": f"beta_{j}", "true": beta_true[j], "exact_mean": ex["mn"][j],
                         "hmc_mean": float(x.mean()), "exact_sd": ex["beta_sd"][j], "hmc_sd": float(x.std()),
                         "err_over_mcse": float((x.mean() - ex["mn"][j]) / (x.std() / np.sqrt(dg.ess(x)))),
                         "ess_bulk": dg.ess_bulk(x), "rhat": dg.rank_rhat(x)})
    s2c = np.exp(2 * dr[:, :, d])
    lin_rows.append({"param": "sigma^2", "true": 1.5 ** 2, "exact_mean": ex["s2_mean"], "hmc_mean": float(s2.mean()),
                     "exact_sd": ex["s2_sd"], "hmc_sd": float(s2.std()),
                     "err_over_mcse": float((s2.mean() - ex["s2_mean"]) / (s2.std() / np.sqrt(dg.ess(s2c)))),
                     "ess_bulk": dg.ess_bulk(s2c), "rhat": dg.rank_rhat(s2c)})
    print(f"Linear regression, n={n}: HMC (4x2000) vs exact NIG posterior; "
          f"divergences {sum(r['n_divergent'] for r in rs)}")
    print_table(lin_rows)
    save_csv(lin_rows, "ch06_linear_hmc_vs_exact")

    # ------------------------------------------------------------ 2. posterior predictive checks
    Xp = np.column_stack([np.ones(n), rng.standard_normal((n, d - 1))])
    y_good = Xp @ beta_true + rng.normal(0, 1.5, n)
    y_heavy = Xp @ beta_true + 1.5 * rng.standard_t(2, n) / np.sqrt(2)
    ppc_good = ppc(Xp, y_good, SEED + 1)
    ppc_heavy = ppc(Xp, y_heavy, SEED + 2)
    ppc_rows = [{"data": "gaussian noise (model correct)", "T_obs_kurtosis": ppc_good["t_obs"],
                 "T_rep_median": float(np.median(ppc_good["t_rep"])), "ppp_value": ppc_good["ppp"],
                 "pi90_coverage": ppc_good["pi90_cover"]},
                {"data": "student-t(2) noise (model wrong)", "T_obs_kurtosis": ppc_heavy["t_obs"],
                 "T_rep_median": float(np.median(ppc_heavy["t_rep"])), "ppp_value": ppc_heavy["ppp"],
                 "pi90_coverage": ppc_heavy["pi90_cover"]}]
    print("\nPosterior predictive check, T = excess kurtosis of OLS residuals:")
    print_table(ppc_rows)
    save_csv(ppc_rows, "ch06_ppc")

    # ------------------------------------------------------------ 3. logistic calibration
    n_tr, n_te, k = 60, 5000, 8
    prior_sd = 1.0
    bt = rng.normal(0, prior_sd, k)
    Xtr, Xte = rng.standard_normal((n_tr, k)), rng.standard_normal((n_te, k))
    ytr = (rng.uniform(size=n_tr) < expit(Xtr @ bt)).astype(float)
    yte = (rng.uniform(size=n_te) < expit(Xte @ bt)).astype(float)
    dr, rs = sp.run_chains(sp.hmc, 4, lambda r: r.standard_normal(k) * 0.1, SEED,
                           logp_grad=logistic_logp(Xtr, ytr, prior_sd), n_draws=1500, n_warmup=750,
                           path_length=1.5, adapt_mass=True)
    bd = dr.reshape(-1, k)
    p_bayes = expit(Xte @ bd.T).mean(1)
    b_mle = logistic_mle(Xtr, ytr)
    p_mle = expit(Xte @ b_mle)
    p_map = expit(Xte @ optimize.minimize(lambda b: -logistic_logp(Xtr, ytr, prior_sd)(b)[0], np.zeros(k),
                                          jac=lambda b: -logistic_logp(Xtr, ytr, prior_sd)(b)[1]).x)
    p_true = expit(Xte @ bt)
    cal_rows = [{"predictor": "true coefficients (oracle)", **scores(p_true, yte)},
                {"predictor": "Bayes posterior predictive (HMC)", **scores(p_bayes, yte)},
                {"predictor": "MAP plug-in (same prior)", **scores(p_map, yte)},
                {"predictor": "MLE plug-in", **scores(p_mle, yte)}]
    print(f"\nLogistic regression, n_train={n_tr}, {k} features, held-out n={n_te}:")
    print_table(cal_rows)
    save_csv(cal_rows, "ch06_logistic_calibration")

    # coverage: truth drawn from the prior -> 50/80/90% central intervals should cover at that rate
    levels = [0.5, 0.8, 0.9]
    hits = {lv: [] for lv in levels}
    n_rep = 60
    for rep in range(n_rep):
        r2 = np.random.default_rng([SEED, rep])
        b0 = r2.normal(0, prior_sd, k)
        Xr = r2.standard_normal((n_tr, k))
        yr = (r2.uniform(size=n_tr) < expit(Xr @ b0)).astype(float)
        res = sp.hmc(logistic_logp(Xr, yr, prior_sd), np.zeros(k), 1000, r2, n_warmup=500,
                     path_length=1.5, adapt_mass=True)
        for lv in levels:
            lo, hi = np.quantile(res["draws"], [(1 - lv) / 2, (1 + lv) / 2], axis=0)
            hits[lv].extend(((b0 >= lo) & (b0 <= hi)).tolist())
    cov_rows = [{"nominal": lv, "empirical": float(np.mean(hits[lv])),
                 "binomial_se": float(np.sqrt(lv * (1 - lv) / len(hits[lv]))), "n_intervals": len(hits[lv])}
                for lv in levels]
    print(f"\nCredible-interval coverage, {n_rep} datasets x {k} coefficients drawn from the prior:")
    print_table(cov_rows)
    save_csv(cov_rows, "ch06_logistic_coverage")
    save_json({"beta_true_logistic": bt, "b_mle": b_mle, "max_abs_mle_coef": float(np.abs(b_mle).max())},
              "ch06_summary")

    # ------------------------------------------------------------ figure 1: PPC
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for i, (pc, title) in enumerate(((ppc_good, "Gaussian noise: model is right"),
                                     (ppc_heavy, "t(2) noise: model is wrong"))):
        hi_x = max(pc["t_obs"], np.quantile(pc["t_rep"], 0.999)) * 1.1
        ax[i].hist(pc["t_rep"], bins=60, color=C["sky"], density=True, range=(-1.5, hi_x),
                   label="replicated data T(y_rep)")
        ax[i].axvline(pc["t_obs"], color=C["red"], lw=2, label=f"observed T(y) = {pc['t_obs']:.2f}")
        ax[i].set(title=f"{title}\nposterior predictive p = {pc['ppp']:.3f}", xlabel="residual excess kurtosis")
        ax[i].legend(fontsize=7)
    b1 = ppc_heavy["draws"][:, :, 1].ravel()
    ex_h = linear_exact(Xp, y_heavy)
    ax[2].hist(b1, bins=60, density=True, color=C["sky"], label="HMC beta_1 (t-noise data)")
    bb = np.linspace(b1.min(), b1.max(), 200)
    nu = 2 * ex_h["an"]
    ax[2].plot(bb, stats.t.pdf(bb, nu, ex_h["mn"][1], np.sqrt(ex_h["bn"] / ex_h["an"] * ex_h["Vn"][1, 1])),
               color=C["black"], label="exact conjugate posterior")
    ax[2].axvline(beta_true[1], color=C["green"], ls=":", label="true beta_1")
    ax[2].set(title="Sampler still matches the (wrong) model exactly", xlabel="beta_1")
    ax[2].legend(fontsize=7)
    savefig(fig, "ch06_posterior_predictive_checks")

    # ------------------------------------------------------------ figure 2: calibration
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot([0, 1], [0, 1], color=C["black"], ls="--", lw=1)
    for p, name, col in ((p_bayes, "Bayes posterior predictive", C["blue"]), (p_mle, "MLE plug-in", C["red"]),
                         (p_map, "MAP plug-in", C["orange"])):
        cf, ac, ece = calibration(p, yte)
        ax[0].plot(cf, ac, "o-", color=col, label=f"{name} (ECE {ece:.3f})")
    ax[0].set(xlabel="predicted probability", ylabel="observed frequency",
              title=f"Held-out calibration, n_train={n_tr}, {k} features")
    ax[0].legend(fontsize=7)
    xx = np.arange(len(levels))
    ax[1].bar(xx - 0.2, levels, width=0.4, color=C["grey"], label="nominal")
    ax[1].bar(xx + 0.2, [r["empirical"] for r in cov_rows], width=0.4, color=C["blue"],
              yerr=[2 * r["binomial_se"] for r in cov_rows], capsize=4, label="empirical (+-2 se)")
    ax[1].set(xticks=xx, xticklabels=[f"{int(lv * 100)}%" for lv in levels], ylim=(0, 1),
              title=f"Interval coverage, truth drawn from the prior ({n_rep * k} intervals)")
    ax[1].legend()
    savefig(fig, "ch06_logistic_calibration_and_coverage")
