"""Chapter 9: WAIC and PSIS-LOO, scored against exact LOO and the true out-of-sample log score.

Author: Abyn Scaria. MIT License.

Setup: x ~ U(-1, 1), y = sin(pi x) + N(0, 0.3^2), n = 30. Candidate models are Bayesian
polynomial regressions of degree 0..8 with a conjugate normal-inverse-gamma prior, so:
  * posterior draws are exact (no MCMC error to confuse the comparison),
  * the exact leave-one-out predictive is a closed-form Student-t (refit without point i),
  * the true expected log predictive density (elpd) of each fitted model can be computed on
    a large fresh test set from the real data-generating process.
That lets each estimator be scored against two references: exact LOO (what it approximates)
and the true elpd (what we actually care about).

Also: a planted outlier to show what Pareto k-hat flags, and the GPD shape estimator checked
on draws with a known tail index.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from bayeslab import model_compare as mc  # noqa: E402
from bayeslab.plotting import C, plt, print_table, save_csv, save_json, savefig  # noqa: E402

SEED = 909
A0, B0, V0 = 2.0, 0.5, 4.0
NOISE = 0.3
DEGREES = list(range(9))
S_DRAWS = 4000


def design(x, deg):
    return np.vander(x, deg + 1, increasing=True)


def nig_post(X, y):
    d = X.shape[1]
    Vn = np.linalg.inv(np.eye(d) / V0 + X.T @ X)
    mn = Vn @ X.T @ y
    an = A0 + y.size / 2
    bn = B0 + 0.5 * (y @ y - mn @ np.linalg.solve(Vn, mn))
    return mn, Vn, an, bn


def predictive_logpdf(Xnew, ynew, post):
    """Exact posterior predictive: Student-t_{2an}(x'mn, bn/an (1 + x'Vn x))."""
    mn, Vn, an, bn = post
    scale = np.sqrt(bn / an * (1 + np.einsum("ij,jk,ik->i", Xnew, Vn, Xnew)))
    return stats.t.logpdf(ynew, 2 * an, Xnew @ mn, scale)


def draws_loglik(X, y, post, rng):
    """Exact posterior draws -> pointwise log-likelihood matrix (S, n)."""
    mn, Vn, an, bn = post
    s2 = stats.invgamma.rvs(an, scale=bn, size=S_DRAWS, random_state=rng)
    L = np.linalg.cholesky(Vn)
    beta = mn + (rng.standard_normal((S_DRAWS, mn.size)) @ L.T) * np.sqrt(s2)[:, None]
    mu = beta @ X.T
    return stats.norm.logpdf(y[None, :], mu, np.sqrt(s2)[:, None])


def exact_loo(X, y):
    out = np.empty(y.size)
    for i in range(y.size):
        m = np.ones(y.size, bool)
        m[i] = False
        out[i] = predictive_logpdf(X[i:i + 1], y[i:i + 1], nig_post(X[m], y[m]))[0]
    return out


def evaluate_dataset(x, y, xt, yt, rng):
    rows = []
    for deg in DEGREES:
        X = design(x, deg)
        post = nig_post(X, y)
        ll = draws_loglik(X, y, post, rng)
        w = mc.waic(ll)
        lo = mc.psis_loo(ll)
        ex = exact_loo(X, y)
        true_elpd = y.size * float(np.mean(predictive_logpdf(design(xt, deg), yt, post)))
        lppd = float(np.sum(np.log(np.mean(np.exp(ll), 0))))
        rows.append({"degree": deg, "lppd_in_sample": lppd, "waic": w["elpd"], "psis_loo": lo["elpd"],
                     "is_loo": mc.is_loo(ll)["elpd"], "exact_loo": float(ex.sum()), "true_elpd": true_elpd,
                     "p_waic": w["p"], "p_loo": lo["p"], "max_khat": float(lo["khat"].max()),
                     "loo_se": lo["se"]})
    return rows


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    n = 30
    xt = rng.uniform(-1, 1, 20000)
    yt = np.sin(np.pi * xt) + rng.normal(0, NOISE, xt.size)

    # ------------------------------------------------------------ one dataset in detail
    x = rng.uniform(-1, 1, n)
    y = np.sin(np.pi * x) + rng.normal(0, NOISE, n)
    one = evaluate_dataset(x, y, xt, yt, rng)
    print(f"One dataset (n={n}); all elpd values on the sum-over-{n}-points scale:")
    print_table(one)
    save_csv(one, "ch09_one_dataset")

    # ------------------------------------------------------------ repeated datasets
    reps = 100
    err = {k: [] for k in ("lppd_in_sample", "waic", "psis_loo", "is_loo", "exact_loo")}
    err_vs_loo = {k: [] for k in ("waic", "psis_loo", "is_loo")}
    picks = {k: [] for k in ("lppd_in_sample", "waic", "psis_loo", "is_loo", "exact_loo", "true_elpd")}
    regret = {k: [] for k in picks}
    for rep in range(reps):
        r2 = np.random.default_rng([SEED, rep])
        xr = r2.uniform(-1, 1, n)
        yr = np.sin(np.pi * xr) + r2.normal(0, NOISE, n)
        rows = evaluate_dataset(xr, yr, xt, yt, r2)
        true = np.array([r["true_elpd"] for r in rows])
        for k in err:
            err[k].extend((np.array([r[k] for r in rows]) - true).tolist())
        for k in err_vs_loo:
            err_vs_loo[k].extend((np.array([r[k] for r in rows]) - np.array([r["exact_loo"] for r in rows])).tolist())
        for k in picks:
            j = int(np.argmax([r[k] for r in rows]))
            picks[k].append(DEGREES[j])
            regret[k].append(float(true.max() - true[j]))
    rep_rows = []
    for k in err:
        e = np.array(err[k])
        rep_rows.append({"estimator": k, "mean_error_vs_true_elpd": float(e.mean()),
                         "rmse_vs_true_elpd": float(np.sqrt(np.mean(e ** 2))),
                         "rmse_vs_exact_loo": float(np.sqrt(np.mean(np.array(err_vs_loo[k]) ** 2)))
                         if k in err_vs_loo else (0.0 if k == "exact_loo" else float("nan")),
                         "mean_regret_of_pick": float(np.mean(regret[k])),
                         "modal_degree_picked": int(np.bincount(picks[k]).argmax())})
    rep_rows.append({"estimator": "true_elpd (oracle)", "mean_error_vs_true_elpd": 0.0, "rmse_vs_true_elpd": 0.0,
                     "rmse_vs_exact_loo": float("nan"), "mean_regret_of_pick": 0.0,
                     "modal_degree_picked": int(np.bincount(picks["true_elpd"]).argmax())})
    print(f"\n{reps} datasets x {len(DEGREES)} models: estimator error and model-selection regret")
    print_table(rep_rows)
    save_csv(rep_rows, "ch09_repeated_datasets")

    # ------------------------------------------------------------ outlier + k-hat
    yo = y.copy()
    i_out = int(np.argmax(x))                 # an outlier at the edge of the design: high leverage
    yo[i_out] += 2.5
    X5 = design(x, 5)
    post5 = nig_post(X5, yo)
    ll5 = mc.psis_loo(draws_loglik(X5, yo, post5, rng))
    ll5_is = mc.is_loo(draws_loglik(X5, yo, post5, rng))
    ex5 = exact_loo(X5, yo)
    w5 = mc.waic(draws_loglik(X5, yo, post5, rng))
    out_rows = [{"quantity": "outlier point elpd_i", "exact_loo": ex5[i_out], "psis_loo": ll5["pointwise"][i_out],
                 "is_loo": ll5_is["pointwise"][i_out], "waic": w5["pointwise"][i_out],
                 "khat": ll5["khat"][i_out]},
                {"quantity": "all other points (sum)", "exact_loo": float(np.delete(ex5, i_out).sum()),
                 "psis_loo": float(np.delete(ll5["pointwise"], i_out).sum()),
                 "is_loo": float(np.delete(ll5_is["pointwise"], i_out).sum()),
                 "waic": float(np.delete(w5["pointwise"], i_out).sum()),
                 "khat": float(np.delete(ll5["khat"], i_out).max())}]
    print("\nDegree-5 model with one planted outlier (khat column: that point / max over the rest):")
    print_table(out_rows)
    save_csv(out_rows, "ch09_outlier")

    # ------------------------------------------------------------ GPD shape recovery
    gpd_rows = []
    for k in (0.1, 0.3, 0.5, 0.7, 0.9, 1.2):
        est = [mc.gpd_fit(stats.genpareto.rvs(k, size=200, random_state=rng))[0] for _ in range(200)]
        gpd_rows.append({"true_k": k, "est_mean": float(np.mean(est)), "est_sd": float(np.std(est))})
    print("\nGPD shape estimator on 200 exceedances (the tail size PSIS uses with ~4000 draws), 200 reps:")
    print_table(gpd_rows)
    save_csv(gpd_rows, "ch09_gpd_recovery")
    save_json({"n": n, "noise": NOISE, "S": S_DRAWS, "outlier_index": i_out,
               "khat_outlier_model": ll5["khat"]}, "ch09_summary")

    # ------------------------------------------------------------ figure 1
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    xx = np.linspace(-1, 1, 200)
    ax[0].scatter(x, y, s=12, color=C["black"], label="data (n=30)")
    ax[0].plot(xx, np.sin(np.pi * xx), color=C["green"], lw=2, label="true sin(pi x)")
    for deg, col in ((1, C["orange"]), (3, C["blue"]), (8, C["red"])):
        mn = nig_post(design(x, deg), y)[0]
        ax[0].plot(xx, design(xx, deg) @ mn, color=col, label=f"degree {deg} posterior mean")
    ax[0].set(ylim=(-2, 2), title="Candidate polynomial models", xlabel="x")
    ax[0].legend(fontsize=7)
    dd = [r["degree"] for r in one]
    for k, col, mk, lab in (("lppd_in_sample", C["grey"], "x", "in-sample lppd"), ("waic", C["orange"], "s", "WAIC"),
                            ("psis_loo", C["blue"], "o", "PSIS-LOO"), ("exact_loo", C["black"], "+", "exact LOO"),
                            ("true_elpd", C["green"], "^", "true elpd (20k test pts)")):
        ax[1].plot(dd, [r[k] for r in one], mk + "-", color=col, label=lab, ms=5)
    ax[1].set(xlabel="polynomial degree", ylabel="elpd (sum over 30 points)", title="One dataset",
              ylim=(min(r["true_elpd"] for r in one) - 15, max(r["lppd_in_sample"] for r in one) + 3))
    ax[1].legend(fontsize=7)
    data = [np.array(err[k]) for k in ("lppd_in_sample", "waic", "psis_loo", "is_loo", "exact_loo")]
    ax[2].boxplot(data, tick_labels=["in-sample", "WAIC", "PSIS-LOO", "IS-LOO", "exact LOO"], showfliers=False)
    ax[2].axhline(0, color=C["green"], ls=":")
    ax[2].set(ylabel="estimate - true elpd", title=f"Error vs truth, {reps} datasets x 9 models")
    savefig(fig, "ch09_waic_loo_vs_truth")

    # ------------------------------------------------------------ figure 2
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    order = np.argsort(x)
    kh = ll5["khat"]
    ax[0].scatter(x, kh, color=[C["red"] if i == i_out else C["blue"] for i in range(n)])
    for th, ls in ((0.5, ":"), (0.7, "--")):
        ax[0].axhline(th, color=C["grey"], ls=ls, label=f"k = {th}")
    ax[0].set(xlabel="x of observation", ylabel="Pareto k-hat",
              title=f"Planted outlier (red) at the design edge, k-hat {kh[i_out]:.2f}")
    ax[0].legend()
    tk = [r["true_k"] for r in gpd_rows]
    ax[1].errorbar(tk, [r["est_mean"] for r in gpd_rows], yerr=[r["est_sd"] for r in gpd_rows], fmt="o",
                   color=C["blue"], capsize=3, label="estimate (mean +- sd)")
    ax[1].plot([0, 1.3], [0, 1.3], color=C["black"], ls="--", lw=1, label="truth")
    ax[1].set(xlabel="true GPD shape k", ylabel="estimated k",
              title="GPD shape estimator, 200 exceedances")
    ax[1].legend()
    savefig(fig, "ch09_khat_diagnostics")
