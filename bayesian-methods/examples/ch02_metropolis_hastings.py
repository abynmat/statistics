"""Chapter 2: random-walk Metropolis-Hastings, and checking the diagnostics themselves.

Author: Abyn Scaria. MIT License.

Parts
  1. Proposal scale: too small / tuned / too large, on N(0, 1). Traces and autocorrelation.
  2. Acceptance rate vs efficiency on a 10-D standard normal (theory says ~0.234 is best).
  3. Diagnostic self-check: our ESS on AR(1) chains vs the exact ESS; our R-hat on chains
     with known defects.
  4. Accuracy table: MH posterior mean / sd vs the exact posterior, with the Monte Carlo
     standard error (MCSE) from our ESS. If ESS is right, |error| / MCSE should be ~N(0,1).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.special import expit  # noqa: E402

from bayeslab import diagnostics as dg  # noqa: E402
from bayeslab import samplers as sp  # noqa: E402
from bayeslab import targets as tg  # noqa: E402
from bayeslab.plotting import C, plt, print_table, save_csv, save_json, savefig  # noqa: E402

SEED = 202


def std_normal_logp(x):
    return -0.5 * float(x @ x)


if __name__ == "__main__":
    # ------------------------------------------------------------ 1. scale on N(0, 1)
    scales = {"too small (0.1)": 0.1, "tuned (2.4)": 2.4, "too large (25)": 25.0}
    traces = {}
    for name, s in scales.items():
        r = sp.rw_metropolis(std_normal_logp, np.array([3.0]), 5000, np.random.default_rng(SEED),
                             n_warmup=0, scale=s, adapt=False)
        x = r["draws"][:, 0]
        traces[name] = (x, r["accept_rate"], dg.ess(x))
        print(f"{name:18s} accept={r['accept_rate']:.3f}  ESS={dg.ess(x):7.1f} / 5000")

    # ------------------------------------------------------------ 2. acceptance vs efficiency
    d = 10
    base = 2.38 / np.sqrt(d)
    sweep = []
    for f in [0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]:
        draws, res = sp.run_chains(sp.rw_metropolis, 4, lambda r: r.standard_normal(d), SEED,
                                   logp=std_normal_logp, n_draws=8000, n_warmup=1000,
                                   scale=f * base, adapt=False)
        ess_min = min(dg.ess(draws[:, :, j]) for j in range(d))
        sweep.append({"scale_factor": f, "accept": float(np.mean([r["accept_rate"] for r in res])),
                      "ess_min_per_draw": ess_min / draws[:, :, 0].size})
    best = max(sweep, key=lambda r: r["ess_min_per_draw"])
    print(f"\n10-D sweep: best efficiency at scale x{best['scale_factor']} (acceptance {best['accept']:.3f})")
    print_table(sweep)
    save_csv(sweep, "ch02_acceptance_sweep")

    # ------------------------------------------------------------ 3. diagnostics self-check
    rng = np.random.default_rng(SEED)
    ess_check = []
    for phi in [0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99]:
        reps = []
        for _ in range(20):
            reps.append(dg.ess(dg.ar1_chains(phi, 4, 2500, rng)))
        truth = 10000 * (1 - phi) / (1 + phi)
        ess_check.append({"phi": phi, "true_ess": truth, "est_mean": float(np.mean(reps)),
                          "est_sd": float(np.std(reps)), "ratio": float(np.mean(reps) / truth)})
    print("\nESS estimator vs exact AR(1) ESS (4 chains x 2500, 20 replicates):")
    print_table(ess_check)
    save_csv(ess_check, "ch02_ess_check")

    scen = {
        "4 well-mixed chains": dict(),
        "one chain shifted by 0.5 sd": dict(offsets=[0, 0, 0, 0.5]),
        "one chain shifted by 2 sd": dict(offsets=[0, 0, 0, 2.0]),
        "one chain 3x wider (same mean)": dict(scales=[1, 1, 1, 3.0]),
    }
    rhat_rows = []
    for name, kw in scen.items():
        vals = {"basic": [], "split": [], "rank": []}
        for _ in range(20):
            x = dg.ar1_chains(0.7, 4, 1000, rng, **kw)
            vals["basic"].append(dg.rhat_basic(x))
            vals["split"].append(dg.split_rhat(x))
            vals["rank"].append(dg.rank_rhat(x))
        rhat_rows.append({"scenario": name, **{k: float(np.mean(v)) for k, v in vals.items()}})
    x = dg.ar1_chains(0.7, 4, 1000, rng) + np.linspace(-1.5, 1.5, 1000)[None, :]
    rhat_rows.append({"scenario": "all chains drift (trend)", "basic": dg.rhat_basic(x),
                      "split": dg.split_rhat(x), "rank": dg.rank_rhat(x)})
    print("\nR-hat variants on chains with known defects (mean of 20 replicates; >1.01 = flag):")
    print_table(rhat_rows)
    save_csv(rhat_rows, "ch02_rhat_check")

    # ------------------------------------------------------------ 4. accuracy vs exact
    acc_rows = []
    zs = []
    g2 = tg.Gaussian([1.0, -2.0], [[1.0, 1.2], [1.2, 4.0]])
    # beta-binomial posterior sampled on the logit scale: z = logit(p).
    # log p(z) = (a+k) log sigmoid(z) + (b+n-k) log(1 - sigmoid(z))  (includes the Jacobian)
    a, b, k, n = 2.0, 2.0, 17, 40
    exact_p = stats.beta(a + k, b + n - k)

    def logit_beta(z):
        return (a + k) * np.log(expit(z[0])) + (b + n - k) * np.log(expit(-z[0]))

    cases = [("2-D Gaussian x1", g2.logp, 2, 0, g2.mean[0], g2.sd[0], None),
             ("2-D Gaussian x2", g2.logp, 2, 1, g2.mean[1], g2.sd[1], None),
             ("beta-binomial p (logit scale)", logit_beta, 1, 0, exact_p.mean(), exact_p.std(), expit)]
    cache = {}
    for name, lp, dim, j, m_true, s_true, transform in cases:
        key = (name.split()[0], dim)
        if key not in cache:
            cache[key] = sp.run_chains(sp.rw_metropolis, 4, lambda r: r.standard_normal(dim) * 2, SEED,
                                       logp=lp, n_draws=5000, n_warmup=2000)
        draws, res = cache[key]
        x = draws[:, :, j]
        if transform is not None:
            x = transform(x)
        e = dg.ess(x)
        mcse = x.std() / np.sqrt(e)
        err = x.mean() - m_true
        zs.append(err / mcse)
        acc_rows.append({"quantity": name, "exact_mean": m_true, "mh_mean": float(x.mean()),
                         "exact_sd": s_true, "mh_sd": float(x.std()), "ess": e, "mcse": mcse,
                         "err_over_mcse": err / mcse, "rhat": dg.rank_rhat(x),
                         "accept": float(np.mean([r["accept_rate"] for r in res]))})
    print("\nMH vs exact posterior (4 chains x 5000 draws):")
    print_table(acc_rows)
    save_csv(acc_rows, "ch02_mh_vs_exact")

    # Calibration of the MCSE: repeat the 2-D run many times, check that err/MCSE ~ N(0,1).
    zrep = []
    for rep in range(60):
        draws, _ = sp.run_chains(sp.rw_metropolis, 4, lambda r: r.standard_normal(2) * 2, 1000 + rep,
                                 logp=g2.logp, n_draws=2000, n_warmup=1000)
        for j in range(2):
            x = draws[:, :, j]
            zrep.append((x.mean() - g2.mean[j]) / (x.std() / np.sqrt(dg.ess(x))))
    zrep = np.array(zrep)
    cover = float(np.mean(np.abs(zrep) < 1.96))
    print(f"\nMCSE calibration over {zrep.size} runs: sd(err/MCSE) = {zrep.std():.3f} (ideal 1), "
          f"95% interval coverage = {cover:.3f} (ideal 0.95)")
    save_json({"z_sd": float(zrep.std()), "coverage_95": cover, "n_runs": int(zrep.size),
               "best_scale_factor": best["scale_factor"], "best_accept": best["accept"],
               "trace_runs": {k: {"accept": v[1], "ess_of_5000": v[2]} for k, v in traces.items()}},
              "ch02_summary")

    # ------------------------------------------------------------ figure 1: traces + ACF
    fig, ax = plt.subplots(3, 2, figsize=(10, 6), gridspec_kw={"width_ratios": [3, 1]})
    for i, (name, (x, acc, e)) in enumerate(traces.items()):
        ax[i, 0].plot(x[:2000], lw=0.6, color=[C["orange"], C["blue"], C["red"]][i])
        ax[i, 0].set_title(f"{name}: acceptance {acc:.2f}, ESS {e:.0f} of 5000", loc="left")
        ax[i, 0].set_ylim(-4, 4)
        ac = dg.autocorrelation(x)[:60]
        ax[i, 1].bar(np.arange(60), ac, color=C["grey"], width=0.8)
        ax[i, 1].set_ylim(-0.2, 1.05)
        ax[i, 1].set_title("autocorrelation", fontsize=8)
    ax[2, 0].set_xlabel("iteration (start at x=3)")
    ax[2, 1].set_xlabel("lag")
    savefig(fig, "ch02_traces_and_acf")

    # ------------------------------------------------------------ figure 2: tuning + ESS check
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    acc = [r["accept"] for r in sweep]
    eff = [r["ess_min_per_draw"] for r in sweep]
    ax[0].plot(acc, eff, "o-", color=C["blue"])
    ax[0].axvline(0.234, color=C["red"], ls=":", label="0.234 (theory)")
    ax[0].set(xlabel="acceptance rate", ylabel="min ESS per draw", title="10-D normal: efficiency vs acceptance")
    ax[0].legend()
    tr = [r["true_ess"] for r in ess_check]
    ax[1].errorbar(tr, [r["est_mean"] for r in ess_check], yerr=[r["est_sd"] for r in ess_check],
                   fmt="o", color=C["green"], capsize=3, label="our ESS (mean +- sd)")
    ax[1].plot([20, 12000], [20, 12000], color=C["black"], ls="--", lw=1, label="exact")
    ax[1].set(xscale="log", yscale="log", xlabel="exact ESS of AR(1) chains", ylabel="estimated ESS",
              title="ESS estimator check")
    ax[1].legend()
    names = [r["scenario"].replace(" (", "\n(").replace("shifted", "shifted\n") for r in rhat_rows]
    xx = np.arange(len(rhat_rows))
    for o, (k, col) in enumerate([("basic", C["grey"]), ("split", C["blue"]), ("rank", C["orange"])]):
        ax[2].bar(xx + (o - 1) * 0.27, [r[k] - 1 for r in rhat_rows], width=0.27, color=col, label=f"{k} R-hat")
    ax[2].axhline(0.01, color=C["red"], ls=":", label="1.01 threshold")
    ax[2].set_yscale("log")
    ax[2].set_xticks(xx)
    ax[2].set_xticklabels(names, fontsize=6)
    ax[2].set(ylabel="R-hat - 1 (log)", title="R-hat variants on defective chains")
    ax[2].legend(fontsize=7)
    savefig(fig, "ch02_tuning_and_diagnostic_checks")
