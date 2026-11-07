"""Chapter 3: Hamiltonian Monte Carlo from scratch, against Metropolis-Hastings.

Author: Abyn Scaria. MIT License.

Parts
  1. Leapfrog: energy error vs step size; the stability limit eps < 2 * smallest sd, beyond
     which the energy error explodes (this is what a divergence is).
  2. Efficiency study: MH vs HMC on correlated Gaussians of dimension 2..100. Metrics:
     min bulk-ESS across coordinates, per second and per density/gradient evaluation.
  3. Accuracy table on the 50-D target against the exact mean and sd.
  4. Neal's funnel: centred vs non-centred HMC, and MH. Divergences, and whether the
     marginal of v matches its exact N(0, 3^2).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from bayeslab import diagnostics as dg  # noqa: E402
from bayeslab import samplers as sp  # noqa: E402
from bayeslab import targets as tg  # noqa: E402
from bayeslab.plotting import C, plt, print_table, save_csv, save_json, savefig  # noqa: E402

SEED = 303
N_CHAINS = 4


def min_bulk_ess(draws):
    return min(dg.ess_bulk(draws[:, :, j]) for j in range(draws.shape[2]))


def run_mh(target, n_draws, n_warmup, seed):
    draws, res = sp.run_chains(sp.rw_metropolis, N_CHAINS, lambda r: r.standard_normal(target.dim) * 0.5,
                               seed, logp=target.logp, n_draws=n_draws, n_warmup=n_warmup)
    return draws, res


def run_hmc(target, n_draws, n_warmup, seed, **kw):
    draws, res = sp.run_chains(sp.hmc, N_CHAINS, lambda r: r.standard_normal(target.dim) * 0.5, seed,
                               logp_grad=target.logp_grad, n_draws=n_draws, n_warmup=n_warmup, **kw)
    return draws, res


def accuracy(draws, target):
    flat = draws.reshape(-1, target.dim)
    return (float(np.max(np.abs(flat.mean(0) - target.mean) / target.sd)),
            float(np.max(np.abs(flat.std(0) / target.sd - 1))))


if __name__ == "__main__":
    # ------------------------------------------------------------ 1. leapfrog energy error
    g2 = tg.Gaussian([0.0, 0.0], [[1.0, 0.95], [0.95, 1.0]])
    sd_min = np.sqrt(np.linalg.eigvalsh(g2.cov).min())          # 0.224: the narrow direction
    rng = np.random.default_rng(SEED)
    x0, p0 = np.array([1.0, 1.2]), rng.standard_normal(2)
    lp0, gr0 = g2.logp_grad(x0)
    h0 = -lp0 + 0.5 * p0 @ p0
    eps_grid = np.geomspace(0.01, 0.6, 30)
    dh = []
    for e in eps_grid:
        n = max(1, int(np.ceil(3.0 / e)))
        x, p, lp, _, _, _ = sp.leapfrog(x0, p0, gr0, e, n, g2.logp_grad, np.ones(2))
        dh.append(abs(-lp + 0.5 * p @ p - h0))
    paths = {}
    for e in (0.1, 0.46):                                      # stable vs past the 2*sd_min limit
        xs, x, p, gr = [x0], x0, p0, gr0
        for _ in range(int(3.0 / e)):
            x, p, _, gr, _, _ = sp.leapfrog(x, p, gr, e, 1, g2.logp_grad, np.ones(2))
            xs.append(x)
        paths[e] = np.array(xs)
    print(f"leapfrog stability limit 2*sd_min = {2 * sd_min:.3f}")

    # ------------------------------------------------------------ 2. efficiency vs dimension
    dims = [2, 5, 10, 25, 50, 100]
    rows = []
    store = {}
    for d in dims:
        t = tg.correlated_gaussian(d, rho=0.9, seed=d)
        mh_draws, mh_res = run_mh(t, 20000, 5000, SEED + d)
        hmc_draws, hmc_res = run_hmc(t, 1000, 1000, SEED + d, path_length=3.0, adapt_mass=True)
        for name, dr, rs in (("MH", mh_draws, mh_res), ("HMC", hmc_draws, hmc_res)):
            e = min_bulk_ess(dr)
            secs = sum(r["seconds"] for r in rs)
            evals = sum(r["n_evals"] for r in rs)
            mean_err, sd_err = accuracy(dr, t)
            rhat = max(dg.rank_rhat(dr[:, :, j]) for j in range(d))
            rows.append({"dim": d, "sampler": name, "draws": dr.shape[0] * dr.shape[1],
                         "min_ess": e, "ess_per_sec": e / secs, "ess_per_1k_evals": 1000 * e / evals,
                         "seconds": secs, "max_rhat": rhat, "max_mean_err_sd": mean_err,
                         "max_sd_rel_err": sd_err,
                         "accept": float(np.mean([r["accept_rate"] for r in rs])),
                         "divergent": int(sum(r.get("n_divergent", 0) for r in rs))})
            store[(d, name)] = dr
        print(f"  dim {d:3d}: MH minESS {rows[-2]['min_ess']:8.1f} ({rows[-2]['ess_per_sec']:8.1f}/s)   "
              f"HMC minESS {rows[-1]['min_ess']:8.1f} ({rows[-1]['ess_per_sec']:8.1f}/s)", flush=True)
    print("\nMH vs HMC on correlated Gaussians (rho=0.9, marginal sds 0.1..1), 4 chains:")
    print_table(rows)
    save_csv(rows, "ch03_efficiency_vs_dim")

    # ------------------------------------------------------------ 3. 50-D accuracy table
    t50 = tg.correlated_gaussian(50, rho=0.9, seed=50)
    acc_rows = []
    for name in ("MH", "HMC"):
        flat = store[(50, name)].reshape(-1, 50)
        z_mean = (flat.mean(0) - t50.mean) / t50.sd
        r_sd = flat.std(0) / t50.sd - 1
        acc_rows.append({"sampler": name, "mean_abs_err_in_sd_units": float(np.mean(np.abs(z_mean))),
                         "max_abs_err_in_sd_units": float(np.max(np.abs(z_mean))),
                         "mean_sd_rel_err": float(np.mean(np.abs(r_sd))),
                         "max_sd_rel_err": float(np.max(np.abs(r_sd)))})
    print("\n50-D Gaussian: error against the exact mean (0) and sd, averaged over 50 coordinates")
    print_table(acc_rows)
    save_csv(acc_rows, "ch03_50d_vs_exact")

    # ------------------------------------------------------------ 4. Neal's funnel
    fun_rows = []
    fun_draws = {}
    exact_v = stats.norm(0, 3)
    configs = [
        ("HMC centred", tg.Funnel(9, True), dict(path_length=4.0, target_accept=0.8)),
        ("HMC centred, target 0.99", tg.Funnel(9, True), dict(path_length=4.0, target_accept=0.99)),
        ("HMC non-centred", tg.Funnel(9, False), dict(path_length=4.0, target_accept=0.8)),
    ]
    for name, f, kw in configs:
        dr, rs = sp.run_chains(sp.hmc, N_CHAINS, lambda r: np.r_[0.0, r.standard_normal(9) * 0.5],
                               SEED, logp_grad=f.logp_grad, n_draws=2000, n_warmup=1000, **kw)
        div = np.stack([r["divergent"] for r in rs])
        fun_draws[name] = (f.to_centered(dr), div)
        v = dr[:, :, 0]
        fun_rows.append({"sampler": name, "divergent_pct": 100 * div.mean(),
                         "v_mean": float(v.mean()), "v_sd": float(v.std()),
                         "v_q05": float(np.quantile(v, 0.05)), "exact_q05": float(exact_v.ppf(0.05)),
                         "ess_bulk_v": dg.ess_bulk(v), "rhat_v": dg.rank_rhat(v),
                         "step_size": float(np.mean([r["step_size"] for r in rs]))})
    f = tg.Funnel(9, True)
    dr, rs = sp.run_chains(sp.rw_metropolis, N_CHAINS, lambda r: np.r_[0.0, r.standard_normal(9) * 0.5],
                           SEED, logp=f.logp, n_draws=40000, n_warmup=10000)
    v = dr[:, :, 0]
    fun_draws["MH centred"] = (dr, np.zeros(v.shape, dtype=bool))
    fun_rows.append({"sampler": "MH centred (40k/chain)", "divergent_pct": float("nan"),
                     "v_mean": float(v.mean()), "v_sd": float(v.std()),
                     "v_q05": float(np.quantile(v, 0.05)), "exact_q05": float(exact_v.ppf(0.05)),
                     "ess_bulk_v": dg.ess_bulk(v), "rhat_v": dg.rank_rhat(v),
                     "step_size": float(np.mean([r["scale"] for r in rs]))})
    print("\nNeal's funnel (exact v ~ N(0, 3): mean 0, sd 3, 5% quantile -4.93):")
    print_table(fun_rows)
    save_csv(fun_rows, "ch03_funnel")
    # Where do divergences start? Compare v at divergent transitions vs all draws.
    dr_c, div_c = fun_draws["HMC centred"]
    v_div = dr_c[:, :, 0][div_c]
    save_json({"stability_limit": 2 * sd_min, "funnel_v_at_divergence_median": float(np.median(v_div)) if v_div.size else None,
               "funnel_v_all_median": float(np.median(dr_c[:, :, 0]))}, "ch03_summary")

    # ------------------------------------------------------------ figure 1: leapfrog + efficiency
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ell = np.linalg.cholesky(g2.cov) @ np.array([np.cos(np.linspace(0, 2 * np.pi, 200)),
                                                 np.sin(np.linspace(0, 2 * np.pi, 200))])
    for s in (1, 2):
        ax[0].plot(s * ell[0], s * ell[1], color=C["grey"], lw=0.8)
    ax[0].plot(*paths[0.1].T, ".-", color=C["blue"], ms=3, label="eps=0.10 (stable)")
    ax[0].plot(*paths[0.46].T, ".-", color=C["red"], ms=3, label=f"eps=0.46 (> limit {2 * sd_min:.2f})")
    ax[0].set(xlim=(-4, 4), ylim=(-4, 4), title="Leapfrog trajectories, rho=0.95 Gaussian")
    ax[0].legend(loc="lower right")
    ax[1].loglog(eps_grid, np.maximum(dh, 1e-12), "o-", color=C["blue"], ms=3)
    ax[1].axvline(2 * sd_min, color=C["red"], ls=":", label="stability limit 2*sd_min")
    ax[1].loglog(eps_grid[:12], 2 * dh[5] * (eps_grid[:12] / eps_grid[5]) ** 2, color=C["grey"], ls="--",
                 label="slope 2")
    ax[1].set(xlabel="step size eps", ylabel="|H(end) - H(start)|", title="Energy error after time 3")
    ax[1].legend()
    for name, col, mk in (("MH", C["orange"], "s"), ("HMC", C["blue"], "o")):
        rr = [r for r in rows if r["sampler"] == name]
        ax[2].loglog([r["dim"] for r in rr], [r["ess_per_sec"] for r in rr], mk + "-", color=col,
                     label=f"{name}: min ESS / sec")
        ax[2].loglog([r["dim"] for r in rr], [r["ess_per_1k_evals"] for r in rr], mk + ":", color=col,
                     alpha=0.6, label=f"{name}: min ESS / 1k evals")
    ax[2].set(xlabel="dimension", title="Sampler efficiency vs dimension")
    ax[2].legend(fontsize=7)
    savefig(fig, "ch03_leapfrog_and_efficiency")

    # ------------------------------------------------------------ figure 2: funnel
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
    for i, name in enumerate(["HMC centred", "HMC non-centred", "MH centred"]):
        dr, div = fun_draws[name]
        xs, vs = dr[:, :, 1].ravel(), dr[:, :, 0].ravel()
        ax[i].scatter(xs[::4], vs[::4], s=2, alpha=0.3, color=C["blue"])
        if div.any():
            ax[i].scatter(dr[:, :, 1][div], dr[:, :, 0][div], s=8, color=C["red"],
                          label=f"divergent ({div.sum()})")
            ax[i].legend(loc="upper right")
        ax[i].set(xlim=(-30, 30), ylim=(-10, 10), xlabel="x1", title=name)
    ax[0].set_ylabel("v (log variance)")
    vv = np.linspace(-10, 10, 300)
    ax[3].plot(vv, exact_v.pdf(vv), color=C["black"], lw=2, label="exact N(0, 3)")
    for name, col in (("HMC centred", C["red"]), ("HMC non-centred", C["blue"]), ("MH centred", C["orange"])):
        ax[3].hist(fun_draws[name][0][:, :, 0].ravel(), bins=60, density=True, histtype="step", color=col,
                   label=name, range=(-10, 10))
    ax[3].set(xlabel="v", title="Marginal of v vs exact")
    ax[3].legend(fontsize=7)
    savefig(fig, "ch03_funnel_divergences")
