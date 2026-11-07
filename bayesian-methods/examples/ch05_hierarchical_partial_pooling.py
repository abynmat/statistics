"""Chapter 5: a hierarchical normal model with known ground truth.

Author: Abyn Scaria. MIT License.

Model (J groups, each reports an estimate y_j with known standard error s_j):
    y_j ~ N(theta_j, s_j^2),  theta_j ~ N(mu, tau^2),  p(mu) flat,  tau ~ half-Cauchy(0, 5)

Because mu and theta can be integrated out analytically given tau (BDA3 section 5.4), the
exact posterior reduces to a 1-D integral over tau. We compute that on a fine grid and use it
as the reference answer for HMC (centred and non-centred), and as the partial-pooling
estimator in a repeated-sampling study against no pooling and complete pooling.
All groups, effects and standard errors are synthetic.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

from bayeslab import diagnostics as dg  # noqa: E402
from bayeslab import samplers as sp  # noqa: E402
from bayeslab.plotting import C, plt, print_table, save_csv, save_json, savefig  # noqa: E402

SEED = 505
J = 8
TAU_SCALE = 5.0


# ---------------------------------------------------------------- exact posterior via tau-grid
TAU_GRID = np.exp(np.linspace(np.log(1e-3), np.log(300.0), 3000))


def exact_posterior(y, s, tau_grid=TAU_GRID):
    """Means and sds of mu, tau, theta_j by quadrature over tau (log-spaced grid)."""
    V = s[None, :] ** 2 + tau_grid[:, None] ** 2               # (G, J) marginal variances
    w = 1.0 / V
    v_mu = 1.0 / w.sum(1)
    mu_hat = v_mu * (w * y).sum(1)
    # log p(tau | y) = log p(tau) + 0.5 log v_mu + sum_j log N(y_j | mu_hat, V_j)  (+ const)
    log_prior = -np.log1p((tau_grid / TAU_SCALE) ** 2)
    ll = 0.5 * np.log(v_mu) - 0.5 * np.sum(np.log(V) + (y - mu_hat[:, None]) ** 2 / V, axis=1)
    lp = log_prior + ll + np.log(tau_grid)                      # + log Jacobian of the log grid
    pt = np.exp(lp - logsumexp(lp))
    # theta_j | tau, y : precision-weighted average of y_j and mu, with mu itself uncertain.
    shrink = s[None, :] ** 2 / V                                # B_j(tau) = s_j^2 / (s_j^2 + tau^2)
    th_mean = (1 - shrink) * y + shrink * mu_hat[:, None]
    th_var = (1 - shrink) * s[None, :] ** 2 + shrink ** 2 * v_mu[:, None]
    E_th = pt @ th_mean
    V_th = pt @ (th_var + th_mean ** 2) - E_th ** 2
    E_mu = pt @ mu_hat
    E_tau = pt @ tau_grid
    return {"mu": (E_mu, np.sqrt(pt @ (v_mu + mu_hat ** 2) - E_mu ** 2)),
            "tau": (E_tau, np.sqrt(pt @ tau_grid ** 2 - E_tau ** 2)),
            "theta": (E_th, np.sqrt(V_th)), "p_tau": pt}


# ---------------------------------------------------------------- HMC log densities
def make_logp(y, s, centered):
    """Parameters: [mu, log_tau, theta_1..J] (centred) or [mu, log_tau, eta_1..J] (non-centred)."""
    def logp_grad(q):
        mu, lt, z = q[0], q[1], q[2:]
        tau = np.exp(lt)
        # half-Cauchy prior on tau, plus log|d tau / d log_tau| = log_tau
        lp = -np.log1p((tau / TAU_SCALE) ** 2) + lt
        g_lt = -2 * tau ** 2 / (TAU_SCALE ** 2 + tau ** 2) + 1.0
        if centered:
            th = z
            lp += -J * lt - 0.5 * np.sum((th - mu) ** 2) / tau ** 2
            g_mu = np.sum(th - mu) / tau ** 2
            g_lt += -J + np.sum((th - mu) ** 2) / tau ** 2
            r = (y - th) / s ** 2
            lp += -0.5 * np.sum((y - th) ** 2 / s ** 2)
            g_z = -(th - mu) / tau ** 2 + r
        else:
            th = mu + tau * z
            lp += -0.5 * np.sum(z ** 2)
            r = (y - th) / s ** 2
            lp += -0.5 * np.sum((y - th) ** 2 / s ** 2)
            g_mu = np.sum(r)
            g_lt += np.sum(r * tau * z)
            g_z = -z + r * tau
        return lp, np.concatenate([[g_mu, g_lt], g_z])
    return logp_grad


def to_theta(draws, centered):
    if centered:
        return draws[..., 2:]
    return draws[..., :1] + np.exp(draws[..., 1:2]) * draws[..., 2:]


def complete_pool(y, s):
    w = 1 / s ** 2
    return np.full_like(y, np.sum(w * y) / w.sum())


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    s = np.round(rng.uniform(8, 18, J), 1)
    mu_true, tau_true = 4.0, 6.0
    theta_true = rng.normal(mu_true, tau_true, J)
    y = rng.normal(theta_true, s)
    print("synthetic groups:\n  s     =", s, "\n  theta =", theta_true.round(2), "\n  y     =", y.round(2))

    ex = exact_posterior(y, s)

    # ------------------------------------------------------------ HMC vs exact
    out = {}
    for name, centered in (("centred", True), ("non-centred", False)):
        lpg = make_logp(y, s, centered)
        dr, rs = sp.run_chains(sp.hmc, 4, lambda r: np.r_[r.normal(0, 2), r.normal(1, 0.5), r.normal(0, 1, J)],
                               SEED, logp_grad=lpg, n_draws=2000, n_warmup=1000, path_length=3.0,
                               target_accept=0.9)
        out[name] = {"draws": dr, "theta": to_theta(dr, centered), "tau": np.exp(dr[:, :, 1]),
                     "div": int(sum(r["n_divergent"] for r in rs)),
                     "div_mask": np.stack([r["divergent"] for r in rs]),
                     "seconds": sum(r["seconds"] for r in rs)}
    rows = []
    for q in ["mu", "tau"] + [f"theta_{j + 1}" for j in range(J)]:
        if q == "mu":
            ref = ex["mu"]
        elif q == "tau":
            ref = ex["tau"]
        else:
            j = int(q.split("_")[1]) - 1
            ref = (ex["theta"][0][j], ex["theta"][1][j])
        row = {"param": q, "exact_mean": float(ref[0]), "exact_sd": float(ref[1])}
        for name in ("centred", "non-centred"):
            o = out[name]
            x = o["draws"][:, :, 0] if q == "mu" else (o["tau"] if q == "tau" else o["theta"][:, :, j])
            tag = "c" if name == "centred" else "nc"
            row[f"{tag}_mean_err"] = float(x.mean() - ref[0])
            row[f"{tag}_sd_err"] = float(x.std() - ref[1])
            row[f"{tag}_ess"] = dg.ess_bulk(x)
            row[f"{tag}_rhat"] = dg.rank_rhat(x)
        rows.append(row)
    print(f"\nHMC vs exact (tau-grid) posterior. Divergences: centred {out['centred']['div']}, "
          f"non-centred {out['non-centred']['div']} (of 8000 draws)")
    print_table(rows)
    save_csv(rows, "ch05_hmc_vs_exact")

    # ------------------------------------------------------------ repeated-sampling study
    study = []
    for tau_t in [0.0, 2.5, 5.0, 10.0, 20.0, 40.0]:
        se = {"no pooling": [], "complete pooling": [], "partial pooling (exact posterior mean)": []}
        cover = []
        for _ in range(400):
            th = rng.normal(mu_true, tau_t, J) if tau_t > 0 else np.full(J, mu_true)
            yy = rng.normal(th, s)
            e = exact_posterior(yy, s, TAU_GRID[::6])
            se["no pooling"].append(np.mean((yy - th) ** 2))
            se["complete pooling"].append(np.mean((complete_pool(yy, s) - th) ** 2))
            se["partial pooling (exact posterior mean)"].append(np.mean((e["theta"][0] - th) ** 2))
            cover.append(np.mean(np.abs(e["theta"][0] - th) < 1.645 * e["theta"][1]))
        row = {"tau_true": tau_t, **{k: float(np.sqrt(np.mean(v))) for k, v in se.items()},
               "partial_90pct_normal_interval_coverage": float(np.mean(cover))}
        study.append(row)
    print("\nRMSE of group-effect estimates over 400 synthetic datasets per tau (s_j fixed, 8-18):")
    print_table(study)
    save_csv(study, "ch05_pooling_study")
    save_json({"s": s, "theta_true": theta_true, "y": y, "mu_true": mu_true, "tau_true": tau_true,
               "divergences": {k: v["div"] for k, v in out.items()}}, "ch05_data")

    # ------------------------------------------------------------ figure 1: shrinkage + tau
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    cp = complete_pool(y, s)[0]
    pp = ex["theta"][0]
    order = np.argsort(y)
    for k, j in enumerate(order):
        ax[0].plot([0, 1], [y[j], pp[j]], color=C["grey"], lw=0.8)
    ax[0].scatter(np.zeros(J), y, color=C["orange"], zorder=3, label="no pooling (y_j)")
    ax[0].scatter(np.ones(J), pp, color=C["blue"], zorder=3, label="partial pooling")
    ax[0].scatter(np.full(J, 1.15), theta_true, color=C["green"], marker="x", zorder=3, label="true theta_j")
    ax[0].axhline(cp, color=C["red"], ls=":", label="complete pooling")
    ax[0].set(xticks=[0, 1], xticklabels=["raw", "posterior mean"], xlim=(-0.2, 1.3),
              title="Shrinkage of 8 group estimates")
    ax[0].legend(fontsize=7)
    ax[1].plot(TAU_GRID, ex["p_tau"] / np.gradient(TAU_GRID), color=C["black"], lw=2, label="exact p(tau | y)")
    for name, col in (("centred", C["red"]), ("non-centred", C["blue"])):
        ax[1].hist(out[name]["tau"].ravel(), bins=80, range=(0, 30), density=True, histtype="step",
                   color=col, label=f"HMC {name} ({out[name]['div']} div.)")
    ax[1].axvline(tau_true, color=C["green"], ls=":", label="true tau")
    ax[1].set(xlim=(0, 30), xlabel="tau", title="Posterior of tau: HMC vs exact")
    ax[1].legend(fontsize=7)
    o = out["centred"]
    lt = o["draws"][:, :, 1].ravel()
    t1 = o["theta"][:, :, 0].ravel()
    m = o["div_mask"].ravel()
    ax[2].scatter(t1, lt, s=2, alpha=0.3, color=C["sky"], label="centred draws")
    ax[2].scatter(t1[m], lt[m], s=10, color=C["red"], label="divergent")
    o2 = out["non-centred"]
    ax[2].scatter(o2["theta"][:, :, 0].ravel()[::4], o2["draws"][:, :, 1].ravel()[::4], s=2, alpha=0.2,
                  color=C["blue"], label="non-centred draws")
    ax[2].set(xlabel="theta_1", ylabel="log tau", title="The funnel in a hierarchical model")
    ax[2].legend(fontsize=7, markerscale=3)
    savefig(fig, "ch05_shrinkage_and_tau")

    # ------------------------------------------------------------ figure 2: pooling study
    fig, ax = plt.subplots(figsize=(6, 3.8))
    tt = [r["tau_true"] for r in study]
    for k, col, mk in (("no pooling", C["orange"], "o"), ("complete pooling", C["red"], "s"),
                       ("partial pooling (exact posterior mean)", C["blue"], "^")):
        ax.plot(tt, [r[k] for r in study], mk + "-", color=col, label=k)
    ax.set(xlabel="true between-group sd tau", ylabel="RMSE of theta estimates",
           title="Partial pooling never loses by much, and often wins")
    ax.legend()
    savefig(fig, "ch05_pooling_rmse_vs_tau")
