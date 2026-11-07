"""Chapter 1: Bayes by brute force (a grid) versus Bayes by algebra (conjugate priors).

Author: Abyn Scaria. MIT License.

Three models with closed-form posteriors:
  A. beta-binomial               p ~ Beta(a, b),  k | p ~ Binomial(n, p)
  B. normal-normal, known sigma  mu ~ N(m0, s0^2), y_i | mu ~ N(mu, sigma^2)
  C. normal, unknown mean and variance, normal-inverse-gamma prior (2-D grid)
For each we compute the posterior on a grid, compare its mean and sd with the exact answer,
and measure how the grid error falls as the grid gets finer.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

from bayeslab.plotting import C, plt, print_table, save_csv, save_json, savefig  # noqa: E402

rng = np.random.default_rng(101)

# ---------------------------------------------------------------- data (synthetic, seeded)
N_TRIALS, P_TRUE = 40, 0.3
K = int(rng.binomial(N_TRIALS, P_TRUE))
A0, B0 = 2.0, 2.0

SIGMA, MU_TRUE = 2.0, 2.5
Y_NN = rng.normal(MU_TRUE, SIGMA, 10)
M0, S0 = 0.0, 5.0

Y_NIG = rng.normal(1.0, 1.5, 12)
NIG0 = dict(m0=0.0, k0=1.0, a0=2.0, b0=2.0)


def grid_moments(log_post, grid, weights):
    """Normalise a log posterior on a grid (with quadrature weights) and return mean, sd."""
    lw = log_post + np.log(weights)
    w = np.exp(lw - logsumexp(lw))
    mean = np.sum(w * grid)
    return mean, np.sqrt(np.sum(w * (grid - mean) ** 2)), w


# ---------------------------------------------------------------- A. beta-binomial
def bb_grid(G):
    p = (np.arange(G) + 0.5) / G                               # midpoint rule on (0, 1)
    lp = stats.beta.logpdf(p, A0, B0) + stats.binom.logpmf(K, N_TRIALS, p)
    return grid_moments(lp, p, np.full(G, 1.0 / G)) + (p,)


bb_exact = stats.beta(A0 + K, B0 + N_TRIALS - K)


# ---------------------------------------------------------------- B. normal-normal
def nn_exact():
    prec = 1 / S0 ** 2 + Y_NN.size / SIGMA ** 2
    mean = (M0 / S0 ** 2 + Y_NN.sum() / SIGMA ** 2) / prec
    return mean, 1 / np.sqrt(prec)


def nn_grid(G, lo=-5.0, hi=10.0):
    mu = lo + (np.arange(G) + 0.5) * (hi - lo) / G
    lp = stats.norm.logpdf(mu, M0, S0) + stats.norm.logpdf(Y_NN[:, None], mu, SIGMA).sum(0)
    return grid_moments(lp, mu, np.full(G, (hi - lo) / G)) + (mu,)


# ---------------------------------------------------------------- C. normal-inverse-gamma
def nig_exact():
    y, n = Y_NIG, Y_NIG.size
    m0, k0, a0, b0 = NIG0.values()
    kn, an = k0 + n, a0 + n / 2
    mn = (k0 * m0 + y.sum()) / kn
    bn = b0 + 0.5 * np.sum((y - y.mean()) ** 2) + k0 * n * (y.mean() - m0) ** 2 / (2 * kn)
    # mu | y ~ Student-t_{2 an}(mn, bn / (an kn));  sigma^2 | y ~ InvGamma(an, bn)
    scale2 = bn / (an * kn)
    sd_mu = np.sqrt(scale2 * (2 * an) / (2 * an - 2))
    ig = stats.invgamma(an, scale=bn)
    return {"mu_mean": mn, "mu_sd": sd_mu, "s2_mean": ig.mean(), "s2_sd": ig.std(),
            "t": stats.t(2 * an, mn, np.sqrt(scale2))}


def nig_grid(G):
    """2-D grid over (mu, log sigma^2); log-scale grid needs the Jacobian sigma^2."""
    mu = np.linspace(-3, 5, G)
    ls2 = np.linspace(np.log(0.1), np.log(20.0), G)
    M, LS = np.meshgrid(mu, ls2, indexing="ij")
    s2 = np.exp(LS)
    m0, k0, a0, b0 = NIG0.values()
    lp = (stats.invgamma.logpdf(s2, a0, scale=b0) + LS                   # prior + Jacobian
          + stats.norm.logpdf(M, m0, np.sqrt(s2 / k0))
          + stats.norm.logpdf(Y_NIG[:, None, None], M, np.sqrt(s2)).sum(0))
    w = np.exp(lp - logsumexp(lp))
    mu_mean = np.sum(w * M)
    s2_mean = np.sum(w * s2)
    return {"mu_mean": mu_mean, "mu_sd": np.sqrt(np.sum(w * (M - mu_mean) ** 2)),
            "s2_mean": s2_mean, "s2_sd": np.sqrt(np.sum(w * (s2 - s2_mean) ** 2)),
            "mu": mu, "s2": np.exp(ls2), "w": w}


if __name__ == "__main__":
    print(f"data: beta-binomial k={K}/{N_TRIALS};  normal-normal n={Y_NN.size};  NIG n={Y_NIG.size}")
    G = 200
    rows = []
    m, s, _, _ = bb_grid(G)
    rows.append({"model": "beta-binomial: p", "exact_mean": bb_exact.mean(), "grid_mean": m,
                 "exact_sd": bb_exact.std(), "grid_sd": s})
    em, es = nn_exact()
    m, s, _, _ = nn_grid(G)
    rows.append({"model": "normal-normal: mu", "exact_mean": em, "grid_mean": m, "exact_sd": es, "grid_sd": s})
    ex, gr = nig_exact(), nig_grid(G)
    rows.append({"model": "NIG: mu", "exact_mean": ex["mu_mean"], "grid_mean": gr["mu_mean"],
                 "exact_sd": ex["mu_sd"], "grid_sd": gr["mu_sd"]})
    rows.append({"model": "NIG: sigma^2", "exact_mean": ex["s2_mean"], "grid_mean": gr["s2_mean"],
                 "exact_sd": ex["s2_sd"], "grid_sd": gr["s2_sd"]})
    for r in rows:
        r["abs_err_mean"] = abs(r["grid_mean"] - r["exact_mean"])
        r["rel_err_sd"] = abs(r["grid_sd"] / r["exact_sd"] - 1)
    print(f"\nGrid ({G} points per axis) vs exact posterior:")
    print_table(rows)
    save_csv(rows, "ch01_grid_vs_exact")

    # ---- grid convergence
    sizes = [4, 8, 16, 32, 64, 128, 256, 512]
    conv = []
    for g in sizes:
        bm, bs, _, _ = bb_grid(g)
        nm, ns, _, _ = nn_grid(g)
        ng = nig_grid(g)
        conv.append({"G": g,
                     "bb_mean_err": abs(bm - bb_exact.mean()), "bb_sd_err": abs(bs - bb_exact.std()),
                     "nn_mean_err": abs(nm - em), "nn_sd_err": abs(ns - es),
                     "nig_mean_err": abs(ng["mu_mean"] - ex["mu_mean"]),
                     "nig_sd_err": abs(ng["mu_sd"] - ex["mu_sd"]), "nig_points": g * g})
    print("\nGrid convergence:")
    print_table(conv, fmt="{:.2e}")
    save_csv(conv, "ch01_grid_convergence")

    # ---- prior sensitivity: posterior mean of p as data accumulates, three priors
    stream = rng.binomial(1, P_TRUE, 400)
    priors = {"Beta(1,1) flat": (1, 1), "Beta(20,5) wrong, confident": (20, 5), "Beta(0.5,0.5) Jeffreys": (0.5, 0.5)}
    ns = np.arange(1, stream.size + 1)
    ks = np.cumsum(stream)
    sens = {name: (a + ks) / (a + b + ns) for name, (a, b) in priors.items()}
    save_json({"n_at": [10, 50, 100, 400],
               **{name: [float(v[i - 1]) for i in (10, 50, 100, 400)] for name, v in sens.items()}},
              "ch01_prior_sensitivity")

    # ---------------------------------------------------------------- figure 1
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    _, _, w, p = bb_grid(40)
    ax[0].bar(p, w * 40, width=1 / 40, color=C["sky"], alpha=0.6, label="grid, 40 pts")
    pp = np.linspace(0, 1, 400)
    ax[0].plot(pp, bb_exact.pdf(pp), color=C["blue"], label=f"exact Beta({A0 + K:.0f},{B0 + N_TRIALS - K:.0f})")
    ax[0].plot(pp, stats.beta.pdf(pp, A0, B0), color=C["grey"], ls="--", label="prior Beta(2,2)")
    ax[0].axvline(P_TRUE, color=C["red"], lw=1, label="true p")
    ax[0].set(title=f"A. beta-binomial, k={K}/{N_TRIALS}", xlabel="p")
    ax[0].legend()
    _, _, w, mu = nn_grid(60)
    ax[1].bar(mu, w * 60 / 15, width=15 / 60, color=C["sky"], alpha=0.6, label="grid, 60 pts")
    mm = np.linspace(-5, 10, 400)
    ax[1].plot(mm, stats.norm.pdf(mm, em, es), color=C["blue"], label="exact normal")
    ax[1].plot(mm, stats.norm.pdf(mm, M0, S0), color=C["grey"], ls="--", label="prior N(0, 5^2)")
    ax[1].axvline(MU_TRUE, color=C["red"], lw=1, label="true mu")
    ax[1].set(title="B. normal-normal, sigma known", xlabel="mu")
    ax[1].legend()
    g2 = nig_grid(120)
    cs = ax[2].contourf(g2["mu"], np.log(g2["s2"]), g2["w"].T, levels=12, cmap="Blues")
    ax[2].set(title="C. NIG posterior on a 120x120 grid", xlabel="mu", ylabel="log sigma^2", ylim=(-1, 2.5),
              xlim=(-1, 3.5))
    ins = ax[2].inset_axes([0.6, 0.62, 0.38, 0.36])
    ins.plot(g2["mu"], g2["w"].sum(1) / (g2["mu"][1] - g2["mu"][0]), color=C["orange"], lw=2, label="grid")
    ins.plot(g2["mu"], ex["t"].pdf(g2["mu"]), color=C["black"], ls=":", label="Student-t")
    ins.set_xlim(-1, 3.5)
    ins.set_yticks([])
    ins.set_title("marginal mu", fontsize=7)
    ins.tick_params(labelsize=6)
    ins.legend(fontsize=6)
    savefig(fig, "ch01_posteriors")

    # ---------------------------------------------------------------- figure 2
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    c = conv
    gs = [r["G"] for r in c]
    ax[0].loglog(gs, [max(r["bb_mean_err"], 1e-16) for r in c], "o-", color=C["blue"], label="beta-binomial (1-D)")
    ax[0].loglog(gs, [max(r["nn_mean_err"], 1e-16) for r in c], "s-", color=C["green"], label="normal-normal (1-D)")
    ax[0].loglog(gs, [max(r["nig_mean_err"], 1e-16) for r in c], "^-", color=C["orange"], label="NIG mu (2-D grid)")
    ax[0].loglog(gs, [max(r["nig_sd_err"], 1e-16) for r in c], "^--", color=C["orange"], alpha=0.6, label="NIG sd(mu)")
    ax[0].set(title="Grid error in the posterior mean vs points per axis", xlabel="grid points per axis",
              ylabel="|grid - exact| (floored at 1e-16)")
    ax[0].legend()
    for name, v in sens.items():
        ax[1].semilogx(ns, v, label=name)
    ax[1].axhline(P_TRUE, color=C["red"], lw=1, ls=":", label="true p")
    ax[1].set(title="Prior sensitivity: posterior mean of p as data arrive", xlabel="observations",
              ylabel="E[p | data]")
    ax[1].legend()
    savefig(fig, "ch01_grid_convergence_and_priors")
