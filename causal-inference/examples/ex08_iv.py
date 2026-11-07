"""Example 8: instrumental variables. Strong vs weak instruments, and LATE.

Author: Abyn Scaria. MIT License.

Part A, linear IV with an unobserved confounder U (true effect beta = 1.0):
    Z ~ N(0,1),  U ~ N(0,1)
    T = pi * Z + U + 0.3 * N(0,1)
    Y = 1.0 * T + 2.0 * U + 0.3 * N(0,1)
(The first-stage and structural errors are highly correlated, rho ~ 0.95, which
is the regime where weak instruments distort inference most.)
OLS is biased upward because U drives both T and Y. 2SLS is consistent for any
pi != 0, but when the first stage is weak (small pi) its sampling distribution
is wide, centred near the OLS bias, and nominal 95% CIs undercover. We sweep pi
so the expected first-stage F covers ~1.5 to ~200.

Part B, binary instrument with heterogeneous effects:
    compliers (60%), always-takers (20%), never-takers (20%), no defiers.
    Effect is 2.0 for compliers, 5.0 for always-takers, 0.5 for never-takers.
    The Wald / 2SLS estimate recovers the complier effect (LATE = 2.0), not the
    population ATE (0.6*2 + 0.2*5 + 0.2*0.5 = 2.3).
"""
from common import PALETTE, plt, save_json, savefig  # noqa: I001

import numpy as np

BETA = 1.0


def tsls(Y, T, Z):
    """Just-identified 2SLS with an intercept, plus the robust SE and first-stage F."""
    n = len(Y)
    Zc = np.column_stack([np.ones(n), Z])
    Xc = np.column_stack([np.ones(n), T])
    # first stage
    g, *_ = np.linalg.lstsq(Zc, T, rcond=None)
    v = T - Zc @ g
    s2 = v @ v / (n - 2)
    se_pi = np.sqrt(s2 * np.linalg.inv(Zc.T @ Zc)[1, 1])
    F = (g[1] / se_pi) ** 2
    # 2SLS: beta = (Z'X)^-1 Z'Y
    b = np.linalg.solve(Zc.T @ Xc, Zc.T @ Y)
    e = Y - Xc @ b
    A = np.linalg.inv(Zc.T @ Xc)
    meat = (Zc * e[:, None] ** 2).T @ Zc
    cov = A @ meat @ A.T
    return b[1], np.sqrt(cov[1, 1]), F


def part_a(n=500, n_sims=2000, noise=0.3, seed=8):
    rng = np.random.default_rng(seed)
    pis = [0.025, 0.05, 0.1, 0.2, 0.4, 0.8]
    rows, draws = {}, {}
    for pi in pis:
        est, F, cover, ols_b = [], [], 0, []
        for _ in range(n_sims):
            Z, U = rng.normal(size=n), rng.normal(size=n)
            T = pi * Z + U + noise * rng.normal(size=n)
            Y = BETA * T + 2.0 * U + noise * rng.normal(size=n)
            b, se, f = tsls(Y, T, Z)
            est.append(b)
            F.append(f)
            cover += abs(b - BETA) < 1.96 * se
            ols_b.append(np.polyfit(T, Y, 1)[0])
        est = np.array(est)
        rows[str(pi)] = {"mean_first_stage_F": float(np.mean(F)), "median_2sls": float(np.median(est)),
                         "iqr_2sls": float(np.subtract(*np.percentile(est, [75, 25]))),
                         "median_abs_error": float(np.median(np.abs(est - BETA))),
                         "ci95_coverage": cover / n_sims, "ols_mean": float(np.mean(ols_b))}
        draws[pi] = est
        print(f"pi={pi:<5} F={np.mean(F):7.1f}  2SLS median={np.median(est):.3f}  "
              f"IQR={rows[str(pi)]['iqr_2sls']:.3f}  coverage={cover / n_sims:.3f}  OLS={np.mean(ols_b):.3f}")
    return rows, draws


def part_b(n=20000, n_sims=300):
    rng = np.random.default_rng(81)
    wald, ate_draws = [], []
    for _ in range(n_sims):
        typ = rng.choice(3, size=n, p=[0.6, 0.2, 0.2])  # 0 complier, 1 always, 2 never
        effect = np.array([2.0, 5.0, 0.5])[typ]
        Z = rng.binomial(1, 0.5, n)
        T = np.where(typ == 0, Z, np.where(typ == 1, 1, 0))
        base = 1.0 + 1.5 * (typ == 1) - 1.0 * (typ == 2) + rng.normal(size=n)  # types differ in Y(0) too
        Y = base + effect * T
        wald.append((Y[Z == 1].mean() - Y[Z == 0].mean()) / (T[Z == 1].mean() - T[Z == 0].mean()))
        ate_draws.append(effect.mean())
    naive_ols = float(np.polyfit(T, Y, 1)[0])
    res = {"late_true": 2.0, "ate_true": 0.6 * 2.0 + 0.2 * 5.0 + 0.2 * 0.5,
           "wald_mean": float(np.mean(wald)), "wald_sd": float(np.std(wald)), "naive_ols_last_draw": naive_ols}
    print("Part B:", res)
    return res


def main():
    rows, draws = part_a()
    print("mild endogeneity (noise sd 1.0, error correlation ~0.63):")
    rows_mild, _ = part_a(noise=1.0, seed=9)
    late = part_b()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    bins = np.linspace(-2, 4, 121)
    for i, pi in enumerate([0.025, 0.1, 0.8]):
        F = rows[str(pi)]["mean_first_stage_F"]
        axes[0].hist(np.clip(draws[pi], -2, 4), bins=bins, histtype="step", lw=2, color=PALETTE[i],
                     density=True, label=f"pi={pi}, mean F={F:.0f}")
    axes[0].axvline(BETA, color="k", ls="--", label="true effect 1.0")
    axes[0].axvline(rows["0.8"]["ols_mean"], color=PALETTE[3], ls=":", label="OLS (biased)")
    axes[0].set_xlabel("2SLS estimate (clipped to [-2, 4])")
    axes[0].set_title("Weak instruments: 2SLS drifts toward OLS and spreads out")
    axes[0].legend(fontsize=8)
    Fs = [rows[k]["mean_first_stage_F"] for k in rows]
    cov = [rows[k]["ci95_coverage"] for k in rows]
    axes[1].plot(Fs, cov, "o-", color=PALETTE[0])
    axes[1].axhline(0.95, color="k", ls="--", label="nominal 95%")
    axes[1].axvline(10, color=PALETTE[3], ls=":", label="F = 10 rule of thumb")
    axes[1].axvline(104.7, color=PALETTE[2], ls=":", label="F = 104.7 (Lee et al. 2022)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("mean first-stage F")
    axes[1].set_ylabel("coverage of 2SLS 95% CI")
    axes[1].set_title("Confidence-interval coverage vs instrument strength")
    axes[1].legend(fontsize=8)
    savefig(fig, "ex08_iv_weak_strong")
    save_json({"true_beta": BETA, "weak_iv_sweep": rows, "weak_iv_sweep_mild_endogeneity": rows_mild,
               "late": late}, "ex08_iv")


if __name__ == "__main__":
    main()
