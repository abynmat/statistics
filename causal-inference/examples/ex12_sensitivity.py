"""Example 12: sensitivity analysis. How strong must a hidden confounder be?

Author: Abyn Scaria. MIT License.

Part A, omitted-variable-bias sensitivity (Cinelli & Hazlett 2020).
    X ~ N(0,1) observed, U ~ N(0,1) unobserved
    D = 0.6*X + 0.5*U + N(0,1)
    Y = 0.5*D + 0.8*X + 0.6*U + N(0,1)        true effect 0.5
The analyst regresses Y on D and X (U omitted), so the estimate is biased.
In OLS the size of that bias depends on only two numbers: the partial R^2 of U
with D (given X), and the partial R^2 of U with Y (given D and X). We draw the
contour of the bias-adjusted estimate over those two numbers, mark where the
real U sits, and report the robustness value RV: the strength (equal on both
axes) that would bring the estimate to zero.

Part B, E-values (VanderWeele & Ding 2017): for an observed risk ratio RR, the
minimum risk-ratio association a confounder needs with both treatment and outcome
to fully explain the result away.

Part C, Rosenbaum bounds for a matched-pairs sign test: how large can the
hidden bias Gamma (odds of treatment within a pair) be before the result is no
longer significant at 5%?
"""
from common import PALETTE, ols, plt, save_json, savefig  # noqa: I001

import numpy as np
from scipy import stats

TRUE = 0.5


def partial_r2(t, df):
    return t ** 2 / (t ** 2 + df)


def adjusted_estimate(est, se, df, r2dz, r2yz):
    """Cinelli-Hazlett: |bias| = se * sqrt(df) * sqrt(r2yz * r2dz / (1 - r2dz)). We assume the
    confounder pushes the estimate up (the worrying direction), so subtract it."""
    bias = se * np.sqrt(df) * np.sqrt(r2yz * r2dz / (1 - r2dz))
    return est - bias


def robustness_value(t, df, q=1.0):
    f = q * abs(t) / np.sqrt(df)
    return 0.5 * (np.sqrt(f ** 4 + 4 * f ** 2) - f ** 2)


def part_a(n=2000):
    rng = np.random.default_rng(12)
    X, U = rng.normal(size=n), rng.normal(size=n)
    D = 0.6 * X + 0.5 * U + rng.normal(size=n)
    Y = TRUE * D + 0.8 * X + 0.6 * U + rng.normal(size=n)

    b, se = ols(Y, np.column_stack([D, X]))
    est, se_d = b[1], se[1]
    df = n - 3
    t = est / se_d
    b_or, _ = ols(Y, np.column_stack([D, X, U]))  # oracle: include U
    # True partial R2 of U, computed from the simulated U
    bu, seu = ols(D, np.column_stack([X, U]))
    r2dz = partial_r2(bu[2] / seu[2], n - 3)
    by, sey = ols(Y, np.column_stack([D, X, U]))
    r2yz = partial_r2(by[3] / sey[3], n - 4)
    adj_at_truth = adjusted_estimate(est, se_d, df, r2dz, r2yz)
    rv = robustness_value(t, df)
    # observed covariate X as an informal benchmark
    bx, sex = ols(D, X)
    r2dx = partial_r2(bx[1] / sex[1], n - 2)
    byx, seyx = ols(Y, np.column_stack([D, X]))
    r2yx = partial_r2(byx[2] / seyx[2], df)
    res = {"estimate_omitting_U": est, "se": se_d, "t": t, "oracle_with_U": b_or[1], "true_effect": TRUE,
           "partial_r2_U_with_D": r2dz, "partial_r2_U_with_Y": r2yz,
           "adjusted_estimate_at_true_U_strength": adj_at_truth, "robustness_value_q1": rv,
           "partial_r2_D_estimate_on_Y": partial_r2(t, df),
           "benchmark_X_partial_r2_D": r2dx, "benchmark_X_partial_r2_Y": r2yx}
    print("Part A:", {k: round(float(v), 4) for k, v in res.items()})

    fig, ax = plt.subplots(figsize=(6.2, 5))
    g = np.linspace(0.001, 0.5, 200)
    R2D, R2Y = np.meshgrid(g, g)
    Z = adjusted_estimate(est, se_d, df, R2D, R2Y)
    cs = ax.contour(R2D, R2Y, Z, levels=np.round(np.arange(-0.4, 0.8, 0.1), 2), colors="grey", linewidths=0.8)
    ax.clabel(cs, fontsize=7)
    ax.contour(R2D, R2Y, Z, levels=[0], colors=PALETTE[3], linewidths=2)
    ax.plot(r2dz, r2yz, "o", color=PALETTE[0], ms=9,
            label=f"actual hidden U: adjusted {adj_at_truth:.3f} (truth {TRUE})")
    ax.plot(r2dx, r2yx, "s", color=PALETTE[2], ms=8, label="observed X's strength (informal benchmark)")
    ax.plot(rv, rv, "D", color=PALETTE[3], ms=7, label=f"robustness value RV = {rv:.3f}")
    ax.set_xlabel("partial R$^2$ of confounder with treatment")
    ax.set_ylabel("partial R$^2$ of confounder with outcome")
    ax.set_title(f"Bias-adjusted estimate (unadjusted {est:.3f}); red = 0")
    ax.legend(fontsize=7, loc="upper right")
    savefig(fig, "ex12_sensitivity_contour")
    return res


def e_value(rr):
    rr = np.where(rr < 1, 1 / rr, rr)
    return rr + np.sqrt(rr * (rr - 1))


def part_b():
    rrs = np.linspace(1.0, 5, 200)
    ev = e_value(rrs)
    table = {str(r): float(e_value(np.array(r))) for r in (1.2, 1.5, 2.0, 3.0, 4.0)}
    print("Part B E-values:", table)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(rrs, ev, color=PALETTE[0], lw=2)
    axes[0].set_xlabel("observed risk ratio")
    axes[0].set_ylabel("E-value")
    axes[0].set_title("E-value grows quickly with the observed effect")
    # Bias factor B = RR_UD * RR_UY / (RR_UD + RR_UY - 1); explain away RR = 2 when B >= 2
    g = np.linspace(1, 8, 300)
    A, B = np.meshgrid(g, g)
    BF = A * B / (A + B - 1)
    axes[1].contourf(A, B, BF >= 2.0, levels=[-0.5, 0.5, 1.5], colors=["white", "#F4C7A1"])
    axes[1].contour(A, B, BF, levels=[2.0], colors=PALETTE[3])
    e2 = float(e_value(np.array(2.0)))
    axes[1].plot(e2, e2, "o", color=PALETTE[0], label=f"E-value for RR=2: {e2:.2f}")
    axes[1].set_xlabel("RR(confounder, treatment)")
    axes[1].set_ylabel("RR(confounder, outcome)")
    axes[1].set_title("Shaded: confounders strong enough to explain away RR = 2")
    axes[1].legend(fontsize=8)
    savefig(fig, "ex12_evalue")
    return table


def part_c(n_pairs=400, effect=0.25):
    rng = np.random.default_rng(13)
    diffs = effect + rng.normal(0, 1, n_pairs)  # treated - control outcome within matched pairs
    pos = int((diffs > 0).sum())
    gammas = np.linspace(1, 3, 201)
    pvals = [1 - stats.binom.cdf(pos - 1, n_pairs, g / (1 + g)) for g in gammas]
    gamma_star = float(gammas[np.argmax(np.array(pvals) > 0.05)]) if max(pvals) > 0.05 else float("nan")
    print(f"Part C: {pos}/{n_pairs} pairs favour treatment; p(Gamma=1)={pvals[0]:.2e}; "
          f"significance lost at Gamma ~ {gamma_star:.2f}")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(gammas, pvals, color=PALETTE[0], lw=2)
    ax.axhline(0.05, color="k", ls="--")
    ax.axvline(gamma_star, color=PALETTE[3], ls=":", label=f"Gamma* = {gamma_star:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("Gamma (max odds ratio of treatment within a matched pair)")
    ax.set_ylabel("worst-case one-sided p-value")
    ax.set_title("Rosenbaum bounds, sign test")
    ax.legend()
    savefig(fig, "ex12_rosenbaum")
    return {"n_pairs": n_pairs, "pairs_favouring_treatment": pos, "p_at_gamma_1": float(pvals[0]),
            "gamma_star": gamma_star}


def main():
    a = part_a()
    b = part_b()
    c = part_c()
    save_json({"ovb": a, "e_values": b, "rosenbaum": c}, "ex12_sensitivity")


if __name__ == "__main__":
    main()
