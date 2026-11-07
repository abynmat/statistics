"""Example 7: synthetic control vs DiD when parallel trends fail.

Author: Abyn Scaria. MIT License.

Interactive fixed-effects (factor) model, 1 treated unit + 30 donors,
30 pre-periods and 10 post-periods:
    Y_jt(0) = delta_t + lambda_t . mu_j + eps_jt,  lambda_t has 2 trending factors
The treated unit's loadings mu_1 equal a convex combination of 4 donors, but they
are far from the donor average. Its untreated path therefore drifts away from the
average donor (parallel trends fails for plain DiD) while a weighted
combination of donors can track it.
True effect after treatment: -3.0 in every post period.

Estimators:
  * DiD vs the equally weighted donor pool
  * Synthetic control (Abadie et al.): non-negative weights summing to 1 that
    match the pre-period path
  * A simplified synthetic DiD (Arkhangelsky et al. 2021): regularised unit weights
    with an intercept, plus time weights, then a weighted DiD
Inference: in-space placebo test, re-running SC with each donor as the fake treated unit.
"""
from common import PALETTE, plt, save_json, savefig  # noqa: I001

import numpy as np
from scipy.optimize import minimize

T0, T1, J = 30, 10, 30
EFFECT = -3.0


def simulate(rng):
    T = T0 + T1
    t = np.arange(T)
    lam = np.column_stack([0.15 * t + rng.normal(0, 0.3, T), np.sin(t / 6) * 2 + rng.normal(0, 0.3, T)])
    delta = 0.05 * t
    mu_d = rng.uniform(0, 2, size=(J, 2))
    w_true = np.zeros(J)
    hi = np.argsort(mu_d[:, 0] + mu_d[:, 1])[-4:]  # 4 donors with high loadings
    w_true[hi] = [0.4, 0.3, 0.2, 0.1]
    mu_1 = w_true @ mu_d
    Y0 = delta[None, :] + np.vstack([mu_1, mu_d]) @ lam.T + rng.normal(0, 0.3, (J + 1, T))
    Y = Y0.copy()
    Y[0, T0:] += EFFECT
    return Y, Y0


def simplex_weights(target, donors, ridge=0.0, intercept=False):
    """argmin_w ||target - (w0) - donors.T w||^2 + ridge ||w||^2  s.t. w >= 0, sum w = 1."""
    J = donors.shape[0]

    def loss(w):
        r = target - donors.T @ w
        if intercept:
            r = r - r.mean()
        return r @ r + ridge * w @ w

    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1},)
    res = minimize(loss, np.full(J, 1 / J), bounds=[(0, 1)] * J, constraints=cons, method="SLSQP",
                   options={"maxiter": 500, "ftol": 1e-10})
    return res.x


def sc_estimate(Y, treated=0):
    donors = np.delete(np.arange(Y.shape[0]), treated)
    w = simplex_weights(Y[treated, :T0], Y[donors, :T0])
    synth = w @ Y[donors]
    gap = Y[treated] - synth
    pre_rmspe = np.sqrt(np.mean(gap[:T0] ** 2))
    post_rmspe = np.sqrt(np.mean(gap[T0:] ** 2))
    return gap[T0:].mean(), gap, w, pre_rmspe, post_rmspe


def did_estimate(Y):
    c = Y[1:].mean(0)
    return (Y[0, T0:].mean() - Y[0, :T0].mean()) - (c[T0:].mean() - c[:T0].mean()), Y[0] - c


def sdid_estimate(Y, ridge_scale=1.0):
    """Simplified SDID: unit weights match pre-trends up to a constant (intercept) with a
    ridge penalty scaled like the paper's zeta; time weights match the post-period mean of
    the controls from the pre-periods (also up to a constant)."""
    donors = Y[1:]
    sigma = np.std(np.diff(donors[:, :T0], axis=1))
    zeta2 = (1 * T1) ** 0.5 * sigma ** 2  # ~ (N_tr*T_post)^(1/2) * sigma^2 (heuristic scale)
    omega = simplex_weights(Y[0, :T0], donors[:, :T0], ridge=ridge_scale * zeta2 * T0, intercept=True)
    lam = simplex_weights(donors[:, T0:].mean(1), donors[:, :T0].T, intercept=True)
    tr_post, tr_pre = Y[0, T0:].mean(), lam @ Y[0, :T0]
    co_post, co_pre = omega @ donors[:, T0:].mean(1), omega @ (donors[:, :T0] @ lam)
    return (tr_post - tr_pre) - (co_post - co_pre), omega, lam


def main(n_sims=100):
    rng = np.random.default_rng(99)
    did, sc, sdid, sdid_lo = [], [], [], []
    for _ in range(n_sims):
        Y, _ = simulate(rng)
        did.append(did_estimate(Y)[0])
        sc.append(sc_estimate(Y)[0])
        sdid.append(sdid_estimate(Y)[0])
        sdid_lo.append(sdid_estimate(Y, ridge_scale=0.01)[0])
    summ = {k: {"mean": float(np.mean(v)), "bias": float(np.mean(v) - EFFECT), "sd": float(np.std(v)),
                "rmse": float(np.sqrt(np.mean((np.array(v) - EFFECT) ** 2)))}
            for k, v in {"DiD vs donor average": did, "Synthetic control": sc, "Synthetic DiD (simplified, default ridge)": sdid,
                                "Synthetic DiD (simplified, ridge x0.01)": sdid_lo}.items()}
    for k, v in summ.items():
        print(f"  {k:<42} mean {v['mean']:.3f}  bias {v['bias']:+.3f}  sd {v['sd']:.3f}  rmse {v['rmse']:.3f}")

    # One panel: paths, gaps and placebo inference
    Y, Y0 = simulate(np.random.default_rng(1))
    est, gap, w, pre, post = sc_estimate(Y)
    _, did_gap = did_estimate(Y)
    ratios = [post / pre]
    placebo_gaps = []
    for j in range(1, J + 1):
        Yp = np.delete(Y, 0, axis=0)  # drop the real treated unit from the placebo pool
        _, gj, _, prej, postj = sc_estimate(Yp, treated=j - 1)
        ratios.append(postj / prej)
        placebo_gaps.append(gj)
    p_value = np.mean(np.array(ratios) >= ratios[0])
    print(f"one panel: SC est {est:.3f}, pre-RMSPE {pre:.3f}, placebo p-value {p_value:.3f}, "
          f"top weights {sorted(np.round(w, 2))[-4:]}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    t = np.arange(T0 + T1)
    donors = np.delete(np.arange(J + 1), 0)
    axes[0].plot(t, Y[0], color="k", lw=2, label="treated unit (observed)")
    axes[0].plot(t, Y0[0], color="k", ls=":", lw=1.5, label="treated unit, true Y(0)")
    axes[0].plot(t, w @ Y[donors], color=PALETTE[0], lw=2, label="synthetic control")
    axes[0].plot(t, Y[1:].mean(0), color=PALETTE[3], lw=1.5, ls="--", label="donor average")
    axes[0].axvline(T0 - 0.5, color="grey")
    axes[0].set_xlabel("period")
    axes[0].set_title("Synthetic control tracks the treated unit pre-treatment")
    axes[0].legend(fontsize=8)
    for gj in placebo_gaps:
        axes[1].plot(t, gj, color="grey", alpha=0.3, lw=1)
    axes[1].plot(t, gap, color=PALETTE[0], lw=2.5, label="treated gap (SC)")
    axes[1].axhline(EFFECT, color="k", ls="--", label="true effect")
    axes[1].axvline(T0 - 0.5, color="grey")
    axes[1].set_title(f"Placebo-in-space: grey = each donor as fake treated, p = {p_value:.3f}")
    axes[1].set_xlabel("period")
    axes[1].set_ylabel("observed - synthetic")
    axes[1].legend(fontsize=8)
    savefig(fig, "ex07_synthetic_control")

    save_json({"true_effect": EFFECT, "n_sims": n_sims, "estimators": summ,
               "one_panel": {"sc_estimate": est, "pre_rmspe": pre, "placebo_p_value": p_value,
                             "n_donors": J}}, "ex07_synthetic_control")


if __name__ == "__main__":
    main()
