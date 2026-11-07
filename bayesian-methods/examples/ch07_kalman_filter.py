"""Chapter 7: the Kalman filter and RTS smoother as exact Bayesian inference.

Author: Abyn Scaria. MIT License.

Parts
  1. 1-D local-level model: filter and smoother, and a check that the smoother equals the
     batch Gaussian posterior over the whole trajectory (to machine precision).
  2. 2-D constant-velocity tracking from noisy position fixes: RMSE of raw fixes, filter and
     smoother; consistency via NEES (should average dim(x) = 4 if the filter is honest).
  3. Robustness study: run the filter with Q and R scaled by 0.01..100 relative to the truth.
     RMSE, NEES and the marginal likelihood over the grid. Does the marginal likelihood find
     the true noise levels?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from bayeslab import filters as fl  # noqa: E402
from bayeslab.plotting import C, plt, print_table, save_csv, save_json, savefig  # noqa: E402

SEED = 707


def rmse(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)

    # ------------------------------------------------------------ 1. local level, batch check
    T1, q1, r1 = 120, 0.05, 1.0
    F1, H1, Q1, R1 = np.eye(1), np.eye(1), np.eye(1) * q1, np.eye(1) * r1
    x1, y1 = fl.simulate_lgssm(F1, H1, Q1, R1, np.zeros(1), T1, rng)
    m0, P0 = np.zeros(1), np.eye(1) * 10.0
    kf1 = fl.kalman_filter(y1, F1, H1, Q1, R1, m0, P0)
    sm1 = fl.rts_smoother(kf1, F1)
    D = np.eye(T1) - np.eye(T1, k=-1)
    prec = D.T @ D / q1
    prec[0, 0] += 1 / (P0[0, 0] + q1) - 1 / q1
    post_cov = np.linalg.inv(prec + np.eye(T1) / r1)
    post_mean = post_cov @ (y1[:, 0] / r1)
    batch = {"max_abs_mean_diff": float(np.max(np.abs(sm1["m"][:, 0] - post_mean))),
             "max_abs_var_diff": float(np.max(np.abs(sm1["P"][:, 0, 0] - np.diag(post_cov))))}
    # The filtered mean at t must equal the batch posterior using y_1..t only (check at t = T/2).
    # (The prior precision of x_1..t is rebuilt for length t: slicing the T-step matrix would
    # condition on the not-yet-seen x_{t+1}.)
    t_half = T1 // 2
    Dh = np.eye(t_half) - np.eye(t_half, k=-1)
    prec_h = Dh.T @ Dh / q1
    prec_h[0, 0] += 1 / (P0[0, 0] + q1) - 1 / q1
    pc = np.linalg.inv(prec_h + np.eye(t_half) / r1)
    batch["filter_vs_batch_at_T/2"] = float(abs(kf1["m"][t_half - 1, 0] - (pc @ (y1[:t_half, 0] / r1))[-1]))
    print("RTS smoother vs batch Gaussian posterior:", batch)
    rows1 = [{"estimate": "raw measurements", "rmse": rmse(y1, x1)},
             {"estimate": "Kalman filter", "rmse": rmse(kf1["m"], x1)},
             {"estimate": "RTS smoother", "rmse": rmse(sm1["m"], x1)}]

    # ------------------------------------------------------------ 2. 2-D constant velocity
    dt, q, r, T = 1.0, 0.05, 4.0, 150
    F, H, Q, R = fl.constant_velocity_model(dt, q, r)
    x0 = np.array([0.0, 0.0, 1.0, 0.5])
    m0, P0 = np.zeros(4), np.diag([25.0, 25.0, 4.0, 4.0])
    xs, ys = fl.simulate_lgssm(F, H, Q, R, x0, T, rng)
    kf = fl.kalman_filter(ys, F, H, Q, R, m0, P0)
    sm = fl.rts_smoother(kf, F)
    rows2 = [{"estimate": "raw position fixes", "pos_rmse": rmse(ys, xs[:, :2]), "vel_rmse": float("nan")},
             {"estimate": "Kalman filter", "pos_rmse": rmse(kf["m"][:, :2], xs[:, :2]),
              "vel_rmse": rmse(kf["m"][:, 2:], xs[:, 2:])},
             {"estimate": "RTS smoother", "pos_rmse": rmse(sm["m"][:, :2], xs[:, :2]),
              "vel_rmse": rmse(sm["m"][:, 2:], xs[:, 2:])}]
    # NEES averaged over 200 Monte Carlo runs (skip the first 10 steps of prior transient)
    nees_runs = []
    for _ in range(200):
        xr, yr = fl.simulate_lgssm(F, H, Q, R, x0, T, rng)
        k = fl.kalman_filter(yr, F, H, Q, R, x0, np.eye(4) * 1e-6)   # x0 is known exactly in the simulator
        nees_runs.append(fl.nees(xr, k["m"], k["P"])[10:].mean())
    nees_mean = float(np.mean(nees_runs))
    print(f"\n1-D local level (q={q1}, r={r1}):")
    print_table(rows1)
    print(f"\n2-D constant velocity (q={q}, r={r}):  mean NEES over 200 runs = {nees_mean:.3f} (ideal 4)")
    print_table(rows2)
    save_csv(rows1, "ch07_local_level_rmse")
    save_csv(rows2, "ch07_cv_tracking_rmse")

    # ------------------------------------------------------------ 3. mis-specified Q and R
    factors = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
    n_mc = 40
    datasets = [fl.simulate_lgssm(F, H, Q, R, x0, T, rng) for _ in range(n_mc)]
    rm_f = np.zeros((factors.size, factors.size))
    rm_s = np.zeros_like(rm_f)
    nees_g = np.zeros_like(rm_f)
    ll_g = np.zeros_like(rm_f)
    for i, fq in enumerate(factors):
        for j, fr in enumerate(factors):
            acc_f, acc_s, acc_n, acc_l = [], [], [], []
            for xr, yr in datasets:
                k = fl.kalman_filter(yr, F, H, Q * fq, R * fr, m0, P0)
                s_ = fl.rts_smoother(k, F)
                acc_f.append(rmse(k["m"][:, :2], xr[:, :2]))
                acc_s.append(rmse(s_["m"][:, :2], xr[:, :2]))
                acc_n.append(fl.nees(xr, k["m"], k["P"])[10:].mean())
                acc_l.append(k["loglik"])
            rm_f[i, j], rm_s[i, j] = np.mean(acc_f), np.mean(acc_s)
            nees_g[i, j], ll_g[i, j] = np.mean(acc_n), np.mean(acc_l)
    i0 = int(np.where(factors == 1.0)[0][0])
    best_ll = np.unravel_index(np.argmax(ll_g), ll_g.shape)
    best_rm = np.unravel_index(np.argmin(rm_f), rm_f.shape)
    rob = []
    for (name, fi, fj) in [("true Q, R", i0, i0), ("Q x0.01 (too confident in the model)", 0, i0),
                           ("Q x100 (too little smoothing)", 8, i0), ("R x0.01 (trusts fixes)", i0, 0),
                           ("R x100 (ignores fixes)", i0, 8), ("both x0.1", 2, 2), ("both x10", 6, 6)]:
        rob.append({"setting": name, "filter_rmse": rm_f[fi, fj], "smoother_rmse": rm_s[fi, fj],
                    "rmse_vs_true": rm_f[fi, fj] / rm_f[i0, i0], "mean_NEES": nees_g[fi, fj]})
    print(f"\nRobustness to mis-specified noise ({n_mc} runs each, ideal NEES = 4):")
    print_table(rob)
    print(f"max marginal likelihood at Q x{factors[best_ll[0]]}, R x{factors[best_ll[1]]};  "
          f"min filter RMSE at Q x{factors[best_rm[0]]}, R x{factors[best_rm[1]]}")
    save_csv(rob, "ch07_misspecification")
    save_json({"batch_check": batch, "nees_true_model": nees_mean, "factors": factors,
               "rmse_filter": rm_f, "rmse_smoother": rm_s, "nees": nees_g, "loglik": ll_g,
               "argmax_loglik": [float(factors[best_ll[0]]), float(factors[best_ll[1]])],
               "argmin_rmse": [float(factors[best_rm[0]]), float(factors[best_rm[1]])]},
              "ch07_summary")

    # ------------------------------------------------------------ figure 1: tracking
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    t = np.arange(T1)
    ax[0].scatter(t, y1[:, 0], s=6, color=C["grey"], label="measurements")
    ax[0].plot(t, x1[:, 0], color=C["black"], lw=1.5, label="true state")
    sd = np.sqrt(kf1["P"][:, 0, 0])
    ax[0].plot(t, kf1["m"][:, 0], color=C["orange"], label="filter mean")
    ax[0].fill_between(t, kf1["m"][:, 0] - 2 * sd, kf1["m"][:, 0] + 2 * sd, color=C["orange"], alpha=0.15)
    ax[0].plot(t, sm1["m"][:, 0], color=C["blue"], label="RTS smoother mean")
    ax[0].set(title=f"1-D local level; smoother = batch posterior (max diff {batch['max_abs_mean_diff']:.1e})",
              xlabel="time")
    ax[0].legend(fontsize=7)
    ax[1].scatter(ys[:, 0], ys[:, 1], s=6, color=C["grey"], label="position fixes (sd 2)")
    ax[1].plot(xs[:, 0], xs[:, 1], color=C["black"], lw=1.5, label="true track")
    ax[1].plot(kf["m"][:, 0], kf["m"][:, 1], color=C["orange"], label=f"filter (RMSE {rows2[1]['pos_rmse']:.2f})")
    ax[1].plot(sm["m"][:, 0], sm["m"][:, 1], color=C["blue"], label=f"smoother (RMSE {rows2[2]['pos_rmse']:.2f})")
    ax[1].set(title=f"2-D constant-velocity tracking (raw RMSE {rows2[0]['pos_rmse']:.2f})", xlabel="x",
              ylabel="y")
    ax[1].legend(fontsize=7)
    savefig(fig, "ch07_kalman_tracking")

    # ------------------------------------------------------------ figure 2: robustness heatmaps
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    lab = [f"{f:g}" for f in factors]
    panels = [(rm_f / rm_f[i0, i0], "filter RMSE / RMSE with true Q,R", "viridis", None),
              (np.log10(nees_g / 4.0), "log10(mean NEES / 4): + overconfident", "RdBu_r", (-2, 2)),
              (ll_g - ll_g.max(), "mean log marginal likelihood (vs max)", "magma", (-200, 0))]
    for a, (Z, title, cmap, lim) in zip(ax, panels):
        im = a.imshow(Z, origin="lower", cmap=cmap, vmin=None if lim is None else lim[0],
                      vmax=None if lim is None else lim[1], aspect="auto")
        a.set(xticks=range(len(factors)), xticklabels=lab, yticks=range(len(factors)), yticklabels=lab,
              xlabel="R scale factor", ylabel="Q scale factor", title=title)
        a.scatter([i0], [i0], marker="*", s=120, color="white", edgecolor="black", label="truth")
        plt.colorbar(im, ax=a, fraction=0.046)
        a.grid(False)
    ax[2].scatter([best_ll[1]], [best_ll[0]], marker="o", s=80, facecolor="none", edgecolor=C["sky"], lw=2,
                  label="argmax")
    ax[2].legend(fontsize=7, loc="lower left", frameon=True)
    for (i, j) in np.ndindex(rm_f.shape):
        ax[0].text(j, i, f"{rm_f[i, j] / rm_f[i0, i0]:.2f}", ha="center", va="center", fontsize=5,
                   color="white" if rm_f[i, j] / rm_f[i0, i0] < 1.5 else "black")
    savefig(fig, "ch07_misspecified_noise")
