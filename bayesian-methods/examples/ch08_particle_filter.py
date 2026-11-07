"""Chapter 8: the bootstrap particle filter, against the EKF and against the exact Kalman answer.

Author: Abyn Scaria. MIT License.

Parts
  1. Exactness check on a linear-Gaussian model, where the Kalman filter is the exact answer:
     PF filtering-mean error and log-likelihood error vs number of particles N.
  2. Nonlinear benchmark (the univariate growth model of Kitagawa 1996 / Gordon et al. 1993):
         x_t = x/2 + 25 x / (1 + x^2) + 8 cos(1.2 t) + v_t,  v_t ~ N(0, 10)
         y_t = x_t^2 / 20 + w_t,                          w_t ~ N(0, 1)
     y depends on x^2, so the posterior is often bimodal in the sign of x. EKF vs PF RMSE.
  3. Degeneracy: weight ESS over time with no resampling, adaptive resampling and resampling
     at every step; and the four resampling schemes compared by the variance of the
     log-likelihood estimate.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from bayeslab import filters as fl  # noqa: E402
from bayeslab.plotting import C, plt, print_table, save_csv, save_json, savefig  # noqa: E402

SEED = 808
QV, RW = 10.0, 1.0


def f_ng(x, t):
    return 0.5 * x + 25 * x / (1 + x ** 2) + 8 * np.cos(1.2 * (t + 1))


def simulate_ng(T, rng):
    x, xs, ys = rng.normal(0, np.sqrt(5)), np.empty(T), np.empty(T)
    for t in range(T):
        x = f_ng(x, t) + rng.normal(0, np.sqrt(QV))
        xs[t] = x
        ys[t] = x ** 2 / 20 + rng.normal(0, np.sqrt(RW))
    return xs, ys


def pf_ng(ys, N, rng, scheme="systematic", ess_threshold=0.5):
    return fl.bootstrap_pf(
        ys[:, None],
        lambda n, g: g.normal(0, np.sqrt(5), (n, 1)),
        lambda x, t, g: f_ng(x, t) + g.normal(0, np.sqrt(QV), x.shape),
        lambda y, x, t: -0.5 * (y[0] - x[:, 0] ** 2 / 20) ** 2 / RW,
        N, rng, scheme=scheme, ess_threshold=ess_threshold)


def ekf_ng(ys):
    return fl.ekf(ys[:, None],
                  lambda m, t: f_ng(m, t),
                  lambda m, t: np.array([[0.5 + 25 * (1 - m[0] ** 2) / (1 + m[0] ** 2) ** 2]]),
                  lambda m, t: m ** 2 / 20,
                  lambda m, t: np.array([[m[0] / 10]]),
                  np.eye(1) * QV, np.eye(1) * RW, np.zeros(1), np.eye(1) * 5.0)


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)

    # ------------------------------------------------------------ 1. PF vs exact KF
    phi, q, r, T = 0.9, 0.5, 1.0, 100
    F, H, Q, R = np.eye(1) * phi, np.eye(1), np.eye(1) * q, np.eye(1) * r
    xs, ys = fl.simulate_lgssm(F, H, Q, R, np.zeros(1), T, rng)
    kf = fl.kalman_filter(ys, F, H, Q, R, np.zeros(1), np.eye(1))
    lin_rows = []
    for N in [50, 100, 300, 1000, 3000, 10000]:
        errs, lls = [], []
        t0 = time.perf_counter()
        for rep in range(40):
            pf = fl.bootstrap_pf(ys, lambda n, g: g.standard_normal((n, 1)),
                                 lambda x, t, g: phi * x + np.sqrt(q) * g.standard_normal(x.shape),
                                 lambda y, x, t: -0.5 * (y[0] - x[:, 0]) ** 2 / r - 0.5 * np.log(2 * np.pi * r),
                                 N, rng)
            errs.append(np.sqrt(np.mean((pf["m"][:, 0] - kf["m"][:, 0]) ** 2)))
            lls.append(pf["loglik"])
        lls = np.array(lls)
        lin_rows.append({"N": N, "rmse_pf_vs_kf_mean": float(np.mean(errs)),
                         "loglik_bias": float(lls.mean() - kf["loglik"]), "loglik_sd": float(lls.std()),
                         "ms_per_run": 1000 * (time.perf_counter() - t0) / 40})
    print(f"PF vs exact Kalman filter on a linear-Gaussian AR(1) model, T={T}, 40 reps per N "
          f"(exact log-lik {kf['loglik']:.3f}):")
    print_table(lin_rows)
    save_csv(lin_rows, "ch08_pf_vs_kalman")

    # ------------------------------------------------------------ 2. nonlinear: EKF vs PF
    T = 100
    ng_rows = {"EKF": [], "PF N=100": [], "PF N=1000": [], "PF N=10000": []}
    example = None
    for rep in range(50):
        xt, yt = simulate_ng(T, rng)
        e = ekf_ng(yt)
        ng_rows["EKF"].append(np.sqrt(np.mean((e["m"][:, 0] - xt) ** 2)))
        for N in (100, 1000, 10000):
            p = pf_ng(yt, N, rng)
            ng_rows[f"PF N={N}"].append(np.sqrt(np.mean((p["m"][:, 0] - xt) ** 2)))
            if rep == 0 and N == 1000:
                example = (xt, yt, e, p)
    ng_tab = [{"filter": k, "rmse_mean": float(np.mean(v)), "rmse_median": float(np.median(v)),
               "rmse_sd": float(np.std(v))} for k, v in ng_rows.items()]
    print("\nNonlinear growth model, 50 simulated series of length 100:")
    print_table(ng_tab)
    save_csv(ng_tab, "ch08_ekf_vs_pf")

    # ------------------------------------------------------------ 3. degeneracy + schemes
    xt, yt = simulate_ng(T, np.random.default_rng(SEED + 1))
    N = 500
    ess_tr = {}
    for name, thr in (("never resample", 0.0), ("resample if ESS < N/2", 0.5), ("resample every step", 1.01)):
        p = pf_ng(yt, N, np.random.default_rng(SEED), ess_threshold=thr)
        ess_tr[name] = (p["ess"], np.sqrt(np.mean((p["m"][:, 0] - xt) ** 2)), p["n_resample"])
    deg_rows = [{"policy": k, "rmse": v[1], "median_weight_ess": float(np.median(v[0])),
                 "min_weight_ess": float(v[0].min()), "resample_events": v[2]} for k, v in ess_tr.items()]
    print(f"\nDegeneracy, N={N}:")
    print_table(deg_rows)
    save_csv(deg_rows, "ch08_degeneracy")
    ref_ll = np.mean([pf_ng(yt, 50000, np.random.default_rng(SEED + 99 + i))["loglik"] for i in range(3)])
    sch_rows = []
    for scheme in ("multinomial", "residual", "stratified", "systematic"):
        lls, rm = [], []
        for rep in range(100):
            p = pf_ng(yt, N, np.random.default_rng([SEED, rep]), scheme=scheme, ess_threshold=1.01)
            lls.append(p["loglik"])
            rm.append(np.sqrt(np.mean((p["m"][:, 0] - xt) ** 2)))
        sch_rows.append({"scheme": scheme, "loglik_mean_minus_ref": float(np.mean(lls) - ref_ll),
                         "loglik_sd": float(np.std(lls)), "rmse_mean": float(np.mean(rm))})
    print(f"\nResampling schemes (resample every step, N={N}, 100 reps; reference log-lik from 3 x N=50000):")
    print_table(sch_rows)
    save_csv(sch_rows, "ch08_resampling_schemes")
    save_json({"kalman_loglik": kf["loglik"], "reference_loglik_ng": float(ref_ll)}, "ch08_summary")

    # ------------------------------------------------------------ figure 1
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    xt0, yt0, e0, p0 = example
    tt = np.arange(T)
    ax[0].plot(tt, xt0, color=C["black"], lw=1.5, label="true x")
    ax[0].plot(tt, e0["m"][:, 0], color=C["red"], lw=1, label="EKF mean")
    sd = np.sqrt(p0["var"][:, 0])
    ax[0].plot(tt, p0["m"][:, 0], color=C["blue"], lw=1, label="PF mean (N=1000)")
    ax[0].fill_between(tt, p0["m"][:, 0] - 2 * sd, p0["m"][:, 0] + 2 * sd, color=C["blue"], alpha=0.15,
                       label="PF +-2 sd")
    ax[0].set(xlabel="time", title="Nonlinear growth model: EKF loses the sign of x", ylim=(-35, 35))
    ax[0].legend(fontsize=7, ncol=2)
    Ns = [rw["N"] for rw in lin_rows]
    ax[1].loglog(Ns, [rw["rmse_pf_vs_kf_mean"] for rw in lin_rows], "o-", color=C["blue"],
                 label="RMSE(PF mean - exact KF mean)")
    ax[1].loglog(Ns, [rw["loglik_sd"] for rw in lin_rows], "s-", color=C["orange"], label="sd of PF log-lik")
    ref = lin_rows[0]["rmse_pf_vs_kf_mean"]
    ax[1].loglog(Ns, ref * (np.array(Ns) / Ns[0]) ** -0.5, color=C["grey"], ls="--", label="N^-1/2")
    ax[1].set(xlabel="particles N", title="Linear-Gaussian check: PF converges to the exact answer")
    ax[1].legend(fontsize=7)
    savefig(fig, "ch08_pf_vs_ekf_and_convergence")

    # ------------------------------------------------------------ figure 2
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for (name, (ess, rm, nr)), col in zip(ess_tr.items(), (C["red"], C["blue"], C["green"])):
        ax[0].semilogy(ess, color=col, lw=1, label=f"{name} (RMSE {rm:.2f})")
    ax[0].set(xlabel="time", ylabel="weight ESS (of 500)", title="Weight degeneracy")
    ax[0].legend(fontsize=7)
    xx = np.arange(len(sch_rows))
    ax[1].bar(xx, [rw["loglik_sd"] for rw in sch_rows], color=[C["grey"], C["orange"], C["sky"], C["blue"]])
    ax[1].set(xticks=xx, xticklabels=[rw["scheme"] for rw in sch_rows], ylabel="sd of log-likelihood estimate",
              title=f"Resampling schemes, N={N}, 100 runs")
    savefig(fig, "ch08_degeneracy_and_resampling")
