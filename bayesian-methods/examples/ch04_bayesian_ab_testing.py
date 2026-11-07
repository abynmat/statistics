"""Chapter 4: Bayesian A/B testing as a decision problem.

Author: Abyn Scaria. MIT License.

Parts
  1. One experiment: Beta-binomial posteriors, P(B > A), expected loss of each choice, both by
     Monte Carlo and by exact 1-D quadrature, next to a two-proportion z-test.
  2. Peeking under the null (A = B): how often does each rule declare a winner if you look
     40 times? Shown for the p-value AND for P(B > A) -- the posterior also drifts.
  3. Decision rule: stop when the expected loss of the better-looking arm falls below a
     threshold eps. Over many experiments with true effects drawn from a prior, measure the
     sample size, the chance of shipping the worse arm and the regret, for several eps.

All data are simulated; conversion rates are made-up round numbers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from scipy import integrate, stats  # noqa: E402
from scipy.special import betainc  # noqa: E402

from bayeslab.plotting import C, plt, print_table, save_csv, save_json, savefig  # noqa: E402

SEED = 404
PRIOR = (1.0, 1.0)


def exact_prob_b_gt_a(aA, bA, aB, bB):
    """P(pB > pA) = integral f_B(x) F_A(x) dx for independent Beta posteriors."""
    fB, FA = stats.beta(aB, bB).pdf, stats.beta(aA, bA).cdf
    lo, hi = stats.beta(aB, bB).ppf([1e-10, 1 - 1e-10])
    return integrate.quad(lambda x: fB(x) * FA(x), lo, hi, limit=200)[0]


def exact_expected_loss(a1, b1, a2, b2):
    """E[max(p1 - p2, 0)]: the loss of choosing arm 2 when arm 1 might be better.

    Uses E[(a - p2)^+] = a F2(a) - mean2 * I_a(a2 + 1, b2), then integrates over p1.
    """
    d1 = stats.beta(a1, b1)
    m2 = a2 / (a2 + b2)

    def inner(a):
        return a * betainc(a2, b2, a) - m2 * betainc(a2 + 1, b2, a)

    lo, hi = d1.ppf([1e-10, 1 - 1e-10])
    return integrate.quad(lambda a: d1.pdf(a) * inner(a), lo, hi, limit=200)[0]


def z_test(kA, nA, kB, nB):
    p = (kA + kB) / (nA + nB)
    se = np.sqrt(p * (1 - p) * (1 / nA + 1 / nB))
    z = (kB / nB - kA / nA) / se
    return z, 2 * stats.norm.sf(np.abs(z))


def posterior_stats(kA, nA, kB, nB, rng, n_draws):
    """Vectorised over experiments: P(B>A), loss of shipping B, loss of keeping A."""
    aA, bA = PRIOR[0] + kA, PRIOR[1] + nA - kA
    aB, bB = PRIOR[0] + kB, PRIOR[1] + nB - kB
    pa = rng.beta(aA[:, None], bA[:, None], size=(kA.size, n_draws))
    pb = rng.beta(aB[:, None], bB[:, None], size=(kA.size, n_draws))
    diff = pb - pa
    return (diff > 0).mean(1), np.maximum(-diff, 0).mean(1), np.maximum(diff, 0).mean(1)


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)

    # ------------------------------------------------------------ 1. one experiment
    pA_true, pB_true, n = 0.100, 0.112, 6000
    xa = rng.uniform(size=n) < pA_true
    xb = rng.uniform(size=n) < pB_true
    kA, kB = int(xa.sum()), int(xb.sum())
    aA, bA, aB, bB = PRIOR[0] + kA, PRIOR[1] + n - kA, PRIOR[0] + kB, PRIOR[1] + n - kB
    draws_a = rng.beta(aA, bA, 400000)
    draws_b = rng.beta(aB, bB, 400000)
    d = draws_b - draws_a
    z, pval = z_test(kA, n, kB, n)
    se = np.sqrt(kA / n * (1 - kA / n) / n + kB / n * (1 - kB / n) / n)
    one = [
        {"quantity": "P(B > A)", "monte_carlo": float((d > 0).mean()), "exact_quadrature": exact_prob_b_gt_a(aA, bA, aB, bB)},
        {"quantity": "expected loss if ship B", "monte_carlo": float(np.maximum(-d, 0).mean()),
         "exact_quadrature": exact_expected_loss(aA, bA, aB, bB)},
        {"quantity": "expected loss if keep A", "monte_carlo": float(np.maximum(d, 0).mean()),
         "exact_quadrature": exact_expected_loss(aB, bB, aA, bA)},
    ]
    print(f"one experiment: A {kA}/{n} = {kA / n:.4f},  B {kB}/{n} = {kB / n:.4f}")
    print_table(one, fmt="{:.6f}")
    ci_bayes = np.quantile(d, [0.025, 0.975])
    diff_hat = kB / n - kA / n
    freq = {"z": z, "p_value_two_sided": pval, "diff_hat": diff_hat,
            "wald_ci": [diff_hat - 1.96 * se, diff_hat + 1.96 * se], "bayes_cri": ci_bayes.tolist(),
            "relative_lift_post_mean": float(np.mean(d / draws_a)),
            "P(lift > 5%)": float(np.mean(d / draws_a > 0.05))}
    print(f"z-test: z={z:.3f}, p={pval:.4f};  95% Wald CI {freq['wald_ci'][0]:.4f}..{freq['wald_ci'][1]:.4f};"
          f"  95% credible {ci_bayes[0]:.4f}..{ci_bayes[1]:.4f}")
    save_csv(one, "ch04_single_experiment")
    # trajectory of the same experiment as data accrue
    looks = np.arange(200, n + 1, 200)
    traj = {"n": looks.tolist(), "p_b_gt_a": [], "loss_ship_b": [], "loss_keep_a": [], "p_value": []}
    for m in looks:
        ka, kb = np.array([xa[:m].sum()]), np.array([xb[:m].sum()])
        pr, lb, la = posterior_stats(ka, np.array([m]), kb, np.array([m]), rng, 40000)
        traj["p_b_gt_a"].append(float(pr[0]))
        traj["loss_ship_b"].append(float(lb[0]))
        traj["loss_keep_a"].append(float(la[0]))
        traj["p_value"].append(float(z_test(ka[0], m, kb[0], m)[1]))

    # ------------------------------------------------------------ 2. peeking under the null
    n_sims, look_every, n_looks, n_post = 2000, 500, 40, 1000
    p0 = 0.10
    incA = rng.binomial(look_every, p0, size=(n_sims, n_looks)).cumsum(1)
    incB = rng.binomial(look_every, p0, size=(n_sims, n_looks)).cumsum(1)
    ever_p = np.zeros(n_sims, bool)
    ever_bayes = np.zeros(n_sims, bool)
    ever_loss = {e: np.zeros(n_sims, bool) for e in (0.0002, 0.0005, 0.001)}
    fp_by_look = []
    for j in range(n_looks):
        m = (j + 1) * look_every
        _, pv = z_test(incA[:, j], m, incB[:, j], m)
        prob, lb, la = posterior_stats(incA[:, j], np.full(n_sims, m), incB[:, j], np.full(n_sims, m), rng, n_post)
        ever_p |= pv < 0.05
        ever_bayes |= (prob > 0.975) | (prob < 0.025)
        for e in ever_loss:
            ever_loss[e] |= np.minimum(lb, la) < e
        fp_by_look.append({"looks": j + 1, "n_per_arm": m, "pvalue_any_look": float(ever_p.mean()),
                           "bayes_any_look": float(ever_bayes.mean()),
                           "pvalue_this_look_only": float((pv < 0.05).mean()),
                           "bayes_this_look_only": float(((prob > 0.975) | (prob < 0.025)).mean())})
    print("\nPeeking under the null (A = B = 10%), 2000 experiments, a look every 500 users per arm:")
    print_table([r for r in fp_by_look if r["looks"] in (1, 2, 5, 10, 20, 40)])
    save_csv(fp_by_look, "ch04_peeking_null")
    stop_null = {str(e): float(v.mean()) for e, v in ever_loss.items()}
    print("fraction of null experiments stopped by the expected-loss rule before 20k/arm:", stop_null)

    # ------------------------------------------------------------ 3. expected-loss decision rule
    n_sims = 3000
    lift = rng.normal(0.0, 0.05, n_sims)                     # true relative lift ~ N(0, 5%)
    pA = np.full(n_sims, 0.10)
    pB = np.clip(pA * (1 + lift), 1e-4, 1 - 1e-4)
    incA = rng.binomial(look_every, pA[:, None], size=(n_sims, n_looks)).cumsum(1)
    incB = rng.binomial(look_every, pB[:, None], size=(n_sims, n_looks)).cumsum(1)
    eps_list = [0.00005, 0.0001, 0.0002, 0.0005, 0.001]
    stopped = {e: np.full(n_sims, -1) for e in eps_list}
    choice = {e: np.zeros(n_sims, int) for e in eps_list}   # 1 = ship B
    prob_end = None
    for j in range(n_looks):
        m = (j + 1) * look_every
        prob, lb, la = posterior_stats(incA[:, j], np.full(n_sims, m), incB[:, j], np.full(n_sims, m), rng, n_post)
        best_loss = np.minimum(lb, la)
        pick_b = lb < la
        for e in eps_list:
            now = (stopped[e] < 0) & ((best_loss < e) | (j == n_looks - 1))
            stopped[e][now] = m
            choice[e][now] = pick_b[now]
        if j == n_looks - 1:
            prob_end = prob
    # Fixed-horizon frequentist: run to 20k/arm, ship B only if significantly better.
    _, pv_end = z_test(incA[:, -1], n_looks * look_every, incB[:, -1], n_looks * look_every)
    freq_choice = ((pv_end < 0.05) & (incB[:, -1] > incA[:, -1])).astype(int)
    best = np.maximum(pA, pB)

    def evaluate(name, ch, n_used):
        shipped = np.where(ch == 1, pB, pA)
        regret = best - shipped
        return {"rule": name, "mean_n_per_arm": float(n_used.mean()),
                "P_ship_worse": float(np.mean(shipped < best - 1e-12)),
                "mean_regret_bp": float(1e4 * regret.mean()),
                "P_ship_B_when_B_worse": float(np.mean(ch[pB < pA] == 1)),
                "P_ship_B_when_B_better_by_5pct+": float(np.mean(ch[lift > 0.05] == 1))}

    dec_rows = [evaluate(f"expected loss < {e * 1e4:g} bp", choice[e], stopped[e]) for e in eps_list]
    dec_rows.append(evaluate("fixed 20k/arm, z-test p<0.05", freq_choice, np.full(n_sims, 20000)))
    dec_rows.append(evaluate("fixed 20k/arm, ship if P(B>A)>0.5", (prob_end > 0.5).astype(int), np.full(n_sims, 20000)))
    print("\nDecision rules over 3000 experiments, true lift ~ N(0, 5%) on a 10% baseline (bp = 0.01 pp):")
    print_table(dec_rows)
    save_csv(dec_rows, "ch04_decision_rules")
    save_json({"single": one, "frequentist": freq, "null_stop_by_loss_rule": stop_null}, "ch04_summary")

    # ------------------------------------------------------------ figure 1
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    xx = np.linspace(0.085, 0.13, 400)
    ax[0].plot(xx, stats.beta(aA, bA).pdf(xx), color=C["blue"], label=f"A: {kA}/{n}")
    ax[0].plot(xx, stats.beta(aB, bB).pdf(xx), color=C["orange"], label=f"B: {kB}/{n}")
    ax[0].axvline(pA_true, color=C["blue"], ls=":", lw=1)
    ax[0].axvline(pB_true, color=C["orange"], ls=":", lw=1)
    ax[0].set(title=f"Posteriors; P(B>A) = {one[0]['exact_quadrature']:.3f}", xlabel="conversion rate")
    ax[0].legend()
    ax[1].hist(d * 1e4, bins=80, color=C["sky"], density=True)
    ax[1].axvline(0, color=C["black"], lw=1)
    ax[1].set(title=f"pB - pA (bp); loss if ship B = {one[1]['exact_quadrature'] * 1e4:.2f} bp",
              xlabel="difference in basis points")
    nn = np.array(traj["n"])
    ax[2].semilogy(nn, np.array(traj["loss_ship_b"]) * 1e4, color=C["orange"], label="loss if ship B (bp)")
    ax[2].semilogy(nn, np.array(traj["loss_keep_a"]) * 1e4, color=C["blue"], label="loss if keep A (bp)")
    ax[2].axhline(1.0, color=C["red"], ls=":", label="threshold 1 bp")
    ax2 = ax[2].twinx()
    ax2.plot(nn, traj["p_b_gt_a"], color=C["grey"], lw=1, label="P(B>A)")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("P(B > A)", color=C["grey"])
    ax2.grid(False)
    ax[2].set(xlabel="users per arm", title="Same experiment as data arrive")
    ax[2].legend(loc="lower left")
    savefig(fig, "ch04_ab_posterior_and_loss")

    # ------------------------------------------------------------ figure 2
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    L = [r["looks"] for r in fp_by_look]
    ax[0].plot(L, [r["pvalue_any_look"] for r in fp_by_look], "o-", ms=3, color=C["orange"],
               label="p < 0.05 at any look")
    ax[0].plot(L, [r["bayes_any_look"] for r in fp_by_look], "s-", ms=3, color=C["blue"],
               label="P(B>A) > 0.975 or < 0.025 at any look")
    ax[0].axhline(0.05, color=C["black"], ls=":", label="nominal 5%")
    ax[0].set(xlabel="number of looks (every 500 users/arm)", ylabel="P(declare a winner | A = B)",
              title="Peeking under the null: both rules drift")
    ax[0].legend()
    for r, col in zip(dec_rows, [C["sky"], C["blue"], C["green"], C["orange"], C["red"], C["black"], C["grey"]]):
        mk = "o" if r["rule"].startswith("expected") else ("s" if "z-test" in r["rule"] else "^")
        ax[1].scatter(r["mean_n_per_arm"], r["mean_regret_bp"], s=50, color=col, marker=mk, label=r["rule"])
    ax[1].set(xlabel="mean users per arm", ylabel="mean regret (bp of conversion)",
              title="Sample size vs regret, true lift ~ N(0, 5%)")
    ax[1].legend(fontsize=7)
    savefig(fig, "ch04_peeking_and_decisions")
