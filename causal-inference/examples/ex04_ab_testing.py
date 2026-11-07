"""Example 4: A/B testing pitfalls. Peeking, CUPED, and interference.

Author: Abyn Scaria. MIT License.

(a) Peeking. In an A/A test (no true effect), checking a fixed-horizon z-test
    after every batch and stopping at the first p < 0.05 inflates the false
    positive rate well above 5%. A mixture sequential probability ratio test
    (mSPRT, as used for "always valid" p-values) keeps it below 5% under
    continuous monitoring.
(b) CUPED. Subtracting theta * (pre-period metric - its mean) from the outcome
    leaves the treatment effect unchanged but removes the variance that the
    pre-period metric explains. With correlation rho the variance falls by a
    factor (1 - rho^2).
(c) Interference. Units share a limited resource inside a market, so treated
    units take outcome from control units in the same market. Randomising
    individuals measures the wrong thing; randomising whole markets recovers
    the effect of launching to everyone.
"""
from common import PALETTE, plt, save_json, savefig  # noqa: I001

import numpy as np
from scipy import stats


# ----------------------------------------------------------------------------
# (a) Peeking vs always-valid inference
# ----------------------------------------------------------------------------
def msprt_lambda(diff_sum, n, sigma2, tau2):
    """Mixture SPRT statistic for H0: mean = 0 with N(0, tau2) mixing prior.

    diff_sum is the running sum of per-pair differences (treatment - control),
    each with variance sigma2. Reject when Lambda >= 1/alpha.
    """
    mean = diff_sum / n
    return np.sqrt(sigma2 / (sigma2 + n * tau2)) * np.exp(
        n ** 2 * tau2 * mean ** 2 / (2 * sigma2 * (sigma2 + n * tau2)))


def peeking(n_sims=4000, batch=100, n_looks=50, alpha=0.05, rng=None):
    sigma2 = 2.0  # per-pair difference of two unit-variance outcomes
    tau2 = 0.01   # mixing variance: tuned for effects of roughly 0.1 sd
    fixed_reject = peek_reject = msprt_reject = 0
    first_hits = []
    for _ in range(n_sims):
        d = rng.normal(0, np.sqrt(sigma2), size=batch * n_looks)  # A/A: no effect
        csum = np.cumsum(d)
        ns = np.arange(batch, batch * n_looks + 1, batch)
        z = csum[ns - 1] / np.sqrt(ns * sigma2)
        p = 2 * (1 - stats.norm.cdf(np.abs(z)))
        if p[-1] < alpha:
            fixed_reject += 1
        hits = np.where(p < alpha)[0]
        if len(hits):
            peek_reject += 1
            first_hits.append(int(hits[0]))
        lam = msprt_lambda(csum[ns - 1], ns, sigma2, tau2)
        if np.any(lam >= 1 / alpha):
            msprt_reject += 1
    res = {"fixed_horizon_fpr": fixed_reject / n_sims,
           "peek_every_batch_fpr": peek_reject / n_sims,
           "msprt_fpr": msprt_reject / n_sims,
           "n_looks": n_looks, "batch_pairs": batch, "n_sims": n_sims}
    # FPR as a function of how many looks you take
    looks_grid = [1, 2, 5, 10, 20, 50]
    fpr_by_looks = []
    for k in looks_grid:
        idx = np.unique(np.linspace(0, n_looks - 1, k).round().astype(int))
        # re-simulate cheaply using the same seed stream
        cnt = 0
        for _ in range(1000):
            d = rng.normal(0, np.sqrt(sigma2), size=batch * n_looks)
            csum = np.cumsum(d)
            ns = (idx + 1) * batch
            z = csum[ns - 1] / np.sqrt(ns * sigma2)
            cnt += np.any(2 * (1 - stats.norm.cdf(np.abs(z))) < alpha)
        fpr_by_looks.append(cnt / 1000)
    res["fpr_by_looks"] = dict(zip([str(k) for k in looks_grid], fpr_by_looks))
    return res, looks_grid, fpr_by_looks


# ----------------------------------------------------------------------------
# (b) CUPED
# ----------------------------------------------------------------------------
def cuped(n_sims=2000, n=2000, rho=0.7, effect=0.08, rng=None):
    plain, adj, plain_rej, adj_rej = [], [], 0, 0
    for _ in range(n_sims):
        X = rng.normal(size=n)                              # pre-period metric
        T = rng.binomial(1, 0.5, n)
        Y = rho * X + np.sqrt(1 - rho ** 2) * rng.normal(size=n) + effect * T
        theta = np.cov(X, Y)[0, 1] / np.var(X, ddof=1)       # pooled, treatment-independent
        Yc = Y - theta * (X - X.mean())
        for arr, store in ((Y, plain), (Yc, adj)):
            est = arr[T == 1].mean() - arr[T == 0].mean()
            se = np.sqrt(arr[T == 1].var(ddof=1) / (T == 1).sum() + arr[T == 0].var(ddof=1) / (T == 0).sum())
            store.append((est, se))
        plain_rej += abs(plain[-1][0] / plain[-1][1]) > 1.96
        adj_rej += abs(adj[-1][0] / adj[-1][1]) > 1.96
    plain, adj = np.array(plain), np.array(adj)
    return {"true_effect": effect, "rho": rho, "n": n,
            "plain_mean": plain[:, 0].mean(), "plain_sd": plain[:, 0].std(),
            "cuped_mean": adj[:, 0].mean(), "cuped_sd": adj[:, 0].std(),
            "variance_ratio": (adj[:, 0].var() / plain[:, 0].var()),
            "theory_variance_ratio": 1 - rho ** 2,
            "power_plain": plain_rej / n_sims, "power_cuped": adj_rej / n_sims}, plain, adj


# ----------------------------------------------------------------------------
# (c) Interference inside markets
# ----------------------------------------------------------------------------
def interference(n_sims=1000, n_markets=200, per_market=50, direct=1.0, spill=0.8, rng=None):
    """Y_i = a_market + direct*T_i - spill*(share of market treated) + noise.

    Treated units capture demand partly at the expense of their market peers.
    Global effect (everyone treated vs no one) = direct - spill = 0.2.
    """
    gte = direct - spill
    ind, clu = [], []
    for _ in range(n_sims):
        a = rng.normal(0, 1, n_markets)
        market = np.repeat(np.arange(n_markets), per_market)
        # individual-level randomisation: 50% within every market
        T = rng.binomial(1, 0.5, n_markets * per_market)
        share = np.bincount(market, T) / per_market
        Y = a[market] + direct * T - spill * share[market] + rng.normal(size=len(T))
        ind.append(Y[T == 1].mean() - Y[T == 0].mean())
        # cluster randomisation: whole markets treated or not
        Tm = rng.binomial(1, 0.5, n_markets)
        T = Tm[market]
        Y = a[market] + direct * T - spill * T + rng.normal(size=len(T))
        ybar = np.bincount(market, Y) / per_market
        clu.append(ybar[Tm == 1].mean() - ybar[Tm == 0].mean())
    return {"global_effect": gte, "direct_effect": direct,
            "individual_randomisation_mean": float(np.mean(ind)), "individual_sd": float(np.std(ind)),
            "cluster_randomisation_mean": float(np.mean(clu)), "cluster_sd": float(np.std(clu))}, ind, clu


def main():
    rng = np.random.default_rng(11)
    pk, looks_grid, fpr_by_looks = peeking(rng=rng)
    print("(a) peeking:", {k: v for k, v in pk.items() if k != "fpr_by_looks"})
    print("    FPR by number of looks:", pk["fpr_by_looks"])
    cu, plain, adj = cuped(rng=rng)
    print("(b) CUPED:", cu)
    it, ind, clu = interference(rng=rng)
    print("(c) interference:", it)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(looks_grid, fpr_by_looks, "o-", color=PALETTE[3], label="naive z-test, stop at first p<0.05")
    axes[0].axhline(0.05, color="k", ls="--", label="nominal 5%")
    axes[0].axhline(pk["msprt_fpr"], color=PALETTE[2], ls="-",
                    label=f"mSPRT, checked all {pk['n_looks']} looks ({pk['msprt_fpr']:.3f})")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("number of interim looks (A/A test)")
    axes[0].set_ylabel("false positive rate")
    axes[0].set_title("(a) Peeking inflates false positives")
    axes[0].legend(fontsize=7)

    bins = np.linspace(-0.15, 0.3, 60)
    axes[1].hist(plain[:, 0], bins=bins, alpha=0.6, color=PALETTE[0], label=f"plain diff (sd {cu['plain_sd']:.3f})")
    axes[1].hist(adj[:, 0], bins=bins, alpha=0.6, color=PALETTE[1], label=f"CUPED (sd {cu['cuped_sd']:.3f})")
    axes[1].axvline(cu["true_effect"], color="k", ls="--", label="true effect")
    axes[1].set_title(f"(b) CUPED, rho={cu['rho']}: power {cu['power_plain']:.2f} -> {cu['power_cuped']:.2f}")
    axes[1].set_xlabel("estimated effect")
    axes[1].legend(fontsize=7)

    bins = np.linspace(-0.2, 1.3, 60)
    axes[2].hist(ind, bins=bins, alpha=0.6, color=PALETTE[3], label="randomise individuals")
    axes[2].hist(clu, bins=bins, alpha=0.6, color=PALETTE[2], label="randomise markets")
    axes[2].axvline(it["global_effect"], color="k", ls="--", label="global effect of launch")
    axes[2].set_title("(c) Interference: unit-level A/B overstates")
    axes[2].set_xlabel("estimated effect")
    axes[2].legend(fontsize=7)
    savefig(fig, "ex04_ab_testing")
    save_json({"peeking": pk, "cuped": cu, "interference": it}, "ex04_ab_testing")


if __name__ == "__main__":
    main()
