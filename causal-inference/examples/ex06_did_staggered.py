"""Example 6: staggered difference-in-differences. Why TWFE breaks and what fixes it.

Author: Abyn Scaria. MIT License.

Balanced panel: 600 units x 20 periods. Units adopt an absorbing treatment at
different times (cohorts g = 5, 10, 15) and 100 units are never treated.
Parallel trends hold by construction (unit FE + common time shocks), so every
failure below comes from treatment-effect heterogeneity, not from confounding.

Treatment effects differ by cohort and grow with time since adoption e = t - g:
    g = 5 : tau = 0.5 + 0.40 * e      (early adopters, effect grows fast)
    g = 10: tau = 1.0 + 0.20 * e
    g = 15: tau = 2.0 + 0.10 * e      (late adopters, big immediate effect)

Estimators:
  * Static TWFE: Y ~ D + unit FE + time FE
  * Goodman-Bacon style 2x2 comparisons (shows the "forbidden" comparisons that use
    already-treated units as controls)
  * Dynamic TWFE event study (leads and lags pooled across cohorts)
  * Callaway & Sant'Anna ATT(g,t), never-treated and not-yet-treated controls
  * Sun & Abraham interaction-weighted event study
  * Borusyak-Jaravel-Spiess imputation estimator
"""
from common import PALETTE, plt, save_json, savefig  # noqa: I001

import numpy as np

N_T = 20
COHORTS = {5: 150, 10: 200, 15: 150, 0: 100}  # 0 = never treated
EFFECT = {5: (0.5, 0.40), 10: (1.0, 0.20), 15: (2.0, 0.10)}


def tau_fn(g, t):
    if g == 0 or t < g:
        return 0.0
    a, b = EFFECT[g]
    return a + b * (t - g)


def simulate(rng):
    g = np.concatenate([np.full(n, gg) for gg, n in COHORTS.items()])
    N = len(g)
    alpha = rng.normal(0, 1, N) + 0.3 * (g == 5) - 0.2 * (g == 15)  # cohorts differ in levels: fine for DiD
    lam = np.cumsum(rng.normal(0.1, 0.3, N_T))
    t = np.arange(1, N_T + 1)
    D = ((g[:, None] > 0) & (t[None, :] >= g[:, None])).astype(float)
    TAU = np.vectorize(tau_fn)(g[:, None], t[None, :])
    Y = alpha[:, None] + lam[None, :] + TAU + rng.normal(0, 0.5, (N, N_T))
    return g, D, Y, TAU


def twoway_demean(M):
    """Within transformation for a balanced N x T panel (exact two-way FE)."""
    return M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()


def twfe_static(D, Y):
    d, y = twoway_demean(D).ravel(), twoway_demean(Y).ravel()
    return float(d @ y / (d @ d))


def twfe_event_study(g, Y, leads=8, lags=10):
    """Y ~ sum_k beta_k 1[t - g = k] + unit FE + time FE, k = -1 omitted, ends binned."""
    N = len(g)
    t = np.arange(1, N_T + 1)
    ks = [k for k in range(-leads, lags + 1) if k != -1]
    rel = t[None, :] - g[:, None]
    cols = []
    for k in ks:
        if k == -leads:
            m = (rel <= k)
        elif k == lags:
            m = (rel >= k)
        else:
            m = (rel == k)
        m = m & (g[:, None] > 0)
        cols.append(twoway_demean(m.astype(float)).ravel())
    Xd = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(Xd, twoway_demean(Y).ravel(), rcond=None)
    return dict(zip(ks, beta))


def cs_attgt(g, Y, control="never"):
    """Callaway & Sant'Anna ATT(g,t) with the unconditional (no covariates) 2x2 estimator.

    Base period is g-1 (the 'varying' base for pre-periods, as in the did package
    default, uses t-1; we use the universal g-1 base for both pre and post here).
    """
    out = {}
    t_idx = {tt: tt - 1 for tt in range(1, N_T + 1)}
    for gg in [k for k in COHORTS if k > 0]:
        treat = g == gg
        base = t_idx[gg - 1]
        for tt in range(1, N_T + 1):
            if tt == gg - 1:
                continue
            if control == "never":
                ctrl = g == 0
            else:  # not yet treated by max(t, g-1), excluding own cohort
                ctrl = ((g == 0) | (g > max(tt, gg - 1))) & ~treat
            if ctrl.sum() == 0:
                continue
            att = (Y[treat, t_idx[tt]] - Y[treat, base]).mean() - (Y[ctrl, t_idx[tt]] - Y[ctrl, base]).mean()
            out[(gg, tt)] = (att, int(treat.sum()))
    return out


def aggregate_overall(att, g):
    """Average of post-treatment ATT(g,t), weighted by cohort size (the 'simple' aggregation)."""
    num = den = 0.0
    for (gg, tt), (a, n) in att.items():
        if tt >= gg:
            num += a * n
            den += n
    return num / den


def aggregate_event(att, emin=-8, emax=10):
    es = {}
    for e in range(emin, emax + 1):
        cells = [(a, n) for (gg, tt), (a, n) in att.items() if tt - gg == e]
        if cells:
            es[e] = sum(a * n for a, n in cells) / sum(n for _, n in cells)
    return es


def sun_abraham(g, Y, emin=-8, emax=10):
    """Interaction-weighted estimator: cohort x relative-time dummies in a TWFE regression,
    never-treated units as the control cohort, then average across cohorts with
    cohort-share weights."""
    t = np.arange(1, N_T + 1)
    rel = t[None, :] - g[:, None]
    cols, keys = [], []
    for gg in [k for k in COHORTS if k > 0]:
        for e in range(1 - gg, N_T + 1 - gg):
            if e == -1:
                continue
            m = ((g[:, None] == gg) & (rel == e)).astype(float)
            cols.append(twoway_demean(m).ravel())
            keys.append((gg, e))
    beta, *_ = np.linalg.lstsq(np.column_stack(cols), twoway_demean(Y).ravel(), rcond=None)
    cate = dict(zip(keys, beta))
    es = {}
    for e in range(emin, emax + 1):
        cells = [(cate[(gg, e)], COHORTS[gg]) for gg in COHORTS if gg > 0 and (gg, e) in cate]
        if cells:
            es[e] = sum(b * n for b, n in cells) / sum(n for _, n in cells)
    return es, cate


def imputation(g, D, Y, n_iter=200):
    """Borusyak-Jaravel-Spiess: fit unit + time FE on untreated cells only, impute Y(0)
    for treated cells, average Y - Y(0)_hat over treated cells."""
    mask = D == 0
    a = np.zeros(Y.shape[0])
    lam = np.zeros(Y.shape[1])
    for _ in range(n_iter):  # alternating projections on the untreated cells
        R = np.where(mask, Y - lam[None, :], 0.0)
        a = R.sum(1) / mask.sum(1)
        R = np.where(mask, Y - a[:, None], 0.0)
        lam = R.sum(0) / mask.sum(0)
    Y0 = a[:, None] + lam[None, :]
    te = (Y - Y0)[D == 1]
    t = np.arange(1, N_T + 1)
    rel = t[None, :] - g[:, None]
    es = {int(e): float((Y - Y0)[(D == 1) & (rel == e)].mean()) for e in range(0, N_T)
          if ((D == 1) & (rel == e)).any()}
    return float(te.mean()), es


def bacon_2x2(g, Y):
    """Simple 2x2 DiDs between timing groups (Goodman-Bacon 2021 building blocks).

    For early cohort k and later cohort l:
      * k vs never, l vs never           (clean)
      * k vs l, using periods before l adopts       (clean: l not yet treated)
      * l vs k, using periods after k adopted        ('forbidden': k already treated)
    """
    t = np.arange(1, N_T + 1)
    res = {}

    def did(tr, ct, pre, post):
        yt = Y[g == tr][:, np.isin(t, post)].mean() - Y[g == tr][:, np.isin(t, pre)].mean()
        yc = Y[g == ct][:, np.isin(t, post)].mean() - Y[g == ct][:, np.isin(t, pre)].mean()
        return float(yt - yc)

    gs = sorted(k for k in COHORTS if k > 0)
    for k in gs:
        res[f"g={k} vs never"] = did(k, 0, t[t < k], t[t >= k])
    for i, k in enumerate(gs):
        for l in gs[i + 1:]:
            res[f"g={k} vs g={l} (before {l} treated)"] = did(k, l, t[t < k], t[(t >= k) & (t < l)])
            res[f"g={l} vs g={k} (k already treated) [forbidden]"] = did(l, k, t[(t >= k) & (t < l)], t[t >= l])
    return res


def main(n_sims=200):
    rng = np.random.default_rng(2024)
    keys = ["TWFE static", "CS (never-treated)", "CS (not-yet-treated)", "Imputation (BJS)"]
    draws = {k: [] for k in keys}
    truths = []
    for s in range(n_sims):
        g, D, Y, TAU = simulate(rng)
        truths.append(TAU[D == 1].mean())
        draws["TWFE static"].append(twfe_static(D, Y))
        draws["CS (never-treated)"].append(aggregate_overall(cs_attgt(g, Y, "never"), g))
        draws["CS (not-yet-treated)"].append(aggregate_overall(cs_attgt(g, Y, "notyet"), g))
        draws["Imputation (BJS)"].append(imputation(g, D, Y)[0])
    truth = float(np.mean(truths))
    summary = {k: {"mean": float(np.mean(v)), "bias": float(np.mean(v) - truth), "sd": float(np.std(v))}
               for k, v in draws.items()}
    print(f"True ATT (avg over treated cells) = {truth:.3f}")
    for k, v in summary.items():
        print(f"  {k:<24} mean {v['mean']:.3f}  bias {v['bias']:+.3f}  sd {v['sd']:.3f}")

    # One detailed dataset for event studies + Bacon pieces
    g, D, Y, TAU = simulate(np.random.default_rng(7))
    es_twfe = twfe_event_study(g, Y)
    es_cs = aggregate_event(cs_attgt(g, Y, "never"))
    es_sa, _ = sun_abraham(g, Y)
    _, es_imp = imputation(g, D, Y)
    t = np.arange(1, N_T + 1)
    rel = t[None, :] - g[:, None]
    es_true = {e: float(TAU[(g[:, None] > 0) & (rel == e)].mean()) for e in range(-8, 11)
               if ((g[:, None] > 0) & (rel == e)).any()}
    bacon = bacon_2x2(g, Y)
    print("\n2x2 building blocks (one dataset):")
    for k, v in bacon.items():
        print(f"  {k:<48} {v:+.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    ks = sorted(es_true)
    axes[0].plot(ks, [es_true[k] for k in ks], "k--", lw=2, label="true average effect by event time")
    axes[0].plot(sorted(es_twfe), [es_twfe[k] for k in sorted(es_twfe)], "o-", color=PALETTE[3],
                 label="TWFE dynamic (pooled leads/lags)")
    axes[0].plot(sorted(es_cs), [es_cs[k] for k in sorted(es_cs)], "s-", color=PALETTE[0], label="Callaway-Sant'Anna")
    axes[0].plot(sorted(es_sa), [es_sa[k] for k in sorted(es_sa)], "^-", color=PALETTE[2], alpha=0.8,
                 label="Sun-Abraham")
    ki = [k for k in sorted(es_imp) if k <= 10]
    axes[0].plot(ki, [es_imp[k] for k in ki], "x", color=PALETTE[4], ms=8, label="Imputation (BJS)")
    axes[0].axvline(-0.5, color="grey", lw=1)
    axes[0].axhline(0, color="grey", lw=1)
    axes[0].set_xlabel("event time e = t - g")
    axes[0].set_ylabel("effect")
    axes[0].set_title("Event study with heterogeneous, dynamic effects")
    axes[0].legend(fontsize=7)

    names = list(summary)
    axes[1].barh(names, [summary[k]["mean"] for k in names], xerr=[summary[k]["sd"] for k in names],
                 color=[PALETTE[3], PALETTE[0], PALETTE[5], PALETTE[4]], capsize=4)
    axes[1].axvline(truth, color="k", ls="--", label=f"true ATT {truth:.2f}")
    axes[1].invert_yaxis()
    axes[1].set_xlabel(f"overall ATT estimate (mean +- sd over {n_sims} panels)")
    axes[1].set_title("Static TWFE vs heterogeneity-robust estimators")
    axes[1].legend()
    savefig(fig, "ex06_did_staggered")

    fig, ax = plt.subplots(figsize=(8, 3.8))
    items = list(bacon.items())
    colors = [PALETTE[3] if "forbidden" in k else PALETTE[0] for k, _ in items]
    ax.barh([k for k, _ in items], [v for _, v in items], color=colors)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("2x2 DiD estimate")
    ax.set_title("Goodman-Bacon building blocks (red = already-treated units used as controls)")
    savefig(fig, "ex06_bacon_2x2")

    save_json({"true_att": truth, "n_sims": n_sims, "estimators": summary,
               "event_study_one_panel": {"true": es_true, "twfe": {int(k): v for k, v in es_twfe.items()},
                                         "cs_never": es_cs, "sun_abraham": es_sa, "imputation": es_imp},
               "bacon_2x2_one_panel": bacon}, "ex06_did_staggered")


if __name__ == "__main__":
    main()
