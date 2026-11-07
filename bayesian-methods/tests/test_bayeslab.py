"""One test suite for every component, each checked against a known answer.

Author: Abyn Scaria. MIT License.

Run with either:
    python3 tests/test_bayeslab.py        (no dependencies)
    python3 -m pytest tests/              (if pytest is installed)
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bayeslab import diagnostics as dg  # noqa: E402
from bayeslab import filters as fl  # noqa: E402
from bayeslab import model_compare as mc  # noqa: E402
from bayeslab import samplers as sp  # noqa: E402
from bayeslab import targets as tg  # noqa: E402


def _fd_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


# ---------------------------------------------------------------- diagnostics
def test_ess_matches_ar1_theory():
    rng = np.random.default_rng(1)
    for phi in (0.0, 0.5, 0.9):
        x = dg.ar1_chains(phi, 4, 20000, rng)
        truth = x.size * (1 - phi) / (1 + phi)
        est = dg.ess(x)
        assert abs(est / truth - 1) < 0.12, (phi, est, truth)


def test_rhat_flags_disagreeing_chains():
    rng = np.random.default_rng(2)
    good = dg.ar1_chains(0.5, 4, 2000, rng)
    shifted = dg.ar1_chains(0.5, 4, 2000, rng, offsets=[0, 0, 0, 1.0])
    scaled = dg.ar1_chains(0.5, 4, 2000, rng, scales=[1, 1, 1, 3.0])
    assert dg.split_rhat(good) < 1.01
    assert dg.split_rhat(shifted) > 1.05
    # Same mean, different scale: classic R-hat barely moves, the folded rank version catches it.
    assert dg.rank_rhat(scaled) > dg.split_rhat(scaled)
    assert dg.rank_rhat(scaled) > 1.05


def test_split_rhat_flags_trend_within_chain():
    rng = np.random.default_rng(3)
    x = dg.ar1_chains(0.3, 4, 2000, rng) + np.linspace(0, 2, 2000)[None, :]
    assert dg.rhat_basic(x) < 1.01 < dg.split_rhat(x)


# ---------------------------------------------------------------- targets
def test_gradients_match_finite_differences():
    rng = np.random.default_rng(4)
    for target in (tg.correlated_gaussian(5), tg.Funnel(4, True), tg.Funnel(4, False)):
        x = rng.standard_normal(target.dim) * 0.5
        g = target.logp_grad(x)[1]
        assert np.allclose(g, _fd_grad(target.logp, x), atol=1e-5)


# ---------------------------------------------------------------- samplers
def test_leapfrog_reversible_and_second_order():
    t = tg.correlated_gaussian(5, rho=0.5, scale_range=(0.5, 1.0))
    rng = np.random.default_rng(5)
    x0, p0 = rng.standard_normal(5), rng.standard_normal(5)
    lp0, g0 = t.logp_grad(x0)
    inv_m = np.ones(5)
    x1, p1, _, g1, _, _ = sp.leapfrog(x0, p0, g0, 0.05, 40, t.logp_grad, inv_m)
    x2, p2, _, _, _, _ = sp.leapfrog(x1, -p1, g1, 0.05, 40, t.logp_grad, inv_m)
    assert np.allclose(x2, x0, atol=1e-10) and np.allclose(-p2, p0, atol=1e-10)
    errs = []
    for eps in (0.04, 0.02):
        n = int(round(0.8 / eps))
        x, p, lp, _, _, _ = sp.leapfrog(x0, p0, g0, eps, n, t.logp_grad, inv_m)
        errs.append(abs((-lp + 0.5 * p @ p) - (-lp0 + 0.5 * p0 @ p0)))
    assert 2.5 < errs[0] / errs[1] < 6.0                        # halving eps -> ~4x smaller


def test_mh_recovers_gaussian_moments():
    t = tg.Gaussian([1.0, -2.0], [[1.0, 1.2], [1.2, 4.0]])
    draws, _ = sp.run_chains(sp.rw_metropolis, 4, lambda r: r.standard_normal(2) * 3, 0,
                             logp=t.logp, n_draws=6000, n_warmup=1500)
    flat = draws.reshape(-1, 2)
    assert np.allclose(flat.mean(0), t.mean, atol=0.15)
    assert np.allclose(flat.std(0), t.sd, rtol=0.08)


def test_hmc_recovers_correlated_gaussian():
    t = tg.correlated_gaussian(10, rho=0.9)
    draws, res = sp.run_chains(sp.hmc, 4, lambda r: r.standard_normal(10), 0,
                               logp_grad=t.logp_grad, n_draws=1000, n_warmup=500,
                               path_length=1.5, adapt_mass=True)
    flat = draws.reshape(-1, 10)
    assert np.max(np.abs(flat.mean(0)) / t.sd) < 0.15
    assert np.max(np.abs(flat.std(0) / t.sd - 1)) < 0.12
    # The averaged step size eps_bar is a little smaller than the iterates, so realised
    # acceptance lands at or slightly above the 0.8 target.
    assert all(0.75 < r["accept_rate"] < 0.95 for r in res)
    assert sum(r["n_divergent"] for r in res) == 0


def test_dual_averaging_hits_target_acceptance():
    t = tg.correlated_gaussian(20, rho=0.5)
    rng = np.random.default_rng(6)
    rates = []
    for target in (0.6, 0.75, 0.9):
        r = sp.hmc(t.logp_grad, np.zeros(20), 800, rng, n_warmup=800, step_size=1.0,
                   n_steps=10, target_accept=target)
        rates.append(r["accept_rate"])
        # eps_bar (the averaged iterate) is slightly conservative: allow +0.12 above target.
        assert target - 0.05 < r["accept_rate"] < target + 0.12, (target, r["accept_rate"])
    assert rates[0] < rates[1] < rates[2]


def test_hmc_reports_divergences_in_funnel_neck():
    f = tg.Funnel(9, centered=True)
    draws, res = sp.run_chains(sp.hmc, 2, lambda r: np.r_[0.0, r.standard_normal(9)], 0,
                               logp_grad=f.logp_grad, n_draws=800, n_warmup=500, path_length=2.0,
                               step_size=0.5)
    assert sum(r["n_divergent"] for r in res) > 0


# ---------------------------------------------------------------- filters
def test_rts_equals_batch_gaussian_posterior():
    """For a linear-Gaussian model the smoother means/vars equal the exact joint posterior."""
    rng = np.random.default_rng(7)
    T, q, r = 40, 0.3, 1.0
    F, H, Q, R = np.eye(1), np.eye(1), np.eye(1) * q, np.eye(1) * r
    xs, ys = fl.simulate_lgssm(F, H, Q, R, np.zeros(1), T, rng)
    m0, P0 = np.zeros(1), np.eye(1) * 5.0
    sm = fl.rts_smoother(fl.kalman_filter(ys, F, H, Q, R, m0, P0), F)
    # Batch posterior: prior on x_1..T is Gaussian with precision from the random walk.
    D = np.eye(T) - np.eye(T, k=-1)                 # x_t - x_{t-1}
    prior_prec = D.T @ D / q
    prior_prec[0, 0] += 1.0 / (P0[0, 0] + q) - 1.0 / q   # x_1 ~ N(0, P0 + q)
    post_prec = prior_prec + np.eye(T) / r
    post_cov = np.linalg.inv(post_prec)
    post_mean = post_cov @ (ys[:, 0] / r)
    assert np.allclose(sm["m"][:, 0], post_mean, atol=1e-8)
    assert np.allclose(sm["P"][:, 0, 0], np.diag(post_cov), atol=1e-8)


def test_particle_filter_loglik_close_to_kalman():
    rng = np.random.default_rng(8)
    F, H, Q, R = np.eye(1) * 0.9, np.eye(1), np.eye(1) * 0.5, np.eye(1) * 1.0
    xs, ys = fl.simulate_lgssm(F, H, Q, R, np.zeros(1), 60, rng)
    kf = fl.kalman_filter(ys, F, H, Q, R, np.zeros(1), np.eye(1))
    pf = fl.bootstrap_pf(
        ys, lambda N, g: g.standard_normal((N, 1)),
        lambda x, t, g: 0.9 * x + np.sqrt(0.5) * g.standard_normal(x.shape),
        lambda y, x, t: stats.norm.logpdf(y[0], x[:, 0], 1.0), 20000, rng)
    assert abs(pf["loglik"] - kf["loglik"]) < 0.3
    assert np.max(np.abs(pf["m"][:, 0] - kf["m"][:, 0])) < 0.05


def test_resampling_schemes_unbiased():
    rng = np.random.default_rng(9)
    w = rng.dirichlet(np.ones(50))
    for scheme in ("multinomial", "stratified", "systematic", "residual"):
        counts = np.zeros(50)
        for _ in range(400):
            counts += np.bincount(fl.resample(w, rng, scheme), minlength=50)
        assert np.allclose(counts / (400 * 50), w, atol=0.01), scheme


# ---------------------------------------------------------------- model comparison
def test_gpd_fit_recovers_shape():
    rng = np.random.default_rng(10)
    for k in (0.2, 0.5, 0.8):
        x = stats.genpareto.rvs(k, scale=1.0, size=3000, random_state=rng)
        kh, _ = mc.gpd_fit(x)
        assert abs(kh - k) < 0.08, (k, kh)


def test_psis_loo_matches_exact_loo_conjugate_normal():
    """Normal mean with known sd and a flat-ish prior: exact LOO predictive is Gaussian."""
    rng = np.random.default_rng(11)
    n, sd, tau0 = 30, 1.0, 10.0
    y = rng.normal(0.5, sd, n)
    post_var = 1 / (1 / tau0 ** 2 + n / sd ** 2)
    post_mean = post_var * y.sum() / sd ** 2
    mu = rng.normal(post_mean, np.sqrt(post_var), 8000)
    loglik = stats.norm.logpdf(y[None, :], mu[:, None], sd)
    exact = 0.0
    for i in range(n):
        v = 1 / (1 / tau0 ** 2 + (n - 1) / sd ** 2)
        m = v * (y.sum() - y[i]) / sd ** 2
        exact += stats.norm.logpdf(y[i], m, np.sqrt(v + sd ** 2))
    loo = mc.psis_loo(loglik)
    assert abs(loo["elpd"] - exact) < 0.1
    assert np.all(loo["khat"] < 0.5)


def _run_all():
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"PASS  {name}  ({time.time() - t0:.1f}s)")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
