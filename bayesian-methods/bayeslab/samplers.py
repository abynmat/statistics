"""Two MCMC samplers written from scratch: random-walk Metropolis-Hastings and HMC.

Author: Abyn Scaria. MIT License.

Both samplers work on an unconstrained vector x in R^d.
  * MH needs `logp(x) -> float`.
  * HMC needs `logp_grad(x) -> (float, ndarray)`.

Each returns a dict with the post-warmup draws plus the bookkeeping needed for diagnostics
(acceptance, step size, divergences, energy errors, number of density/gradient evaluations).
"""
import time

import numpy as np


# ----------------------------------------------------------------------------------------------
# Random-walk Metropolis-Hastings
# ----------------------------------------------------------------------------------------------
def rw_metropolis(logp, x0, n_draws, rng, n_warmup=1000, scale=None, chol=None,
                  adapt=True, target_accept=0.234):
    """Gaussian random-walk Metropolis with optional Robbins-Monro scale adaptation in warmup.

    Proposal: x' = x + scale * chol @ z, z ~ N(0, I). The proposal is symmetric, so the
    acceptance probability is min(1, p(x') / p(x)).

    During warmup, log(scale) moves by (accepted - target) / t^0.6, a standard stochastic
    approximation that drives the acceptance rate towards `target_accept`. 0.234 is the
    asymptotically optimal rate for d-dimensional product targets (Roberts, Gelman & Gilks 1997).
    """
    x = np.array(x0, dtype=float)
    d = x.size
    scale = 2.38 / np.sqrt(d) if scale is None else float(scale)
    L = np.eye(d) if chol is None else np.asarray(chol)
    lp = logp(x)
    total = n_warmup + n_draws
    draws = np.empty((n_draws, d))
    accepted = np.zeros(total, dtype=bool)
    log_scale = np.log(scale)
    t0 = time.perf_counter()
    for i in range(total):
        prop = x + np.exp(log_scale) * (L @ rng.standard_normal(d))
        lp_prop = logp(prop)
        if np.log(rng.uniform()) < lp_prop - lp:
            x, lp = prop, lp_prop
            accepted[i] = True
        if adapt and i < n_warmup:
            log_scale += (accepted[i] - target_accept) / (i + 1) ** 0.6
        if i >= n_warmup:
            draws[i - n_warmup] = x
    return {
        "draws": draws,
        "accept_rate": float(accepted[n_warmup:].mean()),
        "scale": float(np.exp(log_scale)),
        "n_evals": total,                     # one log-density evaluation per iteration
        "seconds": time.perf_counter() - t0,
    }


# ----------------------------------------------------------------------------------------------
# Hamiltonian Monte Carlo
# ----------------------------------------------------------------------------------------------
def leapfrog(x, p, grad, eps, n_steps, logp_grad, inv_mass, h0=None, max_dH=1000.0):
    """Run `n_steps` leapfrog steps of size eps for H(x, p) = -logp(x) + 0.5 p' M^{-1} p.

    If `h0` is given, stop early when the energy error exceeds `max_dH` (a divergence).
    Returns (x, p, logp, grad, n_grad_evals, diverged).
    """
    # Written as kick-drift-kick per step; consecutive half kicks merge into the usual full
    # momentum step, so the cost is one gradient per step. Writing it this way keeps p at the
    # same time as x after every step, which lets us check the energy error at each step.
    x = x.copy()
    p = p.copy()
    lp = None
    for s in range(n_steps):
        p = p + 0.5 * eps * grad                   # half kick
        x = x + eps * inv_mass * p                 # drift
        lp, grad = logp_grad(x)
        p = p + 0.5 * eps * grad                   # half kick
        if h0 is not None:
            h = -lp + 0.5 * np.sum(inv_mass * p ** 2)
            if not np.isfinite(h) or h - h0 > max_dH:
                return x, p, lp, grad, s + 1, True
    return x, p, lp, grad, n_steps, False


class DualAveraging:
    """Step-size adaptation by primal-dual averaging (Hoffman & Gelman 2014, Algorithm 5).

    Written from the update equations in section 3.2 of arXiv:1111.4246:
      H_bar_t = (1 - 1/(t+t0)) H_bar_{t-1} + (delta - alpha_t) / (t + t0)
      log eps_t = mu - sqrt(t) / gamma * H_bar_t
      log eps_bar_t = t^-kappa log eps_t + (1 - t^-kappa) log eps_bar_{t-1}
    """

    def __init__(self, eps0, target=0.8, gamma=0.05, t0=10.0, kappa=0.75):
        self.restart(eps0)
        self.target, self.gamma, self.t0, self.kappa = target, gamma, t0, kappa

    def restart(self, eps0):
        self.mu = np.log(10.0 * eps0)
        self.h_bar = 0.0
        self.log_eps_bar = 0.0
        self.t = 0

    def update(self, accept_prob):
        self.t += 1
        t = self.t
        w = 1.0 / (t + self.t0)
        self.h_bar = (1 - w) * self.h_bar + w * (self.target - accept_prob)
        log_eps = self.mu - np.sqrt(t) / self.gamma * self.h_bar
        eta = t ** (-self.kappa)
        self.log_eps_bar = eta * log_eps + (1 - eta) * self.log_eps_bar
        return float(np.exp(log_eps))

    @property
    def final(self):
        return float(np.exp(self.log_eps_bar))


def hmc(logp_grad, x0, n_draws, rng, n_warmup=1000, step_size=0.1, n_steps=None,
        path_length=None, max_steps=1024, target_accept=0.8, adapt_step=True,
        adapt_mass=False, inv_mass=None, jitter=0.1, max_dH=1000.0):
    """Static-trajectory HMC with dual-averaging step size and optional diagonal mass adaptation.

    Trajectory length: either a fixed `n_steps` or a fixed integration time `path_length`
    (n_steps = ceil(path_length / eps), capped at `max_steps`). After warmup the step size is
    jittered uniformly by +-`jitter` to avoid resonant, periodic trajectories.

    A transition is flagged divergent when the Hamiltonian error exceeds `max_dH` along the
    trajectory (the same threshold Stan uses); divergent proposals are rejected.

    Mass adaptation (if on) estimates the posterior variance on the window from 25% to 75% of
    warmup, sets inv_mass to the regularised variance, and restarts dual averaging.
    """
    x = np.array(x0, dtype=float)
    d = x.size
    inv_m = np.ones(d) if inv_mass is None else np.asarray(inv_mass, dtype=float).copy()
    lp, grad = logp_grad(x)
    eps = float(step_size)
    da = DualAveraging(eps, target=target_accept)
    total = n_warmup + n_draws
    w_lo, w_hi = int(0.25 * n_warmup), int(0.75 * n_warmup)
    window = []

    draws = np.empty((n_draws, d))
    accept_prob = np.empty(total)
    divergent = np.zeros(total, dtype=bool)
    delta_h = np.empty(total)
    steps_used = np.empty(total, dtype=int)
    n_grad = 0
    t_start = time.perf_counter()
    for i in range(total):
        warm = i < n_warmup
        e = eps if warm else eps * rng.uniform(1 - jitter, 1 + jitter)
        L = n_steps if n_steps is not None else int(min(max_steps, max(1, np.ceil(path_length / e))))
        p0 = rng.standard_normal(d) / np.sqrt(inv_m)                 # p ~ N(0, M)
        h0 = -lp + 0.5 * np.sum(inv_m * p0 ** 2)
        xn, pn, lpn, gn, used, div = leapfrog(x, p0, grad, e, L, logp_grad, inv_m, h0, max_dH)
        n_grad += used
        steps_used[i] = used
        if div:
            a, dh = 0.0, np.inf
        else:
            h1 = -lpn + 0.5 * np.sum(inv_m * pn ** 2)
            dh = h1 - h0
            a = float(np.exp(min(0.0, -dh))) if np.isfinite(dh) else 0.0
        accept_prob[i], divergent[i], delta_h[i] = a, div, dh
        if rng.uniform() < a:
            x, lp, grad = xn, lpn, gn
        if warm:
            if adapt_step:
                eps = da.update(a)
            if adapt_mass and w_lo <= i < w_hi:
                window.append(x.copy())
            if adapt_mass and i == w_hi - 1 and len(window) > 10:
                n = len(window)
                var = np.var(np.array(window), axis=0, ddof=1)
                inv_m = (n / (n + 5.0)) * var + 1e-3 * (5.0 / (n + 5.0))
                da.restart(eps)
            if i == n_warmup - 1 and adapt_step:
                eps = da.final
        else:
            draws[i - n_warmup] = x
    post = slice(n_warmup, total)
    return {
        "draws": draws,
        "accept_prob": accept_prob[post],
        "accept_rate": float(accept_prob[post].mean()),
        "divergent": divergent[post],
        "n_divergent": int(divergent[post].sum()),
        "delta_h": delta_h[post],
        "steps": steps_used[post],
        "step_size": eps,
        "inv_mass": inv_m,
        "n_evals": n_grad,                      # gradient evaluations, warmup included
        "seconds": time.perf_counter() - t_start,
    }


def run_chains(sampler, n_chains, x0_fn, seed, **kwargs):
    """Run `sampler` independently n_chains times with dispersed starts.

    Returns (draws of shape (n_chains, n_draws, d), list of per-chain result dicts).
    """
    results = []
    for c in range(n_chains):
        rng = np.random.default_rng([seed, c])
        results.append(sampler(x0=x0_fn(rng), rng=rng, **kwargs))
    return np.stack([r["draws"] for r in results]), results
