"""Predictive model comparison from a pointwise log-likelihood matrix.

Author: Abyn Scaria. MIT License.

Input everywhere: `loglik` of shape (S, n) with loglik[s, i] = log p(y_i | theta^(s)) for S
posterior draws and n observations.

  * WAIC        Watanabe (2010); computed as in Gelman, Hwang & Vehtari (2014).
  * PSIS        Pareto-smoothed importance sampling, Vehtari, Simpson, Gelman, Yao & Gabry
                (arXiv:1507.02646). The generalised-Pareto fit is the profile-likelihood
                empirical-Bayes estimator of Zhang & Stephens (2009), with the weakly
                informative shrinkage of k towards 0.5 described in the PSIS paper; the overall
                structure mirrors the reference `gpdfit` routine in the R `loo` package
                (https://github.com/stan-dev/loo), rewritten here in numpy.
  * PSIS-LOO    Vehtari, Gelman & Gabry (arXiv:1507.04544).
"""
import numpy as np
from scipy.special import logsumexp


def waic(loglik):
    """Return dict(elpd, p_waic, se, pointwise). elpd_waic = lppd - p_waic."""
    S = loglik.shape[0]
    lppd_i = logsumexp(loglik, axis=0) - np.log(S)
    p_i = np.var(loglik, axis=0, ddof=1)
    elpd_i = lppd_i - p_i
    n = elpd_i.size
    return {"elpd": float(elpd_i.sum()), "p": float(p_i.sum()),
            "se": float(np.sqrt(n * np.var(elpd_i))), "pointwise": elpd_i}


def gpd_fit(x):
    """Fit a generalised Pareto (shape k, scale sigma, location 0) to positive exceedances x.

    Zhang & Stephens (2009) estimator: average the profile likelihood over a grid of
    theta = -k / sigma values, then shrink k towards 0.5 with a prior worth 10 observations.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    prior = 3.0
    m = 30 + int(np.floor(np.sqrt(n)))
    j = np.arange(1, m + 1)
    x_q1 = x[int(np.floor(n / 4.0 + 0.5)) - 1]
    theta = 1.0 / x[-1] + (1.0 - np.sqrt(m / (j - 0.5))) / (prior * x_q1)
    k_grid = np.array([np.mean(np.log1p(-t * x)) for t in theta])
    prof = n * (np.log(-theta / k_grid) - k_grid - 1.0)        # profile log-likelihood
    w = np.exp(prof - logsumexp(prof))
    theta_hat = np.sum(theta * w)
    k = np.mean(np.log1p(-theta_hat * x))
    sigma = -k / theta_hat
    k = (n * k + 10 * 0.5) / (n + 10)                            # weakly informative prior
    return float(k), float(sigma)


def _gpd_quantile(p, k, sigma):
    if abs(k) < 1e-12:
        return -sigma * np.log1p(-p)
    return sigma * np.expm1(-k * np.log1p(-p)) / k


def psis(log_ratios):
    """Pareto-smooth one vector of log importance ratios. Returns (smoothed log weights, k_hat)."""
    lw = np.asarray(log_ratios, dtype=float).copy()
    S = lw.size
    lw -= lw.max()
    M = int(np.ceil(min(0.2 * S, 3.0 * np.sqrt(S))))
    order = np.argsort(lw)
    tail_idx = order[-M:]
    cutoff = lw[order[-M - 1]]
    exceed = np.exp(lw[tail_idx]) - np.exp(cutoff)
    if np.all(exceed <= 0) or M < 5:
        return lw - logsumexp(lw), np.inf
    k, sigma = gpd_fit(exceed)
    if np.isfinite(k):
        p = (np.arange(1, M + 1) - 0.5) / M
        smoothed = np.log(_gpd_quantile(p, k, sigma) + np.exp(cutoff))
        lw[tail_idx] = np.minimum(smoothed, 0.0)                 # truncate at the raw max
    return lw - logsumexp(lw), k


def psis_loo(loglik):
    """PSIS-LOO: elpd_loo, p_loo, se, pointwise elpd and pointwise Pareto k-hat."""
    S, n = loglik.shape
    lppd = logsumexp(loglik, axis=0) - np.log(S)
    elpd_i = np.empty(n)
    khat = np.empty(n)
    for i in range(n):
        lw, khat[i] = psis(-loglik[:, i])                        # ratio p(theta|y_-i)/p(theta|y)
        elpd_i[i] = logsumexp(lw + loglik[:, i])
    return {"elpd": float(elpd_i.sum()), "p": float((lppd - elpd_i).sum()),
            "se": float(np.sqrt(n * np.var(elpd_i))), "pointwise": elpd_i, "khat": khat}


def is_loo(loglik):
    """Plain (unsmoothed) importance-sampling LOO, for contrast with PSIS."""
    S, n = loglik.shape
    elpd_i = np.empty(n)
    for i in range(n):
        lw = -loglik[:, i]
        lw = lw - logsumexp(lw)
        elpd_i[i] = logsumexp(lw + loglik[:, i])
    return {"elpd": float(elpd_i.sum()), "pointwise": elpd_i}
