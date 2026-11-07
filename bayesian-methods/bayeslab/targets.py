"""Benchmark target densities with analytic gradients and known answers.

Author: Abyn Scaria. MIT License.

Each target exposes `logp(x)`, `logp_grad(x) -> (logp, grad)` and whatever exact moments are
known, so sampler output can always be scored against the truth.
"""
import numpy as np


class Gaussian:
    """Multivariate normal N(mean, cov). Exact mean and marginal sds are known."""

    def __init__(self, mean, cov):
        self.mean = np.asarray(mean, dtype=float)
        self.cov = np.asarray(cov, dtype=float)
        self.prec = np.linalg.inv(self.cov)
        self.chol = np.linalg.cholesky(self.cov)
        self.sd = np.sqrt(np.diag(self.cov))
        self.dim = self.mean.size

    def logp(self, x):
        r = x - self.mean
        return -0.5 * r @ self.prec @ r

    def logp_grad(self, x):
        g = -self.prec @ (x - self.mean)
        return 0.5 * (x - self.mean) @ g, g


def ar1_correlation(dim, rho):
    """Correlation matrix with entries rho^|i-j|: neighbouring coordinates strongly coupled."""
    idx = np.arange(dim)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def correlated_gaussian(dim, rho=0.9, scale_range=(0.1, 1.0), seed=0):
    """Zero-mean Gaussian with AR(1)-type correlation and log-spaced marginal scales.

    The mix of strong correlation and a 10x range of scales is what makes a single
    isotropic random-walk proposal inefficient.
    """
    rng = np.random.default_rng(seed)
    scales = np.exp(np.linspace(np.log(scale_range[0]), np.log(scale_range[1]), dim))
    scales = rng.permutation(scales)
    cov = ar1_correlation(dim, rho) * np.outer(scales, scales)
    return Gaussian(np.zeros(dim), cov)


class Funnel:
    """Neal's funnel: v ~ N(0, 3^2), x_i | v ~ N(0, exp(v/2)^2), i = 1..k (Neal 2003).

    centered=True samples (v, x) directly; the neck at small v needs tiny steps.
    centered=False samples (v, z) with x = exp(v/2) z, z ~ N(0, 1): an isotropic-ish target.
    Exact marginal of v is N(0, 9) in both cases.
    """

    def __init__(self, k=9, centered=True):
        self.k, self.centered, self.dim = k, centered, k + 1

    def logp_grad(self, th):
        v, x = th[0], th[1:]
        if self.centered:
            ev = np.exp(-v)
            lp = -v * v / 18.0 - 0.5 * np.sum(x * x) * ev - 0.5 * self.k * v
            gv = -v / 9.0 + 0.5 * np.sum(x * x) * ev - 0.5 * self.k
            gx = -x * ev
        else:
            lp = -v * v / 18.0 - 0.5 * np.sum(x * x)
            gv = -v / 9.0
            gx = -x
        return lp, np.concatenate([[gv], gx])

    def logp(self, th):
        return self.logp_grad(th)[0]

    def to_centered(self, draws):
        """Map non-centered (v, z) draws to (v, x)."""
        if self.centered:
            return draws
        out = draws.copy()
        out[..., 1:] = np.exp(draws[..., :1] / 2.0) * draws[..., 1:]
        return out
