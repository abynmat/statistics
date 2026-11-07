"""MCMC diagnostics written from their definitions.

Author: Abyn Scaria. MIT License.

Conventions: `draws` is an array of shape (n_chains, n_draws) for one scalar quantity.
Functions that accept (n_chains, n_draws, dim) apply per dimension.

Definitions follow Gelman et al., BDA3 ch. 11, and Vehtari, Gelman, Simpson, Carpenter and
Buerkner (2021), "Rank-normalization, folding, and localization: an improved R-hat"
(arXiv:1903.08008). The code is written from the equations in those texts; the
multi-chain autocorrelation estimator and Geyer's initial-monotone truncation are the ones
described in section 3 of that paper.
"""
import numpy as np
from scipy import stats


def _as_2d(draws):
    x = np.asarray(draws, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError("expected (n_chains, n_draws)")
    return x


def autocovariance(x):
    """Biased (divide-by-n) autocovariance of a 1-D series at every lag, via FFT."""
    x = np.asarray(x, dtype=float)
    n = x.size
    xc = x - x.mean()
    m = 1 << (2 * n - 1).bit_length()           # zero-pad to avoid circular wrap-around
    f = np.fft.rfft(xc, m)
    acov = np.fft.irfft(f * np.conj(f), m)[:n] / n
    return acov


def autocorrelation(x):
    """Autocorrelation rho_t of a 1-D series, rho_0 = 1."""
    acov = autocovariance(x)
    return acov / acov[0] if acov[0] > 0 else np.zeros_like(acov)


def split_chains(draws):
    """Split each chain in half, doubling the chain count (drops a middle draw if odd)."""
    x = _as_2d(draws)
    n = x.shape[1] // 2
    return np.concatenate([x[:, :n], x[:, -n:]], axis=0)


def rhat_basic(draws):
    """Classic potential scale reduction sqrt(var_plus / W) on the chains as given."""
    x = _as_2d(draws)
    m, n = x.shape
    chain_means = x.mean(axis=1)
    chain_vars = x.var(axis=1, ddof=1)
    W = chain_vars.mean()
    B = n * chain_means.var(ddof=1)
    var_plus = (n - 1) / n * W + B / n
    return float(np.sqrt(var_plus / W)) if W > 0 else np.nan


def split_rhat(draws):
    """Split-R-hat (BDA3 eq. 11.4 applied to half-chains)."""
    return rhat_basic(split_chains(draws))


def ess(draws):
    """Effective sample size of the mean, multi-chain, with Geyer initial-monotone truncation.

    rho_t = 1 - (W - mean_m acov_{m,t}) / var_plus  combines chains so that between-chain
    disagreement lowers the ESS. Pairs P_k = rho_{2k} + rho_{2k+1} are summed while positive,
    and forced to be non-increasing. tau = -1 + 2 * sum_k P_k, ESS = M * N / tau.
    """
    x = _as_2d(draws)
    m, n = x.shape
    if n < 4:
        return np.nan
    acov = np.stack([autocovariance(c) for c in x])          # (m, n)
    chain_means = x.mean(axis=1)
    W = (acov[:, 0] * n / (n - 1)).mean()                    # mean within-chain variance
    var_plus = W * (n - 1) / n
    if m > 1:
        var_plus += chain_means.var(ddof=1)
    if var_plus <= 0:
        return np.nan
    rho = 1.0 - (W - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0
    # Geyer: sum adjacent pairs while the pair sum stays positive, then enforce monotone.
    pair_sums = []
    t = 0
    while t + 1 < n:
        p = rho[t] + rho[t + 1]
        if p < 0:
            break
        pair_sums.append(p)
        t += 2
    pair_sums = np.minimum.accumulate(np.array(pair_sums)) if pair_sums else np.array([1.0])
    tau = -1.0 + 2.0 * pair_sums.sum()
    tau = max(tau, 1.0 / np.log10(m * n))                    # cap ESS for antithetic chains
    return float(m * n / tau)


def rank_normalize(draws):
    """Replace draws by normal scores of their pooled ranks (Blom offset 3/8)."""
    x = _as_2d(draws)
    r = stats.rankdata(x, method="average").reshape(x.shape)
    s = x.size
    return stats.norm.ppf((r - 3.0 / 8.0) / (s + 0.25))


def rank_rhat(draws):
    """Rank-normalised split-R-hat: max of bulk and folded (tail-scale) versions."""
    x = _as_2d(draws)
    bulk = split_rhat(rank_normalize(x))
    folded = np.abs(x - np.median(x))
    tail = split_rhat(rank_normalize(folded))
    return float(max(bulk, tail))


def ess_bulk(draws):
    """Bulk-ESS: ESS of rank-normalised split chains."""
    return ess(rank_normalize(split_chains(draws)))


def ess_tail(draws, q=0.05):
    """Tail-ESS: min ESS of the indicator I(x <= quantile) at q and 1-q (split chains)."""
    x = split_chains(draws)
    lo, hi = np.quantile(x, [q, 1 - q])
    return float(min(ess((x <= lo).astype(float)), ess((x <= hi).astype(float))))


def summarize(samples, names=None):
    """Per-dimension summary for samples shaped (n_chains, n_draws, dim).

    Returns a list of dicts with mean, sd, quantiles, bulk/tail ESS and rank R-hat.
    """
    s = np.asarray(samples, dtype=float)
    if s.ndim == 2:
        s = s[:, :, None]
    out = []
    for j in range(s.shape[2]):
        col = s[:, :, j]
        out.append({
            "name": names[j] if names else f"x{j}",
            "mean": float(col.mean()),
            "sd": float(col.std(ddof=1)),
            "q05": float(np.quantile(col, 0.05)),
            "q95": float(np.quantile(col, 0.95)),
            "ess_bulk": ess_bulk(col),
            "ess_tail": ess_tail(col),
            "rhat": rank_rhat(col),
        })
    return out


def ar1_chains(phi, n_chains, n_draws, rng, offsets=None, scales=None):
    """Simulate stationary AR(1) chains x_t = phi x_{t-1} + e_t with unit marginal variance.

    The true integrated autocorrelation time is (1 + phi) / (1 - phi), so the true ESS of the
    mean is n_chains * n_draws * (1 - phi) / (1 + phi). Used to test `ess` and `split_rhat`.
    """
    sd_e = np.sqrt(1 - phi ** 2)
    x = np.empty((n_chains, n_draws))
    x[:, 0] = rng.standard_normal(n_chains)
    e = rng.standard_normal((n_chains, n_draws)) * sd_e
    for t in range(1, n_draws):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    if scales is not None:
        x = x * np.asarray(scales)[:, None]
    if offsets is not None:
        x = x + np.asarray(offsets)[:, None]
    return x
