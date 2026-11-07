"""Bayesian state estimation: Kalman filter, RTS smoother, EKF, bootstrap particle filter.

Author: Abyn Scaria. MIT License.

State-space model (linear-Gaussian case):
    x_t = F x_{t-1} + w_t,  w_t ~ N(0, Q)
    y_t = H x_t     + v_t,  v_t ~ N(0, R)
The filter computes p(x_t | y_1..t); the smoother computes p(x_t | y_1..T). Equations follow
Saerkkae, "Bayesian Filtering and Smoothing" (2013), chapters 4 and 8.
"""
import numpy as np
from scipy.special import logsumexp


# ----------------------------------------------------------------------------------------------
# Linear-Gaussian: Kalman filter and Rauch-Tung-Striebel smoother
# ----------------------------------------------------------------------------------------------
def kalman_filter(ys, F, H, Q, R, m0, P0):
    """Kalman filter. ys has shape (T, dim_y).

    Returns dict with filtered means/covs, one-step predicted means/covs (needed by the
    smoother) and the exact log marginal likelihood log p(y_1..T) from the innovations.
    """
    T = ys.shape[0]
    n = m0.size
    ms, Ps = np.empty((T, n)), np.empty((T, n, n))
    mp, Pp = np.empty((T, n)), np.empty((T, n, n))
    m, P = m0.astype(float), P0.astype(float)
    loglik = 0.0
    for t in range(T):
        # predict
        m = F @ m
        P = F @ P @ F.T + Q
        mp[t], Pp[t] = m, P
        # update
        innov = ys[t] - H @ m
        S = H @ P @ H.T + R
        K = np.linalg.solve(S, H @ P).T          # K = P H' S^-1 (S symmetric)
        m = m + K @ innov
        P = P - K @ S @ K.T
        P = 0.5 * (P + P.T)                      # keep symmetric against round-off
        ms[t], Ps[t] = m, P
        sign, logdet = np.linalg.slogdet(2 * np.pi * S)
        loglik += -0.5 * (logdet + innov @ np.linalg.solve(S, innov))
    return {"m": ms, "P": Ps, "m_pred": mp, "P_pred": Pp, "loglik": loglik}


def rts_smoother(kf, F):
    """Rauch-Tung-Striebel backward pass on the output of `kalman_filter`."""
    ms, Ps, mp, Pp = kf["m"], kf["P"], kf["m_pred"], kf["P_pred"]
    T = ms.shape[0]
    msm, Psm = ms.copy(), Ps.copy()
    for t in range(T - 2, -1, -1):
        G = np.linalg.solve(Pp[t + 1], F @ Ps[t]).T   # G = P_t F' (P_pred_{t+1})^-1
        msm[t] = ms[t] + G @ (msm[t + 1] - mp[t + 1])
        Psm[t] = Ps[t] + G @ (Psm[t + 1] - Pp[t + 1]) @ G.T
    return {"m": msm, "P": Psm}


def constant_velocity_model(dt, q, r, dims=2):
    """Discretised white-noise-acceleration model. State per axis: (position, velocity).

    Q per axis = q * [[dt^3/3, dt^2/2], [dt^2/2, dt]]; only positions are observed.
    State ordering: [p_1..p_dims, v_1..v_dims].
    """
    I = np.eye(dims)
    Z = np.zeros((dims, dims))
    F = np.block([[I, dt * I], [Z, I]])
    Q = q * np.block([[dt ** 3 / 3 * I, dt ** 2 / 2 * I], [dt ** 2 / 2 * I, dt * I]])
    H = np.hstack([I, Z])
    R = r * I
    return F, H, Q, R


def simulate_lgssm(F, H, Q, R, x0, T, rng):
    n, k = F.shape[0], H.shape[0]
    xs, ys = np.empty((T, n)), np.empty((T, k))
    LQ = np.linalg.cholesky(Q + 1e-12 * np.eye(n))
    LR = np.linalg.cholesky(R)
    x = x0.astype(float)
    for t in range(T):
        x = F @ x + LQ @ rng.standard_normal(n)
        xs[t] = x
        ys[t] = H @ x + LR @ rng.standard_normal(k)
    return xs, ys


def nees(x_true, m, P):
    """Normalised estimation error squared, per time step. E[NEES] = dim(x) if consistent."""
    e = x_true - m
    return np.einsum("ti,tij,tj->t", e, np.linalg.inv(P), e)


# ----------------------------------------------------------------------------------------------
# Extended Kalman filter (scalar-or-vector, user supplies Jacobians)
# ----------------------------------------------------------------------------------------------
def ekf(ys, f, F_jac, h, H_jac, Q, R, m0, P0):
    """EKF: linearise f and h around the current mean at every step."""
    T = ys.shape[0]
    n = m0.size
    ms, Ps = np.empty((T, n)), np.empty((T, n, n))
    m, P = m0.astype(float), P0.astype(float)
    for t in range(T):
        Fj = F_jac(m, t)
        m = f(m, t)
        P = Fj @ P @ Fj.T + Q
        Hj = H_jac(m, t)
        S = Hj @ P @ Hj.T + R
        K = np.linalg.solve(S, Hj @ P).T
        m = m + K @ (ys[t] - h(m, t))
        P = (np.eye(n) - K @ Hj) @ P
        ms[t], Ps[t] = m, P
    return {"m": ms, "P": Ps}


# ----------------------------------------------------------------------------------------------
# Bootstrap (SIR) particle filter
# ----------------------------------------------------------------------------------------------
def resample(weights, rng, scheme="systematic"):
    """Return ancestor indices for normalised `weights` under one of four schemes."""
    N = weights.size
    cw = np.cumsum(weights)
    cw[-1] = 1.0
    if scheme == "multinomial":
        u = np.sort(rng.uniform(size=N))
    elif scheme == "stratified":
        u = (np.arange(N) + rng.uniform(size=N)) / N
    elif scheme == "systematic":
        u = (np.arange(N) + rng.uniform()) / N
    elif scheme == "residual":
        counts = np.floor(N * weights).astype(int)
        idx = np.repeat(np.arange(N), counts)
        n_rest = N - idx.size
        if n_rest > 0:
            resid = N * weights - counts
            resid /= resid.sum()
            extra = rng.choice(N, size=n_rest, p=resid)
            idx = np.concatenate([idx, extra])
        return idx
    else:
        raise ValueError(scheme)
    return np.searchsorted(cw, u)


def bootstrap_pf(ys, sample_x0, propagate, loglik_y, N, rng, scheme="systematic",
                 ess_threshold=0.5):
    """Bootstrap particle filter (Gordon, Salmond & Smith 1993).

    sample_x0(N, rng) -> (N, n) initial particles
    propagate(x, t, rng) -> (N, n) draws from p(x_t | x_{t-1})   (the proposal = the prior)
    loglik_y(y, x, t) -> (N,) log p(y_t | x_t)
    Resampling happens when the weight ESS falls below ess_threshold * N
    (ess_threshold=1 means every step, 0 means never).

    Returns filtered means/variances, the weight-ESS trajectory, the log marginal-likelihood
    estimate and the number of resampling events.
    """
    T = ys.shape[0]
    x = sample_x0(N, rng)
    n = x.shape[1]
    logw = np.full(N, -np.log(N))
    means, variances = np.empty((T, n)), np.empty((T, n))
    ess_hist = np.empty(T)
    loglik = 0.0
    n_resample = 0
    for t in range(T):
        x = propagate(x, t, rng)
        inc = loglik_y(ys[t], x, t)
        # log p(y_t | y_1:t-1) ~ log sum_i W_{t-1}^i g(y_t | x_t^i)
        loglik += logsumexp(logw + inc)
        logw = logw + inc
        logw -= logsumexp(logw)
        w = np.exp(logw)
        means[t] = w @ x
        variances[t] = w @ (x - means[t]) ** 2
        ess_hist[t] = 1.0 / np.sum(w ** 2)
        if ess_hist[t] < ess_threshold * N:
            idx = resample(w, rng, scheme)
            x = x[idx]
            logw = np.full(N, -np.log(N))
            n_resample += 1
    return {"m": means, "var": variances, "ess": ess_hist, "loglik": loglik,
            "n_resample": n_resample}
