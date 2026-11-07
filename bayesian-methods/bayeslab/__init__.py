"""bayeslab: small, readable implementations of Bayesian inference and state estimation.

Author: Abyn Scaria. MIT License.

Modules
-------
diagnostics    autocorrelation, effective sample size, split / rank-normalised R-hat
samplers       random-walk Metropolis-Hastings and Hamiltonian Monte Carlo with dual averaging
targets        log-densities (+ gradients) for the benchmark distributions used in the chapters
filters        Kalman filter, RTS smoother, extended Kalman filter, bootstrap particle filter
model_compare  pointwise log-likelihood tools: WAIC, Pareto-smoothed importance sampling, PSIS-LOO
plotting       one shared matplotlib style and save helpers

Everything is numpy + scipy. No probabilistic-programming library is used anywhere.
"""

import warnings as _w

# numpy 2.0 + Apple Accelerate emits spurious "... encountered in matmul" warnings on finite
# inputs (an upstream issue); results are unaffected, so silence that one message only.
_w.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
