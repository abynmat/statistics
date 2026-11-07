# statistics

Two long-form, code-backed guides to the statistics that data scientists are most often asked about and least
often taught well: causal inference and Bayesian computation. Both are written as numbered chapters with runnable
examples on synthetic data whose true answer is known, so every estimator is compared against the truth or an
exact posterior rather than against another estimator.

| Folder | What it is | Headline result |
|---|---|---|
| [`causal-inference/`](causal-inference/) | An 11-chapter guide from Pearl's ladder and potential outcomes through experiments, observational methods, staggered DiD and synthetic control, double ML and causal forests, causal discovery, sensitivity analysis and the 2024-2026 frontier. 12 examples, 20 figures, a one-page cheat sheet, a 117-term glossary, and a referenced reading order. | Static TWFE estimates 1.16 for a true staggered-adoption ATT of 2.70; Callaway-Sant'Anna gets 2.69. Peeking 50 times at an A/A test gives 31% false positives; an always-valid test gives 1.8%. Standardising the data takes NOTEARS from a perfect structural Hamming distance of 0.0 to 6.3. |
| [`bayesian-methods/`](bayesian-methods/) | Nine chapters that build a small library (`bayeslab`): grid and conjugate posteriors, Metropolis-Hastings, Hamiltonian Monte Carlo with divergence detection, Bayesian A/B testing with decision rules, hierarchical models, Bayesian regression, Kalman and particle filters, WAIC and PSIS-LOO. All diagnostics (ESS, rank R-hat, PSIS k-hat) implemented and checked. | On a 50-D correlated Gaussian the from-scratch HMC's worst posterior-mean error is 0.07 sd with 24-33× more effective samples per evaluation than MH. Rank-normalised R-hat flags a mis-scaled chain (1.145) that plain split R-hat misses (1.006). PSIS-LOO tracks exact LOO at 0.19 nats RMSE vs 0.37 for WAIC. |

## How to read

Each guide's README has a table of contents, a reading guide and a results section. Chapters live in `docs/`,
the code that produces each chapter's figures lives in `examples/`, and `run_all` regenerates everything in a
few minutes on a laptop. The causal-inference guide is written for a reader who knows regression; the Bayesian
guide for a reader who knows probability.

## License

MIT, see [LICENSE](LICENSE).
