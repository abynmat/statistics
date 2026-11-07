# 10. Tooling map and cheat sheet

> Previous: [9. The frontier](09-frontier-2024-2026.md) · Back to [README](../README.md)

## 10.1 The software landscape

Package versions and features change quickly. Check each project's documentation before relying on a specific function. The examples in this repo deliberately use only numpy, scipy, scikit-learn, matplotlib and networkx, so every estimator is visible in roughly a page of code.

### Python

| Package | What it is for | Notes |
|---|---|---|
| **[DoWhy](https://www.pywhy.org/dowhy/v0.8/getting_started/intro.html)** (py-why) | End-to-end workflow: **model** (a causal graph) → **identify** (backdoor, front-door, IV via graph criteria and do-calculus) → **estimate** → **refute** (placebo, random common cause, subset, unobserved-confounder simulation). Also graphical causal models (GCM) for attribution and root-cause questions | Combines graphs and potential outcomes ([DoWhy paper, arXiv 2011.04216](https://arxiv.org/pdf/2011.04216.pdf)). Estimators can be delegated to EconML |
| **EconML** (py-why) | Heterogeneous effects with ML: DML (linear, sparse, kernel, `CausalForestDML`), DR-learner, meta-learners, orthogonal IV, DeepIV, policy learning, SHAP-style interpretation of CATE | The main Python home of DML and GRF-style forests |
| **[CausalML](https://github.com/uber/causalml)** | Uplift modelling: S/T/X/R-learners, uplift trees and forests, Qini/AUUC evaluation, feature selection for uplift | Marketing and targeting focus ([arXiv 2002.11631](https://arxiv.org/abs/2002.11631)) |
| **DoubleML** | Faithful implementation of Chernozhukov et al.: PLR, IRM, PLIV, IIVM, DiD, RDD; repeated cross-fitting; sensitivity (Chernozhukov-Cinelli et al.) | Also available in R |
| **[causal-learn](https://github.com/py-why/causal-learn)** (py-why) | Causal discovery: PC, FCI, CD-NOD, GES, exact search, LiNGAM family, GIN, GRaSP; many CI tests | Python port and extension of Tetrad ([JMLR 2024](https://jmlr.org/papers/volume25/23-0970/23-0970.pdf)) |
| **[gCastle](https://arxiv.org/abs/2111.15155)** | Causal discovery toolbox focused on gradient-based methods (NOTEARS variants, GraN-DAG, RL-BIC), simulators and metrics | Optional GPU acceleration |
| `lingam` | LiNGAM family (DirectLiNGAM, VAR-LiNGAM, multi-group, RCD for latent confounders) | ([JMLR 2023](https://jmlr.org/papers/volume24/21-0321/21-0321.pdf)) |
| **statsmodels** | OLS/WLS/GLM with robust and clustered SEs, IV (`IV2SLS` in `linearmodels`), propensity models | The workhorse for classical estimators |
| `linearmodels` | Panel models (fixed effects, between, first difference), IV-2SLS/LIML/GMM | Panel TWFE with clustering |
| `pyfixest` | Fast high-dimensional fixed effects, event studies, some robust DiD estimators | Python port of the R `fixest` interface |
| `differences`, `csdid` | Callaway-Sant'Anna style staggered DiD in Python | Ports of the R `did` package |
| `rdrobust` / `rddensity` | RD local polynomial estimation, robust bias-corrected CIs, density tests | [PyPI](https://www.pypi.org/project/rdrobust/1.2.1) |
| `pysyncon`, `SparseSC` | Synthetic control | |
| `tfcausalimpact`, `pycausalimpact` | Python ports of CausalImpact | Check fidelity against the R version |
| Tigramite | Time-series causal discovery (PCMCI) | |
| `causalnex`, `pgmpy` | Bayesian networks, structure learning, do-operations | |

### R

| Package | What it is for |
|---|---|
| **`grf`** | Generalized random forests: causal forests, IV forests, survival forests, quantile forests, policy evaluation (RATE), honest inference |
| **`did`** | Callaway & Sant'Anna group-time ATTs and aggregations |
| `fixest` | Fast fixed-effects regression, Sun-Abraham (`sunab`) |
| `didimputation`, `did2s` | Borusyak-Jaravel-Spiess imputation; Gardner two-stage DiD |
| `DIDmultiplegt` | de Chaisemartin-D'Haultfœuille estimators |
| `bacondecomp` | Goodman-Bacon decomposition |
| `HonestDiD` | Rambachan-Roth sensitivity for parallel trends |
| **`Synth`**, `augsynth`, `synthdid` | Synthetic control (original ADH), augmented SC, synthetic DiD |
| **[`CausalImpact`](https://google.github.io/CausalImpact/CausalImpact.html)** | Bayesian structural time-series counterfactuals |
| **[`rdrobust`](https://journal.r-project.org/articles/RJ-2015-004)**, `rddensity`, `rdlocrand` | Regression discontinuity |
| `MatchIt`, `WeightIt`, `cobalt` | Matching, weighting, balance diagnostics |
| `tmle`, `ltmle`, `SuperLearner`, `tlverse` | TMLE and targeted learning |
| `AIPW`, `DoubleML` | Doubly robust / DML estimation |
| `mediation` | Causal mediation with sensitivity analysis |
| **[`sensemakr`](https://carloscinelli.com/sensemakr/articles/sensemakr.html)**, `EValue`, `rbounds` | Sensitivity analysis |
| `ivreg`, `AER`, `ivmodel` | IV with diagnostics, Anderson-Rubin intervals |
| `dagitty`, `ggdag` | Draw DAGs, find adjustment sets, list testable implications |
| `pcalg`, `bnlearn` | Causal discovery and Bayesian networks |
| `causaleffect`, [`cfid`](https://journal.r-project.org/articles/RJ-2023-053/RJ-2023-053.pdf) | Automated identification (ID, IDC; ID*, IDC*) |
| `bartCause`, `bcf` | BART and Bayesian Causal Forests |

### Stata

`teffects` (regression adjustment, IPW, AIPW, matching), `didregress`/`xtdidregress`, `csdid`, `did_imputation`, `eventstudyinteract`, `synth`/`synth2`, `rdrobust`, `ivreg2`/`weakivtest`, `honestdid`.

## 10.2 The cheat sheet

**How to use it:** find your *question* and your *data situation*, check that you can defend the *assumptions*, and be clear about which *estimand* the method delivers.

| Question / situation | Method | Key assumptions | Estimand | Python tool | Example here |
|---|---|---|---|---|---|
| Can I randomise? | RCT / A/B test, difference in means | Randomisation, SUTVA, no SRM | ATE (ITT under non-compliance) | `scipy.stats`, `statsmodels` | `ex04` |
| A/B test with pre-period data | CUPED / regression adjustment | Covariate measured pre-treatment | ATE, lower variance | `statsmodels` OLS | `ex04` |
| A/B test monitored continuously | mSPRT / always-valid inference, group sequential | Correct likelihood or prior | ATE with anytime-valid error control | custom, vendor tools | `ex04` |
| Spillovers between units | Cluster / switchback / two-stage randomisation | Interference contained within clusters | Global or spillover effects | custom | `ex04` |
| Observational, all confounders measured | Regression, matching, IPW, **AIPW** | Ignorability, positivity, correct DAG | ATE, ATT | `DoWhy`, `EconML`, `DoubleML`, `sklearn` | `ex01` |
| Many covariates, flexible confounding | **Double ML** / TMLE | Ignorability + nuisances learnable at ~$n^{-1/4}$ | ATE, partially linear $\theta$ | `DoubleML`, `EconML` | `ex10` |
| Who benefits most? | Meta-learners (S/T/X/DR/R), **causal forest** | Ignorability, overlap | CATE $\tau(x)$ | `EconML`, `CausalML` | `ex09` |
| Whom to target? | Uplift modelling, policy learning | As above + randomised holdout to evaluate | Ranking / policy value | `CausalML`, `EconML` | `ex09` |
| Confounder unobserved, but a mediator carries the whole effect | Front-door adjustment | Front-door criterion | ATE | `DoWhy` | `ex03` |
| Unobserved confounding + an instrument | IV / 2SLS / Wald | Relevance, exclusion, independence, monotonicity | **LATE** (compliers) | `linearmodels`, `DoubleML` (PLIV) | `ex08` |
| Treatment assigned by a threshold | Regression discontinuity (sharp/fuzzy) | Continuity at cutoff, no manipulation | Effect **at the cutoff** (LATE if fuzzy) | `rdrobust` | `ex05` |
| Before/after, treated vs control, same timing | 2×2 DiD / TWFE | Parallel trends, no anticipation | ATT | `linearmodels`, `pyfixest` | `ex06` |
| Staggered rollout | **Callaway-Sant'Anna**, Sun-Abraham, imputation, stacked DiD (not static TWFE) | Parallel trends (by cohort), no anticipation | ATT(g,t) and aggregates | `differences`, `csdid`, `pyfixest` | `ex06` |
| One treated aggregate unit, long pre-period | **Synthetic control**, augmented SC | Treated unit is matched by a donor combination; no spillovers | ATT for that unit | `pysyncon` | `ex07` |
| Few treated units, doubtful parallel trends | Synthetic DiD | Weighted parallel trends | ATT | `synthdid` (R) | `ex07` |
| Single time series + control series | CausalImpact / BSTS | Controls unaffected; stable relationship | Cumulative effect on the treated series | `tfcausalimpact` | n/a |
| Is my effect robust to hidden confounding? | Cinelli-Hazlett, E-value, Rosenbaum, HonestDiD | Interpretable bounds on confounding strength | Bias-adjusted estimate, breakdown point | `sensemakr`, custom | `ex12` |
| Direct vs indirect effects | Mediation (NDE/NIE) | Sequential ignorability | NDE, NIE, CDE | `mediation` (R) | `ex02` |
| Proxies of a hidden confounder | Proximal causal inference | Valid proxies, completeness | ATE | research code | n/a |
| What is the graph? | PC/FCI, GES, LiNGAM, NOTEARS (+ LLM priors) | Faithfulness, plus model-class assumptions | Equivalence class or DAG | `causal-learn`, `gCastle`, `lingam` | `ex11` |
| Experimental results, new population | Transportability, generalisation weighting | Selection diagram; shared mechanisms | Target-population ATE | research code, `DoWhy` partially | n/a |

## 10.3 A workflow that avoids the most common errors

1. **Write the question as an estimand.** "ATT of the loyalty programme on 90-day spend, among members enrolled in Q2."
2. **Draw the DAG**, including latent variables. List the adjustment set and the testable implications.
3. **Pick the identification strategy** that fits the data you have: experiment, adjustment, IV, RD, DiD or SC.
4. **Check the design before looking at outcomes:** overlap, balance, first-stage strength, density at the cutoff, pre-period fit, sample ratio.
5. **Estimate with a robust method:** doubly robust with cross-fitting, heterogeneity-robust DiD, local linear RD.
6. **Refute and test sensitivity:** placebos, negative controls, robustness value, HonestDiD.
7. **Report** the estimand, assumptions, diagnostics, point estimate with CI, and the sensitivity bounds.

## 10.4 Common mistakes, one line each

* Adjusting for a mediator or collider because it is correlated with both $T$ and $Y$ (`ex02`).
* Reading the OLS coefficient as the ATE when effects are heterogeneous (`ex01`).
* Peeking at A/B tests with fixed-horizon p-values (`ex04`: 31% false positives).
* Randomising users when the treatment spills across users (`ex04`).
* Trusting static TWFE under staggered adoption (`ex06`: 43% of the true effect).
* Treating F > 10 as safe for IV inference (`ex08`).
* Fitting high-order global polynomials in RD (`ex05`).
* Reporting a CATE model's ATE accuracy as if it validated its CATEs (`ex09`).
* Assuming DML's confidence intervals are valid whatever the nuisance quality (`ex10`: 38% coverage in the hard design).
* Benchmarking discovery algorithms on unstandardised simulated data (`ex11`).
* Equating a tiny p-value with robustness to hidden bias (`ex12`: Γ* = 1.36).

## 10.5 Decision tree (text version)

```
Can you randomise the treatment?
├── Yes → A/B test (Ch. 3)
│        ├── Spillovers between units? → cluster / switchback / two-stage design
│        ├── Pre-period metric available? → add CUPED
│        ├── Need to monitor continuously? → mSPRT or group-sequential
│        └── Imperfect take-up? → ITT, plus IV for the effect of taking treatment (LATE)
└── No → Is there a source of as-good-as-random variation?
         ├── A threshold assigns treatment → regression discontinuity (Ch. 4)
         ├── An instrument moves treatment, excluded from outcome → IV / 2SLS → LATE (Ch. 4)
         ├── Treated vs untreated units observed before and after → panel methods (Ch. 5)
         │        ├── single adoption date, many units → 2x2 DiD
         │        ├── staggered adoption → Callaway-Sant'Anna / Sun-Abraham / imputation
         │        └── one or a few treated aggregates → synthetic control / SDID / CausalImpact
         └── None of the above → selection on observables (Ch. 4, 6)
                  ├── Draw the DAG; choose an adjustment set (Ch. 1)
                  ├── Check overlap; estimate with AIPW/DML + cross-fitting
                  ├── Need heterogeneity? → causal forest / meta-learners (Ch. 6)
                  ├── Have proxies of a hidden confounder? → proximal inference (Ch. 9)
                  └── ALWAYS: sensitivity analysis (Ch. 8)
```

## 10.6 Minimal install for the examples here, and for going further

```bash
# this repo
pip install numpy scipy scikit-learn matplotlib networkx joblib

# the main causal libraries (not needed for the examples; check each project's docs for versions)
pip install dowhy econml doubleml causalml causal-learn
```

The examples deliberately avoid these libraries, so that each estimator's mechanics are visible. Once you understand an estimator from the from-scratch version, switch to the maintained library for real analyses: the libraries handle edge cases, inference and diagnostics that the sketches here do not.

## 10.7 Mapping this repo's code to library calls

| This repo | Closest library equivalent (check current API) |
|---|---|
| `aipw_boosted_crossfit` in `ex01` | `doubleml.DoubleMLIRM`, `econml.dr.LinearDRLearner`, DoWhy `backdoor.econml...` |
| `tsls` in `ex08` | `linearmodels.iv.IV2SLS` |
| `local_linear` in `ex05` | `rdrobust.rdrobust` |
| `cs_attgt` + `aggregate_*` in `ex06` | R `did::att_gt` + `aggte`; Python `csdid`, `differences` |
| `sun_abraham` in `ex06` | R `fixest::sunab` |
| `imputation` in `ex06` | R `didimputation`; Stata `did_imputation` |
| `sc_estimate` in `ex07` | R `Synth`, `augsynth`; Python `pysyncon` |
| `sdid_estimate` in `ex07` | R `synthdid` |
| `x_learner`, `r_learner`, `dr_learner` in `ex09` | `econml.metalearners`, `econml.dr`, `causalml.inference.meta` |
| `HonestCausalForest` in `ex09` | R `grf::causal_forest`; `econml.dml.CausalForestDML` |
| `est_dml` in `ex10` | `doubleml.DoubleMLPLR` |
| `pc`, `notears_linear` in `ex11` | `causallearn.search.ConstraintBased.PC.pc`; `castle.algorithms.Notears` |
| `adjusted_estimate`, `robustness_value` in `ex12` | `sensemakr` (R/Python/Stata) |

## 10.8 Worked case study: the 7-step workflow on a hypothetical question

This is a made-up scenario, used only to show how the pieces connect. No data or numbers are attached to it.

**Question.** A subscription app introduced an optional "study plan" feature. Leadership asks: *did the study plan increase 60-day retention?*

**Step 1: estimand.** Two different questions hide in that sentence:

* *Did it help the people who used it?* That is the ATT among adopters.
* *Should we push it to everyone?* That is the ATE, or the effect of a default-on rollout, which is a different intervention again.

We pick the ATT among users who enabled the plan in its first month, and note that the rollout decision needs a separate experiment.

**Step 2: DAG.** Motivation, a latent variable, drives both adopting the plan and retention. Observed proxies are prior-month activity, tenure, platform and acquisition channel. Pushing a notification about the plan affects adoption but plausibly not retention directly. That is a candidate instrument, with doubts about exclusion. Day-7 engagement is a *mediator*, so it must not be adjusted for. "Contacted support" is affected by both the plan and churn risk, which makes it a *collider*, so it must not be adjusted for either.

**Step 3: identification strategy.** Options, in decreasing order of credibility:

1. *Randomised encouragement* (the notification was A/B tested): IV, giving the LATE for users whose adoption the notification changed.
2. *Staggered availability by country*: heterogeneity-robust DiD on country-level retention.
3. *Selection on observables*: AIPW on the proxies, with a mandatory sensitivity analysis, because motivation is only partly captured.

We plan to run (1) as primary and (3) as a secondary check. If they agree, both are more believable. If they disagree, the difference is informative about how much the proxies miss.

**Step 4: design checks before outcomes.** For IV: the first-stage F (we want it well above 10, and ideally in the range where Lee et al.'s critical values are near 1.96) and the sample ratio of the notification test. For AIPW: propensity overlap by arm, trimming of extreme propensities, and covariate balance after weighting.

**Step 5: estimation.** IV: 2SLS with robust SEs and an Anderson-Rubin interval. AIPW: boosted nuisances, 5-fold cross-fitting, and a CATE model (causal forest) to see who benefits most, evaluated by a RATE/Qini curve on the randomised notification sample.

**Step 6: refutation and sensitivity.**

* Placebo outcome: retention in the month *before* the feature launched should show zero "effect".
* Negative-control outcome: something the plan cannot plausibly affect.
* Robustness value for the AIPW estimate, benchmarked against the strongest observed covariate (prior-month activity).
* For the IV: an argument, and if possible evidence, that the notification affects retention only through adoption.

**Step 7: report.**

> Among users whose adoption was moved by the notification, the study plan increased 60-day retention by X points (95% CI …; first-stage F = …). An observational AIPW estimate for all adopters is Y points; an unmeasured confounder would need to be about Z times as strong as prior-month activity to explain it away. Day-7 engagement is a mediator and was not adjusted for. The effect of making the plan default-on is not identified by these data and needs a rollout experiment.

The template makes three points explicit that analyses often hide: *which* population the number is about, *what* would have to be true for it to be causal, and *how* fragile it is.

## 10.9 Quick reference: one formula per method

| Method | Core formula |
|---|---|
| Difference in means | $\bar Y_1 - \bar Y_0$ |
| CUPED | $Y - \theta(X - \bar X)$, $\theta = \operatorname{Cov}(X,Y)/\operatorname{Var}(X)$ |
| g-formula | $\frac1n\sum_i \hat\mu_1(X_i) - \hat\mu_0(X_i)$ |
| IPW (Hájek) | $\frac{\sum T_iY_i/\hat e_i}{\sum T_i/\hat e_i} - \frac{\sum (1-T_i)Y_i/(1-\hat e_i)}{\sum (1-T_i)/(1-\hat e_i)}$ |
| AIPW | $\frac1n\sum_i \hat\mu_1 - \hat\mu_0 + \frac{T_i(Y_i-\hat\mu_1)}{\hat e} - \frac{(1-T_i)(Y_i-\hat\mu_0)}{1-\hat e}$ |
| Wald / LATE | $\dfrac{E[Y\mid Z=1]-E[Y\mid Z=0]}{E[T\mid Z=1]-E[T\mid Z=0]}$ |
| Sharp RD | $\lim_{x\downarrow c}E[Y\mid x] - \lim_{x\uparrow c}E[Y\mid x]$ |
| 2×2 DiD | $(\bar Y_{1,\text{post}}-\bar Y_{1,\text{pre}}) - (\bar Y_{0,\text{post}}-\bar Y_{0,\text{pre}})$ |
| ATT(g,t) (CS) | $E[Y_t - Y_{g-1}\mid G=g] - E[Y_t - Y_{g-1}\mid C]$ |
| Synthetic control | $\hat Y_{1t}(0) = \sum_j w_j Y_{jt}$, $w \ge 0$, $\sum w_j = 1$ |
| DML (PLR) | $\hat\theta = \sum \tilde D_i\tilde Y_i / \sum \tilde D_i^2$ (cross-fitted residuals) |
| R-learner | $\arg\min_\tau \sum_i[(Y_i-\hat\ell(X_i)) - \tau(X_i)(T_i-\hat e(X_i))]^2$ |
| Front-door | $\sum_m P(m\mid t)\sum_{t'}P(y\mid m,t')P(t')$ |
| E-value | $RR + \sqrt{RR(RR-1)}$ |
| Robustness value | $\tfrac12(\sqrt{f^4+4f^2}-f^2)$, $f = \lvert t\rvert/\sqrt{\text{df}}$ |
| NOTEARS constraint | $\operatorname{tr}(e^{W\circ W}) - d = 0$ |
