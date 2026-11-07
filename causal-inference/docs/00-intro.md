# 0. Why causal inference, and how this guide is organised

> Part of the [causal-inference](../README.md) guide. Next: [1. Foundations: Pearl's graphical view](01-foundations-pearl.md)

## 0.1 The one-sentence version

Prediction asks *what will I see?* Causal inference asks *what will happen if I change something?* The two questions have different answers whenever the data you learned from was produced by a process that your change would disturb.

A model that predicts hospital mortality well may learn that asthma patients with pneumonia die less often. In the training data they were sent straight to intensive care, so they really did die less. Use that model to decide who needs intensive care and it will send asthma patients home. The pattern in the data is real. The causal reading of it is wrong. Most of this guide is about telling those two apart, and about getting the causal answer when you can.

## 0.2 Three kinds of questions

It helps to sort questions before reaching for a method. Judea Pearl calls this sorting the *ladder of causation* ([Pearl & Mackenzie, 2018](https://web.cs.ucla.edu/~kaoru/3-layer-causal-hierarchy.pdf)):

| Rung | Question type | Example | Notation | Needs |
|---|---|---|---|---|
| 1. Association | "What does seeing X tell me about Y?" | Users who get push notifications churn less. | $P(y \mid x)$ | Data |
| 2. Intervention | "What happens to Y if I *set* X?" | If we turn on notifications for everyone, does churn fall? | $P(y \mid do(x))$ | Data + causal assumptions, or an experiment |
| 3. Counterfactual | "What would Y have been for *this* unit had X been different?" | Would *this* user have churned without the notification they got? | $P(y_x \mid x', y')$ | A fully specified structural model |

The key result behind the ladder is that you cannot answer a higher rung using only lower-rung information. No amount of observational data, by itself, tells you an interventional quantity. You always need something extra: randomisation, a known graph, an instrument, a discontinuity, parallel trends. Most of this guide is a catalogue of those "something extras" and what each one buys.

## 0.3 Identification first, estimation second

Every method in this guide answers two separate questions, and confusing them causes most of the errors you see in practice.

1. **Identification.** If I had infinite data from this population, could I compute the causal quantity at all? This is about assumptions: a DAG, unconfoundedness, an exclusion restriction, parallel trends. The output is an *estimand*, a formula written in terms of observable distributions, such as $E_X[E(Y \mid T=1, X) - E(Y \mid T=0, X)]$.
2. **Estimation.** Given finite, noisy data, how do I compute that formula well? This is statistics and machine learning: regression, weighting, matching, boosting, cross-fitting, standard errors.

A fancy estimator cannot rescue a failed identification argument. A double machine learning estimate with a beautiful confidence interval is still biased if an important confounder is missing. Conversely, a correct identification argument estimated with a misspecified linear model can also be badly off. [Chapter 4](04-observational-methods.md) shows both failures side by side, and [Chapter 8](08-sensitivity-and-validation.md) shows how to say how bad an identification failure would have to be to change your conclusion.

## 0.4 Two languages, one subject

There are two main notations. They are mostly interchangeable, and a working practitioner should read both.

* **Structural causal models and DAGs** (Pearl and others; see [Chapter 1](01-foundations-pearl.md)). You draw the data-generating process as a directed graph and ask graphical questions such as *does this set of variables block every backdoor path?* This language is best for deciding **what to adjust for** and for spotting bad controls.
* **Potential outcomes** (Neyman, Rubin, Imbens; see [Chapter 2](02-potential-outcomes.md)). Every unit has an outcome under each treatment, $Y_i(1)$ and $Y_i(0)$, and only one is ever observed. This language is best for **defining estimands precisely** (ATE vs ATT vs LATE) and for the statistical theory of estimators.

Modern libraries such as [DoWhy](https://www.pywhy.org/dowhy/v0.8/getting_started/intro.html) explicitly combine the two: use a graph to model and identify, then use potential-outcome estimators to estimate, then refute.

## 0.5 How this guide is organised

| Chapter | Topic | Runnable example(s) |
|---|---|---|
| [1](01-foundations-pearl.md) | Pearl: SCMs, DAGs, d-separation, backdoor, front-door, do-calculus | `ex02_bad_controls.py`, `ex03_dags_frontdoor.py` |
| [2](02-potential-outcomes.md) | Rubin: potential outcomes, SUTVA, ignorability, ATE/ATT/CATE | conceptual, plus `ex01` |
| [3](03-experiments.md) | Randomised experiments and A/B testing pitfalls | `ex04_ab_testing.py` |
| [4](04-observational-methods.md) | Regression, matching, propensity scores, IPW, AIPW, IV, RDD | `ex01`, `ex05_rdd.py`, `ex08_iv.py` |
| [5](05-panel-methods.md) | DiD, staggered adoption, event studies, synthetic control, SDID | `ex06_did_staggered.py`, `ex07_synthetic_control.py` |
| [6](06-causal-ml.md) | Double ML, meta-learners, causal forests, uplift, TMLE | `ex09_hte_metalearners.py`, `ex10_dml.py` |
| [7](07-causal-discovery.md) | Causal discovery: PC, GES, LiNGAM, NOTEARS, LLM-assisted | `ex11_causal_discovery.py` |
| [8](08-sensitivity-and-validation.md) | Sensitivity analysis, refutation, mediation, interference | `ex12_sensitivity.py` |
| [9](09-frontier-2024-2026.md) | The frontier: causal representation learning, foundation models, proximal CI, LLMs | survey |
| [10](10-tooling-and-cheatsheet.md) | Software map and the full cheat sheet | reference |
| [Glossary](glossary.md) | Definitions of every term used | reference |
| [References](../references.md) | Papers, books, a reading order | reference |

## 0.6 How the examples work

Every example in `examples/` uses **synthetic data with a known true effect**. That is a deliberate choice. On real data you never see the truth, so you cannot tell a good estimator from a lucky one. On synthetic data you can measure bias directly and see *why* an estimator fails. Each script prints the true effect next to each estimate, writes the raw numbers to `results/*.json`, and writes a figure to `figures/`. The tables in this guide are copied from those JSON files.

The trade-off is that a simulation only shows what happens under the assumptions built into it. A method that wins here can lose on your data. Treat the examples as worked demonstrations of mechanisms, not as benchmarks.

## 0.7 Notation used throughout

| Symbol | Meaning |
|---|---|
| $T$ or $D$ | Treatment (binary unless stated). $D$ is common in econometrics. |
| $Y$ | Outcome |
| $X$ | Observed pre-treatment covariates |
| $U$ | Unobserved variables |
| $Z$ | Instrument (Ch. 4), or a generic adjustment set (Ch. 1). Context makes it clear. |
| $Y_i(t)$ | Potential outcome of unit $i$ under treatment level $t$ |
| $e(x) = P(T=1 \mid X=x)$ | Propensity score |
| $\mu_t(x) = E[Y \mid T=t, X=x]$ | Outcome regression in arm $t$ |
| $\tau(x) = E[Y(1) - Y(0) \mid X=x]$ | Conditional average treatment effect (CATE) |
| $do(T=t)$ | Intervention that sets $T$ to $t$ |
| $A \perp B \mid C$ | $A$ independent of $B$ given $C$ |

## 0.8 What this guide is not

It is not a substitute for the textbooks listed in [references.md](../references.md). Several are free, very good, and much longer. The aim here is a single connected map from the basics to the 2024-2026 research frontier, with working code for the core ideas, so you know which book chapter or paper to open next.

## 0.9 Appendix: a 20-line demonstration of the whole problem

The smallest possible version of everything in this guide. Run it in a Python shell:

```python
import numpy as np
rng = np.random.default_rng(0)
n = 100_000

age = rng.normal(size=n)                                  # confounder
offer = rng.binomial(1, 1 / (1 + np.exp(-1.5 * age)))     # older -> more likely offered
spend = 2.0 * offer + 1.5 * age + rng.normal(size=n)      # true effect of the offer = 2.0

naive = spend[offer == 1].mean() - spend[offer == 0].mean()
print(f"naive difference : {naive:.2f}")                  # 3.60: confounded

# Adjust for age (the backdoor set) with a regression
X = np.column_stack([np.ones(n), offer, age])
beta = np.linalg.lstsq(X, spend, rcond=None)[0]
print(f"adjusted for age : {beta[1]:.2f}")                # 1.99

# Randomise instead: the coin flip ignores age
offer_rct = rng.binomial(1, 0.5, n)
spend_rct = 2.0 * offer_rct + 1.5 * age + rng.normal(size=n)
print(f"randomised       : {spend_rct[offer_rct == 1].mean() - spend_rct[offer_rct == 0].mean():.2f}")  # 2.01
```

The three lines of output are the guide in miniature. Observational comparisons are confounded. Adjustment works *if you adjust for the right thing* (Chapter 1). Randomisation works without knowing what the right thing is (Chapter 3).

## 0.10 Exercises

1. In the snippet, add a variable `loyalty = spend + noise` and adjust for it as well. What happens to the estimate, and which DAG pattern from [Chapter 1](01-foundations-pearl.md) explains it?
2. Make the effect heterogeneous, `spend = (2 + age) * offer + ...`. Which estimand does the adjusted regression now target? (Hint: [Chapter 4, §4.1](04-observational-methods.md).)
3. For a question from your own work, write down which rung of the ladder it sits on, and what "something extra" (randomisation, instrument, cutoff, parallel trends) you could use to answer it.
