# 2. Potential outcomes: Rubin's framework

> Previous: [1. Pearl's graphical view](01-foundations-pearl.md) · Next: [3. Experiments](03-experiments.md)

## 2.1 The setup

For each unit $i$ and each treatment level $t$, imagine an outcome $Y_i(t)$: what $i$'s outcome *would be* if $i$ received $t$. With a binary treatment each unit has two potential outcomes, $Y_i(1)$ and $Y_i(0)$. The **individual treatment effect** is

$$\tau_i = Y_i(1) - Y_i(0).$$

We observe only one of the two:

$$Y_i = T_i\, Y_i(1) + (1 - T_i)\, Y_i(0).$$

Holland called this the **fundamental problem of causal inference**: the individual effect is never observed directly, so causal inference is a missing-data problem in which half the table is always missing.

| Unit | $T$ | $Y(1)$ | $Y(0)$ | $\tau$ |
|---|---|---|---|---|
| 1 | 1 | 7 | ? | ? |
| 2 | 0 | ? | 4 | ? |
| 3 | 1 | 5 | ? | ? |
| 4 | 0 | ? | 6 | ? |

Everything that follows is about filling in the question marks *on average* under assumptions you can defend.

## 2.2 Estimands: be precise about whose effect

| Estimand | Definition | Typical question |
|---|---|---|
| ATE | $E[Y(1) - Y(0)]$ | What if we rolled this out to everyone? |
| ATT | $E[Y(1) - Y(0) \mid T = 1]$ | Did the programme help the people who took it? |
| ATC (ATU) | $E[Y(1) - Y(0) \mid T = 0]$ | Would it help the people who did not take it? |
| CATE | $\tau(x) = E[Y(1) - Y(0) \mid X = x]$ | Who benefits most? (targeting, personalisation) |
| LATE | $E[Y(1) - Y(0) \mid \text{complier}]$ | Effect for those whose treatment the instrument moves ([Ch. 4](04-observational-methods.md)) |
| ATT(g, t) | ATT at time $t$ for the cohort first treated at $g$ | Staggered rollouts ([Ch. 5](05-panel-methods.md)) |

These can differ a lot. In `ex01_confounding.py` the effect varies with $X_2$ and treated units have higher $X_2$, so in a large sample (n = 2 million) the ATE is 2.00, the ATT 2.60 and the ATC 1.59. An estimator that "works" for one is biased for the other. **Always state the estimand before choosing the estimator.**

## 2.3 Assumptions

### SUTVA

The **stable unit treatment value assumption** has two parts:

1. **No interference.** Unit $i$'s potential outcomes depend only on $i$'s own treatment, not on anyone else's. This fails in marketplaces, social networks, vaccination, classrooms and ad auctions. [Chapter 3](03-experiments.md) and [Chapter 8](08-sensitivity-and-validation.md) show what happens and how to fix it.
2. **No hidden versions of treatment.** "Treated" means the same thing for everybody. If "took the drug" covers several doses, $Y_i(1)$ is not well defined.

### Ignorability (unconfoundedness, exchangeability)

$$\{Y(0), Y(1)\} \perp T \mid X.$$

Within levels of the observed covariates, treatment is as good as random. Epidemiologists call this *conditional exchangeability*, economists *selection on observables*, and in DAG language it holds when $X$ satisfies the backdoor criterion. It cannot be tested from the data.

### Positivity (overlap)

$$0 < P(T = 1 \mid X = x) < 1 \quad \text{for all } x \text{ in the population.}$$

Every kind of unit must have some chance of each treatment. If no one over 70 is ever treated, the data say nothing about treatment effects for people over 70, and any estimate there is pure extrapolation from the model. Unlike ignorability, positivity *can* be checked, for example by plotting the estimated propensity score by arm as in `ex01`. Strict overlap, meaning propensities bounded away from 0 and 1, is also what keeps weighting estimators stable.

### Consistency

If $T_i = t$ then $Y_i = Y_i(t)$. The observed outcome equals the potential outcome under the treatment actually received. This sounds trivial, but it fails when treatment is vaguely defined, which is part of why "no hidden versions" matters.

## 2.4 From assumptions to an estimand

Under ignorability, positivity and consistency:

$$
\begin{aligned}
E[Y(1)] &= E_X\big[E[Y(1) \mid X]\big] = E_X\big[E[Y(1) \mid T=1, X]\big] = E_X\big[E[Y \mid T=1, X]\big].
\end{aligned}
$$

The first step is iterated expectations, the second is ignorability, and the third is consistency. Positivity ensures the inner conditional expectation is defined for every $X$. So

$$\text{ATE} = E_X\big[\mu_1(X) - \mu_0(X)\big], \qquad \mu_t(x) = E[Y \mid T=t, X=x].$$

This is the **g-formula** (standardisation), and it is the same expression as Pearl's adjustment formula. An equivalent weighting form is

$$\text{ATE} = E\left[\frac{T\,Y}{e(X)} - \frac{(1-T)\,Y}{1 - e(X)}\right].$$

These two identities are the parents of nearly every estimator in [Chapter 4](04-observational-methods.md) and [Chapter 6](06-causal-ml.md). Regression and matching estimate the first, IPW estimates the second, and doubly robust methods combine them.

## 2.5 Why the naive comparison fails

Split the naive difference in means:

$$
\underbrace{E[Y \mid T=1] - E[Y \mid T=0]}_{\text{naive}}
= \underbrace{E[Y(1) - Y(0) \mid T=1]}_{\text{ATT}}
+ \underbrace{E[Y(0) \mid T=1] - E[Y(0) \mid T=0]}_{\text{selection bias}}.
$$

The selection-bias term asks whether the treated would have had different outcomes *even without treatment*. Randomisation sets it to zero by design. Every observational method is a strategy for arguing that it is zero, or for removing it.

## 2.6 Pearl and Rubin side by side

| Concept | Graphical (Pearl) | Potential outcomes (Rubin) |
|---|---|---|
| Causal effect | $P(y \mid do(t))$ | Distribution of $Y(t)$ |
| No confounding given $X$ | $X$ satisfies the backdoor criterion | $Y(t) \perp T \mid X$ |
| Identification result | Adjustment formula | g-formula / standardisation |
| Bad controls | Read off the graph (colliders, mediators) | Often stated as "only adjust for pre-treatment variables", which M-bias shows is not quite enough |
| Strength | Reasoning about *which* assumptions hold | Defining estimands, deriving estimators and their variance |
| Weakness | Graphs can be hard to elicit and defend | Ignorability is asserted, not derived |

They are logically equivalent for most purposes. Single World Intervention Graphs (SWIGs), from Richardson and Robins, draw potential outcomes directly on a graph and make the bridge explicit.

## 2.7 Interference and SUTVA failures, previewed

When SUTVA fails, $Y_i$ depends on the whole treatment vector $\mathbf{T}$. A common simplification is an **exposure mapping**, for example "own treatment and fraction of treated friends", which reduces the outcome to a few exposure levels. Aronow and Samii give inverse-probability estimators for any exposure mapping that is known ([Aronow & Samii, 2017](https://arxiv.org/html/1305.6156v3)). `ex04` part (c) shows the simplest marketplace version. When treated users take demand from control users, individual randomisation estimates 0.98 while the effect of a full launch is 0.20. Randomising whole markets recovers 0.20.

## 2.8 Key takeaways

* Every causal question is a comparison of potential outcomes, and you see only one per unit.
* Say whose effect you want: ATE, ATT, CATE or LATE. They can differ.
* SUTVA, ignorability, positivity and consistency are what turn observable data into a causal estimand. Only positivity can be checked from the data.
* The g-formula and the IPW identity are the two roots of almost every estimator.

**Further reading:** Imbens & Rubin, *Causal Inference for Statistics, Social, and Biomedical Sciences* (2015), chapters 1-3 and 12. Hernán & Robins, *What If*, chapters 1-3. Cunningham, *The Mixtape*, chapter 4 ([free](https://mixtape.scunning.com/)).

## Appendix A: estimands differ, a numerical check

`ex01_confounding.py` draws 2 million units from its data-generating process and averages the true $\tau(X) = 2 + 1.5 X_2$ over different groups:

```python
Xb, Tb, _, _ = simulate(2_000_000, np.random.default_rng(0))
tau_b = TRUE_ATE + 1.5 * Xb[:, 1]
estimands = {"ATE": tau_b.mean(), "ATT": tau_b[Tb == 1].mean(), "ATC": tau_b[Tb == 0].mean()}
# -> ATE 2.00, ATT 2.60, ATC 1.59
```

The treated have higher $X_2$ (the propensity increases with $X_2$) and $X_2$ raises the effect, so ATT > ATE > ATC. A matching estimator that targets the ATT and an AIPW estimator that targets the ATE would disagree by 0.6 on the same data. Both would be right about their own estimand.

## Appendix B: the IPW identity in three lines

Under ignorability and positivity,

$$
E\left[\frac{T Y}{e(X)}\right]
= E\left[\frac{T\, Y(1)}{e(X)}\right]
= E\left[\frac{E[T \mid X]\; E[Y(1) \mid X]}{e(X)}\right]
= E\big[E[Y(1) \mid X]\big] = E[Y(1)].
$$

The first step uses consistency ($TY = TY(1)$), the second iterated expectations plus ignorability (so $T$ and $Y(1)$ are independent given $X$), and the third $E[T \mid X] = e(X)$. Positivity guarantees we never divide by zero. The same argument with $1 - T$ and $1 - e(X)$ gives $E[Y(0)]$.

## Appendix C: exercises

1. Construct a four-unit table (like §2.1, but with both potential outcomes filled in) in which the ATE is positive while the naive difference in means is negative. Identify the selection-bias term.
2. Show that if treatment is randomised with a known probability $p$ that varies by stratum, the naive pooled difference in means can be biased, while the stratified difference is not. What assumption from this chapter does that rely on?
3. A feature launch affects both the users who see it and their friends. Write down two SUTVA-respecting estimands you *could* define with an exposure mapping, and which design would identify each.
4. Pick a positivity violation from your domain (for example, a policy that never applies to a certain region) and state what the data can and cannot say about it.
