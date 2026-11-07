# Glossary

> Back to [README](../README.md)

Terms are grouped loosely by topic and alphabetical within each group. Chapter references point to where the term is used most.

## Core concepts

| Term | Definition |
|---|---|
| **Association** | Statistical dependence between variables, the first rung of the ladder. Says nothing on its own about what would happen under intervention. (Ch. 0) |
| **Causal effect** | A comparison of outcomes under different interventions, e.g. $E[Y \mid do(T=1)] - E[Y \mid do(T=0)]$ or $E[Y(1) - Y(0)]$. (Ch. 1-2) |
| **Counterfactual** | What would have happened to a specific unit under a treatment it did not receive. The third rung of the ladder. (Ch. 1) |
| **Estimand** | The quantity you want to know, defined before any data: ATE, ATT, LATE, CATE. (Ch. 2) |
| **Estimator** | A procedure that turns data into a number meant to approximate an estimand. (Ch. 4) |
| **Identification** | Showing that an estimand can be written as a function of the observable data distribution under stated assumptions. (Ch. 0) |
| **Intervention, $do(\cdot)$** | Setting a variable to a value by external action, replacing its structural equation. (Ch. 1) |
| **Ladder of causation** | Pearl's hierarchy: association, intervention, counterfactual. (Ch. 0) |

## Graphs and SCMs

| Term | Definition |
|---|---|
| **Adjustment set** | Variables whose conditioning identifies a causal effect via the adjustment formula. (Ch. 1) |
| **Backdoor criterion** | A set $Z$ with no descendants of $T$ that blocks every path into $T$. Satisfying it licenses adjustment. (Ch. 1) |
| **Backdoor path** | A path between $T$ and $Y$ that starts with an arrow into $T$. (Ch. 1) |
| **Collider** | A node with two incoming arrows on a path ($A \to K \leftarrow B$). It blocks the path unless it or one of its descendants is conditioned on. (Ch. 1) |
| **Confounder** | A common cause of treatment and outcome (a fork). (Ch. 1) |
| **CPDAG** | Completed partially directed acyclic graph, representing a Markov equivalence class. (Ch. 7) |
| **d-separation** | A graphical criterion: a set blocks every path between two nodes. It implies conditional independence. (Ch. 1) |
| **DAG** | Directed acyclic graph. In a causal DAG, arrows are direct causal effects. (Ch. 1) |
| **do-calculus** | Three rules for removing $do(\cdot)$ from expressions using d-separation. Complete for identification. (Ch. 1) |
| **Faithfulness** | Every conditional independence in the data is implied by the graph, i.e. no exact cancellations. (Ch. 7) |
| **Front-door criterion** | Identification through a mediator that captures all of the effect when the treatment-outcome confounder is unobserved. (Ch. 1) |
| **M-bias** | Bias from adjusting for a pre-treatment collider of two unobserved causes. (Ch. 1) |
| **Markov equivalence class** | All DAGs implying the same conditional independences. (Ch. 7) |
| **Mediator** | A variable on a causal path from treatment to outcome. (Ch. 1, 8) |
| **SCM** | Structural causal model: exogenous noise plus one-way structural assignments. (Ch. 1) |
| **Selection diagram** | A DAG with selection nodes marking mechanisms that differ between populations. Used for transportability. (Ch. 9) |
| **SWIG** | Single World Intervention Graph, which puts potential outcomes on a graph. (Ch. 2) |
| **v-structure** | An unshielded collider $A \to K \leftarrow B$ with $A$, $B$ non-adjacent. Identifiable from observational data. (Ch. 7) |

## Potential outcomes

| Term | Definition |
|---|---|
| **ATE** | Average treatment effect $E[Y(1) - Y(0)]$. (Ch. 2) |
| **ATT / ATC** | Average effect among the treated / among controls. (Ch. 2) |
| **CATE** | Conditional average treatment effect $\tau(x) = E[Y(1) - Y(0) \mid X = x]$. (Ch. 2, 6) |
| **Consistency** | Observed outcome equals the potential outcome under the received treatment. (Ch. 2) |
| **Fundamental problem of causal inference** | Only one potential outcome per unit is observed. (Ch. 2) |
| **g-formula (standardisation)** | $E_X[\mu_1(X) - \mu_0(X)]$, the regression form of identification under ignorability. (Ch. 2) |
| **Ignorability / unconfoundedness / exchangeability** | $\{Y(0), Y(1)\} \perp T \mid X$. (Ch. 2) |
| **ITE** | Individual treatment effect $Y_i(1) - Y_i(0)$. Never observed. (Ch. 2) |
| **Positivity / overlap** | $0 < e(x) < 1$ for all relevant $x$. (Ch. 2, 4) |
| **Potential outcome** | $Y_i(t)$: unit $i$'s outcome if it received treatment $t$. (Ch. 2) |
| **Selection bias** | $E[Y(0) \mid T=1] - E[Y(0) \mid T=0]$, the gap between naive and causal comparisons. (Ch. 2) |
| **SUTVA** | No interference between units and no hidden versions of treatment. (Ch. 2) |

## Experiments

| Term | Definition |
|---|---|
| **A/A test** | An experiment with no actual difference between arms, used to validate the system and error rates. (Ch. 3) |
| **Always-valid p-value** | A p-value that controls the type-I error under continuous monitoring. (Ch. 3) |
| **Cluster randomisation** | Randomising groups (markets, schools) instead of individuals. (Ch. 3) |
| **CUPED** | Variance reduction by subtracting $\theta(X_{\text{pre}} - \bar X_{\text{pre}})$ from the outcome. (Ch. 3) |
| **Interference** | One unit's treatment affects another unit's outcome. Violates SUTVA. (Ch. 3, 8) |
| **ITT** | Intention-to-treat: the effect of *assignment*, ignoring non-compliance. (Ch. 3) |
| **MDE** | Minimum detectable effect at a given power and sample size. (Ch. 3) |
| **mSPRT** | Mixture sequential probability ratio test. Gives always-valid inference. (Ch. 3) |
| **Peeking** | Repeatedly checking a fixed-horizon test and stopping on significance. Inflates false positives. (Ch. 3) |
| **SRM** | Sample-ratio mismatch: the observed arm sizes differ from the design, which signals a broken experiment. (Ch. 3) |
| **Switchback** | Randomising treatment over time slots within a unit or market. (Ch. 3) |

## Observational methods

| Term | Definition |
|---|---|
| **2SLS** | Two-stage least squares, the standard IV estimator. (Ch. 4) |
| **AIPW** | Augmented inverse probability weighting. Doubly robust and efficient. (Ch. 4) |
| **Always-taker / never-taker / complier / defier** | IV compliance types defined by how treatment responds to the instrument. (Ch. 4) |
| **Balance** | Similarity of covariate distributions across arms after matching or weighting. (Ch. 4) |
| **Bandwidth** | Window around the RD cutoff used for local estimation. (Ch. 4) |
| **Doubly robust** | Consistent if either the outcome model or the propensity model is correct. (Ch. 4) |
| **Exclusion restriction** | The instrument affects the outcome only through treatment. (Ch. 4) |
| **First-stage F** | Strength diagnostic for an instrument. (Ch. 4) |
| **Fuzzy RD** | RD in which the treatment *probability* jumps at the cutoff. (Ch. 4) |
| **Hájek estimator** | Normalised IPW, with weights summing to one in each arm. (Ch. 4) |
| **Instrument** | A variable affecting treatment, unconfounded with the outcome, with no direct effect on it. (Ch. 4) |
| **IPW** | Inverse probability weighting by $1/e(X)$ and $1/(1-e(X))$. (Ch. 4) |
| **LATE** | Local average treatment effect: the effect for compliers. (Ch. 4) |
| **Matching** | Pairing treated and control units with similar covariates. (Ch. 4) |
| **Monotonicity** | No defiers: the instrument pushes everyone's treatment in the same direction. (Ch. 4) |
| **Overlap weights** | Weights $e(1-e)$ targeting the population with the most overlap. (Ch. 4) |
| **Propensity score** | $e(x) = P(T=1 \mid X=x)$. (Ch. 4) |
| **Regression discontinuity (RD)** | Identification from a jump in treatment at a threshold of a running variable. (Ch. 4) |
| **Running variable** | The variable whose threshold assigns treatment in RD. (Ch. 4) |
| **Weak instrument** | An instrument with a small first stage, which makes 2SLS biased and inference unreliable. (Ch. 4) |

## Panel methods

| Term | Definition |
|---|---|
| **ATT(g, t)** | Average effect at time $t$ for the cohort first treated at time $g$. (Ch. 5) |
| **Cohort** | Units sharing the same first-treatment period. (Ch. 5) |
| **DiD** | Difference-in-differences. (Ch. 5) |
| **Event study** | Effects by time relative to treatment, including pre-periods. (Ch. 5) |
| **Forbidden comparison** | Using already-treated units as controls in TWFE under staggered adoption. (Ch. 5) |
| **Goodman-Bacon decomposition** | TWFE as a weighted average of all 2×2 DiDs. (Ch. 5) |
| **Imputation estimator** | Fit unit and time effects on untreated cells, impute $Y(0)$ for treated cells. (Ch. 5) |
| **No anticipation** | Outcomes do not respond before treatment starts. (Ch. 5) |
| **Parallel trends** | Untreated outcomes would have evolved the same way in treated and control groups. (Ch. 5) |
| **Placebo-in-space** | SC inference that reassigns treatment to each donor in turn. (Ch. 5) |
| **RMSPE** | Root mean squared prediction error, the pre/post fit measure in SC. (Ch. 5) |
| **Staggered adoption** | Units begin treatment at different times. (Ch. 5) |
| **Synthetic control** | A convex combination of donors that matches the treated unit's pre-period. (Ch. 5) |
| **Synthetic DiD** | DiD with optimised unit and time weights. (Ch. 5) |
| **TWFE** | Two-way fixed effects regression with unit and time effects. (Ch. 5) |

## Causal ML

| Term | Definition |
|---|---|
| **AUUC / Qini** | Area under the uplift curve. A ranking quality metric for CATE models. (Ch. 6) |
| **Causal forest** | A random forest whose splits target effect heterogeneity, with honest estimation. (Ch. 6) |
| **Cross-fitting** | Fit nuisances on some folds, evaluate on the held-out fold, rotate. (Ch. 6) |
| **DML** | Double/debiased machine learning: orthogonal score plus cross-fitting. (Ch. 6) |
| **GRF** | Generalized random forests for any local moment-defined parameter. (Ch. 6) |
| **Honesty** | Using separate samples to choose splits and to estimate leaf values. (Ch. 6) |
| **Meta-learner** | A recipe turning generic regressors into a CATE estimator (S/T/X/DR/R). (Ch. 6) |
| **Neyman orthogonality** | The score's derivative with respect to the nuisances is zero, so nuisance errors enter only at second order. (Ch. 6) |
| **Nuisance function** | An auxiliary function needed for estimation ($e$, $\mu_t$, $\ell$), not of interest itself. (Ch. 6) |
| **PEHE** | Precision in estimating heterogeneous effects: RMSE of $\hat\tau(x)$. (Ch. 6) |
| **Partially linear model** | $Y = \theta D + g(X) + U$. (Ch. 6) |
| **Super Learner** | A cross-validated ensemble used in TMLE. (Ch. 6) |
| **TMLE** | Targeted maximum likelihood estimation. (Ch. 6) |
| **Uplift modelling** | CATE estimation for targeting decisions. (Ch. 6) |

## Discovery, sensitivity, frontier

| Term | Definition |
|---|---|
| **ANM** | Additive noise model $Y = f(X) + N$ with $N \perp X$. Identifiable direction. (Ch. 7) |
| **Benchmarking (sensitivity)** | Expressing a hypothetical confounder's strength relative to an observed covariate. (Ch. 8) |
| **Causal representation learning** | Learning latent variables that behave as causal variables. (Ch. 9) |
| **E-value** | Minimum risk-ratio confounding strength needed to explain away an estimate. (Ch. 8) |
| **FCI** | Fast Causal Inference, a PC extension allowing latent confounders. (Ch. 7) |
| **GES** | Greedy equivalence search, score-based discovery. (Ch. 7) |
| **LiNGAM** | Linear non-Gaussian acyclic model. Fully identifiable DAG. (Ch. 7) |
| **Natural direct / indirect effect** | Mediation effects defined with nested counterfactuals. (Ch. 8) |
| **Negative control** | An outcome or exposure known to have no causal relation to the other side, used to detect bias. (Ch. 8, 9) |
| **NOTEARS** | Continuous-optimisation structure learning with a smooth acyclicity constraint. (Ch. 7) |
| **PC algorithm** | Constraint-based discovery using conditional independence tests. (Ch. 7) |
| **PFN** | Prior-data fitted network: a transformer pretrained on synthetic tasks for in-context inference. (Ch. 9) |
| **Proximal causal inference** | Identification using treatment-side and outcome-side proxies of a hidden confounder. (Ch. 9) |
| **Robustness value** | The equal partial-$R^2$ strength of a confounder needed to reduce an estimate by a given fraction. (Ch. 8) |
| **Rosenbaum Γ** | Bound on the odds ratio of treatment within matched pairs due to hidden bias. (Ch. 8) |
| **Sequential ignorability** | No unmeasured confounding of treatment-outcome, treatment-mediator and mediator-outcome relations. (Ch. 8) |
| **SHD** | Structural Hamming distance between graphs. (Ch. 7) |
| **Transportability** | Licence to carry causal effects from one population to another. (Ch. 9) |
| **Varsortability** | Tendency of marginal variance to increase along the causal order in simulated data. (Ch. 7) |
