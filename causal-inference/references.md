# References and reading order

> Back to [README](README.md)

Everything in this guide is written in my own words. Where a fact or result comes from a source, it is cited inline and listed here. Links were checked during the research pass in November 2026. Preprints are marked *(preprint)*.

## Suggested reading order

**Stage 1: intuition (no heavy maths)**
1. Pearl & Mackenzie, *The Book of Why* (2018). The ladder of causation and why graphs matter.
2. Huntington-Klein, *The Effect*, Part I ([free online](https://theeffectbook.net/introduction.html)). Research design and causal diagrams, very readable.
3. Cunningham, *Causal Inference: The Mixtape*, chapters 1-4 ([free online](https://mixtape.scunning.com/)).

**Stage 2: the working toolkit**
4. Hernán & Robins, *Causal Inference: What If*, Parts I-II ([free PDF](https://miguelhernan.org/whatifbook)). Potential outcomes, DAGs, g-methods, IPW, standardisation.
5. Facure, *Causal Inference for the Brave and True* ([free, Python](https://matheusfacure.github.io/python-causality-handbook/)). Code-first coverage of most chapters here.
6. Angrist & Pischke, *Mostly Harmless Econometrics* ([publisher](https://press.princeton.edu/books/paperback/9780691120355/mostly-harmless-econometrics)). Regression, IV, DiD, RD from the econometrics side.
7. The Mixtape and The Effect, chapters on DiD, synthetic control, RD and IV.

**Stage 3: rigour**
8. Imbens & Rubin, *Causal Inference for Statistics, Social, and Biomedical Sciences* ([publisher](https://www.cambridge.org/core/books/causal-inference-for-statistics-social-and-biomedical-sciences/71126BE90C58F1A431FE9B2DD07938AB)).
9. Pearl, *Causality*, 2nd ed. ([publisher](https://www.cambridge.org/core/books/causality/B0046844FAE10CBF274D4ACBDAEB5F5B); [table of contents](http://bayes.cs.ucla.edu/BOOK-09/book-toc-final.pdf)).
10. Peters, Janzing & Schölkopf, *Elements of Causal Inference* ([MIT Press, open access](https://mitpress.mit.edu/books/elements-causal-inference)). SCMs, discovery and ML.

**Stage 4: modern methods (read the papers)**
11. Chernozhukov et al. on double ML. Wager & Athey and Athey, Tibshirani & Wager on forests. Künzel et al. and Nie & Wager on meta-learners.
12. The staggered-DiD papers: Goodman-Bacon; Callaway & Sant'Anna; Sun & Abraham; de Chaisemartin & D'Haultfœuille; Borusyak, Jaravel & Spiess. Then Baker et al.'s practitioner's guide and Rambachan & Roth.
13. Abadie et al. on synthetic control, and Arkhangelsky et al. on synthetic DiD.
14. Cinelli & Hazlett on sensitivity.

**Stage 5: frontier**
15. Chapter 9's sources: proximal inference, transportability, causal representation learning, causal foundation models, LLMs and causality.

---

## Books

| Book | Link |
|---|---|
| Pearl, J. *Causality: Models, Reasoning, and Inference*, 2nd ed. Cambridge UP, 2009 | [Cambridge](https://www.cambridge.org/core/books/causality/B0046844FAE10CBF274D4ACBDAEB5F5B) |
| Pearl, J. & Mackenzie, D. *The Book of Why*. Basic Books, 2018 | Publisher listing only. A companion summary of the causal hierarchy is at [UCLA](https://web.cs.ucla.edu/~kaoru/3-layer-causal-hierarchy.pdf) |
| Hernán, M. & Robins, J. *Causal Inference: What If*. CRC, 2020 (free) | [miguelhernan.org](https://miguelhernan.org/whatifbook) |
| Imbens, G. & Rubin, D. *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge UP, 2015 | [Cambridge](https://www.cambridge.org/core/books/causal-inference-for-statistics-social-and-biomedical-sciences/71126BE90C58F1A431FE9B2DD07938AB) |
| Cunningham, S. *Causal Inference: The Mixtape*. Yale UP, 2021 (free) | [mixtape.scunning.com](https://mixtape.scunning.com/) |
| Huntington-Klein, N. *The Effect*. CRC, 2021 (free) | [theeffectbook.net](https://theeffectbook.net/introduction.html) |
| Facure, M. *Causal Inference for the Brave and True* (free) | [GitHub Pages](https://matheusfacure.github.io/python-causality-handbook/) |
| Angrist, J. & Pischke, J.-S. *Mostly Harmless Econometrics*. Princeton UP, 2009 | [Princeton UP](https://press.princeton.edu/books/paperback/9780691120355/mostly-harmless-econometrics) |
| Peters, J., Janzing, D. & Schölkopf, B. *Elements of Causal Inference*. MIT Press, 2017 (open access) | [MIT Press](https://mitpress.mit.edu/books/elements-causal-inference) |
| van der Laan, M. & Rose, S. *Targeted Learning*. Springer, 2011 | [Springer chapter: Introduction to TMLE](https://link.springer.com/chapter/10.1007/978-1-4419-9782-1_4) |

## Foundations (Ch. 0-2)

* Pearl's three-layer causal hierarchy. [UCLA PDF](https://web.cs.ucla.edu/~kaoru/3-layer-causal-hierarchy.pdf)
* Shpitser, I. & Pearl, J. (2006). Identification of conditional interventional distributions; completeness of do-calculus. [arXiv 1206.6876](https://arxiv.org/abs/1206.6876)
* Huang, Y. & Valtorta, M. (2006). Pearl's calculus of intervention is complete. [arXiv 1206.6831](http://arxiv.org/pdf/1206.6831.pdf)
* Tikka, S. (2023). Identifying counterfactual queries with the R package cfid. *R Journal*. [PDF](https://journal.r-project.org/articles/RJ-2023-053/RJ-2023-053.pdf)
* Tikka, S. & Karvanen, J. Identifying causal effects with the R package causaleffect. [arXiv 1806.07161](https://arxiv.org/html/1806.07161)
* Rosenbaum, P. & Rubin, D. (1983). The central role of the propensity score. *Biometrika*. [PDF](https://www.math.mcgill.ca/dstephens/SISCR2017/Articles/Rosenbaum-Rubin-Bka83.pdf)
* A distributional perspective on Pearl's causal hierarchy *(preprint)*. [arXiv 2601.20405](https://arxiv.org/html/2601.20405v2)

## Experiments (Ch. 3)

* Deng, A., Xu, Y., Kohavi, R. & Walker, T. (2013). Improving the sensitivity of online controlled experiments by utilizing pre-experiment data (CUPED). WSDM. [ACM](https://dl.acm.org/doi/10.1145/2433396.2433413) · [PDF](https://ai.stanford.edu/~ronnyk/2013-02CUPEDImprovingSensitivityOfControlledExperiments.pdf)
* Johari, R., Koomen, P., Pekelis, L. & Walsh, D. (2017/2022). Peeking at A/B tests; Always valid inference. KDD / *Operations Research*. [arXiv 1512.04922](https://arxiv.org/abs/1512.04922) · [ACM](https://dl.acm.org/doi/10.1145/3097983.3097992) · [INFORMS](https://pubsonline.informs.org/doi/abs/10.1287/opre.2021.2135)
* Variance reduction combining pre- and in-experiment data (2024). [arXiv 2410.09027](https://arxiv.org/html/2410.09027v1)
* Aronow, P. & Samii, C. (2017). Estimating average causal effects under general interference. *Annals of Applied Statistics*. [arXiv 1305.6156](https://arxiv.org/html/1305.6156v3)
* Sävje, F., Aronow, P. & Hudgens, M. (2021). Average treatment effects in the presence of unknown interference. *Annals of Statistics*. [arXiv 1711.06399](https://arxiv.org/html/1711.06399v2)

## Observational methods (Ch. 4)

* Imbens, G. & Angrist, J. (1994). Identification and estimation of local average treatment effects. *Econometrica*. [NBER t0118](https://www.nber.org/papers/t0118)
* Stock, J. & Yogo, M. (2005). Testing for weak instruments. [NBER t0284](https://www.nber.org/papers/t0284.pdf)
* Lee, D., McCrary, J., Moreira, M. & Porter, J. (2022). Valid t-ratio inference for IV (the tF procedure). *AER*. [PDF](https://eml.berkeley.edu/~jmccrary/tfNBERfinal_main_v2.pdf)
* Calonico, S., Cattaneo, M. & Titiunik, R. (2015). rdrobust: an R package for robust nonparametric inference in RD designs. *R Journal*. [R Journal](https://journal.r-project.org/articles/RJ-2015-004)
* Calonico, Cattaneo & Farrell. Optimal bandwidth choice for robust bias-corrected inference in RD. [arXiv 1809.00236](https://arxiv.org/html/1809.00236v2)
* rdrobust for Python. [PyPI](https://www.pypi.org/project/rdrobust/1.2.1)

## Panel methods (Ch. 5)

* Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. *J. Econometrics*. [NBER w25018](https://www.nber.org/papers/w25018)
* Callaway, B. & Sant'Anna, P. (2021). Difference-in-differences with multiple time periods. *J. Econometrics*. [arXiv 1803.09015](https://arxiv.org/abs/1803.09015v3)
* Sun, L. & Abraham, S. (2021). Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. *J. Econometrics*. [arXiv 1804.05785](https://arxiv.org/abs/1804.05785)
* de Chaisemartin, C. & D'Haultfœuille, X. (2020). Two-way fixed effects estimators with heterogeneous treatment effects. *AER*. [arXiv 1803.08807](https://arxiv.org/abs/1803.08807v7) · [NBER w25904](https://www.nber.org/papers/w25904)
* de Chaisemartin & D'Haultfœuille. TWFE and DiD estimators with several treatments. [arXiv 2012.10077](https://arxiv.org/abs/2012.10077v8)
* Borusyak, K., Jaravel, X. & Spiess, J. (2024). Revisiting event-study designs: robust and efficient estimation. *ReStud*. [arXiv 2108.12419](https://arxiv.org/abs/2108.12419)
* Callaway, B., Goodman-Bacon, A. & Sant'Anna, P. Difference-in-differences with a continuous treatment. [arXiv 2107.02637](https://arxiv.org/pdf/2107.02637v7.pdf)
* Baker, A. et al. (2025). Difference-in-differences designs: a practitioner's guide *(preprint)*. [arXiv 2503.13323](https://arxiv.org/html/2503.13323v3)
* Rambachan, A. & Roth, J. (2023). A more credible approach to parallel trends. *ReStud*. [MIT page](https://economics.mit.edu/people/faculty/ashesh-rambachan/more-credible-approach-parallel-trends/) · [HonestDiD](https://github.com/asheshrambachan/HonestDiD/blob/master/README.md)
* Cohort-anchored HonestDiD *(preprint)*. [arXiv 2509.01829](https://arxiv.org/html/2509.01829)
* Difference-in-differences with multiple events *(preprint)*. [arXiv 2409.05184](https://arxiv.org/html/2409.05184v1)
* Abadie, A., Diamond, A. & Hainmueller, J. (2010). Synthetic control methods for comparative case studies. *JASA*. [MIT DSpace](https://dspace.mit.edu/handle/1721.1/59447) · [NBER w12831](https://www.nber.org/system/files/working_papers/w12831/w12831.pdf)
* Arkhangelsky, D., Athey, S., Hirshberg, D., Imbens, G. & Wager, S. (2021). Synthetic difference in differences. *AER*. [arXiv 1812.09970](https://arxiv.org/abs/1812.09970)
* Large-sample properties of synthetic control under selection on unobservables *(preprint)*. [arXiv 2311.13575](https://arxiv.org/html/2311.13575v2)
* Identification and inference for synthetic controls with confounding *(preprint)*. [arXiv 2312.00955](https://arxiv.org/html/2312.00955)
* Brodersen, K. et al. (2015). Inferring causal impact using Bayesian structural time-series models. *Annals of Applied Statistics*. [arXiv 1506.00356](https://arxiv.org/pdf/1506.00356.pdf) · [R package docs](https://google.github.io/CausalImpact/CausalImpact.html)

## Causal ML (Ch. 6)

* Chernozhukov, V. et al. (2018). Double/debiased machine learning for treatment and structural parameters. *Econometrics Journal*. [arXiv 1608.00060](https://arxiv.org/abs/1608.00060)
* Wager, S. & Athey, S. (2018). Estimation and inference of heterogeneous treatment effects using random forests. *JASA*. [arXiv 1510.04342](https://arxiv.org/abs/1510.04342)
* Athey, S., Tibshirani, J. & Wager, S. (2019). Generalized random forests. *Annals of Statistics*. [arXiv 1610.01271](https://arxiv.org/abs/1610.01271)
* Künzel, S., Sekhon, J., Bickel, P. & Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. *PNAS*. [PNAS](https://www.pnas.org/doi/10.1073/pnas.1804597116) · [arXiv 1706.03461](https://arxiv.org/html/1706.03461v6)
* Nie, X. & Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment effects. *Biometrika*. [arXiv 1712.04912](https://arxiv.org/abs/1712.04912) · [rlearner](https://github.com/xnie/rlearner/blob/master/README.md)
* What makes forest-based heterogeneous treatment effect estimators work? [arXiv 2206.10323](http://arxiv.org/pdf/2206.10323v1)
* On honest estimation in causal forests *(preprint)*. [arXiv 2506.13107](https://arxiv.org/pdf/2506.13107v4)
* Debiased front-door learners for heterogeneous effects *(preprint)*. [arXiv 2509.22531](https://arxiv.org/html/2509.22531v1)
* van der Laan, M. & Rose, S. Introduction to TMLE. [Springer](https://link.springer.com/chapter/10.1007/978-1-4419-9782-1_4) · Gruber & van der Laan, a gentle introduction. [PDF](https://www.stat.cmu.edu/~ryantibs/journalclub/gruber_2009.pdf)
* Chen, H. et al. (2020). CausalML: Python package for causal machine learning. [arXiv 2002.11631](https://arxiv.org/abs/2002.11631) · [GitHub](https://github.com/uber/causalml)

## Causal discovery (Ch. 7)

* Chickering, D. M. (2002). Optimal structure identification with greedy search (GES). *JMLR*. [JMLR](https://jmlr.org/papers/v3/chickering02b.html)
* Shimizu, S. et al. (2006). A linear non-Gaussian acyclic model for causal discovery. *JMLR*. [JMLR](https://jmlr.csail.mit.edu/papers/v7/shimizu06a.html)
* Ikeuchi, T. et al. (2023). Python package for causal discovery based on LiNGAM. *JMLR*. [PDF](https://jmlr.org/papers/volume24/21-0321/21-0321.pdf)
* Zheng, X., Aragam, B., Ravikumar, P. & Xing, E. (2018). DAGs with NO TEARS. NeurIPS. [arXiv 1803.01422](https://arxiv.org/abs/1803.01422) · [PDF](http://www.cs.cmu.edu/~epxing/papers/2018/Zheng_etal_nips18.pdf)
* Reisach, A., Seiler, C. & Weichwald, S. (2021). Beware of the simulated DAG! NeurIPS. [arXiv 2102.13647](https://arxiv.org/html/2102.13647v3)
* Reisach, A. et al. (2023). A scale-invariant sorting criterion to find a causal order in additive noise models. [arXiv 2303.18211](https://arxiv.org/html/2303.18211)
* Seng, J. et al. (2022). Tearing apart NOTEARS. [arXiv 2206.07195](https://arxiv.org/pdf/2206.07195.pdf)
* Zheng, Y. et al. (2024). Causal-learn: causal discovery in Python. *JMLR*. [JMLR](https://jmlr.org/papers/volume25/23-0970/23-0970.pdf) · [GitHub](https://github.com/py-why/causal-learn)
* Zhang, K. et al. gCastle: a Python toolbox for causal discovery. [arXiv 2111.15155](https://arxiv.org/abs/2111.15155)
* Kıcıman, E., Ness, R., Sharma, A. & Tan, C. (2023). Causal reasoning and large language models. [arXiv 2305.00050](https://arxiv.org/abs/2305.00050)
* Large language models for causal discovery: survey *(preprint)*. [arXiv 2402.11068](https://arxiv.org/abs/2402.11068)
* LLMs as imperfect experts for causal structure priors *(preprint)*. [arXiv 2405.13551](https://arxiv.org/pdf/2405.13551)

## Sensitivity, mediation, validation (Ch. 8)

* Cinelli, C. & Hazlett, C. (2020). Making sense of sensitivity: extending omitted variable bias. *JRSS-B*. [PDF](https://escholarship.org/content/qt0pg6471b/qt0pg6471b.pdf) · [sensemakr](https://carloscinelli.com/sensemakr/articles/sensemakr.html)
* Cinelli, C. & Hazlett, C. An omitted variable bias framework for sensitivity analysis of IV. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4217915)
* VanderWeele, T. & Ding, P. (2017). Sensitivity analysis in observational research: introducing the E-value. *Annals of Internal Medicine*. [PubMed](https://pubmed.ncbi.nlm.nih.gov/28693043/)
* Ioannidis, J., Tan, Y. J. & Blum, M. (2019). Limitations and misinterpretations of E-values. [PubMed](https://pubmed.ncbi.nlm.nih.gov/30597486)
* Imai, K., Keele, L. & Tingley, D. (2010). A general approach to causal mediation analysis. *Psychological Methods*. [PDF](https://lukekeele.net/wp-content/uploads/2016/03/mediationII.pdf)
* Sharma, A. & Kıcıman, E. DoWhy: an end-to-end library for causal inference. [arXiv 2011.04216](https://arxiv.org/pdf/2011.04216.pdf) · [docs](https://www.pywhy.org/dowhy/main/user_guide/causal_tasks/estimating_causal_effects/index.html)

## Frontier (Ch. 9)

* Tchetgen Tchetgen, E. et al. (2020). An introduction to proximal causal learning. [arXiv 2009.10982](https://arxiv.org/pdf/2009.10982v1.pdf)
* Miao, W. et al. A confounding bridge approach for double negative control inference. [arXiv 1808.04945](https://arxiv.org/html/1808.04945v4)
* Regression-based proximal causal inference *(preprint)*. [arXiv 2402.00335](https://arxiv.org/abs/2402.00335v2)
* Proximal causal learning of heterogeneous treatment effects (P-learner). [arXiv 2301.10913](https://arxiv.org/html/2301.10913v1)
* Fortified proximal causal inference with many invalid proxies *(preprint)*. [arXiv 2506.13152](https://arxiv.org/html/2506.13152v1)
* Bias formulas for violations of proximal identification assumptions. [arXiv 2208.00105](https://arxiv.org/html/2208.00105)
* Demystifying proximal causal inference *(preprint)*. [arXiv 2512.24413](https://arxiv.org/html/2512.24413v1)
* Bareinboim, E. & Pearl, J. (2016). Causal inference and the data-fusion problem. *PNAS*. [PNAS](https://www.pnas.org/doi/10.1073/pnas.1510507113) · Transportability completeness results: [UCLA](https://ftp.cs.ucla.edu/pub/stat_ser/r390-L.pdf)
* Hünermund, P. & Bareinboim, E. Causal inference and data fusion in econometrics. [arXiv 1912.09104](https://arxiv.org/abs/1912.09104v3)
* Schölkopf, B. et al. (2021). Toward causal representation learning. *Proc. IEEE*. [arXiv 2102.11107](https://arxiv.org/html/2102.11107)
* Unifying causal representation learning with the invariance principle *(preprint)*. [arXiv 2409.02772](https://arxiv.org/html/2409.02772v2)
* Unifying causal representation learning and foundation models *(preprint)*. [arXiv 2402.09236](https://arxiv.org/html/2402.09236v2)
* CausalPFN: amortized causal effect estimation via in-context learning *(preprint)*. [arXiv 2506.07918](https://arxiv.org/html/2506.07918v1)
* Do-PFN: in-context learning for causal effect estimation *(preprint)*. [arXiv 2506.06039](http://arxiv.org/pdf/2506.06039v1)
* Foundation models for causal inference via prior-data fitted networks (CausalFM) *(preprint)*. [arXiv 2506.10914](https://arxiv.org/abs/2506.10914v1)
* Causal foundation models with continuous treatments *(preprint)*. [arXiv 2605.15133](https://arxiv.org/html/2605.15133v1)
* Jin, Z. et al. (2023). CLadder: assessing causal reasoning in language models. NeurIPS. [arXiv 2312.04350](https://arxiv.org/abs/2312.04350v2) · [code](https://github.com/causalNLP/cladder)
* Evaluating interventional reasoning capabilities of LLMs *(preprint)*. [arXiv 2404.05545](https://arxiv.org/html/2404.05545v1)
* A critical review of causal reasoning benchmarks for LLMs. [arXiv 2407.08029](https://arxiv.org/html/2407.08029v1)
* Zečević, M. et al. (2023). Causal parrots: LLMs may talk causality but are not causal. *TMLR*. [arXiv 2308.13067](https://arxiv.org/html/2308.13067)
* Causal tongue-tie: LLMs can encode causal direction, but their yes/no outputs fail to express *(preprint, 2026)*. [arXiv 2605.25891](https://arxiv.org/html/2605.25891v1)
* A case study in the challenges of procedural generation (causal benchmarks) *(preprint)*. [arXiv 2610.38100](https://arxiv.org/html/2610.38100)
* Compared to what? Baselines and metrics for counterfactual prompting *(preprint)*. [arXiv 2605.01048](https://arxiv.org/html/2605.01048v1)
* Estimating causal effects of text interventions leveraging LLMs *(preprint)*. [arXiv 2410.21474](https://arxiv.org/pdf/2410.21474v1)
* Large language models as nondeterministic causal models *(preprint)*. [arXiv 2509.22297](https://arxiv.org/abs/2509.22297)
* Causality is key for interpretability claims to generalise *(preprint)*. [arXiv 2602.16698](https://arxiv.org/html/2602.16698)
* Causal methods for LLM development and evaluation *(preprint)*. [arXiv 2605.25998](https://arxiv.org/pdf/2605.25998v1.pdf)
* Variance reduction for heavy-tailed monetization metrics via post-stratification *(preprint)*. [arXiv 2606.04110](https://arxiv.org/html/2606.04110)
* Multivariate covariate adjustment for switchback experiments *(preprint)*. [arXiv 2608.24038](https://arxiv.org/html/2608.24038v1)

## Software

* DoWhy: [docs](https://www.pywhy.org/dowhy/v0.8/getting_started/intro.html). causal-learn: [GitHub](https://github.com/py-why/causal-learn). CausalML: [GitHub](https://github.com/uber/causalml). gCastle: [arXiv](https://arxiv.org/abs/2111.15155). CausalImpact: [docs](https://google.github.io/CausalImpact/CausalImpact.html). rdrobust: [R Journal](https://journal.r-project.org/articles/RJ-2015-004). sensemakr: [docs](https://carloscinelli.com/sensemakr/articles/sensemakr.html). HonestDiD: [GitHub](https://github.com/asheshrambachan/HonestDiD/blob/master/README.md).
* EconML, DoubleML, grf, did, Synth, synthdid, fixest, MatchIt, tmle, mediation, dagitty: see each project's CRAN or PyPI page. They are listed in [Chapter 10](docs/10-tooling-and-cheatsheet.md) and were not separately web-verified for this guide.

*Content from sources was paraphrased for compliance with licensing restrictions. No passage longer than a short phrase is quoted.*
