# Thought-Aware-FreeKV — `verify_dips` analysis

Question: do cos_sim dips during decode actually correspond to ThinKV-style transition markers ("wait", "hmm", "actually", etc.) in the generated text?

Source: `modal_logs/math50/math50/{corr,recall,tokens}_*.csv` — 28 problems, 161,008 aligned decode steps.

**Transition-marked steps: 2,170 (1.3478% of all steps).** Markers determined by regex against the decoded token text (see `T_KEYWORDS` in `scripts/analyze_dips.py`).


## 1. cos_sim at transition vs non-transition

| flag | count | mean | std | median | min |
|---|---|---|---|---|---|
| other | 158,838 | 0.8880 | 0.0573 | 0.8981 | -0.5988 |
| transition | 2,170 | 0.7764 | 0.0290 | 0.7667 | 0.7604 |

## 2. P(transition) per cos_sim quintile

Quintile boundaries are adaptive to the observed distribution. If cos_sim dips predict transitions, the lowest quintile should have a much higher P(T).

| bucket | n | n_T | P(T) | cos range |
|---|---|---|---|---|
| Q1 (lowest) | 32,216 | 2,100 | 6.5185% | [-0.599, 0.848] |
| Q2 | 32,296 | 18 | 0.0557% | [0.848, 0.885] |
| Q3 | 32,097 | 0 | 0.0000% | [0.885, 0.911] |
| Q4 | 32,218 | 4 | 0.0124% | [0.911, 0.933] |
| Q5 (highest) | 32,181 | 48 | 0.1492% | [0.933, 0.990] |

## 3. Predictive quality

- **AUC of (1 − cos_sim)** as transition predictor: **0.9285** — 0.5 is chance, 1.0 is perfect.
- AUC of (1 − sim_ema): 0.7729

A value meaningfully above 0.5 is direct evidence for the project's premise that cos-sim dips carry thought-transition signal, independent of attention-weight features.

## 4. Per-problem transition-to-dip distance

For each transition-marked step, the distance to the nearest local cos_sim dip (rolling-median dip of >0.03 within a 31-step window).

| problem | #trans | #dips | median dist | ≤2 steps | ≤5 steps |
|---|---|---|---|---|---|
| `test_algebra_1349_json` | 5 | 407 | 0 | 100.0% | 100.0% |
| `test_algebra_1837_json` | 94 | 3959 | 0 | 100.0% | 100.0% |
| `test_algebra_2427_json` | 83 | 5288 | 0 | 98.8% | 100.0% |
| `test_counting_and_probability_119_json` | 5 | 405 | 0 | 100.0% | 100.0% |
| `test_counting_and_probability_134_json` | 235 | 4348 | 0 | 100.0% | 100.0% |
| `test_counting_and_probability_525_json` | 81 | 2291 | 0 | 98.8% | 100.0% |
| `test_geometry_434_json` | 240 | 4609 | 0 | 100.0% | 100.0% |
| `test_geometry_627_json` | 33 | 810 | 0 | 100.0% | 100.0% |
| `test_intermediate_algebra_1000_json` | 10 | 482 | 0 | 100.0% | 100.0% |
| `test_intermediate_algebra_1197_json` | 3 | 586 | 0 | 100.0% | 100.0% |
| `test_intermediate_algebra_1388_json` | 11 | 385 | 0 | 100.0% | 100.0% |
| `test_intermediate_algebra_1454_json` | 188 | 4161 | 0 | 100.0% | 100.0% |
| `test_intermediate_algebra_1994_json` | 6 | 905 | 0 | 100.0% | 100.0% |
| `test_intermediate_algebra_428_json` | 2 | 311 | 0 | 100.0% | 100.0% |
| `test_intermediate_algebra_607_json` | 30 | 1433 | 0 | 100.0% | 100.0% |
| `test_number_theory_1032_json` | 195 | 4279 | 0 | 100.0% | 100.0% |
| `test_number_theory_515_json` | 7 | 327 | 0 | 100.0% | 100.0% |
| `test_number_theory_627_json` | 14 | 496 | 0 | 100.0% | 100.0% |
| `test_number_theory_737_json` | 3 | 202 | 0 | 100.0% | 100.0% |
| `test_number_theory_864_json` | 10 | 2309 | 0 | 80.0% | 100.0% |
| `test_prealgebra_1139_json` | 885 | 4237 | 0 | 100.0% | 100.0% |
| `test_precalculus_1199_json` | 8 | 443 | 0 | 100.0% | 100.0% |
| `test_precalculus_285_json` | 10 | 759 | 0 | 100.0% | 100.0% |
| `test_precalculus_990_json` | 12 | 652 | 0 | 100.0% | 100.0% |

## 5. Plots

- [per_problem_transitions.png](plots/per_problem_transitions.png)
- [cos_sim_hist.png](plots/cos_sim_hist.png)
- [p_transition_by_quintile.png](plots/p_transition_by_quintile.png)