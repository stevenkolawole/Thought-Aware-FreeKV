# Thought-Aware-FreeKV — `verify_dips` analysis

Question: do cos_sim dips during decode actually correspond to ThinKV-style transition markers ("wait", "hmm", "actually", etc.) in the generated text?

Source: `modal_logs/profile_aime/profile_aime/{corr,recall,tokens}_*.csv` — 5 problems, 11,559 aligned decode steps.

**Transition-marked steps: 225 (1.9465% of all steps).** Markers determined by regex against the decoded token text (see `T_KEYWORDS` in `scripts/analyze_dips.py`).


## 1. cos_sim at transition vs non-transition

| flag | count | mean | std | median | min |
|---|---|---|---|---|---|
| other | 11,334 | 0.8912 | 0.0666 | 0.9032 | -0.5985 |
| transition | 225 | 0.7690 | 0.0173 | 0.7666 | 0.7659 |

## 2. P(transition) per cos_sim quintile

Quintile boundaries are adaptive to the observed distribution. If cos_sim dips predict transitions, the lowest quintile should have a much higher P(T).

| bucket | n | n_T | P(T) | cos range |
|---|---|---|---|---|
| Q1 (lowest) | 2,332 | 223 | 9.5626% | [-0.598, 0.852] |
| Q2 | 2,302 | 0 | 0.0000% | [0.852, 0.891] |
| Q3 | 2,306 | 0 | 0.0000% | [0.891, 0.912] |
| Q4 | 2,317 | 0 | 0.0000% | [0.912, 0.932] |
| Q5 (highest) | 2,302 | 2 | 0.0869% | [0.932, 0.974] |

## 3. Predictive quality

- **AUC of (1 − cos_sim)** as transition predictor: **0.9494** — 0.5 is chance, 1.0 is perfect.
- AUC of (1 − sim_ema): 0.8708

A value meaningfully above 0.5 is direct evidence for the project's premise that cos-sim dips carry thought-transition signal, independent of attention-weight features.

## 4. Per-problem transition-to-dip distance

For each transition-marked step, the distance to the nearest local cos_sim dip (rolling-median dip of >0.03 within a 31-step window).

| problem | #trans | #dips | median dist | ≤2 steps | ≤5 steps |
|---|---|---|---|---|---|
| `2024-I-1` | 7 | 299 | 0 | 100.0% | 100.0% |
| `2024-I-3` | 205 | 1943 | 0 | 100.0% | 100.0% |
| `2024-I-6` | 3 | 269 | 0 | 100.0% | 100.0% |
| `2024-II-12` | 8 | 585 | 0 | 100.0% | 100.0% |
| `2024-II-4` | 2 | 324 | 0 | 100.0% | 100.0% |

## 5. Plots

- [per_problem_transitions.png](plots/per_problem_transitions.png)
- [cos_sim_hist.png](plots/cos_sim_hist.png)
- [p_transition_by_quintile.png](plots/p_transition_by_quintile.png)