# Thought-Aware-FreeKV — `verify_dips` analysis

Question: do cos_sim dips during decode actually correspond to ThinKV-style transition markers ("wait", "hmm", "actually", etc.) in the generated text?

Source: `modal_logs/verify_dips/verify_dips/{corr,recall,tokens}_*.csv` — 5 problems, 33,221 aligned decode steps.

**Transition-marked steps: 425 (1.2793% of all steps).** Markers determined by regex against the decoded token text (see `T_KEYWORDS` in `scripts/analyze_dips.py`).


## 1. cos_sim at transition vs non-transition

| flag | count | mean | std | median | min |
|---|---|---|---|---|---|
| other | 32,796 | 0.8820 | 0.0610 | 0.8945 | -0.5977 |
| transition | 425 | 0.7745 | 0.0230 | 0.7656 | 0.7656 |

## 2. P(transition) per cos_sim quintile

Quintile boundaries are adaptive to the observed distribution. If cos_sim dips predict transitions, the lowest quintile should have a much higher P(T).

| bucket | n | n_T | P(T) | cos range |
|---|---|---|---|---|
| Q1 (lowest) | 6,872 | 418 | 6.0827% | [-0.598, 0.844] |
| Q2 | 6,911 | 1 | 0.0145% | [0.848, 0.883] |
| Q3 | 7,277 | 0 | 0.0000% | [0.887, 0.906] |
| Q4 | 6,022 | 1 | 0.0166% | [0.910, 0.926] |
| Q5 (highest) | 6,139 | 5 | 0.0814% | [0.930, 0.980] |

## 3. Predictive quality

- **AUC of (1 − cos_sim)** as transition predictor: **0.9128** — 0.5 is chance, 1.0 is perfect.
- AUC of (1 − sim_ema): 0.6991

A value meaningfully above 0.5 is direct evidence for the project's premise that cos-sim dips carry thought-transition signal, independent of attention-weight features.

## 4. Per-problem transition-to-dip distance

For each transition-marked step, the distance to the nearest local cos_sim dip (rolling-median dip of >0.03 within a 31-step window).

| problem | #trans | #dips | median dist | ≤2 steps | ≤5 steps |
|---|---|---|---|---|---|
| `2024-I-1` | 7 | 290 | 0 | 100.0% | 100.0% |
| `2024-I-2` | 112 | 4524 | 0 | 100.0% | 100.0% |
| `2024-I-3` | 19 | 752 | 0 | 94.7% | 100.0% |
| `2024-I-4` | 285 | 3222 | 0 | 100.0% | 100.0% |
| `2024-II-4` | 2 | 307 | 0 | 100.0% | 100.0% |

## 5. Plots

- [per_problem_transitions.png](plots/per_problem_transitions.png)
- [cos_sim_hist.png](plots/cos_sim_hist.png)
- [p_transition_by_quintile.png](plots/p_transition_by_quintile.png)