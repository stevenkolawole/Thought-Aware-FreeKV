# Thought-Aware-FreeKV — `verify_dips` analysis

Question: do cos_sim dips during decode actually correspond to ThinKV-style transition markers ("wait", "hmm", "actually", etc.) in the generated text?

Source: `modal_logs/full_aime/full_aime/{corr,recall,tokens}_*.csv` — 28 problems, 299,392 aligned decode steps.

**Transition-marked steps: 4,992 (1.6674% of all steps).** Markers determined by regex against the decoded token text (see `T_KEYWORDS` in `scripts/analyze_dips.py`).


## 1. cos_sim at transition vs non-transition

| flag | count | mean | std | median | min |
|---|---|---|---|---|---|
| other | 294,400 | 0.8912 | 0.0539 | 0.9039 | -0.5988 |
| transition | 4,992 | 0.7920 | 0.0426 | 0.7719 | 0.7657 |

## 2. P(transition) per cos_sim quintile

Quintile boundaries are adaptive to the observed distribution. If cos_sim dips predict transitions, the lowest quintile should have a much higher P(T).

| bucket | n | n_T | P(T) | cos range |
|---|---|---|---|---|
| Q1 (lowest) | 60,000 | 4,023 | 6.7050% | [-0.599, 0.849] |
| Q2 | 59,862 | 766 | 1.2796% | [0.849, 0.889] |
| Q3 | 59,795 | 0 | 0.0000% | [0.889, 0.915] |
| Q4 | 59,935 | 2 | 0.0033% | [0.915, 0.935] |
| Q5 (highest) | 59,800 | 201 | 0.3361% | [0.935, 0.991] |

## 3. Predictive quality

- **AUC of (1 − cos_sim)** as transition predictor: **0.8960** — 0.5 is chance, 1.0 is perfect.
- AUC of (1 − sim_ema): 0.7831

A value meaningfully above 0.5 is direct evidence for the project's premise that cos-sim dips carry thought-transition signal, independent of attention-weight features.

## 4. Per-problem transition-to-dip distance

For each transition-marked step, the distance to the nearest local cos_sim dip (rolling-median dip of >0.03 within a 31-step window).

| problem | #trans | #dips | median dist | ≤2 steps | ≤5 steps |
|---|---|---|---|---|---|
| `2024-I-1` | 7 | 288 | 0 | 100.0% | 100.0% |
| `2024-I-10` | 26 | 2797 | 0 | 100.0% | 100.0% |
| `2024-I-11` | 141 | 3380 | 0 | 98.6% | 99.3% |
| `2024-I-12` | 219 | 4047 | 0 | 100.0% | 100.0% |
| `2024-I-13` | 380 | 4456 | 0 | 100.0% | 100.0% |
| `2024-I-14` | 71 | 2722 | 0 | 95.8% | 100.0% |
| `2024-I-15` | 91 | 3708 | 0 | 98.9% | 98.9% |
| `2024-I-2` | 63 | 2645 | 0 | 98.4% | 100.0% |
| `2024-I-3` | 8 | 394 | 0 | 100.0% | 100.0% |
| `2024-I-4` | 465 | 4250 | 0 | 100.0% | 100.0% |
| `2024-I-7` | 3 | 484 | 0 | 100.0% | 100.0% |
| `2024-I-8` | 27 | 616 | 0 | 100.0% | 100.0% |
| `2024-I-9` | 1444 | 4339 | 0 | 99.9% | 100.0% |
| `2024-II-1` | 24 | 1337 | 0 | 100.0% | 100.0% |
| `2024-II-10` | 182 | 3814 | 0 | 99.5% | 100.0% |
| `2024-II-11` | 193 | 4029 | 1 | 100.0% | 100.0% |
| `2024-II-12` | 4 | 630 | 0 | 100.0% | 100.0% |
| `2024-II-14` | 32 | 3982 | 0 | 100.0% | 100.0% |
| `2024-II-15` | 262 | 3227 | 0 | 100.0% | 100.0% |
| `2024-II-2` | 292 | 3806 | 0 | 99.3% | 100.0% |
| `2024-II-3` | 80 | 4277 | 0 | 100.0% | 100.0% |
| `2024-II-4` | 2 | 305 | 0 | 100.0% | 100.0% |
| `2024-II-5` | 473 | 3622 | 0 | 100.0% | 100.0% |
| `2024-II-6` | 168 | 4071 | 0 | 100.0% | 100.0% |
| `2024-II-7` | 104 | 4143 | 0 | 99.0% | 100.0% |
| `2024-II-8` | 115 | 3928 | 0 | 100.0% | 100.0% |
| `2024-II-9` | 116 | 3716 | 0 | 100.0% | 100.0% |

## 5. Plots

- [per_problem_transitions.png](plots/per_problem_transitions.png)
- [cos_sim_hist.png](plots/cos_sim_hist.png)
- [p_transition_by_quintile.png](plots/p_transition_by_quintile.png)