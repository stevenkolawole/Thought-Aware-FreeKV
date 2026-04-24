# Thought-Aware-FreeKV analysis — run `baseline`

Source: `modal_logs/baseline/baseline/{corr,recall}_*.csv` (14 problems, 32 layers)

## 1. Headline correction rate

- **Overall**: 0.8936 (4,977,134 of 5,570,016 (step, layer) checks triggered a correction)
- **Per-problem mean**: 0.8917, median 0.8951, range [0.8631, 0.9204]
- Paper's AIME24 claim: 43–52%. **Does not match — investigate.**

### Per-problem

| Problem | Corr rate | Steps | Recall GB |
|---|---|---|---|
| `2024-II-4` | 0.8631 | 3,344 | 29.77 |
| `2024-I-2` | 0.8631 | 11,196 | 379.22 |
| `2024-II-3` | 0.8661 | 31,999 | 1289.63 |
| `2024-II-12` | 0.8670 | 4,202 | 61.48 |
| `2024-I-4` | 0.8749 | 21,891 | 997.62 |
| `2024-I-7` | 0.8873 | 6,754 | 170.94 |
| `2024-II-6` | 0.8922 | 4,865 | 82.30 |
| `2024-I-1` | 0.8980 | 2,993 | 21.11 |
| `2024-I-12` | 0.9013 | 31,999 | 1551.62 |
| `2024-I-11` | 0.9102 | 9,537 | 303.47 |
| `2024-I-8` | 0.9111 | 31,999 | 1584.77 |
| `2024-II-11` | 0.9141 | 31,999 | 1699.48 |
| `2024-I-3` | 0.9151 | 3,749 | 39.93 |
| `2024-II-7` | 0.9204 | 4,607 | 74.52 |

## 2. Thought-type co-occurrence with corrections

Thought-type distribution (per decode step, layer 0):

- **R**: 0.9903
- **E**: 0.0097
- **T**: 0.0001

Correction rate conditioned on thought_type (all layers):

| thought_type | P(need_corr) | rows |
|---|---|---|
| R | 0.8938 | 5,515,744 |
| E | 0.8732 | 53,984 |
| T | 1.0000 | 288 |

Cross-tab (thought × need_corr, all layers):

```
               no_corr     corr    total
thought_label                           
E                 6843    47141    53984
R               586039  4929705  5515744
T                    0      288      288
All             592882  4977134  5570016
```

## 3. `thought_type == T` as a correction predictor

- precision = 1.0000
- recall    = 0.0001
- F1        = 0.0001
- tp=288 fp=0 fn=4,976,846 tn=592,882

Interpretation: a naive "predict correction iff thought==T" classifier. Low recall means most corrections happen during R/E; if so, the EMA thresholds need retuning or the cos-sim signal itself is the better direct predictor.

## 4. Per-layer correction rate

| layer | P(need_corr) |
|---|---|
| 0 | 0.8184 |
| 1 | 0.3153 |
| 2 | 0.9974 |
| 3 | 0.4127 |
| 4 | 0.9433 |
| 5 | 0.9594 |
| 6 | 0.9533 |
| 7 | 0.8769 |
| 8 | 0.9671 |
| 9 | 0.8826 |
| 10 | 0.9978 |
| 11 | 0.9328 |
| 12 | 0.5631 |
| 13 | 0.7836 |
| 14 | 0.9926 |
| 15 | 0.9777 |
| 16 | 0.9837 |
| 17 | 0.8572 |
| 18 | 0.7469 |
| 19 | 0.9640 |
| 20 | 0.9708 |
| 21 | 0.9480 |
| 22 | 0.9210 |
| 23 | 0.9535 |
| 24 | 0.9839 |
| 25 | 0.9914 |
| 26 | 0.9850 |
| 27 | 0.9927 |
| 28 | 0.9897 |
| 29 | 0.9996 |
| 30 | 0.9727 |
| 31 | 0.9598 |

## 5. Plots

- [per_problem_timeseries.png](plots/per_problem_timeseries.png)
- [cos_sim_by_thought.png](plots/cos_sim_by_thought.png)
- [corr_rate_per_layer.png](plots/corr_rate_per_layer.png)
- [per_problem_rate_and_bw.png](plots/per_problem_rate_and_bw.png)