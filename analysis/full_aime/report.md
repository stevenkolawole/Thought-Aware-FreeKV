# `full_aime` — per-head cosine-similarity analysis

Uses the full `[n_steps, n_layers=32, n_q_heads=32]` cosine-sim tensor cached per problem. This lets us compute any aggregation over heads without re-running.

Problems included: `2024-I-1`, `2024-I-10`, `2024-I-11`, `2024-I-12`, `2024-I-13`, `2024-I-14`, `2024-I-15`, `2024-I-2`, `2024-I-3`, `2024-I-4`, `2024-I-6`, `2024-I-7`, `2024-I-8`, `2024-I-9`, `2024-II-1`, `2024-II-10`, `2024-II-11`, `2024-II-12`, `2024-II-14`, `2024-II-15`, `2024-II-2`, `2024-II-3`, `2024-II-4`, `2024-II-5`, `2024-II-6`, `2024-II-7`, `2024-II-8`, `2024-II-9`


## Global rates at τ=0.9 (for quick comparison)

| pid | n_valid steps | need_corr | per-q-head | per-kv-head | mean<0.9 |
|---|---|---|---|---|---|
| `2024-I-1` | 1,104 | 0.8995 | 0.5724 | 0.6063 | 0.6702 |
| `2024-I-10` | 14,693 | 0.9026 | 0.5149 | 0.5419 | 0.5939 |
| `2024-I-11` | 14,430 | 0.9013 | 0.5518 | 0.5858 | 0.6482 |
| `2024-I-12` | 14,435 | 0.8941 | 0.5541 | 0.5829 | 0.6364 |
| `2024-I-13` | 14,419 | 0.8949 | 0.5258 | 0.5553 | 0.6122 |
| `2024-I-14` | 11,964 | 0.8922 | 0.5568 | 0.5838 | 0.6374 |
| `2024-I-15` | 14,449 | 0.9053 | 0.5755 | 0.6075 | 0.6716 |
| `2024-I-2` | 10,273 | 0.8640 | 0.5295 | 0.5567 | 0.6051 |
| `2024-I-3` | 1,528 | 0.9116 | 0.5679 | 0.6032 | 0.6723 |
| `2024-I-4` | 14,477 | 0.9232 | 0.6096 | 0.6413 | 0.6993 |
| `2024-I-6` | 731 | 0.9118 | 0.5615 | 0.5992 | 0.6749 |
| `2024-I-7` | 2,332 | 0.8777 | 0.5428 | 0.5682 | 0.6184 |
| `2024-I-8` | 2,450 | 0.9108 | 0.5773 | 0.6135 | 0.6852 |
| `2024-I-9` | 14,436 | 0.8778 | 0.4748 | 0.4992 | 0.5336 |
| `2024-II-1` | 4,623 | 0.9153 | 0.5887 | 0.6243 | 0.6839 |
| `2024-II-10` | 14,589 | 0.9189 | 0.5855 | 0.6196 | 0.6879 |
| `2024-II-11` | 14,420 | 0.9127 | 0.5782 | 0.6070 | 0.6624 |
| `2024-II-12` | 2,662 | 0.8715 | 0.5285 | 0.5544 | 0.6013 |
| `2024-II-14` | 14,472 | 0.9238 | 0.6073 | 0.6453 | 0.7342 |
| `2024-II-15` | 14,406 | 0.9036 | 0.5413 | 0.5723 | 0.6401 |
| `2024-II-2` | 14,424 | 0.9051 | 0.5600 | 0.5911 | 0.6479 |
| `2024-II-3` | 14,484 | 0.9180 | 0.5682 | 0.6063 | 0.6794 |
| `2024-II-4` | 1,382 | 0.8616 | 0.5279 | 0.5525 | 0.5943 |
| `2024-II-5` | 14,410 | 0.8966 | 0.5517 | 0.5835 | 0.6434 |
| `2024-II-6` | 14,413 | 0.9006 | 0.5663 | 0.5968 | 0.6587 |
| `2024-II-7` | 14,422 | 0.9269 | 0.5840 | 0.6239 | 0.7015 |
| `2024-II-8` | 14,513 | 0.9113 | 0.5720 | 0.6063 | 0.6714 |
| `2024-II-9` | 14,451 | 0.9058 | 0.5324 | 0.5646 | 0.6332 |


## Problem `2024-I-1`

- Decode steps logged: **1,104** (of 3,000 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **11** (0.3665% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9977 | 0.9999 | 0.9688 | 0.9321 | 0.9036 |
| 0.90 | 0.8995 | 0.9840 | 0.6702 | 0.6063 | 0.5724 |
| 0.85 | 0.6750 | 0.9028 | 0.2785 | 0.3031 | 0.3015 |
| 0.80 | 0.3984 | 0.7424 | 0.1032 | 0.1384 | 0.1527 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.1005 |
| 1 | 0.0909 |
| 2 | 0.0747 |
| 3 | 0.0909 |
| 4 | 0.0670 |
| 5 | 0.0716 |
| 6 | 0.1032 |
| 7 | 0.1176 |
| 8 | 0.2837 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6304 |
| 1 | 0.5890 |
| 2 | 0.5958 |
| 3 | 0.5727 |
| 4 | 0.6082 |
| 5 | 0.6307 |
| 6 | 0.6308 |
| 7 | 0.5930 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.822 |  | 8 | 0.970 |  | 16 | 0.972 |  | 24 | 0.969 |
| 1 | 0.388 |  | 9 | 0.930 |  | 17 | 0.875 |  | 25 | 0.966 |
| 2 | 0.994 |  | 10 | 0.996 |  | 18 | 0.824 |  | 26 | 0.975 |
| 3 | 0.462 |  | 11 | 0.967 |  | 19 | 0.912 |  | 27 | 0.993 |
| 4 | 0.949 |  | 12 | 0.628 |  | 20 | 0.924 |  | 28 | 0.990 |
| 5 | 0.966 |  | 13 | 0.845 |  | 21 | 0.907 |  | 29 | 0.998 |
| 6 | 0.967 |  | 14 | 0.991 |  | 22 | 0.906 |  | 30 | 0.973 |
| 7 | 0.921 |  | 15 | 0.945 |  | 23 | 0.915 |  | 31 | 0.944 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-I-1.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-I-1.png) · [head_correlation](plots/head_correlation_2024-I-1.png) · [n_drifted_hist](plots/n_drifted_hist_2024-I-1.png)


## Problem `2024-I-10`

- Decode steps logged: **14,693** (of 16,383 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **50** (0.3052% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9992 | 1.0000 | 0.9793 | 0.9363 | 0.9010 |
| 0.90 | 0.9026 | 0.9922 | 0.5939 | 0.5419 | 0.5149 |
| 0.85 | 0.6218 | 0.9173 | 0.2051 | 0.2433 | 0.2490 |
| 0.80 | 0.3478 | 0.7266 | 0.0678 | 0.1034 | 0.1197 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0974 |
| 1 | 0.1141 |
| 2 | 0.1147 |
| 3 | 0.1194 |
| 4 | 0.0839 |
| 5 | 0.0730 |
| 6 | 0.0833 |
| 7 | 0.0803 |
| 8 | 0.2339 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.5767 |
| 1 | 0.5336 |
| 2 | 0.5150 |
| 3 | 0.5338 |
| 4 | 0.5201 |
| 5 | 0.5776 |
| 6 | 0.5560 |
| 7 | 0.5225 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.884 |  | 8 | 0.971 |  | 16 | 0.983 |  | 24 | 0.988 |
| 1 | 0.302 |  | 9 | 0.854 |  | 17 | 0.887 |  | 25 | 0.992 |
| 2 | 0.999 |  | 10 | 0.998 |  | 18 | 0.812 |  | 26 | 0.990 |
| 3 | 0.408 |  | 11 | 0.960 |  | 19 | 0.972 |  | 27 | 0.997 |
| 4 | 0.975 |  | 12 | 0.513 |  | 20 | 0.978 |  | 28 | 0.996 |
| 5 | 0.983 |  | 13 | 0.757 |  | 21 | 0.953 |  | 29 | 1.000 |
| 6 | 0.977 |  | 14 | 0.996 |  | 22 | 0.954 |  | 30 | 0.984 |
| 7 | 0.870 |  | 15 | 0.989 |  | 23 | 0.967 |  | 31 | 0.990 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-I-10.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-I-10.png) · [head_correlation](plots/head_correlation_2024-I-10.png) · [n_drifted_hist](plots/n_drifted_hist_2024-I-10.png)


## Problem `2024-I-11`

- Decode steps logged: **14,430** (of 16,383 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **164** (1.0010% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9988 | 1.0000 | 0.9733 | 0.9362 | 0.9052 |
| 0.90 | 0.9013 | 0.9885 | 0.6482 | 0.5858 | 0.5518 |
| 0.85 | 0.6430 | 0.9086 | 0.2353 | 0.2632 | 0.2660 |
| 0.80 | 0.3501 | 0.7261 | 0.0634 | 0.1038 | 0.1223 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0987 |
| 1 | 0.1006 |
| 2 | 0.0849 |
| 3 | 0.0912 |
| 4 | 0.0745 |
| 5 | 0.0754 |
| 6 | 0.1046 |
| 7 | 0.1212 |
| 8 | 0.2489 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6174 |
| 1 | 0.5633 |
| 2 | 0.5822 |
| 3 | 0.5482 |
| 4 | 0.5850 |
| 5 | 0.6000 |
| 6 | 0.5997 |
| 7 | 0.5903 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.793 |  | 8 | 0.966 |  | 16 | 0.992 |  | 24 | 0.970 |
| 1 | 0.383 |  | 9 | 0.921 |  | 17 | 0.864 |  | 25 | 0.994 |
| 2 | 0.998 |  | 10 | 0.998 |  | 18 | 0.795 |  | 26 | 0.981 |
| 3 | 0.504 |  | 11 | 0.972 |  | 19 | 0.970 |  | 27 | 0.977 |
| 4 | 0.907 |  | 12 | 0.618 |  | 20 | 0.970 |  | 28 | 0.979 |
| 5 | 0.941 |  | 13 | 0.856 |  | 21 | 0.939 |  | 29 | 0.999 |
| 6 | 0.944 |  | 14 | 0.995 |  | 22 | 0.930 |  | 30 | 0.940 |
| 7 | 0.882 |  | 15 | 0.991 |  | 23 | 0.944 |  | 31 | 0.927 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-I-11.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-I-11.png) · [head_correlation](plots/head_correlation_2024-I-11.png) · [n_drifted_hist](plots/n_drifted_hist_2024-I-11.png)


## Problem `2024-I-12`

- Decode steps logged: **14,435** (of 16,383 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **224** (1.3672% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9977 | 0.9999 | 0.9683 | 0.9309 | 0.9005 |
| 0.90 | 0.8941 | 0.9869 | 0.6364 | 0.5829 | 0.5541 |
| 0.85 | 0.6295 | 0.9019 | 0.2615 | 0.2900 | 0.2909 |
| 0.80 | 0.3611 | 0.7116 | 0.1192 | 0.1427 | 0.1547 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.1059 |
| 1 | 0.1025 |
| 2 | 0.0860 |
| 3 | 0.0940 |
| 4 | 0.0730 |
| 5 | 0.0699 |
| 6 | 0.0907 |
| 7 | 0.1035 |
| 8 | 0.2746 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6109 |
| 1 | 0.5679 |
| 2 | 0.5709 |
| 3 | 0.5507 |
| 4 | 0.5811 |
| 5 | 0.6061 |
| 6 | 0.5948 |
| 7 | 0.5808 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.817 |  | 8 | 0.959 |  | 16 | 0.980 |  | 24 | 0.984 |
| 1 | 0.313 |  | 9 | 0.907 |  | 17 | 0.841 |  | 25 | 0.982 |
| 2 | 1.000 |  | 10 | 0.996 |  | 18 | 0.775 |  | 26 | 0.976 |
| 3 | 0.428 |  | 11 | 0.946 |  | 19 | 0.941 |  | 27 | 0.987 |
| 4 | 0.906 |  | 12 | 0.676 |  | 20 | 0.967 |  | 28 | 0.990 |
| 5 | 0.946 |  | 13 | 0.864 |  | 21 | 0.925 |  | 29 | 0.999 |
| 6 | 0.948 |  | 14 | 0.992 |  | 22 | 0.908 |  | 30 | 0.946 |
| 7 | 0.888 |  | 15 | 0.967 |  | 23 | 0.914 |  | 31 | 0.943 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-I-12.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-I-12.png) · [head_correlation](plots/head_correlation_2024-I-12.png) · [n_drifted_hist](plots/n_drifted_hist_2024-I-12.png)


## Problem `2024-I-13`

- Decode steps logged: **14,419** (of 16,383 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **385** (2.3499% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9989 | 1.0000 | 0.9757 | 0.9354 | 0.9001 |
| 0.90 | 0.8949 | 0.9905 | 0.6122 | 0.5553 | 0.5258 |
| 0.85 | 0.6261 | 0.9086 | 0.2214 | 0.2568 | 0.2601 |
| 0.80 | 0.3404 | 0.7092 | 0.0851 | 0.1164 | 0.1312 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.1051 |
| 1 | 0.1088 |
| 2 | 0.0958 |
| 3 | 0.1071 |
| 4 | 0.0840 |
| 5 | 0.0751 |
| 6 | 0.0942 |
| 7 | 0.0962 |
| 8 | 0.2339 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.5856 |
| 1 | 0.5470 |
| 2 | 0.5339 |
| 3 | 0.5168 |
| 4 | 0.5570 |
| 5 | 0.5802 |
| 6 | 0.5632 |
| 7 | 0.5586 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.841 |  | 8 | 0.956 |  | 16 | 0.981 |  | 24 | 0.973 |
| 1 | 0.291 |  | 9 | 0.909 |  | 17 | 0.839 |  | 25 | 0.990 |
| 2 | 0.998 |  | 10 | 0.999 |  | 18 | 0.732 |  | 26 | 0.990 |
| 3 | 0.404 |  | 11 | 0.939 |  | 19 | 0.961 |  | 27 | 0.998 |
| 4 | 0.931 |  | 12 | 0.572 |  | 20 | 0.966 |  | 28 | 0.995 |
| 5 | 0.966 |  | 13 | 0.805 |  | 21 | 0.930 |  | 29 | 1.000 |
| 6 | 0.971 |  | 14 | 0.997 |  | 22 | 0.922 |  | 30 | 0.986 |
| 7 | 0.888 |  | 15 | 0.987 |  | 23 | 0.943 |  | 31 | 0.978 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-I-13.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-I-13.png) · [head_correlation](plots/head_correlation_2024-I-13.png) · [n_drifted_hist](plots/n_drifted_hist_2024-I-13.png)


## Problem `2024-I-14`

- Decode steps logged: **11,964** (of 13,845 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **87** (0.6283% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9981 | 0.9999 | 0.9730 | 0.9327 | 0.9038 |
| 0.90 | 0.8922 | 0.9848 | 0.6374 | 0.5838 | 0.5568 |
| 0.85 | 0.6420 | 0.8918 | 0.2673 | 0.2938 | 0.2932 |
| 0.80 | 0.3720 | 0.7123 | 0.1108 | 0.1410 | 0.1535 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.1078 |
| 1 | 0.1015 |
| 2 | 0.0856 |
| 3 | 0.0919 |
| 4 | 0.0738 |
| 5 | 0.0689 |
| 6 | 0.0932 |
| 7 | 0.0946 |
| 8 | 0.2826 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6083 |
| 1 | 0.5767 |
| 2 | 0.5744 |
| 3 | 0.5549 |
| 4 | 0.5858 |
| 5 | 0.5968 |
| 6 | 0.5994 |
| 7 | 0.5742 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.908 |  | 8 | 0.957 |  | 16 | 0.957 |  | 24 | 0.972 |
| 1 | 0.298 |  | 9 | 0.899 |  | 17 | 0.847 |  | 25 | 0.982 |
| 2 | 0.994 |  | 10 | 0.996 |  | 18 | 0.748 |  | 26 | 0.979 |
| 3 | 0.475 |  | 11 | 0.954 |  | 19 | 0.908 |  | 27 | 0.992 |
| 4 | 0.934 |  | 12 | 0.617 |  | 20 | 0.943 |  | 28 | 0.988 |
| 5 | 0.956 |  | 13 | 0.801 |  | 21 | 0.878 |  | 29 | 0.999 |
| 6 | 0.968 |  | 14 | 0.985 |  | 22 | 0.899 |  | 30 | 0.969 |
| 7 | 0.909 |  | 15 | 0.963 |  | 23 | 0.907 |  | 31 | 0.968 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-I-14.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-I-14.png) · [head_correlation](plots/head_correlation_2024-I-14.png) · [n_drifted_hist](plots/n_drifted_hist_2024-I-14.png)


## Problem `2024-I-15`

- Decode steps logged: **14,449** (of 16,383 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **94** (0.5737% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9989 | 1.0000 | 0.9770 | 0.9386 | 0.9107 |
| 0.90 | 0.9053 | 0.9893 | 0.6716 | 0.6075 | 0.5755 |
| 0.85 | 0.6781 | 0.9109 | 0.2819 | 0.3037 | 0.3002 |
| 0.80 | 0.4057 | 0.7470 | 0.0990 | 0.1372 | 0.1510 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0947 |
| 1 | 0.0924 |
| 2 | 0.0791 |
| 3 | 0.0910 |
| 4 | 0.0713 |
| 5 | 0.0716 |
| 6 | 0.0989 |
| 7 | 0.1082 |
| 8 | 0.2928 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6317 |
| 1 | 0.5954 |
| 2 | 0.6004 |
| 3 | 0.5768 |
| 4 | 0.6092 |
| 5 | 0.6248 |
| 6 | 0.6290 |
| 7 | 0.5926 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.889 |  | 8 | 0.974 |  | 16 | 0.970 |  | 24 | 0.983 |
| 1 | 0.277 |  | 9 | 0.907 |  | 17 | 0.887 |  | 25 | 0.987 |
| 2 | 0.997 |  | 10 | 0.998 |  | 18 | 0.813 |  | 26 | 0.988 |
| 3 | 0.487 |  | 11 | 0.959 |  | 19 | 0.943 |  | 27 | 0.994 |
| 4 | 0.966 |  | 12 | 0.615 |  | 20 | 0.958 |  | 28 | 0.993 |
| 5 | 0.977 |  | 13 | 0.824 |  | 21 | 0.919 |  | 29 | 0.999 |
| 6 | 0.977 |  | 14 | 0.991 |  | 22 | 0.919 |  | 30 | 0.971 |
| 7 | 0.922 |  | 15 | 0.973 |  | 23 | 0.947 |  | 31 | 0.967 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-I-15.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-I-15.png) · [head_correlation](plots/head_correlation_2024-I-15.png) · [n_drifted_hist](plots/n_drifted_hist_2024-I-15.png)


## Problem `2024-I-2`

- Decode steps logged: **10,273** (of 12,254 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **76** (0.6202% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9968 | 0.9999 | 0.9586 | 0.9119 | 0.8834 |
| 0.90 | 0.8640 | 0.9768 | 0.6051 | 0.5567 | 0.5295 |
| 0.85 | 0.6104 | 0.8677 | 0.2420 | 0.2686 | 0.2695 |
| 0.80 | 0.3441 | 0.6876 | 0.0917 | 0.1222 | 0.1363 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.1360 |
| 1 | 0.1053 |
| 2 | 0.0828 |
| 3 | 0.0914 |
| 4 | 0.0690 |
| 5 | 0.0671 |
| 6 | 0.0957 |
| 7 | 0.0992 |
| 8 | 0.2535 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.5834 |
| 1 | 0.5454 |
| 2 | 0.5490 |
| 3 | 0.5252 |
| 4 | 0.5647 |
| 5 | 0.5673 |
| 6 | 0.5748 |
| 7 | 0.5433 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.835 |  | 8 | 0.928 |  | 16 | 0.941 |  | 24 | 0.956 |
| 1 | 0.218 |  | 9 | 0.866 |  | 17 | 0.837 |  | 25 | 0.967 |
| 2 | 0.997 |  | 10 | 0.996 |  | 18 | 0.730 |  | 26 | 0.976 |
| 3 | 0.354 |  | 11 | 0.922 |  | 19 | 0.887 |  | 27 | 0.994 |
| 4 | 0.888 |  | 12 | 0.558 |  | 20 | 0.910 |  | 28 | 0.991 |
| 5 | 0.903 |  | 13 | 0.785 |  | 21 | 0.841 |  | 29 | 0.999 |
| 6 | 0.951 |  | 14 | 0.967 |  | 22 | 0.860 |  | 30 | 0.972 |
| 7 | 0.851 |  | 15 | 0.929 |  | 23 | 0.872 |  | 31 | 0.969 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-I-2.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-I-2.png) · [head_correlation](plots/head_correlation_2024-I-2.png) · [n_drifted_hist](plots/n_drifted_hist_2024-I-2.png)


## Problem `2024-I-3`

- Decode steps logged: **1,528** (of 3,461 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **21** (0.6066% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9986 | 1.0000 | 0.9775 | 0.9399 | 0.9098 |
| 0.90 | 0.9116 | 0.9894 | 0.6723 | 0.6032 | 0.5679 |
| 0.85 | 0.6813 | 0.9183 | 0.2686 | 0.2902 | 0.2877 |
| 0.80 | 0.4085 | 0.7589 | 0.0808 | 0.1240 | 0.1397 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0884 |
| 1 | 0.0963 |
| 2 | 0.0802 |
| 3 | 0.0950 |
| 4 | 0.0729 |
| 5 | 0.0737 |
| 6 | 0.1039 |
| 7 | 0.1159 |
| 8 | 0.2737 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6298 |
| 1 | 0.5871 |
| 2 | 0.5978 |
| 3 | 0.5626 |
| 4 | 0.6097 |
| 5 | 0.6261 |
| 6 | 0.6205 |
| 7 | 0.5921 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.887 |  | 8 | 0.961 |  | 16 | 0.991 |  | 24 | 0.974 |
| 1 | 0.437 |  | 9 | 0.910 |  | 17 | 0.901 |  | 25 | 0.984 |
| 2 | 0.994 |  | 10 | 1.000 |  | 18 | 0.823 |  | 26 | 0.982 |
| 3 | 0.531 |  | 11 | 0.960 |  | 19 | 0.958 |  | 27 | 0.986 |
| 4 | 0.953 |  | 12 | 0.626 |  | 20 | 0.972 |  | 28 | 0.980 |
| 5 | 0.964 |  | 13 | 0.834 |  | 21 | 0.939 |  | 29 | 0.999 |
| 6 | 0.952 |  | 14 | 0.998 |  | 22 | 0.938 |  | 30 | 0.961 |
| 7 | 0.888 |  | 15 | 0.985 |  | 23 | 0.940 |  | 31 | 0.960 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-I-3.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-I-3.png) · [head_correlation](plots/head_correlation_2024-I-3.png) · [n_drifted_hist](plots/n_drifted_hist_2024-I-3.png)


## Problem `2024-I-4`

- Decode steps logged: **14,477** (of 16,383 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **480** (2.9297% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9992 | 1.0000 | 0.9815 | 0.9498 | 0.9216 |
| 0.90 | 0.9232 | 0.9935 | 0.6993 | 0.6413 | 0.6096 |
| 0.85 | 0.7038 | 0.9339 | 0.3211 | 0.3461 | 0.3413 |
| 0.80 | 0.4300 | 0.7719 | 0.1488 | 0.1763 | 0.1857 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0768 |
| 1 | 0.0925 |
| 2 | 0.0731 |
| 3 | 0.0864 |
| 4 | 0.0640 |
| 5 | 0.0671 |
| 6 | 0.0879 |
| 7 | 0.1040 |
| 8 | 0.3482 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6629 |
| 1 | 0.6299 |
| 2 | 0.6298 |
| 3 | 0.6068 |
| 4 | 0.6385 |
| 5 | 0.6721 |
| 6 | 0.6570 |
| 7 | 0.6338 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.904 |  | 8 | 0.970 |  | 16 | 0.989 |  | 24 | 0.994 |
| 1 | 0.354 |  | 9 | 0.926 |  | 17 | 0.912 |  | 25 | 0.999 |
| 2 | 0.991 |  | 10 | 1.000 |  | 18 | 0.833 |  | 26 | 0.997 |
| 3 | 0.461 |  | 11 | 0.969 |  | 19 | 0.979 |  | 27 | 0.999 |
| 4 | 0.953 |  | 12 | 0.734 |  | 20 | 0.985 |  | 28 | 0.996 |
| 5 | 0.964 |  | 13 | 0.845 |  | 21 | 0.976 |  | 29 | 1.000 |
| 6 | 0.971 |  | 14 | 1.000 |  | 22 | 0.983 |  | 30 | 0.990 |
| 7 | 0.910 |  | 15 | 0.997 |  | 23 | 0.975 |  | 31 | 0.988 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-I-4.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-I-4.png) · [head_correlation](plots/head_correlation_2024-I-4.png) · [n_drifted_hist](plots/n_drifted_hist_2024-I-4.png)


## Problem `2024-I-6`

- Decode steps logged: **731** (of 2,562 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **25** (0.9754% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9986 | 1.0000 | 0.9765 | 0.9392 | 0.9076 |
| 0.90 | 0.9118 | 0.9887 | 0.6749 | 0.5992 | 0.5615 |
| 0.85 | 0.6783 | 0.9179 | 0.2449 | 0.2734 | 0.2744 |
| 0.80 | 0.3874 | 0.7568 | 0.0621 | 0.1089 | 0.1279 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0882 |
| 1 | 0.0906 |
| 2 | 0.0807 |
| 3 | 0.0943 |
| 4 | 0.0773 |
| 5 | 0.0824 |
| 6 | 0.1120 |
| 7 | 0.1301 |
| 8 | 0.2444 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6287 |
| 1 | 0.5776 |
| 2 | 0.5984 |
| 3 | 0.5559 |
| 4 | 0.5987 |
| 5 | 0.6173 |
| 6 | 0.6168 |
| 7 | 0.6005 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.811 |  | 8 | 0.962 |  | 16 | 0.990 |  | 24 | 0.986 |
| 1 | 0.443 |  | 9 | 0.895 |  | 17 | 0.895 |  | 25 | 0.993 |
| 2 | 1.000 |  | 10 | 0.997 |  | 18 | 0.858 |  | 26 | 0.988 |
| 3 | 0.539 |  | 11 | 0.958 |  | 19 | 0.960 |  | 27 | 0.988 |
| 4 | 0.949 |  | 12 | 0.590 |  | 20 | 0.970 |  | 28 | 0.993 |
| 5 | 0.964 |  | 13 | 0.834 |  | 21 | 0.956 |  | 29 | 1.000 |
| 6 | 0.948 |  | 14 | 0.995 |  | 22 | 0.960 |  | 30 | 0.975 |
| 7 | 0.886 |  | 15 | 0.981 |  | 23 | 0.959 |  | 31 | 0.953 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-I-6.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-I-6.png) · [head_correlation](plots/head_correlation_2024-I-6.png) · [n_drifted_hist](plots/n_drifted_hist_2024-I-6.png)


## Problem `2024-I-7`

- Decode steps logged: **2,332** (of 4,319 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **10** (0.2315% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9974 | 0.9999 | 0.9664 | 0.9209 | 0.8910 |
| 0.90 | 0.8777 | 0.9807 | 0.6184 | 0.5682 | 0.5428 |
| 0.85 | 0.6321 | 0.8824 | 0.2614 | 0.2917 | 0.2914 |
| 0.80 | 0.3664 | 0.7045 | 0.1186 | 0.1433 | 0.1557 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.1223 |
| 1 | 0.1047 |
| 2 | 0.0873 |
| 3 | 0.0930 |
| 4 | 0.0699 |
| 5 | 0.0668 |
| 6 | 0.0903 |
| 7 | 0.0933 |
| 8 | 0.2723 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.5927 |
| 1 | 0.5545 |
| 2 | 0.5582 |
| 3 | 0.5441 |
| 4 | 0.5726 |
| 5 | 0.5812 |
| 6 | 0.5881 |
| 7 | 0.5544 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.873 |  | 8 | 0.955 |  | 16 | 0.936 |  | 24 | 0.954 |
| 1 | 0.280 |  | 9 | 0.903 |  | 17 | 0.830 |  | 25 | 0.967 |
| 2 | 0.999 |  | 10 | 0.999 |  | 18 | 0.737 |  | 26 | 0.972 |
| 3 | 0.421 |  | 11 | 0.951 |  | 19 | 0.879 |  | 27 | 0.994 |
| 4 | 0.939 |  | 12 | 0.563 |  | 20 | 0.907 |  | 28 | 0.991 |
| 5 | 0.950 |  | 13 | 0.765 |  | 21 | 0.844 |  | 29 | 0.999 |
| 6 | 0.970 |  | 14 | 0.980 |  | 22 | 0.870 |  | 30 | 0.967 |
| 7 | 0.908 |  | 15 | 0.936 |  | 23 | 0.888 |  | 31 | 0.961 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-I-7.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-I-7.png) · [head_correlation](plots/head_correlation_2024-I-7.png) · [n_drifted_hist](plots/n_drifted_hist_2024-I-7.png)


## Problem `2024-I-8`

- Decode steps logged: **2,450** (of 4,389 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **49** (1.1162% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9988 | 1.0000 | 0.9726 | 0.9376 | 0.9084 |
| 0.90 | 0.9108 | 0.9902 | 0.6852 | 0.6135 | 0.5773 |
| 0.85 | 0.6908 | 0.9207 | 0.2790 | 0.3020 | 0.2985 |
| 0.80 | 0.4048 | 0.7629 | 0.0867 | 0.1299 | 0.1464 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0892 |
| 1 | 0.0909 |
| 2 | 0.0752 |
| 3 | 0.0894 |
| 4 | 0.0749 |
| 5 | 0.0737 |
| 6 | 0.1017 |
| 7 | 0.1205 |
| 8 | 0.2845 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6377 |
| 1 | 0.5960 |
| 2 | 0.6094 |
| 3 | 0.5740 |
| 4 | 0.6164 |
| 5 | 0.6299 |
| 6 | 0.6291 |
| 7 | 0.6152 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.853 |  | 8 | 0.970 |  | 16 | 0.984 |  | 24 | 0.976 |
| 1 | 0.364 |  | 9 | 0.918 |  | 17 | 0.918 |  | 25 | 0.988 |
| 2 | 1.000 |  | 10 | 0.999 |  | 18 | 0.856 |  | 26 | 0.983 |
| 3 | 0.450 |  | 11 | 0.965 |  | 19 | 0.972 |  | 27 | 0.988 |
| 4 | 0.934 |  | 12 | 0.615 |  | 20 | 0.963 |  | 28 | 0.991 |
| 5 | 0.963 |  | 13 | 0.876 |  | 21 | 0.958 |  | 29 | 1.000 |
| 6 | 0.953 |  | 14 | 0.995 |  | 22 | 0.948 |  | 30 | 0.965 |
| 7 | 0.887 |  | 15 | 0.982 |  | 23 | 0.973 |  | 31 | 0.959 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-I-8.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-I-8.png) · [head_correlation](plots/head_correlation_2024-I-8.png) · [n_drifted_hist](plots/n_drifted_hist_2024-I-8.png)


## Problem `2024-I-9`

- Decode steps logged: **14,436** (of 16,383 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **1,611** (9.8328% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 1.0000 | 1.0000 | 0.9646 | 0.9188 | 0.8796 |
| 0.90 | 0.8778 | 0.9937 | 0.5336 | 0.4992 | 0.4748 |
| 0.85 | 0.5611 | 0.8913 | 0.1582 | 0.2112 | 0.2224 |
| 0.80 | 0.2806 | 0.6731 | 0.0719 | 0.0970 | 0.1125 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.1222 |
| 1 | 0.1300 |
| 2 | 0.1282 |
| 3 | 0.1163 |
| 4 | 0.0725 |
| 5 | 0.0725 |
| 6 | 0.0897 |
| 7 | 0.0810 |
| 8 | 0.1876 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.5443 |
| 1 | 0.4767 |
| 2 | 0.4815 |
| 3 | 0.4849 |
| 4 | 0.4740 |
| 5 | 0.5495 |
| 6 | 0.4980 |
| 7 | 0.4847 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.700 |  | 8 | 1.000 |  | 16 | 1.000 |  | 24 | 0.987 |
| 1 | 0.443 |  | 9 | 0.905 |  | 17 | 0.908 |  | 25 | 0.999 |
| 2 | 1.000 |  | 10 | 1.000 |  | 18 | 0.624 |  | 26 | 0.994 |
| 3 | 0.299 |  | 11 | 1.000 |  | 19 | 0.995 |  | 27 | 0.950 |
| 4 | 0.658 |  | 12 | 0.516 |  | 20 | 1.000 |  | 28 | 0.971 |
| 5 | 0.940 |  | 13 | 0.964 |  | 21 | 0.987 |  | 29 | 1.000 |
| 6 | 0.850 |  | 14 | 1.000 |  | 22 | 0.967 |  | 30 | 0.988 |
| 7 | 0.713 |  | 15 | 1.000 |  | 23 | 0.882 |  | 31 | 0.850 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-I-9.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-I-9.png) · [head_correlation](plots/head_correlation_2024-I-9.png) · [n_drifted_hist](plots/n_drifted_hist_2024-I-9.png)


## Problem `2024-II-1`

- Decode steps logged: **4,623** (of 6,554 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **33** (0.5034% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9993 | 1.0000 | 0.9773 | 0.9434 | 0.9160 |
| 0.90 | 0.9153 | 0.9923 | 0.6839 | 0.6243 | 0.5887 |
| 0.85 | 0.6806 | 0.9246 | 0.2913 | 0.3143 | 0.3125 |
| 0.80 | 0.4038 | 0.7560 | 0.1177 | 0.1489 | 0.1622 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0847 |
| 1 | 0.0916 |
| 2 | 0.0748 |
| 3 | 0.0897 |
| 4 | 0.0677 |
| 5 | 0.0708 |
| 6 | 0.0963 |
| 7 | 0.1134 |
| 8 | 0.3109 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6540 |
| 1 | 0.6041 |
| 2 | 0.6078 |
| 3 | 0.5985 |
| 4 | 0.6185 |
| 5 | 0.6497 |
| 6 | 0.6509 |
| 7 | 0.6109 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.883 |  | 8 | 0.977 |  | 16 | 0.984 |  | 24 | 0.989 |
| 1 | 0.295 |  | 9 | 0.939 |  | 17 | 0.896 |  | 25 | 0.995 |
| 2 | 0.998 |  | 10 | 0.999 |  | 18 | 0.849 |  | 26 | 0.994 |
| 3 | 0.510 |  | 11 | 0.975 |  | 19 | 0.948 |  | 27 | 0.995 |
| 4 | 0.970 |  | 12 | 0.685 |  | 20 | 0.967 |  | 28 | 0.994 |
| 5 | 0.962 |  | 13 | 0.897 |  | 21 | 0.935 |  | 29 | 0.999 |
| 6 | 0.968 |  | 14 | 0.995 |  | 22 | 0.937 |  | 30 | 0.971 |
| 7 | 0.923 |  | 15 | 0.976 |  | 23 | 0.930 |  | 31 | 0.950 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-II-1.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-II-1.png) · [head_correlation](plots/head_correlation_2024-II-1.png) · [n_drifted_hist](plots/n_drifted_hist_2024-II-1.png)


## Problem `2024-II-10`

- Decode steps logged: **14,589** (of 16,383 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **203** (1.2390% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9991 | 1.0000 | 0.9824 | 0.9494 | 0.9205 |
| 0.90 | 0.9189 | 0.9919 | 0.6879 | 0.6196 | 0.5855 |
| 0.85 | 0.6841 | 0.9226 | 0.2795 | 0.3058 | 0.3044 |
| 0.80 | 0.3995 | 0.7562 | 0.1080 | 0.1424 | 0.1567 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0811 |
| 1 | 0.0908 |
| 2 | 0.0796 |
| 3 | 0.0906 |
| 4 | 0.0737 |
| 5 | 0.0746 |
| 6 | 0.0988 |
| 7 | 0.1123 |
| 8 | 0.2986 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6478 |
| 1 | 0.6099 |
| 2 | 0.6061 |
| 3 | 0.5909 |
| 4 | 0.6155 |
| 5 | 0.6438 |
| 6 | 0.6367 |
| 7 | 0.6065 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.847 |  | 8 | 0.982 |  | 16 | 0.993 |  | 24 | 0.993 |
| 1 | 0.389 |  | 9 | 0.940 |  | 17 | 0.862 |  | 25 | 0.995 |
| 2 | 1.000 |  | 10 | 0.999 |  | 18 | 0.822 |  | 26 | 0.994 |
| 3 | 0.575 |  | 11 | 0.973 |  | 19 | 0.957 |  | 27 | 0.995 |
| 4 | 0.975 |  | 12 | 0.674 |  | 20 | 0.979 |  | 28 | 0.996 |
| 5 | 0.985 |  | 13 | 0.790 |  | 21 | 0.942 |  | 29 | 1.000 |
| 6 | 0.981 |  | 14 | 0.995 |  | 22 | 0.950 |  | 30 | 0.979 |
| 7 | 0.946 |  | 15 | 0.977 |  | 23 | 0.950 |  | 31 | 0.973 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-II-10.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-II-10.png) · [head_correlation](plots/head_correlation_2024-II-10.png) · [n_drifted_hist](plots/n_drifted_hist_2024-II-10.png)


## Problem `2024-II-11`

- Decode steps logged: **14,420** (of 16,383 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **209** (1.2756% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9993 | 1.0000 | 0.9767 | 0.9385 | 0.9097 |
| 0.90 | 0.9127 | 0.9917 | 0.6624 | 0.6070 | 0.5782 |
| 0.85 | 0.6631 | 0.9155 | 0.2951 | 0.3260 | 0.3259 |
| 0.80 | 0.3977 | 0.7380 | 0.1648 | 0.1836 | 0.1909 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0873 |
| 1 | 0.1013 |
| 2 | 0.0797 |
| 3 | 0.0965 |
| 4 | 0.0747 |
| 5 | 0.0672 |
| 6 | 0.0878 |
| 7 | 0.0991 |
| 8 | 0.3063 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6276 |
| 1 | 0.5920 |
| 2 | 0.5920 |
| 3 | 0.5773 |
| 4 | 0.6145 |
| 5 | 0.6277 |
| 6 | 0.6220 |
| 7 | 0.6032 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.852 |  | 8 | 0.980 |  | 16 | 0.986 |  | 24 | 0.989 |
| 1 | 0.308 |  | 9 | 0.920 |  | 17 | 0.890 |  | 25 | 0.996 |
| 2 | 0.999 |  | 10 | 0.999 |  | 18 | 0.818 |  | 26 | 0.991 |
| 3 | 0.499 |  | 11 | 0.949 |  | 19 | 0.988 |  | 27 | 0.997 |
| 4 | 0.935 |  | 12 | 0.677 |  | 20 | 0.987 |  | 28 | 0.987 |
| 5 | 0.959 |  | 13 | 0.855 |  | 21 | 0.973 |  | 29 | 1.000 |
| 6 | 0.981 |  | 14 | 0.998 |  | 22 | 0.943 |  | 30 | 0.978 |
| 7 | 0.876 |  | 15 | 0.991 |  | 23 | 0.957 |  | 31 | 0.946 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-II-11.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-II-11.png) · [head_correlation](plots/head_correlation_2024-II-11.png) · [n_drifted_hist](plots/n_drifted_hist_2024-II-11.png)


## Problem `2024-II-12`

- Decode steps logged: **2,662** (of 4,513 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **10** (0.2215% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9965 | 0.9999 | 0.9599 | 0.9142 | 0.8848 |
| 0.90 | 0.8715 | 0.9792 | 0.6013 | 0.5544 | 0.5285 |
| 0.85 | 0.6117 | 0.8717 | 0.2389 | 0.2655 | 0.2676 |
| 0.80 | 0.3493 | 0.6885 | 0.0910 | 0.1218 | 0.1351 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.1285 |
| 1 | 0.1089 |
| 2 | 0.0907 |
| 3 | 0.0918 |
| 4 | 0.0684 |
| 5 | 0.0685 |
| 6 | 0.0965 |
| 7 | 0.0997 |
| 8 | 0.2470 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.5779 |
| 1 | 0.5408 |
| 2 | 0.5550 |
| 3 | 0.5207 |
| 4 | 0.5628 |
| 5 | 0.5598 |
| 6 | 0.5716 |
| 7 | 0.5464 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.852 |  | 8 | 0.955 |  | 16 | 0.950 |  | 24 | 0.964 |
| 1 | 0.238 |  | 9 | 0.885 |  | 17 | 0.794 |  | 25 | 0.971 |
| 2 | 0.998 |  | 10 | 0.995 |  | 18 | 0.751 |  | 26 | 0.974 |
| 3 | 0.415 |  | 11 | 0.950 |  | 19 | 0.887 |  | 27 | 0.992 |
| 4 | 0.901 |  | 12 | 0.563 |  | 20 | 0.905 |  | 28 | 0.993 |
| 5 | 0.941 |  | 13 | 0.784 |  | 21 | 0.853 |  | 29 | 0.999 |
| 6 | 0.958 |  | 14 | 0.979 |  | 22 | 0.858 |  | 30 | 0.948 |
| 7 | 0.884 |  | 15 | 0.944 |  | 23 | 0.858 |  | 31 | 0.952 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-II-12.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-II-12.png) · [head_correlation](plots/head_correlation_2024-II-12.png) · [n_drifted_hist](plots/n_drifted_hist_2024-II-12.png)


## Problem `2024-II-14`

- Decode steps logged: **14,472** (of 16,383 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **37** (0.2258% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9987 | 1.0000 | 0.9803 | 0.9559 | 0.9311 |
| 0.90 | 0.9238 | 0.9921 | 0.7342 | 0.6453 | 0.6073 |
| 0.85 | 0.7210 | 0.9263 | 0.2975 | 0.3143 | 0.3092 |
| 0.80 | 0.4346 | 0.7701 | 0.0794 | 0.1312 | 0.1486 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0762 |
| 1 | 0.0676 |
| 2 | 0.0677 |
| 3 | 0.0902 |
| 4 | 0.0756 |
| 5 | 0.0797 |
| 6 | 0.1156 |
| 7 | 0.1250 |
| 8 | 0.3024 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6798 |
| 1 | 0.6363 |
| 2 | 0.6212 |
| 3 | 0.6192 |
| 4 | 0.6464 |
| 5 | 0.6762 |
| 6 | 0.6560 |
| 7 | 0.6270 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.895 |  | 8 | 0.976 |  | 16 | 0.970 |  | 24 | 0.986 |
| 1 | 0.249 |  | 9 | 0.965 |  | 17 | 0.922 |  | 25 | 0.983 |
| 2 | 0.990 |  | 10 | 1.000 |  | 18 | 0.852 |  | 26 | 0.985 |
| 3 | 0.562 |  | 11 | 0.993 |  | 19 | 0.970 |  | 27 | 0.992 |
| 4 | 0.963 |  | 12 | 0.715 |  | 20 | 0.965 |  | 28 | 0.985 |
| 5 | 0.987 |  | 13 | 0.939 |  | 21 | 0.938 |  | 29 | 0.997 |
| 6 | 0.969 |  | 14 | 0.997 |  | 22 | 0.948 |  | 30 | 0.983 |
| 7 | 0.965 |  | 15 | 0.980 |  | 23 | 0.964 |  | 31 | 0.977 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-II-14.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-II-14.png) · [head_correlation](plots/head_correlation_2024-II-14.png) · [n_drifted_hist](plots/n_drifted_hist_2024-II-14.png)


## Problem `2024-II-15`

- Decode steps logged: **14,406** (of 16,383 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **284** (1.7334% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9995 | 1.0000 | 0.9773 | 0.9398 | 0.9051 |
| 0.90 | 0.9036 | 0.9901 | 0.6401 | 0.5723 | 0.5413 |
| 0.85 | 0.6233 | 0.9175 | 0.2101 | 0.2512 | 0.2597 |
| 0.80 | 0.3247 | 0.7133 | 0.0786 | 0.1109 | 0.1286 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0964 |
| 1 | 0.1071 |
| 2 | 0.0875 |
| 3 | 0.0997 |
| 4 | 0.0803 |
| 5 | 0.0812 |
| 6 | 0.0973 |
| 7 | 0.1185 |
| 8 | 0.2320 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6075 |
| 1 | 0.5456 |
| 2 | 0.5490 |
| 3 | 0.5376 |
| 4 | 0.5621 |
| 5 | 0.6102 |
| 6 | 0.5832 |
| 7 | 0.5829 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.749 |  | 8 | 0.972 |  | 16 | 0.993 |  | 24 | 0.988 |
| 1 | 0.402 |  | 9 | 0.936 |  | 17 | 0.809 |  | 25 | 0.995 |
| 2 | 0.999 |  | 10 | 0.999 |  | 18 | 0.781 |  | 26 | 0.992 |
| 3 | 0.467 |  | 11 | 0.970 |  | 19 | 0.984 |  | 27 | 0.987 |
| 4 | 0.898 |  | 12 | 0.657 |  | 20 | 0.991 |  | 28 | 0.987 |
| 5 | 0.931 |  | 13 | 0.850 |  | 21 | 0.980 |  | 29 | 1.000 |
| 6 | 0.915 |  | 14 | 0.999 |  | 22 | 0.969 |  | 30 | 0.978 |
| 7 | 0.828 |  | 15 | 0.987 |  | 23 | 0.970 |  | 31 | 0.955 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-II-15.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-II-15.png) · [head_correlation](plots/head_correlation_2024-II-15.png) · [n_drifted_hist](plots/n_drifted_hist_2024-II-15.png)


## Problem `2024-II-2`

- Decode steps logged: **14,424** (of 16,383 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **309** (1.8860% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9985 | 1.0000 | 0.9743 | 0.9361 | 0.9048 |
| 0.90 | 0.9051 | 0.9885 | 0.6479 | 0.5911 | 0.5600 |
| 0.85 | 0.6493 | 0.9091 | 0.2690 | 0.2975 | 0.2984 |
| 0.80 | 0.3701 | 0.7255 | 0.1225 | 0.1491 | 0.1617 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0949 |
| 1 | 0.1032 |
| 2 | 0.0836 |
| 3 | 0.0944 |
| 4 | 0.0783 |
| 5 | 0.0712 |
| 6 | 0.0922 |
| 7 | 0.1050 |
| 8 | 0.2771 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6158 |
| 1 | 0.5739 |
| 2 | 0.5790 |
| 3 | 0.5495 |
| 4 | 0.5998 |
| 5 | 0.6152 |
| 6 | 0.6006 |
| 7 | 0.5946 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.866 |  | 8 | 0.978 |  | 16 | 0.971 |  | 24 | 0.981 |
| 1 | 0.284 |  | 9 | 0.945 |  | 17 | 0.862 |  | 25 | 0.990 |
| 2 | 0.998 |  | 10 | 0.999 |  | 18 | 0.765 |  | 26 | 0.983 |
| 3 | 0.484 |  | 11 | 0.962 |  | 19 | 0.946 |  | 27 | 0.991 |
| 4 | 0.933 |  | 12 | 0.709 |  | 20 | 0.960 |  | 28 | 0.986 |
| 5 | 0.933 |  | 13 | 0.874 |  | 21 | 0.934 |  | 29 | 1.000 |
| 6 | 0.944 |  | 14 | 0.997 |  | 22 | 0.934 |  | 30 | 0.970 |
| 7 | 0.901 |  | 15 | 0.975 |  | 23 | 0.939 |  | 31 | 0.967 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-II-2.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-II-2.png) · [head_correlation](plots/head_correlation_2024-II-2.png) · [n_drifted_hist](plots/n_drifted_hist_2024-II-2.png)


## Problem `2024-II-3`

- Decode steps logged: **14,484** (of 16,383 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **90** (0.5493% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9991 | 1.0000 | 0.9832 | 0.9475 | 0.9168 |
| 0.90 | 0.9180 | 0.9930 | 0.6794 | 0.6063 | 0.5682 |
| 0.85 | 0.6847 | 0.9312 | 0.2537 | 0.2821 | 0.2824 |
| 0.80 | 0.3894 | 0.7640 | 0.0759 | 0.1186 | 0.1363 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0820 |
| 1 | 0.0920 |
| 2 | 0.0829 |
| 3 | 0.0971 |
| 4 | 0.0747 |
| 5 | 0.0774 |
| 6 | 0.1056 |
| 7 | 0.1250 |
| 8 | 0.2634 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6344 |
| 1 | 0.5942 |
| 2 | 0.5936 |
| 3 | 0.5673 |
| 4 | 0.6073 |
| 5 | 0.6340 |
| 6 | 0.6187 |
| 7 | 0.6009 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.855 |  | 8 | 0.967 |  | 16 | 0.995 |  | 24 | 0.992 |
| 1 | 0.301 |  | 9 | 0.923 |  | 17 | 0.900 |  | 25 | 0.998 |
| 2 | 0.999 |  | 10 | 1.000 |  | 18 | 0.815 |  | 26 | 0.995 |
| 3 | 0.539 |  | 11 | 0.965 |  | 19 | 0.978 |  | 27 | 0.997 |
| 4 | 0.979 |  | 12 | 0.683 |  | 20 | 0.986 |  | 28 | 0.992 |
| 5 | 0.961 |  | 13 | 0.855 |  | 21 | 0.965 |  | 29 | 1.000 |
| 6 | 0.967 |  | 14 | 0.998 |  | 22 | 0.935 |  | 30 | 0.978 |
| 7 | 0.925 |  | 15 | 0.992 |  | 23 | 0.970 |  | 31 | 0.968 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-II-3.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-II-3.png) · [head_correlation](plots/head_correlation_2024-II-3.png) · [n_drifted_hist](plots/n_drifted_hist_2024-II-3.png)


## Problem `2024-II-4`

- Decode steps logged: **1,382** (of 3,266 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **5** (0.1530% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9960 | 0.9998 | 0.9535 | 0.9081 | 0.8806 |
| 0.90 | 0.8616 | 0.9744 | 0.5943 | 0.5525 | 0.5279 |
| 0.85 | 0.6054 | 0.8538 | 0.2474 | 0.2721 | 0.2721 |
| 0.80 | 0.3428 | 0.6729 | 0.1023 | 0.1267 | 0.1397 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.1384 |
| 1 | 0.1074 |
| 2 | 0.0823 |
| 3 | 0.0954 |
| 4 | 0.0709 |
| 5 | 0.0661 |
| 6 | 0.0889 |
| 7 | 0.0903 |
| 8 | 0.2603 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.5738 |
| 1 | 0.5377 |
| 2 | 0.5470 |
| 3 | 0.5234 |
| 4 | 0.5496 |
| 5 | 0.5716 |
| 6 | 0.5767 |
| 7 | 0.5404 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.872 |  | 8 | 0.946 |  | 16 | 0.945 |  | 24 | 0.959 |
| 1 | 0.173 |  | 9 | 0.894 |  | 17 | 0.763 |  | 25 | 0.970 |
| 2 | 0.978 |  | 10 | 0.997 |  | 18 | 0.745 |  | 26 | 0.973 |
| 3 | 0.397 |  | 11 | 0.950 |  | 19 | 0.833 |  | 27 | 0.988 |
| 4 | 0.912 |  | 12 | 0.544 |  | 20 | 0.878 |  | 28 | 0.984 |
| 5 | 0.928 |  | 13 | 0.766 |  | 21 | 0.819 |  | 29 | 0.992 |
| 6 | 0.955 |  | 14 | 0.977 |  | 22 | 0.860 |  | 30 | 0.960 |
| 7 | 0.892 |  | 15 | 0.924 |  | 23 | 0.844 |  | 31 | 0.954 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-II-4.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-II-4.png) · [head_correlation](plots/head_correlation_2024-II-4.png) · [n_drifted_hist](plots/n_drifted_hist_2024-II-4.png)


## Problem `2024-II-5`

- Decode steps logged: **14,410** (of 16,383 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **496** (3.0273% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9980 | 0.9999 | 0.9701 | 0.9330 | 0.9023 |
| 0.90 | 0.8966 | 0.9879 | 0.6434 | 0.5835 | 0.5517 |
| 0.85 | 0.6290 | 0.9038 | 0.2394 | 0.2713 | 0.2759 |
| 0.80 | 0.3401 | 0.7113 | 0.0931 | 0.1247 | 0.1402 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.1034 |
| 1 | 0.1068 |
| 2 | 0.0814 |
| 3 | 0.0909 |
| 4 | 0.0731 |
| 5 | 0.0745 |
| 6 | 0.0938 |
| 7 | 0.1106 |
| 8 | 0.2654 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6125 |
| 1 | 0.5626 |
| 2 | 0.5699 |
| 3 | 0.5513 |
| 4 | 0.5746 |
| 5 | 0.6184 |
| 6 | 0.5911 |
| 7 | 0.5874 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.811 |  | 8 | 0.947 |  | 16 | 0.989 |  | 24 | 0.989 |
| 1 | 0.348 |  | 9 | 0.872 |  | 17 | 0.828 |  | 25 | 0.996 |
| 2 | 0.993 |  | 10 | 0.996 |  | 18 | 0.791 |  | 26 | 0.989 |
| 3 | 0.400 |  | 11 | 0.926 |  | 19 | 0.974 |  | 27 | 0.995 |
| 4 | 0.904 |  | 12 | 0.642 |  | 20 | 0.990 |  | 28 | 0.990 |
| 5 | 0.924 |  | 13 | 0.812 |  | 21 | 0.979 |  | 29 | 1.000 |
| 6 | 0.922 |  | 14 | 0.990 |  | 22 | 0.965 |  | 30 | 0.976 |
| 7 | 0.842 |  | 15 | 0.982 |  | 23 | 0.966 |  | 31 | 0.963 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-II-5.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-II-5.png) · [head_correlation](plots/head_correlation_2024-II-5.png) · [n_drifted_hist](plots/n_drifted_hist_2024-II-5.png)


## Problem `2024-II-6`

- Decode steps logged: **14,413** (of 16,383 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **188** (1.1475% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9978 | 0.9999 | 0.9727 | 0.9358 | 0.9066 |
| 0.90 | 0.9006 | 0.9895 | 0.6587 | 0.5968 | 0.5663 |
| 0.85 | 0.6585 | 0.9086 | 0.2592 | 0.2961 | 0.2988 |
| 0.80 | 0.3686 | 0.7249 | 0.1213 | 0.1480 | 0.1609 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0994 |
| 1 | 0.0995 |
| 2 | 0.0798 |
| 3 | 0.0885 |
| 4 | 0.0760 |
| 5 | 0.0705 |
| 6 | 0.0969 |
| 7 | 0.1036 |
| 8 | 0.2858 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6233 |
| 1 | 0.5798 |
| 2 | 0.5816 |
| 3 | 0.5629 |
| 4 | 0.5979 |
| 5 | 0.6245 |
| 6 | 0.6084 |
| 7 | 0.5959 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.837 |  | 8 | 0.968 |  | 16 | 0.981 |  | 24 | 0.989 |
| 1 | 0.254 |  | 9 | 0.913 |  | 17 | 0.866 |  | 25 | 0.991 |
| 2 | 0.969 |  | 10 | 0.996 |  | 18 | 0.796 |  | 26 | 0.991 |
| 3 | 0.378 |  | 11 | 0.943 |  | 19 | 0.965 |  | 27 | 0.993 |
| 4 | 0.940 |  | 12 | 0.678 |  | 20 | 0.973 |  | 28 | 0.997 |
| 5 | 0.948 |  | 13 | 0.804 |  | 21 | 0.947 |  | 29 | 1.000 |
| 6 | 0.950 |  | 14 | 0.997 |  | 22 | 0.956 |  | 30 | 0.975 |
| 7 | 0.900 |  | 15 | 0.980 |  | 23 | 0.974 |  | 31 | 0.971 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-II-6.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-II-6.png) · [head_correlation](plots/head_correlation_2024-II-6.png) · [n_drifted_hist](plots/n_drifted_hist_2024-II-6.png)


## Problem `2024-II-7`

- Decode steps logged: **14,422** (of 16,383 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **111** (0.6775% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9991 | 1.0000 | 0.9849 | 0.9530 | 0.9237 |
| 0.90 | 0.9269 | 0.9945 | 0.7015 | 0.6239 | 0.5840 |
| 0.85 | 0.6989 | 0.9354 | 0.2640 | 0.2870 | 0.2868 |
| 0.80 | 0.4071 | 0.7776 | 0.0661 | 0.1155 | 0.1343 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0731 |
| 1 | 0.0870 |
| 2 | 0.0758 |
| 3 | 0.0943 |
| 4 | 0.0765 |
| 5 | 0.0775 |
| 6 | 0.1104 |
| 7 | 0.1292 |
| 8 | 0.2761 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6548 |
| 1 | 0.6135 |
| 2 | 0.6128 |
| 3 | 0.5841 |
| 4 | 0.6234 |
| 5 | 0.6549 |
| 6 | 0.6387 |
| 7 | 0.6089 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.852 |  | 8 | 0.979 |  | 16 | 0.993 |  | 24 | 0.992 |
| 1 | 0.452 |  | 9 | 0.945 |  | 17 | 0.925 |  | 25 | 0.996 |
| 2 | 0.991 |  | 10 | 0.999 |  | 18 | 0.831 |  | 26 | 0.992 |
| 3 | 0.588 |  | 11 | 0.975 |  | 19 | 0.973 |  | 27 | 0.991 |
| 4 | 0.970 |  | 12 | 0.673 |  | 20 | 0.991 |  | 28 | 0.990 |
| 5 | 0.983 |  | 13 | 0.867 |  | 21 | 0.957 |  | 29 | 1.000 |
| 6 | 0.971 |  | 14 | 0.996 |  | 22 | 0.952 |  | 30 | 0.973 |
| 7 | 0.927 |  | 15 | 0.992 |  | 23 | 0.974 |  | 31 | 0.972 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-II-7.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-II-7.png) · [head_correlation](plots/head_correlation_2024-II-7.png) · [n_drifted_hist](plots/n_drifted_hist_2024-II-7.png)


## Problem `2024-II-8`

- Decode steps logged: **14,513** (of 16,383 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **130** (0.7935% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9990 | 1.0000 | 0.9764 | 0.9407 | 0.9120 |
| 0.90 | 0.9113 | 0.9914 | 0.6714 | 0.6063 | 0.5720 |
| 0.85 | 0.6760 | 0.9170 | 0.2624 | 0.2911 | 0.2910 |
| 0.80 | 0.3832 | 0.7482 | 0.0904 | 0.1278 | 0.1440 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0887 |
| 1 | 0.0926 |
| 2 | 0.0821 |
| 3 | 0.0936 |
| 4 | 0.0714 |
| 5 | 0.0729 |
| 6 | 0.1046 |
| 7 | 0.1183 |
| 8 | 0.2760 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6392 |
| 1 | 0.5908 |
| 2 | 0.6011 |
| 3 | 0.5706 |
| 4 | 0.6021 |
| 5 | 0.6225 |
| 6 | 0.6279 |
| 7 | 0.5961 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.859 |  | 8 | 0.973 |  | 16 | 0.984 |  | 24 | 0.979 |
| 1 | 0.324 |  | 9 | 0.919 |  | 17 | 0.889 |  | 25 | 0.992 |
| 2 | 0.999 |  | 10 | 0.999 |  | 18 | 0.827 |  | 26 | 0.990 |
| 3 | 0.507 |  | 11 | 0.961 |  | 19 | 0.962 |  | 27 | 0.991 |
| 4 | 0.945 |  | 12 | 0.634 |  | 20 | 0.972 |  | 28 | 0.990 |
| 5 | 0.969 |  | 13 | 0.847 |  | 21 | 0.942 |  | 29 | 0.999 |
| 6 | 0.966 |  | 14 | 0.994 |  | 22 | 0.947 |  | 30 | 0.975 |
| 7 | 0.916 |  | 15 | 0.984 |  | 23 | 0.963 |  | 31 | 0.961 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-II-8.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-II-8.png) · [head_correlation](plots/head_correlation_2024-II-8.png) · [n_drifted_hist](plots/n_drifted_hist_2024-II-8.png)


## Problem `2024-II-9`

- Decode steps logged: **14,451** (of 16,383 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **138** (0.8423% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9991 | 1.0000 | 0.9784 | 0.9440 | 0.9100 |
| 0.90 | 0.9058 | 0.9939 | 0.6332 | 0.5646 | 0.5324 |
| 0.85 | 0.6193 | 0.9168 | 0.1949 | 0.2354 | 0.2449 |
| 0.80 | 0.3091 | 0.7036 | 0.0594 | 0.0942 | 0.1130 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0942 |
| 1 | 0.1068 |
| 2 | 0.0912 |
| 3 | 0.1000 |
| 4 | 0.0874 |
| 5 | 0.0870 |
| 6 | 0.1028 |
| 7 | 0.1191 |
| 8 | 0.2116 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6016 |
| 1 | 0.5365 |
| 2 | 0.5492 |
| 3 | 0.5202 |
| 4 | 0.5567 |
| 5 | 0.5995 |
| 6 | 0.5730 |
| 7 | 0.5802 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.668 |  | 8 | 0.966 |  | 16 | 0.993 |  | 24 | 0.982 |
| 1 | 0.423 |  | 9 | 0.926 |  | 17 | 0.828 |  | 25 | 0.997 |
| 2 | 1.000 |  | 10 | 1.000 |  | 18 | 0.721 |  | 26 | 0.984 |
| 3 | 0.542 |  | 11 | 0.980 |  | 19 | 0.989 |  | 27 | 0.972 |
| 4 | 0.961 |  | 12 | 0.695 |  | 20 | 0.987 |  | 28 | 0.978 |
| 5 | 0.966 |  | 13 | 0.820 |  | 21 | 0.972 |  | 29 | 1.000 |
| 6 | 0.957 |  | 14 | 0.996 |  | 22 | 0.953 |  | 30 | 0.952 |
| 7 | 0.903 |  | 15 | 0.989 |  | 23 | 0.963 |  | 31 | 0.920 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-II-9.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-II-9.png) · [head_correlation](plots/head_correlation_2024-II-9.png) · [n_drifted_hist](plots/n_drifted_hist_2024-II-9.png)
