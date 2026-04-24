# `dips_v2` — per-head cosine-similarity analysis

Uses the full `[n_steps, n_layers=32, n_q_heads=32]` cosine-sim tensor cached per problem. This lets us compute any aggregation over heads without re-running.

Problems included: `2024-I-1`, `2024-I-2`, `2024-I-3`, `2024-I-4`, `2024-II-4`


## Global rates at τ=0.9 (for quick comparison)

| pid | n_valid steps | need_corr | per-q-head | per-kv-head | mean<0.9 |
|---|---|---|---|---|---|
| `2024-I-1` | 1,548 | 0.8911 | 0.5551 | 0.5865 | 0.6466 |
| `2024-I-2` | 15,014 | 0.8703 | 0.5320 | 0.5598 | 0.6098 |
| `2024-I-3` | 1,475 | 0.9185 | 0.5772 | 0.6136 | 0.6869 |
| `2024-I-4` | 1,253 | 0.8913 | 0.5564 | 0.5872 | 0.6470 |
| `2024-II-4` | 1,407 | 0.8567 | 0.5242 | 0.5481 | 0.5887 |


## Problem `2024-I-1`

- Decode steps logged: **1,548** (of 3,444 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **12** (0.3483% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9973 | 0.9999 | 0.9667 | 0.9252 | 0.8961 |
| 0.90 | 0.8911 | 0.9827 | 0.6466 | 0.5865 | 0.5551 |
| 0.85 | 0.6507 | 0.8931 | 0.2548 | 0.2858 | 0.2864 |
| 0.80 | 0.3670 | 0.7197 | 0.0969 | 0.1276 | 0.1432 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.1089 |
| 1 | 0.0968 |
| 2 | 0.0781 |
| 3 | 0.0942 |
| 4 | 0.0713 |
| 5 | 0.0729 |
| 6 | 0.1029 |
| 7 | 0.1100 |
| 8 | 0.2649 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6107 |
| 1 | 0.5657 |
| 2 | 0.5793 |
| 3 | 0.5489 |
| 4 | 0.5909 |
| 5 | 0.6098 |
| 6 | 0.6118 |
| 7 | 0.5747 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.839 |  | 8 | 0.966 |  | 16 | 0.968 |  | 24 | 0.966 |
| 1 | 0.318 |  | 9 | 0.920 |  | 17 | 0.869 |  | 25 | 0.976 |
| 2 | 0.992 |  | 10 | 0.997 |  | 18 | 0.800 |  | 26 | 0.970 |
| 3 | 0.435 |  | 11 | 0.959 |  | 19 | 0.900 |  | 27 | 0.995 |
| 4 | 0.954 |  | 12 | 0.589 |  | 20 | 0.920 |  | 28 | 0.992 |
| 5 | 0.953 |  | 13 | 0.808 |  | 21 | 0.893 |  | 29 | 0.995 |
| 6 | 0.974 |  | 14 | 0.988 |  | 22 | 0.902 |  | 30 | 0.964 |
| 7 | 0.911 |  | 15 | 0.946 |  | 23 | 0.913 |  | 31 | 0.944 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-I-1.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-I-1.png) · [head_correlation](plots/head_correlation_2024-I-1.png) · [n_drifted_hist](plots/n_drifted_hist_2024-I-1.png)


## Problem `2024-I-2`

- Decode steps logged: **15,014** (of 16,995 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **111** (0.6531% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9972 | 1.0000 | 0.9622 | 0.9161 | 0.8875 |
| 0.90 | 0.8703 | 0.9795 | 0.6098 | 0.5598 | 0.5320 |
| 0.85 | 0.6153 | 0.8735 | 0.2427 | 0.2696 | 0.2701 |
| 0.80 | 0.3476 | 0.6905 | 0.0904 | 0.1211 | 0.1355 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.1297 |
| 1 | 0.1044 |
| 2 | 0.0844 |
| 3 | 0.0942 |
| 4 | 0.0700 |
| 5 | 0.0683 |
| 6 | 0.0950 |
| 7 | 0.1008 |
| 8 | 0.2533 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.5879 |
| 1 | 0.5500 |
| 2 | 0.5509 |
| 3 | 0.5316 |
| 4 | 0.5652 |
| 5 | 0.5717 |
| 6 | 0.5765 |
| 7 | 0.5449 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.853 |  | 8 | 0.932 |  | 16 | 0.950 |  | 24 | 0.964 |
| 1 | 0.231 |  | 9 | 0.866 |  | 17 | 0.854 |  | 25 | 0.971 |
| 2 | 0.998 |  | 10 | 0.997 |  | 18 | 0.741 |  | 26 | 0.973 |
| 3 | 0.367 |  | 11 | 0.921 |  | 19 | 0.900 |  | 27 | 0.995 |
| 4 | 0.880 |  | 12 | 0.555 |  | 20 | 0.918 |  | 28 | 0.991 |
| 5 | 0.909 |  | 13 | 0.801 |  | 21 | 0.853 |  | 29 | 0.999 |
| 6 | 0.946 |  | 14 | 0.976 |  | 22 | 0.868 |  | 30 | 0.974 |
| 7 | 0.858 |  | 15 | 0.943 |  | 23 | 0.892 |  | 31 | 0.975 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-I-2.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-I-2.png) · [head_correlation](plots/head_correlation_2024-I-2.png) · [n_drifted_hist](plots/n_drifted_hist_2024-I-2.png)


## Problem `2024-I-3`

- Decode steps logged: **1,475** (of 3,408 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **20** (0.5867% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9993 | 1.0000 | 0.9795 | 0.9442 | 0.9146 |
| 0.90 | 0.9185 | 0.9907 | 0.6869 | 0.6136 | 0.5772 |
| 0.85 | 0.6904 | 0.9230 | 0.2683 | 0.2936 | 0.2920 |
| 0.80 | 0.4101 | 0.7651 | 0.0825 | 0.1259 | 0.1412 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0815 |
| 1 | 0.0914 |
| 2 | 0.0794 |
| 3 | 0.0930 |
| 4 | 0.0739 |
| 5 | 0.0763 |
| 6 | 0.1074 |
| 7 | 0.1186 |
| 8 | 0.2784 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6388 |
| 1 | 0.6000 |
| 2 | 0.6088 |
| 3 | 0.5727 |
| 4 | 0.6193 |
| 5 | 0.6353 |
| 6 | 0.6306 |
| 7 | 0.6030 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.903 |  | 8 | 0.967 |  | 16 | 0.992 |  | 24 | 0.988 |
| 1 | 0.431 |  | 9 | 0.917 |  | 17 | 0.921 |  | 25 | 0.992 |
| 2 | 0.995 |  | 10 | 1.000 |  | 18 | 0.833 |  | 26 | 0.988 |
| 3 | 0.532 |  | 11 | 0.963 |  | 19 | 0.967 |  | 27 | 0.990 |
| 4 | 0.945 |  | 12 | 0.652 |  | 20 | 0.978 |  | 28 | 0.985 |
| 5 | 0.964 |  | 13 | 0.841 |  | 21 | 0.950 |  | 29 | 0.999 |
| 6 | 0.953 |  | 14 | 0.998 |  | 22 | 0.959 |  | 30 | 0.965 |
| 7 | 0.900 |  | 15 | 0.995 |  | 23 | 0.958 |  | 31 | 0.969 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-I-3.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-I-3.png) · [head_correlation](plots/head_correlation_2024-I-3.png) · [n_drifted_hist](plots/n_drifted_hist_2024-I-3.png)


## Problem `2024-I-4`

- Decode steps logged: **1,253** (of 3,159 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **20** (0.6329% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9973 | 1.0000 | 0.9677 | 0.9257 | 0.8961 |
| 0.90 | 0.8913 | 0.9832 | 0.6470 | 0.5872 | 0.5564 |
| 0.85 | 0.6567 | 0.8971 | 0.2617 | 0.2854 | 0.2846 |
| 0.80 | 0.3799 | 0.7269 | 0.0841 | 0.1220 | 0.1385 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.1087 |
| 1 | 0.1010 |
| 2 | 0.0770 |
| 3 | 0.0925 |
| 4 | 0.0686 |
| 5 | 0.0723 |
| 6 | 0.0992 |
| 7 | 0.1114 |
| 8 | 0.2693 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6062 |
| 1 | 0.5672 |
| 2 | 0.5881 |
| 3 | 0.5519 |
| 4 | 0.5886 |
| 5 | 0.6033 |
| 6 | 0.6102 |
| 7 | 0.5823 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.848 |  | 8 | 0.954 |  | 16 | 0.977 |  | 24 | 0.969 |
| 1 | 0.302 |  | 9 | 0.897 |  | 17 | 0.833 |  | 25 | 0.978 |
| 2 | 0.999 |  | 10 | 0.997 |  | 18 | 0.790 |  | 26 | 0.974 |
| 3 | 0.454 |  | 11 | 0.954 |  | 19 | 0.914 |  | 27 | 0.985 |
| 4 | 0.955 |  | 12 | 0.608 |  | 20 | 0.931 |  | 28 | 0.990 |
| 5 | 0.955 |  | 13 | 0.811 |  | 21 | 0.904 |  | 29 | 1.000 |
| 6 | 0.957 |  | 14 | 0.993 |  | 22 | 0.915 |  | 30 | 0.956 |
| 7 | 0.890 |  | 15 | 0.955 |  | 23 | 0.923 |  | 31 | 0.954 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-I-4.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-I-4.png) · [head_correlation](plots/head_correlation_2024-I-4.png) · [n_drifted_hist](plots/n_drifted_hist_2024-I-4.png)


## Problem `2024-II-4`

- Decode steps logged: **1,407** (of 3,291 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **5** (0.1519% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9954 | 0.9996 | 0.9509 | 0.9039 | 0.8768 |
| 0.90 | 0.8567 | 0.9722 | 0.5887 | 0.5481 | 0.5242 |
| 0.85 | 0.6008 | 0.8499 | 0.2442 | 0.2702 | 0.2708 |
| 0.80 | 0.3397 | 0.6670 | 0.1025 | 0.1265 | 0.1393 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.1433 |
| 1 | 0.1086 |
| 2 | 0.0840 |
| 3 | 0.0934 |
| 4 | 0.0688 |
| 5 | 0.0659 |
| 6 | 0.0863 |
| 7 | 0.0921 |
| 8 | 0.2576 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.5668 |
| 1 | 0.5341 |
| 2 | 0.5421 |
| 3 | 0.5196 |
| 4 | 0.5448 |
| 5 | 0.5661 |
| 6 | 0.5730 |
| 7 | 0.5386 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.871 |  | 8 | 0.947 |  | 16 | 0.940 |  | 24 | 0.947 |
| 1 | 0.175 |  | 9 | 0.892 |  | 17 | 0.746 |  | 25 | 0.966 |
| 2 | 0.977 |  | 10 | 0.996 |  | 18 | 0.729 |  | 26 | 0.965 |
| 3 | 0.392 |  | 11 | 0.944 |  | 19 | 0.817 |  | 27 | 0.986 |
| 4 | 0.905 |  | 12 | 0.539 |  | 20 | 0.874 |  | 28 | 0.984 |
| 5 | 0.920 |  | 13 | 0.764 |  | 21 | 0.806 |  | 29 | 0.991 |
| 6 | 0.953 |  | 14 | 0.975 |  | 22 | 0.846 |  | 30 | 0.962 |
| 7 | 0.891 |  | 15 | 0.923 |  | 23 | 0.837 |  | 31 | 0.954 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-II-4.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-II-4.png) · [head_correlation](plots/head_correlation_2024-II-4.png) · [n_drifted_hist](plots/n_drifted_hist_2024-II-4.png)
