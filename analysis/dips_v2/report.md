# `dips_v2` — per-head cosine-similarity analysis

Uses the full `[n_steps, n_layers=32, n_q_heads=32]` cosine-sim tensor cached per problem. This lets us compute any aggregation over heads without re-running.

Problems included: `2024-I-9`, `2024-II-5`, `2024-II-9`


## Global rates at τ=0.9 (for quick comparison)

| pid | n_valid steps | need_corr | per-q-head | per-kv-head | mean<0.9 |
|---|---|---|---|---|---|
| `2024-I-9` | 30,052 | 0.8830 | 0.4817 | 0.5083 | 0.5500 |
| `2024-II-5` | 30,026 | 0.9067 | 0.5540 | 0.5876 | 0.6508 |
| `2024-II-9` | 30,066 | 0.9243 | 0.5646 | 0.6024 | 0.6828 |


## Problem `2024-I-9`

- Decode steps logged: **30,052** (of 31,999 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **3,172** (9.9125% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 1.0000 | 1.0000 | 0.9665 | 0.9243 | 0.8846 |
| 0.90 | 0.8830 | 0.9943 | 0.5500 | 0.5083 | 0.4817 |
| 0.85 | 0.5719 | 0.8998 | 0.1609 | 0.2128 | 0.2250 |
| 0.80 | 0.2844 | 0.6865 | 0.0682 | 0.0959 | 0.1127 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.1170 |
| 1 | 0.1190 |
| 2 | 0.1270 |
| 3 | 0.1224 |
| 4 | 0.0777 |
| 5 | 0.0725 |
| 6 | 0.0893 |
| 7 | 0.0833 |
| 8 | 0.1918 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.5558 |
| 1 | 0.4861 |
| 2 | 0.4841 |
| 3 | 0.4921 |
| 4 | 0.4826 |
| 5 | 0.5564 |
| 6 | 0.5161 |
| 7 | 0.4933 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.700 |  | 8 | 1.000 |  | 16 | 1.000 |  | 24 | 0.989 |
| 1 | 0.442 |  | 9 | 0.901 |  | 17 | 0.922 |  | 25 | 0.976 |
| 2 | 1.000 |  | 10 | 1.000 |  | 18 | 0.711 |  | 26 | 0.997 |
| 3 | 0.299 |  | 11 | 1.000 |  | 19 | 0.995 |  | 27 | 0.959 |
| 4 | 0.710 |  | 12 | 0.472 |  | 20 | 1.000 |  | 28 | 0.990 |
| 5 | 0.946 |  | 13 | 0.943 |  | 21 | 0.995 |  | 29 | 1.000 |
| 6 | 0.855 |  | 14 | 1.000 |  | 22 | 0.978 |  | 30 | 0.995 |
| 7 | 0.731 |  | 15 | 1.000 |  | 23 | 0.866 |  | 31 | 0.886 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-I-9.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-I-9.png) · [head_correlation](plots/head_correlation_2024-I-9.png) · [n_drifted_hist](plots/n_drifted_hist_2024-I-9.png)


## Problem `2024-II-5`

- Decode steps logged: **30,026** (of 31,999 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **762** (2.3813% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9990 | 1.0000 | 0.9762 | 0.9411 | 0.9093 |
| 0.90 | 0.9067 | 0.9910 | 0.6508 | 0.5876 | 0.5540 |
| 0.85 | 0.6450 | 0.9153 | 0.2357 | 0.2671 | 0.2708 |
| 0.80 | 0.3457 | 0.7248 | 0.0784 | 0.1140 | 0.1314 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0933 |
| 1 | 0.1045 |
| 2 | 0.0844 |
| 3 | 0.0955 |
| 4 | 0.0731 |
| 5 | 0.0736 |
| 6 | 0.1013 |
| 7 | 0.1206 |
| 8 | 0.2535 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6233 |
| 1 | 0.5627 |
| 2 | 0.5712 |
| 3 | 0.5523 |
| 4 | 0.5846 |
| 5 | 0.6188 |
| 6 | 0.5956 |
| 7 | 0.5925 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.788 |  | 8 | 0.964 |  | 16 | 0.992 |  | 24 | 0.982 |
| 1 | 0.382 |  | 9 | 0.905 |  | 17 | 0.859 |  | 25 | 0.993 |
| 2 | 0.999 |  | 10 | 0.999 |  | 18 | 0.806 |  | 26 | 0.985 |
| 3 | 0.456 |  | 11 | 0.958 |  | 19 | 0.973 |  | 27 | 0.981 |
| 4 | 0.945 |  | 12 | 0.657 |  | 20 | 0.983 |  | 28 | 0.986 |
| 5 | 0.957 |  | 13 | 0.855 |  | 21 | 0.966 |  | 29 | 1.000 |
| 6 | 0.944 |  | 14 | 0.998 |  | 22 | 0.954 |  | 30 | 0.970 |
| 7 | 0.872 |  | 15 | 0.986 |  | 23 | 0.971 |  | 31 | 0.948 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-II-5.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-II-5.png) · [head_correlation](plots/head_correlation_2024-II-5.png) · [n_drifted_hist](plots/n_drifted_hist_2024-II-5.png)


## Problem `2024-II-9`

- Decode steps logged: **30,066** (of 31,999 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **46** (0.1437% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9998 | 1.0000 | 0.9761 | 0.9473 | 0.9141 |
| 0.90 | 0.9243 | 0.9942 | 0.6828 | 0.6024 | 0.5646 |
| 0.85 | 0.6603 | 0.9335 | 0.2258 | 0.2675 | 0.2766 |
| 0.80 | 0.3411 | 0.7434 | 0.0877 | 0.1189 | 0.1362 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0757 |
| 1 | 0.0922 |
| 2 | 0.0830 |
| 3 | 0.0993 |
| 4 | 0.0845 |
| 5 | 0.0867 |
| 6 | 0.1077 |
| 7 | 0.1226 |
| 8 | 0.2484 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6337 |
| 1 | 0.5722 |
| 2 | 0.5835 |
| 3 | 0.5611 |
| 4 | 0.5843 |
| 5 | 0.6480 |
| 6 | 0.6195 |
| 7 | 0.6167 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.753 |  | 8 | 0.984 |  | 16 | 1.000 |  | 24 | 0.998 |
| 1 | 0.302 |  | 9 | 0.988 |  | 17 | 0.904 |  | 25 | 1.000 |
| 2 | 0.999 |  | 10 | 1.000 |  | 18 | 0.835 |  | 26 | 0.999 |
| 3 | 0.472 |  | 11 | 0.995 |  | 19 | 0.999 |  | 27 | 0.997 |
| 4 | 0.889 |  | 12 | 0.825 |  | 20 | 0.998 |  | 28 | 0.997 |
| 5 | 0.903 |  | 13 | 0.942 |  | 21 | 0.997 |  | 29 | 1.000 |
| 6 | 0.906 |  | 14 | 0.999 |  | 22 | 0.991 |  | 30 | 0.994 |
| 7 | 0.931 |  | 15 | 0.999 |  | 23 | 0.993 |  | 31 | 0.990 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_2024-II-9.png) · [per_kv_head_rate](plots/per_kv_head_rate_2024-II-9.png) · [head_correlation](plots/head_correlation_2024-II-9.png) · [n_drifted_hist](plots/n_drifted_hist_2024-II-9.png)
