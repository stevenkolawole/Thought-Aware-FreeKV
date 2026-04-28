# `math50` — per-head cosine-similarity analysis

Uses the full `[n_steps, n_layers=32, n_q_heads=32]` cosine-sim tensor cached per problem. This lets us compute any aggregation over heads without re-running.

Problems included: `test_algebra_1072_json`, `test_algebra_1098_json`, `test_algebra_1265_json`, `test_algebra_1349_json`, `test_algebra_1837_json`, `test_algebra_2036_json`, `test_algebra_2193_json`, `test_algebra_2214_json`, `test_algebra_2427_json`, `test_algebra_24_json`, `test_algebra_2584_json`, `test_algebra_305_json`, `test_counting_and_probability_119_json`, `test_counting_and_probability_134_json`, `test_counting_and_probability_525_json`, `test_counting_and_probability_666_json`, `test_geometry_178_json`, `test_geometry_248_json`, `test_geometry_434_json`, `test_geometry_627_json`, `test_geometry_967_json`, `test_intermediate_algebra_1000_json`, `test_intermediate_algebra_1197_json`, `test_intermediate_algebra_134_json`, `test_intermediate_algebra_1388_json`, `test_intermediate_algebra_1454_json`, `test_intermediate_algebra_1994_json`, `test_intermediate_algebra_428_json`, `test_intermediate_algebra_607_json`, `test_number_theory_1032_json`, `test_number_theory_45_json`, `test_number_theory_515_json`, `test_number_theory_572_json`, `test_number_theory_627_json`, `test_number_theory_737_json`, `test_number_theory_864_json`, `test_prealgebra_1139_json`, `test_prealgebra_1302_json`, `test_prealgebra_1388_json`, `test_prealgebra_1558_json`, `test_prealgebra_1622_json`, `test_prealgebra_1840_json`, `test_prealgebra_930_json`, `test_precalculus_1199_json`, `test_precalculus_1303_json`, `test_precalculus_285_json`, `test_precalculus_779_json`, `test_precalculus_807_json`, `test_precalculus_927_json`, `test_precalculus_990_json`


## Global rates at τ=0.9 (for quick comparison)

| pid | n_valid steps | need_corr | per-q-head | per-kv-head | mean<0.9 |
|---|---|---|---|---|---|
| `test_algebra_1072_json` | 29 | 0.8103 | 0.4614 | 0.4849 | 0.5431 |
| `test_algebra_1349_json` | 1,274 | 0.9030 | 0.5808 | 0.6158 | 0.6766 |
| `test_algebra_1837_json` | 14,477 | 0.9214 | 0.5890 | 0.6243 | 0.6901 |
| `test_algebra_2193_json` | 92 | 0.8376 | 0.4978 | 0.5165 | 0.5408 |
| `test_algebra_2427_json` | 14,450 | 0.8818 | 0.5390 | 0.5699 | 0.6297 |
| `test_algebra_2584_json` | 580 | 0.8580 | 0.5335 | 0.5574 | 0.6016 |
| `test_counting_and_probability_119_json` | 1,976 | 0.8671 | 0.5399 | 0.5656 | 0.6061 |
| `test_counting_and_probability_134_json` | 14,404 | 0.9019 | 0.5300 | 0.5614 | 0.6214 |
| `test_counting_and_probability_525_json` | 7,748 | 0.9095 | 0.5676 | 0.6017 | 0.6645 |
| `test_geometry_434_json` | 14,499 | 0.9236 | 0.5750 | 0.6128 | 0.6875 |
| `test_geometry_627_json` | 3,314 | 0.9106 | 0.5774 | 0.6111 | 0.6796 |
| `test_intermediate_algebra_1000_json` | 2,190 | 0.8887 | 0.5523 | 0.5797 | 0.6340 |
| `test_intermediate_algebra_1197_json` | 2,649 | 0.8919 | 0.5553 | 0.5845 | 0.6408 |
| `test_intermediate_algebra_1388_json` | 1,958 | 0.8773 | 0.5475 | 0.5736 | 0.6205 |
| `test_intermediate_algebra_1454_json` | 14,427 | 0.9052 | 0.5706 | 0.6029 | 0.6608 |
| `test_intermediate_algebra_1994_json` | 2,914 | 0.8701 | 0.5427 | 0.5687 | 0.6257 |
| `test_intermediate_algebra_428_json` | 1,097 | 0.9074 | 0.5763 | 0.6067 | 0.6644 |
| `test_intermediate_algebra_607_json` | 6,221 | 0.8804 | 0.5665 | 0.5932 | 0.6393 |
| `test_number_theory_1032_json` | 14,377 | 0.8907 | 0.5146 | 0.5430 | 0.5927 |
| `test_number_theory_515_json` | 1,412 | 0.8885 | 0.5496 | 0.5775 | 0.6343 |
| `test_number_theory_627_json` | 2,322 | 0.9065 | 0.6057 | 0.6362 | 0.6926 |
| `test_number_theory_737_json` | 724 | 0.9097 | 0.5847 | 0.6153 | 0.6719 |
| `test_number_theory_864_json` | 14,376 | 0.9141 | 0.6130 | 0.6430 | 0.7026 |
| `test_prealgebra_1139_json` | 14,494 | 0.9061 | 0.5280 | 0.5616 | 0.6175 |
| `test_precalculus_1199_json` | 1,722 | 0.8950 | 0.5565 | 0.5881 | 0.6478 |
| `test_precalculus_285_json` | 3,357 | 0.8731 | 0.5544 | 0.5808 | 0.6280 |
| `test_precalculus_927_json` | 333 | 0.8375 | 0.5053 | 0.5231 | 0.5538 |
| `test_precalculus_990_json` | 3,592 | 0.8601 | 0.5479 | 0.5716 | 0.6098 |


## Problem `test_algebra_1072_json`

- Decode steps logged: **29** (of 2,014 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **9** (0.4467% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9903 | 1.0000 | 0.9332 | 0.8832 | 0.8500 |
| 0.90 | 0.8103 | 0.9472 | 0.5431 | 0.4849 | 0.4614 |
| 0.85 | 0.5474 | 0.8114 | 0.1746 | 0.2099 | 0.2133 |
| 0.80 | 0.2877 | 0.6272 | 0.0366 | 0.0760 | 0.0934 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.1897 |
| 1 | 0.1272 |
| 2 | 0.0776 |
| 3 | 0.1024 |
| 4 | 0.0647 |
| 5 | 0.0647 |
| 6 | 0.0884 |
| 7 | 0.1067 |
| 8 | 0.1789 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.5097 |
| 1 | 0.4601 |
| 2 | 0.4709 |
| 3 | 0.4515 |
| 4 | 0.4903 |
| 5 | 0.5097 |
| 6 | 0.5151 |
| 7 | 0.4720 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.897 |  | 8 | 0.862 |  | 16 | 0.966 |  | 24 | 0.931 |
| 1 | 0.414 |  | 9 | 0.759 |  | 17 | 0.586 |  | 25 | 0.966 |
| 2 | 1.000 |  | 10 | 0.966 |  | 18 | 0.621 |  | 26 | 0.966 |
| 3 | 0.310 |  | 11 | 0.828 |  | 19 | 0.724 |  | 27 | 0.966 |
| 4 | 0.759 |  | 12 | 0.345 |  | 20 | 0.862 |  | 28 | 0.966 |
| 5 | 0.862 |  | 13 | 0.517 |  | 21 | 0.828 |  | 29 | 1.000 |
| 6 | 0.931 |  | 14 | 0.966 |  | 22 | 0.793 |  | 30 | 0.931 |
| 7 | 0.793 |  | 15 | 0.862 |  | 23 | 0.828 |  | 31 | 0.931 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_test_algebra_1072_json.png) · [per_kv_head_rate](plots/per_kv_head_rate_test_algebra_1072_json.png) · [head_correlation](plots/head_correlation_test_algebra_1072_json.png) · [n_drifted_hist](plots/n_drifted_hist_test_algebra_1072_json.png)


## Problem `test_algebra_1349_json`

- Decode steps logged: **1,274** (of 2,977 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **18** (0.6044% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9985 | 1.0000 | 0.9677 | 0.9305 | 0.9034 |
| 0.90 | 0.9030 | 0.9871 | 0.6766 | 0.6158 | 0.5808 |
| 0.85 | 0.6799 | 0.9082 | 0.2954 | 0.3115 | 0.3064 |
| 0.80 | 0.4094 | 0.7470 | 0.1033 | 0.1379 | 0.1514 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0970 |
| 1 | 0.0925 |
| 2 | 0.0735 |
| 3 | 0.0843 |
| 4 | 0.0668 |
| 5 | 0.0685 |
| 6 | 0.0985 |
| 7 | 0.1182 |
| 8 | 0.3007 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6388 |
| 1 | 0.5968 |
| 2 | 0.6105 |
| 3 | 0.5808 |
| 4 | 0.6091 |
| 5 | 0.6385 |
| 6 | 0.6401 |
| 7 | 0.6115 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.808 |  | 8 | 0.967 |  | 16 | 0.973 |  | 24 | 0.962 |
| 1 | 0.395 |  | 9 | 0.925 |  | 17 | 0.885 |  | 25 | 0.987 |
| 2 | 0.999 |  | 10 | 1.000 |  | 18 | 0.827 |  | 26 | 0.969 |
| 3 | 0.443 |  | 11 | 0.974 |  | 19 | 0.939 |  | 27 | 0.984 |
| 4 | 0.924 |  | 12 | 0.699 |  | 20 | 0.944 |  | 28 | 0.987 |
| 5 | 0.962 |  | 13 | 0.874 |  | 21 | 0.927 |  | 29 | 0.998 |
| 6 | 0.956 |  | 14 | 0.987 |  | 22 | 0.917 |  | 30 | 0.951 |
| 7 | 0.885 |  | 15 | 0.973 |  | 23 | 0.936 |  | 31 | 0.937 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_test_algebra_1349_json.png) · [per_kv_head_rate](plots/per_kv_head_rate_test_algebra_1349_json.png) · [head_correlation](plots/head_correlation_test_algebra_1349_json.png) · [n_drifted_hist](plots/n_drifted_hist_test_algebra_1349_json.png)


## Problem `test_algebra_1837_json`

- Decode steps logged: **14,477** (of 16,383 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **103** (0.6287% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9994 | 1.0000 | 0.9787 | 0.9455 | 0.9174 |
| 0.90 | 0.9214 | 0.9912 | 0.6901 | 0.6243 | 0.5890 |
| 0.85 | 0.6824 | 0.9260 | 0.2840 | 0.3128 | 0.3134 |
| 0.80 | 0.3953 | 0.7518 | 0.1236 | 0.1546 | 0.1684 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0786 |
| 1 | 0.0913 |
| 2 | 0.0783 |
| 3 | 0.0915 |
| 4 | 0.0727 |
| 5 | 0.0716 |
| 6 | 0.0959 |
| 7 | 0.1142 |
| 8 | 0.3061 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6530 |
| 1 | 0.6095 |
| 2 | 0.6071 |
| 3 | 0.5898 |
| 4 | 0.6201 |
| 5 | 0.6558 |
| 6 | 0.6334 |
| 7 | 0.6255 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.860 |  | 8 | 0.966 |  | 16 | 0.988 |  | 24 | 0.996 |
| 1 | 0.418 |  | 9 | 0.947 |  | 17 | 0.864 |  | 25 | 1.000 |
| 2 | 1.000 |  | 10 | 0.997 |  | 18 | 0.824 |  | 26 | 0.995 |
| 3 | 0.462 |  | 11 | 0.963 |  | 19 | 0.981 |  | 27 | 0.995 |
| 4 | 0.974 |  | 12 | 0.751 |  | 20 | 0.986 |  | 28 | 0.995 |
| 5 | 0.948 |  | 13 | 0.868 |  | 21 | 0.968 |  | 29 | 1.000 |
| 6 | 0.956 |  | 14 | 0.998 |  | 22 | 0.963 |  | 30 | 0.985 |
| 7 | 0.918 |  | 15 | 0.984 |  | 23 | 0.963 |  | 31 | 0.973 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_test_algebra_1837_json.png) · [per_kv_head_rate](plots/per_kv_head_rate_test_algebra_1837_json.png) · [head_correlation](plots/head_correlation_test_algebra_1837_json.png) · [n_drifted_hist](plots/n_drifted_hist_test_algebra_1837_json.png)


## Problem `test_algebra_2193_json`

- Decode steps logged: **92** (of 2,095 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **6** (0.2863% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9929 | 1.0000 | 0.9457 | 0.8905 | 0.8579 |
| 0.90 | 0.8376 | 0.9630 | 0.5408 | 0.5165 | 0.4978 |
| 0.85 | 0.5744 | 0.8410 | 0.2473 | 0.2676 | 0.2693 |
| 0.80 | 0.3346 | 0.6467 | 0.1168 | 0.1352 | 0.1440 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.1624 |
| 1 | 0.1284 |
| 2 | 0.0870 |
| 3 | 0.0941 |
| 4 | 0.0669 |
| 5 | 0.0611 |
| 6 | 0.0781 |
| 7 | 0.0707 |
| 8 | 0.2514 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.5285 |
| 1 | 0.5020 |
| 2 | 0.5037 |
| 3 | 0.4915 |
| 4 | 0.5268 |
| 5 | 0.5326 |
| 6 | 0.5397 |
| 7 | 0.5071 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.891 |  | 8 | 0.935 |  | 16 | 0.924 |  | 24 | 0.902 |
| 1 | 0.196 |  | 9 | 0.859 |  | 17 | 0.696 |  | 25 | 0.967 |
| 2 | 1.000 |  | 10 | 1.000 |  | 18 | 0.674 |  | 26 | 0.946 |
| 3 | 0.348 |  | 11 | 0.902 |  | 19 | 0.783 |  | 27 | 0.978 |
| 4 | 0.913 |  | 12 | 0.467 |  | 20 | 0.859 |  | 28 | 1.000 |
| 5 | 0.924 |  | 13 | 0.641 |  | 21 | 0.761 |  | 29 | 1.000 |
| 6 | 0.935 |  | 14 | 0.978 |  | 22 | 0.815 |  | 30 | 1.000 |
| 7 | 0.859 |  | 15 | 0.859 |  | 23 | 0.815 |  | 31 | 0.978 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_test_algebra_2193_json.png) · [per_kv_head_rate](plots/per_kv_head_rate_test_algebra_2193_json.png) · [head_correlation](plots/head_correlation_test_algebra_2193_json.png) · [n_drifted_hist](plots/n_drifted_hist_test_algebra_2193_json.png)


## Problem `test_algebra_2427_json`

- Decode steps logged: **14,450** (of 16,383 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **91** (0.5554% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9978 | 1.0000 | 0.9645 | 0.9205 | 0.8924 |
| 0.90 | 0.8818 | 0.9819 | 0.6297 | 0.5699 | 0.5390 |
| 0.85 | 0.6342 | 0.8827 | 0.2387 | 0.2684 | 0.2684 |
| 0.80 | 0.3627 | 0.7079 | 0.0810 | 0.1155 | 0.1304 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.1182 |
| 1 | 0.1033 |
| 2 | 0.0857 |
| 3 | 0.0933 |
| 4 | 0.0724 |
| 5 | 0.0683 |
| 6 | 0.0970 |
| 7 | 0.1026 |
| 8 | 0.2592 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6022 |
| 1 | 0.5574 |
| 2 | 0.5555 |
| 3 | 0.5453 |
| 4 | 0.5658 |
| 5 | 0.5921 |
| 6 | 0.5878 |
| 7 | 0.5531 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.886 |  | 8 | 0.946 |  | 16 | 0.945 |  | 24 | 0.968 |
| 1 | 0.304 |  | 9 | 0.851 |  | 17 | 0.865 |  | 25 | 0.975 |
| 2 | 0.999 |  | 10 | 0.999 |  | 18 | 0.780 |  | 26 | 0.977 |
| 3 | 0.348 |  | 11 | 0.950 |  | 19 | 0.940 |  | 27 | 0.990 |
| 4 | 0.901 |  | 12 | 0.549 |  | 20 | 0.928 |  | 28 | 0.990 |
| 5 | 0.931 |  | 13 | 0.771 |  | 21 | 0.892 |  | 29 | 0.998 |
| 6 | 0.967 |  | 14 | 0.975 |  | 22 | 0.892 |  | 30 | 0.974 |
| 7 | 0.872 |  | 15 | 0.954 |  | 23 | 0.926 |  | 31 | 0.975 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_test_algebra_2427_json.png) · [per_kv_head_rate](plots/per_kv_head_rate_test_algebra_2427_json.png) · [head_correlation](plots/head_correlation_test_algebra_2427_json.png) · [n_drifted_hist](plots/n_drifted_hist_test_algebra_2427_json.png)


## Problem `test_algebra_2584_json`

- Decode steps logged: **580** (of 2,563 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **13** (0.5070% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9931 | 0.9996 | 0.9436 | 0.8984 | 0.8725 |
| 0.90 | 0.8580 | 0.9699 | 0.6016 | 0.5574 | 0.5335 |
| 0.85 | 0.6167 | 0.8513 | 0.2567 | 0.2880 | 0.2897 |
| 0.80 | 0.3600 | 0.6772 | 0.1290 | 0.1490 | 0.1586 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.1420 |
| 1 | 0.0996 |
| 2 | 0.0865 |
| 3 | 0.0900 |
| 4 | 0.0671 |
| 5 | 0.0655 |
| 6 | 0.0907 |
| 7 | 0.0930 |
| 8 | 0.2657 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.5754 |
| 1 | 0.5372 |
| 2 | 0.5520 |
| 3 | 0.5242 |
| 4 | 0.5603 |
| 5 | 0.5818 |
| 6 | 0.5873 |
| 7 | 0.5406 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.831 |  | 8 | 0.955 |  | 16 | 0.943 |  | 24 | 0.950 |
| 1 | 0.162 |  | 9 | 0.912 |  | 17 | 0.710 |  | 25 | 0.952 |
| 2 | 0.995 |  | 10 | 0.991 |  | 18 | 0.759 |  | 26 | 0.962 |
| 3 | 0.348 |  | 11 | 0.947 |  | 19 | 0.802 |  | 27 | 0.995 |
| 4 | 0.928 |  | 12 | 0.574 |  | 20 | 0.859 |  | 28 | 0.993 |
| 5 | 0.933 |  | 13 | 0.834 |  | 21 | 0.812 |  | 29 | 0.988 |
| 6 | 0.959 |  | 14 | 0.988 |  | 22 | 0.848 |  | 30 | 0.962 |
| 7 | 0.898 |  | 15 | 0.883 |  | 23 | 0.831 |  | 31 | 0.953 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_test_algebra_2584_json.png) · [per_kv_head_rate](plots/per_kv_head_rate_test_algebra_2584_json.png) · [head_correlation](plots/head_correlation_test_algebra_2584_json.png) · [n_drifted_hist](plots/n_drifted_hist_test_algebra_2584_json.png)


## Problem `test_counting_and_probability_119_json`

- Decode steps logged: **1,976** (of 3,974 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **21** (0.5283% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9981 | 1.0000 | 0.9608 | 0.9152 | 0.8868 |
| 0.90 | 0.8671 | 0.9790 | 0.6061 | 0.5656 | 0.5399 |
| 0.85 | 0.6165 | 0.8642 | 0.2636 | 0.2867 | 0.2864 |
| 0.80 | 0.3548 | 0.6841 | 0.1102 | 0.1367 | 0.1491 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.1329 |
| 1 | 0.1032 |
| 2 | 0.0817 |
| 3 | 0.0942 |
| 4 | 0.0661 |
| 5 | 0.0606 |
| 6 | 0.0911 |
| 7 | 0.0992 |
| 8 | 0.2709 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.5863 |
| 1 | 0.5530 |
| 2 | 0.5584 |
| 3 | 0.5404 |
| 4 | 0.5660 |
| 5 | 0.5814 |
| 6 | 0.5913 |
| 7 | 0.5484 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.880 |  | 8 | 0.951 |  | 16 | 0.941 |  | 24 | 0.970 |
| 1 | 0.186 |  | 9 | 0.901 |  | 17 | 0.780 |  | 25 | 0.974 |
| 2 | 0.992 |  | 10 | 0.999 |  | 18 | 0.734 |  | 26 | 0.973 |
| 3 | 0.335 |  | 11 | 0.957 |  | 19 | 0.818 |  | 27 | 0.996 |
| 4 | 0.930 |  | 12 | 0.601 |  | 20 | 0.877 |  | 28 | 0.996 |
| 5 | 0.922 |  | 13 | 0.841 |  | 21 | 0.803 |  | 29 | 1.000 |
| 6 | 0.940 |  | 14 | 0.983 |  | 22 | 0.861 |  | 30 | 0.984 |
| 7 | 0.870 |  | 15 | 0.915 |  | 23 | 0.862 |  | 31 | 0.977 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_test_counting_and_probability_119_json.png) · [per_kv_head_rate](plots/per_kv_head_rate_test_counting_and_probability_119_json.png) · [head_correlation](plots/head_correlation_test_counting_and_probability_119_json.png) · [n_drifted_hist](plots/n_drifted_hist_test_counting_and_probability_119_json.png)


## Problem `test_counting_and_probability_134_json`

- Decode steps logged: **14,404** (of 16,383 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **264** (1.6113% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9988 | 1.0000 | 0.9763 | 0.9349 | 0.8997 |
| 0.90 | 0.9019 | 0.9902 | 0.6214 | 0.5614 | 0.5300 |
| 0.85 | 0.6241 | 0.9150 | 0.2090 | 0.2519 | 0.2600 |
| 0.80 | 0.3253 | 0.7083 | 0.0814 | 0.1135 | 0.1302 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0981 |
| 1 | 0.1166 |
| 2 | 0.0954 |
| 3 | 0.0958 |
| 4 | 0.0782 |
| 5 | 0.0783 |
| 6 | 0.0975 |
| 7 | 0.1144 |
| 8 | 0.2258 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.5933 |
| 1 | 0.5403 |
| 2 | 0.5442 |
| 3 | 0.5222 |
| 4 | 0.5581 |
| 5 | 0.5891 |
| 6 | 0.5720 |
| 7 | 0.5717 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.775 |  | 8 | 0.966 |  | 16 | 0.990 |  | 24 | 0.986 |
| 1 | 0.411 |  | 9 | 0.883 |  | 17 | 0.859 |  | 25 | 0.997 |
| 2 | 0.999 |  | 10 | 0.998 |  | 18 | 0.798 |  | 26 | 0.989 |
| 3 | 0.498 |  | 11 | 0.946 |  | 19 | 0.982 |  | 27 | 0.984 |
| 4 | 0.962 |  | 12 | 0.545 |  | 20 | 0.987 |  | 28 | 0.988 |
| 5 | 0.965 |  | 13 | 0.741 |  | 21 | 0.970 |  | 29 | 1.000 |
| 6 | 0.949 |  | 14 | 0.997 |  | 22 | 0.955 |  | 30 | 0.963 |
| 7 | 0.876 |  | 15 | 0.980 |  | 23 | 0.965 |  | 31 | 0.959 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_test_counting_and_probability_134_json.png) · [per_kv_head_rate](plots/per_kv_head_rate_test_counting_and_probability_134_json.png) · [head_correlation](plots/head_correlation_test_counting_and_probability_134_json.png) · [n_drifted_hist](plots/n_drifted_hist_test_counting_and_probability_134_json.png)


## Problem `test_counting_and_probability_525_json`

- Decode steps logged: **7,748** (of 9,726 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **101** (1.0383% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9988 | 1.0000 | 0.9757 | 0.9394 | 0.9085 |
| 0.90 | 0.9095 | 0.9895 | 0.6645 | 0.6017 | 0.5676 |
| 0.85 | 0.6686 | 0.9168 | 0.2694 | 0.2927 | 0.2915 |
| 0.80 | 0.3865 | 0.7479 | 0.0905 | 0.1278 | 0.1436 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0905 |
| 1 | 0.0994 |
| 2 | 0.0819 |
| 3 | 0.0913 |
| 4 | 0.0723 |
| 5 | 0.0735 |
| 6 | 0.0981 |
| 7 | 0.1127 |
| 8 | 0.2803 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6302 |
| 1 | 0.5826 |
| 2 | 0.5944 |
| 3 | 0.5609 |
| 4 | 0.6050 |
| 5 | 0.6188 |
| 6 | 0.6198 |
| 7 | 0.6019 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.820 |  | 8 | 0.967 |  | 16 | 0.988 |  | 24 | 0.972 |
| 1 | 0.463 |  | 9 | 0.907 |  | 17 | 0.899 |  | 25 | 0.992 |
| 2 | 0.999 |  | 10 | 0.999 |  | 18 | 0.816 |  | 26 | 0.978 |
| 3 | 0.558 |  | 11 | 0.965 |  | 19 | 0.968 |  | 27 | 0.978 |
| 4 | 0.943 |  | 12 | 0.598 |  | 20 | 0.968 |  | 28 | 0.983 |
| 5 | 0.961 |  | 13 | 0.836 |  | 21 | 0.952 |  | 29 | 0.999 |
| 6 | 0.957 |  | 14 | 0.998 |  | 22 | 0.932 |  | 30 | 0.940 |
| 7 | 0.891 |  | 15 | 0.988 |  | 23 | 0.961 |  | 31 | 0.928 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_test_counting_and_probability_525_json.png) · [per_kv_head_rate](plots/per_kv_head_rate_test_counting_and_probability_525_json.png) · [head_correlation](plots/head_correlation_test_counting_and_probability_525_json.png) · [n_drifted_hist](plots/n_drifted_hist_test_counting_and_probability_525_json.png)


## Problem `test_geometry_434_json`

- Decode steps logged: **14,499** (of 16,383 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **267** (1.6296% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9997 | 1.0000 | 0.9854 | 0.9552 | 0.9231 |
| 0.90 | 0.9236 | 0.9953 | 0.6875 | 0.6128 | 0.5750 |
| 0.85 | 0.6719 | 0.9370 | 0.2478 | 0.2830 | 0.2878 |
| 0.80 | 0.3774 | 0.7560 | 0.0938 | 0.1284 | 0.1446 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0764 |
| 1 | 0.0935 |
| 2 | 0.0754 |
| 3 | 0.0975 |
| 4 | 0.0815 |
| 5 | 0.0789 |
| 6 | 0.1029 |
| 7 | 0.1239 |
| 8 | 0.2701 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6421 |
| 1 | 0.5927 |
| 2 | 0.5954 |
| 3 | 0.5718 |
| 4 | 0.6092 |
| 5 | 0.6481 |
| 6 | 0.6239 |
| 7 | 0.6190 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.804 |  | 8 | 0.963 |  | 16 | 0.994 |  | 24 | 0.996 |
| 1 | 0.414 |  | 9 | 0.924 |  | 17 | 0.889 |  | 25 | 0.999 |
| 2 | 1.000 |  | 10 | 1.000 |  | 18 | 0.812 |  | 26 | 0.993 |
| 3 | 0.573 |  | 11 | 0.978 |  | 19 | 0.994 |  | 27 | 0.998 |
| 4 | 0.966 |  | 12 | 0.709 |  | 20 | 0.997 |  | 28 | 0.989 |
| 5 | 0.956 |  | 13 | 0.880 |  | 21 | 0.992 |  | 29 | 1.000 |
| 6 | 0.963 |  | 14 | 0.993 |  | 22 | 0.955 |  | 30 | 0.982 |
| 7 | 0.892 |  | 15 | 0.995 |  | 23 | 0.976 |  | 31 | 0.978 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_test_geometry_434_json.png) · [per_kv_head_rate](plots/per_kv_head_rate_test_geometry_434_json.png) · [head_correlation](plots/head_correlation_test_geometry_434_json.png) · [n_drifted_hist](plots/n_drifted_hist_test_geometry_434_json.png)


## Problem `test_geometry_627_json`

- Decode steps logged: **3,314** (of 5,294 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **56** (1.0576% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9984 | 1.0000 | 0.9761 | 0.9368 | 0.9087 |
| 0.90 | 0.9106 | 0.9902 | 0.6796 | 0.6111 | 0.5774 |
| 0.85 | 0.6895 | 0.9183 | 0.2851 | 0.3082 | 0.3053 |
| 0.80 | 0.4110 | 0.7569 | 0.1072 | 0.1440 | 0.1576 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0894 |
| 1 | 0.0899 |
| 2 | 0.0769 |
| 3 | 0.0923 |
| 4 | 0.0746 |
| 5 | 0.0746 |
| 6 | 0.1033 |
| 7 | 0.1154 |
| 8 | 0.2837 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6392 |
| 1 | 0.5950 |
| 2 | 0.6019 |
| 3 | 0.5693 |
| 4 | 0.6155 |
| 5 | 0.6297 |
| 6 | 0.6337 |
| 7 | 0.6041 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.847 |  | 8 | 0.983 |  | 16 | 0.983 |  | 24 | 0.982 |
| 1 | 0.326 |  | 9 | 0.954 |  | 17 | 0.873 |  | 25 | 0.994 |
| 2 | 0.999 |  | 10 | 0.999 |  | 18 | 0.807 |  | 26 | 0.987 |
| 3 | 0.491 |  | 11 | 0.980 |  | 19 | 0.941 |  | 27 | 0.988 |
| 4 | 0.960 |  | 12 | 0.681 |  | 20 | 0.962 |  | 28 | 0.987 |
| 5 | 0.973 |  | 13 | 0.883 |  | 21 | 0.923 |  | 29 | 0.998 |
| 6 | 0.969 |  | 14 | 0.997 |  | 22 | 0.931 |  | 30 | 0.954 |
| 7 | 0.923 |  | 15 | 0.973 |  | 23 | 0.939 |  | 31 | 0.951 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_test_geometry_627_json.png) · [per_kv_head_rate](plots/per_kv_head_rate_test_geometry_627_json.png) · [head_correlation](plots/head_correlation_test_geometry_627_json.png) · [n_drifted_hist](plots/n_drifted_hist_test_geometry_627_json.png)


## Problem `test_intermediate_algebra_1000_json`

- Decode steps logged: **2,190** (of 4,173 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **15** (0.3594% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9976 | 0.9999 | 0.9646 | 0.9242 | 0.8942 |
| 0.90 | 0.8887 | 0.9834 | 0.6340 | 0.5797 | 0.5523 |
| 0.85 | 0.6438 | 0.8988 | 0.2663 | 0.2965 | 0.2964 |
| 0.80 | 0.3741 | 0.7179 | 0.1218 | 0.1467 | 0.1580 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.1113 |
| 1 | 0.1038 |
| 2 | 0.0798 |
| 3 | 0.0956 |
| 4 | 0.0710 |
| 5 | 0.0707 |
| 6 | 0.0947 |
| 7 | 0.1035 |
| 8 | 0.2696 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6066 |
| 1 | 0.5618 |
| 2 | 0.5754 |
| 3 | 0.5397 |
| 4 | 0.5878 |
| 5 | 0.5937 |
| 6 | 0.6010 |
| 7 | 0.5713 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.796 |  | 8 | 0.957 |  | 16 | 0.968 |  | 24 | 0.953 |
| 1 | 0.283 |  | 9 | 0.922 |  | 17 | 0.842 |  | 25 | 0.971 |
| 2 | 0.999 |  | 10 | 0.997 |  | 18 | 0.784 |  | 26 | 0.979 |
| 3 | 0.499 |  | 11 | 0.965 |  | 19 | 0.924 |  | 27 | 0.984 |
| 4 | 0.916 |  | 12 | 0.627 |  | 20 | 0.935 |  | 28 | 0.983 |
| 5 | 0.908 |  | 13 | 0.847 |  | 21 | 0.900 |  | 29 | 0.999 |
| 6 | 0.947 |  | 14 | 0.988 |  | 22 | 0.912 |  | 30 | 0.959 |
| 7 | 0.891 |  | 15 | 0.962 |  | 23 | 0.897 |  | 31 | 0.945 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_test_intermediate_algebra_1000_json.png) · [per_kv_head_rate](plots/per_kv_head_rate_test_intermediate_algebra_1000_json.png) · [head_correlation](plots/head_correlation_test_intermediate_algebra_1000_json.png) · [n_drifted_hist](plots/n_drifted_hist_test_intermediate_algebra_1000_json.png)


## Problem `test_intermediate_algebra_1197_json`

- Decode steps logged: **2,649** (of 4,620 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **15** (0.3246% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9978 | 0.9999 | 0.9689 | 0.9273 | 0.8998 |
| 0.90 | 0.8919 | 0.9838 | 0.6408 | 0.5845 | 0.5553 |
| 0.85 | 0.6554 | 0.8968 | 0.2634 | 0.2887 | 0.2886 |
| 0.80 | 0.3831 | 0.7267 | 0.1007 | 0.1364 | 0.1489 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.1081 |
| 1 | 0.0969 |
| 2 | 0.0877 |
| 3 | 0.0932 |
| 4 | 0.0697 |
| 5 | 0.0690 |
| 6 | 0.1005 |
| 7 | 0.1010 |
| 8 | 0.2738 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6090 |
| 1 | 0.5751 |
| 2 | 0.5759 |
| 3 | 0.5486 |
| 4 | 0.5939 |
| 5 | 0.5966 |
| 6 | 0.6051 |
| 7 | 0.5721 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.854 |  | 8 | 0.971 |  | 16 | 0.972 |  | 24 | 0.978 |
| 1 | 0.245 |  | 9 | 0.913 |  | 17 | 0.841 |  | 25 | 0.983 |
| 2 | 0.998 |  | 10 | 0.998 |  | 18 | 0.787 |  | 26 | 0.983 |
| 3 | 0.448 |  | 11 | 0.958 |  | 19 | 0.897 |  | 27 | 0.995 |
| 4 | 0.943 |  | 12 | 0.612 |  | 20 | 0.940 |  | 28 | 0.994 |
| 5 | 0.955 |  | 13 | 0.817 |  | 21 | 0.887 |  | 29 | 0.997 |
| 6 | 0.971 |  | 14 | 0.993 |  | 22 | 0.895 |  | 30 | 0.958 |
| 7 | 0.920 |  | 15 | 0.957 |  | 23 | 0.915 |  | 31 | 0.966 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_test_intermediate_algebra_1197_json.png) · [per_kv_head_rate](plots/per_kv_head_rate_test_intermediate_algebra_1197_json.png) · [head_correlation](plots/head_correlation_test_intermediate_algebra_1197_json.png) · [n_drifted_hist](plots/n_drifted_hist_test_intermediate_algebra_1197_json.png)


## Problem `test_intermediate_algebra_1388_json`

- Decode steps logged: **1,958** (of 3,909 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **22** (0.5627% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9963 | 0.9997 | 0.9596 | 0.9156 | 0.8870 |
| 0.90 | 0.8773 | 0.9791 | 0.6205 | 0.5736 | 0.5475 |
| 0.85 | 0.6357 | 0.8837 | 0.2690 | 0.2985 | 0.2989 |
| 0.80 | 0.3746 | 0.7063 | 0.1318 | 0.1553 | 0.1652 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.1227 |
| 1 | 0.1012 |
| 2 | 0.0847 |
| 3 | 0.0925 |
| 4 | 0.0682 |
| 5 | 0.0680 |
| 6 | 0.0898 |
| 7 | 0.0939 |
| 8 | 0.2789 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.5999 |
| 1 | 0.5572 |
| 2 | 0.5696 |
| 3 | 0.5338 |
| 4 | 0.5852 |
| 5 | 0.5852 |
| 6 | 0.5945 |
| 7 | 0.5632 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.830 |  | 8 | 0.962 |  | 16 | 0.955 |  | 24 | 0.969 |
| 1 | 0.215 |  | 9 | 0.903 |  | 17 | 0.813 |  | 25 | 0.979 |
| 2 | 0.987 |  | 10 | 0.998 |  | 18 | 0.773 |  | 26 | 0.981 |
| 3 | 0.396 |  | 11 | 0.951 |  | 19 | 0.884 |  | 27 | 0.995 |
| 4 | 0.931 |  | 12 | 0.612 |  | 20 | 0.917 |  | 28 | 0.994 |
| 5 | 0.924 |  | 13 | 0.807 |  | 21 | 0.867 |  | 29 | 0.999 |
| 6 | 0.959 |  | 14 | 0.981 |  | 22 | 0.881 |  | 30 | 0.966 |
| 7 | 0.881 |  | 15 | 0.933 |  | 23 | 0.873 |  | 31 | 0.956 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_test_intermediate_algebra_1388_json.png) · [per_kv_head_rate](plots/per_kv_head_rate_test_intermediate_algebra_1388_json.png) · [head_correlation](plots/head_correlation_test_intermediate_algebra_1388_json.png) · [n_drifted_hist](plots/n_drifted_hist_test_intermediate_algebra_1388_json.png)


## Problem `test_intermediate_algebra_1454_json`

- Decode steps logged: **14,427** (of 16,383 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **205** (1.2512% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9988 | 1.0000 | 0.9752 | 0.9374 | 0.9075 |
| 0.90 | 0.9052 | 0.9914 | 0.6608 | 0.6029 | 0.5706 |
| 0.85 | 0.6576 | 0.9146 | 0.2635 | 0.2986 | 0.3008 |
| 0.80 | 0.3634 | 0.7337 | 0.1188 | 0.1448 | 0.1587 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0948 |
| 1 | 0.0971 |
| 2 | 0.0780 |
| 3 | 0.0916 |
| 4 | 0.0702 |
| 5 | 0.0737 |
| 6 | 0.0990 |
| 7 | 0.1124 |
| 8 | 0.2831 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6329 |
| 1 | 0.5796 |
| 2 | 0.5974 |
| 3 | 0.5638 |
| 4 | 0.6011 |
| 5 | 0.6242 |
| 6 | 0.6206 |
| 7 | 0.6034 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.847 |  | 8 | 0.973 |  | 16 | 0.983 |  | 24 | 0.985 |
| 1 | 0.342 |  | 9 | 0.929 |  | 17 | 0.857 |  | 25 | 0.995 |
| 2 | 0.999 |  | 10 | 0.997 |  | 18 | 0.798 |  | 26 | 0.992 |
| 3 | 0.446 |  | 11 | 0.972 |  | 19 | 0.933 |  | 27 | 0.991 |
| 4 | 0.934 |  | 12 | 0.681 |  | 20 | 0.966 |  | 28 | 0.995 |
| 5 | 0.958 |  | 13 | 0.871 |  | 21 | 0.914 |  | 29 | 1.000 |
| 6 | 0.955 |  | 14 | 0.995 |  | 22 | 0.926 |  | 30 | 0.981 |
| 7 | 0.889 |  | 15 | 0.967 |  | 23 | 0.923 |  | 31 | 0.972 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_test_intermediate_algebra_1454_json.png) · [per_kv_head_rate](plots/per_kv_head_rate_test_intermediate_algebra_1454_json.png) · [head_correlation](plots/head_correlation_test_intermediate_algebra_1454_json.png) · [n_drifted_hist](plots/n_drifted_hist_test_intermediate_algebra_1454_json.png)


## Problem `test_intermediate_algebra_1994_json`

- Decode steps logged: **2,914** (of 4,834 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **18** (0.3723% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9969 | 0.9999 | 0.9632 | 0.9221 | 0.8958 |
| 0.90 | 0.8701 | 0.9799 | 0.6257 | 0.5687 | 0.5427 |
| 0.85 | 0.6155 | 0.8679 | 0.2429 | 0.2744 | 0.2764 |
| 0.80 | 0.3456 | 0.6830 | 0.0995 | 0.1264 | 0.1396 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.1299 |
| 1 | 0.0985 |
| 2 | 0.0800 |
| 3 | 0.0905 |
| 4 | 0.0731 |
| 5 | 0.0685 |
| 6 | 0.0982 |
| 7 | 0.0950 |
| 8 | 0.2663 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.5961 |
| 1 | 0.5644 |
| 2 | 0.5502 |
| 3 | 0.5501 |
| 4 | 0.5583 |
| 5 | 0.5906 |
| 6 | 0.5887 |
| 7 | 0.5513 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.873 |  | 8 | 0.925 |  | 16 | 0.939 |  | 24 | 0.956 |
| 1 | 0.313 |  | 9 | 0.904 |  | 17 | 0.798 |  | 25 | 0.957 |
| 2 | 1.000 |  | 10 | 0.996 |  | 18 | 0.751 |  | 26 | 0.954 |
| 3 | 0.463 |  | 11 | 0.953 |  | 19 | 0.860 |  | 27 | 0.988 |
| 4 | 0.882 |  | 12 | 0.548 |  | 20 | 0.886 |  | 28 | 0.977 |
| 5 | 0.923 |  | 13 | 0.820 |  | 21 | 0.829 |  | 29 | 0.998 |
| 6 | 0.927 |  | 14 | 0.984 |  | 22 | 0.860 |  | 30 | 0.971 |
| 7 | 0.838 |  | 15 | 0.938 |  | 23 | 0.870 |  | 31 | 0.965 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_test_intermediate_algebra_1994_json.png) · [per_kv_head_rate](plots/per_kv_head_rate_test_intermediate_algebra_1994_json.png) · [head_correlation](plots/head_correlation_test_intermediate_algebra_1994_json.png) · [n_drifted_hist](plots/n_drifted_hist_test_intermediate_algebra_1994_json.png)


## Problem `test_intermediate_algebra_428_json`

- Decode steps logged: **1,097** (of 3,099 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **10** (0.3226% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9983 | 1.0000 | 0.9703 | 0.9301 | 0.9037 |
| 0.90 | 0.9074 | 0.9897 | 0.6644 | 0.6067 | 0.5763 |
| 0.85 | 0.6805 | 0.9118 | 0.2979 | 0.3233 | 0.3199 |
| 0.80 | 0.4180 | 0.7500 | 0.1388 | 0.1645 | 0.1745 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0926 |
| 1 | 0.0982 |
| 2 | 0.0792 |
| 3 | 0.0944 |
| 4 | 0.0688 |
| 5 | 0.0685 |
| 6 | 0.0941 |
| 7 | 0.1019 |
| 8 | 0.3022 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6302 |
| 1 | 0.5913 |
| 2 | 0.6019 |
| 3 | 0.5757 |
| 4 | 0.6125 |
| 5 | 0.6262 |
| 6 | 0.6295 |
| 7 | 0.5864 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.905 |  | 8 | 0.975 |  | 16 | 0.989 |  | 24 | 0.977 |
| 1 | 0.236 |  | 9 | 0.942 |  | 17 | 0.834 |  | 25 | 0.990 |
| 2 | 1.000 |  | 10 | 1.000 |  | 18 | 0.828 |  | 26 | 0.988 |
| 3 | 0.494 |  | 11 | 0.977 |  | 19 | 0.928 |  | 27 | 0.992 |
| 4 | 0.952 |  | 12 | 0.650 |  | 20 | 0.955 |  | 28 | 0.992 |
| 5 | 0.970 |  | 13 | 0.851 |  | 21 | 0.930 |  | 29 | 1.000 |
| 6 | 0.975 |  | 14 | 0.994 |  | 22 | 0.924 |  | 30 | 0.972 |
| 7 | 0.933 |  | 15 | 0.973 |  | 23 | 0.950 |  | 31 | 0.961 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_test_intermediate_algebra_428_json.png) · [per_kv_head_rate](plots/per_kv_head_rate_test_intermediate_algebra_428_json.png) · [head_correlation](plots/head_correlation_test_intermediate_algebra_428_json.png) · [n_drifted_hist](plots/n_drifted_hist_test_intermediate_algebra_428_json.png)


## Problem `test_intermediate_algebra_607_json`

- Decode steps logged: **6,221** (of 8,222 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **44** (0.5351% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9966 | 0.9998 | 0.9651 | 0.9227 | 0.8949 |
| 0.90 | 0.8804 | 0.9818 | 0.6393 | 0.5932 | 0.5665 |
| 0.85 | 0.6497 | 0.8840 | 0.2980 | 0.3203 | 0.3169 |
| 0.80 | 0.3926 | 0.7174 | 0.1440 | 0.1652 | 0.1751 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.1196 |
| 1 | 0.0969 |
| 2 | 0.0787 |
| 3 | 0.0881 |
| 4 | 0.0623 |
| 5 | 0.0630 |
| 6 | 0.0894 |
| 7 | 0.0897 |
| 8 | 0.3123 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6151 |
| 1 | 0.5812 |
| 2 | 0.5837 |
| 3 | 0.5683 |
| 4 | 0.5993 |
| 5 | 0.6071 |
| 6 | 0.6105 |
| 7 | 0.5803 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.883 |  | 8 | 0.962 |  | 16 | 0.942 |  | 24 | 0.964 |
| 1 | 0.203 |  | 9 | 0.908 |  | 17 | 0.850 |  | 25 | 0.976 |
| 2 | 1.000 |  | 10 | 0.998 |  | 18 | 0.761 |  | 26 | 0.975 |
| 3 | 0.377 |  | 11 | 0.954 |  | 19 | 0.886 |  | 27 | 0.995 |
| 4 | 0.940 |  | 12 | 0.620 |  | 20 | 0.915 |  | 28 | 0.990 |
| 5 | 0.939 |  | 13 | 0.815 |  | 21 | 0.861 |  | 29 | 0.999 |
| 6 | 0.962 |  | 14 | 0.981 |  | 22 | 0.878 |  | 30 | 0.968 |
| 7 | 0.896 |  | 15 | 0.941 |  | 23 | 0.883 |  | 31 | 0.949 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_test_intermediate_algebra_607_json.png) · [per_kv_head_rate](plots/per_kv_head_rate_test_intermediate_algebra_607_json.png) · [head_correlation](plots/head_correlation_test_intermediate_algebra_607_json.png) · [n_drifted_hist](plots/n_drifted_hist_test_intermediate_algebra_607_json.png)


## Problem `test_number_theory_1032_json`

- Decode steps logged: **14,377** (of 16,383 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **211** (1.2878% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9988 | 1.0000 | 0.9695 | 0.9235 | 0.8905 |
| 0.90 | 0.8907 | 0.9886 | 0.5927 | 0.5430 | 0.5146 |
| 0.85 | 0.5971 | 0.8949 | 0.1988 | 0.2450 | 0.2540 |
| 0.80 | 0.3006 | 0.6795 | 0.0963 | 0.1171 | 0.1316 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.1093 |
| 1 | 0.1277 |
| 2 | 0.0930 |
| 3 | 0.1009 |
| 4 | 0.0827 |
| 5 | 0.0707 |
| 6 | 0.0899 |
| 7 | 0.1016 |
| 8 | 0.2241 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.5762 |
| 1 | 0.5281 |
| 2 | 0.5263 |
| 3 | 0.5063 |
| 4 | 0.5377 |
| 5 | 0.5784 |
| 6 | 0.5455 |
| 7 | 0.5458 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.851 |  | 8 | 0.955 |  | 16 | 0.980 |  | 24 | 0.981 |
| 1 | 0.267 |  | 9 | 0.908 |  | 17 | 0.809 |  | 25 | 0.998 |
| 2 | 0.989 |  | 10 | 1.000 |  | 18 | 0.701 |  | 26 | 0.967 |
| 3 | 0.346 |  | 11 | 0.927 |  | 19 | 0.989 |  | 27 | 0.990 |
| 4 | 0.937 |  | 12 | 0.607 |  | 20 | 0.986 |  | 28 | 0.987 |
| 5 | 0.966 |  | 13 | 0.765 |  | 21 | 0.961 |  | 29 | 1.000 |
| 6 | 0.956 |  | 14 | 0.991 |  | 22 | 0.930 |  | 30 | 0.971 |
| 7 | 0.866 |  | 15 | 0.985 |  | 23 | 0.967 |  | 31 | 0.969 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_test_number_theory_1032_json.png) · [per_kv_head_rate](plots/per_kv_head_rate_test_number_theory_1032_json.png) · [head_correlation](plots/head_correlation_test_number_theory_1032_json.png) · [n_drifted_hist](plots/n_drifted_hist_test_number_theory_1032_json.png)


## Problem `test_number_theory_515_json`

- Decode steps logged: **1,412** (of 3,423 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **21** (0.6133% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9979 | 0.9999 | 0.9662 | 0.9230 | 0.8921 |
| 0.90 | 0.8885 | 0.9850 | 0.6343 | 0.5775 | 0.5496 |
| 0.85 | 0.6491 | 0.8955 | 0.2689 | 0.2956 | 0.2933 |
| 0.80 | 0.3859 | 0.7227 | 0.1128 | 0.1413 | 0.1535 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.1115 |
| 1 | 0.1032 |
| 2 | 0.0812 |
| 3 | 0.0995 |
| 4 | 0.0707 |
| 5 | 0.0704 |
| 6 | 0.0926 |
| 7 | 0.1012 |
| 8 | 0.2697 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6039 |
| 1 | 0.5583 |
| 2 | 0.5690 |
| 3 | 0.5402 |
| 4 | 0.5833 |
| 5 | 0.5960 |
| 6 | 0.5986 |
| 7 | 0.5711 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.837 |  | 8 | 0.969 |  | 16 | 0.968 |  | 24 | 0.970 |
| 1 | 0.280 |  | 9 | 0.917 |  | 17 | 0.841 |  | 25 | 0.977 |
| 2 | 0.995 |  | 10 | 1.000 |  | 18 | 0.748 |  | 26 | 0.969 |
| 3 | 0.407 |  | 11 | 0.967 |  | 19 | 0.935 |  | 27 | 0.979 |
| 4 | 0.922 |  | 12 | 0.594 |  | 20 | 0.944 |  | 28 | 0.984 |
| 5 | 0.939 |  | 13 | 0.856 |  | 21 | 0.907 |  | 29 | 0.999 |
| 6 | 0.948 |  | 14 | 0.996 |  | 22 | 0.928 |  | 30 | 0.949 |
| 7 | 0.895 |  | 15 | 0.965 |  | 23 | 0.919 |  | 31 | 0.928 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_test_number_theory_515_json.png) · [per_kv_head_rate](plots/per_kv_head_rate_test_number_theory_515_json.png) · [head_correlation](plots/head_correlation_test_number_theory_515_json.png) · [n_drifted_hist](plots/n_drifted_hist_test_number_theory_515_json.png)


## Problem `test_number_theory_627_json`

- Decode steps logged: **2,322** (of 4,311 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **29** (0.6725% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9978 | 1.0000 | 0.9697 | 0.9312 | 0.9056 |
| 0.90 | 0.9065 | 0.9858 | 0.6926 | 0.6362 | 0.6057 |
| 0.85 | 0.7042 | 0.9131 | 0.3427 | 0.3610 | 0.3545 |
| 0.80 | 0.4514 | 0.7631 | 0.1763 | 0.1985 | 0.2039 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0935 |
| 1 | 0.0886 |
| 2 | 0.0709 |
| 3 | 0.0795 |
| 4 | 0.0632 |
| 5 | 0.0603 |
| 6 | 0.0933 |
| 7 | 0.0990 |
| 8 | 0.3517 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6552 |
| 1 | 0.6224 |
| 2 | 0.6256 |
| 3 | 0.6041 |
| 4 | 0.6437 |
| 5 | 0.6556 |
| 6 | 0.6570 |
| 7 | 0.6263 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.882 |  | 8 | 0.982 |  | 16 | 0.975 |  | 24 | 0.973 |
| 1 | 0.268 |  | 9 | 0.940 |  | 17 | 0.887 |  | 25 | 0.981 |
| 2 | 0.980 |  | 10 | 0.999 |  | 18 | 0.831 |  | 26 | 0.980 |
| 3 | 0.464 |  | 11 | 0.969 |  | 19 | 0.924 |  | 27 | 0.994 |
| 4 | 0.953 |  | 12 | 0.671 |  | 20 | 0.945 |  | 28 | 0.992 |
| 5 | 0.966 |  | 13 | 0.876 |  | 21 | 0.907 |  | 29 | 1.000 |
| 6 | 0.967 |  | 14 | 0.996 |  | 22 | 0.932 |  | 30 | 0.967 |
| 7 | 0.926 |  | 15 | 0.968 |  | 23 | 0.935 |  | 31 | 0.978 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_test_number_theory_627_json.png) · [per_kv_head_rate](plots/per_kv_head_rate_test_number_theory_627_json.png) · [head_correlation](plots/head_correlation_test_number_theory_627_json.png) · [n_drifted_hist](plots/n_drifted_hist_test_number_theory_627_json.png)


## Problem `test_number_theory_737_json`

- Decode steps logged: **724** (of 2,686 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **23** (0.8560% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9988 | 1.0000 | 0.9748 | 0.9369 | 0.9088 |
| 0.90 | 0.9097 | 0.9904 | 0.6719 | 0.6153 | 0.5847 |
| 0.85 | 0.6799 | 0.9129 | 0.3048 | 0.3260 | 0.3203 |
| 0.80 | 0.4144 | 0.7456 | 0.1350 | 0.1595 | 0.1692 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0903 |
| 1 | 0.0992 |
| 2 | 0.0774 |
| 3 | 0.0861 |
| 4 | 0.0666 |
| 5 | 0.0684 |
| 6 | 0.0925 |
| 7 | 0.1093 |
| 8 | 0.3101 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6377 |
| 1 | 0.5898 |
| 2 | 0.6103 |
| 3 | 0.5818 |
| 4 | 0.6224 |
| 5 | 0.6361 |
| 6 | 0.6341 |
| 7 | 0.6099 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.884 |  | 8 | 0.965 |  | 16 | 0.990 |  | 24 | 0.986 |
| 1 | 0.296 |  | 9 | 0.919 |  | 17 | 0.874 |  | 25 | 0.994 |
| 2 | 1.000 |  | 10 | 0.997 |  | 18 | 0.826 |  | 26 | 0.989 |
| 3 | 0.459 |  | 11 | 0.977 |  | 19 | 0.957 |  | 27 | 0.996 |
| 4 | 0.936 |  | 12 | 0.648 |  | 20 | 0.968 |  | 28 | 0.994 |
| 5 | 0.957 |  | 13 | 0.859 |  | 21 | 0.952 |  | 29 | 0.999 |
| 6 | 0.949 |  | 14 | 0.988 |  | 22 | 0.956 |  | 30 | 0.982 |
| 7 | 0.896 |  | 15 | 0.974 |  | 23 | 0.968 |  | 31 | 0.974 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_test_number_theory_737_json.png) · [per_kv_head_rate](plots/per_kv_head_rate_test_number_theory_737_json.png) · [head_correlation](plots/head_correlation_test_number_theory_737_json.png) · [n_drifted_hist](plots/n_drifted_hist_test_number_theory_737_json.png)


## Problem `test_number_theory_864_json`

- Decode steps logged: **14,376** (of 16,383 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **28** (0.1709% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9999 | 1.0000 | 0.9838 | 0.9496 | 0.9166 |
| 0.90 | 0.9141 | 0.9954 | 0.7026 | 0.6430 | 0.6130 |
| 0.85 | 0.7079 | 0.9197 | 0.3517 | 0.3664 | 0.3559 |
| 0.80 | 0.4632 | 0.7878 | 0.1541 | 0.1901 | 0.2008 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0859 |
| 1 | 0.0886 |
| 2 | 0.0761 |
| 3 | 0.0698 |
| 4 | 0.0714 |
| 5 | 0.0734 |
| 6 | 0.0808 |
| 7 | 0.0758 |
| 8 | 0.3782 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6702 |
| 1 | 0.6313 |
| 2 | 0.6130 |
| 3 | 0.6309 |
| 4 | 0.6471 |
| 5 | 0.6898 |
| 6 | 0.6440 |
| 7 | 0.6175 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.922 |  | 8 | 0.911 |  | 16 | 0.999 |  | 24 | 0.998 |
| 1 | 0.407 |  | 9 | 0.985 |  | 17 | 0.911 |  | 25 | 0.999 |
| 2 | 1.000 |  | 10 | 1.000 |  | 18 | 0.863 |  | 26 | 0.999 |
| 3 | 0.225 |  | 11 | 0.997 |  | 19 | 0.998 |  | 27 | 0.999 |
| 4 | 0.921 |  | 12 | 0.610 |  | 20 | 0.998 |  | 28 | 0.999 |
| 5 | 0.928 |  | 13 | 0.953 |  | 21 | 0.997 |  | 29 | 1.000 |
| 6 | 0.918 |  | 14 | 0.999 |  | 22 | 0.995 |  | 30 | 0.996 |
| 7 | 0.771 |  | 15 | 1.000 |  | 23 | 0.957 |  | 31 | 0.998 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_test_number_theory_864_json.png) · [per_kv_head_rate](plots/per_kv_head_rate_test_number_theory_864_json.png) · [head_correlation](plots/head_correlation_test_number_theory_864_json.png) · [n_drifted_hist](plots/n_drifted_hist_test_number_theory_864_json.png)


## Problem `test_prealgebra_1139_json`

- Decode steps logged: **14,494** (of 16,383 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **910** (5.5542% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9991 | 1.0000 | 0.9780 | 0.9350 | 0.9017 |
| 0.90 | 0.9061 | 0.9895 | 0.6175 | 0.5616 | 0.5280 |
| 0.85 | 0.6288 | 0.9113 | 0.2013 | 0.2429 | 0.2507 |
| 0.80 | 0.3283 | 0.7183 | 0.0675 | 0.1031 | 0.1206 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.0939 |
| 1 | 0.1175 |
| 2 | 0.0906 |
| 3 | 0.1078 |
| 4 | 0.0816 |
| 5 | 0.0746 |
| 6 | 0.0945 |
| 7 | 0.1123 |
| 8 | 0.2273 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.5955 |
| 1 | 0.5427 |
| 2 | 0.5446 |
| 3 | 0.5309 |
| 4 | 0.5510 |
| 5 | 0.5937 |
| 6 | 0.5743 |
| 7 | 0.5600 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.889 |  | 8 | 0.950 |  | 16 | 0.998 |  | 24 | 0.998 |
| 1 | 0.339 |  | 9 | 0.890 |  | 17 | 0.867 |  | 25 | 1.000 |
| 2 | 1.000 |  | 10 | 1.000 |  | 18 | 0.794 |  | 26 | 0.988 |
| 3 | 0.439 |  | 11 | 0.933 |  | 19 | 0.969 |  | 27 | 0.995 |
| 4 | 0.977 |  | 12 | 0.563 |  | 20 | 0.995 |  | 28 | 0.990 |
| 5 | 0.984 |  | 13 | 0.742 |  | 21 | 0.972 |  | 29 | 1.000 |
| 6 | 0.990 |  | 14 | 0.997 |  | 22 | 0.964 |  | 30 | 0.973 |
| 7 | 0.870 |  | 15 | 0.997 |  | 23 | 0.962 |  | 31 | 0.969 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_test_prealgebra_1139_json.png) · [per_kv_head_rate](plots/per_kv_head_rate_test_prealgebra_1139_json.png) · [head_correlation](plots/head_correlation_test_prealgebra_1139_json.png) · [n_drifted_hist](plots/n_drifted_hist_test_prealgebra_1139_json.png)


## Problem `test_precalculus_1199_json`

- Decode steps logged: **1,722** (of 3,545 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **26** (0.7332% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9979 | 0.9999 | 0.9718 | 0.9339 | 0.9022 |
| 0.90 | 0.8950 | 0.9882 | 0.6478 | 0.5881 | 0.5565 |
| 0.85 | 0.6453 | 0.9046 | 0.2476 | 0.2803 | 0.2814 |
| 0.80 | 0.3565 | 0.7212 | 0.0906 | 0.1230 | 0.1392 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.1050 |
| 1 | 0.0969 |
| 2 | 0.0763 |
| 3 | 0.0960 |
| 4 | 0.0725 |
| 5 | 0.0762 |
| 6 | 0.1020 |
| 7 | 0.1159 |
| 8 | 0.2591 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6205 |
| 1 | 0.5698 |
| 2 | 0.5827 |
| 3 | 0.5420 |
| 4 | 0.5897 |
| 5 | 0.6034 |
| 6 | 0.6128 |
| 7 | 0.5841 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.782 |  | 8 | 0.967 |  | 16 | 0.976 |  | 24 | 0.972 |
| 1 | 0.302 |  | 9 | 0.913 |  | 17 | 0.859 |  | 25 | 0.985 |
| 2 | 0.999 |  | 10 | 0.999 |  | 18 | 0.790 |  | 26 | 0.988 |
| 3 | 0.447 |  | 11 | 0.965 |  | 19 | 0.942 |  | 27 | 0.985 |
| 4 | 0.923 |  | 12 | 0.671 |  | 20 | 0.960 |  | 28 | 0.991 |
| 5 | 0.911 |  | 13 | 0.859 |  | 21 | 0.920 |  | 29 | 0.999 |
| 6 | 0.940 |  | 14 | 0.988 |  | 22 | 0.925 |  | 30 | 0.954 |
| 7 | 0.886 |  | 15 | 0.968 |  | 23 | 0.922 |  | 31 | 0.950 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_test_precalculus_1199_json.png) · [per_kv_head_rate](plots/per_kv_head_rate_test_precalculus_1199_json.png) · [head_correlation](plots/head_correlation_test_precalculus_1199_json.png) · [n_drifted_hist](plots/n_drifted_hist_test_precalculus_1199_json.png)


## Problem `test_precalculus_285_json`

- Decode steps logged: **3,357** (of 5,343 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **13** (0.2433% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9966 | 0.9999 | 0.9598 | 0.9185 | 0.8909 |
| 0.90 | 0.8731 | 0.9791 | 0.6280 | 0.5808 | 0.5544 |
| 0.85 | 0.6286 | 0.8806 | 0.2729 | 0.2991 | 0.2989 |
| 0.80 | 0.3636 | 0.6956 | 0.1290 | 0.1499 | 0.1604 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.1269 |
| 1 | 0.0959 |
| 2 | 0.0774 |
| 3 | 0.0910 |
| 4 | 0.0671 |
| 5 | 0.0659 |
| 6 | 0.0908 |
| 7 | 0.0998 |
| 8 | 0.2852 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.6072 |
| 1 | 0.5662 |
| 2 | 0.5764 |
| 3 | 0.5500 |
| 4 | 0.5833 |
| 5 | 0.5920 |
| 6 | 0.6023 |
| 7 | 0.5693 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.831 |  | 8 | 0.944 |  | 16 | 0.950 |  | 24 | 0.955 |
| 1 | 0.216 |  | 9 | 0.905 |  | 17 | 0.824 |  | 25 | 0.971 |
| 2 | 0.999 |  | 10 | 0.996 |  | 18 | 0.763 |  | 26 | 0.980 |
| 3 | 0.394 |  | 11 | 0.954 |  | 19 | 0.886 |  | 27 | 0.993 |
| 4 | 0.867 |  | 12 | 0.633 |  | 20 | 0.909 |  | 28 | 0.988 |
| 5 | 0.895 |  | 13 | 0.840 |  | 21 | 0.859 |  | 29 | 1.000 |
| 6 | 0.932 |  | 14 | 0.985 |  | 22 | 0.884 |  | 30 | 0.970 |
| 7 | 0.855 |  | 15 | 0.942 |  | 23 | 0.874 |  | 31 | 0.948 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_test_precalculus_285_json.png) · [per_kv_head_rate](plots/per_kv_head_rate_test_precalculus_285_json.png) · [head_correlation](plots/head_correlation_test_precalculus_285_json.png) · [n_drifted_hist](plots/n_drifted_hist_test_precalculus_285_json.png)


## Problem `test_precalculus_927_json`

- Decode steps logged: **333** (of 2,297 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **8** (0.3481% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9933 | 0.9997 | 0.9418 | 0.8942 | 0.8649 |
| 0.90 | 0.8375 | 0.9655 | 0.5538 | 0.5231 | 0.5053 |
| 0.85 | 0.5652 | 0.8358 | 0.2407 | 0.2678 | 0.2688 |
| 0.80 | 0.3175 | 0.6241 | 0.1243 | 0.1335 | 0.1419 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.1625 |
| 1 | 0.1242 |
| 2 | 0.0872 |
| 3 | 0.0904 |
| 4 | 0.0639 |
| 5 | 0.0609 |
| 6 | 0.0795 |
| 7 | 0.0732 |
| 8 | 0.2582 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.5377 |
| 1 | 0.5074 |
| 2 | 0.5109 |
| 3 | 0.5009 |
| 4 | 0.5241 |
| 5 | 0.5358 |
| 6 | 0.5451 |
| 7 | 0.5225 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.796 |  | 8 | 0.937 |  | 16 | 0.928 |  | 24 | 0.952 |
| 1 | 0.156 |  | 9 | 0.865 |  | 17 | 0.685 |  | 25 | 0.964 |
| 2 | 0.997 |  | 10 | 0.991 |  | 18 | 0.679 |  | 26 | 0.976 |
| 3 | 0.330 |  | 11 | 0.934 |  | 19 | 0.781 |  | 27 | 0.994 |
| 4 | 0.916 |  | 12 | 0.529 |  | 20 | 0.847 |  | 28 | 0.985 |
| 5 | 0.934 |  | 13 | 0.703 |  | 21 | 0.754 |  | 29 | 0.994 |
| 6 | 0.931 |  | 14 | 0.985 |  | 22 | 0.814 |  | 30 | 0.946 |
| 7 | 0.871 |  | 15 | 0.898 |  | 23 | 0.796 |  | 31 | 0.934 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_test_precalculus_927_json.png) · [per_kv_head_rate](plots/per_kv_head_rate_test_precalculus_927_json.png) · [head_correlation](plots/head_correlation_test_precalculus_927_json.png) · [n_drifted_hist](plots/n_drifted_hist_test_precalculus_927_json.png)


## Problem `test_precalculus_990_json`

- Decode steps logged: **3,592** (of 5,445 total; earlier steps are before budget was exceeded)
- Transition-keyword tokens: **13** (0.2387% of generated tokens)

**Rates of `sim < τ` under five aggregations:**

| τ | any kv-group (= trigger) | any q-head | mean q-heads | single kv-head | single q-head |
|---|---|---|---|---|---|
| 0.95 | 0.9941 | 0.9996 | 0.9497 | 0.9068 | 0.8798 |
| 0.90 | 0.8601 | 0.9720 | 0.6098 | 0.5716 | 0.5479 |
| 0.85 | 0.6149 | 0.8618 | 0.2818 | 0.3059 | 0.3058 |
| 0.80 | 0.3655 | 0.6851 | 0.1434 | 0.1617 | 0.1713 |

**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**

| drifted | fraction |
|---|---|
| 0 | 0.1399 |
| 1 | 0.1027 |
| 2 | 0.0795 |
| 3 | 0.0828 |
| 4 | 0.0647 |
| 5 | 0.0605 |
| 6 | 0.0845 |
| 7 | 0.0890 |
| 8 | 0.2964 |

**Per-kv-head firing rate (τ=0.9):**

| kv_head | P(sim < 0.9) |
|---|---|
| 0 | 0.5938 |
| 1 | 0.5585 |
| 2 | 0.5682 |
| 3 | 0.5411 |
| 4 | 0.5776 |
| 5 | 0.5819 |
| 6 | 0.5936 |
| 7 | 0.5578 |

**Per-layer `need_corr` rate (τ=0.9):**

| layer | rate | | layer | rate | | layer | rate | | layer | rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.846 |  | 8 | 0.923 |  | 16 | 0.933 |  | 24 | 0.929 |
| 1 | 0.183 |  | 9 | 0.881 |  | 17 | 0.806 |  | 25 | 0.953 |
| 2 | 0.997 |  | 10 | 0.992 |  | 18 | 0.743 |  | 26 | 0.971 |
| 3 | 0.397 |  | 11 | 0.938 |  | 19 | 0.840 |  | 27 | 0.993 |
| 4 | 0.922 |  | 12 | 0.603 |  | 20 | 0.879 |  | 28 | 0.987 |
| 5 | 0.896 |  | 13 | 0.764 |  | 21 | 0.834 |  | 29 | 0.998 |
| 6 | 0.949 |  | 14 | 0.970 |  | 22 | 0.845 |  | 30 | 0.958 |
| 7 | 0.865 |  | 15 | 0.917 |  | 23 | 0.849 |  | 31 | 0.962 |

Plots: [per_layer_meanmin](plots/per_layer_meanmin_test_precalculus_990_json.png) · [per_kv_head_rate](plots/per_kv_head_rate_test_precalculus_990_json.png) · [head_correlation](plots/head_correlation_test_precalculus_990_json.png) · [n_drifted_hist](plots/n_drifted_hist_test_precalculus_990_json.png)
