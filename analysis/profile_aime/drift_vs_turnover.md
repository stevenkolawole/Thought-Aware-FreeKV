# Drifted-group count vs cache turnover (full_aime)

Question: when correction fires, is the cache turnover concentrated in the drifted kv-groups, or spread across all 8?

Source: 5 problems with per-head sim cached.

Total correction events analyzed: **330,577**

Pearson correlation between `n_drifted` and `n_pages`: **0.2767**

OLS fit: `n_pages ≈ 2.05 × n_drifted + 21.47`


**Verdict: uniform — turnover is roughly the same regardless of how many groups drifted.** Per-group suppression buys little; the simpler design is per-step suppression (skip whole events when predicted reuse is high).


## Bucket statistics

| n_drifted | n_events | events% | mean n_pages | median | std | p10 | p90 | mean per drifted-group |
|---|---|---|---|---|---|---|---|---|
| 0 | 879 | 0.27% | 23.8 | 23 | 11.9 | 10 | 38 | inf |
| 1 | 40,017 | 12.11% | 24.4 | 24 | 11.9 | 10 | 40 | 24.4 |
| 2 | 35,299 | 10.68% | 27.3 | 27 | 13.3 | 10 | 45 | 13.6 |
| 3 | 38,304 | 11.59% | 27.8 | 27 | 13.1 | 11 | 45 | 9.3 |
| 4 | 29,689 | 8.98% | 28.5 | 28 | 13.6 | 11 | 45 | 7.1 |
| 5 | 27,076 | 8.19% | 29.4 | 29 | 15.6 | 11 | 47 | 5.9 |
| 6 | 35,765 | 10.82% | 30.2 | 29 | 15.6 | 11 | 49 | 5.0 |
| 7 | 39,435 | 11.93% | 33.4 | 32 | 18.5 | 12 | 56 | 4.8 |
| 8 | 84,113 | 25.44% | 40.4 | 37 | 25.0 | 13 | 72 | 5.0 |

## Interpretation

- The *upper bound* if drifted groups contributed ALL the turnover: `n_pages = 32 × n_drifted`. Compare the empirical mean to `32x` in the bar chart.
- The *lower bound* if drifted groups contributed nothing: `n_pages` would be flat with respect to `n_drifted`. We can see how flat or steep the relationship is.
- Slope/intercept tell us: each additional drifted group raises turnover by ~2.0 pages, on top of a baseline of ~21.5 pages that doesn't depend on drift count.


## Plots

- [n_pages_vs_n_drifted_bar.png](plots/n_pages_vs_n_drifted_bar.png) — mean n_pages per bucket with OLS fit and 32×n_drifted upper bound
- [n_drifted_distribution.png](plots/n_drifted_distribution.png) — how many groups typically drift per event
- [n_pages_vs_n_drifted_heatmap.png](plots/n_pages_vs_n_drifted_heatmap.png) — joint distribution heatmap