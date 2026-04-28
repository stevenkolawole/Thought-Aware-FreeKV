# Drifted-group count vs cache turnover (full_aime)

Question: when correction fires, is the cache turnover concentrated in the drifted kv-groups, or spread across all 8?

Source: 28 problems with per-head sim cached.

Total correction events analyzed: **8,667,814**

Pearson correlation between `n_drifted` and `n_pages`: **0.4575**

OLS fit: `n_pages ≈ 4.63 × n_drifted + 28.36`


**Verdict: mixed.** Drifted-group count partially predicts turnover but there's significant baseline churn from non-drifted groups. Per-group suppression has some value, but the gain is smaller than the headline 80%-reuse number suggests.


## Bucket statistics

| n_drifted | n_events | events% | mean n_pages | median | std | p10 | p90 | mean per drifted-group |
|---|---|---|---|---|---|---|---|---|
| 0 | 18,845 | 0.22% | 36.3 | 37 | 11.9 | 21 | 50 | inf |
| 1 | 943,508 | 10.89% | 36.5 | 37 | 12.1 | 21 | 51 | 36.5 |
| 2 | 822,148 | 9.49% | 39.9 | 40 | 12.7 | 24 | 55 | 19.9 |
| 3 | 920,565 | 10.62% | 42.5 | 43 | 13.3 | 25 | 58 | 14.2 |
| 4 | 720,467 | 8.31% | 44.8 | 46 | 14.0 | 27 | 61 | 11.2 |
| 5 | 705,776 | 8.14% | 46.4 | 47 | 14.8 | 27 | 64 | 9.3 |
| 6 | 931,848 | 10.75% | 49.6 | 50 | 16.7 | 28 | 69 | 8.3 |
| 7 | 1,034,418 | 11.93% | 53.3 | 53 | 18.7 | 30 | 76 | 7.6 |
| 8 | 2,570,239 | 29.65% | 70.6 | 64 | 33.6 | 36 | 115 | 8.8 |

## Interpretation

- The *upper bound* if drifted groups contributed ALL the turnover: `n_pages = 32 × n_drifted`. Compare the empirical mean to `32x` in the bar chart.
- The *lower bound* if drifted groups contributed nothing: `n_pages` would be flat with respect to `n_drifted`. We can see how flat or steep the relationship is.
- Slope/intercept tell us: each additional drifted group raises turnover by ~4.6 pages, on top of a baseline of ~28.4 pages that doesn't depend on drift count.


## Plots

- [n_pages_vs_n_drifted_bar.png](plots/n_pages_vs_n_drifted_bar.png) — mean n_pages per bucket with OLS fit and 32×n_drifted upper bound
- [n_drifted_distribution.png](plots/n_drifted_distribution.png) — how many groups typically drift per event
- [n_pages_vs_n_drifted_heatmap.png](plots/n_pages_vs_n_drifted_heatmap.png) — joint distribution heatmap