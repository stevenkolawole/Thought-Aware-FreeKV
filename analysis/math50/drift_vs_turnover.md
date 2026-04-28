# Drifted-group count vs cache turnover (full_aime)

Question: when correction fires, is the cache turnover concentrated in the drifted kv-groups, or spread across all 8?

Source: 50 problems with per-head sim cached.

Total correction events analyzed: **4,642,645**

Pearson correlation between `n_drifted` and `n_pages`: **0.4225**

OLS fit: `n_pages ≈ 4.35 × n_drifted + 26.55`


**Verdict: mixed.** Drifted-group count partially predicts turnover but there's significant baseline churn from non-drifted groups. Per-group suppression has some value, but the gain is smaller than the headline 80%-reuse number suggests.


## Bucket statistics

| n_drifted | n_events | events% | mean n_pages | median | std | p10 | p90 | mean per drifted-group |
|---|---|---|---|---|---|---|---|---|
| 0 | 9,924 | 0.21% | 34.1 | 34 | 13.8 | 16 | 51 | inf |
| 1 | 521,070 | 11.22% | 34.3 | 35 | 13.3 | 16 | 51 | 34.3 |
| 2 | 426,688 | 9.19% | 37.2 | 38 | 14.1 | 18 | 54 | 18.6 |
| 3 | 477,191 | 10.28% | 39.7 | 41 | 14.8 | 19 | 57 | 13.2 |
| 4 | 381,946 | 8.23% | 42.3 | 43 | 15.8 | 21 | 61 | 10.6 |
| 5 | 371,059 | 7.99% | 43.8 | 45 | 16.5 | 21 | 64 | 8.8 |
| 6 | 487,274 | 10.50% | 46.2 | 47 | 18.3 | 22 | 68 | 7.7 |
| 7 | 544,012 | 11.72% | 49.7 | 50 | 20.3 | 23 | 74 | 7.1 |
| 8 | 1,423,481 | 30.66% | 66.2 | 61 | 34.0 | 28 | 112 | 8.3 |

## Interpretation

- The *upper bound* if drifted groups contributed ALL the turnover: `n_pages = 32 × n_drifted`. Compare the empirical mean to `32x` in the bar chart.
- The *lower bound* if drifted groups contributed nothing: `n_pages` would be flat with respect to `n_drifted`. We can see how flat or steep the relationship is.
- Slope/intercept tell us: each additional drifted group raises turnover by ~4.4 pages, on top of a baseline of ~26.6 pages that doesn't depend on drift count.


## Plots

- [n_pages_vs_n_drifted_bar.png](plots/n_pages_vs_n_drifted_bar.png) — mean n_pages per bucket with OLS fit and 32×n_drifted upper bound
- [n_drifted_distribution.png](plots/n_drifted_distribution.png) — how many groups typically drift per event
- [n_pages_vs_n_drifted_heatmap.png](plots/n_pages_vs_n_drifted_heatmap.png) — joint distribution heatmap