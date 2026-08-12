# Results — phase: clean

Quantile grid: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] → intervals below are 80%. **Decile grid, not the pre-registered one.** TiRex-2 emits only these levels, so every model here was recomputed on them; `mean_var` is comparable within this report but not against `results_*.md`. config.yaml is unchanged.

## h=1 losses (mean per day)

| model | n | QLIKE | CRPS | pin0.10 | pin0.90 | 80% cov | p_uc | p_ind |
|---|---|---|---|---|---|---|---|---|
| persistence | 192 | 0.5909 | 0.6825 | 0.1812 | 0.1850 | 0.781 | 0.521 | 0.052 |
| ewma | 192 | 0.4036 | - | - | - | - | - | - |
| har | 192 | 0.3860 | 0.5526 | 0.1585 | 0.1552 | 0.750 | 0.092 | 0.112 |
| har_x | 192 | 0.3734 | 0.5409 | 0.1573 | 0.1521 | 0.750 | 0.092 | 0.225 |
| har_iv | 192 | 0.3167 | 0.5075 | 0.1402 | 0.1427 | 0.812 | 0.662 | 0.848 |
| har_sv | 188 | 0.3593 | 0.5158 | 0.1458 | 0.1415 | 0.830 | 0.298 | 0.202 |
| har_ic | 192 | 0.3229 | 0.5120 | 0.1409 | 0.1429 | 0.792 | 0.774 | 0.716 |
| har_iv_x | 192 | 0.3073 | 0.4949 | 0.1385 | 0.1396 | 0.807 | 0.800 | 0.635 |
| chronos_uni | 192 | 0.3924 | 0.5499 | 0.1499 | 0.1518 | 0.812 | 0.662 | 0.143 |
| chronos_cov | 192 | 0.3851 | 0.5498 | 0.1540 | 0.1522 | 0.760 | 0.180 | 0.453 |
| chronos_cov_iv | 192 | 0.3173 | 0.5036 | 0.1350 | 0.1456 | 0.812 | 0.662 | 0.773 |
| tirex_uni | 192 | 0.3975 | 0.5522 | 0.1510 | 0.1528 | 0.786 | 0.642 | 0.181 |
| tirex_cov | 192 | 0.3885 | 0.5572 | 0.1518 | 0.1538 | 0.771 | 0.321 | 0.252 |
| tirex_cov_iv | 192 | 0.3878 | 0.5518 | 0.1518 | 0.1537 | 0.797 | 0.914 | 0.083 |
| tirex_cov_ivf | 192 | 0.3296 | 0.4929 | 0.1291 | 0.1483 | 0.740 | 0.043 | 0.948 |

## Diebold-Mariano vs HAR (QLIKE; negative = beats HAR)

| model | DM | p | n |
|---|---|---|---|
| persistence | 3.191 | 0.0017 | 192 |
| ewma | 0.538 | 0.5910 | 192 |
| har_x | -1.404 | 0.1621 | 192 |
| har_iv | -3.573 | 0.0004 | 192 |
| har_sv | -1.072 | 0.2850 | 188 |
| har_ic | -3.140 | 0.0020 | 192 |
| har_iv_x | -3.718 | 0.0003 | 192 |
| chronos_uni | 0.619 | 0.5367 | 192 |
| chronos_cov | -0.045 | 0.9639 | 192 |
| chronos_cov_iv | -3.229 | 0.0015 | 192 |
| tirex_uni | 0.950 | 0.3431 | 192 |
| tirex_cov | 0.196 | 0.8445 | 192 |
| tirex_cov_iv | 0.152 | 0.8791 | 192 |
| tirex_cov_ivf | -2.823 | 0.0053 | 192 |

## Diebold-Mariano vs HAR-IV — same information set

HAR-IV is log-HAR plus log(VXN). Any model fed VXN beats plain HAR trivially, because implied vol predicts realized vol. The question this table answers is whether the foundation model extracts more from VXN than four OLS terms do.

| model | DM | p | n |
|---|---|---|---|
| har_ic | 1.647 | 0.1012 | 192 |
| har_iv_x | -1.180 | 0.2394 | 192 |
| chronos_cov_iv | 0.034 | 0.9732 | 192 |
| tirex_cov_iv | 3.629 | 0.0004 | 192 |
| tirex_cov_ivf | 0.839 | 0.4024 | 192 |

## TiRex-2 robustness: origins after publication (2026-07-01)

The date-based leakage rule applied literally to TiRex-2. Tiny sample by construction — this is a sign check against the full window above, not a test. See reports/LEAKAGE_TIREX2.md.

| model | QLIKE (full) | QLIKE (post-pub) | n |
|---|---|---|---|
| tirex_uni | 0.3975 | 0.2409 | 28 |
| tirex_cov | 0.3885 | 0.2287 | 28 |
| tirex_cov_iv | 0.3878 | 0.2303 | 28 |
| tirex_cov_ivf | 0.3296 | 0.1882 | 28 |
| har | 0.3860 | 0.2334 | 28 |
| har_iv | 0.3167 | 0.1991 | 28 |
| chronos_cov | 0.3851 | 0.2435 | 28 |

## Event-sliced QLIKE (mean)

| model | FOMC (n) | CPI (n) | heavy-earnings (n) | quiet (n) |
|---|---|---|---|---|
| persistence | 0.2428 (6) | 0.8017 (8) | 0.7887 (12) | 0.5790 (166) |
| ewma | 0.2360 (6) | 0.2623 (8) | 0.6117 (12) | 0.4014 (166) |
| har | 0.1711 (6) | 0.4041 (8) | 0.5113 (12) | 0.3839 (166) |
| har_x | 0.2340 (6) | 0.3599 (8) | 0.2837 (12) | 0.3856 (166) |
| har_iv | 0.1694 (6) | 0.3070 (8) | 0.3975 (12) | 0.3166 (166) |
| har_sv | 0.2136 (6) | 0.3344 (8) | 0.5314 (12) | 0.3531 (162) |
| har_ic | 0.1713 (6) | 0.3162 (8) | 0.4189 (12) | 0.3218 (166) |
| har_iv_x | 0.2264 (6) | 0.2933 (8) | 0.2089 (12) | 0.3180 (166) |
| chronos_uni | 0.1421 (6) | 0.4116 (8) | 0.5111 (12) | 0.3919 (166) |
| chronos_cov | 0.2152 (6) | 0.4048 (8) | 0.2001 (12) | 0.4036 (166) |
| chronos_cov_iv | 0.2391 (6) | 0.3271 (8) | 0.1507 (12) | 0.3317 (166) |
| tirex_uni | 0.1279 (6) | 0.3987 (8) | 0.5794 (12) | 0.3941 (166) |
| tirex_cov | 0.1665 (6) | 0.4532 (8) | 0.4576 (12) | 0.3884 (166) |
| tirex_cov_iv | 0.1435 (6) | 0.4187 (8) | 0.5120 (12) | 0.3861 (166) |
| tirex_cov_ivf | 0.1347 (6) | 0.2751 (8) | 0.3581 (12) | 0.3372 (166) |

## Full-sample paired per-origin (all origins in phase)

Win rate and mean difference answer different questions. A high win rate with no mean gain is many small wins funded by rare large losses; a low win rate with a mean gain is the reverse. Read both before believing either.

| pair | n | wins | win % | sign p | mean | median | top-10 share of mean gap |
|---|---|---|---|---|---|---|---|
| har_iv_x vs har_iv | 192 | 117 | 60.9% | 0.00299 | 0.3073 vs 0.3167 | 0.1070 vs 0.1189 | 178% |
| har_x vs har | 192 | 113 | 58.9% | 0.017 | 0.3734 vs 0.3860 | 0.1370 vs 0.1389 | 159% |
| har_sv vs har | 188 | 102 | 54.3% | 0.274 | 0.3593 vs 0.3837 | 0.1277 vs 0.1389 | 149% |
| har_iv vs har | 192 | 115 | 59.9% | 0.00742 | 0.3167 vs 0.3860 | 0.1189 vs 0.1389 | 67% |
| chronos_cov vs chronos_uni | 192 | 98 | 51.0% | 0.829 | 0.3851 vs 0.3924 | 0.1517 vs 0.1528 | 481% |
| tirex_cov vs tirex_uni | 192 | 86 | 44.8% | 0.17 | 0.3885 vs 0.3975 | 0.1558 vs 0.1526 | 230% |

### Heavy-earnings slice, paired per-origin (cutoff 5.0% of index weight)

| pair | mean | median | better/n | sign p | top day | % of gap | ex-top |
|---|---|---|---|---|---|---|---|
| har_x vs har | 0.2837 vs 0.5113 | 0.0650 vs 0.2121 | 9/12 | 0.1460 | 2025-11-19 | 40% | 8/11 |
| har_iv_x vs har_iv | 0.2089 vs 0.3975 | 0.0785 vs 0.1947 | 9/12 | 0.1460 | 2025-11-19 | 36% | 8/11 |
| chronos_cov vs chronos_uni | 0.2001 vs 0.5111 | 0.0778 vs 0.2389 | 11/12 | 0.0063 | 2025-11-19 | 45% | 10/11 |
| tirex_cov vs tirex_uni | 0.4576 vs 0.5794 | 0.2602 vs 0.2999 | 8/12 | 0.3877 | 2025-11-19 | 69% | 7/11 |

## 30-calendar-day horizon vs VXN (log variance)

- VXN MZ: alpha=-1.301, beta=0.829 (p[beta=1]=0.284), R2=0.295, n=171
  - At the window's median VXN (24.1), fitted realized variance is 68% of implied — 82% in vol terms, about 4.2 vol points of premium. Read the premium here, not off alpha: with beta=0.83 the intercept is not interpretable alone.

Encompassing: realized = a + b*VXN + c*model. H3 wants c>0 and significant. A negative c is not evidence for the model — with VXN already in the regression it means the forecast enters against realized variance, which collinear forecasts commonly do. Models that consume VXN as an input are excluded: regressing on VXN and on a function of VXN is collinear by construction and the split of the coefficients is not interpretable.

- har_cum: MZ beta=0.293 R2=0.063 | encompassing b_implied=1.170 c_model=-0.360 (p=0.0923, R2=0.339, n=171)
- persistence_cum: MZ beta=0.142 R2=0.088 | encompassing b_implied=0.875 c_model=-0.024 (p=0.3889, R2=0.297, n=171)
- chronos_uni: MZ beta=0.399 R2=0.130 | encompassing b_implied=1.063 c_model=-0.214 (p=0.4028, R2=0.309, n=171)
- chronos_cov: MZ beta=0.513 R2=0.153 | encompassing b_implied=0.960 c_model=-0.143 (p=0.4275, R2=0.299, n=171)
- tirex_uni: MZ beta=0.567 R2=0.134 | encompassing b_implied=1.077 c_model=-0.313 (p=0.2898, R2=0.309, n=171)
- tirex_cov: MZ beta=0.431 R2=0.129 | encompassing b_implied=1.013 c_model=-0.187 (p=0.3484, R2=0.305, n=171)