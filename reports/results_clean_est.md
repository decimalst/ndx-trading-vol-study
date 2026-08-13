# Results — phase: clean

Quantile grid: [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95] → intervals below are 90%.

**Corrected methodology run** — estimator=`smearing`, inference=`naive`. This report is a fork; `results_clean.md` still holds the frozen pre-registered numbers and is unchanged. Differences between the two are methodology, not data. See src/methodology.py for what each correction does and why.

`mean_var` here is the Duan smearing estimate `exp(mu)*mean(exp(resid))`. The frozen estimator integrates `exp(q)` over the quantile grid and divides by its mass, which discards the tails: it returns about 0.87 of the true conditional mean, and the discarded share depends on the grid — which is why every model's QLIKE differs between `results_clean.md` and `results_clean_dec.md` while EWMA's is 0.4036 in both. Only QLIKE changes; CRPS, pinball and coverage read off the quantiles and are identical.

Two caveats, stated rather than buried. (i) Across the HAR family the frozen estimator's bias spans only 0.866–0.873, so it is a near-common factor: correcting it moves QLIKE **levels** a lot and DM statistics little. It was never the reason one model outranked another. (ii) Rows marked `~` below have no recoverable residuals (Chronos-2, TiRex-2) and are reconstructed from their saved quantiles by tail extension, which on the HAR family lands 3.5% low with a 2.5pp spread across models — real, and a wider spread than the 0.70pp it replaces for those rows. Rows without `~` are exact.

## h=1 losses (mean per day)

| model | n | QLIKE | CRPS | pin0.05 | pin0.95 | 90% cov | p_uc | p_ind |
|---|---|---|---|---|---|---|---|---|
| persistence | 192 | 0.5429 | 0.6227 | 0.1070 | 0.1125 | 0.896 | 0.848 | 0.047 |
| ewma | 192 | 0.4036 | - | - | - | - | - | - |
| har | 192 | 0.3624 | 0.5066 | 0.0983 | 0.0915 | 0.849 | 0.027 | 0.386 |
| har_x | 192 | 0.3533 | 0.4956 | 0.0983 | 0.0891 | 0.859 | 0.075 | 0.218 |
| har_iv | 192 | 0.3118 | 0.4649 | 0.0828 | 0.0828 | 0.880 | 0.374 | 0.423 |
| har_sv | 188 | 0.3201 | 0.4711 | 0.0881 | 0.0839 | 0.926 | 0.224 | 0.959 |
| har_ic | 192 | 0.3158 | 0.4686 | 0.0824 | 0.0833 | 0.891 | 0.669 | 0.246 |
| har_iv_x | 192 | 0.3037 | 0.4541 | 0.0825 | 0.0832 | 0.891 | 0.669 | 0.246 |
| chronos_uni ~ | 192 | 0.3604 | 0.5026 | 0.0916 | 0.0877 | 0.906 | 0.771 | 0.528 |
| chronos_cov ~ | 192 | 0.3656 | 0.5030 | 0.0943 | 0.0906 | 0.880 | 0.374 | 0.158 |
| chronos_cov_iv ~ | 192 | 0.3185 | 0.4614 | 0.0816 | 0.0865 | 0.906 | 0.771 | 0.053 |

## Diebold-Mariano vs HAR (QLIKE; negative = beats HAR)

| model | DM | p | n |
|---|---|---|---|
| persistence | 4.017 | 0.0001 | 192 |
| ewma | 1.348 | 0.1792 | 192 |
| har_x | -1.301 | 0.1949 | 192 |
| har_iv | -3.168 | 0.0018 | 192 |
| har_sv | -2.163 | 0.0318 | 188 |
| har_ic | -2.804 | 0.0056 | 192 |
| har_iv_x | -3.440 | 0.0007 | 192 |
| chronos_uni | -0.206 | 0.8369 | 192 |
| chronos_cov | 0.174 | 0.8619 | 192 |
| chronos_cov_iv | -2.612 | 0.0097 | 192 |

## Diebold-Mariano vs HAR-IV — same information set

HAR-IV is log-HAR plus log(VXN). Any model fed VXN beats plain HAR trivially, because implied vol predicts realized vol. The question this table answers is whether the foundation model extracts more from VXN than four OLS terms do.

| model | DM | p | n |
|---|---|---|---|
| har_ic | 1.327 | 0.1861 | 192 |
| har_iv_x | -1.347 | 0.1796 | 192 |
| chronos_cov_iv | 0.429 | 0.6685 | 192 |

## Event-sliced QLIKE (mean)

| model | FOMC (n) | CPI (n) | heavy-earnings (n) | quiet (n) |
|---|---|---|---|---|
| persistence | 0.2710 (6) | 0.6744 (8) | 0.4925 (12) | 0.5500 (166) |
| ewma | 0.2360 (6) | 0.2623 (8) | 0.6117 (12) | 0.4014 (166) |
| har | 0.2298 (6) | 0.3671 (8) | 0.3423 (12) | 0.3684 (166) |
| har_x | 0.3193 (6) | 0.3706 (8) | 0.2053 (12) | 0.3644 (166) |
| har_iv | 0.2397 (6) | 0.3027 (8) | 0.2545 (12) | 0.3190 (166) |
| har_sv | 0.2688 (6) | 0.3187 (8) | 0.3201 (12) | 0.3221 (162) |
| har_ic | 0.2386 (6) | 0.3092 (8) | 0.2666 (12) | 0.3225 (166) |
| har_iv_x | 0.3197 (6) | 0.3317 (8) | 0.1588 (12) | 0.3122 (166) |
| chronos_uni | 0.1972 (6) | 0.3946 (8) | 0.3453 (12) | 0.3658 (166) |
| chronos_cov | 0.2923 (6) | 0.4071 (8) | 0.1374 (12) | 0.3828 (166) |
| chronos_cov_iv | 0.3261 (6) | 0.3748 (8) | 0.1183 (12) | 0.3299 (166) |

## Full-sample paired per-origin (all origins in phase)

Win rate and mean difference answer different questions. A high win rate with no mean gain is many small wins funded by rare large losses; a low win rate with a mean gain is the reverse. Read both before believing either.

| pair | n | wins | win % | sign p | mean | median | top-10 share of mean gap |
|---|---|---|---|---|---|---|---|
| har_iv_x vs har_iv | 192 | 125 | 65.1% | 3.43e-05 | 0.3037 vs 0.3118 | 0.1540 vs 0.1629 | 147% |
| har_x vs har | 192 | 122 | 63.5% | 0.000215 | 0.3533 vs 0.3624 | 0.1644 vs 0.1564 | 162% |
| har_sv vs har | 188 | 103 | 54.8% | 0.215 | 0.3201 vs 0.3604 | 0.1510 vs 0.1564 | 88% |
| har_iv vs har | 192 | 104 | 54.2% | 0.279 | 0.3118 vs 0.3624 | 0.1629 vs 0.1564 | 72% |
| chronos_cov vs chronos_uni | 192 | 93 | 48.4% | 0.718 | 0.3656 vs 0.3604 | 0.1521 vs 0.1883 | -489% |

### Heavy-earnings slice, paired per-origin (cutoff 5.0% of index weight)

| pair | mean | median | better/n | sign p | top day | % of gap | ex-top |
|---|---|---|---|---|---|---|---|
| har_x vs har | 0.2053 vs 0.3423 | 0.0651 vs 0.0835 | 8/12 | 0.3877 | 2025-11-19 | 50% | 7/11 |
| har_iv_x vs har_iv | 0.1588 vs 0.2545 | 0.0483 vs 0.0688 | 7/12 | 0.7744 | 2025-11-19 | 49% | 6/11 |
| chronos_cov vs chronos_uni | 0.1374 vs 0.3453 | 0.0410 vs 0.1509 | 9/12 | 0.1460 | 2025-11-19 | 51% | 8/11 |

## 30-calendar-day horizon vs VXN (log variance)

- VXN MZ: alpha=-1.301, beta=0.829 (p[beta=1]=0.284), R2=0.295, n=171
  - At the window's median VXN (24.1), fitted realized variance is 68% of implied — 82% in vol terms, about 4.2 vol points of premium. Read the premium here, not off alpha: with beta=0.83 the intercept is not interpretable alone.

Encompassing: realized = a + b*VXN + c*model. H3 wants c>0 and significant. A negative c is not evidence for the model — with VXN already in the regression it means the forecast enters against realized variance, which collinear forecasts commonly do. Models that consume VXN as an input are excluded: regressing on VXN and on a function of VXN is collinear by construction and the split of the coefficients is not interpretable.

- har_cum: MZ beta=0.293 R2=0.063 | encompassing b_implied=1.170 c_model=-0.360 (p=0.0923, R2=0.339, n=171)
- persistence_cum: MZ beta=0.142 R2=0.088 | encompassing b_implied=0.875 c_model=-0.024 (p=0.3889, R2=0.297, n=171)
- chronos_uni: MZ beta=0.377 R2=0.121 | encompassing b_implied=1.068 c_model=-0.216 (p=0.3855, R2=0.310, n=171)
- chronos_cov: MZ beta=0.496 R2=0.143 | encompassing b_implied=0.976 c_model=-0.162 (p=0.3971, R2=0.301, n=171)