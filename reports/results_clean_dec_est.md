# Results — phase: clean

Quantile grid: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] → intervals below are 80%. **Decile grid, not the pre-registered one.** TiRex-2 emits only these levels, so every model here was recomputed on them; `mean_var` is comparable within this report but not against `results_*.md`. config.yaml is unchanged.

**Corrected methodology run** — estimator=`smearing`, inference=`naive`. This report is a fork; `results_clean_dec.md` still holds the frozen pre-registered numbers and is unchanged. Differences between the two are methodology, not data. See src/methodology.py for what each correction does and why.

`mean_var` here is the Duan smearing estimate `exp(mu)*mean(exp(resid))`. The frozen estimator integrates `exp(q)` over the quantile grid and divides by its mass, which discards the tails: it returns about 0.87 of the true conditional mean, and the discarded share depends on the grid — which is why every model's QLIKE differs between `results_clean.md` and `results_clean_dec.md` while EWMA's is 0.4036 in both. Only QLIKE changes; CRPS, pinball and coverage read off the quantiles and are identical.

Two caveats, stated rather than buried. (i) The frozen estimator's bias spans 0.866–0.873 across `har`, `har_x`, `har_iv`, `har_iv_x` and `har_ic` — but 0.866–0.883 once `har_sv`, `har_lev` and `har_iv_lev` are included, and 0.812–0.883 across every quantile model scored here, because `persistence` sits at 0.812. It is a near-common factor only on the narrow set. Correcting it moves QLIKE **levels** far more than DM statistics, but **not by cancellation** — QLIKE differentials are not scale-invariant, and rescaling both forecasts by a single common factor reproduces most of the DM movement on its own. It is therefore not safe to assume rankings are unaffected: on this window `har_sv vs har` moves from DM −1.615 (p=0.1080) to −2.163 (p=0.0318) on identical exactly-scored origins, crossing α=0.05. (ii) Rows marked `~` below have no recoverable residuals (Chronos-2, TiRex-2) and are reconstructed from their saved quantiles by tail extension, which on the five pinned HAR models lands 3.5% low with a 2.5pp spread — real, and a wider spread than the 0.70pp it replaces for those rows. A `~` row compared against an unmarked row mixes estimators; read those comparisons as indicative. Rows without `~` are exact.

## h=1 losses (mean per day)

| model | n | QLIKE | CRPS | pin0.10 | pin0.90 | 80% cov | p_uc | p_ind |
|---|---|---|---|---|---|---|---|---|
| persistence | 192 | 0.5429 | 0.6825 | 0.1812 | 0.1850 | 0.781 | 0.521 | 0.052 |
| ewma | 192 | 0.4036 | - | - | - | - | - | - |
| har | 192 | 0.3624 | 0.5526 | 0.1585 | 0.1552 | 0.750 | 0.092 | 0.112 |
| har_x | 192 | 0.3533 | 0.5409 | 0.1573 | 0.1521 | 0.750 | 0.092 | 0.225 |
| har_iv | 192 | 0.3118 | 0.5075 | 0.1402 | 0.1427 | 0.812 | 0.662 | 0.848 |
| har_sv | 188 | 0.3201 | 0.5158 | 0.1458 | 0.1415 | 0.830 | 0.298 | 0.202 |
| har_ic | 192 | 0.3158 | 0.5120 | 0.1409 | 0.1429 | 0.792 | 0.774 | 0.716 |
| har_iv_x | 192 | 0.3037 | 0.4949 | 0.1385 | 0.1396 | 0.807 | 0.800 | 0.635 |
| chronos_uni ~ | 192 | 0.3586 | 0.5499 | 0.1499 | 0.1518 | 0.812 | 0.662 | 0.143 |
| chronos_cov ~ | 192 | 0.3642 | 0.5498 | 0.1540 | 0.1522 | 0.760 | 0.180 | 0.453 |
| chronos_cov_iv ~ | 192 | 0.3186 | 0.5036 | 0.1350 | 0.1456 | 0.812 | 0.662 | 0.773 |
| tirex_uni ~ | 192 | 0.3642 | 0.5522 | 0.1510 | 0.1528 | 0.786 | 0.642 | 0.181 |
| tirex_cov ~ | 192 | 0.3694 | 0.5572 | 0.1518 | 0.1538 | 0.771 | 0.321 | 0.252 |
| tirex_cov_iv ~ | 192 | 0.3647 | 0.5518 | 0.1518 | 0.1537 | 0.797 | 0.914 | 0.083 |
| tirex_cov_ivf ~ | 192 | 0.3127 | 0.4929 | 0.1291 | 0.1483 | 0.740 | 0.043 | 0.948 |

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
| chronos_uni | -0.429 | 0.6687 | 192 |
| chronos_cov | 0.109 | 0.9131 | 192 |
| chronos_cov_iv | -2.601 | 0.0100 | 192 |
| tirex_uni | 0.165 | 0.8694 | 192 |
| tirex_cov | 0.601 | 0.5488 | 192 |
| tirex_cov_iv | 0.219 | 0.8266 | 192 |
| tirex_cov_ivf | -2.662 | 0.0084 | 192 |

## Diebold-Mariano vs HAR-IV — same information set

HAR-IV is log-HAR plus log(VXN). Any model fed VXN beats plain HAR trivially, because implied vol predicts realized vol. The question this table answers is whether the foundation model extracts more from VXN than four OLS terms do.

| model | DM | p | n |
|---|---|---|---|
| har_ic | 1.327 | 0.1861 | 192 |
| har_iv_x | -1.347 | 0.1796 | 192 |
| chronos_cov_iv | 0.465 | 0.6425 | 192 |
| tirex_cov_iv | 2.951 | 0.0036 | 192 |
| tirex_cov_ivf | 0.050 | 0.9602 | 192 |

## TiRex-2 robustness: origins after publication (2026-07-01)

The date-based leakage rule applied literally to TiRex-2. Tiny sample by construction — this is a sign check against the full window above, not a test. See reports/LEAKAGE_TIREX2.md.

| model | QLIKE (full) | QLIKE (post-pub) | n |
|---|---|---|---|
| tirex_uni | 0.3642 | 0.2435 | 28 |
| tirex_cov | 0.3694 | 0.2497 | 28 |
| tirex_cov_iv | 0.3647 | 0.2451 | 28 |
| tirex_cov_ivf | 0.3127 | 0.1852 | 28 |
| har | 0.3624 | 0.2515 | 28 |
| har_iv | 0.3118 | 0.1987 | 28 |
| chronos_cov | 0.3642 | 0.2597 | 28 |

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
| chronos_uni | 0.2027 (6) | 0.3960 (8) | 0.3388 (12) | 0.3639 (166) |
| chronos_cov | 0.2917 (6) | 0.4054 (8) | 0.1366 (12) | 0.3813 (166) |
| chronos_cov_iv | 0.3263 (6) | 0.3753 (8) | 0.1169 (12) | 0.3302 (166) |
| tirex_uni | 0.1717 (6) | 0.3787 (8) | 0.3981 (12) | 0.3680 (166) |
| tirex_cov | 0.2397 (6) | 0.4785 (8) | 0.3054 (12) | 0.3734 (166) |
| tirex_cov_iv | 0.2012 (6) | 0.4256 (8) | 0.3397 (12) | 0.3695 (166) |
| tirex_cov_ivf | 0.1735 (6) | 0.2740 (8) | 0.2684 (12) | 0.3228 (166) |

## Full-sample paired per-origin (all origins in phase)

Win rate and mean difference answer different questions. A high win rate with no mean gain is many small wins funded by rare large losses; a low win rate with a mean gain is the reverse. Read both before believing either.

| pair | n | wins | win % | sign p | mean | median | top-10 share of mean gap |
|---|---|---|---|---|---|---|---|
| har_iv_x vs har_iv | 192 | 125 | 65.1% | 3.43e-05 | 0.3037 vs 0.3118 | 0.1540 vs 0.1629 | 147% |
| har_x vs har | 192 | 122 | 63.5% | 0.000215 | 0.3533 vs 0.3624 | 0.1644 vs 0.1564 | 162% |
| har_sv vs har | 188 | 103 | 54.8% | 0.215 | 0.3201 vs 0.3604 | 0.1510 vs 0.1564 | 88% |
| har_iv vs har | 192 | 104 | 54.2% | 0.279 | 0.3118 vs 0.3624 | 0.1629 vs 0.1564 | 72% |
| chronos_cov vs chronos_uni | 192 | 90 | 46.9% | 0.427 | 0.3642 vs 0.3586 | 0.1543 vs 0.1958 | -434% |
| tirex_cov vs tirex_uni | 192 | 67 | 34.9% | 3.43e-05 | 0.3694 vs 0.3642 | 0.1854 vs 0.1670 | -345% |

### Heavy-earnings slice, paired per-origin (cutoff 5.0% of index weight)

| pair | mean | median | better/n | sign p | top day | % of gap | ex-top |
|---|---|---|---|---|---|---|---|
| har_x vs har | 0.2053 vs 0.3423 | 0.0651 vs 0.0835 | 8/12 | 0.3877 | 2025-11-19 | 50% | 7/11 |
| har_iv_x vs har_iv | 0.1588 vs 0.2545 | 0.0483 vs 0.0688 | 7/12 | 0.7744 | 2025-11-19 | 49% | 6/11 |
| chronos_cov vs chronos_uni | 0.1366 vs 0.3388 | 0.0406 vs 0.1436 | 9/12 | 0.1460 | 2025-11-19 | 54% | 8/11 |
| tirex_cov vs tirex_uni | 0.3054 vs 0.3981 | 0.1160 vs 0.1809 | 7/12 | 0.7744 | 2025-11-19 | 74% | 6/11 |

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