# Results — phase: diagnostic

Quantile grid: [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95] → intervals below are 90%.

**Corrected methodology run** — estimator=`smearing`, inference=`naive`. This report is a fork; `results_diagnostic.md` still holds the frozen pre-registered numbers and is unchanged. Differences between the two are methodology, not data. See src/methodology.py for what each correction does and why.

`mean_var` here is the Duan smearing estimate `exp(mu)*mean(exp(resid))`. The frozen estimator integrates `exp(q)` over the quantile grid and divides by its mass, which discards the tails: it returns about 0.87 of the true conditional mean, and the discarded share depends on the grid — which is why every model's QLIKE differs between `results_clean.md` and `results_clean_dec.md` while EWMA's is 0.4036 in both. Only QLIKE changes; CRPS, pinball and coverage read off the quantiles and are identical.

Two caveats, stated rather than buried. (i) Across the HAR family the frozen estimator's bias spans only 0.866–0.873, so it is a near-common factor: correcting it moves QLIKE **levels** a lot and DM statistics little. It was never the reason one model outranked another. (ii) Rows marked `~` below have no recoverable residuals (Chronos-2, TiRex-2) and are reconstructed from their saved quantiles by tail extension, which on the HAR family lands 3.5% low with a 2.5pp spread across models — real, and a wider spread than the 0.70pp it replaces for those rows. Rows without `~` are exact.

## h=1 losses (mean per day)

| model | n | QLIKE | CRPS | pin0.05 | pin0.95 | 90% cov | p_uc | p_ind |
|---|---|---|---|---|---|---|---|---|
| persistence | 2463 | 0.5426 | 0.6054 | 0.1040 | 0.1084 | 0.892 | 0.214 | 0.000 |
| ewma | 2463 | 0.4506 | - | - | - | - | - | - |
| har | 2463 | 0.3723 | 0.4950 | 0.0840 | 0.0928 | 0.863 | 0.000 | 0.138 |
| har_x | 2463 | 0.3743 | 0.4945 | 0.0867 | 0.0935 | 0.868 | 0.000 | 0.044 |
| har_iv | 2463 | 0.3401 | 0.4714 | 0.0809 | 0.0869 | 0.891 | 0.150 | 0.237 |
| har_sv | 22 | 0.3539 | 0.4758 | 0.0736 | 0.0949 | 0.909 | 0.885 | 0.516 |
| har_ic | 2351 | 0.3490 | 0.4749 | 0.0813 | 0.0875 | 0.886 | 0.031 | 0.121 |
| har_iv_x | 2463 | 0.3417 | 0.4721 | 0.0842 | 0.0866 | 0.892 | 0.214 | 0.607 |
| har_lev | 2463 | 0.3543 | 0.4805 | 0.0816 | 0.0903 | 0.867 | 0.000 | 0.259 |
| har_iv_lev | 2463 | 0.3317 | 0.4632 | 0.0796 | 0.0863 | 0.895 | 0.435 | 0.147 |

## Diebold-Mariano vs HAR (QLIKE; negative = beats HAR)

| model | DM | p | n |
|---|---|---|---|
| persistence | 10.886 | 0.0000 | 2463 |
| ewma | 7.781 | 0.0000 | 2463 |
| har_x | 0.567 | 0.5705 | 2463 |
| har_iv | -5.000 | 0.0000 | 2463 |
| har_sv | -0.178 | 0.8603 | 22 |
| har_ic | -3.382 | 0.0007 | 2351 |
| har_iv_x | -4.036 | 0.0001 | 2463 |
| har_lev | -5.692 | 0.0000 | 2463 |
| har_iv_lev | -5.555 | 0.0000 | 2463 |

## Diebold-Mariano vs HAR-IV — same information set

HAR-IV is log-HAR plus log(VXN). Any model fed VXN beats plain HAR trivially, because implied vol predicts realized vol. The question this table answers is whether the foundation model extracts more from VXN than four OLS terms do.

| model | DM | p | n |
|---|---|---|---|
| har_ic | 1.240 | 0.2150 | 2351 |
| har_iv_x | 0.428 | 0.6690 | 2463 |
| har_iv_lev | -2.166 | 0.0304 | 2463 |

## Event-sliced QLIKE (mean)

| model | FOMC (n) | CPI (n) | heavy-earnings (n) | quiet (n) |
|---|---|---|---|---|
| persistence | 0.5265 (79) | 0.7873 (115) | 0.5609 (157) | 0.5260 (2141) |
| ewma | 0.3626 (79) | 0.4427 (115) | 0.4845 (157) | 0.4517 (2141) |
| har | 0.2973 (79) | 0.4638 (115) | 0.3861 (157) | 0.3682 (2141) |
| har_x | 0.2667 (79) | 0.4575 (115) | 0.3423 (157) | 0.3748 (2141) |
| har_iv | 0.2847 (79) | 0.4050 (115) | 0.3312 (157) | 0.3387 (2141) |
| har_sv | - | - | - | 0.3539 (22) |
| har_ic | 0.2773 (76) | 0.4190 (110) | 0.3079 (146) | 0.3506 (2045) |
| har_iv_x | 0.2783 (79) | 0.3986 (115) | 0.3008 (157) | 0.3436 (2141) |
| har_lev | 0.2821 (79) | 0.4408 (115) | 0.3849 (157) | 0.3490 (2141) |
| har_iv_lev | 0.2728 (79) | 0.4064 (115) | 0.3583 (157) | 0.3269 (2141) |

## Full-sample paired per-origin (all origins in phase)

Win rate and mean difference answer different questions. A high win rate with no mean gain is many small wins funded by rare large losses; a low win rate with a mean gain is the reverse. Read both before believing either.

| pair | n | wins | win % | sign p | mean | median | top-10 share of mean gap |
|---|---|---|---|---|---|---|---|
| har_iv_lev vs har_iv | 2463 | 1457 | 59.2% | 9.67e-20 | 0.3317 vs 0.3401 | 0.1599 vs 0.1708 | 74% |
| har_lev vs har | 2463 | 1481 | 60.1% | 7.58e-24 | 0.3543 vs 0.3723 | 0.1696 vs 0.1768 | 35% |
| har_iv_x vs har_iv | 2463 | 1617 | 65.7% | 3.55e-55 | 0.3417 vs 0.3401 | 0.1694 vs 0.1708 | -315% |
| har_x vs har | 2463 | 1614 | 65.5% | 2.47e-54 | 0.3743 vs 0.3723 | 0.1736 vs 0.1768 | -220% |
| har_iv vs har | 2463 | 1309 | 53.1% | 0.00191 | 0.3401 vs 0.3723 | 0.1708 vs 0.1768 | 32% |

### Heavy-earnings slice, paired per-origin (cutoff 5.0% of index weight)

| pair | mean | median | better/n | sign p | top day | % of gap | ex-top |
|---|---|---|---|---|---|---|---|
| har_x vs har | 0.3423 vs 0.3861 | 0.1545 vs 0.1794 | 64/157 | 0.0251 | 2017-10-26 | 50% | 63/156 |
| har_iv_x vs har_iv | 0.3008 vs 0.3312 | 0.1617 vs 0.1308 | 61/157 | 0.0065 | 2017-10-26 | 54% | 60/156 |

## 30-calendar-day horizon vs VXN (log variance)

- VXN MZ: alpha=-0.040, beta=1.067 (p[beta=1]=0.272), R2=0.550, n=2463
  - At the window's median VXN (20.4), fitted realized variance is 66% of implied — 81% in vol terms, about 3.9 vol points of premium. Read the premium here, not off alpha: with beta=1.07 the intercept is not interpretable alone.

Encompassing: realized = a + b*VXN + c*model. H3 wants c>0 and significant. A negative c is not evidence for the model — with VXN already in the regression it means the forecast enters against realized variance, which collinear forecasts commonly do. Models that consume VXN as an input are excluded: regressing on VXN and on a function of VXN is collinear by construction and the split of the coefficients is not interpretable.

- har_cum: MZ beta=0.810 R2=0.485 | encompassing b_implied=0.851 c_model=0.196 (p=0.0474, R2=0.556, n=2463)
- persistence_cum: MZ beta=0.443 R2=0.360 | encompassing b_implied=0.954 c_model=0.078 (p=0.0110, R2=0.555, n=2463)