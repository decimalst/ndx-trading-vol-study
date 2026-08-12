# Results — phase: diagnostic

Quantile grid: [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95] → intervals below are 90%.

## h=1 losses (mean per day)

| model | n | QLIKE | CRPS | pin0.05 | pin0.95 | 90% cov | p_uc | p_ind |
|---|---|---|---|---|---|---|---|---|
| persistence | 2463 | 0.5574 | 0.6054 | 0.1040 | 0.1084 | 0.892 | 0.214 | 0.000 |
| ewma | 2463 | 0.4506 | - | - | - | - | - | - |
| har | 2463 | 0.3815 | 0.4950 | 0.0840 | 0.0928 | 0.863 | 0.000 | 0.138 |
| har_x | 2463 | 0.3838 | 0.4945 | 0.0867 | 0.0935 | 0.868 | 0.000 | 0.044 |
| har_iv | 2463 | 0.3438 | 0.4714 | 0.0809 | 0.0869 | 0.891 | 0.150 | 0.237 |
| har_sv | 22 | 0.3729 | 0.4758 | 0.0736 | 0.0949 | 0.909 | 0.885 | 0.516 |
| har_ic | 2351 | 0.3479 | 0.4749 | 0.0813 | 0.0875 | 0.886 | 0.031 | 0.121 |
| har_iv_x | 2463 | 0.3437 | 0.4721 | 0.0842 | 0.0866 | 0.892 | 0.214 | 0.607 |
| har_lev | 2463 | 0.3618 | 0.4805 | 0.0816 | 0.0903 | 0.867 | 0.000 | 0.259 |
| har_iv_lev | 2463 | 0.3376 | 0.4632 | 0.0796 | 0.0863 | 0.895 | 0.435 | 0.147 |

## Diebold-Mariano vs HAR (QLIKE; negative = beats HAR)

| model | DM | p | n |
|---|---|---|---|
| persistence | 9.534 | 0.0000 | 2463 |
| ewma | 6.969 | 0.0000 | 2463 |
| har_x | 0.567 | 0.5705 | 2463 |
| har_iv | -5.148 | 0.0000 | 2463 |
| har_sv | -0.290 | 0.7744 | 22 |
| har_ic | -4.115 | 0.0000 | 2351 |
| har_iv_x | -4.425 | 0.0000 | 2463 |
| har_lev | -5.367 | 0.0000 | 2463 |
| har_iv_lev | -5.212 | 0.0000 | 2463 |

## Diebold-Mariano vs HAR-IV — same information set

HAR-IV is log-HAR plus log(VXN). Any model fed VXN beats plain HAR trivially, because implied vol predicts realized vol. The question this table answers is whether the foundation model extracts more from VXN than four OLS terms do.

| model | DM | p | n |
|---|---|---|---|
| har_ic | 0.258 | 0.7967 | 2351 |
| har_iv_x | -0.031 | 0.9750 | 2463 |
| har_iv_lev | -1.410 | 0.1585 | 2463 |

## Event-sliced QLIKE (mean)

| model | FOMC (n) | CPI (n) | heavy-earnings (n) | quiet (n) |
|---|---|---|---|---|
| persistence | 0.6380 (79) | 0.9142 (115) | 0.6452 (157) | 0.5270 (2141) |
| ewma | 0.3626 (79) | 0.4427 (115) | 0.4845 (157) | 0.4517 (2141) |
| har | 0.3524 (79) | 0.5262 (115) | 0.4417 (157) | 0.3702 (2141) |
| har_x | 0.2830 (79) | 0.5156 (115) | 0.3522 (157) | 0.3814 (2141) |
| har_iv | 0.3354 (79) | 0.4569 (115) | 0.3874 (157) | 0.3350 (2141) |
| har_sv | - | - | - | 0.3729 (22) |
| har_ic | 0.3199 (76) | 0.4678 (110) | 0.3541 (146) | 0.3427 (2045) |
| har_iv_x | 0.2856 (79) | 0.4387 (115) | 0.2900 (157) | 0.3441 (2141) |
| har_lev | 0.3319 (79) | 0.4955 (115) | 0.4359 (157) | 0.3498 (2141) |
| har_iv_lev | 0.3250 (79) | 0.4600 (115) | 0.4172 (157) | 0.3254 (2141) |

## Full-sample paired per-origin (all origins in phase)

Win rate and mean difference answer different questions. A high win rate with no mean gain is many small wins funded by rare large losses; a low win rate with a mean gain is the reverse. Read both before believing either.

| pair | n | wins | win % | sign p | mean | median | top-10 share of mean gap |
|---|---|---|---|---|---|---|---|
| har_iv_lev vs har_iv | 2463 | 1366 | 55.5% | 6.47e-08 | 0.3376 vs 0.3438 | 0.1437 vs 0.1486 | 126% |
| har_lev vs har | 2463 | 1417 | 57.5% | 8.06e-14 | 0.3618 vs 0.3815 | 0.1481 vs 0.1592 | 39% |
| har_iv_x vs har_iv | 2463 | 1511 | 61.3% | 1.44e-29 | 0.3437 vs 0.3438 | 0.1418 vs 0.1486 | 5300% |
| har_x vs har | 2463 | 1496 | 60.7% | 1.26e-26 | 0.3838 vs 0.3815 | 0.1520 vs 0.1592 | -244% |
| har_iv vs har | 2463 | 1333 | 54.1% | 4.65e-05 | 0.3438 vs 0.3815 | 0.1486 vs 0.1592 | 31% |

### Heavy-earnings slice, paired per-origin (cutoff 5.0% of index weight)

| pair | mean | median | better/n | sign p | top day | % of gap | ex-top |
|---|---|---|---|---|---|---|---|
| har_x vs har | 0.3522 vs 0.4417 | 0.1533 vs 0.2073 | 71/157 | 0.2638 | 2017-10-26 | 29% | 70/156 |
| har_iv_x vs har_iv | 0.2900 vs 0.3874 | 0.1341 vs 0.1582 | 69/157 | 0.1506 | 2016-01-14 | 22% | 68/156 |

## 30-calendar-day horizon vs VXN (log variance)

- VXN MZ: alpha=-0.040, beta=1.067 (p[beta=1]=0.272), R2=0.550, n=2463
  - At the window's median VXN (20.4), fitted realized variance is 66% of implied — 81% in vol terms, about 3.9 vol points of premium. Read the premium here, not off alpha: with beta=1.07 the intercept is not interpretable alone.

Encompassing: realized = a + b*VXN + c*model. H3 wants c>0 and significant. A negative c is not evidence for the model — with VXN already in the regression it means the forecast enters against realized variance, which collinear forecasts commonly do. Models that consume VXN as an input are excluded: regressing on VXN and on a function of VXN is collinear by construction and the split of the coefficients is not interpretable.

- har_cum: MZ beta=0.810 R2=0.485 | encompassing b_implied=0.851 c_model=0.196 (p=0.0474, R2=0.556, n=2463)
- persistence_cum: MZ beta=0.443 R2=0.360 | encompassing b_implied=0.954 c_model=0.078 (p=0.0110, R2=0.555, n=2463)