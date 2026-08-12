# GBM post-result reviewer diagnostics

**Evidence class: post-result, outcome-conditioned descriptive diagnostic.**

Protocol SHA-256: `7cb14bc8ccdbfccba1ae9a10dae787465409099c48bf0f702d84d76ad314d63b`.

The parent verdict remains **INCONCLUSIVE**. Substantively, GBM was point-estimate worse than HAR-IV on all three frozen splits (about 4%); nothing here changes that registered verdict.

The frozen comparison used the same same-origin Cboe VXN close for both models, so it is an internally common-information comparison, but it is timing-ambiguous relative to the repository's standing 16:00 origin. It is not labeled leakage-free or timing-safe here.

## 16:00 timing-safe VXN sensitivity

Both estimators were rerun unchanged except that `liv` uses the preceding complete session's VXN close. There was no tuning or interaction reselection.

| split | n | HAR-IV QLIKE | GBM QLIKE | improvement | block p | 95% block interval | win rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_diagnostic | 2463 | 0.353289 | 0.367278 | -3.960% | 0.1414 | [-0.001445, +0.037622] | 55.3% |
| discovery | 1006 | 0.352264 | 0.356677 | -1.253% | 0.1942 | [-0.002486, +0.010684] | 53.2% |
| confirmation | 1457 | 0.353996 | 0.374598 | -5.820% | 0.1984 | [-0.004222, +0.059655] | 56.8% |

Pre-specified sensitivity assessment: **SURVIVES_AT_1600_SAFE_BOUNDARY**. The original formal verdict remains **INCONCLUSIVE**.

## QLIKE loss gap by realized-variance decile

Candidate minus baseline loss is reported below; negative values favor the candidate. Fractions attribute the aggregate paired gap and can be negative or exceed 100% when deciles offset one another.

### gbm confirmation

Common origins: 1457 (2020-01-02 through 2025-10-17). Overall candidate improvement: -4.669%; win rate: 54.7%; top-decile share of aggregate gap: +174.6%.

| RV decile | n | baseline QLIKE | candidate QLIKE | mean diff | sum diff | win rate | fraction of total gap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 146 | 0.774934 | 0.776916 | +0.001982 | +0.289360 | 43.8% | +1.2% |
| 2 | 146 | 0.407304 | 0.393811 | -0.013494 | -1.970114 | 52.7% | -8.4% |
| 3 | 145 | 0.259643 | 0.230871 | -0.028772 | -4.171988 | 62.8% | -17.9% |
| 4 | 146 | 0.219003 | 0.188409 | -0.030594 | -4.466664 | 62.3% | -19.1% |
| 5 | 146 | 0.180263 | 0.158672 | -0.021591 | -3.152280 | 64.4% | -13.5% |
| 6 | 145 | 0.163576 | 0.145632 | -0.017944 | -2.601895 | 52.4% | -11.1% |
| 7 | 146 | 0.166855 | 0.140238 | -0.026617 | -3.886047 | 67.8% | -16.6% |
| 8 | 145 | 0.193921 | 0.182618 | -0.011303 | -1.638964 | 58.6% | -7.0% |
| 9 | 146 | 0.253111 | 0.281725 | +0.028615 | +4.177737 | 52.7% | +17.9% |
| 10 | 146 | 0.809399 | 1.088582 | +0.279183 | +40.760674 | 29.5% | +174.6% |

### locked term confirmation

Common origins: 1457 (2020-01-02 through 2025-10-17). Overall candidate improvement: -0.243%; win rate: 57.4%; top-decile share of aggregate gap: +137.9%.

| RV decile | n | baseline QLIKE | candidate QLIKE | mean diff | sum diff | win rate | fraction of total gap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 146 | 0.774934 | 0.774890 | -0.000044 | -0.006453 | 50.0% | -0.5% |
| 2 | 146 | 0.407304 | 0.406844 | -0.000460 | -0.067188 | 61.0% | -5.5% |
| 3 | 145 | 0.259643 | 0.259090 | -0.000553 | -0.080257 | 70.3% | -6.6% |
| 4 | 146 | 0.219003 | 0.218512 | -0.000491 | -0.071683 | 74.0% | -5.9% |
| 5 | 146 | 0.180263 | 0.179856 | -0.000407 | -0.059380 | 71.2% | -4.9% |
| 6 | 145 | 0.163576 | 0.162986 | -0.000590 | -0.085582 | 64.8% | -7.1% |
| 7 | 146 | 0.166855 | 0.166381 | -0.000474 | -0.069146 | 61.0% | -5.7% |
| 8 | 145 | 0.193921 | 0.193760 | -0.000161 | -0.023385 | 49.7% | -1.9% |
| 9 | 146 | 0.253111 | 0.253131 | +0.000021 | +0.003003 | 46.6% | +0.2% |
| 10 | 146 | 0.809399 | 0.820854 | +0.011455 | +1.672441 | 26.0% | +137.9% |

### timing safe gbm confirmation

Common origins: 1457 (2020-01-02 through 2025-10-17). Overall candidate improvement: -5.820%; win rate: 56.8%; top-decile share of aggregate gap: +149.8%.

| RV decile | n | baseline QLIKE | candidate QLIKE | mean diff | sum diff | win rate | fraction of total gap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 146 | 0.781423 | 0.793346 | +0.011923 | +1.740750 | 50.0% | +5.8% |
| 2 | 146 | 0.425344 | 0.405273 | -0.020071 | -2.930417 | 61.6% | -9.8% |
| 3 | 145 | 0.267300 | 0.236679 | -0.030621 | -4.440087 | 69.0% | -14.8% |
| 4 | 146 | 0.213802 | 0.185400 | -0.028401 | -4.146592 | 71.2% | -13.8% |
| 5 | 146 | 0.178161 | 0.161279 | -0.016883 | -2.464868 | 67.8% | -8.2% |
| 6 | 145 | 0.159882 | 0.140163 | -0.019718 | -2.859171 | 58.6% | -9.5% |
| 7 | 146 | 0.162294 | 0.143639 | -0.018654 | -2.723554 | 54.8% | -9.1% |
| 8 | 145 | 0.180652 | 0.174806 | -0.005845 | -0.847597 | 57.2% | -2.8% |
| 9 | 146 | 0.261808 | 0.287357 | +0.025549 | +3.730133 | 50.0% | +12.4% |
| 10 | 146 | 0.906187 | 1.214117 | +0.307930 | +44.957783 | 28.1% | +149.8% |

### earnings with iv diagnostic

Common origins: 2463 (2016-01-04 through 2025-10-17). Overall candidate improvement: +0.038%; win rate: 61.3%; top-decile share of aggregate gap: +478.4%.

| RV decile | n | baseline QLIKE | candidate QLIKE | mean diff | sum diff | win rate | fraction of total gap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 247 | 0.634311 | 0.621780 | -0.012532 | -3.095347 | 88.3% | +957.5% |
| 2 | 246 | 0.297463 | 0.291425 | -0.006038 | -1.485320 | 83.3% | +459.5% |
| 3 | 246 | 0.215618 | 0.209140 | -0.006478 | -1.593655 | 79.7% | +493.0% |
| 4 | 246 | 0.163671 | 0.169833 | +0.006162 | +1.515919 | 65.4% | -468.9% |
| 5 | 247 | 0.131003 | 0.145368 | +0.014364 | +3.547989 | 55.9% | -1097.6% |
| 6 | 246 | 0.170480 | 0.173359 | +0.002879 | +0.708297 | 56.9% | -219.1% |
| 7 | 246 | 0.195889 | 0.199058 | +0.003169 | +0.779666 | 52.4% | -241.2% |
| 8 | 246 | 0.249862 | 0.261388 | +0.011526 | +2.835395 | 48.0% | -877.1% |
| 9 | 246 | 0.325145 | 0.317056 | -0.008088 | -1.989707 | 47.6% | +615.5% |
| 10 | 247 | 1.051796 | 1.045535 | -0.006261 | -1.546497 | 36.0% | +478.4% |

### earnings without iv diagnostic

Common origins: 2463 (2016-01-04 through 2025-10-17). Overall candidate improvement: -0.591%; win rate: 60.7%; top-decile share of aggregate gap: +96.9%.

| RV decile | n | baseline QLIKE | candidate QLIKE | mean diff | sum diff | win rate | fraction of total gap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 247 | 0.624766 | 0.603366 | -0.021400 | -5.285701 | 89.9% | -95.1% |
| 2 | 246 | 0.313355 | 0.302530 | -0.010824 | -2.662762 | 80.9% | -47.9% |
| 3 | 246 | 0.255992 | 0.244570 | -0.011422 | -2.809819 | 78.0% | -50.6% |
| 4 | 246 | 0.176934 | 0.180903 | +0.003969 | +0.976400 | 63.8% | +17.6% |
| 5 | 247 | 0.168956 | 0.181666 | +0.012709 | +3.139208 | 57.5% | +56.5% |
| 6 | 246 | 0.206318 | 0.214041 | +0.007723 | +1.899855 | 53.3% | +34.2% |
| 7 | 246 | 0.244162 | 0.254340 | +0.010178 | +2.503876 | 56.1% | +45.1% |
| 8 | 246 | 0.271606 | 0.281860 | +0.010255 | +2.522651 | 45.9% | +45.4% |
| 9 | 246 | 0.395103 | 0.394656 | -0.000447 | -0.109909 | 46.7% | -2.0% |
| 10 | 247 | 1.154915 | 1.176711 | +0.021796 | +5.383710 | 35.2% | +96.9% |

## Diagnostic reading

The proposed asymmetric-loss mechanism is strongly supported for the GBM comparison, though not literally in all other nine bins: GBM improves mean QLIKE in realized-variance deciles 2–8, loses in deciles 1, 9, and 10, and decile 10 contributes 174.6% of the net deficit. The many-small-gains/rare-large-loss shape therefore explains the worse mean despite a 54.7% daily win rate.

The frozen locked term shows the same concentrated failure more cleanly: it improves deciles 1–8, is nearly flat in decile 9, and decile 10 contributes 137.9% of its net deficit. The timing-safe GBM sensitivity also improves deciles 2–8 while its top decile contributes 149.8% of the deficit.

The transfer is not universal. The earnings-with-IV comparison is a near-zero cancellation (+0.038%), and it improves rather than loses in the top decile, so its 478.4% ratio is unstable against a tiny aggregate denominator and does not support the mechanism. Without IV, the top decile contributes 96.9% of the deficit, but several middle deciles also lose. The defensible finding is therefore concentrated QLIKE tail fragility for added functional flexibility and the locked term, with partial—not four-way—replication.

Every table reconciles to its full-sample paired mean and sum within the frozen 1e-12 tolerance. These are realized-outcome bins, so they explain where an observed loss gap occurred; they do not define a usable ex-ante rule.

## Correlated-feature and SHAP audit

The partial-dependence-selected pair `lrv_w × lrv_m` has discovery-background Pearson correlation 0.8157 and Spearman correlation 0.8022. Both are overlapping averages of the same RV series. Partial dependence can therefore extrapolate into weakly supported combinations, making this selection plausibly a correlation artifact.

A fixed post-result TreeExplainer interaction audit (SHAP 0.51.0) selected `lrv_w × liv`; it did not agree with the PD pair.

| SHAP pair | mean absolute interaction | selected |
|---|---:|:---:|
| lrv_w × liv | 0.04046525 | yes |
| lrv_w × lrv_m | 0.01529280 | no |
| lrv_d × liv | 0.01446888 | no |
| lrv_m × liv | 0.00842109 | no |
| lrv_d × lrv_w | 0.00455226 | no |
| lrv_d × lrv_m | 0.00314066 | no |

The SHAP audit reused the exact frozen discovery-snapshot estimator and full discovery background, with no confirmation reselection, hyperparameter search, or substitute method. It is diagnostic evidence only.
