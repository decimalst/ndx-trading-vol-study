# Relative QQQ–SPX intraday risk: verified result

**The correlation addition did not qualify.** It increased forecast mean squared error by **0.150% in 2016–2019** and **0.035% in the later evaluation period**, relative to controls containing both assets' risk histories and implied-volatility inputs. Both fixed later subperiods also worsened. The two comparisons remain in the cumulative family of 110 enumerated hypotheses; no measurement, model, source, sample or gate changed after the run began.

The experiment produced **7,386 forecasts on 2,462 common dates**, with 118 monthly expanding fits and three models. The target is next-session log intraday Garman–Klass risk for QQQ ETF minus the corresponding log risk for the SPX price index. This is a relative daily OHLC-risk proxy, not exact Nasdaq-100-index risk or a hedged portfolio's variance.

## Forecast comparisons

Positive relative MSE gain means improvement. Each comparison needed at least **+0.25%** in both phases, negative absolute MSE gaps in both fixed evaluation slices, and the registered wave/cumulative statistical corrections.

| Comparison | 2016–2019 MSE gain | 2020–October 2025 MSE gain | Wave Holm p | Cumulative Holm p |
|---|---:|---:|---:|---:|
| Correlation vs baseline | -0.150% | -0.035% | 0.736710 | 1.000000 |
| Correlation vs mean | +31.391% | +13.181% | 0.067912 | 1.000000 |

Against the strong baseline, the absolute error differences are **+0.00035390** earlier and **+0.00007982** later. Their nominal 95% uncertainty envelopes are [−0.00001318, +0.00073847] and [−0.00035167, +0.00058655]. The corresponding conservative phase p-values are 0.06401 and 0.73671. The two fixed later-slice differences are also positive: +0.00005221 for 2020–2022 and +0.00010960 for 2023 onward. Neither the useful-effect nor stability requirements pass.

The augmented model beats the historical mean in average MSE by 31.39% and 13.18%. Those comparisons have negative nominal interval envelopes in both periods, so this is not a finding that every forecast advantage vanishes or every interval crosses zero. However, the mean comparison fails the registered multiple-comparison checks. The stronger baseline performs better than the augmented model in both phases; the average advantage over the mean cannot be credited to the added correlation.

![Both registered comparisons](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/relative_risk/comparison_intervals.png)

## Measurement and coverage

The predeclared measurement gate **passed** on all **4,226 observed SPX reference dates for each asset**. There were no missing complete OHLC rows, nonfinite calculated GK values, nonpositive GK values, or activations of the fixed `1e-10` floor. The minimum raw GK values were **2.3318348e−6 for QQQ** and **8.3223918e−7 for SPX**. The log-risk target therefore did not depend on artificial flooring. This establishes that numerical property, not the absence of ETF/index measurement differences or proxy bias.

The gate examined every observed asset measurement on the full bounded reference calendar before any correlation, feature, target or sample mask. A failed gate would have preserved both hypotheses at p=1 without building downstream features or forecasts. Missing measurements were never filled, and no row was removed to hide a floor activation.

Development contains **1,005 dates**; evaluation contains **1,457**, beginning January 2, 2020. All 253 observed SPX dates in 2020 remain in this study. Annual scored counts for 2016 through the October 2025 cutoff are **252, 251, 251, 251, 253, 252, 251, 250, 252 and 199**. There are 2,463 feature-complete application dates; December 31, 2019 is excluded from scoring because its next-session label matures after the development boundary. It cannot move the monthly fit date.

The first January 4, 2016 fit uses **1,254 mature common training rows**, from January 5, 2011 through December 30, 2015. Its latest label matures December 31, matching the previous-session fit cutoff. The source audit confirms matching QQQ/SPX dates throughout the reference span. The leading SPX row has no predecessor, and previous-close-dependent measurements do not borrow QQQ's longer prehistory to complete it.

## Model and information policy

The baseline has 27 columns: both assets' strict 1/5/22 intraday-risk, total-risk and negative-return histories; VXN, VIX, VIX9D/VIX and VVIX; entry weekday; and intercept. The candidate adds one strict 22-session correlation of their raw open-to-close returns. All market predictors end at the previous observed SPX session. Every model uses the same rows selected for completeness across all 28 columns, with identical training labels, fit dates and scored origins; each model uses its declared predictor subset.

Exact constant correlation inputs are unknown. A synthetic pre-run test caught a rounding case where repeated nonzero values could otherwise falsely appear correlated; the implementation now checks exact constancy before centering. This correction and a stale protocol-validation-key correction were completed before any empirical numerical measurement. No epsilon variance, clipping, alternate window or post-result repair was introduced.

The fixed normalized ridge penalty is 0.01, with training-only population scaling and an unpenalized intercept. Zero and negative log-risk ratios remain valid. A separate algebraic test confirms that subtracting component ridge forecasts with identical rows, inputs, scaling and penalty equals forecasting their difference; that repackaging is not counted as new information.

The input files remain archival Yahoo/Cboe observations. QQQ is an ETF and SPX is a price index; daily extrema need not be synchronized, and source-calendar equality does not certify exchange-calendar completeness. Within-session ratios remove common multiplicative units, while ETF tracking and vendor conventions remain. Raw overnight/close-return controls retain distribution effects. Early VIX9D history is back-calculated, and exact historical publication latency is unverified. The October 20, 2025 numerical fence and November 3 sealed boundary were preserved. No execution or profit claim follows from these forecasts.

## Verification and retained record

**1,114 repository tests passed before the run**, including 83 new tests, and the runner passed **106 focused checks**. Scoped lint passed for all ten new source/test files. Independent verification passed on its first empirical execution with unchanged tolerances.

The verifier separately reconstructed the full measurement gate, all **4,226×28 feature cells**, signed targets, all 118 monthly fits and 7,386 forecasts, four phase comparisons, twelve bootstrap runs of 99,999 draws, both multiplicity adjustments and all 112 ledger events. An augmented least-squares ridge fit independently checked the saved coefficients; the largest full-MSE first-order residual was **4.0254e−15**, below the fixed `1e-10` tolerance.

The manifest pins **179 source/test files, 12 input artifacts and 298 earlier protocol/report artifacts**. All **489 artifact hashes**, plus the protocol hash, match. Checks against all eight preceding wave manifests also passed. The nine recent waves now contain **56 added comparisons and 113,893 forecasts**. The cumulative 110 includes the earlier 54 enumerated model comparisons and is not the repository's entire lifetime experiment count. Historical reuse remains exploratory despite conservative corrections.

- [Prospective design](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/relative_risk/DESIGN.md) and [frozen protocol](/Users/byrons/code/trading-vol/ndx-vol-experiment/relative_risk.yaml)
- [Complete metrics](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/relative_risk/metrics.json), [independent verification](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/relative_risk/verification.json) and [trial ledger](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/relative_risk/trial_ledger.jsonl)
- [Source/date audit](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/relative_risk/SOURCE_FEASIBILITY.md), [design review](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/relative_risk/DESIGN_REVIEW.md) and [feature contracts](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/relative_risk/FEATURE_IMPLEMENTATION.md)
- [Full pre-run tests](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/relative_risk/full_pre_run_tests.txt), [focused checks](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/relative_risk/pre_run_checks.txt) and [verifier log](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/relative_risk/independent_verifier.log)
- [Exportable figure](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/relative_risk/comparison_intervals.pdf)

The [next blinded question](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/relative_risk/NEXT_JOINT_RISK_DESIGN.md) concerns forecasting the pair's **joint risk**, with identical marginal forecasts and a changing dependence component. It requires a separate matrix-loss and measurement contract; a gain in joint score would not by itself establish true correlation predictability under potentially misspecified marginal forecasts. No such new experiment has been registered or run here.
