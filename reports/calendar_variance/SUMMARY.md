# Original-plan calendar replication: verified result

**No new predictive signal qualified.** Adding the joint original CPI, payroll and FOMC plan block to strong SPX risk controls improved the later sample but worsened the earlier sample. That fails the predeclared requirement to improve both periods. Both comparisons remain in the cumulative family of 108 enumerated hypotheses; no threshold, source, model or sample rule changed after fitting began.

The study produced **6,297 forecasts on 2,099 common dates**, using 101 monthly expanding fits and three models. It forecasts next observed session's Garman–Klass-plus-overnight daily OHLC risk proxy. This is a source-and-timing-controlled replication of a broad calendar question already studied in the repository.

## What changed in forecast quality

Negative absolute differences in `log(h)+q/h` mean improvement. These are **not percentage gains**; raw scores may be negative. Both comparisons needed a difference at or below −0.005 in development and evaluation, improvement in both fixed evaluation slices, and the registered wave/cumulative statistical corrections.

| Comparison | 2016–2019 | Later evaluation | Wave Holm p | Cumulative Holm p |
|---|---:|---:|---:|---:|
| Calendar vs baseline | +0.001990 | -0.008997 | 0.318580 | 1.000000 |
| Calendar vs mean | -0.372619 | -0.447228 | 0.019040 | 0.913920 |

The later calendar-versus-baseline difference is −0.008997, with a nominal 95% uncertainty envelope of **[−0.014344, −0.003796]**. Its conservative phase p-value is 0.000698. Both fixed later slices improve: −0.005216 for the retained 2020–2022 dates and −0.011860 for 2023 onward. This is sample-specific evidence of later improvement and should not be described as no effect anywhere.

However, the earlier difference is **+0.001990**, with interval [−0.002024, +0.005877] and phase p=0.31858. It has the wrong sign and fails the fixed −0.005 useful-effect requirement. Combining both phases gives p=0.31858; the cumulative adjusted p is 1. The later improvement cannot be used to discard the earlier period or justify a new post-result regime split.

The augmented model also beats the historical mean in average score in both phases, but that comparison fails the registered corrections. The common market-history model accounts for much of that advantage; the mean comparison alone cannot identify incremental calendar information. No individual CPI, payroll or FOMC coefficient is treated as a separately tested discovery.

![Both registered comparisons and uncertainty](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/calendar_variance/comparison_intervals.png)

## Coverage and information timing

Development has **983 dates**; the later phase has **1,116**. Although the evaluation window was fixed to begin in January 2020, source eligibility leaves its first scored origin at **2020-11-02**. Only 42 dates remain in 2020, all after that point. This study says nothing about full-calendar performance through the early pandemic shock.

| Year | Common scored dates |
|---|---:|
| 2016 | 252 |
| 2017 | 251 |
| 2018 | 251 |
| 2019 | 229 |
| 2020 | 42 |
| 2021 | 208 |
| 2022 | 231 |
| 2023 | 250 |
| 2024 | 207 |
| 2025 through cutoff | 178 |

The first fit on January 4, 2016 has **1,012 mature common training rows**, just 12 above the fixed minimum. These run January 5, 2011 through August 28, 2015; the final admitted label matures August 31. The first fit's information cutoff is December 31. This gap results from original-plan exclusions, not a change to the training minimum or start date.

The unchanged documentary source audit admits 168 CPI plans, 176 payroll plans and 128 original FOMC dates from 16 annual schedules. Among scored development dates, the flags occur 47/46/31 times respectively, with 119 dates containing at least one flag. Later counts are 54/53/36, with 142 dates containing at least one. Flags can overlap; source-plan counts are not forecast-date counts.

All market predictors end at the preceding observed SPX session. The baseline contains strict 1/5/22 own-risk and negative-return histories, VIX, VIX9D/VIX, VVIX, weekday and nominal elapsed hours. Calendar sources must have a publication civil date strictly before that preceding session. Canceled or revised original plans stay original; emergency meetings and explicit BLS reissues remain excluded. Every model loses a row when any required month is unknown.

The predictor window is entry civil day 16:00 Eastern to next Monday–Friday civil day 16:00, skipping weekends only. It preserves elapsed daylight-saving hours and ignores holidays and early closes; future observed trading dates only determine labels. These are current official-document extracts and archival market measurements, not certified historical web or closing-time vintages. VIX9D's early history is back-calculated. The numerical source fence remains October 20, 2025; the sealed November 3 onward period was not used.

## Verification and retained evidence

**1,031 repository tests passed before fitting**, including 73 new tests; the runner also passed 102 focused checks. Scoped lint passed for all new modules/tests. Tests cover source timing, unknown months, canceled original plans, holidays/daylight-saving conventions, future-data invariance, label maturity, common rows, positive fitting, both statistical gates, and publication/verification failures.

Independent verification passed on its first empirical execution with unchanged tolerances. It reconstructed all **4,226×18 feature cells**, 4,226 calendar source joins, targets, all 101 monthly fits and 6,297 forecasts, both new comparisons, four phase comparisons, twelve bootstrap runs of 99,999 draws, both multiplicity corrections and all 110 ledger events. A separate BFGS optimizer confirmed the positive fits. Maximum saved-coefficient first-order residual was 3.5333e−10; the separate optimizer's maximum was 7.8918e−9, both within their fixed tolerances.

The manifest pins **169 code/test files, 808 input artifacts and 272 earlier protocol/report artifacts**. All 1,249 hashes match, alongside the protocol hash. Every earlier wave's registered artifact check also passed. There were no empirical repairs, retries or omitted comparisons.

The eight recent waves now contain **54 added comparisons and 106,507 forecasts**. The cumulative 108 includes the earlier 54 enumerated model comparisons; it is not the repository's complete lifetime trial count. Historical reuse and selective source coverage remain limitations despite conservative corrections.

- [Frozen design](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/calendar_variance/DESIGN.md) and [protocol](/Users/byrons/code/trading-vol/ndx-vol-experiment/calendar_variance.yaml)
- [Complete metrics](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/calendar_variance/metrics.json), [independent verification](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/calendar_variance/verification.json) and [trial ledger](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/calendar_variance/trial_ledger.jsonl)
- [Date-only feasibility](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/calendar_variance/FEASIBILITY.md) and [independent design review](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/calendar_variance/DESIGN_REVIEW.md)
- [Full pre-run tests](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/calendar_variance/full_pre_run_tests.txt), [focused checks](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/calendar_variance/pre_run_checks.txt) and [verifier log](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/calendar_variance/independent_verifier.log)
- [Exportable figure](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/calendar_variance/comparison_intervals.pdf)

## Next bounded question

A blinded [prospective review](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/calendar_variance/NEXT_RELATIVE_RISK_DESIGN.md) identifies a possible different target: relative intraday OHLC risk between QQQ ETF and the SPX price index, with one lagged joint-correlation input after controlling both assets' risk histories. It is unregistered and unrun. The next step is a joint source/calendar and measurement audit. Current sources do not establish an exact Nasdaq-100-versus-SPX index study, and adding a ratio already implied by existing linear controls would not constitute new information.
