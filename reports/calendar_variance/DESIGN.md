# Documentary calendar replication for SPX daily risk

This eighth bounded wave tests whether the joint original CPI, payroll and FOMC plan block adds predictive information to strong SPX risk controls. Full-session calendars were already tested in the repository; this is a replication with a stricter source and timing policy, not a first calendar test. The [earlier overlap review](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/tail_shape/NEXT_CALENDAR_DESIGN.md) establishes that distinction.

The [protocol](/Users/byrons/code/trading-vol/ndx-vol-experiment/calendar_variance.yaml) is written before new features, targets, fits or scores. Three independent date/missingness checks found 1,012 mature common training observations for the first fit on January 4, 2016, against a fixed minimum of 1,000. The previous-session cutoff is December 31, 2015; admitted origins end August 28, with the final admitted label on August 31. The source gaps are substantial. Numerical feature-scale and solver checks still must pass during the registered run; feasibility is not evidence of predictive success.

## Fixed target and models

At entry close t, forecast next observed SPX session's Garman–Klass daily variance, floored at 1e-10, plus log(open_next/close_t) squared. This is a native daily OHLC risk proxy. It is not measured high-frequency integrated variance, a causal event contribution, a fund return, or an execution result.

The baseline contains an intercept, log strict means of own risk over 1/5/22 sessions, mean negative raw close returns over 1/5/22 sessions, log VIX, log(VIX9D/VIX), log VVIX, four entry-weekday indicators, and nominal elapsed hours. Every market input ends at the previous observed session. Add all three original-plan indicators together as one candidate; compare it against both the baseline and the same-sample historical mean. All three models share 18 complete inputs and matured labels.

The positive model minimizes mean log(h)+q/h plus 0.01 times squared standardized slopes. Training population means/scales and target scaling are fitted separately at each monthly first feature-complete origin. The intercept is unpenalized. A fixed Newton/Armijo solver, tolerance 1e-8, and no fallback are inherited from the tested moment model. Any zero/nonfinite scale among 17 nonintercept common columns, insufficient training support, or unconverged fit aborts the entire registered wave. Future query-label availability never chooses the refit date.

## Information timing and limits

The civil predictor window is entry-day 16:00 Eastern through next Monday–Friday civil day 16:00, skipping weekends only. UTC elapsed hours preserve daylight-saving transitions. Holidays and early closes are not inferred from future market sessions. A Friday before a Monday holiday still has nominal Monday as endpoint. The actual next observed SPX date determines only the target and its maturity.

Each CPI/payroll month touched by that nominal window needs its explicit original plan. Each source's publication civil date must be strictly before the previous observed SPX session date. The FOMC indicator matches the original annual plan's final meeting date to the nominal endpoint date; no historical statement clock is invented. Canceled or revised original plans stay original; emergency meetings and explicit BLS reissues remain excluded. Unknown source coverage is missing for every model, including the mean.

The source corpus is the previously audited set of current official-document text extractions and original-plan ledgers, with no acquisition or substitutions. These are not certified historical raw web vintages. Market data are archival Yahoo/Cboe extracts with uncertain revisions and exact release latency. Early VIX9D training history is back-calculated; all forecast origins begin in 2016. The observed SPX reference calendar inherited vendor removal of incomplete OHLC rows and is not certified exchange-complete. The 2025-10-20 numerical fence and 2025-11-03 sealed boundary remain unchanged.

## Fixed evidence requirements

There are exactly two new paired contrasts: joint calendar versus baseline and versus mean. Both need absolute proper-score improvement of at least 0.005 in development 2016–2019 and evaluation 2020–October 2025, negative differences in both fixed evaluation subperiods, wave Holm adjustment at 0.05/(8×9), and cumulative Holm adjustment across all 108 enumerated comparisons at 0.05. This tracked family is not the entire repository's lifetime experiment count. Reusing historical data remains exploratory despite these corrections.

Each phase uses 99,999 circular block resamples at blocks 21/63/126 plus Bartlett HAC126, retaining the largest two-sided p-value across methods and then across phases. Missing dates remain missing; blocks count retained observations. At least 127 paired observations are required without shortening a bandwidth. Absolute gaps, intervals, annual coverage and fixed-period diagnostics are retained for both comparisons. No percentage improvement is computed from potentially negative raw proper scores.

## Verification and failures

Synthetic tests precede empirical construction and fitting. The manifest pins all Python source/tests, the input corpus and earlier protocols/reports. A separately written verifier reconstructs sources, all features and labels, refit membership, positive-model fits with a separate BFGS optimizer, forecast metadata, proper losses, all inference and the complete ledger. Its coefficient and forecast tolerances are specified before fitting. Any registered production, publication or independent-verification failure invalidates both comparisons with p=1 and no lead; partial scores are retained only as unpublished diagnostics. No empirical repair or gate change is permitted.
