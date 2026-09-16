# Wave 7 source and feature review

Completed 2026-09-07 UTC before protocol freeze, empirical event counting, model fitting, or predictive scoring. This task wrote only `src/tail_shape_features.py`, `tests/test_tail_shape_features.py`, and this new report. Earlier Python, source files, protocols, and reports were not changed.

## Test-first record

The initial run of `.venv/bin/python -m unittest tests.test_tail_shape_features -v` exited 1 with `ImportError: cannot import name 'tail_shape_features' from 'src'`: the 21-test synthetic suite existed before its producer. The first implementation run exposed pandas attribute-name collisions: `.skew` resolved to a method in both the implementation and test fixtures. Explicit `['skew']` column indexing fixed those failures before any empirical source read. The normalization fixture was subsequently strengthened with a synthetic downside shock to exercise event 1 as well as the separate valid-zero/event-0 case. No event formula, threshold, tolerance, or training gate was relaxed.

**All 21 tests now pass; scoped Ruff checks pass.** They cover exact lagged raw-return and negative-return means, the strict daily-variance normalizer without annualization, next-observed-session targets and maturity, zero returns, missing labels remaining unknown, raw-price unit invariance, ignored adjusted close, future-value/calendar mutations, missing-window propagation, fixed schemas, training-only square centers, population-scale failure including derived squares, and invalid design rejection without row selection. Source tests cover a fixed-byte hash mismatch before the SKEW parser, manifest mismatch, fixed historical anchor, raw/derived disagreement, and forbidden post-fence numerical tokens.

| New artifact | SHA-256 |
| --- | --- |
| `src/tail_shape_features.py` | `ae62f72530f6dc7014a699f9e3291a4aa49e461756a0e8a150ee0f3fcb234921` |
| `tests/test_tail_shape_features.py` | `409f308710631a8e2e2bd12205af7aef593ff20615fd7081615fa542899d0ad6` |

The producer imports unchanged source/calendar validation helpers from `src/index_hinge.py`, whose hash remains `2e3e9c02f0125c86a4c8296e218918ba3f7f172be28fa36709fe3f9c0ef5bc75`. It imports no earlier fitting routine. Model fitting, matched conditional moments, event-support gates, training maturity, and scoring are the root runner's responsibility and are not established by this source/feature review.

## Fixed sources and bounded admission

Numeric reads stop at **2025-10-20**; the sealed boundary remains **2025-11-03**. Parquet reads select only needed columns and use a date predicate. Raw CSV values remain text until their date passes the bound. Reading full-file hashes and post-bound date tokens does not analyze post-bound numerical observations. The legacy SKEW source is not replaced by the separately cached free-source version.

| Input | Bounded rows | First bounded date | Last bounded date | Missing SPX reference dates |
| --- | ---: | --- | --- | ---: |
| SPX raw OHLC | 4,226 | 2009-01-02 | 2025-10-20 | Reference calendar |
| VIX | 9,040 | 1990-01-02 | 2025-10-20 | 0 |
| VIX9D | 3,721 | 2011-01-04 | 2025-10-20 | 505, all before its first observation |
| VVIX | 4,878 | 2006-03-06 | 2025-10-20 | 2 |
| Legacy SKEW raw CSV | 9,001 | 1990-01-02 | 2025-10-20 | 5 |
| Existing derived SKEW parquet, audit only | 9,001 | 1990-01-02 | 2025-10-20 | Same 5 |

Selected observed values contain no missing fields or invalid observed prices; source-date absences remain separate missing observations after alignment. VVIX is absent on 2010-11-11 and 2013-05-13. SKEW is absent on **2011-02-09, 2017-09-14, 2018-12-03, 2019-07-05, and 2024-11-29**. Each absence remains unknown at the following reference-session entry after the declared lag. No forward fill or calendar compression is used. Bounded SKEW has 4,780 dates outside the SPX reference calendar; those do not create SPX sessions.

The raw SKEW hash must equal both the fixed pin and `data/raw/skew_daily_source.json` before its numerical parser is called. Its original metadata records retrieval at **2026-08-12T06:08:02.978922+00:00**, the Cboe URL `https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv`, and 9,203 source date rows from 1990-01-02 through 2026-08-11. Those source-date fields match the retained date tokens. Only 9,001 bounded values were parsed numerically.

The fixed original anchor **2018-08-13 = 159.03**, tolerance **0.01**, passes in both metadata and bounded raw values. The anchor is fixed independently of the metadata: changing the metadata's anchor to match an altered source cannot admit it. Every bounded raw date and value is **exactly equal** to the existing derived parquet after field-name alignment. This is exact equality of the loaded numeric values, not a tolerance claim. The raw CSV supplies predictors; the derived file is checked for consistency only.

| Exact input path | SHA-256 |
| --- | --- |
| `data/research_paths/spx_daily.parquet` | `3958fbb1eb36689df1596c26b9f0e02e3f1d4fa3b0032381e5287a2a03697bf0` |
| `data/free_sources/raw/cboe/VIX_History.csv` | `a34aabce269632f30904cf482986dd50b6d4cf51f2203dc51a0a9f460f3c90b2` |
| `data/free_sources/raw/cboe/VIX9D_History.csv` | `0d6f600ee71bf6ffb5069d0c583cbe0b1ec97df6da4e4e439d5f0f22f5616abe` |
| `data/free_sources/raw/cboe/VVIX_History.csv` | `f6bc726455fa3859c662875a005e97ba656b9536ae58fa6977ad24c6228e4f6a` |
| `data/raw/SKEW_History.csv` | `becbf3f7510de66a736495df29a84ba1911d362402f05f6c46dedd5e9b971492` |
| `data/raw/skew_daily_source.json` | `d45a9e3080116655b8f613b131903de33eb23543037893c8603624e84d4121d7` |
| `data/raw/skew_daily.parquet` | `159deae9944941c92a454e80ff32d33eddc94e6e4f2e79a64678bd0429ade5a7` |

The four market-source hashes agree with the prior [wave 6 source review](../index_hinge/SOURCE_REVIEW.md). That review documents Yahoo/yfinance `^GSPC` with `auto_adjust=False` and the historical source builder's removal of incomplete OHLC rows. The calendar therefore consists of observed vendor sessions; its completeness is not independently certified. This module preserves every bounded source row.

## Feature and target contract

`RAW` has 19 columns: the prior wave's 15 raw controls, followed by `neg_d,neg_w,neg_m,skew`. Negative-return controls are strict means of `max(-log(C_s/C_(s−1)),0)` over 1/5/22 observed sessions, then shifted one session. SKEW is its raw index level, aligned to SPX sessions before shifting one session. `BASE=ALL_FEATURES` adds `I_square,R_square,skew_square`, centered using the exact supplied training rows. All 21 non-intercept population scales must exceed `1e-12`; the transform aborts otherwise and returns no reduced feature family.

Feature metadata are `feature_cutoff_date`, `normalization_mean`, and `normalization_scale`. The mean is the strict trailing 22-session arithmetic mean of raw SPX log returns through the prior session. The scale is the square root of the strict trailing 22-session mean daily GK-plus-raw-overnight variance, with the existing GK floor `1e-10`, through that same prior session. Unlike the `R` predictor's annualized convention, this normalizer has **no factor 252**.

Target columns are `y,raw_return,event,target_end,available_date`: raw return is `log(C[t+1]/C[t])`; `y` subtracts the fixed prior-information mean and divides by its scale; `event` is float 1 only when finite `y < -1.5`, float 0 for other finite values, and missing for unknown `y`. Both target end and availability are the next observed SPX close. Current-entry weekdays use the current date; later observed dates appear only in outcome metadata. No empirical target frame was constructed by this audit, and no event/non-event support count was inspected.

The source/API outputs are `(daily,iv,source_audit)` from `load_sources`, `(features,targets)` from `build_features`, and `(transformed_train,transformed_apply,{train_n,I_mean,R_mean,skew_mean})` from `transform`. Models are declared as `frequency,constant_shape,skew_shape`; this module fits none of them.

## Interpretation limits

The sources are later archival vintages, not certified records of values available at each historical refit. The historical anchor guards against one documented discrepancy; it does not certify every date or revision. Early VIX9D observations are back-calculated. SKEW's option-implied construction and 30-day horizon do not directly give a physical one-day crash probability. SPX raw price returns exclude reinvested dividends and risk-free subtraction. The normalized event is measured against an ex ante trailing proxy, not an oracle conditional standard deviation or a universal tail probability.

No source-admission blocker was found under these archival assumptions. Class support, density validity, estimator convergence, predictive improvement, and economic usefulness remain unassessed here.
