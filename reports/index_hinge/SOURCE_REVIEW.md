# Wave 6 source and producer review

Review completed 2026-09-07 UTC, before empirical fits or predictive scoring. Only the new producer, its synthetic tests, and this report were written by this task. No earlier Python file or frozen result was changed. Source inspection was limited to dates, field validity/missingness, provenance, and byte hashes; no empirical feature–target association was calculated.

## Tests before implementation

`tests/test_index_hinge.py` was written before `src/index_hinge.py` existed. The first command, `.venv/bin/python -m unittest tests.test_index_hinge -v`, exited 1 with `ImportError: cannot import name 'index_hinge' from 'src'`. After implementation, all **18** prewritten synthetic contracts passed; scoped Ruff checks passed.

The tests establish explicit rolling formulas, one-session market lags, current-entry weekday encoding, observed-session 21/63 targets and maturity, valid zero returns, the declared Garman–Klass floor, raw-price unit invariance, ignored adjusted close, missing-row propagation without filling, and invariance of current predictors to future values or future calendar changes. They also check training-only squares/hinge centers, the raw gap's linear redundancy, a nonredundant synthetic hinge, ridge agreement with independently formed augmented least squares, population scaling, replication-invariant mean-loss normalization, exact historical means, and failure on any common zero-scale input. A synthetic post-fence forbidden CSV token and invalid post-fence parquet observation verify bounded loading.

| New artifact | SHA-256 |
| --- | --- |
| `src/index_hinge.py` | `2e3e9c02f0125c86a4c8296e218918ba3f7f172be28fa36709fe3f9c0ef5bc75` |
| `tests/test_index_hinge.py` | `094fbedc656ce7a4595566aec7ceba418a53b8526cedcdf8dbd6076a978379e6` |

These are producer checks, not independent empirical verification. The root runner owns horizon-specific complete-label admission, minimum training size, monthly fit dates, evaluation fences, and inference; this module does not run an empirical experiment.

## Exact sources and bounded coverage

All numeric reads stop at **2025-10-20**; the sealed boundary is **2025-11-03**. The parquet read selects only raw `open,high,low,close` with a date predicate. Cboe CSVs are initially read as text, their dates are bounded, and only retained value tokens are converted to numbers. Full-file hashing identifies source bytes without analyzing later numeric observations.

| Source | Provider fields | Bounded rows | First date | Last date | Missing SPX reference dates |
| --- | --- | ---: | --- | --- | ---: |
| `data/research_paths/spx_daily.parquet` | `open,high,low,close` | 4,226 | 2009-01-02 | 2025-10-20 | Reference calendar |
| `data/free_sources/raw/cboe/VIX_History.csv` | `DATE,CLOSE` | 9,040 | 1990-01-02 | 2025-10-20 | 0 |
| `data/free_sources/raw/cboe/VIX9D_History.csv` | `DATE,CLOSE` | 3,721 | 2011-01-04 | 2025-10-20 | 505, all before first observation |
| `data/free_sources/raw/cboe/VVIX_History.csv` | `DATE,VVIX` | 4,878 | 2006-03-06 | 2025-10-20 | 2 |

Selected observed fields contain no missing values, nonpositive/infinite prices, or invalid complete OHLC ranges. Source-date absence is separate from field missingness: VVIX is absent on 2010-11-11 and 2013-05-13. The later absence makes entry 2013-05-14 incomplete after the declared one-session lag. VIX9D starts on 2011-01-04, so the first date allowed by this source's lag alone is 2011-01-05. No values are forward-filled and no reference rows are removed before rolling. VIX and VVIX have 4,814 and 654 bounded dates outside the SPX reference calendar, respectively; alignment does not manufacture extra SPX sessions.

The SPX acquisition manifest records Yahoo Finance/yfinance ticker `^GSPC`, `auto_adjust=False`, acquisition `2026-08-12T17:59:14.463673+00:00`, and 4,227 source rows through 2025-10-21. The extra final date is excluded from this wave. Its historical normalization discarded incomplete OHLC rows when that source was originally built; consequently this is an **observed vendor calendar**, not certification of a complete announced exchange calendar. The new producer preserves every row present in the bounded source.

The Cboe adjacent manifests record acquisition `2026-08-12T20:45:00Z`. All four source hashes match their existing acquisition manifests; the three Cboe byte lengths also match. The exact source hashes are:

| Source | SHA-256 |
| --- | --- |
| SPX parquet | `3958fbb1eb36689df1596c26b9f0e02e3f1d4fa3b0032381e5287a2a03697bf0` |
| VIX CSV | `a34aabce269632f30904cf482986dd50b6d4cf51f2203dc51a0a9f460f3c90b2` |
| VIX9D CSV | `0d6f600ee71bf6ffb5069d0c583cbe0b1ec97df6da4e4e439d5f0f22f5616abe` |
| VVIX CSV | `f6bc726455fa3859c662875a005e97ba656b9536ae58fa6977ad24c6228e4f6a` |

Manifest hashes: `data/research_paths/source_manifest.json` = `457afd656244fc527535984874ee11313fd3156815f0d982d0c9f29ab85d244c`; VIX adjacent manifest = `51caa998f0dadc1a707182a0222f793626da93da535a31f49d66b79db0ab2917`; VIX9D = `aee15817f60f0e3cde73a55fa99b807ca4dd10b6b045d8f9b87683df7050cab0`; VVIX = `f79ecf39b83642b0df23743a1f3aa0077bbf8fc8628fdc0b5de49f0a8ccd0a52`.

## Interpretation and API contract

These snapshots are later archival vintages. A one-session lag addresses within-snapshot timing; it does not certify the values available at historical refits. The already documented Cboe provenance distinguishes VIX9D's January 2011 history from its October 2013 launch, so early training uses back-calculated history. See the prior [source review](../overnight_index/SOURCE_FEASIBILITY.md) for the official methodology references and original verification. SPX price returns exclude reinvested dividends and risk-free subtraction and are not fund fills. Squared percentage VIX divided by 10,000 and 252-scaled trailing OHLC variance remain different measurement constructions; their log difference is not a measured conditional variance risk premium.

`RAW` has 15 columns: `const,I,R,ret_d,ret_w,ret_m,ret_q,lr_d,lr_w,term,lvvix,entry_dow_1..4`. `BASE` adds `I_square,R_square`; `ALL_FEATURES` adds `hinge`. The transform computes all three centers using exactly the supplied training rows, before model population standardization. All 17 non-intercept columns must have scale strictly above `1e-12`, including the hinge; otherwise every model aborts. The fixed intercept necessarily has zero raw variance and is exempt. There is no raw-gap augmentation, tuning, clipping of forecasts, or silent row selection.

`load_sources(protocol,root)` returns `(daily,iv,source_audit)`. `build_features(daily,iv)` returns raw features plus `feature_cutoff_date` and a dictionary keyed by 21/63 whose target frames contain `y,target_end,available_date`. Target availability equals the horizon's completed close. `transform(train,apply)` returns two 18-column frames and the training-center audit. `fit_predict(train,y,apply)` returns `predictions`, `model_audit`, and `transform_audit`; models are `mean,baseline,hinge`, and ridge minimizes mean squared error plus `.01` times standardized slope norm squared with an unpenalized intercept.

No source-validity blocker was found for the declared **archival** experiment. Historical-vintage certification, economic execution, and predictive usefulness remain unestablished by this review.
