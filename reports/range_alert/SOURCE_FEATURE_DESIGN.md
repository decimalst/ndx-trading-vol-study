# Range-alert source and feature contract

This document specifies the new source/feature implementation before empirical source admission, new event counts or fits. It follows the [prospective range-location proposal](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/profiled_quarter/NEXT_RESEARCH_DIRECTION.md). The implementation reads no files at import and contains no empirical runner, support-counting routine, fitted model or scoring function. Only the new feature module, its prewritten synthetic tests and current report/log files were written; earlier sources, code, protocols and reports remain unchanged.

## Sources and provenance

`load_sources(protocol, root=ROOT)` returns `(daily, iv, audit)` by explicitly calling the frozen [tail-shape source loader](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/tail_shape_features.py). Its underlying [SPX loader](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/index_hinge.py) is unchanged. The runner must first register and stage single hash-checked byte snapshots at the declared relative paths. The wrapper does not replace this runner responsibility with freshly observed hashes.

| Source key | Retained source path |
| --- | --- |
| `daily` | `data/research_paths/spx_daily.parquet` |
| `vix` | `data/free_sources/raw/cboe/VIX_History.csv` |
| `vix9d` | `data/free_sources/raw/cboe/VIX9D_History.csv` |
| `vvix` | `data/free_sources/raw/cboe/VVIX_History.csv` |
| `skew` | `data/raw/SKEW_History.csv` |
| `skew_source` | `data/raw/skew_daily_source.json` |
| `skew_derived` | `data/raw/skew_daily.parquet` |

The frozen loader selects bounded raw OHLC and preserves the observed SPX calendar. It parses CSV date tokens before numerical values and retains the October 20, 2025 numerical fence and November 3 protected boundary. Raw SKEW remains the predictor source; its original derived parquet is checked only for bounded equality. Its required raw hash, `becbf3f7510de66a736495df29a84ba1911d362402f05f6c46dedd5e9b971492`, and original August 13, 2018 anchor remain binding. There is no fetch, alternate vintage, filled observation or calendar intersection.

The retained [source review](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/tail_shape/SOURCE_REVIEW.md) documents archival Yahoo/yfinance `^GSPC` raw OHLC and Cboe extracts. Exact historical delivery latency and immutable vintages are not established. Early VIX9D is back-calculated; SKEW is an implied index construction, not a physical one-day event probability. The observed vendor calendar is not independently certified as a complete exchange calendar. This task has not newly parsed or counted historical source observations.

The source audit preserves all inherited paths, hashes, source-date/missingness metadata and `skew_provenance`. It removes the old study's `gap_interpretation`, replaces `raw_columns`, `normalization`, `target_interpretation` and `timing_assumption`, and adds `baseline_columns`, `range_extremity_interpretation` and `arithmetic_policy`. Exact strings live in the new wrapper and are part of the independent reconstruction contract. The old normalized signed-return target description is not retained.

## Information and event timing

`RAW` is the frozen tail-shape RAW19 followed by `intraday, intraday_sq, overnight, overnight_sq, range_extremity`. `ALL_FEATURES=RAW` has 24 columns. `BASE` is the old BASE22, including its three training-centered squares, followed by the four nuisance columns; it has 26 columns. `MODELS=('baseline','recent_frequency','location')`.

For a current observed session s, define intraday `d_s=log(C_s/O_s)`, raw overnight `o_s=log(O_s/C_(s-1))`, and `V_s=max(.5*log(H_s/L_s)^2-(2*log(2)-1)*d_s^2,1e-10)+o_s^2`. Overnight uses exactly the previous row of the full SPX calendar; missing rows are not removed. The four nuisance features at origin t are `d_(t-1), d_(t-1)^2, o_(t-1), o_(t-1)^2`.

For a complete valid OHLC bar with positive range, `Z_s=(log(C_s/L_s)-log(H_s/C_s))/log(H_s/L_s)` and `E_s=Z_s^2`. The candidate is `E_(t-1)`. It is a recorded close-location statistic; it does not measure transactions, auction imbalance, an intraday path or an executable quote. The same-GK/different-location synthetic example establishes an algebraic degree of information, not statistical orthogonality or predictive success.

The exact target reference at origin t is `reference_t=math.fsum(V_(t-22),...,V_(t-1))/22`, using precisely 22 finite positive observations. The threshold is then computed as `2*reference_t`. The target is float one iff `V_(t+1)>threshold_t`, float zero for equality or a smaller known value, and NaN if the reference or next-session variance is unknown. Compensated summation is fixed only for this new binary threshold; old rolling controls retain their frozen pandas arithmetic. Threshold/reference nonfinite results or nonpositive computed values abort. No epsilon, fitted quantile or replacement threshold is used.

`build_features(daily, iv)` returns exactly `(features, targets)`. Feature columns are RAW24 followed by `feature_cutoff_date`, the preceding observed SPX session. Target columns are exactly `y,target_end,available_date`; both dates are the next observed SPX session at which the variance is fully observable. The frozen tail feature builder supplies only RAW19 and the prior cutoff; its irrelevant normalized-return target is neither returned, counted, filtered on nor used by the new model. Current entry weekdays are the only current-date market-control exception. Future observed dates appear only in target metadata.

The target has no annualization or conditional-standard-deviation normalization. It means risk exceeds twice its trailing reference, not necessarily twice the immediately preceding day's risk. It remains a full-session daily OHLC proxy, not measured high-frequency integrated variance, an option payoff or a signed crash. Known labels are retained even when missing IV or unknown location makes the corresponding predictor row unusable; downstream common-sample and causal recent-frequency rules remain the runner/model contract.

## Missingness and numerical admission

Every observed OHLC value must be finite and positive. Every available high/low endpoint inequality is checked, including partially missing bars. Ratios are checked whenever their own operands exist, even if another OHLC field is missing. Nonfinite ratios/logs/derived values, nonzero division or square results that underflow to zero, or a computed location outside the literal `[-1,1]` domain abort. Nonzero finite subnormals are not categorically removed. The implied-variance and legacy variance-log primitives are validated before the frozen builder can turn a numerical failure into missing data.

Location requires the entire OHLC bar. A valid zero range yields unknown E without deleting its date; the existing GK floor can still make its variance and event label valid. A positive recorded range with zero computed log width is invalid. No location clipping or range-floor substitution is allowed. The inherited GK floor is the one intentional measurement floor. Missing observations remain in all strict windows, and no new common mask is applied inside this module.

`session_components(daily)` exposes current `intraday,intraday_sq,overnight,overnight_sq,variance,range_extremity` for arithmetic reconstruction. `build_targets(variance)` exposes the strict fixed event/maturity calculation. Neither helper counts classes or chooses rows.

## Training transformation

`transform(train, apply)` returns `(x_train, x_apply, e_train, e_apply, audit)`. The two baseline outputs are pandas BASE26 frames and the two location outputs are centered pandas Series, all retaining the supplied row indexes. All supplied RAW24 values must be finite and real, the intercept must literally equal one, E must lie in `[0,1]`, and at least two training rows are required. This primitive does not select the training cohort or check label maturity.

On exactly those training rows, separately center I, R and SKEW, create their squared terms, then use training population means and standard deviations for all 25 baseline slopes. The intercept has center zero and scale one. Every slope scale must exceed `1e-12`; a failure aborts rather than dropping a variable. Duplicate information and collinearity are retained for the downstream penalized solver. These rules do not replace the model's 1,000-row and class-support gates.

E is centered on its training mean with fixed scale one. Exact constant training E uses its first value as the center, ensuring an exactly zero centered training Series despite mean-rounding residuals. It retains the fold; the model must issue the canonical zero scalar coefficient. Application E is centered at the same value. Application values never affect centers, curvature, scales or the constant-history flag.

The JSON-safe audit has `train_n`, `columns`, `means`, `scales`, `curvature_means={I,R,skew}`, `range_extremity_mean`, `range_extremity_scale=1`, and `range_extremity_constant`. The means/scales arrays follow BASE26, including intercept entries zero/one.

## Prewritten evidence and limits

The [initial absent-module RED](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/range_alert/features_initial_red.txt) was recorded after the tests existed and before implementation. All **22 prewritten synthetic tests pass**, as recorded in [implementation checks](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/range_alert/features_implementation_checks.txt); scoped Ruff checks also pass. Tests cover exact schemas and frozen controls, bar arithmetic, same-GK/different-E identification, compensated threshold ties, strict historical windows and maturity, missing/zero ranges, IV-independent labels, future value/calendar mutation, positive price-unit changes and ignored adjusted close, range endpoints, invalid partial bars and arithmetic, training-only population scaling/curvature, collinearity retention, exact constant E, and pinned-source fencing using temporary files with forbidden postfence numerical tokens.

| Artifact | SHA256 |
| --- | --- |
| New `src/range_alert_features.py` | `dcd0f713547aa46410f2d37dad7219008ba1c3febb32f742ee11fef60f637203` |
| New `tests/test_range_alert_features.py` | `393dcce4872f85dda46f3e55aae6ec136ec23f72f8e81e3e5f9df6ffe2fa07e4` |
| Unchanged `src/tail_shape_features.py` | `ae62f72530f6dc7014a699f9e3291a4aa49e461756a0e8a150ee0f3fcb234921` |
| Unchanged `src/index_hinge.py` | `2e3e9c02f0125c86a4c8296e218918ba3f7f172be28fa36709fe3f9c0ef5bc75` |

No historical range/event count, model fit, new association or predictive score has been computed for this implementation. Empirical support, numerical convergence on historical folds, independent reconstruction and predictive usefulness remain unassessed. Source admission, the common training/query cohort, causal frequency recurrence, all-fold support and the two-comparison failure ledger remain mandatory downstream responsibilities.
