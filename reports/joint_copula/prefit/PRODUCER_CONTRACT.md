# Joint-copula producer, source and scheduling contract

Prospective generated-only implementation. `PRODUCER_RED.log` records missing-module failures after the sixteen meaningful tests were written and before any of the three new producer modules existed. No empirical sources, features, support, fits or predictions were inspected. Frozen source files and earlier producers are unchanged.

## Source interface

`joint_copula_inputs.read_sources(root,pins,*,source_end='2025-10-20')` returns `(qqq,spx,iv,audit)`. `pins` is exactly the six original source paths in module `SOURCES`, mapped to literal lowercase SHA256. All six byte snapshots are captured and authenticated before any decoding. All full date columns are then parsed and checked before any numerical field. Parquet dates must be native date/timestamp fields; Cboe DATE strings use exact MM/DD/YYYY. Full dates must be increasing, unique, nonmissing, timezone-free midnight dates. Future date metadata is checked; numerical rows are bounded through source_end, never later than October20,2025. The same checked snapshots supply numerical decoding, even if files later change. The outer runner must independently rehash original sources at final publication.

QQQ and SPX retain separate original bounded calendars and only raw open/high/low/close. The IV frame is the sorted union of vxn/vix/vix9d/vvix dates, retaining gaps. Actual price/IV numbers are real, nonboolean and pass the frozen positivity/OHLC contracts. No source filling, adjusted prices, compressed common-date sampling or new vintage is introduced. Audit keys are `status='VERIFIED_BOUNDED_SOURCE_BUFFERS'`, `source_end`, `source_sha256`, `source_metadata`, `numerical_columns`, `snapshot_semantics`. Per-source metadata has raw_dates, bounded_dates and excluded_after_ceiling_dates.

## Shared marginal and dependence interface

`joint_copula_models.fit_predict(train,targets,apply)` returns `(predictions,audit)`. Input features contain the exact34 ordered `joint_risk_features.ALL_FEATURES`; targets contain y_qqq,y_spx in exactly matching training order. Individual signed and zero returns remain valid. The existing `joint_risk_models.transform`, `_mean_fit`, `MARGINAL_COLUMNS`, `_serializable` and `macro_second_moment.fit_second_moment` are reused unchanged. Thus corr22's centered square, mean ridge, scaling, positive residual-second-moment fit and their original audit schemas are preserved.

For each asset, construct current-fit residuals and fit the positive moment on their squares using the frozen staged convention. Dependence coordinates are exactly `z=residual/sqrt(.75*h_train)` for the fixed variance-standardized t8 marginal. Nonzero residual-square or coordinate underflow is fatal. Each family calls the new root density helper on the same two-column z: `fit_dependence(z,'t8')` and `fit_dependence(z,'gaussian')`. No row weighting, retrospective issued-residual label or common correlation optimum is substituted.

Prediction keys, in order, are `t8_copula,gaussian_copula,independence`. Each maps to mu(n,2),h(n,2),rho(n), with columns QQQ then SPX. All arms have bit-identical shared mu and h, finite locations and strictly positive h. Dependence rho is bounded by .995; independence rho is exactly zero. The producer computes no realized density or loss. Root's stable density helper and the scorer own those calculations.

The model audit has exactly `moments,transform,residual_staging,dependence`. `moments` has qqq/spx, each with the original mean and variance dictionaries; `transform` is `{'corr22_mean':...}`; residual_staging is `current_fit_training_residuals`. `dependence` contains t8_copula/gaussian_copula, each `{'rho':float,'audit':root_density_certificate}`. Independent verification can reconstruct marginal training predictions and h from these exact saved coefficients, then check the dependence certificate on the corresponding residual coordinates.

## Full-calendar pipeline

`joint_copula_pipeline.produce(qqq,spx,iv,config=None,*,measurement_callback=None)` returns exactly seven keys: features,targets,applications,panel,coverage,schedules,fits. Before any numerical measurement, all three source indices are validated through the fixed ceiling. The inherited joint-risk full-reference measurement audit is computed before any feature/target mask. If provided, the callback receives that audit before require_measurement; the runner can preserve finite gate failures. Existing feature construction retains its own measurement checks and source-predecessor rules.

Features and targets are exactly the frozen joint-risk builder outputs on the entire SPX reference index. The34 feature columns plus feature_cutoff_date are unchanged. Target columns are y_qqq,y_spx,target_end,available_date. Feature cutoff must be the preceding full reference session. Both target dates must be the next full reference session. A known component without that endpoint is invalid; a missing component makes the scored pair unknown. Dates after the endpoint ceiling are not admitted.

Config has exactly `origin_start,origin_end,source_end,development,evaluation,minimum_train`. Production defaults are origin_start2016-01-04,origin_end2025-10-17,source_end2025-10-20,development[2016-01-04,2019-12-31],evaluation[2020-01-01,2025-10-20],minimum_train1000. Literal phase endpoints bound target maturity; origin_end separately limits requested forecast origins. Smaller generated windows/floors are supported only by the pure component; root's exact protocol enforces real execution settings.

Plan every requested civil month before model fitting. The first common34-feature-complete origin is selected without its query target. Training is every earlier complete origin with both observed return labels ending no later than the preceding full reference session. There is no warmup, cold-start fallback or insufficient-month deletion. Every complete requested origin gets all three distributions, including unknown future labels, immature endpoints or phase-end labels that cannot be scored. Offsets use original SPX position modulo5. Scored origin and next-session label must lie inside the same phase.

## Exact output tables and audit

Dates in output tables are datetime64[ns], with NaT for absent linkage. Count fields use pandas nullable Int64; status flags are bool; numerical predictions/labels are floats. Applications, panel and coverage are chronological, with per-origin model order t8_copula,gaussian_copula,independence. Schedules follow month order; fit audits follow attempted month order.

- Applications are **long**: `origin,model,horizon,feature_cutoff_date,mu_qqq,mu_spx,h_qqq,h_spx,rho,fit_origin,training_cutoff,train_n,phase,offset`. Horizon is1 and phase is development/evaluation/outside_phase.
- Panel has the same columns followed by `target_end,available_date,y_qqq,y_spx`. It contains no saved loss; all labels, marginal predictions and monthly metadata are shared across the three rows at each origin.
- Coverage: `origin,phase,feature_complete,missing_features,feature_cutoff_date,offset,target_end,available_date,target_observed,target_within_phase,issued,scored,status,fit_origin,training_cutoff`. Missing feature names are pipe-joined in fixed raw feature order. target_observed requires both finite components. Status precedence: incomplete_features,attempt_aborted,outside_phase,target_not_mature,missing_target,target_after_phase_cutoff,scored. Unissued fit/cutoff linkage is NaT; target and feature dates remain known independently.
- Schedules: `month,status,fit_origin,training_cutoff,requested_n,feature_complete_n,application_n,train_n`. Status is not_attempted/no_complete_origin/no_requested_origins/insufficient_training/fitted/execution_failed. application_n counts issued origin identities, not three model rows. No-query months have NaT fit dates, null train_n and application_n0.
- Fits: one JSON-native dictionary per attempted month with `month,status,fit_origin,training_cutoff,train_origins,train_positions,train_n,planned_application_origins,application_origins,model_audit`. Dates/lists are ISO strings; positions refer to the full SPX calendar. Model audit is null until fitting returns, and actual application_origins remains empty until issuance succeeds.

`PipelineExecutionError` exposes produced/month/stage/measurement_audit. Measurement-stage failures retain the audit and empty seven-key frames. Expert support or model/issuance failure retains full built inputs, all schedules, attempted fit records and earlier issued applications, with an empty scored panel and explicit aborted coverage. Stages are measurement/feature_construction/feature_alignment/monthly_schedule/training_support/fit_predict/prediction_validation/application_issuance/result_assembly. Generic post-measurement errors retain each available input and the most recent completed monthly snapshots; final assembly errors preserve all issued applications and fit audits, while clearing every scored flag and the panel. No failed attempt returns a successful partial panel.

## Test scope

The four model tests verify exact shared predictions, reconstruct current-fit t8 coordinates from saved marginal coefficients, check both original marginal gradient bounds, unit scaling, application independence, valid individual zero returns, invalid types/order/scale and zero total residual risk. Dependence fitting is stubbed in these tests so they isolate the marginal-to-dependence interface; root separately tests the actual density optimizer.

The nine pipeline tests use generated raw OHLC/IV with the actual frozen feature builder and an explicit simple model fixture. They cover full measurement before masks, future dates before arithmetic, full-calendar maturity and offsets, query-label blindness, future source poisoning, unknown source predecessor, unscored issuance, feature-construction/alignment failures, and retained monthly diagnostics even when final coverage or panel assembly fails. The five source tests use private temporary invented CSV/parquet payloads with forbidden future numeric tokens, proving all-hash/all-date-before-numeric ordering and checked-snapshot decoding. Full real-density generated integration is a separate independent verification task; these component checks do not claim historical fit validation.

## Stable generated evidence

`PRODUCER_FAILURE_RETENTION_RED.log` preserves the two additional prewritten failure-retention regressions failing before the generic post-measurement wrapper. `PRODUCER_FINAL_GREEN.log` records all 18 generated component tests passing in 0.907 seconds, scoped Ruff clean, and six files passing format checks. Original missing-module RED and both intermediate implementation logs remain intact. No actual source numbers, historical arrays, support, fit or score was accessed.

Stable file SHA256 values:

- src/joint_copula_inputs.py: `1d6ef58f203682dae6dbdf1f55ac9a50f34308c05bcd647d63dab385bb689cec`
- src/joint_copula_models.py: `4ab1eb02fe7703cf8639f062e9fb976f7ee22ddd4cc13451566c0e4660558c86`
- src/joint_copula_pipeline.py: `0ea290b36f15adfb5cc59147ae7132848358ff84795f3c6008a2e089bc8eecac`
- tests/test_joint_copula_inputs.py: `d3bf6f3ef583527cf224e7af051af1a8af1f281070c147443a88591a02b9e308`
- tests/test_joint_copula_models.py: `a9adbe3c3b79a8e2f5fad232be4d76a18c23e90e0359334a695d6e92e3a68652`
- tests/test_joint_copula_pipeline.py: `bfb1cba23fb987dfb0e3182d5fea08a0bf62ba1cfe5fd3b1b4c53a1ce29ad41c`
- reports/joint_copula/prefit/PRODUCER_FINAL_GREEN.log: `2afecaee6a97064e5587a5204d5177fd99e05588e5d55c3c94710912b0c99067`
