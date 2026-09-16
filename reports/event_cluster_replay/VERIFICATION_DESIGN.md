# Prospective replay verification: exact state-date semantics

**Specified before historical replay; no historical numerical tables or fitted coefficients have been inspected.** This new helper leaves the failed wave20 implementation, outputs, numerical objectives, solvers, tolerances and comparison family unchanged. It can verify supplied retained outputs under a separately registered structural date contract. Passing this helper does not establish predictive value; the separate replay entry must also complete admission, scoring, inference, accounting and publication checks.

## Exact exception and preserved checks

The only newly permitted equivalence concerns these seven application-state columns, in this order: `origin`, `feature_cutoff_date`, `source_fit_origin`, `window_first_available`, `window_last_available`, `window_first_origin`, `window_last_origin`.

Both actual and expected columns must be native NumPy-backed, timezone-naive `datetime64[ms]`, `datetime64[us]` or `datetime64[ns]`. Strings, object arrays of Timestamp objects, timezone-aware or Arrow-extension dates and all other units are inadmissible. Nonmissing values must be midnight. NaT masks must agree exactly. Convert temporary copies to nanosecond units with lossless conversion and verify the roundtrip to each original unit; overflow or any changed value is a failure. The normalized integer nanosecond instants must agree exactly. There is no rounding, clipping, timezone removal, inferred date parsing or coercion of unknowns. Dates outside the nanosecond representable range fail this declared comparison instead of wrapping.

After only those seven columns are normalized on temporary copies, apply the unchanged frozen exact-frame comparison. Numeric, boolean, string and other dtypes, values, shapes, row order, index values/names and column order/names retain that check. No blanket `check_dtype=False` or generic datetime normalization is used. Input frames are not modified. The full memory table, its DatetimeIndex, scored panels, original control records and every other original check remain under the frozen rules; they receive no new date-unit exception. JSON dates in fit metadata remain their exact original strings.

The actual failed wave20 traceback is a reason to pretest this prospective contract, not evidence that later independent model checks will pass. This helper does not repair or relabel the failed study.

## Supplied-table API and numerical reuse

The signature remains:

`verify_pipeline(features, targets, upstream_panel, upstream_fits, upstream_states, original_index_config, panel, newfits, newstates, memory, support)`.

The eleven arguments are supplied admitted tables/records. This module loads no files and calls no original or retained producer optimizer, forecast generator or scoring implementation. It preserves the frozen independent pipeline reconstruction with one explicit replacement at the application-state comparison. The original independent feature/history primitives, source geometry replay, class/support and monthly-cohort rules, Brier construction, nuisance root solve, scalar bisection, coefficient/objective/probability/full-gradient tolerances and exact zero-correction nesting are reused unchanged. Every original application, including unscored applications, is retained. Structural checks still precede the independent stage solves. There is no monkeypatch of frozen implementation globals.

An accepted call reports `forecasts_verified=4*n`, `retained_candidate_scored_forecasts=2*n`, `original_control_forecasts=2*n`, `newly_generated_forecasts=0`, `new_producer_fits=0`, `retained_monthly_schedules=k`, `independent_stage_fits_verified=2*k`, `original_monthly_fits_replayed=k`, `common_scored_origins=n`, `common_application_origins=a`, `application_states_verified=a`, `memory_rows_verified`, and `monthly_stage_audits`. The term independent fits means verification optima for the retained objectives, not newly generated producer models or forecasts.

The additional `date_comparison_audit` records the policy `lossless_naive_midnight_state_dates_v1`, comparison unit `ns`, allowed units `[ms,us,ns]`, ordered columns, row count, frozen-exact index and other-column policies, and one field record per column with actual/expected units, NaT count and successful losslessness/instant equality flags. It contains no date values or predictive scores. Counts and field identities are exact; any separate comparison of repeated optimizer audits must retain the predeclared numerical tolerances rather than require identical solver evaluation counts.

## Tests written before implementation

The initial new test module contains five primitive contract tests and six full-transport tests. Its first run failed because the new helper module did not exist; `verification_initial_red.txt` retains that output.

The full transport fixture first specifies the original generated source and target calendars/date columns in ms, us or ns. It then serializes and reloads those inputs, runs the actual frozen original producer and wave20 producer on generated observations, serializes/reloads original and retained panel/state/memory tables, and calls the unmocked new independent helper, including all independent stage solves. JSON audit records are roundtripped too. Final and phase-boundary unscored applications remain present. No historical values participate.

The prewritten tests require ms and ns versions to falsify the frozen state comparison before its independent solve loop; all three units must pass the new contract. They reject changed days, subday offsets, timezones, object/string date columns, unsupported units, overflow, changed NaT masks, index/name/order changes, non-date dtype changes and changed numeric bits. Further tests preserve strict memory/index comparison, reject saved coefficient and coherent unscored-probability tampering through real independent solves, and forbid original/retained producer calculators during successful verification. Mocking in rejection tests only observes that the solver is not reached; it never supplies a successful numerical proof.

The first implementation passed all11 tests in49.126 seconds; `verification_first_implementation.txt` preserves that output. The ms/us/ns cases each executed the complete generated original/retained production, Parquet/JSON transport and unmocked independent numerical reconstruction. The prewritten ms/ns falsifiers also confirmed that the unchanged wave20 helper rejects the retained state representation before reaching independent stage solves. No numerical implementation, source-field scope, test assertion or tolerance needed adjustment after this first run.

Scoped formatting/lint is applied only to the two new files, followed by the final unchanged-contract test run recorded in `verification_final_green.txt`. No historical admission, new empirical count, fit, score or replay has been run. All earlier files and the failed wave20 record remain unchanged.
