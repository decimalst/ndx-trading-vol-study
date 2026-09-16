# Treasury pre-fit support audit

`audit_support(reference_calendar, features, targets, event_audit, *, event_tenors, config=None)` returns a JSON-safe report and performs no fit, loss calculation or source financial conversion. The supplied reference calendar remains unchanged. `event_tenors` is the complete ledger metadata map from every event ID to confirmed original tenor or None; its keys must exactly equal the producer event-audit IDs. The frozen feature builder does not include original tenor in its audit rows, so this explicit map is required. Extra donor/share/pending provenance remains opaque to this component; the separate feature verifier authenticates those calculations and pending intervals.

Configuration defaults are the fixed unregistered Treasury protocol's forecasting dates and support floors. A generated-test override has exactly `forecast`, `stability`, `support`: forecast is the six-key pipeline dictionary; stability is two ordered nonoverlapping evaluation intervals; support has exactly the ten integer floor names in the YAML. Every floor must be a positive exact integer. Overrides are for internal generated testing; the registered empirical caller must bind immutable production constants.

## Input validation and counting

The component reuses only the frozen pipeline's source-independent `_configuration`/`_inputs` and date-validation primitives, read-only. Those functions validate all32 common feature names, real finite-or-missing values, the exact intercept, full aligned native calendars, positive observed targets, exact fifth-session target endpoints, and previous-full-session Treasury cutoff with the2010 floor. No producer fitting function is called. An independently supplied calendar must agree exactly after native time-unit normalization.

Event metadata requires event_id, auction_date, source_status, origin, status, origin_masked and mask_reason. Dates are bounded by2010-01-01 and2025-10-20; native activation timestamps must lie on the reference calendar and strictly follow the auction. Exact known/unknown/excluded producer status combinations are enforced, with confirmed tenors for known events. Excluded/outside-calendar events keep unknown activation metadata and never count. Supported events at masked origins cannot create complete source rows. The map must retain every audit ID, including unknown and excluded events.

Every finite auction_count and available tenor-count column is checked against supported, unmasked audit identities at that calendar position. Counts must be nonnegative integers. The omitted two-year category is established by the actual two-year event identities and the total-minus-other-tenors agreement. A row enters support only when all32 common features and cutoff are complete. An activation date is auction_count>0, regardless of zero or cancelling dealer_surprise; simultaneous auctions count once as an information date and individually within their original-tenor identity counts. A source-complete auction on a missing-market-feature row does not count toward common model support.

All monthly first queries and application counts are planned from common features before label eligibility. Each scheduled training cohort then uses earlier origins with observed positive targets ending at or before the preceding full session. All scheduled months must pass1000 training rows,200 activation dates and24 identities of each original tenor. No fit is performed, skipped to repair support or replaced by another estimator.

Phase support counts the common scored-eligible cohort: development targets end by the development end, evaluation targets by source_end. Each phase requires505 daily rows,104 activation dates and12 events per tenor. The two stability slices subset the evaluation scored cohort by origin dates; they do not impose an invented slice-target fence. Each slice requires252 daily rows,52 activation dates and6 events per tenor. Every phase offset is original full-calendar position modulo5 before any masking, with63 daily and20 active dates required. Offset tenor counts are reported but have no invented tenor floor.

## Report schema and whole-attempt outcome

Top-level keys are exactly `status, passed, monthly, phases, slices, offsets, discrepancies`. Status is SUPPORT_PASS or INSUFFICIENT_DATA. Malformed source/schema/calendar/count contradictions raise ValueError; genuine support shortfalls return every count and failure without short-circuiting, fitting or rewriting the source.

Phases are a development/evaluation dictionary. Each row has literal origin_start/end, daily_n, activation_dates, event_n, events_per_tenor, passed and failures. The six tenor dictionary keys are strings2/3/5/7/10/30; their counts sum to event_n. Slices are an ordered two-row list with slice_index0/1; offsets map each phase to five ordered rows with offset0..4 and the literal parent-phase bounds.

Monthly rows have month, status, fit_origin, training_cutoff, requested_n, application_n, train_n, train_activation_dates, train_event_n, train_events_per_tenor, passed and failures. Status is SUPPORTED or INSUFFICIENT_DATA for an actual schedule, NO_COMPLETE_ORIGIN or NO_REQUESTED_ORIGINS for an explicit no-fit month. No-fit dates and all four train fields are None, passed isTrue and failures empty; this is not a fit or an invented monthly coverage gate. Phase/slice/offset gates still expose lost support.

Each discrepancy has exactly scope, metric, observed and required. Scopes are monthly:YYYY-MM, phase:development/evaluation, slice:0/1 or offset:phase:0..4. Metrics use literal support keys, minimum_train, or a per-tenor suffix such as training_events_per_tenor:2. Discrepancies are deterministic: monthly order first, then development phase/offsets, evaluation phase/offsets, and slices. Any discrepancy makes the whole attempt INSUFFICIENT_DATA; previous comparisons must remain counted by the caller.

## Prewritten evidence and literal fixture expectations

Fourteen generated methods preceded implementation; SUPPORT_RED.log preserves the missing-module failure. The first implementation passed those14 in1.117seconds. A fifteenth prewritten regression then demonstrated that a masked current event could be forged into a complete certified-zero row; SUPPORT_MASK_REGRESSION_RED.log retains its assertion failure before the implementation added the explicit blocked-origin check. Final15 methods pass in1.187seconds, with scoped Ruff lint/format checks clean in SUPPORT_GREEN.log. No historical sources, cohorts, support or model calculations were run.

The fixture has200 synthetic full sessions, six simultaneous original-tenor identities at each positive even position, and zero dealer surprise. Its first requested query is position60; the training cutoff is59, so labels ending at59 admit positions1..54: exactly54 rows,27 activation dates and27 events of each tenor. Development origins60..109 retain positions60..104 after their target fence:45 rows,23 activations,23 events per tenor, with full-calendar offset daily counts[9,9,9,9,9] and active counts[5,4,5,4,5]. Evaluation110..179 has70 rows/35 activations; its two slices have35/18 and35/17. Removing the market feature at position70 reduces only original offset0, never renumbers later positions. These expected counts are explicit constants in the prewritten tests, not output copied from the implementation. Tests also cover target-blind scheduling, exact maturity boundaries, missing query/future labels, no-fit months, all-gate failure retention, unknown/excluded metadata, incorrect counts/tenors, typed configuration and input preservation across native datetime units.

## Stable files

| File | SHA256 |
| --- | --- |
| `src/treasury_dealer_support.py` | `5de6028a23b00c0bf496a99c2f1be2cf8b027661864c87a9bf570aeeed4cbb7b` |
| `tests/test_treasury_dealer_support.py` | `a824c022c6eb8699e34143f18cba37cbb647159b0cde8e23be436f61d49a8ee8` |
| `reports/treasury_dealer/predictive_prefit/SUPPORT_RED.log` | `bf3bb82cf3db349c080b60b1bee2d0957f2b6440240dc93515f41e716184933a` |
| `reports/treasury_dealer/predictive_prefit/SUPPORT_MASK_REGRESSION_RED.log` | `f4bd7805c61e26796391004d349c4835f740875dbcab55449700d9c8e5f469bd` |
| `reports/treasury_dealer/predictive_prefit/SUPPORT_GREEN.log` | `372e7bf81feb48203923924a62431064295c08141a6d42750972e6e6d4d46df8` |
