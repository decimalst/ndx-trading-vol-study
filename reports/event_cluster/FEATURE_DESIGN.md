# Fixed event-arrival clustering features

This is a prospective, generated-test implementation record for wave 20. It does not contain historical event counts, predictor/target associations, fitted models or scores. The original wave-18 risk-alert labels, source bounds and source limitations remain unchanged. No prior code, test, protocol or report was edited.

## Fixed mathematical summary

Read exactly 22 known binary labels in increasing **availability-date** order, ending at the preceding observed SPX close. Write the oldest as `e_1`, the newest as `e_22`, and define:

`N = sum(e_i)`; `b = e_22`; `C = sum(e_(i-1)*e_i, i=2..22)`.

The five nuisance features, one fitted candidate and one diagnostic are:

| Exported feature | Exact operation order |
|---|---|
| `event_fraction22` | integer `N` divided once by `22` |
| `event_fraction22_sq` | integer `N*N` divided once by `484` |
| `last_event` | exact binary `b` |
| `expected_adjacency22` | integer `(N-b)*(N-1)` divided once by `441` |
| `linear_recency22` | integer `sum((2*i-23)*e_i, i=1..22)` divided once by `462` |
| `adjacency_fraction22` — fitted candidate | integer `C` divided once by `21` |
| `excess_adjacency22` — diagnostic only | integer `21*C-(N-b)*(N-1)` divided once by `441` |

All counts and numerators are Python integers. The squared fraction is not obtained by squaring a rounded fraction; the excess diagnostic is not obtained by subtracting two rounded floating features. Exact zero numerators therefore give exact positive floating zero, without clipping or a tolerance. These bounded integer divisions cannot produce a nonzero floating underflow. Input validation occurs before integer conversion.

Given `N,b`, the first 21 positions contain `q=N-b` events. Under uniform permutation of those positions, the expected number of internal adjacent pairs is `q*(q-1)/21`; the final pair contributes `b*q/21`. Hence

`E(C | N,b) = (N-b)*(N-1)/21`.

Dividing by 21 gives the registered expected-adjacency fraction. This is a combinatorial reference, not an empirical claim that market events are exchangeable. Its mean-zero property is conditional on `N,b` under that reference, **not** conditional on linear recency or the full market state. Including the expected fraction in the matched nuisance block explicitly includes its `N*b` interaction. It does not prove causal or statistical orthogonality to every existing predictor.

The linear-recency weights sum to zero and increase from oldest to newest. Their inclusion makes adjacency a different proposed increment from simply weighting newer events more heavily. For example, event positions `{2,3,8}` and `{1,5,7}`, with all other positions zero, have identical count, latest label, expected fraction and linear recency. Their adjacent-pair counts are one and zero, respectively. Fitted candidate values are `1/21` and zero; descriptive excess values are `5/147` and `-2/147` before floating representation. This is a synthetic algebraic example, not a market pattern count.

## Pre-empirical correction: fit raw adjacency

The initial synthetic implementation exposed excess adjacency as the fitted feature. During review, before registration or historical feature construction, a mathematical attribution problem was identified: adding a penalized coefficient on `C/21-G` to a frozen nuisance offset also introduces a constrained negative coefficient on `G`. It could change predictions through the already-known reference alone, even when `C` is constant. Including `G` in an earlier frozen stage does not eliminate that problem.

The selected fitted feature is therefore **raw `C/21`**, with `G` retained among the nuisance controls. Exact-integer excess remains a separately named diagnostic and is explicitly absent from `FEATURES`. This prevents the new raw-adjacency coefficient from directly reweighting a subtracted reference. It does not establish structural causality or immunity from all baseline misspecification.

A regression was written before this implementation change. The strengthened fixture has events at `{2,5}` versus `{1,4,22}`: count, latest label, expected fraction, linear recency and descriptive excess differ, while both adjacent-pair counts are zero. The old fitted feature failed the required constant-zero assertion (`-0.0045351473922902496 != 0.0`). Both the [first regression red](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/event_cluster/raw_adjacency_regression_red.txt) and [strengthened regression red](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/event_cluster/raw_adjacency_full_regression_red.txt) are retained. The revised fitted feature is exactly zero in both cases. All-one histories have raw adjacency one, all-zero histories zero; the separate model's exact-constant centering rule retains either constant fold with a zero correction.

## Full-calendar chronology and missingness

`build_memory(targets)` accepts exactly the columns `y,target_end,available_date` on a nonempty, unique, ascending, normalized, timezone-naive `DatetimeIndex`, bounded through `2025-10-20`. Both date columns must equal the exact next-reference-date series, including a terminal `NaT`. A finite terminal label with no next reference close is rejected. Known labels must be numeric zero or one; `NaN` is the only unknown. Observed infinities, nonbinary values, strings, objects, complex and boolean arrays are rejected. Date strings, intraday timestamps, timezone-bearing dates, duplicate dates and inconsistent maturity are rejected rather than repaired.

At zero-based application position `i>=23`, the literal 22 arrival closes occupy positions `i-22..i-1`. Their labels belong to target origins `i-23..i-2`, so the implementation uses `y[i-23:i-1]`. Using the immediately preceding origin's target would be premature: its outcome is not available until the application session's close.

The first 23 positions have unknown summaries because a complete arrival history is unavailable. Every original index row is retained. Once the structural window exists, a single missing label makes **all** six features, the excess diagnostic and five integer-audit fields unknown for that window; no partial count is published as a feature. Structural window dates remain recorded. A missing label at origin position `j` affects feature positions `j+2..j+23`, inclusive. Missing dates/labels are never compressed, filled or reclassified as nonevents.

The builder has no score, phase, application or market-feature mask. Mature labels from feature-incomplete or unscored origins therefore contribute normally. Centering, model fitting, common-cohort selection and the three registered comparisons belong to the separate model/runner contracts. In particular, this module does not refit the original baseline, choose new observations by their future query label, or treat a source failure as a null result.

## Public API and saved schema

`cluster_summary(events)` accepts exactly one 22-element real numeric binary vector and returns an ordered, finite JSON-safe dictionary. Its six feature entries are followed by the excess diagnostic and five Python-integer audit entries:

`event_count22, adjacent_pairs22, expected_adjacency_numerator, linear_recency_numerator, excess_adjacency_numerator`.

`build_memory(targets)` returns the complete input index and these 12 numerical columns, followed by:

`feature_cutoff_date, window_first_available, window_last_available, window_first_origin, window_last_origin`.

Numerical columns use float storage so unavailable values can remain `NaN`; known integer audits retain exact integer values. The predecessor cutoff is known from index position one onward. All four window dates are `NaT` before position 23; afterward they describe the full structural window even when one of its labels is missing. At complete windows, the last availability equals the feature cutoff, and the last target origin is the second preceding reference date.

Exports are `L`, `SOURCE_END`, `TARGET_COLUMNS`, `NUISANCE`, `MEMORY`, `DIAGNOSTIC`, `FEATURES`, `AUDIT_FIELDS`, `SUMMARY_COLUMNS`, `DATE_COLUMNS` and `MEMORY_COLUMNS`. `MEMORY='adjacency_fraction22'`; `DIAGNOSTIC='excess_adjacency22'`. There is no adjustable window or alternative formula argument.

## Prewritten verification evidence

The initial 25 [synthetic contracts](/Users/byrons/code/trading-vol/ndx-vol-experiment/tests/test_event_cluster_features.py) were written before [the implementation](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/event_cluster_features.py) existed. The [initial red log](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/event_cluster/features_initial_red.txt) records the absent-module import failure. All 25 passed on the first implementation in 0.942 seconds; [that log](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/event_cluster/features_first_implementation.txt) is retained. After presentation-only formatting and a lint-style correction, the [pre-amendment green log](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/event_cluster/features_final_green.txt) records 25 passing tests in 0.915 seconds. Those original logs are unchanged.

Following the additional prewritten attribution regression and the prospective raw-adjacency correction, the [current green log](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/event_cluster/features_raw_adjacency_green.txt) records **26 passing tests in 0.925 seconds**. The new [scoped lint log](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/event_cluster/features_raw_adjacency_lint.txt) also passes. No assertion tolerance was relaxed; expected fitted-feature identities were changed to reflect the explicitly revised pre-empirical design while all excess-diagnostic oracle checks were retained.

Tests use an independent `Fraction` oracle on generated binary vectors, enumerate every short binary sequence to verify the conditional permutation identity, and exhaustively check fixed-22 sparse permutations against exact summed integer excess numerators. They also verify all-zero/all-one constants, the matched-count/recency counterexample, exact arrival-date joins on an irregular generated calendar, missing-label propagation, future-label and prefix invariance, input preservation, and rejection of malformed schemas, calendars, labels and dates. No historical source admission or empirical feature construction occurred during this implementation.
