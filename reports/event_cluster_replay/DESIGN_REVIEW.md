# Proposed wave 21: technical verification replay of event clustering

**Adaptive, UNREGISTERED design review. No execution authorization or revised wave 20 verdict.** This note reads frozen code and protocol definitions only. It does not inspect saved coefficients, candidate scores, numerical forecast values, or historical event histories. The parent reports that wave 20's first independent verification stopped at a saved structural `feature_cutoff_date` dtype of microseconds versus reconstructed milliseconds, and that a separate date-only review found equal date values and unknown masks. That is the observed failure, not proof that every remaining check will pass.

The proposed replay is defensible with the restrictions below. Prefer reusing the exact failed producer artifacts over generating replacements: this isolates the representation contract while leaving every previously produced prediction open to full independent challenge. Keep the peak-age proposal pending until this unanswered clustering question has a terminal result.

## Why a separately registered replay is appropriate

The frozen [supplied-table verifier](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/event_cluster_verification.py) checks complete histories, cohorts, metadata, application states, and support before calling its independent stage solvers. Its exact DataFrame comparison treats datetime storage units as part of equality. The [producer pipeline](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/event_cluster_pipeline.py) constructs a state's cutoff by parsing a saved date string, while the independent reconstruction takes the date from the typed source feature column. Equal calendar dates can therefore have different storage resolution.

A strict unit mismatch correctly fails the old frozen contract. A new contract may recognize losslessly equivalent representations without changing the underlying date, mathematical feature, label, model, forecast, or numerical tolerance. This is a change to verification admission and must be labeled accordingly. It cannot amend wave 20 in place or retrospectively claim that its verifier passed.

## Inputs and immutable evidence

Before any new numerical reconstruction, register wave 21 and hash a complete immutable input snapshot. Include the exact wave 20 producer `forecasts.parquet`, `fits.json`, `states.parquet`, `memory.parquet`, `support_audit.json`, and `upstream_admission.json`; its protocol, manifest, freeze/test evidence, failed verification and canonical failure records; and the full original source/code/proof closure needed to reconstruct the study. Preserve original bytes and original dtype metadata. Decode the same buffers that were hashed, and recheck the committed inputs before successful publication.

These wave 20 numerical artifacts enter as **UNVERIFIED producer output**. Their existence and a successful producer exit do not establish model correctness. If the failed verifier captured output hashes, match those identities; where it did not, say precisely that the new registration pins the retained current bytes. Do not invent proof that an output was hashed before the failure. Any uncertainty about which producer attempt produced a retained artifact must be resolved documentarily before admission; do not select among alternative artifacts using their values.

Keep the previously verified upstream studies and the failed wave 20 study as distinct admission roles. A new wrapper may explicitly admit this identified failed study as the object to be checked, while continuing to require valid original source/forecast proofs. It must not generically ignore failure markers or treat failed wave 20 as a verified ancestor. Bind its canonical failure files by hash and preserve them permanently.

The failed study's unpublished scored metrics are not evidence and are not the new calculation's input. Preserve them in the old record without reading them to choose a replay. Recalculate wave 21 inference from the pinned forecast panel only after all independent reconstruction checks pass. No old writer, producer optimizer, cleanup routine, or failure guard may run against the frozen wave 20 directory.

## Lossless datetime-unit contract

Use an explicit field-level schema for structural date columns and DatetimeIndex values. A concrete narrow policy is to accept native, timezone-naive `datetime64[ms]`, `[us]`, or `[ns]` for these declared calendar fields only. Keep JSON date-string fields under their existing strict format rules. Do not extend this policy to numeric arrays, object/string columns, timezones, or arbitrary timestamp metadata.

Before comparing any declared datetime field, independently require:

1. The exact field identity, shape, row/index alignment and unknown mask. Required known dates stay required; `NaT` cannot replace a date or vice versa.
2. Known values are normalized midnight calendar dates within their prescribed source/phase/maturity bounds. These are calendar keys; storage normalization supplies no evidence about real historical publication times.
3. Conversion to a common exact representation is lossless. For example, convert finite epoch ticks to nanosecond integers using checked integer arithmetic, reject overflow, and verify exact round-trip to each original unit. The applicable historical fence fits this range, but the implementation must enforce it rather than assume it.
4. Exact equality of the resulting date values, followed by the unchanged predecessor/next-session, monthly scheduling and label-maturity relations. No rounding, date truncation, timezone stripping, nearest-date matching, or compression is allowed.

Do not use blanket `check_dtype=False`, broad `to_datetime(..., errors='coerce')`, or cast-to-day equality: these could conceal a real schema or timing error. Preserve exact comparisons for every nondate field, including numeric dtypes where the old contract requires them. Preserve names, column order, row order and index structure. Any comparison views must be temporary, leave original artifacts untouched, and report their input units and round-trip checks.

The safest implementation is a new small, explicit date-comparison boundary plus a newly named replay orchestrator that retains the frozen independent mathematical helpers. Where the old orchestrator hardcodes strict frame equality, adapt only that comparison in a new module; do not monkeypatch the frozen module or weaken unrelated validators. A broad rewrite or reserialization of all producer outputs would make the change harder to isolate and is unnecessary.

## What must still be independently verified

Reconstruct the full source-bounded calendar, every literal 22-arrival memory window including unknowns, the exact original training/application/scored cohorts, all class/support gates, and every monthly fit record. Preserve unscored applications and months. A missing history at an originally required row remains a whole-study failure, never grounds to drop that row.

Replay the original saved baseline geometry and coefficients without refitting it. Then independently solve and check **every** wave 20 nuisance and conditional adjacency stage using the unchanged objectives, initialization, solver budgets, stationarity gates and coefficient/probability tolerances in the [wave 20 protocol](/Users/byrons/code/trading-vol/ndx-vol-experiment/event_cluster.yaml). The existing independent root solver and scalar bisection remain the proposed methods. Preserve training-only centering, current-fit training offsets, fixed nuisance coefficients in the scalar verification, constant-adjacency retention, and exact zero-correction parent-probability identity.

Replay all application probabilities, scored Brier losses, saved states and original control values. Neither a successful date comparison nor saved gradients substitute for independent optimization and replay. A later arithmetic, solver, support, coefficient, forecast, inference or integrity discrepancy fails the new trial with no repair or relaxed tolerance.

No original or candidate **producer** fit is rerun. New numerical optimizations occur only inside independent verification and cannot replace a saved coefficient or prediction. Report `new_producer_fits=0` and `newly_generated_forecasts=0`; separately count reused producer fits/forecasts, independently reconstructed stage optima, application states and verified scored rows. Do not copy the old metrics' “new fits” accounting into the replay merely because the input contains those saved fits.

## Statistical identity and failure accounting

Keep exactly three contrasts: cluster versus original baseline, fitted nuisance, and original recent frequency. All models, controls, labels, cohorts, effect thresholds, fixed slices, failure rules and mathematical inference remain unchanged. Each contrast still needs at least `.0005` Brier improvement in both phases and negative differences in both fixed evaluation slices; the candidate needs all three contrasts and both multiplicity gates.

Retain all **134** prior hypotheses, including the three failed wave 20 hypotheses at canonical `p=1`. Register three new replay hypotheses, giving **137**, with wave alpha `.05/(21*22)` and cumulative Holm137 at `.05`. Preserve prior hypothesis identities and row provenance; the same mathematical contrast in two registrations is not permission to merge or delete its failed predecessor.

Retain the fixed seed `20260926`, phase/block seed offsets, **399,999** circular-bootstrap draws, blocks 21/63/126, HAC126, and maximum-over-tests/maximum-over-phases rule. Identical seeds avoid introducing a new random draw that could select a different conclusion. Minimum Monte Carlo p remains `1/400000`, below one tenth of the smallest wave 21 Holm3 raw cutoff `1/27720`. This is resolution arithmetic, not a result.

A normal terminal ledger contains 134 inherited, three registered and three evaluated/unevaluable records: **140 events**, distinct from 137 hypotheses. If a scored run later fails verification, preserve its attempted terminal history and append all required verification-failure events under the newly registered ledger policy; do not erase the trail to force a count. Canonical new metrics must immediately carry three unevaluable rows with conservative, wave-Holm and cumulative-Holm p-values all one, no lead, and an explicit failure proof. Retain the old wave 20 failure unchanged on either outcome.

If successful, the conclusion is that wave 21 independently verified the pinned earlier producer output under a newly registered lossless date-unit contract. It is not a second predictive dataset, fresh out-of-sample replication, or a passed wave 20 result. Reused observations and adaptive continuation remain explicit limitations.

## Minimal prewritten falsification checks

Before registration, generated fixtures should cover the actual mixed-unit construction: millisecond source dates, microsecond dates produced from ISO fit metadata, and both scored and unscored application states. Require complete independent stage solves and inference to pass without changing any saved numerical value. Test all declared unit combinations and round-trips, including `NaT`.

Reject a changed date, a sub-day offset even when casting to a day would match, mismatched unknown masks, timezone/object/string date columns, unit conversion overflow, reordered rows, and numeric dtype/value changes. Retain exact predecessor/next-session and phase-maturity tests. Tampering with a coefficient, saved probability, original control, history integer, support count, unscored application or source hash must still fail.

Transaction tests should prove that registration precedes decoding/reconstruction; wave 20 and every prior frozen artifact remain unchanged; no producer fitting is invoked; independent scalar/nuisance fits are actually exercised; unpublished old metrics cannot supply the new result; and any late failure invalidates all three new hypotheses. Test the new 137-family corrections and stricter alpha with generated p-values and unchanged inference settings. Passing these tests would authorize a separately registered attempt, not promise that the previously uncompleted historical checks will pass.

**Recommendation:** proceed to prospective tests and registration if exact retained-artifact identity and full source closure can be established. There is no need for a new producer run. Keep the sole intended change confined to lossless structural datetime representation, retain every substantive gate, and allow the full independent check to determine whether the clustering question is evaluable.
