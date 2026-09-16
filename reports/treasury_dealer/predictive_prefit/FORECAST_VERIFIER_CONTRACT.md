# Independent Treasury feature, support and forecast verification

This is a prospective generated-data contract. No historical auction quantities,
market observations, common cohorts, empirical fits or score results were read or
run while implementing it. The production Treasury feature, model, pipeline and
support modules are never imported or called by the independent verifier.

## APIs

`verify_forecasts(sources, ledger_events, features, targets, produced, config=None)`
returns `status: VERIFIED` with verified calendar, application, scored-origin,
forecast, coverage and fit counts, or raises `ValueError`. `sources` has exactly
`daily`, `cross`, `iv`. The supplied feature frame contains the fixed 32 model
columns, `rv_total` and `treasury_cutoff_date`; targets contain `y`, `target_end`.
The five produced components are `applications`, `panel`, `coverage`, `schedules`
and `fits`. Config is the six-field forecast config accepted by the frozen
pipeline; defaults are the declared 2016–2019 development, 2020–2025 evaluation,
2025-10-20 source ceiling and 1,000 common training rows.

`verify_inputs_and_support(sources, ledger_events, features, targets, event_audit,
support, config=None)` reconstructs the same inputs, every event audit and all
support rows. It returns `status: VERIFIED`, `support_passed`, `support_status`
and checked-object counts. A faithfully reproduced support failure returns
`support_passed: false`, `support_status: INSUFFICIENT_DATA`; it does not fit or
declare a negative predictive result. Its config is exactly the support module's
`forecast`, two `stability` intervals and ten named positive support floors.
Real execution must use frozen defaults. Smaller configurations are for generated
contracts only.

## Reconstruction

All raw-source calendars and event schemas/dates are checked before numerical
market or auction reconstruction. Date indices retain their complete original
order, including missing rows, and reject duplicates, time zones, nonmidnight
values and dates beyond the source ceiling. Event clocks retain the fixed 2010
floor and 2025-10-20 ceiling. Input frames and events are not mutated.

The independent claims verifier's `_market_targets` oracle is explicitly reused
for the established twelve market controls, GK plus raw overnight variance and
the strict next-five-original-session arithmetic target. Its `_compare` utility
is reused for fixed tolerance/type/date comparisons. This is shared independent
verification code, not a third independent reconstruction of that old baseline.
The prior independent commodity verifier's QR/audit and scheduling implementation
is adapted to the new fixed designs and Treasury cutoff. No Treasury producer
math, feature constants, cohorts or output helpers are reused.

TLT is aligned to the full reference calendar, masked before 2010, and delayed
one original session. Log returns, squares and strict five/twenty-two means are
reconstructed locally. Missing prices retain both affected returns and complete
window gaps. Weekdays are the known Tuesday–Friday indicators; the date cutoff is
the previous full session, masked before the source floor.

Auction reconstruction checks every event and rebuilds activation dates, pending
intervals, donor IDs, support reasons and share/mean audit values. A known delayed
release masks origins after its auction through its availability date and pulses
once strictly afterward. Unknown exact or bounded clocks mask through the first
possible session after that bound; unbounded clocks remain pending. Unknown
financial fields cannot be supplied. Excluded events do not activate or donate.

For a known pulse, the twelve latest known events of the same confirmed original
tenor with strictly earlier auction dates are selected before checking their
release clocks. No older replacement donor is permitted. A tied oldest cutoff,
an unreleased selected donor, insufficient history or a relevant intervening
unknown event leaves the pulse unsupported. A null-tenor unknown blocks every
possible tenor. A masked or unsupported simultaneous event masks all auction
columns for that origin. Successful simultaneous reports sum their individual
counts, log offerings, yields, bid-to-cover ratios, frozen prior means and dealer
surprises. Certified empty days are zero; true-zero or cancelling surprises
remain activation events.

Support uses common 32-column rows and a known previous-session cutoff. Monthly
queries use the first feature-complete requested origin, independent of its
target. Training labels must mature by the previous full session. Phase support
additionally requires observed labels and the development/source maturity fence.
Stability slices use evaluation-scored origins, and offsets use the original
calendar position modulo five. Event identities count separately while activation
dates count once. Every monthly, phase, slice, offset and original-tenor floor is
reconstructed with deterministic failure ordering. Supported events on masked or
otherwise feature-incomplete days do not count.

Three separate unpenalized log-target OLS models use exactly the same complete
32-column training rows. Independent QR solves reconstruct the market12,
matched31 and candidate32 predictions and all standardized-model audit fields.
Population standardization, the strict `1e-12` scale/rank thresholds and exact
arm-specific Duan smearing are retained. Scale/rank/arithmetic failures raise;
there is no alternative solver, dropped column or repaired cohort. Every monthly
fit membership, query membership, prediction, QLIKE loss, coverage reason and
unscored application is checked.

## Fixed comparisons and evidence

Dates, counts, statuses, field sets, ordering and identities are exact.
Coefficient absolute/relative tolerances are `1e-7`/`1e-7`; forecast tolerances are
`1e-12`/`1e-7`; other numerical comparisons use `1e-12`/`1e-9`. These match the
established independent-verifier contract and were fixed before empirical use.
No tolerance tuning or retry after an empirical discrepancy is permitted.

`FORECAST_VERIFIER_RED.log` records the genuine absent-module import failure after
the first 16 substantive test methods were written. The first implementation log
retains four support-entry subcase errors while the feature/forecast cases
passed. The second implementation log records those 16 methods passing after
support was implemented. Before final freezing, explicit literal support counts,
zero-pulse counting, original offsets, input preservation and millisecond/
microsecond transport checks were added. The support fixture was also strengthened
to guarantee every tenor in both invented slices, explicitly testing both a
passing report and an imposed training-support failure. These additions are
pre-empirical coverage, not claimed missing-module RED observations.

The final scoped test/lint record is `FORECAST_VERIFIER_GREEN.log`. The generated
fixtures cover source clocks, missingness, strict donor selection, aggregate
features, output tampering, target-blind monthly scheduling, mature training
membership, all three model audits, unscored tail forecasts, source calendar
preflight, support gates and preservation. No repository-wide suite or source
application is performed by this task.

## Limits

This verifier reconstructs calculations from an already authenticated source
ledger. It does not recertify raw Treasury PDFs/XML, original release timestamps,
source acquisition receipts or upstream byte preservation; the separate source
ledger audit and execution checkpoint own those claims. The qualified cached
reported-date assumption for the unresolved Z52 event remains a source limitation
and is not upgraded by a successful calculation check. Scoring, significance,
masked inference and signal qualification require their separate independent
verification. A generated `VERIFIED` result establishes agreement on generated
contracts and does not establish an empirical signal.
