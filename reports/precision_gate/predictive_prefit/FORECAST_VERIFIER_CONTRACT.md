# Independent precision-gate forecast verification

This implementation and its tests use generated observations only. No empirical
source arrays, feature cohorts, fitted models, predictions or results were read
or run. Existing source, test and publication artifacts are unchanged.

`verify_forecasts(sources, features, targets, produced, config=None)` independently
checks the complete successful five-output pipeline. Sources have exactly
`daily`, `cross`, `iv`; supplied features have MARKET12 plus `rv_total`; targets
have `y,target_end`. The produced keys are `applications,panel,coverage,schedules,
fits`. All four arms are required in fixed order: base, adaptive, constant,
contextual. The ten config fields and production defaults match the separate
pipeline contract: issuance starts 2010-01-04, scoring starts 2016-01-01,
development ends 2019-12-31, evaluation/source ends 2025-10-20, minimum expert
training is 1,000, adaptive half-life is 252 sessions, gate window is 1,260
original sessions and gate minimum history is 252 issued mature pairs. Smaller
configurations are exclusively for generated contracts; the runner binds the
exact empirical protocol.

The return has `status: VERIFIED`, calendar/coverage/application/scored counts,
`application_forecasts_verified`, `forecasts_verified`, fitted months, expert
fits, warmup months, independently checked gate fits and cold-start months.
Insufficient expert support during scoring raises `INSUFFICIENT_DATA`; it never
certifies a partial successful experiment. Earlier expert warmup months remain
explicit with no invented forecasts. The producer's separate failure payload is
retained by the runner; this API does not promote that payload to success.

## Independence and fixed calculations

Only the frozen independent claims verifier's `_market_targets` and `_compare`
are imported. The former reconstructs the established twelve market features,
GK plus raw overnight variance and strict next-five-original-session arithmetic
targets. The latter applies the inherited type/date/numerical comparisons. The
calendar/schema checks follow the prior independent commodity verifier. This
reuse is explicit: the old market baseline is not represented as receiving a
third new independent raw-data derivation. No precision-gate producer, expert,
pipeline, scoring or Treasury feature helper is imported or called.

All source calendars and schemas are checked before numerical market
reconstruction. The unchanged original calendar is retained for source
alignment, label endpoints, training distances, gate windows and modulo-five
offsets. Every supplied feature and target cell is compared with the independent
market oracle. State is exactly `lrv_d-lrv_m` at issuance, with no fitted
normalization or label-dependent state construction.

For every requested month, the first complete MARKET12 query is selected without
its label. Expert training uses every earlier common complete row whose observed
positive target matures by the preceding full session. Training can begin before
issuance. The two experts share precisely those rows. The base weights are one;
adaptive weights are `exp2(-(cutoff_position-train_position)/half_life)` using the
full original positions. Zero or nonfinite weights fail. Each arm independently
computes weighted population means/scales, requires every nonconstant scale to
exceed `1e-12`, and solves its square-root-weighted standardized design with QR.
The SVD rank test uses the declared relative `1e-12` threshold. The verifier
reconstructs raw weights, their sum, all coefficient/scaling/singular-value
audits, weighted normal equations and exact arm-specific weighted Duan smearing.
It does not call the producer's least-squares solver or replace a failed design.

The verifier maintains its own chronological map of actually issued expert
forecasts and issuance states. At each new fit, both gates take the same entries
whose positions are in `[fit_position-1260, fit_position)`, with observed labels
mature by the previous session. It does not refit prior experts under the current
model, compress missing sessions, replace a missing pair, borrow a current-month
forecast or refill earlier warmup dates. Below the fixed pair count both gates
have literal `[0,0]` coefficients and equal the base prediction. Cold starts are
retained in every eligible comparison.

Every successful expert audit, gate membership, saved state, fit/application
date, issued prediction, QLIKE loss, coverage reason, missing-feature list and
unscored application is verified. Scoring requires an origin and its complete
target endpoint within the declared phase. No subset of gate-active dates or
favorable offsets is substituted.

## Separate convex gate certificate

For `u=y/base`, `r=base/adaptive`, `z=tanh(state)`, `x=((1-z)/2,(1+z)/2)` and
`w=x*c`, positive relative precision is computed as `(1-w)+w*r`. The independently
evaluated objective is `mean(u*p-log(p)) + .005*(c_low²+c_high²)`. Its full
gradient is reconstructed directly. Constant mode requires exactly equal
coefficients and uses the sum of the two derivatives on its single optimization
coordinate. The projected-gradient residual at the saved coefficient must be
at most `1e-8`.

A separate SLSQP solve starts at 0.5 on each optimization coordinate with
analytic gradient, bounds `[0,1]`, `ftol=1e-14`, and `maxiter=2000`. It must report
success and a reconstructed projected-gradient norm at most `1e-7`; otherwise
verification fails without switching algorithms. Saved coefficients must match
the independent optimum within absolute `5e-6` and relative `1e-7`; objectives
must agree within absolute `1e-10` and relative `1e-9`. These constants were
agreed before empirical use and are separately recorded by the root protocol.
Strict convexity of the penalized problem underlies the optimum check.

The saved coefficient's objective, full and optimization gradients, stationarity,
box, constant flag, training count, successful L-BFGS-B label and bounded integer
iteration count are all checked. L-BFGS-B iteration counts can only be checked as
metadata in `[0,1000]`; a different optimizer cannot independently reproduce
them. The gate's forecasts are then reconstructed from the independently
certified saved coefficients. This keeps direct prediction arithmetic at the
tight forecast tolerance instead of transferring the wider optimizer-coordinate
tolerance into forecasts.

Other comparisons retain fixed expert coefficient absolute/relative tolerances
`1e-7/1e-7`, forecast tolerances `1e-12/1e-7`, and numerical tolerances
`1e-12/1e-9`. Dates, counts, identities, ordering, cold starts and schemas are
exact. There is no empirical tolerance tuning or fallback.

## Generated evidence

`FORECAST_VERIFIER_RED.log` records the genuine missing-module import failure
after 14 substantive test methods were written. Seven independent mathematical
kernel tests passed in `FORECAST_VERIFIER_KERNEL_FIRST.log`, followed by all
initial 14 full tests in `FORECAST_VERIFIER_FIRST_IMPLEMENTATION.log`. Four
additional pre-empirical tests explicitly check issued membership, weights and
gate-coefficient tampering, retained cold starts and fatal scoring-period
insufficiency; they are not represented as missing-module RED observations.

`FORECAST_VERIFIER_GREEN.log` records 18 tests passing in 74.036 seconds, followed
by scoped formatting/lint checks. Generated evidence includes raw weighted
normal equations and exact smearing, calendar-gap decay, known analytic and
boundary gate optima, independent optimizer settings/failure, every gate audit
field, target-blind issuance, mature gate history, all four arms, missing/scored
and unscored outputs, input preservation and future-calendar rejection before
the market oracle runs. No repository-wide suite is run by this implementation
task.

## Limits

Source byte authentication, current-vintage limitations and historical-access
disclosures remain the runner's responsibility. These market-only calculations
do not use Treasury auction quantities or inherit the Z52 timing assumption as a
new model input. Previously accessed historical periods are not untouched
confirmation. Scoring, inference, multiplicity and signal qualification require
their separate independent checks. A successful generated verifier result does
not establish empirical predictability or future performance.
