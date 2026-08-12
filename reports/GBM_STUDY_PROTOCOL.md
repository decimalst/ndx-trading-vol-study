# Frozen protocol: GBM functional form on the HAR-IV information set

Written and frozen on 2026-08-12 before the first result-producing command.
The executable contract is `gbm_study.yaml`. This is an isolated diagnostic
study: it neither edits the standing experiment specification nor reads an NDX
clean-phase origin.

## Timing qualification discovered in independent review

The frozen design gives both HAR-IV and GBM the same-session published Cboe VXN
close. That makes the model comparison internally common-information, but Cboe
daily index closes can incorporate 16:00–16:15 ET while the repository's
standing forecast origin is 16:00. The frozen run is therefore not described as
a leakage-free 16:00 information set. Its verdict remains unchanged; a
separately registered post-result sensitivity repeats the fixed comparison with
one-session-lagged VXN. No same-session VXN result may support the stronger
16:00-origin claim by itself.

## Question and common information set

The study tests the last functional-form axis in the original program. Linear
HAR-IV and a histogram gradient-boosted tree receive exactly four same-origin
inputs: daily log realized variance, log trailing five-session mean variance,
log trailing 22-session mean variance, and log VXN. No leverage, calendar,
earnings, term-slope, skew, correlation, or future value enters either model.

The target is next-session log GK-plus-overnight variance. At an origin `t`, a
training row indexed `s` is usable only when `s < t`, because that row's target
is realized on the following session. Both models refit on the same expanding
sample at every diagnostic origin from 2016-01-04 through 2025-10-17. The last
allowed target is 2025-10-20, before the sealed phase begins on 2025-11-03.

Both models convert a log prediction to a variance mean with exact Duan
smearing computed from that model's current training residuals. Thus the QLIKE
comparison changes only functional form, not inputs, origins, or point-forecast
estimator.

## One fixed GBM, no tuning loop

The candidate is scikit-learn's deterministic
`HistGradientBoostingRegressor`: squared-error loss, learning rate 0.05, 64
iterations, at most seven leaves and depth three, minimum leaf size 30, L2
regularization 1.0, and 64 histogram bins. Early stopping and hyperparameter
search are disabled. The random seed is 20260812.

The full diagnostic score is reported for continuity. A frozen internal split
separates interaction discovery (2016-01-04 through 2019-12-31) from evaluation
(2020-01-02 through 2025-10-17). The split is calendar-defined and cannot move
after results.

## Interaction localization and interpretable repair

SHAP is absent from the frozen environment, so the study uses the registered
exact fallback rather than adding a package after inspecting results. At the
2019-12-31 snapshot, fit the fixed GBM using only rows whose targets are already
known. On a nine-quantile grid for every one of the six feature pairs, average
the fitted model over the discovery background, double-center the resulting
two-dimensional partial-dependence surface, and calculate the fraction of the
centered surface variance attributable to the interaction residual. This is a
deterministic two-way functional-ANOVA decomposition of the fitted model, not a
surrogate approximation.

Select the largest score with feature order as the tie break. Within that pair,
the largest absolute interaction cell fixes two thresholds and the direction of
two one-sided hinges. Their product, scaled by discovery IQRs, is the only term
added back to linear HAR-IV. The pair, thresholds, directions, and scales are
then locked. The augmented linear model is scored only in 2020-2025; there is
no reselection or alternate term if it fails.

## Scoreboard and verdicts

The primary loss is paired variance-level QLIKE. Every model/split reports mean
loss, relative improvement, h=1 Diebold-Mariano inference for continuity, a
paired 21-session moving-block bootstrap interval and two-sided p-value, and
the paired win rate. Bootstrap inference uses 5,000 draws and seed 20260812.

A GBM win requires a negative confirmation loss difference and block-bootstrap
`p < 0.05`. If it does not win, functional-form equivalence is recorded only if
the 90% block-bootstrap interval lies entirely inside plus or minus 3% of the
confirmation HAR-IV mean QLIKE. Anything else is inconclusive. The locked
linear interaction term passes only with a negative confirmation difference
and block-bootstrap `p < 0.05`; otherwise it is reported as not adding.

These are internal diagnostic conclusions on an already-open historical
window, not a reopening of the sealed NDX evaluation.
