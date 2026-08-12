# Target-change and regime-transition protocol

Frozen before the Oxford-Man mirror was downloaded or either study was scored.
The machine-readable contract is `target_regime.yaml`.

The initial branch URL returned HTTP 404 before any response was accepted or
outcome observed. Repository history showed that commit
`308b795fa220a58dea6784fe8e2566bcf8dea334` was the last revision containing
the 15.2 MB archive before its 2023 deletion. The protocol was amended to pin
that immutable object and to end confirmation on 2017-12-29, safely before the
archive's 2018-06-30 commit date. No model, target, feature, or success criterion
changed.

## Why these are separate studies

The jump study changes the economic object being predicted. The regime study
keeps the existing QQQ variance history but changes the forecasting frame to a
transition probability. They are not combined, and neither is permission to
search feature permutations on an already-spent window.

## Jump versus continuous variation

Use the static Oxford-Man realized-measure mirror for **SPX only**. Its Nasdaq
identifier is not accepted because `IXIC` ordinarily denotes the Nasdaq
Composite and cannot safely be relabeled NDX. The source must contain five-minute
realized variance (`rv5`) and five-minute bipower variation (`bv`). Define:

`continuous = min(rv5, bv)`, `jump = max(rv5 - bv, 0)`, and
`jump_share = jump / rv5`.

The truncation makes the components reconcile exactly while acknowledging that
finite-sample bipower variation can exceed realized variance. The primary target
is whether any of the next five sessions has a jump share above the 90th
percentile estimated in the prior annual training fold.

Three fixed logistic models use identical realized-history terms. The history
model is the baseline; ATM adds one-session-lagged VIX; surface adds the same VIX
plus one-session-lagged SKEW. No close is forward-filled. The only primary test
is surface versus ATM on 2014-2017: surface must lower both average five-phase
Brier and log loss. ATM versus history is descriptive and answers whether the
30-day ATM level transfers to this different target.

## Regime transitions

Use the existing QQQ `log_rv` series, but predict a five-session stress entry
rather than the conditional variance mean. At each annual fit, stress is above
the training-fold 80th percentile. Score only origins currently below that
threshold; the event is any exceedance in the next five sessions.

The candidate is a two-state Gaussian hidden Markov model on log RV. Parameters
are estimated only through the prior year. During the scored year, probabilities
are updated with the forward filter; full-sample smoothed states are forbidden.
The benchmark is a supervised logistic transition model using current, five-day,
and 22-day log RV. The HMM passes only if it lowers both average phase Brier and
log loss and its top probability quintile realizes more events than its bottom
quintile.

The 2016-2025-10-17 period is diagnostic because it has already been inspected
for other questions. No origin or five-session target may reach the clean window
beginning 2025-11-03.

## Required pre-run contracts

Tests must pass before acquisition or scoring and cover source identity, exact
jump reconciliation, training-only thresholds, completed-target training rows,
one-session Cboe lags, annual parameter fences, ordered HMM states, row-stochastic
transitions, future-invariant forward probabilities, and the clean-window fence.
