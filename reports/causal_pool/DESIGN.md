# Wave 14: fixed causal probability pooling

This trial asks whether a fixed average of the verified conditional sign-agreement probability and a recent event-rate estimate improves on **both** components. It follows the frozen [prospective note](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/sign_memory/NEXT_CAUSAL_CALIBRATION_DESIGN.md). The prior model's descriptive average underprediction motivates the question; its phase-wide errors never enter a forecast, a seed, or a selected weight.

The target remains strict next-session raw QQQ ETF–SPX price-index directional agreement. Ties are retained nonevents, missing pairs unknown. It is not volatility magnitude, standalone index direction, covariance, or trading profit. Original source vintages, measurement gates, label convention, feature support, scored origins and phases remain enforced by independent read-only reconstruction of wave 13.

## Forecasts and information timing

Preserve the exact original issued baseline. Initialize one event-rate state at the first original application origin's preceding reference close with numerator equal to the original first fit frequency and denominator one. This seed summarizes the first fit's eligible mature feature-complete training subset. Record its last included label availability separately from the state cutoff; intentionally omit any older excluded labels and do not feed dates at or before the initial cutoff again.

The sole half-life is 63 observed SPX sessions, delta = 2**(-1/63). At every subsequent reference close decay numerator and denominator by delta. Add (1−delta)*y and (1−delta) for that date's newly mature finite label. A missing label adds neither term, but time still ages both state components. Include mature labels from original unscored and feature-incomplete origins. The complete reference calendar determines elapsed time; no phase reset, calibration warm-up or sample-dependent deletion is allowed.

At each original application origin use the state only through its previous session cutoff. Divide numerator by denominator for recent frequency. On each original scored origin form two separate half-products and add them: 0.5*issued_baseline + 0.5*recent_frequency. Save state audits at every original application, including unscored applications; do not fabricate an unscored issued baseline. The two new probability series and copied baseline keep identical original scored rows and labels.

## Controls, gates and counts

Exactly two new comparisons are registered: pooled versus frozen baseline and pooled versus recent frequency. The second control tests whether the conditional model contributes beyond the specified moving event rate. No alternate weight, half-life, logit calibrator, target or score is run in this family.

Brier is the only primary loss. Both comparisons need an absolute decrease of at least 0.0005 in both original development and evaluation phases, negative differences in both fixed later slices, Holm2 below 0.05/(14×15), and cumulative Holm121 below 0.05. All 119 earlier hypotheses remain counted. The 21/63/126 circular-block bootstrap uses 99,999 draws, HAC126 and conservative maximum p across methods and phases; seed 20260920 follows the original phase/block offset rule. Original class support gates and the October 20, 2025 source bound remain unchanged. The period beginning November 3, 2025 is excluded.

There are zero newly fitted monthly baseline models. If n scored origins survive the unchanged original contract, n baseline rows are preserved and 2n new forecasts are generated; the combined panel has 3n rows. Application audit counts are separate. No new empirical support, filter state, forecast or score count is calculated before prewritten checks and registration.

## Verification and arithmetic

The producer follows the recurrence. The independent verifier builds an explicit compensated weighted sum from the initial seed and every eligible observed label, using full-calendar age exponents. At k elapsed reference sessions, its fixed roundoff envelope is 64*(k+1) times the larger of epsilon times the expected magnitude and its downward ULP, with exact zero/sign masks. This is specified before any empirical comparison. State and probability domain checks precede tolerances. Saved q replays exactly from saved S/W; pooling replays exactly from its two half-products and all Brier scores replay exactly.

IEEE float64 finite-state, positive-denominator, probability-domain and nonzero-underflow checks fail the whole family. Representable subnormals and exact mathematical zeros are retained. No clipping, reset, alternate horizon, changed tolerance or removed row repairs an empirical failure. All previous code, protocols, data and reports are hash-pinned. Input decoding uses single immutable byte snapshots. Independent verification reconstructs the old proof and every new state, probability, loss, paired difference, inference result and ledger event before interpretation.

This adaptively selected question reuses archival history. Causal timing and multiplicity accounting do not create untouched confirmation. A passing pool would establish value for this specified probability combination under the registered experiment; it would not identify structural dependence dynamics or prove calibration. The user’s broader search remains active unless useful predictive evidence is separately established.
