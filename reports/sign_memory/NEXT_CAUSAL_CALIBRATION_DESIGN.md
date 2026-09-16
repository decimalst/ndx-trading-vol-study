# Prospective next question: causal probability pooling

**Unregistered exploratory proposal, developed after wave 13's descriptive results.** Wave 13 remains VERIFIED with no qualifying model. The reported baseline mean probabilities were about 3.91 percentage points below observed strict agreement in development and 4.20 points below it in evaluation; its sign-memory increment changed this little. Those supplied descriptive summaries motivate this proposal. They are not a new test, evidence of conditional miscalibration at every probability, or an estimate to insert into future forecasts. No new outcomes, event counts, residual associations, scores, fits, or hyperparameter comparisons were calculated for this note.

The question is whether a simple causal update to the overall event rate improves the **conditional probability of the same strict raw-sign agreement event** when combined with the frozen issued baseline. Keep the original target, archival source limitations, labels, score cohort, and [wave-13 verdict](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/sign_memory/results.md). This proposal neither introduces new data nor supplies independent historical confirmation.

## Choice: fixed pooling before a new logit calibrator

Two reasonable constructions answer related but different questions:

| Construction | What it changes | Numerical and identification requirements |
| --- | --- | --- |
| Online intercept or logit tilt | Shift the baseline logit by a scalar learned only from mature prior forecast errors. This directly targets a persistent probability-level bias while holding all baseline slopes fixed. | Applying `logit` to an issued probability of exactly zero or one is undefined. A defensible implementation could replay the original finite logits from saved coefficients, then fit one regularized scalar on mature issued forecast/label pairs. It would need an additional objective, penalty, optimizer, empty-history convention, and distinction between current-fit and genuinely issued past predictions. The reported phase-wide gap must never initialize the correction. |
| Fixed convex pool | Blend the issued baseline probability with a causal estimate of the unconditional event rate. This allows the recent rate to pull a systematically low or high probability toward current experience. | Uses the issued probabilities directly, including valid zero/one endpoints. It needs one fixed memory horizon and one fixed mixing weight; no optimizer, inverse logit, probability clipping, or new baseline fit. It does not claim to estimate an intercept correction or establish calibration. |

**Choose the fixed pool for one next trial.** It is a smaller, well-posed change that makes the moving-event-rate explanation an explicit control. Defer the logit tilt entirely: no alternate tilt arm, retrospective comparison of these options, or replacement if pooling disappoints. The tilt discussion is a design comparison, not another hypothesis being run.

## Exact proposed forecasts and single timescale

Use the frozen issued wave-13 **baseline** probability `p_t`, not its sign-memory arm and not a refitted or retrospectively fitted probability. Propose exactly three outputs:

1. `frozen_baseline`: `p_t`, copied from the verified issued panel.
2. `recent_frequency`: the causal event-rate state `q_t` defined below.
3. `pooled`: `p_pool,t = 0.5*p_t + 0.5*q_t`.

The **only memory timescale is 63 observed SPX sessions**, with per-session decay `delta = 2**(-1/63)`. The mixing weight is fixed at one half before any new calculation. These are prospective round choices, not optima inferred from the observed gap. Do not search alternative half-lives, weights, start dates, target subsets, or other scores. This is an online event-rate filter and a fixed forecast combination, not residual-driven coefficient fitting.

Use the complete frozen SPX reference calendar and the full wave-13 target table. A target row at origin `j` represents strict agreement during the next actual SPX session and has `available_date` equal to that next close. For the rate state, admit **every finite binary target with a newly reached availability date**, even if its own origin had incomplete features, lacked a scored issued forecast, or was excluded from development scoring at the phase boundary. This prevents feature availability or loss-based selection from choosing the memory observations. Missing paired returns contribute no observation; they never become class zero. Genuine zero returns remain valid class zero under the frozen target definition.

## Cold start and exact causal state update

Let `t0` be the first original forecast origin and `k0` its preceding reference-session cutoff. Initialize once, at that cutoff:

`S[k0] = f0`, `W[k0] = 1`, where `f0` is the exact frozen historical-frequency forecast from the same first monthly training fit.

The underlying training labels were already mature by `k0`. This seed uses no later calibration result and creates no new historical baseline forecasts. It summarizes only the first fit's eligible, feature-complete mature training rows; excluded earlier labels are intentionally absent. Do not separately add any label available on or before `k0`. Initialize the state cutoff to `k0` and record the latest included availability separately as that first fit's `train_last_available`; these dates need not coincide. The unit initial mass is part of the fixed exponentially weighted state, not a selected burn-in length.

For each subsequent reference-session close `k`, in chronological order:

- First set `S <- delta*S` and `W <- delta*W`.
- If the unique next-session binary label with `available_date == k` is finite, set `S <- S + (1-delta)*Y` and `W <- W + (1-delta)`.
- Otherwise add nothing to either quantity. Never impute a label or compress the calendar.

At an origin `t`, issue `q_t = S/W` from the state processed **only through the previous SPX session `k=t-1`**. Then combine it with `p_t`. The state therefore uses labels from origins at most `t-2`; the label at `t-1` becomes observable at `t`'s close and cannot affect the forecast at `t`.

For example, a Monday origin uses only the state through Friday. The label for Friday's origin concerns Monday's session and is unavailable to that Monday prediction. A holiday merely changes the observed-session predecessor; do not invent a business-day offset or a next-common-asset calendar. A next-session date is label/maturity metadata, never a predictor of that still-future outcome.

The numerator and denominator decay on **every reference session**, including sessions with a missing label. On such a session their ratio is unchanged, but their remaining weight declines; the next observed label consequently receives its proper calendar-aged weight. A simple update only on observed labels would instead define a different observation-count half-life.

Retain every original scoreable origin, including the first. There is no calibration warm-up deletion or extra rolling-window admission gate. Preserve the original cohort exactly, and evolve the state across feature gaps, unscored application months, the development/evaluation boundary, and fixed evaluation subperiod boundaries. Never reset or initialize it using a phase's full outcomes. For each prediction save the cutoff, latest consumed availability date, cumulative update count, `S`, `W`, `q_t`, and the frozen probability's original identity.

## Endpoint and arithmetic contract to freeze before implementation

The cold-start frequency is interior because the admitted first fit had both classes. Mathematically `0 <= S <= W`, `W > 0`, so the filter and its convex pool stay in `[0,1]`. An issued baseline probability of exactly zero or one remains valid; no inverse logit, epsilon replacement, winsorization, or clipping is needed. The pool is allowed to move such an endpoint toward a nondegenerate `q_t`.

Before a run, pin the update order above, IEEE float64 arithmetic, and separate half-weight products followed by addition. Require finite state, a strictly positive denominator, and finite probabilities in `[0,1]`. Reject nonzero multiplications or divisions that become zero through unsupported underflow; retain representable nonzero subnormals. Exact zeros from the mathematical operations remain valid. Any nonfinite or out-of-range result fails the whole registered trial; do not reinitialize the state, clamp a probability, shorten the horizon, or omit an origin after inspection. Reuse the frozen checked Brier and paired-difference arithmetic for scoring. This paragraph is a prospective contract, not a claim that numerical edge cases have already been tested.

## Two hypotheses and a strong moving-rate control

Register only these paired Brier-loss comparisons, using the same original origins and labels for all three outputs:

1. `pooled` versus `frozen_baseline`.
2. `pooled` versus `recent_frequency`.

**Both must qualify.** The second comparison is essential: beating an expanding historical frequency would not distinguish conditional information from a moving unconditional event rate. Beating the causal recent-frequency control would show that this fixed blend adds forecasting value beyond that specified moving-rate estimate. It would still not prove calibration, a new independent predictor, true dependence dynamics, or that conditional risk is the mechanism. An average of complementary forecast errors can improve a proper score without identifying any one of those explanations.

If registered next as wave 14, the two additions would take the retained family from 119 to **121** comparisons. Propose Holm2 at `.05/(14*15)` and cumulative Holm121 at `.05`, with no deletion of previous null or unevaluable trials. Keep Brier as the only primary score; require an absolute mean loss decrease of at least `.0005` against both controls in both original phases, and negative differences in each original evaluation slice. Carry over the 21/63/126 circular-block bootstrap, HAC126, 99,999 draws, max-p conservatism across methods and phases, and the existing phase/slice support rules. Predeclare seed **20260920**, with `seed + phase_code*10000 + block`, if this exact proposal is registered. No phase-mean calibration gap becomes a fitted parameter or an extra promotion test.

Retain development origins 2016–2019 with their original year-end maturity fence, evaluation origins 2020–2025-10-17, the 2025-10-20 source/label bound, and the protected 2025-11-03-and-later period. This is an adaptively chosen exploratory question on reused history. Multiplicity accounting and causal replay do not convert that reused history into a new untouched validation sample. Forecasting improvement would concern raw directional-agreement probability, not volatility magnitude, covariance, index direction, or trading profit.

## Required work before registration or execution

Hash-admit the frozen wave-13 issued baseline, original frequency seed, target table, full reference calendar, audits, and verification record. Prewrite synthetic tests for the two-session origin-to-state maturity rule; first-origin cold start; phase-boundary carry; missing-label aging; exact-zero labels; endpoint probabilities; unchanged predictions after future-label mutations; duplicate/misaligned availability rejection; preservation of every issued origin; pooling arithmetic; and whole-family failure publication. Independently reproduce the filter from its explicit exponentially weighted sum, rather than calling the recurrence implementation, and compare every state and probability before interpreting scores.

This note records no new registration, code, numerical source read, event count, calculation, or fit. It specifies one concrete candidate for the next experiment while preserving wave 13 unchanged.
