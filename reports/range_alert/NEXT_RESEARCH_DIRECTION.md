# Proposed wave 19: causal calibration from issued alert errors

This is an **adaptive, unregistered proposal** following the verified wave-18 result. That trial's range-extremity correction did not qualify. Its verdict, forecasts, gates and source bounds remain unchanged. This note uses three closed methodological reports and the result supplied by the parent task; it does not construct new features, event counts, residual associations, fits or scores.

The question is whether a single time-varying intercept, informed only by mature errors of the baseline probabilities actually issued earlier, improves the same next-session risk alert. The event remains next-session SPX daily risk exceeding twice the preceding 22-session mean, with the exact existing OHLC proxy, threshold, lag and maturity rules.

## Novelty and overlap

The [model-memory study](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/model_memory_study/SUMMARY.md) already tested retrieval and recalibration using historical issued errors for continuous risk. Error memory is therefore not a new information channel. The [sign-memory trial](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/sign_memory/DESIGN.md) estimated a fixed coefficient on rolling sign dependence. The [causal-pool trial](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/causal_pool/DESIGN.md) combined issued sign-agreement probabilities with a moving label frequency, explicitly excluding a logit calibrator.

The narrower untested question is sequential intercept calibration of this binary risk-alert baseline. A label-frequency filter reacts to outcomes without asking what the model predicted; the proposed filter uses paired issued predictions and their later outcomes. This is a different forecasting rule on reused information, with substantial overlap in purpose and data. A success would support this particular calibration rule, not establish a new structural predictor.

## One fixed rule

Keep every original monthly baseline fit frozen. Let `eta_j` be its finite historical application logit, and `p_j = expit(eta_j)` its originally issued probability. At an application origin `t`, use only records whose label availability is no later than `k(t)`, the preceding observed SPX close.

Use one half-life, 63 full reference sessions: `delta = 2**(-1/63)`. A record arriving at reference position `k_j` receives weight

`w_j(k) = (1-delta) * delta**(k-k_j)`.

At the first original application's preceding-session cutoff `k0`, initialize an empty issued-error history and correction `a=0`. Do not seed calibration with in-sample fitted probabilities or re-feed labels available at or before `k0`. The baseline's original training and intercept already provide its starting forecast. There is no calibration warm-up deletion or phase reset.

For each later application, find the unique minimizer

`F_k(a) = sum_j w_j(k) * logaddexp(0, (1-2*y_j)*(eta_j+a)) + .01*a*a`.

This signed binary-loss evaluation is mathematically equivalent to `logaddexp(0,x)-y*x`, but avoids subtracting nearly equal large values for a correctly classified positive logit. It is the proposed numerical evaluation rule, not a change of objective.

The weights are **not divided by their sum**. Their total is at most one because at most one new next-session label can arrive at each reference close. This fixes the initial information mass and prevents the first observed error from becoming a full-weight calibration sample. Missing arrivals age all existing weights and increase the relative pull toward zero; they do not generate artificial labels.

In real arithmetic the derivative is `g_k(a) = sum_j w_j(k)*(expit(eta_j+a)-y_j) + .02*a`. Evaluate each binary residual stably as `expit(x)` when `y=0`, and `-expit(-x)` when `y=1`, with `x=eta_j+a`. At zero, the mathematical derivative equals the negative weighted sum of issued baseline forecast errors. This is **not a bitwise identity with residuals calculated from rounded saved probabilities**: for a sufficiently positive finite logit, saved `expit(eta)` can equal one while the stable residual `-expit(-eta)` remains negative and nonzero. Saved-probability replay establishes issuance identity; the original finite logit defines the calibration objective and its stable derivative.

The derivative's slope is at least `.02`, so the optimum is unique even with an empty or temporarily single-class history; the empty case gives exactly zero. With `W=sum_j w_j`, `[-(W+1)/.02, +(W+1)/.02]` is a deterministic strict sign bracket. A separately tested scalar derivative-root implementation can avoid stopping on rounded objective differences. Fix its budget, arithmetic checks and full-gradient acceptance before registration; do not reuse an empirical failure to select another solver.

Issue `p_corrected(t)=expit(eta_t+a_k)`. Only this one intercept changes; there is no slope recalibration, regime selector, alternate half-life, estimated pooling weight or additional extremity coefficient.

## Issuance, maturity and numerical admission

Recover each logit from the **original** saved monthly coefficient and transformation records, on its original saved application features. Require strict replay of the corresponding saved application probability before admitting it. Wave 18 retains application probabilities for unscored applications as well as scored rows. This permits those genuinely issued forecasts to contribute after maturity without inventing probabilities on dates with no original application.

Do not invert rounded probabilities: `logit(0)` and `logit(1)` would lose the finite original model logit. Using the original finite logit handles numerically rounded endpoints without clipping; `expit` endpoints remain valid issued probabilities. If a purported issued application cannot replay or its logit is nonfinite, fail the new trial. Do not substitute a fitted retrospective prediction, delete the record or reinterpret a missing audit as a legitimate forecast gap.

Advance age over the full frozen reference calendar, including feature gaps and unscored months. At issue time cache every original application record without consulting its future label. At a later cutoff, include it only after its known binary label has matured. A date with no originally issued baseline supplies no calibration pair; an issued application with an unknown mature label supplies no pair either. Both are recorded, neither stops time. The recent-frequency control intentionally continues to use its original broader stream of known labels, including dates without baseline issuance. That information-set difference is transparent and does not permit changing either control.

Preserve the full original scored cohort and all original class-support requirements. A scored origin cannot consume its own next-session label, a label available after its prior-session cutoff, or a later refit's reconstruction of an earlier forecast. Guard finite logits, weights, gradients and probabilities, and reject nonzero arithmetic underflow rather than silently dropping tiny records. Source-vintage and publication-time limitations remain those of wave 18.

## Falsification and accounting

Propose exactly two new Brier comparisons: corrected baseline versus frozen baseline, and corrected baseline versus the **same frozen recent-frequency control**. Both must pass. Do not register baseline versus recent frequency again to obtain another significance threshold, and do not promote the correction merely because it inherits a strong baseline.

Retain all **129** prior hypotheses, including failures, giving **131** cumulatively. Preserve the absolute Brier decrease of at least `.0005` against both controls in both original phases, negative differences in both fixed evaluation slices, and cumulative Holm131 below `.05`. The wave-19 correction is Holm2 below `.05/(19*20)`. Keep the existing conservative HAC126 and 21/63/126 circular-block scheme with 199,999 draws; fix a new deterministic seed and all numerical tolerances only in a prospective registration. Missing support, provenance failure or arithmetic failure retains both new hypotheses as unevaluable with p=1.

Prewritten tests should establish empty-history nesting, analytically balanced-error symmetry, finite rounded-endpoint replay and its distinction from stable logit residuals, missing-label aging, no refeeding before the seed, full-calendar timing, unscored-application inclusion, deliberate nonissuance exclusion, future-label and future-refit invariance, uniqueness and bracket signs, exact cohort preservation, unchanged controls, and independent reconstruction of every weighted calibration record and score. Balanced-error tests must use an analytic symmetry condition, not pretend that a finite mathematical logit yields an exactly binary probability merely because its saved floating-point sigmoid rounds to an endpoint. Existing model and source audits should be admitted unchanged.

This trial would test whether forecast-aware calibration adds value beyond the issued baseline and a strong moving event-rate control. It does not promise better conditional calibration, executable risk management or untouched confirmation. Selection follows observed wave-18 results; sequential multiplicity accounting and causal construction do not erase that adaptivity.
