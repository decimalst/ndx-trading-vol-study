# Wave 20 design: clustering of mature SPX risk events

Prospective specification, 2026-09-07. No new event histories, support counts, fits or scores have been computed. Registration follows complete implementation, prewritten synthetic checks and an independent verifier. This design does not amend any earlier experiment.

## Question and novelty

Does the arrangement of recent risk events improve prediction beyond their count, latest outcome and simple linear recency? The unchanged response is the wave18 SPX daily-risk alert, with its exact target arithmetic, one-session predictor lag and next-session label maturity. This is a new summary of existing information, not a new source.

The bounded overlap audit found point-state/latent residual retrieval, additive event-rate filtering, forecast-error calibration and cross-asset sign-agreement memory. The earlier neural models could implicitly encode ordering, but no explicit within-series adjacency comparison against matched count/recency controls was found. Signed close location was rejected as a novelty claim because orthogonal_round2 already tested it. A trailing-peak-age return question remains unselected. No alternative was scored in making this choice.

The separate early-session proposal remains source-blocked. It is not replaced by a claim that this daily experiment uses opening information. Its frozen documentary review is preserved.

## Fixed event-history feature

At each origin's previous-session cutoff, take the binary labels arriving at the last 22 reference closes, oldest first. Labels include known outcomes from feature-incomplete and unissued origins. Their use does not require a historical forecast error. Unknown labels invalidate the entire literal window; time is never compressed.

Let N be the event count, B the latest binary label and C the number of adjacent event/event pairs. Conditional on N and B, uniformly permuting the first 21 positions gives expected adjacent pairs `(N-B)*(N-1)/21`. This is a combinatorial reference, not an exchangeability assumption about markets.

Use five matched nuisance coordinates:

- `n = N/22` and its mathematical square, computed as integer `N*N/484`;
- `B`;
- `G = (N-B)*(N-1)/441`;
- `R = sum((2*i-23)*e_i, i=1..22)/462`.

The sole fitted candidate coordinate is `A = C/21`, computed from its integer count. Retain `M = (21*C-(N-B)*(N-1))/441` as a descriptive feature audit only. M is evaluated from its integer numerator, preserving exact zero and avoiding subtraction of rounded ratios. Its permutation centering is conditional on N,B only; it does not establish conditional residualization given R or the other market controls.

The exact expectation G is included in the nuisance model. A pre-empirical design review rejected fitting M: with frozen nuisance coefficients, its subtraction could merely retune G, even if C were constant. Fitting A avoids that specific mechanism. Event-position sets `{2,3,8}` and `{1,5,7}` within a 22-position window have identical nuisance coordinates but different adjacency. This supplies an algebraic falsification fixture, not empirical orthogonality evidence. A may still be collinear with other controls on a particular sample; the claim is relative to the specified models, not information-theoretic independence.

## Models, cohorts and timing

Reuse every verified wave18 baseline fit, application and recent-frequency state. Reconstruct query logits from the original saved transformations and coefficients, and strictly replay their issued probabilities. No inverse sigmoid of rounded probabilities is permitted. Original coefficient fitting is not rerun.

For a monthly nuisance fit, apply that month's saved baseline transformation and coefficients to its own original mature training rows. These are current-fit in-sample training offsets, not historically issued training forecasts. They are valid inputs to this explicitly staged fit because only those mature training observations are used. Application offsets remain the originally issued baseline logits.

On those same training rows, fit an unpenalized calibration intercept and five centered nuisance slopes, each with ridge penalty .01. Centers use training data only; scales remain one. Retain exact constants with their canonical zero slope. Freeze the resulting nuisance offset, then fit only one .01-penalized coefficient on centered A, with no further intercept. A constant A gives exactly zero correction and preserves the nuisance forecast, even if G and M vary.

Validate the new history at every original training and application row for every month before fitting any new parameter. If any required window is unknown, fail this whole trial. Do not change original cohorts, delete folds or choose a shorter history. Future query-label absence does not remove an issued application. Historical event arrivals follow the full reference calendar across phase boundaries and feature gaps.

The three required comparisons are candidate versus (1) original baseline, (2) matched nuisance model and (3) original recent-frequency forecast. The original baseline comparison prevents a weakened nuisance control from manufacturing success. No baseline-versus-frequency or nuisance-versus-baseline promotion hypothesis is added.

## Frozen numerical and decision requirements

Both fitting stages minimize mean Bernoulli negative log likelihood plus their declared slope penalties. Use stable signed-logit likelihood and residuals. The producer uses the existing damped Newton solver for the nuisance stage and a fixed bracketed Brent solve for the scalar. Independent verification uses a separate objective with a different nuisance solver and bisection for the scalar. Full gradient infinity norm must be at most 1e-8; exact budgets, tolerances and failed-solver behavior must be fixed in the executable protocol and prewritten tests before registration. No empirical repair, clipping or fallback is allowed.

Register exactly three comparisons, retaining 131 previous hypotheses: 134 cumulative. Require a Brier decrease of at least .0005 against all three controls in both development and evaluation, negative gaps in both existing evaluation slices, wave Holm3 below `.05/(20*21)` and cumulative Holm134 below .05. Keep HAC126 and circular blocks 21/63/126. Use 399,999 bootstrap draws and seed 20260926 plus the existing phase/block offsets; resolution 1/400000 is below one tenth of the strictest Holm3 raw wave cutoff. Confidence intervals and nominal minimum detectable effects are descriptive.

Keep the original 2016–2019 and 2020–2025-10-17 phases, source ceiling 2025-10-20 and protected period from 2025-11-03. Retain original support gates: 1,000 mature training observations, 50 of each class in training, 127 observations and 30 of each class per phase, and 15 of each class in each fixed evaluation slice. Any source, arithmetic, support, fitting, forecasting, verification or publication failure leaves all three registered comparisons unevaluable with p=1.

## Interpretation and validation

The previous target's overlapping historical threshold can itself induce label persistence. A successful comparison would support a specific prediction of this archived alert functional relative to the declared models. It would not identify an exogenous self-excitation process, latent physical regime, causal mechanism or trading edge. Current-vintage Yahoo/Cboe measurements and back-calculated early VIX9D limitations remain unchanged; repeated historical research is exploratory.

Prewritten tests must cover the exact conditional permutation expectation, same-nuisance/different-adjacency fixtures, availability-time joins, missing arrivals, future mutation, training-only centering, constant nesting, independent numerical optima and gradient checks. The full repository suite must pass before registration. Independent verification must bind source bytes and all earlier success/failure evidence, reconstruct every new state, original cohort, fit, application, score and trial-ledger entry, and preserve unscored applications. Published prior evidence remains immutable.
