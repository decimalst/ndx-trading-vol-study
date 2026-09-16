# Range-alert model implementation record

The synthetic model tests were written before this module. The initial run exited 1 before implementation, with `ImportError: cannot import name 'range_alert_features' from 'src'`; both new source/model interfaces were still absent. This records a missing-module red run, not an empirical failure. No historical inputs, targets, counts or fits were read.

The first implementation passed all 17 original tests once the feature module became available. Four additional tests were written before the final guard change. Late application-geometry failure and future-label state invariance already passed; two panel-validation regressions initially failed because the inherited validator permitted 500 training rows and a boolean horizon. The new wrapper now enforces at least 1,000 rows and real numerical metadata. All **21 model tests pass** in 3.541 seconds, with no relaxation of an earlier test. The original six integration tests passed their first execution in 6.124 seconds. Two additional scoring and inference tests then passed on their first execution; the final **eight integration tests pass** in 6.320 seconds. Scoped lint passes.

## Models and numerical contract

The source transformation supplies 24 raw features and a 26-column baseline design. Its all-25-slope population scaling, three training-centered curvature features, and fixed-unit centered range extremity are retained exactly. Training and application rows are never separately filtered inside fitting. Exact constant extremity is centered at its first value, making its historical correction identically zero; the column and fold remain present.

The baseline reuses only the frozen `sign_memory_models.logistic_state` and `_newton` numerical machinery. Its objective is mean binary logistic negative log likelihood plus `.01` times the squared slope norm, with an unpenalized intercept. The initial intercept is the logit of the exact common training frequency and all slopes start at zero. The location stage freezes the baseline logit, adds one coefficient multiplying centered extremity, and penalizes that scalar by `.01*b²`. It starts at zero. It is a staged fit, not joint reoptimization of the augmented model.

Both stages retain the original deterministic Newton–Armijo controls: at most 200 Newton updates, 60 line-search trials per update, Armijo constant `1e-4`, and full-gradient infinity norm at most `1e-8`. No restart, fallback or clipping is added. After each returned fit, the original objective and every gradient coordinate are recomputed; solver success, saved objective, saved gradient and saved maximum must match that recomputation. A false success flag cannot substitute for the gradient gate.

Issued logits are checked for finite arithmetic and nonzero coefficient-product underflow. `expit` probabilities may legitimately be exactly zero or one; no clipping is applied. Scoring reuses the frozen checked Brier arithmetic, including rejection when a nonzero probability error squares to an unrepresentable zero. A baseline or location probability is validated for every application before any future-label mask, including an entirely unscored month.

## Calendar, preflight and recent-frequency control

The input feature frame is exactly `RAW24 + feature_cutoff_date`; the target frame is exactly `y,target_end,available_date`, on the same full observed SPX index. Dates must retain the exact prior-session cutoff and next-session label maturity. The complete bounded calendar is preserved. Unknown numerical cells may be NaN; observed infinity, nonbinary labels, invalid known extremity or malformed chronology cause failure rather than row repair.

Monthly application dates are selected using complete raw features and known cutoffs, before looking at application labels. Training uses complete common features, known binary labels, origins before the fit, and labels available no later than the fit's preceding-session cutoff. Every fold needs at least 1,000 rows and 50 examples of each class. Preflight validates **all** scheduled training/application transformations and training support before any optimizer is called, then requires at least 127 observations and 30 of each class in both scored phases, plus 15 of each class in each of the two evaluation slices. The slices must partition the scored evaluation cohort. A late unsupported fold or application transformation blocks the entire run before its first fit.

The frequency control uses the frozen 63-session decay `delta=2**(-1/63)` and contribution `1-delta`. At the first application's previous-session cutoff `k0`, set `S` to the first fit's eligible training event mean and `W=1`. The seed's last included availability date is recorded separately from `k0`; its training subset does not represent every historical label. No label with availability at or before `k0` is fed again.

For each subsequent full reference-calendar close, first multiply both `S` and `W` by `delta`. If its unique next-session label arrival is known, add `(1-delta)*y` to `S` and `(1-delta)` to `W`. Unknown arrivals add nothing. This stream includes labels from feature-incomplete and unscored origins. There is no reset at a monthly refit, phase boundary or feature gap. Each application reads the state only through its own previous-session cutoff.

The state uses the frozen checked product/division rules and requires `0 <= S <= W <= 1` and `W > 0`; its forecast is exactly `S/W`. Nonzero multiplication or division underflow fails. State records are saved at every application, and elapsed-session/update metadata accounts for intervening reference closes. The independent verifier reconstructs the filter by explicit aged weighted sums rather than repeating the recurrence.

## Exact public interfaces and saved audits

- `fit_predict(train_features, y_series, apply_features) -> (predictions, audit)`. Predictions contain only `baseline` and `location` arrays. Training labels must have exactly the training feature index.
- `training_mask(features, targets, fit_entry, min_train=1000)` returns the complete mature common-row mask and checks training class support.
- `application_calendar(features, targets, index_config)` returns the full application `DatetimeIndex`, a full-calendar scoreable-label mask, and the development end date.
- `preflight(features, targets, index_config)` performs the optimization-free checks above.
- `forecast_panel(features, targets, index_config) -> (panel, fits, states)` invokes full preflight before fitting.
- `validate_panel(panel)` validates the exact three-model schema, shared cohort/labels/metadata, probability domains, at least 1,000 training rows, monthly chronology and exact saved Brier losses. The runner separately binds literal protocol phase fences and full source-calendar identities.

The panel preserves the frozen 15 columns: `origin,model,horizon,feature_cutoff_date,target_end,available_date,y,probability,loss,fit_origin,fit_cutoff_date,train_n,train_last_target,train_last_available,phase`. Model names are `baseline,recent_frequency,location`, with exactly three scored rows per admitted origin.

Each fit stores `fit_origin,fit_cutoff_date,train_n,train_first_origin,train_last_origin,train_last_target,train_last_available,application_n,model_audit,application_origins,application_probabilities`. Application origins are date strings. The last dictionary saves the `baseline` and `location` arrays for every application, including unscored ones. The recent-frequency probabilities live in the complete application state table.

`model_audit` contains `train_n,application_n,support,transform,frequency,baseline,location`. Support is `{n,events,nonevents}`. Transform is the full feature-module audit: baseline columns/means/scales, curvature means, extremity mean/scale/exact-constant status and training count. Frequency records the training probability and row counts; only the first fit's value seeds the continuous control. Baseline records coefficients, columns, means/scales, penalty, counts and the original solver audit. Location records its column, mean, fixed scale one, scalar coefficient, penalty, counts, `baseline_frozen=true`, `EXACT_CONSTANT_INPUT` or `FITTED`, and the original scalar solver audit.

Preflight returns `common_application_origins,common_scored_origins,monthly_fits,fits,phases`. Each preflight fit has the usual fit/date/count metadata plus support and transform audit. Each phase has name, count, first/last origin, support and slices; slices record their fixed start/end, count and support.

The state table keeps the exact frozen causal-pool 15-column schema: `origin,feature_cutoff_date,source_fit_origin,seed_fit_origin,seed_cutoff_date,seed_last_available,seed_probability,seed_train_n,S,W,recent_frequency,latest_consumed_available,cumulative_updates,elapsed_sessions,scored`.

## Synthetic integration and limits

The eight integration tests generate 2,100 positive OHLC/IV observations, construct the actual new features and risk-event targets, and introduce an explicit unavailable-label first month and an IV-feature gap. The unavailable-label mask is a deliberate model-interface stress case, not a claim about another raw provider. Every monthly producer fit, both conditional forecasts, all recent-frequency states and the complete support audit match the real independent verifier. Tests reject altered coefficients, scalar correction, training support, issued or unscored application probabilities, seed metadata and state values. Changing a later label cannot change earlier fits or states.

The generated panel also passes the actual runner's scoring and the independent verifier's inference reconstruction: four phase comparisons, 12 block-bootstrap calculations, six descriptive calibration rows, two new hypotheses and a cumulative family of 129. This fixture uses 199 bootstrap draws and 127 synthetic inherited comparisons with p-values of one. The prospective protocol's 199,999 draws remain unchanged, and neither helper reads real earlier metrics during the integration. Mutated application counts, family size, inherited rows and phase uncertainty are rejected.

These are implementation tests, not signal evidence. They neither inspect historical event counts nor validate archival original-vintage availability. The main runner and independent verifier retain the registered family, inference, source and failure-publication responsibilities. No empirical forecast is authorized by this implementation record alone.
