# Issued-calibration implementation record

Only generated synthetic inputs are authorized during implementation. Previous models, tests, protocols and reports remain unchanged.

The 16 initial model contracts were written before the implementation existed. The first run exited 1, establishing the absent-module red result before implementation:

```text
python -m unittest tests.test_issued_calibration_models -v
ImportError: cannot import name 'issued_calibration_models' from 'src'
Ran 1 test in 0.000s
FAILED (errors=1)
```

This is an implementation-order record, not an empirical model failure. All 16 tests passed on the first implementation in 8.382 seconds. An additional prewritten identity test also passed: perturb an original unscored application probability by `1e-13`, within the existing strict replay tolerance, and require an exactly zero correction to preserve that saved value rather than replace it with the slightly different replayed sigmoid. The final **17 model tests pass in 10.470 seconds**.

Seven generated integration tests were written before the independent integration interfaces were finished. Their first run exposed the still-absent `verify_forecasts` and `inherited_rows` exports; a subsequent focused run caught an undefined numeric-validator alias in the new independent module. These are pre-freeze integration findings, not historical failures. After those interfaces were completed, all **seven unchanged integration tests pass in 26.141 seconds**, including independent reconstruction of four phase comparisons, 12 bootstrap calculations and the complete 131-hypothesis family. No numerical tolerance or assertion was relaxed. Scoped lint passes for all three owned Python files.

## Fixed probability rule and solver

The sole new forecast is `calibrated`; `baseline` and `recent_frequency` are copied unchanged from the original wave-18 panel. There is no monthly baseline refitting. Historical baseline logits are reconstructed from the original 26-column geometry, curvature centers and coefficients on every original application, including unscored applications. Saved probabilities must replay with relative tolerance `1e-10` and absolute tolerance `1e-12` before any new scalar optimization. Original application lists, monthly provenance, complete mature geometry/support and rate states are checked without calling an old optimizer.

At the first original application's prior-session cutoff, the calibration history is empty and its intercept is zero. Every subsequent full reference-calendar close ages each existing weight by `delta=2**(-1/63)`. The single newly mature label contributes weight `1-delta` only when its origin had an original issued baseline and its label is known. Weights are never normalized by their sum. No label available at or before the initial cutoff is admitted, and no phase, month, feature-gap or missing-label reset occurs.

Arrival precedence is explicit: `NO_ISSUED_FORECAST` if no original application exists, regardless of label availability; otherwise `UNKNOWN_LABEL` for an unknown label, and `ADMITTED` for a known mature binary label. The original rate control deliberately retains its broader known-label stream. Neither missing issuance nor a missing label is repaired by retrospective prediction or by inventing an outcome.

For historical offsets `eta`, labels `y` and weights `w`, the scalar objective is

`sum(w * logaddexp(0, (1-2*y)*(eta+a))) + .01*a²`.

The derivative uses `expit(eta+a)` for a nonevent and `-expit(-(eta+a))` for an event, then adds `.02*a`. Thus rounded probability endpoints do not erase a representable signed residual. Its derivative is at least `.02` and at most `.02 + .25*sum(w)`, proving a unique minimizer even for single-class or empty histories.

The producer uses exactly one Brent derivative solve on `[-(sum(w)+1)/.02, +(sum(w)+1)/.02]`, with `xtol=1e-12`, `rtol=1e-14`, and at most 200 iterations. Strict finite bracket signs, reported convergence and a recomputed original full-gradient magnitude at most `1e-8` are mandatory. Empty history and exactly balanced `g(0)==0` return canonical zero without a root call. There is no restart, alternate bracket, warm start, clipping or fallback. Independent verification uses bisection rather than Brent, with its prospectively fixed gate and comparison tolerances.

At exactly `a==0`, copy the original saved baseline probability after mandatory logit replay; this preserves exact cold-start and balanced-history nesting despite permissible replay roundoff. At nonzero `a`, issue `expit(replayed_eta+a)`. All original controls retain their exact values, labels, metadata and scored cohort. Every application probability is checked before selecting scored labels. Brier scoring uses the frozen checked squared-error implementation; genuine probability endpoints and zero errors remain valid.

## Arithmetic and rejection rules

Scalar inputs must be finite real numeric arrays, with aligned one-dimensional shapes, exact binary labels, strictly positive record weights and total weight at most one. Empty arrays represent only an empty history. Booleans, strings, object arrays, observed infinities and malformed dates are rejected rather than converted into usable observations.

Every scalar reduction uses `math.fsum`; the likelihood and gradient are sums, not averages over records or normalized effective weight. A finite-logit signed softplus and absolute signed binary residual must remain strictly positive. For example, a correctly classified logit of 40 can yield a saved probability of one while retaining its small finite loss and gradient; a correctly classified logit of 1,000 whose loss/residual underflows is rejected. Weighted products, decayed weights, coefficient products, penalties and coordinate divisions reject nonzero underflow. Finite sigmoid rounding to an endpoint remains valid for issuance. No tolerance is an authorization to replace an invalid primitive with zero.

The immutable original baseline geometry and coefficient arithmetic are checked before any new scalar fitting. A malformed unscored application's probability, a missing original monthly fit, an altered original control or a mismatched state audit blocks the whole new trial. A purported numerical solver success is rechecked against the original derivative; it cannot bypass stationarity.

## Public interfaces and audit schemas

- `calibration_objective(a, eta, y, weights) -> (objective, gradient)` uses the fixed signed arithmetic above.
- `fit_intercept(eta, y, weights) -> audit` performs one bounded derivative solve.
- `replay_issued(features, targets, upstream_panel, upstream_fits, upstream_states, original_index_config) -> (issued_records, support)` checks all original applications and controls before new optimization.
- `forecast_panel(features, targets, upstream_panel, upstream_fits, upstream_states, original_index_config) -> (combined_panel, calibration_states, calibration_audit)` applies the causal memory and preserves the original scored cohort.
- `validate_panel(panel)` checks the exact three-model panel and inherited chronology, domains, class-count floor, shared metadata and Brier identities.

The panel retains the original 15-column schema: `origin,model,horizon,feature_cutoff_date,target_end,available_date,y,probability,loss,fit_origin,fit_cutoff_date,train_n,train_last_target,train_last_available,phase`. The fit fields describe the immutable source baseline fit; the new daily scalar's cutoff and diagnostics live in the separate state/audit records.

`STATE_COLUMNS` has 20 fields: `origin,feature_cutoff_date,source_fit_origin,seed_cutoff_date,elapsed_sessions,history_n,weight_sum,latest_admitted_origin,latest_admitted_available,missing_label_arrivals,no_forecast_arrivals,intercept,baseline_logit,baseline_probability,calibrated_probability,scored,status,objective,gradient,gradient_max_abs`. Latest-admitted dates are `NaT` until the first eligible record. One state is retained for every original application.

Each scalar audit records `intercept,alpha,history_n,weight_sum,objective,gradient,gradient_max_abs,gradient_at_zero,bracket,bracket_gradients,iterations,function_calls,converged,status,method,xtol,rtol,maxiter,curvature_lower,curvature_upper`. Status is `EMPTY_HISTORY`, `BALANCED_AT_ZERO` or `FITTED`; method is the fixed `brentq` contract even when the canonical zero branch avoids a solver call. Function-call counts describe only the root solver, while the separately recorded initial and bracket checks occur outside it.

The top JSON audit stores `seed_origin,seed_cutoff_date,issued_applications,admitted_records,arrival_audit,calibrations,common_application_origins,common_scored_origins,replayed_monthly_fits,new_monthly_fits,support`. `new_monthly_fits` is zero. Issued records contain origin/cutoff/source-fit, finite original logit, saved probability and scored flag, with no future label. Admitted records contain origin, actual availability, original logit and mature label. Arrival records preserve every post-seed reference close through the final application cutoff and its fixed status. Calibrations contain origin plus the full scalar audit. Support is the unchanged optimization-free original preflight audit.

## Synthetic validation and limits

The initial tests include a 70-digit Decimal scalar-root oracle, analytic reflection and balanced-history symmetry, the unnormalized first-information effect, genuine empty nesting, representable rounded endpoints, numerical underflow/domain failures and false solver-success injection. A generated 2,100-session original model supplies real monthly saved coefficients and all application forecasts. Tests verify exact copied controls, original input preservation, explicit full-calendar weight sums, unavailable-label and feature-gap behavior, strict probability replay, and altered geometry/provenance rejection before any new scalar solve.

The additional integration suite requires independent reconstruction of every issued logit, mature record, weighted scalar fit and application forecast. Changing later labels or features changes later original fitted coefficients but cannot change earlier fits, issued logits or calibration states. An original unscored development-boundary application enters memory only after its actual label matures; an entire feature-incomplete month never supplies fabricated calibration predictions. Generated inference uses 199 bootstrap draws and 129 synthetic inherited p=1 hypotheses, leaving the prospective 199,999-draw protocol and real previous artifacts untouched.

These checks provide implementation evidence only. No historical calibration records, event associations, coefficients, forecasts or scores were constructed for this record. The separate registered runner and independent verifier retain responsibility for source admission, the complete 131-hypothesis family, inference gates and fail-closed publication.
