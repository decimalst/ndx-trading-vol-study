# Prospective cross-moment target alignment

**Blinded, unregistered proposal.** No current products, scores, metrics, numerical forecast values, or outcomes were inspected. No source, model, or protocol was changed. The question is useful whether the current product-score experiment passes or fails: does a small model trained directly for the residual cross moment improve its prediction? The current test only applies a new score to forecasts trained for matrix QLIKE; it does not answer that training question.

The recommended next experiment adds two small scalar estimators while keeping the existing conditional means, marginal second moments, source features, calendar, and completed-label folds fixed. It uses closed-form constrained least squares. It introduces no new information channel or latent model. It can test a specific modeling failure mode, but cannot by itself prove that any improvement was caused solely by changing the loss: its bounded affine correlation function also differs from the earlier tanh-index function.

## One minimal family

For each original monthly fit, replay the saved shared-mean and marginal-second-moment coefficients on exactly that fit's mature training rows. Do not refit or recalibrate them. Write `e_Q,e_S` for those current-fit training residuals, `Y=e_Q*e_S`, and `D=sqrt(h_Q)*sqrt(h_S)`. Standardize the existing lagged corr22 using its mean and population scale on those exact rows, and set `phi=tanh(z_corr22)`. This transformation is fixed and bounded; its parameters use no application rows.

First fit a strong target-aligned constant-correlation control:

`a = project_[−c,c](mean(D*Y)/mean(D²))`, with fixed `c=0.995`.

Its forecast is `q0=a*D`. This is the constrained least-squares solution for a constant correlation given the fixed conditional diagonals. It corrects an important attribution weakness: comparing a newly MSE-trained slope only against a QLIKE-trained constant could reward changing the intercept's objective rather than useful state dependence.

Next hold `a` fixed, define `s=c−abs(a)` and `w=s*D*phi`, and fit exactly one additional coefficient:

`b = project_[−1,1](mean(w*(Y−q0))/mean(w²))`.

The candidate predicts

`rho=a+b*s*phi`, `q1=D*rho`.

This is a one-dimensional convex squared-loss problem with its exact bounded least-squares solution; zero slope nests the control. It needs no local optimizer, grid, extra window, or penalty search. The fixed bounds constrain the added term. If a denominator is nonfinite or nonpositive, or the baseline saturates at `abs(a)=c` and removes all slope headroom, declare the specified family unevaluable. Do not widen a bound or substitute a fallback after seeing the data. Individual zero targets remain valid; if every training product is zero, the constant coefficient is zero and ordinary candidate fitting remains possible when its regressor is nondegenerate.

Because `abs(phi)<=1` and `abs(b)<=1`, `abs(rho)<=c` for every future input. With finite positive issued diagonals this defines a positive-definite matrix whose normalized correlation eigenvalue is at least `0.005`. The bound is structural, not a forecast clip. It must not be chosen from the largest future application signal. This is compatible with reporting a matrix, but the proposed primary score is the **direct unweighted product squared loss**, not matrix QLIKE. Correct positive-definiteness does not establish correct marginal calibration.

## Controls and the precise new claim

Register three candidate contrasts, all on the same issued means and actual return rows:

1. New target-aligned dynamic cross moment versus the new target-aligned constant correlation, isolating the incremental bounded state term within this family.
2. New candidate versus the original constant residual-matrix cross moment, the same-row historical product-mean benchmark.
3. New candidate versus the frozen original QLIKE-trained dynamic-correlation cross moment, checking whether the new specification actually improves the previously issued alternative for this functional.

Require the candidate to pass all three declared comparisons. There are **three additional hypotheses**, not two reused labels from the current wave. If this is the next wave after 114 registered comparisons, the cumulative family becomes **117**, with the next telescoping wave allocation registered before construction. The exact effect, inference, multiplicity, and failure rules must be frozen anew; reusing the existing decimal-log-return units and `1e-10` reference product-MSE threshold would require an explicit prospective commitment, not selection based on the current result.

A success would establish a better target-aligned bounded cross-moment specification against these controls on reused history. It would not isolate the loss-function change from the affine-versus-tanh model-class change. An experiment attributing causality specifically to the training objective would need the same model class and restrictions under both objectives; that extra experiment is not included here. Neither a win nor a loss alters the original matrix- or product-score verdicts.

## Means, target availability, and timing

The proposed training products use residuals about the **current monthly fit's** shared means. Those coefficients were fitted using labels available by that fit's previous-SPX-session cutoff, and all training returns used to form products must have matured by the same cutoff. This is the existing disclosed staged training convention. It is causal at the fit date but produces in-sample training residuals; it must not be described as a history of previously issued residuals.

For application and scoring, retain the exact means actually issued by the frozen upstream model for that origin. The new realized response is `(next_return_Q−issued_mu_Q)*(next_return_S−issued_mu_S)`, available when the two next-session intraday returns mature at the next observed SPX close. Historical feature means/scales, both new scalar fits, and every training mask use only information available at the monthly fit cutoff. The original full SPX calendar, source-predecessor rules, 2016–2019 development fence, 2020–2025 evaluation periods, 2025-10-20 source/target limit, and sealed-period exclusion remain candidates for the unchanged chronology, subject to registration.

There is a different legitimate target convention: train on products about means that were actually issued at each historical origin. That would require an existing past-issued mean and its subsequently matured return for every training label. The archived forecast sequence begins at the earlier experiment's forecast start, so it cannot silently supply issued-mean training labels from before that start. Maintaining a 1,000-row initial gate would require a separately audited later training/evaluation schedule. This alternative is **not** part of the minimal proposal and must not be mixed with current-fit residuals to fill history.

In either convention, `E[e_Q e_S | information]` equals true conditional covariance plus the product of the two remaining conditional mean errors. Direct fitting can improve that residual functional without discovering pure correlation dynamics. Replaying frozen marginal fits rather than improving them preserves a clear, limited attribution question; it does not prove their correctness.

## Preconditions for any run

Register the complete specification and prewrite tests before constructing new empirical training products. Tests should prove the two constrained least-squares solutions, uniform rather than volatility-weighted MSE, exact zero-slope nesting, positive-definiteness for arbitrary finite application signals, asset/unit invariance, zero versus nonfinite/underflow arithmetic, and failure at zero headroom or degenerate training geometry. Any constant normalization for numerical conditioning must leave these closed-form coefficients and unweighted objective unchanged; it cannot introduce observation-specific weights.

Use the existing read-only upstream admission and byte snapshots. Reconstruct only the declared training geometry from the already bounded sources and frozen fit records, then fit the two new scalars; never alter earlier forecasts or reports. Numerical feasibility, headroom, power, and predictive benefit are untested. This is an eligible next design, not a completed experiment or confirmed signal.
