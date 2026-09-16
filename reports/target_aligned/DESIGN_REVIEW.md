# Target-aligned cross moment: independent prefit review

This review reads the frozen [prospective note](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/cross_moment/NEXT_TARGET_ALIGNMENT_DESIGN.md) and assesses the new proposal algebraically. No new training products, numerical forecasts, scores, outcomes, or fits were inspected. The earlier note and all earlier protocols/results remain unchanged.

The design is sound with the explicit flat-objective convention and numerical distinctions below. It tests a small target-aligned specification using existing information. It cannot attribute a gain exclusively to the new loss because the bounded affine correlation function also differs from the earlier tanh-index function. It adds neither a new source channel nor independent historical evidence.

## The prospective flat-objective amendment is valid

The earlier unregistered prospectus proposed rejecting zero slope headroom. The new wave may prospectively replace that proposed branch **before any new training products are read**. If its entire slope regressor is exactly zero, the training objective does not depend on the slope: every coefficient in the declared interval is a minimizer. Choosing **`b=0`**, recording **`FLAT_OBJECTIVE`**, and retaining the fold, model, and scored observations is an exact deterministic solution. It is not dropping a component, omitting a difficult fold, or repairing a failed numerical fit.

At an intercept boundary with `s=0`, the candidate and aligned constant coincide on every application observation. If training regressors are zero for another admissible reason, canonical `b=0` also gives that identity without selecting a coefficient using application features or labels. Those zero paired gaps must remain in phase averages and uncertainty estimates.

This convention does **not** authorize treating numerical underflow as flatness. Verify regressor construction before testing `all(w==0)`. Any nonzero regressor with a computed squared norm of zero is a numerical failure. A nonzero multiplication or division that lost its value to zero must not enter the flat branch. The inputs and constant objective must remain finite even when the slope is flat; a zero slope cannot bypass an invalid residual or overflowing squared error.

## Normalization and constrained ratios

At one original monthly fit, let `D_i=sqrt(h_Qi)*sqrt(h_Si)>0` and `Y_i=e_Qi*e_Si` on its exact mature common training rows. Set

`k = mean(D_i)`, `d_i=D_i/k`, `v_i=Y_i/k`,

where `k` is one finite strictly positive training-only scalar. Then

`mean((Y_i−a*D_i)²) = k² * mean((v_i−a*d_i)²)`.

The factor `k²` is common to all observations and parameters. It changes neither minimizer nor observation weighting. This is different from minimizing an unweighted loss on `Y_i/D_i`, which would change the product-MSE criterion. Native forecast products and scored MSEs retain their original units.

For the aligned constant, define `A=mean(d_i²)>0`, `B=mean(d_i*v_i)`, and `c=0.995`. The unique unconstrained solution is `B/A`; the constrained solution is

`a = project_[−c,c](B/A)`.

The normalized objective has gradient `g_a=2*(a*A−B)` and Hessian `2*A`. Its KKT condition is gradient zero in the interior, gradient nonnegative at `−c`, or gradient nonpositive at `+c`. Projection here solves the constrained objective; it is not an after-the-fact forecast clip. Nonfinite arithmetic is a failure, rather than permission to project an infinite ratio to a finite endpoint.

Hold that fitted `a` fixed. Set `s=c−abs(a)`, `phi_i=tanh(z_i)`, `w_i=s*d_i*phi_i`, and `r_i=v_i−a*d_i`, with `z` obtained from the lagged corr22 using the exact training mean and population scale. Put `A_b=mean(w_i²)` and `B_b=mean(w_i*r_i)`.

For a nonflat regressor, the candidate is

`b = project_[−1,1](B_b/A_b)`.

Its gradient is `g_b=2*(b*A_b−B_b)` and Hessian `2*A_b`, with the same boundary sign rules. If every regressor is genuinely zero, `A_b=B_b=0`, the gradient and curvature are zero for all `b`, and canonical `b=0` is the declared minimizer. If any regressor is nonzero, require a finite strictly positive `A_b` before forming the ratio. There is no numerical denominator floor.

Each stage is a globally solved one-dimensional convex problem. The staged pair is **not claimed to minimize a jointly refitted two-parameter objective**: the first-stage intercept is held fixed for the slope comparison. Independent tests should reconstruct both ratios, objectives, projected gradients, boundary solutions, exact flat case, and the zero-slope nesting. No iterative optimizer or penalty tuning is required.

Normalization itself needs checks: original positive `D` must not become zero after division, a nonzero `Y` must not disappear into a zero normalized value, and all declared products, norms, residuals, squared losses, and summaries must remain finite and representable. Reject unsupported arithmetic without deleting observations, raising a floor, or changing the normalization after inspection. Exact zero targets, zero covariance forecasts, zero errors, and exact cancellation of a numerator remain valid. Audit `k`, row count, both ratios/coefficients, headroom, slope status, objectives, and KKT quantities using fixed prewritten numerical tolerances.

## Matrix interpretation and unchanged target limitations

At application, use the original issued marginal moments and predict

`rho=a+b*s*tanh(z)`, `q=D*rho`.

Since `abs(b)<=1`, `abs(tanh(z))<=1`, and `s=c−abs(a)`, the correlation magnitude is bounded by `c` for every finite future signal. With finite positive diagonals, the resulting matrix is positive definite and its normalized correlation eigenvalue is at least `0.005`. This guarantee uses no future signal range and needs no clipping or eigenvalue repair. It is a statement about the forecast matrix, not about the accuracy of the shared marginal models.

For application scoring, `Y=(r_Q−issued_mu_Q)*(r_S−issued_mu_S)` is the residual product about the exact common means already issued. Its conditional expectation equals conditional return covariance plus the product of remaining conditional mean errors. Direct unweighted MSE can improve this residual functional without establishing pure correlation dynamics, measured high-frequency covariance, or hedge profitability. Its units remain decimal-log-return to the fourth power; the `1e-10` useful-effect threshold retains its ex-ante reference-error interpretation.

## Causal frozen replay

Use read-only admission of the hash-pinned original source, fit, forecast, and verification artifacts. Replay saved shared mean/variance coefficients on the **exact original mature training mask**. Do not replace that mask with the later scored cohort, refit shared marginals, recalibrate historical means, or pool residuals from a different monthly model. The original correlations' training centers/scales must agree with those same admitted rows.

All source price/IV features end at the original previous-SPX-session cutoff. Training return labels and any current-fit residual products use only labels available by that cutoff. Although those training means were estimated at the current fit date, every training input and label was then available; this is the disclosed current-fit in-sample residual staging. It must not be described as a history of previously issued forecast errors.

For scored application rows, copy the exact original issued means, marginal forecasts, actual returns, dates, phases, and fit metadata. Products about those issued means mature only when the next observed SPX session's paired returns mature. Neither those future labels nor later application features can select a fit, center, scale, bound, or flat-branch coefficient. Preserve the full reference calendar, source predecessor checks, original monthly schedule, same-row controls, source/target fence at 2025-10-20, and sealed-period exclusion. A separate experiment using historically issued means for all training labels would need its own support/timing contract; it is not this design.

## Family and decision

Add `aligned_constant` and `aligned_dynamic` and retain the original constant residual matrix and QLIKE dynamic predictions as controls. Register three new product-MSE contrasts for `aligned_dynamic`: against `aligned_constant`, the original constant matrix, and the original QLIKE dynamic forecast. Require all three to pass the fixed `1e-10` decrease in both phases, both later evaluation signs, Holm3 at `.05/(12*13)`, and cumulative Holm117 at `.05`. The inherited 114 comparisons and their verdicts remain unchanged.

With 99,999 bootstrap draws, the minimum attainable p is `1/100000`, below one tenth of the strictest raw wave-12 Holm threshold, `1/93600`. That establishes resolution, not power. Pin seed 20260918 and the inherited fixed block/HAC/phase rules before construction; retain nominal MDE as a diagnostic without retrospective threshold adjustment or equivalence claims.

The exact flat-objective branch should therefore be admitted as part of the new preregistered estimator, while nonfinite and underflowed nonflat cases remain explicit failures. This preserves the whole common sample and the intended comparison. Numerical feasibility and predictive benefit remain untested; the review supports prospective implementation and testing, not a claim of a signal.

## Final implementation review before fitting

Read-only review of the new runner, panel validation, protocol, and synthetic inference/publication tests found the six phase analyses, fixed `1e-10` inference units, phase/block seed offsets, three required controls, Holm3/Holm117 accounting, and normal 120-event ledger consistent with this design. The panel requires the original complete cohorts and exact shared labels, means, timing, and conditional diagonals. Forecasts, features, targets, and saved fit JSON are decoded from the same byte payloads whose hashes are checked before the new fits. Caught publication failures replace canonical metrics with all three unevaluable `p=1` records and keep any scored metrics explicitly unpublished.

The review identified one protocol-identity gap: parsing before pretests and hashing afterward could associate the manifest with different protocol bytes from the configuration actually used. The corrected runner reads one protocol byte payload, derives both its parsed configuration and SHA256 from that payload, checks the current file against that initial hash after pretests and before input enumeration or registration, and stores that initial hash in the manifest. The final post-fit integrity check remains in place. The prewritten regression mutates the temporary protocol during the synthetic pretest call and requires rejection before input enumeration, manifest creation, or trial registration; the coordinating agent reports it passed with the 19 runner checks. The corrected source and regression assertions were independently inspected.

No remaining substantive issue was found within this bounded review. No new empirical products, fits, forecast values, or scores were read. This is prefit implementation review, not empirical verification or evidence of a predictive signal.
