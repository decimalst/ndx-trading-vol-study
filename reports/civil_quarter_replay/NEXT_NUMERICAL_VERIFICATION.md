# Prospective numerical verification design

This is a proposal for a separately registered verifier, not an approved implementation or a repair of wave 16. It follows a code-only review of [the independent objective](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/verify_civil_quarter.py) and [the producer objective](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/civil_quarter_models.py). No new data, features, targets, fits, scores, or numerical experiments were inspected or constructed for this note. Wave 16's failed verification remains unchanged. A later trial must retain its failure in the research ledger and register its own numerical method before execution.

The recommendation is to eliminate the unpenalized intercept analytically, solve the resulting strictly convex slope problem using directional gradients, and certify the answer against the original full objective. This changes the verification algorithm, not the model, target normalization, design geometry, or penalty. Objective rounding is a numerical mechanism to test; this derivation alone does not diagnose the observed failure.

## The same objective with one fewer parameter

Let the registered design row be `(1, x_i)`, with 30 slope features, and let `q_i = y_i / train_mean > 0` use the existing target mean. Keep the existing transformed feature values and their actual sample means; do not assume that floating-point centering made every mean exactly zero. With intercept `a`, slopes `b`, and `lambda = .01`, the objective is

\[
F(a,b)=a+\bar x^T b+e^{-a} A(b)+\lambda\|b\|_2^2,
\qquad A(b)=\frac1n\sum_i q_i e^{-x_i^T b}.
\]

The intercept derivative is `1 - exp(-a) A(b)`. Its unique zero is

\[
a_*(b)=\log A(b).
\]

Consequently the profiled objective is exactly

\[
J(b)=1+\log A(b)+\bar x^T b+.01\|b\|_2^2.
\]

The linear sample-mean term must remain. Dropping it merely because the inputs were described as centered would change the objective. Likewise, do not renormalize `q` a second time to force its floating-point mean to one. At the proposed zero-slope start, compute the intercept as `log(mean(q))`.

Define normalized positive weights

\[
w_i(b)=\frac{q_i e^{-x_i^Tb}}{\sum_j q_j e^{-x_j^Tb}},
\qquad m_w=\sum_i w_i x_i.
\]

Then

\[
g(b)=\bar x-m_w+.02b,
\qquad
H(b)=\sum_i w_i(x_i-m_w)(x_i-m_w)^T+.02I.
\]

The covariance form should be evaluated directly, rather than subtracting two large second moments. The penalty gives `H >= .02 I`, even with collinear slope columns. The profiled objective is coercive and strongly convex, so its slope minimizer is unique; the intercept is then unique as well. This does not justify changing any existing scientific rank or support gate.

In exact arithmetic, a profile-gradient residual supplies useful global certificates:

\[
\|b-b_*\|_2\le \frac{\|g(b)\|_2}{.02},
\qquad
J(b)-J(b_*)\le \frac{\|g(b)\|_2^2}{.04}.
\]

These are slope and objective bounds, not automatic certificates for the intercept, application predictions, or quarter coefficient. Those retain their separate checks. A floating-point report must identify the computed residual and its arithmetic qualification rather than describe these as interval-arithmetic proofs.

## Stable differences and an independent solve

Compute `log A` by a checked log-sum-exp reduction of `log(q_i) - x_i'b`. Use compensated reductions for weighted means and dot products. Reconstruct the original positive ratios at accepted states; an unrepresentable positive ratio remains an error. No clipping, target flooring, discarded weights, or altered penalty is permitted as a numerical repair.

For a proposed slope displacement `delta`, the exact objective change is

\[
\Delta J=\bar x^T\delta+
\log\!\left(\sum_i w_i e^{-x_i^T\delta}\right)
+.01(2b^T\delta+\delta^T\delta).
\]

For small projected displacements, evaluate the logarithmic term using

\[
\operatorname{log1p}\!\left(
\frac{\sum_i u_i\operatorname{expm1}(-x_i^T\delta)}{\sum_i u_i}
\right),
\]

where `u_i` are the common unnormalized log-sum-exp weights. Keeping the denominator explicit avoids assuming that rounded normalized weights sum to exactly one. A fixed prospective branch, such as `max(abs(x_i'delta)) <= .5`, makes this formula safe from large exponential excursions; use a checked log-sum-exp difference otherwise. Evaluate the penalty change in its expanded form above. Do not subtract two complete objective values near the optimum. The branch and all arithmetic checks must be tested and frozen before a new run.

One practical solver proposal is a zero-start **profiled Newton direction with a bracketed directional-gradient line solve**, limited to 500 outer states. Form the weighted-covariance Hessian independently and obtain a descent direction by Cholesky solution. Normalize it to a unit vector `v`; require `g'v < 0`. Strong convexity gives a derivative bracket without comparing rounded objective totals:

\[
R=\frac{2(-g^Tv)}{.02},
\qquad
\frac{d}{dt}J(b+tv)\bigg|_{t=0}<0,
\qquad
\frac{d}{dt}J(b+tv)\bigg|_{t=R}>0.
\]

Use a fixed Brent derivative-root solve on the dimensionless interval `[0,1]`, evaluating `g(b + s R v)'v`; proposed limits are `xtol=1e-14`, `rtol=1e-14`, and 200 iterations. Check the actual finite endpoint signs rather than treating the analytic bound as a substitute for numerical validation. An invalid endpoint, factorization, bracket, accepted state, or exhausted budget fails the future verifier. There is no empirical restart, warm start from saved producer coefficients, or fallback optimizer. Retain the stable objective change as a descent diagnostic.

Accepted states must satisfy the original positive-ratio domain. Trial endpoints also need representable auxiliary quantities for the proposed derivative evaluator, which is a separate numerical requirement: the analytic `R` can be large enough to underflow a small positive weight even when ratios near the minimizer are representable. Under the proposed rule that is a verifier failure, not proof that the model's optimum is invalid. Do not silently discard the weight. Synthetic tests must establish whether this conservative bracket is practical before registration; changing its policy would require another prospective specification, not an empirical rescue.

This still uses Newton directions, but it is independent of the producer's full-parameter Newton–Armijo implementation: it eliminates a parameter, forms a different Hessian, uses separate reductions and factorization, and selects steps from a monotone derivative root rather than an absolute-objective acceptance test. It also replaces the old verifier's trust-region objective-reduction decision. These differences require independent implementation and synthetic comparison; they are not a claim that shared mathematics constitutes independent evidence by itself.

## Acceptance and retained evidence

Target a stricter internal profile gradient, for example `max(abs(g)) <= 1e-10`. Then recover the intercept and recompute **every coordinate of the original full gradient** using the original design and normalized targets. For the expression `1 - exp(log(q) - eta)`, `-expm1(log(q) - eta)` supplies a useful independent stable evaluation. Do not set the intercept gradient to zero by construction.

Preserve the frozen full-gradient gate `1e-8 + 1e-12`, including its existing arithmetic allowance; neither optimizer success nor an unchanged printed objective can override it. Preserve the existing coefficient, normalized-prediction, saved-coefficient replay, scalar-quarter, domain, source, maturity, and support checks. The quarter correction remains the same separately checked scalar problem after freezing the verified baseline. Any new algorithm must be registered prospectively; this proposal does not authorize another verification attempt on wave 16.

Save the zero start, iterations, endpoint derivatives, line-root status, profile/full gradients, objective-change diagnostics, intercept residual, and strong-convexity bounds. A stationary answer must also pass the original coefficient and prediction comparisons. Stagnation with an unacceptable full gradient is a failure, even if objective changes are below one ULP.

## Prewritten adversarial tests before registration

1. **Profile identity and derivatives:** generated positive targets and deliberately nonzero feature means; compare full/profile objectives and gradients, finite-difference derivatives, weighted-covariance curvature, and analytic intercepts against a higher-precision oracle. Include duplicate columns to verify that `.02 I` preserves uniqueness.
2. **Rounded objective stagnation:** construct a deterministic small problem whose current point and improved point have identical rounded full objective values while the current full gradient exceeds the gate. The derivative solve must either reach the unchanged gate or fail explicitly. Check stable objective differences against decimal or arbitrary-precision arithmetic. Do not depend on one SciPy version reproducing a particular trust-region status.
3. **False convergence injection:** simulate optimizer success, zero reported objective reduction, exhausted steps, and an unchanged parameter vector with an unacceptable gradient. Every path must reject; a library status cannot certify stationarity.
4. **Intercept and penalty traps:** near-constant targets, nearly dependent features, rescaled target units, and nonzero floating-point centering residuals. Catch accidentally penalizing the intercept, using `.01` instead of `.02` in derivatives, renormalizing targets twice, or omitting the sample-mean term.
5. **Line and arithmetic boundaries:** check the derivative bracket and unique line root, representable extreme ratios, checked underflow/overflow failures, tiny displacements, zero-gradient termination before direction normalization, and candidate-versus-full-objective coherence. Fix arithmetic tolerances from these synthetic cases before any empirical use.
6. **End-to-end independence:** retain the existing synthetic monthly schedule, unscored-month, source-unknown, full-cohort and tampering tests. Compare all saved coefficients and application predictions, then confirm that a failed independent gate still invalidates the entire registered family without changing earlier artifacts.

The bounded next step is to implement and challenge this verifier on synthetic problems under a new prospective protocol. The hypothesis, information set, scoring rule, penalty, and scientific admission gates need no relaxation to investigate the numerical failure mode.
