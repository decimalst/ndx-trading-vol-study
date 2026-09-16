# Synthetic-tested independent baseline solver

This new solver is prospective wave 17 work. It imports no producer or previous verifier, reads no files, accepts only supplied numerical arrays, and has been exercised only on generated synthetic inputs. It does not authorize a rerun or reinterpretation of wave 16. The producer's positive-score model, target mean, feature geometry and `.01` slope penalty remain unchanged.

The public interface is `solve_baseline(design, normalized_y) -> (beta_full, audit)`. The design must have 31 columns, a literal leading intercept, and at least one row. Targets must be a finite, strictly positive vector of the same length. Integer and floating numerical arrays are accepted; boolean, string, object and complex arrays are rejected before conversion. The caller supplies its existing normalized targets. The solver does not renormalize them.

The derivation in the [prospective note](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/civil_quarter_replay/NEXT_NUMERICAL_VERIFICATION.md) applies directly. For actual slope means `mean(x)`, the profile is `1 + log(mean(q exp(-x b))) + mean(x)b + .01||b||²`. Its gradient is `mean(x) - weighted_mean(x) + .02b`; its Hessian is the directly formed weighted covariance plus `.02I`. The intercept is recovered analytically at every state. The start is exactly 30 zero slopes with intercept `log(mean(q))`, evaluated through the same checked log-sum-exp arithmetic as subsequent states.

## Fixed solver and arithmetic

The method identifier is `profiled_newton_directional_brent`. The numerical controls saved verbatim in every returned audit are:

```json
{
  "internal_tolerance": 1e-10,
  "full_tolerance": 1.0001e-8,
  "max_iterations": 500,
  "max_bracket_evaluations": 60,
  "max_root_iterations": 200,
  "root_xtol": 1e-14,
  "root_rtol": 1e-14,
  "small_difference": 0.5
}
```

`full_tolerance` is the Python expression `1e-8 + 1e-12`, preserving the existing gate and its existing arithmetic allowance. `alpha` remains `.01` outside this dictionary. There are at most 500 evaluated outer states; a successful final state must have profile gradient infinity norm at most `1e-10` and an independently recomputed original 31-coordinate gradient infinity norm at most `1e-8 + 1e-12`.

Each state uses a Cholesky solve of the profile covariance Hessian to obtain the Newton direction. Normalize that direction to a unit vector `v`, require `g'v < 0`, and retain the analytic bound `R = 2(-g'v)/.02`. The following is one deterministic line-bracketing algorithm:

1. Start at distance `min(norm(Newton_direction), R)` from the current slopes. The initial left endpoint is zero with its finite negative directional derivative.
2. A finite negative derivative replaces the left endpoint. With no invalid upper barrier, double the trial distance, capped at `R`.
3. A trial with invalid numerical arithmetic becomes an upper barrier. Bisect between the last finite negative-derivative point and that barrier. Retain every rejected trial and its error. Subsequent finite negative points still bisect toward the barrier; an invalid midpoint lowers the barrier.
4. The first finite nonnegative derivative supplies a valid right endpoint. Reject exhaustion of 60 bracket evaluations, a nonmoving midpoint, or a finite negative derivative at the analytic cap.
5. An exactly zero right-endpoint derivative is an exact line root and records zero root iterations/calls. Otherwise run Brent's derivative root method on the dimensionless `[0,1]` representation of the finite bracket with the fixed controls above. Require convergence and a changed coefficient vector. There is no optimizer restart, warm start from producer coefficients, fallback method, or trial-dependent change of penalty.

The original full positive-ratio domain is checked at every accepted state. Initial source/model-domain failures are rejected before line search. Invalid trial points may be bracketed around, but no positive weights or ratios are discarded, clipped, floored, or silently replaced by zero. A line root with invalid arithmetic fails the solver. A permanent barrier that prevents a finite derivative sign change also fails within the fixed budget.

Scalar sums, means, weighted means and dot products use `math.fsum`; a nonzero sum divided by its count may not silently underflow to zero. Matrix products use NumPy after checking every primitive product for nonfinite results and nonzero-to-zero multiplication underflow. Matrix-vector products receive the same product admission checks. The covariance is formed from centered weighted observations, not by subtracting two second moments. Finite nonzero subnormal values remain admissible when all required operations are representable. Final full-gradient residuals use `-expm1(log(q)-eta)` instead of subtracting a rounded exponential from one.

Objective changes never use a difference of two complete objective totals. When all absolute projected displacements are at most `.5`, the logarithmic change uses `log1p(sum(u*expm1(-x*delta))/sum(u))`, with the unnormalized weights and their explicit denominator. Otherwise it uses the change in checked profile log normalizers. The sample-mean contribution and expanded penalty change are added by compensated summation. A positive computed stable objective change rejects the step; rounded equality of complete objectives cannot certify convergence. The whole-state gradient gates remain mandatory.

## Why the bracket differs from the initial prospectus

A prewritten synthetic two-row example uses slope observations `(-10,10)` and targets `exp(-10), exp(10)`, with the other 29 slopes zero. The original distant analytic endpoint produces exponential underflow although the unique minimizer is representable. Even the initial Newton distance is too far. The fixed barrier-bisection rule finds a finite derivative bracket and matches a separate 75-digit Decimal scalar solution. This policy was developed before empirical use, solely from generated inputs. It is a replacement prospective specification, not a relaxation after a historical fit.

Another generated example multiplies targets by `exp(100)` and introduces a small asymmetric signal. The initial full gradient exceeds the unchanged gate, while initial and improved complete objectives round to the identical floating value. The new solver reaches stationarity, and its negative stable objective difference agrees with Decimal arithmetic. This test deliberately avoids requiring any particular SciPy trust-region status or version-specific failure.

## Audit schema and tests

The returned full coefficient vector has length 31. The JSON-safe audit contains `method`, `train_n`, `alpha`, `start_slopes`, `start_intercept`, `iterations`, `success`, `objective`, `profile_objective`, `gradient`, `gradient_max_abs`, `profile_gradient`, `profile_gradient_max_abs`, `slope_distance_bound`, `objective_gap_bound`, `controls`, and `history`. The last two bounds are the computed-gradient strong-convexity formulas `||g||₂/.02` and `||g||₂²/.04`; they are numerical residual-based bounds, not formal interval-arithmetic certificates.

Each history row records `iteration`, complete `objective`, both gradient maxima, `newton_direction_norm`, `analytic_radius`, `bracket`, `bracket_derivatives`, `bracket_trials`, `invalid_bracket_trials`, `root_iterations`, `root_function_calls`, `root_fraction`, `step_distance`, and `objective_difference`. A valid bracket trial records `distance`, `valid=true`, and `derivative`; an invalid trial records `distance`, `valid=false`, and `error`. `iterations` equals the number of accepted updates and the length of `history`. A zero-step optimum has an empty history.

The 23 prewritten tests cover high-precision scalar optima; unchanged target units; the unpenalized intercept; actual noncentered feature means; derivative and covariance identities; nearly collinear slopes; sub-ULP objective improvements; finite bracket barriers; permanent/initial domain failures; checked underflow; extreme representable constant targets; exact endpoint roots; row-order stability; invalid types/shapes; failure budgets; false optimizer success; authoritative full-gradient reconstruction; and a generated 3,600-row, 31-column fit. The full group passes in under one second locally with one BLAS thread. No historical timing estimate is claimed.

Initial absent-module failure, intermediate arithmetic/type/endpoint regressions, and final passing output are retained as `solver_*.txt` in this directory. Integration must still preserve the existing scientific, coefficient, normalized-prediction, source, chronology, scalar-quarter and publication checks. A valid numerical baseline alone does not qualify a predictive signal.
