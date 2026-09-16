# Prospective independent shape-calibration verification

This contract and its generated tests were completed before any new historical shape fit or outcome inspection by this reviewer. Existing source, market, baseline forecasts, protocols, and results remain unchanged. The new calibration changes marginal shape on reused history; synthetic validity is not empirical improvement or evidence of profitable option selling.

## Fixed transformation and normalized density

The original standardized issued error is `z=(y-mu)/sqrt(0.75*h)`, with original t8 marginal density `f0`. Define `w=Phi^-1(T8(z))`. The location `a=mean(w)` and scale `b=sqrt(mean((w-a)^2))` are fixed using the same expanding, mature issued-error pairs used by the earlier affine calibration. They are not jointly optimized with shape.

For each asset let `x=(w-a)/b`, `r=asinh(x)`, `u=delta*r-epsilon`, and `v=sinh(u)`. The shape CDF is `Phi(v)` and its log density is

`log(f0) + logphi(v) - logphi(w) + log(delta) - log(b) + logcosh(u) - 0.5*log(1+x^2)`.

The inverse maps a normal latent score `v` to `w=a+b*sinh((asinh(v)+epsilon)/delta)`, then to `T8^-1(Phi(w))`. Both directions use log-tail probability calculations without clipping. The independent inverse uses a scalar bracketed root solve in log t8 magnitude, whereas the new producer uses a vectorized fixed-point incomplete-beta inversion.

Fixed prospective bounds are epsilon in `[-0.75,0.75]`, delta in `[0.75,2]`, and affine scale strictly greater than `1e-12`. Epsilon zero and delta one nest the earlier affine density. Location zero, scale one, epsilon zero, delta one nest the original t8 density. The Jones–Pewsey sinh–arcsinh family supplies this monotone transformation; this study uses only its shape parameters after a separately fixed affine calibration. It does not fit an unrestricted four-parameter family. [Authors' explanation](https://academic.oup.com/jrssig/article/16/2/6/7029435), [original paper record](https://oro.open.ac.uk/22510/).

## Independent global convex certificate

Conditional on the affine estimates, the parameter-dependent mean negative log likelihood is

`F(epsilon,delta)=mean(0.5*sinh(u)^2-logcosh(u))-log(delta)`.

With `q=sinh(u)*cosh(u)-tanh(u)` and `k=cosh(2u)-sech(u)^2`, its gradient is `[-mean(q), mean(q*r)-1/delta]`. Its Hessian is the mean of `k*[-1,r]'*[-1,r]`, plus `1/delta^2` in the second diagonal position. Since `k>=0` and `delta>0`, the objective is convex on the fixed box. These direct hyperbolic derivatives are reconstructed separately from the producer's algebraically equivalent squared-sinh form.

For a saved feasible parameter vector theta, choose each box corner component as its lower bound when the corresponding gradient is nonnegative, otherwise its upper bound. Convexity gives the global lower bound `F(theta)-gradient dot(theta-corner)`. The independent checker requires the reconstructed nonnegative box gap and projected KKT residual each to be at most `1e-7`, and compares every saved objective, gradient, Hessian, lower bound, bound interval and parameter. A self-consistent audit of a materially nonoptimal point is rejected.

Numerical comparisons are fixed before empirical execution: absolute tolerance `1e-10`, relative tolerance `1e-8` for reconstructed coordinates, densities, scores and audit numbers. Copied baseline columns, dates, ordered row membership and categorical/count identities are exact. The frozen independent copula certificate retains its existing numerical domain and gap tolerances. No after-result tolerance adjustment or retry is authorized here.

## Full forecast and chronology replay

API: `verify(archive, baseline_applications, baseline_panel, produced, calendar=None, minimum_train=252)`. The production floor is 252; generated tests use 12. Lower-level public checks are `verify_parameters`, `independent_shape_coordinates`, and `independent_inverse_shape`.

The verifier independently enumerates the original archive's monthly query schedule. Every donor must have an earlier origin, both target end and available date at or before the original previous-session cutoff, an actually issued forecast, and two finite observed outcomes. It reconstructs the complete eligible month sequence and ordered donor lists; only initial unsupported months are warmup. A later support failure is fatal. With the original full calendar supplied, dates, previous-session cutoffs, next-session targets and global modulo-five offsets are checked by the frozen independent archive oracle.

For each eligible month it reconstructs affine location/scale/count from the mature archive, verifies the two shape certificates, and checks the Gaussian and t8 dependence certificates on the same independently transformed pairs. There is no new producer feature, calibration, density or fitting import in the independent module. Reused code is limited to the previously independent archive, t8 log-tail, normal-coordinate and copula-certificate oracles.

Every application is retained, including wholly unscored months and the final missing-outcome query. Every baseline application and scored panel column is preserved exactly. The scored row set must equal the inherited eligible outcome set; a cell cannot choose its own dates. The independent normalized marginal densities, PITs, normal coordinates, both joint losses and all six signed comparisons are reconstructed. A negative difference means the new density has lower negative log loss. The interaction is `(shape_t8-shape_gaussian)-(affine_t8-affine_gaussian)`.

The proof certifies the shape parameters, normalized densities, complete saved output replay, and unchanged baseline values. It relies on external byte authentication and prior verification of the frozen baseline; it does not reauthenticate raw source files, refit original market models, reconstruct baseline numerical certificates, verify statistical scoring or bootstrap intervals, or validate economic payoffs/execution.

## Prewritten evidence and changes before freeze

- `CORE_RED.log`: the initial generated core contracts failed because the producer module did not yet exist.
- `CORE_EXTENDED_FIRST_IMPLEMENTATION.log`: additional generated contracts exposed an omitted wholly unscored month, absent fixed parameter-box validation, and missing exact identity shortcut. Root fixed those before new historical execution. The future-perturbation fixture was narrowed to one frozen query month: changing labels while keeping later months' old affine controls would correctly violate the same-archive requirement, and was a fixture inconsistency.
- `VERIFIER_RED.log`: all eleven verifier methods were written before the new verifier module, and the missing-module failure was observed.
- `VERIFIER_FIRST_IMPLEMENTATION.log`: all scientific comparisons passed; the deliberately corrupted dependence audit raised the legacy checker's `AssertionError` instead of this API's `ValueError`. The new boundary now normalizes that failure without weakening the rejection.
- `VERIFIER_GREEN.log`: 24 generated methods passed (13 core, 11 independent verifier) in 1.274 seconds. Scoped Ruff and formatting checks passed. Tests include quadrature normalization, CDF/Jacobian finite differences, independent SLSQP agreement, derivative/Hessian checks, generated skew correction, extreme inverse roundtrips, future-label invariance, unscored months, source ceilings, saved Parquet/JSON roundtrips, exact baseline retention and certificate/density/contrast tampering.

Stable owned source/test hashes:

| File | SHA-256 |
| --- | --- |
| `src/verify_copula_shape.py` | `fc56df7759645ddc95e4da3f549aaac65163a4fd079508b83d1918c514706e06` |
| `tests/test_copula_shape.py` | `c1b050e1ec27799906cc3f474580cb38740e7d7a7533f2e1ef1819b01133fe49` |
| `tests/test_verify_copula_shape.py` | `3205ef3db6d5541641dccf41b4ac201bd55e9021eaefe4e678e4655585d80ac6` |

## Interpretation limits for the parallel risk study

Normalization and finite quantiles do not imply finite return moments. In particular, some heavy-tail SAS settings can lack positive moments of log returns, and an exponentiated t8 log-return model already lacks ordinary positive exponential moments. This does not prevent expectations of genuinely bounded vertical-spread or iron-condor terminal payoffs. Those payoffs are bounded by the spread widths even for extreme underlying returns; their numerical construction and live-trading costs require separate verification. Neither a PIT correction nor an improved dependence score alone establishes safer option selling, adequate out-of-sample calibration, a causal mechanism or positive net returns.
