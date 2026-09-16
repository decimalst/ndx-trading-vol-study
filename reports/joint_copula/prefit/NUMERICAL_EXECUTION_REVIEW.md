# Numerical prefit review: fixed joint-copula experiment

No blocking numerical or scientific inconsistency was found in the reviewed code and exact protocol. This review used source code, prewritten test contents, and small invented arrays only. It did not decode historical sources, inspect support or fit historical data, rerun the existing suites, or edit any implementation. The separate execution reviewer owns registration, persistence and publication checks.

## Formulas and controls

The producer fits each asset's mean and positive residual second moment once and copies the identical resulting locations and h values into all three arms. Its standardized residual is (y-mu)/sqrt(.75h). A Student-t with eight degrees of freedom and that scale has variance h. The copula constructions preserve those marginal distributions, so the Gaussian control cannot gain an advantage from different mean, variance, marginal shape or training membership. The fixed independence arm uses a zero copula log density. A t8 copula with rho=0 remains dependent and is correctly kept distinct from independence.

The Gaussian formula is the bivariate Gaussian log density divided by its standard-normal marginals after the t8 CDF-to-normal transform. The t8 formula is the bivariate t8 log density divided by the two univariate t8 densities. These are proper density likelihoods, not a covariance loss. Fitting their negative mean copula log density is equivalent to fitting negative joint log density with the shared marginals held fixed. The protocol correctly reports full joint negative log density while calculating each paired difference directly from the two copula log densities, avoiding cancellation of large shared marginal terms.

The t8 normalization simplifies to 1/(2*pi*sqrt(1-rho^2)); the implemented five-times-log-quadratic exponent and fixed univariate normalization agree. Scaling each residual pair by max(1,abs(z1),abs(z2)) avoids squaring huge unscaled values. The marginal log density uses logaddexp. The normal transform computes the smaller log tail: for abs(z)>1, the incomplete-beta/hypergeometric expression with parameters (4,1/2;5) is the t8 survival probability. ndtri_exp then avoids saturated CDF endpoints. No probability clipping or adjustable tail threshold is introduced.

## Scalar optimization and numerical certificate

The Gaussian objective derivative has numerator rho^3-B*rho^2+(A-1)*rho-B and positive denominator (1-rho^2)^2. The implemented candidates include both domain endpoints, zero and every admissible real polynomial root, with root residual checks. The projected-gradient signs at the lower and upper boundaries are correct. Objective ties within 1e-12 choose the lower rho among evaluated candidates. The all-zero generated fixture explicitly checks the symmetric boundary tie.

For the t8 objective, write W(rho)=c*(1-rho^2)+a-2*rho*b, with c>=0, and D=1-rho^2. Its second derivative is

`5*mean(-2*c/W - (Wprime/W)^2) + 9*(1+rho^2)/D^2`.

W is positive and concave on the fixed domain, so its minimum on an interval is at an endpoint. The largest absolute Wprime is also attained at an endpoint. Using those extrema bounds the negative terms from below; using the point nearest zero bounds the final positive term from below. The stored curvature is therefore a valid real-arithmetic lower Hessian bound. The minimum of the corresponding midpoint Taylor quadratic is taken over both endpoints and its interior vertex when convex. This supports the branch-and-bound lower value. All terminal leaves are retained, cover the full closed interval, and support independent reconstruction.

The local optimizer supplies a candidate only; the whole-domain lower-bound gap and final projected-gradient threshold are still required. Midpoint subdivision, bracketed derivative-root evaluation and the declared near-stationary refinement do not remove unresolved intervals. Local optimizer failure, nonfinite arithmetic, exhausted subdivision budget, excessive gap and excessive projected gradient raise errors. No silent fallback, row deletion, correlation-domain expansion or successful partial fit is used.

## Bounded generated checks performed in this review

Four invented 24-pair arrays (t8 draws, positive and negative nearly collinear pairs, and near-zero pairs), six intervals and 29 points per interval supplied 696 checks. I reconstructed the Hessian directly from unscaled residuals and confirmed each stored lower curvature and objective bound. The largest lower-bound-minus-objective value was -1.4752489786717948e-7. The near-zero and both nearly collinear arrays additionally produced six boundary fits across the two families. Every fit had projected gradient zero; t8 certificate gaps were at most 4.074080628413412e-9, and Gaussian gaps at most 4.583000645652646e-13. The fixed protocol hash check and exact six-field pipeline projection also passed. These are additional adversarial examples, not a replacement for the existing prewritten tests or independent forecast reconstruction.

## Execution and interpretation limits

The protocol has the agreed 2016-01-04 through 2025-10-17 origin bounds, numerical source ceiling 2025-10-20, development ending 2019-12-31, evaluation beginning 2020-01-01, previous-full-session fit cutoff and next-full-session intraday target. Its two comparisons retain 149 prior hypotheses and specify 151 cumulative hypotheses upon registration. Both Gaussian and independence comparisons must satisfy the effect, support, stability and multiplicity rules. The protocol expressly distinguishes historical numerical exclusion for this run from the previously accessed period beginning 2025-11-03.

The 256-epsilon inflation and floating-point root calculations are a numerical certificate rather than directed-rounding or exact-arithmetic proof. Near-boundary calculations, cancellation and roots remain subject to the declared numerical tolerances and independent reconstruction. Underflow of negligible terms in scaled tail identities is an approximation at machine precision; it does not constitute an exact-real guarantee for every possible finite input. No historical robustness claim follows from these small generated checks.

Shared marginal misspecification can still create apparent dependence improvements. A successful result would establish only an exploratory joint-density lead under the declared current snapshots and measurement clock. It would not isolate pure tail dependence, improve marginal volatility by construction, demonstrate index mean alpha or trading profit, certify synchronized opening/closing auctions, or provide untouched future confirmation. The reused synthetic inference calibration is a diagnostic envelope check, not a guarantee of nominal market coverage or extreme multiplicity-adjusted tail calibration.

## Reviewed byte bindings

These hashes identify the bytes inspected, not a registration or empirical execution.

- src/joint_copula_density.py: `da502a656cf4ae480ea6377a546f12645442680bb922ed5c45b2c2bde3e48a0e`
- joint_copula.yaml: `cf43700c6c37ba860e9ef8ae4adb4c70158030efeedd3bca9b073f0691cd5d50`
- src/joint_copula_protocol.py: `1ccca48122f36f74a684909126ae71e4586c2fbb4ab732688865e0da1c72f5c2`
- src/joint_copula_models.py: `4ab1eb02fe7703cf8639f062e9fb976f7ee22ddd4cc13451566c0e4660558c86`
- tests/test_joint_copula_density.py: `b48e7908a91b4df32e0b3112ac14522c7cc3d1ed66971422e7c682404086a777`
- src/verify_joint_copula_forecasts.py: `e74326da39c31de64140cedb0f38530a297680bc313e1777942b777cda25060c`
