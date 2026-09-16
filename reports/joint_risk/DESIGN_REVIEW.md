# Joint risk: independent pre-fit design challenge

Reviewed 2026-09-07 before any new empirical measurement, residual construction, fit or score. This review uses the frozen [prospective note](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/relative_risk/NEXT_JOINT_RISK_DESIGN.md), earlier implementation contracts, primary methodological sources, and algebra. The choices below are prospective; the new protocol must pin them before execution. No earlier source, protocol, result or code was changed.

The proposed response is distinct from relative marginal GK risk. It can test improvement in **joint residual second moments with fixed modeled marginals**. It cannot, by itself, identify true correlation predictability. The latter limitation remains even when every numerical check passes.

## Response, score and attribution

Use the exact paired next-SPX-session raw open-to-close log-return vector `r`, with all predictors ending at the preceding observed SPX session. Shared predictable means `m` define `e=r−m`. The estimand is `S=E[e e'|information]`, which equals true conditional covariance plus `(true_mean−m)(true_mean−m)'`. Training residuals are from the current monthly mean fit, not chronological out-of-sample residuals; both dependence arms must reuse them unchanged.

For positive-definite `H`, the **unhalved** score is `log det(H)+e'H^−1e`. Its expectation is `log det(H)+tr(H^−1S)`, uniquely minimized at positive-definite `S`. This expectation argument does not require Gaussian returns. An observed outer product may be rank one or zero: never take its determinant, add target jitter, or discard a signed product. Patton and Sheppard discuss return outer products as covariance proxies and multivariate QLIKE evaluation; noisy proxies affect precision. [Primary chapter, sections 1.1 and 3.6](https://public.econ.duke.edu/~ap172/Patton_Sheppard_29oct07.pdf).

With `H=D R(rho) D`, `D=diag(sqrt(h_Q),sqrt(h_S))`, write `u=e_Q/sqrt(h_Q)`, `v=e_S/sqrt(h_S)`. The correlation-dependent score is

`f(rho)=log(1−rho²)+(u²+v²−2rho*u*v)/(1−rho²)`.

Changing `rho` changes the weights on squared residuals as well as their product. Thus a score gain may compensate for misspecified diagonals. A precise synthetic counterexample should be tested: true means and true correlation are zero in both states; true marginal variances are `0.1` with probability `1/4`, and `1` with probability `3/4`; the model fixes both diagonals to `1`. A unit-standardized signal takes values `sqrt(3)` and `−1/sqrt(3)`, respectively. The best constant correlation is zero because `A=E[u²+v²]=1.55`, `B=E[uv]=0`. Nevertheless, for `rho=.995*tanh(b*z)` with penalty `.01b²`, the expected objective's second derivative at zero is `2[−.35*.995²+.01]<0`. A nonzero slope improves the joint score despite constant zero true correlation. This is an algebraic counterexample, not a new empirical result.

The marginal models should therefore already contain corr22 and its predeclared training-centered square. This is a useful control, not a proof of correct marginal specification. Engle's work establishes the variance/correlation decomposition and staged estimation precedent; this single-slope experiment is not a DCC-GARCH replication. [Author's paper](https://pages.stern.nyu.edu/~rengle/dccfinal.pdf).

## Shared marginals and the second benchmark

Root's proposed contract uses the same 34 complete source-feature columns, adds `(corr22−training_mean_corr22)²` at each fit, and standardizes the resulting nonconstant columns using training population moments. Both assets' means use fixed normalized ridge `.01`, with an unpenalized intercept. Each positive residual-second-moment model uses the unchanged zero-compatible exponential quasi-likelihood helper with `.01` standardized-slope penalty. No separate target standardization changes the mean penalty. All fitting, centering and scaling use mature common training rows only. All-zero residual second moments, nonfinite values, insufficient rows and zero input scales abort rather than trigger an alternative estimator.

The required attribution control estimates a constant correlation with exactly the candidate's means and diagonals. The practical second benchmark is the **uncentered training residual outer-product mean**, `M=n^−1 sum(e_i e_i')`, reused through the monthly application period. It uses the same fitted mean forecasts when scored. Do not substitute an independently centered sample covariance or an `n−1` divisor. Its positive finite diagonals and normalized correlation eigenvalue must pass the fixed gate, without jitter. It is a useful constant joint-risk benchmark; it does not replace the attribution control. Both candidate contrasts remain registered.

## Scalar objectives and derivatives

Fix `c=.995`, `a0,b in [−4,4]`, and candidate penalty `.01b²`. First fit `rho0=c*tanh(a0)` globally on its allowed range; then hold `a0` fixed and fit only `b` in `rho_i=c*tanh(a0+b*z_i)`. The signal's training mean and population scale are shared with its saved feature transform. At `b=0` the full candidate forecast equals the constant-correlation control exactly.

For one row, put `A=u²+v²`, `B=uv`, `d=1−rho²`, and `P=rho³−B*rho²+(A−1)*rho−B`. Then

`df/drho=2P/d²`,

`d²f/drho²=2(3rho²−2B*rho+A−1)/d²+8rho*P/d³`.

With `t=tanh(a0+b*z)`, `r1=c(1−t²)`, `r2=−2ct(1−t²)`, the candidate gradient is `mean[(df/drho)*r1*z]+.02b`, and its Hessian is `mean[z²*((d²f/drho²)*r1²+(df/drho)*r2)]+.02`.

For constant correlation, replace `A,B` by their sample means. All stationary points are real roots of the cubic `P=0`; the compact-domain minimum is among these roots and the two allowed rho endpoints. Evaluate the complete set, deterministically resolving numerical ties, and save all candidates. Independent verification should isolate roots between the quadratic derivative's turning points instead of reusing the producer's polynomial-root implementation. Projected first-order checks supplement complete candidate enumeration.

## Candidate numerical global-value certificate

A dense grid and one local solve do not certify globality. Proposed bounded procedure, coordinated with the density implementer:

1. Start the local bounded solver at `b=0`, and evaluate both endpoints. Use fixed L-BFGS-B options: 1,000 iterations, 50 line-search steps, `ftol=1e−14`, `gtol=1e−9`. Every finite evaluation gives an objective upper bound; a success flag alone is insufficient.
2. Maintain a best-first interval queue on `[−4,4]`. Ties use the smaller left endpoint. For midpoint `m` and radius `delta`, a valid curvature bound `L >= sup|F''|` gives lower bound `F(m)−abs(F'(m))*delta−L*delta²/2`. Bisect the interval with the lowest lower bound. A midpoint improving the incumbent triggers the same deterministic local refinement; retain an audit of every such attempt. Solver failure never justifies pruning an interval.
3. For each row's eta interval, use its maximum absolute rho `r`, minimum denominator `1−r²`, and bounds `|P| <= r³+|B|r²+|A−1|r+|B|`, `|P'| <= 3r²+2|B|r+|A−1|`. These bound the two rho derivatives above. Bound `r1` at the eta value nearest zero. Bound `|r2|` from the tanh endpoints and `±1/sqrt(3)` when contained. Average `z²*(bound_abs_frr*max_r1²+bound_abs_fr*max_abs_r2)` and add `.02` to obtain `L`.
4. Require an absolute penalized-objective upper/lower gap at most `1e−8` and projected KKT at most `1e−7`; use at most **32,768 interval splits**. Unresolved bounds, nonfinite arithmetic, no admissible incumbent, or budget exhaustion fail the fit. No uncertified grid fallback, changed bound, extra window or post-result restart is permitted.

Outward rounding or an explicitly conservative floating-arithmetic allowance must be implemented and tested before freezing this certificate. Call it a **numerical value certificate to the declared tolerance**, not an exact-arithmetic proof or a guarantee of unique parameters. Store the final coefficient, objective, lower bound, gap, KKT, all local starts/statuses, split/evaluation counts and numerical settings. Independent verification must reconstruct the derivatives and certificate from saved training geometry; equal coefficients alone are insufficient. Synthetic tests must establish tractability before any new market calculation.

## Numerical admissibility and score implementation

For dynamic and constant-correlation arms, `.995` bounds the unit-diagonal correlation matrix's smallest eigenvalue at `.005` and its condition number at `399`. This is normalized conditioning, not a bound on the full matrix's condition number under arbitrary return units. The constant-matrix benchmark requires normalized smallest eigenvalue at least `1e−6` (`abs(rho)<=1−1e−6`). Each diagonal must be finite and strictly positive; do not add a unit-dependent variance floor.

Compute scores with diagonal whitening and `log(h_Q)+log(h_S)+log1p(−rho²)`, avoiding `log(det(H))` formed through products that may overflow. The quadratic form can use the Cholesky expression `v²+(u−rho*v)²/(1−rho²)`. Independently compare with direct Cholesky solves on admissible synthetic matrices, including asset exchange and independent positive rescaling of each return series. Any nonfinite score/gradient or unrepresentable matrix entry fails; prediction clipping, eigenvalue repair and candidate-specific omission are prohibited.

## Registration and verification requirements

Root's proposed gates are coherent: two contrasts, absolute **unhalved** score improvement at least `.005` in both phases against both controls, negative gaps in both fixed evaluation slices, wave Holm at `.05/(10*11)`, and cumulative Holm over `112` enumerated comparisons. Percentage gains in this potentially negative score are inappropriate. Training penalties affect estimates, never the reported predictive loss. The effect threshold is a fixed practical criterion, not a trading-profit calibration.

Prewritten tests should cover the derivatives, all cubic roots/endpoints, a multimodal scalar objective, an unresolved certificate, endpoint KKT signs, exact nesting, zero/signed/rank-one targets, constant-zero residual failure, unchanged means/diagonals, training-only transforms, source-calendar/maturity fences, deterministic tie handling, constant-matrix construction, unit/asset-order invariance, and the misspecification counterexample. Publication and independent-verification failures must preserve both hypotheses at `p=1`, retain diagnostics, and expose the failure distinctly from an evaluated null result.

The design is ready for implementation under these explicit controls. Numeric feasibility and optimizer certification remain untested on market data. A passing result would still be exploratory joint-risk evidence from reused archival QQQ ETF/SPX index observations, without synchronized intraday covariance, contemporaneous-vintage, execution or profit claims.
