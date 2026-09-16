# Prospective joint intraday return-risk question

**Unregistered and blinded to wave 9 outcomes, 2026-09-07.** This review inspected specifications, source documentation and code, plus two primary methodological references. It did not read wave 9 empirical results or numerical data, construct features/targets, count a new sample, fit a model, or change an earlier file.

**A two-asset conditional second-moment matrix is a distinct response from relative marginal GK risk.** The available raw QQQ/SPX opens and closes can support a bounded joint-return experiment. The useful question is whether one lagged correlation signal improves the off-diagonal forecast while both compared models share exactly the same fitted marginal means and second moments. This is an established modeling idea applied to a new repository response, not new data, a measured high-frequency covariance, or evidence of hedging profit.

## Novelty and source boundary

The current [relative-risk producer](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/relative_risk_features.py:100) forecasts the difference of two next-session log GK measurements. Its response has no signed return product. [Orthogonal round two](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/orthogonal_round2.py:91) uses QQQ–TLT correlation to predict scalar QQQ risk. The [international study](/Users/byrons/code/trading-vol/ndx-vol-experiment/international_volatility.yaml:32) uses foreign-market states to predict scalar SPX RV. The [univariate tail-shape design](/Users/byrons/code/trading-vol/ndx-vol-experiment/tail_shape.yaml:70) holds one asset's modeled moments fixed while changing distribution shape. None implements this two-asset matrix response or matrix score. The separately specified [SPX correlation-premium target](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/SPX_DISPERSION_DATA_CONTRACT.md:23) needs an exact constituent tracking basket; it is not QQQ–SPX covariance. A search for matrix-loss, covariance-forecast, and multivariate-correlation implementations found no such fitted family in the inspected source/specifications; this is not certification against unrecorded experiments.

The [existing source/date audit](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/relative_risk/SOURCE_FEASIBILITY.md) establishes matching QQQ/SPX observed dates and conceptual history support. It does not turn QQQ ETF prices into Nasdaq-100 index levels or SPX index levels into traded ETF fills. Use the same raw OHLC snapshots, source fence 2025-10-20, sealed boundary 2025-11-03, exact-date alignment, and previous-source-date rules. Do not introduce a new download or historical vintage. Vendor daily opens/closes do not certify simultaneous index/ETF auction sampling. Intraday returns avoid directly crossing a distribution-date close/open gap, but tracking, adjustment consistency, and archival-revision limitations remain.

## Exact target and proper score

At origin `t`, use market information only through the prior observed SPX session. Let `s` be the next observed SPX session, used only for labels and maturity. Define the two-vector

`r[t] = (log(C_QQQ[s]/O_QQQ[s]), log(C_SPX[s]/O_SPX[s]))`.

Use shared predictable mean forecasts `m[t]` and residuals `e[t] = r[t] − m[t]`. The object being forecast is **`S[t] = E[e[t] e[t]' | information at t]`**. Its observed proxy is the rank-one outer product `e e'`, whose off-diagonal is signed and whose diagonal may be zero. Both returns become label-available at their same observed session close. No future session date or opening price enters a predictor.

Drift matters. Writing the true conditional return mean as `mu`,

`S = Cov(r | information) + (mu − m)(mu − m)'`.

Therefore a misspecified shared mean leaves a residual-second-moment target rather than exact conditional covariance. If zero means are deliberately imposed, the target becomes the raw second moment `E[r r']`; it must not silently be called covariance. A single daily signed product is a noisy observation for this conditional estimand, not realized high-frequency covariation.

For a positive-definite forecast `H`, use the **unhalved** matrix score

`L(H,e) = log det(H) + e' H^(-1) e`.

Conditional expectation gives `log det(H) + tr(H^(-1) S)`. For positive-definite `S`, its unique unconstrained minimum is at `H=S`; no Gaussian return distribution is needed for that expected-score statement. The observed outer product need not be invertible: do **not** compute its log determinant or add target jitter. A zero residual vector is valid. Patton and Sheppard discuss return outer products as covariance proxies and multivariate QLIKE evaluation, including the precision limits of noisy proxies. [Primary chapter, sections 1.1 and 3.6](https://public.econ.duke.edu/~ap172/Patton_Sheppard_29oct07.pdf).

## One correlation increment with shared marginals

Fit a common mean function and positive residual-second-moment function for each asset on the same completed training rows. The shared predictors should include both assets' intraday/total-risk and leverage histories, both intraday signed-return histories, the already available IV controls, weekday, **corr22 itself**, and one predeclared centered corr22 square. Giving that signal to the marginal models is a guard against crediting simple signal-dependent mean/scale changes to the correlation addition. The estimator, features, penalty and any admissible bounds must be fixed before new numerical inspection.

At each monthly refit, freeze those two mean functions and two positive diagonal forecasts `h_Q,h_S` for all compared correlation models. Do not reuse GK point forecasts as diagonal return variances without calibration to the new squared-return-residual target. Stage fitting can follow the existing mean-ridge/positive-second-moment workflow, but the protocol must explicitly say whether its training residuals are current-fit residuals or chronological out-of-fold residuals. The bounded initial recommendation is the existing current-fit staging policy, disclosed as such; these are not historical out-of-sample errors. Outer evaluation still uses only forecasts issued before the scored return is available.

Write `D=diag(sqrt(h_Q),sqrt(h_S))` and `H=D R(rho) D`, where `R` has unit diagonal and off-diagonal `rho`. This variance/correlation separation has an established two-stage precedent in Engle's dynamic-correlation work. The proposed one-feature regression below does **not** reproduce a DCC-GARCH recursion or inherit its empirical results. [Engle, 2002](https://doi.org/10.1198/073500102288618487).

Use one fixed slope adjustment:

- **Conditional constant-correlation control:** estimate a scalar `a0` using the common training matrix score; `rho0 = rho_max·tanh(a0)`.
- **Candidate:** hold `a0` fixed and estimate only `b`; `rho[t] = rho_max·tanh(a0 + b·z[t])`, where `z` is the training-standardized strict prior-22-session intraday Pearson correlation. At `b=0`, candidate forecasts exactly equal the control. Use a fixed slope penalty and no alternative correlation windows, signs, recursions or memory search.

`rho_max` must be fixed strictly below one before fitting. This bounds the standardized correlation matrix away from singularity. Both diagonal forecasts must also remain positive and finite under a fixed admissibility/conditioning rule; no candidate-specific variance floor, eigenvalue repair, or prediction clipping is allowed. Correlation windows need the already tested exact-constant guard, and scaling uses training rows only.

The constant-correlation control is the required attribution comparison. A same-row expanding **constant residual-second-moment matrix**, scored using the same shared mean forecasts, is the natural additional practical benchmark. If included, both candidate contrasts belong in the continuing family before scores. That constant matrix must pass an explicit positive-definiteness check rather than receive unregistered jitter. Neither a log variance ratio nor subtraction of linear marginal forecasts supplies the signed covariance term, so this is not the ridge subtraction identity from wave 9.

## Interpretation, optimizer risk, and the next contract

Equal marginal predictions do not prove that an improvement identifies true correlation dynamics. In standardized residual coordinates, the correlation-dependent score is

`log(1−rho²) + (z_Q² + z_S² − 2·rho·z_Q·z_S)/(1−rho²)`.

The weights on the two squared residuals change with `rho` too. If predicted marginal second moments are wrong, an apparent dynamic-correlation gain can partly compensate for those errors. Shared strong mean/scale controls reduce that concern; they do not remove it. Report a passing result as improved **joint residual-second-moment prediction with fixed modeled marginals**, with the correlation interpretation conditional on their adequacy.

The transformed scalar correlation objective is not guaranteed globally convex. Freeze parameter bounds, deterministic starts, gradient/optimality checks, stopping rules and independent verification before any empirical fit. For this small design, bounded one-dimensional global checks for `a0` and `b` are practical candidates for the contract; do not resolve optimizer disagreement by accepting whichever fit produces the better score. Fit failures remain registered unevaluable entries. Near-collinear returns and small predicted marginal scales can make the matrix score highly sensitive to individual days; daily products also have much more measurement noise than a synchronized intraday covariance estimate.

The immediate task is a **new measurement, estimation and verification contract**, followed by synthetic tests—not a forecast run based on this note. Those tests should establish positive definiteness across allowed parameters; unchanged means/diagonals across correlation arms; exact nesting at `b=0`; score equivalence to direct Cholesky calculation; asset-order and return-unit invariance; valid zero/signed products; finite rank-one-target scores; source/timing/maturity fences; and recovery under constant versus changing correlation. Include a case with constant true correlation but misspecified dynamic means/scales to make the attribution limitation visible. Freeze current-fit residual staging, denominator admissibility, sample minimum, exact score normalization, absolute useful-effect criterion, dependence-aware uncertainty, stability periods and cumulative multiplicity before reading new outcomes. Percentage improvement in a possibly negative raw matrix score is inappropriate.

The existing paired date support makes the route plausible, but it does not certify numerical marginal fits, covariance-matrix conditioning, calibration or power. Any decision should preserve the earlier wave's result and gates unchanged. This note makes no actual trading, dynamic hedging, allocation or profit claim.
