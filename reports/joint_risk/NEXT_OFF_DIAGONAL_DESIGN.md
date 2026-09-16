# Prospective direct cross-moment score

**Unregistered and blinded to wave 10 outcomes.** This note proposes no added wave 10 arm, source, measurement, fit, or score. It uses the declared forecast schema and algebra only. It is a possible new scoring question about the same issued predictions, not a claim that new information has been found.

The bounded next question is whether the issued dynamic-correlation forecasts predict the **signed residual cross moment** more accurately than both issued controls. The matrix score can improve when a dependence parameter compensates for incorrect marginal scales: changing correlation changes the coefficients on both squared residuals as well as their product. Direct product squared error isolates a different forecast functional and removes that particular score-weighting mechanism. It does not identify pure correlation dynamics or correct mean misspecification.

## Exact response and proper score

At each retained origin, take the two actual next-session raw log close/open returns and the identical means already issued by all three wave 10 models. Define

`e_Q = r_Q − issued_mu_Q`, `e_S = r_S − issued_mu_S`,

`Y = e_Q * e_S`,

`q_model = issued_rho_model * sqrt(issued_h_Q_model) * sqrt(issued_h_S_model)`.

Score `L(Y,q) = (Y−q)²`. Both the realized product and its forecast may be negative or exactly zero; no logarithm, positive floor, sign filter, or target determinant is appropriate. Any nonfinite required value is an explicit failure rather than a selective omission.

For the information available when the common mean and matrix were issued, let `m_Y = E[Y | information]`. Direct expansion gives

`E[(Y−q)² | information] = Var(Y | information) + (q−m_Y)²`.

Thus squared loss elicits the conditional residual cross moment when the relevant second moment of `Y` exists. The paired loss difference can also be evaluated as

`(q_candidate−q_control) * (q_candidate+q_control−2Y)`.

There is no matrix-inverse diagonal weighting in that expression. Diagonal forecasts still enter each issued `q`; the conditional candidate and constant-correlation control share those diagonal forecasts exactly. This is a direct check of their different off-diagonal predictions, with the constant-matrix forecast providing the existing training historical-product-mean benchmark.

The estimand remains

`E[Y | information] = Cov(r_Q,r_S | information) + (true_mu_Q−issued_mu_Q)*(true_mu_S−issued_mu_S)`.

A gain can therefore reflect remaining common-mean error or correction of a badly calibrated benchmark. Even a lower product MSE does not prove time variation in true correlation, independent information in corr22, high-frequency covariance accuracy, or a profitable hedge. The same existing information is being assessed for a different functional.

## Smallest separate experiment

Use only the three frozen wave 10 issued forecast rows, with their original target, mean, diagonal, correlation, fit, availability, and phase metadata. Do not refit any mean, marginal model, constant matrix, dependence intercept, or slope. Keep every source and forecast byte unchanged. Require exact origin/label/mean agreement and the same three-model finite sample before computing either contrast. The full-source GK gate and original forecast verification must already have passed; numerical or publication failure cannot be bypassed by selecting a convenient subset of forecasts.

Register **two new hypotheses**: dynamic-correlation product MSE versus constant-correlation product MSE, and versus constant-matrix product MSE. These are additional tests of the issued predictions and must enter the cumulative ledger even though there is no new fitting. If they are the next wave after the presently registered 112 comparisons, the total becomes **114**. Under the existing telescoping allocation, a wave 11 study receives `.05/(11*12)` for its two-comparison Holm family, together with cumulative Holm at `.05`. Any intervening study changes those bookkeeping numbers prospectively. Both comparisons must pass the declared gates for one candidate claim.

Preserve the existing development/evaluation boundaries, label-maturity fence, and fixed evaluation subperiods. The original fitted means and residual staging remain untouched: fitting used current-fit training residuals on mature rows, while the new realized product uses the common **issued** means at each scored origin. Do not replace it with retrospective full-sample residuals or pretend the training residuals were historical out-of-sample errors. No new latent-memory or residual calibration fit is part of this proposal.

The existing 21/63/126-observation block and HAC126 uncertainty design is a reasonable fixed starting contract, with missing-calendar gaps disclosed. Reusing 99,999 bootstrap draws would resolve a minimum attainable p below one tenth of the strictest two-comparison wave 11 threshold. This resolution statement is arithmetic, not evidence of statistical power. No significance, tail trimming, or bandwidth choice should be selected after looking at the product losses.

## Units, effect threshold, and power remain registration decisions

With decimal raw log returns, `Y` and `q` have squared-return units, and product MSE has fourth-power return units. Rescaling QQQ returns by `a` and SPX returns by `b` multiplies every product squared loss and absolute loss difference by `(a*b)²`. Fix these units explicitly. The wave 10 `.005` matrix-score threshold cannot be reused for product MSE. Do not divide losses by realized volatility or model diagonal forecasts to make the numbers convenient; that would create a differently weighted target criterion.

A new, meaningful absolute minimum product-MSE decrease must be fixed **before opening the new losses**, with a stated rationale in these units. This blinded note does not assign that numerical threshold from unseen outcomes. The effect threshold, associated power criterion, finite arithmetic rules, and exact failure publication behavior are the remaining decisions needed before registration. A percentage of a near-zero control MSE is not a substitute for that work.

One daily product is a noisy observation. Large joint moves can dominate squared losses, and the second moment of the product involves cross fourth moments of residual returns. The paired-difference algebra cancels the shared `Y²` term, which helps numerical comparison, but does not make dependence uncertainty or tail influence disappear. Nominal row count and fine bootstrap resolution do not establish power. Report the fixed absolute uncertainty interval and detectable-effect calculation; do not reinterpret an imprecise failure to reject as equivalence.

## Interpretation independent of the current result

If the matrix-score study passes, this separate test could distinguish a gain that also improves signed cross-moment prediction from one supported only by the joint matrix criterion. If the matrix-score study fails, a separately registered product-score success would be evidence for this narrower functional while the matrix-score result remains failed. If both fail, neither becomes evidence of equivalence without an adequate predeclared equivalence analysis. A product-score failure alone cannot prove that any matrix-score gain was caused by diagonal misspecification; sampling uncertainty or other forecast tradeoffs remain possible.

If the current wave is unevaluable or its issued forecasts fail verification, there is no eligible frozen forecast set for this reuse-only proposal. A new modeling or source repair would require another contract. In every case, history reuse and the already stated archival ETF/index limitations remain. The prospective product score can sharpen the interpretation of the same forecasts; it cannot retroactively change the wave 10 success criterion.
