# Shape calibration and 0DTE spread safety: declared experiment

Specified 2026-09-13 before fitting the new models or inspecting their historical
results. The user selected leveraged derivatives, then narrowed the application
to selling spreads and explicitly chose same-day/0DTE expiry. The separate
futures-sizing draft was stopped before empirical execution. This study uses
the already inspected historical window; it is exploratory, not a new untouched
confirmation sample.

## Why this experiment

Wave28's location/scale correction left most of the t8 joint-density advantage,
but did not consistently improve individual forecasts. The new question is
whether allowing distributional skewness and tail shape improves those
forecasts, and whether the resulting distributions help identify sessions with
larger hypothetical spread expiration liabilities.

There are three marginal systems (original, affine, shape), each paired with
Gaussian and t8 dependence. The four old cells remain exactly as saved. The two
new cells use the same monthly fits and mature historical issued errors, with
the same fixed dependence families and degrees of freedom. No current-model
in-sample residuals replace those issued errors.

## New marginal correction

After the inherited normal-score location/scale correction, transform
`x=(Phi^-1(T8(z))-a)/b` to `v=sinh(delta*asinh(x)-epsilon)`. The probability is
`Phi(v)`. The density includes the derivative of every transformation. The
normal-score sinh-arcsinh family is motivated by
[Jones and Pewsey (2009)](https://doi.org/10.1093/biomet/asp053); the fixed
two-stage estimation and numerical bounds here are our experiment's choices,
not a claim to reproduce unrestricted four-parameter estimation in that paper.

With a,b fixed to the mature archive's estimates, the shape likelihood is
convex in epsilon and delta. Epsilon is bounded to[-.75,.75], delta to[.75,2].
We check a convex objective-gap certificate rather than relying only on a
solver's success flag. The independent verifier reconstructs coefficients,
gradients, Hessians, certificates, normalized densities and all new contrasts.
Its exact numerical scope is in [the verifier contract](INDEPENDENT_VERIFIER_CONTRACT.md).

The six registered comparisons are shape-minus-affine marginal loss for each
asset, shape-minus-affine joint loss for each copula, the new t8-minus-Gaussian
gap, and its change from the affine gap. Dependence-aware inference preserves
the full original calendar and its masks. Holm6 must pass.05/(29*30), Holm164
must pass.05, and the paired mean must have the same strict sign in both phases.
An observed improvement or a failed gate does not identify a unique mechanism.

## What the spread application can answer

Single-asset breach probabilities and expected intrinsic liabilities depend on
the marginals. Changing the copula alone cannot change them. The copula can
change whether multiple spreads lose together and the portfolio's tail losses.
We therefore examine two equally allocated underlyings and preserve identical
marginal predictions within each dependence comparison.

The fixed main case has short strikes2% away from the opening price, with1%-wide
wings; distances1% and3% are reported as sensitivities. Put credit spreads,
call credit spreads and iron condors are separate experiments in allocation,
not simultaneously held positions. Each policy reserves2% of equity for the
entire two-asset gross expiration liability before credit. It sells only when
the forecast probability of either short-strike breach is at most10%. An
always-sell control uses the same gross-risk budget.

This follows the limited-width expiration payoff of vertical credit spreads;
collecting a premium does not by itself establish favorable expected returns.
[Cboe's 0DTE resources](https://www.cboe.com/tradable-products/0dte) describe the
strike-width risk and the role of the collected premium. Our return-based
calculation uses hypothetical strikes and expiration at the observed close,
without asserting that such contracts existed on every historical date.

The targets are QQQ and SPX log(close/open), not actual option cash flows. The
forecast cutoff is inherited from the original study, including its lag. We
cannot infer intraday stop execution, actual settlement, assignment, broker
margin or slippage from a closing return. Real contract eligibility and actual
settlement must be recorded for a tradable backtest.

Each spread's debit divided by its width is bounded between0and1. A condor's two
wings cannot both lose at one terminal price, so it also has a maximum of one
width. This bounded payoff remains integrable even when a transformed
log-return distribution has no finite arithmetic-return moments. No simulated
mean return, finite-moment assumption or Kelly sizing is used.

Two fixed Sobol integrations estimate joint breach probabilities and portfolio
VaR/ES. Individual probabilities use analytic CDFs. Expected spread debit uses
checked one-dimensional integration of those CDFs, identically within each
copula pair. A prewritten synthetic narrow-density example exposed an inaccurate
48-node quadrature result. Before market evaluation, the code was corrected to
compare48and96nodes and use adaptive integration when necessary; the failure
and passing regression receipts are retained in `prefit/`.

## Credits, costs and claims

All54 model/structure/distance risk cases are retained. The economic supplement
reports all189 policy/case/premium combinations per phase, assuming credits of
5%,10%or20%of width and round-trip costs of2%of width. These fixed hypothetical
credits are not quotes. Quiet days generally need not offer the same premium
as volatile days, so the scenarios cannot establish executable profits.

The more direct quantities are forecast versus observed breach probability,
joint full-width losses, expected versus observed debit, policy coverage and
break-even credit. Avoided liability is reported alongside forgone assumed
credit. Neither a high win rate nor a reduction in losses alone proves the
options were underpriced or worth selling.

All risk and scenario metrics are descriptive: no additional p-values, no
winner selected after inspection, and no promoted trading rule. The unadjusted
tail rates also have few extreme events and serial dependence. A VaR exception
and an ES exceedance have different meanings; spread-payoff atoms at zero and
maximum loss prevent treating every exceedance count as an exact2.5%target.

## New evidence

The implemented prospective ledger and [collection runbook](../../../docs/PROSPECTIVE_BENCHMARK.md)
specify immutable forecast receipts, paired benchmarks, actual option-leg quote
receipts, first-outcome scoring, revisions, coverage and a fixed evaluation
endpoint. Its current built-in scorer covers scalar normal/t and variance
forecasts; the current joint/shape model and actual spread decisions still need
a live adapter and quote-feed entitlement. It is not represented as an active
collector. New prospective collection must begin with real receipt timestamps,
not retrospective historical timestamps.

Original frozen studies and unrelated worktree changes are retained. Source and
test hashes, the selected test gate, immutable registration, saved output bytes
and independent checks will be retained under this study's report and private
data directories. No purchase, order, subscription or automated trading occurs.
