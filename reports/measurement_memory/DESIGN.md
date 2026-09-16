# Fixed experiment before empirical fitting

This wave tests two possible additions to volatility forecasts: a reported
uncertainty width, and an interaction between that width and a recent volatility
innovation. A width-only model can reveal useful information even if the
interaction fails. An interaction model must also beat the width-only model.
The earlier nine-comparison proposal is retained in the preceding wave's report;
this specification expands it to fifteen comparisons before reading any new
predictive association or fitting any model.

The source is the already audited SPY trade-QMLE series. Its reported interval
half-width is displayed in the same native volatility units. The relative-width
feature is `log1p(width / volatility)`. The interaction multiplies this by the
difference between current log squared QMLE and its trailing 22-session log
mean. Both use measurements dated two reference sessions before the forecast.
No statistical confidence level, calibrated noise variance or quarticity is
inferred from the width. Because width can covary with the volatility state or
liquidity, a predictive interaction would not identify a measurement-error cause.

The target is next-session squared provider-native trade-QMLE volatility.
There is no invented conversion into daily integrated variance. Two alternative
measurements, the provider's five- and fifteen-minute trade estimates squared,
score the exact same forecasts. They never refit a model or alter primary-target
scaling. All four models share complete inputs and all three matured targets in
their training and scored samples.

The market-history baseline has own QMLE history over 1/5/22 sessions; VIX,
VIX9D/VIX and VVIX; negative QQQ return magnitudes over 1/5/22 sessions; and entry
weekday indicators. QQQ is a broad-market price control, not SPY's own return
history. Market inputs are one reference session old. All fits use the same
frozen positive log-link proper-score estimator and slope penalty. Scaling uses
training observations only. Zero-scale inputs or failed convergence abort the
entire family without fallback.

The reference calendar uses bounded observed QQQ session labels. Missing SPY
dates stay present as gaps before rolling or shifting. The target is assigned
maturity two reference sessions after its measurement date; all training labels
must mature by the session preceding the fit. Monthly fits start at the first
feature-complete origin, regardless of whether that origin's future measurement
is eventually available. The first model needs at least 1,000 common training
rows. Development is 2016–2019, and evaluation is 2020–October 2025, with the
same source and maturity fences used by the preceding research.

These delays are declared archival assumptions. Historical provider release
times, revisions and exact session boundaries remain unverified. VIX9D's early
back-calculated history is an archival training input, not a claim of availability
at its 2011 observation dates. The data omit the entire February 5–9, 2018 week;
this test cannot establish performance on that missing stress episode. The
[source review](SOURCE_AND_DESIGN_REVIEW.md) preserves these limitations.

The fifteen contrasts are width against baseline and mean, and interaction
against baseline, mean and width, each scored against all three measurements.
A candidate must improve every primary comparison by at least 0.005 absolute
proper-score units in both development and evaluation, improve both fixed
evaluation subperiods, and pass Holm15 at `1/600` plus Holm99 at `0.05`.
Every alternative-measurement comparison must have a negative gap in both
periods. Those ten sensitivity comparisons also enter both correction families;
they have no additional size or significance threshold. There are 99,999 draws
for each of the fixed 21/63/126 observation blocks, with conservative HAC and
two-period checks. Negative raw scores are never converted to percentage gains.

The source, feature, maturity, fit, primary-only training, alternate scoring and
publication-failure contracts have prewritten synthetic tests. A separate
implementation reconstructs the source fields, every feature and fitted model,
all forecasts and all inference before interpretation. The pre-fit manifest
pins every repository Python source/test dependency, actual runtime/backend,
source artifacts and earlier protocols/reports. Failures retain all fifteen
hypotheses as unevaluable with p-values of one. No new empirical fit or score
existed when this document was written.

- [Full protocol](../../measurement_memory.yaml)
- [Source and design review](SOURCE_AND_DESIGN_REVIEW.md)
- [Earlier prospective proposal](../macro_overnight/NEXT_MEASUREMENT_DESIGN.md)
