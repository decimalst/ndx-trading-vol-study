# Medium-horizon SPX returns: independently verified, no qualifying increment

The asymmetric implied-versus-trailing-proxy hinge did not improve on the strong component baseline at either horizon in either fixed period. It also failed the historical-mean comparisons. All four contrasts remain in the results; every wave and cumulative Holm-adjusted probability equals 1. This is a failed registered candidate, not proof that these data contain no predictive information.

## Results

Positive numbers below mean lower mean squared prediction error. Negative numbers mean deterioration. The fixed effect gate required at least +0.25% against both controls in both periods, alongside corrected uncertainty, stability slices, and every nonoverlapping offset.

| Horizon | Required control | Development MSE improvement | Evaluation MSE improvement | Development improving offsets | Evaluation improving offsets |
|---|---|---:|---:|---:|---:|
| 21 sessions | baseline | -1.350% | -1.271% | 8/21 | 0/21 |
| 21 sessions | mean | -9.499% | -1.809% | 2/21 | 7/21 |
| 63 sessions | baseline | -0.408% | -0.075% | 25/63 | 30/63 |
| 63 sessions | mean | -2.171% | +0.086% | 29/63 | 31/63 |

The quarterly result against the historical mean improved by only 0.086% in evaluation, below the 0.25% effect gate. It deteriorated by 2.171% in development, failed the 2020–2022 stability slice, improved in only 31 of 63 evaluation offsets, and had a conservative evaluation probability of 0.9964. It is not a lead.

The monthly candidate deteriorated against the component baseline in all 21 evaluation offsets. Quarterly baseline differences changed sign between the two evaluation slices. The candidate failed the average effect requirements even before correction or the all-offset gate was applied.

## Uncertainty and power

Absolute paired MSE gaps use candidate loss minus control loss; negative favors the candidate. Intervals are envelopes of nominal 95% circular block-bootstrap and HAC intervals, not simultaneous family confidence intervals. Nominal MDE is the HAC normal-approximation effect for 80% power at a two-sided 5% level; it does not incorporate the stricter family and conjunction gates.

| Horizon / control | Period | Absolute MSE gap | Nominal interval envelope | Nominal MDE | Conservative phase p |
|---|---|---:|---|---:|---:|
| 21 / baseline | development | +0.00001435 | [-0.00004883, +0.00007694] | 0.00007473 | 0.669540 |
| 21 / baseline | evaluation | +0.00003909 | [-0.00002750, +0.00011260] | 0.00009519 | 0.249940 |
| 21 / mean | development | +0.00009346 | [-0.00005672, +0.00025850] | 0.00017867 | 0.260900 |
| 21 / mean | evaluation | +0.00005536 | [-0.00015471, +0.00027138] | 0.00016557 | 0.604530 |
| 63 / baseline | development | +0.00000950 | [-0.00003432, +0.00005400] | 0.00005915 | 0.687410 |
| 63 / baseline | evaluation | +0.00000453 | [-0.00005306, +0.00006240] | 0.00005379 | 0.875220 |
| 63 / mean | development | +0.00004969 | [-0.00064496, +0.00071321] | 0.00092755 | 0.893990 |
| 63 / mean | evaluation | -0.00000521 | [-0.00256517, +0.00229871] | 0.00317565 | 0.996400 |

Every interval crosses zero. The 63-session development offsets contain only 14–15 nonoverlapping outcomes each, and evaluation offsets contain 22–23. Long blocks acknowledge overlapping daily targets but do not create more independent quarters. Historical reuse and finite-sample uncertainty remain.

## Actual coverage and validation

| Horizon | Development origins | Last development origin | Evaluation origins | Last evaluation origin | Initial training labels | Monthly fits |
|---|---:|---|---:|---|---:|---:|
| 21 | 985 | 2019-11-29 | 1,437 | 2025-09-19 | 1,234 | 118 |
| 63 | 943 | 2019-10-01 | 1,395 | 2025-07-22 | 1,192 | 118 |

Both horizons begin development on 2016-01-04 and evaluation on 2020-01-02. Complete target ends are bounded at 2019-12-31 and 2025-10-20 respectively. There are 2,463 feature-complete application origins per horizon, including entries whose eventual targets are excluded by those fences. Those future exclusions never select the monthly fit date.

All 874 repository tests passed before the run; 55 are new tests for this wave. The runner then passed its 60 focused checks before registration and fitting. The independent verifier reconstructed 63,390 raw feature cells, 8,452 target values and their date/missingness patterns, all 14,280 forecasts, 236 monthly horizon fits and training-centered transforms, eight phase comparisons, 24 bootstrap calculations with 99,999 draws apiece, 336 nonoverlapping offset summaries, and all 107 ledger events. The maximum independent stationarity residual for saved ridge coefficients was 8.874e-17, below the predeclared 1e-10 tolerance.

The first empirical run and first independent verification succeeded without repair, retry, tolerance changes, or model changes. The four publication-fault tests also establish that registration, report, evaluated-ledger, and protocol-mutation failures would retain all four hypotheses as unevaluable with probability one. Prior measurement identities are preserved in the cumulative ledger.

## Methodology and limits

The candidate adds one training-centered hinge to a 17-column component baseline. That baseline includes implied and trailing realized risk, their separate curves, return and variance histories, IV shape, VVIX, and weekdays. Monthly expanding ridge uses exact common matured labels and an unpenalized same-row mean. All market inputs lag one observed session. A plain linear difference of already-included components would add no predictor span, so it was not tested as an information increment.

This return-target question is motivated by implied-versus-realized variation research. Its published results emphasize accurately measured intraday realized variation; the daily OHLC proxy used here is materially different. [Bollerslev, Tauchen and Zhou (2009)](https://public.econ.duke.edu/~get/wpapers/btz.pdf).

SPX raw price returns exclude dividends and risk-free interest. The VIX options horizon and annualized trailing daily OHLC variance use different conventions; their log gap is not a directly measured variance risk premium. Yahoo/Cboe sources are archival, early VIX9D is back-calculated, and the observed vendor calendar is not a certified complete historical exchange calendar. No executable strategy or profitability claim follows. The protected period beginning 2025-11-03 was not admitted.

The manifest pins 147 Python source/test files, eight source/provenance inputs, and 230 earlier protocol/report artifacts. All earlier frozen hashes remain unchanged. Six recent waves now contain 49 added comparisons and 92,836 scored forecasts. Combined with the 54 previously enumerated model/signal comparisons, the tracked correction family is 103; this is not an exhaustive count of every experiment ever run in the repository.

## Artifacts and continuation

Full per-year effects, all offsets, and all block results remain in the machine-readable metrics. The next prospective question is whether a source predicts standardized downside-tail shape after controls already let it affect conditional mean and scale. Its feasibility and overlap with earlier jump/stress classifications are documented separately; it has not been registered, fitted, or scored.

- [Protocol](/Users/byrons/code/trading-vol/ndx-vol-experiment/index_hinge.yaml)
- [Metrics](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/index_hinge/metrics.json)
- [Independent verification](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/index_hinge/verification.json)
- [Design and source review](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/index_hinge/DESIGN_REVIEW.md)
- [Next target prospectus](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/index_hinge/NEXT_TARGET_DESIGN.md)
- [Exportable comparison figure](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/index_hinge/comparison_intervals.pdf)

![All four registered comparisons](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/index_hinge/comparison_intervals.png)
