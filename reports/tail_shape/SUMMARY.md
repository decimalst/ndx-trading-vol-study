# Signed downside-tail shape: verified, no qualifying signal

Allowing lagged SKEW to change conditional distribution shape added no useful evaluation-period improvement over the constant-shape control. Its full-density score deteriorated in both periods. All three registered comparisons failed; every wave and cumulative Holm-adjusted probability equals 1. Event support and numerical verification passed, so this is an evaluable negative result, not an insufficient-data outcome.

## What was tested

The target is a next-session SPX raw-close log price return, normalized by a trailing mean and risk estimate available one session before entry. The event is normalized return below −1.5. It is not a fitted quantile, a universal 5% VaR event, or a prediction of an executable fund return.

Both density models receive the same 22-column basis, including SKEW and its separate curvature, to predict conditional mean and variance. Those fitted functions are held exactly equal between the models. The candidate adds one SKEW-dependent asymmetry slope to a moment-standardized skewed Student distribution. Tail heaviness, regularization, coefficient bounds, starts, support requirements, and all evaluation gates were fixed before constructing empirical event labels.

Hansen’s distribution changes asymmetry while its internal standardization preserves innovation mean zero and variance one. This permits a matched comparison of modeled shape. It does not eliminate remaining mean, variance, or fixed-tail-heaviness misspecification. [Hansen (1994)](https://www.ssc.wisc.edu/~bhansen/papers/ier_94.pdf).

## All registered results

Absolute paired gaps are candidate loss minus control loss; negative is better. The required improvements in each period were at least 0.0005 for both Brier comparisons and 0.005 for the full-density negative log score. Each also had to improve in both evaluation slices and pass both multiplicity corrections.

| Score / required control | Development gap | Evaluation gap | Development phase p | Evaluation phase p | Corrected verdict |
|---|---:|---:|---:|---:|---|
| brier / constant_shape | -0.00003477 | +0.00000067 | 0.049430 | 0.954000 | Does not qualify |
| brier / frequency | -0.00091665 | -0.00036597 | 0.438099 | 0.747920 | Does not qualify |
| nll / constant_shape | +0.00001161 | +0.00045055 | 0.982620 | 0.303527 | Does not qualify |

The earlier Brier improvement against constant shape had an uncorrected conservative phase probability of 0.04943, but its absolute effect was only 0.00003477—far smaller than the predeclared 0.0005 requirement. In evaluation the effect was slightly adverse, +0.00000067, with probability 0.954. The historical-frequency comparison improved on average in both periods, but failed corrected uncertainty, fell short of the evaluation effect requirement, and deteriorated in the 2023–2025 slice. A favorable early score alone would have given the wrong selection signal.

## Uncertainty

These are envelopes of nominal 95% paired block-bootstrap and HAC intervals, not simultaneous family confidence intervals. Nominal MDE is the HAC normal-approximation effect at 80% power and two-sided 5%, before stricter family and conjunction gates. Losses are not converted to percentage density improvements.

| Score / control | Period | Absolute gap | Nominal interval envelope | Nominal MDE |
|---|---|---:|---|---:|
| brier / constant_shape | development | -0.00003477 | [-0.00007252, -0.00000251] | 0.00004611 |
| brier / constant_shape | evaluation | +0.00000067 | [-0.00002304, +0.00002394] | 0.00003115 |
| brier / frequency | development | -0.00091665 | [-0.00323366, +0.00147931] | 0.00331189 |
| brier / frequency | evaluation | -0.00036597 | [-0.00241572, +0.00210651] | 0.00268297 |
| nll / constant_shape | development | +0.00001161 | [-0.00106509, +0.00105171] | 0.00148669 |
| nll / constant_shape | evaluation | +0.00045055 | [-0.00040773, +0.00131360] | 0.00122681 |

## Actual support and calibration

| Period | Common origins | Downside events | Nonevents | Observed event rate |
|---|---:|---:|---:|---:|
| development | 1,002 | 82 | 920 | 8.184% |
| evaluation | 1,456 | 146 | 1,310 | 10.027% |
| 2020-01-02 to 2022-12-31 | 756 | 83 | 673 | 10.979% |
| 2023-01-01 to 2025-10-17 | 700 | 63 | 637 | 9.000% |

Development origins run from 2016-01-04 to 2019-12-30; evaluation runs from 2020-01-02 to 2025-10-17. There are 2,459 feature-complete application origins and 2,458 scored origins. The development-end outcome fence removes the remaining application from scoring without changing its monthly fit schedule. The first fit uses 1,253 completed labels from 2011-01-05 through 2015-12-30, available by 2015-12-31, with 156 events and 1,097 nonevents. Those are also the minimum training class counts over all 118 monthly fits.

| Model | Development mean probability | Evaluation mean probability | Development Brier | Evaluation Brier |
|---|---:|---:|---:|---:|
| frequency | 11.408% | 10.655% | 0.07625234 | 0.09027937 |
| constant_shape | 11.225% | 9.799% | 0.07537047 | 0.08991274 |
| skew_shape | 11.217% | 9.808% | 0.07533570 | 0.08991340 |

Average predicted probabilities are only a coarse calibration summary. The full metrics retain all 60 fixed probability-bin summaries, including empty bins, plus annual effects and both fixed evaluation slices. These descriptive values do not promote an unregistered baseline comparison.

## Validation and provenance

All 958 repository tests passed before the run; 84 are new tests for this wave. The runner then passed 100 focused checks before registration and the first empirical feature construction or event count. The first empirical run and first independent verification succeeded without repairs, retries, or relaxed tolerances.

The independent verifier reconstructed all 4,226 source-calendar rows and 19 raw features, both normalization inputs, every normalized return and binary event, all 7,374 forecasts, all 118 monthly shared-moment fits and 236 shape optimizations, every class-support check, six phase comparisons, 18 bootstrap calculations with 99,999 draws each, cumulative correction across 106 comparisons, and all 109 ledger events. Both density forecasts have identical predicted mean and variance at every scored origin. Separate formula/quadrature tests verify probability mass, mean, variance, CDF, Jacobian, and shape gradients.

The maximum independently recomputed shared-variance stationarity residual was 1.727e-09; the maximum numerical shape projected-gradient residual was 1.319e-08, within the predeclared tolerances. Optimizer success is a checked stationary fixed-start result, not proof of a global optimum.

The manifest pins 159 Python source/test files, 11 input/provenance files, and 250 earlier protocol/report artifacts. Earlier frozen hashes remain unchanged. Seven recent waves now contain 52 added comparisons and 100,210 scored forecasts. Combined with the 54 previously enumerated comparisons, the tracked family is 106; this is not an exhaustive count of every historical experiment in the repository.

The pinned original SKEW archive matches its metadata, historical anchor, and all 9,001 bounded derived rows. Its five missing SPX dates remain gaps. Yahoo/Cboe inputs are archival; early VIX9D is back-calculated and exact historical release latency/revisions remain unverified. SKEW represents a 30-day option-implied construction, while this target is a one-day physical price-return event. SPX price returns exclude reinvested dividends and risk-free interest. No trading or profitability claim is made.

## Artifacts and next decision

The next calendar audit checks whether a proposed full-session announcement-volatility test would duplicate earlier work. Earlier calendar models already addressed that target; any further run must be labeled a replication with stricter schedule-availability controls, not a new signal discovery. The prospective note records the actual overlap and source-timing differences before another experiment is registered.

- [Protocol](/Users/byrons/code/trading-vol/ndx-vol-experiment/tail_shape.yaml)
- [Complete metrics](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/tail_shape/metrics.json)
- [Independent verification](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/tail_shape/verification.json)
- [Design and pre-fit review](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/tail_shape/DESIGN_REVIEW.md)
- [Source review](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/tail_shape/SOURCE_REVIEW.md)
- [Next calendar audit](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/tail_shape/NEXT_CALENDAR_DESIGN.md)
- [Exportable comparison figure](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/tail_shape/comparison_intervals.pdf)

![All three registered comparisons](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/tail_shape/comparison_intervals.png)
