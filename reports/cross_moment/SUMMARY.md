# Direct QQQ–SPX shared-risk test: verified, no qualifying signal

The dynamic-correlation forecast does not qualify under the separately frozen direct cross-moment test. Both registered comparisons were evaluated on every original issued row and independently verified on the first attempt. No model, forecast, previous score, or earlier verdict was changed.

Against the stronger constant-correlation control, the early-period mean improvement misses the fixed effect threshold, the 2020–2022 slice worsens, and uncertainty includes no improvement in both phases. Against the constant residual matrix, mean improvements are larger and both later slices improve, but neither phase clears even the conservative nominal 5% test. Both comparisons fail the wave and cumulative multiplicity gates.

## What was tested

Use the original 7,386 model/origin rows from 2,462 forecast origins: 1,005 in development (January 4, 2016–December 30, 2019) and 1,457 in evaluation (January 2, 2020–October 17, 2025). These are next-session raw intraday log-return forecasts for the QQQ ETF and SPX price index, with the original label maturity and one-session predictor lag. The source-value fence remains October 20, 2025; November 3 onward remains sealed. There were **zero new model fits and zero newly generated forecasts**.

For common issued means, the realized target is `Y=(r_QQQ-mu_QQQ)*(r_SPX-mu_SPX)`; each model predicts `q=rho*sqrt(h_QQQ)*sqrt(h_SPX)`. Score squared error `(Y-q)^2`. Both conditional models share identical issued means and marginal moments. The constant-matrix control has different marginal moments, so its comparison cannot isolate the dependence increment. Signed and zero products are valid.

All differences below are candidate minus control in fourth-power decimal log-return units; **negative is better**. The predeclared minimum reduction is `1e-10` in both phases against both controls, with negative gaps in both later fixed slices. This threshold is the squared cross-moment error of a 1% by 10-basis-point reference pair, not measured economic utility, a literal RMSE reduction, or profitability.

## Mean loss and uncertainty

| Control | Period | Candidate MSE | Control MSE | Paired difference | Nominal 95% interval envelope | Conservative phase p |
|---|---|---:|---:|---:|---|---:|
| Constant correlation | development | 2.304236922e-08 | 2.308817713e-08 | -4.580791042e-11 | [-1.843346235e-10, +5.970099131e-11] | 0.45572000 |
| Constant correlation | evaluation | 1.302356334e-07 | 1.304602790e-07 | -2.246456302e-10 | [-1.864607373e-09, +1.207425660e-09] | 0.77452000 |
| Constant matrix | development | 2.304236922e-08 | 2.595995611e-08 | -2.917586897e-09 | [-6.940168361e-09, +6.817343609e-11] | 0.06899000 |
| Constant matrix | evaluation | 1.302356334e-07 | 1.483864184e-07 | -1.815078499e-08 | [-4.231699443e-08, +1.047867788e-09] | 0.06387911 |

Intervals are the envelope of the nominal 95% percentile circular-block bootstrap intervals and the Bartlett HAC126 interval. The phase p-value is the maximum of the three centered-null two-sided bootstrap tests and HAC126. Percentile intervals and centered-null tests need not be exact inversions; promotion uses the frozen conservative p-values. Every envelope spans zero. No percentage transformation or alternative score was used.

| Control | Both-phase p | Holm2 p | Holm114 p | Result |
|---|---:|---:|---:|---|
| Constant correlation | 0.77452000 | 0.77452000 | 1.00000000 | Does not qualify |
| Constant matrix | 0.06899000 | 0.13798000 | 1.00000000 | Does not qualify |

Holm2 must be below `0.05/(11*12) = 0.0003787878788`; independently, Holm114 must be below `0.05`. The family retains 112 inherited hypotheses plus these two new score comparisons. All 116 ledger events—112 inherited, two registered, two evaluated—were verified. No failed comparison was dropped.

## Stability and measurement precision

| Control | 2020–2022 gap | 2023–October 2025 gap | Development nominal MDE / threshold | Evaluation nominal MDE / threshold |
|---|---:|---:|---:|---:|
| Constant correlation | +7.265850805e-10 | -1.250509278e-09 | 1.226 | 20.470 |
| Constant matrix | -1.578783657e-08 | -2.069912880e-08 | 42.678 | 274.422 |

The minimum detectable effects are nominal HAC-based 80% diagnostics at the usual two-sided 5% level, not power estimates at the much stricter registered multiplicity thresholds. In evaluation, the nominal detectable effect against constant correlation is about 20.47 times the fixed useful-effect reference. This is substantial uncertainty; nonqualification is not an equivalence finding or proof that predictability is absent. All annual diagnostics remain in the metrics; no favorable year was selected for promotion.

## What this changes in the research

The earlier joint matrix score showed nominal improvements, but its dynamic dependence parameter can also compensate for imperfect variance forecasts. Direct product loss evaluates a different functional and does not use the matrix inverse to reweight squared residuals. The current result provides no reliable evidence that the same issued forecasts improve direct residual cross-moment prediction. It does not identify the cause of the earlier joint-score gain.

The conditional expectation of Y equals conditional covariance plus the product of the two conditional mean errors. An individual daily Y is a noisy observation, and the conditional mean being predicted is not automatically true covariance. The forecast still uses issued marginal moments. Daily products are noisy, their uncertainty depends on cross fourth moments, and the archived ETF/index history has already been reused. No measured high-frequency covariance, exact Nasdaq-100 index, fresh-holdout, execution, or profit claim follows.

Target alignment remains a distinct next question: these issued models were fitted for joint matrix loss, whereas this wave evaluates product squared error. A separately registered model fitted directly to a clearly defined cross-moment target could test that mismatch. It must specify training-label availability, appropriate strong controls, structural constraints, and all thresholds before fitting. The current findings do not justify changing this wave or presuming such a model will succeed.

## Verification and preservation

- Full repository discovery: **1,289 tests passed** in 38.756 seconds. A final presentation-only chart fix followed; all **76 focused checks passed** afterward, including rendered chart bounds. The runner reran the same prewritten checks before registration. Lint passed for all nine new source/test files. The unrelated pre-existing whole-repository lint issue was not changed.
- The pre-run freeze and run manifest agree on all 200 source/test files. **All 583 manifest entries**—200 code/test, 57 upstream/input, and 326 preserved—remain unchanged. All ten previous wave manifests were also checked before the run. The 57 inputs include the original raw-input pins and every upstream protocol, report, and private output.
- Admission reconstructed the prior verified experiment through read-only verifier functions. The new scorer and verifier each hashed and decoded the same immutable issued byte buffer. Neither called an earlier writing or invalidation entrypoint.
- Independent verification rebuilt **22,158 primitive score cells**, **4,924 paired gaps**, all four phase analyses, **12 bootstrap runs of 99,999 draws**, both Holm families, and all 116 ledger events. It passed without repairs, retries, or tolerance changes.
- Across the eleven recent waves, 60 comparisons have been added to the earlier 54 enumerated hypotheses. The generated-forecast total remains **121,279**; this wave rescored 7,386 of those existing rows. These are not repository-lifetime totals.

Protocol SHA256: `edeb72e2d7708c0a5227318d1ea54742050f9be0b761941ce598cdda28c26ed0`. Independent verifier SHA256: `d1cb8c3fc69e44646359b839aac160acf6ed4be6cc043e2f2ca23f7c979430fa`.

## Files

- [Frozen protocol](/Users/byrons/code/trading-vol/ndx-vol-experiment/cross_moment.yaml)
- [Pre-score design](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/cross_moment/DESIGN.md)
- [Independent design and implementation review](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/cross_moment/DESIGN_REVIEW.md)
- [Pre-run freeze record](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/cross_moment/freeze_record.json)
- [Final focused checks](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/cross_moment/focused_pre_fit_checks.txt)
- [Full repository checks](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/cross_moment/full_pre_fit_checks.txt)
- [Run manifest](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/cross_moment/manifest.json)
- [Complete metrics and annual diagnostics](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/cross_moment/metrics.json)
- [Independent verification](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/cross_moment/verification.json)
- [Complete trial ledger](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/cross_moment/trial_ledger.jsonl)
- [Unchanged preceding joint-risk result](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/joint_risk/SUMMARY.md)
- [Comparison figure (PNG)](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/cross_moment/comparison_intervals.png)
- [Comparison figure (PDF)](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/cross_moment/comparison_intervals.pdf)
- [Blinded proposal for direct target-aligned fitting](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/cross_moment/NEXT_TARGET_ALIGNMENT_DESIGN.md)
