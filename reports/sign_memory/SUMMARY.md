# Sign-agreement memory: small later improvement, no qualifying signal

The directional-agreement memory model does not qualify under either registered comparison. Against the strong conditional baseline, it increases mean Brier loss in development and decreases it in evaluation. The later improvement is consistent across both fixed evaluation slices and its nominal interval excludes zero, but its size is only about 5.6% of the predeclared 0.0005 effect requirement. The complete both-period and multiplicity gates fail.

This is a completed, independently verified experiment. All 118 monthly fits produced the three probability models, with 7,386 new forecast rows. No fit had constant memory, and no support or numerical failure removed a fold. No code, protocol, threshold, window or tolerance changed after fitting.

## Target and experiment

At each origin, predict whether QQQ ETF and SPX price-index raw intraday log(close/open) returns will be strictly in the same direction on the next observed SPX session. Both positive or both negative is an event. A zero return in either asset is a retained nonevent; the complement therefore includes ties as well as opposite directions. Missing pairs stay unknown. This is a probability target for directional agreement, not standalone direction, volatility magnitude or covariance.

The memory feature is the previous 22-session agreement fraction minus the independence reference formed from the same window’s four marginal positive/negative fractions. All summaries use an identical complete paired window ending one observed session before origin. The baseline contains the existing lagged risk, signed-return and continuous-correlation controls, their training-centered correlation square, and those marginal sign controls. Existing nonintercept terms use training population scaling; five bounded additions and the memory use fixed scale one and training centering.

The frequency control uses the identical mature training sample. The baseline fits normalized ridge logistic likelihood; the candidate freezes that logit and adds one regularized memory coefficient, with no additional intercept. Both penalties are 0.01. These are two convex stages, not a joint augmented fit. Likelihood training and Brier evaluation elicit the same probability functional, but need not improve together in a misspecified restricted model.

Brier loss is (event − probability)², bounded in [0,1]; lower is better. Every paired difference below is candidate minus control in absolute squared-probability-error units. The expected loss is pi(1−pi)+(p−pi)², avoiding return-fourth-moment requirements while retaining serial dependence and sampling uncertainty.

## Every registered result

| Control | Period | Candidate Brier | Control Brier | Paired difference | Nominal 95% interval envelope | Conservative phase p |
|---|---|---:|---:|---:|---|---:|
| Conditional baseline | development | 0.1553028914 | 0.1552880108 | +1.488056330e-05 | [-4.051634759e-07, +3.261612701e-05] | 0.06597000 |
| Conditional baseline | evaluation | 0.1179525672 | 0.1179806351 | -2.806789870e-05 | [-4.668131215e-05, -9.974106281e-06] | 0.00236233 |
| Expanding historical frequency | development | 0.1553028914 | 0.1552095473 | +9.334406015e-05 | [-2.434193421e-03, +2.606067315e-03] | 0.94171000 |
| Expanding historical frequency | evaluation | 0.1179525672 | 0.1226971325 | -4.744565328e-03 | [-7.011837243e-03, -2.403617386e-03] | 0.00008000 |

The envelope includes Bartlett HAC126 and percentile circular-block bootstrap intervals for blocks 21, 63 and 126. Each phase p is the maximum of the corresponding two-sided tests, with centered-null bootstrap tests. Percentile intervals and centered-null tests are not exact inversions. Both phases enter each hypothesis through their maximum p; no favorable method or period is selected.

| Control | Both-period p | Holm2 p | Holm119 p | Verdict |
|---|---:|---:|---:|---|
| Conditional baseline | 0.06597000 | 0.13194000 | 1.00000000 | Does not qualify |
| Expanding historical frequency | 0.94171000 | 0.94171000 | 1.00000000 | Does not qualify |

The candidate must beat both controls by at least 0.0005 in both phases, improve both fixed later slices, pass Holm2 at 0.05/(13×14) = 0.0002747252747, and separately pass cumulative Holm119 at 0.05. All 117 earlier hypotheses and both new comparisons remain counted. The .0005 reference is a fixed squared-probability-error threshold, not a measured profit or literal probability-point improvement.

## Stability and uncertainty

| Control | 2020–2022 difference | 2023–October 2025 difference | Development nominal MDE / effect | Evaluation nominal MDE / effect |
|---|---:|---:|---:|---:|
| Conditional baseline | -2.441166617e-05 | -3.201099683e-05 | 0.043698 | 0.051726 |
| Expanding historical frequency | -4.638714130e-03 | -4.858721543e-03 | 6.639727 | 6.481587 |

Against the baseline, both later slices improve, but by much less than the required effect. Development worsens rather than improves. The nominal HAC-based 80% minimum detectable effect is about 0.044–0.052 times the effect reference for this matched comparison. That is a descriptive diagnostic at ordinary two-sided 5%, not a power guarantee under the stricter repeated-search gates. The experiment does not support equivalence, nor does its small later association establish a usable new signal.

## Descriptive calibration and support

| Period | Model | Observed agreement | Mean issued probability | Probability minus observed rate |
|---|---|---:|---:|---:|
| development | frequency | 0.81094527 | 0.76398834 | -0.04695693 |
| development | baseline | 0.81094527 | 0.77183499 | -0.03911029 |
| development | memory | 0.81094527 | 0.77173818 | -0.03920709 |
| evaluation | frequency | 0.86273164 | 0.79487043 | -0.06786121 |
| evaluation | baseline | 0.86273164 | 0.82068623 | -0.04204541 |
| evaluation | memory | 0.86273164 | 0.82090244 | -0.04182920 |

All three models underpredict agreement on average in both periods. The baseline’s average shortfall is about 3.91 percentage points in development and 4.20 in evaluation; the new memory coefficient changes this little. These predeclared summaries describe calibration in the large, not conditional calibration or an independently tested temporal trend. A causal calibration-memory experiment with a strong recent-frequency control is a separate prospective question; these diagnostics do not repair or promote this run.

| Cohort | Observations | Agreement events | Nonevents, including ties |
|---|---:|---:|---:|
| development | 1005 | 815 | 190 |
| evaluation | 1457 | 1257 | 200 |
| 2020-01-02–2022-12-31 | 756 | 646 | 110 |
| 2023-01-01–2025-10-17 | 701 | 611 | 90 |

Each expanding training sample meets the frozen minimum of 1,000 rows and 50 examples per class. Each phase meets 30 per class and each fixed evaluation slice 15 per class. Fits occur at the first feature-complete monthly origin before query-label filtering, with training labels mature by the preceding session. The 2,463 application origins and 2,462 scored origins retain the original timing convention. Development labels stop at December 31, 2019; all source and target dates stop at October 20, 2025. The period beginning November 3, 2025 remains sealed.

## Validation, accounting and limits

- The full repository suite passed **1,461 tests** in 45.937 seconds. The runner then passed **100 prewritten focused checks** in 4.673 seconds before registration or empirical construction. Scoped lint passed for all twelve new source/test files; the unrelated earlier whole-repository lint issue remains unchanged.
- First-attempt independent verification reconstructed 4,226 full source rows and 40 raw predictors, all 118 monthly fits and 236 convex stages, 292,596 mature training-row instances, all 7,386 forecasts and Brier scores, 4,924 paired differences, four phase analyses, twelve bootstrap runs of 99,999 draws, and six calibration summaries. A separate trust-region baseline solve and bracketed scalar solve checked the producer’s Newton fits; primary scores use strict saved-coefficient replay.
- All **684 manifest entries** remain unchanged: 223 source/test files, 135 input/upstream artifacts and 326 preserved records. Before the run, all twelve earlier wave manifests were rechecked, covering 1,454 unique pinned files. Previous verification records were reconstructed through read-only functions.
- The successful ledger contains **121 events**: 117 inherited hypotheses, two registrations and two evaluated outcomes. The cumulative family is **119**. Across thirteen recent waves, 65 comparisons have been added to the initial 54 enumerated hypotheses, and **133,589 new forecasts** have been generated (126,203 previously plus 7,386 here). These are not repository-lifetime totals.

Archival Yahoo/Cboe vintages, ETF versus price-index construction, unsynchronized reported opens, early back-calculated VIX9D, and repeated reuse of history remain limitations. Exact historical publication latency, independent holdout confirmation, executable hedges and profit are not established. The frozen verdict is unchanged by the later-period nominal improvements.

Protocol SHA256: `a825704da953cd84302ee37430e446f8558d700859ccc7d2b44fbf5e2268f3d0`. Independent verifier SHA256: `44c0dc5ab288dfd95cc483b98ea7f198bd587830a24141baa210d259b13b5c27`.

## Files

- [Frozen protocol](/Users/byrons/code/trading-vol/ndx-vol-experiment/sign_memory.yaml)
- [Prefit design](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/sign_memory/DESIGN.md)
- [Independent design and implementation review](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/sign_memory/DESIGN_REVIEW.md)
- [Pre-run freeze record](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/sign_memory/freeze_record.json)
- [Full repository checks](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/sign_memory/full_pre_fit_checks.txt)
- [Runner prefit checks](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/sign_memory/pre_run_checks.txt)
- [Run manifest](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/sign_memory/manifest.json)
- [Complete metrics and annual diagnostics](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/sign_memory/metrics.json)
- [Independent verification](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/sign_memory/verification.json)
- [Complete trial ledger](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/sign_memory/trial_ledger.jsonl)
- [Source audit](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/sign_memory/source_audit.json)
- [Whole-source measurement audit](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/sign_memory/measurement_audit.json)
- [Comparison figure (PNG)](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/sign_memory/comparison_intervals.png)
- [Comparison figure (PDF)](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/sign_memory/comparison_intervals.pdf)
- [Prospective causal probability-pooling experiment](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/sign_memory/NEXT_CAUSAL_CALIBRATION_DESIGN.md)
- [Separate prospective calendar audit](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/sign_memory/NEXT_CALENDAR_AUDIT.md)
- [Unchanged previous direct-fitting study](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/target_aligned/SUMMARY.md)
