# Causal probability pooling: verified results

The fixed pool does not qualify under the complete registered gates.

The pool lowers mean Brier loss versus the frozen conditional baseline by 0.00205469 in development and 0.00113666 in evaluation. Both point estimates exceed the fixed 0.0005 effect requirement, and both later slices improve. However, evaluation uncertainty includes zero and the both-period Holm2 p is 0.48489. Against recent frequency, the pool worsens development loss by 0.00025215; its later improvement does not repair that failed comparison. Three of the four nominal interval envelopes include zero. Neither comparison passes the complete gates.

The experiment combines the unchanged issued QQQ–SPX directional-agreement probability with a causal estimate of the recent agreement rate. It asks whether the combination improves on **both** components. The memory half-life is 63 observed SPX sessions and the blend is exactly 50:50; neither was selected from this run’s results.

Independent verification reconstructed 4,924 new forecasts and 2,462 preserved baseline rows across 2,462 scored origins. No baseline model was refitted. It checked 2,463 saved application states against 2,463 full-calendar states, including 2,462 post-seed label arrivals.

## Every registered result

Brier loss is squared probability error; lower is better. Differences are pooled minus control. Intervals and phase p-values below are nominal, before the repeated-search corrections.

| Control | Period | Pooled Brier | Control Brier | Difference | Nominal 95% interval envelope | Conservative phase p |
|---|---|---:|---:|---:|---|---:|
| Frozen conditional baseline | development | 0.1532333233 | 0.1552880108 | -2.054687531e-03 | [-4.061753782e-03, -1.199272939e-04] | 0.03738891 |
| Frozen conditional baseline | evaluation | 0.1168439784 | 0.1179806351 | -1.136656612e-03 | [-3.131468598e-03, +7.692900145e-04] | 0.24244689 |
| Recent event-rate estimate | development | 0.1532333233 | 0.1529811715 | +2.521518316e-04 | [-1.225298828e-03, +1.729602491e-03] | 0.73799701 |
| Recent event-rate estimate | evaluation | 0.1168439784 | 0.1181229857 | -1.279007215e-03 | [-2.568821831e-03, +2.445944740e-05] | 0.05194647 |

| Control | Both-period p | Holm2 p | Holm121 p | Verdict |
|---|---:|---:|---:|---|
| Frozen conditional baseline | 0.24244689 | 0.48489379 | 1.00000000 | DOES_NOT_QUALIFY |
| Recent event-rate estimate | 0.73799701 | 0.73799701 | 1.00000000 | DOES_NOT_QUALIFY |

Both comparisons must improve by at least 0.0005 in both development and evaluation, improve each fixed later slice, pass Holm2 at 0.05/(14×15), and pass cumulative Holm121 at 0.05. All 119 inherited comparisons remain counted. The threshold is an absolute squared-probability-error decrease, not a probability-point change or a profit threshold.

## Stability and calibration diagnostics

| Control | 2020–2022 difference | 2023–October 2025 difference | Development nominal MDE / effect | Evaluation nominal MDE / effect |
|---|---:|---:|---:|---:|
| Frozen conditional baseline | -1.953857461e-03 | -2.553387197e-04 | 5.531016 | 5.448645 |
| Recent event-rate estimate | -9.190144809e-04 | -1.667244743e-03 | 4.223677 | 3.687271 |

The HAC-based 80% minimum detectable effect is a nominal diagnostic at ordinary two-sided 5%. It is not power under the stricter search gates, an equivalence test, or evidence that a failed comparison has no predictability.

| Period | Model | Observed agreement | Mean probability | Probability minus observed rate |
|---|---|---:|---:|---:|
| development | Frozen conditional baseline | 0.81094527 | 0.77183499 | -0.03911029 |
| development | Recent event-rate estimate | 0.81094527 | 0.79611779 | -0.01482748 |
| development | Fixed half-and-half pool | 0.81094527 | 0.78397639 | -0.02696888 |
| evaluation | Frozen conditional baseline | 0.86273164 | 0.82068623 | -0.04204541 |
| evaluation | Recent event-rate estimate | 0.86273164 | 0.86388699 | +0.00115535 |
| evaluation | Fixed half-and-half pool | 0.86273164 | 0.84228661 | -0.02044503 |

These predeclared summaries describe average calibration only. They do not establish conditional calibration, a new temporal trend, or a mechanism. A smaller absolute mean calibration gap cannot promote a model whose registered loss gates fail.

## Timing, controls and independent checks

The target is strict next-session raw intraday sign agreement between the QQQ ETF and SPX price index. Opposite directions and exact ties are retained nonevents; missing paired returns stay unknown. The first original monthly fit’s eligible mature training frequency initializes the memory at the first application’s preceding-session cutoff, with unit mass. Its last included label date is recorded separately. Labels at or before that cutoff are never fed again.

Every later reference close ages the state. Newly mature labels enter from the full target table, including origins with missing features or no scored forecast. Missing labels age numerator and mass together without adding an observation. The state never resets at a phase boundary or an unscored month. A forecast uses only its previous-session state, so the preceding origin’s still-unavailable next-session label cannot enter.

The producer uses a recurrence; the independent checker uses explicit compensated sums of calendar-aged weights. Its roundoff bound was fixed before computation. Probability domains and zero/sign masks are checked first; saved quotients, half-product blends, baseline values and Brier scores replay exactly. Input JSON/parquet and output snapshots are hash-bound. Independent verification replays the complete prior proof through frozen read-only functions.

Inference uses Bartlett HAC126 and 99,999-draw circular-block bootstraps at lengths 21, 63 and 126, with seed 20260920 and the fixed phase/block offsets. Each hypothesis uses the largest two-sided p across methods and phases. Percentile interval envelopes and centered-null tests are not exact inversions.

| Cohort | Observations | Agreement events | Nonevents including ties |
|---|---:|---:|---:|
| development | 1005 | 815 | 190 |
| evaluation | 1457 | 1257 | 200 |
| 2020-01-02–2022-12-31 | 756 | 646 | 110 |
| 2023-01-01–2025-10-17 | 701 | 611 | 90 |

The complete repository suite passed **1,533 tests** before the run, followed by **77 focused pre-run checks**. Scoped lint passed for all ten new source/test files. All **747 manifest entries** remain unchanged: 233 source/test files, 199 inputs and 315 preserved records. The ledger contains 119 inherited hypotheses, two registrations and two evaluated outcomes: 123 events for a 121-comparison family.

Across fourteen recent waves, 67 comparisons have been added to the initial 54 enumerated hypotheses, with 138,513 new forecasts generated (133,589 previously plus 4,924 here). Reused baseline rows and the cross-moment rescoring run do not count as new forecasts. These are not repository-lifetime totals. All thirteen earlier wave manifests were also rechecked before this run, covering 1,504 unique pinned files.

No code, protocol, memory length, blend weight, threshold, window or tolerance changed after registration. The October 20, 2025 numerical fence and the excluded period beginning November 3, 2025 remain unchanged.

Archival source revisions, ETF versus index construction, unsynchronized reported opens, unverified historical availability and early back-calculated VIX9D remain limitations. This adaptively selected test reuses history; causal replay and multiplicity correction do not supply untouched confirmation. It concerns directional-agreement probability, not volatility magnitude, standalone index direction, covariance or trading profit.

Protocol SHA256: `1e329ee208612ed8db1fede873b92a938820d51a3fcf2ec583c813d37ac000ad`. Independent verifier SHA256: `cb34da06786f61e50b89f494e4071abb8d250539a52145d782cc1e122aeaa900`.

## Files

- [Prefit design](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/causal_pool/DESIGN.md)
- [Independent design and implementation review](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/causal_pool/DESIGN_REVIEW.md)
- [Pre-run freeze record](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/causal_pool/freeze_record.json)
- [Full repository tests](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/causal_pool/full_pre_fit_checks.txt)
- [Focused pre-run checks](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/causal_pool/pre_run_checks.txt)
- [Complete manifest](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/causal_pool/manifest.json)
- [Every metric and annual diagnostic](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/causal_pool/metrics.json)
- [Independent verification](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/causal_pool/verification.json)
- [Complete trial ledger](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/causal_pool/trial_ledger.jsonl)
- [Comparison figure (PNG)](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/causal_pool/comparison_intervals.png)
- [Comparison figure (PDF)](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/causal_pool/comparison_intervals.pdf)
- [Frozen protocol](/Users/byrons/code/trading-vol/ndx-vol-experiment/causal_pool.yaml)
- [Prospective civil quarter-end volatility experiment](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/causal_pool/NEXT_CALENDAR_VARIANCE_DESIGN.md)
- [Unchanged previous memory study](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/sign_memory/SUMMARY.md)
