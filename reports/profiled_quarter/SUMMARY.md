# Civil quarter-end SPX risk: verified replay

**No candidate passes all registered predictive gates.** Independent verification completed for this attempt. The two earlier attempts retain their separate failure records and p=1 comparisons; their failures are not statistical null results.

The quarter-end addition slightly worsened average forecast loss in both phases and both later slices. Against the full controlled baseline, the evaluation difference was **+0.0000621** in proper-score units, with wave-adjusted p=**0.39746**. The full baseline already forecasts much better than the historical mean on these comparisons; adding the quarter-end term provides no observed incremental improvement.

The run generated and independently reconstructed **6,297 forecasts** for **2,099 common scored origins**, with **101 monthly fitting blocks** and **2,099 applications**. The protocol also requires checks of applications without mature scored labels; this run had no additional unscored applications.

The experiment asks whether the final five civil dates of March, June and September improve next-session SPX daily risk forecasts after controlling for market volatility, original macro-plan flags, eleven month indicators, ordinary month-end and December year-end. The candidate adds one penalized centered slope to a fitted baseline. It must also beat the same-sample historical mean.

The target is the inherited floored daily Garman–Klass proxy plus squared raw overnight return. The nominal calendar skips weekends only. Neither the producer model nor the information set changed in this attempt; the independent baseline optimizer changed to exact intercept profiling and a bounded directional-gradient solve.

## Verified comparisons

The nominal evaluation window begins in 2020, but the first common scored evaluation origin is **November 2, 2020**, with only 42 scored dates in that year. The evaluation and its 2020–2022 slice therefore do not cover the early pandemic shock.

Negative candidate-minus-control loss differences mean improvement. The score is log(h)+y/h. Differences and the .005 minimum decrease are absolute natural-log-score units, not percentage returns or trading profits. The intervals below are envelopes of the declared HAC and block-bootstrap intervals.

| Control | Phase | Candidate loss | Control loss | Difference | 95% interval envelope | Conservative p |
|---|---|---:|---:|---:|---:|---:|
| Full controlled baseline | development | -9.2184835901 | -9.2186747497 | +1.911596084e-04 | [+8.991819258e-06, +3.793469264e-04] | 0.04095000 |
| Full controlled baseline | evaluation | -8.6210638388 | -8.6211259672 | +6.212846490e-05 | [-8.171108015e-05, +2.097897466e-04] | 0.39746000 |
| Historical mean | development | -9.2184835901 | -8.8554223132 | -3.630612770e-01 | [-5.217928275e-01, -2.301569942e-01] | 0.00007000 |
| Historical mean | evaluation | -8.6210638388 | -8.1769193571 | -4.441444817e-01 | [-8.115249799e-01, -1.245031821e-01] | 0.01054000 |

| Control | Conservative p across phases | Wave Holm p | Cumulative Holm p | Result |
|---|---:|---:|---:|---|
| Full controlled baseline | 0.39746000 | 0.39746000 | 1.00000000 | DOES_NOT_QUALIFY |
| Historical mean | 0.01054000 | 0.02108000 | 1.00000000 | DOES_NOT_QUALIFY |

Both controls must show at least a .005 improvement in both phases, improve in both fixed late slices, pass wave Holm2 at .05/(17×18), and pass cumulative Holm127 at .05. A favorable control, period or fitted coefficient alone does not qualify.

| Control | 2020–2022 difference | 2023–2025 difference | Development nominal MDE/effect | Evaluation nominal MDE/effect |
|---|---:|---:|---:|---:|
| Full controlled baseline | +5.846742917e-05 | +6.490162739e-05 | 0.051630 | 0.036437 |
| Historical mean | -5.398374422e-01 | -3.716589478e-01 | 37.994150 | 91.377790 |

The minimum detectable effect is an ordinary 5% HAC power diagnostic, not adjusted power or an equivalence test. Failure to qualify does not prove a zero predictive effect.

![Verified comparison intervals](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/profiled_quarter/comparison_intervals.png)

## Support and numerical verification

| Scored period | Observations | Quarter-end present | Quarter-end absent |
|---|---:|---:|---:|
| development | 983 | 43 | 940 |
| evaluation | 1,116 | 49 | 1,067 |
| 2020-01-02–2022-12-31 | 481 | 20 | 461 |
| 2023-01-01–2025-10-17 | 635 | 29 | 606 |

All monthly training support, civil-rank and conditional nonredundancy checks passed before optimization. The independent solver analytically eliminated the intercept, solved the strongly convex slope problem, then recomputed all coordinates of the original gradient. The original full-gradient, coefficient, normalized-prediction and scalar-quarter gates remained mandatory. Every application was checked; the protocol also checks retained unscored fits when present.

The complete registered source-document closure and both original successful upstream proof records were independently reconstructed. Source completeness does not establish historical release latency or immutable vintages. Existing archival revisions, original-plan coverage, early back-calculated VIX9D, nominal calendars and the daily-risk measurement limits remain.

## Preparation and accounting

Before freezing, **1,775 repository tests** passed. The run manifest records **267 code files**, **1,195 inputs** and **290 preserved artifacts**. All prior frozen artifacts remain unchanged.

The ledger has 129 events: 125 inherited comparisons, two new registrations and two verified evaluated outcomes. The cumulative family contains 127 hypotheses, including both failed predecessors. The independent inference reconstruction checked four phase comparisons and twelve bootstrap runs.

The seventeen-wave sequence contains **151,107 generated forecast rows**, including **6,297 unverified rows** retained from wave16. Completed independently verified waves account for **144,810 newly generated rows**. The preceding source-packaging failure generated zero. There are **73 new registrations** beyond the initial 54 enumerated hypotheses. These are sequence counts, not a repository-lifetime census.

These results use adaptively reused historical data and do not provide untouched confirmation, a causal quarter-end mechanism or demonstrated trading profitability.

- [Protocol](/Users/byrons/code/trading-vol/ndx-vol-experiment/profiled_quarter.yaml)
- [Independent verification](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/profiled_quarter/verification.json)
- [Metrics](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/profiled_quarter/metrics.json)
- [Solver design](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/profiled_quarter/SOLVER_DESIGN.md)
- [Independent pre-run review](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/profiled_quarter/DESIGN_REVIEW.md)
- [Previous numerical failure](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/civil_quarter_replay/SUMMARY.md)

The next unregistered proposal tests whether the previous close’s position inside its high–low range adds information about next-session risk exceeding twice its prior 22-session mean. It fixes one feature and one event definition, with conditional and recent-frequency controls. Its arithmetic contract and tests must be finalized before inspecting event counts or fitting. See the [next research direction](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/profiled_quarter/NEXT_RESEARCH_DIRECTION.md).
