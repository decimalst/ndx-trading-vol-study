# Closing-range location and SPX risk alerts

**No candidate passes all registered predictive gates.** The previous close's position within its daily range did not improve on the conditional baseline: average Brier loss was slightly worse in both phases. It was worse in the 2020–2022 slice and better in the 2023–2025 slice, so the direction was not stable.

The conditional forecasts scored substantially better than recent event frequency on this sample. That improvement does not establish value from the added location feature: the baseline alone had slightly lower loss than the candidate in both phases. The candidate-versus-frequency comparison also failed its wave-adjusted threshold, despite passing the cumulative threshold. Both comparisons and both thresholds were required.

The target is **next-session daily OHLC risk exceeding twice the mean of the 22 sessions ending at the previous-session cutoff**. It is not doubling relative to the immediately preceding day, measured high-frequency variance, a signed crash, or a trading payoff. The input is squared closing-location extremity inside the previous daily high–low range. One fitted slope adjusts the fixed fitted baseline logit; the second control continuously updates recent event frequency with a 63-session half-life.

The run independently verified **7,374 forecasts**, **118 monthly fits**, **2,458 common scored origins**, and **2,459 applications**. The unscored application was 2019-12-31, retained in prediction and state checks; its next-session label crossed the development-period maturity boundary.

Actual scored development runs from **2016-01-04 through 2019-12-30**; evaluation runs from **2020-01-02 through 2025-10-17**. These dates describe retained vendor-calendar observations, with common feature and label requirements.

## Verified comparisons

Negative candidate-minus-control Brier differences mean improvement. Brier loss is squared probability error. The registered 0.0005 minimum decrease is an absolute score difference, not a percentage return or a probability-point gain. Intervals are envelopes of the declared HAC and block-bootstrap intervals.

| Control | Phase | Candidate loss | Control loss | Difference | 95% interval envelope | Conservative p |
|---|---|---:|---:|---:|---:|---:|
| Conditional baseline | development | 0.0978393667 | 0.0978367509 | +2.615825855e-06 | [-1.894418138e-04, +1.768284611e-04] | 0.97899000 |
| Conditional baseline | evaluation | 0.1074079892 | 0.1073631656 | +4.482359986e-05 | [-6.920996013e-05, +1.606052583e-04] | 0.44104809 |
| Recent event frequency | development | 0.0978393667 | 0.1118620336 | -1.402266695e-02 | [-1.944301614e-02, -8.986664925e-03] | 0.00000500 |
| Recent event frequency | evaluation | 0.1074079892 | 0.1245947439 | -1.718675470e-02 | [-2.570475824e-02, -8.967306274e-03] | 0.00010500 |

| Control | Conservative p across phases | Wave Holm p | Cumulative Holm p | Result |
|---|---:|---:|---:|---|
| Conditional baseline | 0.97899000 | 0.97899000 | 1.00000000 | DOES_NOT_QUALIFY |
| Recent event frequency | 0.00010500 | 0.00021000 | 0.01354500 | DOES_NOT_QUALIFY |

Both controls must improve by at least 0.0005 in both phases, improve in both fixed later slices, pass wave Holm2 at **0.05/(18×19) = 0.0001461988304**, and pass cumulative Holm129 at 0.05. A favorable control, subperiod or cumulative p-value alone does not qualify.

| Control | 2020–2022 difference | 2023–2025 difference | Development nominal MDE/effect | Evaluation nominal MDE/effect |
|---|---:|---:|---:|---:|
| Conditional baseline | +1.589370852e-04 | -7.841896429e-05 | 0.498032 | 0.325995 |
| Recent event frequency | -2.125481166e-02 | -1.279325317e-02 | 13.380633 | 23.497434 |

Nominal minimum detectable effects use ordinary 5% HAC power calculations. They are not multiplicity-adjusted power, an equivalence test, or evidence that the true location effect is exactly zero.

![Verified Brier comparison intervals](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/range_alert/comparison_intervals.png)

## Support and descriptive probability checks

| Scored period | Observations | Events | Nonevents |
|---|---:|---:|---:|
| development | 1,002 | 127 | 875 |
| evaluation | 1,456 | 210 | 1,246 |
| 2020-01-02–2022-12-31 | 756 | 117 | 639 |
| 2023-01-01–2025-10-17 | 700 | 93 | 607 |

| Phase | Model | Observations | Event frequency | Mean predicted probability | Prediction minus frequency | Brier loss |
|---|---|---:|---:|---:|---:|---:|
| development | Conditional baseline | 1,002 | 0.12674651 | 0.12400369 | -0.00274282 | 0.0978367509 |
| development | Recent event frequency | 1,002 | 0.12674651 | 0.13100538 | +0.00425887 | 0.1118620336 |
| development | Location candidate | 1,002 | 0.12674651 | 0.12433020 | -0.00241631 | 0.0978393667 |
| evaluation | Conditional baseline | 1,456 | 0.14423077 | 0.12340816 | -0.02082261 | 0.1073631656 |
| evaluation | Recent event frequency | 1,456 | 0.14423077 | 0.14106570 | -0.00316507 | 0.1245947439 |
| evaluation | Location candidate | 1,456 | 0.14423077 | 0.12339128 | -0.02083949 | 0.1074079892 |

These are calibration-in-the-large summaries, not conditional calibration tests. In evaluation the conditional baseline predicted an average event probability of 12.34%, against 14.42% observed. Its lower Brier loss than recent frequency does not imply that its average probability was better calibrated. No recalibration was fitted after observing this discrepancy.

## Preparation, verification and accounting

All **1,888 repository tests passed before freeze**. The manifest records **279 code files**, **1,277 inputs**, and **269 preserved artifacts**. All 1,774 prior unique pins checked before freeze remained unchanged.

Independent verification reconstructed all 118 monthly baseline fits and 118 scalar fits, every issued application probability and frequency state, 7,374 primitive Brier scores, 4,916 paired differences, four phase comparisons, twelve bootstrap runs of 199,999 draws, and six descriptive calibration rows. It saved exact hashes for all ten checked output snapshots. No optimizer retry or numerical-gate relaxation was used.

The ledger contains **131 events: 127 inherited comparisons, two registrations, and two verified evaluated outcomes**. The cumulative family contains 129 hypotheses, including both earlier failed quarter-end attempts. Those failures retain their status and p=1 records; they are not reclassified as statistical null results.

Across the eighteen-wave sequence, **158,481 forecast rows were newly generated**, including **6,297 unverified rows** retained from the failed numerical verification in wave16. Completed independently verified waves account for **152,184 newly generated rows**. There are **75 new registrations** beyond the initial 54 enumerated hypotheses. Reused or rescored forecast rows are not counted again. These are sequence counts, not a repository-lifetime census.

The raw inputs are archival Yahoo/Cboe extracts. Original release latency and immutable vintages are not established, early VIX9D values are back-calculated, and the observed vendor calendar is not an independently certified exchange calendar. The daily risk proxy, missing observations and reused historical sample remain limitations. This result provides neither untouched confirmation nor demonstrated trading profitability.

- [Protocol](/Users/byrons/code/trading-vol/ndx-vol-experiment/range_alert.yaml)
- [Independent verification](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/range_alert/verification.json)
- [Metrics](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/range_alert/metrics.json)
- [Study design](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/range_alert/DESIGN.md)
- [Independent prefit review](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/range_alert/DESIGN_REVIEW.md)
- [Prior completed study](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/profiled_quarter/SUMMARY.md)

The next [unregistered experiment](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/range_alert/NEXT_RESEARCH_DIRECTION.md) tests a single probability adjustment informed by mature errors from previously issued baseline forecasts. It keeps both controls and the same event definition. This is an adaptive follow-up selected after these results; it has not been run or shown to improve forecasting.
