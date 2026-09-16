# Issued-error memory for SPX risk alerts

**No candidate passes all registered predictive gates.** The issued-error memory improves average calibration in both phases, but does not establish reliable incremental forecasting accuracy. Against the unchanged conditional baseline, Brier loss is worse by 0.0002606504 in development and better by 0.0002514406 in evaluation. The evaluation improvement is below the required 0.0005 absolute decrease, and the conservative across-phase p-value is 0.794055. Both evaluation slices improve, but their favorable direction cannot override the failed phase/effect and significance gates.

In evaluation, the average predicted event probability moves from 12.34% to 13.91%, against 14.42% observed. This reduces average underprediction, but it is a descriptive calibration-in-the-large result, not proof of conditional calibration or a new orthogonal information channel. The development phase illustrates the distinction: average calibration improves while mean Brier loss worsens slightly.

The calibrated model scores better than recent event frequency, as the original baseline already did descriptively. That comparison has wave Holm p=0.00049, above the required 0.0001315789474, although cumulative Holm p=0.03087 is below 0.05. Both controls and both multiplicity gates were required. No new contrast promotes the existing baseline, and no half-life, penalty, threshold or solver was changed after seeing the result.

The target remains next-session daily OHLC risk exceeding twice the mean of the 22 sessions ending at the previous-session cutoff. The candidate keeps the original monthly model and adjusts only its intercept, using mature outcomes paired with genuinely issued earlier baseline forecasts. A 63-session half-life and fixed .01 ridge determine the correction. Memory starts empty and advances through feature gaps and missing labels; no calibration warm-up or phase reset is used.

This run independently verified **2,458 new candidate forecasts**, **4,916 unchanged control forecasts**, **7,374 combined rows**, **2,458 common scored origins** and **2,459 original applications**. There are **zero new monthly baseline fits**. All original unscored applications remain in the memory timing audit.

Actual scored development: **2016-01-04–2019-12-30**; evaluation: **2020-01-02–2025-10-17**. The same original controls, labels, cohort and training metadata were preserved.

## Verified Brier comparisons

Negative differences mean lower squared probability error. The required .0005 decrease is an absolute Brier-score gain, not a probability-point change or return. Intervals are nominal 95% HAC/block-bootstrap envelopes.

| Control | Phase | Candidate loss | Control loss | Difference | 95% interval envelope | Conservative p |
|---|---|---:|---:|---:|---:|---:|
| Conditional baseline | development | 0.0980974012 | 0.0978367509 | +2.606503847e-04 | [-1.570871941e-03, +1.952306123e-03] | 0.76296500 |
| Conditional baseline | evaluation | 0.1071117250 | 0.1073631656 | -2.514406198e-04 | [-2.304597531e-03, +1.480968592e-03] | 0.79405500 |
| Recent event frequency | development | 0.0980974012 | 0.1118620336 | -1.376463239e-02 | [-1.915343518e-02, -8.716420702e-03] | 0.00000500 |
| Recent event frequency | evaluation | 0.1071117250 | 0.1245947439 | -1.748301892e-02 | [-2.686256344e-02, -8.591809052e-03] | 0.00024500 |

| Control | Conservative p across phases | Wave Holm p | Cumulative Holm p | Result |
|---|---:|---:|---:|---|
| Conditional baseline | 0.79405500 | 0.79405500 | 1.00000000 | DOES_NOT_QUALIFY |
| Recent event frequency | 0.00024500 | 0.00049000 | 0.03087000 | DOES_NOT_QUALIFY |

Both controls must improve by at least .0005 in both phases, improve in both later slices, pass wave Holm2 below **.05/(19×20)=.0001315789474**, and pass cumulative Holm131 below .05. A favorable comparison or adjusted p alone does not qualify. No new baseline-versus-frequency hypothesis is registered.

| Control | 2020–2022 difference | 2023–2025 difference | Development nominal MDE/effect | Evaluation nominal MDE/effect |
|---|---:|---:|---:|---:|
| Conditional baseline | -4.223679119e-04 | -6.683914429e-05 | 2.860642 | 3.820792 |
| Recent event frequency | -2.183611666e-02 | -1.278167335e-02 | 13.692654 | 25.417839 |

Nominal 80% minimum detectable effects use ordinary 5% HAC power calculations. They are not multiplicity-adjusted power or equivalence tests.

![Verified Brier comparisons](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/issued_calibration/comparison_intervals.png)

## Support and descriptive calibration

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
| development | Issued-error calibration | 1,002 | 0.12674651 | 0.12721588 | +0.00046937 | 0.0980974012 |
| evaluation | Conditional baseline | 1,456 | 0.14423077 | 0.12340816 | -0.02082261 | 0.1073631656 |
| evaluation | Recent event frequency | 1,456 | 0.14423077 | 0.14106570 | -0.00316507 | 0.1245947439 |
| evaluation | Issued-error calibration | 1,456 | 0.14423077 | 0.13914801 | -0.00508276 | 0.1071117250 |

These phase averages describe calibration in the large. They do not measure full conditional calibration and were never used as a retrospective correction.

## Verification and scope

All **1,988 repository tests passed before freeze**. The manifest records 291 code files, 1,872 input references and 24 preserved artifacts. Prior code and evidence remain unchanged. The source admission closure contains 1,866 pinned files, validated without changing original sources.

Independent verification reconstructs every original application logit, full-calendar arrival category, eligible weighted record, scalar optimum, probability, Brier score, paired difference, inference result and ledger event. Producer Brent and independent bisection retain their prospective budgets and gradient gates. There is no empirical solver fallback, arithmetic-gate relaxation or removed forecast. Exact hashes bind the checked output snapshots.

The ledger retains **129 inherited hypotheses, two new registrations and two evaluated outcomes: 133 events and 131 cumulative hypotheses**. Both earlier failed quarter-end attempts remain p1 unevaluable records, not statistical null results. Original control forecasts and rescored rows are not newly generated rows.

Across the nineteen-wave sequence, **160,939 forecast rows were newly generated**, including 6,297 unverified rows preserved from the failed wave 16 numerical verification. Completed verified waves account for **154,642 newly generated rows**. There are 77 new hypotheses beyond the initial 54 enumerated comparisons. These are sequence counts, not a repository-lifetime census.

Issued forecasts mean immutable earlier archival walk-forward records, not contemporaneously live-stamped production forecasts. This is an adaptively chosen calibration rule on reused archival information, with substantial overlap with earlier memory experiments. The daily OHLC proxy is not integrated variance or a trading payoff. Current-vintage Yahoo/Cboe inputs have unverified original release latency and revision history; early VIX9D values are back-calculated, and the vendor calendar is not independently certified as complete. Causal timing and multiplicity accounting do not create untouched confirmation or demonstrated trading profitability.

- [Protocol](/Users/byrons/code/trading-vol/ndx-vol-experiment/issued_calibration.yaml)
- [Independent verification](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/issued_calibration/verification.json)
- [Metrics](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/issued_calibration/metrics.json)
- [Design](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/issued_calibration/DESIGN.md)
- [Independent prefit review](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/issued_calibration/DESIGN_REVIEW.md)
- [Prior completed study](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/range_alert/SUMMARY.md)

The [next proposal](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/issued_calibration/NEXT_RESEARCH_DIRECTION.md) asks whether an early-session price shock adds information about risk strictly after a fixed observation cutoff. It is unregistered and requires a separate timestamped-data admission review; daily OHLC alone cannot support the proposed temporal separation. No new measurements or fits have been run for it.
