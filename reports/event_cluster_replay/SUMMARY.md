# Event-clustering verification replay

**No new qualifying predictive signal.** Independent reconstruction completed, but the clustering candidate did not clear all registered improvement gates.

The question was whether adjacency among the last 22 mature SPX risk-event labels adds information beyond their count, latest event, simple recency and existing market forecasts. The candidate adds one scalar correction to a frozen matched count/recency model.

## Results

Negative differences mean lower Brier loss. Both phases must improve by at least 0.0005 against every control; both fixed later slices must also improve. The same fixed dependence checks and multiplicity adjustments apply.

| Control | Development Brier difference | Evaluation Brier difference | Wave Holm p | Cumulative Holm p | Complete gate |
|---|---:|---:|---:|---:|---|
| baseline | +0.00004441 | -0.00013997 | 0.59353500 | 1.00000000 | DOES_NOT_QUALIFY |
| nuisance | +0.00000089 | +0.00000249 | 0.66476000 | 1.00000000 | DOES_NOT_QUALIFY |
| recent_frequency | -0.01398088 | -0.01737155 | 0.00027750 | 0.01267250 | DOES_NOT_QUALIFY |

The direct test of the clustering component is against the matched nuisance model: adding adjacency slightly worsened loss in both periods. Against the original market baseline, development worsened and the evaluation improvement was only 0.000140, below the required 0.0005. The much larger improvement over simple recent frequency does not isolate any benefit from clustering and also fails the stricter wave-level significance gate. None of these results supports promoting the new component.

All controls use 2,458 common scored origins: 2016-01-04 through 2019-12-30 in development, and 2020-01-02 through 2025-10-17 in evaluation. There are 3 new comparisons and 137 cumulative hypotheses, including the original failed clustering family at p=1.

## What was verified

The new verifier checks 9,832 retained scored forecasts: 4,916 nuisance/clustering forecasts and 4,916 original controls. It reconstructs 2,459 complete applications, 118 monthly schedules and 236 new-stage optima. There were zero new producer fits and zero newly generated forecasts in this replay.

The runner first reconstructed all retained histories, cohorts, parameters and probabilities before computing inference. A separate guarded verifier repeated that reconstruction and independently recomputed the three comparisons. Shared source-metadata traversal is disclosed; the numerical checks use the separately implemented objectives and inference.

Before registration, all 2,224 repository tests passed. The freeze preserves 319 Python files, including all 308 earlier files unchanged. Focused tests ran again before the three registrations and historical numerical admission.

## Relationship to the original failure

Wave20 remains FAILED/UNEVALUABLE with all three hypotheses retained at p=1. Its files, parameters and forecasts were not changed. This separately registered replay accepts losslessly equivalent millisecond, microsecond or nanosecond representations only in seven declared application-state date columns. Exact dates, missing-value masks, all other dtypes, model mathematics, cohorts, thresholds and numerical tolerances remain required.

The original timestamp exception was therefore a verification representation issue, not permission to repair fitted models or replace observations. Successful replay certifies the retained output under the new declared comparison rule; it does not retroactively pass the failed study.

## Limits and evidence

This uses the same adaptively reused archival daily risk proxy. It is not an untouched holdout, causal identification of market self-excitation, proof of an orthogonal information source or a trading-profit test. Overlapping risk thresholds can induce persistence. Original source-vintage limitations remain; no new numerical observations from the protected period beginning 2025-11-03 were admitted.

- [Complete phase, slice, uncertainty and calibration estimates](metrics.json).
- [Independent reconstruction, inference and artifact proof](verification.json).
- [Prospective design](DESIGN.md) and [freeze record](freeze_record.json).
- [Original failed research record](../event_cluster/SUMMARY.md).
- [Comparison figure](comparison_intervals.png) and [exportable PDF](comparison_intervals.pdf).
