# Event-clustering experiment: independent verification failed

**This experiment is unevaluable. It supplies no qualifying signal and no statistical null result.** The historical runner produced forecasts, but the independent verifier rejected a timestamp storage-unit mismatch before independently solving the new fitted stages and recomputing their inference. All three comparisons remain in the cumulative family with p=1.

The failed comparison was the `feature_cutoff_date` column of the full application-state table: the saved version uses `datetime64[us]`, while the independently reconstructed version uses `datetime64[ms]`. The frozen exact-frame check rejects that difference. The traceback identifies a representation mismatch; it does not, by itself, establish that all date values or model outputs are otherwise correct.

## What was attempted

The candidate tested whether adjacent past SPX risk events improve prediction beyond their count, latest event, simple linear recency and the existing market forecast. It adds one coefficient on the adjacent-pair fraction among the last 22 mature event labels. A matched count/recency model is fitted first and then frozen. The original baseline and recent-frequency forecasts are preserved controls.

The run generated **4,916 new scored forecasts** across the nuisance and clustering models, alongside **4,916 reused control forecasts**, for 9,832 saved panel rows. The producer reports 118 monthly schedules, each with a new nuisance fit and scalar clustering fit. Those new model stages are **not independently verified**. Detailed outputs remain preserved in the ignored private data directory.

| Registered comparison | Canonical status | Retained p-value |
|---|---|---:|
| Clustering versus original market forecast | UNEVALUABLE | 1 |
| Clustering versus matched count/recency model | UNEVALUABLE | 1 |
| Clustering versus original recent frequency | UNEVALUABLE | 1 |

There are **134 cumulative hypotheses**: 131 inherited and three new. The ledger preserves 140 events: three registrations, 131 inherited records, three producer evaluations and three subsequent verification failures. The evaluated records remain an audit trail; canonical failed metrics determine the current outcome. The earlier failed families remain retained as well.

## Verification and preservation

All **2,137 repository tests passed before freeze**, followed by all 149 focused pre-run tests. The freeze pins 308 Python files, including 291 unchanged earlier files, and 56 prospective documents and test-evidence artifacts. Passing those synthetic tests did not establish compatibility of every historical representation; the actual independent reconstruction caught the mismatch.

A separate metadata-schema defect was corrected before freeze or registration: the old verification proof stores ledger counts by event type, whereas the initial synthetic fixture had assumed one total. A prewritten regression established the failure, the new loader was corrected to require the exact frozen map, and the complete test suite was rerun. The failed check and earlier green suite remain preserved. No new historical feature, support count, fit or score was available during that correction.

After the actual verification failure, the frozen implementation, protocol, thresholds and data were left unchanged. The failure guard published UNEVALUABLE/no-lead metrics, a failure record and FAILED verification, appended all three failure outcomes, and retained scored metrics as unpublished diagnostics. No candidate score or chart is promoted from those diagnostics. There was no empirical retry or tolerance change.

The experiment uses an archived daily SPX risk proxy, with its original prior-session feature cutoff and label-maturity rules. Its overlapping event threshold can itself induce persistence; the design would support only model-relative predictive evidence if eventually verified. Existing historical source-vintage limits and repeated use of research history remain. The protected period beginning 2025-11-03 was not admitted.

## Evidence and next step

- [Exact failure traceback](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/event_cluster/verification_run.log).
- [Independent failure-accounting review](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/event_cluster/FAILURE_REVIEW.md).
- [Canonical metrics](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/event_cluster/metrics.json), [failed verification](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/event_cluster/verification.json), and [retained ledger](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/event_cluster/trial_ledger.jsonl).
- [Frozen protocol](/Users/byrons/code/trading-vol/ndx-vol-experiment/event_cluster.yaml), [design](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/event_cluster/DESIGN.md), and [freeze record](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/event_cluster/freeze_record.json).
- [Earlier supplied-paper and model results](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/model_memory_study/SUMMARY.md).

The event-clustering question remains unanswered by a completed independent verification. A future replay would need a separately specified, tested date-representation contract before registration, while preserving this failed family and its original outputs. A [date-only diagnosis](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/event_cluster/TIMESTAMP_DIAGNOSIS.md) records the representation issue; it does not complete model or statistical verification or relabel the failed run.
