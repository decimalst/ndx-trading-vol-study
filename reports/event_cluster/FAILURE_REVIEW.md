# Event-cluster failure-accounting review

**Status: FAILURE_ACCOUNTING_REVIEWED. The research result remains UNEVALUABLE and numerical verification remains FAILED.** This review confirms the recorded failure, complete comparison accounting and preservation identities. It is not a successful independent model verification, a null result or an authorization to repair/retry the frozen run.

The [actual traceback](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/event_cluster/verification_run.log) identifies the full application-state `feature_cutoff_date` comparison: saved `datetime64[us]` versus reconstructed `datetime64[ms]`. The exact-frame assertion fails at that representation difference. It does not prove that every date value or other model output is correct. Reading the frozen control flow confirms that this assertion precedes the loop calling `verify_stages`; therefore no independent new-stage solve completed, and the subsequent independent inference was not reached. No optimizer, producer, whole verifier or numerical diagnostic was rerun for this review.

The [canonical metrics](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/event_cluster/metrics.json) and failure JSON are identical. They declare whole-wave UNEVALUABLE, no leads and three retained comparisons: clustering against baseline, nuisance and recent frequency. Each has empty phase results and all three canonical p-values equal to1. The verification record is FAILED. The human-readable result also describes the failure. Producer-evaluated rows remain an audit trail; they cannot override the canonical failed record.

| Reviewed item | Recorded count and qualification |
|---|---|
| Cumulative hypotheses |134:131 inherited plus3 new |
| Ledger |140 events in order:3 registered,131 inherited,3 evaluated,3 verification_failed |
| Saved forecast panel |9,832 rows; categorical model metadata shows2,458 per model |
| New forecasts |4,916 producer-generated rows across nuisance and cluster; independently unverified |
| Reused controls |4,916 rows across original baseline and recent frequency |
| Monthly schedules |118 reported by the producer run log; new fitted coefficients were not inspected |
| Other saved tables |2,459 state rows and4,226 full-calendar memory rows, counted from Parquet metadata only |
| Prefit test records |2,137 full repository tests and149 focused pre-run tests |
| Manifest hashes |2,321 matching entries:308 code,1,990 inputs,23 preserved |
| Freeze |308 matching Python pins, including291 unchanged earlier files, plus56 matching prefit artifacts |

Manifest counts are entries in their respective maps, not a claim of that many distinct files. The protocol, manifest, freeze, canonical records, logs, ledger and current private outputs are byte-bound in [failure_review.json](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/event_cluster/failure_review.json). Every current manifest/freeze pin matched. The full prefit log matches its frozen digest. Both earlier failed civil-quarter families still have their two inherited records each. This preserves the entire134-hypothesis family; no unsuccessful family was removed.

The unpublished metrics wrapper remains explicitly `UNPUBLISHED_DIAGNOSTIC_ONLY` and marked ineligible for inheritance or promotion. Its numerical score payload was not inspected. Fit files were hashed without decoding coefficients; forecast inspection was limited to categorical model names and Parquet row/schema metadata. No labels, probabilities, losses or new history values were examined. Current private output hashes establish the bytes captured by this review; they do not manufacture an absent pre-failure successful-output certificate.

The [final summary](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/event_cluster/SUMMARY.md) correctly distinguishes generated forecasts from independently verified results, describes the timestamp-unit failure without asserting that all values match, retains all three p=1 comparisons and all134 hypotheses, and makes no null, calibration or trading-profit claim. Its counts agree with the canonical records, producer log and non-score metadata. No material correction is required. The reviewed summary SHA256 is `b39c1a6af2b3699801c58546363c6fadf1d520ae50019521e2c66fc99ba110d6`.

Any future replay needs a separately specified and pretested date-representation contract before registration. This failed protocol, family, implementation and original outputs remain preserved. The present review proposes no coercion, tolerance relaxation, rerun or relabeling of this study.
