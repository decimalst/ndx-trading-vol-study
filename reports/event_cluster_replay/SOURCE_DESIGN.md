# Source admission for a separately registered event-cluster replay

This new wrapper admits the **exact failed wave20 outputs as unverified artifacts**, together with the unchanged original wave18 records inside their successful wave19 proof. It performs no historical producer run, target or memory construction, coefficient fitting, timestamp normalization or scoring. Admission is a source-identity check; it is not successful independent verification of the retained candidate forecasts.

Only this new module, its tests and this new report directory are changed. All 308 previously frozen Python files, wave20 protocol, source artifacts, reports, canonical three p=1 outcomes and failure history remain unchanged. The proposed new trial would inherit 134 hypotheses, including those three failed comparisons, and register three more for a cumulative family of 137. Source collection itself registers no hypothesis.

## Exact API and source roles

The [source module](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/event_cluster_replay_admission.py) exports:

```python
collect_input_pins(root=ROOT, expected=None) -> dict[str, str]
admit_upstream(root=ROOT, expected=None, registered_pins=None)
    -> (audit, original_loaded, retained)
```

Collection returns deterministic sorted repository-relative file paths and their SHA-256 hashes. It parses only checked proof/protocol/failure metadata; it does not decode Parquet, retained numerical fits/support, unpublished metrics or evaluated ledger payloads. Ordinary documentary counts are read from the already pinned failure review rather than computed from new numerical tables.

Admission first completes the failed-record closure and requires every closure member in the supplied new registration pins. It then calls frozen [event_cluster_admission.admit_upstream](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/event_cluster_admission.py), which admits the still-VERIFIED wave19/wave18 records without calling an old optimizer or verifier writer. Its returned audit must exactly equal the checked wave20 `upstream_admission.json`, both before numerical-table admission through the frozen metadata-only collector and after the original loader returns. `original_loaded` is the original loader's unchanged dictionary, including the original wave18 protocol, features, targets, forecasts, fits, states and support/audit records.

The third returned object has exactly these keys:

| Key | Checked source | Role |
|---|---|---|
| `forecasts` | `data/event_cluster/forecasts.parquet` | Retained unverified candidate rows and preserved controls |
| `states` | `data/event_cluster/states.parquet` | Retained unverified complete application states |
| `memory` | `data/event_cluster/memory.parquet` | Retained unverified full-calendar history table |
| `fits` | `data/event_cluster/fits.json` | Retained unverified fit records; required JSON list |
| `support` | `data/event_cluster/support_audit.json` | Retained unverified support object |
| `upstream_admission` | `data/event_cluster/upstream_admission.json` | Exact identity of the original successful-source admission |

Each retained artifact is decoded from its already checked byte buffer. Frames preserve values, index, columns, dtypes, datetime units and unknown masks. The source layer does not repair the old ms/us timestamp discrepancy. Any new temporal comparison contract and subsequent independent numerical reconstruction belong to the separately registered replay. No unverified field becomes a trusted predictive measurement solely because its bytes match the failure publication audit.

## Three literal anchors and complete closure

The production default anchors are:

| Repository-relative path | SHA-256 |
|---|---|
| `event_cluster.yaml` | `824a9beda7cf293568dd2d16bbd3b76a8235f359a78c764147d75731c22b734c` |
| `reports/event_cluster/publication_audit.json` | `0d901a938c6a88ff4dbf1bb68d114640a50279e38826be29c97ea167ca56be15` |
| `reports/event_cluster/failure_review.json` | `ae554fd105b5a48b50739af4a6e64c3a8895c3e94bb14fbeb0230490d88d3408` |

The anchor paths must match exactly. Explicit `expected` hash overrides exist for synthetic temporary trees and must also retain those exact paths. The new runner and independent entry must enforce their own literal registered anchor map for the actual study.

The publication audit binds the exact six retained output hashes, all report artifacts, original manifest, freeze, FAILED verification and canonical metrics/failure. The failure review independently supplies matching private-output/report hashes and binds its written review. Both maps enter one conflict-checked closure. The manifest and freeze protocol hashes and code maps must agree, along with publication counts and the frozen full-suite log hash. Every referenced `reports/<study>/manifest.json` is recursively followed through `code`, `inputs`, `preserved` and `existing_artifacts_sha256`, including files outside immediate output directories. The original successful admission's full closure is also merged and checked.

The shared frozen path/byte primitives reject absolute or noncanonical paths, traversal, symlink components, malformed hashes, conflicting declarations, missing files, hash changes, duplicate JSON keys and nonfinite JSON numbers. All decoding follows a successful hash check of the same payload. Existing superseded or conflicting preservation declarations are not silently replaced. This wrapper performs no archive scan or numerical source parser operation.

The independent replay entry will share this metadata-only traversal, explicitly disclose that boundary, and separately verify its literal anchors and full declared byte closure. New target/history/model reconstruction must remain independent of the producer; sharing a source-identity utility must not be presented as an independently implemented mathematical verifier.

## Identified failure and opaque diagnostics

The only accepted failed-wave20 record has canonical status `UNEVALUABLE`, `whole_wave_aborted=true`, no leads, exactly three rows against baseline/nuisance/recent-frequency, empty phases, verdict `UNEVALUABLE`, status `INVALID_RUN` and literal float 1.0 for all three p-value fields. The two canonical files must equal the exact expected record. The verification object must be FAILED with this exact error:

```text
Independent verification failed: AssertionError: every saved application state differs from independent reconstruction
```

The pinned failure review must identify column `feature_cutoff_date`, saved `datetime64[us]`, expected `datetime64[ms]`, and the exact full application-state comparison stage. It records zero completed independent new-stage solves, no independent inference and no claim that all other values are correct. A different failure, a success marker, changed error, altered unit/stage declaration or partial promotion is rejected. This is not a general option to ignore failed upstream verification.

The ledger is exactly 140 lines in the reviewed sequence: three registrations, 131 inherited, three evaluated, three verification-failed events. Registration and final failed rows are checked against exact expected dictionaries. The 131 inherited and three evaluated lines are compared as raw byte hashes against the ordered lists in the pinned failure review; their scored payloads are not parsed. This preserves the previous evaluated diagnostics without using them for new inference or promotion. `unpublished_scored_metrics.json` is a hash-only dependency and is never JSON-decoded.

The successful wave18 and wave19 canonical failure markers must remain absent before and after admission and the final rehash. The identified wave20 failure marker instead must remain present with its exact pinned bytes. It is freshly checked again after the final whole-closure rehash. The new runner and independent verifier must repeat their respective input identity and failure-state checks at their publication commit points; the source function alone cannot bind future filesystem changes after it returns.

## Deterministic audit schema

The audit has status `PINNED_FAILED_WAVE20_WITH_VERIFIED_WAVE19_WAVE18`; sorted `anchors`, `files`, `manifest_paths`, `manifest_groups`, `retained_output_hashes` and `report_artifact_hashes`; and metadata `counts` for files, manifests and retained outputs. `failed_wave_identity` binds the old protocol, manifest, freeze, verification, failure, publication audit and failure review hashes. `original_admission` is the exact complete frozen successful-source audit.

`failed_wave_counts` records published `combined_saved_forecasts`, `generated_candidate_forecasts`, `reused_control_forecasts`, `producer_reported_monthly_schedules`, `state_rows` and `memory_rows`, plus `completed_independently_verified_forecasts=0`, three old new hypotheses, cumulative 134 and the exact 140-event role map. These are retained documentary counts, not fresh numerical verification or newly measured support. They do not impose a hardcoded current empirical sample-size gate.

The audit explicitly declares `retained_outputs_numerically_verified=false`, `historical_models_refitted=false`, `source_values_reparsed=false`, `retained_outputs_transformed=false`, `loaded_from_checked_snapshots=true` and `unpublished_scored_payload_decoded=false`. There is no variable wall-clock time or absolute machine path in this canonical audit.

## Prewritten synthetic checks

The initial sixteen [tests](/Users/byrons/code/trading-vol/ndx-vol-experiment/tests/test_event_cluster_replay_admission.py) preceded implementation; [admission_initial_red.txt](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/event_cluster_replay/admission_initial_red.txt) records the expected absent-module failure. Their first implementation passed all sixteen. Four further contracts explicitly cover mixed retained timestamp units/unknowns, opaque middle-ledger handling, symlink/duplicate-key rejection and mutation of the required failed marker after the main rehash loop.

The last boundary test [failed before its correction](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/event_cluster_replay/admission_failure_boundary_red.txt): the old shared absence guard protected successful ancestors but did not freshly check the required failed-wave20 marker after the loop. Only the new wrapper was changed to rehash that marker at both boundaries. This preserves the exact accepted failed record; it does not alter any old failure, numerical objective, gate or tolerance.

All **20 synthetic tests passed in 2.376 seconds**, recorded in [admission_implementation_checks.txt](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/event_cluster_replay/admission_implementation_checks.txt); [scoped lint passed](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/event_cluster_replay/admission_lint.txt). Tests exercise exact output identity and dtypes, unchanged original records, no producer/old-writer calls, registration coverage, early original-audit mismatch, late failure markers, canonical p=1 typing, exact role-count schema, ledger final-row mutation, private dependencies, output/report tampering and checked-buffer-only table reads. All fixtures are synthetic temporary trees.

The actual old failure metadata/schema was inspected before these tests were written, including its typed role-count maps and ordered raw-line hash declarations. No real retained numerical artifact has been admitted or decoded by this new module, and no actual new closure count, event history, predictive support, fit or score is claimed here. Whole-pipeline transport tests, the full-suite freeze, registration and fresh independent verification remain separate requirements.
