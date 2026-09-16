# Immutable issued-forecast admission

This module admits the completed, independently verified wave18 records for a prospective calibration experiment. It does not recreate raw market measurements, event labels, historical logits, prior model fits, calibration states or predictive scores. The next models must separately reconstruct each original application logit and strictly replay its saved probability before using a later mature label. Hash identity supplies the original-record boundary; it does not replace that causal replay.

## Tests before implementation

The initial 18 synthetic admission contracts were written before the new module. The initial import failed because `issued_calibration_admission` did not exist; [admission_initial_red.txt](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/issued_calibration/admission_initial_red.txt) preserves that failure. The first implementation passed all 18 tests. A later prefit fault injection showed that a canonical failure marker created during the final rehash could escape a check placed only before that loop. The [regression red](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/issued_calibration/admission_marker_regression_red.txt) was retained before adding a second absence check after the loop. All **19 tests** now pass in 0.293 seconds and scoped Ruff checks pass. [The final test log](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/issued_calibration/admission_implementation_checks.txt) records the result.

The fixtures create their own tiny parquet and JSON records, including an older manifest whose declared source lies outside the immediate study directory, and an older preservation-only manifest. Tests cover every verified output being modified, checks before JSON/parquet decoding, source conflicts, missing files, recursive preservation, malformed anchors/hashes, noncanonical paths, symlinks, duplicate/nonfinite JSON, status/backlink inconsistencies, missing registration coverage, deterministic read-only operation, a canonical failure marker and a mutation during decoding. A changed original cannot be substituted for the saved checked bytes; the final rehash still rejects the concurrent change.

After these tests, the parent explicitly authorized a metadata-only check of the actual archive. `collect_input_pins` admitted **1,866 files** without a missing path, conflict or hash mismatch. Parquet decoding was patched to raise during this check, and no new model, event count, calibration or score was computed. The older `existing_artifacts_sha256` declarations passed unchanged. This count is a measured closure diagnostic, not a hardcoded admission or support gate.

## Exact interfaces and returned records

`collect_input_pins(root=ROOT, expected=None)` returns a deterministic sorted map from repository-relative paths to SHA-256. It checks metadata/file identity without decoding output values. The runner can use it to build the new registration input inventory.

`admit_upstream(root=ROOT, expected=None, registered_pins=None)` returns `(audit, loaded)`. When registration pins are supplied, every closure member must have the identical registered hash before any output value is decoded; additional new-study input pins are allowed. The returned dictionary has exactly these keys:

- `protocol`: the original wave18 YAML, decoded from its anchored bytes.
- `features`, `targets`, `forecasts`, `states`: original parquet tables decoded from their checked byte buffers.
- `fits`: the original fit list; `support_audit`, `source_audit`, `upstream_admission`: original JSON objects.
- `metrics`, `ledger`: the original verified metrics and complete evaluated trial ledger.

The caller receives the original full bounded feature and target calendars, all original application states, and all monthly fit/application records. The admission layer does not derive a different complete-case cohort, rescore old predictions or fabricate issued probabilities for dates without an original application. Loading a target table does not authorize using an unmatured query label; the new model schedule must enforce the original preceding-session cutoff.

## Identity and dependency closure

`expected` is exactly a seven-path anchor map. The default pins are the frozen `range_alert.yaml`, `manifest.json`, `freeze_record.json`, `verification.json`, `publication_audit.json`, `publication_review.json` and `NEXT_RESEARCH_DIRECTION.md`; the latter six reside under `reports/range_alert/`. Synthetic tests may supply different hashes for those same paths. Production uses the literal map in the new protocol, whose full hash is independently frozen by the runner.

All seven anchor hashes are checked before their metadata is decoded. Their protocol, verifier, manifest, freeze, success status, published summary and verified-output links must agree. The source identities are explicit raw-byte SHA-256 values. JSON duplicate keys, nonfinite values and floating-point overflow are rejected. Cross-record identities preserve types so a boolean cannot replace an integer or a visual-review flag.

The exact ten verified outputs are the eight files under `data/range_alert/` named `upstream_admission.json`, `source_audit.json`, `support_audit.json`, `features.parquet`, `targets.parquet`, `forecasts.parquet`, `states.parquet`, `fits.json`, plus the report’s `metrics.json` and `trial_ledger.jsonl`. Both the original independent verification and publication audit must declare the same complete map.

The closure recursively follows already-pinned `reports/<study>/manifest.json` files. Each optional `code`, `inputs`, `preserved` and `existing_artifacts_sha256` map contributes raw-byte references; absent legacy groups contribute zero entries. The older `sources.*.bounded_content_sha256` fields are numerical-content fingerprints, so they are retained as opaque provenance and never misrepresented as raw-file hashes. No raw source parser is called to recompute them.

The closure also includes the current freeze’s code, prefit designs and full test log, the publication audit’s report-artifact map, and the publication review’s document/figure maps. Conflicting declarations fail; an old hash is never silently superseded. Paths must be canonical, relative POSIX paths without traversal, backslashes or symlink components. Missing, changed and escaping files fail. No recursive directory inventory invents replacement sources.

The entire closure is hashed before any output values are decoded. Metadata and output payloads are retained from their checked reads; opaque older source files are hashed without parsing or retaining their full contents. Parquet decoding uses `BytesIO` of those exact buffers. Original verified table/fit/application counts and the 127 inherited, two registered, two evaluated ledger counts must agree. All pins are rehashed before returning, and canonical failure-marker absence is checked both before and after that final loop. The new runner/verifier must additionally repeat this absence check at their own publication boundary after candidate work. No previous file is written.

## Canonical audit and interpretation

The deterministic audit contains `status=PINNED_WAVE18_VERIFIED_OUTPUTS`, sorted `anchors` and `files`, `manifest_paths`, `manifest_groups` with all four optional group counts, the exact `verified_output_hashes`, expanded `publication_artifact_hashes`, closure `counts`, `upstream_identity`, and `upstream_verified_counts`. It states `historical_models_refitted=false`, `source_values_reparsed=false` and `loaded_from_checked_snapshots=true`. It contains no current timestamp or machine-dependent absolute paths.

This is admission of unchanged successful proof and issued-output identities, not a rerun of every older study. The original Yahoo/Cboe archive, uncertain historical publication latency, revision history, back-calculated early VIX9D and repeatedly reused outcome history keep their earlier limitations. Finite original logits must be recovered from the original saved coefficient/transform/application record, not by inverting a rounded saved probability or applying a later fit. Empty calibration history should preserve the admitted saved baseline probability exactly. Those later replay and calibration checks belong to the new model and independent verifier, not this source module.
