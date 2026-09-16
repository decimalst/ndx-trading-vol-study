# Frozen wave15 source dependency audit

The failed immutable source tree omitted **four required BLS extraction files**. All four exist in the original tree and their exact bytes match hashes contained in the pinned CPI/payroll ledgers. A fifth omitted file, the original four-example capture manifest, is a documentary dependency with its own exact hash in the pinned payroll ledger. It is not opened by the frozen admission scanner.

This is a bounded dependency audit, not a repaired run. No market features, targets, forecasts, scores or model fits were read or computed. Only this report and the [machine-readable inventory](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/civil_quarter/source_dependency_audit.json) were written.

## Exact omitted files

All five are under `data/source_discovery/bls_plan_capture/`. None is directly listed in any wave15 manifest section, in the wave8 input list, or in the **817-file** staging set actually specified by `reconstruct_calendar`. Every listed hash below matches the existing file bytes and a hash declaration in pinned metadata.

| File | Reader role | Exact SHA256 |
| --- | --- | --- |
| `capture_manifest.json` | Documentary capture manifest | `ab2a4d167d7e93e403af682dc30aca40cd05497816e626308fa7d8caa6f7c941` |
| `cpi_09112025.web-extract.txt` | Runtime extraction | `5c5880834bf10c136631c68dfd023f8276ee29e795ad49d4caed95d0646e521c` |
| `cpi_12152015.web-extract.txt` | Runtime extraction | `a29d4c780e626d86167de0584172b9e84c730cf2b91815372ebdba54dea40c00` |
| `empsit_09052025.web-extract.txt` | Runtime extraction | `6483aaebdd8a4e320e5ca433ceaea8a2a9dcca9fb08c90cab87f71f16b1c863e` |
| `empsit_12042015.web-extract.txt` | Runtime extraction | `6bacb96bc54d0242fe5a4c34a98a0b3c3b52d6216305e92a167240eddb2b5454` |

The first chronological missing source in the frozen CPI scan is `cpi_12152015.web-extract.txt`, matching the reported exception. Its ledger pin is `a263e318edf6a6233b44ac8c141a33c9389e7b90e131c0ea9e26678842b5256a`. The selected payroll ledger pin is `608384be1b94c0e0977251efd1c8bfafe8b7249171d49629c6148254e4c92b39`; its `artifact_sha256` map explicitly commits the omitted capture manifest. The JSON inventory records every path, hash, referring field and direct-manifest-to-ledger-to-document chain.

## Closure and integrity checks

The runtime readers require **394 unique extraction files**: **189 CPI**, **189 payroll** and **16 FOMC**. These correspond to 506 document-reference rows because eight annual meeting rows share each FOMC extraction. All 394 files exist in the original tree and match their declared hashes. Exactly four are missing from staging; all 16 FOMC extractions are present. One out-of-fence selected document per BLS ledger is retained as metadata but not opened by these bounded readers.

For the wider documentary closure, this audit enumerated all **389 pinned JSON metadata files** in the existing macro-plan corpus, including selected and retained prior ledgers, capture manifests and coverage metadata, then traversed the referenced four-example capture manifest. All 389 JSON byte hashes match the wave15 pins and all are included in staging. Structured local paths, basename extraction-file fields and path-to-hash maps identify **620 unique source files**. The only omissions are the five files listed above. Full path and declared-hash details are in the JSON.

One retained `vintage_admission_amendment.intermediate_186_SUPERSEDED.json` still points the pathname `ledger_verified.json` at its former intermediate digest `04a86b020b457e2d5692015281134698deb8f16d27b311928a8e254d652d3345`. The final ledger and current amendment identify the matching current digest `608384be1b94c0e0977251efd1c8bfafe8b7249171d49629c6148254e4c92b39`. This historical reference is recorded separately; it is not a missing file, a current selected-source mismatch, or an alternative hash to admit. Nothing was rewritten.

## Cause and implication

`verify_civil_quarter.reconstruct_calendar` copies the wave8 manifest's direct inputs plus its inherited metric paths into a temporary tree. The earlier macro scanner follows each bounded ledger record's `snapshot_path` and verifies `snapshot_sha256`; it checks excluded reissues and ambiguous records as well as admitted plans. The original source corpus reused the four exemplar texts outside the `macro_plans` directory. Direct directory/manifest enumeration therefore did not provide the complete source-reader closure.

The exact snapshot content was already cryptographically committed: changing any of these four texts without detection would require breaking its ledger-declared SHA256 relationship or replacing the pinned ledger. This commitment does not cause the text files to exist in the temporary tree. The capture-manifest bytes are likewise committed through the payroll ledger, but the scanner does not need that manifest to read the four selected snapshots.

A separately registered attempt can explicitly enumerate and hash-check the bounded reader closure before staging, with a synthetic fixture that places a referenced extraction outside the ledger directory. It should preserve the failed wave15 records and retain historical superseded declarations as provenance, not substitute old hashes into the active dependency map. This audit makes no change to the frozen model, evidence gates, source admissibility or study accounting.

The saved texts remain tool-rendered partial extracts with `raw_provider_bytes_sha256 = null`; byte integrity and dependency closure do not establish original provider bytes or an immutable historical archive. The failure provides no evidence for or against the quarter-end predictor and does not itself reclassify earlier verified studies.

Audit anchors: wave15 protocol `8b9fdabcbaf3eae80116cc1efa02e7d32372d9a628faf74996fceefe142da109`; wave15 manifest `212578e5418c5f10e8e4847bbebf335bc3dcbd7b0db63aac853a4902a5605156`; wave8 manifest `521deb0c053c3eb724f2cd84c4644688863ec263ecce62d00c8ea129580bb4c8`. No admission routine was rerun by this auditor; the separate independent agent owns the read-only failure reproduction.
