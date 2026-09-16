# Correction: the later historical period was accessed by regression tests

On September 9, 2026, before the Treasury forecasting experiment was registered or its market inputs decoded, inspection found that six existing tests load the processed market history, reconstruct HAR forecasts on the old clean period, or launch evaluation over that period. The configured old clean period starts November 3, 2025. The local processed history exists. The previous commodity experiment's saved `FULL_PRECHECK.log` records all six tests as passing at lines 1225–1233.

Therefore the blanket wording that numerical QQQ/SPX observations beginning November 3, 2025 “remain unopened” is not supported and is false for the regression-suite execution. Prior published aggregate results had also already exposed that historical period. The current inspection read code, file metadata and saved test outcomes; it did not decode those market observations again. This distinction does not restore untouched confirmation. No future research should claim that this historically evaluated period is pristine.

This is an additive correction to the saved source-preparation summary and earlier equivalent statements. Frozen protocols, source files, old regression tests, full-suite logs and prior result verdicts remain unchanged. The finding concerns validation-workflow access and the strength of an untouched-period claim; it is not evidence that the new Treasury feature builder uses later observations or that any prior signal qualifies.

The new Treasury read path still authenticates the original four source snapshots and decodes numerical rows only through October 20, 2025. Its historical results remain exploratory. True future confirmation requires observations whose outcomes have not already been evaluated, with a prospective protocol fixed before access.

Before any Treasury empirical access, the execution test gate now selects every discovered repository test except these six exact identities:

- `tests.test_methodology.TestDiagnosticOnlyQuarantine.test_qlike_series_refuses_the_clean_window`
- `tests.test_methodology.TestFrozenReportsUnchanged.test_default_evaluate_reproduces_the_frozen_reports`
- `tests.test_methodology.TestFrozenReportsUnchanged.test_corrected_run_writes_a_separate_file`
- `tests.test_methodology.TestEstimatorReconstructionAccuracy.test_reconstruction_matches_exact_smearing_on_the_har_family`
- `tests.test_methodology.TestEstimatorReconstructionAccuracy.test_reconstruction_table_matches_the_published_one`
- `tests.test_methodology.TestEstimatorReconstructionAccuracy.test_the_near_common_factor_claim_is_scoped_to_its_own_model_set`

The selection manifest saves all selected and quarantined IDs before execution, and fails if the six expected IDs are missing or any test identity is duplicated. The executed count, errors, failures and any ordinary skipped tests are reported separately. The research runner requires all selected tests to execute successfully without skips and preserves the six original test files' hashes. This will be described as a selected-suite pass with six disclosed quarantines, not an unqualified full-suite pass. The exclusion is for data-access scope, not a response to failing results or an empirical signal outcome.

The new selection helper was checked on invented test identities before use. Its first combined run exposed an accidental discovery of a fake fixture subclass as a real test; the fixture became a plain callable, preserving the selection assertions. The initial failure log and subsequent successful log are both retained. No historical experiment was registered during these checks.
