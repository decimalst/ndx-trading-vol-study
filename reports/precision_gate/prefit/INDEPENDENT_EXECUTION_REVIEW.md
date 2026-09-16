# Independent prefit execution review

Status: **NO_REMAINING_BLOCKER_IN_BOUNDED_PREFIT_REVIEW**. The new precision-gate execution is ready for the root agent's selected-suite test run and prospective freeze. This is a source and generated-evidence review before empirical execution; it is not a certificate for a completed historical run or a predictive result.

Reviewer: new_study_tests. The reviewer independently implemented the feature/forecast verifier and inspected the new protocol and actual `run` wiring. A separate bounded reading by replay_entry_finish identified the report-binding issue described below. No market arrays, historical fits, empirical scores, or new source values were read for this review. No historical pipeline, full test suite, calibration, or bootstrap was rerun. Existing pinned files were not edited.

The fixed protocol separates issuance beginning 2010-01-04 from development 2016–2019 and evaluation 2020–2025-10-20. The ten-field configuration projection matches the producer and independent verifier: the first common 12-feature query in each month schedules fitting without its target; training labels must mature by the preceding original calendar session. Insufficient pre-development expert history produces explicit warmup; insufficient development/evaluation expert history fails. Both gates use the same actually issued, mature expert records in the last 1,260 original calendar positions. Cold starts remain issued and scored. The wrapper supplies only the established market sources and the 12-feature baseline plus realized-variance target input. No Treasury auction quantity or Z52 source-clock assumption enters this model experiment.

The preserved expert and gate tolerances match the prewritten independent verifier contract. The independent path reconstructs market features and five-session targets, separate weighted/unweighted QR expert fits and Duan smearing, issued-history membership, both convex gate optima with SLSQP, the saved coefficients' objective and stationarity, and all scored and unscored outputs. Iteration counts are checked for type and bounds rather than falsely reproduced across solvers. Saved-coefficient forecast reconstruction keeps the tight forecast tolerance; the separate optimizer's coefficient tolerance does not relax that check.

The wrapper verifies the protocol, complete source/test inventory, inherited input/preservation maps, and exact selected-test receipts before execution. The selection retains the declared six historical-replay quarantines; it does not claim an unqualified full-suite pass. The prior 146 comparison dictionaries are authenticated and copied unchanged. Three new registrations and all 146 inherited rows are durably written before the market reader callback. The three fixed controls are base, adaptive, and constant, each compared with contextual; a missing or reordered comparison invalidates the entire attempt. A completed ordinary journal has 152 rows: three registrations, 146 inheritances, and three evaluations. Cumulative multiplicity is 149 comparisons. This is a prospective expected count, not an observed run count.

Both independent verifiers run before completion. The scorer retains cold starts, checks the prescribed daily and fitted-gate support, and uses the original full phase calendars and global modulo-five offsets. The three-contrast and cumulative Holm gates, phase effects, offsets, and evaluation slices agree with the fixed protocol. Failure of any required comparison prevents a lead. Numerical, support, verification, preservation, or publication failure records all three new comparisons as unevaluable with probability one and preserves provisional evidence. Generic monthly expert/gate failures retain earlier issued forecasts and audits through `PipelineExecutionError`; successful-output mathematics is unchanged.

Three concrete preservation issues were corrected in the new runner before empirical access:

1. The original freeze payload is hashed before JSON decoding and the same signature is checked at every completion boundary. `FREEZE_BINDING_RED.log` records the prewritten missing-helper failure; `EXECUTION_BINDING_GREEN.log` records the subsequent passing regression.
2. Every one of the seven data outputs is pinned immediately after its first save. The exact inventory and original signatures are checked before verification, before its certificate, around metrics publication, and before/after terminal publication. This also detects feature or target drift while models are running. `OUTPUT_BINDING_RED.log` records the prewritten missing-helper failure; the same binding GREEN log covers its correction. The final actual-run wiring was read in addition to inspecting those helper tests.
3. Reports previously could be rehashed after a mutation and thereby receive newly accepted terminal hashes. `REPORT_BINDING_RED.log` reproduces that actual incorrect COMPLETED result after an injected metrics mutation. Reports now bind the intended JSON serialization at first write, while the journal binds its original payload and every append against the previously checked prefix. The manifest, forecast/score/combined verification reports, metrics, journal and terminal are checked against their expected bytes at completion boundaries. The JSON and journal serializers were compared with their frozen write helpers. The final 13-test log is `EXECUTION_REPORT_BINDING_GREEN.log` (0.917 seconds, all passing). This correction changes preservation enforcement and no scientific parameter.

Generated evidence inspected: 18 independent forecast-verifier tests passed in 74.036 seconds; 15 expert/pipeline tests passed in 0.573 seconds; 10 scoring tests passed in 0.998 seconds and five independent scoring tests passed in 1.838 seconds; the final 13 execution/protocol tests passed as above. Scoped formatting/lint evidence is retained with those logs. These component counts are not a substitute for the root's forthcoming selected-suite receipt.

The exact result qualifications disclose reused development/evaluation history, earlier regression access beginning 2025-11-03, the current vendor-vintage limitation, and this experiment's numerical ceiling of 2025-10-20. They do not represent the later period as untouched confirmation. Reused synthetic calibration is evidence about its fixed generated process, not certified market coverage or adjusted-tail probabilities. This review certifies no historical source vintage, future predictive benefit, or practical trading value.

Reviewed byte identities follow. They identify the files inspected here; the forthcoming root freeze must independently bind the complete relevant inventory. Later empirical results and render artifacts are outside this review.

| File | SHA-256 |
| --- | --- |
| `precision_gate.yaml` | `c9ac4c6f238cc898e219b8e1b1e9c8f1bf84371e40bf363a7572ee63182f11a2` |
| `src/precision_gate_protocol.py` | `04a8111110d51df1ee353fddb0c2b6419de2b9c3f208c371f2bf39403b476f1a` |
| `src/precision_gate_search.py` | `3516fff89461503074d3e7e4c74a0c2dcc10f9c262435275e89f1b88f3a4202d` |
| `tests/test_precision_gate_search.py` | `69296a6f2cc77241ba5762ca873b81e25c36a733785596b47f27b58a5aaae5d7` |
| `src/precision_gate_models.py` | `4e175974e39e1faf3e91dfb01894729b986b892020f707d037107b3e2f83c290` |
| `src/precision_gate_pipeline.py` | `b09fc13ef027919d3c3c96f81224476d9dc282e53c1f89a57ab097a76e5db1dc` |
| `src/verify_precision_gate_forecasts.py` | `820771fd50954042be4739adad8690f53e74c52a53b9bcc1cfbf81fa58cb8d72` |
| `tests/test_verify_precision_gate_forecasts.py` | `c1b141db5aa652ec08a4801832e89d8f57cbe83f759339ebdc43d829a7b878e3` |
| `src/precision_gate_score.py` | `77629299d5b0f54da0683392a4465605196c628b5f3d4038f06e0ef333967b6b` |
| `src/verify_precision_gate_scores.py` | `1d625705a198e8de8f638ce721603ee26b6be4969765d45bb37e9e03b3d77516` |
| `reports/precision_gate/prefit/EXECUTION_CONTRACT.md` | `5f7ef45f282775b636190bffe88b1ceda5fbe892ff2e6f72590d159f81c597fd` |
| `reports/precision_gate/prefit/EXECUTION_REPORT_BINDING_GREEN.log` | `3614ae11d8c052b6fcf48617cb6ac6c5a8d9da74c74a2c12ab469e050d0c91a2` |
| `reports/precision_gate/prefit/PIPELINE_FAILURE_RETENTION_FINAL_GREEN.log` | `45d54775ce424b66f01232ef3d37cbbd7bd3acdf32cc1c6f017db0d4715e54ea` |
| `reports/precision_gate/predictive_prefit/FORECAST_VERIFIER_GREEN.log` | `f94cec506c056c31541c9b2543c249093ae7dddb0392f4083cc81c8a2fd8b553` |
| `reports/precision_gate/predictive_prefit/SCORE_GREEN.log` | `32bcf78b3667e273d26194ee9edcf2f9f9c2869fbe9b54c775d31f81b56199c3` |
| `reports/precision_gate/predictive_prefit/SCORE_VERIFICATION_GREEN.log` | `2d9ab17e10dad88d44605e4deb19309c931075802c6228ddfb82e748287a15ea` |
