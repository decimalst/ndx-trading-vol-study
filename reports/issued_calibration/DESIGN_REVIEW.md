# Prefit review of issued-alert calibration

This is a bounded review of the new protocol, producer schedule, runner, failure publication and plotting contract. No empirical calibration has been run or evaluated for this review. The source/admission author performed this review, so the admission implementation’s independent proof must still come from the separately written verifier. Source preservation and successful synthetic tests are preparation evidence, not a candidate result.

No material contract issue remains in the reviewed runner and prospective mathematics. All 100 distinct scoped preparation checks pass: 19 admission tests, 17 model tests, 21 independent-verifier tests, seven shared integration tests and 36 runner/publication/plot checks. Integration includes independent full inference, and the complete independent publication entrypoint is now implemented and tested. The full repository suite passed all 1,988 tests in 180.307 seconds, with process exit status zero. No frozen earlier file was changed.

## What this experiment can establish

The new question is whether a single causal intercept adjustment improves the existing binary risk alert. It uses the same target and unchanged baseline/recent-frequency controls. It does not re-register the older baseline-versus-frequency contrast, add an exogenous information source or establish a new structural predictor.

Here **issued** means the immutable earlier walk-forward forecast records produced by the archival experiment. These were not contemporaneously timestamped live production predictions. Replaying their original coefficients and transformations preserves the earlier walk-forward information boundary; it does not establish original-vintage market availability. The prior Yahoo/Cboe revision/latency caveats and repeated adaptive use of the same history remain material.

The target remains risk in the next observed session exceeding twice the mean of exactly 22 sessions ending at the previous-session cutoff. This is an alert on the declared daily OHLC proxy, not doubling from the immediately preceding day, a signed crash, measured high-frequency variance or a trading payoff. Brier improvement is an absolute probability-score improvement. It does not establish conditional calibration in every state or profitability.

## Causal replay and scalar objective

All original application records must replay before any new scalar optimization. The source baseline is recovered from its own saved fit, column order, training geometry and coefficients; a newer fit cannot reconstruct an earlier prediction. Rounded probabilities are not inverted. Original unscored applications remain issued records, while dates without original applications cannot acquire retrospective predictions.

The source’s original preflight checks unchanged mature common training rows, all original monthly transforms and the scored cohort. It performs no new monthly baseline optimization. The new calibration then advances through every full reference-calendar close. Only a previously issued record with a label available by the application’s prior-session cutoff can enter. Feature gaps and unknown labels age existing weights. Arrivals are classified as no issuance, unknown label or admitted, with no resampling, warm-up deletion, pre-seed refeeding or phase reset.

At empty initialization, the correction is zero. Each later eligible record receives the fixed unnormalized weight `(1-delta)*delta**age`, with a 63-session half-life. The signed logistic objective adds `.01*a²` without dividing by record count or weight mass. Its derivative is monotone with slope at least `.02`; the fixed bracket `±(sum(weights)+1)/.02` has strict opposite signs. This makes the optimum unique, including empty or single-class histories. Brent versus independently implemented bisection supplies a distinct scalar check, with the prospectively fixed original-gradient gates and no fallback.

Signed softplus and binary residual evaluation avoid cancellation, but saved rounded probabilities and mathematical finite-logit errors are explicitly different objects. When the fitted correction is exactly zero, the producer copies the admitted saved baseline probability after strict original replay; otherwise it evaluates the shifted finite logit. This fixed identity branch preserves empty/balanced nesting without fitting a new baseline or silently clipping probabilities. Nonzero arithmetic underflow remains a failure rather than permission to omit a record.

The recent-frequency control retains its original broader known-label stream, including dates without baseline issuance. Its recomputed state identity and both controls’ probabilities/metadata must agree with the immutable records. It is not recalibrated or silently changed to match the candidate’s narrower paired-error stream.

## Admission, registration and terminal accounting

The new admission layer checks seven immutable wave18 anchors, all ten independently verified output snapshots, publication/documentary identities and the recursive prior manifest closure. Output decoding consumes the same checked bytes, with registration coverage required and a final whole-closure rehash. Earlier raw-source bytes are hashed without parsing their numerical values. Legacy `existing_artifacts_sha256` maps are enforced; numerical `bounded_content_sha256` fingerprints are not mislabeled as raw-file hashes.

After its initial 18 prewritten tests passed, the parent-authorized actual metadata-only check admitted 1,866 closure files with no conflict, missing file or hash mismatch. Parquet decoding was forbidden during that check. This verifies preservation/admission feasibility and is not a new feature, event count, fit or predictive analysis. A subsequent failure-marker regression brings the admission suite to 19 passing tests.

The runner now registers both hypotheses immediately after freeze/pretest validation, before source-closure enumeration or manifest construction. Therefore an admission or manifest failure cannot disappear as an unregistered attempt. An existing ledger or failure record also blocks a rerun even if no manifest was completed. Canonical failure metrics retain both new p=1 outcomes and no lead.

Successful full-history accounting is **129 inherited comparisons + two registrations + two evaluated outcomes = 133 ledger events**, corresponding to **131 cumulative hypotheses**. Producer-failure recovery is atomic and preserves interrupted ledger bytes. If inherited records were not yet available, it writes the two registrations and two unevaluable outcomes with an explicit zero reconstructed-history count; missing old records are not fabricated. Prior failed attempts retain their original p=1 and failure interpretations.

There are **zero new monthly baseline fits**, one new candidate forecast per original scored origin, two reused control rows per origin, and three combined rows. Every original application receives a calibration state, including an unscored application. The scalar calibration audit is separate from copied source monthly-fit metadata. Counting combined rows as newly generated forecasts would overstate the new experiment; the runner’s separate fields prevent that error.

## Inference and publication

The two contrasts are calibrated versus baseline and calibrated versus recent frequency. Both must improve by at least `.0005` mean Brier loss in both phases, improve in both fixed later slices, pass Holm2 at `.05/(19*20)` and cumulative Holm131 at `.05`. No lone control, calibration summary or selected subperiod can qualify the candidate. The new seed is 20260925. The 199,999 draws retain adequate minimum-p resolution for one tenth of the strictest two-comparison wave cutoff. HAC126, blocks21/63/126 and the maximum p across declared methods/phases remain fixed. Nominal MDE/effect ratios remain descriptive, not equivalence or adjusted-power gates.

The runner checks complete shared panels, original date predecessors/target maturity, literal study phase fences, training metadata and phase/class support before inference. The plot requires current successful independent verification, exact verified metrics bytes, a complete two-control/four-phase cohort, matching new/reused/application counts and zero new monthly fits. Separate horizontal scales retain readability for differently sized contrasts, with literal interval endpoints and the −.0005 reference.

The parent reports 36 passing runner/publication/plot checks. Its added closure-failure test confirms registration and terminal p=1 records exist even before manifest creation. Atomic partial-registration, missing-history, report and evaluated-write faults are also covered.

The review identified a second publication boundary: checking existing-file hashes does not establish the absence of a newly created canonical upstream `failure.json`. The source helper now checks absence after its final rehash as well as before it; a fault injected during the last read failed before the correction and passes afterward. The producer similarly repeats the old-marker absence check after candidate work and before publishing scored metrics, with a prewritten failing regression. Final read-only inspection confirmed that the independent verifier repeats absence checks after closure hashing, after output decoding/rehashing and immediately before writing `VERIFIED`, after candidate reconstruction, inference and final artifact checks. These are synthetic safeguards; the full independent historical replay and empirical verdict remain unrun.

## Preservation and completion boundary

A read-only comparison checked all 279 prior Python files against the frozen wave18 manifest, with no mismatch. The new inventory currently adds five source modules and seven test modules, totaling 291 Python files. No actual calibration records, intercepts, forecasts, event associations or losses were inspected or computed for this review.

The complete test evidence is [full_repository_tests.txt](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/issued_calibration/full_repository_tests.txt), SHA-256 `af90dd19704a8204340e14dcb2c1c1c8e182197edb577dc30880ec7f6b91d5d1`. Its recorded 1,988-test success completes prefit testing. The next step is explicit freeze. Historical execution and any result publication require the parent’s registered run and the new independent verification; this review cannot substitute for either.

The following stable bytes were reviewed and rechecked after the full-suite result. These hashes identify prepared implementation and design, not empirical outputs:

| File | SHA-256 |
|---|---|
| [issued_calibration.yaml](/Users/byrons/code/trading-vol/ndx-vol-experiment/issued_calibration.yaml) | `f4c03c25718439432c176aba36ce3bc572776b67a5ff80f906ddc00cd85b2051` |
| [issued_calibration_admission.py](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/issued_calibration_admission.py) | `1b53be9ae7dea920179de1d55251de391ad79ee09bb9c11351c262bb627526a2` |
| [issued_calibration_models.py](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/issued_calibration_models.py) | `c6e33df945df45f4b55293fca90389668afa46c2bac10379783b79552fa8763a` |
| [issued_calibration_search.py](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/issued_calibration_search.py) | `ae7187152bc797111aea9b9ac786ade9a87d5faeedc96359d29f7ec0dc39e11e` |
| [verify_issued_calibration.py](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/verify_issued_calibration.py) | `6d2724eb5c9996e0560a9c4bab3759e8d64600cbd1a51d374a045ea418ee755d` |
| [plot_issued_calibration.py](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/plot_issued_calibration.py) | `3bf9cd45f69ae3904d18ab69478b3e02992e88c6c6915ace1bc0197bce0dc039` |
| [DESIGN.md](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/issued_calibration/DESIGN.md) | `29851428d210c242e3d3300353f77e98f9e04ff151e265be4935a742c8cd7c24` |
| [SOURCE_DESIGN.md](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/issued_calibration/SOURCE_DESIGN.md) | `401c3ec3fbce0e414d0f81a9c670750d146ec37689091967e0076b3f011635b0` |
| [MODEL_DESIGN.md](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/issued_calibration/MODEL_DESIGN.md) | `b291b5efad443987080330323e19483afcae08d0067f4555268d92db5bac6042` |
| [VERIFIER_DESIGN.md](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/issued_calibration/VERIFIER_DESIGN.md) | `f08848a68d8209c819033a42f1edee8078cfa8632e925e66a9995eebdf376a29` |
