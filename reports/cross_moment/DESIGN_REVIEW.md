# Direct cross-moment score: independent prefit challenge

This review uses the declared forecast schema, frozen source code, and algebra. No wave 11 products, losses, numerical forecast values, fits, or empirical associations were read or constructed. No upstream file was changed. The proposal is a separately registered score of existing issued predictions; it does not change the wave 10 criterion or add information to the models.

## Threshold and the claim it supports

The proposed absolute useful-effect threshold **1e-10** is dimensionally coherent:

`(0.01 * 0.001)² = 1e-10`.

The two reference numbers are decimal log-return magnitudes, approximately a 1% and a 10-basis-point price move. Their product has squared-return units, and its squared error has fourth-power return units. The threshold is the squared error associated with a reference cross-moment error of `1e-5`. It is **not** a statement that the model improves covariance RMSE by `1e-5`, correctly predicts such a return pair, or earns a corresponding trading profit. A difference of mean squared errors is not a difference of root mean squared errors.

This is a transparent ex-ante decision scale, rather than a calibrated economic definition of usefulness. That limitation is acceptable if stated before inspecting the new losses and the value remains fixed afterward. No additional empirical power gate is needed. The proposed two-phase/two-control effect and significance gates already define promotion; adding a scale, tail, or power gate after seeing the scores would change the test. Keep the fixed negative-sign requirements for both later evaluation subperiods.

HAC126 nominal 80% minimum detectable effect is useful **as a diagnostic in the same fourth-power units**. Label it nominal: the usual 5% two-sided calculation is not the detectable effect at the much stricter wave/cumulative multiplicity thresholds, and estimated standard errors do not establish power retrospectively. Report uncertainty and detectable-effect diagnostics without declaring equivalence or lowering the useful-effect threshold when uncertainty is large.

## Exact functional and stable paired arithmetic

Use the common means actually issued at each origin. With `e_Q=r_Q−mu_Q`, `e_S=r_S−mu_S`, the realized response is `Y=e_Q*e_S`. Each model predicts `q=H_QS=rho*sqrt(h_Q)*sqrt(h_S)`. The target remains a conditional residual cross moment:

`E[Y | information] = Cov(r_Q,r_S | information) + (true_mu_Q−issued_mu_Q)*(true_mu_S−issued_mu_S)`.

Squared loss `(Y−q)²` is proper for this functional when its expectation exists, since conditional expectation decomposes into `Var(Y | information)+(q−E[Y | information])²`. An individual realized outer product may be zero or rank one, and the product and forecast may be signed. A zero target, zero forecast, zero error, or zero loss is valid.

For candidate and control, calculate the paired difference as

`(q_candidate−q_control) * ((q_candidate−Y)+(q_control−Y))`.

This equals candidate squared loss minus control squared loss. It cancels the common `Y²` term without subtracting two large squared losses and avoids explicitly forming `2Y`. Independently check both individual squared errors and compare the factored expression against higher-precision arithmetic on synthetic fixtures, including nearly equal forecasts and large common target terms. The final preregistered direct/factored consistency allowance is the sum, over `L_candidate`, `L_control`, and `d`, of `64*max(eps*abs(term), abs(term)−nextafter(abs(term),0))`. This uses the relative floating allowance or one downward ULP, whichever is larger, and has no fixed absolute unit floor. Summing already scaled terms avoids overflowing a raw sum of large losses. The final allowance must be finite. Subnormal terms receive their own representable spacing; an exact-zero term contributes zero. Penalties used during the original fits must not enter the new score.

This direct criterion removes the matrix inverse's reweighting of squared residuals when correlation changes. It still uses issued diagonal forecasts to construct `q`, and it does not establish correct shared means or scales. Both dependence arms must retain exactly the same issued means and diagonals; all three arms retain exactly the same means and actual returns. A gain establishes better prediction of the residual product against those controls, not pure correlation dynamics, high-frequency covariance measurement, or new information in the source set.

## Numerical admission without a value floor

Use the declared decimal-log-return units throughout. Compute the forecast's geometric-mean diagonal as `sqrt(h_Q)*sqrt(h_S)`, then multiply by `rho`. This avoids the unnecessary direct product `h_Q*h_S` and avoids multiplying a very small square root by `rho` before combining the other square root. Diagonals must remain finite and strictly positive under the upstream gate. Do not replace small values, rescale observations by their forecast variance, clip, or drop individual rows.

The proposed finite-arithmetic domain should be explicit in the protocol:

- Reject nonfinite residual differences, products, forecast cross moments, errors, squared losses, paired factors, paired differences, or aggregate/inference results. A finite input alone does not guarantee its derived square is representable.
- Preserve exact zeros. If both factors in a multiplication are nonzero but the result becomes zero, reject the run as unsupported underflow instead of treating that value as a genuine observation of zero. Apply the same rule to a nonzero error whose square becomes zero and to the factored paired difference. Exact cancellation making a paired factor zero is valid.
- Finite nonzero subnormal values may remain; no smallest-normal-number floor is justified. The relevant limit is representability in the registered float64 arithmetic, including the smallest positive subnormal, not the useful-effect threshold. Do not compare values to `1e-10` to decide whether a target or loss is numerically zero.
- Failure of this arithmetic contract makes the whole new two-comparison wave unevaluable, retaining both p-values at one. There is no alternate precision, outcome-dependent rescaling, selective deletion, or after-score repair fallback.

These are computational admission rules, not additional evidence gates. Synthetic tests should distinguish genuine zero products from nonzero-factor underflow, check overflow and sign changes, confirm asset-order symmetry, and show that independent return-unit changes `r_Q→a*r_Q`, `r_S→b*r_S` multiply every product-loss difference by `(a*b)²`. The absolute threshold is meaningful only under the original fixed units. Tail-dominated sampling variation remains possible: the product's second moment involves cross fourth moments of residual returns. The paired algebra improves computation; it does not remove that uncertainty.

Root's proposed fixed inference scale `s=1e-10` is also coherent: perform bootstrap and HAC on `d/s`, define the authoritative mean difference as `mean(d/s)*s`, and restore CI, SE, and MDE to raw loss units. This is one preregistered positive unit conversion for every observation, not volatility weighting or a change to individual product MSE. Apply the same mean convention to annual and fixed-subperiod diagnostics. Check scaled inputs, intermediate summaries, restored values, and consistency allowances for finiteness, and test mathematical p-value invariance under the conversion. Keep individual MSEs in the original units. This conditioning step adds no empirical arm or power gate.

## Pin and revalidate the actual issued forecasts before products

An existing `verification.json` status does not bind the current forecast output bytes. It is necessary evidence, not sufficient admission. Before constructing any new residual product or loss, the new wave must register and hash-pin the **current** upstream files:

- `joint_risk.yaml`; `reports/joint_risk/manifest.json`, `verification.json`, `metrics.json`, and `trial_ledger.jsonl`.
- `data/joint_risk/source_audit.json`, `measurement_audit.json`, `features.parquet`, `targets.parquet`, `forecasts.parquet`, and `fits.json`.
- The unchanged source, provenance, code, test, and preservation records named by the upstream manifest, plus all new wave 11 code/tests/protocol.

Require the upstream verification to be `VERIFIED`, its protocol/verifier identities to match, the numerical measurement gate to have passed, and no superseding terminal failure. A verified non-passing predictive result is eligible for this distinct new scoring question; a failed verification or unevaluable forecast construction is not.

Implement admission as a new read-only adapter, for example `validate_upstream(root) -> {pinned_hashes, replay_evidence, issued_forecasts}`. The frozen independent verifier already exposes reusable components in [verify_joint_risk.py](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/verify_joint_risk.py): `validate_protocol`, `verify_manifest_coverage`, bounded `load_source_tables`, `measurement_audit`, `require_measurement`, `feature_target_tables`, `verify_forecasts`, `verify_metrics`, and `verify_ledger`. Compose their checks, including all upstream manifest hash groups, exact saved feature/target schemas and dates, source-audit equality, all three forecast arms, exact common issued means/diagonals, maturity and phase fences, fit provenance, and coefficient/certificate reconstruction.

The read-only replay may numerically reconstruct the **same frozen fits** to verify them; that is verification, not an additional fitted candidate or permission to overwrite predictions. The new products must be computed from the hash-pinned **issued forecast columns**, not replayed values that merely agree within the upstream tolerances. Save admission evidence only under `reports/cross_moment` or the new output directory.

Do **not** call the frozen verifier's `verify()`, `verify_with_failure_guard()`, or `invalidate_publication()`: those entry points write or invalidate earlier reports. A wave 11 admission failure belongs to wave 11 and must leave all wave 10 artifacts unchanged. Recheck upstream hashes after replay and immediately before product construction, or load immutable byte snapshots and verify that the decoded forecasts came from the registered bytes. Recheck hashes again before publication. A marker with a tampered forecast, a changed file between admission and scoring, altered means/labels, or a source/certificate mismatch must fail a prewritten synthetic contract.

## Registration and recommendation

Register exactly two new comparisons: dynamic-correlation product MSE against constant correlation and against the constant residual matrix. Require an absolute paired mean loss decrease of at least `1e-10` in both development and evaluation against both controls, both fixed later subperiod gaps negative, Holm2 below `.05/(11*12)`, and cumulative Holm114 below `.05`. Keep the prior phase/maturity/common-row boundaries and inherited 112 hypotheses. Signed and zero losses need no percentage calculation or positive-baseline-loss assumption. With 99,999 bootstrap draws, the attainable minimum p is below one tenth of the strictest raw two-comparison wave threshold; that is numerical resolution rather than power.

The design is coherent with the clarified threshold interpretation, strict finite arithmetic, and read-only upstream admission. No further predictive gate is warranted from this blinded review. Any success remains a reused-history result for one narrower functional of the same archived ETF/index forecasts. It cannot retroactively change the earlier matrix-score verdict.

## Read-only implementation follow-up before freeze

The new runner, protocol, scorer, and publication tests were inspected without running upstream validation or computing empirical products. The fixed inference conversion is used for phase, annual, and fixed-subperiod differences, with restored dimensional results; the useful-effect threshold, both controls, both phase requirements, later signs, and 2/114 Holm accounting match this design. Canonical publication failures retain both new comparisons as unevaluable p-values of one and leave earlier report paths untouched. Nonserializable scored diagnostics are retained as unpromoted text rather than bypassing terminal accounting.

The review identified one concrete gap: upstream admission originally returned before the runner reopened forecast and calendar paths, so those later reads were not bound to the bytes just admitted. Root added a `read_pinned_parquet` helper that reads bytes, compares their SHA256 to the new manifest input pin, and decodes that same byte buffer. Both the issued forecast and calendar are loaded this way before the scorer. Separate publication fault tests change each file after successful admission and require that scoring never starts. The helper and test intent were checked read-only; this review did not execute the new numerical study. No remaining integration blocker was found in the reviewed root runner.

The independent verifier's final schema/admission review also found no substantive compatibility blocker. Its separate `read_issued_snapshot` binds decoded issued forecasts to the same registered bytes; its scalar product arithmetic, zero/sign checks, restored-dimensional inference, two new/cumulative 114 comparison accounting, and 116-event successful ledger align with the producer. Upstream reconstruction composes read-only helpers and validates the complete earlier verification record without invoking its writing entry points. The new verification-failure guard writes only the cross-moment report directory and records both terminal p-values before optional diagnostic backups. This was a code review, not an execution of upstream admission or the new scores.
