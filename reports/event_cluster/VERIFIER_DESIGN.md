# Prospective independent verification: event clustering

**Prospective contract and supplied-table independent core; no historical calculation.** The selected wave20 mechanism has an independent construction and numerical implementation with prewritten synthetic tests. The whole-study typed protocol guard, source admission, pipeline reconstruction, inference and guarded publication entry point are now implemented and tested with synthetic inputs; no historical execution is authorized by this document. This document does not change earlier studies, admit another source or register comparisons. No historical event histories, support counts, associations, candidate forecasts or fits have been constructed.

## Exact statistic and its limited interpretation

At each application, let `e_1,...,e_22` be the original binary SPX risk-alert labels ordered by their actual availability closes, oldest first, ending at the prior-session cutoff. Every position must be known; no compression of missing arrivals is allowed. Define integer counts `N=sum(e)`, `b=e_22`, and `C=sum(e_i*e_(i-1), i=2..22)`.

The five nuisance inputs, in order, are:

1. `event_fraction22 = N/22`;
2. `event_fraction22_sq = N*N/484`, one integer-product division before training centering;
3. `last_event = b`;
4. `expected_adjacency22 = (N-b)*(N-1)/441`;
5. `linear_recency22 = sum((2*i-23)*e_i, i=1..22)/462`.

The sole fitted candidate input is `A=adjacency_fraction22=C/21`. Keep `M=excess_adjacency22=(21*C-(N-b)*(N-1))/441` as a descriptive audit only; it never enters either fitted design.

This prospectively replaces the initially proposed fitted M before implementation or historical calculations. If C is constant while G varies, adding M to a frozen nuisance offset can simply retune G, which is already a nuisance input. Fitting centered A eliminates that guaranteed redundant correction: constant training C forces coefficient zero, including when application C changes. It permits only a model-relative predictive adjacency claim. Realized A may still be collinear with nuisance, so no conditional orthogonality or information-theoretic novelty is claimed.

Compute the count products and numerator as exact signed integers before the fixed divisions. Descriptive M is algebraically `[C-E(C|N,b)]/21`, without subtracting rounded floating expectations. All weights, windows and divisors are fixed; the recency divisor is **462**.

Under uniform permutations of the first21 binary positions conditional on `N,b`, set `q=N-b`. The20 internal adjacent pairs have expected total `q(q-1)/21`; the last pair contributes `b*q/21`. Their sum is `(N-b)*(N-1)/21`. Thus the proposed expectation is correct. This is a combinatorial centering reference, not an assertion that market labels are exchangeable and not a randomization-test null. It does not imply mean zero conditional on the additional recency variable or the market controls.

There is an exact arrangement distinction beyond the proposed nuisance inputs: the synthetic sequences with ones at positions `{1,2,6}` and `{1,3,5}`, and zeros elsewhere through22, have the same `N=3`, `b=0`, squared count, expectation and linear recency sum. Their adjacency counts are respectively1 and0. This proves a degree of variation in the candidate; it proves no predictive value. The statistic reflects runs of the thresholded alert, whose overlapping risk references can themselves induce dependence. A result would concern that archived alert functional, not identify a physical latent regime, causal mechanism or new exogenous source.

## Chronology and proof that rows cannot be discarded

Reuse the original full SPX calendar and wave18 target table. For original origin position `i`, the required source labels occupy `y[i-23:i-1]`: their availability closes are the22 positions `i-22,...,i-1`. Select by actual source availability and independently check the origin/target/availability mapping. The last eligible label is not the current origin's later outcome. Labels attached to unscored or feature-incomplete original origins remain eligible once known and mature. Every source position remains on the full calendar.

The original alert compares next-session GK-plus-overnight risk with twice its strict22 preceding risk reference. A22-label history therefore needs more raw history than22 bars. Original feature completeness does not prove availability of the new label history. Neither zero filling nor the last22 known observations is an admissible replacement.

Let `T_j` and `A_j` be the exact original training and application indices for monthly fit `j`, including applications with unknown eventual labels. Before any new optimization, require a complete, valid new history at **every member of every `T_j ∪ A_j`**. Compare these indices directly with the admitted original metadata and independently reconstructed original masks. If any required history is missing, fail the whole new study; do not shrink a training set, skip an application, move a monthly fit or remove an unscored month. This explicit set-containment check is the proof obligation behind the unchanged-cohort promise. It remains to be exercised after registration; no historical completeness assertion is made here.

Retain original source/date fences, maturity rules, development and evaluation windows, original class/count support gates and both evaluation slices. Histories update through the prior close, never through a future target. Archived source revisions and historical release latency retain their prior limitations.

## Independent staged objectives

For each original monthly fit, independently reconstruct its saved baseline geometry and coefficients. Apply that saved monthly baseline to its own mature historical training rows to obtain finite logits `eta0`. These are **current-fit in-sample training offsets**, not historically issued training forecasts. They use only data available to that monthly fit. Requiring issued logits for all those training rows would contradict the unchanged first-fit training cohort because original forecasts do not cover that whole pretraining history. No original baseline optimizer is rerun.

For application rows, replay the saved original baseline logits and verify their probabilities against the actual issued probabilities at `rtol=1e-10, atol=1e-12`. Preserve the actual issued baseline and recent-frequency control probabilities exactly.

Center each of the five nuisance columns on the exact original training rows, with fixed scale1. An exactly constant column uses its first value as its center and stays present as zero. Do the same for `A=adjacency_fraction22`. Descriptive M is excluded. There is no empirical-dispersion scaling, rank-based deletion or replacement basis.

**Nuisance stage.** With centered five-vector `H`, fit one unpenalized intercept `a` and five slopes `theta`:

`F(a,theta) = mean(logaddexp(0,(1-2*y)*(eta0+a+H@theta))) + .01*sum(theta^2)`.

For `z=[1,H]`, signed stable Bernoulli residual `r`, and `w=expit(eta)*expit(-eta)`, the full gradient is `mean(z*r)+[0,.02*theta]`; its Jacobian is `mean(w*z*z.T)+diag(0,.02,...,.02)`. Both-class support and the positive slope penalty give a finite unique optimum for finite admitted offsets and arithmetic. The intercept must remain unpenalized; it supplies matched calibration rather than letting the candidate's geometry compensate for a missing global adjustment.

Use independent `scipy.optimize.root(method='hybr')` from exactly six zeros, analytic Jacobian, `xtol=1e-10`, `maxfev=2000`, `factor=1`. Require solver success **and** the recomputed full gradient infinity norm at most `1e-8+1e-12`. Record status, message, function/Jacobian evaluations, final coefficients, objective and full gradient. No alternative start, retry or solver fallback follows an empirical failure.

**Clustering stage.** Freeze the fitted nuisance logit `etaN=eta0+a+H@theta`; with `z=A-mean_training(A)`, fit only `beta`:

`Q(beta) = mean(logaddexp(0,(1-2*y)*(etaN+beta*z))) + .01*beta^2`.

Its derivative is `mean(z*r)+.02*beta` and curvature is `mean(z^2*w)+.02 >= .02`. Independently solve the derivative using `bisect` on fixed `[-R,R]`, where `R=(mean(abs(z))+1)/.02`, with `xtol=1e-12`, `rtol=1e-14`, `maxiter=200`. Check finite endpoint gradients and strict opposite signs, convergence and the same full-gradient gate. Exactly constant training memory is centered to exact zero and retains canonical `beta=0` with the fold/model present.

After independently validating the nuisance optimum, reconstruct the scalar problem using the saved, checked nuisance coefficients. This verifies the conditional optimization actually used, without compounding permissible differences between independent nuisance solvers. The candidate must never refit nuisance or original baseline coefficients.

## Numerical and prediction comparisons

The producer's saved full-gradient gate remains `1e-8`; the independently recomputed/solved gate is `1e-8+1e-12`. Saved objective replay uses `rtol=1e-10, atol=1e-12`. Compare independently solved coefficients at `rtol=1e-7, atol=1e-6`; strict saved-logit/probability replay uses `rtol=1e-10, atol=1e-12`. Apply strict finite-value and probability-domain checks before tolerances, and reconstruct all application outputs including unscored ones.

The nuisance stage retains the frozen finite-value logistic objective/Jacobian policy. The scalar stage additionally requires signed softplus and signed Bernoulli residual to remain nonzero for finite logits; a probability rounded to an endpoint is allowed when those quantities remain representable. Checked products and original-coordinate divisions reject nonzero-to-zero underflow. No inverse-logit reconstruction or probability clipping is permitted. Independent calculations must not introduce unrelated admissibility tests through unused diagnostic curvature or other optional quantities. The confirmed exact nesting rule copies the parent's checked saved probability wherever the added correction is exactly zero; in particular `beta=0` reproduces the nuisance forecast exactly.

Use the exact original binary targets and independently reconstruct each new Brier score and stable paired difference. Old control rows, timing, origin cohorts and values must remain unchanged. Nuisance parameters are optimized for normalized log loss; improved training likelihood cannot replace the registered Brier gates.

## Admission, inference and required prospective checks

Bind the original wave18 source/proof/outputs through the new wave19 preservation wrapper, including the zero-hypothesis early-session documentary pass. No old source/model/inference writer is invoked. Decode only checked byte snapshots after complete closure admission. Check both relevant prior failure-marker absences after decoding, after final hash rechecks and before publishing a new successful verification. The final verifier must save the exact output hashes it verified.

The candidate has three required comparisons: original baseline, fitted nuisance control, and original recent frequency. Retain all131 earlier hypotheses, giving134 only when this new family is registered. Use wave20 Holm3 at `.05/(20*21)`, cumulative Holm134 at `.05`, the unchanged `.0005` Brier gain in both phases against all controls, and favorable signs in both later slices. The selected399,999 bootstrap draws resolve to `1/400000`, below one tenth of the smallest wave Holm3 cutoff `1/25200`. Keep the declared HAC126 and blocks21/63/126. Inferential counts and any generated-forecast counts must come from the eventual verified record, not be asserted in this design.

Prewritten synthetic tests must precede implementation and cover: exact combinatorial expectation; integer arithmetic, all-zero/all-one/tie cases and malformed labels; the same-nuisance/different-adjacency example; constant adjacency with varying count/expectation forcing exact parent forecasts; full-calendar availability and no compressed unknowns; strict rejection of any newly incomplete original training/application row; future-label and future-coefficient invariance; all original unscored applications/months; current-fit training offsets versus original issued query controls; nuisance and scalar gradients, independent optima, zero-correction nesting and fail-closed numerical guards; all three paired comparisons and full family/ledger accounting; and source, output, failure-marker and publication tampering. A failed source/support/optimizer/reconstruction/publication gate must leave all three new comparisons unevaluable with p=1 and preserved diagnostics, never trigger a data-dependent repair.

## Supplied-table implementation evidence

The fitted raw-adjacency revision above was made before writing the independent tests or implementation. The20 prewritten tests initially failed at import because the new module did not exist; that original output is retained in `verifier_initial_red.txt`. The first implementation passed19 tests and found one date-storage-unit mismatch: structural date fields used nanoseconds while the generated reference calendar used microseconds. The implementation now preserves the reference calendar dtype; no date values, formulas, tolerances or tests were changed to resolve it. The original failure remains in `verifier_first_implementation.txt`.

All20 tests now pass, including independent integer formulas and permutation centering, exact full-calendar history reconstruction against a separately implemented producer, missing-history rejection without cohort changes, future-label invariance, fixed training centers, checked saved baseline geometry, nuisance analytic gradient/Jacobian checks, separate zero-start root and scalar bisection solves, saved-stage reconstruction and tamper rejection. The constant-C/varying-G regression passes: query adjacency can change, but the fitted cluster coefficient remains exactly zero and issued cluster probabilities equal their nuisance parents. Source/tests pass scoped Ruff. `verifier_core_green.txt` records the final formatted-core test run.

The20-test record above covers the original pure core. The subsequent entry layer has its own prewritten RED and final GREEN evidence described below. No historical verification status is claimed.

## Complete prospective entry and publication checks

The final entry consumes exactly six new data outputs (`upstream_admission.json`, `forecasts.parquet`, `states.parquet`, `memory.parquet`, `fits.json`, `support_audit.json`) plus metrics and trial ledger. Each decoder receives the same captured byte buffer whose SHA256 is saved in `verified_output_hashes`. All registered code, input and preservation hashes, complete inventories, protocol/manifest identities, freeze design hashes and full prefit test-log identity are checked before and after reconstruction. Both original failure-marker absences and the new failure-marker absence are checked immediately before the successful publication commit.

Source metadata traversal is deliberately shared with the new admission module; it is not represented as a second independent source-interpretation implementation. The verifier separately checks the literal anchored documents, every admitted file hash, all registered input coverage, the prior VERIFIED identity and exact saved output hash map, then repeats those checks after checked-snapshot admission. No shared producer feature, fit, prediction, score or inference routine establishes the new reconstructed values. The separately owned `event_cluster_verification.verify_pipeline` reconstructs original training/application cohorts and every new monthly stage using this verifier's primitives and frozen independent helpers. It retains all unscored applications/months and rejects any newly missing history before mathematical fitting.

The entry tests were written before this layer existed; `verifier_entry_initial_red.txt` retains the missing-function errors. The first implementation's five entry errors came from a metadata-only synthetic source fixture that intentionally lacked an original index contract; the fixture was extended to include its declared synthetic index before any hashes were formed. No behavioral assertion, numerical tolerance or source-admission rule changed. `verifier_entry_first_implementation.txt` retains the initial errors, and the corrected34-test run is retained separately.

Final audit review then prewrote three regressions that failed: a boolean zero could masquerade as an integer refit count; a tiny negative saved gradient maximum could pass a comparison tolerance; and an unterminated malformed ledger record could absorb the first appended failure event. The fixes require literal count types, nonnegative gradient maxima and a separating newline before appending all three canonical failure records. `verifier_domain_initial_red.txt` preserves those failures. None changes the mathematical objective, solver, budget or tolerance.

All35 owned tests pass. They include real admission of a wholly temporary two-layer source package; actual six-phase inference with18 independently reconstructed199-draw synthetic bootstraps, all134 hypotheses and eight descriptive calibration rows; strict137-event success ledger identity; output mutation and final prior-marker rejection; and all-three-p1 invalidation with malformed/NaN metrics and an unterminated ledger. Reduced199 draws occur only in the explicitly generated test fixture; the registered protocol remains399999. The temporary entry wiring test mocks already separately tested numerical/inference substeps to isolate complete paths and byte commitments; it is not claimed to be an empirical end-to-end run. Full supplied-table mathematical integration is provided by the separately prewritten pipeline-verification suite.

The canonical failure transition writes UNEVALUABLE/no-lead metrics, failure JSON, a FAILED verification and human-readable failed result before optional diagnostic backups. It appends all three failure records and preserves prior files. Existing scored metrics, if any, remain explicitly unpublished diagnostics. A failure is not a null result and cannot trigger a historical retry, alternate optimizer or changed tolerance.
