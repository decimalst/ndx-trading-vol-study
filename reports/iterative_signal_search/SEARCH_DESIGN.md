# Sequential search design for volatility and index predictability

Design recommendation, 2026-09-06. No new empirical fit or score was produced
for this document. The available 2011–2025 history has already been inspected.
The existing protected phase remains closed.

## What a successful historical iteration can establish

Use historical experiments to identify a reproducible, useful **exploratory
lead**. A newly quarantined algorithm-validation period is data that a new
candidate's development process has not used; it is not newly unobserved market
history. Its results must retain the history-reused label. This permits further
research now without pretending that another date split erases earlier work.

Distinguish three outcomes: a useful exploratory lead, an independently
verified implementation, and prospective confirmation. The first two can be
achieved with existing data. None of them implies profitable implementation or
guarantees that repeated searching will eventually produce a valid signal.

## Append-only trial ledger

Before generating each new score, append an immutable specification with:

- Study, wave, candidate and parent IDs; specification timestamp; code, input,
  checkpoint and protocol hashes; intended target, horizon and decision time.
- Exact features, source availability and lags; preprocessing; fit schedule;
  hyperparameters or a finite, training-only selection rule; random seeds.
- Required controls, scored origins, date roles, target maturity/purge rules,
  primary loss, minimum useful effect and stability requirements.
- Every intended contrast and its multiplicity-family ID; whether the dates
  were inspected in older studies, this candidate's development, or both.
- After execution: all contrasts, counts, uncertainty, period results,
  verification status and artifact hashes, including unsuccessful outcomes.

Backfill known frozen studies, including the eight round-two contrasts and the
46 modeling/reference contrasts, without altering their original reports.
Inventory other discoverable historical trials separately. Label any incomplete
retrospective inventory explicitly; do not claim that a new ledger captures
unlogged experiments. The ledger must be complete prospectively from this
iteration onward.

A changed feature, target, horizon, model, outer evaluation window, selection
rule, or post-score hyperparameter creates a new entry. Failed fits and missing
data remain entries, with unevaluable registered tests assigned p=1. An exact
rerun with identical hashes is verification, not a new experiment. A numerical
repair requires an archived explanation and tests; a repair after scores must
preserve the original result and receive a new version. Record every inner
tuning configuration, even when honest nested selection makes the resulting
selection policy the single outer hypothesis.

## Temporal development and replay

Recommended working roles are 2011–2018 for development, 2019–2021 for screening,
and 2022–2025-10-10 for a candidate-frozen historical check. Do not inspect the
last slice for a new candidate until its specification and tests are frozen.
Existing aggregate knowledge of those years must still be disclosed. Once its
scores inform a revision, that revision cannot describe the same slice as
algorithm-unseen.

For the actual scored experiment, prefer a predeclared walk-forward replay:
outer test blocks 2016–17, 2018–19, 2020–21, 2022–23 and 2024–2025-10-10. Each
outer block trains only on earlier completed labels. Any permitted tuning uses
chronological inner splits inside that training history. Record the selected
configuration at every outer origin or refit. Thus a changing model is evaluated
as a causal selection policy, not as retrospectively chosen best parameters.

Use the actual session calendar and declare the earliest valid block before
scores if a target lacks the initial 500 training observations. Overlapping
targets must mature before training, calibration, retrieval or ensemble updates.
Fit normalization and projections inside the same eligible training sample.
Never remove sessions to manufacture consecutive context windows. Keep paired
model/control origins identical within every target family; publish exclusions.

Do not repeatedly interpret the final historical slice as fresh confirmation.
Further waves remain allowed: freeze the revised policy, replay it honestly,
append its results and treat them as increasingly adaptive historical research.
Preserve successful policies for subsequent forward accrual without reopening
the repository's protected phase during this search.

## Multiplicity and sequential decisions

Maintain cumulative Holm-adjusted results across all registered comparable
contrasts, retaining earlier unsuccessful trials. Also report each fixed wave's
family and the complete known search count. A change of target is a research
opportunity, not a way to reset the ledger.

For a declared sequence of prospectively specified test waves, a practical
additional error-budget rule is wave w receiving alpha = .05/[w(w+1)], with
Holm inside that wave. Its total allocation is .05 without requiring a fixed
number of waves. Record the wave index before scores. This guards repeated
testing only to the extent that the underlying p-values are valid; it does not
repair adaptive reuse of an already inspected historical sample. Therefore
historical leads remain exploratory even when they clear this rule.

Use paired dependence-aware inference, preserving the existing conservative
combination of block lengths 21/63/126 and HAC126 unless a target-specific
change is specified before scores. Do not treat overlapping daily outcomes as
independent observations. Set bootstrap resolution before running: for M
contrasts in a wave, require 1/(B+1) to be no greater than one tenth of that
wave's smallest Holm threshold, alpha/M. A fixed 4,999-draw bootstrap can become
incapable of clearing a stricter sequential threshold; more draws address
resolution, not statistical power or model quality. Use a fixed seed and
chunked computation. If that budget is impractical, report unresolved inference
rather than changing thresholds after results.

## Actionable gates and controls

1. **Valid construction:** input timing, training selection, paired samples,
   positivity where needed, and an independent reconstruction all pass.
2. **Useful effect:** predeclare a minimum of 1% relative improvement in the
   primary proper loss unless the target-specific protocol justifies another
   value before scores. Require improvement in each of the three existing
   stability periods; report annual and nonoverlapping-horizon phases as well.
3. **Appropriate control:** volatility candidates must beat the strong
   HAR-IV/leverage/stress benchmark. A new feature or representation must also
   beat a model with the same fitting objective and information set minus that
   addition. Index returns need zero-return and expanding historical-mean
   controls; direction probabilities need a training-only frequency control and
   a proper probability loss. Accuracy against 50% alone is insufficient.
4. **Honest uncertainty:** report the effect, uncertainty envelope, cumulative
   multiplicity result, wave result and power limitations together. Significant
   deterioration is evidence of harm in this setting, not an improvement lead.
5. **Correct attribution:** adding existing benchmark features to a weak model
   is not evidence that the new model contributes information. Memory must beat
   global recalibration; a representation must beat observable-state retrieval;
   dynamic weighting must beat its fixed ensemble control.

Advance a passing historical lead to a frozen forward forecast file and a
separately specified practical decision test. Continue other registered
experiments when an arm fails. If a wave finds nothing useful, record that
outcome and design the next wave from a stated mechanism or data question;
never rename failure, relax a frozen gate, or hide earlier trials to satisfy
the desire for a positive result.
