# HMM calibration and incremental-state protocol

Frozen after the comparison critique and before inspecting transition labels or
scores from 2025-11-03 onward. The machine-readable contract is
`regime_repair.yaml`.

This is not a pristine project-wide holdout: other variance experiments already
used these dates. It is a **target-specific holdout** because the transition
label, calibrated HMM forecast, and augmented supervised model have not been
scored there.

## Locked training and evaluation

- Use only the annual out-of-fold transition forecasts dated 2016 through 2024
  to fit calibration and both supervised models.
- Fit the two-state HMM once through 2024-12-31. During evaluation, update only
  its forward filter; do not refit parameters.
- Evaluate calm origins from 2025-11-03 through 2026-08-04 whose full next-five-
  session targets are present by 2026-08-11.
- Keep the stress threshold fixed at the 80th percentile of log RV through
  2024-12-31.

## Comparisons

First, fit one Platt calibrator to the clipped logit of the raw HMM five-session
exceedance probability. Calibration succeeds only if it lowers both average
five-phase Brier and log loss versus the raw HMM.

Second, refit the supervised logistic benchmark on exactly the same out-of-fold
rows available to the augmented model. The benchmark uses current, five-session,
and 22-session log RV. The augmented model adds the Platt-calibrated HMM
probability as one fixed feature. Incremental state information succeeds only if
the augmented model lowers both losses versus the benchmark.

No calibration method selection, probability rescaling, threshold tuning, or
holdout refit is permitted after scoring.
