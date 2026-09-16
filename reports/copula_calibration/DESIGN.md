# Crossed marginal calibration and dependence diagnostic

Specified 2026-09-09 before new market fits. Wave 28 adds seven registered
comparisons to the 151 inherited comparisons. This is an exploratory mechanism
diagnostic on reused history. The original study and its findings remain frozen.

## The comparison

| Individual return distributions | Gaussian dependence | Fixed t8 dependence |
| --- | --- | --- |
| Original issued marginals | `orig_gaussian` | `orig_t8` |
| Causally recalibrated marginals | `cal_gaussian` | `cal_t8` |

All cells preserve the same original issued means and variance parameters, use
the same mature historical forecast pairs, and score the same subsequent dates.
Both dependence models are refitted separately within each marginal system.
Unlike wave 27, even the original-marginal dependence models are fitted using
**historical issued errors**, rather than current-fit in-sample residuals.
Therefore their saved correlations and score gaps need not reproduce wave 27.
A matched-date bridge to the old result is descriptive only.

## Causal archive and schedule

Authenticate the original applications, targets, coverage, calendar, scored
panel, fit certificates and completion record, plus their original source
closure and old report hashes. Validate that the original three arms have
identical marginal parameters; retain one application per origin and join its
original outcome. A genuine issued forecast whose target crosses an old phase
boundary can enter the archive when mature even though it was not scored in
that phase. Read only the date index from the features file.

At each **original monthly fit date**, admit only complete issued pairs with
origins before the fit and with both target end and availability no later than
the previous full SPX session. Start with the first monthly fit having 252 such
pairs; every earlier whole month is warm-up. There is no mid-month startup.
Use the expanding archive thereafter. Fit calibration and all four dependence
parameters on exactly the same ordered pairs, and issue forecasts without the
current outcomes. No earlier OOS history or new marginal model is generated.

Retain the literal 2016–2019 and 2020–2025-10-20 phases and original calendar.
Warm-up and missing positions retain mask zero; global modulo-five offsets are
never renumbered. All cells require 505 observations per phase, 252 per fixed
later slice and 63 per phase/offset group. Exact support is checked only after
registration, rather than chosen to produce a favorable sample.

## A normalized marginal correction

For each asset let `F0` and `f0` be its original issued t8 CDF and density:

```
z = (y - mu) / sqrt(0.75 * h)
w = Phi_inverse(T8_CDF(z))
a = mean(w_archive)
b = sqrt(mean((w_archive - a)^2))
v = (w - a) / b
Fcal(y) = Phi(v)
log fcal(y) = log f0(y) + log phi(v) - log phi(w) - log b
```

The last line is the exact Jacobian and must enter every full-density score.
Identity parameters `a=0, b=1` reproduce the original marginal. Fit each asset
separately on the common mature pairs, with population scaling and fixed
`b > 1e-12`. Failure does not permit clipping, shrinking, dropping an asset or
changing the rule. Evaluate probability transforms with stable unclipped tails.

Gaussian dependence uses `w` or `v`; t8 dependence uses `z` or
`T8_inverse(Phi(v))`. Reuse the frozen bounded dependence solvers and numerical
certificates. The calibration corrects normal-score location and scale; it does
not guarantee correct skewness or other distributional shape. The resulting
density is normalized, but its variance need not equal the stored base `h` or
even be finite. This study evaluates density forecasts.

## Seven comparisons and interpretation

In fixed order, test the original t8-minus-Gaussian loss gap, the calibrated
gap, their difference, calibration's full-density gain under Gaussian
dependence, its gain under t8 dependence, and its individual marginal-score
gains for QQQ and SPX. Define interaction as **calibrated gap minus original
gap**; a positive value indicates attenuation of the t8 advantage.

Use shared full-calendar bootstrap draws across all seven contrasts, block
lengths 21/63/126, 399,999 draws, seed 20260910 with the inherited phase/block
offsets, and Bartlett HAC126. Take the maximum two-sided probability across
methods and both phases. A difference receives `adjusted_difference_detected`
only with the same strict sign in both phases and passing Holm7 at
`0.05/(28*29)` and cumulative Holm158 at `0.05`.

Report full joint and individual marginal scores, uncertainty, fixed slices
and offsets. Describe held-out PIT tails at 2.5%/97.5% and normal-score mean,
variance and skewness. An attenuated dependence gap is not evidence that
calibration improved: examine the held-out marginal scores and PIT descriptions
together. Persistence supports robustness to this correction only. No contrast
automatically promotes a predictive signal or identifies a unique mechanism.

## Critique reconciliation and execution

The subsequently supplied critique scripts resolve two apparent count
differences: its 25 extreme days used a quantile boundary near `max|z| > 4.043`,
whereas the assessment's literal `>4` rule counted 27. Its trimming uses strict
empirical-quantile inequalities; the assessment removed a fixed number from
each end. Those definitions need not retain the same observations.

The simulation program is now available. It generates pairs using the saved
per-origin correlations, then fits and scores constant dependence parameters
on the **same simulated sample**, with 30 repetitions. This mixture of
per-origin correlations and in-sample refitting does not reproduce the original
monthly forecasting procedure and supplies no bound on the actual score gap.
Its code is inspected, not rerun; no simulated value becomes a target or gate.

Freeze the new protocol, executable inventory, prewritten targeted-test
receipts, exact old source closure and old-report hash map before numerical
access, then durably register all seven comparisons. Independently verify
calibration, Jacobians, inverse tails, monthly maturity, shared masks,
dependence fits/certificates and inference. Relevant existing copula and
inference tests join the new tests; the unrelated entire test suite is not a
prerequisite. Any source, support, numerical, fitting, verification or
preservation failure retains the complete seven-comparison family with `p=1`.
No unrecorded retry or alternative model is allowed.

New private numerical outputs belong in
`data/model_memory_study/copula_calibration_wave28`; reports belong in
`reports/copula_calibration/predictive`. No old frozen artifact is edited.
