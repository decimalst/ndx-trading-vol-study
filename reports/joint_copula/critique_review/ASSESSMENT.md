# Assessment of the supplied joint-copula critique

Reviewed 2026-09-09. This is an additive, post-result assessment. The original
protocol, results, qualification gates and publication artifacts remain unchanged.
No new model was fitted, selected or recalibrated for this assessment.

The critique identifies a real limitation: the saved individual return forecasts
show calibration departures. The stronger assertion that marginal error explains
most of the joint-score improvement is not established. In particular, the
simulation's reported expected gain is not a ceiling on the gain possible in the
actual data.

The defensible result is **a verified historical joint-score improvement under
imperfect shared marginal forecasts, with the mechanism unresolved**. This remains
an exploratory result on reused history. It does not demonstrate better standalone
volatility forecasting, uniquely identify the true dependence distribution, or
establish trading usefulness. My earlier plain-language description of the result
as an improved relationship forecast needed this qualification.

## What this assessment reproduced

The supplied critique reports independent checks of all densities, inference,
timing and provenance. This assessment did not repeat its bootstrap or source-data
reconstruction. It authenticated the saved panel and metrics, independently
reconstructed the t8-versus-Gaussian contrast using SciPy distributions, and
computed the residual descriptions below from all 2,462 scored pairs. Each phase's
mean contrast agrees with the original result within 1e-14.

Standardized coordinates are `(return - predicted mean) / sqrt(0.75 * predicted variance)`.
Their reference distribution is t8, whose variance is 8/6, not one.

| Description | QQQ | SPX | Reference |
| --- | ---: | ---: | ---: |
| Residual variance, population convention | 1.487963 | 1.354078 | 1.333333 |
| Below predicted 2.5th percentile | 100 / 2,462 = 4.06% | 83 / 2,462 = 3.37% | 2.5% |
| Above predicted 97.5th percentile | 47 / 2,462 = 1.91% | 25 / 2,462 = 1.02% | 2.5% |

These observed coverage departures support the critique's calibration concern.
They are post-result descriptions, without a new dependence-adjusted significance
test. The lower-versus-upper imbalance also means a symmetric variance adjustment
alone is not assured to solve the problem.

Other reproduced descriptions:

- Squared-coordinate correlation is 0.761413. The critique's model-implied values
  of 0.68 and 0.60 were not independently reproduced in this assessment.
- The 625 days with both absolute coordinates below 0.53 are 25.39% of the sample
  and contribute 43.01% of the signed total score improvement.
- The stated threshold of either absolute coordinate above four identifies
  **27 days**, rather than the critique's 25. They contribute 10.31% of the gain.
- Removing the lowest and highest 1% of daily score gains, separately, gives a
  mean gain of 0.029778 nats versus 0.030834 untrimmed. The gain survives this
  check, although it is not literally unchanged. This removes 24 observations
  from each end of the gain distribution; other trimming definitions may differ.

Contributions are signed score sums, not causal attributions. A t copula changes
the central density as well as the tails, so substantial gain on central days
does not rule out a dependence explanation.

Four generated checks covering scale, tail counts, contribution arithmetic and
invalid inputs were written before the reconstruction. The missing-implementation
failure and subsequent passing run are retained in `TEST_RED.log` and
`TEST_GREEN.log`. Exact results and input hashes are in `DIAGNOSTICS.json`.

## Why the simulation is not a ceiling

The critique reports an expected candidate-minus-control loss of -0.011 when a
fitted t8 model generates the data, versus about -0.031 observed. Its program,
seeds, sample sizes and meaning of the reported standard deviation were not
provided, so those simulation numbers have not been independently reproduced.
It is also unclear whether refitting and scoring used separate simulated samples
and reproduced the original monthly training and evaluation design.

Even if those numbers are correct, the bound does not follow. For the actual
conditional law P and the two complete density forecasts qT and qG,

`E_P[loss_T - loss_G] = KL(P || qT) - KL(P || qG)`.

Under a simulation assuming P equals qT, an oracle Gaussian comparison yields
`-KL(qT || qG*)`. That is a benchmark for the assumed distribution and fitting
design. It does not bound the contrast under another distribution, nor does a
simulated mean bound every realized sample. The underlying log-score/KL identity
is described by [Gneiting and Raftery (2007)](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).

A larger observed gain could reflect marginal errors, different dependence,
asymmetry, changing regimes, estimation effects or a combination. This simulation
could motivate a model check; it cannot isolate which explanation dominates.

Shared marginal density terms cancel from the existing paired score contrast,
but their probability transformations still enter both copulas. Consequently,
marginal error can influence the measured dependence-model advantage. Conversely,
changing only the copula does not repair either individual forecast's marginal
distribution. The proposed common-scale explanation is plausible, not identified.

## A follow-up that can distinguish the explanations better

A separately preregistered experiment should cross two marginal systems with
the two dependence models:

| | Gaussian dependence | Fixed t8 dependence |
| --- | --- | --- |
| Original individual forecasts | Existing control | Existing candidate |
| Prespecified recalibrated individual forecasts | New matched control | New matched candidate |

Use identical forecast dates, available inputs and training schedules in all
cells. Learn recalibration from mature, previously issued forecasts or nested
chronological training folds. Do not calibrate from the entire inspected sample
or residuals whose outcomes trained their own forecasts. Refit both dependence
models under each marginal system. A fixed scale-only arm could probe the
narrower scale hypothesis; an asymmetric arm could address the tail imbalance.
Any extra choices and comparisons need their own declared multiplicity treatment.

Compare full normalized joint densities across marginal systems, including the
density factors introduced by recalibration. For example, replacing F with G(F)
requires the density factor g(F)f. Copula-only scores do not provide a fair
comparison across different marginal systems.

Prespecify the change in the t8-minus-Gaussian gap, together with held-out
individual log scores and calibration diagnostics. If individual calibration
improves and the copula gap shrinks, that supports sensitivity to marginal error.
If the gap persists, it supports robustness to that particular correction.
Neither outcome proves a unique causal mechanism, and the simulated -0.011 should
not be used as a target or pass threshold.

The reused historical sample can support this mechanism diagnostic. Confirmation
of a newly selected forecasting system still requires subsequently untouched
outcomes. The calibration comparison should precede that confirmation; another
historical recalibration run has not been launched as part of this assessment.

## Reproducibility and chronology

The critique's preservation concern is valid. The frozen inventory contains 514
source/test files: 74 are tracked and 440 untracked in the current repository.
The new study files are untracked, and the last commit is from August 13. Hashes
detect changes but do not themselves retain the original bytes. A durable private
snapshot or scoped commit would strengthen reproducibility. The existing dirty
`.gitignore` and `Makefile` have been preserved. No commit or publication was made
for this review.

No joint-copula Makefile target was found. This affects discoverability and
convenience, rather than invalidating the saved numerical result.

The fixed eight degrees of freedom were inherited from the earlier tail-shape
study on reused history. This is not an independently confirmed modeling choice.
However, the earlier specification itself says eight was fixed before fitting
without a degrees-of-freedom search; inheritance alone does not show that eight
was optimized on those data.

Local artifacts record the full test gate completing at 06:20:08 UTC, the full
precheck completing at 06:20:09, the executable freeze at 06:20:10, registration
at 06:20:33 and the completed run at 06:20:54 on September 9. These support the
locally observed ordering but are not an external trusted timestamp. A short
elapsed interval alone is not evidence of post-result registration. Stronger
chronology attestation is a valid improvement for future experiments.
