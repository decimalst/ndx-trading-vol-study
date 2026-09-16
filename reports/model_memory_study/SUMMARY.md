# Modeling, memory and reference-paper experiments

The existing HAR-IV/leverage/stress benchmark remains the supported choice.
None of the 14 alternatives qualified as an improvement over it. This result
covers 15 models including the benchmark, two forecast horizons, and 46 fixed
comparisons on 2,458 common origins per horizon, 2016-01-04 through 2025-10-10.
It is evidence about these specific configurations on reused history, not a
claim that every possible model or representation must fail.

## What we learned

Positive percentages below mean lower QLIKE forecast loss than the existing
benchmark; negative percentages mean worse performance.

| Selected approach | One session | Five sessions |
|---|---:|---:|
| Direct Gamma objective | -0.88% | -13.46% |
| Observable-state error memory | +0.04% | -2.68% |
| TiRex latent error memory | -0.09% | -2.10% |
| Dynamic ensemble | +0.02% | -8.49% |
| Matched NLinear neural control | -3.19% | -6.85% |
| Small xLSTM adaptation | -8.68% | -26.14% |
| Univariate Moirai with Gamma calibration | -14.63% | -20.32% |
| Baseline inputs plus Moirai, fitted with Gamma | -0.95% | -13.94% |

The crucial Moirai comparison holds the Gamma objective and historical training
rows fixed: adding its forecast summary changed loss by **-0.07% at one session
and -0.42% at five sessions**, with adjusted p=1 for both. That provides no
incremental-signal evidence. Similarly, latent retrieval failed to improve on
observable retrieval or ordinary global recalibration.

The only positive comparison clearing the complete statistical gate was adding
our established benchmark inputs to univariate Moirai at one session:
**+11.94%, adjusted p=0.0092**. This establishes the value of the existing inputs
within that comparison; it is not a new signal supplied by Moirai.

The machine-readable label `INCONCLUSIVE` means that a comparison did not
qualify for an improvement shortlist. It does not mean every loss difference
is statistically indistinguishable from zero. Several deteriorations are
statistically significant after correcting all 46 tests, including one-session
xLSTM versus the benchmark and its matched linear control, five-session Gamma
versus the benchmark, and both univariate Moirai horizons.

## Methods and controls

The core experiment tests changed fitting objectives, recency weighting,
regularization, smooth nonlinear functions, global calibration, observable and
latent retrieval, and equal and dynamically weighted ensembles. It uses the
same baseline inputs throughout. Nonlinear functions have a matched regularized
linear control; adaptive models have nonadaptive controls; memory has global
calibration and observable-state controls; dynamic weighting has equal weights
and the constituent models for comparison.

Training uses completed targets only. Core and calibration models refit monthly.
Historical forecast-error memory stores predictions actually made at historical
origins, waits an additional 22 trading sessions after target completion, and
fits its scaling/PCA geometry on eligible historical records. Ensemble weights
use only completed historical expert losses. No random train/test split,
future-fitted transformation, validation-based choice, or after-score tuning
was used. The old protected phase and 97 existing protocol/report artifacts
remain unchanged.

The xLSTM pair uses 22-session input windows, a fixed small CPU architecture,
12 training epochs, and annual fits using up to 1,500 eligible windows. Before
any neural fit, the initial warm-up date was explicitly moved from 2013-02-01
to 2013-02-07: the former had only 496 complete windows and the latter was the
first date meeting the unchanged 500-window requirement. All scored origins
remain intact. The initial failed eligibility check and four omitted warm-up
origins are retained in the audit.

QLIKE is evaluated on daily high/low/open/close plus overnight variance, not
five-minute realized variance. Inference uses paired block bootstrap at 21,
63 and 126 sessions, HAC126, and Holm correction across all 46 comparisons.
Improvement additionally requires at least 1% lower loss and favorable signs
in each of three fixed periods. Intervals are nominal, not simultaneous; the
simulation calibration does not certify their exact coverage on market data.

## How the supplied references were applied

- **Moirai 2.0:** ran the pinned official small checkpoint in a separate
  environment. Each forecast receives exactly 512 past log-variance observations.
  Its quantile summaries enter causal univariate or augmented Gamma models;
  exponentiated medians are not assumed to be conditional means.
- **xLSTM-Mixer:** ran an explicitly adapted small CPU model and matched NLinear
  control. This is not a reproduction of the paper's architecture or benchmark.
- **TSLib:** reviewed as a source of implementations and controls. Its generic
  benchmark settings were not substituted for this repository's methodology.
- **TimeCopilot:** reviewed its orchestration and ensemble approach. Its package
  and LLM-driven model selection were not run as an extra forecasting model.
- **Timer-S1:** reviewed its 8.3-billion-parameter checkpoint. It supports CPU
  inference, but its approximate BF16 weight footprint alone exceeds the
  observed available memory. The large-model run was deferred; no Timer-S1
  forecasts were produced.

Moirai and TiRex were released/trained after parts or all of this historical
sample. Their origin-causal local inputs do not establish absence of pretraining
exposure to market history. All foundation-model findings therefore remain
exploratory.

## Results, verification and reproduction

- [Complete results and all mechanism comparisons](combined_results.md)
- [All 46 estimates, uncertainty, periods, years and sampling phases](combined_metrics.json)
- [Comparison figure](model_comparison.png) and [exportable PDF](model_comparison.pdf)
- [Moirai and xLSTM primary-source review](REFERENCE_REVIEW.md)
- [TSLib, TimeCopilot and Timer-S1 review](LIBRARY_AND_LARGE_MODEL_REVIEW.md)
- [Core independent audit](verification.json)
- [Complete reference and 46-comparison audit](reference_verification.json)
- [Moirai full checkpoint replay](moirai_independent_verification.json)
- [Pre-run core checks](pre_run_checks.txt), [pre-run reference checks](reference_pre_run_checks.txt), and [full final suite](final_tests.txt)

The full suite passes **526 tests**. The core independent verifier reconstructs
54,076 scored forecasts, all 629,248 neighbor records, all 32 core inference
comparisons, and all historical component fits. Moirai's independent replay
reconstructs all 3,968 input windows and all 178,560 native quantile values
exactly.

The reference-model audit also passes: it checks all **73,740 scored forecasts**
in the complete panel, all 46 combined comparisons, all 472 Moirai calibration
fits, and all 12,752 historical neural forecasts. All 26 saved annual neural
models replay their predictions exactly. The first annual pair was independently
retrained to identical parameter hashes; later training was checked through
independently reconstructed eligibility and transforms plus saved-state replay.
The independent neural implementation preserves the original float32 window
layout because changing strides can select different reduction kernels. No
model, forecast, protocol or numerical tolerance was changed to achieve replay.

The repository-wide linter reports one cosmetic import-order finding in the
now hash-frozen core verifier test. Its bytes were retained to preserve the
pre-run audit; no numerical or test failure is being concealed by that finding.

Use `make verify-model-memory-study` and `make verify-model-memory-reference`
to verify the local artifacts. `make model-memory-study` and
`make model-memory-reference` reproduce the scoring stages from the pinned
local inputs. Separate `model-memory-neural` and `model-memory-moirai` targets
cover extraction. The Moirai environment, raw model weights, detailed forecasts,
neighbors, neural states and third-party-derived inputs remain local under
ignored data directories. Their revisions, hashes and execution evidence are
retained in the manifests and audit files.
