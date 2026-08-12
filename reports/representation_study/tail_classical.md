# Extended-history tail ranking: classical models

The supervised benchmark uses current log RV and trailing 5-/22-session means
of log RV, matching the earlier transition study. It is not the standard HAR
`log(mean variance)` transform.

Origins: 5,592 (2002-01-02 through 2025-10-10); event rate 13.2%.
The 736 positive origins map to 222 trigger sessions and 118 registered transition episodes. The positive-origin count is not treated as the independent-event denominator.
At a 13.2% five-session crossing rate this target measures recurrent movement out of a calm band—threshold proximity—not rare-crisis anticipation.

The phase mean averages the five fixed offsets of this five-session target—not the 21 offsets used by earlier 21-session designs.

| model | AUC mean [min, max] | lift mean [min, max] | top-decile event rate |
|---|---:|---:|---:|
| benchmark | 0.8704 [0.8642, 0.8775] | 4.804x [4.623, 5.030] | 63.2% |
| hmm_augmented | 0.8714 [0.8653, 0.8770] | 4.668x [4.489, 4.854] | 61.4% |
| hmm_platt | 0.8294 [0.8214, 0.8346] | 4.587x [4.418, 4.823] | 60.4% |
| rv_percentile | 0.8111 [0.7943, 0.8219] | 3.501x [2.855, 4.065] | 46.1% |

Post-result leave-one-episode-out jackknife intervals (118 transition episodes):

These are positive-episode influence intervals: each replicate removes one
threshold-trigger episode while all negative origins remain fixed. They do not
capture negative-origin or residual serial-score sampling variability and are
not full cluster-robust standard errors.

| model | AUC 95% interval | lift 95% interval |
|---|---:|---:|
| benchmark | [0.8410, 0.8997] | [4.263, 5.345]x |
| hmm_augmented | [0.8425, 0.9002] | [4.335, 5.000]x |
| hmm_platt | [0.7927, 0.8660] | [4.178, 4.995]x |
| rv_percentile | [0.7840, 0.8382] | [3.102, 3.899]x |

Paired episode-clustered differences (candidate minus baseline):

| comparison | AUC delta [95% interval] | lift delta [95% interval] | top-decile rate delta [95% interval] |
|---|---:|---:|---:|
| benchmark_minus_rv_percentile | +0.0593 [+0.0432, +0.0753] | +1.304x [+0.820, +1.787] | +17.1% [+10.7%, +23.6%] |
| hmm_augmented_minus_benchmark | +0.0010 [-0.0021, +0.0041] | -0.136x [-0.494, +0.221] | -1.8% [-6.5%, +2.9%] |
| hmm_platt_minus_benchmark | -0.0410 [-0.0620, -0.0199] | -0.218x [-0.504, +0.069] | -2.9% [-6.6%, +0.9%] |

The protocol attempted scoring in 2001, but the frozen 400 completed calm-label minimum delayed the first eligible fold to 2002. Thus 1999-2001 is training-only; the forward scoreboard captures 2002 and 2008, not the onset of the 2000 transition.

Adding the calibrated HMM state changed AUC by +0.0010 and top-decile lift by -0.136x. That is negligible ranking gain, not evidence that the HMM beats the supervised benchmark.

The one-variable `rv_percentile` row was requested only after the headline was seen. It is a reviewer diagnostic for volatility-level persistence, not pre-specified confirmation.

The event and scoreboard were frozen before this run. Brier/log loss are not verdict metrics here.
