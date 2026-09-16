# Broad modeling and memory experiment

Exploratory on previously inspected history. Eleven models, two horizons, 32 prespecified comparisons.
Positive improvement means lower QLIKE. All arms use the same target and scoring origins.

| model | h1 improvement | h5 improvement | h1 adjusted p | h5 adjusted p |
|---|---:|---:|---:|---:|
| gamma | -0.88% | -13.46% | 1.0000 | 0.0320 |
| adaptive_ols | -1.39% | -9.37% | 1.0000 | 0.1680 |
| adaptive_gamma | -1.52% | -22.05% | 1.0000 | 0.1998 |
| ridge_gamma | -0.87% | -10.93% | 1.0000 | 0.0434 |
| spline_gamma | -3.36% | -23.89% | 0.7632 | 0.3068 |
| global_calibration | -0.09% | -1.70% | 1.0000 | 1.0000 |
| observable_memory | +0.04% | -2.68% | 1.0000 | 0.6649 |
| latent_memory | -0.09% | -2.10% | 1.0000 | 1.0000 |
| equal_ensemble | +0.03% | -9.24% | 1.0000 | 0.0638 |
| dynamic_ensemble | +0.02% | -8.49% | 1.0000 | 0.0600 |

## Mechanism controls

Memory must beat global recalibration. Latent versus observable retrieval isolates the representation.
Dynamic weighting must beat equal weighting and the best component point estimate before an ensemble claim.

| candidate vs control | h | improvement | adjusted p | verdict |
|---|---:|---:|---:|---|
| adaptive_gamma vs gamma | 1 | -0.64% | 1.0000 | INCONCLUSIVE |
| spline_gamma vs ridge_gamma | 1 | -2.47% | 1.0000 | INCONCLUSIVE |
| observable_memory vs global_calibration | 1 | +0.12% | 1.0000 | INCONCLUSIVE |
| latent_memory vs global_calibration | 1 | -0.00% | 1.0000 | INCONCLUSIVE |
| latent_memory vs observable_memory | 1 | -0.13% | 1.0000 | INCONCLUSIVE |
| dynamic_ensemble vs equal_ensemble | 1 | -0.01% | 1.0000 | INCONCLUSIVE |
| adaptive_gamma vs gamma | 5 | -7.57% | 1.0000 | INCONCLUSIVE |
| spline_gamma vs ridge_gamma | 5 | -11.68% | 1.0000 | INCONCLUSIVE |
| observable_memory vs global_calibration | 5 | -0.96% | 1.0000 | INCONCLUSIVE |
| latent_memory vs global_calibration | 5 | -0.40% | 1.0000 | INCONCLUSIVE |
| latent_memory vs observable_memory | 5 | +0.56% | 1.0000 | INCONCLUSIVE |
| dynamic_ensemble vs equal_ensemble | 5 | +0.68% | 1.0000 | INCONCLUSIVE |

## Uncertainty and stability

Candidate-minus-control gaps below: negative favors candidate. Intervals envelope block21/63/126 and HAC126;
they are nominal and not simultaneous. Adjusted p-values use Holm over all 32 core comparisons; the reference extension also reports the complete 46-comparison family.

| comparison | h | gap | 95% envelope | 2016–19 | 2020–22 | 2023–25 | verdict |
|---|---:|---:|---|---:|---:|---:|---|
| gamma vs baseline | 1 | +0.002827 | [-0.002273, +0.008484] | -0.00061 | +0.00054 | +0.01028 | INCONCLUSIVE |
| adaptive_ols vs baseline | 1 | +0.004483 | [-0.001665, +0.010901] | +0.00724 | -0.00571 | +0.01157 | INCONCLUSIVE |
| adaptive_gamma vs baseline | 1 | +0.004901 | [-0.003169, +0.013020] | +0.00725 | -0.00989 | +0.01757 | INCONCLUSIVE |
| ridge_gamma vs baseline | 1 | +0.002793 | [-0.001565, +0.007459] | -0.00005 | +0.00026 | +0.00966 | INCONCLUSIVE |
| spline_gamma vs baseline | 1 | +0.010820 | [+0.002322, +0.022072] | +0.00673 | +0.01982 | +0.00696 | INCONCLUSIVE |
| global_calibration vs baseline | 1 | +0.000282 | [-0.000212, +0.000838] | +0.00073 | -0.00009 | +0.00005 | INCONCLUSIVE |
| observable_memory vs baseline | 1 | -0.000118 | [-0.001979, +0.001743] | -0.00032 | -0.00283 | +0.00312 | INCONCLUSIVE |
| latent_memory vs baseline | 1 | +0.000293 | [-0.001587, +0.001937] | -0.00057 | +0.00077 | +0.00103 | INCONCLUSIVE |
| equal_ensemble vs baseline | 1 | -0.000093 | [-0.003549, +0.003610] | -0.00038 | -0.00544 | +0.00613 | INCONCLUSIVE |
| dynamic_ensemble vs baseline | 1 | -0.000076 | [-0.003480, +0.003597] | -0.00026 | -0.00539 | +0.00596 | INCONCLUSIVE |
| adaptive_gamma vs gamma | 1 | +0.002074 | [-0.005098, +0.009224] | +0.00785 | -0.01043 | +0.00730 | INCONCLUSIVE |
| spline_gamma vs ridge_gamma | 1 | +0.008028 | [-0.000768, +0.018484] | +0.00678 | +0.01956 | -0.00270 | INCONCLUSIVE |
| observable_memory vs global_calibration | 1 | -0.000400 | [-0.002353, +0.001567] | -0.00105 | -0.00274 | +0.00308 | INCONCLUSIVE |
| latent_memory vs global_calibration | 1 | +0.000011 | [-0.001683, +0.001620] | -0.00130 | +0.00086 | +0.00098 | INCONCLUSIVE |
| latent_memory vs observable_memory | 1 | +0.000411 | [-0.001560, +0.002358] | -0.00025 | +0.00360 | -0.00210 | INCONCLUSIVE |
| dynamic_ensemble vs equal_ensemble | 1 | +0.000017 | [-0.000171, +0.000218] | +0.00012 | +0.00005 | -0.00017 | INCONCLUSIVE |
| gamma vs baseline | 5 | +0.025128 | [+0.012039, +0.038335] | +0.04085 | +0.02787 | -0.00058 | INCONCLUSIVE |
| adaptive_ols vs baseline | 5 | +0.017489 | [+0.005050, +0.030524] | +0.02447 | +0.01311 | +0.01214 | INCONCLUSIVE |
| adaptive_gamma vs baseline | 5 | +0.041163 | [+0.011838, +0.071557] | +0.06826 | +0.02965 | +0.01449 | INCONCLUSIVE |
| ridge_gamma vs baseline | 5 | +0.020411 | [+0.008626, +0.032912] | +0.03072 | +0.02738 | -0.00206 | INCONCLUSIVE |
| spline_gamma vs baseline | 5 | +0.044604 | [+0.013380, +0.080946] | +0.08340 | +0.02425 | +0.01064 | INCONCLUSIVE |
| global_calibration vs baseline | 5 | +0.003174 | [-0.000517, +0.007364] | +0.00919 | -0.00241 | +0.00054 | INCONCLUSIVE |
| observable_memory vs baseline | 5 | +0.005000 | [+0.000580, +0.009828] | +0.01144 | +0.00091 | +0.00013 | INCONCLUSIVE |
| latent_memory vs baseline | 5 | +0.003930 | [-0.002053, +0.010388] | +0.01187 | +0.00116 | -0.00455 | INCONCLUSIVE |
| equal_ensemble vs baseline | 5 | +0.017252 | [+0.006684, +0.028404] | +0.03206 | +0.01286 | +0.00063 | INCONCLUSIVE |
| dynamic_ensemble vs baseline | 5 | +0.015859 | [+0.006094, +0.026110] | +0.02913 | +0.01247 | +0.00036 | INCONCLUSIVE |
| adaptive_gamma vs gamma | 5 | +0.016035 | [-0.004452, +0.037574] | +0.02741 | +0.00178 | +0.01507 | INCONCLUSIVE |
| spline_gamma vs ridge_gamma | 5 | +0.024193 | [-0.006966, +0.059928] | +0.05269 | -0.00314 | +0.01269 | INCONCLUSIVE |
| observable_memory vs global_calibration | 5 | +0.001826 | [-0.002340, +0.006485] | +0.00225 | +0.00332 | -0.00042 | INCONCLUSIVE |
| latent_memory vs global_calibration | 5 | +0.000756 | [-0.003516, +0.005394] | +0.00268 | +0.00357 | -0.00509 | INCONCLUSIVE |
| latent_memory vs observable_memory | 5 | -0.001071 | [-0.005803, +0.003979] | +0.00043 | +0.00025 | -0.00467 | INCONCLUSIVE |
| dynamic_ensemble vs equal_ensemble | 5 | -0.001394 | [-0.002924, -0.000041] | -0.00293 | -0.00039 | -0.00027 | INCONCLUSIVE |

## Memory and ensemble audit

- Memory records per query: 708–1238; exactly 64 neighbors per retrieval arm.
- Memory target must finish at least 22 QQQ sessions before query; lookback 1260 sessions. Global and local corrections use identical pools and 25% shrinkage.
- Monthly observable scaling and eight-dimensional PCA use only records eligible at the month's first scoring origin. No whitening; no future-fitted geometry.
- Ensemble weights use completed expert losses only: 252-session lookback, 63-session half-life, 10% equal-weight floor.
- Historical forecasts begin 2013-02-01 to warm up the memory. Scoring 2016-01-04–2025-10-10, targets at or before 2025-10-20.
- Same-history inputs do not create new external information; pretrained latent-cache exposure remains unknown.
- The target is daily GK-plus-overnight variance, not measured five-minute RV. No economic-return or equivalence claim is made.
- All annual, phase, MDE, component-loss and inference details are in metrics.json. Neighbor traces and fits are local under data/model_memory_study.
- Reproduce: make model-memory-study; independent check: make verify-model-memory-study.
