# Complete modeling and reference comparison

All 46 prespecified comparisons enter the final multiplicity correction. Positive improvement means lower QLIKE.
These historical experiments reuse inspected data. A statistical shortlist is not prospective confirmation.

| Model | h1 improvement | h5 improvement | h1 adjusted p | h5 adjusted p |
|---|---:|---:|---:|---:|
| gamma | -0.88% | -13.46% | 1.0000 | 0.0400 |
| adaptive_ols | -1.39% | -9.37% | 1.0000 | 0.2160 |
| adaptive_gamma | -1.52% | -22.05% | 1.0000 | 0.2590 |
| ridge_gamma | -0.87% | -10.93% | 1.0000 | 0.0546 |
| spline_gamma | -3.36% | -23.89% | 0.9540 | 0.4012 |
| global_calibration | -0.09% | -1.70% | 1.0000 | 1.0000 |
| observable_memory | +0.04% | -2.68% | 1.0000 | 0.8511 |
| latent_memory | -0.09% | -2.10% | 1.0000 | 1.0000 |
| equal_ensemble | +0.03% | -9.24% | 1.0000 | 0.0814 |
| dynamic_ensemble | +0.02% | -8.49% | 1.0000 | 0.0760 |
| nlinear | -3.19% | -6.85% | 0.9235 | 1.0000 |
| xlstm | -8.68% | -26.14% | 0.0092 | 0.6468 |
| moirai_univariate_gamma | -14.63% | -20.32% | 0.0092 | 0.0092 |
| moirai_augmented_gamma | -0.95% | -13.94% | 1.0000 | 0.0168 |

## Mechanism comparisons

| Candidate vs control | h | improvement | adjusted p | verdict |
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
| xlstm vs nlinear | 1 | -5.32% | 0.0168 | INCONCLUSIVE |
| moirai_augmented_gamma vs gamma | 1 | -0.07% | 1.0000 | INCONCLUSIVE |
| moirai_augmented_gamma vs moirai_univariate_gamma | 1 | +11.94% | 0.0092 | EXPLORATORY_SHORTLIST |
| xlstm vs nlinear | 5 | -18.05% | 1.0000 | INCONCLUSIVE |
| moirai_augmented_gamma vs gamma | 5 | -0.42% | 1.0000 | INCONCLUSIVE |
| moirai_augmented_gamma vs moirai_univariate_gamma | 5 | +5.31% | 1.0000 | INCONCLUSIVE |

The neural pair uses a fixed small CPU adaptation and a matched linear control. Moirai summaries receive causal Gamma calibration; exponentiated quantiles are not presumed to be mean forecasts.
Moirai and TiRex pretraining overlap with market history is unknown. Timer-S1 and the TimeCopilot orchestration service are reviewed separately, not counted as fitted models.
All periods, annual slices, nonoverlapping phases, uncertainty intervals and nominal minimum detectable effects are retained in combined_metrics.json.
