# Eidos-derived context-noise robustness diagnostic

This is an adaptation of Eidos Appendix A.1.2, not an evaluation of Eidos.
All tests and corruption grids were frozen before model inference.

## Result

Under each model's native preprocessing, the foundation forecasts were less sensitive to injected context noise than the expanding HAR control. At maximum Gaussian noise, relative decile-grid CRPS approximation was 1.053 for Chronos-2, 1.032 for TiRex-2, and 1.212 for HAR. At 20% impulse contamination it was 1.314, 1.145, and 5.562, respectively.

TiRex-2 degraded less than Chronos-2 under impulse corruption; their paired interval excluded zero only at probabilities 0.15 and 0.20. No registered Gaussian comparison between the two foundation models excluded zero. The slight CRPS improvements at the lowest noise levels are retained as observed and should be read as finite-sample/noise-regularization behavior, not as a tuned forecasting gain.

Within the two foundation adapters, this makes surface-noise fragility an unlikely explanation for their earlier null. The HAR magnitude is not an apples-to-apples architectural contrast: unlike Eidos Appendix A.1.2, this adaptation did not impose one verified noisy-statistics renormalization pipeline across all three models. It does not establish a clean accuracy advantage: the diagnostic window was already used, and the corruption experiment measures stability rather than forecast value.

## Design

- 124 origins, every twentieth session from 2016-01-04 through 2025-10-15, forecasting the next session.
- The input is the trailing 1,024 log-RV observations. Gaussian and impulse paths use a hash-derived seed per origin and exact common random numbers across all models and intensities.
- Chronos-2 and TiRex-2 use locally cached pinned checkpoints. TiRex uses its checkpoint inference defaults, matching the existing univariate runner.
- HAR is expanding OLS on daily log variance and the log of trailing 5/22-session mean variance. It is fit only on clean history through the origin; only the forecast-origin state is corrupted.
- The model adapters receive the same raw corrupted log-RV arrays, then retain their native preprocessing. Chronos-2 and TiRex-2 may normalize internally; that behavior was not independently audited. HAR does not renormalize the corrupted origin state.
- The stored `crps` score is a decile-grid CRPS approximation: twice the trapezoidal integral of pinball loss over quantiles 0.1-0.9, normalized by that grid width. Relative values divide its mean at an intensity by the same model's clean mean. Bootstrap intervals are secondary paired model differences, not model-accuracy tests.

## Registered degradation curves

| model | corruption | intensity | mean decile-grid CRPS approximation | relative score |
|---|---:|---:|---:|---:|
| chronos_2_univariate | gaussian | 0.00 | 0.584572 | 1.0000 |
| chronos_2_univariate | gaussian | 0.20 | 0.582116 | 0.9958 |
| chronos_2_univariate | gaussian | 0.40 | 0.587627 | 1.0052 |
| chronos_2_univariate | gaussian | 0.60 | 0.598773 | 1.0243 |
| chronos_2_univariate | gaussian | 0.80 | 0.615583 | 1.0530 |
| har_univariate_expanding_clean_fit | gaussian | 0.00 | 0.596950 | 1.0000 |
| har_univariate_expanding_clean_fit | gaussian | 0.20 | 0.612415 | 1.0259 |
| har_univariate_expanding_clean_fit | gaussian | 0.40 | 0.637548 | 1.0680 |
| har_univariate_expanding_clean_fit | gaussian | 0.60 | 0.674042 | 1.1291 |
| har_univariate_expanding_clean_fit | gaussian | 0.80 | 0.723293 | 1.2116 |
| tirex_2_univariate | gaussian | 0.00 | 0.590211 | 1.0000 |
| tirex_2_univariate | gaussian | 0.20 | 0.587873 | 0.9960 |
| tirex_2_univariate | gaussian | 0.40 | 0.588072 | 0.9964 |
| tirex_2_univariate | gaussian | 0.60 | 0.594755 | 1.0077 |
| tirex_2_univariate | gaussian | 0.80 | 0.609363 | 1.0324 |
| chronos_2_univariate | impulse | 0.00 | 0.584572 | 1.0000 |
| chronos_2_univariate | impulse | 0.05 | 0.596201 | 1.0199 |
| chronos_2_univariate | impulse | 0.10 | 0.621185 | 1.0626 |
| chronos_2_univariate | impulse | 0.15 | 0.682872 | 1.1682 |
| chronos_2_univariate | impulse | 0.20 | 0.768392 | 1.3145 |
| har_univariate_expanding_clean_fit | impulse | 0.00 | 0.596950 | 1.0000 |
| har_univariate_expanding_clean_fit | impulse | 0.05 | 1.486915 | 2.4909 |
| har_univariate_expanding_clean_fit | impulse | 0.10 | 2.235495 | 3.7449 |
| har_univariate_expanding_clean_fit | impulse | 0.15 | 2.860718 | 4.7922 |
| har_univariate_expanding_clean_fit | impulse | 0.20 | 3.320027 | 5.5616 |
| tirex_2_univariate | impulse | 0.00 | 0.590211 | 1.0000 |
| tirex_2_univariate | impulse | 0.05 | 0.589262 | 0.9984 |
| tirex_2_univariate | impulse | 0.10 | 0.596624 | 1.0109 |
| tirex_2_univariate | impulse | 0.15 | 0.630080 | 1.0675 |
| tirex_2_univariate | impulse | 0.20 | 0.675933 | 1.1452 |

## Paired 95% moving-block intervals

A positive difference means model A degrades more than model B on relative CRPS.
The 22-session dependence choice maps to two origins on this twentieth-session sampling grid.

| corruption | intensity | model A | model B | A-B | 95% interval |
|---|---:|---|---|---:|---:|
| gaussian | 0.20 | chronos_2_univariate | har_univariate_expanding_clean_fit | -0.0301 | [-0.0480, -0.0122] |
| gaussian | 0.20 | chronos_2_univariate | tirex_2_univariate | -0.0002 | [-0.0104, 0.0103] |
| gaussian | 0.20 | tirex_2_univariate | har_univariate_expanding_clean_fit | -0.0299 | [-0.0473, -0.0124] |
| gaussian | 0.40 | chronos_2_univariate | har_univariate_expanding_clean_fit | -0.0628 | [-0.1040, -0.0230] |
| gaussian | 0.40 | chronos_2_univariate | tirex_2_univariate | 0.0089 | [-0.0082, 0.0268] |
| gaussian | 0.40 | tirex_2_univariate | har_univariate_expanding_clean_fit | -0.0716 | [-0.1097, -0.0347] |
| gaussian | 0.60 | chronos_2_univariate | har_univariate_expanding_clean_fit | -0.1049 | [-0.1726, -0.0414] |
| gaussian | 0.60 | chronos_2_univariate | tirex_2_univariate | 0.0166 | [-0.0049, 0.0377] |
| gaussian | 0.60 | tirex_2_univariate | har_univariate_expanding_clean_fit | -0.1214 | [-0.1862, -0.0605] |
| gaussian | 0.80 | chronos_2_univariate | har_univariate_expanding_clean_fit | -0.1586 | [-0.2551, -0.0657] |
| gaussian | 0.80 | chronos_2_univariate | tirex_2_univariate | 0.0206 | [-0.0046, 0.0446] |
| gaussian | 0.80 | tirex_2_univariate | har_univariate_expanding_clean_fit | -0.1792 | [-0.2754, -0.0890] |
| impulse | 0.05 | chronos_2_univariate | har_univariate_expanding_clean_fit | -1.4710 | [-2.0141, -1.0049] |
| impulse | 0.05 | chronos_2_univariate | tirex_2_univariate | 0.0215 | [-0.0239, 0.0622] |
| impulse | 0.05 | tirex_2_univariate | har_univariate_expanding_clean_fit | -1.4925 | [-2.0268, -1.0273] |
| impulse | 0.10 | chronos_2_univariate | har_univariate_expanding_clean_fit | -2.6822 | [-3.4883, -1.9586] |
| impulse | 0.10 | chronos_2_univariate | tirex_2_univariate | 0.0518 | [-0.0075, 0.1074] |
| impulse | 0.10 | tirex_2_univariate | har_univariate_expanding_clean_fit | -2.7340 | [-3.5554, -2.0108] |
| impulse | 0.15 | chronos_2_univariate | har_univariate_expanding_clean_fit | -3.6241 | [-4.6095, -2.7624] |
| impulse | 0.15 | chronos_2_univariate | tirex_2_univariate | 0.1006 | [0.0359, 0.1668] |
| impulse | 0.15 | tirex_2_univariate | har_univariate_expanding_clean_fit | -3.7247 | [-4.7140, -2.8594] |
| impulse | 0.20 | chronos_2_univariate | har_univariate_expanding_clean_fit | -4.2472 | [-5.3300, -3.3118] |
| impulse | 0.20 | chronos_2_univariate | tirex_2_univariate | 0.1692 | [0.1094, 0.2438] |
| impulse | 0.20 | tirex_2_univariate | har_univariate_expanding_clean_fit | -4.4164 | [-5.5392, -3.4556] |

## Interpretation limits

The diagnostic asks how these particular model pipelines respond to controlled context corruption. Eidos re-normalizes each corrupted sequence using its noisy statistics; this adaptation retained model-native preprocessing and therefore does not reproduce that step uniformly. Cross-model HAR-versus-foundation magnitudes mix architecture with preprocessing. It does not establish that the Eidos architecture would be robust, nor does it create a new clean predictive claim. The 2016-2025 window was already diagnostic, and both foundation checkpoints may have encountered financial histories during pretraining.
