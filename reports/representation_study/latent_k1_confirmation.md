# TiRex sparse-k=1: separate 99-control diagnostic

**Post-result formal-randomization verdict: FORMAL SELECTIVITY EVIDENCE.** This is not a pristine historical holdout: the single k=1 rung was registered after the earlier ten-control ladder had been seen. It does not change that frozen ladder or its verdict.

## Registered result

- Held-out origins: **5,592** across **24** annual forward folds; latest origin **2025-10-10**, before the sealed clean window.
- Actual sparse-k=1 phase-mean AUC: **0.8153**; top-decile lift: **3.68x**.
- 99-control AUC median / 95th percentile: **0.5202 / 0.5640**.
- Corrected exact randomization p: **0.0100** (0 controls at least as high as actual; attainable minimum **0.01**).
- Actual AUC phase range: **0.8058-0.8193**; lift range: **3.48x-3.83x**.

The exact p-value is the registered inference. There is one registered rung, so no ladder-wide family correction is invoked. Controls are first-order Markov paths estimated from completed fold-training labels; each control reselects its own coordinate using only its synthetic training labels.

## Was k=1 selected out of sample?

Yes. For every annual fold, all 512 dimensions were ranked on completed training labels, one coordinate was selected, and its ridge-logistic score was evaluated only on that year's held-out origins. The verifier reconstructs both actual and all 2,376 fold-by-control selections.

## Coordinate identity and stability

The 24 folds selected **9** distinct coordinates. The recurring coordinates were:

| coordinate | folds | fraction |
|---:|---:|---:|
| z499 | 6 | 25.0% |
| z386 | 5 | 20.8% |
| z410 | 3 | 12.5% |
| z412 | 3 | 12.5% |
| z442 | 3 | 12.5% |
| z046 | 1 | 4.2% |
| z280 | 1 | 4.2% |
| z356 | 1 | 4.2% |
| z401 | 1 | 4.2% |

This is a fold-specific selector, not evidence for one universal TiRex neuron. Signs below are oriented using the training-only event/non-event effect so positive values have a common event-facing interpretation.

## Held-out coordinate characterization

| held-out variable | valid folds | median Pearson [min, max] | median Spearman [min, max] |
|---|---:|---:|---:|
| log_rv_level | 24 | 0.573 [0.111, 0.724] | 0.576 [0.118, 0.739] |
| prior_fold_empirical_rv_percentile | 24 | 0.567 [0.072, 0.729] | 0.576 [0.119, 0.739] |
| prior_session_vxn_level | 17 | 0.755 [0.564, 0.867] | 0.734 [0.501, 0.879] |
| trailing_1_session_return | 24 | 0.039 [-0.079, 0.240] | 0.060 [-0.092, 0.237] |
| trailing_22_session_mean_log_rv | 24 | 0.804 [0.571, 0.945] | 0.802 [0.523, 0.944] |
| trailing_22_session_return | 24 | -0.368 [-0.754, 0.018] | -0.309 [-0.740, 0.068] |
| trailing_5_session_mean_log_rv | 24 | 0.856 [0.416, 0.910] | 0.848 [0.408, 0.928] |
| trailing_5_session_return | 24 | -0.056 [-0.270, 0.238] | -0.081 [-0.268, 0.256] |

Every correlation uses the coordinate chosen before that fold and only that fold's held-out origins. VXN is shifted on its complete source calendar before reindexing, so it is at least one full published session old; VXN rows before September 2009 remain missing.

## Scope

A positive result means the registered single-coordinate representation separates the actual transition-proximity label from the structurally matched synthetic controls under this reused historical sample. It does not establish incremental information beyond direct RV history, causal neuron semantics, checkpoint pretraining cleanliness, or a pristine out-of-sample discovery.

## Reproducibility

- Protocol SHA-256: `b51e0213fe8ff0f7c4fc4e1ef9b22f1ddbdae4d2ef899f58ae6d198de33c6680`
- Embeddings SHA-256: `a2c33b4cd907bb4456a21fda66a9bfeaa44d3e27cab00de1ba186780dc893f40`
- Classical origins SHA-256: `502ab51d1ec3e2d2d0115179a5edba23f5d06896418a49da5a6417aaf06f81db`
- Forecast artifact: `data/representation_study/latent_k1_confirmation_forecasts.parquet`
- Control artifact: `data/representation_study/latent_k1_confirmation_controls.parquet`
- Per-fold selection artifact: `data/representation_study/latent_k1_confirmation_selections.parquet`
- Held-out correlation artifact: `data/representation_study/latent_k1_coordinate_correlations.parquet`
