# TiRex-2 latent probe: full coordinates, controls, and episode uncertainty

This diagnostic follows the frozen latent contract in `representation_study.yaml`.
Scores were produced under protocol SHA `1a79821ae34f…`; the
final reviewed file is `4083043dd8a5…`.
Post-score changes were documentation/audit corrections and did not change latent
labels, fits, scores, seeds, or controls. No PCA or
post-result capacity selection is used: each representation is the complete
512-coordinate `stack_out_norm` state at zero-based token 63 from pinned TiRex-2
0.2.1 and checkpoint revision `05e5b26`.

Common scored origins: **5,592**, 2002-01-02 through
2025-10-10; event rate **13.2%** across
**118 positive-trigger episodes**. This 13.2% target is
recurrent five-session threshold proximity among calm origins, not a count of rare
independent regime breaks.

**Result:** every latent-only rung separates the actual label from its ten
capacity-matched controls on the frozen descriptive scorecard. None adds usable
ranking over the RV-history benchmark. Sparse k=1 essentially ties it, k=5 has an
interval including zero, k=10 is worse, and full ridge and the fixed MLP are
materially worse.

## Frozen probe ladder

| rung | latent AUC | control median / 95th | AUC selectivity (episode CI) | latent lift / control median | augmented AUC | Δ vs RV-history (CI) | Δ vs HMM+RV-history (CI) | descriptive heuristic met | exact p |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| full_ridge | 0.738 | 0.488 / 0.550 | +0.250 [0.203, 0.297] | 2.25× / 0.98× | 0.754 | -0.116 [-0.148, -0.084] | -0.117 [-0.149, -0.085] | yes | 0.091 |
| sparse_k1 | 0.815 | 0.519 / 0.550 | +0.296 [0.245, 0.348] | 3.68× / 1.18× | 0.869 | -0.001 [-0.006, 0.004] | -0.002 [-0.008, 0.005] | yes | 0.091 |
| sparse_k5 | 0.827 | 0.500 / 0.547 | +0.327 [0.269, 0.385] | 3.86× / 1.19× | 0.866 | -0.005 [-0.012, 0.002] | -0.006 [-0.013, 0.002] | yes | 0.091 |
| sparse_k10 | 0.827 | 0.503 / 0.535 | +0.324 [0.267, 0.381] | 3.72× / 1.09× | 0.862 | -0.008 [-0.015, -0.000] | -0.009 [-0.017, -0.001] | yes | 0.091 |
| small_mlp | 0.737 | 0.524 / 0.548 | +0.214 [0.170, 0.257] | 2.36× / 1.05× | 0.757 | -0.113 [-0.138, -0.089] | -0.114 [-0.139, -0.089] | yes | 0.091 |

The frozen heuristic is **not 5% randomization evidence**. With ten controls,
the smallest exact corrected Monte Carlo p-value is `(0+1)/(10+1) = 0.0909`.
`formal_evidence` is therefore false for every rung and selectivity is descriptive.
No controls were added after results were seen.

Actual-trigger and each control-seed episode have separate jackknife variance
components. A control positive origin is its proxy trigger, clustered within its
annual fold because the training-estimated Markov path resets yearly. The interval
conditions on all negative origins and captures positive-episode influence; it is
not a full serial-score or negative-origin uncertainty estimator. The exact method
was recorded before aggregate output in `latent_probe_uncertainty_method.md`.

## Five-phase range

| rung | latent AUC min–max | latent lift min–max | augmented AUC min–max | augmented lift min–max |
|---|---:|---:|---:|---:|
| full_ridge | 0.732–0.748 | 2.14×–2.39× | 0.749–0.765 | 2.31×–2.61× |
| sparse_k1 | 0.806–0.819 | 3.48×–3.83× | 0.864–0.876 | 4.49×–4.85× |
| sparse_k5 | 0.824–0.829 | 3.48×–4.27× | 0.859–0.873 | 4.21×–4.85× |
| sparse_k10 | 0.825–0.830 | 3.55×–3.90× | 0.857–0.868 | 4.08×–4.65× |
| small_mlp | 0.723–0.746 | 2.00×–2.60× | 0.734–0.775 | 2.61×–2.96× |

## Top-decile lift and incremental ranking

| rung | lift selectivity (episode CI) | augmented lift | Δ vs RV-history (CI) | Δ vs HMM+RV-history (CI) |
|---|---:|---:|---:|---:|
| full_ridge | +1.275× [0.760, 1.791] | 2.470× | -2.335× [-2.931, -1.738] | -2.198× [-2.636, -1.761] |
| sparse_k1 | +2.501× [1.870, 3.133] | 4.668× | -0.136× [-0.428, 0.156] | +0.000× [-0.297, 0.298] |
| sparse_k5 | +2.670× [2.123, 3.217] | 4.437× | -0.367× [-0.770, 0.036] | -0.231× [-0.526, 0.065] |
| sparse_k10 | +2.627× [2.242, 3.013] | 4.275× | -0.529× [-1.032, -0.027] | -0.393× [-0.887, 0.101] |
| small_mlp | +1.314× [0.804, 1.825] | 2.742× | -2.062× [-2.600, -1.525] | -1.926× [-2.254, -1.597] |

Control episode counts by seed: 4200: 112, 4201: 120, 4202: 127, 4203: 123, 4204: 122, 4205: 103, 4206: 123, 4207: 115, 4208: 114, 4209: 106.

## Sparse-coordinate stability

| rung | distinct coordinates | mean adjacent-fold Jaccard | most recurrent coordinates (fold count) |
|---|---:|---:|---|
| sparse_k1 | 9 | 0.565 | z499 (6), z386 (5), z410 (3), z412 (3), z442 (3), z046 (1), z280 (1), z356 (1), z401 (1) |
| sparse_k5 | 36 | 0.525 | z442 (10), z499 (9), z401 (8), z046 (7), z402 (6), z386 (6), z398 (5), z385 (5), z318 (4), z412 (4) |
| sparse_k10 | 70 | 0.503 | z401 (13), z442 (13), z499 (10), z046 (10), z491 (10), z402 (9), z398 (9), z297 (7), z386 (6), z318 (6) |

Coordinates are selected only from each fold's completed training labels by
absolute standardized event/non-event mean difference. Test labels and embeddings
never enter selection; the selected coordinates are scored on that year's disjoint
held-out common rows. All yearly identities and signed effects are retained in
`data/representation_study/latent_selected_dimensions.parquet`.

## Fixed MLP optimizer audit

| task | fits | converged before cap | hit 500-iteration cap | warnings |
|---|---:|---:|---:|---:|
| latent_actual | 24 | 3 | 21 | 21 |
| augmented_actual | 24 | 3 | 21 | 21 |
| control | 240 | 9 | 231 | 231 |

The cap was not increased post-result. Capped fits are fixed-optimizer
endpoints, not evidence that nonlinear capacity was exhausted.

## Interpretation limits

- The ten-control selectivity result is descriptive, not formal 5% evidence.
- Positive-episode influence is clustered; negative origins and residual serial
  score uncertainty are held fixed rather than fully resampled.
- Augmented probes use exactly the frozen benchmark/HMM common rows.
- TiRex-2 may have encountered market histories during pretraining, so this is a
  causal-origin diagnostic rather than a pristine corpus holdout.
- The fixed MLP does not license wider/deeper post-result searches.
