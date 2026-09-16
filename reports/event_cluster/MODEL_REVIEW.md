# Independent pre-fit model review

**No material model/feature-contract mismatch remains in the reviewed version.** This was a read-only review of the new model module, its synthetic tests, the draft protocol and design. The reviewer did not change those files or inspect historical source rows, event counts, fits, predictions or scores. The source/pipeline/runner admission and whole-study independent verifier remain separate required checks; this review is not empirical verification.

## Mathematical and timing checks

The [model implementation](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/event_cluster_models.py) consumes the six fitted columns exported by the [feature module](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/event_cluster_features.py). They are the five count/endpoint/reference/recency nuisances followed by raw `adjacency_fraction22=C/21`. The excess diagnostic is absent from fitting. Nuisance coordinates and adjacency are centered using training-only compensated means, with exact-constant centers taken from the first value and fixed scale one.

The original 26 coefficients and original training geometry are replayed without fitting. Curvature uses the saved monthly centers; mean/scale identity and finite coefficient/coordinate arithmetic are checked. For the new training stage, the caller supplies offsets made by applying that same month's saved original model to its own mature training rows. These are explicitly current-fit in-sample offsets, not a substituted history of issued forecasts. Application offsets replay the original issued probabilities within the declared tolerances. The pure model helper has no access to future application labels. Exact calendar membership and the selection of the correct saved fit are the pipeline's responsibilities, not claims inferred from this helper alone.

The nuisance objective is mean binary negative log likelihood with the fixed original offset, a free calibration intercept and five `.01`-penalized slopes. The inherited solver starts at six zeros, uses its fixed Newton/Armijo limits, and is followed by recomputation of its original full gradient and audit. Both-class training support is enforced. Application values do not determine centers, either fitted coefficient vector or the scalar offset.

The candidate objective fixes the nuisance training logit, adds a coefficient on centered raw adjacency and penalizes that coefficient by `.01*b²`. Its derivative is strictly increasing because the penalty contributes `.02`; the proposed bracket `±(mean(abs(z))+1)/.02` has strict opposite signs in exact arithmetic. The implementation checks finite endpoint calculations and signs, uses one fixed Brent solve when needed, and recomputes the original derivative before accepting its `1e-8` stationarity gate. Constant centered input and an exactly zero initial gradient use canonical zero. No alternate bracket, retry, clipping or fitted intercept is introduced.

Scalar likelihood and derivative calculations use the signed binary forms, retaining a representable residual when a sigmoid rounds to an endpoint. The scalar's checked-product and strictly positive finite-logit loss/residual rules are distinct from, and consistent with, the protocol's explicit reuse of the frozen nuisance numerical contract. The code does not impose an additional unused scalar-curvature admission test. The independent solver settings and probability/coefficient comparison tolerances in [the protocol](/Users/byrons/code/trading-vol/ndx-vol-experiment/event_cluster.yaml) remain separate from the strict saved-value replay requirements.

## Attribution and exact nesting

The constant-adjacency/varying-expectation regression confirms the pre-empirical correction: changes in `G` or the descriptive excess cannot drive a scalar fit when raw `C/21` is constant. The model retains the zero coefficient and copies the nuisance probability even if application adjacency differs from its training constant. This resolves the specific frozen-offset subtraction problem; it does not prove a structural clustering mechanism or remove every form of model misspecification.

For each query, an exactly zero nuisance correction copies the checked original saved probability. An exactly zero cluster correction copies the nuisance probability. Nonzero corrections use the corresponding finite parent logit plus correction. Thus the stated nesting concerns exact saved values, not merely equality to a newly evaluated sigmoid.

The initial test set checked these branches on naturally matching probabilities. Review requested one focused falsifier: a saved parent probability of `0.5+1e-13`, within replay tolerance, with balanced training labels and exactly zero offsets/corrections. Root added the test before any empirical run. It requires both new models to preserve that original value exactly; substituting a recomputed `0.5` would fail. No implementation correction was needed.

## Synthetic evidence and reviewed snapshot

An independent invocation of [the current model tests](/Users/byrons/code/trading-vol/ndx-vol-experiment/tests/test_event_cluster_models.py) passed **all 14 tests in 0.117 seconds**. Coverage includes separately solved nuisance equations, scalar derivative/reflection checks, current saved geometry replay, application-mutation invariance, constant slopes, constant raw adjacency despite varying expectation, probability identity, extreme logits, invalid inputs and false solver-success rejection. These are generated inputs only. Root's corresponding test output is retained separately.

The reviewed snapshot hashes below document the bytes inspected; they are not a claim that the still-prospective full study has already been frozen. Later presentation-only formatting requires the final registration to pin the actual resulting bytes.

| File | SHA256 at review |
|---|---|
| `src/event_cluster_models.py` | `26453771762d7cc1d880ff30cadc5699dcdf21fd5636a1adb0ce0068d5927fb6` |
| `tests/test_event_cluster_models.py` | `9d29ff83264fb4e18e5da5fe73abc91125cc4818a27f383f5d48f923a38f97c7` |
| `src/event_cluster_features.py` | `8f43b5fa694ceb001d7d33a59fed0aed12fa2b23a8a6fdf00554eaf9f3bc260e` |
| `event_cluster.yaml` | `824a9beda7cf293568dd2d16bbd3b76a8235f359a78c764147d75731c22b734c` |
| `reports/event_cluster/DESIGN.md` | `89e310189ccd3d31218fab952c5905037f388805e470fc1fd22e827999d65127` |

The three required candidate comparisons against original baseline, matched nuisance and original recent frequency are consistent with the design. This review supplies no new comparison, numerical diagnostic on historical fits, tolerance change or permission to skip the independent full-calendar and publication checks.
