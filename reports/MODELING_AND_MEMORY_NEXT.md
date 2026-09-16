# Modeling and latent-memory options

Reviewed 2026-09-06. Status: proposed research design, not frozen or scored.
This note follows the round-two feature study and does not amend any existing
protocol, result, or evaluation gate.

## Why there is still a modeling question

The last round tested individual additions to a global, linear log-variance
model. It does not settle whether the existing data can help through changing
coefficients, a different fitting objective, or nonlinear similarity to past
episodes. These are ways to use existing information; none creates a new
external signal.

The repository has already tried a fixed boosted-tree model, calibrated HMM
states, TiRex latent ridge/sparse/MLP probes, and a residualized one-coordinate
probe. The time-safe GBM was about 4% worse overall, with its largest failures
concentrated in high-variance observations. The latent studies addressed stress
classification. The residualized probe projected out three HAR terms, not the
full implied-volatility/leverage/stress benchmark, and tested one coordinate.
Those results do not evaluate a memory of continuous-variance forecast errors.

## Priority 1: fit the quantity we evaluate

Keep the same predictors and train Gamma regression with a log link directly
on positive variance. The local environment already supports GammaRegressor.
Its data-fit loss is Gamma deviance, equal to twice QLIKE. A synthetic numerical
identity check passed; no market forecasting score was produced for this model.

For actual variance y and positive prediction v:

    QLIKE(y,v) = y/v - log(y/v) - 1
    GammaDeviance(y,v) = 2 * QLIKE(y,v)

The existing OLS models minimize squared log-variance error and subsequently
smear the prediction. Smearing is appropriate under its residual-distribution
assumptions but does not make the fitted coefficients minimize QLIKE. Keeping
the same regressors and changing only the objective isolates this distinction.
A direct Gamma model predicts mean variance without a separate smearing step.
Regularization, input scaling, and optimizer convergence need pre-run contracts.

Source: [scikit-learn Gamma regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html)
and [Gamma deviance](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_gamma_deviance.html).

## Priority 2: adapt gradually to changing relationships

Use the same HAR-IV/leverage/stress inputs with exponentially decaying training
weights, or a small state-space model for changing coefficients. A fixed
forgetting half-life should be the first experiment, with an equally weighted
fit as its control. If fitting log variance, smearing must use those same
weights. A half-life of roughly one trading year is an example to specify
before scoring, not a value selected from these data.

This tests whether persistent historical regimes dilute a currently useful
relationship. More rapid adaptation increases estimation noise, so it should
not be optimized by repeatedly inspecting the historical scoreboard.

Dynamic model averaging is a later extension: update weights on a small set of
experts from their matured out-of-sample forecasting losses. Compare any such
combination with both its component models and an equal-weight ensemble.

Source: [Wang et al. (2016), dynamic model averaging for realized volatility](https://www.sciencedirect.com/science/article/pii/S0378426615003647).
Its SPX evidence motivates a test; it does not establish an NDX improvement.

## Priority 3: a memory of forecast mistakes

Maintain records containing the state at a past origin, its genuine historical
forecast, the subsequently observed variance, and the time that outcome became
available. At the current origin, retrieve similar completed episodes and use
their errors for a small correction to the benchmark.

For variance, store a scale-free error ratio:

    r_s = actual_variance_s / forecast_made_at_s
    correction_t = (1 - lambda) + lambda * weighted_mean(retrieved r_s)
    forecast_t = baseline_t * correction_t

With nonnegative weights and 0 <= lambda <= 1 this preserves positivity.
Lambda shrinks the correction toward leaving the baseline unchanged.
The numerical choice, number of neighbors, history window, distance rule, and
any fixed bounds must be specified before scores. Under QLIKE, the optimal
constant multiplicative correction on a historical pool is its arithmetic mean
actual/forecast ratio. Simply exponentiating a mean log residual estimates a
different object.

Use errors from forecasts made by the historical forecasting procedure using
information available then. Do not create memory values by fitting today's
model to all its training data and treating its in-sample residuals as historical
forecast errors. Record the model/refit version; errors from evolving models
need not transfer just because their input states resemble one another.

The critical comparison is ordinary recalibration: a local memory can appear
useful merely because the baseline has a global scale bias.

| Arm | Forecast construction | What the comparison establishes |
|---|---|---|
| Benchmark | Existing strong model, unchanged | Reference |
| Global calibration | Shrunk mean OOS variance ratio over the eligible memory pool | Calibration without state matching |
| Observable-state memory | Same correction using neighbors in standardized market features | Whether matching states helps beyond calibration |
| Latent-state memory | Same memory values and correction using compressed TiRex states | Whether the representation improves retrieval |

Memory arms must beat global calibration as well as the benchmark. A latent
arm beating the benchmark alone cannot establish a benefit from latent geometry.
If a learned retrieval gate is added later, it needs a no-correction option and
must be compared with a fixed similarity-weighted retrieval rule.

The general idea has established precedent in
[ResMem (NeurIPS 2023)](https://papers.neurips.cc/paper_files/paper/2023/file/bf0857cb9a41c73639f028a80301cdf0-Paper-Conference.pdf).
The recent [RATL preprint, September 3, 2026](https://arxiv.org/html/2609.03937v1)
specifically studies causal retrieval of forecast-error trajectories. Its
financial Exchange benchmark deteriorates at longer horizons, so broad gains
on other domains are not evidence of a finance edge. Our ratio correction and
historical out-of-sample memory design would be adaptations, not a claimed
reproduction of either paper.

## Applying latent memory to this checkout

The existing cache has 6,668 rows by 512 dimensions, 1999-04-12 through
2025-10-10, with 2,458 origins from 2016 onward. It was inspected for schema
and coverage only in this review. There is no need to rerun TiRex extraction.

- Cache: `data/representation_study/latent_embeddings.parquet`.
- Manifest: `data/representation_study/latent_embeddings_manifest.json`.
- Checkpoint: frozen TiRex-2 revision `05e5b26db52bfb256f1ae1bdf785589850482de3`.
- Input: up to 2,048 prior log-variance observations; this is a representation
  of RV history, not a multi-asset state or an additional external source.

A modest compression, such as eight training-fitted principal components,
would make a first nearest-neighbor comparison tractable. This is a proposed
choice, not evidence that eight dimensions are optimal. Fit centering,
projection, and scaling only on eligible historical rows and transform both
stored keys and queries with the same map at each refit. Do not compare keys
stored under incompatible PCA versions. Whitening can amplify low-variance
noise and must be an explicit decision. The original latent protocol forbids
PCA; this separate study would leave that protocol untouched.

Origin-causal embeddings do not resolve possible exposure to market history
during foundation-model pretraining. Any historical latent result remains
exploratory, including when its nearest-neighbor search is causal.

## Proposed test order and gates

First test the fitting objective and gradual adaptation as separate changes to
the benchmark. Run the four-arm memory comparison with the existing benchmark
as a separately identifiable experiment. Combining all changes at once would
make a gain difficult to attribute. More complex gates or mixtures come after
the individual mechanisms demonstrate value.

Before running, complete synthetic and independent checks for:

1. Historical predictions and memory values use only information available at
   their own origin; a five-session error enters memory only after all five
   outcomes finish, with a fixed additional retrieval gap.
2. Scalers, PCA, selectors, and any hyperparameters use prior data only. Fit
   transformations inside each temporal fold; prohibit random train/test splits.
3. All arms use the same origins, targets, and eligible memory pool. Specify a
   common warm-up with enough past OOS errors; do not fill missing embeddings.
4. Cap common latent comparisons at 2025-10-10, and separately enforce complete
   targets before the existing protected phase. No reopening of sealed data.
5. Check positivity, Gamma optimizer convergence, weighted smearing, memory
   identities, and independent reconstruction of retrieved neighbors and errors.
6. Fix the complete comparison family before scoring, including memory versus
   calibration contrasts; use paired dependence-aware inference and multiplicity
   correction. Keep one- and five-session QLIKE as the primary targets.
7. Retain the existing research standard of meaningful effect size and period
   stability. Historical results identify leads; prospective forecasts establish
   whether they persist. Economic utility needs a separately specified decision
   rule and costs and is not implied by better forecast loss.

These are specific mechanisms to test, not a claim that additional complexity
will beat a strong implied-volatility benchmark. No new models were scored in
this review.
