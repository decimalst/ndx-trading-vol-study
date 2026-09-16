# Fixed CPU xLSTM-Mixer-inspired experiment

Specified 2026-09-06 before empirical neural forecasts or scores. This extension
leaves previous protocols unchanged and belongs to the master's combined
32-core plus 14-reference, 46-hypothesis family. The complete synthetic suite
was written first, failed because its module was absent, then passed after
implementation. No package was installed for this adaptation.

## Purpose and controls

Test whether a small scalar-memory neural mixer can use the existing benchmark
information more effectively. The arms are `nlinear` and `xlstm`. Both consume
the same inputs, targets, training rows and schedule. Compare each to the strong
benchmark, and compare `xlstm` directly to `nlinear`; a standalone neural
forecast beating its linear neural control alone does not establish a useful
improvement over HAR-IV/leverage/stress.

The inspiration is the authors' combination of linear temporal forecasting,
scalar-memory mixing, memory tokens and a second representation view. The
authors also provide a linear-only ablation. Our architecture is an explicitly
smaller CPU adaptation, with a different output and training objective; it is
not a replication of their benchmark results.
[Official implementation](https://github.com/mauricekraus/xLSTM-Mixer/blob/main/xlstm_mixer/models/xlstm_mixer.py)

## Fixed model and optimization

- Input: 22 actual consecutive trading sessions of the 11 nonconstant baseline
  features, in this exact order: `lrv_d,lrv_w,lrv_m,lev_d,lev_w,lev_m,liv,lvix,
  term,xasset_stress,market_stress`. Prior implied-volatility lags remain intact.
- Normalize each feature using the population mean and standard deviation of
  flattened eligible training windows. Repeated observations therefore receive
  the explicit multiplicity induced by overlapping windows. No query, future
  window or uncompleted training label enters these fitted statistics.
- Shared linear stage: subtract the latest observation of each feature; apply
  a shared 22-to-2 temporal linear map; restore that feature's latest value;
  project each of the 11 resulting two-dimensional variate tokens to width 16.
- `nlinear`: flatten the 11 tokens and apply a 176-to-2 linear readout. This
  entire map is affine in the standardized input window.
- `xlstm`: prepend two learned width-16 tokens. Apply one shared scalar-memory
  sLSTM layer in forward and reversed variate order, restore the latter order,
  and add the mean of the two mixed views to the original variate tokens.
  Flatten and use the same readout form. The two views contain only data
  already observed at the origin. Recurrent state resets for every sample.
- sLSTM configuration: width 16, four heads, zero convolution kernel, zero
  dropout, `vanilla` backend, float32 throughout, automatic mixed precision off.
  Installed versions used by the synthetic checks: `xlstm==2.0.5` and
  `torch==2.13.0`; versions are recorded again in every empirical fit audit.
- Joint outputs: log predicted mean variance for horizons 1 and 5. Divide each
  target by its training median for numerical units, optimize mean QLIKE over
  both outputs, and restore target units after exponentiating the predictions.
  No Duan smearing is applied to the directly trained variance outputs.
- Initialize both models with seed 20260906. Their common layers have exactly
  matched initial parameters. Train from scratch at every refit, using Adam
  with learning rate .001, betas (.9,.999), epsilon 1e-8, weight decay zero,
  gradient norm cap 1, 12 epochs, batch size 128, and chronological batches.
  Run on one CPU thread with deterministic operations. No shuffling,
  checkpoint selection, early stopping, validation-based choices or tuning.

The two learned tokens are model parameters trained on past examples. They do
not represent the separate store of matured historical forecast errors.

## Temporal contract and output

Refit at the first eligible forecast origin in each calendar year, beginning
with the master's historical 2013 forecast start. Each fit uses the most recent
1,500 eligible training origins, with at least 500 required. Training origins
must precede the fit origin, and both the one- and five-session target must
have completed by the fit origin. Five-session targets are the arithmetic mean
of the next five daily variances. Context windows retain the actual session
calendar even when a date is excluded from the common origin pool.

Use the master's same admissible origins, score dates 2016-01-04 through
2025-10-10 and completed-target fence 2025-10-20. The sealed phase beginning
2025-11-03 remains unused. Missing contexts and insufficient training history
must be reported explicitly; they cannot be filled or silently discarded after
scores. Fixed-epoch training is a bounded optimization procedure, not a claim
of optimizer convergence or an exhaustive search of the architecture.

`yearly_forecasts(features, eligible_origins, forecast_origins, progress=None,
fit_callback=None)` returns a long forecast table and full annual fit audits.
Rows contain `origin,horizon,model,prediction,y,target_end,fit_origin,train_n,
train_last_target`. The fit callback permits saving every annual trained state
for independent replay. Audits record exact training origins, normalization,
target units, parameter counts, optimization steps, training losses, gradients,
package versions and a state hash. The module writes no files itself.

These honest historical neural predictions can also serve as log-forecast
features in a future Gamma stacking test. Such augmentation requires its own
enumerated comparison; it must use predictions made at the historical origin,
never retrospective predictions from the latest fitted neural model.

## Pre-run evidence

All nine synthetic tests passed in 2.53 seconds on the local CPU. They include
actual vanilla sLSTM forward/backward and optimizer updates; matched linear
initialization; no cross-sample recurrent state; the exact log-output QLIKE
gradient; one/five-session target construction and maturity; the 1,500-window
cap and 500-window minimum; real-session context boundaries; and two complete
12-epoch fits that reproduce exactly after future values are perturbed.

This supports a bounded yearly CPU run. It does not establish empirical
predictive performance. Numerical failures stop the arm; there is no silent
backend change, architecture change or replacement forecast.
