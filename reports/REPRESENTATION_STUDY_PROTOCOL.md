# Tail ranking, latent probing, and noise-robustness protocol

The core target, scoreboard, noise grid, and probe ladder were frozen on
2026-08-12 before their result-producing commands. The machine-readable
contract is [`representation_study.yaml`](../representation_study.yaml); its
explicitly marked `reviewer_controls` section was appended only after the first
classical ranking headline and remains diagnostic rather than pre-specified
confirmation. The
1999 source panel is built and hash-locked first under
[`history_extension.yaml`](../history_extension.yaml); this protocol cannot
silently fall back to the mutable project master.

## Ordered design

The order is part of the test:

1. Reconstruct the price-only QQQ realized-variance panel from March 1999 and
   lock the source hash.
2. Freeze the transition event and ranking scoreboard.
3. Extract causal TiRex-2 representations and run the probe.

The GBM functional-form study is separate and may run alongside these steps.
The noise experiment is also a representation mechanism diagnostic, but it
does not use transition labels and cannot influence the tail definition.

No origin on or after 2025-11-03 is permitted. This work does not reopen the
sealed NDX clean accrual window.

## Tail event and scoreboard

The event definition is inherited, unchanged, from the registered HMM study:
at each annual fold, estimate the 80th percentile of `log_rv` using observations
through the prior-year cutoff. Score only calm origins at or below that
threshold. The label is one when any of the next five complete sessions exceeds
the same threshold.

The runner first attempts a 2001 fold, but it requires at least 400 completed
calm-origin training labels after feature construction. The realized eligible
sample did not clear that frozen minimum until the 2002 fold. Consequently,
1999–2001 is training-only: this design scores the final portion of the dot-com
transition and 2008 forward, but not the onset in 2000. That limitation is
reported rather than relaxing the threshold after seeing results.

The primary metric is ROC AUC. The secondary metric is top-decile lift: the
event rate among the highest-scored `ceil(10%)` of origins divided by the full
event rate. Both are computed independently on the five fixed non-overlapping
origin phases and averaged without sample-size weighting. Pooled daily scores,
Brier loss, and log loss are diagnostic only. Every model uses the same rows.

## Classical comparators

The supervised benchmark is ridge logistic regression on current log realized
variance plus trailing 5- and 22-session **means of log realized variance**,
inheriting the earlier transition-study convention. These are not the standard
HAR transforms `log(mean variance)`. The two-state Gaussian HMM is fit on each
fold's prior price history, forward-filtered, converted to a five-session
exceedance probability, and Platt calibrated using only completed prior labels.
The HMM incremental model adds that calibrated state probability to the three
benchmark features.

## Post-result reviewer control

After the 0.8704 benchmark AUC and 4.80x lift were visible, review requested a
single-variable persistence control. It is therefore registered as a
**post-result diagnostic**, not retroactively described as confirmatory. For
each annual fold, current `log_rv` is converted to its empirical percentile in
that fold's prior-year training history and entered alone in the same ridge
logistic model. This asks how much of the three-feature benchmark is simply the
current volatility level within calm origins.

A VXN-percentile alternative is not put on the common extended sample because
the locked free VXN series begins in 2009; using it would discard the 1999
history extension. It can be reported later on a shorter matched sample, but it
is not interchangeable with this full-history control.

The same review observed that five phases are a coarse robustness grid and
that 736 positive origins overstate independent information when crossings
cluster. The amended diagnostic therefore reports each metric's five-phase
minimum, maximum, and spread, plus a 95% leave-one-transition-episode-out
jackknife interval. The episode rule was already frozen for the latent study:
attach a positive origin to its first crossing and merge trigger sessions no
more than five market sessions apart. Applying that uncertainty to the
classical headline is post-result reporting, not a rewritten original test.

## TiRex-2 probe ladder and control tasks

The checkpoint, revision, package version, layer, and pooling rule are fixed.
The representation is the 512-dimensional last valid target token after
`stack_out_norm` and before the forecast head. The context contains at most
2,048 causal `log_rv` observations; shorter early contexts are retained rather
than backfilled. Test-time sign flipping and differencing are disabled. There
is no layer, pooling, or regularization search.

PCA is forbidden. Variance is not label relevance, so an unsupervised top-PC
projection could discard a low-variance regime direction before testing it.
Instead every probe uses the full 512-dimensional space unless the rung is
explicitly sparse. The fixed capacity ladder is:

1. Ridge logistic over all dimensions.
2. Ridge logistic on the 1, 5, and 10 dimensions with the largest absolute
   standardized event/non-event mean difference in that fold's **training
   rows only**. All three k values are reported; none is selected as a winner.
3. One eight-unit tanh MLP with fixed L2 penalty, solver, iteration cap, and
   seed.

Every rung is fitted twice: latent-only, to measure representation selectivity,
and with the three supervised RV-history features, to measure incremental
value against the benchmark and calibrated HMM. Direct joint fitting avoids
feeding an in-sample probe logit into a second-stage model.

Raw AUC is not sufficient evidence because probe capacity can manufacture an
impressive fit. Each fold therefore also generates ten first-order Markov
surrogate-label paths. Their transition probabilities are estimated from that
fold's completed training labels only, and each path continues into the test
segment independently of the representations. The identical probe, feature
scaling, sparse selection, and optimization are applied to real and control
tasks. The primary representation quantity is **selectivity**: actual-label AUC
minus the median matched-control AUC. Top-decile-lift selectivity is secondary.

Uncertainty is clustered on transition episodes, not daily origins. A positive
origin is attached to the first future threshold-exceedance session; trigger
sessions within five market sessions form one episode. Leave-one-episode-out
jackknife intervals are reported for AUC selectivity and model deltas. The
number and influence of episodes travel with every headline score.
These are conditional influence diagnostics: positive-trigger episodes are
deleted while negative origins stay fixed. They are not full cluster-robust
standard errors for all serial score variation.

The result remains diagnostic: the TiRex pretraining corpus may include
financial price histories.

## Eidos-derived context corruption

This is an adaptation of Appendix A.1.2 of the
[Eidos paper](https://arxiv.org/html/2602.14024v1), not an evaluation of Eidos.
The input is the original harness's 1,024-session `log_rv` history (Eidos used
2,048, so this is explicitly an adaptation). Gaussian noise uses local-context
standard deviation and intensities `{0,.2,.4,.6,.8}`. Impulse noise occurs with
probabilities `{0,.05,.10,.15,.20}`, has magnitude eight local standard
deviations, and receives a random sign. Seed 42 is converted into an
order-invariant per-origin seed from the first 64 bits of
`SHA-256("42|YYYY-MM-DD")`. One Gaussian draw and one impulse uniform/sign draw
per origin are reused across models and intensities; impulse masks are nested.
The frozen implementation passes the same raw corrupted log-RV array to each
adapter and retains model-native preprocessing. Chronos-2 and TiRex-2 may
normalize internally, but that behavior was not independently audited; the HAR
adapter does not renormalize its corrupted origin state. This differs from
Eidos Appendix A.1.2, which explicitly re-normalizes a corrupted sequence using
its noisy statistics, so HAR-versus-foundation degradation magnitudes are not
an apples-to-apples architectural contrast.

Origins are every twentieth session in the already diagnostic 2016-01-04
through 2025-10-17 window. Chronos-2 and TiRex-2 are evaluated univariately on a
common decile grid. Expanding HAR is refit on uncorrupted history at each origin;
only its origin-time lag features are recomputed from the corrupted context. The
sole primary statistic is a relative decile-grid CRPS approximation at every
intensity: the trapezoidal pinball integral over quantiles 0.1-0.9 for a noisy
input divided by the same model's clean-input score on the same origins.

Paired model differences use 5,000 moving-block bootstrap draws with seed
420042. A 22-session dependence block maps to two sampled origins because the
diagnostic grid advances 20 sessions at a time. These intervals are secondary
to the full registered degradation curves.

This tests how the three observed model pipelines respond to surface-level
corruption. Within the two foundation adapters it bears on their earlier null;
the HAR comparison also mixes preprocessing with architecture. It cannot show
that an untested Eidos checkpoint would be robust, and contamination of the
diagnostic history prevents a new forecast-accuracy claim.

## Post-result 99-control sparse-k=1 follow-up

Review after the latent ladder identified that ten controls make a 5% exact
randomization claim unattainable. The original ladder remains unchanged and
descriptive. A separate machine-readable protocol,
`latent_k1_confirmation.yaml`, registers exactly one k=1 rung and 99 fixed
controls; it is explicitly a reused-history, post-result diagnostic rather than
a pristine holdout. Its report is
[`representation_study/latent_k1_confirmation.md`](representation_study/latent_k1_confirmation.md).
