# TSLib, TimeCopilot and Timer-S1 applicability

Reviewed 2026-09-06. These findings complement REFERENCE_REVIEW.md and the
separately fixed Moirai and xLSTM implementation plans. No benchmark leaderboard
is treated as evidence of an orthogonal NDX variance signal.

## Time-Series-Library

TSLib is useful as a source of implementations and controlled architectural
ablations. Its collection includes linear models, temporal mixers, patch
transformers and TimeXer, which explicitly handles exogenous variables. The
maintainers' April 2026 notice says active feature additions have slowed and
some older benchmarks no longer provide a useful measure of current progress;
they still regard the baseline implementations as correct.
[Official library](https://github.com/thuml/Time-Series-Library)

For this repository, the reusable element is the model implementation. The
experiment must retain our trading-session targets, delayed implied-volatility
inputs, historical training availability, matched scoring dates, and QLIKE
inference. A generic forecasting benchmark's split, preprocessing and metric
would change the question. The current small linear/mixer comparison follows
this approach. We did not install the complete TSLib dependency stack or run
every listed architecture.

## TimeCopilot

TimeCopilot combines forecasting models with an LLM-controlled workflow for
analysis, model selection, validation, forecasting and ensembles. It is an
orchestration framework, so a result attributed to it also depends on which
models and selection procedure it runs.
[Authors' paper](https://arxiv.org/abs/2509.00616)

The applicable idea here is transparent comparison and ensemble construction.
The current experiment uses fixed experts and weights updated only from
completed historical forecast errors. There is no additional TimeCopilot
forecasting arm: we did not run its package, call an LLM API, or use generated
explanations as predictors. Autonomous selection over already inspected
results would enlarge the search and require explicit accounting.

## Timer-S1

Timer-S1 has 8.3 billion total parameters, with 0.75 billion active per token,
and an 11,520-point maximum context. Its mixture of experts and serial prediction
objective are interesting candidates for long-context forecasting. The model
card supports CPU inference and recommends at least 40 GB of GPU memory for
GPU use. It emits nine quantiles; the active parameter count is not the total
memory requirement.
[Paper](https://arxiv.org/abs/2603.04791)
[Official checkpoint and execution guidance](https://huggingface.co/thuml/Timer-S1)

The current runtime reported 36 GiB total memory, about 13.76 GiB available, and
no accessible CUDA or MPS accelerator. As a rough lower bound, 8.3 billion BF16
parameters alone occupy about 15.46 GiB, before runtime state or temporary
allocations. This motivates deferring this large model in the current run;
it does not prove CPU execution is impossible. Its weights were not downloaded,
and no Timer-S1 forecast was produced.

A future isolated run should first measure a single fixed-context CPU/GPU
forecast, then use a pinned checkpoint and the same causal calibration design
as Moirai. It would be a new registered comparison, with potential market-data
pretraining exposure still unresolved. This run instead tests the much smaller
Moirai checkpoint and a locally trained xLSTM adaptation.
