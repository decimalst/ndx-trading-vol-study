# Focused reference review: Moirai 2.0 and xLSTM-Mixer

Reviewed 2026-09-06 from the authors' papers and official implementations.
This is an applicability and dependency inspection. Neither model was installed,
trained, or scored during this review, and this note does not change a frozen
comparison family. TSLib, TimeCopilot, and Timer-S1 are being reviewed separately.

## Practical findings

| Reference | Useful experiment for this repository | Current local status |
|---|---|---|
| Moirai 2.0 small | Frozen univariate RV forecast summaries added to the existing strong benchmark, with causal calibration | Official code is available; a separate compatible environment and checkpoint are required |
| xLSTM-Mixer | A small jointly trained mixer of past observable market features, compared with its linear component | The supplied model defaults to CUDA; the underlying sLSTM has a vanilla backend that could support an explicit CPU adaptation |

Neither paper establishes incremental QLIKE performance on NDX variance. Any
addition needs a separately fixed specification before its first score; broad
benchmark rankings are motivation for tests, not evidence of a finance signal.

## Moirai 2.0: suitable as a univariate expert

The paper reports an 11.4 million parameter small model, trained on 36 million
series using quantile forecasting. Its limitations section explicitly says that
multivariate forecasting and covariate support were dropped. This rules out
treating the released model as a demonstrated multivariate replacement for our
HAR-IV, leverage, and stress model. Its paper also reports that larger variants
and longer horizons did not consistently help. [Paper, sections 5.3 and 6](https://arxiv.org/html/2511.11698v1)

The model card identifies the released checkpoint as
`Salesforce/moirai-2.0-R-small`. Its stated training sources include GIFT-Eval
subsets, generated data, and internal Salesforce data. Consequently, the public
description alone cannot prove absence of exposure to this market history.
The checkpoint is labeled CC-BY-NC-4.0; the code's license is a separate item.
[Official checkpoint and model card](https://huggingface.co/Salesforce/moirai-2.0-R-small)

The official predictor accepts a `device` argument and emits quantile forecasts.
The convenience prediction method states that it supports univariate forecasts.
A CPU execution route is therefore plausible from code inspection; inference
speed and runtime compatibility have not been measured here. A quantile grid or
its median is not automatically the conditional mean needed by QLIKE.
[Official forecasting implementation](https://github.com/SalesforceAIResearch/uni2ts/blob/main/src/uni2ts/model/moirai2/forecast.py)

The official dependency file requires PyTorch >=2.1 and <2.5, NumPy 1.26, and
SciPy 1.11, plus GluonTS and Lightning. The repository's currently installed
versions, checked locally, are PyTorch 2.13.0, NumPy 2.4.6, and SciPy 1.17.1;
Uni2TS, GluonTS, and Lightning are absent. Installing those requirements into
the research environment would alter the existing model stack. A separate
environment is the concrete compatibility requirement, not a claim that a Mac
cannot run the model. [Official dependency specification](https://github.com/SalesforceAIResearch/uni2ts/blob/main/pyproject.toml)

Proposed minimal extension, not a frozen specification: pin the official code
and checkpoint revisions, use a fixed 512-session context of log daily RV, and
produce one- and five-session forecast summaries. Add a predeclared summary as
an input to the same Gamma benchmark using only previous completed labels.
Compare benchmark alone, the summary alone with causal calibration, and the
augmented benchmark. Treat an exponentiated log-median as a feature with a clear
name, never as an already calibrated mean-variance prediction. This tests whether
the pretrained forecasting map contributes structure beyond the benchmark while
keeping the final training objective aligned with QLIKE. These are proposed
design choices, not a reproduction of a paper experiment.

## xLSTM-Mixer: suitable for a controlled adaptation

The paper combines a linear forecasting stage with scalar-memory xLSTM mixing
and a second view of the representation. It evaluates multivariate forecasting,
which is closer to the question of jointly modeling our observable inputs than
another univariate foundation model. Its results still require a task-specific
test at our short horizons. [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/09e38101e74a89129ccd0d0756ed36b3-Abstract-Conference.html)

The official repository was tested with Python 3.11, PyTorch 2.4, Ubuntu 22.04,
and CUDA 12.1. Its CUDA sLSTM path requires compute capability >=8.0. The
requirements pin `xlstm==1.0.3`; this research environment has `xlstm==2.0.5`.
[Official setup instructions](https://github.com/mauricekraus/xLSTM-Mixer)
[Official requirements](https://github.com/mauricekraus/xLSTM-Mixer/blob/main/requirements.txt)

The released model constructs `sLSTMLayerConfig` without exposing a backend
parameter, so its default path requests CUDA. The code includes learned memory
tokens, reversible instance normalization, a linear backbone, and explicit
ablations including a linear-only arm. Learned tokens are parameters fitted on
training data; they are not the proposed database of matured forecast errors.
[Official mixer model](https://github.com/mauricekraus/xLSTM-Mixer/blob/main/xlstm_mixer/models/xlstm_mixer.py)

The underlying official xLSTM implementation offers both `vanilla` and `cuda`
backends and float32 configuration. This was also confirmed in the installed
package by reading its source. Thus, CUDA absence alone is insufficient to call
the architecture infeasible; a local test requires an explicit backend
adaptation and verification of the version difference.
[Official sLSTM cell configuration](https://github.com/NX-AI/xlstm/blob/main/xlstm/blocks/slstm/cell.py)

Proposed minimal extension, not a frozen specification: train a small mixer on
fixed trailing windows of the existing observable features, with an output
parameterization that guarantees positive variance and uses QLIKE. Compare an
otherwise matched linear-only model, the full mixer, and the strong benchmark.
Freeze width, window length, seeds, optimizer, iteration limit, refit schedule,
and any validation rule before scoring. Verify synthetic forward and backward
passes, target availability, chronological normalization, and repeatability on
the chosen backend first. Restrict all reverse-view operations to past input
windows. Report this as an adapted xLSTM-Mixer experiment, including the
backend, output objective, and configuration changes.

## What would establish usefulness

Both extensions should use the same predeclared horizons and admissible dates
as their controls, preserve the sealed phase, and produce honest historical
predictions for calibration and any later ensemble. A model that only improves
its standalone forecast over a weak univariate control has not established
orthogonal information. An augmented model must beat the strong benchmark, and
the complete added comparison family must enter multiplicity correction.
Possible pretraining overlap makes historical foundation-model findings
exploratory even when the local forecast construction is causal.
