# Moirai 2.0 fixed extraction plan

Specified before any checkpoint forward or historical model extraction on 2026-09-06.
The adapter is additive and uses an isolated Python environment. The existing
research environment and all frozen experiments are unchanged.

The official `Salesforce/moirai-2.0-R-small` checkpoint revision is
`30f43ff08c8494f4943ae1521e9d4e94a0fbb389`; the official Uni2TS code revision is
`cfd46d4510ed8896f263116f32928eede05b0a75`. The checkpoint is labeled
CC-BY-NC-4.0 and code Apache-2.0. Its mixed public/internal pretraining corpus
does not establish absence of exposure to historical market data. Results remain
exploratory even when the extraction procedure uses causal input windows.

Each eligible origin from 2010-01-04 through 2025-10-10 receives exactly 512
actual sessions of log strictly-positive daily GK-plus-overnight variance,
including the origin session. Source: local `data/orthogonal_round2/features.parquet`,
column `rv_total`. Read only this column and rows through 2025-10-10; the network
is used only to acquire public code and model weights. The model receives no
dates, targets, future covariates, padded observations, or imputed values. Its own
pretrained normalization operates separately on each supplied historical window.

One five-step native forecast emits the fixed checkpoint quantile grid. Select
the numeric 0.5 quantile independently at each step. Retain `moirai_log_h1` as
the first step's median log-variance and `moirai_log_h5` as
`log(mean(exp(step_median_log_variance[1:5])))`, computed by stable log-mean-exp.
These are frozen predictor summaries, not conditional-mean estimates. Root's
separate protocol governs train-only Gamma calibration and augmented comparisons.
The extraction module never reads labels, evaluates predictive losses, tunes a
parameter, or ranks an alternative.

Use CPU float32 inference, evaluation mode, deterministic PyTorch operations,
seed 20260906, two CPU threads, and batches of 32 windows. Synthetic unit tests
must pass before the first checkpoint forward. First test the real checkpoint
only with synthetic sequences, including repeated and permuted batch inputs.
Historical extraction requires the root agent's explicit go after these checks.

Checkpoint SHA-256: `fb5652a3db8ea572606221b7cb1e77bb8962b168e4d4cc752cf31ceb04074669`.
Config SHA-256: `6b74b03c8ec199fabc352c0203465958142ca468183da68549652734836f853d`.
Code archive SHA-256: `e1de1cbff2b6e131e1988f4f9dddce51e0801bb41c87d7e7d8609a783dd89f18`.
Raw downloads, the isolated environment, install records, and source-derived
outputs remain local under `data/model_memory_reference/`. Hashes and the
installed dependency lock accompany the extraction audit.

Primary references: [official checkpoint](https://huggingface.co/Salesforce/moirai-2.0-R-small/tree/30f43ff08c8494f4943ae1521e9d4e94a0fbb389),
[official code](https://github.com/SalesforceAIResearch/uni2ts/tree/cfd46d4510ed8896f263116f32928eede05b0a75),
[paper](https://arxiv.org/abs/2511.11698).
