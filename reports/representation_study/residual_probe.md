# Residualized latent probe — is anything orthogonal to RV history?

**Post-result and additive.** Governed by `residual_probe.yaml`, frozen before `src/residual_probe.py` was written. Runs on a sample the parent study already inspected, so it is a diagnostic and not a new holdout. It rewrites no frozen verdict.

## The question

The parent probe showed transition proximity is *decodable* from TiRex-2's latent state. But its selected coordinates track smoothed volatility and prior-session VXN, and [`pooling_diagnostic.md`](pooling_diagnostic.md) showed the pooled statistic is mostly between-fold — a score with zero within-year information scores 0.8317, above the parent's 0.8153. So: project the HAR feature set out of every coordinate, then ask whether what remains ranks transitions **within** a fold.

## How much of the latent is just RV history?

Median R² of `[1, log_rv_d, log_rv_w, log_rv_m]` on a latent coordinate, across 19 folds: **0.305** (fold range 0.242–0.748). Whatever the ranking result below, this is the first-order answer: three HAR terms explain that share of a typical TiRex coordinate.

## Ranking

`within-fold` is the registered statistic — AUC inside each (phase, fold) cell, pair-count weighted. `pooled` is the parent's phase-mean statistic, shown for continuity only; it is **not** registered here and is the one the pooling diagnostic discredits.

| score | within-fold AUC | pooled phase-mean AUC |
|---|---|---|
| parent `p_benchmark` (continuity only) | 0.6985 | 0.8716 |
| HAR benchmark, refit in this pipeline | 0.7162 | 0.8719 |
| **residualized k=1 alone** | 0.6298 | 0.7129 |
| **HAR + residualized k=1** | 0.7176 | 0.8716 |

## The registered comparison

`HAR + residualized k=1` minus `HAR`, within fold, paired over the **19 annual folds** (the five phases inside a fold are offsets of the same year and are averaged first — treating them as independent would repeat a pseudo-replication this repository has already corrected twice).

| quantity | value |
|---|---|
| mean ΔAUC | -0.0006 |
| median ΔAUC | +0.0022 |
| folds improved | 10 / 19 |
| sign test p | 1.0000 |
| bootstrap 95% CI | [-0.0165, +0.0155] |
| MDE at 80% power | 0.0234 |
| equivalence margin (declared) | ±0.010 |
| p_TOST | 0.1383 |
| **verdict** | **`inconclusive`** |

**`inconclusive` means this sample cannot answer it**, not that there is no effect. The minimum detectable effect is 0.0234 AUC against an observed -0.0006 — roughly 37x. Reporting this row as a null would be the exact defect this repository has corrected repeatedly.

**And the design, not the sample, is the binding constraint.** Resolving an effect the size of the declared ±0.010 margin at 80% power would take **~104 annual folds** — 104 years of daily data, against the 19 usable here. The fold-to-fold spread (−0.075 to +0.086) is an order of magnitude larger than the effect being looked for, so accruing more history does not fix this. A conclusive answer needs a different unit of inference — not a longer sample.

## Per-fold deltas

The spread, not just the mean — a mean with no spread shown is how a handful of folds gets mistaken for a result.

| fold | ΔAUC | | fold | ΔAUC |
|---|---|---|---|---|
| 2003 | -0.0371 | | 2017 | +0.0490 |
| 2007 | +0.0327 | | 2018 | -0.0100 |
| 2008 | -0.0474 | | 2019 | -0.0292 |
| 2009 | +0.0104 | | 2020 | +0.0117 |
| 2010 | +0.0330 | | 2021 | -0.0239 |
| 2011 | +0.0037 | | 2022 | -0.0748 |
| 2012 | +0.0857 | | 2023 | +0.0058 |
| 2014 | -0.0000 | | 2024 | +0.0137 |
| 2015 | +0.0022 | | 2025 | -0.0150 |
| 2016 | -0.0225 | |  |  |

## Limits

- Post-result on an inspected sample; not a new holdout.
- One registered rung (k=1), matched to the parent. A richer residualized probe might find something k=1 cannot, and this run does not exclude that.
- The residualization is linear. A component non-linearly related to HAR would survive projection and be counted as orthogonal.
- HAR here is the parent's three-term feature set, not the full HAR-IV information set; VXN is not projected out.
