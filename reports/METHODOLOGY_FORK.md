# Methodology fork — what was corrected, and what it changed

Written 2026-08-12. This documents a parallel evaluation path added in response
to four defects in the frozen pipeline. It is **additive**: the pre-registered
reports are unchanged and still reproduce byte-for-byte from the default
command line. `tests/test_methodology.py::TestFrozenReportsUnchanged` asserts
that, so a correction cannot quietly rewrite a result it was meant to be
compared against.

## Scenarios

| file | estimator | inference | what it isolates |
|---|---|---|---|
| `results_{phase}.md` | `trunc` | `naive` | frozen pre-registration |
| `results_{phase}_est.md` | `smearing` | `naive` | the point-forecast fix alone |
| `results_{phase}_inf.md` | `trunc` | `corrected` | the inference fixes alone |
| `results_{phase}_v2.md` | `smearing` | `corrected` | both |

```bash
make scenarios
```

## The four corrections

### 1. Point forecast: `trunc` → Duan smearing (`src/models.py`)

`trunc_mean_var` integrates `exp(q)` over the quantile grid and divides by the
grid's mass, discarding everything outside `[tau_lo, tau_hi]`. Measured against
exact smearing it returns **0.87 of the true conditional mean**, and the
discarded share depends on the grid — which is why every model's QLIKE differs
between `results_clean.md` and `results_clean_dec.md` while EWMA's is 0.4036 in
both. EWMA emits a native variance forecast and is not understated at all, yet
sits in the same column and the same DM table as models that are.

`--estimator smearing` uses `exp(mu)*mean(exp(resid))`, computed exactly for
every model whose residuals exist (`baselines --estimator smearing` writes
`*_sm.parquet`). Chronos-2 and TiRex-2 have no recoverable residuals and are
reconstructed from their saved quantiles by tail extension; those rows are
marked `~`.

**Honest limit, since the original review overstated this.** Across the HAR
family the frozen estimator's bias spans only 0.866–0.873 — a spread of
**0.70pp**. It is a near-common factor, so it largely cancels in paired
comparisons: correcting it moves QLIKE *levels* materially and DM statistics
only slightly. It was never the reason one model outranked another. The earlier
claim of a 4.7pp spread came from using a lognormal as the reference, which is
itself dispersion-biased. Reconstruction accuracy is pinned in the tests:

| method | ratio to exact | level | cross-model spread |
|---|---|---|---|
| `trunc` | 0.866–0.873 | −13.0% | 0.70pp |
| `lognormal` | 0.945–0.968 | −4.6% | 2.28pp |
| `tail_ext` | 0.952–0.976 | −3.5% | 2.49pp |

`tail_ext` is exact on a true lognormal; the residual 3.5% is genuine fat-
tailedness in log-RV residuals. It trades a large near-common bias for a
smaller dispersion-dependent one, which is why it is used only where nothing
better exists.

### 2. Power and equivalence (`src/methodology.py`)

`metrics.dm_test` reports a p-value and nothing else, so a failure to reject
reads as a null. Every DM row now carries the minimum detectable effect, the
origins required to resolve the gap actually observed, a TOST equivalence test
against a declared margin (3% of the benchmark's loss), and a three-way
verdict. **A non-significant DM alone can never produce the verdict
`equivalent`** — without the equivalence test it is `inconclusive`.

### 3. Specification gate (`spec_registry.yaml`)

`config.yaml` is frozen and records no specification dates. The evaluator
already suppressed `har_lev`/`har_iv_lev` from clean-phase output by hand "so
testing them cannot become a peek"; the registry applies that rule to
everything. A model is confirmatory in a phase only from
`max(phase_start, specified_on, available_from)`. Undetermined dates are
treated as exploratory — the burden sits on the specification, not the reader.

### 4. Overlapping-window inference (`src/methodology.py`)

The 30-day MZ and encompassing regressions run on daily origins whose targets
share ~21 trading days, and report `n=171` / `n=2463`. HAC(32) on n=171 is a
lag/n ratio of 19%. Replaced with a circular moving-block bootstrap and refits
on all 21 non-overlapping subsamples, reporting `n_eff` alongside `n`. This is
the standard `config.yaml`'s own `carry_study` block already sets.

## What actually changed

### QLIKE levels move; rankings do not (clean window, fix 1 alone)

| model | frozen | + estimator | Δ |
|---|---|---|---|
| persistence | 0.5565 | 0.5429 | −0.0136 |
| ewma | 0.4036 | 0.4036 | +0.0000 |
| har | 0.3731 | 0.3624 | −0.0107 |
| har_iv | 0.3095 | 0.3118 | +0.0023 |
| har_iv_x | 0.3006 | 0.3037 | +0.0031 |
| chronos_uni | 0.3716 | 0.3604 | −0.0112 |

EWMA is the control: unchanged, because it never used the estimator. No model
ordering changes.

### The clean-window "null" was never a null (fix 2, clean window)

| model vs har_iv | DM (p) | MDE | origins needed | verdict |
|---|---|---|---|---|
| har_ic | 1.327 (0.186) | 0.0085 | 856 | **equivalent** |
| har_iv_x | −1.347 (0.180) | 0.0169 | 831 | inconclusive |
| chronos_cov_iv | 0.429 (0.669) | 0.0435 | 8,191 | inconclusive |

Only one of the three earns a null. `chronos_cov_iv` would need 8,191 origins
to resolve the gap it shows; the window has 192. Reporting that row as evidence
that the foundation model extracts nothing extra was never supported.

### The earnings result is refuted on the well-powered sample (diagnostic)

| model vs har_iv | DM (p) | p_TOST | verdict |
|---|---|---|---|
| har_iv_x | 0.428 (0.669) | **0.013** | **equivalent** |

At n=2463 the equivalence test *rejects non-equivalence*: `har_iv_x` is
statistically indistinguishable from `har_iv` within 3% of its loss. That is a
positive finding of no effect, not a failure to reject — and it is the finding
the clean window was too short to produce.

The replication panel and the heavy-earnings slice agree:

| pair | clean win% | clean gap | flip k | diagnostic win% | diagnostic gap | replicates? |
|---|---|---|---|---|---|---|
| har_iv_x vs har_iv | 65.1% | +0.0081 | 4 | 65.7% | −0.0016 | **no** |
| har_x vs har | 63.5% | +0.0090 | 4 | 65.5% | −0.0020 | **no** |
| har_iv vs har | 54.2% | +0.0506 | never | 53.1% | +0.0322 | yes |
| har_lev vs har | 58.9% | +0.0137 | never | 60.1% | +0.0180 | yes |

The win rate replicates; the mean gap inverts. Four of 192 clean origins flip
its sign. On the diagnostic heavy-earnings slice the event terms lose on a
*majority* of days — 61/157 and 64/157, sign p = 0.0065 and 0.0251, significant
in the direction opposite to the claim.

Note what survives: `har_iv vs har` and `har_lev vs har` replicate cleanly in
both windows on both statistics. The corrections are not indiscriminate.

### The 30-day section reports 8 independent observations as 171 (fix 4)

```
frozen     beta=0.829 (p[beta=1]=0.284), R2=0.295, n=171
corrected  beta=0.829, n=171, n_eff=8
           bootstrap se=0.220, 95% CI [0.325, 1.183], p[beta=1]=0.293
           across the 21 non-overlapping subsamples: beta in [0.298, 1.710],
                                                     honest se=0.458
           premium 4.1 vol points, 95% CI [1.7, 5.9]
```

The point estimates are unchanged — only the uncertainty is, and the honest
standard error is 2–3x the HAC(32) one. The variance risk premium the frozen
report prints as "about 4.2 vol points" has a 95% interval of [1.7, 5.9].

## Consequences for the headline

1. **"Foundation models add nothing beyond HAR-IV"** is not established. The
   comparison is `inconclusive`, not `equivalent`, and would need 8,191 origins.
2. **"Earnings concentration carries information VXN does not price"** is
   refuted, not merely unsupported: `equivalent` at p_TOST = 0.013 on n=2463,
   with the clean-window effect inverting and the heavy-earnings slice
   significantly negative.
3. **The 30-day VXN results** cannot support any conclusion at n_eff = 8, in
   either direction.
4. **The point-forecast estimator** should be fixed for correctness of levels
   and cross-grid comparability, but it does not change any ranking and is not
   a reason to doubt the model comparisons.

Items 1–3 need either a much longer accrual window or a different design. Item
4 is done.
