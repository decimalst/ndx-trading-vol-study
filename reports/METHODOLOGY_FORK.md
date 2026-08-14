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

**Honest limit, since the original review overstated this — and the walk-back
was itself overstated, corrected 2026-08-13.** Across `har`, `har_x`, `har_iv`,
`har_iv_x` and `har_ic` the frozen estimator's bias spans 0.866–0.873, a spread
of **0.70pp**. That band is a property of those five models. Widen the set and
it does not hold: 0.866–0.883 (1.64pp) once `har_sv`, `har_lev` and
`har_iv_lev` are included, and **0.812–0.883 (7.05pp)** across every quantile
model scored in the same DM column, because `persistence` sits at 0.812. All
three scopes are pinned in `tests/test_methodology.py`.

Two claims that stood here were wrong:

- *"It largely cancels in paired comparisons."* Backwards. QLIKE differentials
  are **not** scale-invariant — `d = rv·(1/v_a − 1/v_b) + log(v_a/v_b)` — so
  rescaling *both* forecasts by a single common factor moves the DM statistic
  on its own, and in fact reproduces most of the observed movement. The common
  factor is the mechanism, not the reason there is no effect.
- *"It was never the reason one model outranked another."* False on this
  window. `har_sv vs har`, with **both sides exactly smeared on identical 188
  origins**, moves from DM −1.615 (p=0.1080) to −2.163 (p=0.0318) — crossing
  the pre-registered α=0.05 on the estimator switch alone. Two further orderings
  become undetermined (`har`/`chronos_cov`, `har_ic`/`chronos_cov_iv`), though
  both of those pairs are inferentially empty (DM p ≥ 0.60 under both
  estimators) and mix an exact row against a reconstructed one.

What survives is the narrower and still-useful statement: correcting the
estimator moves QLIKE **levels** far more than it moves DM statistics, and no
*confirmatory* headline turns on the difference — `har_sv` is an exploratory
specification with n_req=315 against n=188. But "rankings are unaffected" is
not something this repository is entitled to say.

The earlier claim of a 4.7pp spread came from using a lognormal as the
reference, which is itself dispersion-biased. Reconstruction accuracy is pinned
in the tests:

Measured on `har`, `har_x`, `har_iv`, `har_iv_x`, `har_ic` over the clean
window, against exact smearing. Pinned by
`tests/test_methodology.py::test_reconstruction_table_matches_the_published_one`
so the table and the estimator cannot drift apart:

| method | ratio to exact | level | cross-model spread |
|---|---|---|---|
| `trunc` | 0.866–0.873 | −13.0% | 0.70pp |
| `lognormal` | 0.952–0.962 | −4.3% | 1.00pp |
| `tail_ext` | 0.952–0.976 | −3.5% | 2.49pp |

`tail_ext` interpolates the quantile function linearly in `z = Phi^-1(tau)` and
extrapolates past the outermost knot at the end-segment slope, which makes it
exact on a true lognormal on any grid; the residual 3.5% is genuine fat-
tailedness in log-RV residuals. It trades a large near-common bias for a
smaller dispersion-dependent one, which is why it is used only where nothing
better exists.

The `lognormal` row was previously published as 0.945–0.968 / −4.6% / 2.28pp.
That was not reproducible from any code in the repository and is corrected
here to what `models.mean_var_from_quantiles(..., "lognormal")` returns. See
the 2026-08-13 entry in `AMENDMENTS.md`.

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
| chronos_uni ~ | 0.3716 | 0.3579 | −0.0137 |

EWMA is the control: unchanged, because it never used the estimator. Ordering
within this table does not change — but see the correction above: `har_sv vs
har` crosses α=0.05 under the estimator fix, and `har`/`chronos_cov` inverts.
"No model ordering changes" was true only of the six rows printed here.

### The clean-window "null" was never a null

The heading of this section used to read "(fix 2, clean window)" while the
table below it carried the **fixes 1+2** numbers. Both are shown now, because
the difference is itself the finding:

Fix 2 alone (`results_clean_inf.md`, trunc estimator + corrected inference):

| model vs har_iv | DM (p) | MDE | origins needed | p_TOST | verdict |
|---|---|---|---|---|---|
| har_ic | 1.496 (0.136) | 0.0097 | 673 | 0.119 | inconclusive |
| har_iv_x | −1.245 (0.215) | 0.0200 | 972 | 0.478 | inconclusive |
| chronos_cov_iv ~ | 0.285 (0.776) | 0.0461 | 18,583 | 0.390 | inconclusive |

Fixes 1+2 (`results_clean_v2.md`, smearing estimator + corrected inference):

| model vs har_iv | DM (p) | MDE | origins needed | p_TOST | verdict |
|---|---|---|---|---|---|
| har_ic | 1.327 (0.186) | 0.0085 | 856 | 0.039 | **equivalent** |
| har_iv_x | −1.347 (0.180) | 0.0169 | 831 | 0.418 | inconclusive |
| chronos_cov_iv ~ | 0.553 (0.581) | 0.0413 | 4,919 | 0.468 | inconclusive |

Under the inference fix alone, **none** of the three earns a null. The single
`equivalent` verdict in this repository's clean window depends on the estimator
fix as well — which is another reason the "the estimator changes no verdicts"
claim above had to go. Either way `chronos_cov_iv` would need 4,919 origins
(18,583 under the frozen estimator) to resolve the gap it shows; the window has
192. Reporting that row as evidence that the foundation model extracts nothing
extra was never supported.

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
| har_iv_x vs har_iv | 65.1% | +0.0081 | 5 | 65.7% | −0.0016 | **no** |
| har_x vs har | 63.5% | +0.0090 | 4 | 65.5% | −0.0020 | **no** |
| har_iv vs har | 54.2% | +0.0506 | 20 | 53.1% | +0.0322 | yes |

The win rate replicates; the mean gap inverts. Five of 192 clean origins flip
its sign.

A `har_lev vs har` row was published here (58.9% / +0.0137 / never), and it
should not have been. `har_lev` and `har_iv_lev` are quarantined from the clean
**window**, not merely from the clean **report**, and the replication panel
reached past the report filter to score them on 192 clean origins. Removed
2026-08-13; see `AMENDMENTS.md`. The `flip k = never` in that row was also the
diagnostic value printed in a clean column. On the diagnostic heavy-earnings slice the event terms lose on a
*majority* of days — 61/157 and 64/157, sign p = 0.0065 and 0.0251, significant
in the direction opposite to the claim.

Note what survives: `har_iv vs har` replicates cleanly in both windows on both
statistics. The corrections are not indiscriminate. (`har_lev vs har` was also
cited here as replicating; that citation rested on a clean-window score the
model is quarantined from, and is withdrawn. Its diagnostic evidence at
n=2463 is untouched.)

### The 30-day section reports 8 independent observations as 171 (fix 4)

```
frozen     beta=0.829 (p[beta=1]=0.284), R2=0.295, n=171
corrected  beta=0.829, n=171, n_eff=8
           HAC(32) se=0.160   bootstrap se=0.220
           sd of beta across the 21 starting offsets = 0.360
           bootstrap 95% CI [0.325, 1.183], p[beta=1]=0.293
           across the 21 non-overlapping subsamples: beta in [0.298, 1.710]
           premium 4.1 vol points, 95% CI [1.7, 5.9]
```

The point estimates are unchanged; only the uncertainty is.

**Two corrections to what stood here, both from the 2026-08-13 audit.**

1. The figure previously published as the "honest standard error" (0.458) was
   the *median within-subsample* standard error — the se of a beta fitted to
   ~8 points, which is algebraically ≈ √21 × the full-sample OLS se. It is not
   a standard error of the reported full-sample beta and formed no valid ratio
   against HAC. The ratio "2.9x", and the earlier "3x", are both withdrawn.
   The report now prints `sd across starting offsets` (0.360) as the headline
   dispersion and labels the within-subsample figure for what it is.
2. **The bootstrap that replaced HAC is itself too narrow.** Measured against a
   DGP at this data's persistence, its nominal-95% interval covers ~82–85% and
   `se_boot` is ~0.75× the truth; the `p_zero`/`p_one` tests have true size
   ~10–16%. So `[0.325, 1.183]` is a floor on the uncertainty, not a calibrated
   interval, and the premium interval `[1.7, 5.9]` is likewise too tight.
   Pinned by `tests/test_methodology.py::TestBootstrapCalibration`.

Abandoning HAC(32) at lag/n = 0.19 remains correct — it is biased lower still.
The honest summary is that **no interval in this section is calibrated**, which
only strengthens the conclusion below: at n_eff = 8 the 30-day results support
nothing in either direction.

This is a clean-window statement. On the diagnostic window the same section
prints n=2463, n_eff=117, a lag/n ratio of 0.01, and HAC/bootstrap/refit
standard errors of 0.061/0.060/0.090 — HAC is essentially fine there, and the
generator previously printed the clean window's "lag/n near 0.19 ... understates
by about 3x" into the diagnostic report regardless. Corrected 2026-08-13. The variance risk premium the frozen
report prints as "about 4.2 vol points" has a 95% interval of [1.7, 5.9].

## Consequences for the headline

1. **"Foundation models add nothing beyond HAR-IV"** is not established. The
   comparison is `inconclusive`, not `equivalent`, and would need 4,919 origins.
2. **"Earnings concentration carries information VXN does not price"** is
   refuted, not merely unsupported: `equivalent` at p_TOST = 0.013 on n=2463,
   with the clean-window effect inverting and the heavy-earnings slice
   significantly negative.
3. **The 30-day VXN results** cannot support any conclusion at n_eff = 8, in
   either direction.
4. **The point-forecast estimator** should be fixed for correctness of levels
   and cross-grid comparability. It does not overturn any *confirmatory*
   comparison, but it is not ranking-neutral: `har_sv vs har` crosses α=0.05
   under it, and the only `equivalent` verdict in the clean window
   (`har_ic vs har_iv`) requires it. Treat frozen QLIKE levels as wrong by
   ~13% and frozen DM columns as estimator-dependent near the threshold.

Items 1–3 need either a much longer accrual window or a different design. Item
4 is done.
