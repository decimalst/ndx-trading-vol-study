# Results — phase: clean

Quantile grid: [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95] → intervals below are 90%.

**Corrected methodology run** — estimator=`trunc`, inference=`corrected`. This report is a fork; `results_clean.md` still holds the frozen pre-registered numbers and is unchanged. Differences between the two are methodology, not data. See src/methodology.py for what each correction does and why.

## h=1 losses (mean per day)

| model | n | status | QLIKE | CRPS | pin0.05 | pin0.95 | 90% cov | p_uc | p_ind |
|---|---|---|---|---|---|---|---|---|---|
| persistence | 192 | confirmatory | 0.5565 | 0.6227 | 0.1070 | 0.1125 | 0.896 | 0.848 | 0.047 |
| ewma | 192 | confirmatory | 0.4036 | - | - | - | - | - | - |
| har | 192 | confirmatory | 0.3731 | 0.5066 | 0.0983 | 0.0915 | 0.849 | 0.027 | 0.386 |
| har_x | 192 | confirmatory | 0.3618 | 0.4956 | 0.0983 | 0.0891 | 0.859 | 0.075 | 0.218 |
| har_iv | 192 | confirmatory | 0.3095 | 0.4649 | 0.0828 | 0.0828 | 0.880 | 0.374 | 0.423 |
| har_sv | 188 | **exploratory** | 0.3371 | 0.4711 | 0.0881 | 0.0839 | 0.926 | 0.224 | 0.959 |
| har_ic | 192 | **exploratory** | 0.3147 | 0.4686 | 0.0824 | 0.0833 | 0.891 | 0.669 | 0.246 |
| har_iv_x | 192 | **exploratory** | 0.3006 | 0.4541 | 0.0825 | 0.0832 | 0.891 | 0.669 | 0.246 |
| chronos_uni | 192 | confirmatory | 0.3716 | 0.5026 | 0.0916 | 0.0877 | 0.906 | 0.771 | 0.528 |
| chronos_cov | 192 | confirmatory | 0.3719 | 0.5030 | 0.0943 | 0.0906 | 0.880 | 0.374 | 0.158 |
| chronos_cov_iv | 192 | confirmatory | 0.3142 | 0.4614 | 0.0816 | 0.0865 | 0.906 | 0.771 | 0.053 |

### Specification status

A model is *confirmatory* in this phase only from `max(phase_start, specified_on, available_from)` onward. Anything specified after the window opened was tested on the data that produced it, so its p-values in the tables below are descriptive, not inferential. The evaluator already applied this rule to `har_lev`/`har_iv_lev` by hand; `spec_registry.yaml` applies it to everything.

| model | why not confirmatory | confirmatory origins available | gate |
|---|---|---|---|
| har_sv | specification date not recorded | 0 (date unrecorded) | 500 |
| har_ic | specification date not recorded | 0 (date unrecorded) | 500 |
| har_iv_x | specified 2026-08-11, after the phase opened | 0 | 500 |

Every row above is below its gate. Those models are reported for completeness and must not be quoted as confirmatory results.


## Diebold-Mariano vs HAR (QLIKE; negative = beats HAR)

`MDE` is the smallest QLIKE gap this sample could detect at 80% power; `n_req` is how many origins it would take to resolve the gap actually observed. `p_TOST` tests EQUIVALENCE against a margin of 3% of the benchmark's loss. A non-significant DM alone never earns the verdict `equivalent` — without the TOST it is `inconclusive`.

| model | DM | p | n | MDE | n_req | p_TOST | verdict |
|---|---|---|---|---|---|---|---|
| persistence | 3.325 | 0.0011 | 192 | 0.1545 | 136 | 0.999 | B better |
| ewma | 0.983 | 0.3267 | 192 | 0.0868 | 1,559 | 0.733 | inconclusive |
| har_x | -1.376 | 0.1704 | 192 | 0.0231 | 796 | 0.508 | inconclusive |
| har_iv | -3.477 | 0.0006 | 192 | 0.0513 | 125 | 0.998 | A better |
| har_sv * | -1.615 | 0.1080 | 188 | 0.0587 | 566 | 0.860 | inconclusive |
| har_ic * | -3.084 | 0.0023 | 192 | 0.0531 | 158 | 0.993 | A better |
| har_iv_x * | -3.660 | 0.0003 | 192 | 0.0555 | 113 | 0.999 | A better |
| chronos_uni | -0.166 | 0.8682 | 192 | 0.0258 | 54,537 | 0.147 | inconclusive |
| chronos_cov | -0.065 | 0.9483 | 192 | 0.0525 | 357,040 | 0.297 | inconclusive |
| chronos_cov_iv | -2.945 | 0.0036 | 192 | 0.0561 | 174 | 0.991 | A better |

`*` = exploratory specification (see the table above).

## Diebold-Mariano vs HAR-IV — same information set

HAR-IV is log-HAR plus log(VXN). Any model fed VXN beats plain HAR trivially, because implied vol predicts realized vol. The question this table answers is whether the foundation model extracts more from VXN than four OLS terms do.

`MDE` is the smallest QLIKE gap this sample could detect at 80% power; `n_req` is how many origins it would take to resolve the gap actually observed. `p_TOST` tests EQUIVALENCE against a margin of 3% of the benchmark's loss. A non-significant DM alone never earns the verdict `equivalent` — without the TOST it is `inconclusive`.

| model | DM | p | n | MDE | n_req | p_TOST | verdict |
|---|---|---|---|---|---|---|---|
| har_ic * | 1.496 | 0.1363 | 192 | 0.0097 | 673 | 0.119 | inconclusive |
| har_iv_x * | -1.245 | 0.2146 | 192 | 0.0200 | 972 | 0.478 | inconclusive |
| chronos_cov_iv | 0.285 | 0.7761 | 192 | 0.0461 | 18,583 | 0.390 | inconclusive |

`*` = exploratory specification (see the table above).

## Event-sliced QLIKE (mean)

| model | FOMC (n) | CPI (n) | heavy-earnings (n) | quiet (n) |
|---|---|---|---|---|
| persistence | 0.2442 (6) | 0.7321 (8) | 0.6458 (12) | 0.5528 (166) |
| ewma | 0.2360 (6) | 0.2623 (8) | 0.6117 (12) | 0.4014 (166) |
| har | 0.1867 (6) | 0.3866 (8) | 0.4481 (12) | 0.3738 (166) |
| har_x | 0.2595 (6) | 0.3592 (8) | 0.2517 (12) | 0.3736 (166) |
| har_iv | 0.1902 (6) | 0.3002 (8) | 0.3395 (12) | 0.3121 (166) |
| har_sv | 0.2305 (6) | 0.3224 (8) | 0.4345 (12) | 0.3345 (162) |
| har_ic | 0.1910 (6) | 0.3082 (8) | 0.3565 (12) | 0.3165 (166) |
| har_iv_x | 0.2565 (6) | 0.3026 (8) | 0.1846 (12) | 0.3105 (166) |
| chronos_uni | 0.1649 (6) | 0.3972 (8) | 0.4265 (12) | 0.3739 (166) |
| chronos_cov | 0.2467 (6) | 0.4004 (8) | 0.1671 (12) | 0.3899 (166) |
| chronos_cov_iv | 0.2755 (6) | 0.3438 (8) | 0.1311 (12) | 0.3274 (166) |

## Full-sample paired per-origin (all origins in phase)

Win rate and mean difference answer different questions. A high win rate with no mean gain is many small wins funded by rare large losses; a low win rate with a mean gain is the reverse. Read both before believing either.

| pair | n | wins | win % | sign p | mean | median | top-10 share of mean gap |
|---|---|---|---|---|---|---|---|
| har_iv_x vs har_iv | 192 | 120 | 62.5% | 0.000655 | 0.3006 vs 0.3095 | 0.1210 vs 0.1266 | 165% |
| har_x vs har | 192 | 112 | 58.3% | 0.025 | 0.3618 vs 0.3731 | 0.1427 vs 0.1431 | 160% |
| har_sv vs har | 188 | 102 | 54.3% | 0.274 | 0.3371 vs 0.3709 | 0.1300 vs 0.1431 | 108% |
| har_iv vs har | 192 | 111 | 57.8% | 0.0361 | 0.3095 vs 0.3731 | 0.1266 vs 0.1431 | 68% |
| chronos_cov vs chronos_uni | 192 | 95 | 49.5% | 0.942 | 0.3719 vs 0.3716 | 0.1354 vs 0.1570 | -9410% |

## Replication: clean vs diagnostic

`flip k` is how many origins must be dropped from the winning tail before the mean gap changes sign — a direct read on whether a result is a property of the sample or of a handful of days. A pair that wins in clean and loses in diagnostic has not replicated, whatever its p-value in either window.

| pair | clean win% | clean gap | flip k | diagnostic win% | diagnostic gap | diagnostic n | replicates? |
|---|---|---|---|---|---|---|---|
| har_iv_x vs har_iv | 62.5% | +0.0089 | 3 | 61.3% | +0.0001 | 2463 | yes |
| har_x vs har | 58.3% | +0.0114 | 4 | 60.7% | -0.0023 | 2463 | **no** |
| har_sv vs har | 54.3% | +0.0338 | 9 | — | — | — | not testable |
| har_iv vs har | 57.8% | +0.0636 | 24 | 54.1% | +0.0377 | 2463 | yes |
| chronos_cov vs chronos_uni | 49.5% | -0.0003 | 0 | — | — | — | not testable |

### Heavy-earnings slice, paired per-origin (cutoff 5.0% of index weight)

| pair | mean | median | better/n | sign p | top day | % of gap | ex-top |
|---|---|---|---|---|---|---|---|
| har_x vs har | 0.2517 vs 0.4481 | 0.0429 vs 0.1629 | 9/12 | 0.1460 | 2025-11-19 | 42% | 8/11 |
| har_iv_x vs har_iv | 0.1846 vs 0.3395 | 0.0636 vs 0.1430 | 8/12 | 0.3877 | 2025-11-19 | 39% | 7/11 |
| chronos_cov vs chronos_uni | 0.1671 vs 0.4265 | 0.0518 vs 0.2178 | 10/12 | 0.0386 | 2025-11-19 | 49% | 9/11 |

## 30-calendar-day horizon vs VXN (log variance)

Origins are daily but each target spans 21 trading days, so consecutive rows share almost all of their target. **n = 171, but n_eff = 8 independent windows.** Standard errors below come from a circular moving-block bootstrap (block = 21, 2000 reps) and from refitting on all 21 non-overlapping subsamples — not from HAC(32), which at this n has a lag/n ratio near 0.19 and understates se(beta) by about 3x. config.yaml's own `carry_study` block already requires non-overlapping inference for exactly this reason.

- VXN MZ: alpha=-1.301, beta=0.829, R2=0.295, n=171, n_eff=8
  - bootstrap se(beta)=0.220, 95% CI [0.325, 1.183], p[beta=1]=0.293
  - across the 21 non-overlapping subsamples (~8 obs each): beta ranges [0.298, 1.710], honest se=0.458
  - variance risk premium at the window's median VXN (23.8): **4.1 vol points, 95% CI [1.7, 5.9]**. The frozen report prints this as a single number; at n_eff=8 it is an interval or it is nothing.

Encompassing: realized = a + b*VXN + c*model. H3 wants c>0 and significant. A negative c is not evidence for the model — with VXN already in the regression it means the forecast enters against realized variance, which collinear forecasts commonly do. Models that consume VXN as an input are excluded: regressing on VXN and on a function of VXN is collinear by construction and the split of the coefficients is not interpretable.

- har_cum: MZ beta=0.293 R2=0.063 | encompassing b_implied=1.170 c_model=-0.360 (bootstrap p=0.135, 95% CI [-0.892, 0.096], n=171, n_eff=8, across-subsample c range [-1.076, 0.561])
- persistence_cum: MZ beta=0.142 R2=0.088 | encompassing b_implied=0.875 c_model=-0.024 (bootstrap p=0.296, 95% CI [-0.102, 0.023], n=171, n_eff=8, across-subsample c range [-0.486, 0.606])
- chronos_uni: MZ beta=0.377 R2=0.121 | encompassing b_implied=1.068 c_model=-0.216 (bootstrap p=0.347, 95% CI [-0.761, 0.213], n=171, n_eff=8, across-subsample c range [-1.068, 0.803])
- chronos_cov: MZ beta=0.496 R2=0.143 | encompassing b_implied=0.976 c_model=-0.162 (bootstrap p=0.439, 95% CI [-0.620, 0.262], n=171, n_eff=8, across-subsample c range [-2.028, 1.508])

At n_eff=8 this section cannot reject anything, and a non-significant `c_model` here is not evidence that the model adds nothing beyond VXN — it is evidence that the window is too short to tell. Read the CIs, not the p-values.

---

**Specification dates missing from the registry:** `har_sv`, `har_ic`. These are treated as exploratory until `spec_registry.yaml` records when they were written.