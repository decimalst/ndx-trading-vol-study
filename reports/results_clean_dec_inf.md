# Results — phase: clean

Quantile grid: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] → intervals below are 80%. **Decile grid, not the pre-registered one.** TiRex-2 emits only these levels, so every model here was recomputed on them; `mean_var` is comparable within this report but not against `results_*.md`. config.yaml is unchanged.

**Corrected methodology run** — estimator=`trunc`, inference=`corrected`. This report is a fork; `results_clean_dec.md` still holds the frozen pre-registered numbers and is unchanged. Differences between the two are methodology, not data. See src/methodology.py for what each correction does and why.

## h=1 losses (mean per day)

| model | n | status | QLIKE | CRPS | pin0.10 | pin0.90 | 80% cov | p_uc | p_ind |
|---|---|---|---|---|---|---|---|---|---|
| persistence | 192 | confirmatory | 0.5909 | 0.6825 | 0.1812 | 0.1850 | 0.781 | 0.521 | 0.052 |
| ewma | 192 | confirmatory | 0.4036 | - | - | - | - | - | - |
| har | 192 | confirmatory | 0.3860 | 0.5526 | 0.1585 | 0.1552 | 0.750 | 0.092 | 0.112 |
| har_x | 192 | confirmatory | 0.3734 | 0.5409 | 0.1573 | 0.1521 | 0.750 | 0.092 | 0.225 |
| har_iv | 192 | confirmatory | 0.3167 | 0.5075 | 0.1402 | 0.1427 | 0.812 | 0.662 | 0.848 |
| har_sv | 188 | **exploratory** | 0.3593 | 0.5158 | 0.1458 | 0.1415 | 0.830 | 0.298 | 0.202 |
| har_ic | 192 | **exploratory** | 0.3229 | 0.5120 | 0.1409 | 0.1429 | 0.792 | 0.774 | 0.716 |
| har_iv_x | 192 | **exploratory** | 0.3073 | 0.4949 | 0.1385 | 0.1396 | 0.807 | 0.800 | 0.635 |
| chronos_uni | 192 | confirmatory | 0.3924 | 0.5499 | 0.1499 | 0.1518 | 0.812 | 0.662 | 0.143 |
| chronos_cov | 192 | confirmatory | 0.3851 | 0.5498 | 0.1540 | 0.1522 | 0.760 | 0.180 | 0.453 |
| chronos_cov_iv | 192 | confirmatory | 0.3173 | 0.5036 | 0.1350 | 0.1456 | 0.812 | 0.662 | 0.773 |
| tirex_uni | 192 | **exploratory** | 0.3975 | 0.5522 | 0.1510 | 0.1528 | 0.786 | 0.642 | 0.181 |
| tirex_cov | 192 | **exploratory** | 0.3885 | 0.5572 | 0.1518 | 0.1538 | 0.771 | 0.321 | 0.252 |
| tirex_cov_iv | 192 | **exploratory** | 0.3878 | 0.5518 | 0.1518 | 0.1537 | 0.797 | 0.914 | 0.083 |
| tirex_cov_ivf | 192 | **exploratory** | 0.3296 | 0.4929 | 0.1291 | 0.1483 | 0.740 | 0.043 | 0.948 |

### Specification status

A model is *confirmatory* in this phase only from `max(phase_start, specified_on, available_from)` onward. Anything specified after the window opened was tested on the data that produced it, so its p-values in the tables below are descriptive, not inferential. The evaluator already applied this rule to `har_lev`/`har_iv_lev` by hand; `spec_registry.yaml` applies it to everything.

| model | why not confirmatory | confirmatory origins available | gate |
|---|---|---|---|
| har_sv | specification date not recorded | 0 (date unrecorded) | 500 |
| har_ic | specification date not recorded | 0 (date unrecorded) | 500 |
| har_iv_x | specified 2026-08-11, after the phase opened | 0 | 500 |
| tirex_uni | specified 2026-07-01, after the phase opened; model available 2026-07-01 | 28 | 500 |
| tirex_cov | specified 2026-07-01, after the phase opened; model available 2026-07-01 | 28 | 500 |
| tirex_cov_iv | specified 2026-07-01, after the phase opened; model available 2026-07-01 | 28 | 500 |
| tirex_cov_ivf | specified 2026-07-01, after the phase opened; model available 2026-07-01 | 28 | 500 |

Every row above is below its gate. Those models are reported for completeness and must not be quoted as confirmatory results.


## Diebold-Mariano vs HAR (QLIKE; negative = beats HAR)

`MDE` is the smallest QLIKE gap this sample could detect at 80% power; `n_req` is how many origins it would take to resolve the gap actually observed. `p_TOST` tests EQUIVALENCE against a margin of 3% of the benchmark's loss. A non-significant DM alone never earns the verdict `equivalent` — without the TOST it is `inconclusive`.

| model | DM | p | n | MDE | n_req | p_TOST | verdict |
|---|---|---|---|---|---|---|---|
| persistence | 3.191 | 0.0017 | 192 | 0.1799 | 148 | 0.999 | B better |
| ewma | 0.538 | 0.5910 | 192 | 0.0916 | 5,201 | 0.573 | inconclusive |
| har_x | -1.404 | 0.1621 | 192 | 0.0252 | 765 | 0.546 | inconclusive |
| har_iv | -3.573 | 0.0004 | 192 | 0.0544 | 118 | 0.998 | A better |
| har_sv * | -1.072 | 0.2850 | 188 | 0.0638 | 1,284 | 0.714 | inconclusive |
| har_ic * | -3.140 | 0.0020 | 192 | 0.0563 | 153 | 0.994 | A better |
| har_iv_x * | -3.718 | 0.0003 | 192 | 0.0593 | 109 | 0.999 | A better |
| chronos_uni | 0.619 | 0.5367 | 192 | 0.0287 | 3,934 | 0.305 | inconclusive |
| chronos_cov | -0.045 | 0.9639 | 192 | 0.0581 | 731,842 | 0.304 | inconclusive |
| chronos_cov_iv | -3.229 | 0.0015 | 192 | 0.0597 | 145 | 0.996 | A better |
| tirex_uni * | 0.950 | 0.3431 | 192 | 0.0340 | 1,668 | 0.498 | inconclusive |
| tirex_cov * | 0.196 | 0.8445 | 192 | 0.0351 | 39,066 | 0.234 | inconclusive |
| tirex_cov_iv * | 0.152 | 0.8791 | 192 | 0.0324 | 64,950 | 0.198 | inconclusive |
| tirex_cov_ivf * | -2.823 | 0.0053 | 192 | 0.0560 | 189 | 0.987 | A better |

`*` = exploratory specification (see the table above).

## Diebold-Mariano vs HAR-IV — same information set

HAR-IV is log-HAR plus log(VXN). Any model fed VXN beats plain HAR trivially, because implied vol predicts realized vol. The question this table answers is whether the foundation model extracts more from VXN than four OLS terms do.

`MDE` is the smallest QLIKE gap this sample could detect at 80% power; `n_req` is how many origins it would take to resolve the gap actually observed. `p_TOST` tests EQUIVALENCE against a margin of 3% of the benchmark's loss. A non-significant DM alone never earns the verdict `equivalent` — without the TOST it is `inconclusive`.

| model | DM | p | n | MDE | n_req | p_TOST | verdict |
|---|---|---|---|---|---|---|---|
| har_ic * | 1.647 | 0.1012 | 192 | 0.0107 | 555 | 0.200 | inconclusive |
| har_iv_x * | -1.180 | 0.2394 | 192 | 0.0222 | 1,082 | 0.492 | inconclusive |
| chronos_cov_iv | 0.034 | 0.9732 | 192 | 0.0505 | 1,335,780 | 0.311 | inconclusive |
| tirex_cov_iv * | 3.629 | 0.0004 | 192 | 0.0549 | 114 | 0.999 | B better |
| tirex_cov_ivf * | 0.839 | 0.4024 | 192 | 0.0431 | 2,140 | 0.587 | inconclusive |

`*` = exploratory specification (see the table above).

## TiRex-2 robustness: origins after publication (2026-07-01)

The date-based leakage rule applied literally to TiRex-2. Tiny sample by construction — this is a sign check against the full window above, not a test. See reports/LEAKAGE_TIREX2.md.

| model | QLIKE (full) | QLIKE (post-pub) | n |
|---|---|---|---|
| tirex_uni | 0.3975 | 0.2409 | 28 |
| tirex_cov | 0.3885 | 0.2287 | 28 |
| tirex_cov_iv | 0.3878 | 0.2303 | 28 |
| tirex_cov_ivf | 0.3296 | 0.1882 | 28 |
| har | 0.3860 | 0.2334 | 28 |
| har_iv | 0.3167 | 0.1991 | 28 |
| chronos_cov | 0.3851 | 0.2435 | 28 |

## Event-sliced QLIKE (mean)

| model | FOMC (n) | CPI (n) | heavy-earnings (n) | quiet (n) |
|---|---|---|---|---|
| persistence | 0.2428 (6) | 0.8017 (8) | 0.7887 (12) | 0.5790 (166) |
| ewma | 0.2360 (6) | 0.2623 (8) | 0.6117 (12) | 0.4014 (166) |
| har | 0.1711 (6) | 0.4041 (8) | 0.5113 (12) | 0.3839 (166) |
| har_x | 0.2340 (6) | 0.3599 (8) | 0.2837 (12) | 0.3856 (166) |
| har_iv | 0.1694 (6) | 0.3070 (8) | 0.3975 (12) | 0.3166 (166) |
| har_sv | 0.2136 (6) | 0.3344 (8) | 0.5314 (12) | 0.3531 (162) |
| har_ic | 0.1713 (6) | 0.3162 (8) | 0.4189 (12) | 0.3218 (166) |
| har_iv_x | 0.2264 (6) | 0.2933 (8) | 0.2089 (12) | 0.3180 (166) |
| chronos_uni | 0.1421 (6) | 0.4116 (8) | 0.5111 (12) | 0.3919 (166) |
| chronos_cov | 0.2152 (6) | 0.4048 (8) | 0.2001 (12) | 0.4036 (166) |
| chronos_cov_iv | 0.2391 (6) | 0.3271 (8) | 0.1507 (12) | 0.3317 (166) |
| tirex_uni | 0.1279 (6) | 0.3987 (8) | 0.5794 (12) | 0.3941 (166) |
| tirex_cov | 0.1665 (6) | 0.4532 (8) | 0.4576 (12) | 0.3884 (166) |
| tirex_cov_iv | 0.1435 (6) | 0.4187 (8) | 0.5120 (12) | 0.3861 (166) |
| tirex_cov_ivf | 0.1347 (6) | 0.2751 (8) | 0.3581 (12) | 0.3372 (166) |

## Full-sample paired per-origin (all origins in phase)

Win rate and mean difference answer different questions. A high win rate with no mean gain is many small wins funded by rare large losses; a low win rate with a mean gain is the reverse. Read both before believing either.

| pair | n | wins | win % | sign p | mean | median | top-10 share of mean gap |
|---|---|---|---|---|---|---|---|
| har_iv_x vs har_iv | 192 | 117 | 60.9% | 0.00299 | 0.3073 vs 0.3167 | 0.1070 vs 0.1189 | 178% |
| har_x vs har | 192 | 113 | 58.9% | 0.017 | 0.3734 vs 0.3860 | 0.1370 vs 0.1389 | 159% |
| har_sv vs har | 188 | 102 | 54.3% | 0.274 | 0.3593 vs 0.3837 | 0.1277 vs 0.1389 | 149% |
| har_iv vs har | 192 | 115 | 59.9% | 0.00742 | 0.3167 vs 0.3860 | 0.1189 vs 0.1389 | 67% |
| chronos_cov vs chronos_uni | 192 | 98 | 51.0% | 0.829 | 0.3851 vs 0.3924 | 0.1517 vs 0.1528 | 481% |
| tirex_cov vs tirex_uni | 192 | 86 | 44.8% | 0.17 | 0.3885 vs 0.3975 | 0.1558 vs 0.1526 | 230% |

## Replication: clean vs diagnostic

`flip k` is how many origins must be dropped from the winning tail before the mean gap changes sign — a direct read on whether a result is a property of the sample or of a handful of days. A pair that wins in clean and loses in diagnostic has not replicated, whatever its p-value in either window.

| pair | clean win% | clean gap | flip k | diagnostic win% | diagnostic gap | diagnostic n | replicates? |
|---|---|---|---|---|---|---|---|
| har_iv_x vs har_iv | 60.9% | +0.0093 | 3 | 59.4% | +0.0012 | 2463 | yes |
| har_x vs har | 58.9% | +0.0126 | 4 | 58.9% | -0.0021 | 2463 | **no** |
| har_sv vs har | 54.3% | +0.0244 | 5 | — | — | — | not testable |
| har_iv vs har | 59.9% | +0.0694 | 25 | 55.2% | +0.0385 | 2463 | yes |
| chronos_cov vs chronos_uni | 51.0% | +0.0073 | 1 | — | — | — | not testable |
| tirex_cov vs tirex_uni | 44.8% | +0.0091 | 2 | — | — | — | not testable |

### Heavy-earnings slice, paired per-origin (cutoff 5.0% of index weight)

| pair | mean | median | better/n | sign p | top day | % of gap | ex-top |
|---|---|---|---|---|---|---|---|
| har_x vs har | 0.2837 vs 0.5113 | 0.0650 vs 0.2121 | 9/12 | 0.1460 | 2025-11-19 | 40% | 8/11 |
| har_iv_x vs har_iv | 0.2089 vs 0.3975 | 0.0785 vs 0.1947 | 9/12 | 0.1460 | 2025-11-19 | 36% | 8/11 |
| chronos_cov vs chronos_uni | 0.2001 vs 0.5111 | 0.0778 vs 0.2389 | 11/12 | 0.0063 | 2025-11-19 | 45% | 10/11 |
| tirex_cov vs tirex_uni | 0.4576 vs 0.5794 | 0.2602 vs 0.2999 | 8/12 | 0.3877 | 2025-11-19 | 69% | 7/11 |

### Pre-committed gate: reachability

`next_evaluation.earnings_slice_confirmatory` requires **40 heavy-earnings days** before the registered confirmatory DM test may run. This phase has **12**. Heavy-earnings days arrive at 5.35% of origins over the full sample, so the `at_origins: 500` trigger projects to **~27** — a shortfall of ~13. Reaching 40 takes roughly **748 origins** at the observed rate.

**The gate is therefore not reachable at its own trigger.** Over every rolling window in this sample the maximum count attained in 500 sessions is well below 40. This is disclosed rather than repaired: `config.yaml` is frozen, so the floor stands and the shortfall is published beside it. See `reports/AMENDMENTS.md`.


## 30-calendar-day horizon vs VXN (log variance)

Origins are daily but each target spans 21 trading days, so consecutive rows share almost all of their target. **n = 171, but n_eff = 8 independent windows.** Standard errors below come from a circular moving-block bootstrap (block = 21, 2000 reps) and from refitting on all 21 non-overlapping subsamples — not from HAC(32), which at this n has a lag/n ratio of 0.19 and returns se(beta)=0.160 against 0.220 from the bootstrap and a 0.360 spread of the point estimate across the 21 starting offsets. The bootstrap interval is itself too narrow — see the calibration note below. config.yaml's own `carry_study` block already requires non-overlapping inference for exactly this reason.

- VXN MZ: alpha=-1.301, beta=0.829, R2=0.295, n=171, n_eff=8
  - bootstrap se(beta)=0.220, 95% CI [0.325, 1.183], p[beta=1]=0.293
  - across the 21 non-overlapping subsamples (~8 obs each): beta ranges [0.298, 1.710], sd across starting offsets=0.360; median within-subsample se=0.458 (the se of a beta fitted to ~8 points, **not** of the full-sample beta — it was previously published as "honest se", which overstated the reported coefficient's uncertainty)
  - variance risk premium at the window's median VXN (23.8): **4.1 vol points, 95% CI [1.7, 5.9]**. The frozen report prints this as a single number; at n_eff=8 it is an interval or it is nothing.

Encompassing: realized = a + b*VXN + c*model. H3 wants c>0 and significant. A negative c is not evidence for the model — with VXN already in the regression it means the forecast enters against realized variance, which collinear forecasts commonly do. Models that consume VXN as an input are excluded: regressing on VXN and on a function of VXN is collinear by construction and the split of the coefficients is not interpretable.

- har_cum: MZ beta=0.293 R2=0.063 | encompassing b_implied=1.170 c_model=-0.360 (bootstrap p=0.135, 95% CI [-0.892, 0.096], n=171, n_eff=8, across-subsample c range [-1.076, 0.561])
- persistence_cum: MZ beta=0.142 R2=0.088 | encompassing b_implied=0.875 c_model=-0.024 (bootstrap p=0.296, 95% CI [-0.102, 0.023], n=171, n_eff=8, across-subsample c range [-0.486, 0.606])
- chronos_uni: MZ beta=0.399 R2=0.130 | encompassing b_implied=1.063 c_model=-0.214 (bootstrap p=0.370, 95% CI [-0.759, 0.233], n=171, n_eff=8, across-subsample c range [-1.138, 0.996])
- chronos_cov: MZ beta=0.513 R2=0.153 | encompassing b_implied=0.960 c_model=-0.143 (bootstrap p=0.468, 95% CI [-0.578, 0.273], n=171, n_eff=8, across-subsample c range [-2.233, 1.546])
- tirex_uni: MZ beta=0.567 R2=0.134 | encompassing b_implied=1.077 c_model=-0.313 (bootstrap p=0.300, 95% CI [-0.941, 0.301], n=171, n_eff=8, across-subsample c range [-2.328, 1.768])
- tirex_cov: MZ beta=0.431 R2=0.129 | encompassing b_implied=1.013 c_model=-0.187 (bootstrap p=0.309, 95% CI [-0.641, 0.179], n=171, n_eff=8, across-subsample c range [-1.635, 0.658])

At n_eff=8 this section cannot reject anything, and a non-significant `c_model` here is not evidence that the model adds nothing beyond VXN — it is evidence that the window is too short to tell. Read the CIs, not the p-values.

---

**Specification dates missing from the registry:** `har_sv`, `har_ic`. These are treated as exploratory until `spec_registry.yaml` records when they were written.