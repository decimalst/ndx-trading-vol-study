# Results — phase: diagnostic

Quantile grid: [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95] → intervals below are 90%.

**Corrected methodology run** — estimator=`trunc`, inference=`corrected`. This report is a fork; `results_diagnostic.md` still holds the frozen pre-registered numbers and is unchanged. Differences between the two are methodology, not data. See src/methodology.py for what each correction does and why.

## h=1 losses (mean per day)

| model | n | status | QLIKE | CRPS | pin0.05 | pin0.95 | 90% cov | p_uc | p_ind |
|---|---|---|---|---|---|---|---|---|---|
| persistence | 2463 | confirmatory | 0.5574 | 0.6054 | 0.1040 | 0.1084 | 0.892 | 0.214 | 0.000 |
| ewma | 2463 | confirmatory | 0.4506 | - | - | - | - | - | - |
| har | 2463 | confirmatory | 0.3815 | 0.4950 | 0.0840 | 0.0928 | 0.863 | 0.000 | 0.138 |
| har_x | 2463 | confirmatory | 0.3838 | 0.4945 | 0.0867 | 0.0935 | 0.868 | 0.000 | 0.044 |
| har_iv | 2463 | confirmatory | 0.3438 | 0.4714 | 0.0809 | 0.0869 | 0.891 | 0.150 | 0.237 |
| har_sv | 22 | **exploratory** | 0.3729 | 0.4758 | 0.0736 | 0.0949 | 0.909 | 0.885 | 0.516 |
| har_ic | 2351 | **exploratory** | 0.3479 | 0.4749 | 0.0813 | 0.0875 | 0.886 | 0.031 | 0.121 |
| har_iv_x | 2463 | **exploratory** | 0.3437 | 0.4721 | 0.0842 | 0.0866 | 0.892 | 0.214 | 0.607 |
| har_lev | 2463 | **exploratory** | 0.3618 | 0.4805 | 0.0816 | 0.0903 | 0.867 | 0.000 | 0.259 |
| har_iv_lev | 2463 | **exploratory** | 0.3376 | 0.4632 | 0.0796 | 0.0863 | 0.895 | 0.435 | 0.147 |

### Specification status

A model is *confirmatory* in this phase only from `max(phase_start, specified_on, available_from)` onward. Anything specified after the window opened was tested on the data that produced it, so its p-values in the tables below are descriptive, not inferential. The evaluator already applied this rule to `har_lev`/`har_iv_lev` by hand; `spec_registry.yaml` applies it to everything.

| model | why not confirmatory | confirmatory origins available | gate |
|---|---|---|---|
| har_sv | specification date not recorded | 0 (date unrecorded) | 500 |
| har_ic | specification date not recorded | 0 (date unrecorded) | 500 |
| har_iv_x | specified 2026-08-11, after the phase opened | 0 | 500 |
| har_lev | specified 2026-08-11, after the phase opened | 0 | 500 |
| har_iv_lev | specified 2026-08-11, after the phase opened | 0 | 500 |

Every row above is below its gate. Those models are reported for completeness and must not be quoted as confirmatory results.


## Diebold-Mariano vs HAR (QLIKE; negative = beats HAR)

`MDE` is the smallest QLIKE gap this sample could detect at 80% power; `n_req` is how many origins it would take to resolve the gap actually observed. `p_TOST` tests EQUIVALENCE against a margin of 3% of the benchmark's loss. A non-significant DM alone never earns the verdict `equivalent` — without the TOST it is `inconclusive`.

| model | DM | p | n | MDE | n_req | p_TOST | verdict |
|---|---|---|---|---|---|---|---|
| persistence | 9.534 | 0.0000 | 2463 | 0.0517 | 213 | 1.000 | B better |
| ewma | 6.969 | 0.0000 | 2463 | 0.0278 | 398 | 1.000 | B better |
| har_x | 0.567 | 0.5705 | 2463 | 0.0111 | 60,046 | 0.010 | equivalent |
| har_iv | -5.148 | 0.0000 | 2463 | 0.0205 | 729 | 1.000 | A better |
| har_sv * | -0.290 | 0.7744 | 22 | 0.2604 | 2,048 | 0.563 | inconclusive |
| har_ic * | -4.115 | 0.0000 | 2351 | 0.0259 | 1,090 | 0.998 | A better |
| har_iv_x * | -4.425 | 0.0000 | 2463 | 0.0239 | 987 | 0.999 | A better |
| har_lev * | -5.367 | 0.0000 | 2463 | 0.0103 | 671 | 0.988 | A better |
| har_iv_lev * | -5.212 | 0.0000 | 2463 | 0.0236 | 712 | 1.000 | A better |

`*` = exploratory specification (see the table above).

## Diebold-Mariano vs HAR-IV — same information set

HAR-IV is log-HAR plus log(VXN). Any model fed VXN beats plain HAR trivially, because implied vol predicts realized vol. The question this table answers is whether the foundation model extracts more from VXN than four OLS terms do.

`MDE` is the smallest QLIKE gap this sample could detect at 80% power; `n_req` is how many origins it would take to resolve the gap actually observed. `p_TOST` tests EQUIVALENCE against a margin of 3% of the benchmark's loss. A non-significant DM alone never earns the verdict `equivalent` — without the TOST it is `inconclusive`.

| model | DM | p | n | MDE | n_req | p_TOST | verdict |
|---|---|---|---|---|---|---|---|
| har_ic * | 0.258 | 0.7967 | 2351 | 0.0158 | 278,037 | 0.056 | inconclusive |
| har_iv_x * | -0.031 | 0.9750 | 2463 | 0.0117 | 19,719,557 | 0.008 | equivalent |
| har_iv_lev * | -1.410 | 0.1585 | 2463 | 0.0123 | 9,717 | 0.175 | inconclusive |

`*` = exploratory specification (see the table above).

## Event-sliced QLIKE (mean)

| model | FOMC (n) | CPI (n) | heavy-earnings (n) | quiet (n) |
|---|---|---|---|---|
| persistence | 0.6380 (79) | 0.9142 (115) | 0.6452 (157) | 0.5270 (2141) |
| ewma | 0.3626 (79) | 0.4427 (115) | 0.4845 (157) | 0.4517 (2141) |
| har | 0.3524 (79) | 0.5262 (115) | 0.4417 (157) | 0.3702 (2141) |
| har_x | 0.2830 (79) | 0.5156 (115) | 0.3522 (157) | 0.3814 (2141) |
| har_iv | 0.3354 (79) | 0.4569 (115) | 0.3874 (157) | 0.3350 (2141) |
| har_sv | - | - | - | 0.3729 (22) |
| har_ic | 0.3199 (76) | 0.4678 (110) | 0.3541 (146) | 0.3427 (2045) |
| har_iv_x | 0.2856 (79) | 0.4387 (115) | 0.2900 (157) | 0.3441 (2141) |
| har_lev | 0.3319 (79) | 0.4955 (115) | 0.4359 (157) | 0.3498 (2141) |
| har_iv_lev | 0.3250 (79) | 0.4600 (115) | 0.4172 (157) | 0.3254 (2141) |

## Full-sample paired per-origin (all origins in phase)

Win rate and mean difference answer different questions. A high win rate with no mean gain is many small wins funded by rare large losses; a low win rate with a mean gain is the reverse. Read both before believing either.

| pair | n | wins | win % | sign p | mean | median | top-10 share of mean gap |
|---|---|---|---|---|---|---|---|
| har_iv_lev vs har_iv | 2463 | 1366 | 55.5% | 6.47e-08 | 0.3376 vs 0.3438 | 0.1437 vs 0.1486 | 126% |
| har_lev vs har | 2463 | 1417 | 57.5% | 8.06e-14 | 0.3618 vs 0.3815 | 0.1481 vs 0.1592 | 39% |
| har_iv_x vs har_iv | 2463 | 1511 | 61.3% | 1.44e-29 | 0.3437 vs 0.3438 | 0.1418 vs 0.1486 | 5300% |
| har_x vs har | 2463 | 1496 | 60.7% | 1.26e-26 | 0.3838 vs 0.3815 | 0.1520 vs 0.1592 | -244% |
| har_iv vs har | 2463 | 1333 | 54.1% | 4.65e-05 | 0.3438 vs 0.3815 | 0.1486 vs 0.1592 | 31% |

## Replication: diagnostic vs clean

`flip k` is how many origins must be dropped from the winning tail before the mean gap changes sign — a direct read on whether a result is a property of the sample or of a handful of days. A pair that wins in diagnostic and loses in clean has not replicated, whatever its p-value in either window.

| pair | diagnostic win% | diagnostic gap | flip k | clean win% | clean gap | clean n | replicates? |
|---|---|---|---|---|---|---|---|
| har_iv_lev vs har_iv | 55.5% | +0.0062 | 7 | 55.2% | -0.0073 | 192 | **no** |
| har_lev vs har | 57.5% | +0.0198 | never | 56.2% | +0.0113 | 192 | yes |
| har_iv_x vs har_iv | 61.3% | +0.0001 | 1 | 62.5% | +0.0089 | 192 | yes |
| har_x vs har | 60.7% | -0.0023 | 0 | 58.3% | +0.0114 | 192 | **no** |
| har_iv vs har | 54.1% | +0.0377 | never | 57.8% | +0.0636 | 192 | yes |

### Heavy-earnings slice, paired per-origin (cutoff 5.0% of index weight)

| pair | mean | median | better/n | sign p | top day | % of gap | ex-top |
|---|---|---|---|---|---|---|---|
| har_x vs har | 0.3522 vs 0.4417 | 0.1533 vs 0.2073 | 71/157 | 0.2638 | 2017-10-26 | 29% | 70/156 |
| har_iv_x vs har_iv | 0.2900 vs 0.3874 | 0.1341 vs 0.1582 | 69/157 | 0.1506 | 2016-01-14 | 22% | 68/156 |

## 30-calendar-day horizon vs VXN (log variance)

Origins are daily but each target spans 21 trading days, so consecutive rows share almost all of their target. **n = 2463, but n_eff = 117 independent windows.** Standard errors below come from a circular moving-block bootstrap (block = 21, 2000 reps) and from refitting on all 21 non-overlapping subsamples — not from HAC(32), which at this n has a lag/n ratio near 0.19 and understates se(beta) by about 3x. config.yaml's own `carry_study` block already requires non-overlapping inference for exactly this reason.

- VXN MZ: alpha=-0.040, beta=1.067, R2=0.550, n=2463, n_eff=117
  - bootstrap se(beta)=0.060, 95% CI [0.951, 1.179], p[beta=1]=0.273
  - across the 21 non-overlapping subsamples (~117 obs each): beta ranges [0.994, 1.116], honest se=0.090
  - variance risk premium at the window's median VXN (20.4): **3.9 vol points, 95% CI [3.1, 4.5]**. The frozen report prints this as a single number; at n_eff=117 it is an interval or it is nothing.

Encompassing: realized = a + b*VXN + c*model. H3 wants c>0 and significant. A negative c is not evidence for the model — with VXN already in the regression it means the forecast enters against realized variance, which collinear forecasts commonly do. Models that consume VXN as an input are excluded: regressing on VXN and on a function of VXN is collinear by construction and the split of the coefficients is not interpretable.

- har_cum: MZ beta=0.810 R2=0.485 | encompassing b_implied=0.851 c_model=0.196 (bootstrap p=0.034, 95% CI [0.017, 0.383], n=2463, n_eff=117, across-subsample c range [0.056, 0.325])
- persistence_cum: MZ beta=0.443 R2=0.360 | encompassing b_implied=0.954 c_model=0.078 (bootstrap p=0.002, 95% CI [0.027, 0.133], n=2463, n_eff=117, across-subsample c range [-0.116, 0.174])

At n_eff=117 this section cannot reject anything, and a non-significant `c_model` here is not evidence that the model adds nothing beyond VXN — it is evidence that the window is too short to tell. Read the CIs, not the p-values.

---

**Specification dates missing from the registry:** `har_sv`, `har_ic`. These are treated as exploratory until `spec_registry.yaml` records when they were written.