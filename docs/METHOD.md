# Method, leakage policy, and pre-registered hypotheses

*Split out of `README.md` on 2026-08-13 to keep the top-level document short.
Nothing here changed in the move except the added verdict-vocabulary section at the end, which documents the corrected fork.*

## Pre-registered hypotheses

- **H1 (primary):** Chronos-2 with calendar covariates (`chronos_cov`) achieves
  lower mean QLIKE than rolling log-HAR-RV for h=1 forecasts over the clean
  window. Test: Diebold–Mariano, HAC errors, α=0.05.
- **H2:** covariates help where they should — `chronos_cov` beats `chronos_uni`
  on FOMC/CPI/heavy-earnings days (descriptive until event counts grow; see
  Power, below).
- **H3 (market test):** in the encompassing regression at the 30-calendar-day
  horizon, the model's forecast carries a significant coefficient alongside
  VXN-implied variance. This is the only result that would suggest a trade
  rather than a good vol model.
- **Interval quality gate:** 90% intervals must not show violation *clustering*
  (Christoffersen independence p > 0.05). Calibrated-on-average but clustered
  is a fail for any premium-selling use.

Config is frozen at clean-phase start. Changes after observing clean-phase
results go in `reports/AMENDMENTS.md` and restart the accrual clock.

## Leakage policy (the reason this design looks the way it does)

The Chronos-2 checkpoint was published 2025-10-17. Anything before that date
may be in or adjacent to its training corpus, so:

- **Diagnostic phase** (2016 → 2025-10-17): pipeline validation and baseline
  benchmarking only. Chronos results here are labeled contaminated and never
  reported as evidence.
- **Clean phase** (2025-11-03 → open): the only window where Chronos results
  count. Data accrue forward daily, but the frozen specification is not
  re-evaluated until the pre-committed gate of **500 scored origins or
  2027-06-30**, whichever comes first. No monthly peeking.

The date rule is a fallback for models whose training corpus is undisclosed. For
a model that publishes an enumerable corpus, check the corpus instead — see
[`reports/LEAKAGE_TIREX2.md`](../reports/LEAKAGE_TIREX2.md), which does this for
TiRex-2 and explains why its results are reported on the full clean window with
the post-publication subwindow as a robustness check.


## Method summary

- **Target:** every original QQQ headline table in this repository was produced with
  `SOURCE=daily`, because Polygon was not purchased. Thus `rv_total` is the
  yfinance daily-OHLC **Garman–Klass estimate plus the squared overnight gap**;
  it is not five-minute realized variance. The code supports a preferred future
  5-minute RTH sum plus the same overnight gap. Total variance is used so the
  30-day comparison lines up with VXN's calendar-time quote: expected
  30-calendar-day variance = (VXN/100)² × 30/365.
- **Covariates (all known ex ante):** FOMC/CPI/NFP flags and capped trading-day
  countdowns, `earnings_wt` (sum of NDX weights printing BMO that day + AMC the
  prior trading day), day-of-week.
- **Models:** persistence (+250d empirical Δ quantiles), EWMA λ=0.94 (QLIKE
  only), log-HAR-RV, HAR-X (+event terms), **HAR-IV** (log-HAR + log VXN),
  Chronos-2 and TiRex-2 each as `uni` / `cov` / `cov_iv` (`cov_iv` adds VXN as
  a past-observed covariate — a separate information set; a win for `cov_iv` is
  a weaker claim than a win for `cov`).
- **Controls:** any model handed VXN is scored against **HAR-IV**, not plain
  HAR. Beating plain HAR with VXN in hand only re-establishes that implied vol
  predicts realized vol. This control is what closed the Chronos thread on the
  first clean window — see [`reports/FINDINGS.md`](../reports/FINDINGS.md).
- **Scoring:** QLIKE on variance (primary; robust to RV proxy noise), CRPS and
  pinball on log-RV quantiles, 90% coverage + Kupiec + Christoffersen, DM vs
  HAR, event-sliced QLIKE, and at 30 calendar days: Mincer–Zarnowitz plus the
  encompassing regression against VXN.
- **Point-variance convention:** quantile models get a truncated-mean estimator
  (trapezoid of exp(q) over the τ grid). It understates the true conditional
  mean by **0.866–0.873** across five HAR specifications, but by **0.812–0.883
  (7.05pp)** across every quantile model actually scored — `persistence` sits
  at 0.812. So QLIKE *levels* are wrong by ~13%, are not comparable across
  quantile grids at all, and rankings are **not** guaranteed safe: on the clean
  window `har_sv vs har` crosses α=0.05 when the estimator is corrected. EWMA
  uses its native variance forecast, is not understated, and is flagged
  accordingly. `--estimator smearing` replaces this with exact Duan smearing;
  see [`reports/METHODOLOGY_FORK.md`](../reports/METHODOLOGY_FORK.md).

## Power — read before believing anything

Clean window as of 2026-08-11 (measured, not estimated): **193 trading days, 192
scored origins, 6 FOMC decisions, 8 CPI prints, 9 NFP prints, 30 days carrying
earnings weight, 12 in the frozen ≥5% heavy-earnings slice** (157 in the
diagnostic window). The full-window DM test has
reasonable power for large differences only. Event-sliced comparisons on n=6–12
are descriptive, full stop. The value of
the harness is that it accrues clean data forward mechanically; the honest
timeline for event-conditional conclusions is 1–2 more years of accrual.
Resist the urge to peek monthly and stop at the first significant p.


## Verdict vocabulary (added 2026-08-13 with the corrected fork)

The frozen reports carry a Diebold–Mariano statistic and a p-value. That is not
enough to distinguish "these two forecasts are the same" from "this window
cannot tell them apart", and the repository spent months conflating the two.
The corrected reports (`results_*_inf.md`, `results_*_v2.md`) carry four extra
columns and a three-way verdict:

| column | what it answers |
|---|---|
| `MDE` | the smallest QLIKE gap this sample could detect at 80% power |
| `n_req` | origins needed to resolve the gap *actually observed* |
| `p_TOST` | two one-sided tests against a declared 3%-of-benchmark margin |
| `verdict` | `A better` / `B better` / `equivalent` / `inconclusive` |

**A non-significant DM alone can never produce `equivalent`.** Without the
equivalence test rejecting, it is `inconclusive`. This is the single most
important thing to know when reading any "no difference" claim in this
repository — including ones written before the fork existed.

Two further guards:

- **Specification status.** `spec_registry.yaml` records when each model spec
  was written. A model is *confirmatory* in a phase only from
  `max(phase_start, specified_on, available_from)`; everything else is
  *exploratory*, reported but never quotable as a confirmatory result.
  Undetermined dates count as exploratory.
- **`n_eff`, not `n`.** The 30-calendar-day regressions run on daily origins
  whose targets overlap ~21 trading days. They report effective sample size
  beside the raw count — n=171 is n_eff=8.

### Known miscalibration, stated rather than buried

The block bootstrap that replaced HAC(32) in the 30-day section is itself
anti-conservative: measured nominal-95% coverage is ~82%, and the coefficient
p-values have true size ~10–16%. It is still better than HAC at lag/n = 0.19,
which is why it is used, but its intervals are a **floor on the uncertainty,
not a calibrated interval**. Pinned by
`tests/test_methodology.py::TestBootstrapCalibration`.

### The point estimator is not neutral

`--estimator smearing` replaces the frozen truncated-mean point forecast with
exact Duan smearing. QLIKE is `rv/v − log(rv/v) − 1`, which is **not**
scale-invariant, so a common rescaling of both forecasts moves the DM statistic
rather than cancelling out of it. Every DM row involving a quantile model is
therefore estimator-dependent. Three rows cross α=0.05 today:
`har_sv vs har` (both grids, clean) and `har_iv_lev vs har_iv` (diagnostic).
See `reports/METHODOLOGY_FORK.md`.
