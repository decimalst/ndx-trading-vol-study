# NDX volatility forecasting: what the market already knows

This repository documents a leakage-controlled research project on Nasdaq-100
realized volatility. I started with a straightforward question: can a modern
time-series model, plus calendar and market data known at the forecast origin,
predict QQQ/NDX variance better than classical models—and, more importantly,
better than the options market already does through VXN?

The short answer is: **volatility is forecastable, but the forecastable part is
largely already priced.** A compact HAR model using VXN is difficult to beat.
Across roughly ten original designs spanning QQQ/NDX and SPX—with
underlying-price, option-implied, and cross-sectional/holdings evidence—extra
index-level signals were usually redundant, non-stationary, contaminated by how
they were discovered, or too weak to survive a frozen holdout. A five-path
extension changed the question from “can this beat the surface?” to “what does
the surface absorb, and what can a 30-day scalar not represent?” That produced a
positive single-name earnings mechanism, an absorption map, and a VRP term
curve, alongside another clean rejection. These designs are not statistically
independent, so their count is not a meta-test; their breadth is the important
result.

This is research code and a record of negative as well as positive evidence. It
is not a trading recommendation or financial advice.

## What I did

I built a point-in-time forecasting harness with explicit leakage fences,
rolling or expanding fits, proper loss functions, event slices, interval tests,
and independent recomputation of the most important verdicts. Before empirical
runs, tests froze source identity, publication lags, target completion, model
comparisons, and holdout boundaries.

The project then moved through several questions:

1. Compare persistence, EWMA, HAR, Chronos-2, and TiRex-2 on QQQ realized
   variance.
2. Add VXN so every model competes against the information already embedded in
   30-day implied volatility.
3. Test FOMC, CPI, payroll, earnings, leverage, cross-asset, liquidity-adjacent,
   overnight, and term-structure features without using future information.
4. Examine volatility carry and whether lagged SKEW repairs its adverse
   selection.
5. Change the target to SPX jump variation and change the model frame to a
   two-state QQQ transition model.
6. Repair the HMM comparison with prior-only Platt calibration and ask the fair
   incremental question: does calibrated state probability add to a supervised
   model trained on the same rows?
7. Reconstruct point-in-time QQQ holdings from SEC N-PORT and specify a properly
   matched future SPX correlation-premium study.
8. Replace the current 13-name selection with the top 25 issuers in each
   SEC-accepted quarterly QQQ snapshot.
9. Measure how much VXN attenuates leverage, weekday, earnings, macro, and
   post-FOMC regularities.
10. Trace VXN's contribution over 1, 5, 10, 21, 42, and 63 sessions.
11. Measure the matched 9/30/93-day SPX volatility-risk-premium curve and run
    historically gated single-name earnings/own-IV diagnostics.
12. Test the regime-conditional VIX9D/VIX hypothesis on an untouched 2014-2015
    SPX window.

## Results in plain English

| Question | What happened | Interpretation | Details |
|---|---|---|---|
| Can foundation models beat a simple volatility model? | No. Chronos-2 and TiRex-2 did not beat HAR-IV, the HAR model that also sees VXN. | Model complexity did not add information beyond a small linear model plus the option-implied level. | [Clean results](reports/results_clean.md), [full findings](reports/FINDINGS.md) |
| Do macro calendars and earnings improve the forecast? | Not reliably. The well-powered earnings comparison was flat, and the concentration defence pointed the wrong way. | Widely known scheduled information appears priced or too small at this horizon. | [Diagnostic results](reports/results_diagnostic.md), [holdings audit](reports/qqq_nport_audit.md) |
| How much leverage information does VXN absorb? | The joint Wald statistic on three return-asymmetry terms falls from 103.19 without VXN to 62.63 with it: about 40% attenuation, with a strongly significant remainder. | This is the clearest positive measurement in the project. The residual information is real in-sample but worth only about 1.8% of diagnostic QLIKE out of sample (DM p=0.159). | [Research amendments](reports/AMENDMENTS.md#2026-08-12-fourth--ledger-closing-test-har_iv_lev), [standing findings](reports/FINDINGS.md) |
| Does short-end term slope help? | It improved QLIKE by 5.3%/7.0% in 2022-2023 but only 0.2%/0.9% in 2024-2025. | The interesting hypothesis is regime-conditional: slope may matter when the front end is dislocated. That yearly pattern is now inspected, so it is prospective-only. | [Independent signal verification](reports/signal_study/verification.md) |
| Does SKEW repair short-vol carry? | Historically it raised mean P&L and improved CVaR, worst loss, and drawdown while keeping 63.3% of entries. | The registered participation gate failed, but the decisive problem is contamination: February 2020 motivated the rule and remains in the sample. | [SKEW diagnostic](reports/skew_carry_diagnostic.md), [protocol](reports/SKEW_CARRY_PROTOCOL.md) |
| Does the SPX jump target reveal a surface signal? | No under the registered comparison. Lagged SKEW worsened the five-minute Oxford-Man RV/BPV proxy model. | This was not based on the local hourly bars, but it still lacks realized quarticity for a formal BNS jump test. | [Target findings](reports/TARGET_REGIME_FINDINGS.md), [jump report](reports/jump_target.md) |
| Does a two-state HMM predict transitions? | Calibration helped substantially; adding the calibrated state to the supervised benchmark slightly worsened both Brier and log loss. | The HMM summarizes and ranks the current state, but supplied no detectable incremental transition information beyond RV history. | [Repair result](reports/regime_repair.md), [repair protocol](reports/REGIME_REPAIR_PROTOCOL.md) |
| Can COR1M be compared with QQQ realized correlation? | No—the universes do not match. | The clean study moves to SPX, where COR1M, DSPX, and VIXEQ match by construction. It awaits exact historical top-50 weights and returns. | [SPX data contract](reports/SPX_DISPERSION_DATA_CONTRACT.md) |
| What does VXN absorb? | About 39% of the leverage Wald statistic and 33% of the point-in-time top-25 earnings statistic; little weekday/macro structure, while it amplifies the post-FOMC contrast. | The surface is informative but not a sufficient statistic for every regularity. This is measurement, not a forecast win. | [Absorption map](reports/research_paths/absorption_map.md) |
| Does VXN's forecast contribution peak near 30 days? | No. Its diagnostic OOS gain peaks at 5 sessions (11.45%) and decays to 6.91% at 21 and 2.64% at 42. | The 30-day risk-neutral quote adds most relative value against HAR at shorter physical horizons in this sample. | [Horizon curve](reports/research_paths/horizon_curve.md) |
| What does the SPX VRP curve look like? | Mean implied-minus-realized vol is 3.61, 3.47, and 4.72 points at 9, 30, and 93 calendar days. | The longest measured premium is richest; only its leverage-state difference excludes zero under the frozen block interval. | [VRP term structure](reports/research_paths/vrp_term_structure.md) |
| Was the index earnings mechanism merely absent? | No. After own implied vol, historically top-25 AAPL/AMZN/Alphabet earnings sessions retain a 2.05-log-variance pooled residual contrast. | Constant-maturity single-name IV does not localize the event to one day; this explains the index null better than a current-name concentration story. | [Single-name earnings](reports/research_paths/single_name_earnings.md) |
| Does regime-conditioned VIX9D/VIX replicate on SPX? | No. It worsens 2014-2015 QLIKE by 0.60%; DM p=0.0325 points against it. | The inspected NDX non-stationarity did not become a portable regime rule. | [SPX replication](reports/research_paths/spx_term_slope_replication.md) |

The most defensible overall conclusion is not that variance is unpredictable.
It is that **historical data describes the current volatility regime much
better than it anticipates the transition into the next one, while options
already price most of the forecastable conditional mean.** That shifts the
economic question from “can I forecast this better?” toward “is there a risk
premium I am willing and able to bear?” The latter is a portfolio decision, not
something a lower forecasting loss establishes on its own.

## Where to start

- [Standing findings](reports/FINDINGS.md) — the current conclusions and the
  evidence hierarchy.
- [Five-path findings](reports/RESEARCH_PATHS_FINDINGS.md) — absorption, horizon,
  VRP term structure, historically gated single names, and the SPX replication.
- [Quarterly top-25 reconstruction](reports/research_paths/quarterly_top25.md) —
  the point-in-time SEC N-PORT ranking and earnings-universe audit.
- [Independent five-path verification](reports/research_paths/verification.md) —
  source hashes, membership/as-of checks, and metric recomputation.
- [Clean-window results](reports/results_clean.md) — the original QQQ forecast
  comparison.
- [Orthogonal-signal verification](reports/signal_study/verification.md) — the
  sealed term-slope holdout and year-by-year stability.
- [Target and regime findings](reports/TARGET_REGIME_FINDINGS.md) — SPX jumps,
  QQQ latent states, and the calibrated incremental repair.
- [Research amendments](reports/AMENDMENTS.md) — what changed after results were
  observed and why none of those changes silently rewrote a baseline.
- [Next research program](reports/NEXT_RESEARCH_PROGRAM.md) — prospective-only
  hypotheses and data worth purchasing.
- [Paused historical-weight work](reports/PAUSED_HISTORICAL_WEIGHTS.md) — exact
  state of the incomplete public-data reconstruction and safe restart rules.

## How to read the verdicts

`FAIL` means a registered criterion was not met; it does not always mean the
mechanism was useless. The term-slope series was non-stationary, the SKEW veto
was contaminated despite improving every risk outcome, and the original HMM
comparison was calibration-confounded. Those qualifications are recorded, but
the frozen verdicts are never rewritten after seeing results. Likewise, a
descriptive improvement is not called a tradable strategy without a genuinely
unseen transition and an implementable portfolio/risk test.

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
[`reports/LEAKAGE_TIREX2.md`](reports/LEAKAGE_TIREX2.md), which does this for
TiRex-2 and explains why its results are reported on the full clean window with
the post-publication subwindow as a robustness check.

## Data sourcing

| Series | Source | Cost | Notes |
|---|---|---|---|
| QQQ daily OHLC (1999→) | yfinance | free | drives the Garman–Klass + overnight target used in the original QQQ headline tables |
| QQQ 5-min bars | Polygon.io aggregates | ~$29/mo plan, or FirstRate Data one-time bundle | preferred future RV estimator; not purchased or used for the headline results |
| VXN daily history | CBOE (`cdn.cboe.com/.../VXN_History.csv`) | free | the market benchmark; if 404, update URL from the CBOE VXN page |
| Macro calendar (FOMC/CPI/NFP) | Fed + BLS schedules | free | `calendars/*.csv`, filled and verified 2026-08-11 for 2016→2026 (FOMC to 2027). See "Calendar provenance" below — the seeds contained errors |
| Earnings dates, top-weight NDX names | yfinance (`make fetch-earnings`) | free, no API key | session inferred from the ET announcement timestamp; FMP path kept as `make fetch-earnings-fmp` |
| Index weights | Invesco QQQ "Complete Holdings" CSV | free | **manual download** — Invesco returns HTTP 406 to scripted requests. Save to `data/raw/qqq_holdings_YYYY-MM-DD.csv`; newest is used |
| Historical QQQ holdings | SEC Form N-PORT (`make fetch-nport-weights`) | free | quarterly public snapshots from 2019-09-30; point-in-time use begins at SEC acceptance, 50–62 days later |
| Quarterly QQQ top 25 | audited SEC N-PORT holdings, aggregated by CUSIP issuer | free | 27 snapshots / 44 issuers; exactly 25 per snapshot and never backfilled with current names |
| SPX term surface | Cboe VIX9D, VIX, VIX3M | free | official 9-, 30-, and 93-calendar-day expected-volatility histories |
| Single-name implied vol | Cboe VXAPL, VXAZN, VXGOG, VXIBM | free | fixed source family; analysis retains only historically eligible top-25 issuer-sessions |
| Orthogonal signal inputs | Yahoo Finance ETFs + Cboe VIX/VIX9D histories | free | separate diagnostic-only study; Cboe closes delayed one session at the 16:00 ET forecast origin |

### Calendar provenance

Filled 2026-08-11 from `bls.gov` (CPI, Employment Situation) and
`federalreserve.gov` (FOMC). Both sites 403/406 scripted fetches, so the dates
were read off the rendered schedule pages and are checked into `calendars/`
with per-file provenance headers rather than re-fetched at runtime.

What the seed files got wrong — all inside the clean window, so these were not
cosmetic:

- `cpi.csv`: **2026-02-11 → 2026-02-13** and **2026-04-14 → 2026-04-10**.
- `nfp.csv`: **2026-02-06 → 2026-02-11**; five in-window releases missing
  (2025-11-20, 2025-12-16, 2026-01-09, 2026-07-02, 2026-08-07).
- `earnings_top.csv` was empty, so `earnings_wt` was identically zero and the
  heavy-earnings slice was empty.
- `fomc.csv` was correct for 2022–2026; extended back to 2016 and out to 2027.

The 2025 shutdown gaps are **real, not missing data**: September 2025 CPI slipped
to 2025-10-24 and no October 2025 CPI was ever published; September 2025 payrolls
slipped to 2025-11-20 and October payrolls were folded into the 2025-12-16
release. Do not "repair" these gaps.

Not included in v1 (extension hooks): 1-DTE implied vol for a short-horizon
market benchmark (no 1-day VXN exists; would need ThetaData/ORATS/CBOE DataShop
options data), and NQ futures 23-hour-session RV (Databento GLBX).

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
  first clean window — see [`reports/FINDINGS.md`](reports/FINDINGS.md).
- **Scoring:** QLIKE on variance (primary; robust to RV proxy noise), CRPS and
  pinball on log-RV quantiles, 90% coverage + Kupiec + Christoffersen, DM vs
  HAR, event-sliced QLIKE, and at 30 calendar days: Mincer–Zarnowitz plus the
  encompassing regression against VXN.
- **Point-variance convention:** quantile models get a truncated-mean estimator
  (trapezoid of exp(q) over the τ grid). It understates the true mean
  identically across quantile models, so their QLIKE ranking is fair; EWMA uses
  its native variance forecast and is flagged accordingly.

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

## Run order

Needs **Python 3.11+** (pandas ≥2.1 and torch). `make setup` creates `.venv`
with `python3.11` and every other target uses it automatically; override with
`make PY=/path/to/python <target>`.

```bash
make setup                      # creates .venv, installs requirements.txt
make smoke                      # synthetic end-to-end check, no data needed
make fetch-free                 # yfinance daily OHLC + CBOE VXN
# calendars/{fomc,cpi,nfp}.csv are checked in and verified through 2026
# (FOMC through 2027) — extend them from the source pages when they run out.
# For earnings, first save the Invesco QQQ "Complete Holdings" CSV to
# data/raw/qqq_holdings_YYYY-MM-DD.csv (the site blocks scripted downloads):
make fetch-earnings             # yfinance; weights read from that holdings file
#    review calendars/earnings_fetched.csv (esp. the session column), then:
make merge-earnings             # -> calendars/earnings_top.csv
make features                   # SOURCE=bars after a polygon fetch, else GK daily
make baselines PHASE=all
make chronos-clean              # ~190 origins x 3 variants; ~2 min on CPU
make evaluate PHASE=clean       # -> reports/results_clean.md
make evaluate PHASE=diagnostic  # baseline sanity on the long window
```

`baselines PHASE=all` writes `*_all.parquet`, and `evaluate` slices those to the
requested phase. Every model refits per origin from history up to that origin, so
slicing an `all` run is identical to having run the phase directly.

Forward accrual: `make daily-update` (cron it after the close if you want, e.g.
`30 18 * * 1-5`). Infrastructure is deliberately one box + parquet + Make — no
cloud, no scheduler, nothing to babysit. State lives in `data/`, results in
`reports/`, and `config.yaml` is the pre-registration artifact.

The five-path extension is separate from the frozen harness and runs its safety
tests before every empirical command:

```bash
.venv/bin/python -m unittest tests.test_research_paths -v
.venv/bin/python -m src.research_paths top25
.venv/bin/python -m src.research_paths fetch-external
.venv/bin/python -m src.research_paths fetch-top25-earnings
.venv/bin/python -m src.research_paths absorption-map
.venv/bin/python -m src.research_paths horizon-curve
.venv/bin/python -m src.research_paths vrp-term-structure
.venv/bin/python -m src.research_paths single-name-earnings
.venv/bin/python -m src.research_paths spx-term-slope
.venv/bin/python -m src.verify_research_paths
```

## Diagnostic-only orthogonal-signal study

The independently frozen protocol is in
[`reports/ORTHOGONAL_SIGNALS_PROTOCOL.md`](reports/ORTHOGONAL_SIGNALS_PROTOCOL.md).
It never reads the original clean window. Safety tests run before both empirical
stages:

```bash
make fetch-signal-inputs
make signals-discover            # tests first; locks at most one candidate
make signals-confirm             # tests first; spends confirmation once
make verify-signals              # independent metric recomputation
```

Seven combinations of term structure, cross-asset stress, and QQQ market state
were evaluated in discovery. Lagged `log(VIX9D/VIX)` won discovery, improved
sealed confirmation QLIKE by 2.92%, but **failed confirmation** (DM p=0.1016).
Its improvement decayed from 5.3%/7.0% in 2022-2023 to 0.2%/0.9% in 2024-2025,
making regime-conditional front-end dislocation a prospective-only hypothesis.
See [`reports/signal_study/verification.md`](reports/signal_study/verification.md)
and the [weight/source audit](reports/HISTORICAL_WEIGHTS_AND_SIGNAL_BACKLOG.md).

Historical QQQ holdings can be refreshed independently with
`make fetch-nport-weights`. The target runs nine parser/integrity/as-of tests
first, then writes 27 SEC filing snapshots plus a quarterly concentration
summary. These are not exact daily Nasdaq-100 weights and are not fed into the
completed signal holdout. Top-10 weight fell 55.13% to 46.85% and HHI fell
0.0459 to 0.0312, with a sharp step after the 2023 anti-concentration rebalance.
That adverse direction, flat earnings win rates, and the 2025 low effectively
close the earnings-concentration defence on existing project data.

An experimental 2004-2018 annual-report backfill plus official Nasdaq
membership parser was started test-first and then paused before a complete
dataset was produced. It remains disconnected from every model. See
[`reports/PAUSED_HISTORICAL_WEIGHTS.md`](reports/PAUSED_HISTORICAL_WEIGHTS.md)
for the exact passing contracts, parser repairs, SEC 403 stop, and safe restart
conditions.

The failed carry rule also has one frozen post-hoc mechanism diagnostic:
`make skew-carry`, documented in
[`reports/SKEW_CARRY_PROTOCOL.md`](reports/SKEW_CARRY_PROTOCOL.md). It tests a
single lagged Cboe SKEW veto and mechanically excludes clean origins. Its strict
verdict is FAIL because it retained 63.3% of eligible trades versus a registered
70% floor, despite rejecting the three known pre-COVID adverse entries and
improving the descriptive tail aggregates. This is not a validated strategy.
The participation gate was a poor proxy for the risk-adjusted outcome; the
decisive limitation is that February 2020 motivated the rule and remains in its
sample.

The resulting ranked research program, verified public cross-asset/surface
coverage, and pre-run safety gate are recorded in
[`reports/NEXT_RESEARCH_PROGRAM.md`](reports/NEXT_RESEARCH_PROGRAM.md).

Two target/model-frame diagnostics are frozen in `target_regime.yaml`:

```bash
make fetch-jump-target       # tests first; commit-pinned Oxford-Man SPX source
make jump-target             # 2014-2017 jump-event comparison
make regime-transition       # forward-filtered two-state QQQ diagnostic
make regime-repair           # Platt calibration + incremental-state holdout
make verify-target-regime    # independent target/timing/metric audits
```

The SPX jump surface comparison failed. It used Oxford-Man five-minute RV and
bipower variation—not the local hourly bars—but still lacks the formal BNS
quarticity statistic. The original HMM comparison was calibration-confounded:
Platt scaling improved both holdout scores. The fair incremental test still
failed; adding the calibrated state to the same-row supervised benchmark made
both losses slightly worse. A correctly matched correlation-premium study is
specified for SPX and remains data-blocked. See
[`reports/TARGET_REGIME_FINDINGS.md`](reports/TARGET_REGIME_FINDINGS.md) and the
[`SPX data contract`](reports/SPX_DISPERSION_DATA_CONTRACT.md).

## Known limitations

- GK-based RV is noisier than 5-min RV; QLIKE is chosen partly because it is
  robust to proxy noise, but buy the intraday history if results get close.
- Earnings weights in the original harness are assigned per announcement from the latest
  point-in-time reconstruction available before that announcement; a future
  weight snapshot cannot change an earlier feature. The broader reconstruction
  is still approximate: the tracked 13-name basket uses current membership and
  is normalized to a fixed 55.43% aggregate weight, so it captures relative
  drift rather than exact historical Nasdaq-100 membership and weights.
- The five-path extension does not reuse that membership approximation: its
  earnings studies start with the first SEC-accepted 2019 quarterly top-25
  snapshot. That removes current-name selection but leaves a delayed quarterly
  QQQ-fund proxy rather than exact daily NDX weights.
- `predict_df` gets synthetic contiguous timestamps per origin (trading days
  are irregular); mapping back to real dates is positional and exact.
- 30-day distributional eval for Chronos is out of scope (quantiles don't sum
  across steps); the cumulative point forecast sums per-step means, which is
  exact for the mean regardless of dependence.
- Chronos's 30-calendar-day cumulative forecast needs 21 future covariate rows,
  which only exist for dates already in the master frame. The newest ~20 origins
  therefore get `log_cum_var_hat = NaN` (h=1 is unaffected). They fill in as data
  accrues; their 30-day targets aren't realized yet anyway.
- Earnings **sessions are inferred** from yfinance announcement timestamps, not
  verified against IR pages. See `reports/AMENDMENTS.md`.
- Calendars are checked in, not fetched at runtime: bls.gov and invesco.com both
  reject scripted requests. They currently run through 2026 (FOMC through 2027)
  and must be extended by hand after that, or `is_cpi`/`is_nfp` silently become 0.
