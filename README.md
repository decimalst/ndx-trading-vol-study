# NDX Realized-Variance Forecasting Experiment

Tests whether Chronos-2 (zero-shot, arXiv 2510.15821), fed known-future calendar
covariates, forecasts Nasdaq-100 realized variance better than classical
baselines — and whether any of it adds information beyond what the options
market already prices. Research code; nothing here is trading or financial advice.

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
  count. Accrues forward daily; re-evaluate monthly.

The date rule is a fallback for models whose training corpus is undisclosed. For
a model that publishes an enumerable corpus, check the corpus instead — see
[`reports/LEAKAGE_TIREX2.md`](reports/LEAKAGE_TIREX2.md), which does this for
TiRex-2 and explains why its results are reported on the full clean window with
the post-publication subwindow as a robustness check.

## Data sourcing

| Series | Source | Cost | Notes |
|---|---|---|---|
| QQQ daily OHLC (1999→) | yfinance | free | drives Garman–Klass RV fallback + overnight gaps |
| QQQ 5-min bars | Polygon.io aggregates | ~$29/mo plan, or FirstRate Data one-time bundle | preferred RV estimator; check plan history depth |
| VXN daily history | CBOE (`cdn.cboe.com/.../VXN_History.csv`) | free | the market benchmark; if 404, update URL from the CBOE VXN page |
| Macro calendar (FOMC/CPI/NFP) | Fed + BLS schedules | free | `calendars/*.csv`, filled and verified 2026-08-11 for 2016→2026 (FOMC to 2027). See "Calendar provenance" below — the seeds contained errors |
| Earnings dates, top-weight NDX names | yfinance (`make fetch-earnings`) | free, no API key | session inferred from the ET announcement timestamp; FMP path kept as `make fetch-earnings-fmp` |
| Index weights | Invesco QQQ "Complete Holdings" CSV | free | **manual download** — Invesco returns HTTP 406 to scripted requests. Save to `data/raw/qqq_holdings_YYYY-MM-DD.csv`; newest is used |
| Historical QQQ holdings | SEC Form N-PORT (`make fetch-nport-weights`) | free | quarterly public snapshots from 2019-09-30; point-in-time use begins at SEC acceptance, 50–62 days later |
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

- **Target:** `log_rv` where `rv_total` = intraday RV (5-min sum of squared log
  returns, RTH) + squared overnight gap. Garman–Klass + overnight as the free
  fallback. Total variance is used so the 30-day comparison lines up with
  VXN's calendar-time quote: expected 30-cal-day variance = (VXN/100)² × 30/365.
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
earnings weight, 8 in the heavy-earnings slice.** The full-window DM test has
reasonable power for large differences only. Event-sliced comparisons on n=6–8
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
- Earnings-weight covariate uses approximately-current weights; drift over the
  diagnostic window is real (documented, accepted).
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
- The reconstructed 13-name weight file holds the tracked basket at a constant
  55.43% and uses current membership. It is approximate relative drift, not
  exact historical NDX weights; see the weight/source audit above.
- Calendars are checked in, not fetched at runtime: bls.gov and invesco.com both
  reject scripted requests. They currently run through 2026 (FOMC through 2027)
  and must be extended by hand after that, or `is_cpi`/`is_nfp` silently become 0.
