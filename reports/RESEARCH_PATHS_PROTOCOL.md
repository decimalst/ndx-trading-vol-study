# Frozen protocol: five paths beyond the implied-volatility level

Written and frozen 2026-08-12 before any result-producing run below. The
machine-readable contract is `research_paths.yaml`; tests enforce the timing
rules. This extension does not alter `config.yaml`, does not read a single NDX
clean origin, and does not convert a diagnostic result into confirmation.

## Why these five paths belong together

The existing project asked whether new features could beat a 30-day implied
level. The extension asks what that scalar cannot represent: which effects it
only partly absorbs, how its contribution changes with horizon, how the
variance premium changes across the surface, whether index-level aggregation
masked a single-name earnings mechanism, and whether short-end slope works only
in an inverted front-end regime.

No study selects a winner from a feature menu. All registered rows are reported.

## Historical universe: quarterly QQQ top 25

The new earnings paths do not inherit the current 13-name basket. For every
public N-PORT snapshot from 2019 onward, positive equity share classes are
aggregated to the CUSIP issuer identifier, ranked by disclosed portfolio weight,
and cut at 25 issuers. A snapshot becomes usable only at its SEC `accepted_at`
timestamp; before the first accepted snapshot, the universe is missing. Between
filings, the latest accepted ranking is carried as an explicit stale snapshot,
not reconstructed from today's membership. This prevents the recent NVDA/MU
concentration regime from being projected backward.

Quarterly QQQ holdings remain a delayed fund proxy, not exact daily NDX weights.
The paused pre-2019 annual backfill remains paused and is never silently replaced
with current names.

## 1. NDX absorption map

On 2016-01-04 through 2025-10-17, regress next-session log GK-plus-overnight
variance on HAR and one frozen group. Refit the same regression with same-origin
log(VXN), using the common complete sample, and report

`1 - Wald(group | HAR + VXN) / Wald(group | HAR)`.

Groups are leverage, target weekday, point-in-time top-25 target earnings
weight, scheduled macro
events, and the session after FOMC. HAC uses ten lags, matching the already
published leverage calculation. This is a measurement of
Wald attenuation. It is not a literal information decomposition; it may be
negative, and no ranking or threshold is applied. The known leverage value is
an internal anchor, not a selection criterion.

Implementation audit: the first dry run used a guessed five-lag HAC and stopped
when it failed to reconstruct the existing leverage ledger. Before interpreting
any row, the code was repaired to use the ledger's ten-lag convention and to
allow the 2025-10-17 origin's next-session target to complete on 2025-10-20,
still before the 2025-11-03 clean start. That exactly recovers 103.1875 →
62.6305 at n=2463. The provisional output is retained as
`absorption_map_metrics_pre_anchor_repair.json`; it is not evidence.

## 2. NDX horizon curve

For h in {1, 5, 10, 21, 42, 63} trading sessions, directly forecast the log of
mean future variance with expanding HAR and HAR+log(VXN). A training row is
eligible only once all h target sessions have occurred. Both models use exactly
the same origins and Duan smearing. The primary curve is paired OOS QLIKE
improvement; secondary curves are incremental adjusted R-squared and partial
R-squared. Overlap inference uses h-1 lags. There is no best-horizon test.

## 3. SPX VRP term structure

Match official Cboe VIX9D, VIX, and VIX3M closes to 9-, 30-, and 93-calendar-day
forward SPX close-to-close variance. Cboe's current methodology specifies those
constant maturities. Report the common-origin mean premium in volatility points
and the same curve when the trailing five-session SPX return is negative versus
nonnegative. Fixed 21-session moving-block intervals describe sampling
uncertainty. This is not an option return series and cannot establish a
tradable premium net of skew, jumps, hedging, spreads, or margin.

## 4. Single-name earnings mechanism

Use the fixed free matched Cboe family AAPL/VXAPL, AMZN/VXAZN, GOOG/VXGOG, and
IBM/VXIBM, but retain an issuer-session only when it belongs to the latest
SEC-accepted quarterly QQQ top 25. For each eligible asset, produce an expanding
HAR plus its own 30-day implied-vol forecast without any earnings input. Only
after those forecasts are fixed, label BMO announcement sessions and the next
session after AMC/unknown announcements. Measure the event versus non-event log
forecast residual for every eligible name and an equal-name pool. IBM is
reported as zero coverage if it never passes the historical membership gate.

This fence matters: `earnings_top.csv` records realized announcement timestamps,
not a versioned history of what date was scheduled at each old forecast origin.
Using it inside HAR-IV-X would manufacture point-in-time certainty. The primary
study therefore measures whether an earnings mechanism survives own implied
volatility; it does not claim an executable forecast improvement.

## 5. SPX regime-conditional term slope

The NDX year-by-year pattern was first inspected on 2016-2025. The external
historical holdout is SPX in 2014-2015, with VIX9D data from 2011 supplying the
training history. Compare the same expanding HAR-IV-leverage baseline with two
fixed additions: full log(VIX9D/VIX), and its positive part, which is nonzero
only when VIX9D exceeds VIX. The dislocation-only comparison is primary; the
unconditional slope is a registered reference, not an alternate winner.

To reproduce the information set that generated the hypothesis, this path uses
a 16:00 ET origin and delays both Cboe closes by one complete SPX session. This
clarification was made before the first SPX score was run.

Success requires lower dislocation-only QLIKE than both baseline and the
unconditional form, two-sided DM p below 0.05 versus baseline, and more than
half of paired origins won. A failure is reported as-is.

## Source and timing contract

- Cboe index values are used at a post-16:15 ET decision time and forecast only
  subsequent underlying sessions.
- Exact-date joins only; no forward fill across missing market sessions.
- Yahoo Finance supplies underlying OHLC. Raw downloads, URLs, fetch timestamp,
  date coverage, row counts, and SHA-256 hashes are recorded in a manifest.
- All score windows end before the NDX clean phase. The SPX confirmation score
  window also ends before the prior NDX term-slope discovery begins.
- Tests run before acquisition and again before every empirical command.
