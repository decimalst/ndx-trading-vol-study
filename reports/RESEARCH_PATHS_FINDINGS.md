# Findings from all five proposed paths

Completed 2026-08-12 under the frozen protocol in
[`RESEARCH_PATHS_PROTOCOL.md`](RESEARCH_PATHS_PROTOCOL.md). The independent
recomputation in [`research_paths/verification.md`](research_paths/verification.md)
passed every source-hash, timing, sample, metric, and verdict check.

These studies do not reopen the NDX clean window. The NDX work is explicitly
diagnostic; the SPX term-slope score uses 2014-2015, before the inspected NDX
window begins. The single-name event labels are attached only after forecasts
are fixed because the historical earnings archive is not versioned as-of each
old forecast origin.

## Executive result

The five paths produce two genuinely useful measurements and one clean
rejection:

1. Same-origin VXN attenuates, but does not eliminate, known regularities. It
   absorbs about 39% of the leverage joint-Wald statistic and about 33% of the
   point-in-time top-25 earnings statistic. It absorbs very little weekday or
   scheduled-macro structure and amplifies the post-FOMC contrast.
2. Index earnings were the wrong place to look for the mechanism. For AAPL,
   AMZN, and Alphabet while each issuer was actually in the SEC-accepted QQQ
   top 25, earnings-session variance remains far above an expanding HAR plus the
   name's own 30-day implied level. The equal-asset event/non-event forecast-
   residual contrast is 2.049 log-variance points: a 7.76× geometric ratio of
   actual/forecast variance on event versus non-event sessions (95% block
   interval 1.829 to 2.272 in logs).
3. The proposed regime-conditional VIX9D/VIX repair does not replicate. On the
   untouched SPX 2014-2015 window it worsens QLIKE by 0.60%, loses on a majority
   of origins, and is worse even inside front-end inversions.

The positive single-name result does not imply a free earnings trade. A 30-day
constant-maturity index spreads one event's variance across a month; this study
asks whether the next session is still exceptional, not whether short-dated
options underprice it after bid/ask, skew, jumps, and the exact announced date.

## 1. Absorption map

| regularity | common n | Wald without VXN | Wald with VXN | attenuation |
|---|---:|---:|---:|---:|
| Leverage | 2,463 | 103.19 | 62.63 | 39.3% |
| Point-in-time top-25 earnings weight | 1,479 | 39.68 | 26.42 | 33.4% |
| Weekday | 2,463 | 12.55 | 11.39 | 9.2% |
| Scheduled FOMC/CPI/NFP | 2,463 | 39.83 | 37.80 | 5.1% |
| Post-FOMC | 2,463 | 14.99 | 17.76 | −18.4% |

The leverage row exactly reconstructs the existing ledger after the
implementation audit restored its ten-lag HAC convention and n=2,463 target
completion. The pre-repair output is retained and explicitly non-evidentiary.
Details: [`research_paths/absorption_map.md`](research_paths/absorption_map.md).

## 2. Horizon curve

| horizon (sessions) | n | HAR+VXN QLIKE gain | DM p | partial R² |
|---:|---:|---:|---:|---:|
| 1 | 2,462 | 8.14% | 2.35e-7 | 9.03% |
| 5 | 2,458 | 11.45% | 0.00088 | 15.12% |
| 10 | 2,453 | 9.67% | 0.0216 | 14.93% |
| 21 | 2,442 | 6.91% | 0.0799 | 12.73% |
| 42 | 2,421 | 2.64% | 0.332 | 8.28% |
| 63 | 2,400 | 3.64% | 0.303 | 7.74% |

The contribution peaks at five sessions, not near the quote's roughly 21-session
trading horizon, then decays. That is evidence about the relative value of VXN
against HAR, not a literal maturity mapping: VXN is a calendar-time risk-neutral
expectation while the target is mean physical daily GK-plus-overnight variance.
Details and chart: [`research_paths/horizon_curve.md`](research_paths/horizon_curve.md).

## 3. SPX VRP term structure

All three points use the same 3,656 origins from 2011-01-04 through 2025-07-18.

| horizon | implied vol | realized vol | mean premium | 95% block interval |
|---:|---:|---:|---:|---:|
| 9 calendar days | 17.58 | 13.98 | 3.61 | [3.05, 4.10] |
| 30 calendar days | 18.18 | 14.71 | 3.47 | [2.50, 4.29] |
| 93 calendar days | 20.04 | 15.32 | 4.72 | [3.53, 5.79] |

The 93-day premium is 1.02 vol points richer after a negative trailing five-day
SPX return than otherwise, with a block interval of [0.11, 2.10]. The 9- and
30-day state differences include zero. These are implied-minus-realized index
levels, not option strategy returns. Details:
[`research_paths/vrp_term_structure.md`](research_paths/vrp_term_structure.md).

## 4. Single-name earnings with historical membership

| asset / own IV | eligible origins | events | event − non-event log residual | geometric variance-ratio effect |
|---|---:|---:|---:|---:|
| AAPL / VXAPL | 1,478 | 23 | 1.715 | 5.56× |
| AMZN / VXAZN | 1,478 | 23 | 2.139 | 8.49× |
| GOOG / VXGOG | 1,478 | 23 | 2.295 | 9.92× |
| IBM / VXIBM | 0 | 0 | n/a | n/a |

IBM was in the fixed free-IV source family but never in the available
SEC-accepted QQQ top 25, so it correctly contributes no rows. No asset was
selected on its result. Details:
[`research_paths/single_name_earnings.md`](research_paths/single_name_earnings.md).

## 5. SPX regime-conditional term slope

Registered verdict: **FAIL**.

| model | n | QLIKE | change vs baseline | DM p | win rate |
|---|---:|---:|---:|---:|---:|
| Baseline HAR-IV-leverage | 503 | 0.2950 | — | — | — |
| Unconditional log(VIX9D/VIX) | 503 | 0.2955 | −0.16% | 0.863 | 54.1% |
| Positive slope only (VIX9D > VIX) | 503 | 0.2968 | −0.60% | 0.0325 | 48.1% |

The positive-slope model worsened QLIKE by 1.66% on 141 inverted origins and by
0.17% on the other 362. The significant DM statistic points against the
candidate, not for it. Details:
[`research_paths/spx_term_slope_replication.md`](research_paths/spx_term_slope_replication.md).

## What changed in the data

The new earnings work uses 27 quarterly QQQ N-PORT snapshots, 675 top-25 issuer
rows, and 44 distinct issuers. Positive equity share classes are combined by
CUSIP issuer before ranking. Each snapshot becomes usable only at its SEC
acceptance timestamp; before the first accepted 2019 snapshot, the universe is
missing rather than filled with today's names. All 43 active mapped symbols have
realized earnings-date coverage, producing 578 eligible issuer-events across 39
issuers.

This directly addresses the current-name problem. NVIDIA moves from rank 17 at
1.27% in the first disclosed snapshot to rank 1 at 8.68% in 2026; that rise is
not projected backward. Micron appears only in quarters where its disclosed
issuer weight actually reaches the top 25. See
[`research_paths/quarterly_top25.md`](research_paths/quarterly_top25.md).

## Remaining boundaries

- N-PORT starts in 2019 and arrives with a filing lag. It is a quarterly QQQ
  fund-weight proxy, not licensed daily NDX weights.
- Historical earnings timestamps are suitable for ex-post mechanism labels but
  not for an executable scheduled-date forecast without versioned calendars.
- Single-name realized variance still uses adjusted daily Garman-Klass plus the
  overnight gap; intraday data would sharpen event-day measurement.
- The horizon and absorption findings are diagnostic on inspected NDX history.
- The VRP curve is not a delta-hedged or investable option return.
