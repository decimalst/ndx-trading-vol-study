# Index-return opportunities: unscored design audit

2026-09-06. This audit inspected schemas, bounded coverage, data-building code,
and existing research. It fitted no model, constructed no new performance
statistic, and examined no market observations after 2025-10-20. It is a design
recommendation, not a frozen experimental protocol or evidence of a signal.

The most useful immediate experiment is QQQ **next-session open-to-close
return**, using the separate signed overnight and daytime history. The earlier
studies mainly asked whether unsigned stress measures improve volatility.
Cancellation between overnight and daytime mechanisms could hide information
in a close-to-close target. This is a new target and use of information already
present in the data, not a new external data source.

## Data that can support the experiment now

All counts below apply only to dates at or before 2025-10-20.

| Local input | Usable coverage observed | Provenance and limitation |
|---|---|---|
| `data/raw/daily_ohlc.parquet` | 6,696 QQQ sessions, 1999-03-10–2025-10-20; complete open, high, low, close, adjusted close, volume | Yahoo via `src/fetch.py::fetch_daily_ohlc`, `auto_adjust=False`; latest-vintage history, not an archived point-in-time feed |
| `data/research_paths/spx_daily.parquet` | 4,226 SPX sessions, 2009-01-02–2025-10-20; complete OHLC | Yahoo `^GSPC`, source manifest records `auto_adjust=False`; index levels are not tradable fills and exclude reinvested dividends |
| `data/raw/cross_asset_daily.parquet` | HYG: 4,663 sessions from 2007-04-11; TLT: 5,845 from 2002-07-30; GLD: 5,263 from 2004-11-18; USO: 4,914 from 2006-04-10; UUP: 4,691 from 2007-03-01; all end 2025-10-20 | Yahoo adjusted-close preference in `fetch_signal_inputs`; the builder has a raw-close fallback, so its present values are not accompanied by a per-column adjustment provenance assertion |
| `data/raw/hourly_bars.parquet` | 3,669 bars, 2023-09-13 09:30 ET–2025-10-17 15:30 ET | Too short for an independent long historical split; hourly bars cannot reconstruct the literature's first-half-hour return |
| `calendars/fomc.csv` | Scheduled decisions plus two explicitly listed emergency 2020 events | Emergency dates 2020-03-03 and 2020-03-16 must not be advertised as known at the preceding close; the header documents this distinction |

Existing experiment search covered `reports/FINDINGS.md`,
`reports/HISTORICAL_WEIGHTS_AND_SIGNAL_BACKLOG.md`,
`reports/orthogonal_round2/METHODOLOGY_REVIEW.md`, and `docs/OPEN_FINDINGS.md`.
The existing cross-asset and market-state nulls concern volatility, not the
signed-return hypotheses below. Previously inspected history remains reused
history even after changing the target.

## A small fixed first family

Use a forecast origin after session `t` closes and execute at the opening
auction of session `t+1`, with a planned closing-auction exit that day. The
primary target is `close[t+1] / open[t+1] - 1`. Nothing from `t+1`, including
its opening gap, enters a decision assumed to be submitted before that open.
Prices at the auctions are execution proxies, so costs and slippage remain
necessary. The target needs no dividend reinvestment assumption because the
position is flat overnight.

Use three candidate information groups, each added alone to one common
baseline. No combination search or model-class search is needed to ask whether
these information channels matter.

| Group | Fixed information | Matched control and interpretation |
|---|---|---|
| Session decomposition | Signed overnight-minus-daytime log return, averaged over 1, 5, and 22 completed sessions | Baseline already includes total adjusted close-to-close returns at the same three scales. This tests decomposition conditional on the total, avoiding mechanically redundant inclusion of both components and their sum. |
| Signed cross-asset transmission | Five separate one-session adjusted log returns of HYG, TLT, GLD, USO, UUP, through `t` | Include own-market returns and volatility in the baseline. The earlier root-mean-square stress composite discarded direction; this proposal retains it. This is a hypothesis, not evidence that these ETF returns predict QQQ. |
| Calendar phase | An indicator that the next session is one of the first three sessions observed in its calendar month, plus its weekday | Use the same lagged-market baseline. Do not use an ex-post "last session of month" label that depends on a future unscheduled closure. The first-three-session indicator can be calculated from prior dates and the known date of the next opening, without observing any future price. |

Recommended market baseline: intercept; total adjusted log-return averages over
1, 5, and 22 sessions; trailing realized-variance level; previous-session VXN.
Use identical fixed ridge regularization, standardization fitted on training
rows, monthly refits, and at least 1,000 completed training examples across
baseline and augmentations. Pin the exact penalty and variance transform before
running. Also score the zero forecast and training-only historical mean;
beating a poorly fitted market model alone is insufficient.

For decomposition, define adjusted overnight log return algebraically as total
adjusted close-to-close log return minus raw open-to-close log return. Their sum
then exactly recovers the total. This is an adjusted-history predictor, not a
claim that a cash dividend was received at the opening auction. Add a synthetic
dividend/split test and report the vendor-vintage limitation. Never combine an
adjusted close with a raw open directly.

Use the common feature-complete calendar for every arm, never per-model
available-case samples. A missing session or price is not a zero return.

## Evaluation and execution controls

An orderly historical split is training through 2015; discovery 2016–2019;
separate evaluation 2020–2025-10-20. Parameters may expand using only labels
completed before each refit. Freeze all three candidate groups and their
hyperparameters before any score; report all of them on both periods without
selecting a discovery winner or dropping weak candidates. These periods are
chronologically disjoint **historical discovery and evaluation**, not pristine
confirmation: the repository has already inspected these market years for
other purposes. The clean window beginning 2025-11-03 stays sealed.

Primary statistical criterion: paired squared-error improvement against both
the market baseline and training historical mean. Correct the six group/control
comparisons as one family; use the maximum p-value from dependence-aware block
lengths 21, 63, 126 and HAC(126). Declare a practical improvement threshold and
stability periods before scoring. Do not call a high direction-hit rate a
forecast improvement without accounting for the unconditional positive rate.

A separate executable screen can use a fixed long/flat position rule, taking
the position only when predicted return exceeds the prespecified round-trip
cost. Charge costs on both entry and exit even when successive days have the
same signal, since positions close each afternoon. Fix 2 basis points per side
as a central cost scenario and 5 basis points per side as stress assumptions;
these are sensitivity assumptions, not claims about measured historical spreads.
Show always-long daytime exposure and the historical-mean decision as controls.
Report turnover, exposure, net return, drawdown, and mean-variance utility with
a fixed risk-aversion coefficient. Lower drawdown alone can reflect lower
exposure and does not establish timing skill. Short-selling is unnecessary for
the first screen and would require borrow-cost assumptions absent from this
data. A price-only SPX replication can test directional generality, but its
result must not be presented as executable SPY performance.

Required tests before scores: future-perturbation invariance; exact origin and
target dates; training-label maturity; decomposition identity through a dividend
and split; zero and missing-price rejection; training-only scaling; common-row
parity; calendar construction that never inspects future prices; and a two-leg
cost check. An independent implementation should reconstruct every feature,
forecast, and trade from bounded raw data.

## Literature and the next information upgrade

The decomposition hypothesis is motivated by [Lou, Polk, and Skouras (2019)](https://personal.lse.ac.uk/polk/research/TugOfWar.pdf),
which documents firm-level persistence within overnight/daytime components and
opposing cross-period effects. Its cross-sectional strategy evidence does not
establish the same mechanism for one aggregate ETF; the proposed test is an
adaptation. [McConnell and Xu (2008)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1135217)
motivate a calendar-phase hypothesis. Their turn-of-month result concerns a
different window and does not validate the first-three-session daytime version.

[Gao, Han, Li, and Zhou (2018)](https://profiles.wustl.edu/en/publications/market-intraday-momentum/)
provide a more direct index-return opportunity: previous close to the first
half-hour endpoint predicting the final half hour. The current hourly history
cannot reproduce the predictor. A provenance-verified, longer 5- or 30-minute
QQQ/SPY history would make this a genuinely different information experiment.
Do not silently substitute first-hour returns and call that a replication.

[Lucca and Moench's pre-FOMC study](https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr512.pdf)
motivates announcement-time segmentation, but its 24-hour pre-announcement
window also needs intraday prices. A [later direct replication](https://doi.org/10.1016/j.frl.2020.101781)
reports decay after 2015. A broad all-day FOMC dummy is not an assured edge, and
the repository's emergency-event timing must first be corrected in an additive
calendar source for any new experiment.

[Bollerslev, Tauchen, and Zhou (2009)](https://public.econ.duke.edu/~get/wpapers/btz.pdf)
offer a slower-horizon index-return route through variance risk premia, with
stronger evidence at a quarterly horizon and dependence on accurate high
frequency realized variance. The existing GK-plus-overnight proxy does not
satisfy that measurement standard. It can support an explicitly named implied
minus proxy-realized variance screen, but not a faithful replication or a claim
to have measured an option variance risk premium.

The strongest external-data upgrade is consequently reliable intraday history,
followed by historical auction imbalance messages where accessible with proven
timestamps. [NYSE's own imbalance documentation](https://www.nyse.com/data-insights/nyse-introduces-closing-auction-imbalance-analysis-tool)
describes actual order imbalances and indicative clearing prices. Daily close
location is not a substitute for those measurements. Nasdaq-listed QQQ also
requires the correct primary-market auction feed; NYSE instrument examples do
not prove QQQ auction coverage.

Finally, [Moreira and Muir (2017)](https://www.nber.org/papers/w22208)
motivate evaluating the economic use of an already competent volatility model
through risk allocation. That is a separate experiment with risk-matched
controls, turnover, financing, and cash returns. It may uncover practical use
without discovering a new return signal, so the two conclusions should remain
distinct.
