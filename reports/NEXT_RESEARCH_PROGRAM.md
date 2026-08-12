# Research program after the orthogonal-signal study

Status as of 2026-08-12. Historical-weight collection is paused. This note
separates what the completed work now supports from hypotheses that still need
new data or genuinely untouched observations.

## Decision from the SKEW diagnostic

The surface-shape idea produced a useful mechanism result, but not a strategy.
The one frozen SKEW veto removed the three known pre-COVID adverse entries and
materially improved average tail statistics, while retaining only 63.3% of the
richness trades against a registered 70% floor. The strict verdict is therefore
**FAIL** and the threshold must not be tuned on this window.

The informative result is the overlap: the backward-looking richness rule
traded on 61.8% of high-SKEW origins versus 45.1% of lower-SKEW origins. In
other words, it disproportionately called ATM variance expensive while the
wings were already charging for tail risk. Lagged SKEW had only -0.270 Pearson
correlation with lagged VXN, so this is not merely another transform of the ATM
level.

This makes surface shape the first idea in the project with a clear mechanism
tied to an observed failure. It still requires a new transition to validate.
The existing clean period has already been inspected and cannot be reused.

## What is cheaply available now

A read-only coverage check of Cboe's public daily files found the following
histories through 2026-08-11:

| series | first observation | observations | role |
|---|---:|---:|---|
| VIX | 1990-01-02 | 9,248 | SPX 30-day ATM level |
| VXN | 2009-09-14 | 4,255 | NDX 30-day ATM level |
| RVX | 2009-09-16 | 4,246 | RUT 30-day ATM level |
| SKEW | 1990-01-02 | 9,203 | SPX surface shape / tail proxy |
| VVIX | 2006-03-06 | 5,080 | VIX vol-of-vol |
| VIX9D | 2011-01-04 | 3,923 | SPX short-end term point |
| VIX3M | 2009-09-18 | 4,249 | SPX three-month term point |
| VIX6M | 2008-01-02 | 4,681 | SPX six-month term point |
| VIX1Y | 2007-01-03 | 4,926 | SPX one-year term point |
| VXAPL, VXAZN, VXGOG, VXIBM | 2011-01-07 | 3,916 each | single-name 30-day levels |

This means SPX term slope, VVIX, SPX/RUT replication, and a small single-name
replication do not require purchased option chains. Public index histories do
not provide NDX skew or NDX term points, so SPX surface variables would remain
regime proxies rather than exact NDX measurements.

Sources: [Cboe daily index data](https://www.cboe.com/us/indices/market_statistics/historical_data/),
[selected volatility-index methodology](https://cdn.cboe.com/api/global/us_indices/governance/Volatility_Index_Methodology_Selected_Broad_Based_Index_Equity_and_ETF_Volatility_Indices.pdf),
and [SKEW methodology](https://cdn.cboe.com/resources/indices/documents/SKEWwhitepaperjan2011.pdf).

## Ranked next studies

### 1. Cross-sectional replication with the existing harness

This is the cheapest next executable study and the strongest defense against a
one-index story. Run the unchanged target, timing, baselines, phase sampling,
and metrics on SPX/VIX and RUT/RVX first. Add the four public single-name
volatility indices only under one pre-registered replication family; do not
select the names after seeing results.

The assets share market shocks, so five nominal tests are not five independent
replications. Report per-asset estimates plus a date-blocked pooled estimate,
not a count of significant assets. The concentration prediction should be
directional and frozen in advance: event effects should be larger for the
single names, then NDX, then broader SPX/RUT indices.

### 2. One future surface-state protocol, observed without tuning

Preserve the failed 80th-percentile SKEW rule exactly as a forward monitor. A
new confirmation must freeze the historical richness cutoff at +0.386812814
and use either one-session-lagged Cboe closes or timestamped pre-close quotes.
Do not silently refit either cutoff.

VVIX and term slope are plausible separate hypotheses, but they are not
fallbacks for the failed participation gate. If pursued, register only one next
surface hypothesis on an untouched asset or future period. A defensible first
choice is a scale-free SPX slope such as `log(VIX3M) - log(VIX)`, using the same
one-session publication lag. Any SKEW/slope/VVIX combination must be specified
as one fixed composite before its validation data are opened; enumerating all
permutations on the spent NDX window would only manufacture a winner.

### 3. Change the target when point-in-time constituents are ready

Implied-versus-realized correlation is genuinely different from forecasting
total variance. Treat COR1M as the implied forecast and construct realized
correlation from point-in-time constituent membership, weights, and returns.
The paused holdings work is therefore a prerequisite, not an optional feature.
Do not substitute current constituents into history.

A jump-versus-continuous target is also well motivated, but it needs intraday
data. Pre-register the sampling interval, overnight treatment, bipower-variation
estimator, jump threshold, and market-hours calendar before acquiring the data.

### 4. Spend a data budget on intraday shape, not more daily macros

If paid data are purchased, 5-minute QQQ/NQ bars offer more information per
dollar than another daily macro feed. The first fixed targets should stay close
to the original question: realized range, fraction of variance in the first and
last hour, deviation from the ordinary intraday U-shape, and close location
within the day's range. Event-day deformation can then be tested without
pretending daily total variance is the same object.

Option-chain archiving is valuable but only prospectively. A useful archive
needs timestamped quotes, strikes, expiries, volume, open interest, and contract
adjustments. Open interest alone cannot identify whether dealers are long or
short gamma, and a cron job cannot recreate the missing past.

### 5. Treat regime switching as a frame, not new information

A Markov model fitted only to RV and VXN may simply repackage the ATM level. It
becomes useful when the state includes a registered surface-shape input and the
target is transition risk. Only filtered probabilities available at the origin
may enter forecasts; smoothed full-sample state probabilities are lookahead.
Estimate all state and transition parameters inside each training fold and
score next-period tail-event probability with log loss or Brier score alongside
the existing variance metrics.

### 6. Keep CFTC positioning as a low-frequency secondary study

CFTC Traders in Financial Futures data are free and plausibly orthogonal, but
the report is a Tuesday position snapshot normally released Friday at 15:30 ET.
Use the consolidated Nasdaq-100 leveraged-fund net position normalized by open
interest as one frozen feature. Availability must follow the documented release
date, not the Tuesday report date. Historical interpretation must account for
the consolidated Nasdaq-100 series beginning in 2010 and Micro E-mini Nasdaq-100
being added in 2023.

Sources: [CFTC COT release schedule](https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm),
[COT explanatory notes](https://www.cftc.gov/MarketReports/CommitmentsofTraders/AbouttheCOTReports/index.htm),
and [historical special announcements](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm).

## Required test gate before any next run

Every new study must begin with failing tests for:

- the exact as-of timestamp and publication lag of every input;
- no backward fill and explicit flat/missing behavior;
- train-fold-only fitting of transforms, thresholds, and model parameters;
- online/filtered rather than smoothed regime probabilities;
- immutable discovery, confirmation, and clean-window fences;
- unchanged baselines and phase construction across assets;
- source coverage, schema, historical anchors, and content hashes; and
- one frozen success criterion with the full hypothesis family counted.

Acquisition or scoring starts only after that gate passes. A parser repair adds
a regression and restarts the gate. A failed criterion remains a failed result;
no threshold or expected baseline is moved to accommodate it.
