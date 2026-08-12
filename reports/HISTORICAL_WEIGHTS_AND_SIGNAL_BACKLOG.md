# Historical weights and next-signal backlog

Audited 2026-08-12. This is a source-and-timing audit, not another model run.
No clean-window origin was read and no additional confirmation test was spent.

## Historical Nasdaq-100 weights

### What the repository currently has

`data/raw/pit_weights.parquet` is a market-cap reconstruction for the 13 names
in the 2026 Invesco snapshot. It is not exact historical Nasdaq-100 membership
or weight history:

- 6,439 indexed sessions are present, but the first non-empty date is
  2015-10-23; the file is empty before then.
- The tracked-basket total is exactly 55.43% every valid day (standard deviation
  1.6e-14) because the builder deliberately renormalizes to a fixed snapshot
  total. It cannot measure changes in the basket's aggregate index weight.
- The basket has survivorship bias: it uses today's 13 names over all history.
  The `META` column is empty until 2022-06-09 because the old `FB` identifier is
  not mapped, and historical additions/removals are absent.
- A within-basket HHI can be calculated, but it is not an index HHI. For
  example, it moves from 0.1371 on 2016-01-04 to 0.1045 on 2025-10-17 while the
  fixed aggregate remains 55.43%. Missing membership, missing `FB`, and the
  fixed total make that unsuitable as a validated NDX-concentration feature.

The file remains useful for approximate relative drift among surviving names,
especially for earnings weights. It must not be described as exact historical
index weights or used to claim a concentration effect.

### Reliable acquisition paths

1. **Exact daily history:** Nasdaq's licensed [Global Index Watch / Index Data
   service](https://www.nasdaq.com/solutions/global-indexes/data) provides
   historical compositions and weights. This is the only identified route that
   answers the exact daily-membership question for the whole 2016+ study.
2. **Free, delayed proxy (now downloaded):** public QQQ Form N-PORT filings
   reconstruct quarter-end fund holdings from late 2019 onward. The SEC
   publishes bulk
   [Form N-PORT datasets](https://www.sec.gov/data-research/sec-markets-data/form-n-port-data-sets)
   and documents unauthenticated [EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces).
   A snapshot may enter a point-in-time feature only after its EDGAR acceptance
   timestamp, never on its portfolio report date. The public series is
   quarterly and delayed, so it is a slow regime proxy, not a daily weight
   signal; it also cannot cover 2016 through most of 2019.

The supplied EDGAR page resolved the archive route. After nine pre-written
parser, integrity, and timing tests passed, `make fetch-nport-weights` downloaded
all 27 public QQQ snapshots: 2,746 holding rows covering report dates 2019-09-30
through 2026-03-31. Each contains 101–103 investments, sums to 99.503%–100.092%
of portfolio value, and was accepted 50–62 calendar days after its report date.
The raw normalized history is in `data/raw/qqq_nport_holdings.parquet`; the
filing index and tested concentration summary are adjacent. See
`reports/qqq_nport_audit.md` for every accession and acceptance timestamp.

The disclosure history is informative. On positive equity positions, top-10
weight moved from 55.13% in the 2019-09-30 snapshot to 46.85% in 2026-03-31;
security-level HHI moved from 0.0459 to 0.0312 (effective holdings 21.8 to
32.1). Those numbers are known to a simulated forecaster only after the
respective SEC acceptance dates. With just 27 slow snapshots and no coverage
for most of the discovery window, they were not inserted post hoc into the
already-spent signal holdout.

## What the completed signal study already answers

The registered cross-asset family included credit (HYG), duration/rates (TLT),
gold (GLD), crude oil (USO), and the dollar (UUP). Its discovery QLIKE was
0.344417 versus 0.342753 for the safe baseline, a **0.49% deterioration**.
Adding it to other families did not beat the singleton term-slope winner.

The QQQ market-state family combined abnormal volume and overnight-variance
share. It deteriorated discovery QLIKE by **2.08%**. Thus commodity, dollar,
credit/rate, volume, and overnight-share information have been tested once in a
low-multiplicity composite; none earned access to confirmation.

The one locked candidate, lagged `log(VIX9D/VIX)`, improved confirmation QLIKE
by **2.92%** and won 54.5% of origins, but failed the registered DM threshold
(p=0.1016). The effect also decayed from 5.3%/7.0% in 2022/2023 to 0.2%/0.9%
in 2024/2025. Treat it as a promising feature for future data, not a confirmed
signal.

## Ranked follow-up candidates

These are a backlog, not permission to reuse the spent 2022–2025 confirmation
window.

| priority | family | reliable timing at a 16:00 ET origin | assessment |
|---:|---|---|---|
| 1 | Option convexity: VVIX, skew, or VIX futures curve | Cboe daily closes must be delayed one session because volatility indexes can update through 16:15 ET | Best economic complement to VXN level; use one pre-specified shape factor. Free history/terms must be verified before registration. |
| 2 | Breadth/dispersion | A tradable 16:00 ET proxy such as QQQ versus an equal-weight Nasdaq-100 ETF can be same-origin; constituent breadth needs historical membership | More orthogonal than another macro level and available daily. Prefer a single signed spread, frozen before a new forward holdout. |
| 3 | Funding/liquidity state | ON RRP results are known after the 12:45–13:15 operation; SOFR is published the following morning for prior-day trades; Treasury cash data is delayed | Reliable point-in-time sources, but most variation is regime-specific and slow. Use one liquidity composite and structural-break guards. |
| 4 | Individual FX stress | Tradable ETF closes are same-origin; Fed H.10 is weekly and released Monday at 16:15 for the prior week | UUP already failed inside the cross-asset composite. A future EUR/JPY/CHF breadth factor is defensible only as a new, single registered family. |
| 5 | Dealer gamma | Requires historical strike/expiry option open interest plus quotes/Greeks; OI is previous-night inventory | Economically attractive but not identifiable from VXN or the underlying. Cboe [Option Quote Intervals](https://datashop.cboe.com/option-quote-intervals) offers paid history with optional OI/Greeks from 2012; do not synthesize gamma without it. |
| 6 | M1/M2 money supply | Monthly release, revisions, and a May-2020 definition break | Too slow for next-day variance and weakly incremental to market prices. Suitable only as a preregistered regime interaction, not a daily predictor. |
| 7 | Weather | Historical observations are revised/reconstructed and station availability is not consistently known at the close | Weak causal prior for NDX variance; keep excluded. NOAA describes [GHCN-Daily](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily) as quality controlled and routinely reconstructed. |

Official timing references: [NY Fed ON RRP FAQ](https://www.newyorkfed.org/markets/rrp_faq.html),
[NY Fed SOFR](https://www.newyorkfed.org/markets/reference-rates/sofr),
[Treasury Daily Statement](https://fiscaldata.treasury.gov/datasets/daily-treasury-statement/operating-cash-balance),
[Federal Reserve H.10](https://www.federalreserve.gov/releases/H10/), and
[Federal Reserve H.6 technical Q&A](https://www.federalreserve.gov/releases/h6/h6_technical_qa.htm).

## Decision

Do not promote a new signal from this round. Keep `term_slope` in forward
observation without refitting or repeated testing. For the next genuinely new
holdout, the strongest order is: one option-shape factor, then one breadth
factor, then (only if enough post-regime history exists) one liquidity factor.
Exact daily historical weights require licensed Nasdaq data; the tested free
N-PORT dataset can support only a delayed quarterly robustness feature in a
future, newly frozen holdout.
