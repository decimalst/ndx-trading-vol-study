# Overnight QQQ source feasibility

2026-09-06. Independent source and completeness audit for `overnight_index.yaml`.
No model was fitted, no forecast loss or return association was examined, and no
older source or frozen artifact was changed. Numeric source reads were bounded
through 2025-10-20; the protected phase starting 2025-11-03 was not inspected.

**The fixed proxy study has sufficient history: its first fit has 1,254 common
eligible training rows, above the required 1,000.** The source limitations are
archival adjustment provenance and missing corporate-action accounting, rather
than a shortage of observations. They limit interpretation of any eventual lead.

## Exact input contract

The following paths are relative to the repository. Both parquets use a
timezone-naive `date` index stored as the parquet `date` field. Do not interpret
the storage index as an additional predictor.

| Path | Fields used | Bounded source coverage |
|---|---|---|
| `data/raw/daily_ohlc.parquet` | `open`, `high`, `low`, `close`, `adj close`, `volume` | 6,696 QQQ sessions, 1999-03-10–2025-10-20 |
| `data/raw/cross_asset_daily.parquet` | `hyg`, `tlt`, `gld`, `uso`, `uup` | Per-column coverage below |
| `data/free_sources/raw/cboe/VXN_History.csv` | `DATE`, `CLOSE` | 4,053 observations, 2009-09-14–2025-10-20 |
| `data/free_sources/raw/cboe/VIX_History.csv` | `DATE`, `CLOSE` | 9,040 observations, 1990-01-02–2025-10-20 |
| `data/free_sources/raw/cboe/VIX9D_History.csv` | `DATE`, `CLOSE` | 3,721 observations, 2011-01-04–2025-10-20 |
| `data/free_sources/raw/cboe/VVIX_History.csv` | `DATE`, `VVIX` | 4,878 observations, 2006-03-06–2025-10-20 |

VXN, VIX and VIX9D raw headers are `DATE,OPEN,HIGH,LOW,CLOSE`; VVIX's header is
`DATE,VVIX`. Parse the raw CSV fields directly and align to the QQQ observation
calendar without filling. Do not substitute a previously processed IV series.
CSV dates are US month/day/year. No duplicate dates were found in these bounded
inputs.

| Cross column | First nonmissing date | Observations through 2025-10-20 | Missing QQQ sessions after its first date |
|---|---|---:|---:|
| `hyg` | 2007-04-11 | 4,663 | 0 |
| `tlt` | 2002-07-30 | 5,845 | 0 |
| `gld` | 2004-11-18 | 5,263 | 0 |
| `uso` | 2006-04-10 | 4,914 | 0 |
| `uup` | 2007-03-01 | 4,691 | 0 |

QQQ has no missing or nonpositive retained fields and no rows violating the
ordinary OHLC bounds. The cross series and retained IV values are positive and
finite. VXN and VIX9D have no missing QQQ dates after their first observation.
VIX lacks 1999-12-31, well before this study's complete training pool. VVIX has
61 absent QQQ dates after its first observation: 59 in 2006, 2010-11-11, and
2013-05-13. The last is material: it makes entry 2013-05-14 incomplete after the
one-session feature lag. There are no subsequent common-input holes. These are
checks against the archived QQQ observation calendar, not certification against
an independently acquired exchange calendar.

## Provenance and meaning

`src/fetch.py::fetch_daily_ohlc` calls Yahoo through yfinance with
`auto_adjust=False` and explicitly retains the six QQQ fields. It discards any
dividend and split event columns. The file's matching earlier source manifest
is `data/history_extension/source_manifest.json`; it is a later acquired
historical snapshot, not a sequence of historical publication vintages.

`src/fetch.py::fetch_signal_inputs` selects `Adj Close` when present and otherwise
uses `Close`. The saved cross parquet does not preserve the field selected for
each ticker. **These are cached Yahoo close series built by an
adjusted-close-preferred/raw-close-fallback rule; this audit does not certify
that adjusted close was selected for every column.** There is no need to
replace a frozen exploratory covariate on that account. Any passing cross arm
would require confirmation of those fields before promotion beyond an
exploratory lead. A fresh download would be a different vintage, not proof of
what the old downloader received.

The Cboe raw files each have an adjacent `.csv.manifest.json`, declaring
`source_id=cboe:daily-index:SYMBOL`, `source_version=daily-live`, and retrieval
time `2026-08-12T20:45:00Z`. Their declared byte counts and SHA-256 values match
the current files. The manifests do not contain a URL; the primary download
pattern recorded in `src/fetch.py` is
`https://cdn.cboe.com/api/global/us_indices/daily_prices/{SYMBOL}_History.csv`.
Cboe's [historical-data page](https://www.cboe.com/tradable_products/vix/vix_historical_data)
identifies VIX, VIX9D and VVIX historical series.

VXN measures expected Nasdaq-100 volatility over 30 days; VIX measures expected
S&P 500 volatility over 30 days. VIX9D is the S&P 500 nine-day measure, so its
ratio to VIX is a cross-index term-structure input for QQQ. VVIX measures implied
volatility associated with the 30-day forward VIX, using nearby VIX option
prices; it is not another QQQ volatility level. These identities follow
[Cboe's broad-index methodology](https://cdn.cboe.com/api/global/us_indices/governance/Volatility_Index_Methodology_Selected_Broad_Based_Index_Equity_and_ETF_Volatility_Indices.pdf),
[SPX term methodology](https://cdn.cboe.com/api/global/us_indices/governance/Volatility_Index_Methodology_Selected_SPX_Target_Expected_Volatility_Term_Indices.pdf),
and [VIX factsheet](https://cdn.cboe.com/resources/futures/VIX_fact_sheet_2019.pdf).

Cboe's SPX term methodology distinguishes VIX9D's first historical value
(January 2011) from launch (October 2013). Thus the early training history is
back-calculated history, not evidence of an index published live in 2011. All
scored origins start in 2016, after launch, but the 2026 download still does not
establish the vintage available to each historical refit. Conservative date
lags address information timing within the cache; they do not repair this
archival-vintage limitation.

## Training and scoring-row feasibility

An independent reconstruction used all 25 input columns together: intercept;
own adjusted-overnight and daytime log-return means over 1/5/22 observations;
log mean GK-plus-raw-overnight variance over 1/5/22; log VXN and log VIX; entry
Tuesday–Friday indicators; five individual cross log returns; volume-pressure
latest value and strict five-session mean; log(VIX9D/VIX); and log VVIX.

All market transforms end on the previous actual QQQ session. The volume
reference excludes its own observation: preceding 252 log-volume observations,
minimum 126, sample standard deviation. Raw GK is floored at `1e-10` before
adding squared raw overnight log return. No finite-input check depends on a
forecast or on the size/sign of the target return.

| Completeness/maturity check | Result |
|---|---|
| First complete entry | 2011-01-05, determined by lagged VIX9D |
| First scheduled fit entry | 2016-01-04 |
| Fit's market cutoff | 2015-12-31 |
| Complete eligible training entries | 1,254 |
| First/last training entries | 2011-01-05 / 2015-12-30 |
| Latest admitted training-label availability | 2015-12-31 |
| Expected development entries, label available by 2019-12-31 | 1,005 |
| Expected evaluation entries through 2025-10-17, target by 2025-10-20 | 1,457 |

An entry's next-opening target uses the next session's stored adjustment factor,
so label availability is conservatively that next session's **close**. Before
the entry close, a monthly fit can therefore use labels available through the
previous session only. In particular, the first fit excludes entry 2015-12-31:
its next-session label is available after the 2016-01-04 close. Prior-session
Cboe inputs require one shift from entry, not the extra shift used in earlier
after-close studies. Entry weekday is known in advance; the realized next
opening date is not a predictor.

## Adjustment-factor measurement audit

The frozen flag is `abs(log(F_next/F_entry)) > 1e-5`, where
`F = adj close / close`. Counts below inspect only that factor and eligibility;
they do not associate the flag with returns, forecasts, or forecast losses.

| Sample | Eligible entries | Flagged | Unflagged |
|---|---:|---:|---:|
| All bounded QQQ entry/next-session pairs | 6,695 | 86 | 6,609 |
| First fit's complete training pool | 1,254 | 22 | 1,232 |
| Mature development sample | 1,005 | 16 | 989 |
| Evaluation sample | 1,457 | 23 | 1,434 |

A flag is a potential adjustment boundary, not an independently verified action.
It cannot filter entries or become a predictor. The primary sample and all
training rows retain these observations; the protocol's unflagged sensitivity
uses the same frozen forecasts after fitting.

[Yahoo's adjustment documentation](https://help.yahoo.com/kb/SLN28256.html)
describes multiplicative split and dividend adjustments to earlier prices. The
resulting adjusted overnight proxy can differ from exact price-plus-dividend
entitlement profit when a market move accompanies a distribution. The factor
also cannot establish original share units around splits. A common later
adjustment multiplier cancels from adjacent returns algebraically, but revised
historical prices or action records can still change them. The
[yfinance history contract](https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html)
distinguishes automatic OHLC adjustment from repair; its
[repair documentation](https://ranaroussi.github.io/yfinance/advanced/price_repair.html)
describes missing dividend adjustments and wrong action dates. No repair or
replacement was performed here.

The issuer's [QQQ product distribution table](https://www.invesco.com/us/financial-products/etfs/product-detail?audienceType=investors&productId=QQQ&ticker=QQQ)
is a primary source that distinguishes ex-date, record date, payment date, and
per-share amount. Its historical table was located, but a complete historical
action file with declaration timestamps was not added. The current product URL
can redirect to the general QQQ site, and the cached source fields alone cannot
establish a historical cash ledger. This does not prevent the explicitly
adjusted-return proxy study. It does prevent treating a result as demonstrated
cash profit or verified auction execution.

## Frozen input fingerprints

Full-file byte hashes below identify the acquired snapshots; hashing bytes did
not expose protected-period numeric observations.

| Input | SHA-256 |
|---|---|
| `data/raw/daily_ohlc.parquet` | `710290d8ad8569172559334b23e423b50ecf7cd10f0fd8ec24a9fc29608bca4d` |
| `data/raw/cross_asset_daily.parquet` | `53f11ec90fc4f68d8d5592af7d893f7b5707bfbd7799b0735681acc18e72ac53` |
| `VXN_History.csv` | `753b5a406a7a888a9e4ca1a3e37a71fbb415a3883bc4b02ff20dbf0c5bdc6c98` |
| `VIX_History.csv` | `a34aabce269632f30904cf482986dd50b6d4cf51f2203dc51a0a9f460f3c90b2` |
| `VIX9D_History.csv` | `0d6f600ee71bf6ffb5069d0c583cbe0b1ec97df6da4e4e439d5f0f22f5616abe` |
| `VVIX_History.csv` | `f6bc726455fa3859c662875a005e97ba656b9536ae58fa6977ad24c6228e4f6a` |
