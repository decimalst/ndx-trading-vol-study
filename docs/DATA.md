# Data sourcing and calendar provenance

*Split out of `README.md` on 2026-08-13 to keep the top-level document short.
Nothing here changed in the move; it is the same text.*

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
| Wider free-source program | CFTC, Cboe, Kaggle, Hugging Face, and Zenodo | free to download; rights vary | raw mirrors remain local where upstream rights are unclear; see the [source ledger](../reports/FREE_DATA_SOURCES.md) before using any derived panel |

The source program also provides exact revision/hash inventories, disk
preflight, local sidecar manifests, parsers, and compact audit artifacts. It
does **not** treat uploader license labels as proof that exchange-derived data
may be redistributed. The 478 GB Hugging Face companion stock repository is the
only listed free source intentionally not downloaded: the frozen two-times
preflight exceeds available disk, and its upstream APIs and corporate-action
handling are undocumented.

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

