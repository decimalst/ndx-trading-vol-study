# Data licensing and provenance

The MIT grant in `LICENSE` covers code and written material authored here. It
does **not** relicense third-party data. Everything under `data/` and
`calendars/` is redistributed so the published results can be verified from a
clean clone, and each source retains its original terms.

If you intend to use any of this data for anything other than reproducing the
results in this repository, check the upstream terms yourself. The
classifications below are the author's reading, not legal advice.

| Path | Source | Terms as understood |
|---|---|---|
| `data/raw/SKEW_History.csv` | Cboe published historical index values | Free download from cboe.com for informational use. Redistributed here as the input to a published result. |
| `data/free_sources/processed/cboe_daily_local.parquet` | Derived from the above plus other Cboe close files | Normalized panel built by `src/free_data_sources.py`. Same provenance as its inputs. |
| `data/raw/oxford_man_realized.zip`, `data/raw/oxford_man_spx.parquet` | Oxford-Man Institute Realized Library, pinned to commit `308b795f…` | Academic dataset, archived. Pinned to an immutable commit because the upstream branch 404s. |
| `data/raw/qqq_holdings_2026-08-10.csv` | Invesco QQQ "Complete Holdings" | Published daily by Invesco. A point-in-time snapshot, redistributed as a model input. |
| `data/raw/qqq_nport_*.parquet` | SEC EDGAR N-PORT filings | US public records. No restriction. |
| `data/raw/daily_ohlc.parquet`, `hourly_bars.parquet`, `cross_asset_daily.parquet` | Yahoo Finance via `yfinance` | Personal/research use per Yahoo's terms. Not redistributable for commercial purposes. |
| `data/raw/implied_corr.parquet`, `short_dated_iv.parquet` | Cboe index values (COR1M, VIXEQ, VIX9D) | As above. |
| `calendars/fomc.csv`, `cpi.csv`, `nfp.csv` | federalreserve.gov, bls.gov | US public records. No restriction. |
| `calendars/earnings_top.csv`, `earnings_fetched.csv` | Derived from `yfinance` announcement timestamps | Same terms as the Yahoo data. **Sessions are inferred from timestamps, not verified** — see the limitations in `README.md`. |

## Excluded from the repository

`data/free_sources/raw/` is not committed: bulk third-party archives that are
large, are not needed to reproduce any published number, or whose upstream
provenance is undocumented. Some of these are exchange-derived mirrors whose
uploader-declared licences (CC0, MIT, CC BY) do not match their apparent
provenance; `reports/FREE_DATA_SOURCES.md` records which, and those are marked
private-research-only regardless of the declared label. Fetch commands are in
`docs/DATA.md`.

## Removal

If you hold rights to any file here and want it removed, open an issue and it
will be taken out of the working tree and out of git history.
