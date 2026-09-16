# Relative intraday risk: date and source feasibility

Review completed **2026-09-07 07:19 UTC**, before numerical measurement construction. The [prospective question](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/calendar_variance/NEXT_RELATIVE_RISK_DESIGN.md) has sufficient **date/source support** at its proposed start: **1,254 mature training rows** at **2016-01-04**, using the previous observed SPX session cutoff **2015-12-31**. This count was reconstructed from the two raw date indexes and all four raw IV date columns; it was not copied from a prior single-asset study.

Only bounded date indexes, schema names, source/manifest bytes, and earlier frozen source-validity reports were read. No market numerical values, GK measurements, targets, correlations, feature scales, fitted outputs, or associations were constructed or inspected. Full-file hashes identify unchanged bytes without analyzing protected-period values. Every source-date calculation is bounded at **2025-10-20**, before the sealed **2025-11-03** boundary.

## Source identity and paired calendar

| Input | Exact meaning and selected fields | Bounded date coverage |
| --- | --- | --- |
| `data/raw/daily_ohlc.parquet` | QQQ ETF vendor OHLC: `open,high,low,close`; stored `adj close` and `volume` are not needed for the proposed raw-price measurements | 6,696 dates, 1999-03-10–2025-10-20 |
| `data/research_paths/spx_daily.parquet` | Yahoo `^GSPC` SPX price-index OHLC: `open,high,low,close`; stored `volume` is not used | 4,226 reference dates, 2009-01-02–2025-10-20 |
| `data/free_sources/raw/cboe/VXN_History.csv` | `DATE,CLOSE`; Nasdaq-100 implied volatility, a proxy for the QQQ side | 4,053 dates, 2009-09-14–2025-10-20 |
| `data/free_sources/raw/cboe/VIX_History.csv` | `DATE,CLOSE`; SPX implied-volatility level | 9,040 dates, 1990-01-02–2025-10-20 |
| `data/free_sources/raw/cboe/VIX9D_History.csv` | `DATE,CLOSE`; SPX short implied-volatility horizon | 3,721 dates, 2011-01-04–2025-10-20 |
| `data/free_sources/raw/cboe/VVIX_History.csv` | `DATE,VVIX`; volatility-of-VIX index | 4,878 dates, 2006-03-06–2025-10-20 |

Both parquet date indexes are unique, sorted, normalized, and timezone-naive. Their physical schemas contain the named OHLC fields and a `date` index field. Within the entire SPX reference span, the QQQ date set **equals** the SPX date set:

- QQQ dates missing from the SPX reference calendar: **none (`[]`)**.
- Extra QQQ dates inside the SPX reference span: **none (`[]`)**.
- QQQ/SPX preceding-source-date mismatches when an SPX predecessor exists: **none (`[]`)**.

QQQ has **2,470 earlier dates**, 1999-03-10–2008-12-31, outside the SPX reference span. They are excluded prehistory, not extra sessions inside the matched experiment. At the first reference date, **2009-01-02**, QQQ has predecessor **2008-12-31**, while the selected SPX source has no predecessor. That first reference observation is therefore **unknown for both assets' common prior-close-dependent histories**. It must not be made complete by borrowing QQQ's longer history or assuming an unavailable SPX close. Every later paired date has the same preceding date in both source calendars.

The reference calendar is preserved before all rolling operations. There is no intersection followed by compressed rolling, interpolated date, carried-forward value, or substitution of a later common session.

## Field validity and IV gaps

After confirming exact raw-file identity, this audit relies on the previously frozen [QQQ source-validity review](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/overnight_index/SOURCE_FEASIBILITY.md) and [SPX source-validity review](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/index_hinge/SOURCE_REVIEW.md). Those audits found complete positive finite selected OHLC/IV values, valid complete OHLC ranges, and no missing observed fields. This task did not recalculate numerical missingness or price validity. Date absence remains distinct from a missing field on an observed row.

| IV source | Missing SPX dates before its first observation | Missing SPX dates after its first observation |
| --- | ---: | --- |
| VXN | 175 | None |
| VIX | 0 | None |
| VIX9D | 505 | None |
| VVIX | 0 | **2010-11-11, 2013-05-13** |

Each IV input is aligned to the **exact previous SPX session**, with no fill. The 2010 VVIX gap occurs before all-input support begins. The later VVIX gap excludes entry **2013-05-14**; it is the only subsequent date-support hole inside the first training window. VXN and VIX9D inception dates are source-coverage constraints, not missing-at-random observations.

## Strict history support and initial maturity gate

The date-only calculation separately requires 22 complete paired intraday observations and 22 complete paired prior-close-dependent observations, followed by the one-reference-session feature lag. Shorter 1/5-session windows are also supported whenever these complete 22-session windows exist. The latter condition covers total GK-plus-raw-overnight history and raw close-return leverage history. It is stricter at the source's leading boundary than the intraday-only history.

| Date-support fact | Reconstructed result |
| --- | --- |
| First entry with 22 paired intraday observations through its preceding session | 2009-02-04 |
| First entry with 22 paired prior-close-dependent observations through its preceding session | 2009-02-05 |
| First entry with all histories and VXN/VIX/VIX9D/VVIX date support | **2011-01-05** |
| First feature-date-complete January 2016 fit | **2016-01-04** |
| Fit's preceding observed SPX cutoff | **2015-12-31** |
| Prior origins with an actual next-reference-session date mature by cutoff, before IV/history restrictions | 1,761 |
| Common mature date-supported training rows | **1,254** |
| Fixed training minimum | 1,000 |
| First / last supported training origin | **2011-01-05 / 2015-12-30** |
| Latest supported training label maturity date | **2015-12-31** |

For maturity, the next observed SPX date is used only as a prospective label date, and QQQ must have that exact date. The training origin must precede the fit and this next-date label must mature by the fit cutoff. No numerical label is computed. Entry 2015-12-31 cannot train the January 4 fit: its next-session label matures after that fit's cutoff.

| Training origin year | Mature date-supported rows |
| --- | ---: |
| 2011 | 250 |
| 2012 | 250 |
| 2013 | 251 |
| 2014 | 252 |
| 2015 | 251 |

These sum to the independently reconstructed **1,254**. They are an upper bound on final numeric common rows: a zero correlation denominator, nonfinite transform, or other frozen numerical measurement failure could remove support or abort the design. No empirical correlation or training-feature scale was examined here.

## Exact byte pins and provenance

All six source hashes equal their existing manifests:

| Source | SHA-256 |
| --- | --- |
| QQQ parquet | `710290d8ad8569172559334b23e423b50ecf7cd10f0fd8ec24a9fc29608bca4d` |
| SPX parquet | `3958fbb1eb36689df1596c26b9f0e02e3f1d4fa3b0032381e5287a2a03697bf0` |
| VXN CSV | `753b5a406a7a888a9e4ca1a3e37a71fbb415a3883bc4b02ff20dbf0c5bdc6c98` |
| VIX CSV | `a34aabce269632f30904cf482986dd50b6d4cf51f2203dc51a0a9f460f3c90b2` |
| VIX9D CSV | `0d6f600ee71bf6ffb5069d0c583cbe0b1ec97df6da4e4e439d5f0f22f5616abe` |
| VVIX CSV | `f6bc726455fa3859c662875a005e97ba656b9536ae58fa6977ad24c6228e4f6a` |

The matching QQQ manifest is `data/history_extension/source_manifest.json`, SHA-256 `3a0f35ea40c102bfcea3ddd38dc323f2fb5be1990b9b9b366c81089d35fd3ec3`. It pins the QQQ source and records `frozen_on: 2026-08-12`; it does **not** record an exact QQQ acquisition timestamp. The [downloader](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/fetch.py) uses yfinance `Ticker('QQQ').history(..., auto_adjust=False)`, retains vendor OHLC plus adjusted close/volume, and strips the timestamp timezone before normalizing to civil dates.

The SPX manifest is `data/research_paths/source_manifest.json`, SHA-256 `457afd656244fc527535984874ee11313fd3156815f0d982d0c9f29ab85d244c`. It records ticker `^GSPC`, `auto_adjust: false`, and acquisition **2026-08-12T17:59:14.463673+00:00**. Its historical normalization discarded incomplete OHLC rows, so matched date equality establishes equality of two archived observed calendars, not completeness against an independent exchange-session ledger.

The four Cboe adjacent `.csv.manifest.json` files record retrieval **2026-08-12T20:45:00Z**, `source_version: daily-live`, and source IDs `cboe:daily-index:SYMBOL`. Their hashes are VXN `1314c602a9e2c2ce6561c450fe4e7ea3c310e975d386e5ff7d7ecf51e88d3391`; VIX `51caa998f0dadc1a707182a0222f793626da93da535a31f49d66b79db0ab2917`; VIX9D `aee15817f60f0e3cde73a55fa99b807ca4dd10b6b045d8f9b87683df7050cab0`; VVIX `f79ecf39b83642b0df23743a1f3aa0077bbf8fc8628fdc0b5de49f0a8ccd0a52`. The source identities and Cboe methodology references are documented in the earlier [source review](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/overnight_index/SOURCE_FEASIBILITY.md).

## What the matched dates cannot establish

The instruments remain **QQQ ETF and SPX price index**. QQQ is not a raw Nasdaq-100 index-level feed, and the current configuration's `^NDX` cross-check comment does not supply one. Their vendor daily open/high/low/close fields do not certify synchronized auction fills, identical sampling rules for intraday extrema, or exact historical 16:00 publication snapshots. A one-session predictor lag is a declared archival timing convention; exact historical release latency and revisions remain unverified.

“Raw OHLC” here means the selected vendor OHLC fields, rather than the separately stored adjusted-close field. `auto_adjust=False` does not itself certify a complete historical corporate-action accounting convention. The QQQ file omits dividend and split event columns. Within-session OHLC ratios cancel a common multiplicative unit/adjustment factor and avoid directly including the previous-close distribution gap in the proposed intraday response; they do not remove tracking effects, inconsistent adjustments, or source error. The proposed raw overnight and close-return **controls** still carry those intersession corporate-action limitations. No source repair, adjusted-close substitution, or distribution inference is authorized by this audit.

Both market feeds and the Cboe inputs are later archival vintages. Early VIX9D observations are back-calculated, not proof of contemporaneous live publication. VXN is an index-IV proxy for the QQQ ETF; VIX/VIX9D concern SPX, and VVIX is not either asset's realized variance. The proposed target remains a relative daily OHLC-risk measurement, not integrated variance, a hedged portfolio variance, a matched index-return spread, or an executable relative-value strategy.

**Conclusion:** date/source support clears the unchanged 1,000-row initial gate with 254 rows of margin. Exact paired calendar alignment is established within the source span, subject to the explicit first-row boundary. A separately frozen numerical measurement contract must still assess transform validity and feature-scale feasibility before any forecasting claim or run. No lower data threshold or changed asset/target convention follows from this source-only result.
