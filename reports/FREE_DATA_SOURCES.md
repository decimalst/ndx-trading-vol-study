# Free-data source program

This ledger records what was actually acquired, what its provenance permits us
to claim, and whether it produced an empirical result. These are separate
questions. A file being free to download does **not** establish that an uploader
had the right to relicense exchange-derived data, and a successfully parsed
file is not a successful predictive test.

The acquisition contract is
[`free_data_sources.yaml`](../free_data_sources.yaml). The frozen empirical
contracts are [`free_signal_study.yaml`](../free_signal_study.yaml),
[`nq_intraday_study.yaml`](../nq_intraday_study.yaml), and
[`surface_data_study.yaml`](../surface_data_study.yaml). Raw third-party files
stay under `data/free_sources/raw/`, are ignored by Git, and are not redistributed.
Only compact audits, non-reconstructive aggregates, forecasts, and metrics may
be checked in when their source-specific policy permits it.

## Status vocabulary

- **Official** means the publisher is the originating agency or index
  administrator. It does not mean unrestricted redistribution.
- **Uploader** means the repository or archive is a third-party mirror. Its
  license label is recorded as a claim, not treated as proof of upstream rights.
- **Quarantine** means local private-research use only; raw data and
  reconstructive derivatives are not publishable from this repository.
- **Verified result** means a second implementation recomputed timing fences,
  rows, and metrics without importing the study implementation. **Pending**
  means no empirical conclusion exists yet.

## Source and acquisition ledger

| Source | Provenance and rights posture | Local acquisition | Promotion status | Empirical status |
|---|---|---|---|---|
| [CFTC Traders in Financial Futures](https://dev.socrata.com/foundry/publicreporting.cftc.gov/gpe5-46if), Nasdaq-100 consolidated code `20974+` | **Official** U.S. government source; public-domain class with CFTC acknowledgement | Acquired through Socrata view `gpe5-46if`; raw kept local | Compact normalized weekly panel promoted with a recorded SHA-256 | **Verified result:** the positioning augmentation failed its registered two-metric gate |
| [Cboe VIX, VXN, VIX1D, VIX9D, SKEW, and VVIX histories](https://www.cboe.com/tradable_products/vix/vix_historical_data) | **Official** Cboe files; personal/non-commercial download terms, so redistribution is not assumed | All six acquired as raw local CSVs | 26,874 delayed close rows normalized locally; the panel is ignored by Git. Forty-seven historical VIX rows fail OHLC geometry, so only their valid closes are retained and their OHLC fields are quarantined | Source audit complete; no new combined-panel prediction result |
| [Hugging Face `misikoff/SPX`](https://huggingface.co/datasets/misikoff/SPX), revision `498783…` | **Uploader** claims MIT; upstream price-data provenance is undocumented | Pinned CSV acquired and hash matched | Quarantine; 5,075 of 24,167 data rows have zero opens, so the strict OHLC parser rejects the mirror and no values are imputed | Source audit complete; no predictive result |
| [Hugging Face `fabhaus/equities_5m_options`](https://huggingface.co/datasets/fabhaus/equities_5m_options), revision `99d9d3…` | **Uploader** uses an `other` license and describes the bars only as coming from “various sources” | All 22 frozen monthly shards from 2024-01 through 2025-10 matched pinned hashes | Quarantine; activity-only trade bars lack NBBO, open interest, IV, and Greeks and therefore cannot identify dealer GEX | **Verified `INSUFFICIENT_DATA`:** the fixed four-component composite is undefined because near-expiry share has zero scale; zero forecasts |
| [Hugging Face `fabhaus/equities_5m_stockprices`](https://huggingface.co/datasets/fabhaus/equities_5m_stockprices), revision `f17c0b…` | **Uploader**; several unnamed APIs and unadjusted corporate actions | Not bulk-downloaded | Disabled by the acquisition contract: about 478 GB, unclear upstream rights, and no justified need for the companion archive | No result; intentionally not acquired |
| [Zenodo TSLA options](https://doi.org/10.5281/zenodo.15496947) | **Uploader/depositor** claims CC BY 4.0; original market-data source is undocumented | 1,035,799,239 bytes; pinned MD5 and schema matched; 4,584,740 data rows from 2018-08-06 through 2023-08-31 | Quarantine despite successful validation; the file actually contains bid/offer, open interest, IV, and Greeks | Source audit complete; no frozen predictive protocol or result |
| [Kaggle QQQ option chains, version 1](https://www.kaggle.com/datasets/kylegraupe/qqq-daily-option-chains-q1-2020-to-q4-2022) | **Uploader** claims CC0; upstream provenance is undisclosed and the advertised 2020 start is not in the file | Archive acquired; 1,775,749 rows, actual coverage 2021-01-04 through 2022-12-30 | Raw quarantine; only daily shape aggregates and forecasts promoted | **Verified private diagnostic:** augmented QLIKE was worse by 0.04247 on 166 rows; verdict `INCONCLUSIVE` |
| [Kaggle SPY option surface, version 2](https://www.kaggle.com/datasets/dudesurfin/spy-options-eod-volatility-surface-2010-2023) | **Uploader** claims MIT and says the source was OptionsDX; upstream redistribution rights were not verified | Archive acquired; 14 yearly files, 2010-01-04 through 2023-12-29 after calendar checks | Raw quarantine; aggregate-only output | **Verified private diagnostic:** late-confirmation augmented QLIKE was worse by 0.01332 on 1,464 rows; verdict `INCONCLUSIVE` |
| [Kaggle AAPL option chains, version 4](https://www.kaggle.com/datasets/kylegraupe/aapl-options-data-2016-2020) | **Uploader** claims CC0; upstream provenance is undisclosed | Archive acquired; 1,563,515 rows, 2016-01-04 through 2023-03-31 after calendar checks | Raw quarantine; the 2020 split was audited and the crossing return was not rewritten | **Verified private diagnostic:** augmented QLIKE was worse by 0.00450 on 1,253 rows; verdict `INCONCLUSIVE` |
| [Kaggle NQ one-minute bars, version 1](https://www.kaggle.com/datasets/tgtanalytics/nq-futures-1min-bar-2022-2025) | **Uploader** claims CC0; contract identity and continuous-futures roll method are undocumented; the 1,048,575-row file ends mid-session | Archive and CSV acquired | Raw quarantine; stitch-like neighborhoods and incomplete sessions excluded conservatively | **Verified no-evaluable-fold outcome:** the frozen 180-row training gate failed in both 2024 and 2025, so zero forecasts were scored |

The GitHub entries in the supplied list are software rather than additional
market-data archives. This repo already consumes CFTC's official Socrata data
directly, so a third-party COT downloader would add code dependency but no
observations. Optopsy is an options research framework, not a data license or
history. The NQ repositories describe research workflows and commercial data
vendors but do not provide the long, auditable CME history needed here. They
were therefore evaluated as tooling references, not mislabeled as acquired
datasets. WRDS/OptionMetrics and the named exchange vendors were excluded from
this free-source pass because they require institutional or paid access.

## What the completed diagnostics say

### CFTC positioning: ranking lift moved, AUC did not

The weekly CFTC study used one QQQ origin per release, delayed each Tuesday
report by ten calendar days and then to the first eligible QQQ session, and
removed the known 2018-2019 shutdown and 2023 ION-backlog periods. It scored 591
calm origins from 2013-01-03 through 2025-10-10, with 99 transition labels
(16.75%).

| Model | AUC | Top-decile lift | Top-decile event rate |
|---|---:|---:|---:|
| RV-history benchmark | 0.8308 | 3.482x | 58.33% |
| + leveraged-money net/open-interest share | 0.8304 | 3.681x | 61.67% |

The augmentation lost 0.0004 AUC while gaining 0.199x lift. Because the frozen
gate required improvement on **both**, the result is **FAIL**, not a partial
win. The independent verifier passed 15 checks, reconstructed all 591 rows and
492 valid negative labels with missing trigger dates, and matched the normalized
source SHA-256 `776bcde3479bd8250f5a10a6d92fbf37473107d7d5562b8186b8b794e4a7c09d`.
See the [CFTC report](free_signal_study/cftc_positioning.md).

### NQ intraday: the data cannot evaluate the frozen model

The NQ pipeline retained 278,041 RTH minute bars, produced 726 sessions, found
678 quality-eligible sessions, and excluded 16 sessions around possible stitch
or bad-print triggers. It then stopped at the pre-specified training gate:

| Test year | Eligible training rows | Frozen minimum | Evaluable? |
|---|---:|---:|---|
| 2024 | 63 | 180 | No |
| 2025 | 135 | 180 | No |

The independent verifier reports `VERIFIED_NO_EVALUABLE_FOLDS`. This is neither
a positive nor a null predictive result; the free file is too short after
quality, timing, and completed-target requirements. Relaxing the minimum after
seeing this would change the experiment. The unknown contract stitch also keeps
the source exploratory even if a future longer file clears the sample gate.
The [dedicated NQ audit](nq_intraday_study.md) records all fold counts and the 48
descriptive BNS-significant jump sessions without treating them as a forecast
result.

### Option-surface shape: better tail ordering did not lower mean QLIKE

The frozen surface study measured 30-day ATM IV, 25-delta skew, 9-day minus
30-day term slope, and gamma-weighted reported volume. The last quantity is
explicitly **not dealer gamma exposure** because these archives have no open
interest. Every surface snapshot was delayed one full exchange session; the
target was the following session.

On the governing SPY late-confirmation split, the augmented model's AUC rose
from 0.8298 to 0.8349, but mean QLIKE worsened from 1.5359 to 1.5492 and
top-decile lift fell from 3.787x to 3.683x. Its paired QLIKE interval
`[-0.0189, 0.0493]` spans zero, so the frozen verdict is `INCONCLUSIVE`.
QQQ and AAPL show the same central pattern—higher AUC but worse mean QLIKE—and
are mechanism diagnostics only. The independent verifier passed 20 checks over
3,850 saved forecasts, including the measurement/origin/target ordering and
raw-archive hashes. See the [surface report](surface_data_study.md).

### HF option flow: a registered source field has no variation

The 22 frozen shards contain about 25.5 million raw lines. Streaming ingestion
hashed every byte while parsing only QQQ/SPY rows and produced 406 QQQ and 439
SPY activity days; absent sessions were never manufactured as zero. The
registered `near_expiry_volume_share_7d` is nevertheless exactly zero on every
retained day for both symbols. Its strictly-prior standard deviation is zero,
so the prewritten equal-weight four-component composite has no defined value.
Dropping the component, substituting a zero z-score, or reweighting the other
three after learning this would change the protocol. The independent verifier
therefore confirms `INSUFFICIENT_DATA`, 0 finite composite rows, and 0 forecasts.
See the [HF flow report](free_signal_study/hf_option_flow.md).

## Completed source audits that are not predictive results

- **HF SPX:** the pinned hash matches and dates are unique, but 5,075 rows from
  the historical file have zero opens. The strict OHLC parser quarantines all
  24,167 rows rather than imputing an open or silently switching target fields.
- **Zenodo TSLA:** the pinned MD5
  `63722cfae0ee22bfa5fabcd1720a3d7c` and frozen schema match. The file has
  4,584,740 data rows, 1,277 quote dates from 2018-08-06 through 2023-08-31,
  and—contrary to the original listing—bid/offer and open-interest columns.
  This makes it potentially useful for a future private TSLA protocol, but
  undocumented upstream rights still bar redistribution and no result exists.
- **Cboe combined panel:** 26,874 official-source close observations were mapped
  only to the next locked QQQ session. The normalized file remains local-only.
  VIX contains 47 rows whose OHLC geometry is internally inconsistent; their
  close is valid and retained for close-only use, while the OHLC fields remain
  quarantined. No combined-panel hypothesis was run.
- **CFTC pre-consolidation code `209742`:** 1,052 official rows from 2006-06-13
  through 2026-08-04 were normalized for continuity audits. The predictive
  result above deliberately remains on the non-overlapping consolidated
  `20974+` contract and its stricter ten-day/blackout timing protocol.

## Reproduction and review pointers

- Source identities, pinned revisions, expected sizes, licenses, and download
  commands: [`free_data_sources.yaml`](../free_data_sources.yaml)
- Acquisition and parser contracts: [`src/free_data_sources.py`](../src/free_data_sources.py)
- CFTC implementation and independent verifier:
  [`src/free_signal_study.py`](../src/free_signal_study.py) and
  [`src/verify_cftc_signal_study.py`](../src/verify_cftc_signal_study.py)
- NQ implementation and independent verifier:
  [`src/nq_intraday_study.py`](../src/nq_intraday_study.py) and
  [`src/verify_nq_intraday_study.py`](../src/verify_nq_intraday_study.py)
- Surface implementation and independent verifier:
  [`src/surface_data_study.py`](../src/surface_data_study.py) and
  [`src/verify_surface_data_study.py`](../src/verify_surface_data_study.py)
- HF flow implementation and independent verifier:
  [`src/free_signal_study.py`](../src/free_signal_study.py) and
  [`src/verify_free_signal_option_flow.py`](../src/verify_free_signal_option_flow.py)
- Compact auxiliary source audit:
  [`src/free_source_panels.py`](../src/free_source_panels.py)

All of these are post-program diagnostics on reused or third-party data. None
reopens, extends, or silently enters the sealed NDX clean window.
