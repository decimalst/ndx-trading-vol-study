# Extended realized-volatility data discovery

2026-09-06 local date / 2026-09-07 UTC retrievals. Source discovery only: no
forecast was fitted, scored, selected, or compared. Observation-date metadata was
inspected; numerical estimates were not analyzed. Existing studies and their
protected period remain unchanged. New files are under `data/source_discovery`.

**A dated Oxford-Man extension through January 2020 was found for SPX and all six
fixed foreign indexes. A verified dated extension through 2022 was not found.**
For a separately defined ETF target, the strongest broader source lead is the
University of Chicago Risk Lab's SPY dataset. It contains extensive additional
years, subject to missing dates, measurement validation, and research-only terms.

## Source decisions

| Source | Verified discovery | Current status |
|---|---|---|
| HARNet author supplement | All seven fixed Oxford symbols dated through 2020-01-31; 384–407 additional observations per symbol beyond the current archive | **Dated extension found, quarantined pending measurement/provenance audit** |
| Son et al. spillover supplement | All seven symbols plus OMX; source reportedly downloaded August 2022; 2,615 rows with integer labels only | **Blocked: observation dates absent** |
| Volterra-signature author supplement | Full Oxford CSV has the same Git blob hash as our current 2018 source | **No extension** |
| Dugo et al. / `mfou` | Author documentation explicitly says the distributed data are not their Oxford dataset because dissemination permission is absent | **Not an Oxford replacement** |
| VOLARE university platform | Documented ES futures from 2009; public extended menu also lists NQ; no SPY/QQQ listed | **Blocked: authenticated access and unresolved terminal coverage/session/roll contract** |
| Chicago Risk Lab | Public SPY, QQQ, ES and NQ endpoints; exact date counts below | **Research-source lead; SPY has the best continuous coverage** |

No country was added or removed based on outcomes. Broader ETF/futures options
would constitute new targets, not a splice into the previous cash-index study.

## Dated Oxford-Man extension: HARNet

The [HARNet authors' repository](https://github.com/mdsunivie/HARNet) links their
[paper](https://arxiv.org/abs/2205.07719) and explicitly attributes
`data/MAN_data.csv` to the Oxford-Man Institute website. The file's exact size,
header, fixed revision, and January 2020 tail were inspected before the bounded
download. The original 24,524,969 bytes were then archived as an 8,521,385-byte
gzip, with a 25 MB retrieval cap. No model code from that repository was run.

Local archive:
`data/source_discovery/quarantine/harnet_2020/MAN_data.csv.gz`

Local provenance/date record:
`data/source_discovery/quarantine/harnet_2020/source_manifest.json`

Pinned source:
`https://raw.githubusercontent.com/mdsunivie/HARNet/3b2832f163ec52cab2435163e82f2e816ae348c9/data/MAN_data.csv`

| Fixed symbol | First observation | Last observation | Total rows | Additional dates after 2018-06-27 |
|---|---|---|---:|---:|
| `.SPX` | 2000-01-03 | 2020-01-31 | 5,038 | 397 |
| `.N225` | 2000-02-02 | 2020-01-31 | 4,885 | 384 |
| `.HSI` | 2000-01-03 | 2020-01-31 | 4,919 | 387 |
| `.KS11` | 2000-01-04 | 2020-01-31 | 4,939 | 389 |
| `.FTSE` | 2000-01-04 | 2020-01-31 | 5,064 | 404 |
| `.GDAXI` | 2000-01-03 | 2020-01-31 | 5,092 | 400 |
| `.FCHI` | 2000-01-03 | 2020-01-31 | 5,120 | 407 |

Each selected symbol has no duplicated dates. Its date set through 2018-06-27
exactly matches that symbol in the existing pinned archive: no old dates were
lost or added. This is a **calendar comparison only**, not a comparison of
realized-measure values or an assertion that the data vintage is unchanged.
The full source has 141,537 rows and 31 symbols.

Exact columns: `date, Symbol, open_to_close, rsv, medrv, rsv_ss, nobs, rv5,
close_price, rv10, open_price`. Dates retain local timestamps with offsets;
preserve their local date labels rather than converting through UTC. The source
does not contain `rk_parzen`, `bv`, or the complete older estimator set.
It could therefore support additional dates for the fixed semivariance or
international-information question after validation; it cannot extend the
original kernel arm unchanged. It ends before the main 2020 pandemic period.

The repository declares MIT licensing for its material. That does not establish
the underlying provider's redistribution terms or independently certify the
file's retained-field transformations. It remains quarantined and is not an
admitted model input. The next bounded audit should verify rv5/rsv definitions,
units, missing/invalid values, and overlap consistency, then register a new
evaluation period without touching the earlier results. No such numerical
validation was performed during discovery.

Fingerprints:

- Git blob: `f1f5ed593bdb42304cad042632cb953a522f78fb`.
- Original CSV SHA-256: `c6be6c68280e9100ce734721a4c05bd557ec9009053ad7526fda20b6bd62851e`.
- Local gzip SHA-256: `30ef85340891ea9b8fdd69b8f15e459037c2586505096a0fa27c9ea546685143`.

## Oxford candidates that do not solve the problem

[Son et al.'s data-availability statement](https://onlinelibrary.wiley.com/doi/abs/10.1002/for.2975)
links the author repository now at
[bumhoson/SpilloverVolPrediction](https://github.com/bumhoson/SpilloverVolPrediction).
Its README attributes five-minute realized variances to Oxford-Man and says
they were downloaded in August 2022. The 450,127-byte file was stored in
`data/source_discovery/quarantine/son_spillover_2022/rv_dataset.csv` at commit
`666127515b1ead9cd3a1c675a1c2b7976d1ee738`; SHA-256 is
`9f6065e3ac046a1a18512668fa71457dba5dc51bc1cb6e802a1441cd96a0098c`.

It contains `.SPX,.GDAXI,.FCHI,.FTSE,.OMXSPI,.N225,.KS11,.HSI`, but its first
column is only `0..2614`. Actual date coverage, missing-local-session handling,
and preprocessing cannot be reconstructed safely from this file. No repository
license was detected. A retrieval month is not an observation end date. Dates
were not invented or inferred by matching volatility values. This supplement
remains unusable for the intended historical timing contract.

The [Volterra-signature author repository](https://github.com/lucapelizzari/Volterra_signature_learning)
contains a 46,914,079-byte Oxford CSV. Small uncompressed range requests showed
the 2018-06-27 tail. More conclusively, its Git blob identifier
`48f52aaf37e8419844a18b803c493fa4f0e1b45f` exactly matches the locally computed
Git blob identifier of our existing archive's CSV. No full duplicate was
downloaded. A recently published paper/repository does not imply recent data.

The [multivariate-rough-volatility paper](https://www.tandfonline.com/doi/full/10.1080/14697688.2026.2638519)
reports an Oxford download in June 2022, but its linked `mfou` package's
[data documentation](https://github.com/ranieridugo/mfou/blob/master/man/lrv.Rd)
explicitly distinguishes its supplied Risk Lab stock data from the paper's
Oxford data, citing lack of permission to disseminate the latter. Package code
availability does not establish data availability.

The original Oxford download endpoint could not be opened. The
[VOLARE authors](https://arxiv.org/abs/2602.19732) describe Oxford's discontinuation
in mid-2022. Older Google TFT download code still points to that endpoint rather
than supplying a new archive. Two other author repositories inspected advertised
Oxford data in documentation but did not expose a current corresponding raw CSV
in the checked default-branch tree. No secondary marketing copy was treated as
an authoritative extension.

## VOLARE: promising new measures, access still required

The [VOLARE paper](https://arxiv.org/html/2602.19732v1) documents realized variance,
quarticity, signed semivariances, bipower variation and kernels derived from
Kibot high-frequency data. Its asset appendix lists ES from 2009-09-28, US
stocks from 2015-01-02, and FX from 2009. Futures use second-resolution bid/ask
midpoints; the discussion describes the Sunday 18:00–Friday 17:00 ET week and
daily 17:00–18:00 break. It does not establish the exact daily label boundary or
contract-roll rule needed for an ES target. No such rule was inferred.

The live [university platform](https://volare.unime.it) public menu also lists NQ
among extended futures, but neither SPY nor QQQ appears in its listed assets.
This is menu evidence, not an authenticated inventory. Its FAQ says downloads
contain CSV plus README and update on the second day of each month. The public
access check `/api/financial-data/limits` returned HTTP 401, `Not authenticated`.
The site offers a free-access request form. No account was created or request
sent. Current final dates, actual ES/NQ completeness, dataset-specific terms,
and the README's session/roll definitions remain unverified. No dataset was
downloaded and no platform model was run.

## Chicago Risk Lab: a broader ETF research source

[Dacheng Xiu's Risk Lab](https://dachxiu.chicagobooth.edu/#risklab) identifies its
products as daily **annualized realized volatility** from high-frequency trades
and NBBO mid-quotes. It provides QMLE and five-/fifteen-minute subsampled
comparisons. Its terms restrict use to research and prohibit investment or
commercial use. That permits a research-source assessment, not deployment or
cash-trading claims. The exact annualization conversion and equity session
boundary still require a measurement contract; these fields must not silently
be treated as Oxford daily variance.

Public symbol-catalog queries resolved SPY to permanent identifier `84398` and
QQQ to `86755`. The public per-series endpoints are small (0.20–0.74 MB each)
and require no account. Only headers, date labels, row widths and fingerprints
were retained in `data/source_discovery/risklab_date_metadata.json`.
No estimate file was stored. The metadata record includes full available date
extents, while counts for prospective research are bounded through 2025-10-20.

| Series | Endpoint query | First date | Last available date | Rows through 2025-10-20 | Date rows in 2018–2022 |
|---|---|---|---|---:|---:|
| SPY | `data.php?ticker=84398` | 1996-01-02 | 2026-08-21 | 7,473 | 1,241 |
| QQQ | `data.php?ticker=86755` | 1999-03-10 | 2026-08-21 | 3,136 | 477 |
| ES | `data.php?ticker=ES&ft=1` | 1997-09-09 | 2023-05-26 | 6,123 | 1,096 |
| NQ | `data.php?ticker=NQ&ft=1` | 2015-07-27 | 2023-05-26 | 1,699 | 1,016 |

The first six response lines identify symbol, permanent identifier, description,
row count, first date and last date. Equity records then have 12 whitespace
fields; these futures records have 16. The provider's public chart code reads
field 1 as `YYYYMMDD` and maps zero-based fields `2,5,6,7,10,11` to QMLE-trade,
five-minute-trade, fifteen-minute-trade, QMLE-quote, five-minute-quote and
fifteen-minute-quote series. Remaining fields were not decoded or numerically
validated. The chart applies display clipping, so future ingestion must use
source records, not values copied from the visualization.

Coverage is not continuous merely because first and last dates are far apart:

- SPY lacks 18 QQQ-calendar dates in 2018–2022, including every session from
  2018-02-05 through 2018-02-09. Its annual date counts for 2018–2022 are
  244, 252, 252, 247, 246. A full-session-calendar model must preserve gaps.
- QQQ has a gap from 2006-12-29 to 2020-11-09; 2010–2019 have no records, and
  2020 has only one. It cannot serve as the intended continuous 2018–2022 target.
- ES and NQ have many date omissions: respectively 164 and 244 dates in the
  QQQ calendar during 2018–2022. Their source describes GLOBEX futures, so the
  QQQ-calendar counts are diagnostic comparisons, not exchange-calendar
  certifications. Contract selection, roll handling and date assignment remain
  unresolved. Both returned histories stop in May 2023.

These are date-availability observations only; no relationship to returns,
volatility magnitude, or forecast performance was examined. Missingness may
limit representativeness, so it cannot be repaired by silently rolling across
the remaining rows. SPY merits a separately registered measurement audit;
QQQ/futures need their gaps and session definitions resolved first.

The discovery supports two concrete paths without changing earlier studies:
validate the seven-symbol HARNet extension for additional pre-February-2020
time coverage, or validate Risk Lab SPY for a distinct research-only ETF
realized-volatility target with a broader period. Neither is yet a scored
experiment or a validated predictive signal.
