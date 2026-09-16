# Additional Treasury result archive recovery

Reviewed September 8, 2026; retrieval receipts use September 9 UTC. **Two archived result PDFs were recovered with verified TLS, matching Internet Archive content digests and matching dated release headers.** The earlier [bounded search review](CLOCK_RESULT_RECOVERY_REVIEW.md) remains unchanged. Its SHA-256 is `89018dc17b9e299974da701bae45dc348c6684783a8a8013d51a7ee8b5615e0b`.

This additional method was authorized after a separate exact-URL Internet Archive request succeeded for the previously missing YY0 notice. The earlier browsing-tool safe-open failures did not describe Internet Archive availability.

## Requests and observed evidence

One batch queried CDX for the three exact original result URLs, filtering HTTP200/application-pdf records. All three index requests returned HTTP200 over certificate-verified HTTPS. The earliest matching snapshot was then requested once for each of two matches. A separately authorized non-www Z52 query was attempted once. Six requests were made in total; no automatic retry, alternate snapshot, redirect following or insecure TLS option was used. Each request had a30-second transfer limit and35-second subprocess limit. Body limits were2MiB for CDX and8MiB for PDF.

| Event | Archive evidence | Visual original-document check |
| --- | --- | --- |
| 2019-12-23 / 912828YZ7 | One matching CDX record; earliest capture **2019-12-23T22:57:21Z**. Its PDF returned HTTP200. | Complete single page rendered and viewed. The release header says December23,2019; title is TREASURY AUCTION RESULTS; own CUSIP is912828YZ7 and term is2-Year Note. [Exact snapshot](https://web.archive.org/web/20191223225721id_/https://www.treasurydirect.gov/instit/annceresult/press/preanre/2019/R_20191223_3.pdf). |
| 2020-12-09 / 91282CAV3 | Two matching CDX records; earliest capture **2020-12-10T11:25:17Z**. Only that earliest PDF was requested; it returned HTTP200. | Complete single page rendered and viewed. The release header says December09,2020; title is TREASURY AUCTION RESULTS; own CUSIP is91282CAV3 and term is9-Year11-Month Note. [Exact snapshot](https://web.archive.org/web/20201210112517id_/https://www.treasurydirect.gov/instit/annceresult/press/preanre/2020/R_20201209_3.pdf). |
| 2020-01-27 / 912828Z52 | The www exact-URL CDX query returned an empty JSON result under the fixed filters. The non-www query timed out after30 seconds with no HTTP response body. | No original PDF recovered. The separately pinned provider response still contains Treasury-attributed result text with the matching January27,2020 release date, result heading and CUSIP. Neither the filtered empty result nor the timeout establishes universal archive absence. |

For both recovered files, SHA1 encoded in base32 exactly matches the selected CDX content digest. Their SHA-256 values are:

- YZ7: `115da0980757d996e0f2393b6570bef03aa7fa42e60740e2649d1120cc32b104`
- AV3: `cbf003d71196720a77262c28833c19e60c8726a33469f4f7b0292469fbed96a1`

The CDX capture times provide archival upper-bound evidence, separately from the printed release dates. Conservatively using the UTC capture dates yields bounds of December23,2019 and December10,2020 respectively; mapping those bounds to forecast sessions belongs to the separately tested clock stage. Neither timestamp is asserted to be the exact first-publication time.

## Z52 clock assessment

**The saved Z52 text is defensible as QUALIFIED_REPORTED_CLOCK_ONLY, provided that evidence class and its limitation are explicit.** The exact Treasury result URL is attributed by the provider; the returned body has a release date, competitive-result heading and matching CUSIP. The date claim comes from document-body text, not its filename, AuctionDate field or the search provider's publication-age label. This is consistent with using supported agency-reported dates for exploratory timing without demanding historical cryptographic publication stamps.

Its provenance is weaker than the recovered PDF/CDX evidence: no original binary, byte-level PDF identity check or contemporaneous archive timestamp was recovered. January27 is therefore a **reported release date**, not an independently timestamped upper bound. If the new clock contract requires archival timestamp evidence for every missing-clock event, Z52 remains blocked; it must not silently inherit the stronger evidence class of YZ7/AV3. Under an explicitly qualified reported-date rule, the date can support a missing-value activation clock while its financial quantities and original PDF identity remain unknown. This review recommends that narrow distinction and does not implement it.

## Preservation and scope

New private files are under `data/source_discovery/treasury_auction/clock_recovery_v1/result_archive_v1/`. They include pre-request plans, complete CDX and PDF bodies, response headers, complete curl write-out/stderr/receipts, render receipts, page images and a metadata-only header review. The directory is private; files were created with a restrictive umask.

- `artifact_hashes.json` binds41 evidence files; SHA-256 `d8c3699680a30df62801df64f5c2484de8e59d560776046cce5ac95703c2f204`.
- `header_clock_review.json` records the two header checks and clock distinctions; SHA-256 `f5b3426128883f155e9fdf4e575aa0c315fe96b36ada505f3fd5fad54ad6d5e7`.
- Z52's prior provider response remains at `clock_recovery_v1/result_search_v1/02_official_cusip_date_search.receipt.json`, SHA-256 `1af4ff99ba58bebbd8c9bef4b8c444c1b7e37e5e38c1069e3a4bd4440b3299ea`.

Pre-ceiling auction amounts were incidentally visible on the full source pages and in the earlier cached text. No financial field was converted, compared or used in arithmetic; only header/date/CUSIP/term metadata was transcribed. No model, market array, feature panel or original identity/amount parser was run. Existing404 captures, source dispositions, prior reports and frozen artifacts remain intact. Recovery supplies new evidence for a separate source decision; it does not itself promote identity, amount, first-vintage or forecasting status.
