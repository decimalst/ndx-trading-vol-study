# Bounded Treasury result-clock recovery review

Reviewed 2026-09-08. **The search reproduced dated, Treasury-attributed cached result text for 912828Z52. It recovered no original PDF bytes and established no release-clock bound for 912828YZ7 or 91282CAV3.** The cached Z52 result was already reported by the parent; this independent query confirms reproducibility of that discovery, rather than a new source recovery.

Exactly eight targeted operations were attempted: three filename/month searches, three official-domain CUSIP/date searches, and two exact-URL Wayback CDX opens. No retries, alternate-path guessing, model calls, or further retrievals followed. Existing failed captures and frozen artifacts were not edited.

| Event | Attributable evidence returned | Disposition |
| --- | --- | --- |
| 2019-12-23 / 912828YZ7 | The official-domain query returned the December 19 offering announcement, identifying the scheduled December 23 auction. No dated competitive result was returned. | The announcement establishes the scheduled auction, not the result's actual release date. Clock remains unresolved. [Offering source](https://treasurydirect.gov/instit/annceresult/press/preanre/2019/A_20191219_2.pdf). |
| 2020-01-27 / 912828Z52 | The provider returned indexed full text attributed to the exact dead Treasury result URL. Its body contains “For Immediate Release,” January 27, 2020, “TREASURY AUCTION RESULTS,” and CUSIP 912828Z52. | A concrete dated-content candidate for a release-clock bound. This is a saved search-provider representation, with no recovered PDF, direct HTTP receipt, or original-file hash. It does not independently repair missing financial-source verification. [Attributed result source](https://www.treasurydirect.gov/instit/annceresult/press/preanre/2020/R_20200127_4.pdf). |
| 2020-12-09 / 91282CAV3 | No relevant result was returned by either targeted search. | Clock remains unresolved. An empty search result does not establish absence of public release or absence from every archive. |

The Z52 search also returned its January 27 noncompetitive report and January 23 offering. The noncompetitive document identifies its totals as tentative and distinguishes them from later official results; it cannot substitute for the competitive-result release clock. [Noncompetitive source](https://www.treasurydirect.gov/instit/annceresult/press/preanre/2020/NCR_20200127_3.pdf).

Both CDX attempts—YZ7 and AV3, exact original URLs, HTTP-200 filter and timestamp/original/status/mimetype/digest fields—were rejected by the browsing tool as non-retryable safe-open errors. No CDX rows or Internet Archive HTTP response were obtained. This is a tool-access failure, not evidence that archived copies do not exist. Exact requested URLs and returned errors are retained in the private receipt.

## Reproducible receipts

These are complete tool request/response records, not fabricated network receipts or recovered original documents. The private directory is mode 0700 and the three receipt files are mode 0600.

| Repository-relative receipt | SHA-256 |
| --- | --- |
| `data/source_discovery/treasury_auction/clock_recovery_v1/result_search_v1/01_exact_filename_search.receipt.json` | `db49c556279cdc5b86f22439aa1446051a7e7c7a1df84656d86f3ef76bc353a6` |
| `data/source_discovery/treasury_auction/clock_recovery_v1/result_search_v1/02_official_cusip_date_search.receipt.json` | `1af4ff99ba58bebbd8c9bef4b8c444c1b7e37e5e38c1069e3a4bd4440b3299ea` |
| `data/source_discovery/treasury_auction/clock_recovery_v1/result_search_v1/03_archive_cdx.receipt.json` | `43cbd473123e0d33b41015ef901e4726e6f07ca964c826bb3332e67056681f5c` |

The provider's “Published” or crawl-age labels were not interpreted as original release timestamps. The Z52 date claim comes from the returned document-body text. Search responses incidentally exposed pre-ceiling auction amounts and yields for Z52 and offering documents; these remain in the private raw provider receipt, were not numerically converted or used in any calculation, and are omitted here. No protected market series or later market quotes were accessed.

## Consequence for preparation

No first-vintage source admission or forecast eligibility is asserted. Z52 now has reproducible cached dated-content evidence to assess under the source contract; it still lacks a recovered original PDF. YZ7 and AV3 retain their unbounded release-clock disposition under the strict actual-release design. The bounded search is complete and does not justify replacing release dates with filename or scheduled-auction dates.
