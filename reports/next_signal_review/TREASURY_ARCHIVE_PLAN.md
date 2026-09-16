# Treasury auction archive metadata contract

Recorded before acquiring annual API response bodies. This is source-route inspection, not source-value admission or a predictive trial. No hypothesis count is incremented.

## Discovery evidence

The official archive page advertises historical announcement/results releases from July 1998 for non-TIPS securities:
https://www.treasurydirect.gov/auctions/announcements-data-results/announcement-results-press-releases/

Its linked public client code specifies the annual archive request and filename mapping:
https://www.treasurydirect.gov/scripts/auctions-section/annceresult.js

The grid is a current Treasury database query, not a static historical annual index. This distinction matters: current inventory links alone do not prove original publication or exhaustive historical correction coverage.

The query route is `/TA_WS/securities/search`, with `startDate`, `endDate`, `compact=true`, `dateFieldName=auctionDate` and JSON output. Use separate calendar-year requests from January 1, 2010 through October 20, 2025 inclusive. No request extends beyond that ceiling. Omit the UI's `pdfFilenameAnnouncement=notNull` filter so missing announcement links remain visible rather than silently disappearing. Request all security types for inventory completeness; no eligibility filter is applied during acquisition.

## Prewritten metadata projection

Before real captures, synthetic contracts must pass for `src.treasury_auction_archive.project_archive`. Read UTF-8 JSON bytes with numeric tokens represented by a tagged string type, without converting numerical source fields. Reject duplicate JSON keys and nonstandard nonfinite constants. Accept a list or the single-key `securityList` wrapper; reject JSONP and other envelopes.

Date-preflight every record before projecting any document link. Require a real ISO date or midnight ISO datetime; auction dates must lie within the exact requested interval, itself within 2010-01-01 through 2025-10-20. Announcement dates may precede the source floor but cannot follow their auction. The six identity/date fields must be actual strings. A future or malformed row fails the entire response rather than being discarded. The result contains only CUSIP, security term/type, term bucket, dates, and document URLs. All other source fields stay quarantined and uninspected, including numerical values which the compact response may supply.

The official mapping is `a` CUSIP, `d` security term, `z3a` security type, `t3a` term bucket, `h` announcement date, `i` auction date; `e3` announcement PDF, `f3` competitive result PDF, `f31` noncompetitive result PDF, `f32` special-announcement PDF(s), `e4` announcement XML, and `ia1` competitive result XML. Missing document fields are retained as empty lists. Special-announcement filenames split only on comma-space. Filenames must be simple safe PDF/XML basenames; reject paths, queries, traversal, and external hosts. Announcement PDFs use the announcement year; result/special PDFs use the auction year; XML filenames use the official `/xml/` prefix. Do not invent missing filenames.

Reject repeated `(auction date, CUSIP)` identities, including identical duplicates. Reopenings may reuse a CUSIP on a different date. Sort the metadata records by auction date and CUSIP. This does not infer original term, release date, correction status, or numerical eligibility from filename/term labels.

## Acquisition and evidence

Use the system certificate-verified HTTP client, with finite timeouts and no disabled TLS checks. Python's initial discovery request could not validate the local certificate chain; the system client succeeded with its trust store. Save raw response bytes, response headers, requested/effective URLs, HTTP status, retrieval time, response byte count and SHA256 under the already ignored `data/source_discovery/treasury_auction/`. Do not print raw response bodies. HTTP or parsing failure stays visible; do not overwrite or silently retry an existing captured attempt.

Pin the discovery page/client bytes and the tested projector, tests, and acquisition script before requesting the annual responses. Each response must have HTTP 200, an unchanged official URL, and an acceptable JSON content type before projection. Record metadata counts and every missing/special document link, without reading linked auction documents yet. A second independent metadata reconstruction must check the retained identities/dates/links and hashes before relying on the inventory.

The inventory only establishes what the official endpoint currently returns. Its completeness, the original documents' integrity, correction coverage, exact numerical schema, and publication clock require further checks. No numerical auction history, feature, target panel, estimator, or predictive score is authorized by this metadata checkpoint alone.
