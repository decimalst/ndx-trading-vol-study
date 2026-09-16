# Payroll original next-release plans: source audit

Status: **bounded source collection complete; corrected admission is 176 plans**.
This is documentary source work only. No market outcomes, model fits, macro
surprises, forecast scores, or protected market data were examined.

## Pre-fit admission correction

The consumer input is now `data/source_discovery/macro_plans/nfp/ledger_verified.json`,
SHA-256 `608384be1b94c0e0977251efd1c8bfafe8b7249171d49629c6148254e4c92b39`.
It preserves all 190 records: **176 admitted, 13 original-vintage-uncertain,
one outside the publication fence**. The original ledger, source rules and all
snapshots remain byte-identical. The original collection counts below describe
that preserved documentary ledger; they are superseded for model admission.

The complete source-only review found explicit reissue notices in the releases
dated January 8, 2010; February 6, 2015; December 6, 2019; January 10, February 7,
March 6, April 3, May 8, June 5, July 2, August 7 and September 4, 2020; and
May 2, 2025. Passive and active wording both count. The same conservative rule
used for CPI excludes these selected documents even when a notice says the text
or another format was unaffected. It does not substitute a different format or
reissue timestamp. All original publication and plan fields remain preserved.

Unknown original-plan months are **February 2010; March 2015; January–October
2020; and June 2025**. They are unknown, not non-events. January 2010 remains
outside the source lower bound. Admitted sources now begin February 5, 2010,
and admitted plans begin March 5, 2010; the final admitted plan remains October
3, 2025. There are no duplicate admitted plan months.

`vintage_admission_amendment.json` retains the exact numbered notices, hashes,
all 380 CPI/payroll source classifications, and preservation checks. Thirteen
synthetic assertions passed before the completed source comparison. The two
independent full inventories agree on all 380 queue identifiers, supplied
source hashes, 17 CPI reissues and 13 payroll reissues. The initially incomplete
186-plan review is retained under `*.intermediate_186_SUPERSEDED.json` and must
not be used. This is a pre-fit documentary correction; no performance result
informed it. The separate model verifier checks final consumer admission.

## Frozen handling and selection

Source rules were written before release collection in
`data/source_discovery/macro_plans/nfp/source_handling_rules.json`, SHA-256
`404dabf2ea5fb0d4503f7d73997b21cfa682f8f735dd58694ab54f5b9be0d925`.
The [official BLS Employment Situation archive index](https://www.bls.gov/bls/news-release/empsit.htm)
was saved as exact web-tool extraction text. Its SHA-256 is
`d036c85dbc3de40f4d521e9e05be171304348ec1f1e9f76843db35af3812cb94`.

The queue contains 190 actual HTML links, in ascending reference-month order,
from the December 2009 report through the September 2025 report. The first
reference-month entry is necessary to capture a source published in January
2010. Final admission depends on the document's explicitly printed publication
date falling within **2010-01-01 through 2025-10-20**, inclusive. URL filenames
were not invented, and actual announcement dates were not used to select earlier
documents. The September 2025 reference-month candidate explicitly states
publication on 2025-11-20. Its captured evidence is preserved as
`OUTSIDE_PUBLICATION_FENCE` and is not an admitted plan source.

The next plan printed inside each preceding publication is retained, including
later-cancelled plans. No actual-event calendar, later revision, or market outcome
was used to replace it. A missing original plan is unknown; it is never silently
encoded as a non-event. Publication time is the stated embargo expiry, not a
measurement of website latency. Source-time eligibility belongs to the separately
registered downstream parser and forecast-origin contract.

## Evidence capture and machine records

Successful new captures use `web__run` clicks on the official index's actual
link identifiers, with `response_length=long`. The exact returned string is
stored as UTF-8 without manual rewriting. Each capture has its source URL,
request, start/end UTC time, byte length, SHA-256, and source-kind metadata.
These are **currently rendered partial official-document text extracts**.
`raw_provider_bytes_sha256=null` and `tool_version=null` throughout; a snapshot
hash does not identify original BLS HTML/PDF bytes or a historical web vintage.
Nearby source text remains in the saved extraction but macro values are not
extracted into the ledger.

The two earlier payroll exemplars are reused by their original immutable paths
and hashes under `data/source_discovery/bls_plan_capture/`. Their files and old
capture manifest were not modified. No direct provider requests, BLS 403 bypass,
accounts, or external messages were used.

The source-specific extractor passed 12 synthetic assertions before processing
the captured documents. It requires explicit publication/plan dates, times and
timezone labels, requires the plan to follow the publication, rejects missing
years rather than inventing them, and preserves reference-month names without
inventing reference years. Generic ET uses `America/New_York`; explicit EST/EDT
uses the stated offset. Exact snapshot bytes and source line numbers remain
available for the independent common parser.

Original collection artifacts (preserved; use the admission overlay above):

- `data/source_discovery/macro_plans/nfp/ledger.json`: 190 queue records,
  parsed source fields, evidence/line references, hashes, and monthly counts.
- `data/source_discovery/macro_plans/nfp/coverage.json`: count-only source
  coverage, admitted date bounds, and explicit boundary limitations.
- `data/source_discovery/macro_plans/nfp/captures/`: additive extraction snapshots
  and per-record capture metadata; reused records point to the older exemplars.
- `data/source_discovery/macro_plans/nfp/capture_failures_initial.json`: exact
  initially rejected queue identifiers and rate-limit status; no source text
  is claimed for a failed request. `capture_retry_completion.json` records their
  successful single retries after the root-coordinated pause.
- `data/source_discovery/macro_plans/nfp/ledger.initial.json`: the preserved
  intermediate ledger from that pause; it is not the completed consumer input.
- `data/source_discovery/macro_plans/nfp/extract_ledger.py`: source-only parser
  and its synthetic assertions; it does not fetch sources or read market data.

## Original collection counts and capture history

There are **189 usable explicit original-plan records**: 187 newly captured
release extracts and two reused exemplars. All 189 contain a stated publication
timestamp, original planned timestamp, and document identifier. Zero captured
admitted records have missing/ambiguous plan statements. All snapshot hashes
and byte lengths match, source URLs are unique, and stated publication timestamps
increase in the queue order. The admitted snapshots total 5,317,034 bytes.
Including the single excluded publication, the 190 release snapshots total
5,345,287 bytes. The exact final ledger SHA-256 is
`b64110b8cf4951c0adf7ceb3f444e7970e069eaf06f7bb59fcd72b68ced0b248`.

The shared web tool initially rejected 36 requests with HTTP 429 / Rate limit
exceeded. Collection stopped during a coordinated pause. After the root task's
single successful probe, each pending request was retried exactly once,
sequentially with at least 1.1 seconds between requests and no overlapping web
calls. All 36 retries succeeded; **zero requests remain pending**. The initial
failure record and intermediate ledger remain unchanged for provenance. A rate
limit was not classified as a missing BLS document or missing calendar event.

Coverage below counts original-plan calendar months, not realized release
months. There is exactly one admitted original plan for every calendar month
from February 2010 through October 2025, with no internal missing or duplicate
months. The JSON retains every individual calendar-month count. A month's plan
becomes known only when its source document was published; complete retrospective
coverage does not authorize backfilling that knowledge before publication.

| Original-plan year | Captured distinct calendar months |
|---|---:|
| 2010 | 11 |
| 2011 | 12 |
| 2012 | 12 |
| 2013 | 12 |
| 2014 | 12 |
| 2015 | 12 |
| 2016 | 12 |
| 2017 | 12 |
| 2018 | 12 |
| 2019 | 12 |
| 2020 | 12 |
| 2021 | 12 |
| 2022 | 12 |
| 2023 | 12 |
| 2024 | 12 |
| 2025 | 10 |

Admitted sources run from 2010-01-08 through 2025-09-05; their original plans
run from 2010-02-05 through 2025-10-03. The source-publication lower bound cannot
establish the January 2010 original plan from a preceding 2009 document. The
post-fence November 2025 publication is retained only as excluded evidence.
Every admitted plan states the release year explicitly; none explicitly states
the planned reference month's year, so that separate field remains null.

The independent model verifier identified two documentary details that are
retained without correction. The December 7, 2012 publication header literally
prints **EDT**; its recorded `08:30-04:00` follows the frozen explicit-zone rule,
even though that label differs from New York's civil winter offset. The October
22, 2013 source states November 8 as its next plan and then mentions a former
November 1 plan. The ledger retains the first explicit scheduled-next-release
clause, not the later historical mention. These source peculiarities do not
relax the separate rule that publication date must precede the previous market-
session date. No source timestamp or raw evidence was corrected after review.

The original October 2013 and October 2025 payroll plans remain present exactly
as preannounced; cancellation/rescheduling did not replace them. Source retrieval
is complete for the fixed queue and publication fence. Remaining limitations
are documentary: current tool-rendered archive extracts are not historical
network vintages, exact provider bytes remain unavailable, and the January 2010
plan is outside the declared source window. This is not a validation of a
forecast predictor. The parent task controls independent verification, the
common parser, timing contract, and registration before empirical testing.
