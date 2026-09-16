# Independent verification of macro plan source evidence

Offline source verification only. No producer modules were imported, no source
bytes were changed, no web calls were made, and no market values, model fits,
forecast scores or event outcomes were inspected. Dates are reconstructed from
the exact saved official-document extracts before comparison with the ledgers.

## FOMC original annual plans: verified

Independent reconstruction agrees with **all 128 original planned meeting
spans in 16 annual documents for 2010–2025**. The separate implementation is
`data/source_discovery/macro_plans/independent_source_verification/verify_sources.py`;
its result and reconstructed evidence are retained in `fomc_verification.json`
in that directory. Eighteen meaningful synthetic assertions passed before source
parsing, including cross-month date ranges, printed-weekday contradictions,
single-day plans, modern prose ranges, explicit next-year carryovers, exclusion
of a revised prior-year block, unknown publication clocks, and strict date-lag
eligibility. PDF line labels with page ranges are also covered for the CPI phase.

Checks agree on each source's exact SHA-256 and byte length, official URL,
publication date, stated publication clock when present, original meeting start
and final date, printed weekdays, and recorded conservative availability bound.
All 16 annual coverage records contain eight plans and cover the selected annual
schedule from January 1 through December 31. Every source predates its covered
year. The ledger, coverage, manifest and producer-file hashes recorded in the
producer audit match; the producer file was hashed only, not imported or run.
All 16 source files remained byte-identical after verification.

The original-plan distinctions survive the reconstruction. The selected 2011
source retains September 20 as a single-day plan. The selected 2012 source retains
July 31, September 12 and December 11 as its original final dates. The document
providing 2013 also contains revised 2012 meetings; those rows are excluded from
the 2013 block and cannot replace the independently selected earlier 2012 plan.
The selected 2020 annual source retains its March 17–18 plan. Next-year January
carryovers are excluded, including the separate 2026 block in the 2025 document.
These checks establish what the selected documents originally planned; they do
not assert that every meeting occurred as planned or reconstruct emergency events.

Six annual sources (2010–2014 and 2016) say “For immediate release” without a
clock. Their exact publication timestamp stays empty. The recorded next-calendar-
day midnight bound is correctly distinguished from an actual publication time.
Ten sources state a clock, which agrees with the ledger. No FOMC statement time
is assigned: all 128 statement-time fields remain empty. Publication clocks are
not used as meeting clocks, and a modern statement-time convention is not
projected backward onto earlier plans.

The source-eligibility policy is **publication date strictly before the previous
market-session date**. Independent synthetic checks reject publication on that
same date even if a stated clock would be earlier than the market close. Annual
coverage is not backfilled before its source publication. This verifier does
not replace the caller's previous-session date with the next realized session
or an exchange-calendar library. Causal integration with the actual feature
builder remains the separate model-verification task.

These are currently rendered partial official-document extracts. Their hashes
identify the saved extraction bytes, not original provider HTML or immutable
historical website vintages. Raw-provider hashes and tool versions remain null.
Independent agreement does not remove that documentary limitation.

## CPI original next-release plans: verified with unknown coverage retained

Independent reconstruction agrees with all **190 queue records** in the final
CPI ledger, SHA-256
`a263e318edf6a6233b44ac8c141a33c9389e7b90e131c0ea9e26678842b5256a`.
The actual index labels cover December 2009 through September 2025 exactly once.
Publication headers determine admission within 2010-01-01 through 2025-10-20;
the boundary document explicitly states 2025-10-24 and remains excluded.

Thirteen CPI synthetic assertions passed before the final raw-source parsing.
They cover multiple documents in a batch snapshot, repeated excerpts, PDF
page-range continuation tags, absent release years, missing zones, past plans,
weekday contradictions, identifiers on media-contact lines, and explicit
this-release reissues versus unrelated routine data revisions. Source headers,
next-plan sentences and identifiers were reconstructed from the exact selected
URL sections; producer evidence snippets were comparison material, not inputs
to the reconstruction. Literal EST/EDT offsets are preserved; generic ET uses
America/New_York. Reference-period years do not supply omitted release years.

| Independently confirmed source status | Records | Admission |
|---|---:|---|
| Explicit original plan without a detected this-release reissue | 168 | Admitted under the source policy |
| Explicit notice that this release was reissued | 17 | Original vintage uncertain; not admitted |
| Next-release year omitted from the printed plan | 4 | Timestamp/month remain unknown; not admitted |
| Publication after the fixed fence | 1 | Excluded |

All four missing-year cases occur in the publications dated August 19,
September 16, October 15 and November 17, 2015. Their reference-period years are
printed, but the next release's date clause omits its year. Those clauses remain
unresolved under the frozen policy. The separately retained September 16 PDF
also omits the year; it does not resolve the ambiguity. The August 2021 document's
USDL identifier is printed on a media-contact line; independent whole-document
identifier extraction agrees with the ledger without inventing an ID.

The 17 reissue flags are independently visible in the selected raw source
sections. Same-day reissues without an original clock remain uncertain, as do
later reissues. Statements that only tables changed do not override the chosen
conservative policy. Their printed dates and next-plan timestamps remain in the
ledger as documentary fields, while `historical_plan_admitted` remains false.
Reissue dates do not replace publication dates or make a historical plan eligible.

The resulting **21 unknown original-plan calendar months** are:

| Year | Months without an admitted, unambiguous original plan |
|---|---|
| 2010 | March–July |
| 2011 | March–April; June–August |
| 2012 | March |
| 2015 | September–December |
| 2019 | March |
| 2021 | February–March |
| 2022 | January |
| 2024 | July–August |

There are no duplicate admitted plan months. Independently reconstructed monthly
counts match the producer's coverage dictionary exactly. January 2010 also lacks
a preceding publication within the source lower bound and is outside the
February 2010–October 2025 coverage span. The original October 2025 plan remains
in the source record despite its later cancellation. None of these coverage
results says an unknown month contained no announcement.

All **198 retained capture files** match their declared byte lengths and SHA-256
hashes. The 359 retained source-URL variants within those files were independently
inspected: 189 have complete explicit timestamp evidence, 164 are partial views
without enough text for a complete reconstruction, and six have missing or
ambiguous evidence. Variant counts are not counts of distinct releases or
admitted plans. No comparable publication or next-plan timestamp contradicts the
selected ledger. Partial views are preserved and do not fill missing years or
override vintage exclusions. The corpus, original ledger and source bytes stayed
unchanged throughout verification.

Independent results are saved under
`data/source_discovery/macro_plans/independent_source_verification/` in
`cpi_verification.json` and `cpi_variant_verification.json`, with their separate
verification scripts. No empirical result was used to resolve a source issue.

## Cross-source reissue consistency check

A complete follow-up inventories all 380 CPI/payroll source records and agrees
with the separate model verifier on 17 CPI and **13 payroll same-release
reissues**, including passive and active wording. The original payroll ledger
had not applied the CPI vintage screen. The pre-fit admission overlay now has
**176 admitted plans, 13 uncertain vintages and one out-of-fence source**.
Thirteen synthetic amendment assertions passed before the completed comparison.
The original ledger, rules and source snapshots remain unchanged; an initially
incomplete 186-plan review is explicitly preserved as superseded. Exact notices,
the complete source inventory, and all preservation checks are retained in
`nfp/vintage_admission_amendment.json`. The final ledger hash and 13 unknown
months are documented in `NFP_SOURCE_AUDIT.md`. No forecasting result informed
this source-consistency correction.
