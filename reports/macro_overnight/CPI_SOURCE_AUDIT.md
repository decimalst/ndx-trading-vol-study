# CPI original next-release plan source audit

The bounded source pass is complete. It recovered **189 CPI publication documents dated January 2010 through September 2025**, plus one boundary document excluded because its printed publication date is October 24, 2025. The usable ledger contains **168 explicit original-plan records** under the frozen source rules. Seventeen reissued archives and four incomplete next-date clauses remain unadmitted. No market outcomes, event-return comparisons, model fits, forecasts, or predictive scores were read or produced.

| Final status | Documents | Historical plan admitted |
|---|---:|---|
| `VERIFIED_EXPLICIT_PLAN` | 168 | Yes, only after the printed publication timestamp |
| `ORIGINAL_VINTAGE_UNCERTAIN` | 17 | No |
| `AMBIGUOUS_PRINTED_TIMESTAMP_OR_ID` | 4 | No; all four omit the next-date year |
| `OUTSIDE_PUBLICATION_FENCE` | 1 | No |
| Unresolved access or incomplete capture | 0 | — |

The source handling contract was frozen at 2026-09-07 04:10:09 UTC, before collection. Its SHA256 is `5bb5e932ee311cc67627febda97d01bb86d9e04b3488bd41bb8c23517674ddbd`. Enumeration used actual links from the [official CPI release index](https://www.bls.gov/bls/news-release/cpi.htm), whose titles describe reference months. The saved queue covers December 2009 through August 2025 reference months, then inspects the September 2025 reference-month link solely to establish the publication fence. That boundary document prints October 24, 2025 and is excluded from admission. Its date was read from the header, not inferred from a filename or a realized event calendar.

The predictor represented by this collection is the **next-release plan printed in a preceding monthly publication**, retained even when later canceled or changed. It is not a reconstruction of the latest revised schedule. Each record retains the publication header, exact next-plan wording, document identifier, explicitly stated reference month, source URL, selected source section, retrieval bounds, snapshot path, byte length, SHA256, and admission status. Printed dates and times are interpreted in `America/New_York`; omitted years and timezone contradictions are not repaired. Month coverage becomes known only after the source publication, never retroactively at the start of that month.

Current archives explicitly state that 17 releases were reissued. These have printed publication dates 2010-02-19, 2010-03-18, 2010-04-14, 2010-05-19, 2010-06-17, 2011-02-17, 2011-03-17, 2011-05-13, 2011-06-15, 2011-07-15, 2012-02-17, 2019-02-13, 2021-01-13, 2021-02-10, 2021-12-10, 2024-06-12, and 2024-07-11. The ledger keeps their old headers, printed next plans, and revision notices. An old header on a revised copy cannot certify the original plan or its availability. Same-day revisions are also unadmitted because the original publication-time version is not certified. For example, the [July 2011 PDF](https://www.bls.gov/news.release/archives/cpi_07152011.pdf) explicitly describes a later August reissue.

The [August 19](https://www.bls.gov/news.release/archives/cpi_08192015.htm), [September 16](https://www.bls.gov/news.release/archives/cpi_09162015.htm), [October 15](https://www.bls.gov/news.release/archives/cpi_10152015.htm), and [November 17, 2015](https://www.bls.gov/news.release/archives/cpi_11172015.htm) next-plan clauses omit the release-date year. Their `original_plan_timestamp` and `planned_calendar_month` remain null. A bounded check of the same-publication [September 2015 PDF](https://www.bls.gov/news.release/archives/cpi_09162015.pdf) also found no year in that clause. Both snapshots are retained; no year was inferred from the reference month, weekday, source date, or filename.

Verified original-plan calendar-month coverage comprises 168 distinct months from February 2010 through October 2025. Counts by calendar year are:

| Year | Verified plan months | Year | Verified plan months |
|---|---:|---|---:|
| 2010 | 6 | 2018 | 12 |
| 2011 | 7 | 2019 | 11 |
| 2012 | 11 | 2020 | 12 |
| 2013 | 12 | 2021 | 10 |
| 2014 | 12 | 2022 | 11 |
| 2015 | 8 | 2023 | 12 |
| 2016 | 12 | 2024 | 10 |
| 2017 | 12 | 2025 through October | 10 |

Within January 2010–October 2025, verified coverage is absent for 2010-01, 2010-03 through 2010-07, 2011-03, 2011-04, 2011-06 through 2011-08, 2012-03, 2015-09 through 2015-12, 2019-03, 2021-02, 2021-03, 2022-01, 2024-07, and 2024-08. These months remain unknown; the gaps do not establish a non-event. January 2010 has no preceding publication within the chosen source-publication window. Source publication coverage itself is twelve documents in every year 2010–2024 and nine in 2025 before the cutoff.

All saved evidence is permitted **tool-rendered partial page text**, including rendered text from four official linked PDF fallbacks. It is not raw provider HTML or PDF bytes. Every record has `raw_provider_bytes_sha256=null` and `tool_version=null`, because those bytes and renderer versions were unavailable. Hashes identify the exact saved UTF-8 extraction files only. Current archive text and a printed original header do not prove immutable historical provider bytes, even for admitted records without explicit reissue notices.

The selected records use 190 distinct snapshots, including the two unchanged earlier CPI exemplars. The complete capture manifest pins 198 saved or reused extraction files, including index, resolution, alternative, and unsuccessful partial-extraction evidence. Selected evidence retrieval bounds span 2026-09-07 03:53:52–04:36:17 UTC, including exemplar reuse. One initial link-resolution capture has explicitly documented enclosing time bounds rather than a direct tool-call clock bracket. A transient HTTP429 batch was retained as an access failure, then completed through coordinated sequential requests after cooldown. Three earlier HTML timeouts and one empty HTML rendering used the already linked official PDFs. No direct-fetch workaround or access-control bypass was used.

The local integrity pass completed 1,287 assertions covering selected extraction byte lengths and SHA256 hashes, terminal newlines, null raw-provider hashes and tool versions, required timestamp/identifier/capture fields, causal ordering for admitted plans, and exact admission-status consistency. Both prior exemplar hashes still match. Independent source verification is separate from this producer audit and is recorded in the companion verification report when complete.

Machine-readable deliverables are ignored local research artifacts under `data/source_discovery/macro_plans/cpi/`: `ledger.json`, `final_index_queue.json`, `capture_manifest.json`, `source_audit.json`, the frozen `contract.json`, exact captures, and capture metadata. The final ledger SHA256 is `a263e318edf6a6233b44ac8c141a33c9389e7b90e131c0ea9e26678842b5256a`; the local parser SHA256 is `f4f6f7bead6ccf9e4353c2a4e8176e7b86b374043178fa373952f95a6dfd512e`. Only `VERIFIED_EXPLICIT_PLAN` records are eligible for downstream plan flags. The later experiment must assess training support with these unknown months intact and preserve its predeclared minimum sample; this source audit makes no predictive claim.
