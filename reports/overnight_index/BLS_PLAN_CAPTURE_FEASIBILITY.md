# Capturing four official BLS advance-plan examples

Captured 2026-09-07 03:53:52–03:53:53 UTC, corresponding to the evening of
2026-09-06 in New York. This bounded task accessed only the four previously
verified BLS exemplar URLs. It did not reconstruct a calendar, read market
outcomes, fit models, compare event returns, alter existing calendars, or
change earlier reports.

**A reproducible local evidence snapshot is feasible through the permitted
web tool. Raw provider HTML/PDF bytes remain unavailable.** Four saved text
extractions now contain the original publication headers and next-release
sentences. Their exact stored UTF-8 bytes are hashed, and the timestamps were
checked independently. The hashes identify these saved tool-rendered extracts;
they do not identify original BLS response bytes or a historical web vintage.

## What was captured

The ignored directory
[`data/source_discovery/bls_plan_capture/`](/Users/byrons/code/trading-vol/ndx-vol-experiment/data/source_discovery/bls_plan_capture)
contains four `*.web-extract.txt` files and
[`capture_manifest.json`](/Users/byrons/code/trading-vol/ndx-vol-experiment/data/source_discovery/bls_plan_capture/capture_manifest.json).
The manifest records source URL, capture start/end UTC, source kind, byte size,
SHA-256, publication and planned timestamps, original page-line references,
the two evidence sentences, checks, and the request/method metadata available
from this tool session. These files are ignored by git; the source text has
not been published or added to tracked calendars.

Each capture used `web.run` with two operations on the same official URL:
open the page at line 170 and find `scheduled to be released`, with
`response_length=long`. The returned value was a text string containing
tool-rendered page excerpts and tool metadata. The string was stored verbatim
as UTF-8 with its original single terminal newline. No manual reconstruction
of a source sentence was used in the snapshot. This is a **partial page text
extraction**, not a complete page or raw network response. Nearby text supplied
by the tool remains in the snapshot; only schedule fields were extracted into
the ledger.

The public callable name is `web__run`. Tool implementation and renderer
versions are not exposed, so the manifest records `tool_version: null` and
explains the limitation rather than inventing a version. The request arguments
and exact returned text are retained. Retrieval timestamps are from the clock
tool, not inferred from BLS page dates.

| Extract | Bytes | SHA-256 of exact stored extraction bytes |
| --- | ---: | --- |
| `cpi_12152015.web-extract.txt` | 28,023 | `a29d4c780e626d86167de0584172b9e84c730cf2b91815372ebdba54dea40c00` |
| `empsit_12042015.web-extract.txt` | 28,868 | `6bacb96bc54d0242fe5a4c34a98a0b3c3b52d6216305e92a167240eddb2b5454` |
| `cpi_09112025.web-extract.txt` | 26,158 | `5c5880834bf10c136631c68dfd023f8276ee29e795ad49d4caed95d0646e521c` |
| `empsit_09052025.web-extract.txt` | 29,241 | `6483aaebdd8a4e320e5ca433ceaea8a2a9dcca9fb08c90cab87f71f16b1c863e` |

`raw_provider_bytes_sha256` is null for every record. These extraction hashes
replace no source-byte hash: the canonical ledger-record hashes in the earlier
feasibility report are a third, different object and are not reused here.
Earlier direct requests received HTTP 403. This capture task made no further
direct-fetch attempts, changed no access controls, and created no account; it
used the explicitly permitted web-tool text-snapshot route.

## Extracted original plans and independent spotchecks

The publication header states an embargo expiry time, which is used as the
document's stated public-release time. It is not a measurement of actual
website response latency. All four statements specify 08:30. `EST` maps to
UTC−05:00 for the December/January examples; the September/October dates use
UTC−04:00 under `America/New_York`.

| Event / document ID | Publication timestamp | Original next-release timestamp | Official source |
| --- | --- | --- | --- |
| CPI / USDL-15-2390 | 2015-12-15T08:30:00-05:00 | 2016-01-20T08:30:00-05:00 | [2015-12-15 CPI report](https://www.bls.gov/news.release/archives/cpi_12152015.htm) |
| Payroll / USDL-15-2292 | 2015-12-04T08:30:00-05:00 | 2016-01-08T08:30:00-05:00 | [2015-12-04 Employment Situation](https://www.bls.gov/news.release/archives/empsit_12042015.htm) |
| CPI / USDL-25-1356 | 2025-09-11T08:30:00-04:00 | 2025-10-15T08:30:00-04:00 | [2025-09-11 CPI report](https://www.bls.gov/news.release/archives/cpi_09112025.htm) |
| Payroll / USDL-25-1344 | 2025-09-05T08:30:00-04:00 | 2025-10-03T08:30:00-04:00 | [2025-09-05 Employment Situation](https://www.bls.gov/news.release/archives/empsit_09052025.htm) |

Local checks read the actual saved bytes, locate the publication and plan
lines, verify both contain the corresponding date and 08:30 time, match the
release identifier, and require publication before planned release. All four
records pass. A separate reviewer independently opened the four official URLs
and confirmed every publication/plan pair and timezone label without examining
market outcomes. This independent check did not use the parsed ledger as its
source of truth.

The independent reviewer also checked all four saved byte lengths and SHA-256
hashes against the manifest, verified exactly one terminal newline in each,
and matched the saved evidence sentences to that separate browser spotcheck.
All four integrity checks passed. The manifest records the independent review
and its source-vintage limits.

The evidence supports a fixed policy: **the next-release plan printed in the
preceding monthly publication**. The two 2025 records deliberately retain the
original October dates. Later cancellation or rescheduling must not alter
these predictors, and these plans must not be labelled actual release dates
or the latest information about releases. Plans become eligible only after
their document's publication time and the registered prior-close cutoff.
Missing statements are not inferred. Emergency FOMC decisions are outside this
BLS ledger and remain excluded from the separate scheduled timing control.

## What this establishes and what remains

Reproducibility here means another reviewer can load these exact saved bytes,
verify their hashes, and reproduce the timestamp extraction. It does **not**
mean a new browser call must return the same bytes: citation identifiers,
navigation, page excerpts, renderer versions, and currently served archive
content can change. Saving an extract today also does not prove those exact
bytes existed at the historical publication time. It provides documentary
evidence of a preannouncement inside an official archived release, with a
clearly stated capture limitation.

These four examples establish a workable acquisition path for source evidence;
they do not establish complete monthly coverage, parser generality, enough
training history, or predictive performance. Before a larger study, freeze the
fixed-plan policy and source-handling rules, then obtain the required bounded
release documents and independently verify missing/ambiguous statements.
Every record must preserve its source extraction and source kind. A raw PDF or
HTML capture, if subsequently available through permitted access, should be
stored as an additional distinct artifact with its own provider-byte hash;
existing extraction snapshots must not be silently replaced.

No experiment was run, and no empirical result was used to choose this policy.
