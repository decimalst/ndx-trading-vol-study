# DOL annual archive metadata parser

Prospective contract written 2026-09-08 before implementation and before annual-index acquisition for this component. Existing source-timing review supplied HTML href patterns only. This component authenticates dated link metadata, not release contents or numerical observations.

`parse_archive_index(payload: bytes, year: int, expected_sha256: str, ceiling='2025-10-20') -> dict` verifies the exact immutable input buffer hash before HTML decoding. It requires a four-digit year in 1900–2099, the exact fixed ceiling 2025-10-20, an HTML document, and at least one dated release link. Every release path must match both the requested year and its filename year. It accepts the observed `/press/YYYY/MMDDYY.asp` and `.pdf` patterns and validates actual calendar dates without imposing a weekday. The request year is checked against every dated path, including excluded later links.

The HTML parser reads anchor href metadata only and never follows a link. Latin-1 decoding preserves ASCII paths in both legacy and UTF-8 HTML; non-ASCII prose does not enter records. The final release identity must use the exact official host and an ASCII path. Relative links resolve against the official archive endpoint; HTTP and HTTPS identities normalize to HTTPS. Release-shaped foreign, credentialed, port-qualified, query-bearing, fragment-bearing, encoded or unsupported-extension links fail. Unrelated navigation, text and script contents do not create records. A base href or ambiguous duplicate href attributes fails rather than changing resolution silently.

Equivalent links deduplicate by canonical URL. Different valid paths for one date, including `.asp` versus `.pdf`, fail even after the ceiling. Output records sort by date. Dates equal to the ceiling remain eligible as link metadata; later links are separately recorded with `reason='after_source_ceiling'`. No record certifies an authenticated report value or usable market-origin state.

Return fields are `status='METADATA_ONLY_NOT_RELEASE_RECONCILED'`, `year`, `source_ceiling`, `source_sha256`, `records`, `excluded_postcutoff`, and `duplicate_links_removed`. Ordinary rows contain exactly `release_date` and `release_url`; excluded rows add `reason`.

Generated prewritten tests exercise exact schema, hash-before-decode ordering, typed arguments, legacy ASP links, weekday exceptions, cutoff equality and exclusion, duplicate versus competing links, leap/invalid dates, requested/path/filename year mismatch, unsafe release identities, navigation/script exclusion, invalid documents, legacy text and ambiguous link resolution. RED must be captured before creating the implementation; GREEN and focused lint precede acquisition.

After GREEN, acquire only the 2009–2025 annual index HTML via system curl with verified TLS, HTTPS-only transport and the documented POST fields `report=press`, `year=YYYY`, `submit=Submit` at `https://oui.doleta.gov/unemploy/archive.asp`. Use at most two concurrent requests. Require successful HTTP status and an HTML content type before parsing, save exact payload hashes and transport receipts under private `data/claims_release/dol_indexes`, and write a metadata-only combined ledger. No release href is requested by this component. Later release fetchers must enforce the ceiling again before transport.

No prior code, protocol or report is edited. No features, model support, fits or scores are computed. This prospective component uses Python 3.11 in the repository `.venv`; branch and remote were inspected as `main` and the configured SSH origin. The existing mixed worktree is preserved.

After the initial 17 tests passed, a single 2009 index HTML transport confirmed the archive's explicit year header is `<b>2009</b>`; it is the page's only bold element. Before further acquisition, three additional generated header tests require exactly one bold four-digit year header equal to the requested year. Missing, mismatched, conflicting and duplicate year headers fail even when href years are valid. This metadata-based strengthening changes no numerical parser or source values. The extra RED and final GREEN are retained separately.

## Completed bounded acquisition

Final generated result: **20 tests passed**, with focused Ruff clear. The preimplementation RED, subsequent header-contract RED and final GREEN are `ARCHIVE_RED.log`, `ARCHIVE_HEADER_RED.log` and `ARCHIVE_GREEN.log`. Acquisition requested all 17 annual HTML indexes for 2009–2025, with at most two concurrent system-curl requests, normal TLS verification and no redirects or release-body requests.

The strict parser accepted 15 annual indexes and rejected the 2012 and 2019 pages because their explicit headers and path years conflict with three filename years. Those failures are retained; no year was dropped for a model and no filename was corrected.

| Requested/header year | Exact anchor | Calendar cell metadata |
| --- | --- | --- |
| 2012 | `<a href="/press/2012/010313.asp">3</a>` | January, `headers="January_Tuesday"`; the following Thursday cell also has its own January 5 link. |
| 2012 | `<a href="/press/2012/080113.asp">1</a>` | August, `headers="August_Wednesday"`; the following Thursday cell separately links `/press/2012/080212.asp`. |
| 2019 | `<a href="/press/2019/010318.pdf">3</a>` | January, `headers="January_Thursday"`. |

`ARCHIVE_ANCHOR_CONTEXT.log` preserves each exact source cell and surrounding HTML with byte offsets and raw-file hashes. These are discrepancies for authoritative identity resolution, not instructions to infer the intended URL.

Private, ignored output: `data/claims_release/dol_indexes/index_ledger.json`, SHA-256 `6ca298807623aac53da3c273c358ec341e0972158132cf17ec1b58901a67e423`, status **INCOMPLETE_ARCHIVE_METADATA**. It lists all requested years and both failed years explicitly, with 770 retained date-link records from valid indexes and 7 later links separately excluded. These counts describe link metadata, not admitted observations, support or a predictive result. Each valid index has `{year}.html`, `.headers`, `.receipt.json` and `.metadata.json` files.

The two rejected payloads and headers remain under `data/claims_release/dol_indexes/quarantine/`, with `2012.receipt.json` and `2019.receipt.json`. Their detailed curl JSON was not retained because parsing raised before normal receipt publication. The receipts disclose this and retain the exact response headers, payload hashes and failed checks. The transport routine had already checked HTTP 200, successful TLS verification, exact endpoint and HTML type before invoking the parser; no unavailable response field was reconstructed.

All annual jobs had been queued before the first parser failure. The executor completed the other HTML requests while propagating that error; every response was subsequently inspected. No further network request followed the discovered defects. The combined ledger remains explicitly incomplete and is not a complete first-release source admission. Root will handle any independent resolution using authoritative metadata.
