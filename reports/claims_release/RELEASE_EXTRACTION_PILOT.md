# Original-release extraction pilot

Status: FIVE_RELEASE_EXTRACTION_PILOT_PASSED, 2026-09-08. Exactly five dated DOL releases were acquired and the unchanged, pretested parser extracted the advance seasonally adjusted initial-claims headline and matching reference week from every one. No format failure occurred. This validates these five extraction cases; it is not complete source reconciliation or a predictive experiment.

## Selection fixed before body access

Selections used existing archive anchors and their calendar-cell date/weekday metadata only. The private `data/claims_release/dol_pilot/selection.json` was written before requesting bodies; SHA-256 is `4b131f1904b5b12bcf8eb58df2174a760750a2bea2be04e720eb3bb6707c882d`.

| Actual release | Format and selection reason | Parsed reference week | Advance SA initial claims |
|---|---|---|---:|
| [2009-06-04](https://oui.doleta.gov/press/2009/060409.asp) | HTML; first correct link after the excluded initial archive boundary | 2009-05-30 | 621,000 |
| [2012-01-05](https://oui.doleta.gov/press/2012/010512.asp) | HTML; correct January/Thursday link, with year rollover | 2011-12-31 | 372,000 |
| [2014-12-31](https://oui.doleta.gov/press/2014/123114.pdf) | PDF; final same-year PDF, with explicit Wednesday release metadata | 2014-12-27 | 298,000 |
| [2020-09-03](https://oui.doleta.gov/press/2020/090320.pdf) | PDF; documented seasonal-adjustment transition date | 2020-08-29 | 881,000 |
| [2025-09-25](https://oui.doleta.gov/press/2025/092525.pdf) | PDF; latest actual archive link on or before 2025-10-20 | 2025-09-20 | 218,000 |

Each selected anchor's day and containing calendar-cell month/weekday agree with its dated URL. The 2012 selection uses `/press/2012/010512.asp`, not one of the mismatched-year links. Its parent annual index remains quarantined; accepting this single genuine link does not validate that whole index. Neither source values nor market observations determined selection.

## Transport and extraction evidence

Initial sandbox attempts failed DNS resolution with curl exit 6 and zero body bytes. Their exact headers/receipts remain in `transport_attempt_1/`. The same five selected URLs were then retrieved with system curl through the authorized network path. All five returned HTTP 200 with TLS verification result 0 and unchanged effective URLs. TLS verification was never disabled; no redirect, rolling PDF or other release URL was followed. At most two requests ran concurrently. Successful request timestamps span 2026-09-08T20:53:17.821950Z through 20:53:18.176607Z and are retrieval clocks, not historical availability timestamps.

Every original response body, response-header file and full curl transport receipt is private under `data/claims_release/dol_pilot/`. Receipts retain requested/effective URL, start/completion times, HTTP/TLS outcome, byte count and body/header hashes. Original body SHA-256 values are:

| Release | Body SHA-256 |
|---|---|
| 2009-06-04 | a751d297a68d4163872550aced544955aaa054b35fc106481690cd1c3f38ec6c |
| 2012-01-05 | 645f7c8defb0fa1e6df7f3e55a84dfb5f4b021d4a91149f90199dc78d73283b8 |
| 2014-12-31 | 3e933e634843a744b94f3399eb4125d1d4055abf8c99a287271d07f1a084deef |
| 2020-09-03 | 710992338b31021418cc6f9e775825d821600a2579d6a79cca65eb051b1119a8 |
| 2025-09-25 | a5ee3eb9712e480a09307f00fa2a818ba2b30f119b93bfa4f74d948782244558 |

HTML extraction used BeautifulSoup 4.15.0 with `html.parser`, removal of script/style content and space-separated text extraction. PDF extraction used bundled pypdf 6.10.0 from checked body buffers, processing all 8/12/9 pages respectively with default text extraction and newline joining. Exact extracted text, extractor/version/page metadata and text hashes are retained separately. Both HTML headline passages were manually inspected. The three PDFs' complete first pages were rendered with bundled Poppler and visually inspected; their release headers and headline week/count agree with the parser outputs. Previous-week revisions and insured/continued-claims figures were not substituted for the initial headline.

The parser recognized explicit body release dates in all five cases, agreeing with the supplied dated URL. It correctly handled the 2012 previous-year reference week and 2014 Wednesday release. This is date verification only: the old January 2012 HTML itself labels its clock “EDT”; no historical UTC timestamp or independent ALFRED ingestion clock was inferred from that label.

## Outcome and limits

At pilot completion, the parser and its tests matched their pre-body-access hashes, respectively `8c298b9acab5b0c104e498fab83f3a3649fa82191aaedd7462e441b459b720f9` and `3db2ad43f2cc7e3a077dca7284af92e91e87728f40f71e71af0e45e12588f0fd`. All 336 frozen Python files from the preceding completed experiment were independently checked unchanged. No source/test correction was made for this pilot. A later separately authorized URL-identity extension must retain this completion record and its original hashes.

Private `pilot_results.json` retains every parser result. `pilot_review.json` has status FIVE_RELEASE_EXTRACTION_PILOT_PASSED and binds 45 acquisition/extraction/visual/result artifacts; its SHA-256 is `9f81cc92f27881071cc232242dca3cd5f5c2cad930762006b2a93903f05efdac`. The failed initial transport evidence is included. These files expose the actual five source counts and are source-validation evidence, not invented fixtures or previously unseen-source claims.

No ALFRED observation values or market tables were opened here; no complete-series comparison, feature construction, model fitting or scoring was performed. Root owns the separate ALFRED comparison and broader coverage decision. Complete archival coverage, original-publication authenticity beyond these dated records, source revisions, ingestion timing and missing-release policy remain unestablished by this pilot. The fixed source ceiling remains 2025-10-20.
