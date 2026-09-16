# December 2019 conditional notice: original archive recovery

The missing notice is recovered as an attributable archived original PDF. Its full single page was extracted, read and visually inspected. The bytes match the Internet Archive CDX digest exactly. This is sufficient evidence for a separately tested, explicitly pinned notice-purpose and date-bound adjudication. **Source admission remains false.** No market data, feature construction, auction amount conversion or predictive calculation was performed.

The [exact original Treasury URL](https://www.treasurydirect.gov/instit/annceresult/press/preanre/2019/BPD_SPL_20191219_3.pdf) remains represented by its original failed capture; it was not replaced. The recovered [raw archived snapshot](https://web.archive.org/web/20191219193126id_/https://www.treasurydirect.gov/instit/annceresult/press/preanre/2019/BPD_SPL_20191219_3.pdf) is a separate source artifact.

## Identity, purpose and applicability

The page carries the Treasury News identity and a December 19, 2019 immediate-release header. It describes a conditional possibility that the December 24, 2019 five-year note auction could instead reopen the existing seven-year notes of Series T-2024, CUSIP 9128283P3, originally issued January 2, 2018. It does not report an actual substitution. The body explicitly says actual additional issuance would be identified by the auction results release and a special announcement. It also states the net-long-position reporting scope. No correction to realized dealer allocations, offering amount, auction schedule or result fields is stated. Financial thresholds and amounts remain only in the private original/text, not in this review.

The frozen metadata maps this URL to December 24, 2019 / 912828YY0 / 5-Year Note, announced December 19. That is a date-and-term association: this notice does not print 912828YY0. The warning must be retained alongside separately authenticated final announcement/results identity. It does not itself authorize replacing YY0 with 3P3 or treating a conditional alternative as exhaustive.

## Clock evidence and its limits

| Evidence | Observed value | Meaning |
| --- | --- | --- |
| Original PDF release header | December 19, 2019; immediate release, no clock | Dated publication statement |
| Exact-URL CDX snapshot | 2019-12-19 19:31:26 UTC | Independently recorded archive capture |
| Replay Memento-Datetime | Same timestamp | Replay agrees with selected CDX capture |
| Preserved original HTTP Date | Same timestamp | Original response metadata agrees with capture |
| Preserved original Last-Modified | 2019-12-19 16:18:32 UTC | Supporting server metadata; not first-delivery proof |

The same-day snapshot supports a qualified availability upper bound of **2019-12-19**, before the auction. It does not establish the exact first publication time, exclude an earlier version, prove actual historical investor receipt, or authenticate the later auction result. The timestamp is an archive observation, distinct from the September 2026 retrieval time. A prospective date-only rule may use this independently evidenced bound only through the parent's separately tested adjudication. No frozen unavailable-notice ledger has been altered.

## Transport and byte evidence

System curl used HTTPS-only transport, normal certificate verification, bounded timeouts, and no retries, redirects or insecure flags. CDX and raw replay each returned HTTP 200. The replay is application/pdf, 56,058 bytes, with a PDF signature, unchanged requested/effective URL and zero redirects. Its SHA1 base32 is `BTDU54GMBAUJVYJRDV5XKPKCGLIG2K67`, exactly the CDX digest. The CDX length field is an archive-record length and is not used as the raw-PDF size.

Earlier exact-original and search-tool archive openings did not recover bytes. Sandbox curl initially failed DNS; its receipt is retained and was not treated as authoritative network unavailability. The explicitly authorized escalated read-only curl succeeded. Cached official-URL search text agrees with the recovered purpose but is retained separately and does not substitute for original-byte authentication.

All artifacts are private under `data/source_discovery/treasury_auction/clock_recovery_v1/yy0_notice_v1/`:

| Artifact | SHA256 |
| --- | --- |
| `archive_cdx_escalated.body` | `f503542e7f55ca6742cd3fe4cf6cbd3436dc0b82b49b3b08c738d8f1f0f8d66e` |
| `archived_original.pdf` | `cb4e7092d6ead3104730e633dae71653969a13483bcd39354b9d6763fb7d9a43` |
| `archived_original.headers.txt` | `884998c654513a75f134859c9944bd05b4316673a5ebc99dfb16beafafda9d34` |
| `archived_original.receipt.json` | `68a259709fc61c1fe303dbda69296bf50c3e98125fa7b13b86e3930d00baabb6` |
| `archived_original.txt` | `9931837b8a3aecb05c322e556c55855d99b2a54ed08c69623dc9968394fbe1bb` |
| `archived_original-1.png` | `3589c7ee94bd8fc43af4fba6faa370d7c8972cbd210cbaeb6a1cebd67ceda575` |
| `inspection_manifest.json` | `3342980481277b59d83ec2660206782f1c4a90991e7c443a65cd25545c729638` |
| `recovery_verification.json` | `7f2c253bae7210397e292d758b67b76cc4edeb3ccf18a59511ab9f113cdfbcfe` |

The inspection manifest binds every saved request response, receipt, extracted text and complete-page render. `recovery_verification.json` records the exact digest checks, metadata association, reviewed purpose, limitations and original frozen input hashes. Extraction used bundled pypdf because pdftotext was unavailable; the complete original was rendered at 110 dpi and viewed. No fetched object was executed.

The immediate next step is bounded adjudication against the existing final result identity, preserving both original unavailable-source evidence and this recovered version. Recovery closes the missing notice body/date evidence; it does not by itself admit the historical feature ledger or establish predictive support.
