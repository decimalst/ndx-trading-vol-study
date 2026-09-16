# Treasury history term-label exception review

All 252 `Unrecognized PDF offered term` events were preserved. Their announcement labels include a coupon token between the remaining term and `Note`/`Bond`, a layout absent from the frozen parser's grammar. The result PDF labels omit that token. Both XML records agree on remaining term, security type and reopening status for every case; their selected identity metadata also agree.

| Remaining term and type | Events |
| --- | ---: |
| 9-Year 10-Month Note | 64 |
| 9-Year 11-Month Note | 62 |
| 29-Year 10-Month Bond | 63 |
| 29-Year 11-Month Bond | 63 |

There are 251 ordinary coupon-label cases and one additional extraction anomaly. The observed coupon syntax is an integer-percent token or a mixed-fraction-percent token; the review does not disclose or convert coupon values. Exact original-line hashes, redacted layouts and occurrence counts are retained in the companion JSON, together with all 252 initial error records, original PDF/text hashes and XML bindings.

The additional anomaly is auction 2016-01-13, CUSIP `912828M56`, in [announcement A_20160107_5.pdf](https://www.treasurydirect.gov/instit/annceresult/press/preanre/2016/A_20160107_5.pdf). Its retained term text contains `10-Mont h`. The original PDF SHA-256 is `98eb25fe51b700ea15a2bee2bc143a3aa59204ba1c85bd256da7fb3620130b43`; extracted text SHA-256 is `e3d9b1f698de754b38fc7c13bc3a3a318a3e17a52655bc09cff40d4ac59dc96e`. Both XMLs and the result label specify the same remaining term. A complete-page render at 110 dpi was visually inspected: the original clearly prints `10-Month`, with the reopening label on the following line. The review records the unchanged extracted `Mont h` and printed `Month` separately. This is a confirmed extraction-spacing artifact. A prospective adapter can recognize that exact additional unit-token spelling inside the term field; no raw-text rewrite or general spelling repair is needed. The retained private render manifest is `data/source_discovery/treasury_auction/history_term_layout_visual_v1/render_manifest.json` (SHA-256 `7c131248fca018b680b541a297eac95928d7df8f5ca2c862d19c999b9ff731d9`).

A future separate adapter should recognize only the bounded coupon-token position and syntax, keep the original text intact, and preserve all own-security, date, reopening, original-tenor and XML term/type checks. Generated regressions must precede implementation and a new checkpoint. These are term-layout candidates; a later reconciliation may reveal other identity or lineage issues.

Before label inspection, all 26,875 identity-checkpoint V2 pins were verified. The metadata-to-inspection-checkpoint link was checked. This review inspected 504 exact PDF term lines and compared 504 pinned XML metadata records; reviewed source/report pins were checked again before publication. The one complete original page was rendered and viewed solely to resolve the printed term. No financial quantity was converted or transcribed, no source rerun, no code changed and no history admitted. All 252 observed term layouts are now classified; their original reconciliation failures remain unchanged.

Companion review JSON SHA-256: `c1187307ad014144d09b2f3de7582f504298258a4d803816888bbb75b3ec7bf5`.
