# Auction history reconciled for the prospective dealer-demand experiment

The complete source pass now reconciles 1,132 auctions across all required bidder allocations, accounting totals, high yield, bid-to-cover and offering size. The final paired identity pass reconciles 1,133 of the 1,138 selected events. No Treasury feature panel, market fit or forecast score has been produced; the predictive comparison family remains 144, with no new qualifying signal.

| Completed check | Result |
| --- | ---: |
| Selected auction events retained in their original order | 1,138 |
| Final paired PDF/XML identities | 1,133 |
| Confirmed substitutions reconciled without rewriting original identities | 18 |
| Reviewed amended announcements reconciled with separate original/revised dates | 4 |
| Complete bidder-amount and offering matches | 1,132 |
| Exact paired result-field comparisons | 18,112 |
| Offering-source values agreeing under the fixed scale | 3,396 |
| Scoped prewritten tests passed before final source application | 297 |
| Accounting parser failures or cross-source amount disagreements | 0 |

Both independent saved-output audits completed with zero discrepancies. The identity audit verified all 1,138 events, all 18 substitutions and four amendments, and preservation of every earlier match. The accounting audit independently decoded 18,112 raw XML result fields and 3,396 offering tokens, checked all 1,132 private outputs, and retained the twenty prior pilot comparisons. Its sixteen PDF result fields were checked as saved outputs against independently decoded XML values; it did not claim another independent PDF-table parse. The original production PDF and XML parsers are separate implementations.

Five events retain their prior identity dispositions: three unavailable competitive-result PDFs, the January 2016 announcement's printed CUSIP conflict, and the documented June 2019 contingency test. A sixth event, the December 24, 2019 auction, has matching paired identities but an unavailable required special notice; its quantities were not applied. These exceptions remain explicit records and must not become zeros or disappear from the preceding-auction reference pool.

## What changed

Separately tested adapters now recognize the observed announcement coupon labels, one visually confirmed extracted-word spacing artifact, the four column-order layouts, genuine amended headings, February month-end anniversaries and distinct current/original dates for confirmed substitutions. Exact source dates and identity lexemes remain intact. Independent review found and fixed missing filename-date validation, ignored conflicting CUSIP punctuation forms, and a missing required prior-result pin before the relevant source run.

The first adapted identity pass reached 1,115 matches and exposed the next check on 18 substitutions. Their result OriginalCUSIP explicitly equals their original announced CUSIP, while the final security differs. A separate tested adapter requires that exact relationship, explicit AnnouncedCUSIP and the full dated final confirmation. The canonical offering mix is nine two-year, eight five-year and one three-year note; the existing canonical records required no correction.

The amount pass independently parses sixteen fields from each competitive result PDF and XML, then checks three explicit offering amounts. XML offering scale remains a fixed one-billion multiplier checked against the PDF's stated dollar amount for each event. It is not inferred from accepted totals or selected dynamically. Source quantities and partial-stage records are private under `data/source_discovery/treasury_auction/history_accounting_v1/`.

## Next step and limits

Build the complete release-availability ledger and preceding-twelve-auction history masks, retaining the six explicit dispositions. Then freeze the single proposed dealer-share surprise and matched market/auction controls, its next-five-session QQQ variance target, support rules and serial-dependence inference before fitting. The new feature must improve on both the matched controls and the established market baseline under the cumulative comparison adjustment. No threshold, tenor subset, horizon or extra model search has been chosen from these source amounts.

Dated original content, paired representations and known notice review support qualified exploratory research. Current capture hashes establish reproducibility, not historical cryptographic delivery or intraday publication timing. The first permissible origin remains strictly after the latest required supported release date. Source reconciliation does not establish predictive usefulness, and no new signal is promoted.

## Saved evidence

- `TREASURY_HISTORY_IDENTITY_CHECKPOINT_V3.json` and `TREASURY_HISTORY_IDENTITY_V2.json` preserve the 1,115-match intermediate pass.
- `TREASURY_HISTORY_ORIGINAL_CUSIP_REVIEW.json` binds the eighteen observed identity relationships to existing source evidence.
- `TREASURY_HISTORY_IDENTITY_CHECKPOINT_V4.json` and `TREASURY_HISTORY_IDENTITY_V3.json` preserve the final 1,133-match identity pass.
- `TREASURY_HISTORY_ADAPTED_PRECHECK.json` and its log record 297 passing tests.
- `TREASURY_HISTORY_ACCOUNTING_CHECKPOINT.json` and `TREASURY_HISTORY_ACCOUNTING.json` bind the complete 1,132-event quantity comparison.
- `TREASURY_HISTORY_IDENTITY_V3_AUDIT.json` and `TREASURY_HISTORY_ACCOUNTING_AUDIT.json` record the independent saved-output checks and their exact scope.
- `TREASURY_HISTORY_RECONCILED_TERMINAL_AUDIT.json` pins this completed stage and preserves earlier studies.

The preceding goal turn completed a terminal checkpoint; this turn implemented, tested and applied the observed source rules and completed the full quantity reconciliation. Both constitute progress. The research goal remains active.
