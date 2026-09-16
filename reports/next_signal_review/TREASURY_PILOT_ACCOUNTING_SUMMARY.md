# Treasury source pilot: paired accounting is usable for further source work

**Twenty of the fixed 22 auctions passed independent PDF/XML amount reconciliation and security-lineage checks. No predictive signal has been tested or admitted in this Treasury stage.** The registered comparison count remains 144, and no new feature, target, forecast or empirical score was produced.

The [accounting audit](TREASURY_PILOT_ACCOUNTING.json) preserves all 22 selected events, all 118 document associations and the 116 original captured documents. Each of the 20 processed auctions has 16 semantically identical source values in the two independently decoded formats: 14 integral par-dollar amounts, the published bid-to-cover ratio and high yield in percentage points. Thus 320 named cross-format comparisons match. Both parsers enforce category and overall accounting, preserve explicit zero, and use public totals excluding SOMA for bid-to-cover. A format match is not an independent historical timestamp.

| Pilot disposition | Auctions | Consequence |
| --- | ---: | --- |
| Paired source amounts and lineage match | 20 | Continue bounded source admission; no model admission yet. |
| Documented June 21, 2019 contingency test | 1 | Retained as an event, excluded from the regular-auction candidate and its same-tenor history. No amount processing in this stage. |
| January 14, 2016 announcement-identity conflict | 1 | Retained as unresolved. Its competitive result identifies 912810RP5, while the announcement PDF prints 912828RP5. No automatic typo repair or amount processing. |

## Original tenor and conditional notices

Original tenor was established for the 20 processed events from matching stated dated/maturity dates, original issue lineage for reopenings, PDF/XML security descriptions, and explicit reopening flags. Exact dated-to-maturity calendar years distinguish original ten-/thirty-year securities from their shortened remaining-term descriptions. Dates moved for settlement do not replace the dated date. The June 12, 2019 note retains ten-year lineage. Nonexact anniversaries remain unsupported; no rounding rule was introduced.

The final documents retain 912828P20 as a new two-year issue on January 26, 2016 and 91282CNX5 as a new five-year issue on August 27, 2025. Their prior conditional warnings do not establish substitution into the alternative original five-/seven-year securities. This conclusion uses final identity and lineage, not a calculation involving the yield condition. A future realized substitution with different announced/final identifiers still requires explicit reconciliation under the strict identity guard.

All seven pilot special URLs now have complete-page documentary review in the [first notice findings](TREASURY_PILOT_NOTICE_FINDINGS.md) and [remaining notice findings](TREASURY_PILOT_REMAINING_NOTICES.md). Two January 2, 2025 URLs contain visually identical schedule notices despite different PDF byte hashes. Their applicability extends beyond the archive's narrow per-URL memberships; original associations and the broader documentary relationships are both retained. The [result identity review](TREASURY_PILOT_RESULT_IDENTITY_REVIEW.md) separately resolves competitive versus tentative noncompetitive roles and same-day event identities.

## June 2010: the original affected figure is still recoverable

The [June 25, 2010 amendment](https://www.treasurydirect.gov/instit/annceresult/press/preanre/2010/BPD_SPL_20100625_1.pdf) expressly corrects SOMA and associated overall totals for the June 22/23/24 auctions, leaving the other result details unchanged. In this stage, the parent independently viewed the complete already-rendered amendment page after its source and image hashes were pinned.

For selected June 22 CUSIP 912828NS5, the amendment's table identifies original and corrected SOMA awards separately. Both independently parsed captured June 22 result formats retain the **original** award for SOMA accepted and tendered. Their accounting reconciles with that original amount. The literal comparison is to the displayed original/corrected cells; it does not synthesize a missing original by subtracting a later correction. The exact values and comparison evidence remain in the private `accounting_pilot_v1/2010_correction_original_match.json` record.

This is evidence that the original affected figure remains recoverable in the captured PDF/XML pair, not a case where only a demonstrated revised result survived. The June 25 correction must still have its own dated ledger entry. Competitive dealer categories are outside the amendment's stated changed-field set. Do not backfill corrected overall totals into the original report, treat all fields as revised, or generalize this finding to every historical correction. The other two amended auctions are retained as notice applicability, without opening their unselected result documents.

Evidence binds to amendment PDF SHA256 `c00fa6a50da70ee9dae1abca28d95f0a71de4f57ffc3412b78b3741be246ce27`, complete-page PNG SHA256 `3258b799453116f93718b0f4e25284d0cd33c98420f40545ce965508b882c3f4`, and the June 22 private accounting file/hash recorded in the public audit. The original accounting artifacts remain unchanged.

## Checks and exposure

All **140 scoped Treasury tests passed** before application. These include prewritten amount contracts, independently authored PDF accounting contracts, date/identity/lineage guards and earlier capture/inventory tests. The independent lineage review found a missing auction-to-issue chronology constraint; its generated failing test and the fix preceded source application. A malformed synthetic fixture mutation was also corrected before application, as documented in the [plan](TREASURY_ACCOUNTING_PLAN.md). Scoped lint and formatting passed for the seven new source/test/runner files. A previously disclosed import-order warning in an older frozen inventory script remains outside this change.

The [prospective checkpoint](TREASURY_ACCOUNTING_CHECKPOINT.json), SHA256 `e5f0bea7cf0364e4dd7b991018c7e8f94f7a7980d9d2e1a1048976f3f65ab3be`, pinned 366 artifacts before numerical application. The run completed successfully once; it does not overwrite or replay existing outputs. Source quantities remain private under the ignored source-discovery directory. The public accounting audit SHA256 is `0ffc9634ecf32a0f4b2863050b23da694c733d03394982a9b10d3e9aab6d194d`.

Financial figures were previously visible in the fixed documentary pilot and some were read as strings for units/layout. This stage numerically decoded the 20 declared eligible document pairs. No market arrays were opened; protected QQQ/SPX observations from November 3, 2025 onward remain unopened. Earlier historical evaluation periods have already been reused and earlier aggregate clean-window reports were read; this does not establish untouched confirmation.

Rechecking the previous commodity-study freeze found zero drift across its 394 code pins, 1,538 input pins, 1,181 preserved artifacts and five prefit artifacts. The 329 prior pilot identity pins also remained unchanged before this checkpoint.

## Work still required before a forecast test

The pilot supports continuing the dealer-allocation source investigation. It does not establish full historical completeness or absence of revisions. The full provisional inventory contains 253 distinct linked nominal special URLs; seven are now reviewed, leaving 246 known URLs beyond this pilot. All relevant known notices and their actual cross-event applicability must be reconciled before a full-history first-release claim.

The full auction result history, offering-size controls and paired announcement quantities, field-specific originals/corrections, original-tenor exceptions and distinct release support still require admission checks. The unresolved 2016 announcement must be handled explicitly; it is not a predictive failure. Only after source admission will a separate prospective experiment fix the dealer-allocation pulse, same-tenor reference history, matched controls, strictly later-session availability, sample support and inference gates. No experiment has been selected or tuned from these pilot financial values.
