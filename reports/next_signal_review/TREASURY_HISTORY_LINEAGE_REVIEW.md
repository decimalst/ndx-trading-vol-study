# Treasury history lineage review

All 39 specified first-failing events have been retained with their initial errors and exact source bindings in `TREASURY_HISTORY_LINEAGE_REVIEW.json`. The reviewed date and term labels support two source conventions. No adapter was implemented and no identity result was promoted.

| Observed convention | Events | Initial failure |
| --- | ---: | --- |
| February end-of-month dates crossing February 28/29 | 22 | Exact original-tenor anniversary rejected |
| Confirmed substitution with distinct current and original dated dates | 18 | 17 original-lineage rejections; one masked by the earlier February check |
| Both conventions | 1 | 2016-02-23 / 912828UR9 |
| Distinct reviewed events | 39 | All initial failures preserved |

## February dates are printed source dates

All 22 exact-tenor failures pair the last day of February with the last day of February in the maturity year. The explicit two-, five- or seven-year descriptors and both PDF/XML date labels agree. The strict helper requires identical month/day, so it rejects February 28 versus February 29 even when both are printed month ends.

For example, 2010-02-23 / 912828MQ0 is explicitly a two-year note dated 2010-02-28, issued 2010-03-01 and maturing 2012-02-29. Those separate dates agree across the [original announcement](https://www.treasurydirect.gov/instit/annceresult/press/preanre/2010/A_20100218_1.pdf) and [result](https://www.treasurydirect.gov/instit/annceresult/press/preanre/2010/R_20100223_2.pdf) and their XML counterparts. The 2012, 2016, 2020 and 2024 cases also provide the reverse leap-day direction. No source date should be repaired or shifted.

A prospective adapter can recognize an exact same-month end-of-month anniversary, with the supported integer-year tenor corroborated by the explicit source term. It should preserve ordinary exact anniversaries and reject arbitrary day tolerances, wrong months, wrong tenor counts and contradictory paired dates. This is a source-calendar rule, not a volatility feature or a trading-calendar assumption.

## Reopening sources retain two dated-date meanings

The 17 original-lineage failures all describe confirmed realized substitutions. Their final XML `DatedDate` and PDF `Dated Date` refer to the current offering period, while `OriginalDatedDate` and the printed `Original Issue Date` refer to the older security. The older original issue can precede the current dated date. Requiring these dated-date fields to be equal rejects that explicit layout.

For 2013-09-25 / 912828RH5, the advertised five-year offering was replaced by the confirmed seven-year original security. The final pair retains current dated date 2013-09-30, original dated/issue date 2011-09-30 and maturity 2018-09-30. The [final result](https://www.treasurydirect.gov/instit/annceresult/press/preanre/2013/R_20130925_1.pdf) and the already reviewed realized notice agree on the final identity and lineage. The 2015-09-22 / 912828TS9 case additionally distinguishes original dated date 2012-09-30 from original issue date 2012-10-01; both must remain visible.

All 18 substitutions in this review match the existing canonical confirmations on final CUSIP, original issue date, offered term and series. The canonical transcription audit and its source bindings were authenticated. This review retains that evidence without changing its original `result_documents_bound` status or declaring a new completed reconciliation.

A prospective adapter should retain current offering dated date separately from original security dated date, derive original tenor from the explicit original dated-to-maturity context, and require the bound actual confirmation for a substitution. It must continue to reject absent confirmations, contradictory original dates, wrong identities or series, and invalid original/current issue chronology. A conditional warning remains insufficient.

The 2016-02-23 / 912828UR9 event has both conventions. Its announcement first fails the February 29-to-February 28 anniversary check; its result then has a current dated date of 2016-02-29 and original dated/issue date of 2013-02-28 for the confirmed five-year original security. A generated regression must exercise both layers together.

## Evidence and limits

The saved identity result binds `TREASURY_HISTORY_IDENTITY_CHECKPOINT_V2.json`, SHA-256 `b43e9aeff7e3bb72de6185b114cd58009a5db7319191b3328b6967c8221dc2a5`. All 26,875 checkpoint pins matched before review. The ledger retains 78 PDF date/term label sets with line numbers, 78 XML identity descriptor sets, original records, initial failures and per-event findings. All 261 consumed evidence paths were rechecked before the ledger was written.

No conflict was observed in the reviewed date/term/identity labels. This does not establish that all later checks will pass: no adapter or full identity rerun occurred. The 835 initially reconciled events, 302 requiring review and one excluded contingency test remain the unchanged initial result.

First-failure triage is not exhaustive. Root relayed that the separately reviewed 2016-02-26 P79 and 2017-02-23 W48 cases also print February end-of-month pairs but first failed CUSIP reading order. They are outside these 39 reviewed events and are not counted here.

This review read pinned metadata and relevant retained PDF labels, not complete original-page visuals. It exported no financial amounts or yields, performed no financial decoding, fetched no source, edited no frozen helper, fitted no model and added no predictive comparison. Source-history admission remains false; the cumulative registered comparison count remains 144.
