# Treasury pilot: three special-notice findings

Reviewed 2026-09-08. **The June 2019 entry is a documented live contingency test; the June 2010 notice is an actual correction limited to SOMA and overall totals; the January 2016 notice describes a conditional reopening before an auction. These require different ledger dispositions.** All three complete one-page images were viewed with the image tool. This report makes documentary classifications only: no candidate feature, numerical support count, model, score or market outcome was calculated.

The original PDFs were already captured by the parent. Their current byte hashes were independently checked against [TREASURY_DOCUMENT_CAPTURE.json](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/next_signal_review/TREASURY_DOCUMENT_CAPTURE.json), whose SHA256 at review was `a1d954d4b8ffa9b840cef478e9a409db547abc94de2dbb3a76d37ac7c9c4f6b3`. The supplied [render manifest](/Users/byrons/code/trading-vol/ndx-vol-experiment/data/source_discovery/treasury_auction/pilot_visual_v1/render_manifest.json) binds the image paths to those originals. No new download occurred, no original capture or existing report was changed, and no parser was edited or rerun.

| Notice | Supported release date | Documentary disposition | Implication |
| --- | --- | --- | --- |
| June 2019 contingency operation | 2019-06-20, visible immediate-release header | Pre-event terms notice for a live contingency infrastructure test on June 21 | Retain the event, exclude it from the proposed regular-auction universe; it is not a correction of the June 12 auction merely because the CUSIP repeats. |
| June 2010 amended auction results | 2010-06-25, visible immediate-release header | Correction of SOMA awards and associated overall tendered/accepted totals for three named auctions | Record a real correction and its affected fields. The notice expressly leaves other result details unchanged; it does not establish revision of competitive dealer allocations. |
| January 2016 possible reopening | 2016-01-21, visible immediate-release header | Conditional pre-auction security-identity notice | Final auction results and any follow-up notice must establish whether the condition occurred. This page alone neither changes the event's original tenor nor supplies allocations. |

These are supported dates of the notices, not asserted intraday delivery timestamps, not automatically the first release dates of the underlying result figures, and not cryptographic proof of historical immutability. All three notice dates are inside the 2025-10-20 source ceiling.

## 2019: live contingency operation, not a second ordinary allocation observation

Primary URL: [June 20, 2019 special notice](https://www.treasurydirect.gov/instit/annceresult/press/preanre/2019/BPD_SPL_20190620_4.pdf).

Original PDF SHA256: `f55544bfc9f0e03b59f8108de0b139614476afa2d8c383bc0e3d186fd4e12630`.

Viewed complete page: [p03-1.png](/Users/byrons/code/trading-vol/ndx-vol-experiment/data/source_discovery/treasury_auction/pilot_visual_v1/p03-1.png), SHA256 `df3296efdda031af621c4addefe212a56f93cf97c175111b10a24ee758fc6312`.

The immediate-release block at the upper left visibly says June 20, 2019. The operation date in the body is June 21. The latter must not replace the former because a text extractor concatenates the contact block with the release label or encounters the body date more cleanly. This visually resolves this notice's date; it does not silently repair or certify the automated parser.

The heading identifies a live small-value contingency auction operation for the 9-year 11-month note, CUSIP **9128286T2**. The opening paragraph says Treasury had announced its intent on May 1 and explains that it regularly tests its contingency auction infrastructure. Thus June 20 is this specific terms notice's date, not a claim that it was the first public mention of a planned operation.

The terms require telephone bidding through the Federal Reserve Bank of New York, allow only designated primary dealers to submit bids, prohibit customer bids and noncompetitive tenders including FIMA, and separately permit SOMA. The final paragraph describes a forthcoming offering announcement at 9:30 a.m. Eastern on June 21; that scheduled announcement time is not the June 20 notice's release time and is not a result publication time.

This is an actual live operation used to test infrastructure, not an imaginary record or a demonstrated duplicate/republication of June 12. The repeated CUSIP is therefore insufficient for deduplication. The terms directly explain why ordinary noncompetitive participation should not be assumed, although this notice alone does not prove why a particular archive link is absent.

For the proposed regular nominal-auction study, disposition should be **documented contingency operation outside the regular-auction universe**, with the row and all documents retained. Its restricted participant rules would make any allocation comparison fundamentally different from an ordinary auction with the usual competitive bidder categories; no allocation share was calculated here. It should not become an observation in the regular-auction feature or its same-tenor reference history. The notice provides eligibility and event-purpose evidence, not accepted dealer amounts, and it supplies no basis for assigning the June 12 result figures to June 21. The June 12 event's own identity and result availability still require its own documents.

## 2010: a genuine correction with an explicit field boundary

Primary URL: [June 25, 2010 amended auction results](https://www.treasurydirect.gov/instit/annceresult/press/preanre/2010/BPD_SPL_20100625_1.pdf).

Original PDF SHA256: `c00fa6a50da70ee9dae1abca28d95f0a71de4f57ffc3412b78b3741be246ce27`.

Viewed complete page: [p06-1.png](/Users/byrons/code/trading-vol/ndx-vol-experiment/data/source_discovery/treasury_auction/pilot_visual_v1/p06-1.png), SHA256 `3258b799453116f93718b0f4e25284d0cd33c98420f40545ce965508b882c3f4`.

The page is explicitly titled **Amended Treasury Auction Results**, with an immediate-release date of June 25, 2010. It says a clerical error understated awards to the Federal Reserve's System Open Market Account. Its table identifies the affected events:

| Auction date | Security description | CUSIP |
| --- | --- | --- |
| 2010-06-22 | 2-year note | 912828NS5 |
| 2010-06-23 | 5-year note | 912828NL0 |
| 2010-06-24 | 7-year note | 912828NK2 |

The first is the selected pilot membership. The other two are explicit applicability evidence in the same selected notice; their result documents were not opened for this review. A full notice ledger must retain all three relationships, even when the pilot selected the notice through only one auction.

The table separately prints original and corrected SOMA awards. Those amounts were visible during the requested page inspection, but were neither transcribed into a numerical dataset nor used for arithmetic. The paragraph below the table states that SOMA tendered, SOMA accepted, total tendered and total accepted increased accordingly, and that all other result details remain unchanged.

Disposition should be **dated result correction: SOMA and associated overall totals**. This is concrete evidence of an auction-result correction, refining the earlier metadata-only source-clock review, which had not inspected a correction example. That older report remains preserved as the record of its narrower evidence scope.

The correction does not reclassify these auctions as tests or nonregular events. It also does not by itself make the proposed dealer competitive share revised: the intended numerator and competitive denominator exclude SOMA, and Treasury explicitly says other details are unchanged. The documentary evidence supports treating the correction at the field level, subject to checking the original/current result representations and any other applicable notices. It would be incorrect either to ignore a genuine amendment or to declare all required dealer-allocation fields unrecoverable solely because a SOMA correction exists.

For first-release reconstruction, retain the June 25 correction clock separately. If any input or accounting check uses an affected overall total, its original version and applicable date must be handled explicitly; do not mix corrected SOMA with an original overall total, and do not backdate corrected totals. The stated unchanged-field boundary can support first-release continuity of unaffected fields when their original result identity and release date are established. This review did not compare those result amounts, verify any accounting identity, or authenticate the initial dealer-allocation publication date. The correction date should not automatically become the availability date of unchanged competitive allocations.

## 2016: conditional reopening warning, not an observed outcome

Primary URL: [January 21, 2016 possible unscheduled reopening notice](https://www.treasurydirect.gov/instit/annceresult/press/preanre/2016/BPD_SPL_20160121_1.pdf).

Original PDF SHA256: `5d00d0da915c447215214e794190e9752a37b79e5822159b4bdb42fe44540666`.

Viewed complete page: [p07-1.png](/Users/byrons/code/trading-vol/ndx-vol-experiment/data/source_discovery/treasury_auction/pilot_visual_v1/p07-1.png), SHA256 `af6d757de16039f63f42689c1a6cca23386898061bc2b2d2b31d393b81bb003f`.

The immediate-release header says January 21, 2016. The body concerns the scheduled January 26 two-year note auction. It says that if the auction's high yield falls in a specified range, the notes would instead be an additional issuance of an outstanding five-year security: Series U-2018, CUSIP **912828UJ7**, originally issued January 31, 2013. That 2013 date is a security-lineage descriptor, not the notice's release date.

Treasury states that an actual additional issuance of the outstanding security would be identified in the auction-results press release and a special announcement. The notice therefore supplies its own confirmation route. Its wording is conditional; neither the headline nor the range establishes that the condition was met. No auction yield was inspected or compared with the range in this review.

Disposition should be **pre-auction conditional terms/identity notice**. It is not a correction to a January 26 result that had already happened, and it contains no first dealer-allocation amounts. The archive's selected event has CUSIP **912828P20**; the different outstanding-security CUSIP in the notice is the stated conditional alternative, so its appearance is not by itself a conflicting document identity.

Keep the selected event pending final identity confirmation from its own results and any follow-up notice. If the condition was not realized, the provisional two-year description may remain appropriate once authenticated. If it was realized, original five-year lineage and reopening status would matter despite the offering's two-year description. Do not use remaining maturity, the pre-auction title, or the word “unscheduled” alone to assign the original-tenor group or discard the event. A scheduled auction with a conditional security substitution is not automatically the same event class as the 2019 contingency test. Final eligibility must apply the already declared regular-auction and original-tenor rules to the authenticated realized event.

The warning's January 21 date makes these terms known before the auction; it does not make final security identity or allocations available before results. Any later document required to establish the initial event identity/categories retains its own supported release clock. This review makes no finding about the separate 2016 result-layout conflict being examined by the parent.

## Limits and handoff

These findings resolve the purpose and visible release dates of three specific notices, and identify a real correction's stated scope. They do not certify the entire 22-record pilot, complete PDF/XML agreement, the full known-special-notice set, original result recovery, numerical field mappings or historical support. No notice here has a release date beyond the ceiling. Current hashes pin the inspected evidence; they do not provide an independent historical timestamp.

The parent can retain three distinct documentary dispositions: excluded contingency event; regular-auction result correction with a limited affected-field set; and conditional terms notice requiring final event identity. No parser success should be claimed for the visually resolved June 2019 date until the implementation owner preserves the failure and tests any required extraction change. No experiment was registered and no comparison count changed in this review.
