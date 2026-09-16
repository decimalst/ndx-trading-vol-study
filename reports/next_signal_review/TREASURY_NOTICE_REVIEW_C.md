# Treasury known-notice review C

**All 82 assigned notices were read completely and all 94 selected archive associations are preserved with documentary dispositions. No source history is admitted.** The [ledger](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/next_signal_review/TREASURY_NOTICE_LEDGER_C.json) has SHA256 `313e63d983eafeb88b5dcef014964b86a3c0b3ce56067c179a226f0398103ec9` and binds checkpoint `d63de42754c25a1999f83b31b43f776d4d0eff1658f36b9831a67e3616d2e8a2`.

The 2,042 checkpoint pins were verified before reading and again after writing the ledger. Every assigned original-body/text hash, release date and ordered membership list agrees with the pinned source index. Each alternative or realized CUSIP, series and original issue date recorded manually was additionally checked against its own literal source text. This was verification of documentary identity, not financial decoding.

| Individually read notice purpose | Documents |
| --- | ---: |
| Bid closing times | 27 |
| Conditional security identity / reopening terms | 50 |
| Auction rescheduling after a technical issue | 1 |
| Actual realized-reopening confirmation | 3 |
| Coupon-auction and TIPS-buyback schedule changes | 1 |

No allocation-result correction or test/contingency event is stated in these 82 notices. This is a finding about this assignment, not the other reviewers' documents or unlinked history. Original amounts and yield ranges are omitted from the public ledger. Every document was read individually; template grouping was not used to infer its classification.

## Concrete findings

Three notices explicitly confirm realized original-seven-year reopenings:

| Dated notice / offered auction | Explicit realized security | Original issue date |
| --- | --- | --- |
| [April 25, 2017, two-year offering](https://www.treasurydirect.gov/instit/annceresult/press/preanre/2017/BPD_SPL_20170425_1.pdf) | 912828ST8, Series K-2019, original seven-year | 2012-04-30 |
| [March 27, 2019, five-year offering](https://www.treasurydirect.gov/instit/annceresult/press/preanre/2019/BPD_SPL_20190327_1.pdf) | 912828W71, Series J-2024, original seven-year | 2017-03-31 |
| [April 23, 2019, two-year offering](https://www.treasurydirect.gov/instit/annceresult/press/preanre/2019/BPD_SPL_20190423_1.pdf) | 912828WG1, Series K-2021, original seven-year | 2014-04-30 |

The April 20, 2017 warning had described a different possible original-five-year alternative, 912828D23. The later April 25 confirmation names original-seven-year 912828ST8 instead. Conditional warnings therefore cannot be assumed to enumerate every possible realized identity. Neither notice was silently rewritten, and no yield trigger was calculated. These are dated identity confirmations relative to the offering, not statements that previously released dealer amounts were wrong. Final result/category documents still need to be bound to the confirmed securities and their original availability dates.

The January 22, 2015 warning names a January 28 five-year auction while its current archive association is January 29 / 912828H52. The separately authorized [January 26 rescheduling notice](https://www.treasurydirect.gov/instit/annceresult/press/preanre/2015/BPD_SPL_20150126_1.pdf), also read in full with its hashes checked, explicitly moves that CUSIP from January 28 to January 29 because of weather. The ledger preserves both dated states and the cross-reference, including its broader bill/FRN/note scope. This is an explained schedule change, not an unresolved source error.

The February 25, 2016 notice postpones that day's seven-year auction close to February 26 because of a technical issue. It expressly preserves settlement and other announcement terms, and says already submitted bids stand while allowing updates until the new close. The notice changes schedule and bid-update opportunity; it reports no corrected allocation results.

Body applicability often extends beyond the selected API memberships. The ledger retains every explicitly named bill and FRN schedule event, the January 9, 2025 TIPS buyback and the January 8 coupon auction covered by the January 2 notice despite its narrower selected links. Date/term/security-type matches to pinned projected metadata are labelled as metadata-derived CUSIPs. The August 26, 2020 FRN reference uses the body's two-year group while metadata records a remaining one-year eleven-month term; both descriptions are preserved. Original issue dates for conditional securities are not mislabelled as auction dates.

## Visual checks and limits

The original January 22, 2015 page and all three realized-reopening pages were each rendered once at 110 dpi and viewed completely. All four renderer processes exited 0, using the exact returned execution handle until terminal. The [private render manifest](/Users/byrons/code/trading-vol/ndx-vol-experiment/data/source_discovery/treasury_auction/full_notice_review_C_v1/render_manifest.json), SHA256 `33ea656cc37caf7956e3741317c8f8acd098dade2836a250523687c6a2b49b14`, records original hashes, all-page outputs and individual render logs. Previously rendered pilot pages remain unchanged; their full saved texts were reread in this assignment.

The ledger records precise final-result binding obligations for conditional/realized identity notices. These are remaining source-reconciliation tasks, not claims that originals are missing or that an experiment failed. Other reviewers' dated confirmations may resolve further conditions during consolidation. No release header remains unresolved in this set, and the rescheduling evidence explains its apparent auction-date discrepancy.

Only this report, its JSON ledger and the authorized private render directory were written. No network request, new document acquisition, market-array access, financial-value calculation, feature construction, support calculation, model fit or forecast scoring occurred. All prior pins remain unchanged. Registered comparisons remain 144; completion of this reading assignment does not admit numerical history or certify completeness of unlinked notices.
