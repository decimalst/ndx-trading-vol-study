# Treasury notice review B

**All 83 assigned complete saved texts were individually read and classified, preserving all 103 archive associations.** The [ledger](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/next_signal_review/TREASURY_NOTICE_LEDGER_B.json) records each document's identity hashes, release date, purposes, actual body scope, metadata-mapping basis, association disposition and remaining requirements. Source admission remains false and the comparison family remains 144.

The review checkpoint SHA256 is `d63de42754c25a1999f83b31b43f776d4d0eff1658f36b9831a67e3616d2e8a2`. All 2,042 checkpoint pins were independently verified before document reading and again before ledger construction. Each assigned PDF/text binding was checked. The assignment manifest hash is `ffe7c7d63ddcb586c4e6d5cf50e9e605648f7c00bed8529b0b98a779aa798f09`; the complete text index hash is `e6dfcb9f403286805b45970125b502726068ef437012ccb00a02d5ee9d2f94a8`.

The documents comprise 40 conditional substitution notices, six notices explicitly confirming realized reopenings, and 37 schedule/payment notices. None of these 83 states an allocation correction, a test/contingency event, or a first allocation addendum. That finding does not cover the other reviewers' documents. Template fingerprints did not substitute for reading any assigned text.

| Explicit realized result notice | Offered term → stated original term | Realized CUSIP / original issue date |
|---|---|---|
| 2013-08-28 | 5-Year → 7-Year | `912828RE2` / 2011-08-31 |
| 2016-02-23 | 2-Year → 5-Year | `912828UR9` / 2013-02-28 |
| 2017-11-27 | 5-Year → 7-Year | `912828M80` / 2015-11-30 |
| 2019-03-26 | 2-Year → 7-Year | `912828C57` / 2014-03-31 |
| 2019-05-28 | 5-Year → 7-Year | `912828XT2` / 2017-05-31 |
| 2019-11-05 | 3-Year → 10-Year | `912828TY6` / 2012-11-15 |

These are direct statements of realized additional issuance, not outcomes inferred by evaluating yield conditions. Pair them with final competitive-result identity and original lineage before admitting source quantities. Conversely, each conditional notice remains a warning until supported final identity or a realized-outcome notice establishes the outcome. Equality between its alternative CUSIP and an archive CUSIP alone is not sufficient.

The January 22, 2015 notice names a January 27 two-year auction, while the archive records January 28 / `912828H78`. I viewed its complete original page and independently read the parent's identified, pinned January 26 rescheduling notice. The latter expressly moves that named CUSIP from January 27 to January 28 and retains settlement dates. Both dated stages are preserved in the ledger; this is an explained schedule amendment, not an unresolved source-date contradiction. The cross-reference's original/text hashes are recorded, and no new source was fetched.

The January 24, 2019 notice names only the January 28 two-year note and thirteen-week bill. Complete-page visual review confirms that the additionally linked five-year event `9128285Z9` is not named. Its API association remains in the ledger with an explicit limitation: this body does not establish a five-year closing time. No claim that the auction itself is invalid follows from that limited scope discrepancy.

Two shuffled-layout texts also received complete-page visual review. The November 5, 2020 page supplies a Veterans Day schedule covering notes, bonds and multiple bill terms, including no auctions on November 11. The March 29, 2022 page confirms April 18 settlement for nominal coupon auctions announced April 7 while principal/interest payments scheduled for April 15 remain on that date. Individual April auction dates/CUSIPs come from metadata matching the announcement date and stated terms, not from the notice body. Both headers state an 11:00 A.M. embargo without an explicit header timezone; preserve date evidence without inventing a historical intraday delivery timestamp.

The four complete page images and render logs are retained under the private `data/source_discovery/treasury_auction/full_notice_review_B_v1/` directory, with hashes in the ledger. The first rendering emitted font-cache warnings but its full page was readable and checked; the later three rendered without warnings using a private font cache. No existing capture or report was changed.

Body scopes explicitly retain bills, FRNs, broad auction/announcement schedules, payment dates, and alternative security references outside the selected Note/Bond memberships. Financial amounts, coupon values and yield ranges are omitted from the public ledger. No financial calculation, market array, source cohort, model, forecast or score was produced. Documentary classification and final source-history admission remain separate.
