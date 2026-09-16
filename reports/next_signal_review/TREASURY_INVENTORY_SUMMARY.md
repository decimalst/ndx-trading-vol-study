# Treasury auction source inventory

**The official document route is usable; no numerical auction source or predictive signal is admitted yet.**

The bounded metadata inventory covers **5,347 records** from 2010 through October 20, 2025 and exposes **27,139 unique document URLs**. All 16 annual responses passed independent metadata reconstruction. No announcement or competitive-result PDF link is absent in these returned records; linked document availability and original-release integrity have not yet been tested.

The six proposed nominal coupon display groups contain **1,138 provisional records**. They include 235 rows with special-announcement links, corresponding to 253 distinct linked special PDFs. These are current inventory classifications, not admitted original-tenor or original-vintage labels.

| Year | All records | Provisional coupon rows | Coupon rows with special links | All rows with special links |
| --- | ---: | ---: | ---: | ---: |
| 2010 | 301 | 72 | 3 | 49 |
| 2011 | 269 | 72 | 1 | 8 |
| 2012 | 264 | 72 | 4 | 9 |
| 2013 | 267 | 72 | 6 | 15 |
| 2014 | 270 | 72 | 9 | 13 |
| 2015 | 272 | 72 | 24 | 27 |
| 2016 | 266 | 72 | 14 | 18 |
| 2017 | 277 | 72 | 30 | 51 |
| 2018 | 284 | 72 | 10 | 49 |
| 2019 | 325 | 73 | 28 | 49 |
| 2020 | 503 | 72 | 23 | 94 |
| 2021 | 445 | 72 | 16 | 68 |
| 2022 | 384 | 72 | 18 | 48 |
| 2023 | 428 | 72 | 17 | 87 |
| 2024 | 440 | 72 | 15 | 76 |
| 2025 | 352 | 57 | 17 | 65 |

## The next source questions

The 2019 inventory has 73 provisional coupon rows rather than 72. Two 10-year-group entries share CUSIP 9128286T2 in June: June 12 and June 21, with a special notice linked to the latter. Both are retained. The original notices must establish whether the extra entry is a regular auction, a test, or another event; the inventory alone does not justify excluding it.

Reopenings display a shortened remaining term. For example, the January 13, 2010 entry has a 9-Year 10-Month security label in a 10-Year display group. Actual original-term identity must come from dated notices and CUSIP lineage.

A fixed **22-record documentary pilot** is saved in `TREASURY_DOCUMENT_PILOT_SELECTION.json`: first observations in every proposed term group in 2010, 2016 and 2025; the earliest/latest linked nominal special notice; both members of the anomalous 2019 group; and the earliest shortened-term label. Selection used metadata only. No pilot document body has been acquired.

The full known-special-notice review remains necessary; a pilot cannot establish absence of corrections for the remaining history. Current index dates and file names are routing clues, not independently authenticated publication times. Dated primary content and resolved known corrections can support qualified exploratory use without claiming cryptographic immutability.

## Verification and preservation

All **64 generated source-contract and integration tests passed before the successful inventory capture**. This is a targeted source-audit suite, not a new full-repository test run or a predictive experiment. The initial strict capture stopped because cash-management bills have an empty display grouping. That attempt remains preserved; the amended inventory retains the blank label without assigning a term, and reused the original 2010 response bytes. The remaining 15 annual responses were newly captured with certificate verification.

The final scoped style check reported one import-order diagnostic in the new capture wrapper. It has no effect on the generated checks or source reconstruction; the checkpointed wrapper was retained unchanged. No claim that every lint check passed is made.

Both capture jobs are terminal. The 394 prior frozen Python files, all original study inputs/preserved artifacts, the initial metadata checkpoint and its pinned files, and the successful inventory checkpoint are checked for preservation in `TREASURY_INVENTORY_AUDIT.json`.

**Registered comparisons remain 144. No new features, market outcome arrays, model fits, forecast scores, or acceptance gates were created or inspected by this inventory.** All nonmetadata API fields remain quarantined without numerical conversion.

Official route: [Treasury archive](https://www.treasurydirect.gov/auctions/announcements-data-results/announcement-results-press-releases/) and its [public archive client](https://www.treasurydirect.gov/scripts/auctions-section/annceresult.js).
