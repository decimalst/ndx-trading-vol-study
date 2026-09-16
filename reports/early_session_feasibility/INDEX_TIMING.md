# SPX index timing and measurement evidence

Documentary review dated 2026-09-07. No market observations were downloaded or parsed. This is source feasibility, not a registered experiment or a signal result.

## Verified primary-source facts

S&P's November 7, 2016 announcement makes the intraday change effective **December 5, 2016**: real-time U.S. equity indices, including SPX, switch from primary-market constituent trades to consolidated-tape trades. Official closing values and special opening quotations retain primary-exchange pricing. The earlier March announcement gave a planned quarter, not the final effective date. [Effective-date announcement, page 1](https://www.spglobal.com/spdji/en/documents/index-news-and-announcements/20161107-changes-to-spdji-us-indices-intraday-calculations.pdf), [earlier announcement](https://press.spglobal.com/2016-03-01-S-P-Dow-Jones-Indices-Announces-Changes-to-U-S-Indices-Intraday-Calculations).

The August 2026 Equity Indices Policies & Practices describes fixed-interval intraday calculations using the latest constituent prices, with older prices used when updates are unavailable. Its example interval is not an SPX-specific historical cadence guarantee. Official closing calculations can be revised; the policy says disrupted or erroneous intraday histories are not recomputed. Its change log also identifies an index-level auto-hold policy change effective after the November 17, 2023 close. These are current policy and retrospective change-log statements, not a certification of any vendor's archive. [Policies, pages 45–46, 50–51 and 62](https://www.spglobal.com/spdji/en/documents/methodologies/methodology-sp-equity-indices-policies-practices.pdf).

The July 2026 U.S. Indices Methodology separately says real-time indices are not restated and identifies SPX / .SPX as S&P 500 vendor symbols. Neither statement establishes historical delivery times or a downstream vendor's correction practice. [U.S. methodology, pages 23–24](https://www.spglobal.com/spdji/en/documents/methodologies/methodology-sp-us-indices.pdf).

## Implications for the proposed experiment

These are research-design inferences, not additional claims by S&P:

- The unchanged development window straddles the December 2016 measurement change, while training history can use the earlier convention. Source admission must establish whether historical files preserve the contemporaneous stream or reconstruct it. The dates above do not authorize dropping periods or adding scored subgroups after observing outcomes.
- A fresh timestamp on an index message does not imply simultaneous fresh trades for all constituents. The proposed shock measures an observed published index move; it cannot automatically be interpreted as an efficient-price innovation or a tradable opening execution.
- Keep publisher event time, vendor receipt time, bar interval, archive version and the forecast cutoff distinct. A policy against publisher intraday restatement does not prove that a vendor never fills gaps, rebuilds bars or corrects its own archive.
- The prior official close used in the predictor needs an availability/version rule. A subsequently corrected close must not be silently supplied to an earlier forecast.
- A closing index value and the last intraday message need not be the same observation or become available together. Define the target endpoint and label maturity from documentary evidence before constructing any target. A next-day fit must wait for the required endpoint/version to become available.
- A stale starting value can include a pre-cutoff gap in an alleged post-cutoff range measure. Exact starting-value age and the intended event-time or feed-state target remain admission decisions; missing cutoff information cannot be repaired with a later print.

The [admission contract](ADMISSION_CONTRACT.md) records the outstanding choices. None has been resolved by inspecting prices. Historical measurement changes are an additional provenance requirement, not evidence that the signal exists or does not exist.

## Evidence limits

The documents above were read through public web retrieval. This review retains URLs, document dates and page locators; it does not claim locally archived or byte-pinned copies of the publishers' PDFs. General policy documents are insufficient to admit a particular data product. A vendor must supply product-specific history, field definitions, revision treatment and appropriate research rights before acquisition and numerical validation.
