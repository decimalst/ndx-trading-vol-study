# Early-session SPX experiment: source feasibility

**The documentary review is complete; the experiment remains source-blocked and unregistered.** Two providers offer plausible routes to the needed historical observations, but neither is admitted. No new empirical result or statistical null follows from this review.

The proposed question is whether information observable at 09:35 New York time improves prediction of risk over the remaining session. It changes the information set after repeated tests of existing daily information failed to establish a qualifying signal. It needs timestamped SPX cash-index observations and a new, strictly subsequent target. Daily highs and lows cannot be separated into before- and after-cutoff components.

## What the review established

| Route | Finding | Remaining issue |
| --- | --- | --- |
| Audited local sources | Daily SPX OHLC and daily realized-risk summaries; no qualifying SPX price path found in the bounded inventory. | Insufficient time resolution. Existing NQ minute bars are a different asset and history. |
| Cboe Main Channel ticks | Advertises disseminated SPX values and an active-symbol archive beginning in 2004. | Confirm SPX-specific completeness, historical clock/vintage semantics and permission for index-derived research. |
| FirstRate SPX bars | Advertises history beginning in 2008 and publishes private-research/model-use terms. | Establish original bar boundaries, cutoff availability, upstream history and correction vintages. |
| Massive and ThetaData | Their advertised index histories start too late. | They cannot supply the unchanged training and development windows. |
| algoseek | A product-specific SPX cash-index archive was not established in the bounded search. | Product identity and coverage remain unconfirmed; this is not proof of unavailability. |

Provider statements were checked on 2026-09-07. Exact supporting URLs, product distinctions and limitations are in [PROVIDERS.md](PROVIDERS.md). The main technical lead is [Cboe Main Channel Tick Data](https://datashop.cboe.com/main-channel-tick-data); [FirstRate SPX](https://firstratedata.com/i/index/SPX) is an alternative whose bars need further timing evidence. Cboe's generic internal-use grant does not settle its separate index-derived-use requirements. [Cboe license, section I.1(e)(viii)](https://datashop.cboe.com/documents/Cboe_LiveVol_DataShop_License_Agreement.pdf).

Historical measurement also matters. S&P changed the constituent-price source for intraday U.S. index calculation in December 2016, inside the unchanged development period. The vendor must document whether its archive preserves the original measurement conventions. The review adds no outcome-based exclusions or new scored subgroups. [S&P effective-date announcement](https://www.spglobal.com/spdji/en/documents/index-news-and-announcements/20161107-changes-to-spdji-us-indices-intraday-calculations.pdf), [timing evidence and implications](INDEX_TIMING.md).

## What must be resolved before a trial

There are three substantive decisions:

1. **Source and clock.** Establish the cash-index identity, complete training history, original dissemination/receipt boundary, bar/price sampling, corrections, session calendars and permitted private use. A timestamp at 09:35 does not alone prove availability by 09:35. A stale pre-cutoff reference can contaminate the claimed future-only range.
2. **Comparable models.** The proposed normalized shock also uses new post-cutoff historical risk. The matched baseline must receive the declared historical information, and the intended claim must be bounded to what the comparison identifies. The [admission contract](ADMISSION_CONTRACT.md) records two unselected prospective designs; neither has been chosen by examining scores.
3. **Independent numerical validation.** Once the documentary requirements and final estimand are resolved, freeze the source plan, model, meaningful synthetic checks, support gates and independent verifier before numerical measurement or fitting. Keep the original phases and protected-period fence. Build controls for the new target; do not reuse probabilities for the older full-session target.

The precise missing provider evidence is now a bounded package: SPX symbol/date and schema inventory; original timestamp, latency and revision definitions; interpretation of historical calculation changes; and terms/quote explicitly covering retained data plus private derived features, labels, models and permitted reporting. No inquiry was sent and no purchase was made. Receiving this package would permit a source-admission decision, not automatically a favorable feasibility or signal result.

## Accounting and scope

This pass added documentation only. It downloaded no market observations, parsed no market rows or protected-period values, constructed no features/targets/support counts, ran no fits, and registered no hypotheses. Numerical tests were not rerun for this documentary change.

The previous completed study remains [issued forecast-error calibration](../issued_calibration/SUMMARY.md), independently verified and publication-audited. Its two contrasts bring the cumulative ledger to **131 hypotheses**. This feasibility pass adds **zero**. A possible future wave20 with the proposed two contrasts would bring that count to 133 only upon registration. No signal has been promoted.

This source route is pending external documentary evidence. That is a limitation of this proposed experiment, not a conclusion that the broader volatility/index search has no remaining directions. No additional statistical test is selected in this report.

Companion evidence: [local inventory](LOCAL_PROVENANCE.md), [provider comparison](PROVIDERS.md), [index timing](INDEX_TIMING.md), [prospective admission contract](ADMISSION_CONTRACT.md). These reports retain source URLs and document locators; public source pages/PDFs have not been archived as byte-pinned local originals. The preservation audit and peer review are separate from empirical validation.
