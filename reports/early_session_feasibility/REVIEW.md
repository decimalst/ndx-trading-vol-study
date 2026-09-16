# Documentary peer review of early-session feasibility

**Review completed 2026-09-07. No remaining material reporting defect was found after the FirstRate license clarification below. No provider is admitted, and no empirical validation is claimed.** The source-blocked, unregistered conclusion is consistent across the five reviewed documents.

This was a read-only review of the other authors' reports and their consistency with the admission contract. The reviewer authored that contract; this report is not a second independent assessment of its own authorship. Bounded primary-document rereads checked the central timing, historical-coverage and license claims. No market data, new sample/support counts, features, targets, model results or protected-period values were inspected or computed. No empirical tests were run. The local inventory's documented scope was assessed without repeating its file/schema inspection.

## Reviewed versions

The following SHA256 values identify the exact local report versions reviewed, not the external publishers' PDF/page bytes.

| Document | SHA256 |
| --- | --- |
| [Summary](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/early_session_feasibility/SUMMARY.md) | `8bea8e0e8936c78e019ff0e322d78f1ecb7c287309b665c23942edefa5f0a10c` |
| [Local provenance](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/early_session_feasibility/LOCAL_PROVENANCE.md) | `5f303a3be7e825b6f3d8ffe7bf156c0aea6d05ec6a56ba082d5578b410e28783` |
| [Index timing](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/early_session_feasibility/INDEX_TIMING.md) | `1dd68639b769137f3088cf294efd3e594cc73e3334873cc07be324ff3b6bc5cd` |
| [Providers](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/early_session_feasibility/PROVIDERS.md) | `7dfea7c2f0af1354e10be2337a4559ba7c158b18d4a0909f5698fe55d1a33ecf` |
| [Admission contract](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/early_session_feasibility/ADMISSION_CONTRACT.md) | `7a93081cf31ba4d1ea6acb4c1d2e06b4a4916189bff27c31d53d0c00a0373f84` |

## Findings and correction

**Index timing is appropriately qualified.** The primary announcement establishes December 5, 2016 as the intraday consolidated-tape effective date, with primary-exchange pricing retained for official closes and SOQ. The current policy supports the distinction between historical intraday values and revised official closing files, and dates the index-level auto-hold change after the November 17, 2023 close. The report does not turn a generic calculation-interval example into an SPX historical cadence guarantee or equate an index mark with simultaneous constituent execution. [S&P effective-date announcement](https://www.spglobal.com/spdji/en/documents/index-news-and-announcements/20161107-changes-to-spdji-us-indices-intraday-calculations.pdf), [Equity Indices Policies & Practices, pages 46, 50–51 and 62](https://www.spglobal.com/spdji/en/documents/methodologies/methodology-sp-equity-indices-policies-practices.pdf).

**The central Cboe rights gate is supported.** Section I.1(e)(viii), PDF page4, explicitly places index-derived-data creation and specified indicator/strategy uses behind a separate written executed agreement. Treating ordinary purchase/internal-use wording as insufficient authorization for this proposed derived research is justified; the reports do not purport to classify the user's license or authorize an order. The product/specification also support the distinction between dissemination timestamps and original customer receipt, including the 2019 timestamp-precision change. No missing receipt or revision fields are inferred into existence. [Cboe license](https://datashop.cboe.com/documents/Cboe_LiveVol_DataShop_License_Agreement.pdf), [Main Channel specification](https://datashop.cboe.com/documents/Main_Channel_Tick_Data_v1.pdf), [product description](https://datashop.cboe.com/main-channel-tick-data).

**One FirstRate wording correction was requested and made by the provider report's author.** The earlier blanket redistribution prohibition omitted explicit allowances for insubstantial disclosures and certain service-provider uses. The final version states the commercial third-party restriction and these allowances separately. It still correctly distinguishes permitted private/derived research from historical timeliness, completeness and revision proof. This correction does not change the unresolved source-admission decision. [FirstRate license, grant clauses (i), (ii), (iv), (v), and Subscriber obligations](https://firstratedata.com/about/license).

**The advertised-history exclusions are bounded.** The independent reread confirmed the index-specific 2017 first-access date for ThetaData's earliest listed tier and February14,2023 for Massive's index-value files. Those published offerings cannot supply the unchanged pretraining and first development year. The reports avoid equating their longer equity/options histories with index history and avoid claiming an undiscovered custom product is impossible. [ThetaData index subscription table](https://docs.thetadata.us/Articles/Getting-Started/Subscriptions.html), [Massive index-value history](https://massive.com/docs/flat-files/indices/values).

## Cross-document scientific consistency

- The local-source conclusion is restricted to the inspected inventory. Daily OHLC and daily realized-risk summaries are not represented as recoverable intraday price paths; NQ/SPY are not substituted for SPX.
- The cutoff remains a prospective as-of convention. Event time, receipt time, both minute-bar labeling conventions, stale starting marks, revised prior closes and label maturity are kept separate. No source is admitted by timestamp formatting alone.
- Early-close duration and exceptional closures remain prospective measurement decisions. No new historical exclusion or scored slice is introduced from the documented methodology changes.
- The denominator remains an unresolved information/control issue. The summary does not claim that a finite `log(K_t)` adjustment isolates current-shock information; neither of the contract's alternative directions is selected.
- Both controls must be reconstructed for the changed event. Old probabilities are not carried into a new target, and missing eventual labels do not erase already issued applications.
- The summary retains 131 prior hypotheses and adds zero for this documentary pass. The possible133 count is explicitly contingent on future registration. “Source-blocked” is not presented as a statistical null, exhaustive research failure, acquisition approval or predictive result.

## Limits of this review

The external documents were inspected through public web retrieval. No locally byte-pinned publisher originals were created, so this is not immutable historical documentary provenance. The provider search is deliberately bounded; this review does not certify every current price, every vendor's full license applicability or any actual archive's completeness. It does not establish a receipt-time historical record, adequate class support or a ready-to-run experiment. Those limitations are disclosed in the report set rather than deferred to a numerical result. The root's separate byte/link preservation audit is distinct from this documentary review.
