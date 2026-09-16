# First-release claims: source audit checkpoint

**No claims forecasting experiment has run yet.** The source audit found that ALFRED's initial-release-only export cannot be used unchanged as an independently authenticated first-publication history. The next experiment will use an original-DOL ledger with explicit source-quality gaps, subject to its prewritten admission checks.

The controlled export contains 852 weekly records from reference week 2009-05-30 through 2025-09-20. Original documents were acquired for all 852 bounded archive links, plus an independently located official release omitted from the annual index. Full extraction covered 253 HTML files and 600 PDFs containing 5,594 pages. The parser identifies 852 genuine claims reports with matching displayed body dates; the remaining document is an agency dummy file.

| Finding | Consequence |
| --- | --- |
| 850 genuine reports match ALFRED exactly on week, value and date. | Retain the original reports with the separate archive cross-check. |
| The March 22, 2018 report first published 229,000 for the March 17 reference week. The March 29 report revised it to 227,000, which ALFRED associates with the later date. | Use the authenticated original report in the new ledger; preserve the contaminated export record and its mismatch. |
| The preserved March 23, 2017 DOL document is marked CORRECTED and says 261,000; ALFRED says 258,000 in both inspected March 23/24 vintages. | The original DOL version and correction time remain unresolved. Keep that reference week explicitly unavailable. |
| Four requested reference Saturdays at the end have no admitted report within the source ceiling. | Preserve those gaps. This is not an assertion that all four releases were due before the ceiling. |

The intended source disposition is **851 supported first-report values, one unresolved value and four explicit absences**. Those counts describe source records, not forecast origins or sufficient training support. The unchanged raw-export reconciliation retains its failed status; it was not relabeled as successful after the source disagreements were explained.

The three annual-index filename anomalies were resolved through official bodies. A further omitted 2019 release was located in DOL's newsroom. Every original payload, request receipt, extraction and exception remains pinned. Text-format corrections were tested with generated examples before their use on real reports; the initial failed pass remains preserved. The corrected source components pass 87 synthetic tests. The final repository-wide regression passes **2,465 tests in 378.109 seconds**, with the checked code unchanged during the run (`FULL_SOURCE_FINAL_CHECK.json`). This certifies the completed source components, not the still-unimplemented forecasting study.

The [source disposition](SOURCE_DISPOSITION.md) fixes the handling of disagreements and missingness before predictive calculations. The [forecasting draft](PREDICTIVE_DESIGN_DRAFT.md) proposes the signed change from four prior first reports, one five-session QQQ variance target, and two required comparisons against matched labor/calendar controls and the established market benchmark. No estimator grid, numerical support calculation, model fit, score, or new hypothesis registration has occurred. The cumulative family remains 140.

The next work is to build and independently verify the explicitly incomplete first-report ledger, then freeze and test the forecasting protocol before running it. Dated public archives do not prove immutable historical downloads or exact intraday publication clocks. All later forecasting findings remain exploratory on reused market history.

Detailed evidence: [complete corrected source comparison](FULL_RECONCILIATION_FORMATS_RESULT.json), [independent discrepancy review](SOURCE_DISCREPANCY_REVIEW.md), [unresolved 2017 correction timing](2017_CORRECTION_TIMING.md), [archive exceptions](ARCHIVE_RECONCILIATION.md) and [full text extraction](FULL_TEXT_EXTRACTION.md).
