# Treasury full known-notice inventory: preflight and review contract

**The fixed notice selection matches an independent reconstruction of the pinned projected metadata. This establishes the known linked-document inventory, not the contents, first-release history, or predictive usefulness of those documents.** The next documentary review can proceed under the existing source gates. No new approval step is introduced.

The [fixed selection](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/next_signal_review/TREASURY_FULL_NOTICE_METADATA_SELECTION.json) has SHA256 `7f1fb67ffb6b8543d40ec15fd51d54cea76bf89575dfa32cac14ecae17599971`. The independent [preflight record](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/next_signal_review/TREASURY_FULL_NOTICE_PREFLIGHT.json), SHA256 `d54db2d79854bb33e6458fe82ddcb363e08dc26db65cbb2298fd117784e62b6b`, contains the 18 verified source pins, annual counts, exact membership fingerprint, reused-body/receipt/header bindings, and complete mapping observations. Verification used only the 16 projected annual metadata files, pinned inventory/capture manifests, and hashes of the seven retained pilot originals. No raw annual numerical fields, new PDFs, market data, or network requests were accessed. Only this contract and the new preflight JSON were written.

## Fixed scope and accounting

The scope is the existing literal metadata rule: SecurityType `Note` or `Bond`, with term bucket `2-Year`, `3-Year`, `5-Year`, `7-Year`, `10-Year`, or `30-Year`. Auction dates stay within 2010-01-01 through 2025-10-20. This provisional display-label scope does not establish final nominal-security eligibility, original tenor, regular-auction status, or coverage of instruments outside these six groups.

| Metadata quantity independently reconstructed | Count |
|---|---:|
| Projected annual files / all projected events | 16 / 5,347 |
| All projected document associations, across all event types and document roles | 27,537 |
| Events satisfying the six-term Note/Bond metadata rule | 1,138 |
| Such events with at least one special-notice link | 235 |
| Distinct selected special-notice URLs | 253 |
| Selected special-notice/event associations | 298 |
| Originals present in the pinned pilot capture / remaining URLs | 7 / 246 |

Every selected URL and its full ordered membership tuple matched the reconstruction: announcement date, auction date, CUSIP, security term, security type and term bucket. The seven reused originals matched their recorded PDF, receipt and header hashes and requested/effective URLs. The preflight verifies their capture bindings; the selection's already-reviewed label inherits the prior documentary work rather than claiming that work was repeated here. Preserve all existing pilot findings and failures.

The inventory has 41 URLs linked to multiple selected events, 35 same-day shared-URL groups, and 56 selected events with multiple notices. Seventy-seven selected notice URLs are also linked to nonselected event types in the same projected metadata. A notice can therefore address a wider schedule than its selected Note/Bond memberships. Reuse one retained body per URL while keeping every event association; do not join events by date alone or carry one event's disposition across all others.

There are no duplicate selected event identities, cross-auction-year URLs, filename-year conflicts, missing filename-date tokens, invalid date tokens or tokens after the ceiling in this reconstruction. Those are metadata checks, not authenticated public-release dates. Four associations have filename dates after the linked auction: `BPD_SPL_20100625_1.pdf` for June 22–24, 2010, and `BPD_SPL_20150911_1.pdf` for September 10, 2015 / `912810RN0`. The former's full inventory adds June 23 / `912828NL0` and June 24 / `912828NK2` to the pilot's June 22 / `912828NS5` membership. Preserve that expansion. The latter needs body classification; its filename alone does not prove a correction.

## Required documentary record

Account for all 253 documents and all 298 selected associations, including any excluded or unresolved event. Each notice record should retain:

1. **Evidence and clock:** exact URL, original-body hash, text/extraction hash when used, page or section location, the literal dated release header, and any explicit clock/timezone. Keep auction, announcement, issue, maturity, correction and retrieval dates separate. If a header is missing, conflicting or illegible, retain an unresolved clock or explicit agreeing originating-agency evidence; never substitute a filename, HTTP Last-Modified, PDF creation date, or the auction date for public release.
2. **Purpose and affected fields:** identify correction/replacement, first publication of a required addendum, schedule/terms change, cancellation, test/contingency event, conditional identity/reopening notice, other supported purpose, or unresolved. Multiple purposes may coexist. Name the exact affected fields and any explicit unchanged-field boundary. A special slot, result filename prefix, or later date does not by itself establish revision.
3. **Actual applicability:** record the events, securities, documents and conditions the body actually addresses, separately from API memberships. Match date plus CUSIP and additional stated identity where needed. Preserve broad schedule applicability, shared notices, reused CUSIPs across auctions, omitted linked events and conflicting/incorrect API associations. A newly observed cross-reference becomes a traceable follow-up obligation; do not invent a replacement URL or silently declare the original inventory exhaustive.
4. **Disposition and unresolved work:** give an evidence-backed conclusion for each association, distinguishing unchanged fields, changed fields, delayed first publication, regular versus test/other events, and missing original evidence. Record precise remaining questions rather than promoting parser success to source admission.

## Original reports, addenda and conditions

A correction must identify its affected original and its own supported release date. Retain recoverable originals and corrected versions separately. If a required first-report field was demonstrably overwritten and its original cannot be recovered, that field remains unknown; corrected numbers cannot reconstruct an earlier information set. An explicit unchanged-field statement can support a field-specific finding when the original result identity and release evidence are also established. Do not mark all competitive allocations revised merely because SOMA or an overall total changed, and do not ignore a genuine correction because the intended numerator appears unaffected.

A first addendum is the first availability of any required fields it newly supplies, even when the main result was published earlier. Use the latest supported first-release date of the required inputs. A schedule-only change does not automatically reset unrelated result availability. Conditional substitution or reopening warnings require realized result identity and any stated follow-up confirmation: the possible alternative security is not the actual outcome. Preserve announced and realized CUSIPs, reopening flags and original lineage separately, and resolve their relationship explicitly rather than silently changing the selected event or inferring realization from a headline.

An apparent or actual post-ceiling release stays outside the pre-ceiling feature information set. Record relevant later correction evidence as an audit relationship when reviewed within the authorized source work; do not import its corrected values into the past. Unknown release dates remain unknown. Future issue/maturity descriptors can be facts already announced before the ceiling and are not automatically future observations. Neither a visible date/time nor today's capture hash proves actual historical delivery of these exact bytes. Dated primary releases with coherent identity and resolved known notices can support qualified exploration without a contemporaneous cryptographic archive; lack of that archive alone is not a new blanket blocker.

## Efficient independent review and completion limits

Every document needs an independently checkable classification. Repeated templates may use exact captured-text and evidence comparisons, with each document's date, identity, event applicability and substantive differences checked. Identical templates do not justify blindly copying a prior conclusion. Manual rendering of every duplicate is unnecessary when the exact text and evidence classification can be independently confirmed. Every unresolved/ambiguous item, actual correction/replacement, or test/contingency case requires whole-page visual review of all relevant pages and footnotes. Use the retained seven pilot originals and prior visual evidence where applicable; do not overwrite or relabel prior findings.

Scalar figures seen for documentary authentication remain source evidence, not model inputs. Do not calculate candidate features, yield-trigger outcomes, support floors or forecast results during notice review. A missing original, extraction failure, unresolved identity or inadequate clock is a source problem, not a null predictive result. Preserve it explicitly rather than dropping the event, filling zero or changing an experiment gate.

Completion means every known selected notice and association has a supported disposition or precise unresolved requirement, with cross-referenced applicability accounted for. It does not establish that the archive linked every relevant notice, validate all result amounts, admit the full history, or register an experiment. Existing comparisons remain **144**. Full-history source admission and any prospective forecasting protocol remain separate work under the already established methodology.
