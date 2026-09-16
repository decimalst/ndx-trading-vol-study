# Treasury pilot: four remaining special-notice classifications

Reviewed 2026-09-08 local date. **The four remaining special URLs contain two copies of one schedule-change notice, a separate auction-closing-times notice, and a conditional reopening warning. None describes a correction of already published allocation results.** This completes documentary classification of the pilot's seven unique special-notice URLs together with the preserved [three-notice findings](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/next_signal_review/TREASURY_PILOT_NOTICE_FINDINGS.md). It does not complete the future review of all 253 distinct special URLs in the provisional nominal inventory.

## Evidence and rendering

The four URLs were selected as the exact set difference between the pilot's seven special URLs and the three already reviewed URLs. No selection changed. Before rendering, each original PDF's SHA256 was checked against [TREASURY_DOCUMENT_CAPTURE.json](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/next_signal_review/TREASURY_DOCUMENT_CAPTURE.json), [TREASURY_PILOT_IDENTITY.json](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/next_signal_review/TREASURY_PILOT_IDENTITY.json) and [TREASURY_PDF_METADATA.json](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/next_signal_review/TREASURY_PDF_METADATA.json). Each capture receipt was hash-checked and recorded HTTP 200 with identical requested/effective official URLs. Each identity record already had `METADATA_MATCH`, a checked release date and a single supported release-header date inside the ceiling. These checks preceded inspection of the pages.

Each original was rendered once with `pdftoppm -r 110 -png`, without limiting the page range. All four are one-page originals, and all four complete pages were inspected through the image tool. Every renderer subprocess ended with exit 0; its command/stdout/stderr are retained in a separate `.render.log`. The completed [render manifest](/Users/byrons/code/trading-vol/ndx-vol-experiment/data/source_discovery/treasury_auction/pilot_remaining_notice_visual_v1/render_manifest.json), SHA256 `5983b8cb9f2674a8f997cda8bc2820f280d82cb2aff65696145dcfbe06c16dd1`, records exact original paths, source hashes, archive memberships, release metadata, render paths and image hashes. The source hashes were checked again after rendering. No PDF was downloaded again, modified or replaced.

| Source and complete rendered page | Original PDF SHA256 | Visible release header |
| --- | --- | --- |
| [SPL_20250102_1.pdf](https://www.treasurydirect.gov/instit/annceresult/press/preanre/2025/SPL_20250102_1.pdf); [complete page](/Users/byrons/code/trading-vol/ndx-vol-experiment/data/source_discovery/treasury_auction/pilot_remaining_notice_visual_v1/SPL_20250102_1-1.png) | `9a4568e10d061b80e60d5f8b13e002fe30c2c8c346fa8328aa411d70956ecb75` | January 02, 2025 |
| [SPL_20250102_2.pdf](https://www.treasurydirect.gov/instit/annceresult/press/preanre/2025/SPL_20250102_2.pdf); [complete page](/Users/byrons/code/trading-vol/ndx-vol-experiment/data/source_discovery/treasury_auction/pilot_remaining_notice_visual_v1/SPL_20250102_2-1.png) | `eda36643457ec1b8a73f5b13df099fe15c68d913a24ee4164026108f02d8acd8` | January 02, 2025 |
| [SPL_20250123_1.pdf](https://www.treasurydirect.gov/instit/annceresult/press/preanre/2025/SPL_20250123_1.pdf); [complete page](/Users/byrons/code/trading-vol/ndx-vol-experiment/data/source_discovery/treasury_auction/pilot_remaining_notice_visual_v1/SPL_20250123_1-1.png) | `c816f66faa5add5cd89550622c11696d8f0e4a5ca41855f1d49e0f0efc39a50e` | January 23, 2025 |
| [SPL_20250821_2.pdf](https://www.treasurydirect.gov/instit/annceresult/press/preanre/2025/SPL_20250821_2.pdf); [complete page](/Users/byrons/code/trading-vol/ndx-vol-experiment/data/source_discovery/treasury_auction/pilot_remaining_notice_visual_v1/SPL_20250821_2-1.png) | `f140a3e098c8fc292b0a79b4ceb6dccc36b2737344bbd59701c320062b8891a1` | August 21, 2025 |

The notice dates above come from visible immediate-release headers. Bid closing times and event dates in the bodies are different clocks; none is promoted to the time at which final allocations became public. All four notice dates are before 2025-10-20. Today's capture hashes establish reproducibility, not an independent historical publication timestamp.

## January 2: one schedule notice served at two URLs

Both January 2 pages have the same heading about auction and buyback schedule changes for the week of January 6. They explain that the National Day of Mourning for former President Carter changes auction timing. The three-year note auction is scheduled for January 6, the 9-year 10-month note for January 7, and the 29-year 10-month bond for January 8. The notice explicitly says these auctions take place one day before the dates tentatively announced at the last quarterly refunding and settle as originally scheduled. All three retain noon Eastern noncompetitive and 1 p.m. Eastern competitive closing times.

The same pages also change the time of the January 9 TIPS liquidity-support buyback in the one-year to 7.5-year maturity sector. They specify a 12:40-1 p.m. Eastern window instead of the usual 1:40-2 p.m. window and say settlement is unchanged. This is a buyback operation, not an additional note auction or a competitive dealer-allocation observation in the proposed universe.

Disposition for both URLs: **pre-auction schedule-change notice, with separate buyback timing information**. Neither page supplies allocation results, changes a bidder-category definition, or corrects an already released result. Moving an ordinary auction's calendar date does not turn it into the contingency-test event class identified in the earlier findings. The actual rescheduled date should govern the subsequent result identity and release checks; an expected historical weekday must not override this documented schedule.

The archive associations differ: `_1` is attached to January 6/7 pilot records, while `_2` is attached to January 8. Their actual text applies to all three auctions and the buyback. Both original PDFs remain preserved because they have different byte hashes. Their independently rendered PNGs are byte-identical at the specified 110 dpi, with SHA256 `3f76ea367e7fc6bd2f64e9301945d76788ba0a7790d0cf06e7d2f32b0f4dbff1`, consistent with the identical visible content. This is not evidence of two revisions, two independent release times or two distinct schedule announcements; the metadata difference inside the PDFs was not investigated.

Neither page names a CUSIP. The following mappings to selected securities use the pilot's date/term identities and therefore remain distinct from a CUSIP explicitly printed in the notice:

| Body event | Corresponding pilot identity | Applicability beyond the URL's narrow archive membership |
| --- | --- | --- |
| January 6, 2025 three-year note auction | 91282CMF5 | Both January 2 notices address it, although `_2` is linked through the January 8 event. |
| January 7, 2025 9-year 10-month note auction | 91282CLW9 | Both address it; the heading places it in the ten-year note group, but full original-issuance lineage still belongs to announcement/result identity checks. |
| January 8, 2025 29-year 10-month bond auction | 912810UE6 | Both address it, although `_1` is linked only through January 6/7 in the pilot. |
| January 9, 2025 TIPS liquidity-support buyback, one-year to 7.5-year sector | No individual CUSIP is supplied by these notices | Outside the selected coupon-auction records; retain this cross-reference without inventing a security or fetching buyback documents. |

The notices do not change the stated three-/ten-/thirty-year auction groupings or report a security substitution. They corroborate the scheduling relationship between the heading's long-tenor groups and the shortened remaining terms, but cannot replace full original-term/CUSIP lineage evidence. First allocation availability still depends on the subsequent result/category documents; it is not January 2 simply because the schedule was then known.

## January 23: closing times for four January 27 auctions

The January 23 page specifies noncompetitive/competitive closing times for Monday, January 27, 2025. It groups the 26-week bill and two-year note at 11 a.m./11:30 a.m. Eastern, and the 13-week bill and five-year note at noon/1 p.m. Eastern. The page does not state that these are amended results or provide accepted bidder amounts. Its heading and body concern bid submission cutoffs.

Disposition: **pre-auction closing-times notice**. It keeps separate events on the same calendar day and makes no original-tenor substitution. There is no basis here to merge the two selected coupon auctions, move their result clocks to the announcement day, or treat either as outside the regular auction universe. Actual results remain necessary to establish event identity and when final categories became available; the listed competitive closing times are not authenticated result release times.

| Body event on January 27, 2025 | Pilot relationship |
| --- | --- |
| Two-year note | Selected event 91282CMH1; CUSIP comes from the pilot, not this notice's text. |
| Five-year note | Selected event 91282CMG3; CUSIP comes from the pilot, not this notice's text. |
| 26-week bill | Explicit additional applicability outside the selected coupon events; the notice supplies no CUSIP. |
| 13-week bill | Explicit additional applicability outside the selected coupon events; the notice supplies no CUSIP. |

No bill result or outside-pilot document was requested. The notice does not resolve the separate competitive/noncompetitive filename-suffix identity question for the five-year auction; that belongs to the relevant result bodies.

## August 21: conditional five-year auction / original seven-year reopening

The August 21 immediate-release header precedes a warning about the scheduled August 27, 2025 five-year note auction. If its high yield falls within a stated range, Treasury says the notes would be an additional issue of the outstanding seven-year Series P-2030, CUSIP **91282CHW4**, originally issued August 31, 2023. The page says the additional issue would have that existing CUSIP. It further says that an actual reopening instead of a new five-year note would be identified in the auction-results press release and a special announcement.

Disposition: **conditional pre-auction security-identity/terms notice**. It is neither a correction of an already published August 27 allocation result nor a finding that the condition occurred. The notice contains a yield condition and an outstanding amount as background; these were visible during documentary inspection but were not converted, compared with any auction result or used in a calculation.

The selected archive event is August 27 / **91282CNX5**. The alternative **91282CHW4** and its August 31, 2023 original issue date are explicit cross-references in the notice. The different CUSIP is explained as a conditional alternative; it is not by itself a contradictory source identity. The 2023 date is original issuance, not the release date of this notice or an authenticated auction date. No original 2023 auction document or follow-up announcement was fetched.

This is the same kind of prospective identity question as the January 2016 warning in the earlier findings: the scheduled auction can require final security-lineage confirmation. If the conditional substitution occurred, original seven-year lineage and reopening status would matter despite the current five-year auction description. If it did not, the notice alone cannot reclassify a new five-year issue. Neither the hypothetical range nor the word “unscheduled” alone is a valid exclusion rule. Apply the predetermined regular-auction and original-tenor rules to the authenticated final event, using its results and any relevant follow-up notice. This report does not determine that final outcome.

August 21 establishes availability of the conditional terms. It does not establish availability of final security identity or first dealer allocations. Any document required to establish those retains its own supported first-release date; the original seven-year issue date cannot substitute for it.

## Completion and limits

All four remaining special URLs and their six selected archive associations have a documentary disposition here. The review also retains the January 8 applicability not listed under the first January 2 URL, the January 9 TIPS buyback, the two January 27 bill auctions, and the conditional original seven-year security. No cross-reference caused an unselected document fetch. Both January 2 PDFs remain separate provenance records despite identical rendered content.

Together with the earlier three-page review, all seven unique pilot special URLs are now classified by notice purpose and visible dated header. The known June 2010 SOMA amendment remains the actual result correction among this reviewed set; the conditional notices still require final event-identity confirmation. Completion of notice classification does not mean all pilot results agree, that every original term is admitted, or that the full historical correction review is complete.

Only this new report and the authorized private render directory were written. The private directory contains four complete PNGs, four render logs and the completion manifest. All original PDFs remain byte-identical to their capture pins. No market data, feature panel, cohort, numerical source dataset, model, forecast, support calculation, empirical score or predictive gate was accessed or computed. No experiment was registered and no comparison count changed.
