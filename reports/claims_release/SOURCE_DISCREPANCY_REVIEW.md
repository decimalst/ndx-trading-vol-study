# Bounded source-discrepancy review

Completed 2026-09-08 against the first complete-text reconciliation result. The reported source discrepancies are real, but they have different causes: an archived correction with an unestablished correction clock in 2017, a later revision substituted for the first headline in the 2018 ALFRED export, one dummy archive document, and seven recoverable text-format failures. None is evidence of predictive value or source admission.

This review manually read 12 selected pinned DOL texts, checked 11 literal CSV lines directly inside the preserved ALFRED ZIP without importing either repository parser, and independently verified each selected response/text/extraction/receipt hash chain. Seven PDF first pages were rendered and visually inspected. The private evidence is `data/claims_release/discrepancy_review/evidence.json`; it records source paths and hashes, exact CSV lines, and excerpt hashes. The review changed no source text, parser, test, ALFRED record or reconciliation result and performed no market access or model calculation.

## Value and date discrepancies

| Reference week | ALFRED value / realtime-start date | Preserved DOL evidence | Finding |
|---|---|---|---|
| 2017-03-18 | 258,000 / 2017-03-23; CSV line 409 | The March 23 PDF is explicitly marked **CORRECTED** and reports an advance initial-claims headline of 261,000. The March 30 PDF describes 261,000 as the previous week's unrevised level. | The parser correctly reads the currently archived corrected document. That document does not establish the value or posting time of its superseded version. ALFRED's 258,000 is neither disproved as an initial observation nor independently authenticated by this archive comparison. Do not replace it with 261,000 under the original 08:30 clock. |
| 2018-03-17 | 227,000 / 2018-03-29; CSV line 461 | The March 22 PDF reports 229,000 for March 17. The March 29 PDF explicitly revises the previous week downward by 2,000, from 229,000 to 227,000, while describing the annual seasonal-factor revision. | ALFRED's exported value and date match a later revision, not the March 22 first headline. This is a source discrepancy, not a headline parser error. |
| 2018-03-24 | 215,000 / 2018-03-29; CSV line 462 | March 29 reports an advance headline of 215,000 for March 24. | This valid current-week row shares its ALFRED date with the preceding week's delayed row. The duplicate-date rejection is therefore faithful to the existing contract and must not be bypassed by silently deduplicating. |

For cross-checking, ALFRED line 410 reports 258,000 for week March 25, 2017 with date March 30; the March 30 DOL current-week headline agrees. That separate week's equality does not resolve the March 18 discrepancy. Both March 2017 PDFs' body dates and values, and both March 2018 PDFs' body dates and revision wording, were also inspected visually. A release's printed date, a retrieved archive's present contents, ALFRED's realtime-start field, and the time when a correction became available are distinct evidence. This review supplies no immutable ingestion timestamp or intraday correction time.

## The extra March 15, 2014 document

The complete extracted text at the actually linked `https://oui.doleta.gov/press/2014/031514.pdf` is a dummy-file label and filename. The rendered single page confirms that it contains no release header, reference-week claims headline or claims value. Its Saturday filename is not evidence of an additional weekly release. The strict parse failure is correct: classify this preserved artifact explicitly as a nonreport, retain its bytes and receipt, and never manufacture a numeric observation from it. This review does not alter archive coverage or the reconciliation output.

## Text-format exceptions

The following body dates, reference weeks and first seasonally adjusted initial-claims values are directly readable in the pinned texts. Each value and date also agrees with the indicated literal ALFRED CSV line. These manual checks diagnose the failures; they do not replace the failed automated pass.

| DOL body release date | Reference week | Headline | CSV line | Observed format |
|---|---|---:|---:|---|
| 2011-08-04 | 2011-07-30 | 400,000 | 115 | Explicit header contains `Aug. 4 , 2011`; body date was unverified. |
| 2012-08-02 | 2012-07-28 | 365,000 | 167 | Reference clause has `July 28 the advance`, omitting the comma after the day. |
| 2012-09-06 | 2012-09-01 | 365,000 | 172 | Explicit header contains `September 6 , 2012`; body date was unverified. |
| 2012-09-13 | 2012-09-08 | 382,000 | 173 | Explicit header contains `September 13 , 2012`; body date was unverified. |
| 2012-12-06 | 2012-12-01 | 370,000 | 185 | Explicit header contains `December 6 , 2012`; body date was unverified. |
| 2022-04-14 | 2022-04-09 | 185,000 | 673 | PDF text extraction splits the exact headline word as `a dvance`; the rendered page reads normally. |
| 2022-10-06 | 2022-10-01 | 219,000 | 698 | PDF text extraction splits the exact headline word as `adv ance`; the rendered page reads normally. |

All seven printed release dates are Thursdays and the indicated reference dates are Saturdays. No current headline was substituted with the surrounding revised prior-week or continued-claims values. Some old headers use a seasonal time-zone abbreviation inconsistent with the month; this review verifies their printed calendar dates and does not derive UTC publication instants from those abbreviations.

## Scope and retained result

The root result reports 853 acquired documents, 849 parsed reports, 845 verified body dates, 852 ALFRED rows, 847 exact diagnostic pairs, four headline failures and four body-date failures. The four headline failures comprise the dummy document plus the three headline-format cases above; the four remaining document failures are the header-format cases. These counts are the retained complete-pass result, not a claim that this 12-document review independently reprocessed all documents. Full-scope reconciliation remains rejected for the duplicate release date; the 847 diagnostic pair matches do not admit a partial panel.

The four reported Saturday gaps, 2025-09-27, 2025-10-04, 2025-10-11 and 2025-10-18, are retained exactly. They are gaps in the supplied observation-date range. The last Saturday's ordinary following-Thursday release would fall after the 2025-10-20 source ceiling, so an observation-date gap must not automatically be described as a missing publication due before the ceiling. No missing-body timing or reason is established by this review.

Root separately authorized prospective synthetic regressions for only the demonstrated optional reference-day comma, whitespace inside the headline word `advance`, and whitespace before the body-date comma. Any such later parser version and full-text rerun must preserve this first-pass result and failures. The dummy document and 2017/2018 disagreements remain source adjudication issues, unaffected by those format changes.
