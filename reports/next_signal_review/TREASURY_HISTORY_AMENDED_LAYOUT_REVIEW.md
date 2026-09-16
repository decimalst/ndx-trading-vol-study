# Seven Treasury announcement exceptions

All seven complete saved texts were read and all seven original one-page PDFs were rendered at 110 dpi and visually reviewed. Their PDF/text hashes match the terminal capture and frozen metadata. The JSON companion records exact inputs, individual evidence and remaining limits; the private render manifest binds every original to its complete PNG.

| Selected auction / CUSIP | Printed release date | Finding |
| --- | --- | --- |
| 2015-01-28 / 912828H78 | 2015-01-26 | Genuine amended announcement. The January 28 auction date is red. The existing January 26 weather notice moves the two-year auction from January 27 and preserves settlement dates. |
| 2015-01-29 / 912828H52 | 2015-01-26 | Genuine amended announcement. The January 29 auction date and 11:00/11:30 a.m. ET closing times are red. The same weather notice documents the five-year auction's move from January 28. |
| 2016-02-26 / 912828P79 | 2016-02-25 | Genuine amended announcement plus text reading-order failure. The original page clearly aligns the own CUSIP with its label. The red February 26 date and 11:00/11:30 a.m. ET deadlines agree with the existing technical-issue reschedule notice. |
| 2017-02-21 / 912828W30 | 2017-02-16 | Ordinary offering announcement. The own CUSIP and auction-date rows are visually unambiguous; extraction emitted values before their labels. |
| 2017-02-22 / 912828W55 | 2017-02-16 | The same text-order problem, independently confirmed on this original page. No amended heading or highlighted correction appears. |
| 2017-02-23 / 912828W48 | 2017-02-16 | The same text-order problem, independently confirmed on this original page. No amended heading or highlighted correction appears. |
| 2017-06-12 / 912828XU9 | 2017-06-08 | Genuine same-date amendment. The heading and 11:00/11:30 a.m. ET closing times are red. The existing June 8 notice gives those deadlines for this three-year auction. |

The four empty-CUSIP extractions are layout failures. The original pages do not have missing or competing own CUSIPs. Any future extractor needs a tested rule for this observed two-column structure; arbitrary standalone identifiers must not become aliases, and saved source text must remain unchanged.

The three revised-release cases retain different original archive/XML announcement dates: January 22, 2015 for H78/H52 and February 18, 2016 for P79. The amended PDF release date must remain separate. For XU9, an unchanged release date does not establish that the captured announcement is the first version.

The scheduling evidence is already in the pinned consolidated review and original A/C ledgers:

- [January 26, 2015 weather rescheduling](https://www.treasurydirect.gov/instit/annceresult/press/preanre/2015/BPD_SPL_20150126_1.pdf): moves the relevant auctions and explicitly preserves settlement dates.
- [February 25, 2016 technical rescheduling](https://www.treasurydirect.gov/instit/annceresult/press/preanre/2016/BPD_SPL_20160225_1.pdf): moves the close to February 26, retains settlement and other auction aspects, and permits updates to submitted bids until the new close.
- [June 8, 2017 closing-time notice](https://www.treasurydirect.gov/instit/annceresult/press/preanre/2017/BPD_SPL_20170608_1.pdf): distinguishes the June 12 three-year and ten-year auction deadlines; only the three-year scope is applied to XU9 here.

Two literal calendar boundaries need separate lineage handling. P79 prints dated date February 29, 2016 and maturity February 28, 2023; W48 prints dated date February 28, 2017 and maturity February 29, 2024. The maturity dates also agree with the frozen XML metadata. These are real source dates, not extraction errors to repair.

The three February 2017 ordinary announcements show an 11:00 A.M. embargo without a timezone in that header. All closing-time rows explicitly use ET. None of these labels authenticates actual historical delivery time. Red highlights and the existing schedule notices establish scheduling changes; they do not provide a complete comparison with an unavailable pre-amendment version or prove unchanged numerical offering terms.

No financial values were converted, no downloads or code changes were made, and no source-history admission occurred. The prior January 2016 RP5 announcement typo remains unchanged and was not re-rendered. All seven cases still require the separately controlled final-result and source-admission steps.
