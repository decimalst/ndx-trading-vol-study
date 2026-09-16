# Civil-time windows and original FOMC plans

Source assessment completed 2026-09-07, before registration or fitting of the
proposed macro overnight study. No prices, target values, forecast scores or
previously sealed outcomes were read for this task. Earlier studies were not
modified.

**The FOMC source ledger is ready for independent verification. A complete
historical exchange-session plan ledger has not been established.** The agreed
alternative is an explicitly nominal civil-time window, with no claim that it
equals a planned or actual exchange holding interval.

## Original FOMC annual plans

Sixteen dated Federal Reserve announcements supply eight original planned
meetings per year, or **128 meetings for 2010–2025**. Use each year's selected
original full annual announcement. Do not replace it with a later revision,
an eventual decision date, or an emergency meeting. Discard the next-year
January carryover from that document, since that next year's full annual
announcement supplies its own eight-plan ledger.

| Plan year | Dated official annual announcement | Stated publication time, Eastern |
|---|---|---|
| 2010 | [2009-06-04](https://www.federalreserve.gov/newsevents/pressreleases/monetary20090604a.htm) | Immediate release; clock time unstated |
| 2011 | [2010-05-28](https://www.federalreserve.gov/newsevents/pressreleases/monetary20100528b.htm) | Immediate release; clock time unstated |
| 2012 | [2011-03-25](https://www.federalreserve.gov/newsevents/pressreleases/monetary20110325a.htm) | Immediate release; clock time unstated |
| 2013 | [2012-05-16](https://www.federalreserve.gov/newsevents/pressreleases/monetary20120516b.htm) | Immediate release; clock time unstated |
| 2014 | [2013-04-05](https://www.federalreserve.gov/newsevents/pressreleases/monetary20130405a.htm) | Immediate release; clock time unstated |
| 2015 | [2014-06-05](https://www.federalreserve.gov/newsevents/pressreleases/monetary20140605b.htm) | 14:00 EDT |
| 2016 | [2015-05-11](https://www.federalreserve.gov/newsevents/pressreleases/monetary20150511a.htm) | Immediate release; clock time unstated |
| 2017 | [2016-06-28](https://www.federalreserve.gov/newsevents/pressreleases/monetary20160628a.htm) | 14:00 EDT |
| 2018 | [2017-05-11](https://www.federalreserve.gov/newsevents/pressreleases/monetary20170511b.htm) | 14:00 EDT |
| 2019 | [2018-05-25](https://www.federalreserve.gov/newsevents/pressreleases/monetary20180525a.htm) | 12:00 EDT |
| 2020 | [2019-05-17](https://www.federalreserve.gov/newsevents/pressreleases/monetary20190517a.htm) | 10:00 EDT |
| 2021 | [2020-07-02](https://www.federalreserve.gov/newsevents/pressreleases/monetary20200702a.htm) | 10:00 EDT |
| 2022 | [2021-06-04](https://www.federalreserve.gov/newsevents/pressreleases/monetary20210604a.htm) | 11:00 EDT |
| 2023 | [2022-06-24](https://www.federalreserve.gov/newsevents/pressreleases/monetary20220624a.htm) | 14:00 EDT |
| 2024 | [2023-06-23](https://www.federalreserve.gov/newsevents/pressreleases/monetary20230623a.htm) | 11:30 EDT |
| 2025 | [2024-08-09](https://www.federalreserve.gov/newsevents/pressreleases/monetary20240809a.htm) | 13:30 EDT |

Publication time is distinct from a meeting's eventual statement time. The
ledger assigns **no statement times**. The Fed's
[2013 timing announcement](https://www.federalreserve.gov/newsevents/pressreleases/monetary20130313a.htm)
introduced 14:00 Eastern for all regularly scheduled statements; applying that
time throughout 2010–2012 would need separate evidence. The proposed FOMC
feature instead asks whether the original planned meeting's final **date**
equals the nominal next weekday. It is an anticipation/timing comparison,
without a claim that an announcement fell within or after an actual held
interval.

Preserved source distinctions matter. The original 2011 September plan ends
September 20. The original 2012 July, September and December plans end July 31,
September 12 and December 11. The document used for the original **2013**
schedule also contains a **revised 2012** block; the parser explicitly excludes
that block from the 2013 extraction and keeps the earlier 2012 source. The
2020 ledger retains the original March 17–18 plan and contains none of the
March 3, 15 or 16 emergency/retrospective replacements. These are immutable
plan identities, not assertions that the meetings occurred as planned.

The source policy for model inputs is deliberately conservative:
`source_publication_date < feature_cutoff_date`, where the cutoff date is the
previous completed market observation's date. The exact header time is retained
for provenance but does not relax this extra date lag. For date-only documents,
the ledger also records next-calendar-day 00:00 Eastern as a conservative
availability bound; this is a bound, not an invented publication timestamp.
Every annual source predates its covered year. Complete annual coverage means
coverage of the selected original schedule, not all realized policy events.

## Why the calendar window is explicitly nominal

Primary exchange notices establish that advance session information exists.
Examples identified in this bounded review include Nasdaq's
[2010 holiday announcement](https://www.nasdaqtrader.com/TraderNews.aspx?id=dn2009-041),
[2011 holiday announcement](https://www.nasdaqtrader.com/TraderNews.aspx?id=dn2010-030),
and [November 2010 early-close notice](https://www.nasdaqtrader.com/TraderNews.aspx?id=UVA2010-016).
The 2010 annual page is labeled revised; that label cannot be silently treated
as an unmodified original publication. Instrument scope also matters: futures
and options notices can specify hours different from regular Nasdaq equities.

These examples do not establish all annual holiday rules, changes, early
closes, publication dates and prior-cutoff schedule revisions for 2010–2025.
Historical exchange rules might support a complete reconstruction with more
document collection; that work was stopped once the nominal alternative was
chosen. Neither a current exchange-calendar library nor the next observed QQQ
date supplies the missing historical publication evidence.

The fixed construction is:

1. Anchor entry civil date `t` at **16:00 America/New_York**.
2. Find the first Monday–Friday civil date strictly after `t`, skipping weekends
   only; anchor it at **09:30 America/New_York**.
3. Use this nominal interval for eligible original CPI/payroll plan timestamps.
   For FOMC, compare only the nominal ending date with the original meeting's
   final date.
4. Calculate nominal elapsed hours after converting both endpoints to UTC.
   Subtracting Python datetimes with the same timezone directly can incorrectly
   retain wall-clock duration across a DST change.

Ordinary weekday nights are 17.5 hours; ordinary Friday windows are 65.5 hours.
Spring and autumn DST Friday windows are respectively 64.5 and 66.5 hours.
The local zoneinfo file is hash-recorded for reproducibility. This pins civil
time conversion, not an exchange's historical schedule publication.

The limitations are part of the definition. Thursday before Good Friday points
to Friday morning, so an 08:30 Friday plan is included even though equities
remain closed that day. Friday before a Monday holiday still points to Monday
morning and misses a Tuesday release that could occur before the actual next
opening. Early-close dates retain the nominal 16:00 anchor. An unplanned future
closure never changes the nominal endpoint. Consequently this is a coarse
advance-plan calendar predictor of overnight risk; its coefficient cannot
establish complete announcement exposure or an actual holding-duration effect.
Actual observed sessions may determine labels and label maturity, while
contemporaneously observed entry dates define opportunities. No future
observed session enters these nominal predictors.

## Saved artifacts and remaining gates

All source and derived data are ignored under
`data/source_discovery/macro_plans/calendar/`:

- `fomc_capture_manifest.json` and sixteen `fomc_YEAR_official_extract.txt`
  files preserve exact UTF-8 web-tool response strings, their hashes, capture
  times and official URLs. They are current rendered official-document
  extracts, not raw HTML bytes or historical web captures. Raw-provider hashes
  and tool versions are explicitly null; the current dated pages do not prove
  unchanged bytes since original publication.
- `fomc_original_annual_plans.csv` contains 128 rows, source text and hashes,
  original start/final dates, publication precision and plan policy.
  `fomc_annual_coverage.json` contains sixteen complete annual plan intervals.
- `build_fomc_ledger.py` reproduces the ledger from saved extracts. Five
  synthetic parser checks run before documentary parsing; all 128 printed
  weekday labels and sixteen saved hashes are checked. Its audit explicitly
  says that independent verification remains outstanding.
- `nominal_weekday_windows_2010_2025.csv` contains **4,174 civil weekdays**,
  including exchange holidays. It is not a list of trading sessions.
  `build_nominal_calendar.py` reproduces it with eight prior synthetic checks
  covering DST, weekends, Good Friday, a Monday holiday, early-close anchoring
  and immunity to future closures. Its audit pins the timezone file and output
  hashes.

Before any fit, independently reconstruct the FOMC dates and nominal windows;
freeze the source-eligibility rule, unknown-coverage behavior and final study
fences; and test the actual feature builder's causal joins. Missing BLS plan
coverage must remain unknown rather than zero. The nominal substitute changes
the proposed interpretation and baseline duration feature, so the complete
numerical protocol must state it before scoring. No forecasting hypothesis
was added or tested by this source task.
