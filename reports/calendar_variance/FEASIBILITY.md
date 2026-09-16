# Calendar variance: initial date/source feasibility

Date/source-only review completed 2026-09-07 before wave 8 numerical feature construction, targets, fits, or scoring. **The fixed initial 1,000-row requirement is feasible: 1,012 common mature training rows are supported at the first feature-complete January 2016 origin, 2016-01-04.** Its previous observed SPX session cutoff is **2015-12-31**. This is a source-support finding, not a test of predictive usefulness, feature rank, or optimizer convergence.

The proposed question remains the source-and-timing-controlled replication described in [the prospective design](../tail_shape/NEXT_CALENDAR_DESIGN.md). The repository already tested broader full-session macro-calendar variance models; documentary original plans with these controls are the narrower distinction.

## Independent reconstruction

This review reads only the bounded SPX parquet **date index**, the three Cboe CSV **DATE columns**, and existing documentary plan records. No price, implied-volatility value, numerical variance target, event flag, fitted output, or feature/target association was read or constructed. Full market-file SHA-256 checks match the prior [wave 6 source-validity audit](../index_hinge/SOURCE_REVIEW.md); that audit establishes positive finite complete observed OHLC/IV fields. It is therefore sufficient to reconstruct numerical availability from these unchanged sources' dates without rereading their values. The existing read-only `macro_search.load_plan_records` checks admitted source extraction hashes and source statuses; coverage dates are then reconstructed separately, without calling a market feature/target builder or the old event-flag builder.

The 4,226 bounded SPX dates run from 2009-01-02 through 2025-10-20. Every reference date remains in the rolling calendar. A daily return and GK-plus-overnight observation need the preceding close; 22 such observations followed by the declared one-session lag first have date support at reference position 23 (zero-based). Each lagged IV series must have an observation on the exact previous SPX date; nothing is filled. VIX9D first supports an entry on 2011-01-05. The missing VVIX observation on 2013-05-13 excludes entry 2013-05-14. Earlier VVIX absence does not affect the VIX9D-complete training window.

For each origin, the nominal window starts at 16:00 America/New_York and ends at 16:00 on the next Monday–Friday civil date, skipping weekends only. This computation does not accept the next observed SPX date. Both BLS series need an admitted original plan for every calendar month touched by that nominal window, and each source's publication civil date must be **strictly earlier** than the previous observed SPX date. The nominal ending year must have its eligible complete original FOMC annual plan. An absent or not-yet-eligible plan is unknown, even on dates when no announcement would ultimately occur. Actual next observed dates are used only for label maturity: a historical origin must precede the fit, and its next-session close must be no later than the fit cutoff.

| Initial fit admission | Rows |
| --- | ---: |
| Market/date-complete mature candidates | 1,254 |
| Unknown CPI coverage | 220 |
| Unknown NFP coverage | 23 |
| Unknown for both CPI and NFP | 1 |
| Unknown FOMC annual coverage | 0 |
| Excluded union | 242 |
| Common mature training rows | **1,012** |

The common origins run from **2011-01-05 through 2015-08-28**, and the latest admitted label matures **2015-08-31**. Market-only maturity would allow origins through 2015-12-30; the documentary coverage mask, rather than the target horizon, causes the substantial training staleness at the first fit.

| Origin year | Common mature rows |
| --- | ---: |
| 2011 | 140 |
| 2012 | 227 |
| 2013 | 250 |
| 2014 | 252 |
| 2015 | 143 |

## Excluded coverage

These counts refer to otherwise market/date-complete mature candidates at the initial fit. “Absent” means no admitted plan under the frozen source policy, including excluded ambiguous/reissued sources; it does not claim the release itself did not occur. A boundary row can touch two missing months, so the following month counts are **not additive** within an event.

| Series | Required month | Reason | Affected candidate rows |
| --- | --- | --- | ---: |
| CPI | 2011-03 | Absent admitted plan | 24 |
| CPI | 2011-04 | Absent admitted plan | 21 |
| CPI | 2011-06 | Absent admitted plan | 23 |
| CPI | 2011-07 | Absent admitted plan | 21 |
| CPI | 2011-08 | Absent admitted plan | 24 |
| CPI | 2012-03 | Absent admitted plan | 23 |
| CPI | 2013-11 | Publication not strictly before cutoff | 1 |
| CPI | 2015-03 | Publication not strictly before cutoff | 1 |
| CPI | 2015-09 | Absent admitted plan | 22 |
| CPI | 2015-10 | Absent admitted plan | 23 |
| CPI | 2015-11 | Absent admitted plan | 21 |
| CPI | 2015-12 | Absent admitted plan | 22 |
| NFP | 2015-03 | Absent admitted plan | 23 |

The unchanged complete source corpus admits 168 of 190 CPI records and 176 of 190 payroll records. CPI excludes 17 original-vintage-uncertain reissues, four ambiguous printed timestamp/ID records, and one publication outside the fence; payroll excludes 13 reissues and one publication outside the fence. Sixteen annual FOMC documents supply exactly 128 original meeting dates. Canceled original plans remain plans; no revised realized dates or emergency FOMC events are substituted.

## Limits and decision

The source publication window is 2010-01-01 through 2025-10-20; numerical market access remains bounded at 2025-10-20, before the sealed 2025-11-03 boundary. Market bytes are later archival snapshots, and early VIX9D training history is back-calculated. The SPX calendar is the observed vendor calendar, whose original builder removed incomplete OHLC records, not a certified historical exchange calendar. Documentary snapshots are current official-page extractions with exact extraction hashes, not immutable original provider vintages.

The nominal window ignores holidays and early closes and uses Eastern civil time. Those assumptions remain fixed and do not permit filtering after inspecting the actual next session. Extending the prior 09:30 endpoint to 16:00 leaves touched-month and annual-coverage eligibility unchanged; no new announcement frequency is inferred here.

Proceed with prewritten synthetic tests and protocol registration while preserving **all** controls, the strict unknown-month mask, and minimum training size 1,000. The initial margin is only 12 rows, and the old training observations cannot be described as recent complete coverage. A later source/admission discrepancy must fail the gate rather than trigger a smaller minimum or a removed control.

## Subsequent synthetic implementation checks

After the date-only feasibility decision, `tests/test_calendar_variance_features.py` was written before its producer existed. Its initial run exited 1 with `ImportError: cannot import name 'calendar_variance_features' from 'src'`. The new `src/calendar_variance_features.py` was then implemented; **all 22 synthetic contracts pass**, and scoped Ruff checks pass. The two lint-only test cleanups did not change a contract. No empirical feature or target construction, fit, or score was run by this subtask.

The tests check the 18-column common feature schema, exact raw-price risk/negative-return histories and market lags, observed next-session target maturity, missing-window propagation, native risk units, the fixed GK floor for flat prices, ignored adjusted close, raw-price unit invariance, invalid OHLC rejection, current-entry calendar encoding, and predictor invariance to future values or observed-session changes. Calendar contracts include both DST directions, nominal holidays and early closes, open-left/closed-right timestamps, strict source cutoff dates, both touched months and both BLS series, unknown annual coverage, incomplete annual plans, original cancellations, and future-source invariance. Synthetic source files place forbidden numeric tokens after the fence and verify their exclusion; the unchanged original-plan source adapter is mocked for that isolated loader test.

The new wrapper reuses the frozen bounded market parser and original-plan source loader. Its nested market audit updates only the current feature/target description and removes the old gap interpretation; original source hashes and source/date diagnostics remain intact. Its plan evidence preserves the previous schema, with the new nominal ending clock and durations. Fitting and all empirical gates remain the root runner's responsibility.

| New artifact | SHA-256 after synthetic checks |
| --- | --- |
| `src/calendar_variance_features.py` | `a8d595a4009abc605e04bf02f07a7b71192f7db971a38ffcc19df003e59b8dad` |
| `tests/test_calendar_variance_features.py` | `ab0e61a38fccc0714f91c246134100a660f007ac32d4f29bdf884b5ecb5370e1` |
