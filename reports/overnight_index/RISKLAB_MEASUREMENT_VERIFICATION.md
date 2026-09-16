# Risk Lab SPY measurement audit: independent verification

2026-09-07. **Verified as a measurement audit, without admitting the source to a
model.** An independent calculation reproduced 1,722 scalar or list checks from
the saved raw provider response: source metadata, six estimator mappings,
validity counts, positive-value quantiles, four auxiliary fields, all three
fixed periods and all sixteen annual panels for 2010–2025. No forecast was
fitted, no predictive comparison was made, and no source or frozen study was
modified.

The [verification JSON](/Users/byrons/code/trading-vol/ndx-vol-experiment/data/source_discovery/quarantine/risklab_spy/measurement_verification.json)
records hashes, methods and counts. It checks the existing
[measurement audit](/Users/byrons/code/trading-vol/ndx-vol-experiment/data/source_discovery/quarantine/risklab_spy/measurement_audit.json)
against the fixed
[measurement contract](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/overnight_index/RISKLAB_MEASUREMENT_CONTRACT.md).
Both source and numerical verification artifacts remain under ignored data
paths.

The response's six-line header identifies SPY, permanent identifier 84398,
description `SPDR S & P 500 E T F TRUST`, and 7,680 unique increasing dates from
1996-01-02 to 2026-08-21. Every row identifier agrees with the header. The
737,911 saved bytes match acquisition SHA-256
`b7802ef21565b13831420c8fdd2b15176e57265fd5afe981be73693c8350e392`.
The independent parser read date and identity tokens first. It parsed ten
numerical fields only for the 7,473 rows through 2025-10-20; it parsed **zero
numerical fields** from the 207 later rows. Full-history dates were used only
to check declared count and date extent.

The saved provider chart script independently confirms zero-based volatility
fields 2/5/6/7/10/11: QMLE, five-minute and fifteen-minute estimates from trades,
then their quote counterparts. Auxiliary fields 3/4/8/9 contain MA orders and
displayed confidence widths. The audit retains native values. It applies
neither the chart's lower-value removal and upper clipping nor its percentage
display conversion; the old square-root expression is commented out in the
provider script. Source-documentation byte lengths and hashes also matched
their saved manifest.

| Fixed interval | Present source dates | QQQ reference sessions | Absent source dates | Present invalid primary estimates | Complete 1 / 5 / 22-session windows |
|---|---:|---:|---:|---:|---:|
| 2000–2015 | 4,025 | 4,025 | 0 | 0 | 4,025 / 4,021 / 4,004 |
| 2016–2019 | 999 | 1,006 | 7 | 0 | 999 / 983 / 927 |
| 2020–2025-10-20 | 1,438 | 1,458 | 20 | 0 | 1,438 / 1,358 / 1,056 |

All six estimators are positive and finite on every present source date in
these three intervals; zero, negative, NaN and infinite counts are zero there.
The pre-2016 quote fields contain 4, 7 and 6 observations above the chart cap
for QMLE, five-minute and fifteen-minute estimates respectively. Those values
were preserved. These are counts by field, not a claim about distinct dates
or relative estimator quality. All reported quantiles, order-validity counts
and confidence-width counts matched the separate reconstruction.

Only the bounded timestamp column was read from the QQQ parquet file using an
Arrow predicate; no QQQ price fields were loaded. The checker counted
consecutive valid sessions directly, resetting at missing or invalid dates,
rather than using the producer's rolling-window implementation. Windows also
reset at the beginning of each separately reported period or year, matching
the measurement contract. They are complete **trailing measurement windows**,
not counts of scored or tradable forecasts. No source date falls outside the
reference calendar in the three intervals, but the reference is an observed
QQQ calendar, not a historically announced SPY calendar.

The implementation history was checked explicitly. The first CLI failed when
the calendar reader compared a parquet timestamp against a string, after the
source parser returned and before diagnostic output. The original source and
ten-test snapshots match the first manifest. The second attempt added a typed
calendar reader and one regression test. Its current source, eleven-test file
and unchanged contract match the second manifest, and all eleven tests passed
again. AST comparisons confirm that the source parser and numerical diagnostic
functions are identical between attempts. **The original audit implementation
was amended; it was not unchanged.** The retained records place the amendment
before the first successful diagnostic output, without changing measurement
definitions, periods or numerical rules.

The independent numerical calculation did not import the producer audit
module. It used scalar classifications, sorted linear quantile interpolation
and consecutive-run counters, with exact count/date comparisons and fixed
floating tolerances of relative 1e-12 and absolute 1e-14. All 187 pinned
code/input/prior-artifact hash checks in the latest frozen overnight model
study's manifest also passed.

This establishes faithful parsing and reproduction of the specified source
diagnostics. It does not resolve the provider's annualization factor, equity
session boundaries, historical publication latency, revisions or deployment
rights. Values remain in native annualized-volatility units. No squaring,
square rooting, calibration, Oxford-variance conversion, model admission or
predictive claim follows from this verification.
