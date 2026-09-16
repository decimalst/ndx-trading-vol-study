# Risk Lab SPY source audit, fixed before numerical inspection

This is a measurement audit with no forecast fits, target relationships, model
selection or predictive tests. It does not register another hypothesis family.
The primary prospective measure is fixed now as the provider's QMLE from trades.
Inspect all six supplied estimators without replacing that choice based on their
values. New source bytes stay under ignored `data/source_discovery`; no source
is spliced into the existing frozen Oxford studies.

The [provider](https://dachxiu.chicagobooth.edu/#risklab) describes annualized
daily volatility and permits research use only. The linked public
`volatility.js` maps equity fields 2/5/6/7/10/11 (zero-based) to QMLE, five-minute
and fifteen-minute estimates from trades and quotes. It multiplies raw levels
by 100 for percentage display. It also drops values below 1e-6 and clips values
above 3 in the chart; neither operation belongs in this audit. The old square
root expression in that script is commented out and must not be applied.

Download only SPY permanent identifier 84398, at most 1 MB, from the provider's
public endpoint. Preserve raw bytes and hash them. Validate the six-line header,
declared count, date extent, increasing unique dates, identity, and 12-field
historical rows. Preserve raw zero, negative, NaN and infinite numeric values in
the audit. Reject malformed historical values instead of guessing or repairing.
Parse dates before numerical fields, and do not parse any numerical estimate
after 2025-10-20. Full-source header/date metadata may be inspected for source
identity and consistency; it is not a predictor.

All quantities stay in native provider volatility units. There is no
annualization-factor inference, squaring, square rooting, calibration, or
conversion into Oxford daily variance. The provider's exact equity session
boundary, annualization factor, revision history and historical publication
latency remain unresolved. The cited methodological paper describes the
estimator, not a versioned contract for every downloadable field.

Report counts of positive finite, zero, negative, NaN, infinite and above-chart-cap
observations for all six measures; positive-value min/1st/50th/99th/max quantiles;
nonnegative-integer MA orders and nonnegative finite displayed confidence widths.
Those are source diagnostics, not forecast comparisons or reasons to remove
otherwise valid positive extreme values.

Use only bounded QQQ session labels as a US equity calendar diagnostic. This
does not certify a historically announced SPY calendar. Separately count absent
source dates and present dates with invalid primary estimates. Retain every
calendar gap. Count complete trailing windows of 1/5/22 actual reference
sessions, never windows over just the remaining nonmissing records. Report
2000–2015, 2016–2019 and 2020–2025-10-20 plus fixed annual counts for 2010–2025.
No phase or year is selected by numerical behavior.

Any malformed source or calendar stops the audit. Incomplete or invalid series
remain documented as such. A successful parser or a positive estimate count
does not establish point-in-time usability or admit the source to a model.
Ten synthetic contracts were written and failed on the absent implementation
before the implementation was added; all ten then passed before numerical data
were downloaded or inspected. Freeze source/test and this contract's hashes in
the acquisition manifest before fetching the numerical endpoint.
