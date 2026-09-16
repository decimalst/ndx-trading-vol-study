# Overnight index returns: completed third wave

No overnight-return candidate qualified. Signed cross-asset returns, signed
volume pressure and implied-volatility shape all performed worse than the
historical-mean forecast in both development and evaluation. The result is
unchanged in the fixed sensitivity excluding adjustment-factor changes.

Positive numbers below mean less squared forecast error; negative numbers mean
more error. These are forecast comparisons, not trading returns.

| Addition | Development vs market model | Evaluation vs market model | Development vs historical mean | Evaluation vs historical mean |
|---|---:|---:|---:|---:|
| Signed cross-asset returns | −0.358% | −0.194% | −1.705% | −0.912% |
| Signed volume pressure | +0.014% | −0.188% | −1.328% | −0.907% |
| Implied-volatility shape | −0.097% | +0.049% | −1.440% | −0.668% |

The comparison covers 2,462 entry dates, 2016-01-04 through 2025-10-17. All
market inputs use the previous completed session's daily observations, making
the information cutoff compatible with a decision before the entry closing
auction. This is a previous-session information state, not a certified 16:00
snapshot of Cboe's end-of-day fields. Weekday
features describe that known entry date. No realized future opening date enters
a predictor.

The baseline includes separate past overnight and daytime returns, realized
and implied volatility, and weekdays. Every model shares complete training
and evaluation rows. Ridge scaling and fitting are historical-only; each
label must be available by the fit's previous-session information cutoff.
The first fit has 1,254 eligible training rows, above the fixed 1,000 minimum.

The target is a vendor-adjusted overnight-return proxy. Its dependence on the
next closing price cancels when the adjustment factor is held fixed, but its
corporate-action convention differs from exact cash profit. Label availability
therefore remains conservatively assigned to the next session close. All 39
flagged adjustment changes remain in the primary sample; the extra sensitivity
uses identical forecasts and is never a trading filter. Cross-series field
adjustment provenance and VIX9D's back-calculated early history are documented
in the source audit.

The independent verifier reconstructed **12,310 forecasts across 118 monthly
fits**, all 6,696 rows of 25 source-derived predictors, all 12 phase comparisons,
36 bootstrap runs, 12 adjustment sensitivities and the full 78-comparison
cumulative correction. Verification passed with unchanged frozen source and
tolerances. The complete pre-run repository suite passed **665 tests**, and
the new source and tests pass scoped lint.

The three recent waves now contain **24 new comparisons and 60,053 independently
reconstructed forecasts**. None produced a qualifying new signal. The 78-count
cumulative family also includes the 54 previously enumerated modeling and
orthogonal-input comparisons; it does not conceal the separately inventoried
older repository search. Historical reuse remains exploratory.

- [All six comparisons and measurement sensitivities](results.md)
- [Numerical estimates and uncertainty](metrics.json), [figure](comparison.png), and [PDF](comparison.pdf)
- [Independent verification](verification.json) and [full pre-run tests](full_pre_run_tests.txt)
- [Source audit and limitations](SOURCE_FEASIBILITY.md)
- [Frozen protocol](../../overnight_index.yaml), [manifest](manifest.json), and [trial ledger](trial_ledger.jsonl)
- [Previous international-volatility wave](../international_volatility/SUMMARY.md)
- [Original modeling and supplied-reference experiments](../model_memory_study/SUMMARY.md)

The broader search remains active. A dated seven-market Oxford supplement
through January 2020 was recovered; its overlap agrees within predeclared
numerical tolerance, with small representation differences rather than exact
reproduction. A separate SPY high-frequency source extends through the permitted
October 2025 cutoff, subject to documented missing dates and measurement/timing
limitations. Four official advance-announcement examples were also captured
and independently checked. None of these source findings is a predictive result.

- [Oxford overlap audit](HARNET_EXTENSION_NUMERIC_AUDIT.md), [independent verification](HARNET_EXTENSION_VERIFICATION.md), and [documentary limitations](HARNET_EXTENSION_PROVENANCE.md)
- [SPY measurement audit](RISKLAB_MEASUREMENT_AUDIT.md) and [independent verification](RISKLAB_MEASUREMENT_VERIFICATION.md)
- [Overnight second-moment proposal](MACRO_OVERNIGHT_VARIANCE_FEASIBILITY.md) and [official-plan capture route](BLS_PLAN_CAPTURE_FEASIBILITY.md)
- [Next experiment and readiness requirements](NEXT_WAVE.md)
