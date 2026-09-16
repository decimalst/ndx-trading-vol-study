# Model weights conditioned on current risk

**No new qualifying predictive improvement.** This fixed experiment compares a state-dependent precision blend against the market baseline, its adaptive counterpart, and a fitted constant blend. Both experts use the same twelve market predictors. The state is daily minus monthly log realized risk; the combination stays between the experts.

| Control | Period | Paired dates | QLIKE difference | 95% interval envelope |
| --- | --- | ---: | ---: | --- |
| Market baseline | 2016–2019 | 1,001 | +0.019423 | [+0.001138, +0.039608] |
| Market baseline | 2020–Oct 2025 | 1,453 | +0.002861 | [-0.000842, +0.007156] |
| Adaptive model | 2016–2019 | 1,001 | -0.004411 | [-0.011403, +0.001494] |
| Adaptive model | 2020–Oct 2025 | 1,453 | -0.010679 | [-0.025879, +0.003013] |
| Fixed blend | 2016–2019 | 1,001 | -0.000037 | [-0.000545, +0.000492] |
| Fixed blend | 2020–Oct 2025 | 1,453 | +0.000260 | [-0.000078, +0.000681] |

Negative differences mean lower forecast loss. A lead requires an improvement of at least0.005 in both phases against all three controls, negative differences in every fixed later slice and calendar offset, all support floors, and both wave and cumulative multiple-testing corrections. Intervals shown here are unadjusted diagnostic envelopes; individual cells do not establish a lead.

![Saved model-blend estimates and uncertainty](comparison_intervals.png)

The common scored sample contains **2,454 origins**, including **39 cold-start origins** and **2,415 origins with fitted gate weights**. Cold starts use baseline-valued gate forecasts and remain included. Expert forecasts were newly issued with monthly target-blind schedules. Each gate fit uses only previously issued forecast pairs whose outcomes matured by the preceding full-calendar session, within the fixed1260-session window. It never recalculates old forecasts using a newer expert.

All three new comparisons retain the146 inherited entries, giving **149 registered comparisons**. Previous unevaluable attempts remain counted. Verification independently reconstructs market inputs, targets, every expert fit, gate membership and optimum, scored and unscored applications, and the full-calendar paired inference. The independent gate optimizer differs from the producer; solver iteration counts are range checked rather than claimed to reproduce exactly.

## Qualifications

Historical development and evaluation windows have been reused. Prior regression tests accessed the period beginning 2025-11-03; it is not untouched confirmation. Current vendor snapshots do not certify complete point-in-time price or implied-volatility vintages. Numerical inputs in this experiment end 2025-10-20.

The exact six historical-replay regression tests are excluded from this run with their original files and prior-access correction preserved. The existing synthetic inference calibration is reused for the identical daily-mask procedure; it does not establish exact market interval coverage or adjusted-tail calibration. This model experiment introduces no Treasury auction quantities or release clocks. Historical nonqualification is not proof of equivalence or proof that all conditional combinations are useless. A historical lead would still require future confirmation and separate trading evaluation.

Evidence: [metrics](metrics.json), [independent reconstruction](verification.json), [freeze](freeze_record.json), [selected tests](TEST_GATE_RESULT.json), [registration journal](trial_ledger.jsonl), [terminal record](terminal.json), and [PDF figure](comparison_intervals.pdf).
