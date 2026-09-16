# Wave27 saved scientific result review

Review outcome: **the two saved comparisons satisfy the predeclared gates, supporting a qualified exploratory joint-density lead.** No discrepancy was found in the reviewed effects, uncertainty, subgroup support, family arithmetic, fitted metadata or density interpretation. The appropriate practical statement is: *On the reused 2016–2025 history, a fixed t8 dependence model assigned better joint predictive densities to next-session QQQ and SPX intraday return pairs than either separately fitted Gaussian dependence or independence, using identical conditional marginal forecasts.*

This is an additive review of saved artifacts and frozen formulas after the terminal COMPLETED result. I read the metrics and both verification proofs, checked saved fit metadata, compared exact file hashes and reconciled reported counts and gate arithmetic. I did not read raw market arrays, refit models, recompute forecasts or row densities, rerun tests, or resample inference. The existing computational verification proof remains the evidence for those operations. This review does not replace the separate source-preservation and registration audit.

## Effects and uncertainty

Differences below are candidate minus control **full joint negative log density**, in nats per observed return pair. Lower is better. Both phase differences must be <=-0.005.

| Control | Phase | Paired origins | Mean difference | Reported 95% interval envelope |
|---|---|---:|---:|---|
| Gaussian copula | Development | 1005 | -0.030568715957402973 | [-0.04125171516797149, -0.01988571674683445] |
| Gaussian copula | Evaluation | 1457 | -0.031017081500978507 | [-0.04117759022050474, -0.02085657278145227] |
| Independence | Development | 1005 | -0.6067968120341651 | [-0.7043435042556597, -0.5092501198126705] |
| Independence | Evaluation | 1457 | -0.7469353249639794 | [-0.8422456481825161, -0.6498442246294044] |

Every interval envelope remains below zero and below the -0.005 effect threshold. Each envelope spans that endpoint's HAC126 and block21/63/126 nominal 95% intervals. These are reported interval envelopes, not simultaneous confidence bands adjusted for the full research search.

For **each** comparison, the maximum two-sided probability across both phases and all four inference methods is **0.0000025**. All twelve block probabilities equal that simulation floor, 1/(399999+1); they should not be interpreted as exact probabilities known below that resolution. The four HAC probabilities are smaller: Gaussian development2.0418901363320526e-08, Gaussian evaluation2.186472233666833e-09, independence development3.4150268656316497e-34, and independence evaluation3.0226705932620458e-53. Using the maximum retains the more conservative stored probability.

Both Holm-wave probabilities are **0.000005**, below the inclusive wave threshold **0.00006613756613756614**. Both Holm151 probabilities are **0.0003775**, below **0.05**. I independently reconstructed those two Holm adjustments from the saved probabilities, including all149 inherited rows. Both verdicts are COMPARISON_GATE_PASS, and the saved sole lead is joint_copula. The family contains exactly two new comparisons and151 cumulative comparisons. The terminal metrics hash binds this result.

## Sample and subgroup coverage

All three arms use the same **2462 scored origins**, producing7386 scored model rows. Development spans scored origins2016-01-04 through2019-12-30:1005 pairs within1006 literal phase-calendar positions. Evaluation spans2020-01-02 through2025-10-17:1457 pairs within1458 positions. The development phase-end origin with a target beyond its fence and the final unissued evaluation-calendar position remain in inference's uncompressed time domain. Global offsets use the original full SPX calendar, not a calendar renumbered after missingness.

Every development offset has201 pairs. Evaluation offsets0/1/2 have291 pairs each, and offsets3/4 have292 pairs each. These exceed the63 floor. Every offset mean is strictly negative for both controls in both phases; for the stronger Gaussian comparison, development offset means range from -0.035124879147211636 to -0.02735536102875954, and evaluation means from -0.03872331631145984 to -0.027338664563368954.

The fixed2020–2022 slice has756 pairs and the2023–2025-10-20 slice has701 pairs, exceeding252 each. Gaussian-comparison means are respectively **-0.02466274254157803** and **-0.03786997772538187**. Independence-comparison means are **-0.6922714067894689** and **-0.8058881382877026**. The annual rows reconcile with both phase totals and their full-calendar counts. Their signs are also negative throughout, but annual signs are descriptive and did not add tests or alter the gates.

The saved forecast proof records4226 reference-calendar rows,2463 coverage origins,7389 issued application forecasts,118 monthly fits,472 marginal-model fits and236 dependence fits. The fit artifact contains118 fitted months,2016-01 through2025-10, with1254–3704 common training rows. Every saved training count matches both stored training identity lists. Planned and actually issued query identities match in every month, totaling2463 applications. All118 records explicitly identify current-fit training residuals and both separately fitted dependence arms. The initial fit origin2016-01-04 has training cutoff2015-12-31; the last fit origin2025-10-01 has cutoff2025-09-30. The missing one scored origin was retained among issued applications, as required.

The independent forecast proof reports a largest reconstructed global objective gap of9.974728820694168e-09, within the declared1e-8 bound. This is the saved floating-point numerical certificate, not an exact-arithmetic global proof. This review did not reproduce optimizer messages or certificates. The separate score proof reports verification of7386 row densities, two comparisons, four phase endpoints, twelve bootstrap runs, four HAC runs and149 inherited comparisons. Its subproof exactly matches the combined verification artifact.

## Why this is a proper joint-density comparison

All arms share locations mu, variances h and fixed t8 marginal distributions. The scale sqrt(0.75h) gives varianceh because a standard t8 variable has variance8/6. Each arm's original-return-space density is c(F1(y1),F2(y2))*f1(y1)*f2(y2). The Gaussian copula code uses its bivariate normal density divided by the two univariate normal densities after the t8-CDF-to-normal transform. Those denominator terms, together with the retained original marginal densities and scale factors, account for the transformation; it is not an unnormalized normal likelihood applied directly to transformed data. The t8 arm similarly divides its bivariate t8 density by its univariate t8 marginals. Independence has copula density1; t8 with correlation0 would still not be independence.

Consequently, the reported full losses are proper log scores in return units. The negative full-loss levels are valid for continuous densities; they are not negative probabilities. Their paired differences equal control log-copula minus candidate log-copula because the shared marginals cancel. The saved phase full-loss subtraction agrees with each reported mean difference. Changing return units shifts both complete losses equally and cannot create this contrast. A percentage improvement obtained by dividing these negative, unit-dependent full-loss levels would not be an appropriate effect interpretation.

The Gaussian comparison is the substantive matched control: it has its own likelihood-fitted scalar correlation, the same data, the same marginal forecasts and the same schedule. The larger independence gain alone would mainly establish that joint dependence matters. The smaller but consistent Gaussian gain is evidence favoring this fixed heavy-tailed dependence family for these joint predictions. It does **not** isolate a pure tail-dependence mechanism, because changing the copula changes the entire joint density.

## Claim limits and practical use

The conditional marginal distributions may be misspecified even though they are identical across controls. A more suitable dependence family can compensate for systematic marginal errors, including remaining common scale variation. The staged in-sample residual construction is explicitly saved; it is not a sequence of historical issued forecast errors. Neither identity matching nor proper normalization proves marginal calibration or identifies a structural economic source of tail dependence.

This experiment leaves each market's marginal mean and variance unchanged across arms. It therefore establishes no improvement in stand-alone volatility or directional forecasts. The supported use is an exploratory candidate for **joint return-scenario and dependence modelling**. Portfolio loss calibration, tail-event frequency, VaR/expected-shortfall coverage, hedging performance and net trading profit were not tested by this endpoint.

The development and evaluation windows have been reused. Prior regression tests accessed the period starting2025-11-03, so that period is not untouched confirmation. The present numerical input ceiling is2025-10-20. Current vendor snapshots do not certify complete historical price or implied-volatility vintages; VIX9D back-calculated training history remains declared. QQQ is an ETF and SPX a price index, with exact synchronized auction measurements and original publication latency unverified. Predictors stop at the preceding SPX session of each origin, and the target is the following SPX session's intraday return pair, preserving the conservative extra lag.

Passing the registered multiplicity gates is meaningful evidence within this recorded family; it does not turn adaptively reused history into independent confirmation. The reused daily-mask synthetic calibration addresses limited serial-inference behavior, not universal market-score coverage or the precision of extreme multiplicity-adjusted tails. The appropriate next evidential step would be prospective confirmation under a fixed joint-risk task, without relabeling this result as a newly discovered marginal-volatility signal. This review starts no new experiment.

## Exact saved artifact pins

The following SHA-256 digests were read directly for this review. The terminal report pins matched their current bytes, its output hash map matched the combined verification output map, and the fit artifact matched its recorded output pin. Full old-history preservation is assigned to the separate publication audit.

| Artifact | SHA-256 |
|---|---|
| joint_copula.yaml | cf43700c6c37ba860e9ef8ae4adb4c70158030efeedd3bca9b073f0691cd5d50 |
| reports/joint_copula/prefit/SCORE_CONTRACT.md | ca26abf44cfc67ce10d207b50697bbcc33a5291849702909d98b620856650053 |
| reports/joint_copula/predictive/metrics.json | d1de728c8fb6909a4ce8ecd6edb055d8b3652693550de140dfca3ff688593442 |
| reports/joint_copula/predictive/verification.json | 6abe3595a307ba61860093b12d891696d95ca41451f68397ad470f8840d907d4 |
| reports/joint_copula/predictive/forecast_verification.json | a712dc8016689c0df0b3a2c8a4eedc9e3f995563baa80d408839851a65052dc2 |
| reports/joint_copula/predictive/score_verification.json | e37fa59942dae4249a0ea41a31eff022916b14f929b41dfe5cb44bf4c1239639 |
| reports/joint_copula/predictive/terminal.json | 9dfbe34029b5b8095fd68ff0c3e1e624bd854de287449a1045b6700906742b2f |
| data/model_memory_study/joint_copula_wave27/fits.json | be070e131e96faa7939be052838beb16594825a1afe451c5deb01448fe1bae91 |
