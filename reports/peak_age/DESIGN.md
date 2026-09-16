# Peak age and SPX 21-session returns

Prospective wave22 design, 2026-09-08. The experiment is unregistered. No historical numerical tables, new feature/support calculations, model fits or scores have been admitted. The completed wave21 clustering replay remains verified with no qualifying lead; all 137 preceding hypotheses, including failed families, remain in the next comparison family.

## Question and fixed experiment

Does the position of a trailing closing-price maximum add useful return forecasts after controlling for drawdown depth, the window return and the existing market state? The pre-existing [independent design review](../event_cluster_replay/NEXT_PEAK_AGE_DESIGN.md) supplies the rationale and limits. The complete literal executable contract is [peak_age.yaml](../../peak_age.yaml).

Use exactly 252 consecutive closes through the previous observed SPX session. Select the most recent exact stored-price tie at the window maximum. Divide its age in reference-session positions by251. Match it with log drawdown depth, its single square, and the window return across251 intervals. Retain all 15 original raw baseline fields and training-centered implied/realized curvature. There is one 21-session forward raw-close log-price-return target and no second horizon, peak window, threshold or model grid.

Four arms share exactly the same complete mature training rows and full application dates: training mean; original17-column baseline;20-column baseline with depth/depth-square/window-return; and its sequential peak-age correction. The16 and19 nuisance slopes use training-only population standardization and ridge.01; each declared scale must exceed1e-12. No unused hinge is calculated, required or fitted. The depth model is fitted first and frozen; the single age coefficient uses its current-fit in-sample residuals, raw training-centered age and ridge.01. No extra intercept, joint refit, age standardization or historical-error-memory substitution is allowed. Constant training age or exactlyzero numerator gives canonical beta0 and exact copied depth predictions.

The sequential coefficient may adjust ridge shrinkage even when age lies in the existing nuisance span. A passing result would therefore establish model-relative forecast improvement, not a new independent information source. A generated counterexample explicitly tests this distinction.

## Source, clocks and complete applications

The admitted sources are the existing pinned SPX raw daily OHLC and Cboe VIX/VIX9D/VVIX extracts. Their complete provenance and prior-result closure is anchored to wave21's verified publication. New admission decodes only checked source snapshots bounded at2025-10-20. Numerical observations from2025-11-03 onward remain outside the experiment. Original archival-vintage, exact-publication-latency and VIX9D backcalculation limitations remain.

Keep the full observed reference calendar before any missing-value filtering. A feature window never fills, compresses or changes length around missing observations. Arithmetic faults on fully observed inputs fail the trial; they cannot silently remove a row. New log ratios use the bounded-mantissa algorithm in the literal protocol and are tested against an independent high-precision oracle, including extreme and adjacent floating-point prices.

Every monthly fit begins on its first common-feature-complete application within the fixed origin/phase fences, before checking that query's eventual label. Training requires at least1000 earlier common-complete rows whose labels have matured by the previous reference session. All four arms refit on those same rows. Save each fit's complete training/application date lists and every application prediction, including the unscored final applications. Only afterward select the phase-eligible, mature scoring subset. Missing labels, future prices or an unscored final month cannot silently change the historical fit schedule.

Date comparison accepts only native naive normalized-midnight ms/us/ns representations, exact instants and unknown masks, with checked lossless nanosecond conversion and roundtrip. Numeric values/dtypes and integer peak metadata retain their declared checks. No old source or prior forecast is transformed or replaced by a new control.

## Inference and acceptance

Register exactly three contrasts: peak_age against baseline, depth and mean, all at horizon21. Inherit137 prior hypotheses for cumulative140. All three controls must show at least0.25% relative MSE improvement in both phases, negative paired differences in both fixed later slices, and negative nonempty differences for every offset0..20 in both phases. Offsets use positions in the full original calendar before filtering; no favorable offset may be selected.

Each phase requires505 paired observations. Preserve blocks126/252/504 and Bartlett HAC504,399999 bootstrap draws, seed20260928 and the specified horizon/phase/block offsets. The conservative hypothesis p is the maximum across dependence checks and both phases. Require Holm3 below.05/(22*23) and cumulative Holm140 below.05. The simulation resolution is below one tenth of the strictest raw wave cutoff. Intervals, annual summaries and power diagnostics remain descriptive.

A successful ordinary ledger has143 events: three registrations,137 inherited and three evaluated. Register before fallible historical source enumeration/admission. Any source, feature, support, numerical, fitting, scoring or publication failure retains all three hypotheses as unevaluable p1. A later independent failure preserves attempted evaluated history and appends three verification-failed records. No thresholds, windows, columns or solver tolerances change after outcome access.

## Independent verification and prewritten evidence

Feature/model, scheduling, inference and transaction contracts were written before their corresponding implementation. Missing-module and subsequent synthetic failure logs remain retained. All current code, protocol and prefit evidence must be frozen after full repository tests and before registration.

The independent numerical implementation reconstructs the full feature/target/state tables, monthly common cohorts, nuisance objectives and all scalar corrections. It uses augmented least squares rather than the producer's normal equations and closed scalar formula. Full generated pipelines pass through Parquet/JSON before reconstruction under ms/us/ns source dates. Unscored predictions and exact peak/tie/missing metadata are included in tampering tests. Independent entry inference uses the separately implemented existing bootstrap/HAC/statistical path and verifies complete trial identities and source/output hashes. Shared source traversal is disclosed rather than described as independently authored provenance logic.

Root scheduling's first implementation incorrectly assumed a NumPy datetime dtype exposed a `.unit` attribute. The11 prewritten pipeline tests and independent generated integration both caught this before any historical access. Unit extraction now uses `np.datetime_data`; the original RED log remains. A pure-inference fixture also used scalar exponentiation for stored loss while the declared panel computes one multiplication; a one-bit rounding difference was corrected in the invented fixture, keeping the exact saved-loss check and model/inference rules unchanged. These are prefreeze synthetic corrections, not empirical repairs.

Detailed responsibilities and evidence are recorded in FEATURE_MODEL_DESIGN.md, SOURCE_DESIGN.md, VERIFICATION_DESIGN.md and the independent entry/publication reviews. Passing synthetic checks establishes implementation coverage, not historical support or predictive success. The actual registered run and independent verification remain necessary.

Final integration review found direct current-output writes that could follow a symbolic link into an existing artifact. Four additional generated publication regressions exposed nine sentinel-preservation failures before the repair. Current report text, check evidence, interrupted-ledger backups and Parquet outputs now use a temporary regular file followed by atomic destination replacement, extending the existing JSON and ledger protection. The 46-test root scheduling, inference and publication suite passed after this correction. These are pre-empirical implementation regressions; no statistical rule, historical observation or fitted result motivated the repair.

## Interpretation

A rolling maximum can expire without a new high. Peak age is not an all-time-high duration or a verified economic recovery state. SPX price returns exclude dividends and risk-free subtraction and do not specify a feasible trade. Repeated archival selection and overlapping targets remain limitations even if every gate passes. A favorable historical result would need genuinely future confirmation.
