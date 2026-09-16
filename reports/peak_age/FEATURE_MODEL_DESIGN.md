# Peak-age feature and model contracts

Wave22 generated feature/model implementation follows the literal prospective
`reports/event_cluster_replay/NEXT_PEAK_AGE_DESIGN.md` before historical admission.
No historical numerical table, support count, fit, or score is read here.

`src/peak_age_features.py` exposes original `RAW`15, `BASE`17, `COMMON`19,
`stable_log_ratio(x,y)`, and `build_features(daily,iv)`. The builder returns three
full-reference-calendar DataFrames: features (`COMMON` then cutoff date), the
sole21-session target, and peak-window state. New columns are `peak_age`,
`drawdown`, `drawdown_sq`, and `window_return`. Raw baseline formulas keep the
original lag and arithmetic. Fully observed arithmetic failures raise rather
than becoming missing rows. Genuine source NaNs retain their original positions.

State positions and integer age use nullable Int64. Origin position always
exists; cutoff exists after row0. Window endpoints exist from row252 even for
missing windows, while peak position/date/age are unavailable until all252
closes are present. This exposes uncompressed missing windows. Date columns and
indexes are normalized to nanosecond DatetimeIndex representation after accepting
equivalent generated ms/us/ns input dates. Exact timestamp equality is semantic
calendar equality, never equality of raw integer storage units.

`src/peak_age_models.py` exposes `transform`, `fit_scalar`, `scalar_objective`,
and `fit_predict`. The final fit requires at least1000 aligned common finite
rows; primitive transforms/scalar fits accept two generated observations for
analytic contracts. Prediction arrays align exactly to the supplied application
index. Four keys are returned: `predictions`, `model_audit`, `transform_audit`,
`scalar_audit`. Root scheduling records exact common training/application dates.

Model audits for mean/baseline/depth contain columns, means, scales, beta, alpha,
train_n, objective, full half-objective gradient, and maximum absolute gradient.
The baseline has16 nuisance slopes; depth has19. Curvature centers are fitted
once on common training rows. Population means/scales remain the original
float64 conventions. Every declared nuisance scale must exceed1e-12.

Scalar audits contain coefficient, age_center, age_scale1, alpha.01, train_n,
numerator, second_moment, denominator, objective, half-objective gradient,
maximum absolute gradient and status EXACT_CONSTANT_INPUT/BALANCED_AT_ZERO/FITTED.
The scalar stage freezes the current fitted depth model, retains centered raw
age scale1, and solves the specified one-dimensional penalized correction.
Zero coefficients copy depth predictions exactly. The generated in-span-age
counterexample expressly demonstrates that this is a sequential shrinkage
adjustment and does not establish an orthogonal information channel.

Prewritten tests cover literal windows/ties/expiry, one-origin algebraic
distinction, full-calendar missingness and sole21 targets, source timing,
separate Decimal160-digit log-ratio oracles, reversible binary scaling,
generated ms/us/ns Parquet and JSON transport, independent augmented ridge and
scalar least-squares calculations, all nuisance scales, common1000-row support,
constant/balanced stages, finite arithmetic, and training-only transformations.
Required saved/independent half-gradients are at most1e-10; coefficients compare
at relative1e-7/absolute1e-12 and predictions at relative1e-8/absolute1e-12.
Generated high-precision log comparisons use relative7e-16 and zero absolute
tolerance, with exact sign/nonzero/equality requirements. No empirical result
may change these definitions or tolerances.

Before admission, the generated arithmetic-failure fixture was corrected to
pair min-subnormal and max-float closes: a lone max-float close against an
ordinary prior close still has a finite original ratio. The algebraic
matched-controls test uses the stated reconstruction tolerance because the
unchanged pandas rolling accumulator can retain about1.7e-17 of history
roundoff after an altered old return expires. New D/D2/U and peak/tie identities
remain exact. Neither correction uses historical observations.
