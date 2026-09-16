# Peak-age final publication review

Status: REVIEWED_VERIFIED_PEAK_AGE_PUBLICATION, 2026-09-08. The final summary and figure accurately describe the independently verified experiment: no registered comparison qualified and no lead is promoted.

This review inspected SUMMARY.md, metrics.json, verification.json, the actual PNG, freeze/test evidence, the exact ledger and artifact identities. It did not rerun fitting, forecasting, bootstrap/HAC inference or tests, and decoded no numerical source or forecast table. Byte hashing of retained artifacts is distinct from numerical reconstruction. The existing VERIFIED proof supplies the mathematical verification.

## Results and interpretation

All three peak_age contrasts are present at the sole 21-session horizon. Relative MSE gains, in percent, agree with the saved metrics:

| Control | Development | Evaluation | Wave/cumulative Holm p |
|---|---:|---:|---:|
| Original market baseline | +3.962295% | -10.920809% | 1 / 1 |
| Matched depth/window-return model | -0.179647% | -0.604370% | 1 / 1 |
| Training mean | -3.760081% | -11.510409% | 1 / 1 |

The age correction slightly increases measured error against its matched parent in both periods. Both reported interval envelopes include zero; the summary correctly avoids claiming reliable deterioration. The standalone evaluation block-504 interval is positive, which is why the final wording explicitly refers to the envelopes. Improvement over the original baseline in development reverses in evaluation and cannot identify an age effect independently of the added depth/window-return controls. The +0.25% effect requirement and every registered control, period, stability, dependence and multiplicity gate remain in force.

The final wording also correctly distinguishes preceding-session market inputs from known entry-calendar information. The stated overlapping-target, archival-vintage/VIX9D-backcalculation, reused-history, price-index and execution limits agree with the protocol and proof. The sequential coefficient can adjust ridge shrinkage within the nuisance span; neither this result nor this publication review establishes an orthogonal information channel.

## Counts and preservation

The verified common cohorts are 985 development origins, 2016-01-04 through 2019-11-29, and 1,437 evaluation origins, 2020-01-02 through 2025-09-19. The proof certifies 9,688 scored predictions and squared losses across 2,422 origins; 9,852 full application predictions across 2,463 origins, including 41 unscored applications; 118 monthly schedules; 236 nuisance fits; 118 scalar corrections; and 118 training means. It certifies six phase comparisons, 18 bootstrap runs with 399,999 draws each, HAC504, 126 offset checks and the 3/140 new/cumulative family.

The ledger was compared to complete canonical registration/inherited/evaluated rows, not merely counted: 143 ordered records, with roles 3/137/3. Every current output in the exact ten-path verification set matches its recorded hash. The manifest's 336 code and 2,199 input hashes, all 75 prefit hashes and the complete current Python inventory match. Its separate preserved group is empty because preceding artifacts are already included among inputs. All 319 preceding Python files and all 55 wave21 publication report artifacts remain byte-exact under the pinned prior publication.

The old wave20 canonical failure/metrics/verification trio remains unchanged, including its exact FAILED error record. Its three canonical conservative/wave/cumulative p-values remain one, and all three inherited rows retain conservative p=1 with the correct canonical source hash and row index. No current or required successful index-hinge/wave18/wave19/wave21 failure marker is present, including dangling links. Final old-failure hashes and marker absence were checked again after the general identity checks.

The frozen full-suite log records 2,378 tests passing in 384.561 seconds. The separately bound pre-run log records 154 tests passing in 66.211 seconds. These are inspected execution records, not new test runs by this reviewer. Protocol, manifest, freeze, executing-verifier and output identities agree with the independent proof.

## Figure and outcome

The actual comparison_intervals.png was visually inspected. All six effects, three control labels, both period colors, nominal interval envelopes, percent axis, +0.25% gate, legend and limitations are visible without clipping. Directions and approximate magnitudes match the saved effects; the chart does not conceal the unfavorable comparisons. PNG and PDF identities are bound with the final summary, metrics, verification and this review in publication_review.json. The PDF was hash-checked; visual inspection was of the PNG.

No unresolved publication error remains after the two precise summary wording corrections. The separate final publication-audit script may now check and seal these artifacts. This approval concerns faithful reporting and artifact integrity; it adds no empirical hypothesis and makes no trading or future-confirmation claim.
