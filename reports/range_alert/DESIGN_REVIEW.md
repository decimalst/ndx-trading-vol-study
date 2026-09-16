# Independent prefit design and publication review

Status: **NO REMAINING MATERIAL FINDING IN THE REVIEWED CONTRACT**. This is a prefit code/design review, not approval of historical results or a claim that the experiment has passed independent reconstruction. Full-repository checks, the parent’s freeze, registration and the new independent historical verification remain separate requirements.

The review covered the new protocol, runner, model schedule, plotting admission, publication tests and their integration contracts. It used source code, prospective documents, hash metadata and synthetic test evidence only. It did not parse historical numerical sources, construct historical event labels, inspect new event counts or associations, fit historical models, or read new scores. The reviewer implemented the new source/feature module; independent reconstruction of that module’s mathematics belongs to the separate verifier author. No review statement here substitutes for that independent check.

## Identifiable question and limits

The event is next-session SPX daily risk greater than twice the mean risk of the **22 sessions ending at the prior-session cutoff**. It is not doubling from the immediately preceding session. Risk is the declared floored Garman–Klass plus raw overnight-square proxy, not measured high-frequency variance. The exact compensated reference and strict inequality fix boundary labels before inspection.

Previous closing-range extremity can differ for bars with identical high–low range and open-to-close return. That establishes a distinct bar statistic, not statistical orthogonality or an order-flow mechanism. The declared baseline already includes daily risk, IV, leverage, return-history and weekday controls, three training-centered curvature terms, and signed intraday/overnight returns and their squares. The candidate changes one centered extremity coefficient around the frozen fitted baseline logit. It cannot change the baseline coefficients. Exact constant extremity retains a zero correction and its fold; collinear baseline columns are retained under ridge. There is no association-driven feature removal or rank-based rescue.

The frequency comparison matters: it tests whether the candidate improves on a continuously updated event-rate estimate as well as the conditional baseline. Brier improvement is a proper probability-score improvement for this event, not a probability-point increase, variance-magnitude forecast gain, trading return or profit. Reused archival histories and adaptive selection of this next question prevent an untouched-confirmation claim. Original publication latency and vintage caveats, including early back-calculated VIX9D history, remain explicit.

## Timing, common cohorts and immutable sources

The model schedule selects each month’s first feature-complete application before applying the future query-label mask. Historical training requires complete common raw features, a known cutoff, a known binary label and maturity no later than that fit’s preceding observed SPX close. Each fit requires 1,000 rows and 50 examples of each class. All folds, application transforms and phase/slice support are checked before the first optimizer call. An unsupported late fold aborts the whole experiment. Unscored applications still require valid conditional forecasts and a saved frequency state.

The recent-frequency filter starts from the first common training mean at the initial application cutoff, then advances on every full reference-calendar close. It never re-adds labels already available at the seed cutoff and never resets across monthly fits, phase boundaries or feature gaps. Known labels from feature-incomplete or unscored origins enter only on their declared availability date. Unknown labels contribute no update. This is deliberately a broader subsequent label stream than the complete-case fitting cohort, with the same causal clock.

The runner checks the protocol from one byte payload and verifies the freeze before and after pretests. Registration precedes new source decoding and all historical feature construction. Each of the 11 declared source/documentary files is read once, checked against both the protocol hash and registration input pin, and copied from those same bytes into an isolated source tree before the frozen loader runs. Provenance records, raw-versus-derived SKEW checks and source-date fences are retained. There is no new data acquisition or vintage substitution. Prior successful studies are anchored through unchanged proof records and artifact hashes; this wave does not rerun their historical models.

## Fixed inference and failure accounting

There are exactly two new Brier contrasts: location versus baseline and location versus recent frequency. All 127 inherited comparisons, including the two original failed quarter-end attempts, remain in the cumulative family of 129. Each new contrast uses both development and evaluation; promotion requires at least 0.0005 absolute Brier reduction in both phases, a negative difference in each of the two fixed later slices, wave Holm significance at `0.05/(18*19)` and cumulative Holm significance at 0.05. Both controls must pass. The six phase/model calibration summaries are descriptive and add no alternate promotion route.

The 199,999 bootstrap draws give minimum attainable p-value `1/200000`, below one tenth of the strictest raw two-comparison wave cutoff. Blocks 21/63/126, HAC lag 126, seed 20260924 and the conservative maximum across declared phase tests are fixed. The nominal 80% minimum-detectable-effect diagnostic does not establish equivalence, multiplicity-adjusted power or an alternate acceptance gate.

The review identified a publication defect before freeze: a failed inherited read or partial registration/evaluated write could leave an incomplete or malformed ledger even though canonical metrics became p=1. The corrected runner renders ledger JSON before an atomic replacement and preserves pre-recovery bytes separately. Recoverable producer failures rebuild a terminal ledger from two registrations, the captured validated inherited rows when available, and two unevaluable p=1 outcomes. Complete-history success or producer failure has 131 events. If the inherited snapshot was unavailable, the four explicit new registration/terminal events remain, with `inherited_rows_available=false` and zero reconstructed inherited rows; 127 missing records are not fabricated. Diagnostics cannot replace the canonical unevaluable metrics. The 38 root search/publication/plot tests pass, including early inherited-read, partial registration, serialization, report and partial evaluated-write faults.

The plot entry point refuses a failure record and requires the current verified protocol identity, complete two-control/four-phase cohort, matching reconstruction counts and the exact metrics-byte hash saved by independent verification. The two controls use separate horizontal scales, literal confidence endpoints and the fixed −0.0005 reference. A single passing control cannot produce a lead. Synthetic render tests support this review; no historical figure was inspected.

## Reviewed identity and evidence

The current inventory has **12 new Python files**: five source modules and seven test modules, bringing the total to 279. A read-only hash comparison against the prior profiled-quarter manifest checked all **267 prior code files**, with no mismatch or missing path. This is a code-preservation check only; the parent owns the complete prior source/report preservation proof.

The reviewed non-owned implementation identities are:

| File | SHA-256 |
|---|---|
| [range_alert.yaml](/Users/byrons/code/trading-vol/ndx-vol-experiment/range_alert.yaml) | `ff7b2a7fee497da11f1f88613a17314d3b92d683f506bc4cc51d5089d2bc16a3` |
| [range_alert_search.py](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/range_alert_search.py) | `2697504b0c158857f2df11d1c079b9b18c888a26d56942ce951b441d402e41b8` |
| [range_alert_models.py](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/range_alert_models.py) | `2a8288a9b8026edd4f9d80d1b5ed192850cb4b8dd4fa92e6619177855e66f632` |
| [plot_range_alert.py](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/plot_range_alert.py) | `4aa3aee15a38df6480f7a9fa36b9541e58a6080efa71706bb4f044f553c9c423` |
| [DESIGN.md](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/range_alert/DESIGN.md) | `c1720f933480135dc8084fe4a5ce012dea3ee317e85f83bcdfa49873a7d2d7ec` |

The feature author’s separate [source/feature design](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/range_alert/SOURCE_FEATURE_DESIGN.md) records its tests-first implementation and exact APIs. Its 22 checks passed; the model author records 21 model checks. The final recovery evidence is [root_recovery_checks.txt](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/range_alert/root_recovery_checks.txt), SHA-256 `2a39bced47f2e1d523e394049c5a9ec786df71a8c6fe783aada428a58af47e65`. Independent mathematical and integration tests continue under their respective owners before the parent’s full freeze. No historical run or favorable result is implied by this review.
