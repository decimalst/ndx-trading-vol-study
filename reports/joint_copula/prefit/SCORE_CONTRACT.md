# Prospective joint-copula scoring and independent verification

This wave27 component is specified and tested on invented inputs before historical access. It adds exactly two comparisons: fixed t8 copula against Gaussian copula and independence, with identical t8 marginal forecasts in all three arms. It does not add a marginal model, horizon, parameter search or secondary endpoint.

## Clock and common cohort

The scorer accepts the pipeline's long `origin` panel and the unchanged full SPX reference calendar. Each origin has exactly one row for `t8_copula`, `gaussian_copula` and `independence`. All fields except model and correlation agree exactly across those rows. Shared values include the two realized returns, conditional means and positive variances, source and fit clocks, horizon, phase, offset and training count. Independence has exactly zero correlation; other correlations lie in [-0.995, 0.995]. Numeric booleans, missing/nonfinite values, invalid clocks and unmatched rows fail the attempt.

Origins run from 2016-01-04 through 2025-10-17. Features stop at the preceding full SPX session of the origin. The target and its available date are the next full SPX session. This retains the expressly chosen extra lag between feature cutoff and predicted next-session intraday return. A monthly fit uses the full calendar session preceding its fit origin as training cutoff; it has at least 1000 training rows. Development origins and mature targets end on 2019-12-31. Evaluation is the literal 2020-01-01 through 2025-10-20 phase. All reference dates and endpoints end by 2025-10-20. Exact label-blind scheduling, complete training cohorts and forecast reconstruction belong to the separate forecast verifier; the scorer checks the supplied per-row clock identities.

## Proper score and units

For each instrument, z=(y-mu)/(sqrt(h)*sqrt(0.75)); a t8 variate's scale squared is 0.75 times its variance. The reported loss is minus the sum of the two t8 marginal log densities in original return units and the copula log density. It is a proper full joint negative log density in nats per return pair and may legitimately be negative. The reported candidate/control losses are the means of that complete loss.

The paired difference is control log-copula density minus candidate log-copula density. Shared marginals cancel analytically; the implementation does not subtract two potentially large full losses to obtain the contrast. Negative differences favor t8. A common return-unit rescaling shifts all full losses equally and preserves every paired difference. No clipping, epsilon, nonfinite replacement or tail probability truncation is permitted.

## Support, inference and decision

Support is checked for both phases and all diagnostics before resampling: at least 505 paired scored origins per phase, 63 per each of five phase offsets, and 252 per fixed evaluation slice (2020-01-01 through 2022-12-31 and 2023-01-01 through 2025-10-20). Offsets are origin positions modulo five on the original full SPX calendar, never phase-local or compressed positions.

Each phase's inference time domain contains every reference-calendar position within that literal phase, including internal missing origins and immature phase-end positions. A mask selects scored origins. It does not compress time or pad the phase with other years. Frozen `treasury_dealer_inference.masked_mean_inference` supplies masked-mean circular-block bootstrap inference at block lengths 21, 63 and 126, with 399999 draws, and HAC lag126. The phase seed is 20260909 plus phase_code*10000 (development0, evaluation1); the frozen core adds the block length. Both contrasts receive the same phase/block random draws.

Each comparison's conservative two-sided probability is the maximum across both phases and all four methods. Holm adjustment includes the two new tests at alpha0.05/(27*28), and separately all 151 cumulative tests at alpha0.05. Both probability comparisons are inclusive. Each phase's mean difference must be <=-0.005 nats per pair; every evaluation slice and each phase offset must be strictly negative. Only passing both controls produces the single `joint_copula` lead. Annual rows and reported interval envelopes are descriptive, with each envelope spanning the HAC and three bootstrap intervals.

Exactly 149 previous records are retained, including failed records, probabilities and source identities. The inheritance function appends the precision gate's three records to its 146 inherited records in their existing order and assigns the pinned precision-gate report identity. The caller authenticates its actual file hash. Any source, support, numeric, optimizer or verification failure terminates the family with two unevaluable p=1 records; the runner preserves the complete inherited family. Missing support uses INSUFFICIENT_DATA; other failures use INVALID_RUN. No partial comparison is promoted.

## APIs and report schema

Producer API: `evaluate(panel, calendar, prior, protocol)`. Family APIs: `inherit_family(previous, signature)`, `_prior(prior, count=149)`, `_probabilities(row)` and `failure_metrics(error, signature, prior=None)`.

Completed metrics have status, rows, inherited_rows, leads, hypothesis_count2, cumulative_hypothesis_count151, common_scored_origins, evidence_class and evidence_limitation. Rows are ordered Gaussian then independence and contain study/candidate/control/horizon/score identities, two phase records, conservative and both Holm probabilities, and verdict. Phase records contain mean/n/full_calendar_n, HAC and block results, conservative probability, candidate/control full losses, first/last origin, interval envelope, offsets, stability slices and annual counts. Full calendar and diagnostic counts are explicitly reported. The fixed reused-history, current-vintage and QQQ-ETF/SPX-price-index qualifications are attached to completed and failed metrics.

Independent API: `verify_scores(panel, calendar, metrics, prior, protocol)`. Success is VERIFIED with comparisons_verified2, endpoints_verified4, bootstrap_runs_verified12, hac_runs_verified4, inherited_comparisons_verified149, density_rows_verified and common_scored_origins. Failure raises and must be handled as a whole-family failure by the runner.

## Independent reconstruction boundary

The verifier does not import the scoring module. It independently reconstructs standardized returns, t8 marginal log densities, both copula formulas, all paired scores, calendar masks, diagnostics, Holm corrections and gates. Gaussian normal coordinates use an independent central hypergeometric integral and a log incomplete-beta tail representation with inverse log-normal-tail evaluation. Scaled quadratic forms avoid overflow in the t8 copula. Generated tests compare those formulas to direct distribution ratios and 100-digit arithmetic, including z of magnitude1e200, near zero and boundary correlations.

The root density module is called only as the subject being checked: every producer marginal log density, copula log density and full row loss is compared with independently established expected values. Those producer values do not establish the verifier's expected scores or inference. This explicit check enforces rowwise agreement despite the intentionally unchanged pipeline panel having no saved loss column.

The verifier reuses frozen independent `_indexed_inference` from `verify_treasury_dealer_scores`, which reconstructs circular blocks with explicit indexed observations rather than producer prefix sums. It also reuses generic independent date/numeric validation and Holm helpers. Score/density/report numeric tolerance is absolute1e-10 plus relative1e-8. Continuous HAC, conservative and Holm probabilities use absolute1e-12 plus relative1e-8. Bootstrap count probabilities, all identities/counts and inherited records are exact. Changed verdicts are rejected even when a small numeric difference lies within tolerance. Both implementations reject altered scientific protocol fields; the root authenticates the complete frozen protocol.

## Generated evidence and limits

The 11 producer and 6 verifier tests were written before their modules; SCORE_RED.log and SCORE_VERIFICATION_RED.log preserve actual missing-module failures. The first implementations passed all 17 tests without mathematical or expected-value changes. A later verifier-test edit only sorted imports and replaced an equivalent map/lambda with a generator for lint. Final receipts preserve that distinction.

Tests exercise full-density units and cancellation, negative losses, arbitrary-precision tails, shared marginal and clock corruption, full-calendar gaps and boundaries, support before inference, seeds, two-control qualification, exact discrete probabilities, continuous probability/Holm corruption, all diagnostic groups, and preservation of failed historical family records. Generated integration invokes the actual producer and independent inference cores with 31 draws through private test patches; the public scientific protocol remains fixed at399999. Other gate tests inject deterministic inference outputs so every support and qualification branch is testable without an empirical run.

This component does not rerun a calibration study. It reuses the previously tested daily masked-inference core and its synthetic serial-null calibration. That checks limited inference behavior and does not establish calibration for every copula score distribution. No historical source values, cohorts, forecasts, scores or outcomes were read for this implementation. Reused historical windows and current-vintage limitations remain explicit; neither successful verification nor a passing exploratory gate would certify untouched confirmation or investment usefulness.
