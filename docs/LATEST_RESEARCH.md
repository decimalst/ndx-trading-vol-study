# Follow-up research index

Status reviewed **2026-09-16**. This index is additive: original protocols,
failed runs, corrections and frozen findings retain their recorded bytes and
verdicts. Intermediate plans can predate completed results; the links below
point to the latest relevant outcome.

No new standalone volatility predictor has cleared the relevant study gates.
Wave 27 produced an exploratory joint-density lead under shared marginals.
The subsequent attribution experiments did not establish its mechanism, and
the spread replays did not establish executable options profitability.

## Models, signals and data

“No qualifying increment” means the study's registered promotion criteria were
not met. It does not establish equivalence, and it does not turn unavailable
data or failed numerical checks into evidence against a hypothesis.

| Study | Latest saved outcome |
|---|---|
| [Orthogonal round 2](../reports/orthogonal_round2/results.md) | Eight inconclusive comparisons; no qualified increment. |
| [Model, latent memory and reference-paper alternatives](../reports/model_memory_study/SUMMARY.md) | None of 14 alternatives qualified across 46 comparisons. |
| [Finer-measurement search](../reports/iterative_signal_search/SUMMARY.md) | Completed; no qualifying candidate. |
| [International volatility](../reports/international_volatility/SUMMARY.md) | Completed; no qualifying increment. |
| [Overnight index returns](../reports/overnight_index/SUMMARY.md) | Completed; no qualifying increment. |
| [Macro and overnight information](../reports/macro_overnight/SUMMARY.md) | Completed and verified; no qualifying increment. |
| [Measurement memory](../reports/measurement_memory/SUMMARY.md) | Completed and verified; no qualifying increment. |
| [Index hinge](../reports/index_hinge/SUMMARY.md) | Completed and verified; no qualifying increment. |
| [Tail shape](../reports/tail_shape/SUMMARY.md) | Completed and verified; no qualifying increment. |
| [Calendar variance](../reports/calendar_variance/SUMMARY.md) | Completed; no qualifying increment. |
| [Relative risk](../reports/relative_risk/SUMMARY.md) | Completed; no qualifying increment. |
| [Joint risk](../reports/joint_risk/SUMMARY.md) | Better historical joint score, but failed search-adjusted gates. |
| [Cross moment](../reports/cross_moment/SUMMARY.md) | Completed; no qualifying increment. |
| [Target aligned](../reports/target_aligned/SUMMARY.md) | Completed; no qualifying increment. |
| [Sign memory](../reports/sign_memory/SUMMARY.md) | Completed; no qualifying increment. |
| [Causal pool](../reports/causal_pool/SUMMARY.md) | Completed; no qualifying increment. |
| [Civil quarter](../reports/civil_quarter/SUMMARY.md) / [replay](../reports/civil_quarter_replay/SUMMARY.md) | Admission and numerical failures preserved as separate outcomes. |
| [Profiled quarter](../reports/profiled_quarter/SUMMARY.md) | Verified follow-up; no qualifying increment. |
| [Range alert](../reports/range_alert/SUMMARY.md) | Completed; no qualifying increment. |
| [Issued calibration](../reports/issued_calibration/SUMMARY.md) | Completed; no qualifying increment. |
| [Event cluster](../reports/event_cluster/SUMMARY.md) / [replay](../reports/event_cluster_replay/SUMMARY.md) | Original timestamp-verification failure retained; replay verified and nonqualifying. |
| [Peak age](../reports/peak_age/SUMMARY.md) | Completed; no qualifying increment. |
| [Claims releases](../reports/claims_release/predictive/SUMMARY.md) | Completed and verified; no qualifying signal. The earlier top-level summary is an intermediate plan. |
| [Commodity implied volatility](../reports/commodity_implied/predictive/SUMMARY.md) | Completed and verified; no qualifying signal. |
| [Treasury dealer information](../reports/treasury_dealer/predictive/SUMMARY.md) | Completed and verified; no qualifying signal. |
| [Precision gate](../reports/precision_gate/predictive/FINDINGS.md) | State-dependent blend did not qualify against the market benchmark. |
| [Early-session feasibility](../reports/early_session_feasibility/SUMMARY.md) | Source-blocked feasibility result; no registered predictive experiment completed. |

These are related searches over overlapping history, not independent
replications. The later search ledger reaches **164 tracked comparisons**;
this is the declared family count, not a census of every historical analysis
in the repository. Current-vintage inputs and reused observations limit claims
of prospective predictive value even when timing and numerical checks pass.

## Copula result and unresolved attribution

**[Joint copula, wave 27](../reports/joint_copula/predictive/FINDINGS.md).**
With identical per-asset marginal forecasts, the t8 dependence arm improved
mean joint negative log likelihood versus Gaussian by 0.030568716 nats in
development and 0.031017082 in evaluation. It also beat independence. Both
comparisons passed the registered cumulative adjustment (151 comparisons at
that wave; adjusted p=0.0003775). This qualifies as an exploratory joint-density
lead. It does not establish better standalone volatility forecasts, calibrated
tail risk or trading profits.

**[Crossed marginal calibration, wave 28](../reports/copula_calibration/predictive/FINDINGS.md).**
On 2,190 common sessions, recalibration reduced the t8–Gaussian gap by about
9.6% / 11.6% in development/evaluation; most of the descriptive gap remained.
None of the seven attribution comparisons cleared the adjusted gates.
Marginal calibration was mixed, and the mechanism remains unresolved.
A simulation under the fitted t8 law is a benchmark under that assumed law,
not a ceiling or an attribution test. Central-day gains do not distinguish
scale error from dependence misspecification.

**[Shape calibration, wave 29](../reports/copula_shape/predictive/terminal.json).**
The formal experiment is UNEVALUABLE: its sampled tail-quantile consistency
discrepancy exceeded the fixed numerical tolerance. All six comparisons remain
in the search family as p=1. Passing density checks did not override that
failure. Later descriptive use of saved outputs does not promote this model or
repair its formal verdict.

## Same-day spread risk and hindsight geometry

**[Verified descriptive replay](../reports/copula_spread_replay/FINDINGS.md).**
The study uses QQQ and SPX open-to-close returns over 2,190 saved sessions,
with 733 development and 1,457 evaluation sessions. The primary hypothetical
spreads have short strikes 2% from the open and wings 1% wide. Each model's
rule accepts a day when its predicted probability of either underlying
breaching is at most 10%. The portfolio reserves 2% of equity in gross maximum
expiration liability, split equally across assets.

For evaluation condors, the shape-t8 rule accepts 917 sessions (62.94%) and
has 29 breach days (3.16%), compared with 146/1,457 (10.02%) when always selling.
The original t8 rule has 33/964 breaches (3.42%); an affine Gaussian alternative
is also competitive, at about 3.25%. This is evidence of descriptive risk
selection, not a consistent upgrade from the new model or a dependence-only
benefit. Marginal forecasts contribute to the filter too.

**[Hindsight geometry search](../reports/spread_geometry_hindsight/FINDINGS.md).**
The search covers 20 short-strike distances, eight wing widths and three
structures: 480 geometries, seven fixed day masks and four hypothetical premium
scenarios. It retains 10,080 geometry summaries, 40,320 account scenarios and
252 within-period selections. The original day masks are unchanged; probabilities
were not recomputed for the new strikes. All geometry winners use hindsight.

The shape-t8 condor cohort has zero observed expiration liability at ±3.75%
over 917 evaluation days, but two development breaches at that distance.
The nearest tested zero-liability distance over all 1,542 accepted sessions is
±4.25%. All eight widths tie in that zero-liability region. Premium assumptions
reverse the apparent preference for narrow versus wide wings, so the search
does not identify an economically optimal width. See [all selected cases](../reports/spread_geometry_hindsight/ALL_WINNERS.md).

Neither study has actual executable option premiums, contract rounding,
intraday exits, assignment or broker-margin accounting. Hypothetical credit
scenarios are sensitivity calculations, not observed returns. Closing inside
a strike says nothing about an earlier intraday breach. A zero historical
breach count does not imply zero future risk.

The earlier [futures risk-backtest directory](../reports/copula_risk_backtest/)
contains a superseded, explicitly unrun design; it is retained as provenance.

## Fresh data and calibration

The [prospective benchmark plan](PROSPECTIVE_BENCHMARK.md) specifies immutable
forecasts, content-addressed source receipts, availability timestamps, missing
and late observations, revisions and scoring. The ledger is implemented and
tested. Live provider collection, the option quote/execution adapter and
integrated joint-forecast scoring still require integration; no live collector
or trading service is activated by these results.

For spread validation, record exact contracts and synchronized bid/ask quotes
at the decision time, the frozen entry and sizing rules, costs and settlement.
Any stop or intraday exit rule also requires the corresponding intraday data.
Use fresh observations to assess calibration and compare the joint filter
against marginal-only and simpler benchmark rules.

## Reproduction and publication

The commit includes protocols, code, tests, compact results, generated figures
and preserved audit/provenance records. Private data, fitted forecast tables,
raw source captures, model weights and environments stay local. A clean clone
therefore does not contain every empirical input needed to regenerate every
study. Review each study's protocol and freeze manifest before running it.

Use Python 3.11 in the project environment. `make test-fast` runs the original
methodology contracts and smoke check; `make test` discovers the broader
contract suite; `make lint` checks source and tests. Environment-dependent
checks are separate (`make test-env`). Study-specific pre-run tests and
independent verification receipts sit alongside the relevant reports.

**Historical access:** `make test` and `make test-fast` include tests that read
and reconstruct forecasts over previously evaluated market history. They are
not data-free checks. See the [historical test-access correction](../reports/treasury_dealer/predictive_prefit/HISTORICAL_TEST_ACCESS_CORRECTION.md)
and use its explicit six-test quarantine when preserving that access boundary.
Passing a regression suite does not make those observations untouched.

The [2026-09-16 archival verification record](../reports/archive_20260916/README.md)
documents the checks run for this commit, historical access and remaining lint
findings. It is separate from the original study verification receipts.

Do not refit studies, overwrite outputs or repin hashes merely to make checks
pass. Frozen failures remain part of the evidence. Corrections and new research
belong in additive, explicitly labeled outputs.
