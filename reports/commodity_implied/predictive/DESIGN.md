# Joint commodity option-implied risk and QQQ variance

This prospective experiment asks whether delayed OVX and GVZ jointly improve the five-session QQQ variance forecast beyond both the established market baseline and matched commodity-return risk controls. Source admission is complete, but no new commodity feature panel, forecast cohort, fit or score has been inspected when this design is written. The fixed executable protocol is [commodity_implied.yaml](../../../commodity_implied.yaml).

The admitted original Cboe captures each contain 4,045 in-scope observations from September 18, 2009 through October 20, 2025. This is source metadata, not a claim about usable model rows or calendar completeness. The predeclared source floor remains January 2, 2009. Current source bytes and bounded outputs are authenticated and independently reconstructed; immutable historical publication vintages remain unverified. No later source numerical cells enter the bounded parser. See the [source certificate](../SOURCE_ADMISSION.json) and [source review](../SOURCE_FEASIBILITY.md).

## Fixed information set

| Model | Inputs | Purpose |
|---|---|---|
| Market | The existing 12 columns, including intercept, realized variance histories, leverage, equity implied volatility, term and market stress | Preserve the established benchmark |
| Matched | Market plus eight controls: USO and GLD each contribute a delayed signed one-session return, its square, and log five- and 22-session mean squared returns | Account for the commodities' observed return and risk state |
| Candidate | Matched plus delayed log implied-variance units from OVX and GVZ together | Test the joint incremental option-information block |

Reindex the commodity sources to the full observed QQQ calendar first. Mask new commodity price and index cells before the fixed source floor, then delay both sets of levels by one observed QQQ session. Compute returns and strict rolling histories from those delayed levels. Missing observations never carry forward or compress a window. Zero returns and their squares are valid; a genuinely zero rolling mean leaves its logarithm unknown. Invalid arithmetic, insufficient rank or a zero-scale regressor aborts without clipping, substitution or removing a column.

For each implied-volatility index, the feature is `2*(log(index)-log(100))-log(252)`. With an intercept this fixed normalization is only a linear reparameterization of log index levels. It does not create a measured variance risk premium. Both option columns enter together; there is no individual-index selection, latent-factor search, parameter grid or alternate horizon in this attempt.

The existing cross-asset file is an inherited vendor price snapshot. Its acquisition code selects adjusted close when present and otherwise close; original per-series field selection, revision vintage and split-adjustment metadata have not been separately authenticated here. That limits economic interpretation. The experiment preserves these exact bytes and does not repair the data after seeing forecasts. Cboe's dated 2022 quotation-source and 2025 strike-selection changes remain inside the unchanged evaluation history.

## Forecast clock and estimator

The target is the arithmetic mean of the next five full-calendar daily GK-plus-overnight variance proxies, requiring all five constituents. The established market-feature formulas and GK floor of 1e-10 remain unchanged. This target differs from the commodity indexes' underlying ETFs and 30-calendar-day option horizon.

All three models use exactly the same 22-feature-complete training and application rows. Expanding monthly fits select the earliest feature-complete query before looking at its target. Training labels must finish by the immediately previous full-calendar QQQ session; at least 1,000 common mature training origins are required. The estimator is a separate OLS regression of log target for each arm, with training-population standardization and each arm's own exact training-residual Duan smearing. Relative rank cutoff and minimum feature scale remain 1e-12. Numerical failures and unsupported scheduled fits stop the whole registered attempt.

Requested origins run from January 1, 2016 through October 20, 2025. Development origins and labels must finish by December 31, 2019. Evaluation begins January 1, 2020, and labels must finish by October 20, 2025. These calendar boundaries include all available queries, including the final unscored applications; they do not move in response to target availability. Every requested observed origin remains in coverage. Every complete application retains its forecasts even when its target cannot be scored. Five calendar offsets are assigned on the original QQQ calendar before missingness exclusions. Equity observations from November 3, 2025 onward remain protected; no numerical source beyond October 20 is admitted.

## Two comparisons and fixed decision rules

Candidate-minus-matched and candidate-minus-market are the only two registered hypotheses. Lower paired QLIKE loss is better. Both comparisons must improve mean loss by at least 0.005 in both development and evaluation, have negative differences in every five-session calendar offset and both fixed evaluation slices (2020–2022 and 2023–the source ceiling), and satisfy the statistical gates. Each phase requires 505 paired origins; each fixed evaluation slice requires 252; each phase/offset requires 63. Annual coverage and calendar gaps are disclosed. Claims-specific release counts and equal-release weighting do not apply to this daily source.

Inference uses two-sided centered-null plus-one block bootstrap probabilities with block lengths 21, 63 and 126 and 399,999 draws, together with Bartlett HAC126. Shared paired draws use seed 20260930 plus the fixed horizon, phase and block offsets. Each hypothesis takes the maximum probability across blocks, HAC and both phases. Holm correction must pass both within this wave at `0.05/(24*25)` and across all 144 comparisons at 0.05. Nominal intervals are reported without claiming simultaneous coverage.

The 142 inherited comparisons retain their identities, source hashes and probabilities. Both new hypotheses are journaled before any historical feature/cohort construction. Any registered invalid or unsupported attempt retains both hypotheses as unevaluable with probability one. There is no restart, threshold relaxation, parameter change or selective supported-month run after registration.

## Verification and evidence class

Source, input, feature, model, scheduler, protocol and registration tests use invented data and precede empirical runs. Independent reconstruction checks the entire forecast output, including unscored applications, clocks, shared cohorts, all model audits and individual smearing factors. A separate verifier reconstructs all statistical calculations and accounting. The serial-null calibration uses 200 generated AR(1) trials with coefficient 0.8, length 1,500 after 200 warmup points, 499 draws and a minimum conservative interval-envelope coverage of 0.90. This calibration is not a financial-data coverage guarantee.

The complete repository tests, unchanged code/source checks, fixed protocol, independent calibration and pre-existing evidence are recorded in a prospective freeze before registration. No frozen code or prior results may change afterwards. Even a full gate pass would be an exploratory finding on reused history, subject to source-vintage and measurement limitations and requiring untouched confirmation before promotion. It would establish an increment conditional on these controls, not causality, executable profit or universal orthogonality.
