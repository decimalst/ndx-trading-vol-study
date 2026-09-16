# Prospective commodity implied-volatility design review

2026-09-08. Design only; unregistered and unscored. The exact candidate and matched controls below were accepted by root before any commodity numerical source, feature, support calculation or outcome was inspected. This review read the frozen claims protocol as a methodological template and official index documentation. It did not read old claims cohorts, fits or results, obtain the new index CSVs, or change Python or existing reports. Current accounting remains 142 comparisons; wave 24 would be next if this study is selected.

**Recommendation: one jointly fitted two-column OVX/GVZ block, tested against both the established market baseline and a fixed matched commodity-risk baseline.** Two separate economic exposures are retained without choosing an index, averaging them, estimating a latent factor, or searching for signs or weights. A successful joint block would establish an increment conditional on the stated controls, without identifying which index supplied it or establishing causality.

## Fixed candidate and matched controls

Retain the existing twelve market columns: const, lrv_d, lrv_w, lrv_m, lev_d, lev_w, lev_m, liv, lvix, term, xasset_stress, market_stress. Their current-close QQQ/ETF convention and prior-session equity-IV convention remain unchanged. Existing VXN, VIX and term inputs control much of the equity option common state; the pooled cross-asset stress measure alone does not separate oil from gold risk.

Let t index the full observed QQQ calendar. Let P_j be the existing adjusted-close series for j in {USO, GLD}, aligned to that calendar without filling. Let I_j be OVX for USO and GVZ for GLD, also aligned without filling. Define:

    r_j[t] = log(P_j[t-1] / P_j[t-2])
    R_j,w[t] = mean(r_j[t]^2, ..., r_j[t-w+1]^2), for w in {5,22}
    z_j[t] = 2*log(I_j[t-1]/100) - log(252)

Equivalently, align each source first, delay its level series by exactly one QQQ position, then construct the additional features. Do not shift on the source's smaller calendar, compress a missing day, or apply a second delay after constructing returns. All 5/22 squared-return constituents must exist. No contemporary t commodity-IV value enters the forecast.

| Arm | Columns, including intercept | Purpose |
|---|---:|---|
| Market |12| Established equity/market information. |
| Matched |20| Market plus, for each of USO and GLD, r1, r1², log(R5), log(R22). |
| Candidate |22| Matched plus z_USO and z_GLD, jointly fitted. |

The four controls per ETF cover the signed immediate move, its magnitude, near-target realized risk and approximately monthly realized risk. Raw r1² remains zero on a genuine zero-return day. A 5/22 mean of zero makes only its logarithm unavailable; no epsilon or replacement is added.

The full alternative of six controls per asset would also add mean signed returns over 5 and 22 sessions. Those are additional drift histories, rather than essential controls for the specific realized-risk explanation. Omitting them keeps the comparison focused, but limits any claim to the chosen information set: an improvement could still reflect nonlinear effects or omitted momentum. A 1+22-only design is smaller but omits recent 5-session risk at the target's own horizon. The accepted four-control specification retains that control at the same dimension as several competing four-control designs. These alternatives are conceptual comparisons only; there will be no numerical selection among them.

Fit the two candidate columns together with all matched controls in ordinary unpenalized regression. Do not regress each index sequentially on a different nuisance set, use test-period residualization, or choose a principal component. Training-only partial-regression algebra may explain the fit, but the authoritative estimator is the complete joint design. Collinearity must trigger the frozen rank/scale rule rather than removal of a column.

## Units and interpretation

Cboe identifies GVZ and OVX as 30-day measures based respectively on GLD and USO options. Those underlyings determine the interpretation: these are ETF option signals, not spot-gold or front-month crude-futures variance series. [Cboe methodology, sections 1–2](https://cdn.cboe.com/api/global/us_indices/governance/Volatility_Index_Methodology_Selected_Broad_Based_Index_Equity_and_ETF_Volatility_Indices.pdf).

The quoted index is volatility in percentage-point units. Squaring after dividing by 100 produces an annualized implied-variance convention; division by 252 is a predetermined conversion to a daily trading-session scale. Compute its logarithm as the expression above to avoid unnecessarily forming a potentially extreme square. This normalization does not convert a 30-calendar-day option measure into an exact forecast of five future QQQ sessions. The commodity controls are past adjusted-close squared returns, while the QQQ target remains the arithmetic mean of five future GK-plus-overnight variance proxies. The quantities differ in underlying, horizon and construction.

With an intercept and training-only centering/scaling, multiplying a strictly positive variance feature by a fixed unit constant adds only a constant to its logarithm. The 252 choice is therefore a declared unit convention, not another model variant. Do not call z minus log realized variance a measured variance risk premium: trailing ETF returns are neither a matched forward physical expectation nor the same payoff as the option index. This study does not estimate such a premium.

## Source history, timing and the 2020 episode

Cboe's 2008 annual report records the introduction of both indices in 2008. Its current methodology separately identifies pre-2008 first-value dates and 2008 launches. Root's accepted common source admission date is **2009-01-02**, with ceiling **2025-10-20**; the design does not need an unverified exact OVX launch day. The prelaunch portion of any historical download must remain excluded. [Cboe 2008 annual report, printed page 11](https://cdn.cboe.com/resources/annual_reports/annualreport2008.pdf), [index information, section 6](https://cdn.cboe.com/api/global/us_indices/governance/Volatility_Index_Methodology_Selected_Broad_Based_Index_Equity_and_ETF_Volatility_Indices.pdf).

The live-history floor does not authenticate historical file vintages. The current methodology is version 9.0, revised March 30, 2026, after the study ceiling. It cannot simply be asserted to govern every old observation. Cboe also documented a methodology change effective before the open on June 6, 2022 and a strike-selection change effective February 10, 2025 that includes both OVX and GVZ. Preserve dated notices and source snapshots; do not retrospectively rebuild earlier values under today's rules or split the sample to select a favorable regime. [Current document information](https://cdn.cboe.com/api/global/us_indices/governance/Volatility_Index_Methodology_Selected_Broad_Based_Index_Equity_and_ETF_Volatility_Indices.pdf), [2022 implementation notice](https://cdn.cboe.com/resources/release_notes/2022/Update-EQ-ETF-Volatility-Indices-Consultation-Summary.pdf), [2025 strike-selection notice](https://cdn.cboe.com/resources/release_notes/2025/Modifications-to-the-Strike-Selection-in-Volatility-Index-Calculations.pdf).

Use only the previous QQQ session's dated observation. This conservative delay handles the prospective research clock; it does not prove when today's historical CSV was published or whether its older values were later corrected. Missing prior-session IV remains missing. There is no claims-style seven-day carry or last-available fallback. The source-feasibility review owns historical-file restatement and publication-vintage evidence; any residual limitation must remain explicit in the eventual evidence class.

The 2020 negative-WTI-futures episode must not be handled by inserting negative futures prices into positive ETF-price logarithms. The modeled price is adjusted USO, and its fund exposure and corporate actions require their own metadata checks. The existing cross-asset input has adjusted closes, not commodity OHLC: the matched risk measure is squared close return, never an invented commodity GK proxy. Validate split adjustment and preserve genuine extreme returns or IV values. Index readings above 100 are not automatically invalid. Observed nonpositive ETF prices or IV and nonfinite values fail validation; authentic positive extremes are retained. No winsorization, crisis-year deletion, rolling-contract replacement or post-inspection robust estimator is proposed.

## Uniform samples and transferable methodology

Use the sole five-session QQQ target and original development/evaluation origin fences from claims_release.yaml; construct the new cohort afresh. All three arms must use identical rows complete for all 22 features, both in training and application. The market-only arm must not gain earlier or more complete rows. Keep the full calendar before lags, rolling histories, target endpoints and modulo5 assignment. Feature warmup, missing sources and lack of target maturity are distinct missingness reasons.

Retain expanding monthly OLS of log target, three separate exact training-residual Duan smear factors, first complete monthly application scheduling before query-label checks, and labels mature no later than the preceding observed QQQ session. Preserve population scaling, the 1e-12 scale/rank cutoffs, finite positive forecasts, saved unscored applications and independent mathematical reconstruction. The training floor remains 1000 common mature origins. An unavailable or rank-deficient model is an invalid/unsupported attempt under its fixed rules, without column removal or an estimator fallback.

Do not copy claims-specific calendar/age features, five-consecutive-release logic, distinct-weekly-release floors or equal-report-weighting gates. Daily index observations are a different information process. Retain the 505-per-phase, 252-per-fixed-evaluation-slice and 63-per-full-calendar-offset floors as prospective daily support gates. Report valid source dates, missing calendar stretches and annual coverage. Overlapping targets and persistent predictors still require dependence-aware inference; daily timestamps do not create independent economic events.

## Hypotheses and unresolved freeze items

Register only two formal predictive contrasts if proceeding: candidate minus matched and candidate minus market. Both must pass. The second comparison prevents a result that depends on deterioration of the matched benchmark. Two jointly entered IV coefficients do not themselves create two extra hypotheses, because there is no singleton selection or formal per-index claim. Any later individual-index, transformed-factor or alternate-horizon comparison would be an additional experiment.

The proposed accounting is wave 24 with alpha 0.05/(24*25)=1/12000, two new comparisons and 144 cumulative comparisons. Retain both hypotheses as p=1 if an attempt fails after registration. Recommend the inherited absolute QLIKE effect gate of at most −0.005 in both phases against each control, negative fixed-slice/offset differences, conservative two-sided bootstrap lengths 21/63/126 and HAC 126, 399999 draws, and the existing synthetic dependence calibration. Require both within-wave Holm 2 and cumulative Holm 144 gates. Remove only the claims-specific gates described above; any replacement must be written before source-dependent support inspection. Historical reuse remains exploratory even after a pass.

Before registration, root still needs to fix the artifact/column parser contract and source-version evidence; write the common 2009 floor and exact rolling-history warmup into executable tests; bind the new independent source/feature/model verification; and assign the new seed and final daily-data gate literals. A conservative warmup recommendation is to mask the newly used commodity histories before the admission date, then wait for the full required return windows, leaving the established market-baseline definition unchanged. Neither the application endpoint nor a lag/window may move in response to support or results. These are implementation/source-admission decisions, not permission to reopen the accepted 22-column specification.

No commodity time series, source CSV header, QQQ data, feature/cohort, fit or score was inspected for this review. The cited documents establish identity and methodological limitations; they do not establish usable numerical coverage. Only this new review document was written.
