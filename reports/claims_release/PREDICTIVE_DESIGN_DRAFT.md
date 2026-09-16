# Draft: first-release claims and five-session QQQ variance

**For review, 2026-09-08. Not frozen, registered or executed.** This proposal uses existing protocol prose and the previously proposed mechanism. No numerical source table, feature panel, model output or score was read for this design. Source authentication must finish before registration.

The question is whether the latest genuine first report improves volatility prediction beyond both established market information and matched controls for the recent labor level and release timing. It is not a consensus-surprise test or a causal claim.

## Measurement and preserved boundaries

Use the established daily raw-OHLC proxy:

    v[t] = max(0.5*log(H[t]/L[t])^2 - (2*log(2)-1)*log(C[t]/O[t])^2, 1e-10)
           + log(O[t]/C[t-1])^2
    q5[t] = mean(v[t+1], ..., v[t+5])

The sole target is that arithmetic mean in native squared-log-return units, available at the fifth subsequent observed QQQ session's close. Preserve the full bounded reference calendar before rolling, lagging or filtering. Missing constituents make the target missing; never shorten or compress the window. This remains an OHLC risk proxy with inherited ETF/vendor limitations.

| Fence | Exact dates |
| --- | --- |
| Application origins | 2016-01-04 through 2025-10-17. |
| Development | 2016-01-04 through 2019-12-31; target must mature by 2019-12-31. |
| Evaluation | 2020-01-02 through 2025-10-17; target must mature by 2025-10-20. |
| Evaluation stability slices | 2020-01-02 through 2022-12-31; 2023-01-01 through 2025-10-17. |
| Source ceiling / protected period | Nothing after 2025-10-20 is used; 2025-11-03 onward remains protected. |

Target maturity determines the last scored origin; do not move endpoints to recover rows. Measurement and phase precedents are [orthogonal_round2.yaml](/Users/byrons/code/trading-vol/ndx-vol-experiment/orthogonal_round2.yaml), [iterative_signal_search.yaml](/Users/byrons/code/trading-vol/ndx-vol-experiment/iterative_signal_search.yaml) and [calendar_variance.yaml](/Users/byrons/code/trading-vol/ndx-vol-experiment/calendar_variance.yaml). These remain reused historical partitions.

## One feature and a fixed date rule

For reference Saturday k, let c[k] be its authenticated positive **advance seasonally adjusted initial-claims first report**:

    m4[k] = mean(log(c[k-1]), ..., log(c[k-4]))
    x[k] = log(c[k]) - m4[k]

Here k-j means exactly 7*j calendar days earlier. Require five consecutive reference Saturdays and genuine first reports. Retain contemporaneous seasonal treatment: no revised-history substitution, archive backfill, alternate lookback, sign restriction or pandemic exclusion.

At the idealized after-close origin on civil date t, use only admitted original DOL reports whose displayed public-release date is strictly earlier than t. Select the latest date-eligible report before checking its numerical availability. As specified in `SOURCE_DISPOSITION.md`, ALFRED's date remains comparison evidence, not a second availability clock: the authentic March 2018 first report and ALFRED's later revised record are different observations of the publication process. The unresolved March 2017 first-report value remains missing and must not be replaced by either conflicting count.

**The civil day of a new release still uses the previously eligible information, never today's new count.** A normal Thursday report first becomes eligible on Friday if Friday is an observed origin. Require age **1–7 calendar days inclusive**, measured from the actual DOL release date. At age 8 the state is unknown. A Wednesday release follows the same rule, without a future holiday or announcement calendar.

A missing/ambiguous latest eligible value or incomplete five-report chain makes the feature unknown; do not fall back to an older complete report or skip a reference week. An entirely absent release lets the last known state expire under the same seven-day bound. The rule intentionally permits that older state through day 7 and does not assert that another report was scheduled. Preserve missing and holiday gaps explicitly, without extending expiry or backdating availability from later evidence.

## Three arms and two required comparisons

Retain the established twelve market-baseline columns, including the intercept:

    const, lrv_d, lrv_w, lrv_m, lev_d, lev_w, lev_m,
    liv, lvix, term, xasset_stress, market_stress

Preserve their definitions and source identities in [orthogonal_round2.yaml](/Users/byrons/code/trading-vol/ndx-vol-experiment/orthogonal_round2.yaml): HAR variance history, leverage, delayed VXN/VIX/term inputs and prior-only cross-asset/market stress scaling. QQQ/ETF inputs retain the inherited idealized current-close convention; Cboe inputs are delayed one reference session. Existing archival/back-calculation limitations remain disclosed.

| Arm | Columns |
| --- | --- |
| Market | Those 12 established columns. |
| Matched control | Market plus m4, age/7, and entry-weekday indicators Tuesday–Friday: 18 columns. Monday is the reference. |
| Candidate | Matched control plus x: 19 columns. |

**Both formal comparisons must pass:** candidate versus matched control, and candidate versus market. The first tests the latest report conditional on prior labor level/timing; the second prevents success from depending on deterioration of the matched control. Neither proves universal orthogonality.

All three arms use identical training, application and scoring rows, complete for all 19 candidate columns. Calendar features describe the known current origin. Retain unavailable origins and reasons; the market arm receives no additional rows.

## Fixed estimator and causal fitting

Fit expanding **OLS of log(q5)**, with separate exact training-residual Duan smearing for each arm. No estimator, penalty, feature or horizon grid. Refit at the first common-feature-complete origin of each month, selected before checking its future query label. Earlier training origins must have all five target sessions mature by the **previous observed QQQ session** of the fit. A month with no complete application origin is an explicit coverage gap.

Use common-training population means/standard deviations for non-intercept columns. Require finite scales greater than 1e-12 and full declared design rank under a fixed relative singular-value threshold of 1e-12. Do not delete columns or switch estimators after inspection.

For each arm, let e_i be its training log-target residual; predict exp(predicted_log_q5) times mean(exp(e_i)), computed stably from those matured training rows only. No future residual calibration or clipping. Save every complete application prediction, including origins whose targets cannot be scored. The estimator follows the log-target OLS/smearing precedent in the round2 and iterative protocols.

## Support, effect and dependence gates

These thresholds are prospective, not conclusions about available support.

| Gate | Requirement |
| --- | --- |
| Each monthly fit | At least 1,000 mature training origins and 200 distinct latest-release identities. |
| Each phase | At least 505 paired origins and 104 distinct latest-release identities. |
| Each fixed evaluation slice | At least 252 paired origins and 52 distinct latest-release identities. |
| Five-session offsets | Full-calendar position modulo 5 before filtering: each phase/offset has at least 63 paired origins and negative mean paired loss difference for both comparisons. |
| Practical effect | Candidate-minus-control mean QLIKE difference at most **−0.005 in both phases**, against each control. |
| Stability | Negative differences against both controls in each evaluation slice, and in both phases when within-report mean losses are averaged equally across latest-release identities. |

Score with L(q,f)=q/f−log(q/f)−1; its paired difference equals that of log(f)+q/f. Report absolute improvements. The 0.005 threshold follows [calendar_variance.yaml](/Users/byrons/code/trading-vol/ndx-vol-experiment/calendar_variance.yaml); training floors and full-calendar offset discipline follow later protocols, including [peak_age.yaml](/Users/byrons/code/trading-vol/ndx-vol-experiment/peak_age.yaml). Distinct-release floors are explicit prospective additions. Daily rows sharing a report are not independent labor observations; report-weighted stability is a robustness gate, not another formal p-value.

For each contrast, use two-sided centered-null circular bootstrap lengths **21, 63 and 126 retained paired observations**, plus Bartlett HAC **126**. Each phase p-value is their maximum; each hypothesis p-value is the maximum across development/evaluation. Use **399,999 draws**, plus-one p-values and seed 20260929 + 5*1000000 + phase_code*10000 + block, with development 0 and evaluation 1; share draw designs across contrasts. Report calendar gaps, annual coverage, distinct reports, uncertainty and nominal MDE. Never select a favorable block, offset or year.

Before scoring, prewrite timing, missingness, overlap and independent-reconstruction tests, including release-day exclusion, age-7/age-8 boundaries and missing Saturdays. Retain the existing serial-null calibration precedent: AR(1) rho 0.8, n=1,500, 200 repetitions, 499 resamples and at least 90% coverage of the conservative interval envelope. These are future validation requirements; no such run occurred for this draft.

## Accounting and decision

Future registration adds **two hypotheses to the 140 prior comparisons, for 142 total**. Preserve every inherited status and p-value. If this is the next numbered wave, proposed wave 23 alpha is exactly 0.05/(23*24)=1/11040: require both wave Holm-2 adjusted p-values below it and both cumulative Holm-142 adjusted p-values below 0.05, plus every gate above. Root must reconcile numbering before freezing if another study is registered first.

Before registration, source failure leaves the count at 140. After registration, insufficient support or an invalid fit/arithmetic/verification/publication attempt retains **both** hypotheses as unevaluable p=1, distinguishing INSUFFICIENT_DATA from an invalid run and a completed negative result. Do not relax gates after inspection.

Even a pass is only an **exploratory candidate for future untouched evaluation**. Independent verification must reconstruct source timing, all common cohorts, fits/smearing, scored and unscored forecasts, paired losses and family accounting before publication. This draft authorizes no model run, execution claim or position sizing.
