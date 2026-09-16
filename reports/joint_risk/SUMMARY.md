# Joint QQQ–SPX intraday risk: verified exploratory improvement

**The changing-dependence model improved the joint-risk score, but did not qualify under the registered search-wide corrections.** Against the control with identical predicted means and marginal second moments, the absolute score improvement was **0.03325 in 2016–2019** and **0.08363 in 2020–October 2025**. Both fixed later subperiods improved too. The useful-effect and stability requirements pass; the statistical correction requirements do not. No protocol, source, model, sample, tolerance or acceptance gate changed after the run began.

The experiment produced **7,386 forecasts on 2,462 common dates**, across three models and 118 monthly expanding refits. Independent verification passed on its first empirical execution. This is evidence about archival QQQ ETF/SPX index **joint return residual second moments**, with important attribution limits; it is not a validated trading signal or a high-frequency covariance measurement.

## What improved, and what did not pass

Negative paired score differences mean improvement. The score is the unhalved `logdet(H)+e' H^-1 e`; its level can be negative and depends on return units, so these differences are not percentages. Each comparison required an absolute decrease of at least **0.005 in both phases**, negative differences in both fixed later slices, Holm adjustment over this wave's two comparisons at **0.05/110 = 0.00045455**, and cumulative Holm over **112** enumerated hypotheses at 0.05.

| Dynamic dependence compared with | 2016–2019 difference | 2020–October 2025 difference | Wave Holm p | Cumulative Holm p |
|---|---:|---:|---:|---:|
| Constant correlation, identical modeled marginals | -0.033250 | -0.083632 | 0.017584 | 1.000000 |
| Constant residual-second-moment matrix | -0.881073 | -0.881852 | 0.009020 | 0.460020 |

All four nominal 95% interval envelopes exclude zero. For the required constant-correlation comparison, they are **[-0.060699, -0.005802]** earlier and **[-0.120951, -0.046313]** later. The conservative phase p-values are **0.0175838** and **0.00001121**. Taking the weaker phase, as specified, leaves insufficient evidence for the wave and cumulative corrections. The constant-matrix comparison also fails both corrections despite its larger average improvements.

The fixed 2020–2022 and 2023+ differences against constant correlation are **-0.064537** and **-0.104226**. Earlier annual diagnostics are less uniform: 2016 improved slightly (-0.002317), 2017 worsened slightly (+0.001842), and 2018–2019 contributed larger improvements. These are retained diagnostics, not grounds for dropping years or changing the registered periods.

The much larger advantage over the constant matrix cannot all be credited to the dependence slope: that benchmark also lacks the conditional marginal second-moment forecasts. The comparison with identical modeled marginals is the relevant incremental test. Its result is encouraging for follow-up, while its registered verdict remains **DOES_NOT_QUALIFY**.

![Both registered joint-risk comparisons](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/joint_risk/comparison_intervals.png)

The dotted vertical lines mark the fixed -0.005 useful-effect threshold; solid vertical lines mark zero improvement.

## What the experiment measures

The response is the two next-session raw open-to-close log returns. Shared mean forecasts define `e=r-m`; the observed matrix is `e e'`. It may be rank one, have signed off-diagonal entries, or be zero. The score never takes a determinant of that observed matrix or adds target jitter.

Both conditional arms use exactly the same mean and positive diagonal forecasts. Their predictors include both markets' strict 1/5/22 risk, negative-return and signed intraday-return histories; implied-volatility inputs; weekday; lagged corr22; and its training-centered square. All market predictors end at the preceding observed SPX session. The 34 source columns become 35 model columns after the training-only square is added. Every nonconstant training scale must be finite and exceed 1e-12. No row, feature or failed model is silently dropped.

Shared mean ridge and positive-moment penalties are fixed at 0.01. Training residuals are from the current monthly mean fit, not historical out-of-sample errors. Constant correlation is fitted first. The candidate holds its intercept fixed and fits only one penalized slope in `.995*tanh(a0+b*z_corr22)`, with both scalar parameters bounded to [-4,4]. The practical second control uses the uncentered mean of those training residual outer products, without jitter or an n-1 divisor.

Equal modeled diagonals do not establish true correlation predictability. Changing correlation in the matrix inverse also changes the weights on squared residuals. Before the historical run, an analytic synthetic counterexample demonstrated a joint-score gain with constant zero true correlation and incorrect shared diagonal forecasts. Common-mean error is another limitation: the target equals conditional covariance plus the outer product of the true-minus-predicted mean. Any stronger correlation interpretation therefore depends on the adequacy of the shared marginal models.

## Data and numerical checks

The full-reference measurement gate passed on all **4,226 observed SPX dates for each asset**. No complete OHLC row was missing and no GK value activated the fixed 1e-10 floor. Here the GK gate protects risk-history predictors; the response itself consists of signed returns. The source audit counted **23 zero QQQ intraday returns and 2 zero SPX intraday returns**, which are valid observations. Nonfinite computed returns on any observed open/close pair would abort, even if high or low were missing.

Development has **1,005** scored dates and evaluation **1,457**. There are 2,463 feature-complete application dates; December 31, 2019 is excluded from scoring because its label matures after the development boundary. The first January 4, 2016 fit uses **1,254** mature common rows from January 5, 2011 through December 30, 2015, with the latest training label available December 31. Query-label availability cannot move a monthly refit, even when a whole month's query labels are absent.

The constant dependence fit compares every admissible real cubic stationary point and both endpoints. Each slope fit has a numerical global-value certificate with a finite nonnegative gap at most 1e-8 and projected gradient at most 1e-7. The independent verifier checked all **236 scalar fits and 14,946 certificate intervals**; the largest gap was **9.9473e-9**, the largest dependence first-order residual **2.7023e-8**, and no fit used more than **139** of the allowed 32,768 interval splits. This is a numerical certificate to the declared tolerance, not an exact-arithmetic proof of unique parameters.

The raw data remain archival Yahoo/Cboe snapshots. QQQ is an ETF and SPX a price index; observed-date equality does not establish synchronized auction sampling or exchange-calendar completeness. Back-calculated early VIX9D training history, source revisions, ETF tracking and uncertain historical publication latency remain disclosed limitations. The October 20, 2025 numerical fence and November 3 sealed boundary were preserved.

## Verification and retained record

**1,218 repository tests passed before the run**, including 104 new tests. The runner then passed **127 focused checks**. Scoped lint passed for all twelve new source/test files. Pre-run regression failures and their fixes are retained for partial-OHLC arithmetic, an entirely unscored month, inconsistent numerical certificate bounds, and malformed/nonfinite publication records.

Independent verification reconstructed all **4,226×34 source feature cells**, both targets, shared mean and positive-moment fits, every forecast, all certificate intervals, four phase comparisons, twelve bootstrap runs of 99,999 draws, both multiplicity corrections and **114 ledger events** (110 inherited, 2 registered, 2 evaluated). Separate marginal-fit tolerances were specified before any market calculation; no tolerance was changed after fitting.

The manifest pins **191 source/test files, 12 input artifacts and 326 earlier protocol/report artifacts**. All **529 hashes**, plus the protocol hash, match. All nine preceding wave manifests also passed preservation checks after the full test suite. Ten recent waves now contain **58 added comparisons and 121,279 forecasts**; combined with the earlier 54 enumerated comparisons, this is the 112-hypothesis family, not the repository's entire lifetime experiment count. Reused history remains exploratory despite conservative corrections.

- [Prospective design](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/joint_risk/DESIGN.md), [frozen protocol](/Users/byrons/code/trading-vol/ndx-vol-experiment/joint_risk.yaml) and [independent design review](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/joint_risk/DESIGN_REVIEW.md)
- [Complete metrics](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/joint_risk/metrics.json), [independent verification](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/joint_risk/verification.json) and [trial ledger](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/joint_risk/trial_ledger.jsonl)
- [Source feasibility](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/joint_risk/SOURCE_FEASIBILITY.md), [feature contracts](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/joint_risk/FEATURE_IMPLEMENTATION.md) and [numerical certificate protocol](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/joint_risk/DENSITY_NUMERICAL_PROTOCOL.md)
- [Full pre-run tests](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/joint_risk/full_pre_run_tests.txt), [focused checks](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/joint_risk/pre_run_checks.txt) and [independent run log](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/joint_risk/independent_verifier.log)
- [Exportable figure](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/joint_risk/comparison_intervals.pdf)

The [next blinded question](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/joint_risk/NEXT_OFF_DIAGONAL_DESIGN.md) tests the issued forecasts' signed cross products directly, to remove the matrix-inverse weighting mechanism from the score. It would require a separate registered effect threshold, two additional hypotheses and its own verification. It has not been run and cannot change this experiment's verdict.
