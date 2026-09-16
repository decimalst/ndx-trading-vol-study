# Methodology review and new orthogonal-signal tests

Review date: 2026-09-06. Scope: the current repository and a new, additive
historical experiment. Existing specifications, reports, and accrual gates
remain unchanged. The checkout began clean on `main`, using Python 3.11.7.

## Conclusions from the review

The useful standard is incremental out-of-sample forecast improvement beyond
what realized variance, implied volatility, and negative returns already tell
us. Low correlation alone is insufficient, as is a significant fitted
coefficient. The original historical confirmation sample has been inspected;
another historical experiment can identify leads, but cannot provide new
confirmation merely by naming a split "confirmation".

1. **Information timing must be explicit.** Generic HAR-IV still uses same-day
   VXN (`src/models.py`), whereas the older orthogonal study correctly delays
   daily Cboe closes one trading session (`src/signal_study.py`). Cboe closes
   can include observations after 16:00. The new experiment delays VXN, VIX,
   VIX9D, and VVIX. QQQ and ETF session summaries assume idealized observation
   at the close; this is a forecasting experiment, not an executable fill test.
2. **Use a consistent variance point forecast.** The old signal runner still
   uses truncated quantile integration, a known frozen estimator limitation.
   The new experiment uses exact Duan smearing from each model's own training
   residuals. This corrects log-to-level conversion under a stable residual
   distribution assumption; it does not guarantee calibration under regime shifts.
3. **Serial dependence exists even at a one-day horizon.** The old daily DM
   implementation uses zero covariance lags. An independent rescore of the
   stored diagnostic HAR-IV-X minus HAR-IV smearing losses gives lag-one
   correlation 0.136894 and Ljung–Box(10) p=3.76e-48. Applying HAC(22) to the
   old earnings equivalence calculation raises p from 0.012663 to 0.035393;
   this check does not overturn that verdict. The new study fixes 21-, 63-,
   and 126-session block sensitivities plus HAC(126) before scoring.
4. **Ranking and forecast loss answer different questions.** The documented
   annual-fold AUC artifact can reward between-year differences with no
   within-year information. This study directly scores continuous variance
   forecasts with paired QLIKE, with annual and fixed-period breakdowns.
5. **Keep targets and samples comparable.** The repository uses both
   log(mean variance) and mean(log variance) as HAR features. This study fixes
   the former, rebuilds a uniform daily GK-plus-overnight target, and gives
   all arms identical training and scoring rows. Five-day targets require all
   five future sessions and cannot enter training until the last is observed.
6. **The target remains a proxy.** GK-plus-overnight is not five-minute realized
   variance. Historical Yahoo prices are a current-vintage snapshot, not a
   vintage archive; raw overnight gaps retain dividend/corporate-action
   limitations. QLIKE's theoretical proxy robustness requires assumptions
   about proxy bias that this experiment does not establish. Any surviving
   lead needs another variance measurement and genuinely new observations.

Relevant existing records: `reports/METHODOLOGY_FORK.md`,
`reports/AMENDMENTS.md`, `docs/OPEN_FINDINGS.md`, and `reports/FINDINGS.md`.

## Four fixed candidate hypotheses

| Candidate | Hypothesis | What it is not |
|---|---|---|
| VVIX | VIX-option uncertainty may predict future QQQ variance beyond volatility levels and short-term slope. | Direct NDX implied volatility or dealer inventory. |
| Realized-variance dispersion | Equal HAR levels can conceal different recent variability in log variance. | Realized quarticity or an independent external data source. |
| QQQ–TLT correlation | Recent stock–bond co-movement may distinguish hedging regimes beyond unsigned cross-asset stress. | Proof of a causal inflation or policy mechanism. |
| QQQ close location | A close near a range extreme may add information beyond net negative returns. | Measured order flow or an intraday semivariance estimate. |

The last three are new summaries of existing information. VVIX is the one
additional external measurement. There is no combination search. The stronger
baseline includes HAR, lagged VXN and VIX, leverage, lagged VIX9D/VIX, and the
previously tested cross-asset and QQQ market-state composites. Each candidate
is projected on that baseline with training-only coefficients and added alone.
The projection's R-squared describes linear redundancy, not economic independence.

## Design fixed before the first new score

The executable specification is `orthogonal_round2.yaml`. Forecast origins are
2016-01-04 through 2025-10-17, with all targets ending no later than 2025-10-20.
The protected phase starting 2025-11-03 contributes neither inputs nor targets.
Models refit at the first eligible origin of each calendar month, use at least
500 expanding training rows, and predict one- and five-session average variance.
All models use the same refit schedule; no model or threshold is selected.

Eight candidate/horizon tests comprise the family. Each uses the largest
two-sided p-value across the three block lengths and HAC(126), followed by
Holm correction across all eight. A lead qualifies for an **exploratory
shortlist** only if corrected p<0.05, full-sample QLIKE improves at least 1%,
and the gap favors it in each of 2016–2019, 2020–2022, and 2023–2025.
Otherwise evidence is inconclusive; a failed gate is not an equivalence test.
The three intervals, years, five-day sampling phases, and nominal minimum
detectable effects remain visible regardless of direction.

Synthetic timing, future-perturbation, missingness, target-completion,
residualization, and estimator checks precede empirical scoring. A persistent
AR(1) null simulation also checks that the conservative interval envelope
covers zero in at least 90% of 200 replications before scoring. Passing this
limited calibration check does not establish coverage for nonstationary market
losses or adjust for the repository's entire historical search. An independent
verifier reconstructs raw features, targets, training samples, and every
prediction using direct augmented regressions rather than producer functions.

## Sources supporting the design

- [Cboe VVIX factsheet](https://cdn.cboe.com/resources/education/research_publications/vixfactsheet2019.pdf)
  describes the VIX-option-based uncertainty measure; it does not establish
  the forecasting hypothesis tested here.
- [Cboe VIX data timing](https://datashop.cboe.com/faqs) documents late index
  calculations, supporting a conservative daily-close delay.
- [Patton (2011), volatility forecast comparison with imperfect proxies](https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf)
  motivates QLIKE with explicit conditions on the proxy.
- [Federal Reserve discussion of monetary policy and stock–bond correlation](https://www.federalreserve.gov/newsevents/speech/clarida20191112a.htm)
  motivates the regime hypothesis; it is not evidence of predictive value here.
