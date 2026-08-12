# Findings — clean phase, first evaluation

**Window:** 2025-11-03 → 2026-08-11. 192 scored origins. Frozen `config.yaml`,
pre-registered 7-point quantile grid. Source: `reports/results_clean.md`.

## Verdict: the primary hypothesis failed. Record it as a negative.

**H1 — FAILED.** `chronos_cov` mean QLIKE 0.3688 vs log-HAR-RV 0.3734.
DM = −0.259, **p = 0.796**. The calendar covariates — the original thesis of the
experiment — bought 1.2% of QLIKE, which is indistinguishable from zero at
n=192. This is a well-powered null for a difference of that size, not an
underpowered maybe.

**H3 — FAILED.** Every encompassing coefficient on a non-VXN-fed model is
negative (`har_cum` −0.360, `chronos_uni` −0.216, `chronos_cov` −0.159; none
significant). Nothing here carries information beyond the option surface. A
negative `c_model` is not a backwards win — it is collinearity.

**Interval quality gate — PASSED**, with a caveat. All Chronos variants show no
violation clustering (`p_ind` 0.11–0.53). `chronos_cov_iv` clears it by a hair
at **p_ind = 0.053**; violation clustering is exactly what makes a
calibrated-on-average band useless for premium selling, so if that number drifts
below 0.05 as the sample grows, the best-scoring model fails the gate.

**H2 — descriptive only** at n=6–8, as pre-registered. See below.

## The `cov_iv` result was the input, not the model

`chronos_cov_iv` beats plain HAR at DM = −3.056, p = 0.0026, which survives
Bonferroni across the six tests. But plain HAR is the wrong control: that
comparison only re-establishes that implied vol predicts realized vol, which is
one of the oldest results in the volatility literature. Nearly the whole effect
arrives with VXN — `cov_iv` beats `uni` by 16% QLIKE while `cov` beats `uni` by
0.8%.

**HAR-IV** (log-HAR + log(VXN), five terms, added for this reason) settles it:

| comparison | QLIKE | DM | p |
|---|---|---|---|
| har_iv vs har | 0.3098 vs 0.3734 | −3.477 | 0.0006 |
| **chronos_cov_iv vs har_iv** | **0.3131 vs 0.3098** | **+0.218** | **0.828** |

All of `cov_iv`'s apparent edge over HAR was the VXN input.

**Read that p = 0.828 correctly: it is a failure to reject, not evidence of
equivalence.** At n=192 there is almost no power to detect a 1% QLIKE
difference, so the data are equally consistent with a small real Chronos edge.
The reason to close the thread is not "the models are the same" — it is that a
120M-parameter foundation model whose *upper bound* on advantage over a
five-term OLS is "too small to measure in nine months" does not justify the
operational cost of running it. **The Chronos thread closes on cost-benefit, not
on a proof of equivalence.** If a later window separates them, that is not a
contradiction of this result.

## What survived

**Calibration — weaker than it first appeared.** Plain HAR's 90% intervals cover
84.9% (`p_uc` = 0.027): significantly overconfident, the failure mode that
bankrupts a premium seller. All Chronos variants land at 0.885–0.906 with clean
Kupiec p-values. But HAR-IV covers **0.880 with p_uc = 0.374** — adding VXN fixes
most of HAR's overconfidence too. Against the right control the calibration edge
narrows to a nominal one (0.880 vs 0.885–0.906, none significantly off), so this
is a mild preference, not a result.

**Heavy-earnings slice — the one place the original thesis showed life.**
HAR-X 0.2401 vs HAR 0.5192, and `chronos_cov` 0.1807 vs `chronos_uni` 0.5143:
two independently specified models, same covariate, both roughly halving loss in
the same direction. That is the NDX-concentration argument's first sign of life.
It is n=8 and pre-registered as descriptive — a reason to keep accruing, not a
finding.

**FOMC slice — event features made things worse** (HAR-X 0.2591 vs HAR 0.1867;
`cov` 0.2406 vs `uni` 0.1649). The obvious suspect — a truncated FOMC calendar
mislabelling pre-2022 training rows — does **not** apply: `calendars/fomc.csv`
covers 2016-01-27 → 2027-12-08 (49 decision days in 2016–2021 alone) and all
forecasts here were built after that fix. So this is n=6 noise or a genuinely
badly-fit term, not attenuation from a mislabelled sample.

## Variance risk premium

The one economically real thing in the table. Do **not** read α = −1.301 as a 73%
premium: β = 0.829 ≠ 1, so the intercept is not interpretable alone. Evaluated
on the fitted line at the window's median VXN of 24.1, realized variance is 68%
of implied — **82% in vol terms, about 4.2 vol points of premium** (at VXN = 20
it is 72% / 85% / ~3.0 points).

That is a normal, healthy VRP for the window, and it is visible to anyone who can
chart VXN minus trailing RV. Harvesting it is a short-convexity carry trade whose
risk lives entirely in the tail, and 192 days containing no volatility event says
nothing about that tail.

---

# Update — 2026-08-12: point-in-time weights, new features, TiRex-2

All numbers below are on **point-in-time index weights**. The previous run used
one Invesco snapshot for all history, a look-ahead sitting directly on the
earnings slice. Removing it made `chronos_cov` *worse*, so part of its already
negligible edge was the look-ahead.

## H1 is dead past arguing

`chronos_cov` 0.3722 is worse than `chronos_uni` 0.3719. DM vs HAR **−0.064,
p = 0.95**. The calendar covariates are not merely unhelpful to Chronos-2, they
are marginally harmful. `chronos_cov_iv` vs HAR-IV: **+0.285, p = 0.777**, point
estimate favouring five OLS terms.

## Ranking (clean window, 192 origins, pre-registered grid)

| model | QLIKE | 90% cov | p_uc | p_ind |
|---|---|---|---|---|
| **har_iv_x** | **0.3006** | 0.891 | 0.669 | 0.246 |
| har_iv | 0.3095 | 0.880 | 0.374 | 0.423 |
| chronos_cov_iv | 0.3142 | 0.906 | 0.771 | 0.053 |
| har_ic | 0.3147 | 0.891 | 0.669 | 0.246 |
| har_sv | 0.3371 | 0.926 | 0.224 | **0.959** |
| har_x | 0.3618 | 0.859 | 0.075 | 0.218 |
| chronos_uni | 0.3716 | 0.906 | 0.771 | 0.528 |
| chronos_cov | 0.3719 | 0.880 | 0.374 | 0.158 |
| har | 0.3731 | 0.849 | 0.027 | 0.386 |

## Signed semivariance (Patton-Sheppard) — the best value-per-effort item

`har_sv` swaps the daily term for its signed halves; no new market data, only a
split share from free hourly bars. QLIKE 0.3371 vs HAR 0.3731 (−9.6%,
DM −1.615, **p = 0.108**) — not significant, but ~8x the effect the calendar
covariates ever produced, and it beats every Chronos variant not fed VXN. It
also has the cleanest violation independence in the table (**p_ind = 0.959**).
Constraint: yfinance caps 5-minute history at 60 days, so the split comes from
~7 hourly returns/session and only reaches back to 2023-09-13 (532 pre-window
training rows, just over `min_train_days`). A true 5-minute split needs paid
intraday data and would likely sharpen this.

## Implied correlation — a clean negative for forecasting, a positive for mechanism

`har_ic` (HAR-IV + log COR1M + log VIXEQ) scores 0.3147, **worse** than HAR-IV
(DM +1.496, p = 0.136). The correlation split adds nothing beyond the VXN level.

But it makes the earnings mechanism observable, and the signature is present:

| measure, heavy-earnings vs other days | heavy | other | diff | p |
|---|---|---|---|---|
| avg constituent IV (VIXEQ) | 43.94 | 41.29 | **+2.65** | 0.083 |
| implied correlation (COR1M) | 10.57 | 14.02 | **−3.45** | 0.176 |
| index IV (VXN) | 24.91 | 24.58 | +0.33 | 0.791 |

Constituent vol up, correlation down, **index vol flat** — the two offset, which
is exactly why a 30-day index measure cannot see an earnings day. Directionally
consistent on all three plus Spearman(earnings_wt, COR1M) = −0.293 (p = 0.116).
None individually significant at n=12 vs 166. A check that could have falsified
the story and did not.

## HAR-IV-X — the one pre-specified new model, and the strongest result yet

Motivation, fixed before running: VXN is 30-day constant-maturity and
structurally cannot know a 9%-weight name prints tonight, yet `har_x` and
`chronos_cov` beat `har_iv` on heavy-earnings days while losing everywhere else.

HAC coefficient on `earnings_wt` **alongside** log(VXN):

| sample | coef | HAC t | p |
|---|---|---|---|
| clean window (n=192) | 0.0534 | **3.81** | **0.0001** |
| all data (n=4252) | 0.0314 | **7.06** | **<0.0001** |

Significant in both, same sign — H3-flavoured evidence at h=1 that the earnings
covariate carries information the 30-day surface does not price. Roughly +5.5%
next-day variance per point of index weight reporting.

**Do not overstate this.** It is an *in-sample coefficient*. Out of sample,
`har_iv_x` vs `har_iv` is DM −1.245, **p = 0.215** — the forecast improvement is
not demonstrated at n=192. A reliably nonzero coefficient with an insignificant
out-of-sample gain is the classic pattern, and the honest reading is "worth
accruing toward", not "established". Note also that in the clean window all
three HAR terms are insignificant while log(VXN) carries a coefficient of 3.91 —
a steep slope on 192 observations that may not be stable.

## Heavy-earnings slice: threshold frozen, sample now n=12

The old in-sample 80th-percentile cutoff moved the slice from n=8 to n=7 on a
data correction alone — a researcher degree of freedom on top of the only live
result. Replaced with a frozen absolute cutoff (**≥5% of index weight reports**)
in `config.yaml`, which also yields n=12. Paired per-origin, now printed in
every report:

| pair | mean | better/n | sign p | top day % of gap | ex-top |
|---|---|---|---|---|---|
| har_x vs har | 0.2517 vs 0.4481 | 9/12 | 0.146 | 42% | 8/11 |
| har_iv_x vs har_iv | 0.1846 vs 0.3395 | 8/12 | 0.388 | 39% | 7/11 |
| chronos_cov vs chronos_uni | 0.1671 vs 0.4265 | 10/12 | **0.039** | 49% | 9/11 |

Three independent pairs, all directionally consistent. Bonferroni across them
puts the best at 0.12, so treat none as significant; 2025-11-19 (NVDA) still
carries ~40–50% of each mean gap.

## FOMC: coherent across four specifications

Plain HAR (0.1867) and `chronos_uni` (0.1649) win; everything with an upward
event loading does worse — `har_x` 0.2595, `chronos_cov` 0.2467, `har_iv` 0.1902,
`har_iv_x` 0.2565. Four specifications, three information sets, one direction.
Reading: FOMC days in this window realized *below* what any forward-looking
measure expected, so pricing in a bump over-forecasts. Still n=6, descriptive,
but now a specific testable claim about the regime rather than an anomaly.

## TiRex-2

Univariate performance is on par with Chronos-2 (0.3975 vs 0.3924 on the decile
grid). Its **past-covariate channel is inert here**: an oracle probe — a past
covariate whose final value *is* the target — left RMSE unchanged (0.725 vs
0.707 baseline), while the same oracle in the *future* channel collapsed it to
0.049 (Chronos-2: 0.156). So `tirex_cov_iv` never meaningfully saw VXN, and its
DM +3.63 vs HAR-IV measures a broken channel, not a model. Routing VXN through
the working channel (`tirex_cov_ivf`, last observed value carried flat — same
information set) fixes it: QLIKE 0.3878 → **0.3296**, DM vs HAR-IV +0.839,
p = 0.402. Same verdict as Chronos-2: matches the linear control, does not beat it.

## Dealer gamma — not feasible retrospectively

Requires historical end-of-day option open interest by strike. yfinance exposes
only the *current* chain, and the clean window is in the past, so it cannot be
backfilled from free sources; CBOE DataShop or equivalent is required. It could
be accrued forward by archiving chains daily from now, which would make it
testable in 6–12 months. Not implemented.

---

# Update — 2026-08-12 (second): the diagnostic window reverses the earnings story

Every number in this section was independently recomputed by a separate
verification pass before being written down; all matched.

## The most important table was in `results_diagnostic.md`, and it points at null

The leakage boundary exists for pretrained models. HAR, HAR-X, HAR-IV and
HAR-IV-X are expanding-window rolling regressions with no pretraining, so the
diagnostic window is a **valid out-of-sample test for the OLS family**: 2,463
origins and **157 heavy-earnings days** — thirteen times the clean window's 12.
On that sample:

- `har_iv_x` vs `har_iv`: **DM = −0.031, p = 0.975**. Dead flat at n=2463.
- Paired on heavy-earnings days: `har_x` beats `har` on **71/157 (45%)**;
  `har_iv_x` beats `har_iv` on **69/157 (44%)**. Both **below half**, while the
  slice *means* still improve (0.3522 vs 0.4417; 0.2900 vs 0.3874).

Win rate below half with an improving mean is the signature of a term that pays
off on a handful of large days and costs a little on all the rest. That is a
kind of usefulness for a vol forecaster, but it is **not** "the covariate
carries information the surface doesn't price," and it is the opposite
direction from the clean window's 9/12. The in-sample coefficient (t = 7.06 on
4,252 rows) sitting next to an out-of-sample DM of p = 0.975 on 2,463 origins
is the textbook in-sample/out-of-sample gap. **The well-powered test points at
null; yesterday's HAR-IV-X section is superseded by this one.**

## Concentration test: pre-specified, then the regressor turned out degenerate

The one testable defence was pre-specified in `config.yaml` before running:
regress the paired win rate on contemporaneous concentration; b>0 at one-sided
α=0.05 → regime effect; otherwise the clean-window 9/12 is noise.

Disclosure: **the pre-specified regressor was degenerate by construction.**
`pit_weights.parquet` renormalizes the tracked basket to its snapshot total
(55.4%) on every date — a property documented in `build_pit_weights`'s own
docstring and not connected when the test was specified. The basket-sum
regressor is constant; the logit is unestimable. This was discoverable ex ante
and is a specification error, not an outcome-driven change. Because the yearly
win-rate table had been seen by the time this was caught, **no amended
regressor can claim pre-registration status**; everything below is labeled
post-hoc.

- Yearly win rate (`har_x` vs `har`, 157 days): no trend. Best year 2024
  (0.733, n=15); **worst year 2025 (0.286, n=14) — the lowest win rate in the
  highest-concentration regime**, the opposite of the defence's prediction.
- Post-hoc day-level logit, win ~ reporting weight: b = +0.040, one-sided
  p = 0.086; tercile win rates 0.358 / 0.519 / 0.481 — not monotone.

**Decision, per the pre-committed criterion: the regime defence is not
supported. The clean-window 9/12 is treated as consistent with noise, and the
earnings-covariate story rests entirely on the n=500 accrual gate.**

## H3 at proper HAC lags: one modest genuine positive survives

`maxlags=21` sat at the floor for a 30-calendar-day (~21-trading-day overlap)
horizon; the code default is now 32 (≈1.5h; logged in AMENDMENTS), with 40 as a
robustness check. Diagnostic window:

| model | c_model | p @21 | p @32 | p @40 |
|---|---|---|---|---|
| har_cum | +0.196 | 0.035 | 0.047 | 0.051 |
| persistence_cum | +0.078 | 0.006 | 0.011 | 0.014 |

`persistence_cum` survives cleanly; `har_cum` sits exactly on the boundary.
So there is a genuine H3-flavoured pass on the well-powered window: RV history
carries statistically detectable information beyond VXN at 30 days over
2016–2025. Keep its size honest: adding the model forecast moves R² from 0.550
to ~0.556. Detectable, economically thin, and it does not replicate in the
clean window (c negative, n.s.) — a real but small effect, not a trade.

## TiRex-2 `cov_ivf`: fails the interval gate, and the sign-flip rule fired

Two corrections to yesterday's TiRex verdict, both against it:

1. **It fails the interval gate** — the only model in the decile table to do
   so: 80% interval covers **0.740** (p_uc = 0.043), uniformly overconfident
   rather than clustered (p_ind = 0.948). Carrying VXN flat across 21 future
   steps plausibly tells the model the future is more determined than it is.
   Since the object of interest here is the range, a QLIKE win from a model
   that under-covers is the wrong trade, and yesterday's QLIKE-only framing
   overstated it.
2. **The pre-stated sign-disagreement rule fired and is now applied.** Full
   window: `har_iv` 0.3167 beats `tirex_cov_ivf` 0.3296. Post-publication
   subwindow (n=28): `tirex_cov_ivf` 0.1882 beats `har_iv` 0.1991. Per
   `LEAKAGE_TIREX2.md`: believe neither; keep accruing. (The level drop on
   those 28 days is a calm-regime effect common to all models — only the
   ranking is readable there, and it flips.)

## HAR-SV: the window-matched control attributes the gain to the split

`har_sv` trains from 2023-09-13 (532 rows); `har` on ~2,500 — so the −9.6%
could have been recency. Refitting plain HAR on the identical window and
scoring identical origins (n=188):

| model | QLIKE |
|---|---|
| HAR, full history | 0.3709 |
| HAR, window-matched | 0.3731 |
| HAR-SV, same window | **0.3388** |

Recency contributes ~nothing (window-matched HAR is marginally *worse*;
DM +0.23, p = 0.82). The semivariance split itself is the whole gain:
**DM = −1.98, p = 0.049** on the matched sample. That survives the control it
had to survive — with two caveats stated plainly: p = 0.049 is a boundary
number, and this is one more test drawn against the same 192 clean days.
Patton–Sheppard has strong prior support, which is why it stays the most
promising non-VXN direction; it is not yet a finding here.

## Multiplicity, said out loud

This update round added seven specifications across two grids. Most were
motivated in advance and every null above is reported, but with ~10 models in
the pool, "har_iv_x is best at 0.3006" carries almost no ranking information
when its edge over second place is p = 0.21 — and the diagnostic window now
shows its defining term flat at n=2463. **No further specifications until the
pre-committed gate (500 origins or 2027-06-30).** The gate is permission to
look, not permission to stop at a favourable p.

---

# Update — 2026-08-12 (third): H3 withdrawn, asymmetry confirmed, carry fails

## H3 does not survive the removal of overlap — withdrawing yesterday's positive

Newey-West is asymptotic in *independent* observations, and 2,463 origins with
30-calendar-day overlapping targets contain only ~117 non-overlapping blocks.
Raising the lags corrected the direction without reaching the right size. Taking
every 21st origin and refitting with ordinary errors, and — the step that
mattered — **repeating it at all 21 phase offsets** rather than one:

| model | HAC n=2463 | blocks n=118 (phase 0) | median p across 21 phases | share p<0.05 | c range |
|---|---|---|---|---|---|
| har_cum | +0.196, p=0.047 | +0.232, p=0.123 | 0.212 | **0.05** | +0.056 … +0.325 |
| persistence_cum | +0.078, p=0.011 | +0.160, p=0.013 | 0.187 | **0.24** | −0.117 … +0.174 |

The phase-0 `persistence_cum` result that looks significant is one of the 24% of
phases that happen to reach it; the median phase is p = 0.19 and the coefficient
changes sign depending on where the blocks start. **The HAC number was borrowing
power from overlap. The "genuine H3 positive" reported in the previous update is
withdrawn.** Reporting a single block phase would have replaced one artefact
with another.

## Orthogonalisation: the diagnosis was right, the mechanism needs one correction

Regressing the forecast on log(VXN) and using the residual **cannot change the
p-value** — by Frisch-Waugh-Lovell, `[iv, e]` spans the same column space as
`[iv, m]`, so the coefficient and t-stat on `e` are identical to those on `m`.
Verified empirically: both +0.1958, t = +1.98, p = 0.0474 for `har_cum`.

But the underlying diagnosis was exactly right, and the orthogonalisation is
what exposes it:

| model | R²(m ~ iv) | sd(e) | c | standardised effect c·sd(e) |
|---|---|---|---|---|
| har_cum | **0.798** | 0.337 | +0.196 | **+0.0660** |
| persistence_cum | 0.555 | 0.788 | +0.078 | **+0.0616** |

`har_cum` is far more collinear with VXN, its orthogonal component has less than
half the spread, and that inflates the standard error on its coefficient. Larger
point estimate, weaker p-value — a collinearity signature, as diagnosed. Put on
a common scale, **the two carry essentially identical incremental information**
(+0.066 vs +0.062). "Persistence survives, HAR doesn't" was never a statement
about which forecast is more informative. The report now carries the
standardised effect so the comparison is interpretable.

## Return asymmetry earns its place across a decade — semivariance is real

LHAR (Corsi-Renò): HAR plus negative-return terms at daily/weekly/monthly
aggregation, from daily data back to 1999, so it is not capped by yfinance's
intraday history. Diagnostic window, **n = 2463**:

| model | QLIKE | DM vs HAR | p |
|---|---|---|---|
| har | 0.3815 | — | — |
| **har_lev** | **0.3618** | **−5.367** | **<0.0001** |

Decisive on a well-powered sample. The mechanism underneath signed semivariance
is real, which retrospectively supports the intraday split as a sharper
instrument for it rather than a 188-day fluke — while leaving that boundary
p = 0.049 exactly as weak as it was. `har_lev` does not beat `har_iv` (0.3438):
VXN remains the stronger single input. But it is the best model here that uses
**no market data at all**.

`har_lev` is registered in `DIAGNOSTIC_ONLY` and is **mechanically suppressed
from clean-phase reports**, so testing it during the freeze cannot become a
peek. It enters the clean window at the gate or not at all.

## The carry study: conditional selection FAILS its pre-committed criterion

Pre-registered in `config.yaml` before implementation; diagnostic window only;
the clean window stays untouched as confirmation at the gate. Short a 30-day
variance swap when richness (implied vs the RV-history-only `har_cum` forecast,
which never sees VXN) exceeds its median, versus always short. Inference on
~117 non-overlapping trades, all 21 phase offsets. P&L in vega-equivalent vol
points per trade.

| | unconditional | conditional |
|---|---|---|
| trades | 117 | 59 |
| mean P&L | +1.714 | **+2.381** |
| hit rate | 0.816 | 0.830 |
| 5% CVaR | −33.30 | **−22.71** |
| worst trade | −105.34 | −49.63 |
| max drawdown | 118.16 | 55.13 |

Selection improves the mean by 39% *and* improves the tail, and 17/21 phases
agree on the mean. But **bootstrap median p = 0.50, and not one of the 21 phases
reaches p < 0.05.** Mean improves ✓, CVaR not worsened ✓, significant ✗ →
**VERDICT: FAIL** on the pre-committed criterion. The honest reading is that at
~117 independent trades this design cannot distinguish a 39% edge from noise.

### The tail, examined rather than summarised

The five worst trades in the entire decade are all the same event — entering
short variance in the week before the COVID crash:

| origin | VXN | P&L (vol pts) | taken by the signal? |
|---|---|---|---|
| 2020-02-19 | 18.58 | **−196.7** | **yes** |
| 2020-02-18 | 19.12 | −185.6 | **yes** |
| 2020-02-20 | 20.33 | −177.4 | no |
| 2020-02-21 | 22.46 | −158.2 | **yes** |
| 2020-02-24 | 29.23 | −124.8 | no |

**Three of the five worst trades were taken by the conditional rule.** Variance
looked rich right up to the crash — that is what a low-VXN pre-event regime
looks like, and richness-based selection walked straight into it. A single trade
lost 196.7 vol points against a mean gain of 1.71, a ratio of ~115:1, and the
unconditional max drawdown of 118 is ~59% of the decade's entire cumulative P&L.

This is the short-convexity risk stated as a measurement instead of a caveat.
The signal improved the average tail while providing no protection against the
one tail that mattered — precisely the failure mode that averages hide.

### This is a mechanism failure, not an underpowered result

The p = 0.50 says only that conditional cannot be distinguished from
unconditional. The tail table says something categorically worse: **the signal
selected into the worst event of the decade.** That is not sampling noise that
more origins would average away — it is the mechanism running backwards.
Implied variance looks rich relative to a backward-looking forecast *precisely
when* realized vol has been quiet and the shock has not arrived. A rule that
trades on "implied is rich versus my forecast" is therefore structurally biased
toward maximum exposure immediately before regime breaks. More data reproduces
this rather than repairing it.

The economics kill it independently of the statistics. A mean of +1.71 vol
points against a worst realization of −196.7 is ~115:1; surviving that draw
requires sizing at which the edge is economically meaningless. And the baseline
being improved on is itself unattractive — unconditional carry surrendered ~59%
of the decade's cumulative P&L in a single drawdown. Both legs fail, for
different reasons.

**Conclusion: forecasting variance well does not translate into trading the
premium here, and the conditional-selection idea has an identified adverse
mechanism rather than an inconclusive test.** The clean window cannot
rehabilitate it: 192 days containing no vol event has strictly less tail
information than the ten years just examined. Anything past this point needs
someone who trades short convexity professionally; the binding constraint is
not a p-value.

## Ledger-closing test: does leverage survive alongside VXN? No — it survives.

VXN embeds skew, which is the same economics as the leverage effect, so
`har_lev` might have been VXN by another route. Diagnostic window, n=2463:

| model | QLIKE | DM vs HAR-IV | p |
|---|---|---|---|
| har_iv | 0.3438 | — | — |
| **har_iv_lev** | **0.3376** | −1.410 | 0.159 |

Forecast gain over HAR-IV is directional but not significant. The coefficient
test is not ambiguous:

| term | coef | HAC t | p |
|---|---|---|---|
| lev_d | −6.49 | −2.73 | 0.0063 |
| lev_w | −35.02 | −6.26 | <0.0001 |
| lev_m | −29.41 | −2.33 | 0.0197 |
| liv (log VXN) | +1.54 | +11.36 | <0.0001 |

**Joint Wald on the three leverage terms alongside log(VXN): χ² = 62.63,
p < 1e-6.** All three individually significant, all with the correct sign
(more negative returns → higher forecast variance). Adding VXN drops the joint
statistic from 103.19 to 62.63, so VXN absorbs roughly 40% of the leverage
information and a large, decisively significant remainder survives. R² rises
0.5347 → 0.5644.

### The symmetric test — the one that decided the earnings case

An in-sample Wald next to an insignificant out-of-sample DM is exactly the
pattern used to reject HAR-IV-X. The check that settled that case is the paired
per-origin count, and it had not been run here. At n=2463 the sign test is very
sharp (53% is already p<0.01):

**`har_iv_lev` vs `har_iv`: 1366/2463 = 55.5% of days individually improve,
sign p = 6.5e-08.** The median improves too (0.1437 vs 0.1486), and the win rate
is **unchanged at 55.3% after dropping the ten largest days** — broad-based, not
a handful of wins carrying a mean. Leverage passes the same standard that the
earnings term failed, on out-of-sample evidence rather than an in-sample Wald.

### What "not subsumed" does and does not claim

Two things are true at once and both belong in the sentence:

- The leverage coefficients are **reliably nonzero alongside log(VXN)** —
  statistical existence, decisively (χ² = 62.63, p < 1e-6), corroborated
  out-of-sample by the 55.5% paired count.
- The forecast value they buy is **small**: QLIKE 0.3438 → 0.3376, **+1.8%**,
  DM p = 0.159. At n=2463 that test has real power against moderate effects, so
  p = 0.159 is not merely "underpowered" — it is mild evidence that whatever
  gain exists is genuinely small.

**Correct statement: not subsumed, and worth about 1.8% of QLIKE that does not
survive a significance test.**

### Context, so this does not inflate in the retelling

Return asymmetry is among the most replicated results in the volatility
literature — Black (1976), then EGARCH, then Corsi-Renò. **This experiment did
not discover it and did not need to.** Calling it "the one signal that survived"
invites reading a fifty-year-old effect as a finding.

The genuinely novel number here is the **partial absorption**: the joint Wald on
the leverage terms falls from **103.19 to 62.63** when log(VXN) enters the
model. The option surface prices roughly 40% of the leverage information and no
more. That is the sentence worth keeping — smaller, more defensible, and
actually new.

### A correction the full-sample test forced on the earnings verdict

Running the paired count on all origins rather than the heavy-earnings subset
reverses the shape I previously described. Earlier, from the n=157 slice, I
characterised the earnings term as "paying off on a handful of large days and
costing a little on the rest." On the full n=2463 sample it is the opposite:

| pair | win rate | sign p | mean Δ | top-10 share of mean gap |
|---|---|---|---|---|
| har_iv_x vs har_iv | 61.3% | 1.4e-29 | **+0.0%** | **5300%** |
| har_x vs har | 60.7% | 1.3e-26 | **−0.6%** | −244% |

The earnings terms win on ~61% of days by small amounts and give it all back on
a few large ones, netting to zero (or worse). Many small wins funded by rare
large losses — a *less* attractive profile than the one I described, and the
same shape as the carry study's failure. Both readings are correct on their own
samples; the full-sample one is the relevant one for "is this term useful," and
the verdict stays null. The report now prints win rate and mean difference side
by side at every sample size so this cannot be read one-sidedly again.

The original-model ledger has no open items.

## Separate orthogonal-signal holdout: directional term slope, not confirmed

This study was frozen in `signal_study.yaml` and is mechanically unable to read
the original clean window. It used discovery (2016–2021) to select at most one
of seven combinations, then spent confirmation (2022–2025-10-17) once.

Discovery selected lagged `log(VIX9D/VIX)`: QLIKE 0.3311 versus 0.3428 for a
timing-safe HAR-IV-LEV baseline (+3.39%). Cross-asset stress (HYG/TLT/GLD/USO/UUP)
and QQQ market state (volume/overnight share) were null or harmful, alone and in
combinations.

On 952 sealed confirmation origins, term slope reached QLIKE 0.3260 versus
0.3358 (+2.92%) and won 519/952 days (54.5%), with sound interval independence
(p=0.910). It **failed** because the pre-registered two-sided DM p-value was
0.1016, above 0.05. An independent implementation reproduced every stored
metric to 1e-12. Conservative post-hoc HAC checks were also non-significant.

The improvement decayed from 5.3%/7.0% in 2022/2023 to 0.2%/0.9% in 2024/2025.
Correct disposition: observe it prospectively without tuning; do not call it a
validated signal and do not reuse this confirmation period.

## Historical weights: free QQQ history recovered, but intentionally not scored

The repository's old `pit_weights.parquet` is a current-13-name market-cap
reconstruction with a mechanically constant 55.43% aggregate, not historical
Nasdaq-100 weights. The public SEC path supplied by the user yielded 27 tested
QQQ N-PORT snapshots (2,746 holdings) from 2019-09-30 through 2026-03-31.
Disclosures arrive 50–62 days after the portfolio date and are joined by exact
SEC acceptance timestamp at a 16:00 ET origin.

The disclosed positive-equity top-10 weight moved 55.13% → 46.85%, and HHI
0.0459 → 0.0312. This is a valid quarterly point-in-time fund proxy, not exact
daily index history. It begins too late for most of discovery and contains only
27 slow snapshots, so inserting it after seeing the signal results would be
post-hoc contamination. It is stored for a future, separately frozen holdout.

An attempted annual backfill to 2004 was later paused before completion. The
2004 and 2005 schedules parse and reconcile under pre-written regressions, but
the next SEC archive request repeatedly returned HTTP 403. No combined dataset
or concentration feature was promoted; see `PAUSED_HISTORICAL_WEIGHTS.md`.

## SKEW repairs the observed tail mechanism, but fails its frozen usability gate

The carry rule's identified defect suggested exactly one post-hoc diagnostic:
retain the original richness signal, but veto a trade when one-session-lagged
Cboe SKEW is above its trailing 252-session 80th percentile. The rule and a
70%-participation floor were frozen before SKEW was fetched or joined to P&L.

Across all 21 non-overlapping phase offsets, the SKEW veto rejected the three
known pre-COVID adverse entries and improved the average 5% CVaR from -22.71 to
-10.90, worst trade from -49.63 to -14.32, and maximum drawdown from 55.13 to
19.18. Mean P&L rose from +2.38 to +2.80 vol points. But it retained only 63.3%
of richness trades versus the frozen 70% minimum, so the strict verdict is
**FAIL**.

The diagnostic still identifies something real about the failure mechanism.
Richness selected 61.8% of high-SKEW origins versus 45.1% of lower-SKEW origins;
36.7% of richness trades occurred in a high-SKEW regime. Lagged SKEW's
correlation with lagged VXN was only -0.270, so the veto is not another ATM-level
copy. The backward-looking rule calls variance rich disproportionately often
while option wings are already expensive—the adverse-selection story is
supported, but the simple veto is too indiscriminate under the registered
criterion.

This cannot validate a strategy: February 2020 motivated the rule, and the
original carry frame itself used a full-window median and same-date published
VXN daily close. Future validation must freeze the historical cutoff and use
lagged Cboe inputs or timestamped pre-close quotes.

---

# Standing summary — what is true at the end

Written 2026-08-12 at the close of the pre-gate work. This is the section to
read first.

**Volatility is forecastable, and a five-term linear regression does it
competently.** HAR-IV reaches QLIKE 0.3095 on the clean window with honest
intervals (coverage 0.880, p_uc 0.374, p_ind 0.423). It is both the
best-scoring and among the best-calibrated models.

**Nearly all of that comes from VXN.** The option surface prices next-day
variance well, and neither foundation model extracted anything a five-term
regression missed — Chronos-2 vs HAR-IV p = 0.777, TiRex-2 (via its working
covariate channel) p = 0.402. Two 2026-vintage foundation models, one linear
control, no daylight.

**Return asymmetry is real, well known, and only partly priced by the surface.**
LHAR beats HAR at DM −5.367 (n=2463); the leverage terms stay jointly
significant alongside log(VXN) (χ² = 62.63, p < 1e-6) and improve 55.5% of days
individually (p = 6.5e-08). The effect itself is Black (1976) and this
experiment neither discovered nor needed to. **The novel number is the partial
absorption: the joint Wald falls 103.19 → 62.63 when VXN enters, so the option
surface prices ~40% of the leverage information and no more.** The forecast
value of the unpriced remainder is small — +1.8% QLIKE, DM p = 0.159.

**Null:** calendar covariates (H1, p = 0.95, marginally *harmful*); implied
correlation beyond the VXN level; the earnings mechanism out of sample (DM
p = 0.975, win rate 45% at n=157, no concentration trend); information beyond
the surface at 30 days (H3, withdrawn after non-overlapping-block testing);
cross-asset and market-state composites. Lagged VIX9D/VIX was directionally
positive in a separate holdout but did not confirm (DM p = 0.1016).

**The variance risk premium is real and structurally hard to harvest.** ~3.9–4.2
vol points, visible to anyone charting VXN minus trailing RV. Conditional
selection does not pass against unconditional carry. A post-hoc SKEW veto
supports the adverse-selection mechanism and dramatically improves historical
tail aggregates, but fails its frozen 70%-participation floor (63.3%) and is not
out-of-sample evidence.

**Four clean negatives and three self-withdrawn positives.** The version of this
project that found significance would very likely have been wrong. What is worth
keeping is not a model — it is the harness: the leakage boundary, the frozen
pre-registration, the mechanically enforced diagnostic-only fence, the paired
per-origin reporting that never lets a mean travel without its sign count, and
the block testing that took back two of my own results.

## Standing rule

Let the clean window accrue. Do not re-run hunting for significance; every
additional model tested on these same 192 days is another comparison against the
same noise. Any model handed VXN must be reported against HAR-IV, not HAR.
