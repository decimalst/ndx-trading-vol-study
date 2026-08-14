# Findings — clean phase, first evaluation

**Window:** 2025-11-03 → 2026-08-11. 192 scored origins. Frozen `config.yaml`,
pre-registered 7-point quantile grid. Source: `reports/results_clean.md`.

## Verdict: the primary hypothesis failed. Record it as a negative.

**H1 — FAILED TO REJECT. Not a null.** `chronos_cov` mean QLIKE 0.3719 vs
log-HAR-RV 0.3731, DM = −0.065, **p = 0.948**, n=192. The calendar covariates —
the original thesis of the experiment — bought 0.3% of QLIKE.

> **Corrected 2026-08-13.** This block previously read "0.3688 vs 0.3734,
> DM = −0.259, p = 0.796 … a well-powered null for a difference of that size,
> not an underpowered maybe." Both halves were wrong. The numbers appear
> nowhere else in the repository and do not reproduce from the code; the
> figures above do. And the design is not well-powered — it is the opposite:
>
> | quantity | value |
> |---|---|
> | minimum detectable effect at 80% power | 0.0525 = **14.1% of HAR's loss** |
> | MDE ÷ observed gap | **43×** |
> | power against the gap actually observed | **0.050** |
> | origins needed to resolve that gap | **357,040** |
> | 95% CI on the gap | [−9.5%, +10.2%] of HAR QLIKE |
> | TOST vs a 3% margin | p = 0.297 — does **not** reject non-equivalence |
>
> Power of 0.050 against the observed effect means the test is no more likely
> to detect it than to fire by chance. The corrected evaluator's verdict is
> **`inconclusive`**, not `equivalent` (`reports/results_clean_inf.md`). H1 is
> unresolved on this window, not answered.
>
> What *is* established, on the 13× larger diagnostic window: the calendar
> covariates are **equivalent** to their controls at n=2463, p_TOST = 0.013
> (`reports/results_diagnostic_v2.md`). Cite the diagnostic window for the
> calendar-covariate null. The clean window cannot carry it.

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

## H1 is unresolved on this window — and settled against on the diagnostic one

*(This section was headed "H1 is dead past arguing". Retitled 2026-08-13: on
the clean window the corrected verdict is `inconclusive`, with power 0.050
against the observed gap. See the correction in the verdict block above.)*

`chronos_cov` 0.3722 is worse than `chronos_uni` 0.3719. DM vs HAR **−0.064,
p = 0.95**. `chronos_cov_iv` vs HAR-IV: **+0.285, p = 0.777**, point estimate
favouring five OLS terms.

Every one of those is a failure to reject on a design whose MDE is 14% of HAR's
loss — they are consistent with the covariates being useless *and* with their
being worth several times any plausible effect. The sentence "the calendar
covariates are not merely unhelpful to Chronos-2, they are marginally harmful"
is a reading of a point estimate with a 95% interval spanning ±10% of HAR's
QLIKE, and is withdrawn.

The claim that survives is the diagnostic-window one, where n=2463 gives the
equivalence test real power: `har_x vs har` is **equivalent** at p_TOST = 0.006
and `har_iv_x vs har_iv` at p_TOST = 0.013. That is a positive finding of no
effect. It is the finding the clean window was always too short to produce, and
it is what should be cited for H1.

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
covariates ever produced, and it beats every Chronos variant not fed VXN.

**Estimator-dependent, flagged 2026-08-13.** Under the corrected point-forecast
estimator (`reports/METHODOLOGY_FORK.md`), with both sides exactly smeared on
the identical 188 origins, this reads QLIKE 0.3201 vs 0.3624, DM **−2.163,
p = 0.0318** — significant at the pre-registered α. Nothing about the data
changed; only the estimator did. This is the one comparison in the repository
whose significance turns on that fix, and it is why the fork's "the estimator
changes no rankings" claim was withdrawn. It does not promote `har_sv` to a
confirmatory result: its specification date is unrecorded, so
`spec_registry.yaml` treats it as exploratory, and n_req = 315 against n = 188.

It
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
peek.

**That claim was too strong, and the exception is recorded rather than
quietly repaired.** Until 2026-08-13 the suppression covered the clean
*report* but not the clean *window*: the corrected fork's replication panel
scored `har_lev` and `har_iv_lev` on all 192 clean origins and published their
win rates, mean gaps and a `replicates? yes` verdict. The draw is therefore
~38% spent for these two models (192 of the ≥500 gate origins), with direction
known. The guard now lives in `_qlike_series` and is enforced by
`tests/test_methodology.py::TestDiagnosticOnlyQuarantine`. When the gate opens,
`har_lev`'s clean-window result must be read as carrying a 192-origin peek —
not as a clean draw. Nothing in the return-asymmetry conclusion above depends
on it; that rests on the diagnostic window at n=2463.

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
  DM p = 0.159 under the frozen point-forecast estimator.

> **This one reverses under the corrected estimator, 2026-08-13.** With both
> sides exactly smeared on the same 2,463 origins, `har_iv_lev vs har_iv` reads
> **DM −2.166, p = 0.0304** — significant, verdict `A better`
> (`reports/results_diagnostic_v2.md`). It is one of only three DM comparisons
> in the entire repository that cross α=0.05 when the estimator is corrected,
> and it is the one that matters most, because it is the project's clearest
> positive measurement.
>
> The sentence that stood here — "worth about 1.8% of QLIKE that does not
> survive a significance test" — is therefore **estimator-dependent, and false
> under the estimator the methodology fork calls correct**. It is withdrawn.
>
> Do not over-read the reversal either. `har_iv_lev` is an exploratory
> specification (`spec_registry.yaml`: specified 2026-08-11, after the phase
> opened), the TOST returns `inconclusive` at p_TOST = 0.321 under both
> estimators, and n_req = 4,120. The honest statement is: **the gain is small,
> its significance depends on which point estimator is used, and the frozen
> report's p = 0.159 should not be quoted on its own.**

**Correct statement: not subsumed; worth about 1.8% of QLIKE, whose statistical
significance is estimator-dependent (p = 0.159 frozen, p = 0.030 corrected) and
which no equivalence test resolves either way.**

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

The improvement decayed monotonically from 5.3%/7.0% in 2022/2023 to 0.2%/0.9%
in 2024/2025. That non-stationarity is more informative than p=0.1016. The
natural interpretation is a regime-conditional term-slope signal: useful while
the front end is dislocated, negligible otherwise. Because the yearly pattern
has now been seen, this is a prospective-observation hypothesis only. Do not
fit the interaction or reuse this confirmation period.

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
post-hoc contamination. The sharp 2023-Q3 drop reflects Nasdaq's special
rebalance to reduce overconcentration. This is adverse—not neutral—to the
earnings-concentration defence and joins the flat yearly win rates and 2025 low
as a third independent contradiction. The direction is now known, so the
existing N-PORT data cannot support a later clean test. Within this project the
hypothesis is effectively closed unless genuinely future observations or an
untouched asset are acquired.

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

The gate was probably mis-specified. Participation was a proxy for avoiding a
degenerate "never trade" rule, while the outcomes actually cared about were
mean and tail risk; the veto improved every registered outcome axis while
rejecting 36.7% of trades. The frozen verdict still stands, but contamination—not
the 70% proxy—is the decisive limitation: February 2020 motivated the rule and
is included in the diagnostic. A future overlay should gate directly on
pre-specified risk-adjusted outcomes, plus a minimal non-degeneracy condition.

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

## A different target and a regime frame both produce informative negatives

A new, commit-pinned Oxford-Man SPX dataset allowed one frozen jump-target test
without buying intraday bars. The target was a material `max(RV5-BV,0)/RV5`
event within five sessions. ATM VIX barely improved the history-only Brier
(0.21421 to 0.21334); adding lagged SKEW worsened Brier to 0.21765 and log loss
to 0.62576. The strict surface-versus-ATM verdict is **FAIL**. BPV exceeded RV
on 16.9% of days and the 2017 event rate collapsed to 2.0%, so the next version
needs realized quarticity and a formal BNS jump target rather than tuning this
proxy.

A separate two-state Gaussian HMM targeted five-session entry into a high-RV
state. Its direct loss to a supervised classifier was calibration-confounded.
A pre-specified Platt repair on a target-specific 2025-2026 holdout improved the
HMM's Brier from 0.21819 to 0.20531 and log loss from 0.67383 to 0.60468 while
retaining a 54.5%-versus-18.2% top/bottom quintile spread. Calibration is real.
The fair incremental test still failed: adding the calibrated HMM probability
to the same-row supervised model worsened Brier from 0.19572 to 0.19662 and log
loss from 0.57822 to 0.57915. The state is descriptive and well ranked, but its
useful information was already absorbed by direct RV-history features.

The correlation-premium target is now explicitly an SPX study, matching COR1M,
DSPX, and VIXEQ by construction. It remains unscored only because the repository
lacks the exact point-in-time top-50 SPX tracking basket and constituent return
panel. Full results and the acquisition gate are in
`TARGET_REGIME_FINDINGS.md` and `SPX_DISPERSION_DATA_CONTRACT.md`.

---

# Standing summary — what is true at the end

Written 2026-08-12 at the close of the pre-gate work. This is the section to
read first.

**The central result is roughly ten designs across two assets and three
instrument families—not any individual null.** Daily variance models, option-
level controls, calendar/earnings features, cross-asset composites, term shape,
carry, jump decomposition, and latent-state transition work repeatedly find
that forecastable variance is already priced or that apparent additions are
regime-bound, contaminated, or redundant. These are not statistically
independent trials, so do not turn the count into a meta-p-value; breadth is the
robustness claim.

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
p = 0.975, win rate 45% at n=157, declining concentration); information beyond
the surface at 30 days (H3, withdrawn after non-overlapping-block testing);
cross-asset and market-state composites. Lagged VIX9D/VIX was directionally
positive but decayed from 5.3%/7.0% in 2022-2023 to 0.2%/0.9% in 2024-2025;
front-end dislocation is now a prospective-only interaction hypothesis.

**The variance risk premium is real and structurally hard to harvest.** ~3.9–4.2
vol points, visible to anyone charting VXN minus trailing RV. Conditional
selection does not pass against unconditional carry. A post-hoc SKEW veto
supports the adverse-selection mechanism and dramatically improves historical
tail aggregates, but fails its frozen 70%-participation floor (63.3%) and is not
out-of-sample evidence. The gate was misaligned with the risk objective, but
February 2020 contamination is the binding problem.

**Four clean negatives and three self-withdrawn positives.** The version of this
project that found significance would very likely have been wrong. What is worth
keeping is not a model — it is the harness: the leakage boundary, the frozen
pre-registration, the mechanically enforced diagnostic-only fence, the paired
per-origin reporting that never lets a mean travel without its sign count, and
the block testing that took back two of my own results.

**The repeated transition result is now better specified.** On the extended
history, direct RV features rank recurrent five-session threshold crossings
well (AUC 0.870), but this 13.2%-base-rate target mostly measures proximity to a
state boundary rather than rare discontinuous breaks. Calibration repairs the
HMM's probability scale, yet its state adds no usable top-decile ordering.
TiRex's continuous latent also encodes the threshold-proximity state but adds
nothing beyond RV history. The information boundary is therefore
architecture-invariant on this target: price history carries the state, while
the HMM and latent representation are redundant re-encodings rather than new
transition information.

## Standing rule

Let the clean window accrue. Do not re-run hunting for significance; every
additional model tested on these same 192 days is another comparison against the
same noise. Any model handed VXN must be reported against HAR-IV, not HAR.

## Frozen extension: five research paths (2026-08-12)

The follow-up changed the question from “can another feature beat the surface?”
to “what does the surface absorb, where does its value live, and where should the
mechanism be tested?” All five registered paths were run; none touched the sealed
NDX clean window.

- VXN absorbs 39.3% of the leverage joint Wald statistic and 33.4% of the
  point-in-time top-25 earnings statistic, but little calendar structure.
- Its diagnostic QLIKE contribution peaks at five sessions and decays beyond
  ten; the curve does not peak at its nominal 30-calendar-day horizon.
- The SPX implied-minus-realized premium is positive at 9, 30, and 93 calendar
  days: 3.61, 3.47, and 4.72 annualized volatility points.
- AAPL, AMZN, and GOOG earnings each leave large positive realized-variance
  residuals after conditioning on their own 30-day implied indices. This is
  mechanism evidence, not a tradable calendar forecast.
- The clean 2014–2015 SPX replication rejects the prospective term-slope idea:
  both the unconditional slope and the pre-specified dislocation interaction
  worsen QLIKE.

The constituent universe is now reconstructed independently each quarter from
accepted SEC N-PORT filings: share classes are combined by issuer, the top 25 are
ranked by filed portfolio weight, and a snapshot becomes usable only after its
acceptance timestamp. See [the frozen extension findings](RESEARCH_PATHS_FINDINGS.md)
and [the independent verification report](research_paths/verification.md).

## Extended-history functional-form and representation program (2026-08-12)

Four additional paths were run under isolated protocols without reopening the
sealed NDX clean window.

**The nonlinear escape hatch closed on point estimates, while its frozen formal
verdict remains INCONCLUSIVE.** A fixed histogram GBM using exactly the HAR-IV
information set was worse than HAR-IV by 3.78% in discovery, 4.67% in
confirmation, and 4.31% over the full diagnostic sample. The confirmation
21-session block interval spans zero, so equality cannot be ruled out and the
registered label is not rewritten. A timing-safe sensitivity that lags VXN one
complete session is also worse on all three splits (5.82% in confirmation).

**QLIKE explains the win-rate/mean-loss divergence.** GBM wins 54.7% of
confirmation origins while losing on mean. Realized-variance deciles 2 through
8 improve, but decile 10 contributes 174.6% of the net loss gap. The frozen
locked term has the same shape: 57.4% wins, a slightly worse mean, and 137.9% of
its gap in decile 10. The timing-safe GBM repeats it at 149.8%. This is a
post-result, outcome-conditioned mechanism diagnostic—not an ex-ante rule—and
the earnings-with-IV pair does not cleanly replicate it. Partial dependence had
selected the highly correlated weekly/monthly RV pair (Pearson 0.816); a fixed
post-result SHAP audit instead selected weekly RV × implied volatility,
supporting the correlation-artifact caveat.

**History can rank recurrent threshold crossings.** The price-only extension
contains 6,694 QQQ sessions from 1999-03-11 through 2025-10-17 and is locked by
raw-source, transform-code, protocol, and derived-output hashes. The first
scored annual fold is 2002 because 1999-2001 supplies the required completed
training labels. Across 5,592 calm origins, the five-session event rate is
13.16%: 736 positive origins collapse to 222 trigger sessions and 118 positive
episodes. The RV-history classifier reaches 0.8704 phase-mean AUC, 4.804x
top-decile lift, and a 63.2% top-decile event rate. The reviewer-requested
single RV-percentile control reaches 0.8111 AUC and 3.501x lift. This is a strong
ranking result, but the target is proximity to a recurrent 80th-percentile
boundary, not rare-crisis prediction.

**Ranking metrics confirm the HMM null.** The calibrated HMM alone reaches
0.8294 AUC. Adding it to RV history moves AUC only +0.0010 while reducing lift
by 0.136x and the top-decile event rate by 1.79 percentage points. Thus the
proper-score objection to the original HMM comparison has been removed: the
state probability does not add usable ordering where the risk overlay would
act.

**A continuous TiRex latent reaches the same information boundary.** With no
PCA, full ridge, sparse k={1,5,10}, and a fixed eight-unit MLP were trained in
annual forward folds. Sparse k=1 and k=5 reach 0.8153 and 0.8270 latent-only
AUC, showing descriptive decodability, but their augmented AUCs are 0.8694 and
0.8657 versus 0.8704 for direct RV history. Every sparse coordinate is selected
from completed fold-training labels and scored on disjoint forward rows; nine
different coordinates appear in the k=1 path, so it is not one universal
neuron. The original ten-control ladder cannot provide formal 5% evidence: its
minimum exact corrected p-value is 1/11 = 0.0909. The old empirical-percentile
flag is retained only as a descriptive heuristic. The fixed MLP also hit its
500-iteration cap in 273 of 288 fits, so its result is a frozen optimizer
endpoint, not proof that nonlinear probe capacity was exhausted.

**A separate 99-control k=1 run confirms selectivity, not incremental value.**
After the ten-control ceiling was identified, one rung—not the whole ladder—was
registered under 99 fixed Markov controls. The exact same annual forward
procedure selects from all 512 dimensions on completed training labels and
scores only the held-out year; every synthetic control reselects from its own
training labels. Actual phase-mean AUC is 0.8153 versus control median 0.5202,
95th percentile 0.5640, and maximum 0.6080. With zero exceedances, the corrected
exact randomization p-value is 1/100 = 0.01. This is formal selectivity evidence
within an explicitly post-result, reused-history diagnostic—not a pristine
holdout and not evidence of value beyond the 0.8704 RV-history benchmark.

The coordinate characterization makes the mechanism concrete. Nine different
coordinates are selected across 24 folds; z499 appears six times and z386 five,
so there is no universal neuron. On held-out rows, the event-oriented median
Pearson correlation is 0.856 with trailing five-session mean log RV, 0.804 with
the 22-session mean, 0.755 with prior-session VXN (17 eligible folds), 0.573
with current log RV, and only 0.039/-0.056 with one-/five-session return. TiRex
really encodes the state, but principally as smoothed volatility level and
lagged implied volatility—the same information the direct benchmark extracts
more efficiently.

**The Eidos-derived corruption result is narrower than first reported.** On the
registered context-noise grid, Chronos-2 and TiRex-2 degrade modestly, making
surface-noise fragility an unlikely explanation for their earlier forecasting
null. But this adaptation supplies raw corrupted log-RV to each native adapter;
it does not impose Eidos Appendix A.1.2's common noisy-statistics
renormalization. HAR therefore is not an apples-to-apples architectural
contrast, and the stored score is a 0.1-0.9 decile-grid CRPS approximation rather
than full-tail CRPS.

Taken together, the three obvious model-side explanations are closed on the
available diagnostic evidence: a compact implied-aware model already has the
capacity, a flexible GBM does not reveal useful functional form, and neither a
two-state filter nor a rich continuous foundation representation contributes
transition information beyond direct RV features. What remains binding is the
information set and the definition of the economic risk, not another model
class.

## Post-program free-source diagnostics (2026-08-12)

This extension is deliberately outside the evidence hierarchy of the sealed NDX
clean window. Its protocols were frozen before their respective source
transforms and scores, but the assets, targets, and historical windows had
already been inspected elsewhere. Results are therefore **post-program
diagnostics**, not new clean confirmations.

**Weekly CFTC positioning does not clear the incremental gate.** One
conservatively delayed Nasdaq TFF release was mapped to each eligible calm QQQ
origin. Across 591 origins, the RV-history benchmark scored 0.8308 AUC and
3.482x top-decile lift. Adding leveraged-money net/open-interest share scored
0.8304 AUC and 3.681x lift. The lift gain is real in the saved ranking, but the
registered rule required both metrics to improve, so the verdict is **FAIL**.
An independent implementation passed 15 checks and reproduced all rows,
labels, structural-break sensitivities, and metrics.

**The free NQ minute file produced no predictive verdict.** The frozen pipeline
retained 678 quality-eligible RTH sessions, but only 63 eligible training rows
were available for the 2024 fold and 135 for 2025, below the pre-specified 180
in both cases. The independently verified status is
`VERIFIED_NO_EVALUABLE_FOLDS`, with zero forecast origins. This is a data-
adequacy result—not evidence for or against the BNS jump/shape features—and the
undocumented continuous-contract stitch remains a second binding limitation.

**Free option-surface mirrors did not improve mean QLIKE.** The private Kaggle
diagnostic delayed each close snapshot one full exchange session and compared
ATM-only HAR information with fixed skew, term-slope, and gamma-weighted-volume
features. On the governing SPY late-confirmation split, AUC moved 0.8298 to
0.8349, but mean QLIKE worsened 1.5359 to 1.5492 and top-decile lift fell 3.787x
to 3.683x; the block interval spans zero, so the verdict is `INCONCLUSIVE`.
QQQ and AAPL likewise improved AUC while worsening mean QLIKE. Those two are
mechanism diagnostics only, and gamma-weighted volume is not dealer GEX because
the files contain no open interest. A second implementation passed 20 checks
over 3,850 forecasts and verified timing and source hashes.

**The frozen HF option-flow feature is not measurable on the advertised
archive.** All 22 pinned shards through 2025-10 matched their immutable hashes
and yielded 406 QQQ and 439 SPY activity days. But the registered
`near_expiry_volume_share_7d` component is exactly zero on every day for both
symbols, so its strictly-prior scale is zero and the fixed four-component
composite is undefined. The protocol was not rewritten to drop or reweight the
component; zero forecasts were scored. The independently reconstructed status
is `INSUFFICIENT_DATA`, not a model null.

The acquisition result matters independently of those scores. Official CFTC
and Cboe files are distinguished from uploader mirrors; all ambiguous
exchange-derived archives remain raw-local/quarantined; uploader license labels
are not treated as proof of upstream redistribution rights. The auxiliary audit
validated a 26,874-row, one-session-lag local Cboe close panel, quarantined the
HF SPX mirror because 5,075 rows have zero opens, and verified the Zenodo TSLA
file's pinned MD5/schema and 4,584,740 data rows. The last source really does
contain bid/offer and open interest, contrary to the original listing, but its
upstream provenance is undocumented, so it remains private-research-only and
supports no new prediction claim. The complete state and source-by-source
caveats are in [`FREE_DATA_SOURCES.md`](FREE_DATA_SOURCES.md).
