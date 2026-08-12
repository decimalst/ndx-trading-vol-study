# Amendments log

`config.yaml` is the pre-registration artifact. Any change to it *after*
clean-phase results have been observed must be logged here and restarts the
accrual clock.

---

## 2026-08-11 — Setup entry (first clean-phase run). Clock starts.

**`config.yaml` was not modified.** Everything below is environment, code-defect,
and data-input work done to make the pre-registered design executable for the
first time. No clean-phase result had been observed before these changes, so
none of it is an amendment in the sense above; it is recorded here because it
changed what the covariates mean.

### Environment

- Built `.venv` on Python 3.11 (`python3` on this box was Anaconda 3.8.8, below
  the pandas ≥2.1 / torch floor). `Makefile` now resolves `PY` to `.venv` when present.
- Added `lxml` to `requirements.txt` (`yfinance.get_earnings_dates()` needs it).
- Verified stack: pandas 3.0.5, numpy 2.4.6, torch 2.13.0, chronos-forecasting 2.3.1.

### Code defects fixed (all pre-existing, all affecting reported numbers)

1. **`chronos_runner.run_chronos` dropped whole batches at the data frontier.**
   `_frames_for_batch` returned `(None, None)` when *any* origin in a batch had
   fewer than 21 future covariate rows, and the caller `continue`d — discarding
   up to `batch_origins` (32) origins, always the most recent ones. Since the
   experiment's entire value is forward accrual, this silently deleted the
   newest clean data on every run. Origins are now grouped by supportable
   horizon: every origin keeps its h=1 forecast, and only the 30-day cumulative
   is `NaN` where the horizon is not covariate-supported (20 of 192 origins).
2. **`evaluate` could not read the forecasts `baselines` wrote.** The documented
   order (`baselines PHASE=all` then `evaluate PHASE=clean`) looked for
   `*_clean.parquet` while `baselines` had written `*_all.parquet`, producing a
   report with an empty results table and no error. `evaluate` now falls back to
   the `_all` run and slices it to the phase window.
3. **Diagnostic-phase VXN benchmark scored the wrong window.** The Mincer–Zarnowitz
   row used a lower bound only, so the diagnostic report's VXN line was computed
   over 2016→2026-08-11 (n=2644) including the clean window, while every other
   row in the same table covered 2016→2025-10-17 (n=2463). Now bounded on both sides.
4. **Quantile columns matched by string formatting.** `str(tau)` happened to match
   `predict_df`'s column names for this quantile grid; it silently would not for
   e.g. `0.025`. Now matched on float value, raising if a level is absent.
5. **`encompassing` output was uninterpretable.** The report printed `c_model`
   without `b_implied`, so a negative coefficient could not be diagnosed as
   collinearity rather than evidence against the model. Both are printed now.

### Data inputs — this is the part that changes what the covariates mean

`calendars/*.csv` were seed files with errors. Verified 2026-08-11 against
bls.gov and federalreserve.gov (both 403/406 scripted fetches; read from the
rendered schedule pages). Corrections, all inside the clean window:

| File | Was | Now |
|---|---|---|
| `cpi.csv` | 2026-02-11, 2026-04-14 | **2026-02-13, 2026-04-10** (wrong dates) |
| `cpi.csv` | no 2025 dates | added 2025-12-18; **no Oct-2025 CPI exists** (shutdown) |
| `nfp.csv` | 2026-02-06 | **2026-02-11** (wrong date) |
| `nfp.csv` | 5 dates | 131 dates; added 2025-11-20, 2025-12-16, 2026-01-09, 2026-07-02, 2026-08-07 |
| `fomc.csv` | 2022–2026 | verified correct; extended to 2016–2027 |
| `earnings_top.csv` | empty | 1171 rows, 13 names ≥2% weight |

Clean-window event counts moved from (6 FOMC, 7 CPI, 4 NFP, 0 earnings) to
**(6 FOMC, 8 CPI, 9 NFP, 30 earnings days)**. Any event-sliced number produced
before this date is void.

The 2025 shutdown gaps are real: September 2025 CPI slipped to 2025-10-24 and
October 2025 CPI was never published; September 2025 payrolls slipped to
2025-11-20 and October payrolls were folded into 2025-12-16. Do not fill them.

**Earnings sessions are inferred**, not verified: yfinance reports announcement
timestamps in ET and the session is derived from the hour (<09:30 → BMO,
≥16:00 → AMC). The distribution is consistent with reality (WMT 100% BMO, the
mega-caps ~100% AMC) and no clean-window row is ambiguous, but this has **not**
been checked against company IR pages. A wrong session shifts the vol day by
one, so treat the heavy-earnings slice as provisional until it is.

**Index weights are not point-in-time.** Weights come from the Invesco QQQ
holdings file dated 2026-08-10 and are applied to the whole history. Known and
accepted per the README; it matters most in the diagnostic window.

---

## 2026-08-11 — Post-result entry: HAR-IV control, TiRex-2, decile grid

**`config.yaml` still unmodified.** Clean-phase results have now been observed
and are recorded in `reports/FINDINGS.md`. Everything below is additive — new
models and a parallel report — not a change to a pre-registered quantity.

### HAR-IV added as the control for VXN-fed models

`chronos_cov_iv` beat plain HAR at p=0.0026, but plain HAR is the wrong control:
that comparison only shows implied vol predicts realized vol. Added `har_iv`
(log-HAR + log(VXN), a five-term OLS) and a dedicated DM table against it.

Result: `chronos_cov_iv` vs `har_iv` gives DM = +0.218, p = 0.828 — the
foundation model does not beat the linear model on the same information set, and
the point estimate favours the regression. **This closes the Chronos thread on
this window.** Any future model handed VXN is reported against HAR-IV.

No `har_iv_cum` counterpart exists on purpose: the encompassing regression
already conditions on VXN, so an IV-fed cumulative forecast would be collinear
with the regressor by construction.

### Encompassing table: VXN-consuming models removed

`chronos_cov_iv` produced `b_implied` = 2.478 with `c_model` = −1.230 — a
collinearity blow-up, not a result, since the model eats VXN as an input.
Regressing realized variance on VXN *and* on a function of VXN does not admit an
interpretable coefficient split. Models in `IV_FED` are now excluded from that
table with the reason printed in the report.

### Variance-risk-premium reporting corrected

The old report labelled α < 0 as the premium. With β = 0.829 ≠ 1 the intercept
is not interpretable alone. The report now evaluates the fitted line at the
window's median VXN and states the premium in both variance and vol terms
(68% of implied / 82% in vol / ~4.2 vol points at VXN 24.1).

### TiRex-2 (arXiv 2607.01204v1) added — parallel decile-grid report

Leakage assessed by corpus enumeration rather than publication date; see
`reports/LEAKAGE_TIREX2.md`. The corpus contains no equity index, realized-
variance, or implied-vol series, and its real components end around 2023.

TiRex-2 emits only deciles (0.1…0.9) and accepts no quantile-level argument.
Since `mean_var` is a truncated mean over the quantile grid, it is not
comparable across grids, so `--quantile-grid deciles` recomputes **every** model
on TiRex's grid and writes a parallel `*_dec` report. `config.yaml`'s `quantiles`
is untouched and the pre-registered results stand as-is. On the decile grid the
widest interval is 80%, so the interval gate there is an analogue of the
registered 90% gate, not the registered gate.

### Also fixed

`_h1_scores` hardcoded the 0.05/0.95 pinball levels and `cmd_evaluate` hardcoded
`nominal=0.90` for the coverage tests. Both now derive from the active grid,
which is what makes the decile report correct rather than silently mislabelled.

---

## 2026-08-12 (second) — HAC lags, frozen earnings cutoff, concentration test

### HAC maxlags for the 30-day horizon: 21 → 32

`mz_regression` and `encompassing` used `maxlags=21`, exactly at the floor for
forecasts with ~21 trading days of target overlap; standard practice is ≥1.5h.
Default is now 32 (`metrics.HAC_LAGS_30D`), 40 run as a robustness check.
Effect on previously reported numbers: diagnostic `har_cum` c_model p
0.035→0.047 (0.051 at 40); `persistence_cum` 0.006→0.011 (0.014 at 40); no
clean-window conclusion changes (those were already null). This is a
methodology tightening that makes a marginal result more marginal — the
direction a correction should cut.

### Heavy-earnings threshold frozen at an absolute cutoff

`heavy_earnings_quantile: 0.8` (in-sample, recomputed every run; slice moved
n=8→n=7 on a data correction alone) superseded by
`heavy_earnings_min_wt: 5.0` — days on which ≥5% of index weight reports.
Frozen before any result under the new definition was observed. Slice is n=12
in the clean window, n=157 in the diagnostic window.

### Concentration test: pre-specified, regressor found degenerate, decision recorded

Pre-specified in `config.yaml` before running (single test, stated criterion).
The regressor — tracked-basket total weight from `pit_weights.parquet` — is
**constant by construction** (the build renormalizes the basket to its snapshot
total daily), so the test as specified is unestimable. Specification error,
discoverable ex ante, disclosed in FINDINGS; post-hoc substitutes (yearly
win-rate table, day-level win~weight logit) are labeled post-hoc and do not
support the concentration defence. Decision recorded under the pre-committed
criterion: defence not supported; earnings story rests on the n=500 gate.
No further re-specification of this test.

### New specifications this round (for the multiplicity ledger)

`har_sv`, `har_ic`, `har_iv_x` (pre-specified as the single new model before
running), `tirex_uni/cov/cov_iv/cov_ivf`, plus the window-matched `har`
control. All nulls reported. Specification freeze until the pre-committed gate.

---

## 2026-08-12 (third) — H3 withdrawn; LHAR; carry study

**`config.yaml` gained two pre-registration blocks (`carry_study`, and earlier
`concentration_test`); no pre-registered forecasting quantity was altered.**

### H3 positive withdrawn

Non-overlapping-block refit (every 21st origin, ~117 obs, ordinary errors) run
at **all 21 phase offsets**. `har_cum` reaches p<0.05 in 5% of phases,
`persistence_cum` in 24%, median p 0.21 and 0.19, coefficient sign unstable for
persistence. The HAC result was borrowing power from overlap. The "genuine H3
pass" recorded in the second 2026-08-12 entry is **withdrawn**; H3 is null on
both windows.

### Orthogonalised encompassing added to the diagnostic

Frisch-Waugh-Lovell makes the orthogonalised coefficient and p-value identical
to the original (verified). The informative outputs are R²(m~iv) and sd(e), plus
the standardised effect c·sd(e), which is what makes models comparable:
har_cum +0.0660 vs persistence_cum +0.0616 — near-identical incremental content,
so the apparent ordering between them was a collinearity artefact.

### har_lev (LHAR / Corsi-Renò) — diagnostic only, mechanically enforced

Daily-data return asymmetry, testable from 1999 and therefore not capped by
yfinance's 730-day intraday limit. Diagnostic n=2463: QLIKE 0.3618 vs HAR
0.3815, DM −5.367, p<0.0001. Added to `DIAGNOSTIC_ONLY` in experiment.py, which
suppresses it from clean-phase reports so that testing during the freeze cannot
become a peek.

### Carry study — new module `src/carry.py`, `make carry`

Pre-registered before implementation. Diagnostic window only; clean window
untouched. Result: **FAIL** on the pre-committed criterion (mean improves
+1.714→+2.381 vol points and CVaR improves −33.3→−22.7, but bootstrap median
p=0.50 with 0/21 phases significant). Three of the five worst trades in the
decade — all in the week before the COVID crash — were taken by the conditional
rule. Worst trade −196.7 vol points against a mean of +1.71.

---

## 2026-08-12 (fourth) — ledger-closing test: har_iv_lev

`config.yaml` unchanged. One diagnostic-only specification added to answer the
last open question: is the leverage effect just VXN by another route, given that
VXN embeds skew?

`har_iv_lev` = HAR + log(VXN) + Corsi-Renò leverage terms. Registered in
`DIAGNOSTIC_ONLY`, verified suppressed from clean-phase reports.

Answer: **not subsumed.** Joint Wald on the three leverage terms alongside
log(VXN), diagnostic n=2463: χ² = 62.63, p < 1e-6; all three individually
significant with the correct sign; R² 0.5347 → 0.5644. VXN absorbs ~40% of the
leverage information (joint χ² falls 103.19 → 62.63) and a decisive remainder
survives. Out-of-sample forecast gain over HAR-IV is directional but not
significant (QLIKE 0.3376 vs 0.3438, DM −1.410, p = 0.159).

Return asymmetry is the single non-market signal that survived the project.
Documented, not acted on. **No open items remain before the pre-committed gate
(500 clean origins or 2027-06-30).**

---

## 2026-08-12 (fifth) — symmetric test applied to har_iv_lev; two corrections

`config.yaml` unchanged.

### The check that decided the earnings case, applied to leverage

An in-sample Wald beside an insignificant OOS DM is the pattern that rejected
HAR-IV-X; the paired per-origin count is what decided it, and it had not been
run for har_iv_lev. Diagnostic n=2463: **1366 wins (55.5%), sign p = 6.5e-08**,
median improves, win rate 55.3% after dropping the ten largest days. Leverage
passes the same standard the earnings term failed.

### Two claims tightened

1. "Not subsumed" now reads "not subsumed, and worth ~1.8% of QLIKE that does
   not survive a significance test." At n=2463 the DM has real power, so
   p = 0.159 is mild evidence the gain is small, not merely underpowered.
2. The novelty claim is narrowed from "the one signal that survived" (a
   fifty-year-old effect — Black 1976, EGARCH, Corsi-Renò) to the partial
   absorption: joint Wald 103.19 → 62.63 when VXN enters, i.e. the surface
   prices ~40% of the leverage information and no more.

### Correction to the earnings characterisation

The full-sample paired count reverses the shape previously reported from the
n=157 heavy-earnings slice. On all 2463 origins, har_iv_x wins 61.3% of days
(p=1.4e-29) with **+0.0% mean improvement**, top-10 days carrying 5300% of the
(near-zero) gap; har_x wins 60.7% with mean **−0.6%**. Many small wins funded by
rare large losses — the opposite of "pays off on a few large days," and a less
attractive profile. Verdict unchanged (null); characterisation corrected.

### Permanent reporting change

`cmd_evaluate` now prints a **full-sample paired per-origin table** (win rate,
sign p, mean, median, top-10 concentration) for the key pairs in every phase,
alongside the existing heavy-earnings slice table. A mean difference and a win
rate answer different questions and can point opposite ways; neither now travels
alone.

---

## 2026-08-12 (sixth) — separate orthogonal-signal protocol and holdout

**`config.yaml` unchanged.** This is a separate diagnostic-only study governed
by `signal_study.yaml`. The original clean window is fenced out in code and by
pre-written tests.

Before fetching signal inputs or producing a score, commits `ed313fb` and
`340e8dd` registered the safety contract. Before any result, commit `352b59f`
expanded the user-authorized combination search to exactly three families and
their seven non-empty combinations. Both discovery and confirmation Make
targets run the 14-test safety suite first.

Discovery (2016–2021, n=1511) locked one winner: one-session-lagged
`log(VIX9D/VIX)`, +3.39% QLIKE versus timing-safe HAR-IV-LEV. Cross-asset stress
(HYG/TLT/GLD/USO/UUP), QQQ market state (abnormal volume/overnight share), and
their combinations did not win.

Confirmation was spent once (2022–2025-10-17, n=952). Term slope improved mean
QLIKE by 2.92% and won 54.5% of origins, but failed the registered two-sided DM
threshold (p=0.1016). Verdict: **FAIL**. An independent verifier that imports
neither signal implementation nor shared metrics reproduced all confirmation
numbers to 1e-12 and confirmed the last scored target precedes the clean fence.

Historical-weight audit: `pit_weights.parquet` is a survivorship-biased
13-current-name reconstruction with a mechanically fixed 55.43% aggregate, not
exact historical NDX weights. After the user supplied the working EDGAR page,
pre-run commits `30f49a2`, `844d38e`, and `5eefd8f` registered the N-PORT parser,
liability, as-of timing, and concentration tests before live processing. Nine
tests passed, then all 27 public QQQ N-PORT snapshots were downloaded (2,746
holdings, report dates 2019-09-30 through 2026-03-31, disclosure lag 50–62
days). They are aligned by SEC acceptance timestamp and stored for future use;
no weight signal was scored on the spent holdout.

---

## 2026-08-12 (seventh) — paused annual backfill; one SKEW carry diagnostic

### Historical holdings backfill paused

The proposed 2004-2018 QQQ annual-report parser and Nasdaq public-membership
XLSX parser were written behind ten pre-run contracts. Real 2004 and 2005
fixed-width variants exposed parser gaps; each stopped the run, received a
regression test, and then reconciled exactly to disclosed total investments.
The subsequent SEC archive request repeatedly returned HTTP 403 despite bounded
backoff. The user paused collection. No combined output was promoted and the
code remains disconnected from every model. Full state and restart rules are in
`reports/PAUSED_HISTORICAL_WEIGHTS.md`.

### SKEW-conditioned carry

A single post-hoc mechanism diagnostic was frozen before SKEW acquisition. It
preserved the failed richness rule and vetoed trades when one-session-lagged
Cboe SKEW exceeded a trailing 252-session 80th percentile estimated through the
prior observation. No VVIX fallback or threshold search was permitted. Twelve
pre-run tests covered the clean fence, Cboe schema and historical anchor,
lagging, rolling-threshold timing, missing values, rule application, and phase
sampling.

The official file supplied 9,203 rows from 1990-01-02 through 2026-08-11 and
retained Cboe's independently published 2018-08-13 close of 159.03. Source hash:
`becbf3f7510de66a736495df29a84ba1911d362402f05f6c46dedd5e9b971492`.

Strict frozen verdict: **FAIL**. The veto rejected all three known adverse
pre-COVID entries and improved average phase mean (+2.381 to +2.800), CVaR
(-22.711 to -10.897), worst trade (-49.629 to -14.317), and drawdown (55.127 to
19.179), but retained only 63.3% of richness trades versus the registered 70%
floor. An independent implementation reproduced timing, masks, 21 phase
metrics, tail statistics, and verdict inputs.

This is not a strategy backtest. It is motivated by the already-observed 2020
failure and inherits the old carry study's full-window richness median and
same-date VXN daily close. A future version must freeze +0.386812814 and use
lagged Cboe data or timestamped pre-close quotes.
