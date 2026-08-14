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

---

## 2026-08-12 (eighth) — changed target and transition frame

Before downloading the new target source or scoring either model, eleven tests
and `target_regime.yaml` froze two diagnostics: an SPX five-session jump event
from Oxford-Man RV/BPV, and a two-state QQQ stress-transition model using only
forward-filtered probabilities.

The initial Oxford-Man branch URL returned HTTP 404 before any file was
accepted. Public repository history identified commit
`308b795fa220a58dea6784fe8e2566bcf8dea334` as the last containing revision.
The source was pinned to that immutable commit and confirmation conservatively
ended 2017-12-29, before the archive snapshot. No target, model, or decision
criterion changed. A real mixed-timezone date layout then stopped parsing; a
regression was added to preserve the stated trading date without UTC date
roll-back.
The repaired pre-run gate therefore contains twelve contracts.

Both frozen verdicts were **FAIL**. On 1,001 SPX origins, the surface model lost
to ATM on Brier (0.217646 vs 0.213344) and log loss (0.625756 vs 0.614335). On
2,205 calm QQQ origins, the HMM lost to the supervised transition benchmark on
Brier (0.137616 vs 0.126650) and log loss (0.457171 vs 0.420496), despite strong
top-versus-bottom quintile ranking. An independent verifier reproduced source
hash, component reconciliation, source lags, annual thresholds, completed
targets, phase losses, fences, and both verdicts.

---

## 2026-08-12 (ninth) — calibrated and incremental HMM repair

User review identified that the direct supervised-versus-unsupervised HMM
comparison confounded discrimination with calibration. Before inspecting this
transition target after 2025-11-03, six tests and `regime_repair.yaml` froze a
Platt calibration and a same-row incremental-state comparison. These dates had
already been used elsewhere in the project, so the result is explicitly a
target-specific holdout rather than a pristine global holdout. All fitted
objects use annual out-of-fold rows through 2024; the HMM, threshold, Platt
mapping, and supervised coefficients are not refit on holdout labels.

On 166 calm origins, Platt calibration passed: HMM Brier improved 0.218189 to
0.205309 and log loss improved 0.673834 to 0.604679. The fair incremental test
failed narrowly: adding calibrated HMM probability to the supervised benchmark
worsened Brier 0.195725 to 0.196617 and log loss 0.578219 to 0.579151. This
replaces the interpretation of the original comparison: calibration was a real
defect, but the state supplied no detectable information beyond direct RV
history in the registered holdout.

The jump-data objection was also checked against implementation lineage. The
jump module reads Oxford-Man five-minute `rv5` and `bv`; it never reads the
seven-bars-per-session yfinance file. The jump negative remains limited by the
absence of a formal BNS quarticity statistic, but it is not an hourly-sampling
result.

### Interpretive corrections from final review

- The term-slope result is now led by non-stationarity, not its aggregate
  p-value: confirmation improvement decayed 5.3%/7.0% in 2022-2023 to
  0.2%/0.9% in 2024-2025. A dislocated-front-end interaction is prospective
  only because the yearly split has been inspected.
- The SKEW overlay passed every registered outcome check and failed only the
  70% participation proxy. The verdict is unchanged, but February 2020
  contamination—not gate design—is the binding reason it is not evidence. A
  future overlay should gate direct risk-adjusted outcomes plus minimal
  non-degeneracy.
- N-PORT top-10 weight fell 55.13% to 46.85% and HHI fell 0.0459 to 0.0312,
  with the large step following Nasdaq's July 2023 anti-concentration special
  rebalance. Together with flat yearly earnings wins and the 2025 low, that is
  a third line against the earnings-concentration defence. This sample is now
  spent for the question; existing-data refits cannot restore falsifiability.

---

## 2026-08-12 (tenth) — history, GBM, ranking, and representation extensions

These are isolated diagnostic protocols. They do not modify `config.yaml`,
reopen the sealed NDX clean window, or rewrite any prior frozen verdict.

### Price-only history extension

`history_extension.yaml` froze QQQ/VXN source hashes, the 2025-10-17 endpoint,
the Garman-Klass-plus-overnight transform, the derived parquet hash, and the
exact transform implementation hash before downstream use. The resulting panel
has 6,694 sessions from 1999-03-11. Cboe's free VXN file begins 2009-09-14, so
no proxy or splice extends HAR-IV backward. A downstream verifier now requires
the panel to match both the frozen protocol and build manifest.

### GBM functional-form study

`gbm_study.yaml` froze one histogram gradient-boosted tree, one HAR-IV
information set, fixed discovery/confirmation dates, and 21-session block
inference. GBM is point-estimate worse than HAR-IV on all three splits, but the
confirmation block interval spans zero; the registered verdict remains
**INCONCLUSIVE**. The verifier now recomputes every metric directly from the
saved forecasts.

After that headline was observed, `gbm_post_result.yaml` froze three reviewer
diagnostics: realized-RV-decile loss attribution, one-session-lagged VXN timing
sensitivity, and a fixed TreeExplainer interaction audit. They are explicitly
outcome-conditioned/post-result and cannot revise the parent verdict. GBM's
highest RV decile contributes 174.6% of its net confirmation deficit; the
locked term contributes 137.9%. Lagged-VXN GBM is 5.82% worse in confirmation
and decile 10 contributes 149.8%. Partial dependence selected correlated
weekly/monthly RV, whereas SHAP 0.51.0 selected weekly RV × implied vol. The
post-result verifier reconstructs timing-safe metrics and every decile table
from hash-locked inputs.

### Tail ranking and reviewer controls

`representation_study.yaml` froze the inherited five-session/80th-percentile
event and five-phase AUC/lift scoreboard before scoring the extended sample.
The 5,592 calm origins yield 0.8704 AUC and 4.804x lift for RV history. The HMM
augmentation adds +0.0010 AUC while reducing lift by 0.136x. The trailing-RV
percentile row, phase min/max/spread, and positive-episode jackknife were added
only after the headline in response to review and remain labeled as such. The
jackknife is a positive-episode influence diagnostic with negative origins held
fixed, not a full cluster-robust standard error. The benchmark intentionally
inherits the transition module's `mean(log RV)` convention rather than
standard HAR's `log(mean variance)`.

### Latent ladder corrections and formal follow-up

The original TiRex ladder uses full 512-dimensional ridge, sparse k={1,5,10},
and one fixed MLP with no PCA. Sparse dimensions are selected inside each
annual training fold and scored on disjoint forward rows. Review caught that
ten synthetic controls impose a minimum exact corrected p-value of 1/11. No
controls were silently added: the stored empirical-percentile flag is now
called a descriptive heuristic, `formal_evidence` is false, and exact p=0.0909
is shown for every rung. Chunk reuse now requires a current run signature and
per-chunk hashes. The MLP convergence audit records 273 of 288 fits at the
frozen iteration cap; the old scores were not refit or tuned.

A separate `latent_k1_confirmation.yaml` was then frozen for one sparse k=1
rung and 99 predetermined controls. It is a post-result formal-randomization
diagnostic, not a new historical holdout. None of 99 controls reached the
actual 0.8153 AUC, so corrected exact p=0.01. The verifier reconstructs all
2,376 fold/control selections and 553,608 control rows. Coordinate
characterization uses held-out rows and one-full-session-lagged VXN; it shows a
fold-varying encoding dominated by smoothed volatility rather than a new
transition variable.

### Eidos-derived noise report correction

Review of the source paper identified two overstatements. Eidos Appendix A.1.2
re-normalizes corrupted inputs using noisy statistics; this implementation
passes the common raw corruption to each model's native preprocessing, while
HAR does not renormalize the origin state. The HAR/foundation magnitude is
therefore not an apples-to-apples architecture comparison. The stored `crps`
is also a trapezoidal approximation over quantiles 0.1-0.9, not full-tail CRPS.
The report and protocol now say both explicitly. No forecasts, corruption
paths, or bootstrap draws changed, and the verifier recomputes curves,
intervals, hashes, and the rendered report.

---

## 2026-08-12 (eleventh) — wider free-source acquisition and diagnostics

This is another isolated post-program extension. It does not modify
`config.yaml`, reopen the sealed NDX clean window, or revise any prior verdict.
`free_data_sources.yaml` first froze source identities, revisions, storage
budgets, raw-local policies, availability rules, and provenance classifications.
Raw third-party archives are Git-ignored. Ambiguous exchange-derived mirrors
remain private-research-only even when their uploader labels say CC0, MIT, or CC
BY.

Three empirical protocols were frozen before their own transforms and scores:

- `free_signal_study.yaml` specified one conservatively delayed CFTC release per
  QQQ origin, known publication-backlog exclusions, annual forward fits, and a
  two-metric AUC/lift gate.
- `nq_intraday_study.yaml` specified strict ET localization, the RTH window,
  completeness and stitch guards, five-minute RV/BPV/tripower/BNS construction,
  a 180-row training minimum, and no relaxation after source inspection.
- `surface_data_study.yaml` specified exact exchange-session filtering, a full-
  session surface lag, fixed ATM/skew/term/gamma-volume construction, and a
  governing late SPY split. It also precluded calling gamma-weighted volume
  dealer GEX because the mirrors lack open interest.

The CFTC augmentation **failed** its frozen joint gate: AUC moved 0.8308 to
0.8304 while top-decile lift moved 3.482x to 3.681x across 591 origins. The NQ
study returned **no evaluable folds**, not a model null: its 2024/2025 training
counts were 63/135 against the frozen 180 minimum. The surface augmentations
were point-estimate worse on mean QLIKE for QQQ, AAPL, and both SPY splits; all
registered verdicts are `INCONCLUSIVE`. Independent verifiers reproduced the
CFTC score (15 checks), NQ no-fold outcome and data fences, and surface scores
(20 checks over 3,850 forecasts).

The remaining source gates were then completed without changing the frozen
features. All 22 pinned HF option shards matched their hashes, but the registered
near-expiry-volume-share component was exactly zero on every QQQ and SPY day;
the fixed composite was therefore undefined and the verified result is
`INSUFFICIENT_DATA` with zero forecasts, not a predictive null. The auxiliary
audit also (i) produced a local-only 26,874-row Cboe close panel with a full-
session availability lag, while quarantining 47 internally inconsistent
historical VIX OHLC rows from any OHLC use; (ii) quarantined the HF SPX mirror
because 5,075 opens are zero; and (iii) matched the Zenodo TSLA file's pinned
MD5 and schema across 4,584,740 data rows. Zenodo does contain bid/offer and
open interest, correcting the original source description, but undocumented
upstream provenance keeps it private-research-only. None of these audits creates
a new predictive result. See `reports/FREE_DATA_SOURCES.md` for the current
acquisition and rights ledger.

---

## 2026-08-13 — the methodology fork was documented but not implemented

**`config.yaml` unchanged. No frozen report moved: `results_clean.md`,
`results_diagnostic.md` and `results_clean_dec.md` are byte-for-byte identical
before and after this entry, re-verified by
`tests/test_methodology.py::TestFrozenReportsUnchanged`.**

### What was wrong

`reports/METHODOLOGY_FORK.md`, `src/methodology.py` and
`tests/test_methodology.py` all described the four conclusion-changing
corrections, and the `_est`/`_inf`/`_v2` reports sat in `reports/`. The code
that produced them did not exist. `src/models.py` had no smearing or
reconstruction estimator, `src/config.py` had no registry loader,
`src/experiment.py` had no `--estimator`/`--inference` flags, and the
`make scenarios` command the fork document tells the reader to run was not a
target. Of the 28 tests in the fork's own suite, 12 failed on a clean checkout:
11 errors and one failure, the latter being the CLI rejecting its own
documented arguments.

The corrected reports were therefore unreproducible artifacts. The guardrail
that catches this for the pre-registered reports — byte-for-byte reproduction
from the default command line — had no counterpart for the fork, which is
precisely why the fork is where it happened.

### What was restored

- `models.smearing_mean_var`, `models.mean_var_from_quantiles`
  (`trunc`/`lognormal`/`tail_ext`), `models.rescore_mean_var`, and
  `estimator=` on `run_har`/`run_persistence`.
- `config.load_spec_registry` / `config.spec_status`, applying
  `max(phase_start, specified_on, available_from)` by rule.
- `--estimator` / `--inference` on `src.experiment`, with the 2x2 scenario
  routing, the specification-status section, the power/equivalence DM columns,
  the replication panel, and overlap-aware 30-day inference.
- `make test-methodology`, `make baselines-smearing`, `make scenarios`,
  `make scenarios-all`. The empirical targets are gated on the test suite, as
  every other empirical target in this repository is.

Fidelity check: the restored smearing estimator reproduces the surviving
`*_sm.parquet` forecasts to 0.0 relative error on `har`, `har_iv`, `har_iv_x`
and `persistence`, and the restored `tail_ext` reproduces the published
`trunc` and `tail_ext` rows of the fork document's reconstruction table
exactly. The restoration is the original method, not a lookalike.

### What changed in the numbers, and what did not

**No verdict changed.** Every `A better` / `B better` / `equivalent` /
`inconclusive` in every corrected table is what it was.

Three things did move, all of them defects rather than data:

1. **Chronos-2 / TiRex-2 reconstructed rows (`~`).** Their previously published
   values are not reproducible by `trunc`, `lognormal` or `tail_ext`, i.e. by
   any method the repository contains or documents. They came from a fourth
   scheme that no longer exists and was never specified precisely enough to
   rebuild. The rows are now `tail_ext` throughout. The visible consequence is
   `chronos_cov_iv` vs `har_iv`: DM 0.429 → 0.553, and origins-to-resolve
   8,191 → 4,919. The verdict, `inconclusive`, is unchanged, as is the
   headline that the comparison was never a null. `METHODOLOGY_FORK.md` and
   the affected reports are updated. A new test,
   `test_reconstruction_table_matches_the_published_one`, pins the published
   table to the code so this cannot recur silently — the pre-existing test
   asserted only inequalities, which a wide family of schemes satisfies.
   The `lognormal` reference row is corrected from 0.945–0.968 / −4.6% /
   2.28pp to 0.952–0.962 / −4.3% / 1.00pp for the same reason.

2. **Two clean-window sentences were printed into the diagnostic report as
   fact.** The 30-day section stated "a lag/n ratio near 0.19 and understates
   se(beta) by about 3x" in both phases; on the diagnostic window the ratio is
   0.01 and the three standard errors are 0.061 (HAC), 0.060 (bootstrap) and
   0.090 (refits) — HAC is essentially fine there. The section also closed with
   "At n_eff=117 this section cannot reject anything", directly contradicting
   the `har_cum` row above it (c_model=0.196, bootstrap p=0.034). Both are now
   computed per phase; the closing caveat is conditional on n_eff.

3. **A premium labelled with a median from a different sample.** The premium
   line printed the phase-wide median VXN (24.1) beside a figure evaluated on
   the shorter sample that has a realized 30-day target (median 23.8). Now
   both come from the same frame.

Also reconciled: `flip k` for `har_iv_x vs har_iv` was published as 3 in
`methodology.py`, 4 in `METHODOLOGY_FORK.md` and 5 in `results_clean_v2.md`.
The computed value is 5; the first two were stale.

### What this does not fix

The fork's own outputs still have no byte-for-byte fence — only the frozen
pre-registered reports do. The new pinning test covers the estimator table,
not the reports. Nothing here revisits any frozen verdict or changes a
pre-registered quantity.

**Correction to the sentence that stood here.** It originally ended "…or
reopens the clean window." That was false, and an adversarial audit run the
same day caught it: the restored replication panel reopened the clean window
for the two quarantined models. See the next entry.

---

## 2026-08-13 (second) — adversarial audit of the restoration

**`config.yaml` unchanged. The three frozen reports remain byte-for-byte
identical.** An eight-lens adversarial audit was run against the repository
immediately after the restoration above, with every finding put through two
independent refutation passes. 39 raw findings, 20 verified, **14 survived**,
6 refuted. The eight items acted on here are the ones that were reproduced
independently before being fixed.

### The clean window was reopened for two quarantined models

`har_lev` and `har_iv_lev` are `DIAGNOSTIC_ONLY`: admitted for diagnostic-window
testing during the freeze, suppressed from clean-phase reports "so testing them
cannot become a peek". The guard lived inside the h=1 table loop only. The
corrected fork's replication panel is driven by `FULL_PAIRS` and calls
`_qlike_series(...)` directly, which had no guard — so `results_diagnostic_v2.md`
and `results_diagnostic_inf.md` published clean-window win rates, mean gaps and
a `replicates? yes` verdict for both models over all 192 clean origins, and
`METHODOLOGY_FORK.md` cited `har_lev vs har` as replicating cleanly.

Suppressing a row from a report is not a quarantine. **~38% of the eventual
≥500-origin gate draw is spent for these two models, with direction known**, and
no re-run un-spends it. When the gate opens their clean result must be read as
carrying a 192-origin peek. Nothing in the return-asymmetry conclusion depends
on it — that rests on the diagnostic window at n=2463.

Fixed at the source: `_quarantined()` reads `diagnostic_only` from
`spec_registry.yaml` (a field previously computed by `config.spec_status` and
read by nothing) and is enforced in both the table loop and `_qlike_series`.
`TestDiagnosticOnlyQuarantine` pins it, including a test that greps the
generated reports for the leak.

### "The estimator fix changes no rankings" was false

`har_sv vs har`, with **both sides exactly smeared on identical 188 origins**,
moves from DM −1.615 (p=0.1080) to −2.163 (**p=0.0318**) — across the
pre-registered α on the estimator switch alone. The only `equivalent` verdict in
the clean window (`har_ic vs har_iv`, p_TOST=0.039) also requires the estimator
fix: under the inference fix alone it is `inconclusive` at p_TOST=0.119, and
**none** of the three rows in that table earns a null. The section heading
claiming those numbers were "(fix 2)" was showing fixes 1+2; both tables are
now printed.

The stated mechanism was also backwards. QLIKE differentials are not
scale-invariant, so a common multiplicative factor moves DM on its own rather
than cancelling. And the "0.866–0.873 / 0.70pp near-common factor" holds only
over the five models it was measured on: 0.866–0.883 across the HAR family,
**0.812–0.883 (7.05pp)** across every quantile model scored, with `persistence`
at 0.812. The pinning test quantified over the narrow set while the prose
quantified over all models, so it passed while the claim was false; it now pins
all three scopes.

### Both 30-day standard errors were misdescribed

The figure published as the "honest standard error" (0.458) was the median
*within-subsample* SE — the SE of a beta fitted to ~8 points, algebraically
≈ √21 × the full-sample OLS se. It is not a standard error of the reported
coefficient, and the ratios formed against HAC from it (3x, then 2.9x) are
withdrawn. The key is renamed `se_within_subsample`; the reports now lead with
`beta_sd_across_phases`, which was computed all along and never printed.

Worse, **the bootstrap that replaced HAC is itself anti-conservative**: against
a DGP at this data's measured persistence its nominal-95% interval covers
**82.2%** (reproduced independently) and `p_zero` has true size ~10–16%. So
`results_diagnostic_v2.md`'s "a significant `c_model` above is a real rejection"
— resting on `har_cum` p=0.034 — was unsupported and is struck.
`TestBootstrapCalibration` now measures coverage in CI so this cannot be
forgotten again. Abandoning HAC(32) at lag/n=0.19 is still correct; the honest
statement is that no interval in that section is calibrated, which strengthens
rather than weakens the n_eff=8 conclusion.

### Failures to reject were reported as established nulls

`FINDINGS.md` opened with H1 as "a well-powered null for a difference of that
size, not an underpowered maybe", under a section headed "H1 is dead past
arguing". Measured: MDE = 14.1% of HAR's loss, **43× the observed gap**; power
against the gap actually observed = **0.050**; origins needed = **357,040**;
95% CI on the gap = [−9.5%, +10.2%] of HAR QLIKE; TOST p = 0.297. The corrected
verdict is `inconclusive`. The headline numbers quoted there (0.3688 / 0.3734 /
DM −0.259 / p 0.796) also reproduce nowhere — the code gives 0.3719 / 0.3731 /
−0.065 / 0.9483.

H1's real answer lives on the diagnostic window, where n=2463 gives the
equivalence test power: `har_x vs har` equivalent at p_TOST=0.006,
`har_iv_x vs har_iv` at 0.013. Corrected in `FINDINGS.md` and in README's
"Results in plain English", which had rendered it as a flat "No."

### The pre-committed earnings gate cannot fire at its own trigger

`next_evaluation.earnings_slice_confirmatory.min_heavy_earnings_days: 40` gates
the registered confirmatory DM test. Heavy-earnings days arrive at ~6.2–6.4% of
origins. Over **every** rolling 414-session window in the sample (the sessions
between `clean_start` and the `at_date: 2027-06-30` trigger) the maximum count
attained is **34**; 40 is reached in 0.1% of 500-session windows. Forty heavy
days accrues at ~640–750 origins, roughly 2028. The field was also read by no
code, so nothing computed the shortfall.

This is the repository's signature defect found once more: a registered test
that cannot deliver its verdict at its own registered date. `config.yaml` is
frozen, so it is **disclosed, not repaired** — `cmd_evaluate` now reads the
field and prints the reachability arithmetic beside the gate in every corrected
report.

### One verifier claimed independence it does not have

`verify_target_regime.py` re-derives features, thresholds, targets, lags and
fences from raw, but never re-derives a single forecast probability — it
aggregates the pipeline's own stored columns. The audit demonstrated the
consequence by injecting real look-ahead into `src.regime_transition` (full-
sample HMM fit, smoothed rather than filtered gammas) and watching the verifier
still print `VERIFICATION PASSED … both frozen verdicts independently match`.
Currently inert: an independent re-derivation reproduces every stored
probability at max abs error 0.0, and leakage pushes toward PASS while both
frozen verdicts are FAIL, so it cannot have manufactured the published
negatives. The banner and the README line now state the actual scope.
`src/verify_regime_repair.py` already meets the stronger standard and is cited
as the target.

### Not fixed here — recorded so the next reviewer starts from them

- **F6 — since confirmed and published additively.** The tail-ranking
  scoreboard pools 24 annual folds whose event rates range 0.000–0.808, four
  containing no event at all, so most scored pairs straddle two different years.
  Reproduced independently: a fold-constant score with **zero within-year
  information** reaches phase-mean AUC **0.8317**, above the published
  `p_rv_percentile` (0.8111), `p_hmm_platt` (0.8294) and the latent sparse k=1
  rung (0.8153) that cleared the 99-control gate. Within-fold AUC for every
  model is far lower (0.677–0.731) and the ordering **reorders** — the
  calibrated HMM is best within fold while ranking third pooled, so
  "HMM augmentation adds +0.0010 AUC while reducing lift" is a pooled-metric
  artifact. The prior-folds-only version of the same score sits at chance
  (0.5072), confirming the between-fold component is not itself forecastable.
  Both the pooling and the control design are *registered*, so no verdict is
  rewritten: `src/pooling_diagnostic.py` and
  `reports/representation_study/pooling_diagnostic.md` publish it alongside
  (`make pooling-diagnostic`). A future ranking protocol must stratify within
  fold before comparing.
- **F7.** The NQ study's "no evaluable folds" appears to be a rolling-window
  construction artifact rather than a shortage of history; a gap-tolerant
  reading reportedly yields 316 origins and a null. Changing it alters a frozen
  protocol and must be run as an additive, labelled sensitivity.
- **F12.** `noise_robustness`'s HAR arm aggregates on the variance scale while
  the impulse is injected on the log scale; all eight Gaussian HAR-vs-foundation
  intervals reportedly flip significance under the repo's other convention. The
  direction survives, the interval exclusions may not. The within-foundation
  Chronos-vs-TiRex comparison is untouched.
- 19 lower-severity findings were not adversarially verified and carry no
  claim here.
