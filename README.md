# NDX volatility forecasting: what the market already knows

A leakage-controlled, pre-registered study of Nasdaq-100 realized volatility.
The question: can a modern time-series model, plus calendar and market data
known at the forecast origin, predict QQQ/NDX variance better than classical
models — and better than the options market already does through VXN?

**The volatility answer is null and correctly so.** A compact HAR model using
VXN is difficult to beat. Across ~16 designs spanning QQQ/NDX and SPX, extra
index-level signals were redundant, non-stationary, contaminated by how they
were discovered, too weak to survive a frozen holdout — or, in several cases
the project originally miscounted as nulls, simply **unresolved by a window too
short to resolve them**.

**The finding with actual signal is the process one.** This repository was built
with heavy AI assistance over about three days, under pre-registration,
byte-for-byte reproduction fences, and adversarial review. Twenty-two
corrections were caught. The most instructive is that an entire set of
methodology corrections was **documented, tested, cited, and never implemented**
— 12 of its own 28 tests failed on a clean checkout, while its output sat in
`reports/` looking exactly like reproducible artifacts. Several later
corrections then **inherited the defect they were correcting**: the fix to
"failures to reject reported as nulls" was itself reported as a null; the fix to
"HAC understates the standard error" shipped a bootstrap covering 82%.

If you read one thing here, read that ledger:

### → [How this was built](reports/HOW_THIS_WAS_BUILT.md)

It states the division of labour between model and human, the guardrails, all 22
corrections with what each one cost, and what would count as this process having
converged (it has not: 8 → 12 → 22 findings across rounds, each round surviving
every prior one).

This is research code and a record of negative as well as positive evidence. It
is not a trading recommendation or financial advice.

## The honest headline

| | |
|---|---|
| **Does anything beat HAR-IV?** | Nothing tested does. But on the clean window that is **`inconclusive`, not a null**: for `chronos_cov_iv` vs HAR-IV the minimum detectable effect is 0.0413 QLIKE — **13% of HAR-IV's 0.3118 loss** — and resolving the gap actually observed would take **4,919 origins** against the 192 available (18,583 under the frozen estimator). That row's corrected point forecast is reconstructed by tail extension, so read it as indicative. |
| **What is genuinely established?** | On the 13× larger diagnostic window, where equivalence tests have power: calendar and earnings covariates are **`equivalent`** to their controls (p_TOST 0.006 / 0.013). That is a positive finding of no effect, and it is the one claim of this kind the project can actually support. |
| **What survives as positive?** | Return asymmetry (leverage) alongside VXN: joint Wald 103.19 → 62.63, so the surface prices ~40% of it and a decisive remainder survives. Its out-of-sample value is small and its **significance is estimator-dependent** — p = 0.159 frozen, p = 0.030 corrected. |
| **What was retracted today?** | Three ranking results. A score with *zero within-year information* scores 0.8317 AUC on the annual-fold scoreboard, above several published headlines. See below. |

Full results table and evidence hierarchy: **[standing findings](reports/FINDINGS.md)**.
Every "no difference" claim written before 2026-08-13 should be read against
**[the methodology fork](reports/METHODOLOGY_FORK.md)**, which supersedes several
of them.

## Three retracted ranking claims

The tail-ranking scoreboard pools 24 annual folds whose event rates run 0.000 to
0.808 — four contain no event at all — so most scored pairs straddle two
different years. A **fold-constant score with zero within-year information**
reaches phase-mean AUC **0.8317**. Measured in
[`pooling_diagnostic.md`](reports/representation_study/pooling_diagnostic.md)
(`make pooling-diagnostic`):

| claim as previously published | status |
|---|---|
| "Price history ranks stress-state crossings: 0.870 AUC, 4.80× lift" | Survives the ceiling (0.8704 > 0.8317) but **most of it is between-fold**; within-fold AUC is 0.7034. |
| "The calibrated HMM adds no usable information" | **Sign flip.** Within fold the calibrated HMM is the *best* of the four scores (0.7314 vs benchmark 0.7034). The pooled ordering that produced "no usable addition" is a metric artifact. |
| "TiRex k=1 survives a 99-control test at p=0.01" | The p-value is real but is the **floor** of a 99-control design, and its 0.8153 AUC is **below** the 0.8317 zero-information ceiling. |

The pooling and the control design are both *registered*, so no frozen verdict is
rewritten; the diagnostic is published alongside. A future ranking protocol must
stratify within fold before comparing.

### The follow-up that should settle it — and can't

If the latent probe's selected coordinates track smoothed volatility, the
question worth asking is not "is transition proximity decodable?" but "is
anything in there **orthogonal to realized-volatility history**?" So:
project the HAR feature set out of every one of the 512 coordinates using
training-fold coefficients only, reselect, and score within fold
([`residual_probe.md`](reports/representation_study/residual_probe.md),
`make residual-probe`).

| | |
|---|---|
| HAR's median R² on a latent coordinate | **0.305** (fold range 0.242–0.748) — so the latent is *not* mostly RV history in a variance sense |
| Residualized k=1 alone, within fold | 0.6298 — above chance, so the orthogonal part is not noise |
| HAR alone → HAR + residualized k=1 | 0.7162 → 0.7176; mean fold ΔAUC **−0.0006**, 10/19 folds improved |
| verdict | **`inconclusive`** — MDE is 0.0234, **37× the observed effect** |

The useful output is the last line, and it is a statement about the design
rather than the data: the fold-to-fold spread (−0.075 to +0.086) is an order of
magnitude larger than the effect being looked for, so resolving a one-AUC-point
effect would need **~104 annual folds**. More history does not fix that. Anyone
wanting to answer this needs a different unit of inference — not a longer
sample. Reported as `inconclusive` rather than as the null it superficially
resembles.

## Where to start

- [How this was built](reports/HOW_THIS_WAS_BUILT.md) — the correction ledger. Start here.
- [Standing findings](reports/FINDINGS.md) — current conclusions and evidence hierarchy.
- [Methodology fork](reports/METHODOLOGY_FORK.md) — the four defects that changed
  conclusions, and the `_est`/`_inf`/`_v2` reports that isolate each.
- [Amendments](reports/AMENDMENTS.md) — everything changed after results were observed.
- [Pooling diagnostic](reports/representation_study/pooling_diagnostic.md) — what the
  ranking scoreboard actually measures.
- [Open backlog](docs/OPEN_FINDINGS.md) — audit findings not yet resolved, enumerated.
- Full results table → [`docs/RESULTS.md`](docs/RESULTS.md)
- Method, leakage policy, hypotheses, verdict vocabulary → [`docs/METHOD.md`](docs/METHOD.md)
- Reproduce anything → [`docs/RUNBOOK.md`](docs/RUNBOOK.md)
- Data sourcing and provenance → [`docs/DATA.md`](docs/DATA.md) · licensing → [`DATA-LICENSE.md`](DATA-LICENSE.md)
- Orthogonal-signal study → [`docs/SIGNAL_STUDY.md`](docs/SIGNAL_STUDY.md)
- Per-study reports → [`reports/`](reports/)

Code and prose are MIT ([`LICENSE`](LICENSE)); third-party data under `data/`
and `calendars/` keeps its original terms ([`DATA-LICENSE.md`](DATA-LICENSE.md)).

## How to read a verdict

`FAIL` means a registered criterion was not met; it does not always mean the
mechanism was useless. **`inconclusive` means the design could not tell** —
it is not evidence of no effect, and the repository spent months conflating the
two. Only `equivalent`, which requires an equivalence test to reject
non-equivalence, is a positive finding of no difference. Exploratory
specifications are marked and must not be quoted as confirmatory.
Full vocabulary: [`docs/METHOD.md`](docs/METHOD.md#verdict-vocabulary-added-2026-08-13-with-the-corrected-fork).

## Known limitations

Read these before believing any number here.

- **Several headline comparisons are `inconclusive`, not null.** The clean window
  is 192 origins. A failure to reject is not evidence of no effect.
- **The RV proxy is not 5-minute realized variance.** Every original QQQ headline
  table was produced with `SOURCE=daily` — a Garman–Klass daily estimate plus the
  squared overnight gap, **not five-minute realized variance**.
- **Event slices are descriptive, full stop.** n=6–12 in the clean window: 6 FOMC,
  8 CPI, 9 NFP, 30 earnings-weight days, 12 in the frozen ≥5% heavy-earnings slice
  (157 in the diagnostic window).
- **The clean window contains no volatility event** and cannot settle tail questions.
- **The ~16 studies are not statistically independent**; their count is not a meta-test.
- **`har_lev` / `har_iv_lev` no longer have a clean draw.** 192 clean origins were
  scored and published for them before a quarantine bug was fixed — ~38% of the
  gate draw, spent with direction known.
- **The pre-committed earnings gate cannot fire at its own trigger.** It requires 40
  heavy-earnings days; the maximum attainable in any window of that length across
  the whole sample is 34. `config.yaml` is frozen, so this is disclosed, not repaired.
- **The 30-day bootstrap is anti-conservative** (~82% coverage at nominal 95%).
  Its intervals are a floor on the uncertainty, not a calibrated interval.
- **Every DM row involving a quantile model is estimator-dependent.** QLIKE is not
  scale-invariant. Three rows cross α=0.05 when corrected.
- **The corrected fork's own reports have no byte-for-byte fence** — only the frozen
  pre-registered reports do. That asymmetry is exactly how the fork rotted once.
- **Earnings sessions are inferred**, not verified against company IR pages. A wrong
  session shifts a volatility day by one.
- **Index weights are approximate.** Earnings weights are assigned per announcement from
  the latest point-in-time reconstruction available before that announcement, so no future
  snapshot can change an earlier feature — but the tracked 13-name basket uses current
  membership normalized to a fixed 55.43% aggregate, capturing relative drift rather than
  exact historical NDX membership. The five-path extension avoids this by starting from
  the first SEC-accepted 2019 quarterly top-25 snapshot, which still leaves a delayed
  quarterly fund proxy rather than exact daily weights.
- **Calendars are checked in, not fetched at runtime** (bls.gov and invesco.com reject
  scripted requests). They run through 2026, FOMC through 2027; after that `is_cpi` /
  `is_nfp` silently become 0 unless extended by hand.
- **Ranking AUCs are mostly between-fold** — see the retraction table above.
- **No monthly peeking.** The next evaluation is pre-committed to 500 scored origins or
  2027-06-30, whichever comes first. Reaching a gate is permission to look, not
  permission to stop at a favourable p.
- **Two audit findings are recorded and *not* fixed**, because each would amend a frozen
  protocol rather than correct a computation. (i) The NQ study's "no evaluable folds"
  looks like a rolling-window construction artifact, not a shortage of history — strict
  5/22 rolling means over a gap-containing series destroy ~142 of 205 otherwise-usable
  2024 training rows — and a gap-tolerant reading *reportedly* yields 316 origins and a
  null. (ii) `noise_robustness`'s HAR arm aggregates the corrupted state on the
  **variance** scale while the impulse is injected on the **log** scale; all eight
  Gaussian HAR-vs-foundation intervals *reportedly* flip to including zero under the
  repo's other convention. Direction survives, exclusions may not; the within-foundation
  Chronos-2 vs TiRex-2 comparison is untouched. Both need additive, labelled
  sensitivities. Plus 19 unverified lower-severity leads — all enumerated in
  [`docs/OPEN_FINDINGS.md`](docs/OPEN_FINDINGS.md).

## Related work

The latent-probe and context-noise studies sit in the 2026 literature on probing
time-series foundation models (TSFMs) — linear/sparse probes over frozen
representations, and context-corruption robustness grids in the style of Eidos.
The probing *method* here is not a contribution. What this repository adds is the
**incremental-value test against a strong domain benchmark**: not "is transition
proximity decodable from the latent state?" (it is, at 0.8153 AUC) but "does it
beat direct realized-volatility history, and does it beat a score containing no
within-year information at all?" (no, and no). That comparison is routinely
absent from the probing literature, which tends to establish decodability and
stop. The same standard applied to this repository's own results is what produced
the retraction table above.

## Status

Provisional by the project's own standard. Three rounds of adversarial review
produced 8, then 12, then 22 findings, each round surviving every prior one. This
process will count as converged when a round produces **zero conclusion-changing
findings**. That has not happened.
