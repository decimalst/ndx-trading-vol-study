# Open findings

Enumerated backlog from the 2026-08-13 adversarial audit. Each entry is scoped
to be filed as a single issue. "Unverified" means the finding was produced by a
review agent but **never put through the two adversarial passes** the confirmed
findings went through — treat the claim as a lead, not a result. Reproduce
before acting.

Confirmed-and-fixed findings are in
[`reports/AMENDMENTS.md`](../reports/AMENDMENTS.md) (the two 2026-08-13
entries). The correction ledger is in
[`reports/HOW_THIS_WAS_BUILT.md`](../reports/HOW_THIS_WAS_BUILT.md).

## Confirmed, not yet fixed

These survived adversarial verification. They require protocol amendments or
refits rather than edits, which is why they are open rather than done.

| # | Area | Finding | Why it is still open |
|---|---|---|---|
| O1 | `src/nq_intraday_study.py` | "No evaluable folds" appears to be a rolling-window construction artifact, not a shortage of history: strict 5/22 rolling means over a gap-containing series destroy ~142 of 205 otherwise-usable 2024 training rows. A gap-tolerant reading reportedly yields 316 origins and a null (AUC ≈ 0.50). | Changing the construction alters a **frozen protocol**. Must be run as an additive, labelled sensitivity, not an edit. README currently says "NQ lacked enough training history", which would become "the study runs and returns a null". |
| O2 | `src/noise_robustness.py` | The HAR arm aggregates the corrupted origin state on the **variance** scale, `log(mean(exp(x)))`, while the impulse is injected on the **log** scale — so one spike enters the 5/22-day lag as a multiplicative `e^(8σ)` factor. All eight registered Gaussian HAR-vs-foundation intervals reportedly flip from excluding zero to including zero under the repo's other aggregation convention. Direction survives; magnitude and interval exclusion may not. | Requires re-running the corruption grid under both conventions and publishing both columns. The load-bearing within-foundation comparison (Chronos-2 vs TiRex-2) is untouched either way. |
| O3 | `representation_study.yaml`, `latent_k1_confirmation.yaml` | The ranking scoreboard pools 24 annual folds with event rates 0.000–0.808. Diagnosed and published in [`pooling_diagnostic.md`](../reports/representation_study/pooling_diagnostic.md), but the **protocol itself** is unamended: the control label paths do not reproduce the actual between-fold event-rate dispersion (actual sd 0.181), and the statistic is not stratified within fold. | A future ranking protocol must stratify within fold before comparing. Amending a registered protocol is a pre-registration decision, not a code change. |

## Guardrail holes

| # | Area | Finding | Status |
|---|---|---|---|
| O4 | `tests/test_methodology.py::TestFrozenReportsUnchanged` | The byte-for-byte fence reads current bytes into memory, **overwrites the file**, then compares — so it can only detect drift introduced by that subprocess, and is structurally incapable of detecting that a frozen report *already* diverged. It is destructive and self-healing: with a corrupted dependency, consecutive invocations launder the corruption into the "frozen" record and go green. | **Fixed 2026-08-13** by pinning SHA-256 digests in `reports/FROZEN_REPORT_HASHES.json`. Kept here for the record. |
| O5 | test suite reachability | `python -m unittest discover` runs **0 tests and reports OK** (no `tests/__init__.py`). Most test functions were unreachable from any make target. | **Fixed 2026-08-13** — `make test` runs every module; CI runs it. |
| O6 | `src/experiment.py` | The frozen-report fence re-derives QLIKE/DM/coverage/MZ at write time but reads `mean_var` from cached forecast parquets. An error introduced into `models.trunc_mean_var` is therefore **invisible** to the fence — a `*1.001` patch leaves it green. `TestEstimatorReconstructionAccuracy` covers this only partially. | Open. Wants a forecast-level hash pin or a fence that refits. |
| O7 | `Makefile`, `docs/RUNBOOK.md` | `make daily-update` refreshes only the frozen clean path. The `_est`/`_v2` reports are scored off `*_sm.parquet`, which the loader prefers whenever present, so skipping `baselines-smearing` after an accrual round leaves the corrected reports scored on **fewer origins than the frozen one, silently**. | **Mitigated 2026-08-13** by documenting the three-step accrual sequence and adding `make repin-frozen-reports`. Not eliminated: nothing yet *detects* the origin-count mismatch. A guard asserting `n` parity between `results_clean.md` and `results_clean_v2.md` would close it. |
| O8 | `Makefile` | `baselines-smearing` passes no `--quantile-grid`, so the decile-grid `_est`/`_v2` reports have no make target and need a hand-run `src.experiment baselines --estimator smearing --quantile-grid deciles`. | Open. Wants a `GRID=` parameter on the target. |
| O9 | `pyproject.toml` | 128 ruff findings are **deferred, not clean** — 94 unnecessary `int()` casts, 31 `zip()` without `strict=`, and a tail of style rules. All sit in code that regenerates byte-for-byte-fenced artifacts, so autofixing them risks a silent numerical edit for a cosmetic gain. | Open by choice, with reasons in `pyproject.toml`. Clear one rule at a time, each followed by a frozen-report reproduction check. `ruff format` is also not enforced: it would reflow 69 numerical files. |
| O10 | repository size | 119 MB clone, 62 MB in `data/` — 132 parquet files plus a 15 MB zip. Reproduce-from-clean-clone is a real argument for committing inputs, but git is the wrong store for them: every clone pays for every historical revision of every binary. | Open. Release assets or an LFS/DVC pointer would preserve reproducibility at a fraction of the clone cost. Moving them is a history rewrite, so it needs a deliberate decision. |
| O11 | `src/verify_skew_carry.py`, `src/verify_regime_repair.py` | Both import the primitive they verify (`build_trades`, `fit_gaussian_hmm`), so a defect inside those functions reproduces identically on both sides. | **Disclosed 2026-08-13** — both banners now state the limit. Not closed: making them genuinely independent means reimplementing the primitive, as `src/verify_regime_repair.py` already does for the forward filter. |

## Unverified leads

Not adversarially verified. Ordered roughly by how load-bearing they look.

| # | Area | Lead |
|---|---|---|
| U1 | `src/carry.py` | `_boot_diff` reportedly permutes a sample against its own superset, giving a claimed true size ~0.0007 rather than the nominal level. Would affect the carry study's bootstrap p-values. |
| U2 | `src/skew_carry.py` | The registered 70% participation floor reportedly sits 0.27pp below the veto's own no-selectivity ceiling — an AND-gate in arithmetic self-opposition, the same shape as the 1/11 p-value-floor precedent already in the ledger. |
| U3 | `src/metrics.py` | `crps_from_quantiles` divides by the tau range, which would make CRPS silently **incomparable across the 7-point and 9-point grids** — the same class of defect as the point estimator already corrected. |
| U4 | `src/methodology.py` | `dm_tost`'s stated justification (`\|r1\| < 0.09` on both phases) is reportedly false on the diagnostic window: measured r1 = +0.137, Ljung–Box p = 1e-11, **for the exact pair producing the "earnings refuted" verdict**. If true, that verdict's TOST is anti-conservative. |
| U5 | `src/methodology.py` | `paired_summary`'s `flip_k` search is capped at 25, so `never` can mean "more than 25"; true counts reportedly 63 and 116 in some rows. Cosmetic unless `never` is read as "robust". |
| U6 | `calendars/fomc.csv` | Unscheduled 2020 emergency FOMC dates are fed as "forward-published" calendar covariates. 2020-03-16 was announced Sunday evening after the Friday close, so it was **not** known at the origin. Affects diagnostic-window event slices. |
| U7 | `src/research_paths.py` | `absorption_map`'s HAC(10) lag reportedly chosen because it reproduced a target number; the earnings absorption headline reportedly swings 20.1% → 33.9% across lag choices. |
| U8 | `src/regime_repair.py` | The incremental-state FAIL reportedly rests on a gap 3.6× smaller than its own standard error, with 102/166 origins pointing the other way — i.e. an `inconclusive` reported as a FAIL, the item-10 pattern again. |
| U9 | ingest layer | The fetch/ingest layer beyond the joins already checked was not audited at all. |
| U10–U19 | various | Ten further lower-severity findings from the audit's raw set were neither verified nor triaged. |

## What would count as converged

This process has produced 8, then 12, then 22 findings across three rounds, each
round surviving every prior one. It will count as converged when **a round
produces zero conclusion-changing findings**. Until then the current state is
provisional by the project's own standard, not merely by a reader's suspicion.
