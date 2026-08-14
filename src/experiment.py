"""Orchestration CLI.

  python -m src.experiment features  [--source daily|bars]
  python -m src.experiment baselines [--phase diagnostic|clean|all]
  python -m src.experiment chronos   [--phase clean] [--variant uni|cov|cov_iv]
  python -m src.experiment tirex     [--phase clean] [--variant uni|cov|cov_iv]
  python -m src.experiment evaluate  [--phase clean] [--source daily|bars]

`evaluate` writes reports/results_{phase}.md with: h=1 loss table, interval
coverage tests, DM matrix vs HAR, event-sliced QLIKE, and the 30-calendar-day
MZ + encompassing regressions against VXN.

--quantile-grid
    `preregistered` (default) uses config.yaml's `quantiles` — the frozen
    pre-registration grid, [0.05 .. 0.95], widest interval 90%.
    `deciles` overrides it in memory with 0.1 .. 0.9 and suffixes every output
    with `_dec`. TiRex-2 emits exactly those deciles and cannot be asked for
    other levels, so comparing it against the other models requires recomputing
    all of them on that grid — the truncated-mean estimator behind `mean_var`
    is only comparable across models sharing a grid. This never mutates
    config.yaml, so the pre-registered results stand alongside it untouched.

--estimator / --inference
    The corrected-methodology fork (reports/METHODOLOGY_FORK.md). Both default
    to the frozen pre-registration, so the default command line reproduces
    `results_{phase}.md` byte-for-byte — asserted by
    tests/test_methodology.py::TestFrozenReportsUnchanged, which is what makes
    a correction unable to quietly rewrite the result it was meant to be
    compared against.

        estimator  trunc    | smearing     point forecast behind QLIKE
        inference  naive    | corrected    power, equivalence, spec gate, overlap

    The four combinations write four files, so each correction can be read in
    isolation rather than as one undifferentiated "corrected" number:

        trunc    + naive      results_{phase}.md        frozen
        smearing + naive      results_{phase}_est.md    estimator fix alone
        trunc    + corrected  results_{phase}_inf.md    inference fixes alone
        smearing + corrected  results_{phase}_v2.md     both
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from . import config, features, methodology, metrics, models


DECILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# (estimator, inference) -> report suffix. Appended AFTER the grid suffix, so
# the decile-grid fork lands in results_clean_dec_v2.md.
SCENARIO_SUFFIX = {
    ("trunc", "naive"): "",
    ("smearing", "naive"): "_est",
    ("trunc", "corrected"): "_inf",
    ("smearing", "corrected"): "_v2",
}

# Overlap of the 30-calendar-day target in trading days, and the bootstrap
# settings behind the corrected inference. Fixed, not tuned per report.
OVERLAP_30D = 21
N_BOOT = 2000
BOOT_SEED = 20260812


def apply_grid(cfg: dict, grid: str) -> str:
    """Set cfg['quantiles'] for the run and return the output-file suffix."""
    if grid == "deciles":
        cfg["quantiles"] = list(DECILES)
        return "_dec"
    return ""


def apply_scenario(cfg: dict, estimator: str, inference: str) -> str:
    """Record the fork scenario on cfg and return the report suffix."""
    if (estimator, inference) not in SCENARIO_SUFFIX:
        raise ValueError(f"unknown scenario {(estimator, inference)!r}")
    cfg["_estimator"], cfg["_inference"] = estimator, inference
    return SCENARIO_SUFFIX[(estimator, inference)]


def _est_suffix(cfg: dict) -> str:
    """Forecast-file suffix for the active estimator (`_sm` for smearing)."""
    return "_sm" if cfg.get("_estimator") == "smearing" else ""


def _heavy_cutoff(cfg: dict, wt: pd.Series) -> float:
    """Heavy-earnings threshold: a frozen absolute weight, not a sample quantile.

    An in-sample quantile is recomputed on every run, so the slice reshuffles
    whenever the data changes — it moved from n=8 to n=7 on the point-in-time
    weight fix alone. See config.yaml.
    """
    if cfg.get("heavy_earnings_min_wt") is not None:
        return float(cfg["heavy_earnings_min_wt"])
    pos = wt[wt > 0]
    return float(pos.quantile(cfg["heavy_earnings_quantile"])) if len(pos) else np.inf


def _phase_bounds(cfg: dict, phase: str) -> tuple[str, str]:
    if phase == "diagnostic":
        return cfg["diagnostic_start"], cfg["diagnostic_end"]
    if phase == "clean":
        return cfg["clean_start"], "2099-01-01"
    return cfg["diagnostic_start"], "2099-01-01"


def _phase_origins(df: pd.DataFrame, cfg: dict, phase: str) -> pd.DatetimeIndex:
    lo, hi = _phase_bounds(cfg, phase)
    idx = df.index[(df.index >= lo) & (df.index <= hi)]
    return idx[:-1]  # last date has no realized t+1 yet


def _load_master(cfg: dict, source: str) -> pd.DataFrame:
    return pd.read_parquet(cfg["paths"]["processed"] / f"master_{source}.parquet")


def _forecast_path(cfg: dict, name: str, phase: str, est: str):
    """Existing forecast file for `name`/`phase` at estimator suffix `est`."""
    fdir = cfg["paths"]["forecasts"]
    sfx = cfg.get("_grid_suffix", "")
    for stem in (f"{name}_{phase}{sfx}{est}", f"{name}_all{sfx}{est}"):
        p = fdir / f"{stem}.parquet"
        if p.exists():
            return p
    return None


def _load_forecast(cfg: dict, name: str, phase: str) -> pd.DataFrame | None:
    """Forecasts for `name` in `phase`, falling back to a `_all` run.

    Every model refits per origin from history up to that origin, so an origin's
    forecast does not depend on which origins were requested alongside it —
    slicing an `all` run by date is identical to having run that phase directly.
    This is what makes the documented `baselines PHASE=all` then
    `evaluate PHASE=clean` order work.

    Under `--estimator smearing` the `*_sm` run is preferred and the frozen run
    is the fallback, because only models with recoverable residuals have one.
    Chronos-2 and TiRex-2 do not, so they land here on the frozen file and the
    caller reconstructs their point forecast from the saved quantiles.
    """
    est = _est_suffix(cfg)
    p = _forecast_path(cfg, name, phase, est)
    if p is None and est:
        p = _forecast_path(cfg, name, phase, "")
    if p is None:
        return None
    fc = pd.read_parquet(p)
    lo, hi = _phase_bounds(cfg, phase)
    return fc[(fc.index >= lo) & (fc.index <= hi)]


def cmd_baselines(cfg: dict, phase: str, source: str) -> None:
    df = _load_master(cfg, source)
    origins = _phase_origins(df, cfg, phase)
    est = cfg.get("_estimator", "trunc")
    runs = {
        "persistence": lambda: models.run_persistence(df, origins, cfg, estimator=est),
        "ewma": lambda: models.run_ewma(df, origins, cfg),
        "har": lambda: models.run_har(df, origins, cfg, with_x=False, estimator=est),
        "har_x": lambda: models.run_har(df, origins, cfg, with_x=True, estimator=est),
        # Control for the VXN-fed models. No `_cum` counterpart on purpose: the
        # encompassing regression already conditions on VXN, so a HAR-IV
        # cumulative forecast would be collinear with the regressor by design.
        "har_iv": lambda: models.run_har(df, origins, cfg, with_x=False, with_iv=True,
                                         estimator=est),
        # Signed semivariance (Patton-Sheppard): swaps the daily term for its
        # signed halves. No new data — hourly bars only supply the split share.
        "har_sv": lambda: models.run_har(df, origins, cfg, with_sv=True, estimator=est),
        # HAR-IV plus the implied-correlation decomposition.
        "har_ic": lambda: models.run_har(df, origins, cfg, with_iv=True, with_ic=True,
                                         estimator=est),
        # HAR-IV-X: log-HAR + log(VXN) + event terms. Pre-specified as the ONLY
        # new specification after the clean-phase results were seen, on this
        # reasoning: VXN is a 30-day constant-maturity measure and structurally
        # cannot know that a 9%-weight name prints tonight, yet har_x and
        # chronos_cov both beat har_iv on heavy-earnings days while losing to it
        # everywhere else. A significant earnings coefficient ALONGSIDE VXN is
        # H3-flavoured evidence at h=1: information the surface does not price.
        "har_iv_x": lambda: models.run_har(df, origins, cfg, with_x=True, with_iv=True,
                                           estimator=est),
        # LHAR: return asymmetry from daily data (1999-), the mechanism under
        # signed semivariance, on a sample big enough to settle it. DIAGNOSTIC
        # ONLY — see DIAGNOSTIC_ONLY below; it must not spend a clean-window draw.
        "har_lev": lambda: models.run_har(df, origins, cfg, with_lev=True,
                                          estimator=est),
        # Ledger-closing test: does return asymmetry survive alongside VXN?
        # VXN embeds skew, which is the same economics, so leverage may just be
        # VXN by another route. DIAGNOSTIC ONLY. Either answer closes the map:
        # subsumed -> no open items; not subsumed -> the one non-market signal
        # that survived everything, documented and not acted on.
        "har_iv_lev": lambda: models.run_har(df, origins, cfg, with_iv=True,
                                             with_lev=True, estimator=est),
    }
    sfx = cfg.get("_grid_suffix", "") + _est_suffix(cfg)
    for name, fn in runs.items():
        out = fn()
        path = cfg["paths"]["forecasts"] / f"{name}_{phase}{sfx}.parquet"
        out.to_parquet(path)
        print(f"{name}: {len(out)} forecasts -> {path}")
    if est != "trunc":
        # The 30-day models emit a log point forecast that goes straight into a
        # log regression; there is no exp() to take, so the point-forecast
        # estimator does not apply and re-emitting them under `_sm` would only
        # create a second identical file to drift out of sync.
        print("30c cumulative forecasts are estimator-independent — not rewritten")
        return
    gsfx = cfg.get("_grid_suffix", "")
    for name in ("har", "persistence"):
        cum = models.predict_cum(df, origins, cfg, model=name)
        path = cfg["paths"]["forecasts"] / f"{name}_cum_{phase}{gsfx}.parquet"
        cum.to_parquet(path)
        print(f"{name} 30c cumulative: {len(cum)} -> {path}")


def cmd_chronos(cfg: dict, phase: str, variant: str, source: str) -> None:
    from . import chronos_runner

    df = _load_master(cfg, source)
    origins = _phase_origins(df, cfg, phase)
    if phase != "clean":
        print("NOTE: pre-checkpoint origins are leakage-contaminated; "
              "diagnostic-phase Chronos results are plumbing checks only.")
    out = chronos_runner.run_chronos(df, origins, cfg, variant=variant)
    sfx = cfg.get("_grid_suffix", "")
    path = cfg["paths"]["forecasts"] / f"chronos_{variant}_{phase}{sfx}.parquet"
    out.to_parquet(path)
    print(f"chronos_{variant}: {len(out)} forecasts -> {path}")


def cmd_tirex(cfg: dict, phase: str, variant: str, source: str) -> None:
    from . import tirex_runner

    df = _load_master(cfg, source)
    origins = _phase_origins(df, cfg, phase)
    if phase != "clean":
        print("NOTE: diagnostic-phase TiRex results are plumbing checks only.")
    out = tirex_runner.run_tirex(df, origins, cfg, variant=variant)
    # TiRex only emits deciles; writing them under the pre-registered suffix
    # would put a differently-gridded mean_var next to the others.
    sfx = cfg.get("_grid_suffix", "")
    if sfx != "_dec":
        sys.exit("tirex requires --quantile-grid deciles (the model emits only "
                 "0.1..0.9 and mean_var is not comparable across grids)")
    path = cfg["paths"]["forecasts"] / f"tirex_{variant}_{phase}{sfx}.parquet"
    out.to_parquet(path)
    print(f"tirex_{variant}: {len(out)} forecasts -> {path}")


def _h1_scores(df: pd.DataFrame, fc: pd.DataFrame, taus: np.ndarray) -> pd.DataFrame:
    nxt = df["log_rv"].shift(-1).rename("y_log")
    nxt_var = df["rv_total"].shift(-1).rename("y_var")
    j = fc.join(nxt).join(nxt_var).dropna(subset=["y_log"])
    qcols = [f"q{t:.2f}" for t in taus]
    out = pd.DataFrame(index=j.index)
    out["qlike"] = metrics.qlike(j["y_var"].values, j["mean_var"].values)
    if all(c in j.columns for c in qcols):
        qmat = j[qcols].values
        out["crps"] = metrics.crps_from_quantiles(j["y_log"].values, qmat, taus)
        # Outermost available levels — 0.05/0.95 on the pre-registered grid,
        # 0.1/0.9 on the decile grid. Never hardcode: TiRex-2 has no 0.05/0.95.
        out["pin_lo"] = metrics.pinball(j["y_log"].values, j[qcols[0]].values, taus[0])
        out["pin_hi"] = metrics.pinball(j["y_log"].values, j[qcols[-1]].values, taus[-1])
        out["_lo"], out["_hi"], out["_y"] = j[qcols[0]], j[qcols[-1]], j["y_log"]
    return out


# --------------------------------------------------------------------------
# Corrected-fork helpers. None of these run in the frozen default path.
# --------------------------------------------------------------------------
DM_HEADER_NOTE = (
    "`MDE` is the smallest QLIKE gap this sample could detect at 80% power; "
    "`n_req` is how many origins it would take to resolve the gap actually "
    "observed. `p_TOST` tests EQUIVALENCE against a margin of 3% of the "
    "benchmark's loss. A non-significant DM alone never earns the verdict "
    "`equivalent` — without the TOST it is `inconclusive`.")


# Specifications admitted for diagnostic-window testing during the freeze. They
# are quarantined from the clean WINDOW, not merely from the clean REPORT --
# scoring them on clean origins spends the draw whether or not the number is
# printed in the headline table. `spec_registry.yaml` carries the same fact as
# `diagnostic_only: true`; `_quarantined` reads the registry and this tuple is
# the fallback, with `tests/test_methodology.py` asserting the two agree.
DIAGNOSTIC_ONLY = ("har_lev", "har_iv_lev")

_REGISTRY_CACHE: dict | None = None


def _registry() -> dict:
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        _REGISTRY_CACHE = config.load_spec_registry()
    return _REGISTRY_CACHE


def _quarantined(name: str, phase: str) -> bool:
    """Is `name` barred from being SCORED at all in `phase`?

    The h=1 table had this guard; the corrected fork's replication panel did
    not, and published clean-window win rates and mean gaps for both
    diagnostic-only models. A quarantine that only filters the report is not a
    quarantine -- the point is that the clean window is not looked at.
    """
    if phase == "diagnostic":
        return False
    entry = (_registry().get("models") or {}).get(name) or {}
    return bool(entry.get("diagnostic_only")) or name in DIAGNOSTIC_ONLY


def _phase_start(cfg: dict, phase: str) -> str:
    return _phase_bounds(cfg, phase)[0]


def _dm_extras(loss_a: pd.Series, loss_b: pd.Series) -> str:
    """MDE / n_req / p_TOST / verdict, the columns that turn a p into a verdict."""
    a, b = loss_a.values, loss_b.values
    mde = methodology.dm_mde(a, b)
    margin = methodology.equivalence_margin(b)
    tost = methodology.dm_tost(a, b, margin=margin)
    return mde, tost, margin


def _other_phase(phase: str) -> str:
    return "diagnostic" if phase == "clean" else "clean"


def _qlike_series(cfg: dict, df: pd.DataFrame, name: str, phase: str,
                  taus: np.ndarray) -> pd.Series | None:
    """QLIKE per origin for `name` in `phase`, scored under the active estimator.

    Used by the replication panel to score the OTHER window. It re-reads the
    forecasts rather than reusing the report's own table, so the two windows
    are scored by identical code.

    Returns None for a model quarantined in `phase`. Without this the panel
    reaches into the clean window for the diagnostic-only models and publishes
    their clean win rate and mean gap -- which is the peek the quarantine
    exists to prevent, arriving through the back door of the OTHER-window
    column rather than through the report's own table.
    """
    if _quarantined(name, phase):
        return None
    fc = _load_forecast(cfg, name, phase)
    if fc is None or fc.empty:
        return None
    fc = _apply_estimator(cfg, name, fc, phase, taus)[0]
    s = _h1_scores(df, fc, taus)
    return None if s.empty else s["qlike"]


def _apply_estimator(cfg: dict, name: str, fc: pd.DataFrame, phase: str,
                     taus: np.ndarray) -> tuple[pd.DataFrame, bool]:
    """Return (forecasts, exact) with `mean_var` on the active estimator.

    `exact` is False when the model has no recoverable residuals and its point
    forecast had to be rebuilt from saved quantiles — those rows are flagged
    `~` so a reconstruction is never mistaken for a measurement.
    """
    if cfg.get("_estimator") != "smearing":
        return fc, True
    if _forecast_path(cfg, name, phase, "_sm") is not None:
        return fc, True                      # a real smearing run exists
    rescored = models.rescore_mean_var(fc, taus, method="tail_ext")
    if rescored.isna().all():
        return fc, True                      # no quantiles (EWMA): native forecast
    return fc.assign(mean_var=rescored), False


def cmd_evaluate(cfg: dict, phase: str, source: str) -> None:
    df = _load_master(cfg, source)
    taus = np.asarray(cfg["quantiles"])
    estimator = cfg.get("_estimator", "trunc")
    corrected = cfg.get("_inference", "naive") == "corrected"
    registry = config.load_spec_registry() if corrected else None
    specs: dict[str, dict] = {}
    recon: set[str] = set()
    names = ["persistence", "ewma", "har", "har_x", "har_iv", "har_sv", "har_ic", "har_iv_x", "har_lev", "har_iv_lev",
             "chronos_uni", "chronos_cov", "chronos_cov_iv",
             "tirex_uni", "tirex_cov", "tirex_cov_iv", "tirex_cov_ivf"]
    # Models handed VXN must be judged against the VXN-fed linear control, not
    # against plain HAR — beating HAR only shows that implied vol predicts
    # realized vol, which has been known for decades.
    # DIAGNOSTIC_ONLY is module-level so `_qlike_series` enforces the same
    # quarantine; see `_quarantined`.
    IV_FED = ("har_ic", "har_iv_x", "har_iv_lev", "chronos_cov_iv", "tirex_cov_iv", "tirex_cov_ivf")
    # Interval implied by the grid: 90% pre-registered, 80% on deciles.
    nominal = float(taus[-1] - taus[0])
    scores, lines = {}, []
    lines.append(f"# Results — phase: {phase}\n")
    grid_note = (f"Quantile grid: {[round(t, 3) for t in taus.tolist()]} "
                 f"→ intervals below are {nominal:.0%}.")
    if cfg.get("_grid_suffix") == "_dec":
        grid_note += (" **Decile grid, not the pre-registered one.** TiRex-2 emits "
                      "only these levels, so every model here was recomputed on "
                      "them; `mean_var` is comparable within this report but not "
                      "against `results_*.md`. config.yaml is unchanged.")
    lines.append(grid_note + "\n")

    gsfx = cfg.get("_grid_suffix", "")
    if estimator != "trunc" or corrected:
        lines.append(
            f"**Corrected methodology run** — estimator=`{estimator}`, "
            f"inference=`{cfg.get('_inference', 'naive')}`. This report is a fork; "
            f"`results_{phase}{gsfx}.md` still holds the frozen pre-registered "
            f"numbers and is unchanged. Differences between the two are "
            f"methodology, not data. See src/methodology.py for what each "
            f"correction does and why.\n")
    if estimator == "smearing":
        lines.append(
            "`mean_var` here is the Duan smearing estimate "
            "`exp(mu)*mean(exp(resid))`. The frozen estimator integrates `exp(q)` "
            "over the quantile grid and divides by its mass, which discards the "
            "tails: it returns about 0.87 of the true conditional mean, and the "
            "discarded share depends on the grid — which is why every model's "
            "QLIKE differs between `results_clean.md` and `results_clean_dec.md` "
            "while EWMA's is 0.4036 in both. Only QLIKE changes; CRPS, pinball "
            "and coverage read off the quantiles and are identical.\n")
        lines.append(
            "Two caveats, stated rather than buried. (i) The frozen estimator's "
            "bias spans 0.866–0.873 across `har`, `har_x`, `har_iv`, `har_iv_x` "
            "and `har_ic` — but 0.866–0.883 once `har_sv`, `har_lev` and "
            "`har_iv_lev` are included, and 0.812–0.883 across every quantile "
            "model scored here, because `persistence` sits at 0.812. It is a "
            "near-common factor only on the narrow set. Correcting it moves "
            "QLIKE **levels** far more than DM statistics, but **not by "
            "cancellation** — QLIKE differentials are not scale-invariant, and "
            "rescaling both forecasts by a single common factor reproduces most "
            "of the DM movement on its own. It is therefore not safe to assume "
            "rankings are unaffected: on this window `har_sv vs har` moves from "
            "DM −1.615 (p=0.1080) to −2.163 (p=0.0318) on identical exactly-"
            "scored origins, crossing α=0.05. (ii) Rows marked `~` below have no "
            "recoverable residuals (Chronos-2, TiRex-2) and are reconstructed "
            "from their saved quantiles by tail extension, which on the five "
            "pinned HAR models lands 3.5% low with a 2.5pp spread — real, and a "
            "wider spread than the 0.70pp it replaces for those rows. A `~` row "
            "compared against an unmarked row mixes estimators; read those "
            "comparisons as indicative. Rows without `~` are exact.\n")

    lines.append("## h=1 losses (mean per day)\n")
    status_col = " status |" if corrected else ""
    status_dash = "---|" if corrected else ""
    lines.append(f"| model | n |{status_col} QLIKE | CRPS | pin{taus[0]:.2f} "
                 f"| pin{taus[-1]:.2f} | {nominal:.0%} cov | p_uc | p_ind |")
    lines.append(f"|---|---|{status_dash}---|---|---|---|---|---|---|")
    for name in names:
        if _quarantined(name, phase):
            continue
        fc = _load_forecast(cfg, name, phase)
        if fc is None or fc.empty:
            continue
        fc, exact = _apply_estimator(cfg, name, fc, phase, taus)
        s = _h1_scores(df, fc, taus)
        if s.empty:
            continue
        scores[name] = s
        if not exact:
            recon.add(name)
        if "_y" in s:
            cov = metrics.coverage_tests(s["_y"].values, s["_lo"].values,
                                         s["_hi"].values, nominal=nominal)
            cov_str = f"{cov['coverage']:.3f} | {cov['p_uc']:.3f} | {cov['p_ind']:.3f}"
            extra = (f"{s['crps'].mean():.4f} | {s['pin_lo'].mean():.4f} | "
                     f"{s['pin_hi'].mean():.4f}")
        else:
            cov_str, extra = "- | - | -", "- | - | -"
        label = f"{name} ~" if name in recon else name
        st = ""
        if corrected:
            specs[name] = config.spec_status(registry, name, _phase_start(cfg, phase))
            st = (" confirmatory |" if specs[name]["confirmatory"]
                  else " **exploratory** |")
        lines.append(f"| {label} | {len(s)} |{st} {s['qlike'].mean():.4f} | "
                     f"{extra} | {cov_str} |")

    # ---- specification gate ----
    if corrected:
        gate = int(cfg.get("next_evaluation", {}).get("at_origins", 0))
        weak = [n for n in scores if not specs[n]["confirmatory"]]
        lines.append("\n### Specification status\n")
        lines.append(
            "A model is *confirmatory* in this phase only from "
            "`max(phase_start, specified_on, available_from)` onward. Anything "
            "specified after the window opened was tested on the data that "
            "produced it, so its p-values in the tables below are descriptive, "
            "not inferential. The evaluator already applied this rule to "
            "`har_lev`/`har_iv_lev` by hand; `spec_registry.yaml` applies it to "
            "everything.\n")
        if not weak:
            lines.append("Every model in this report is confirmatory in this phase.\n")
        else:
            lines.append("| model | why not confirmatory | confirmatory origins "
                         "available | gate |")
            lines.append("|---|---|---|---|")
            below = True
            for n in weak:
                frm = specs[n]["confirmatory_from"]
                if frm is None:
                    avail = "0 (date unrecorded)"
                    n_ok = 0
                else:
                    n_ok = int((scores[n].index >= frm).sum())
                    avail = f"{n_ok:,}"
                below &= n_ok < gate
                lines.append(f"| {n} | {specs[n]['reason']} | {avail} | {gate} |")
            lines.append("")
            if below:
                lines.append("Every row above is below its gate. Those models are "
                             "reported for completeness and must not be quoted as "
                             "confirmatory results.\n")
            else:
                lines.append("A row at or above its gate has accrued enough "
                             "confirmatory origins to be read inferentially from "
                             "`confirmatory origins available` onward — not over "
                             "the full sample shown above.\n")

    def _dm_table(base_name: str, model_names) -> None:
        """One DM block. Adds power/equivalence columns under `corrected`."""
        base = scores[base_name]["qlike"]
        if corrected:
            lines.append(DM_HEADER_NOTE + "\n")
            lines.append("| model | DM | p | n | MDE | n_req | p_TOST | verdict |")
            lines.append("|---|---|---|---|---|---|---|---|")
        else:
            lines.append("| model | DM | p | n |")
            lines.append("|---|---|---|---|")
        starred = False
        for name in model_names:
            if name == base_name or name not in scores:
                continue
            joined = pd.concat([scores[name]["qlike"], base], axis=1,
                               keys=["a", "b"]).dropna()
            r = metrics.dm_test(joined["a"].values, joined["b"].values, h=1)
            if not corrected:
                lines.append(f"| {name} | {r['dm']:.3f} | {r['p']:.4f} | {r['n']} |")
                continue
            mde, tost, _ = _dm_extras(joined["a"], joined["b"])
            gap = float(joined["b"].mean() - joined["a"].mean())
            verdict = methodology.dm_verdict(r["p"], gap, tost)
            label = name
            if not specs.get(name, {}).get("confirmatory", True):
                label, starred = f"{name} *", True
            lines.append(
                f"| {label} | {r['dm']:.3f} | {r['p']:.4f} | {r['n']} | "
                f"{mde['mde']:.4f} | {mde['n_required']:,.0f} | "
                f"{tost['p_tost']:.3f} | {verdict} |")
        if starred:
            lines.append("\n`*` = exploratory specification (see the table above).")

    if "har" in scores:
        lines.append("\n## Diebold-Mariano vs HAR (QLIKE; negative = beats HAR)\n")
        _dm_table("har", list(scores))

    if "har_iv" in scores:
        lines.append("\n## Diebold-Mariano vs HAR-IV — same information set\n")
        lines.append("HAR-IV is log-HAR plus log(VXN). Any model fed VXN beats plain "
                     "HAR trivially, because implied vol predicts realized vol. The "
                     "question this table answers is whether the foundation model "
                     "extracts more from VXN than four OLS terms do.\n")
        _dm_table("har_iv", IV_FED)

    # ---- TiRex-2 strict-leakage robustness subwindow ----
    tirex_names = [n for n in scores if n.startswith("tirex")]
    if tirex_names and phase == "clean":
        cut = pd.Timestamp(cfg.get("tirex", {}).get("publication_date", "2026-07-01"))
        lines.append(f"\n## TiRex-2 robustness: origins after publication ({cut.date()})\n")
        lines.append("The date-based leakage rule applied literally to TiRex-2. Tiny "
                     "sample by construction — this is a sign check against the full "
                     "window above, not a test. See reports/LEAKAGE_TIREX2.md.\n")
        lines.append("| model | QLIKE (full) | QLIKE (post-pub) | n |")
        lines.append("|---|---|---|---|")
        for name in tirex_names + [n for n in ("har", "har_iv", "chronos_cov")
                                   if n in scores]:
            s = scores[name]
            sub = s[s.index >= cut]
            if sub.empty:
                continue
            lines.append(f"| {name} | {s['qlike'].mean():.4f} | "
                         f"{sub['qlike'].mean():.4f} | {len(sub)} |")

    # ---- event slices (descriptive at current sample sizes) ----
    lines.append("\n## Event-sliced QLIKE (mean)\n")
    ev = pd.DataFrame(index=df.index)
    ev["fomc"] = df["is_fomc"].shift(-1)
    ev["cpi"] = df["is_cpi"].shift(-1)
    thr = df["earnings_wt"].shift(-1)
    hq = _heavy_cutoff(cfg, thr)
    ev["heavy_earn"] = (thr >= hq).astype(float)
    lines.append("| model | FOMC (n) | CPI (n) | heavy-earnings (n) | quiet (n) |")
    lines.append("|---|---|---|---|---|")
    for name, s in scores.items():
        e = ev.reindex(s.index)
        quiet = (e[["fomc", "cpi", "heavy_earn"]].sum(axis=1) == 0)
        def _m(mask):
            v = s.loc[mask.fillna(False), "qlike"]
            return f"{v.mean():.4f} ({len(v)})" if len(v) else "-"
        lines.append(f"| {name} | {_m(e['fomc'] == 1)} | {_m(e['cpi'] == 1)} | "
                     f"{_m(e['heavy_earn'] == 1)} | {_m(quiet)} |")

    # ---- FULL-SAMPLE paired per-origin counts ----
    # A mean difference and a win rate answer different questions, and the two
    # can point opposite ways: a term can win on 61% of days and still deliver
    # zero mean improvement if its losses are rare and large. Any claim about a
    # specification is reported on both, always, at every sample size.
    FULL_PAIRS = [("har_iv_lev", "har_iv"), ("har_lev", "har"),
                  ("har_iv_x", "har_iv"), ("har_x", "har"),
                  ("har_sv", "har"), ("har_iv", "har"),
                  ("chronos_cov", "chronos_uni"), ("tirex_cov", "tirex_uni")]
    fp = [(a, b) for a, b in FULL_PAIRS if a in scores and b in scores]
    if fp:
        from scipy import stats as _st2
        lines.append("\n## Full-sample paired per-origin (all origins in phase)\n")
        lines.append("Win rate and mean difference answer different questions. A "
                     "high win rate with no mean gain is many small wins funded by "
                     "rare large losses; a low win rate with a mean gain is the "
                     "reverse. Read both before believing either.\n")
        lines.append("| pair | n | wins | win % | sign p | mean | median | "
                     "top-10 share of mean gap |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for a, b in fp:
            ia = scores[a].index.intersection(scores[b].index)
            if len(ia) < 30:
                continue
            da, db = scores[a].loc[ia, "qlike"], scores[b].loc[ia, "qlike"]
            n, w = len(ia), int((da < db).sum())
            p = float(_st2.binomtest(w, n, 0.5).pvalue)
            gap = db.mean() - da.mean()
            d = db - da
            top = d.nlargest(10).sum() / n
            share = f"{100 * top / gap:.0f}%" if abs(gap) > 1e-9 else "n/a"
            lines.append(f"| {a} vs {b} | {n} | {w} | {100*w/n:.1f}% | {p:.3g} | "
                         f"{da.mean():.4f} vs {db.mean():.4f} | "
                         f"{da.median():.4f} vs {db.median():.4f} | {share} |")

    # ---- replication across the two windows ----
    # A pair that wins here and loses there has not replicated, whatever its
    # p-value in either window. This is the check that decided the earnings
    # case, and it is the reason the corrected report exists.
    if corrected and fp:
        other = _other_phase(phase)
        lines.append(f"\n## Replication: {phase} vs {other}\n")
        lines.append(
            "`flip k` is how many origins must be dropped from the winning tail "
            "before the mean gap changes sign — a direct read on whether a "
            "result is a property of the sample or of a handful of days. A pair "
            f"that wins in {phase} and loses in {other} has not replicated, "
            "whatever its p-value in either window.\n")
        lines.append(f"| pair | {phase} win% | {phase} gap | flip k | "
                     f"{other} win% | {other} gap | {other} n | replicates? |")
        lines.append("|---|---|---|---|---|---|---|---|")
        cache: dict[str, pd.Series | None] = {}
        for a, b in fp:
            ia = scores[a].index.intersection(scores[b].index)
            if len(ia) < 30:
                continue
            here = methodology.paired_summary(scores[a].loc[ia, "qlike"],
                                              scores[b].loc[ia, "qlike"])
            for nm in (a, b):
                if nm not in cache:
                    cache[nm] = _qlike_series(cfg, df, nm, other, taus)
            oa, ob = cache[a], cache[b]
            flip = "never" if here["flip_k"] is None else here["flip_k"]
            if oa is None or ob is None:
                there = None
            else:
                io = oa.index.intersection(ob.index)
                there = (methodology.paired_summary(oa.loc[io], ob.loc[io])
                         if len(io) >= 30 else None)
            if there is None:
                cols, verdict = "— | — | —", "not testable"
            else:
                cols = (f"{there['win_pct']:.1f}% | {there['gap']:+.4f} | "
                        f"{there['n']}")
                verdict = ("yes" if np.sign(here["gap"]) == np.sign(there["gap"])
                           else "**no**")
            lines.append(f"| {a} vs {b} | {here['win_pct']:.1f}% | "
                         f"{here['gap']:+.4f} | {flip} | {cols} | {verdict} |")

    # ---- paired sign test on the heavy-earnings slice ----
    # A mean can be carried by one blown-up day at this sample size, so the
    # report must never show the slice mean without the per-origin count.
    pairs = [("har_x", "har"), ("har_iv_x", "har_iv"),
             ("chronos_cov", "chronos_uni"), ("tirex_cov", "tirex_uni")]
    pairs = [(a, b) for a, b in pairs if a in scores and b in scores]
    if pairs:
        from scipy import stats as _st
        heavy_idx = ev.index[(ev["heavy_earn"] == 1).fillna(False)]
        lines.append(f"\n### Heavy-earnings slice, paired per-origin "
                     f"(cutoff {hq:.1f}% of index weight)\n")
        lines.append("| pair | mean | median | better/n | sign p | top day | "
                     "% of gap | ex-top |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for a, b in pairs:
            ia = scores[a].index.intersection(scores[b].index).intersection(heavy_idx)
            if len(ia) < 3:
                continue
            da, db = scores[a].loc[ia, "qlike"], scores[b].loc[ia, "qlike"]
            n, better = len(ia), int((da < db).sum())
            p = float(_st.binomtest(better, n, 0.5).pvalue)
            gap = db.mean() - da.mean()
            top = (db - da).idxmax()
            share = (100 * (db - da).max() / n / gap) if gap else float("nan")
            keep = ia.drop(top)
            lines.append(
                f"| {a} vs {b} | {da.mean():.4f} vs {db.mean():.4f} | "
                f"{da.median():.4f} vs {db.median():.4f} | {better}/{n} | "
                f"{p:.4f} | {top.date()} | {share:.0f}% | "
                f"{int((da.loc[keep] < db.loc[keep]).sum())}/{len(keep)} |")

    # ---- pre-committed gate: is it reachable at its own trigger? ----
    # config.yaml registers a confirmatory earnings test behind
    # `min_heavy_earnings_days`, and NO CODE read that field -- so nothing ever
    # checked whether the trigger that opens the gate can supply the sample the
    # test requires. It cannot. Printing the arithmetic beside the gate is the
    # cheapest possible guard against the repository's signature defect: a
    # registered test that cannot deliver its verdict.
    ne = cfg.get("next_evaluation") or {}
    slice_gate = (ne.get("earnings_slice_confirmatory") or {}).get(
        "min_heavy_earnings_days")
    if corrected and slice_gate:
        heavy_all = (df["earnings_wt"].shift(-1) >= _heavy_cutoff(
            cfg, df["earnings_wt"].shift(-1))).astype(float)
        rate = float(heavy_all.mean())
        have = int(ev.loc[ev.index.intersection(
            next(iter(scores.values())).index), "heavy_earn"].fillna(0).sum())
        need = int(slice_gate)
        at_origins = int(ne.get("at_origins", 0) or 0)
        proj = rate * at_origins
        lines.append("\n### Pre-committed gate: reachability\n")
        lines.append(
            f"`next_evaluation.earnings_slice_confirmatory` requires "
            f"**{need} heavy-earnings days** before the registered confirmatory "
            f"DM test may run. This phase has **{have}**. Heavy-earnings days "
            f"arrive at {100 * rate:.2f}% of origins over the full sample, so "
            f"the `at_origins: {at_origins}` trigger projects to "
            f"**~{proj:.0f}** — a shortfall of ~{max(0, need - proj):.0f}. "
            f"Reaching {need} takes roughly **{need / rate:,.0f} origins** at "
            f"the observed rate.\n")
        if proj < need:
            lines.append(
                f"**The gate is therefore not reachable at its own trigger.** "
                f"Over every rolling window in this sample the maximum count "
                f"attained in {at_origins} sessions is well below {need}. This "
                f"is disclosed rather than repaired: `config.yaml` is frozen, "
                f"so the floor stands and the shortfall is published beside it. "
                f"See `reports/AMENDMENTS.md`.\n")

    # ---- 30-calendar-day horizon vs VXN ----
    if "vxn_exp_var_30c" in df.columns:
        lines.append("\n## 30-calendar-day horizon vs VXN (log variance)\n")
        y = np.log(df["fwd_var_30c"].clip(lower=1e-12))
        iv = np.log(df["vxn_exp_var_30c"].clip(lower=1e-12))
        # Same window as every other row in this report — an unbounded mask here
        # would score the diagnostic-phase VXN benchmark on clean-phase data too.
        lo, hi = _phase_bounds(cfg, phase)
        mask = (df.index >= lo) & (df.index <= hi)
        r = metrics.mz_regression(y[mask], iv[mask])
        iv_med = float(iv[mask].median())
        vxn_med = float(df.loc[mask, "vxn"].median())

        if corrected:
            mzc = methodology.mz_corrected(y[mask], iv[mask], overlap=OVERLAP_30D,
                                           n_boot=N_BOOT, seed=BOOT_SEED)
            prem = methodology.premium_ci(y[mask], iv[mask], df.loc[mask, "vxn"],
                                          overlap=OVERLAP_30D, n_boot=N_BOOT,
                                          seed=BOOT_SEED)
            n_eff = mzc["n_eff"]
            lag_ratio = r["maxlags"] / max(r["n"], 1)
            # Quote all three standard errors rather than a ratio: "HAC
            # understates by 3x" is true of the non-overlapping refits and not
            # of the bootstrap, and a single ratio invites reading one as the
            # other. Measured per report, so it cannot go stale.
            se_txt = f"se(beta)={r['se_beta']:.3f} against {mzc['boot']['se_boot']['f']:.3f}"
            if mzc["phases"].get("n_phases"):
                se_txt += (f" from the bootstrap and a "
                           f"{mzc['phases']['beta_sd_across_phases']['f']:.3f} "
                           f"spread of the point estimate across the "
                           f"{mzc['phases']['n_phases']} starting offsets")
            else:
                se_txt += " from the bootstrap"
            se_txt += (". The bootstrap interval is itself too narrow — see the "
                       "calibration note below")
            lines.append(
                f"Origins are daily but each target spans {OVERLAP_30D} trading "
                f"days, so consecutive rows share almost all of their target. "
                f"**n = {r['n']}, but n_eff = {n_eff:.0f} independent windows.** "
                f"Standard errors below come from a circular moving-block "
                f"bootstrap (block = {OVERLAP_30D}, {N_BOOT} reps) and from "
                f"refitting on all {OVERLAP_30D} non-overlapping subsamples — not "
                f"from HAC({r['maxlags']}), which at this n has a lag/n ratio of "
                f"{lag_ratio:.2f} and returns {se_txt}. config.yaml's own "
                f"`carry_study` block already requires non-overlapping inference "
                f"for exactly this reason.\n")
            lines.append(f"- VXN MZ: alpha={r['alpha']:.3f}, beta={r['beta']:.3f}, "
                         f"R2={r['r2']:.3f}, n={r['n']}, n_eff={n_eff:.0f}")
            b = mzc["boot"]
            lines.append(f"  - bootstrap se(beta)={b['se_boot']['f']:.3f}, 95% CI "
                         f"[{b['ci_lo']['f']:.3f}, {b['ci_hi']['f']:.3f}], "
                         f"p[beta=1]={b['p_one']['f']:.3f}")
            ph = mzc["phases"]
            if ph.get("n_phases"):
                lines.append(
                    f"  - across the {ph['n_phases']} non-overlapping subsamples "
                    f"(~{ph['n_per_phase']:.0f} obs each): beta ranges "
                    f"[{ph['beta_min']['f']:.3f}, {ph['beta_max']['f']:.3f}], "
                    f"sd across starting offsets="
                    f"{ph['beta_sd_across_phases']['f']:.3f}; median "
                    f"within-subsample se="
                    f"{ph['se_within_subsample']['f']:.3f} (the se of a beta "
                    f"fitted to ~{ph['n_per_phase']:.0f} points, **not** of the "
                    f"full-sample beta — it was previously published as "
                    f"\"honest se\", which overstated the reported "
                    f"coefficient's uncertainty)")
            ci = prem.get("ci")
            if ci:
                # prem's own median, not the phase-wide one: the premium is
                # evaluated on rows with a realized 30-day target, which is a
                # slightly shorter sample, and labelling it with a median taken
                # from a different sample is how a number stops matching its
                # caption.
                lines.append(
                    f"  - variance risk premium at the window's median VXN "
                    f"({prem['vxn_median']:.1f}): **{prem['point']:.1f} vol points, 95% CI "
                    f"[{ci[0]:.1f}, {ci[1]:.1f}]**. The frozen report prints this "
                    f"as a single number; at n_eff={n_eff:.0f} it is an interval "
                    f"or it is nothing.")
        else:
            lines.append(f"- VXN MZ: alpha={r['alpha']:.3f}, beta={r['beta']:.3f} "
                         f"(p[beta=1]={r['p_beta1']:.3f}), R2={r['r2']:.3f}, n={r['n']}")
            # alpha alone is not the premium unless beta == 1. Evaluate the fitted
            # line at a representative implied level instead.
            ratio = float(np.exp(r["alpha"] + (r["beta"] - 1.0) * iv_med))
            lines.append(f"  - At the window's median VXN ({vxn_med:.1f}), fitted realized "
                         f"variance is {ratio:.0%} of implied — {np.sqrt(ratio):.0%} in vol "
                         f"terms, about {vxn_med * (1 - np.sqrt(ratio)):.1f} vol points of "
                         f"premium. Read the premium here, not off alpha: with beta="
                         f"{r['beta']:.2f} the intercept is not interpretable alone.")
        lines.append("\nEncompassing: realized = a + b*VXN + c*model. H3 wants c>0 and "
                     "significant. A negative c is not evidence for the model — with "
                     "VXN already in the regression it means the forecast enters "
                     "against realized variance, which collinear forecasts commonly "
                     "do. Models that consume VXN as an input are excluded: regressing "
                     "on VXN and on a function of VXN is collinear by construction and "
                     "the split of the coefficients is not interpretable.\n")
        for name in ("har_cum", "persistence_cum",
                     "chronos_uni", "chronos_cov",
                     "tirex_uni", "tirex_cov"):
            fc = _load_forecast(cfg, name, phase)
            if fc is None or "log_cum_var_hat" not in fc.columns:
                continue
            m = fc["log_cum_var_hat"].dropna()
            if m.empty:
                continue
            r1 = metrics.mz_regression(y, m)
            r2 = metrics.encompassing(y, iv, m)
            if not corrected:
                lines.append(f"- {name}: MZ beta={r1['beta']:.3f} R2={r1['r2']:.3f} | "
                             f"encompassing b_implied={r2['b_implied']:.3f} "
                             f"c_model={r2['c_model']:.3f} "
                             f"(p={r2['p_model']:.4f}, R2={r2['r2']:.3f}, n={r2['n']})")
                continue
            ec = methodology.encompassing_corrected(y, iv, m, overlap=OVERLAP_30D,
                                                    n_boot=N_BOOT, seed=BOOT_SEED)
            eb, eph = ec["boot"], ec["phases"]
            rng = ""
            if eph.get("n_phases"):
                rng = (f", across-subsample c range [{eph['beta_min']['m']:.3f}, "
                       f"{eph['beta_max']['m']:.3f}]")
            lines.append(f"- {name}: MZ beta={r1['beta']:.3f} R2={r1['r2']:.3f} | "
                         f"encompassing b_implied={r2['b_implied']:.3f} "
                         f"c_model={r2['c_model']:.3f} "
                         f"(bootstrap p={eb['p_zero']['m']:.3f}, 95% CI "
                         f"[{eb['ci_lo']['m']:.3f}, {eb['ci_hi']['m']:.3f}], "
                         f"n={ec['n']}, n_eff={ec['n_eff']:.0f}{rng})")

        if corrected:
            n_eff = methodology.effective_n(int(r["n"]), OVERLAP_30D)
            if n_eff < 30:
                # The window that motivated the whole correction. Say so.
                lines.append(
                    f"\nAt n_eff={n_eff:.0f} this section cannot reject anything, "
                    f"and a non-significant `c_model` here is not evidence that "
                    f"the model adds nothing beyond VXN — it is evidence that the "
                    f"window is too short to tell. Read the CIs, not the p-values.")
            else:
                # n_eff is respectable here, and saying otherwise would be its own
                # overstatement: the diagnostic window DOES resolve some of these.
                # But a bootstrap p just under 0.05 is still not a 5% rejection --
                # the measured size of this test is ~10-16%, so the earlier
                # "a significant c_model above is a real rejection" was wrong.
                lines.append(
                    f"\nn_eff={n_eff:.0f} independent windows can resolve moderate "
                    f"effects, unlike the clean window. But these are not tests on "
                    f"{int(r['n'])} observations: every interval here is built on "
                    f"{n_eff:.0f}, and the frozen report's n is {OVERLAP_30D}x "
                    f"that. **And the bootstrap is anti-conservative** — measured "
                    f"nominal-95% coverage is ~82-85% and the true size of the "
                    f"`c_model` p-value is ~10-16%, not 5% "
                    f"(`tests/test_methodology.py::TestBootstrapCalibration`). A "
                    f"bootstrap p just under 0.05 here is not a 5% rejection; "
                    f"treat these intervals as a floor on the uncertainty and "
                    f"read the across-subsample ranges beside them.")

    # ---- registry gaps, stated where they are actionable ----
    if corrected:
        missing = [n for n in scores
                   if specs[n]["confirmatory_from"] is None and specs[n]["known"]]
        if missing:
            lines.append("\n---\n")
            lines.append("**Specification dates missing from the registry:** "
                         + ", ".join(f"`{m}`" for m in missing)
                         + ". These are treated as exploratory until "
                           "`spec_registry.yaml` records when they were written.")

    sfx = cfg.get("_grid_suffix", "") + cfg.get("_scenario_suffix", "")
    out = cfg["paths"]["reports"] / f"results_{phase}{sfx}.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["features", "baselines", "chronos", "tirex",
                                    "evaluate"])
    ap.add_argument("--phase", default="clean", choices=["diagnostic", "clean", "all"])
    ap.add_argument("--variant", default="uni",
                    choices=["uni", "cov", "cov_iv", "cov_ivf"])
    ap.add_argument("--source", default="daily", choices=["daily", "bars"])
    ap.add_argument("--quantile-grid", default="preregistered",
                    choices=["preregistered", "deciles"],
                    help="deciles = TiRex-2's native 0.1..0.9; suffixes outputs "
                         "with _dec and never mutates config.yaml")
    ap.add_argument("--estimator", default="trunc", choices=["trunc", "smearing"],
                    help="point forecast behind QLIKE. trunc is the frozen "
                         "pre-registered estimator; smearing is the correction "
                         "and writes forecasts to *_sm.parquet")
    ap.add_argument("--inference", default="naive", choices=["naive", "corrected"],
                    help="corrected adds power/equivalence verdicts, the "
                         "specification gate, and overlap-aware standard errors")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = config.load(a.config)
    cfg["_grid_suffix"] = apply_grid(cfg, a.quantile_grid)
    cfg["_scenario_suffix"] = apply_scenario(cfg, a.estimator, a.inference)
    if a.cmd == "features":
        features.assemble(cfg, source=a.source)
    elif a.cmd == "baselines":
        cmd_baselines(cfg, a.phase, a.source)
    elif a.cmd == "chronos":
        cmd_chronos(cfg, a.phase, a.variant, a.source)
    elif a.cmd == "tirex":
        cmd_tirex(cfg, a.phase, a.variant, a.source)
    elif a.cmd == "evaluate":
        cmd_evaluate(cfg, a.phase, a.source)


if __name__ == "__main__":
    main()
