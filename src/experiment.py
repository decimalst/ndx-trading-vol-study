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
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from . import config, features, metrics, models


DECILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def apply_grid(cfg: dict, grid: str) -> str:
    """Set cfg['quantiles'] for the run and return the output-file suffix."""
    if grid == "deciles":
        cfg["quantiles"] = list(DECILES)
        return "_dec"
    return ""


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


def _load_forecast(cfg: dict, name: str, phase: str) -> pd.DataFrame | None:
    """Forecasts for `name` in `phase`, falling back to a `_all` run.

    Every model refits per origin from history up to that origin, so an origin's
    forecast does not depend on which origins were requested alongside it —
    slicing an `all` run by date is identical to having run that phase directly.
    This is what makes the documented `baselines PHASE=all` then
    `evaluate PHASE=clean` order work.
    """
    fdir = cfg["paths"]["forecasts"]
    sfx = cfg.get("_grid_suffix", "")
    p = fdir / f"{name}_{phase}{sfx}.parquet"
    if not p.exists():
        p = fdir / f"{name}_all{sfx}.parquet"
        if not p.exists():
            return None
    fc = pd.read_parquet(p)
    lo, hi = _phase_bounds(cfg, phase)
    return fc[(fc.index >= lo) & (fc.index <= hi)]


def cmd_baselines(cfg: dict, phase: str, source: str) -> None:
    df = _load_master(cfg, source)
    origins = _phase_origins(df, cfg, phase)
    runs = {
        "persistence": lambda: models.run_persistence(df, origins, cfg),
        "ewma": lambda: models.run_ewma(df, origins, cfg),
        "har": lambda: models.run_har(df, origins, cfg, with_x=False),
        "har_x": lambda: models.run_har(df, origins, cfg, with_x=True),
        # Control for the VXN-fed models. No `_cum` counterpart on purpose: the
        # encompassing regression already conditions on VXN, so a HAR-IV
        # cumulative forecast would be collinear with the regressor by design.
        "har_iv": lambda: models.run_har(df, origins, cfg, with_x=False, with_iv=True),
        # Signed semivariance (Patton-Sheppard): swaps the daily term for its
        # signed halves. No new data — hourly bars only supply the split share.
        "har_sv": lambda: models.run_har(df, origins, cfg, with_sv=True),
        # HAR-IV plus the implied-correlation decomposition.
        "har_ic": lambda: models.run_har(df, origins, cfg, with_iv=True, with_ic=True),
        # HAR-IV-X: log-HAR + log(VXN) + event terms. Pre-specified as the ONLY
        # new specification after the clean-phase results were seen, on this
        # reasoning: VXN is a 30-day constant-maturity measure and structurally
        # cannot know that a 9%-weight name prints tonight, yet har_x and
        # chronos_cov both beat har_iv on heavy-earnings days while losing to it
        # everywhere else. A significant earnings coefficient ALONGSIDE VXN is
        # H3-flavoured evidence at h=1: information the surface does not price.
        "har_iv_x": lambda: models.run_har(df, origins, cfg, with_x=True, with_iv=True),
        # LHAR: return asymmetry from daily data (1999-), the mechanism under
        # signed semivariance, on a sample big enough to settle it. DIAGNOSTIC
        # ONLY — see DIAGNOSTIC_ONLY below; it must not spend a clean-window draw.
        "har_lev": lambda: models.run_har(df, origins, cfg, with_lev=True),
        # Ledger-closing test: does return asymmetry survive alongside VXN?
        # VXN embeds skew, which is the same economics, so leverage may just be
        # VXN by another route. DIAGNOSTIC ONLY. Either answer closes the map:
        # subsumed -> no open items; not subsumed -> the one non-market signal
        # that survived everything, documented and not acted on.
        "har_iv_lev": lambda: models.run_har(df, origins, cfg, with_iv=True,
                                             with_lev=True),
    }
    sfx = cfg.get("_grid_suffix", "")
    for name, fn in runs.items():
        out = fn()
        path = cfg["paths"]["forecasts"] / f"{name}_{phase}{sfx}.parquet"
        out.to_parquet(path)
        print(f"{name}: {len(out)} forecasts -> {path}")
    for name in ("har", "persistence"):
        cum = models.predict_cum(df, origins, cfg, model=name)
        path = cfg["paths"]["forecasts"] / f"{name}_cum_{phase}{sfx}.parquet"
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


def cmd_evaluate(cfg: dict, phase: str, source: str) -> None:
    df = _load_master(cfg, source)
    taus = np.asarray(cfg["quantiles"])
    names = ["persistence", "ewma", "har", "har_x", "har_iv", "har_sv", "har_ic", "har_iv_x", "har_lev", "har_iv_lev",
             "chronos_uni", "chronos_cov", "chronos_cov_iv",
             "tirex_uni", "tirex_cov", "tirex_cov_iv", "tirex_cov_ivf"]
    # Models handed VXN must be judged against the VXN-fed linear control, not
    # against plain HAR — beating HAR only shows that implied vol predicts
    # realized vol, which has been known for decades.
    # Specifications admitted for diagnostic-window testing during the freeze.
    # Suppressed from clean-phase reports so testing them cannot become a peek;
    # they enter the clean window only at the pre-committed gate.
    DIAGNOSTIC_ONLY = ("har_lev", "har_iv_lev")
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
    lines.append("## h=1 losses (mean per day)\n")
    lines.append(f"| model | n | QLIKE | CRPS | pin{taus[0]:.2f} | pin{taus[-1]:.2f} "
                 f"| {nominal:.0%} cov | p_uc | p_ind |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name in names:
        if phase != "diagnostic" and name in DIAGNOSTIC_ONLY:
            continue
        fc = _load_forecast(cfg, name, phase)
        if fc is None or fc.empty:
            continue
        s = _h1_scores(df, fc, taus)
        if s.empty:
            continue
        scores[name] = s
        if "_y" in s:
            cov = metrics.coverage_tests(s["_y"].values, s["_lo"].values,
                                         s["_hi"].values, nominal=nominal)
            cov_str = f"{cov['coverage']:.3f} | {cov['p_uc']:.3f} | {cov['p_ind']:.3f}"
            extra = (f"{s['crps'].mean():.4f} | {s['pin_lo'].mean():.4f} | "
                     f"{s['pin_hi'].mean():.4f}")
        else:
            cov_str, extra = "- | - | -", "- | - | -"
        lines.append(f"| {name} | {len(s)} | {s['qlike'].mean():.4f} | {extra} | {cov_str} |")

    if "har" in scores:
        lines.append("\n## Diebold-Mariano vs HAR (QLIKE; negative = beats HAR)\n")
        lines.append("| model | DM | p | n |")
        lines.append("|---|---|---|---|")
        base = scores["har"]["qlike"]
        for name, s in scores.items():
            if name == "har":
                continue
            joined = pd.concat([s["qlike"], base], axis=1, keys=["a", "b"]).dropna()
            r = metrics.dm_test(joined["a"].values, joined["b"].values, h=1)
            lines.append(f"| {name} | {r['dm']:.3f} | {r['p']:.4f} | {r['n']} |")

    if "har_iv" in scores:
        lines.append("\n## Diebold-Mariano vs HAR-IV — same information set\n")
        lines.append("HAR-IV is log-HAR plus log(VXN). Any model fed VXN beats plain "
                     "HAR trivially, because implied vol predicts realized vol. The "
                     "question this table answers is whether the foundation model "
                     "extracts more from VXN than four OLS terms do.\n")
        lines.append("| model | DM | p | n |")
        lines.append("|---|---|---|---|")
        base_iv = scores["har_iv"]["qlike"]
        for name in IV_FED:
            if name not in scores:
                continue
            joined = pd.concat([scores[name]["qlike"], base_iv], axis=1,
                               keys=["a", "b"]).dropna()
            r = metrics.dm_test(joined["a"].values, joined["b"].values, h=1)
            lines.append(f"| {name} | {r['dm']:.3f} | {r['p']:.4f} | {r['n']} |")

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
        lines.append(f"- VXN MZ: alpha={r['alpha']:.3f}, beta={r['beta']:.3f} "
                     f"(p[beta=1]={r['p_beta1']:.3f}), R2={r['r2']:.3f}, n={r['n']}")
        # alpha alone is not the premium unless beta == 1. Evaluate the fitted
        # line at a representative implied level instead.
        iv_med = float(iv[mask].median())
        vxn_med = float(df.loc[mask, "vxn"].median())
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
            lines.append(f"- {name}: MZ beta={r1['beta']:.3f} R2={r1['r2']:.3f} | "
                         f"encompassing b_implied={r2['b_implied']:.3f} "
                         f"c_model={r2['c_model']:.3f} "
                         f"(p={r2['p_model']:.4f}, R2={r2['r2']:.3f}, n={r2['n']})")

    out = cfg["paths"]["reports"] / f"results_{phase}{cfg.get('_grid_suffix', '')}.md"
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
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = config.load(a.config)
    cfg["_grid_suffix"] = apply_grid(cfg, a.quantile_grid)
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
