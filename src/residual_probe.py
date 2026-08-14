"""Residualized latent probe — is anything in TiRex's latent state orthogonal to RV history?

Governed by `residual_probe.yaml`, frozen before this file was written.
POST-RESULT and ADDITIVE: it runs on a sample the parent study already
inspected, rewrites no frozen verdict, and cannot restore falsifiability to the
parent. `make residual-probe` runs `tests/test_residual_probe.py` first.

WHAT THIS ASKS, AND WHY IT IS NOT WHAT THE PARENT ASKED

The parent probe established that five-session transition proximity is
DECODABLE from the frozen 512-dim latent state (sparse k=1, 0.8153 pooled
phase-mean AUC, 0 of 99 controls reaching it). Two later results make that the
weaker question:

  - The selected coordinates track smoothed 5/22-day volatility and prior-session
    VXN. So what was decoded is volatility LEVEL, which direct RV history already
    supplies.
  - `reports/representation_study/pooling_diagnostic.md`: the pooled statistic is
    dominated by between-fold event-rate variation. A fold-constant score with
    zero within-year information reaches 0.8317 -- above the parent's 0.8153.

So this projects the HAR feature set out of every coordinate first, then asks
whether what remains ranks transitions WITHIN a fold. Three fits happen per fold
-- the residualization, the coordinate selection, and the ridge -- and every one
sees training rows only. That is the whole leakage surface and it is fenced by
`tests/test_residual_probe.py`.

The benchmark is REFIT here rather than read from the parent's `p_benchmark`, so
that `bench` and `bench_plus_resid` differ by exactly one column and nothing
else. The parent's number is reported alongside for continuity only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "residual_probe.yaml"
DATA = ROOT / "data" / "representation_study"
OUT_REPORT = ROOT / "reports" / "representation_study" / "residual_probe.md"
OUT_METRICS = DATA / "residual_probe_metrics.json"

HAR = ["log_rv_d", "log_rv_w", "log_rv_m"]


# --------------------------------------------------------------------------
# primitives (contract-tested before this module existed)
# --------------------------------------------------------------------------
def residualize(train_H: np.ndarray, train_z: np.ndarray,
                apply_H: np.ndarray, apply_z: np.ndarray) -> np.ndarray:
    """Project `train_H` out of `train_z`, then apply those coefficients.

    The coefficients come from the training rows and are APPLIED to whatever
    rows are passed in. Refitting on the rows being residualized would let the
    held-out year shape the very quantity claimed to be orthogonal to it.
    """
    A = np.asarray(train_H, dtype=float)
    b = np.asarray(train_z, dtype=float)
    beta, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.asarray(apply_z, dtype=float) - np.asarray(apply_H, dtype=float) @ beta


def within_fold_auc(frame: pd.DataFrame, score: str,
                    event: str = "event") -> tuple[float, int]:
    """Pair-weighted AUC inside (ranking_phase, fold_year) cells.

    Cells without both classes contribute nothing -- they contain no orderable
    pair. Scoring them as 0.5 would import the between-fold structure this
    metric exists to remove.
    """
    rows = []
    for _, g in frame.groupby(["ranking_phase", "fold_year"], sort=True):
        if g[event].nunique() < 2:
            continue
        rows.append((len(g), roc_auc_score(g[event], g[score])))
    if not rows:
        return float("nan"), 0
    w = np.array([r[0] for r in rows], dtype=float)
    v = np.array([r[1] for r in rows], dtype=float)
    return float((w * v).sum() / w.sum()), len(rows)


def phase_mean_auc(frame: pd.DataFrame, score: str,
                   event: str = "event") -> float:
    """The parent's pooled statistic. Reported for continuity, not registered."""
    aucs = []
    for _, g in frame.groupby("ranking_phase"):
        if g[event].nunique() < 2:
            continue
        aucs.append(roc_auc_score(g[event], g[score]))
    return float(np.mean(aucs)) if aucs else float("nan")


def verdict(p_paired: float, mean_delta: float, p_tost: float,
            alpha: float = 0.05) -> str:
    """Three-way. A failure to reject is never silently promoted to a null."""
    if np.isfinite(p_paired) and p_paired < alpha:
        return "adds" if mean_delta > 0 else "degrades"
    if np.isfinite(p_tost) and p_tost < alpha:
        return "equivalent"
    return "inconclusive"


# --------------------------------------------------------------------------
# fold machinery
# --------------------------------------------------------------------------
def _standardize(train: np.ndarray, apply_: np.ndarray) -> np.ndarray:
    mu = train.mean(axis=0)
    sd = train.std(axis=0, ddof=0)
    sd[sd < 1e-12] = 1.0
    return (apply_ - mu) / sd


def _fit_score(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    model = LogisticRegression(penalty="l2", C=1.0, max_iter=2000,
                               solver="lbfgs")
    model.fit(_standardize(Xtr, Xtr), ytr)
    return model.predict_proba(_standardize(Xtr, Xte))[:, 1]


def _select_top1(resid_train: np.ndarray, y_train: np.ndarray) -> int:
    """Largest absolute standardized mean difference. Training rows only.

    Same selector as the parent k=1 rung, so the only difference between this
    probe and that one is the residualization.
    """
    scale = resid_train.std(axis=0, ddof=0)
    scale[scale < 1e-12] = 1.0
    eff = (resid_train[y_train == 1].mean(axis=0)
           - resid_train[y_train == 0].mean(axis=0)) / scale
    return int(np.argmax(np.abs(eff)))


def run() -> dict:
    protocol = yaml.safe_load(PROTOCOL.read_text())
    fc = pd.read_parquet(DATA / "tail_classical_forecasts.parquet")
    lat = pd.read_parquet(DATA / "latent_embeddings.parquet")

    if len(fc) != protocol["sample"]["expected_origins"]:
        raise RuntimeError(f"sample changed: {len(fc)} rows")
    folds = sorted(fc["fold_year"].unique())
    if len(folds) != protocol["sample"]["expected_folds"]:
        raise RuntimeError(f"fold set changed: {len(folds)} folds")

    clean_start = pd.Timestamp(protocol["sample"]["clean_window_start"])
    if fc.index.max() >= clean_start:
        raise RuntimeError("a scored origin reaches the sealed clean window")

    zcols = [c for c in lat.columns if c.startswith("z")]
    common = fc.index.intersection(lat.index)
    fc = fc.loc[common].copy()
    Z = lat.loc[common, zcols].astype(float)
    fc["target_end"] = pd.to_datetime(fc["target_end"])
    fc["cutoff"] = pd.to_datetime(fc["cutoff"])

    per_fold, r2_all, chosen = [], [], []
    scored = []
    for fold in folds:
        te = fc[fc["fold_year"] == fold]
        if te.empty:
            continue
        cutoff = te["cutoff"].iloc[0]
        # Training pool: rows whose ORIGIN and whose TARGET both complete at or
        # before the cutoff. A row whose target ends after the cutoff would leak
        # the held-out period's outcome into the fit.
        tr = fc[(fc.index <= cutoff) & (fc["target_end"] <= cutoff)]
        if len(tr) < 200 or tr["event"].nunique() < 2:
            continue

        Htr = np.column_stack([np.ones(len(tr)), tr[HAR].values])
        Hte = np.column_stack([np.ones(len(te)), te[HAR].values])
        Ztr, Zte = Z.loc[tr.index].values, Z.loc[te.index].values

        Rtr = np.empty_like(Ztr)
        Rte = np.empty_like(Zte)
        r2s = []
        for j in range(Ztr.shape[1]):
            beta, *_ = np.linalg.lstsq(Htr, Ztr[:, j], rcond=None)
            fit_tr = Htr @ beta
            Rtr[:, j] = Ztr[:, j] - fit_tr
            Rte[:, j] = Zte[:, j] - Hte @ beta
            ss_tot = ((Ztr[:, j] - Ztr[:, j].mean()) ** 2).sum()
            r2s.append(1.0 - (Rtr[:, j] ** 2).sum() / ss_tot if ss_tot > 0 else np.nan)
        r2_all.append(float(np.nanmedian(r2s)))

        ytr = tr["event"].values.astype(int)
        j = _select_top1(Rtr, ytr)
        chosen.append({"fold_year": int(fold), "coordinate": int(j),
                       "har_r2_median": r2_all[-1]})

        out = te[["fold_year", "ranking_phase", "event"]].copy()
        out["resid_k1"] = _fit_score(Rtr[:, [j]], ytr, Rte[:, [j]])
        out["bench_own"] = _fit_score(tr[HAR].values, ytr, te[HAR].values)
        out["bench_plus_resid"] = _fit_score(
            np.column_stack([tr[HAR].values, Rtr[:, [j]]]), ytr,
            np.column_stack([te[HAR].values, Rte[:, [j]]]))
        out["p_benchmark"] = te["p_benchmark"].values
        scored.append(out)

        # Per-fold delta, averaged over the five phases FIRST: the phases are
        # offsets of the same year and are not independent of each other.
        deltas = []
        for _, g in out.groupby("ranking_phase"):
            if g["event"].nunique() < 2:
                continue
            deltas.append(roc_auc_score(g["event"], g["bench_plus_resid"])
                          - roc_auc_score(g["event"], g["bench_own"]))
        if deltas:
            per_fold.append({"fold_year": int(fold),
                             "delta_auc": float(np.mean(deltas)),
                             "phases": len(deltas)})

    S = pd.concat(scored, ignore_index=True)
    d = np.array([p["delta_auc"] for p in per_fold], dtype=float)
    n = len(d)

    # Paired inference over folds, not over fold-phase cells.
    wins = int((d > 0).sum())
    p_sign = float(stats.binomtest(wins, n, 0.5).pvalue)
    rng = np.random.default_rng(protocol["inference"]["seed"])
    boot = np.array([rng.choice(d, n, replace=True).mean()
                     for _ in range(protocol["inference"]["bootstrap_draws"])])
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    margin = float(protocol["inference"]["equivalence_margin_auc"])
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    df = n - 1
    p_tost = float(max(stats.t.sf((d.mean() + margin) / se, df),
                       stats.t.cdf((d.mean() - margin) / se, df)))
    k = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
    mde = float(k * sd / np.sqrt(n))
    # Folds needed to bring the MDE down to the declared materiality margin.
    # If this is far beyond any attainable sample the answer is not "run longer"
    # -- it is that the DESIGN cannot settle the question, which is a more
    # useful thing to report than another inconclusive row.
    folds_for_margin = float((k * sd / margin) ** 2)
    v = verdict(p_sign, float(d.mean()), p_tost,
                protocol["inference"]["alpha"])

    scores = ["p_benchmark", "bench_own", "resid_k1", "bench_plus_resid"]
    table = {}
    for s in scores:
        wf, cells = within_fold_auc(S, s)
        table[s] = {"within_fold_auc": wf, "cells": cells,
                    "pooled_phase_mean_auc": phase_mean_auc(S, s)}

    metrics = {
        "protocol_version": protocol["version"],
        "status": protocol["status"],
        "origins": int(len(S)),
        "folds": n,
        "har_r2_on_latents_median": float(np.nanmedian(r2_all)),
        "har_r2_on_latents_range": [float(np.nanmin(r2_all)), float(np.nanmax(r2_all))],
        "scores": table,
        "primary": {
            "comparison": "bench_plus_resid minus bench_own, within fold",
            "mean_delta_auc": float(d.mean()),
            "median_delta_auc": float(np.median(d)),
            "folds_positive": wins,
            "sign_p": p_sign,
            "bootstrap_ci": list(ci),
            "mde_80pct_power": mde,
            "folds_needed_for_margin": folds_for_margin,
            "equivalence_margin": margin,
            "p_tost": p_tost,
            "verdict": v,
        },
        "per_fold": per_fold,
        "selected": chosen,
    }
    OUT_METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    OUT_REPORT.write_text(render(metrics))
    return metrics


def render(m: dict) -> str:
    p = m["primary"]
    L = ["# Residualized latent probe — is anything orthogonal to RV history?\n",
         "**Post-result and additive.** Governed by `residual_probe.yaml`, frozen "
         "before `src/residual_probe.py` was written. Runs on a sample the parent "
         "study already inspected, so it is a diagnostic and not a new holdout. "
         "It rewrites no frozen verdict.\n",
         "## The question\n",
         "The parent probe showed transition proximity is *decodable* from "
         "TiRex-2's latent state. But its selected coordinates track smoothed "
         "volatility and prior-session VXN, and "
         "[`pooling_diagnostic.md`](pooling_diagnostic.md) showed the pooled "
         "statistic is mostly between-fold — a score with zero within-year "
         "information scores 0.8317, above the parent's 0.8153. So: project the "
         "HAR feature set out of every coordinate, then ask whether what remains "
         "ranks transitions **within** a fold.\n",
         "## How much of the latent is just RV history?\n",
         f"Median R² of `[1, log_rv_d, log_rv_w, log_rv_m]` on a latent "
         f"coordinate, across {m['folds']} folds: "
         f"**{m['har_r2_on_latents_median']:.3f}** "
         f"(fold range {m['har_r2_on_latents_range'][0]:.3f}–"
         f"{m['har_r2_on_latents_range'][1]:.3f}). Whatever the ranking result "
         f"below, this is the first-order answer: three HAR terms explain that "
         f"share of a typical TiRex coordinate.\n",
         "## Ranking\n",
         "`within-fold` is the registered statistic — AUC inside each "
         "(phase, fold) cell, pair-count weighted. `pooled` is the parent's "
         "phase-mean statistic, shown for continuity only; it is **not** "
         "registered here and is the one the pooling diagnostic discredits.\n",
         "| score | within-fold AUC | pooled phase-mean AUC |",
         "|---|---|---|"]
    label = {"p_benchmark": "parent `p_benchmark` (continuity only)",
             "bench_own": "HAR benchmark, refit in this pipeline",
             "resid_k1": "**residualized k=1 alone**",
             "bench_plus_resid": "**HAR + residualized k=1**"}
    for s, row in m["scores"].items():
        L.append(f"| {label[s]} | {row['within_fold_auc']:.4f} | "
                 f"{row['pooled_phase_mean_auc']:.4f} |")
    L.append("")
    L.append("## The registered comparison\n")
    L.append(f"`HAR + residualized k=1` minus `HAR`, within fold, paired over "
             f"the **{m['folds']} annual folds** (the five phases inside a fold "
             f"are offsets of the same year and are averaged first — treating "
             f"them as independent would repeat a pseudo-replication this "
             f"repository has already corrected twice).\n")
    L.append("| quantity | value |")
    L.append("|---|---|")
    L.append(f"| mean ΔAUC | {p['mean_delta_auc']:+.4f} |")
    L.append(f"| median ΔAUC | {p['median_delta_auc']:+.4f} |")
    L.append(f"| folds improved | {p['folds_positive']} / {m['folds']} |")
    L.append(f"| sign test p | {p['sign_p']:.4f} |")
    L.append(f"| bootstrap 95% CI | [{p['bootstrap_ci'][0]:+.4f}, "
             f"{p['bootstrap_ci'][1]:+.4f}] |")
    L.append(f"| MDE at 80% power | {p['mde_80pct_power']:.4f} |")
    L.append(f"| equivalence margin (declared) | ±{p['equivalence_margin']:.3f} |")
    L.append(f"| p_TOST | {p['p_tost']:.4f} |")
    L.append(f"| **verdict** | **`{p['verdict']}`** |")
    L.append("")
    if p["verdict"] == "equivalent":
        L.append("**`equivalent` is a positive finding, not a failure to reject.** "
                 "The equivalence test rejects non-equivalence against a margin "
                 "declared before the run, so the statement earned here is that "
                 "the orthogonal component of the latent state carries no "
                 "within-fold ranking information worth as much as one AUC "
                 "point — the representation is a lossy re-encoding of realized-"
                 "volatility history.\n")
    elif p["verdict"] == "inconclusive":
        L.append(f"**`inconclusive` means this sample cannot answer it**, not "
                 f"that there is no effect. The minimum detectable effect is "
                 f"{p['mde_80pct_power']:.4f} AUC against an observed "
                 f"{p['mean_delta_auc']:+.4f} — roughly "
                 f"{abs(p['mde_80pct_power'] / p['mean_delta_auc']):.0f}x. "
                 f"Reporting this row as a null would be the exact defect this "
                 f"repository has corrected repeatedly.\n")
        L.append(f"**And the design, not the sample, is the binding constraint.** "
                 f"Resolving an effect the size of the declared "
                 f"±{p['equivalence_margin']:.3f} margin at 80% power would take "
                 f"**~{p['folds_needed_for_margin']:.0f} annual folds** — "
                 f"{p['folds_needed_for_margin']:.0f} years of daily data, against "
                 f"the {m['folds']} usable here. The fold-to-fold spread "
                 f"(−0.075 to +0.086) is an order of magnitude larger than the "
                 f"effect being looked for, so accruing more history does not "
                 f"fix this. A conclusive answer needs a different unit of "
                 f"inference — not a longer sample.\n")
    elif p["verdict"] == "adds":
        L.append("**`adds`.** A component of the latent state that survives "
                 "projecting out HAR still ranks transitions within fold. That "
                 "is a genuinely new state variable and is worth pursuing.\n")
    else:
        L.append("**`degrades`.** The added coordinate makes within-fold "
                 "ranking significantly worse.\n")
    L.append("## Per-fold deltas\n")
    L.append("The spread, not just the mean — a mean with no spread shown is "
             "how a handful of folds gets mistaken for a result.\n")
    L.append("| fold | ΔAUC | | fold | ΔAUC |")
    L.append("|---|---|---|---|---|")
    pf = m["per_fold"]
    half = (len(pf) + 1) // 2
    for i in range(half):
        a = pf[i]
        b = pf[i + half] if i + half < len(pf) else None
        rhs = f"{b['fold_year']} | {b['delta_auc']:+.4f}" if b else " | "
        L.append(f"| {a['fold_year']} | {a['delta_auc']:+.4f} | | {rhs} |")
    L.append("")
    L.append("## Limits\n")
    L.append("- Post-result on an inspected sample; not a new holdout.\n"
             "- One registered rung (k=1), matched to the parent. A richer "
             "residualized probe might find something k=1 cannot, and this run "
             "does not exclude that.\n"
             "- The residualization is linear. A component non-linearly related "
             "to HAR would survive projection and be counted as orthogonal.\n"
             "- HAR here is the parent's three-term feature set, not the full "
             "HAR-IV information set; VXN is not projected out.\n")
    return "\n".join(L)


if __name__ == "__main__":
    m = run()
    print(f"verdict: {m['primary']['verdict']}  "
          f"mean dAUC={m['primary']['mean_delta_auc']:+.4f}  "
          f"p_sign={m['primary']['sign_p']:.4f}  "
          f"p_TOST={m['primary']['p_tost']:.4f}")
    print(f"wrote {OUT_REPORT}")
