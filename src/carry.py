"""Conditional vs unconditional variance carry — see config.yaml `carry_study`.

Forecasting variance well is not the same as trading the premium. P&L depends on
the CONDITIONAL variance risk premium — whether implied is rich right now — not
the unconditional average. This module tests whether selecting on richness beats
always being short, on the diagnostic window only.

This is an evaluation, not a strategy. Read `known_limits` in config.yaml before
reading any number here: it is a linear variance-swap proxy with no costs, no
financing, and no gap risk, on an instrument retail cannot trade.

  python -m src.carry
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

STEP = 21  # trading days spanned by a 30-calendar-day target


def build_trades(cfg: dict) -> pd.DataFrame:
    df = pd.read_parquet(cfg["paths"]["processed"] / f"master_daily.parquet")
    fc = pd.read_parquet(cfg["paths"]["forecasts"] / "har_cum_all.parquet")
    lo, hi = cfg["diagnostic_start"], cfg["diagnostic_end"]

    d = pd.concat(
        [df["vxn_exp_var_30c"].rename("iv"), df["fwd_var_30c"].rename("rv"),
         df["vxn"], fc["log_cum_var_hat"].rename("f")], axis=1).dropna()
    d = d[(d.index >= lo) & (d.index <= hi)]

    # short-variance P&L in 30-day total-variance units
    d["pnl_var"] = d["iv"] - d["rv"]
    # vega-equivalent vol points: (K^2 - RV^2) / (2K), annualised
    ann = 365.0 / cfg["cum_horizon_cal_days"]
    k = np.sqrt(d["iv"] * ann)
    d["pnl_vol"] = (d["iv"] * ann - d["rv"] * ann) / (2 * k) * 100.0
    d["richness"] = np.log(d["iv"]) - d["f"]
    return d


def _stats(p: np.ndarray) -> dict:
    if not len(p):
        return {}
    srt = np.sort(p)
    k = max(1, int(np.ceil(0.05 * len(p))))
    return {
        "n": len(p), "mean": p.mean(), "median": float(np.median(p)),
        "hit": float((p > 0).mean()), "sd": p.std(ddof=1) if len(p) > 1 else np.nan,
        "worst": float(srt[0]), "cvar5": float(srt[:k].mean()),
        "maxdd": float((np.maximum.accumulate(np.cumsum(p)) - np.cumsum(p)).max()),
        "total": float(p.sum()),
    }


def _boot_diff(a: np.ndarray, b: np.ndarray, n: int = 20000, seed: int = 0) -> float:
    """Two-sided bootstrap p for mean(a) - mean(b), independent resampling."""
    rng = np.random.default_rng(seed)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    na = len(a)
    diffs = np.empty(n)
    for i in range(n):
        s = rng.permutation(pool)
        diffs[i] = s[:na].mean() - s[na:].mean()
    return float((np.abs(diffs) >= abs(obs)).mean())


def run(cfg: dict) -> None:
    d = build_trades(cfg)
    thr = float(d["richness"].median())
    print(f"diagnostic window: {len(d)} overlapping origins, "
          f"{d.index.min().date()} .. {d.index.max().date()}")
    print(f"richness threshold (median, fixed): {thr:+.4f}")
    print(f"unconditional mean premium: {d['pnl_vol'].mean():+.2f} vol points\n")

    rows = []
    for off in range(STEP):
        sub = d.iloc[off::STEP]
        if len(sub) < 40:
            continue
        p_all = sub["pnl_vol"].to_numpy()
        sel = sub[sub["richness"] > thr]
        p_sel = sel["pnl_vol"].to_numpy()
        if len(p_sel) < 10:
            continue
        sa, sb = _stats(p_sel), _stats(p_all)
        rows.append({
            "phase": off, "n_all": sb["n"], "n_sel": sa["n"],
            "mean_all": sb["mean"], "mean_sel": sa["mean"],
            "cvar_all": sb["cvar5"], "cvar_sel": sa["cvar5"],
            "worst_all": sb["worst"], "worst_sel": sa["worst"],
            "hit_all": sb["hit"], "hit_sel": sa["hit"],
            "dd_all": sb["maxdd"], "dd_sel": sa["maxdd"],
            "p_mean": _boot_diff(p_sel, p_all, seed=off),
        })
    r = pd.DataFrame(rows)

    print("=== NON-OVERLAPPING TRADES, all 21 block phases "
          "(P&L in vega-equivalent vol points per trade)")
    print(f"  trades taken: {r['n_sel'].mean():.0f} of {r['n_all'].mean():.0f} available")
    print(f"  mean P&L   unconditional {r['mean_all'].mean():+.3f}  "
          f"conditional {r['mean_sel'].mean():+.3f}  "
          f"diff {r['mean_sel'].mean() - r['mean_all'].mean():+.3f}")
    print(f"  hit rate   unconditional {r['hit_all'].mean():.3f}  "
          f"conditional {r['hit_sel'].mean():.3f}")
    print(f"  5% CVaR    unconditional {r['cvar_all'].mean():+.3f}  "
          f"conditional {r['cvar_sel'].mean():+.3f}   "
          f"(less negative = better)")
    print(f"  worst trade unconditional {r['worst_all'].mean():+.3f}  "
          f"conditional {r['worst_sel'].mean():+.3f}")
    print(f"  max drawdown unconditional {r['dd_all'].mean():.3f}  "
          f"conditional {r['dd_sel'].mean():.3f}")
    print(f"  bootstrap p (mean diff): median {r['p_mean'].median():.4f}, "
          f"share < 0.05 = {(r['p_mean'] < 0.05).mean():.2f}")
    print(f"  phases where conditional mean > unconditional: "
          f"{int((r['mean_sel'] > r['mean_all']).sum())}/{len(r)}")
    print(f"  phases where conditional CVaR is WORSE: "
          f"{int((r['cvar_sel'] < r['cvar_all']).sum())}/{len(r)}")

    mean_better = r["mean_sel"].mean() > r["mean_all"].mean()
    sig = r["p_mean"].median() < 0.05
    tail_ok = r["cvar_sel"].mean() >= r["cvar_all"].mean()
    print(f"\n=== PRE-COMMITTED CRITERION")
    print(f"  mean improves ........ {mean_better}")
    print(f"  significant (median p<0.05) ... {sig}")
    print(f"  CVaR not worsened .... {tail_ok}")
    print(f"  VERDICT: {'PASS' if (mean_better and sig and tail_ok) else 'FAIL'}")

    print("\n=== worst 5 trades in the full overlapping series (tail, not summarised)")
    w = d.nsmallest(5, "pnl_vol")[["vxn", "iv", "rv", "richness", "pnl_vol"]]
    w = w.assign(taken=w["richness"] > thr)
    print(w.round(4).to_string())


if __name__ == "__main__":
    run(config.load())
