"""Corrected inference for the forecasting experiment.

This module is a FORK, not a replacement. Nothing here is imported by the
default evaluation path, so `results_{phase}.md` and `results_{phase}_dec.md`
still reproduce byte-for-byte from the frozen pre-registration. The corrected
statistics land in a parallel `_v2` report, and the two are meant to be read
side by side.

Four defects motivated it. Each is fixed by one section below.

1. NO POWER ACCOUNTING. `metrics.dm_test` reports a p-value and nothing else,
   so a failure to reject reads as a null. On the clean window (n=192) the
   minimum detectable QLIKE gain is ~5.4% of HAR-IV's loss for the har_iv_x
   comparison and ~13% for chronos_cov_iv — i.e. the design cannot see effects
   several times larger than any plausible one. `dm_verdict` refuses to emit
   "no difference" unless an equivalence test actually says so.

2. OVERLAPPING 30-DAY WINDOWS. `metrics.mz_regression` and
   `metrics.encompassing` run on daily origins whose 30-calendar-day targets
   share ~21 trading days, and report n=171 (clean) / n=2463 (diagnostic).
   The independent counts are ~8 and ~117. HAC(32) on n=171 is a lag/n ratio
   of 19%, far past where Newey-West is reliable. On the diagnostic window the
   ratio is 1% and HAC is fine there, so the report computes it per phase
   rather than quoting one remembered ratio. NOTE, added after the 2026-08-13
   audit: the replacement bootstrap is also too narrow (see
   `block_bootstrap_ols`), so this section buys a better floor on the
   uncertainty, not a calibrated interval. Earlier claims that the "honest"
   standard error was 3x / 2.9x the HAC one are withdrawn -- they compared HAC
   against a median within-subsample se, which estimates a different quantity.
   This is also the standard config.yaml's own `carry_study` block already
   sets -- "NON-OVERLAPPING trades only ... Overlapping series is descriptive
   only".

3. POST-HOC SPECIFICATIONS SCORED AS CONFIRMATORY. Handled in
   `spec_registry.yaml` and the evaluator, not here.

4. GRID-DEPENDENT POINT FORECAST. Handled in `models.py`.

Everything in this module is deterministic given a seed; bootstraps take one
explicitly so a re-run reproduces the report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------
# 1. Power, minimum detectable effect, and equivalence
# --------------------------------------------------------------------------
def dm_mde(loss_a: np.ndarray, loss_b: np.ndarray,
           alpha: float = 0.05, power: float = 0.80) -> dict:
    """Smallest QLIKE gap the DM test could detect on this sample.

    Reported next to every DM p-value. A p of 0.78 means one thing when the
    MDE is 1% of the benchmark's loss and something entirely different when it
    is 15%; without this column the reader cannot tell which they are looking
    at. `n_required` inverts the same formula against the OBSERVED gap and
    answers "how long would this have to accrue to settle it".
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return {"mde": np.nan, "n_required": np.nan, "sd": np.nan, "n": n}
    sd = float(d.std(ddof=1))
    k = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    mde = k * sd / np.sqrt(n)
    gap = abs(float(d.mean()))
    n_req = float((k * sd / gap) ** 2) if gap > 1e-12 else np.inf
    return {"mde": float(mde), "n_required": n_req, "sd": sd, "n": n}


def dm_tost(loss_a: np.ndarray, loss_b: np.ndarray, margin: float,
            alpha: float = 0.05) -> dict:
    """Two one-sided tests for equivalence of two loss series.

    H0 is NON-equivalence (|E d| >= margin); rejecting it is the only way to
    earn the phrase "these two forecasts are the same". `margin` is in QLIKE
    units and must be chosen from the benchmark's loss level, not from the
    data -- see `equivalence_margin`.

    Uses the same plain-variance standard error as `metrics.dm_test` at h=1,
    which the measured loss-differential autocorrelations (|r1| < 0.09 on both
    phases) justify.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10 or margin <= 0:
        return {"p_tost": np.nan, "equivalent": False, "lo": np.nan,
                "hi": np.nan, "n": n}
    dbar, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(n))
    df = n - 1
    if se <= 0:
        # Two forecasts that are identical origin by origin. Degenerate, but it
        # happens when a variant collapses onto its parent, and it must not
        # raise: the answer is "equivalent" exactly when the constant gap is
        # inside the margin.
        eq = abs(dbar) < margin
        return {"p_tost": 0.0 if eq else 1.0, "equivalent": bool(eq),
                "lo": dbar, "hi": dbar, "n": n}
    # H0a: mu <= -margin   H0b: mu >= +margin. Equivalence needs both rejected.
    p_a = float(stats.t.sf((dbar + margin) / se, df))
    p_b = float(stats.t.cdf((dbar - margin) / se, df))
    p_tost = max(p_a, p_b)
    crit = stats.t.ppf(1 - alpha, df)          # (1-2*alpha) CI, the TOST dual
    return {"p_tost": p_tost, "equivalent": bool(p_tost < alpha),
            "lo": dbar - crit * se, "hi": dbar + crit * se, "n": n}


def equivalence_margin(benchmark_loss: np.ndarray, frac: float = 0.03) -> float:
    """Equivalence margin as a fraction of the benchmark's mean loss.

    Fixed at declaration time, never tuned to make a verdict come out. 3% of
    HAR-IV's QLIKE is roughly a third of the gap HAR-IV itself opens over
    plain HAR, so an "equivalent" verdict at this margin means the challenger
    is not delivering a third of the value that adding VXN delivers.
    """
    m = np.asarray(benchmark_loss, dtype=float)
    return float(frac * np.nanmean(m))


def dm_verdict(dm_p: float, gap: float, tost: dict, alpha: float = 0.05) -> str:
    """Three-way verdict. The point of the whole exercise.

    `gap` is mean(loss_b) - mean(loss_a): positive = A better. A non-significant
    DM alone is NEVER reported as evidence of no difference; it maps to
    `inconclusive` unless the equivalence test independently rejects.
    """
    if np.isfinite(dm_p) and dm_p < alpha:
        return "A better" if gap > 0 else "B better"
    if tost.get("equivalent"):
        return "equivalent"
    return "inconclusive"


# --------------------------------------------------------------------------
# 2. Inference under overlapping forecast windows
# --------------------------------------------------------------------------
def effective_n(n: int, overlap: int) -> float:
    """Independent observations behind an overlapping-window regression.

    n daily origins whose targets each span `overlap` trading days contain
    about n/overlap non-overlapping windows. Reporting n instead of this is
    the single biggest overstatement in the 30-day tables.
    """
    return float(n) / max(int(overlap), 1)


def _ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def nonoverlap_phases(y: pd.Series, X: pd.DataFrame, step: int) -> dict:
    """Refit on all `step` non-overlapping subsamples (origins t, t+step, ...).

    Each subsample is genuinely free of target overlap, so its textbook OLS
    standard error is honest. The spread of the point estimate ACROSS phases is
    reported too: if beta swings from 0.30 to 1.71 depending on which day of
    the month you start on, the full-sample point estimate is not a fact about
    the market.
    """
    d = pd.concat([y.rename("_y"), X], axis=1).dropna()
    cols = list(X.columns)
    betas, ses, ns = [], [], []
    for p in range(step):
        s = d.iloc[p::step]
        if len(s) < len(cols) + 3:
            continue
        Xs = s[cols].values
        ys = s["_y"].values
        b = _ols(ys, Xs)
        resid = ys - Xs @ b
        dof = len(s) - Xs.shape[1]
        if dof <= 0:
            continue
        xtx_inv = np.linalg.pinv(Xs.T @ Xs)
        se = np.sqrt(np.diag(xtx_inv) * (resid @ resid) / dof)
        betas.append(b)
        ses.append(se)
        ns.append(len(s))
    if not betas:
        return {"n_phases": 0}
    B, S = np.asarray(betas), np.asarray(ses)
    return {
        "n_phases": len(betas),
        "n_per_phase": float(np.mean(ns)),
        "beta_median": dict(zip(cols, np.median(B, axis=0))),
        "beta_min": dict(zip(cols, B.min(axis=0))),
        "beta_max": dict(zip(cols, B.max(axis=0))),
        "beta_sd_across_phases": dict(zip(cols, B.std(axis=0, ddof=1))),
        # Median WITHIN-subsample SE. This is the standard error of a beta
        # fitted to ~n/step observations, NOT of the full-sample beta -- it is
        # algebraically close to sqrt(step) x the full-sample OLS se, so it
        # neither is nor measures the uncertainty of the reported coefficient.
        # It was previously published as "honest se" beside the full-sample
        # beta, which overstated the uncertainty of that estimate by ~1.3-1.4x.
        # Kept because the per-subsample scale is worth seeing; renamed so it
        # cannot be read as the thing it is not. Prefer beta_sd_across_phases.
        "se_within_subsample": dict(zip(cols, np.median(S, axis=0))),
    }


def block_bootstrap_ols(y: pd.Series, X: pd.DataFrame, block: int,
                        n_boot: int = 2000, seed: int = 20260812,
                        stat=None) -> dict:
    """Circular moving-block bootstrap for an overlapping-window regression.

    Uses every observation (unlike the phase split) while still respecting the
    dependence induced by the shared target window. Block length is set to the
    overlap, so a resampled block carries its own serial correlation with it.

    KNOWN MISCALIBRATION, measured rather than assumed. Resampling blocks of
    ROWS conditions on the one realized near-unit-root regressor path, so the
    intervals are too narrow: against a DGP at this data's measured persistence
    (regressor AR(1) 0.895, MA(20) errors) the nominal-95% interval covers
    ~82-85% and se_boot is ~0.75x the truth. Lengthening the block makes it
    worse, not better. `tests/test_methodology.py::TestBootstrapCalibration`
    pins this so it cannot be forgotten. Consequently `p_zero` / `p_one` have
    true size ~10-16% rather than 5%, and a bootstrap p just under 0.05 is NOT
    a 5% rejection. These intervals are still far better than HAC(32) at
    lag/n = 0.19, which is why they are used -- but they are a floor on the
    uncertainty, not a calibrated interval.

    `stat` optionally maps a coefficient vector plus the resampled frame to a
    scalar, which is how the "vol points of premium" gets a confidence
    interval instead of a point.
    """
    d = pd.concat([y.rename("_y"), X], axis=1).dropna()
    cols = list(X.columns)
    Xa, ya = d[cols].values, d["_y"].values
    n = len(d)
    if n < block + 5:
        return {"n": n, "n_boot": 0}
    beta_hat = _ols(ya, Xa)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    draws, stats_ = [], []
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(np.arange(s, s + block) % n) for s in starts])[:n]
        Xb, yb = Xa[idx], ya[idx]
        try:
            b = _ols(yb, Xb)
        except np.linalg.LinAlgError:
            continue
        draws.append(b)
        if stat is not None:
            stats_.append(stat(b, Xb))
    D = np.asarray(draws)
    out = {
        "n": n, "n_boot": len(D), "block": block,
        "beta": dict(zip(cols, beta_hat)),
        "se_boot": dict(zip(cols, D.std(axis=0, ddof=1))),
        "ci_lo": dict(zip(cols, np.percentile(D, 2.5, axis=0))),
        "ci_hi": dict(zip(cols, np.percentile(D, 97.5, axis=0))),
    }
    # Two-sided bootstrap p-value for each coefficient against zero, and for
    # the slope against one (the MZ efficiency null).
    out["p_zero"] = {c: float(2 * min((D[:, j] <= 0).mean(),
                                      (D[:, j] >= 0).mean()))
                     for j, c in enumerate(cols)}
    out["p_one"] = {c: float(2 * min((D[:, j] <= 1).mean(),
                                     (D[:, j] >= 1).mean()))
                    for j, c in enumerate(cols)}
    if stats_:
        s = np.asarray(stats_, dtype=float)
        s = s[np.isfinite(s)]
        out["stat_median"] = float(np.median(s))
        out["stat_ci"] = (float(np.percentile(s, 2.5)),
                          float(np.percentile(s, 97.5)))
    return out


def mz_corrected(realized_log: pd.Series, forecast_log: pd.Series,
                 overlap: int = 21, n_boot: int = 2000,
                 seed: int = 20260812) -> dict:
    """Mincer-Zarnowitz with overlap-aware inference.

    Same point estimates as `metrics.mz_regression`; the standard errors and
    the p-value on beta=1 come from the block bootstrap and the non-overlapping
    phase split instead of HAC(32).
    """
    d = pd.concat([realized_log.rename("y"), forecast_log.rename("f")],
                  axis=1).dropna()
    X = pd.DataFrame({"const": 1.0, "f": d["f"]}, index=d.index)
    boot = block_bootstrap_ols(d["y"], X, block=overlap, n_boot=n_boot, seed=seed)
    phases = nonoverlap_phases(d["y"], X, step=overlap)
    r2 = np.nan
    if boot.get("n_boot"):
        fit = X.values @ np.array([boot["beta"]["const"], boot["beta"]["f"]])
        ss = ((d["y"] - fit) ** 2).sum()
        r2 = float(1 - ss / ((d["y"] - d["y"].mean()) ** 2).sum())
    return {"n": len(d), "n_eff": effective_n(len(d), overlap), "r2": r2,
            "boot": boot, "phases": phases}


def encompassing_corrected(realized_log: pd.Series, implied_log: pd.Series,
                           model_log: pd.Series, overlap: int = 21,
                           n_boot: int = 2000, seed: int = 20260812) -> dict:
    """realized = a + b*implied + c*model, with overlap-aware inference."""
    d = pd.concat([realized_log.rename("y"), implied_log.rename("iv"),
                   model_log.rename("m")], axis=1).dropna()
    X = pd.DataFrame({"const": 1.0, "iv": d["iv"], "m": d["m"]}, index=d.index)
    boot = block_bootstrap_ols(d["y"], X, block=overlap, n_boot=n_boot, seed=seed)
    phases = nonoverlap_phases(d["y"], X, step=overlap)
    return {"n": len(d), "n_eff": effective_n(len(d), overlap),
            "boot": boot, "phases": phases}


def premium_ci(realized_log: pd.Series, implied_log: pd.Series,
               vxn_level: pd.Series, overlap: int = 21, n_boot: int = 2000,
               seed: int = 20260812) -> dict:
    """Variance risk premium in vol points, with a bootstrap interval.

    The frozen report evaluates the fitted MZ line at the median VXN and prints
    one number ("about 4.2 vol points"). On the clean window that number ranges
    3.1 to 5.0 across non-overlapping subsamples, so a point estimate with no
    interval is the wrong object. Same estimator, resampled.
    """
    d = pd.concat([realized_log.rename("y"), implied_log.rename("f"),
                   vxn_level.rename("vxn")], axis=1).dropna()
    if d.empty:
        return {"n": 0}
    X = pd.DataFrame({"const": 1.0, "f": d["f"]}, index=d.index)
    iv_med, vxn_med = float(d["f"].median()), float(d["vxn"].median())

    def _prem(beta, _Xb):
        ratio = np.exp(beta[0] + (beta[1] - 1.0) * iv_med)
        return vxn_med * (1.0 - np.sqrt(ratio))

    boot = block_bootstrap_ols(d["y"], X, block=overlap, n_boot=n_boot,
                               seed=seed, stat=_prem)
    point = _prem([boot["beta"]["const"], boot["beta"]["f"]], None) \
        if boot.get("n_boot") else np.nan
    return {"n": len(d), "n_eff": effective_n(len(d), overlap),
            "vxn_median": vxn_med, "point": point,
            "ci": boot.get("stat_ci"), "n_boot": boot.get("n_boot", 0)}


# --------------------------------------------------------------------------
# 3. Replication across phases
# --------------------------------------------------------------------------
def paired_summary(loss_a: pd.Series, loss_b: pd.Series) -> dict:
    """Win rate, sign test, mean gap, and leave-top-k-out fragility.

    `flip_k` is the number of origins that must be removed from the winning
    tail before the mean gap changes sign. On the clean window har_iv_x needs
    only 5 of 192; the frozen report's "top-10 share of mean gap" column says
    the same thing less directly, and is easy to read past.
    """
    ix = loss_a.index.intersection(loss_b.index)
    a, b = loss_a.loc[ix], loss_b.loc[ix]
    n = len(ix)
    if n < 3:
        return {"n": n}
    wins = int((a < b).sum())
    d = (b - a).astype(float)                     # positive = A better
    flip_k = None
    order = d.sort_values(ascending=False).index
    for k in range(0, min(n - 1, 25) + 1):
        if d.drop(order[:k]).mean() <= 0:
            flip_k = k
            break
    top = d.max()
    return {
        "n": n, "wins": wins, "win_pct": 100.0 * wins / n,
        "sign_p": float(stats.binomtest(wins, n, 0.5).pvalue),
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "gap": float(d.mean()),
        "flip_k": flip_k,
        "top_day": str(d.idxmax().date()) if n else None,
        "top_share": float(100 * top / d.sum()) if abs(d.sum()) > 1e-12 else np.nan,
    }
