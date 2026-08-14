"""Chronos-2 zero-shot walk-forward forecasts of log realized variance.

Variants (info sets):
  uni     - target history only
  cov     - + known-future calendar covariates (FOMC/CPI/NFP countdowns & flags,
             earnings weight, day-of-week)
  cov_iv  - cov + VXN level as a *past-observed* covariate (market-information set;
             a win here means less than a win for `cov`)

Implementation notes:
- predict_df() infers frequency from timestamps, and trading days are irregular,
  so each origin's context is re-timestamped onto a synthetic contiguous daily
  grid. The mapping back to real dates is positional and exact.
- Each origin is its own series id; origins are batched `batch_origins` at a
  time through one predict_df call.
- h=1 quantiles come from forecast step 1. The 30-calendar-day point forecast
  sums per-step truncated means over the next 21 steps (expectation is linear,
  so no joint-dependence assumption is needed for the mean).
- Known-future covariates can only be supplied for dates that already exist in
  the master frame, so the newest origins have fewer than 21 future rows. Those
  origins are run at a shorter prediction_length (h=1 is always available, since
  origins exclude the last date) and get log_cum_var_hat = NaN. Origins are
  therefore grouped by how many future rows they have, and each group is batched
  separately — a short tail must never silently drop a whole batch of origins.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import models

FUTURE_COLS = ["is_fomc", "is_cpi", "is_nfp", "days_to_fomc", "days_to_cpi",
               "days_to_nfp", "earnings_wt", "dow"]
CUM_STEPS = 21  # trading days approximating 30 calendar days


def _device(cfg: dict) -> str:
    want = cfg["chronos"].get("device", "auto")
    if want != "auto":
        return want
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_pipeline(cfg: dict):
    from chronos import Chronos2Pipeline
    return Chronos2Pipeline.from_pretrained(cfg["chronos"]["model_id"],
                                            device_map=_device(cfg))


def _frames_for_batch(df: pd.DataFrame, batch: list[pd.Timestamp], cfg: dict,
                      variant: str, pred_len: int):
    """Context and future frames for one batch of origins, all at `pred_len`.

    Callers must only pass origins that have >= pred_len future rows in df
    (see `_avail_steps`); a short origin here is a programming error, not a
    condition to silently skip.
    """
    ctx_rows, fut_rows = [], []
    n_ctx = cfg["context_days"]
    for t in batch:
        hist = df.loc[:t].tail(n_ctx)
        m = len(hist)
        t0 = pd.Timestamp("2000-01-01")
        stamps = t0 + pd.to_timedelta(np.arange(m), unit="D")
        ctx = pd.DataFrame({"id": str(t.date()), "timestamp": stamps,
                            "target": hist["log_rv"].values})
        if variant in ("cov", "cov_iv"):
            for c in FUTURE_COLS:
                ctx[c] = hist[c].values
        if variant == "cov_iv":
            ctx["vxn"] = hist["vxn"].values
        ctx_rows.append(ctx)

        fut_stamps = t0 + pd.to_timedelta(np.arange(m, m + pred_len), unit="D")
        fut = pd.DataFrame({"id": str(t.date()), "timestamp": fut_stamps})
        if variant in ("cov", "cov_iv"):
            pos = df.index.get_loc(t)
            future_real = df.iloc[pos + 1: pos + 1 + pred_len]
            if len(future_real) < pred_len:
                raise ValueError(
                    f"origin {t.date()} has {len(future_real)} future rows, "
                    f"needs {pred_len} — group origins with _avail_steps first"
                )
            for c in FUTURE_COLS:
                fut[c] = future_real[c].values  # calendar features: known ex ante
        fut_rows.append(fut)
    return pd.concat(ctx_rows, ignore_index=True), pd.concat(fut_rows, ignore_index=True)


def _avail_steps(df: pd.DataFrame, t: pd.Timestamp, max_steps: int) -> int:
    """Forecast steps supportable at origin t, capped at max_steps.

    Bounded by the future covariate rows present in df. Origins exclude the last
    date, so this is >= 1 for every valid origin.
    """
    return int(min(max_steps, len(df.index) - 1 - df.index.get_loc(t)))


def _qcol_map(pred: pd.DataFrame, taus: list[float]) -> list[str]:
    """Map each tau to its output column, matching on float value.

    predict_df names quantile columns by their level ('0.05', '0.1', ...), so
    string formatting of the config value is not reliably an exact match.
    """
    numeric = {}
    for c in pred.columns:
        try:
            numeric[float(c)] = c
        except (TypeError, ValueError):
            continue
    missing = [t for t in taus if t not in numeric]
    if missing:
        raise KeyError(f"quantile columns {missing} absent from predict_df output "
                       f"(got {sorted(numeric)})")
    return [numeric[t] for t in taus]


def run_chronos(df: pd.DataFrame, origins: pd.DatetimeIndex, cfg: dict,
                variant: str = "uni") -> pd.DataFrame:
    taus = list(cfg["quantiles"])
    pipe = _load_pipeline(cfg)
    if variant == "cov_iv" and "vxn" not in df.columns:
        sys.exit("cov_iv requires the VXN series — run the free fetcher first")
    rows = []
    B = cfg["chronos"]["batch_origins"]
    origins = [t for t in origins if t in df.index]

    # Group by supportable horizon so a short tail costs those origins their
    # 30-day cumulative forecast only — never their h=1 forecast, and never
    # the other origins batched alongside them. `uni` needs no future covariates
    # and could always run the full horizon, but it is grouped identically so
    # every variant conditions on the same horizon at each origin; otherwise the
    # uni-vs-cov comparison would differ by more than its information set.
    by_len: dict[int, list[pd.Timestamp]] = {}
    for t in origins:
        by_len.setdefault(_avail_steps(df, t, CUM_STEPS), []).append(t)
    short = sum(len(v) for k, v in by_len.items() if k < CUM_STEPS)
    if short:
        print(f"  {short} origin(s) near the data frontier run at a shorter "
              f"horizon (h=1 kept, 30d cumulative = NaN)", flush=True)

    done = 0
    for pred_len in sorted(by_len, reverse=True):
        group = by_len[pred_len]
        for i in range(0, len(group), B):
            batch = group[i: i + B]
            ctx, fut = _frames_for_batch(df, batch, cfg, variant, pred_len)
            pred = pipe.predict_df(
                ctx,
                future_df=fut if variant in ("cov", "cov_iv") else None,
                prediction_length=pred_len,
                quantile_levels=taus,
                id_column="id",
                timestamp_column="timestamp",
                target="target",
            )
            qcols = _qcol_map(pred, taus)
            for t in batch:
                p = pred[pred["id"] == str(t.date())].sort_values("timestamp")
                if len(p) < pred_len:
                    print(f"  warning: origin {t.date()} returned {len(p)}/"
                          f"{pred_len} steps — skipped", flush=True)
                    continue
                q = np.array([float(p.iloc[0][c]) for c in qcols])
                if pred_len == CUM_STEPS:
                    cum = sum(
                        models.trunc_mean_var(np.asarray(taus),
                                              np.array([float(r[c]) for c in qcols]))
                        for _, r in p.iterrows()
                    )
                    log_cum = float(np.log(max(cum, 1e-12)))
                else:
                    log_cum = np.nan  # horizon not fully covariate-supported
                rows.append({
                    "origin": t,
                    **{f"q{tau:.2f}": v for tau, v in zip(taus, q)},
                    "mean_var": models.trunc_mean_var(np.asarray(taus), q),
                    "log_cum_var_hat": log_cum,
                })
            done += len(batch)
            print(f"  {done}/{len(origins)} origins", flush=True)
    return pd.DataFrame(rows).set_index("origin").sort_index()
