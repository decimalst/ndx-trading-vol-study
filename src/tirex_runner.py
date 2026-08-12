"""TiRex-2 zero-shot walk-forward forecasts of log realized variance.

Same variants and output schema as `chronos_runner`, so the two model families
drop into the same evaluation path:
  uni     - target history only
  cov     - + known-future calendar covariates
  cov_iv  - cov + VXN as a *past-observed* covariate (market information set)

Differences from the Chronos-2 runner, all forced by the model's API:

- **Fixed decile output.** TiRex-2 emits exactly 9 quantiles (0.1 ... 0.9;
  `model.quantiles`) and takes no quantile_levels argument. It cannot produce
  the 0.05/0.95 levels the pre-registered grid asks for, so anything compared
  against TiRex must be recomputed on the decile grid — see the
  `--quantile-grid deciles` path in experiment.py. The widest interval
  available is therefore 80%, not 90%.
- **No timestamps.** Inputs are plain tensors, so the synthetic contiguous
  daily grid the Chronos runner needs does not exist here; position in the
  tensor is the only index. One fewer thing to get wrong.
- **Future covariates span context + horizon.** TiRex wants [V_f, T+H] rather
  than horizon-only rows, so context and future covariate blocks are
  concatenated.

Origins are grouped by supportable horizon for the same reason as in the
Chronos runner: an origin near the data frontier must still yield its h=1
forecast, and must not take other origins down with it.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import models

FUTURE_COLS = ["is_fomc", "is_cpi", "is_nfp", "days_to_fomc", "days_to_cpi",
               "days_to_nfp", "earnings_wt", "dow"]
CUM_STEPS = 21  # trading days approximating 30 calendar days
MODEL_ID = "NX-AI/TiRex-2"


def _device(cfg: dict) -> str:
    want = cfg.get("tirex", {}).get("device", cfg["chronos"].get("device", "auto"))
    if want != "auto":
        return want
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def native_quantiles(cfg: dict) -> list[float]:
    """The model's fixed output levels, read off the loaded checkpoint."""
    model = _load_model(cfg)
    return [round(float(q), 4) for q in model.quantiles]


_MODEL_CACHE: dict = {}


def _load_model(cfg: dict):
    dev = _device(cfg)
    key = (cfg.get("tirex", {}).get("model_id", MODEL_ID), dev)
    if key not in _MODEL_CACHE:
        from tirex2 import load_model
        _MODEL_CACHE[key] = load_model(key[0], device=dev)
    return _MODEL_CACHE[key]


def _avail_steps(df: pd.DataFrame, t: pd.Timestamp, max_steps: int) -> int:
    return int(min(max_steps, len(df.index) - 1 - df.index.get_loc(t)))


def _item(df: pd.DataFrame, t: pd.Timestamp, cfg: dict, variant: str, pred_len: int):
    import torch

    from tirex2 import TimeseriesType

    pos = df.index.get_loc(t)
    n_ctx = cfg["context_days"]
    hist = df.iloc[max(0, pos + 1 - n_ctx): pos + 1]
    target = torch.tensor(hist["log_rv"].to_numpy(), dtype=torch.float32).unsqueeze(0)

    future_cov = None
    if variant in ("cov", "cov_iv", "cov_ivf"):
        fut = df.iloc[pos + 1: pos + 1 + pred_len]
        if len(fut) < pred_len:
            raise ValueError(
                f"origin {t.date()} has {len(fut)} future rows, needs {pred_len} — "
                "group origins with _avail_steps first"
            )
        # [V_f, T+H]: calendar features over context and horizon, known ex ante
        past_blk = hist[FUTURE_COLS].to_numpy()
        fut_blk = fut[FUTURE_COLS].to_numpy()
        if variant == "cov_ivf":
            # VXN delivered through the known-future channel: history over the
            # context, then the LAST OBSERVED value carried flat across the
            # horizon. Uses nothing after t, so the information set matches
            # cov_iv — but it arrives via the channel TiRex-2 actually reads.
            # An oracle probe showed its past-covariate channel inert here while
            # the future channel is near-perfect, which makes cov_iv unmeasurable
            # for this model. See reports/FINDINGS.md.
            vx_hist = hist["vxn"].to_numpy()
            past_blk = np.concatenate([past_blk, vx_hist[:, None]], axis=1)
            fut_blk = np.concatenate(
                [fut_blk, np.full((pred_len, 1), vx_hist[-1])], axis=1)
        block = np.concatenate([past_blk, fut_blk], axis=0).T
        future_cov = torch.tensor(np.ascontiguousarray(block), dtype=torch.float32)

    past_cov = None
    if variant == "cov_iv":
        past_cov = torch.tensor(hist[["vxn"]].to_numpy().T.copy(), dtype=torch.float32)

    return TimeseriesType(target=target, past_covariates=past_cov,
                          future_covariates=future_cov)


def run_tirex(df: pd.DataFrame, origins: pd.DatetimeIndex, cfg: dict,
              variant: str = "uni") -> pd.DataFrame:
    model = _load_model(cfg)
    taus = [round(float(q), 4) for q in model.quantiles]
    if variant in ("cov_iv", "cov_ivf") and "vxn" not in df.columns:
        sys.exit("cov_iv requires the VXN series — run the free fetcher first")
    B = cfg["chronos"]["batch_origins"]
    origins = [t for t in origins if t in df.index]

    by_len: dict[int, list[pd.Timestamp]] = {}
    for t in origins:
        by_len.setdefault(_avail_steps(df, t, CUM_STEPS), []).append(t)
    short = sum(len(v) for k, v in by_len.items() if k < CUM_STEPS)
    if short:
        print(f"  {short} origin(s) near the data frontier run at a shorter "
              f"horizon (h=1 kept, 30d cumulative = NaN)", flush=True)

    rows, done = [], 0
    for pred_len in sorted(by_len, reverse=True):
        group = by_len[pred_len]
        for i in range(0, len(group), B):
            batch = group[i: i + B]
            items = [_item(df, t, cfg, variant, pred_len) for t in batch]
            out = model.forecast(items, prediction_length=pred_len,
                                 output_type="numpy")
            for t, fc in zip(batch, out):
                a = np.asarray(fc)[0]           # (n_quantiles, pred_len)
                q = a[:, 0].astype(float)       # h=1
                if pred_len == CUM_STEPS:
                    cum = sum(models.trunc_mean_var(np.asarray(taus),
                                                    a[:, s].astype(float))
                              for s in range(pred_len))
                    log_cum = float(np.log(max(cum, 1e-12)))
                else:
                    log_cum = np.nan
                rows.append({
                    "origin": t,
                    **{f"q{tau:.2f}": v for tau, v in zip(taus, q)},
                    "mean_var": models.trunc_mean_var(np.asarray(taus), q),
                    "log_cum_var_hat": log_cum,
                })
            done += len(batch)
            print(f"  {done}/{len(origins)} origins", flush=True)
    return pd.DataFrame(rows).set_index("origin").sort_index()
