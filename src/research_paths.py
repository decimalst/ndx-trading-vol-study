"""Frozen extensions that measure information beyond the 30-day IV level.

This module is deliberately independent of ``src.models`` and
``src.experiment``. Those files may continue evolving under the original
protocol; these studies consume frozen source artifacts and never write into the
sealed clean-window outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import time
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import yaml


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "research_paths.yaml"
LN2 = np.log(2.0)


def load_protocol(path: Path | str = PROTOCOL_PATH) -> dict:
    with open(path) as handle:
        protocol = yaml.safe_load(handle)
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: dict) -> None:
    """Fail closed if any boundary that protects the original study is lost."""
    required = {
        "fences", "quarterly_top25", "absorption_map", "horizon_curve",
        "vrp_term_structure", "single_name_earnings",
        "spx_term_slope_replication",
    }
    missing = required - set(protocol)
    if missing:
        raise ValueError(f"protocol missing sections: {sorted(missing)}")
    fences = protocol["fences"]
    if not protocol.get("frozen_before_empirical_run"):
        raise ValueError("protocol must be frozen before empirical work")
    if not fences.get("forbid_sealed_ndx_clean_origins"):
        raise ValueError("NDX clean origins must remain forbidden")
    if not fences.get("no_forward_fill"):
        raise ValueError("missing observations may not be forward-filled")
    if not fences.get("no_candidate_selection"):
        raise ValueError("candidate selection is forbidden")
    clean = pd.Timestamp(fences["sealed_ndx_clean_start"])
    for section in ("absorption_map", "horizon_curve"):
        if pd.Timestamp(protocol[section]["end"]) >= clean:
            raise ValueError(f"{section} overlaps the sealed NDX clean phase")
    replication = protocol["spx_term_slope_replication"]
    if pd.Timestamp(replication["score_end"]) >= pd.Timestamp(
        fences["ndx_diagnostic_start"]
    ):
        raise ValueError("SPX score window overlaps NDX term-slope discovery")
    if int(protocol["quarterly_top25"]["keep"]) != 25:
        raise ValueError("quarterly universe must remain top 25")


def _normal_dates(values: Iterable) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(list(values)))
    if index.tz is not None:
        index = index.tz_localize(None)
    return index.normalize()


def align_exact_dates(source: pd.Series, target_index: Iterable) -> pd.Series:
    """Exact-date join with no stale-value fill across a missing observation."""
    series = source.copy()
    source_index = pd.DatetimeIndex(pd.to_datetime(series.index))
    if source_index.tz is not None:
        source_index = source_index.tz_localize(None)
    series.index = source_index.normalize()
    if series.index.duplicated().any():
        raise ValueError("source has duplicate dates")
    target = _normal_dates(target_index)
    return series.sort_index().reindex(target)


def forward_mean_variance(series: pd.Series, horizon: int) -> pd.Series:
    """Mean of exactly the next ``horizon`` observed sessions, excluding t."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for i in range(max(0, len(values) - horizon)):
        future = values[i + 1 : i + horizon + 1]
        if len(future) == horizon and np.isfinite(future).all():
            out[i] = float(future.mean())
    return pd.Series(out, index=series.index, name=f"fwd_mean_var_{horizon}")


def training_rows_completed_by(index: Iterable, origin, horizon: int) -> np.ndarray:
    """Rows whose entire forward target is observable at ``origin``."""
    dates = _normal_dates(index)
    when = pd.Timestamp(origin).tz_localize(None).normalize()
    positions = np.arange(len(dates))
    origin_pos = int(dates.searchsorted(when, side="right") - 1)
    return positions + int(horizon) <= origin_pos


def parse_cboe_history(text: str, symbol: str) -> pd.Series:
    frame = pd.read_csv(io.StringIO(text))
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if frame.empty:
        raise ValueError(f"{symbol}: empty Cboe history")
    date_col = "date" if "date" in frame else frame.columns[0]
    close_col = "close" if "close" in frame else frame.columns[-1]
    frame[date_col] = pd.to_datetime(frame[date_col], errors="raise").dt.normalize()
    if frame[date_col].duplicated().any():
        raise ValueError(f"{symbol}: duplicate dates")
    close = pd.to_numeric(frame[close_col], errors="raise")
    if (close <= 0).any():
        raise ValueError(f"{symbol}: non-positive close")
    out = pd.Series(close.to_numpy(), index=frame[date_col],
                    name=symbol.lower()).sort_index()
    out.index.name = "date"
    return out


def gk_plus_overnight(daily: pd.DataFrame) -> pd.DataFrame:
    """Daily Garman-Klass RTH variance plus the preceding close/open gap."""
    frame = daily.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    required = {"open", "high", "low", "close"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"OHLC frame missing columns: {sorted(missing)}")
    if frame.index.duplicated().any():
        raise ValueError("OHLC frame has duplicate dates")
    frame = frame.sort_index()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if (frame[column] <= 0).any():
            raise ValueError(f"OHLC frame has non-positive {column}")
    high_low = np.log(frame["high"] / frame["low"])
    close_open = np.log(frame["close"] / frame["open"])
    intraday = (0.5 * high_low.pow(2) - (2 * LN2 - 1) * close_open.pow(2)) \
        .clip(lower=1e-10)
    overnight_return = np.log(frame["open"] / frame["close"].shift(1))
    total = intraday + overnight_return.pow(2)
    result = pd.DataFrame({
        "rv_intraday": intraday,
        "var_overnight": overnight_return.pow(2),
        "r_overnight": overnight_return,
        "rv_total": total,
        "log_rv": np.log(total.clip(lower=1e-12)),
        "ret_cc": np.log(frame["close"] / frame["close"].shift(1)),
    })
    result.index = _normal_dates(result.index)
    result.index.name = "date"
    return result


def positive_front_slope(vix9d: pd.Series, vix: pd.Series) -> pd.Series:
    slope = np.log(pd.to_numeric(vix9d) / pd.to_numeric(vix))
    return slope.clip(lower=0.0)


def event_target_flags(trading_sessions: Iterable, events: pd.DataFrame) -> pd.Series:
    """Map BMO to the announcement day and AMC/unknown to the next session."""
    sessions = _normal_dates(trading_sessions).sort_values().unique()
    out = pd.Series(0, index=sessions, dtype=int, name="is_earnings")
    required = {"date", "session"}
    if missing := required - set(events):
        raise ValueError(f"events missing columns: {sorted(missing)}")
    for row in events.itertuples(index=False):
        date = pd.Timestamp(row.date).tz_localize(None).normalize()
        session = str(row.session).lower()
        if session == "bmo":
            target = date
        else:
            later = sessions[sessions > date]
            if not len(later):
                continue
            target = later[0]
        if target in out.index:
            out.loc[target] = 1
    return out


def build_single_name_design(frame: pd.DataFrame) -> pd.DataFrame:
    """HAR-IV features only; the signature intentionally accepts no events."""
    required = {"rv_total", "log_rv", "own_iv"}
    if missing := required - set(frame):
        raise ValueError(f"single-name frame missing columns: {sorted(missing)}")
    design = pd.DataFrame(index=frame.index)
    design["const"] = 1.0
    design["lrv_d"] = frame["log_rv"]
    design["lrv_w"] = np.log(frame["rv_total"].rolling(5).mean())
    design["lrv_m"] = np.log(frame["rv_total"].rolling(22).mean())
    design["liv"] = np.log(frame["own_iv"].where(frame["own_iv"] > 0))
    return design


def wald_attenuation(without_market: float, with_market: float) -> float:
    if not np.isfinite(without_market) or without_market <= 0:
        raise ValueError("uncontrolled Wald statistic must be positive")
    if not np.isfinite(with_market) or with_market < 0:
        raise ValueError("controlled Wald statistic must be non-negative")
    return 1.0 - float(with_market) / float(without_market)


def build_absorption_design(master: pd.DataFrame,
                            top25_earnings_weight: pd.Series) -> pd.DataFrame:
    """Build origin-t features and explicitly target-t+1 measurement labels."""
    required = {
        "rv_total", "log_rv", "ret_cc", "vxn", "is_fomc", "is_cpi", "is_nfp",
    }
    if missing := required - set(master):
        raise ValueError(f"master frame missing absorption columns: {sorted(missing)}")
    frame = master.sort_index()
    design = pd.DataFrame(index=frame.index)
    design["const"] = 1.0
    design["lrv_d"] = frame["log_rv"]
    design["lrv_w"] = np.log(frame["rv_total"].rolling(5).mean())
    design["lrv_m"] = np.log(frame["rv_total"].rolling(22).mean())
    returns = frame["ret_cc"]
    weekly = returns.rolling(5).mean()
    monthly = returns.rolling(22).mean()
    design["lev_d"] = returns.where(returns < 0, 0.0)
    design["lev_w"] = weekly.where(weekly < 0, 0.0)
    design["lev_m"] = monthly.where(monthly < 0, 0.0)
    design["liv"] = np.log(frame["vxn"].where(frame["vxn"] > 0))

    target_dow = pd.Series(frame.index.dayofweek, index=frame.index).shift(-1)
    for day, label in zip((1, 2, 3, 4),
                          ("tuesday", "wednesday", "thursday", "friday")):
        design[f"target_{label}"] = (target_dow == day).astype(float)
    design["target_fomc"] = frame["is_fomc"].shift(-1)
    design["target_cpi"] = frame["is_cpi"].shift(-1)
    design["target_nfp"] = frame["is_nfp"].shift(-1)
    # If the origin itself is an FOMC session, t+1 is the post-FOMC target.
    design["target_post_fomc"] = frame["is_fomc"]
    earnings = align_exact_dates(top25_earnings_weight, frame.index)
    design["target_top25_earnings_weight"] = earnings.shift(-1)
    design["y_next"] = frame["log_rv"].shift(-1)
    return design


def forward_calendar_realized_vol(returns: pd.Series, horizon_days: int) -> pd.Series:
    """Annualized realized vol from returns dated in the open-right H-day window."""
    if horizon_days < 1:
        raise ValueError("calendar horizon must be positive")
    series = returns.copy().sort_index()
    index = _normal_dates(series.index)
    series.index = index
    values = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=index, dtype=float,
                    name=f"realized_vol_{horizon_days}c")
    if not len(index):
        return out
    for date in index:
        end = date + pd.Timedelta(days=int(horizon_days))
        # A target is incomplete until the underlying history reaches its
        # calendar endpoint, even if the last few days are a weekend/holiday.
        if index[-1] < end:
            continue
        window = values.loc[(values.index > date) & (values.index <= end)].dropna()
        if not len(window):
            continue
        out.loc[date] = 100.0 * np.sqrt((365.0 / horizon_days)
                                        * float(window.pow(2).sum()))
    return out


def moving_block_bootstrap(values, *, block: int, draws: int, seed: int,
                           statistic) -> np.ndarray:
    """Fixed-length circular moving-block bootstrap over the first dimension."""
    array = np.asarray(values)
    if array.ndim == 0:
        raise ValueError("bootstrap input must have observations")
    n = len(array)
    if n == 0 or block < 1 or draws < 1:
        raise ValueError("bootstrap requires data, positive block, and draws")
    rng = np.random.default_rng(seed)
    result = []
    blocks_needed = int(np.ceil(n / block))
    offsets = np.arange(block)
    for _ in range(draws):
        starts = rng.integers(0, n, size=blocks_needed)
        positions = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        result.append(statistic(array[positions]))
    return np.asarray(result)


def walk_forward_log_ols(features: pd.DataFrame, target_log: pd.Series,
                         target_var: pd.Series, origins: Iterable,
                         *, label_horizon: int,
                         min_train_rows: int) -> pd.DataFrame:
    """Expanding direct forecasts with only fully completed training labels."""
    design = features.copy().sort_index()
    if design.index.duplicated().any():
        raise ValueError("features have duplicate dates")
    y_log = target_log.reindex(design.index)
    y_var = target_var.reindex(design.index)
    index = pd.DatetimeIndex(design.index)
    rows = []
    for origin in _normal_dates(origins):
        if origin not in design.index:
            continue
        pos = int(index.get_loc(origin))
        completed = np.arange(len(index)) + int(label_horizon) <= pos
        finite_x = np.isfinite(design.to_numpy(dtype=float)).all(axis=1)
        finite_y = np.isfinite(y_log.to_numpy(dtype=float))
        train_mask = completed & finite_x & finite_y
        X = design.to_numpy(dtype=float)[train_mask]
        y = y_log.to_numpy(dtype=float)[train_mask]
        if len(y) < int(min_train_rows):
            continue
        x_t = design.loc[origin].to_numpy(dtype=float)
        if not np.isfinite(x_t).all():
            continue
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        residual = y - X @ beta
        mu = float(x_t @ beta)
        mean_var = float(np.exp(mu) * np.exp(residual).mean())
        actual = float(y_var.loc[origin]) if np.isfinite(y_var.loc[origin]) else np.nan
        rows.append({
            "origin": origin,
            "mean_var": mean_var,
            "actual_var": actual,
            "log_forecast": mu,
            "train_n": int(len(y)),
        })
    if not rows:
        empty = pd.DataFrame(columns=["mean_var", "actual_var", "log_forecast", "train_n"])
        empty.index = pd.DatetimeIndex([], name="origin")
        return empty
    return pd.DataFrame(rows).set_index("origin")


def qlike_loss(actual: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if (actual <= 0).any() or (forecast <= 0).any():
        raise ValueError("QLIKE requires positive actual and forecast variance")
    ratio = actual / forecast
    return ratio - np.log(ratio) - 1.0


def dm_test_hac(loss_a: np.ndarray, loss_b: np.ndarray,
                *, horizon: int) -> dict:
    """Two-sided DM with Bartlett HAC through h-1 and Harvey adjustment."""
    from scipy import stats

    difference = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    difference = difference[np.isfinite(difference)]
    n = len(difference)
    if n < 10:
        return {"dm": np.nan, "p": np.nan, "n": n}
    variance = float(np.var(difference, ddof=0))
    for lag in range(1, int(horizon)):
        covariance = float(np.cov(difference[lag:], difference[:-lag], ddof=0)[0, 1])
        variance += 2.0 * (1.0 - lag / horizon) * covariance
    if not np.isfinite(variance) or variance <= 0:
        return {"dm": np.nan, "p": np.nan, "n": n}
    dm = float(difference.mean() / np.sqrt(variance / n))
    harvey_term = (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
    if harvey_term > 0:
        dm *= float(np.sqrt(harvey_term))
    p_value = float(2 * stats.t.sf(abs(dm), df=n - 1))
    return {"dm": dm, "p": p_value, "n": n}


def _joint_wald(result, columns: list[str]) -> tuple[float, float]:
    matrix = np.zeros((len(columns), len(result.params)))
    names = list(result.params.index)
    for row, column in enumerate(columns):
        matrix[row, names.index(column)] = 1.0
    test = result.wald_test(matrix, scalar=True)
    return float(test.statistic), float(test.pvalue)


def _partial_r2(y: pd.Series, restricted: pd.DataFrame,
                full: pd.DataFrame) -> float:
    import statsmodels.api as sm

    restricted_fit = sm.OLS(y, restricted).fit()
    full_fit = sm.OLS(y, full).fit()
    denominator = float(np.square(restricted_fit.resid).sum())
    numerator = denominator - float(np.square(full_fit.resid).sum())
    return numerator / denominator if denominator > 0 else np.nan


def fit_absorption_group(design: pd.DataFrame, group_columns: list[str],
                         start, end, hac_maxlags: int) -> dict:
    """Joint Wald attenuation on one fixed common sample."""
    import statsmodels.api as sm

    base = ["const", "lrv_d", "lrv_w", "lrv_m"]
    required = [*base, "liv", *group_columns, "y_next"]
    sample = design.loc[pd.Timestamp(start):pd.Timestamp(end), required] \
        .replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) <= len(required) + 10:
        raise ValueError(f"insufficient common rows for {group_columns}: {len(sample)}")
    without_columns = [*base, *group_columns]
    with_columns = [*base, "liv", *group_columns]
    without = sm.OLS(sample["y_next"], sample[without_columns]).fit(
        cov_type="HAC", cov_kwds={"maxlags": int(hac_maxlags)}
    )
    with_market = sm.OLS(sample["y_next"], sample[with_columns]).fit(
        cov_type="HAC", cov_kwds={"maxlags": int(hac_maxlags)}
    )
    wald_without, p_without = _joint_wald(without, group_columns)
    wald_with, p_with = _joint_wald(with_market, group_columns)
    return {
        "n_without": int(without.nobs), "n_with": int(with_market.nobs),
        "wald_without": wald_without, "p_without": p_without,
        "wald_with": wald_with, "p_with": p_with,
        "attenuation": wald_attenuation(wald_without, wald_with),
        "r2_without": float(without.rsquared), "r2_with": float(with_market.rsquared),
        "partial_r2_without": _partial_r2(
            sample["y_next"], sample[base], sample[without_columns]
        ),
        "partial_r2_with": _partial_r2(
            sample["y_next"], sample[[*base, "liv"]], sample[with_columns]
        ),
        "coefficients_without": {column: float(without.params[column])
                                  for column in group_columns},
        "coefficients_with": {column: float(with_market.params[column])
                               for column in group_columns},
        "first_origin": str(sample.index.min().date()),
        "last_origin": str(sample.index.max().date()),
    }


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run_absorption_map(protocol: dict | None = None) -> dict:
    protocol = protocol or load_protocol()
    study = protocol["absorption_map"]
    all_master = pd.read_parquet(ROOT / study["source"])
    end_position = int(all_master.index.get_loc(pd.Timestamp(study["end"])))
    if end_position + 1 >= len(all_master):
        raise ValueError("absorption target after final origin is incomplete")
    completion = all_master.index[end_position + 1]
    clean_start = pd.Timestamp(protocol["fences"]["sealed_ndx_clean_start"])
    if completion >= clean_start:
        raise ValueError("absorption target completion reaches sealed clean phase")
    master = all_master.loc[:completion]
    earnings_path = ROOT / protocol["outputs"]["data_dir"] / "top25_earnings_daily.parquet"
    earnings = pd.read_parquet(earnings_path)["top25_earnings_weight"]
    design = build_absorption_design(master, earnings)
    results = {}
    for name, columns in study["groups"].items():
        results[name] = fit_absorption_group(
            design, list(columns), study["start"], study["end"],
            int(study["hac_maxlags"]),
        )
    payload = {
        "study": "absorption_map", "evidence_class": study["evidence_class"],
        "source_estimator": study["source_estimator"], "groups": results,
    }
    data_dir = ROOT / protocol["outputs"]["data_dir"]
    report_dir = ROOT / protocol["outputs"]["report_dir"]
    _write_json(data_dir / "absorption_map_metrics.json", payload)
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# What the 30-day implied level absorbs", "",
        "**Evidence class: diagnostic measurement.** The original NDX clean window is not read.", "",
        "The reported percentage is `1 - Wald with VXN / Wald without VXN`. It is a",
        "descriptive attenuation of a joint Wald statistic—not a literal information share,",
        "not bounded to [0, 100%], and not a forecast or trading result.", "",
        "| regularity | common n | Wald without VXN | Wald with VXN | attenuation | partial R² without / with |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in results.items():
        lines.append(
            f"| {name} | {row['n_with']} | {row['wald_without']:.2f} "
            f"(p={row['p_without']:.3g}) | {row['wald_with']:.2f} "
            f"(p={row['p_with']:.3g}) | {100 * row['attenuation']:+.1f}% | "
            f"{row['partial_r2_without']:.4f} / {row['partial_r2_with']:.4f} |"
        )
    lines += [
        "", "The earnings row starts only when an SEC-accepted quarterly top-25",
        "snapshot exists. Realized announcement labels are used after the fact for",
        "measurement; they are not claimed as a versioned ex-ante calendar.", "",
        "All other rows use 2016-01-04 through 2025-10-17. Same-origin VXN is valid",
        "because this extension fixes the decision time after the 16:15 ET Cboe close.",
    ]
    (report_dir / "absorption_map.md").write_text("\n".join(lines) + "\n")

    import matplotlib.pyplot as plt
    names = list(results)
    values = [100 * results[name]["attenuation"] for name in names]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(names, values, color=["#35618f" if value >= 0 else "#a64b3c" for value in values])
    axis.axhline(0, color="black", linewidth=.8)
    axis.set_ylabel("joint-Wald attenuation (%)")
    axis.set_title("How much same-origin VXN attenuates each regularity")
    axis.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(report_dir / "absorption_map.png", dpi=180)
    plt.close(fig)
    return payload


def _har_features(frame: pd.DataFrame, *, iv: pd.Series | None = None,
                  leverage: bool = False) -> pd.DataFrame:
    features = pd.DataFrame(index=frame.index)
    features["const"] = 1.0
    features["lrv_d"] = frame["log_rv"]
    features["lrv_w"] = np.log(frame["rv_total"].rolling(5).mean())
    features["lrv_m"] = np.log(frame["rv_total"].rolling(22).mean())
    if leverage:
        returns = frame["ret_cc"]
        weekly = returns.rolling(5).mean()
        monthly = returns.rolling(22).mean()
        features["lev_d"] = returns.where(returns < 0, 0.0)
        features["lev_w"] = weekly.where(weekly < 0, 0.0)
        features["lev_m"] = monthly.where(monthly < 0, 0.0)
    if iv is not None:
        features["liv"] = np.log(iv.reindex(frame.index).where(iv.reindex(frame.index) > 0))
    return features


def _in_sample_increment(y: pd.Series, base: pd.DataFrame,
                         candidate: pd.DataFrame, hac_lags: int) -> dict:
    import statsmodels.api as sm

    joined = pd.concat([y.rename("y"), base.add_prefix("b_"),
                        candidate[[column for column in candidate if column not in base]]], axis=1)
    joined = joined.replace([np.inf, -np.inf], np.nan).dropna()
    base_columns = [f"b_{column}" for column in base]
    added = [column for column in candidate if column not in base]
    restricted = sm.OLS(joined["y"], joined[base_columns]).fit()
    full_columns = [*base_columns, *added]
    full = sm.OLS(joined["y"], joined[full_columns]).fit(
        cov_type="HAC", cov_kwds={"maxlags": int(hac_lags)}
    )
    partial = _partial_r2(joined["y"], joined[base_columns], joined[full_columns])
    return {
        "n": int(full.nobs), "r2_base": float(restricted.rsquared),
        "r2_candidate": float(full.rsquared), "adj_r2_candidate": float(full.rsquared_adj),
        "partial_r2": float(partial),
        "iv_coefficient": float(full.params[added[0]]),
        "iv_p_hac": float(full.pvalues[added[0]]),
    }


def run_horizon_curve(protocol: dict | None = None) -> dict:
    protocol = protocol or load_protocol()
    study = protocol["horizon_curve"]
    # Slice before target construction so no clean-window realization can enter.
    master = pd.read_parquet(ROOT / study["source"]).loc[:study["end"]].copy()
    base = _har_features(master)
    iv = np.log(master["vxn"].where(master["vxn"] > 0)).rename("liv")
    candidate = base.join(iv)
    # Fair comparison: baseline training/scoring uses the candidate's availability.
    base_fair = base.copy()
    base_fair.loc[~np.isfinite(iv), "const"] = np.nan
    rows = []
    forecast_frames = []
    for horizon in study["horizons_trading_sessions"]:
        horizon = int(horizon)
        actual = forward_mean_variance(master["rv_total"], horizon)
        target_log = np.log(actual.where(actual > 0))
        origins = master.loc[study["start"]:study["end"]].index
        origins = origins[actual.reindex(origins).notna() & iv.reindex(origins).notna()]
        base_fc = walk_forward_log_ols(
            base_fair, target_log, actual, origins, label_horizon=horizon,
            min_train_rows=int(study["min_train_rows"]),
        )
        iv_fc = walk_forward_log_ols(
            candidate, target_log, actual, origins, label_horizon=horizon,
            min_train_rows=int(study["min_train_rows"]),
        )
        common = base_fc.index.intersection(iv_fc.index)
        base_fc, iv_fc = base_fc.loc[common], iv_fc.loc[common]
        good = np.isfinite(base_fc["actual_var"]) & np.isfinite(iv_fc["actual_var"])
        base_fc, iv_fc = base_fc.loc[good], iv_fc.loc[good]
        actual_values = base_fc["actual_var"].to_numpy()
        base_loss = qlike_loss(actual_values, base_fc["mean_var"].to_numpy())
        iv_loss = qlike_loss(actual_values, iv_fc["mean_var"].to_numpy())
        dm = dm_test_hac(iv_loss, base_loss, horizon=horizon)
        in_sample = _in_sample_increment(
            target_log.reindex(common), base.reindex(common), candidate.reindex(common),
            max(0, horizon - 1),
        )
        row = {
            "horizon": horizon, "n": int(len(common)),
            "qlike_har": float(base_loss.mean()),
            "qlike_har_iv": float(iv_loss.mean()),
            "qlike_improvement_pct": float(100 * (base_loss.mean() - iv_loss.mean())
                                           / base_loss.mean()),
            "paired_win_rate": float((iv_loss < base_loss).mean()),
            "dm": dm["dm"], "dm_p": dm["p"], **in_sample,
            "first_origin": str(common.min().date()),
            "last_origin": str(common.max().date()),
        }
        rows.append(row)
        for model, forecasts, loss in (("har", base_fc, base_loss),
                                       ("har_iv", iv_fc, iv_loss)):
            saved = forecasts.copy()
            saved["qlike"] = loss
            saved["model"] = model
            saved["horizon"] = horizon
            forecast_frames.append(saved.reset_index())
    payload = {"study": "horizon_curve", "evidence_class": study["evidence_class"],
               "rows": rows}
    data_dir = ROOT / protocol["outputs"]["data_dir"]
    report_dir = ROOT / protocol["outputs"]["report_dir"]
    _write_json(data_dir / "horizon_curve_metrics.json", payload)
    pd.concat(forecast_frames, ignore_index=True).to_parquet(
        data_dir / "horizon_curve_forecasts.parquet", index=False
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Where VXN contributes along the horizon curve", "",
        "**Evidence class: diagnostic forecast-shape measurement.** Targets are built",
        "only from returns through 2025-10-17; the sealed clean phase is never read.", "",
        "| h sessions | n | HAR QLIKE | HAR+VXN QLIKE | improvement | DM p | win rate | partial R² |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['horizon']} | {row['n']} | {row['qlike_har']:.4f} | "
            f"{row['qlike_har_iv']:.4f} | {row['qlike_improvement_pct']:+.2f}% | "
            f"{row['dm_p']:.4g} | {row['paired_win_rate']:.1%} | {row['partial_r2']:.4f} |"
        )
    lines += [
        "", "Every point is reported; no horizon is selected. OOS forecasts are expanding",
        "direct regressions with exact Duan smearing. Training labels enter only after all",
        "h future sessions are complete, and DM uses h-1 overlap lags.",
    ]
    (report_dir / "horizon_curve.md").write_text("\n".join(lines) + "\n")

    import matplotlib.pyplot as plt
    frame = pd.DataFrame(rows)
    fig, left = plt.subplots(figsize=(8, 4.5))
    left.plot(frame["horizon"], frame["qlike_improvement_pct"], marker="o",
              color="#35618f", label="OOS QLIKE improvement")
    left.axhline(0, color="black", linewidth=.8)
    left.set_xlabel("forecast horizon (trading sessions)")
    left.set_ylabel("HAR+VXN QLIKE improvement (%)", color="#35618f")
    right = left.twinx()
    right.plot(frame["horizon"], 100 * frame["partial_r2"], marker="s",
               color="#a64b3c", label="partial R²")
    right.set_ylabel("in-sample partial R² (%)", color="#a64b3c")
    left.set_title("Incremental contribution of the 30-day implied level")
    fig.tight_layout()
    fig.savefig(report_dir / "horizon_curve.png", dpi=180)
    plt.close(fig)
    return payload


def _percentile_interval(draws: np.ndarray) -> list[float]:
    values = np.asarray(draws, dtype=float)
    return [float(np.nanpercentile(values, 2.5)),
            float(np.nanpercentile(values, 97.5))]


def run_vrp_term_structure(protocol: dict | None = None) -> dict:
    protocol = protocol or load_protocol()
    study = protocol["vrp_term_structure"]
    data_dir = ROOT / protocol["outputs"]["data_dir"]
    report_dir = ROOT / protocol["outputs"]["report_dir"]
    spx = pd.read_parquet(data_dir / "spx_daily.parquet").sort_index()
    cboe = pd.read_parquet(data_dir / "cboe_indices.parquet").sort_index()
    close_returns = np.log(spx["close"] / spx["close"].shift(1))
    table = pd.DataFrame(index=spx.index)
    premium_columns = []
    for symbol, horizon in study["horizons_calendar_days"].items():
        horizon = int(horizon)
        implied = align_exact_dates(cboe[symbol], spx.index)
        realized = forward_calendar_realized_vol(close_returns, horizon)
        table[f"{symbol}_implied"] = implied
        table[f"{symbol}_realized"] = realized
        column = f"{symbol}_premium"
        table[column] = implied - realized
        premium_columns.append(column)
    table["trailing_5d_return"] = np.log(spx["close"] / spx["close"].shift(5))
    table["negative_leverage_state"] = table["trailing_5d_return"] < 0
    required = [column for symbol in study["horizons_calendar_days"]
                for column in (f"{symbol}_implied", f"{symbol}_realized",
                               f"{symbol}_premium")]
    sample = table.loc[study["start"]:study["end"]].dropna(subset=required)
    if sample.empty:
        raise ValueError("no common SPX VRP origins")
    parameters = study["uncertainty"]
    block, draws, seed = 21, 5000, 20260812
    values = sample[premium_columns].to_numpy()
    boot_mean = moving_block_bootstrap(
        values, block=block, draws=draws, seed=seed,
        statistic=lambda array: np.mean(array, axis=0),
    )
    state_values = np.column_stack([
        values, sample["negative_leverage_state"].astype(float).to_numpy()
    ])

    def state_difference(array):
        premiums, state = array[:, :-1], array[:, -1].astype(bool)
        if state.all() or (~state).all():
            return np.full(premiums.shape[1], np.nan)
        return premiums[state].mean(axis=0) - premiums[~state].mean(axis=0)

    boot_difference = moving_block_bootstrap(
        state_values, block=block, draws=draws, seed=seed + 1,
        statistic=state_difference,
    )
    rows = []
    state = sample["negative_leverage_state"]
    for position, (symbol, horizon) in enumerate(study["horizons_calendar_days"].items()):
        premium = sample[f"{symbol}_premium"]
        negative_mean = float(premium[state].mean())
        nonnegative_mean = float(premium[~state].mean())
        rows.append({
            "symbol": symbol, "horizon_calendar_days": int(horizon),
            "n": int(len(sample)), "implied_mean": float(sample[f"{symbol}_implied"].mean()),
            "realized_mean": float(sample[f"{symbol}_realized"].mean()),
            "premium_mean": float(premium.mean()), "premium_median": float(premium.median()),
            "premium_mean_ci95": _percentile_interval(boot_mean[:, position]),
            "negative_state_n": int(state.sum()),
            "nonnegative_state_n": int((~state).sum()),
            "negative_state_premium": negative_mean,
            "nonnegative_state_premium": nonnegative_mean,
            "negative_minus_nonnegative": negative_mean - nonnegative_mean,
            "state_difference_ci95": _percentile_interval(boot_difference[:, position]),
        })
    payload = {
        "study": "vrp_term_structure", "evidence_class": study["evidence_class"],
        "common_n": int(len(sample)), "first_origin": str(sample.index.min().date()),
        "last_origin": str(sample.index.max().date()), "rows": rows,
        "uncertainty": {"method": "circular moving-block bootstrap",
                        "block_sessions": block, "draws": draws, "seed": seed},
    }
    _write_json(data_dir / "vrp_term_structure_metrics.json", payload)
    sample.to_parquet(data_dir / "vrp_term_structure_observations.parquet")
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SPX volatility-risk-premium term structure", "",
        "**Evidence class: descriptive SPX measurement.** This matches official Cboe",
        "9-, 30-, and 93-calendar-day implied-volatility indices to subsequent SPX",
        "close-to-close realized volatility on one common-origin sample.", "",
        "| horizon | common n | implied vol | realized vol | premium (95% block CI) | premium after negative 5d return | otherwise | difference (95% block CI) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ci, diff_ci = row["premium_mean_ci95"], row["state_difference_ci95"]
        lines.append(
            f"| {row['horizon_calendar_days']}d | {row['n']} | {row['implied_mean']:.2f} | "
            f"{row['realized_mean']:.2f} | {row['premium_mean']:+.2f} "
            f"[{ci[0]:+.2f}, {ci[1]:+.2f}] | {row['negative_state_premium']:+.2f} | "
            f"{row['nonnegative_state_premium']:+.2f} | "
            f"{row['negative_minus_nonnegative']:+.2f} [{diff_ci[0]:+.2f}, {diff_ci[1]:+.2f}] |"
        )
    lines += [
        "", "The premium is an index-level implied-minus-realized volatility difference,",
        "not an option P&L. It excludes skew, jumps, delta hedging, transaction costs,",
        "margin, and the path dependence that determines whether it can be harvested.",
    ]
    (report_dir / "vrp_term_structure.md").write_text("\n".join(lines) + "\n")

    import matplotlib.pyplot as plt
    frame = pd.DataFrame(rows)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(frame["horizon_calendar_days"], frame["premium_mean"], marker="o",
              label="all origins", color="#222222")
    axis.plot(frame["horizon_calendar_days"], frame["negative_state_premium"],
              marker="s", label="negative trailing 5d return", color="#a64b3c")
    axis.plot(frame["horizon_calendar_days"], frame["nonnegative_state_premium"],
              marker="^", label="nonnegative trailing 5d return", color="#35618f")
    axis.axhline(0, color="black", linewidth=.8)
    axis.set_xlabel("calendar horizon (days)")
    axis.set_ylabel("implied minus realized volatility points")
    axis.set_title("SPX volatility-risk-premium term structure")
    axis.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "vrp_term_structure.png", dpi=180)
    plt.close(fig)
    return payload


def _event_residual_effect(values: np.ndarray) -> float:
    residual = values[:, 0]
    event = values[:, 1].astype(bool)
    if event.all() or (~event).all():
        return np.nan
    return float(residual[event].mean() - residual[~event].mean())


def run_single_name_earnings(protocol: dict | None = None) -> dict:
    protocol = protocol or load_protocol()
    study = protocol["single_name_earnings"]
    data_dir = ROOT / protocol["outputs"]["data_dir"]
    report_dir = ROOT / protocol["outputs"]["report_dir"]
    cboe = pd.read_parquet(data_dir / "cboe_indices.parquet")
    top = pd.read_parquet(ROOT / protocol["quarterly_top25"]["output"])
    events = pd.read_parquet(data_dir / "top25_earnings_events.parquet")
    rows, saved_frames, bootstrap_effects = [], [], []
    for offset, (asset, definition) in enumerate(study["matched_iv_family"].items()):
        daily = pd.read_parquet(data_dir / f"{asset.lower()}_daily.parquet")
        rv = gk_plus_overnight(daily)
        iv_symbol = str(definition["iv"]).lower()
        frame = rv.copy()
        frame["own_iv"] = align_exact_dates(cboe[iv_symbol], frame.index)
        design = build_single_name_design(frame)
        target_log = frame["log_rv"].shift(-1)
        target_var = frame["rv_total"].shift(-1)
        origins = frame.loc[study["start"]:study["end"]].index
        forecasts = walk_forward_log_ols(
            design, target_log, target_var, origins, label_horizon=1,
            min_train_rows=int(study["min_train_rows"]),
        )
        issuer = str(definition["issuer_id"])
        membership = assign_top25_asof(top, forecasts.index)
        eligible_origins = pd.DatetimeIndex(
            membership.loc[membership["issuer_id"] == issuer, "origin"].unique()
        )
        eligible = forecasts.loc[forecasts.index.intersection(eligible_origins)].copy()
        eligible_events = events.loc[events["issuer_id"] == issuer]
        event_origins = set(pd.to_datetime(eligible_events["origin"]))
        eligible["is_earnings"] = eligible.index.isin(event_origins)
        eligible["log_forecast_residual"] = np.log(
            eligible["actual_var"] / eligible["mean_var"]
        )
        eligible["qlike"] = qlike_loss(
            eligible["actual_var"].to_numpy(), eligible["mean_var"].to_numpy()
        ) if len(eligible) else np.array([])
        if eligible["is_earnings"].any() and (~eligible["is_earnings"]).any():
            event_mean = float(eligible.loc[eligible["is_earnings"],
                                             "log_forecast_residual"].mean())
            other_mean = float(eligible.loc[~eligible["is_earnings"],
                                             "log_forecast_residual"].mean())
            effect = event_mean - other_mean
            array = eligible[["log_forecast_residual", "is_earnings"]].to_numpy(dtype=float)
            boot = moving_block_bootstrap(
                array, block=21, draws=5000, seed=20260812 + offset,
                statistic=_event_residual_effect,
            )
            ci = _percentile_interval(boot)
            bootstrap_effects.append(boot)
        else:
            event_mean = other_mean = effect = np.nan
            ci = [np.nan, np.nan]
        rows.append({
            "asset": asset, "issuer_id": issuer, "iv_symbol": definition["iv"],
            "eligible_n": int(len(eligible)),
            "event_n": int(eligible["is_earnings"].sum()) if len(eligible) else 0,
            "non_event_n": int((~eligible["is_earnings"]).sum()) if len(eligible) else 0,
            "mean_qlike": float(eligible["qlike"].mean()) if len(eligible) else np.nan,
            "event_mean_log_residual": event_mean,
            "non_event_mean_log_residual": other_mean,
            "event_effect_log_variance": effect,
            "event_effect_variance_pct": float(100 * np.expm1(effect))
            if np.isfinite(effect) else np.nan,
            "effect_ci95": ci,
            "first_eligible_origin": str(eligible.index.min().date()) if len(eligible) else None,
            "last_eligible_origin": str(eligible.index.max().date()) if len(eligible) else None,
        })
        eligible["asset"] = asset
        if len(eligible):
            eligible.index.name = "origin"
            saved_frames.append(eligible.reset_index())
    finite_rows = [row for row in rows if np.isfinite(row["event_effect_log_variance"])]
    pooled_effect = float(np.mean([row["event_effect_log_variance"] for row in finite_rows])) \
        if finite_rows else np.nan
    if bootstrap_effects:
        pooled_boot = np.nanmean(np.column_stack(bootstrap_effects), axis=1)
        pooled_ci = _percentile_interval(pooled_boot)
    else:
        pooled_ci = [np.nan, np.nan]
    pooled = {
        "assets": [row["asset"] for row in finite_rows],
        "equal_asset_effect_log_variance": pooled_effect,
        "equal_asset_effect_variance_pct": float(100 * np.expm1(pooled_effect))
        if np.isfinite(pooled_effect) else np.nan,
        "effect_ci95": pooled_ci,
    }
    payload = {
        "study": "single_name_earnings", "evidence_class": study["evidence_class"],
        "rows": rows, "equal_asset_pool": pooled,
        "event_label_fence": study["crucial_fence"],
    }
    _write_json(data_dir / "single_name_earnings_metrics.json", payload)
    if saved_frames:
        pd.concat(saved_frames, ignore_index=True).to_parquet(
            data_dir / "single_name_earnings_forecasts.parquet", index=False
        )
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Single-name earnings residual after own implied volatility", "",
        "**Evidence class: cross-sectional mechanism diagnostic.** Forecasts are fixed",
        "without earnings inputs. Realized announcement labels are attached afterward.",
        "An issuer-session is retained only while the latest SEC-accepted quarterly QQQ",
        "snapshot ranks that issuer in the top 25.", "",
        "| asset / own IV | eligible n | event n | event − other log residual | variance ratio effect | 95% block CI (log) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["eligible_n"]:
            ci = row["effect_ci95"]
            lines.append(
                f"| {row['asset']} / {row['iv_symbol']} | {row['eligible_n']} | "
                f"{row['event_n']} | {row['event_effect_log_variance']:+.3f} | "
                f"{row['event_effect_variance_pct']:+.1f}% | [{ci[0]:+.3f}, {ci[1]:+.3f}] |"
            )
        else:
            lines.append(f"| {row['asset']} / {row['iv_symbol']} | 0 | 0 | n/a | n/a | n/a |")
    lines += [
        "", f"Equal-asset pooled effect across {', '.join(pooled['assets']) or 'no assets'}: "
        f"**{pooled_effect:+.3f} log variance ({pooled['equal_asset_effect_variance_pct']:+.1f}%)** "
        f"with 95% block interval [{pooled_ci[0]:+.3f}, {pooled_ci[1]:+.3f}].",
        "", "This asks whether earnings variance remains unusually large after each name's",
        "own 30-day Cboe implied level. It does not claim an executable earnings-date",
        "forecast: the historical announcement archive is not versioned as-of each origin.",
    ]
    (report_dir / "single_name_earnings.md").write_text("\n".join(lines) + "\n")
    return payload


def _comparison_row(name: str, forecasts: pd.DataFrame,
                    baseline: pd.DataFrame, horizon: int) -> dict:
    common = forecasts.index.intersection(baseline.index)
    candidate = forecasts.loc[common]
    base = baseline.loc[common]
    finite = (
        np.isfinite(candidate["actual_var"])
        & np.isfinite(candidate["mean_var"])
        & np.isfinite(base["mean_var"])
    )
    candidate, base = candidate.loc[finite], base.loc[finite]
    actual = candidate["actual_var"].to_numpy()
    candidate_loss = qlike_loss(actual, candidate["mean_var"].to_numpy())
    baseline_loss = qlike_loss(actual, base["mean_var"].to_numpy())
    dm = dm_test_hac(candidate_loss, baseline_loss, horizon=horizon)
    return {
        "name": name, "n": int(len(candidate)),
        "qlike": float(candidate_loss.mean()),
        "baseline_qlike": float(baseline_loss.mean()),
        "improvement_pct": float(100 * (baseline_loss.mean() - candidate_loss.mean())
                                 / baseline_loss.mean()),
        "paired_win_rate": float((candidate_loss < baseline_loss).mean()),
        "dm": dm["dm"], "dm_p": dm["p"],
        "index": candidate.index,
        "candidate_loss": candidate_loss,
        "baseline_loss": baseline_loss,
    }


def run_spx_term_slope_replication(protocol: dict | None = None) -> dict:
    protocol = protocol or load_protocol()
    study = protocol["spx_term_slope_replication"]
    data_dir = ROOT / protocol["outputs"]["data_dir"]
    report_dir = ROOT / protocol["outputs"]["report_dir"]
    spx_daily = pd.read_parquet(data_dir / "spx_daily.parquet").loc[:study["score_end"]]
    frame = gk_plus_overnight(spx_daily).loc[study["training_start"]:study["score_end"]]
    cboe = pd.read_parquet(data_dir / "cboe_indices.parquet")
    delay = int(study["cboe_close_delay_sessions"])
    vix = align_exact_dates(cboe["vix"], frame.index).shift(delay)
    vix9d = align_exact_dates(cboe["vix9d"], frame.index).shift(delay)
    slope = np.log(vix9d.where(vix9d > 0) / vix.where(vix > 0)).rename("term_slope")
    base = _har_features(frame, iv=vix, leverage=True)
    unconditional = base.join(slope)
    dislocation = base.join(slope.clip(lower=0.0).rename("front_dislocation"))
    availability = np.isfinite(slope)
    base_fair = base.copy()
    base_fair.loc[~availability, "const"] = np.nan
    target_log = frame["log_rv"].shift(-1)
    target_var = frame["rv_total"].shift(-1)
    origins = frame.loc[study["score_start"]:study["score_end"]].index
    origins = origins[target_var.reindex(origins).notna()]
    baseline_fc = walk_forward_log_ols(
        base_fair, target_log, target_var, origins, label_horizon=1,
        min_train_rows=int(study["min_train_rows"]),
    )
    unconditional_fc = walk_forward_log_ols(
        unconditional, target_log, target_var, origins, label_horizon=1,
        min_train_rows=int(study["min_train_rows"]),
    )
    dislocation_fc = walk_forward_log_ols(
        dislocation, target_log, target_var, origins, label_horizon=1,
        min_train_rows=int(study["min_train_rows"]),
    )
    common = baseline_fc.index.intersection(unconditional_fc.index).intersection(
        dislocation_fc.index
    )
    baseline_fc = baseline_fc.loc[common]
    unconditional_fc = unconditional_fc.loc[common]
    dislocation_fc = dislocation_fc.loc[common]
    unconditional_metrics = _comparison_row(
        "unconditional_slope", unconditional_fc, baseline_fc, 1
    )
    dislocation_metrics = _comparison_row(
        "dislocation_only", dislocation_fc, baseline_fc, 1
    )
    # Compare both registered additions directly on their common rows.
    actual = dislocation_fc.loc[common, "actual_var"].to_numpy()
    dislocation_loss = qlike_loss(actual, dislocation_fc.loc[common, "mean_var"].to_numpy())
    unconditional_loss = qlike_loss(actual, unconditional_fc.loc[common, "mean_var"].to_numpy())
    inversion = slope.reindex(common) > 0
    baseline_loss = qlike_loss(actual, baseline_fc.loc[common, "mean_var"].to_numpy())
    by_state = {}
    for label, mask in (("inverted", inversion.to_numpy()),
                        ("not_inverted", (~inversion).to_numpy())):
        by_state[label] = {
            "n": int(mask.sum()),
            "baseline_qlike": float(baseline_loss[mask].mean()) if mask.any() else np.nan,
            "dislocation_qlike": float(dislocation_loss[mask].mean()) if mask.any() else np.nan,
            "improvement_pct": float(
                100 * (baseline_loss[mask].mean() - dislocation_loss[mask].mean())
                / baseline_loss[mask].mean()
            ) if mask.any() else np.nan,
        }
    success = bool(
        dislocation_metrics["qlike"] < dislocation_metrics["baseline_qlike"]
        and dislocation_metrics["dm_p"] < 0.05
        and dislocation_metrics["paired_win_rate"] > 0.5
        and dislocation_metrics["qlike"] < unconditional_metrics["qlike"]
    )

    def clean_metrics(row: dict) -> dict:
        return {key: value for key, value in row.items()
                if key not in {"index", "candidate_loss", "baseline_loss"}}

    payload = {
        "study": "spx_term_slope_replication",
        "evidence_class": study["evidence_class"],
        "verdict": "PASS" if success else "FAIL",
        "baseline_qlike": float(baseline_loss.mean()),
        "unconditional_slope": clean_metrics(unconditional_metrics),
        "dislocation_only": clean_metrics(dislocation_metrics),
        "dislocation_minus_unconditional_qlike": float(
            dislocation_loss.mean() - unconditional_loss.mean()
        ),
        "by_origin_state": by_state,
        "first_origin": str(common.min().date()),
        "last_origin": str(common.max().date()),
        "cboe_delay_sessions": delay,
        "success_rule": study["success_requires"],
    }
    _write_json(data_dir / "spx_term_slope_metrics.json", payload)
    saved = []
    for name, forecasts, loss in (
        ("baseline", baseline_fc, baseline_loss),
        ("unconditional_slope", unconditional_fc, unconditional_loss),
        ("dislocation_only", dislocation_fc, dislocation_loss),
    ):
        output = forecasts.loc[common].copy()
        output["qlike"] = loss
        output["model"] = name
        output["front_inverted"] = inversion.to_numpy()
        saved.append(output.reset_index())
    pd.concat(saved, ignore_index=True).to_parquet(
        data_dir / "spx_term_slope_forecasts.parquet", index=False
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SPX regime-conditional VIX9D/VIX replication", "",
        f"**Registered verdict: {payload['verdict']}.** This is a different asset and a",
        "2014-2015 score window that ends before the inspected 2016-2025 NDX study.",
        "Both Cboe closes are delayed one full session at the 16:00 ET origin.", "",
        "| model | n | QLIKE | improvement vs baseline | DM p | paired win rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metrics in (unconditional_metrics, dislocation_metrics):
        lines.append(
            f"| {metrics['name']} | {metrics['n']} | {metrics['qlike']:.4f} | "
            f"{metrics['improvement_pct']:+.2f}% | {metrics['dm_p']:.4g} | "
            f"{metrics['paired_win_rate']:.1%} |"
        )
    lines += [
        "", "| origin state | n | baseline QLIKE | dislocation QLIKE | improvement |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, row in by_state.items():
        lines.append(
            f"| {label} | {row['n']} | {row['baseline_qlike']:.4f} | "
            f"{row['dislocation_qlike']:.4f} | {row['improvement_pct']:+.2f}% |"
        )
    lines += [
        "", "The primary rule required the dislocation-only form to beat baseline and",
        "the unconditional form, DM p<0.05, and a paired win rate above 50%.",
    ]
    (report_dir / "spx_term_slope_replication.md").write_text("\n".join(lines) + "\n")
    return payload


def _representative_name(group: pd.DataFrame) -> str:
    ranked = group.sort_values(["pct_value", "name"], ascending=[False, True])
    return str(ranked.iloc[0]["name"])


def build_quarterly_top25(holdings: pd.DataFrame, keep: int = 25) -> pd.DataFrame:
    """Aggregate security rows to issuers, then retain each quarter's top 25.

    The first six CUSIP characters identify the issuer. This folds multiple
    listed share classes (notably Alphabet) before ranking.
    """
    required = {
        "accession", "report_date", "accepted_at", "name", "cusip",
        "pct_value", "value_usd", "asset_category",
    }
    if missing := required - set(holdings):
        raise ValueError(f"N-PORT holdings missing columns: {sorted(missing)}")
    frame = holdings.copy()
    frame["pct_value"] = pd.to_numeric(frame["pct_value"], errors="raise")
    frame["value_usd"] = pd.to_numeric(frame["value_usd"], errors="raise")
    frame["cusip"] = frame["cusip"].fillna("").astype(str).str.strip().str.upper()
    frame = frame.loc[
        (frame["asset_category"] == "EC")
        & (frame["pct_value"] > 0)
        & (frame["cusip"].str.len() >= 6)
    ].copy()
    frame["issuer_id"] = frame["cusip"].str[:6]
    frame["report_date"] = pd.to_datetime(frame["report_date"]).dt.tz_localize(None).dt.normalize()
    frame["accepted_at"] = pd.to_datetime(frame["accepted_at"], utc=True)
    if frame.empty:
        raise ValueError("no positive equity holdings")
    keys = ["accession", "report_date", "accepted_at", "issuer_id"]
    rows = []
    for key, group in frame.groupby(keys, sort=True, dropna=False):
        rows.append({
            "accession": key[0], "report_date": key[1], "accepted_at": key[2],
            "issuer_id": key[3], "name": _representative_name(group),
            "pct_value": float(group["pct_value"].sum()),
            "value_usd": float(group["value_usd"].sum()),
            "share_classes": int(len(group)),
            "cusips": "|".join(sorted(group["cusip"].unique())),
        })
    issuers = pd.DataFrame(rows)
    equity_total = issuers.groupby("accession")["pct_value"].sum().rename("equity_pct_value")
    issuers = issuers.sort_values(
        ["accession", "pct_value", "issuer_id"],
        ascending=[True, False, True], kind="stable",
    )
    issuers["rank"] = issuers.groupby("accession").cumcount() + 1
    out = issuers.loc[issuers["rank"] <= int(keep)].copy()
    counts = out.groupby("accession").size()
    if not (counts == int(keep)).all():
        bad = counts[counts != int(keep)].to_dict()
        raise ValueError(f"snapshots do not contain exactly {keep} issuers: {bad}")
    if out.duplicated(["accession", "issuer_id"]).any():
        raise ValueError("duplicate issuer inside top-25 snapshot")
    out = out.join(equity_total, on="accession")
    top_total = out.groupby("accession")["pct_value"].transform("sum")
    out["top25_pct_value"] = top_total
    out["weight_within_top25"] = out["pct_value"] / top_total
    return out.sort_values(["report_date", "rank"], kind="stable").reset_index(drop=True)


def assign_top25_asof(top25: pd.DataFrame, origins: Iterable,
                      origin_hour_et: int = 16) -> pd.DataFrame:
    """Expand the latest SEC-accepted top-25 snapshot over supplied origins."""
    required = {"accession", "report_date", "accepted_at", "issuer_id", "rank"}
    if missing := required - set(top25):
        raise ValueError(f"top-25 frame missing columns: {sorted(missing)}")
    dates = _normal_dates(origins)
    left = pd.DataFrame({"origin": dates})
    left["origin_at"] = (
        left["origin"].dt.tz_localize("America/New_York")
        + pd.Timedelta(hours=origin_hour_et)
    ).dt.tz_convert("UTC")
    snapshots = top25[["accession", "report_date", "accepted_at"]].drop_duplicates()
    snapshots["accepted_at"] = pd.to_datetime(snapshots["accepted_at"], utc=True)
    snapshots = snapshots.sort_values("accepted_at")
    if snapshots["accepted_at"].duplicated().any():
        raise ValueError("duplicate snapshot acceptance timestamp")
    assigned = pd.merge_asof(
        left.sort_values("origin_at"), snapshots,
        left_on="origin_at", right_on="accepted_at",
        direction="backward", allow_exact_matches=True,
    ).dropna(subset=["accession"])
    if assigned.empty:
        return pd.DataFrame(columns=["origin", *top25.columns])
    result = assigned.merge(
        top25.drop(columns=["report_date", "accepted_at"]),
        on="accession", how="left", validate="many_to_many",
    )
    result = result.drop(columns="origin_at")
    result["snapshot_age_days"] = (
        result["origin"] - pd.to_datetime(result["report_date"])
    ).dt.days
    if (pd.to_datetime(result["accepted_at"], utc=True)
            > (result["origin"].dt.tz_localize("America/New_York")
               + pd.Timedelta(hours=origin_hour_et)).dt.tz_convert("UTC")).any():
        raise AssertionError("future filing escaped as-of join")
    return result.sort_values(["origin", "rank"], kind="stable").reset_index(drop=True)


def build_top25_earnings_events(events: pd.DataFrame, top25: pd.DataFrame,
                                trading_sessions: Iterable,
                                symbol_map: pd.DataFrame) -> pd.DataFrame:
    """Attach realized announcements only to the historically eligible issuer."""
    required_events = {"date", "ticker", "session"}
    if missing := required_events - set(events):
        raise ValueError(f"earnings events missing columns: {sorted(missing)}")
    if missing := {"issuer_id", "ticker"} - set(symbol_map):
        raise ValueError(f"symbol map missing columns: {sorted(missing)}")
    sessions = _normal_dates(trading_sessions).sort_values().unique()
    mapping = symbol_map[["issuer_id", "ticker"]].copy()
    mapping["ticker"] = mapping["ticker"].astype(str).str.upper()
    if mapping["ticker"].duplicated().any() or mapping["issuer_id"].duplicated().any():
        raise ValueError("symbol map must be one-to-one")
    frame = events.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["event_date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    frame = frame.merge(mapping, on="ticker", how="inner", validate="many_to_one")
    rows = []
    for row in frame.itertuples(index=False):
        if str(row.session).lower() == "bmo":
            target = row.event_date if row.event_date in sessions else None
        else:
            later = sessions[sessions > row.event_date]
            target = later[0] if len(later) else None
        if target is None:
            continue
        prior = sessions[sessions < target]
        if not len(prior):
            continue
        rows.append({
            "event_date": row.event_date,
            "target_date": target,
            "origin": prior[-1],
            "ticker": row.ticker,
            "issuer_id": row.issuer_id,
            "session": str(row.session).lower(),
        })
    if not rows:
        return pd.DataFrame(columns=[
            "event_date", "target_date", "origin", "ticker", "issuer_id",
            "session", "accession", "report_date", "accepted_at", "rank",
            "pct_value",
        ])
    mapped = pd.DataFrame(rows).drop_duplicates(["target_date", "issuer_id"])
    membership = assign_top25_asof(top25, mapped["origin"].unique())
    result = mapped.merge(
        membership,
        on=["origin", "issuer_id"], how="inner", validate="many_to_one",
        suffixes=("", "_holding"),
    )
    return result.sort_values(["target_date", "rank"], kind="stable").reset_index(drop=True)


def _normal_yahoo_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(-1):
            data = data.xs(ticker, axis=1, level=-1)
        else:
            data.columns = data.columns.get_level_values(0)
    data.columns = [str(column).strip().lower() for column in data.columns]
    required = ["open", "high", "low", "close"]
    if missing := set(required) - set(data):
        raise ValueError(f"{ticker}: Yahoo history missing {sorted(missing)}")
    keep = required + (["volume"] if "volume" in data else [])
    data = data.loc[:, keep].dropna(subset=required)
    index = pd.DatetimeIndex(pd.to_datetime(data.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    data.index = index.normalize()
    data.index.name = "date"
    if data.index.duplicated().any():
        raise ValueError(f"{ticker}: duplicate Yahoo dates")
    if data.empty or (data[required] <= 0).any().any():
        raise ValueError(f"{ticker}: invalid Yahoo OHLC history")
    return data.sort_index()


def _fetch_yahoo(ticker: str, start: str, end: str,
                  *, auto_adjust: bool) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        ticker, start=start, end=end, auto_adjust=auto_adjust,
        actions=False, repair=True, progress=False, threads=False,
    )
    return _normal_yahoo_frame(raw, ticker)


def fetch_external_inputs(protocol: dict | None = None) -> dict:
    """Fetch the frozen SPX surface and matched single-name source family."""
    protocol = protocol or load_protocol()
    directory = ROOT / protocol["outputs"]["data_dir"]
    raw_dir = directory / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "retrieved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "protocol_version": protocol["protocol_version"],
        "sources": {},
    }

    symbols = {}
    vrp = protocol["vrp_term_structure"]
    for name, url in vrp["implied_sources"].items():
        symbols[name.upper()] = url
    for asset, definition in protocol["single_name_earnings"]["matched_iv_family"].items():
        iv_symbol = definition["iv"]
        symbols[iv_symbol] = protocol["single_name_earnings"]["iv_url_template"].format(
            symbol=iv_symbol
        )
    session = requests.Session()
    session.headers.update({"User-Agent": "ndx-vol-experiment/1.0 research"})
    cboe_series = {}
    for symbol, url in symbols.items():
        response = session.get(url, timeout=45)
        response.raise_for_status()
        path = raw_dir / f"{symbol}_History.csv"
        path.write_bytes(response.content)
        series = parse_cboe_history(response.text, symbol)
        cboe_series[symbol.lower()] = series
        manifest["sources"][symbol.lower()] = {
            "provider": "Cboe", "url": url, "rows": int(len(series)),
            "first_date": str(series.index.min().date()),
            "last_date": str(series.index.max().date()),
            "sha256": _sha256(path),
        }
        time.sleep(0.1)
    surface = pd.DataFrame(cboe_series).sort_index()
    surface.index.name = "date"
    surface_path = directory / "cboe_indices.parquet"
    surface.to_parquet(surface_path)

    end = (pd.Timestamp(max(vrp["end"], protocol["single_name_earnings"]["end"]))
           + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    spx = _fetch_yahoo("^GSPC", "2009-01-01", end, auto_adjust=False)
    spx_path = directory / "spx_daily.parquet"
    spx.to_parquet(spx_path)
    manifest["sources"]["spx"] = {
        "provider": "Yahoo Finance via yfinance", "ticker": "^GSPC",
        "auto_adjust": False, "rows": int(len(spx)),
        "first_date": str(spx.index.min().date()),
        "last_date": str(spx.index.max().date()), "sha256": _sha256(spx_path),
    }

    for asset in protocol["single_name_earnings"]["matched_iv_family"]:
        daily = _fetch_yahoo(asset, "2009-01-01", end, auto_adjust=True)
        path = directory / f"{asset.lower()}_daily.parquet"
        daily.to_parquet(path)
        manifest["sources"][asset.lower()] = {
            "provider": "Yahoo Finance via yfinance", "ticker": asset,
            "auto_adjust": True, "rows": int(len(daily)),
            "first_date": str(daily.index.min().date()),
            "last_date": str(daily.index.max().date()), "sha256": _sha256(path),
        }
    manifest_path = directory / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _infer_yahoo_earnings(ticker: str, limit: int = 40) -> pd.DataFrame:
    import yfinance as yf

    dates = yf.Ticker(ticker).get_earnings_dates(limit=limit)
    if dates is None or not len(dates):
        return pd.DataFrame(columns=["date", "ticker", "session"])
    rows = []
    for value in dates.index:
        timestamp = pd.Timestamp(value)
        minutes = timestamp.hour * 60 + timestamp.minute
        if minutes and minutes < 9 * 60 + 30:
            session = "bmo"
        elif minutes >= 16 * 60:
            session = "amc"
        else:
            session = "unknown"
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_localize(None)
        rows.append({"date": timestamp.normalize(), "ticker": ticker,
                     "session": session})
    return pd.DataFrame(rows).drop_duplicates(["date", "ticker"])


def fetch_top25_earnings(protocol: dict | None = None) -> dict:
    """Build a realized-event panel for the accepted quarterly top-25 universe."""
    protocol = protocol or load_protocol()
    directory = ROOT / protocol["outputs"]["data_dir"]
    directory.mkdir(parents=True, exist_ok=True)
    top_path = ROOT / protocol["quarterly_top25"]["output"]
    top = pd.read_parquet(top_path) if top_path.exists() else write_top25(protocol)
    mapping = pd.read_csv(ROOT / "calendars/top25_symbol_map.csv", comment="#", dtype=str)
    existing = pd.read_csv(ROOT / "calendars/earnings_top.csv", comment="#",
                           parse_dates=["date"])
    existing = existing[["date", "ticker", "session"]].copy()
    existing["ticker"] = existing["ticker"].str.upper()
    frames = [existing]
    existing_symbols = set(existing["ticker"])
    failures = []
    for row in mapping.itertuples(index=False):
        ticker = str(row.ticker).upper()
        if ticker in existing_symbols or str(getattr(row, "active", "true")).lower() != "true":
            continue
        try:
            fetched = _infer_yahoo_earnings(ticker, limit=40)
            if fetched.empty:
                failures.append({"ticker": ticker, "reason": "no dates returned"})
            else:
                frames.append(fetched)
        except Exception as exc:  # one missing issuer is audited, not hidden
            failures.append({"ticker": ticker,
                             "reason": f"{type(exc).__name__}: {exc}"})
        time.sleep(0.25)
    events = pd.concat(frames, ignore_index=True).drop_duplicates(["date", "ticker"])
    study = protocol["single_name_earnings"]
    events = events.loc[
        (events["date"] >= pd.Timestamp("2019-01-01"))
        & (events["date"] <= pd.Timestamp(study["end"]) + pd.Timedelta(days=7))
    ].sort_values(["date", "ticker"])
    events.to_parquet(directory / "top25_earnings_raw.parquet", index=False)

    master = pd.read_parquet(ROOT / protocol["absorption_map"]["source"])
    sessions = master.loc[:protocol["fences"]["ndx_diagnostic_end"]].index
    eligible = build_top25_earnings_events(events, top, sessions, mapping)
    eligible.to_parquet(directory / "top25_earnings_events.parquet", index=False)

    first_accept = pd.to_datetime(top["accepted_at"], utc=True).min()
    session_times = (pd.DatetimeIndex(sessions).tz_localize("America/New_York")
                     + pd.Timedelta(hours=16)).tz_convert("UTC")
    valid_sessions = pd.DatetimeIndex(sessions)[session_times >= first_accept]
    daily = pd.Series(np.nan, index=pd.DatetimeIndex(sessions),
                      name="top25_earnings_weight", dtype=float)
    if len(valid_sessions):
        daily.loc[valid_sessions] = 0.0
    if len(eligible):
        totals = eligible.groupby("target_date")["pct_value"].sum()
        daily.loc[daily.index.intersection(totals.index)] = totals.reindex(
            daily.index.intersection(totals.index)
        )
    daily.to_frame().to_parquet(directory / "top25_earnings_daily.parquet")

    covered = set(events["ticker"])
    required = set(mapping.loc[mapping["active"].str.lower() == "true", "ticker"])
    audit = {
        "raw_events": int(len(events)), "eligible_events": int(len(eligible)),
        "eligible_issuers": int(eligible["issuer_id"].nunique()) if len(eligible) else 0,
        "covered_active_symbols": int(len(required & covered)),
        "required_active_symbols": int(len(required)),
        "missing_active_symbols": sorted(required - covered),
        "fetch_failures": failures,
        "first_point_in_time_session": str(valid_sessions.min().date())
        if len(valid_sessions) else None,
    }
    (directory / "top25_earnings_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    return audit


def write_top25(protocol: dict | None = None) -> pd.DataFrame:
    protocol = protocol or load_protocol()
    source = ROOT / protocol["quarterly_top25"]["source"]
    output = ROOT / protocol["quarterly_top25"]["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    top = build_quarterly_top25(pd.read_parquet(source),
                                int(protocol["quarterly_top25"]["keep"]))
    top.to_parquet(output, index=False)
    return top


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["top25", "fetch-external",
                                            "fetch-top25-earnings", "absorption-map",
                                            "horizon-curve", "vrp-term-structure",
                                            "single-name-earnings",
                                            "spx-term-slope"])
    args = parser.parse_args()
    if args.command == "top25":
        top = write_top25()
        print(
            f"wrote quarterly top 25: {len(top)} rows, "
            f"{top['accession'].nunique()} snapshots, "
            f"{top['issuer_id'].nunique()} issuers"
        )
    elif args.command == "fetch-external":
        manifest = fetch_external_inputs()
        print(f"wrote external inputs: {len(manifest['sources'])} sources")
    elif args.command == "fetch-top25-earnings":
        audit = fetch_top25_earnings()
        print(json.dumps(audit, indent=2, sort_keys=True))
    elif args.command == "absorption-map":
        result = run_absorption_map()
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "horizon-curve":
        result = run_horizon_curve()
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "vrp-term-structure":
        result = run_vrp_term_structure()
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "single-name-earnings":
        result = run_single_name_earnings()
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "spx-term-slope":
        result = run_spx_term_slope_replication()
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
