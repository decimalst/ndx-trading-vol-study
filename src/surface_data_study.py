"""Private normalization and frozen option-surface diagnostic.

The protocol in ``surface_data_study.yaml`` was written and its tests were run
before this implementation existed. Raw Kaggle mirrors never leave the ignored
``data/free_sources/raw`` tree. Only non-reconstructive daily aggregates,
forecasts, hashes, and aggregate metrics may be written.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import pathlib
import zipfile
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "surface_data_study.yaml"

OPTION_COLUMNS = (
    "QUOTE_UNIXTIME", "QUOTE_READTIME", "QUOTE_DATE", "QUOTE_TIME_HOURS",
    "UNDERLYING_LAST", "EXPIRE_DATE", "EXPIRE_UNIX", "DTE",
    "C_DELTA", "C_GAMMA", "C_IV", "C_VOLUME", "C_BID", "C_ASK",
    "STRIKE", "P_BID", "P_ASK", "P_DELTA", "P_GAMMA", "P_IV",
    "P_VOLUME",
)
BASELINE_FEATURES = ("lrv_d", "lrv_w", "lrv_m", "log_atm_iv_30")
AUGMENTED_FEATURES = (
    *BASELINE_FEATURES, "skew_25d", "term_9d_30d",
    "log1p_gamma_weighted_volume",
)


def load_protocol(path: str | pathlib.Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    with pathlib.Path(path).open() as handle:
        spec = yaml.safe_load(handle)
    validate_protocol(spec)
    return spec


def protocol_sha256(path: str | pathlib.Path = DEFAULT_PROTOCOL) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def validate_protocol(spec: dict[str, Any]) -> None:
    if spec.get("status") != "frozen_before_first_empirical_run":
        raise ValueError("surface protocol is not frozen before results")
    if spec.get("evidence_class") != "private_diagnostic_only":
        raise ValueError("surface study must remain private and diagnostic")
    if spec["provenance"]["raw_redistribution"] != "forbidden":
        raise ValueError("raw redistribution must be forbidden")
    shape = spec["shape_construction"]
    if shape["open_interest_available"]:
        raise ValueError("source does not contain open interest")
    if not shape["gamma_weighted_volume_is_not_dealer_gex"]:
        raise ValueError("gamma-volume proxy must not be called dealer GEX")
    if tuple(spec["study"]["baseline_features"]) != BASELINE_FEATURES:
        raise ValueError("baseline feature set drifted")
    if tuple(spec["study"]["augmented_features"]) != AUGMENTED_FEATURES:
        raise ValueError("augmented feature set drifted")
    if int(spec["timing"]["surface_measurement_lag_sessions"]) != 1:
        raise ValueError("surface must be lagged one full session")
    if float(spec["parser_contract"]["quote_time_hours"]) != 16.0:
        raise ValueError("only strict 16:00 snapshots are permitted")
    if spec["study"]["aapl"]["earnings_overlay"] != "omitted":
        raise ValueError("AAPL earnings labels are not point-in-time safe")


def normalize_headers(columns: Iterable[Any]) -> list[str]:
    out = []
    for value in columns:
        text = str(value).lstrip("\ufeff").strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1].strip()
        out.append(text)
    return out


def _canonicalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = normalize_headers(out.columns)
    missing = sorted(set(OPTION_COLUMNS) - set(out.columns))
    if missing:
        raise ValueError(f"option source missing required columns: {missing}")
    return out.loc[:, OPTION_COLUMNS].copy()


def _time_is_strict_1600(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.match(
        r"^\d{4}-\d{2}-\d{2}[ T]16:00(?::00)?$", na=False
    )


def normalize_option_frame(
    frame: pd.DataFrame,
    symbol: str,
    session_dates: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    spec: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the frozen schema, clock, calendar, and quote-quality filters."""
    spec = spec or load_protocol()
    raw = _canonicalize_frame(frame)
    n_input = len(raw)
    numeric = [
        "QUOTE_TIME_HOURS", "UNDERLYING_LAST", "DTE", "C_DELTA",
        "C_GAMMA", "C_IV", "C_VOLUME", "C_BID", "C_ASK", "STRIKE",
        "P_BID", "P_ASK", "P_DELTA", "P_GAMMA", "P_IV", "P_VOLUME",
    ]
    for col in numeric:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["QUOTE_DATE"] = pd.to_datetime(
        raw["QUOTE_DATE"].astype("string").str.strip(), errors="coerce"
    ).dt.normalize()
    raw["EXPIRE_DATE"] = pd.to_datetime(
        raw["EXPIRE_DATE"].astype("string").str.strip(), errors="coerce"
    ).dt.normalize()

    pc = spec["parser_contract"]
    time_ok = (
        np.isclose(raw["QUOTE_TIME_HOURS"], float(pc["quote_time_hours"]),
                   rtol=0.0, atol=1e-12)
        & _time_is_strict_1600(raw["QUOTE_READTIME"])
    )
    rejected_wrong_time = int((~time_ok).sum())
    raw = raw.loc[time_ok].copy()

    sessions = pd.DatetimeIndex(pd.to_datetime(session_dates)).normalize()
    if sessions.has_duplicates or not sessions.is_monotonic_increasing:
        sessions = pd.DatetimeIndex(sorted(set(sessions)))
    session_ok = raw["QUOTE_DATE"].isin(sessions)
    rejected_non_session = int((~session_ok).sum())
    raw = raw.loc[session_ok].copy()

    quote_cols = ["C_BID", "C_ASK", "P_BID", "P_ASK"]
    quote_values = raw[quote_cols].to_numpy(float)
    quote_ok = np.isfinite(quote_values).all(axis=1)
    quote_ok &= (quote_values >= 0.0).all(axis=1)
    quote_ok &= raw["C_BID"].to_numpy(float) <= raw["C_ASK"].to_numpy(float)
    quote_ok &= raw["P_BID"].to_numpy(float) <= raw["P_ASK"].to_numpy(float)
    rejected_invalid_quotes = int((~quote_ok).sum())
    raw = raw.loc[quote_ok].copy()

    dte = raw["DTE"].to_numpy(float)
    core_ok = np.isfinite(dte)
    core_ok &= dte >= float(pc["dte_min"])
    core_ok &= dte <= float(pc["dte_max"])
    core_ok &= np.isfinite(raw["UNDERLYING_LAST"].to_numpy(float))
    core_ok &= raw["UNDERLYING_LAST"].to_numpy(float) > 0.0
    core_ok &= np.isfinite(raw["STRIKE"].to_numpy(float))
    core_ok &= raw["STRIKE"].to_numpy(float) > 0.0
    for col in ("C_IV", "P_IV"):
        core_ok &= np.isfinite(raw[col].to_numpy(float))
        core_ok &= raw[col].to_numpy(float) > 0.0
    rejected_invalid_core = int((~core_ok).sum())
    raw = raw.loc[core_ok].copy()

    duplicate_key = ["QUOTE_DATE", "EXPIRE_DATE", "STRIKE"]
    duplicate_count = int(raw.duplicated(duplicate_key, keep=False).sum())
    if duplicate_count:
        raise ValueError(
            f"{symbol} contains {duplicate_count} duplicate contract rows after filtering"
        )

    raw = raw.rename(columns={
        "QUOTE_DATE": "quote_date", "QUOTE_READTIME": "quote_readtime",
        "QUOTE_TIME_HOURS": "quote_time_hours",
        "UNDERLYING_LAST": "underlying_last", "EXPIRE_DATE": "expire_date",
        "DTE": "dte", "C_DELTA": "c_delta", "C_GAMMA": "c_gamma",
        "C_IV": "c_iv", "C_VOLUME": "c_volume", "C_BID": "c_bid",
        "C_ASK": "c_ask", "STRIKE": "strike", "P_BID": "p_bid",
        "P_ASK": "p_ask", "P_DELTA": "p_delta", "P_GAMMA": "p_gamma",
        "P_IV": "p_iv", "P_VOLUME": "p_volume",
    })
    raw["symbol"] = str(symbol).upper()
    raw = raw.sort_values(["quote_date", "expire_date", "strike"],
                          kind="mergesort").reset_index(drop=True)
    audit = {
        "symbol": str(symbol).upper(), "input_rows": n_input,
        "accepted_rows": len(raw), "rejected_wrong_time": rejected_wrong_time,
        "rejected_non_session": rejected_non_session,
        "rejected_invalid_quotes": rejected_invalid_quotes,
        "rejected_invalid_core": rejected_invalid_core,
        "duplicate_contract_rows": duplicate_count,
    }
    return raw, audit


def _select_maturity(group: pd.DataFrame, key: str,
                     spec: dict[str, Any]) -> tuple[pd.DataFrame, float]:
    rule = spec["shape_construction"]["maturity_selection"][key]
    lo, hi = map(float, rule["permitted_dte"])
    target = float(rule["target_dte"])
    candidates = group.loc[group["dte"].between(lo, hi), ["dte", "expire_date"]]
    candidates = candidates.drop_duplicates()
    if candidates.empty:
        return group.iloc[0:0], math.nan
    candidates = candidates.assign(distance=(candidates["dte"] - target).abs())
    chosen = candidates.sort_values(
        ["distance", "dte", "expire_date"], kind="mergesort"
    ).iloc[0]
    selected = group.loc[
        np.isclose(group["dte"], float(chosen["dte"]), rtol=0.0, atol=1e-9)
        & (group["expire_date"] == chosen["expire_date"])
    ]
    return selected, float(chosen["dte"])


def _match_iv(frame: pd.DataFrame, side: str, target: float,
              spec: dict[str, Any]) -> tuple[float, float]:
    delta_col, iv_col = f"{side}_delta", f"{side}_iv"
    valid = frame.loc[
        np.isfinite(frame[delta_col]) & np.isfinite(frame[iv_col]),
        [delta_col, iv_col, "strike"],
    ].copy()
    if valid.empty:
        return math.nan, math.nan
    valid["delta_error"] = (valid[delta_col] - float(target)).abs()
    valid = valid.loc[
        valid["delta_error"]
        <= float(spec["shape_construction"]["delta_match"]["maximum_absolute_error"])
    ]
    if valid.empty:
        return math.nan, math.nan
    chosen = valid.sort_values(["delta_error", "strike"], kind="mergesort").iloc[0]
    return float(chosen[iv_col]), float(chosen["delta_error"])


def _atm_iv(frame: pd.DataFrame, spec: dict[str, Any]) -> tuple[float, float]:
    rule = spec["shape_construction"]["delta_match"]
    call, ce = _match_iv(frame, "c", float(rule["atm_call"]), spec)
    put, pe = _match_iv(frame, "p", float(rule["atm_put"]), spec)
    if not np.isfinite(call) or not np.isfinite(put):
        return math.nan, math.nan
    return float((call + put) / 2.0), float(max(ce, pe))


def build_daily_surface(frame: pd.DataFrame, symbol: str,
                        spec: dict[str, Any] | None = None) -> pd.DataFrame:
    """Reduce valid full-chain rows to the four frozen daily shape measures."""
    spec = spec or load_protocol()
    rows: list[dict[str, Any]] = []
    dm = spec["shape_construction"]["delta_match"]
    for date, group in frame.groupby("quote_date", sort=True):
        nine, nine_dte = _select_maturity(group, "nine_day", spec)
        thirty, thirty_dte = _select_maturity(group, "thirty_day", spec)
        atm9, atm9_error = _atm_iv(nine, spec)
        atm30, atm30_error = _atm_iv(thirty, spec)
        call25, call25_error = _match_iv(thirty, "c", float(dm["wing_call"]), spec)
        put25, put25_error = _match_iv(thirty, "p", float(dm["wing_put"]), spec)
        skew = put25 - call25 if np.isfinite(put25) and np.isfinite(call25) else math.nan
        term = atm9 - atm30 if np.isfinite(atm9) and np.isfinite(atm30) else math.nan
        cvol = pd.to_numeric(group["c_volume"], errors="coerce").fillna(0.0).clip(lower=0.0)
        pvol = pd.to_numeric(group["p_volume"], errors="coerce").fillna(0.0).clip(lower=0.0)
        cgamma = pd.to_numeric(group["c_gamma"], errors="coerce").abs().fillna(0.0)
        pgamma = pd.to_numeric(group["p_gamma"], errors="coerce").abs().fillna(0.0)
        gamma_volume = float((cgamma * cvol + pgamma * pvol).sum())
        rows.append({
            "symbol": str(symbol).upper(), "quote_date": pd.Timestamp(date),
            "underlying_last": float(group["underlying_last"].median()),
            "atm_iv_30": atm30, "skew_25d": skew, "term_9d_30d": term,
            "gamma_weighted_volume": gamma_volume,
            "gamma_measure_label": "gamma_weighted_volume_not_dealer_gex",
            "selected_dte_9": nine_dte, "selected_dte_30": thirty_dte,
            "atm_delta_error_9": atm9_error, "atm_delta_error_30": atm30_error,
            "call_25d_error": call25_error, "put_25d_error": put25_error,
            "valid_chain_rows": int(len(group)),
        })
    return pd.DataFrame(rows).sort_values("quote_date").reset_index(drop=True)


def add_full_session_lag(daily: pd.DataFrame,
                         session_dates: Sequence[pd.Timestamp] | pd.DatetimeIndex
                         ) -> pd.DataFrame:
    """Map measurement t -> origin t+1 -> target t+2 on exact sessions."""
    sessions = pd.DatetimeIndex(pd.to_datetime(session_dates)).normalize()
    sessions = pd.DatetimeIndex(sorted(set(sessions)))
    position = {date: i for i, date in enumerate(sessions)}
    outputs = []
    groups = daily.groupby("symbol", sort=True) if "symbol" in daily else [(None, daily)]
    for _, part in groups:
        part = part.copy()
        part["quote_date"] = pd.to_datetime(part["quote_date"]).dt.normalize()
        close_map = (
            part.dropna(subset=["underlying_last"])
            .drop_duplicates("quote_date", keep=False)
            .set_index("quote_date")["underlying_last"]
            if "underlying_last" in part else pd.Series(dtype=float)
        )
        rows = []
        for _, row in part.iterrows():
            measurement = pd.Timestamp(row["quote_date"])
            pos = position.get(measurement)
            if pos is None or pos + 2 >= len(sessions):
                continue
            item = row.to_dict()
            item["measurement_date"] = measurement
            item["origin"] = sessions[pos + 1]
            item["target_date"] = sessions[pos + 2]
            item["surface_lag_sessions"] = 1
            item["measurement_session_number"] = int(pos)
            item["origin_session_number"] = int(pos + 1)
            item["target_session_number"] = int(pos + 2)
            if len(close_map):
                item["measurement_underlying_last"] = close_map.get(measurement, np.nan)
                item["origin_underlying_last"] = close_map.get(sessions[pos + 1], np.nan)
                item["target_underlying_last"] = close_map.get(sessions[pos + 2], np.nan)
            rows.append(item)
        outputs.append(pd.DataFrame(rows))
    if not outputs:
        return pd.DataFrame()
    out = pd.concat(outputs, ignore_index=True)
    if len(out):
        if not (pd.to_datetime(out["measurement_date"]) < pd.to_datetime(out["origin"])).all():
            raise AssertionError("surface measurement is not before origin")
        if not (pd.to_datetime(out["origin"]) < pd.to_datetime(out["target_date"])).all():
            raise AssertionError("target is not after origin")
    return out


def aapl_split_audit(daily: pd.DataFrame,
                     spec: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = spec or load_protocol()
    rule = spec["study"]["aapl"]["split_audit"]
    frame = daily.copy()
    frame["quote_date"] = pd.to_datetime(frame["quote_date"]).dt.normalize()
    values = frame.groupby("quote_date")["underlying_last"].median()
    pre, post = pd.Timestamp(rule["pre_session"]), pd.Timestamp(rule["post_session"])
    if pre not in values or post not in values:
        return {
            "status": "not_available", "pre_session": str(pre.date()),
            "post_session": str(post.date()), "within_expected_range": False,
        }
    ratio = float(values.loc[pre] / values.loc[post])
    lo, hi = map(float, rule["expected_pre_over_post_range"])
    return {
        "status": "audited_without_rewriting",
        "pre_session": str(pre.date()), "post_session": str(post.date()),
        "pre_underlying": float(values.loc[pre]),
        "post_underlying": float(values.loc[post]),
        "pre_over_post_ratio": ratio, "expected_range": [lo, hi],
        "within_expected_range": bool(lo <= ratio <= hi),
    }


def build_forecast_design(surface: pd.DataFrame,
                          spec: dict[str, Any] | None = None) -> pd.DataFrame:
    """Build HAR + lagged-surface features and a next-session RV target."""
    spec = spec or load_protocol()
    if surface.empty:
        return pd.DataFrame()
    symbols = surface["symbol"].dropna().unique() if "symbol" in surface else ["UNKNOWN"]
    if len(symbols) != 1:
        raise ValueError("build_forecast_design accepts exactly one symbol")
    frame = surface.copy()
    for col in ("measurement_date", "origin", "target_date"):
        frame[col] = pd.to_datetime(frame[col]).dt.normalize()
    frame = frame.sort_values("origin", kind="mergesort")
    if frame["origin"].duplicated().any():
        raise ValueError("duplicate origins in lagged surface panel")
    if not {"measurement_underlying_last", "origin_underlying_last",
            "target_underlying_last"}.issubset(frame.columns):
        close_map = (
            frame.drop_duplicates("measurement_date", keep=False)
            .set_index("measurement_date")["underlying_last"]
        )
        frame["measurement_underlying_last"] = frame["measurement_date"].map(close_map)
        frame["origin_underlying_last"] = frame["origin"].map(close_map)
        frame["target_underlying_last"] = frame["target_date"].map(close_map)
    for col in ("measurement_underlying_last", "origin_underlying_last",
                "target_underlying_last"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    positive = (
        (frame["measurement_underlying_last"] > 0)
        & (frame["origin_underlying_last"] > 0)
        & (frame["target_underlying_last"] > 0)
    )
    frame = frame.loc[positive].copy()
    origin_return = np.log(
        frame["origin_underlying_last"] / frame["measurement_underlying_last"]
    )
    target_return = np.log(
        frame["target_underlying_last"] / frame["origin_underlying_last"]
    )
    if str(symbols[0]).upper() == "AAPL":
        split = pd.Timestamp(spec["study"]["aapl"]["split_audit"]["post_session"])
        origin_crosses = (
            (frame["measurement_date"] < split) & (frame["origin"] >= split)
        )
        target_crosses = (frame["origin"] < split) & (frame["target_date"] >= split)
        origin_return = origin_return.mask(origin_crosses)
        target_return = target_return.mask(target_crosses)
    frame["rv_origin"] = np.square(origin_return)
    frame["actual_var"] = np.square(target_return)
    floor = float(spec["study"]["log_floor"])
    frame["lrv_d"] = np.log(frame["rv_origin"].clip(lower=floor))
    if "origin_session_number" in frame:
        positions = frame["origin_session_number"].astype(int)
        full = pd.Series(
            frame["rv_origin"].to_numpy(float), index=positions.to_numpy()
        ).reindex(range(int(positions.min()), int(positions.max()) + 1))
        weekly = full.rolling(5, min_periods=5).mean()
        monthly = full.rolling(22, min_periods=22).mean()
        frame["lrv_w"] = np.log(positions.map(weekly).clip(lower=floor))
        frame["lrv_m"] = np.log(positions.map(monthly).clip(lower=floor))
    else:
        # Used only by compact synthetic fixtures; production ingestion always
        # attaches exact session numbers above.
        frame["lrv_w"] = np.log(
            frame["rv_origin"].rolling(5, min_periods=5).mean().clip(lower=floor)
        )
        frame["lrv_m"] = np.log(
            frame["rv_origin"].rolling(22, min_periods=22).mean().clip(lower=floor)
        )
    frame["y_next"] = np.log(frame["actual_var"].clip(lower=floor))
    atm = pd.to_numeric(frame["atm_iv_30"], errors="coerce")
    frame["log_atm_iv_30"] = np.log(atm.where(atm > 0))
    gamma_volume = pd.to_numeric(frame["gamma_weighted_volume"], errors="coerce")
    frame["log1p_gamma_weighted_volume"] = np.log1p(gamma_volume.where(gamma_volume >= 0))
    frame["surface_lag_sessions"] = 1
    if not (frame["measurement_date"] < frame["origin"]).all():
        raise ValueError("surface lookahead: measurement is not before origin")
    if not (frame["origin"] < frame["target_date"]).all():
        raise ValueError("target must follow origin")
    frame = frame.set_index("origin", drop=True)
    frame.index.name = "origin"
    return frame


def _smear(log_prediction: float, residuals: np.ndarray) -> float:
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if not len(residuals):
        raise ValueError("cannot smear without finite training residuals")
    return float(np.exp(log_prediction) * np.mean(np.exp(residuals)))


def _fit_ols(train: pd.DataFrame, row: pd.Series,
             features: Sequence[str]) -> float:
    X = np.column_stack([np.ones(len(train)), train.loc[:, features].to_numpy(float)])
    y = train["y_next"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    x = np.r_[1.0, row.loc[list(features)].to_numpy(float)]
    mu = float(x @ beta)
    return _smear(mu, y - fitted)


def forecast_symbol(design: pd.DataFrame, symbol: str,
                    spec: dict[str, Any] | None = None) -> pd.DataFrame:
    """Expanding same-row baseline and augmented forecasts, with no tuning."""
    spec = spec or load_protocol()
    common = design.dropna(subset=[*AUGMENTED_FEATURES, "y_next", "actual_var",
                                   "measurement_date", "target_date"]).copy()
    common = common.sort_index()
    min_train = int(spec["study"]["min_training_rows"])
    rows = []
    for origin, row in common.iterrows():
        train = common.loc[pd.to_datetime(common["target_date"]) <= pd.Timestamp(origin)]
        if len(train) < min_train:
            continue
        baseline = _fit_ols(train, row, BASELINE_FEATURES)
        augmented = _fit_ols(train, row, AUGMENTED_FEATURES)
        threshold = float(train["actual_var"].quantile(
            float(spec["study"]["tail"]["quantile"]), interpolation="linear"
        ))
        rows.append({
            "symbol": str(symbol).upper(), "origin": pd.Timestamp(origin),
            "measurement_date": pd.Timestamp(row["measurement_date"]),
            "target_date": pd.Timestamp(row["target_date"]),
            "surface_lag_sessions": int(row["surface_lag_sessions"]),
            "measurement_session_number": int(row["measurement_session_number"]),
            "origin_session_number": int(row["origin_session_number"]),
            "target_session_number": int(row["target_session_number"]),
            "actual_var": float(row["actual_var"]),
            "baseline_var": baseline, "augmented_var": augmented,
            "tail_threshold": threshold,
            "tail_event": int(float(row["actual_var"]) > threshold),
            "training_rows": int(len(train)),
        })
    if not rows:
        return pd.DataFrame(columns=[
            "symbol", "measurement_date", "target_date", "actual_var",
            "baseline_var", "augmented_var", "tail_threshold", "tail_event",
            "training_rows", "split",
        ]).rename_axis("origin")
    out = pd.DataFrame(rows).set_index("origin")
    if str(symbol).upper() == "SPY":
        early = spec["study"]["spy"]["early"]
        late = spec["study"]["spy"]["late_confirmation"]
        out["split"] = "outside_frozen_splits"
        out.loc[out.index.to_series().between(pd.Timestamp(early[0]), pd.Timestamp(early[1])),
                "split"] = "early"
        out.loc[out.index.to_series().between(pd.Timestamp(late[0]), pd.Timestamp(late[1])),
                "split"] = "late_confirmation"
    else:
        out["split"] = "all"
    return out


def qlike(actual: Sequence[float], forecast: Sequence[float]) -> np.ndarray:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if (actual < 0).any() or (forecast <= 0).any():
        raise ValueError("QLIKE requires nonnegative actuals and positive forecasts")
    z = np.maximum(actual, 1e-18) / forecast
    return z - np.log(z) - 1.0


def _auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if not n_pos or not n_neg:
        return math.nan
    ranks = pd.Series(scores).rank(method="average").to_numpy(float)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _ranking_metrics(labels: np.ndarray, scores: np.ndarray,
                     fraction: float) -> dict[str, float]:
    n = len(labels)
    base = float(np.mean(labels)) if n else math.nan
    count = max(1, int(math.ceil(n * fraction))) if n else 0
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    top_rate = float(np.mean(np.asarray(labels)[order[:count]])) if count else math.nan
    lift = float(top_rate / base) if base > 0 else math.nan
    return {
        "auc": _auc(labels, scores), "top_decile_lift": lift,
        "top_decile_event_rate": top_rate, "base_rate": base,
    }


def score_forecasts(frame: pd.DataFrame,
                    spec: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = spec or load_protocol()
    required = ["actual_var", "baseline_var", "augmented_var", "tail_event"]
    clean = frame.dropna(subset=required).copy()
    if clean.empty:
        raise ValueError("no common forecast rows to score")
    actual = clean["actual_var"].to_numpy(float)
    base_loss = qlike(actual, clean["baseline_var"].to_numpy(float))
    aug_loss = qlike(actual, clean["augmented_var"].to_numpy(float))
    labels = clean["tail_event"].to_numpy(int)
    fraction = float(spec["study"]["tail"]["top_fraction"])
    base_rank = _ranking_metrics(labels, clean["baseline_var"].to_numpy(float), fraction)
    aug_rank = _ranking_metrics(labels, clean["augmented_var"].to_numpy(float), fraction)
    return {
        "n": int(len(clean)),
        "baseline": {"mean_qlike": float(base_loss.mean()), **base_rank},
        "augmented": {"mean_qlike": float(aug_loss.mean()), **aug_rank},
        "paired": {
            "mean_qlike_difference": float(np.mean(aug_loss - base_loss)),
            "improvement_pct": float(
                100.0 * (base_loss.mean() - aug_loss.mean()) / base_loss.mean()
            ) if base_loss.mean() != 0 else math.nan,
            "win_rate": float(np.mean(aug_loss < base_loss)),
            "auc_difference": float(aug_rank["auc"] - base_rank["auc"]),
            "top_decile_lift_difference": float(
                aug_rank["top_decile_lift"] - base_rank["top_decile_lift"]
            ),
        },
    }


def moving_block_interval(values: Sequence[float], spec: dict[str, Any]) -> list[float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    block = int(spec["study"]["inference"]["moving_block_sessions"])
    draws = int(spec["study"]["inference"]["bootstrap_draws"])
    rng = np.random.default_rng(int(spec["study"]["inference"]["seed"]))
    if n < 2:
        return [math.nan, math.nan]
    block = min(block, n)
    starts = np.arange(n - block + 1)
    means = np.empty(draws, dtype=float)
    blocks_needed = int(math.ceil(n / block))
    for i in range(draws):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([values[s:s + block] for s in chosen])[:n]
        means[i] = sample.mean()
    alpha = 1.0 - float(spec["study"]["inference"]["confidence_level"])
    return [float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2))]


def summarize_split(frame: pd.DataFrame, spec: dict[str, Any]) -> dict[str, Any]:
    metrics = score_forecasts(frame, spec)
    actual = frame["actual_var"].to_numpy(float)
    differential = (
        qlike(actual, frame["augmented_var"].to_numpy(float))
        - qlike(actual, frame["baseline_var"].to_numpy(float))
    )
    interval = moving_block_interval(differential, spec)
    metrics["paired"]["moving_block_ci95"] = interval
    mean = metrics["paired"]["mean_qlike_difference"]
    if mean < 0 and interval[1] < 0:
        verdict = "IMPROVES"
    elif mean > 0 and interval[0] > 0:
        verdict = "WORSE"
    else:
        verdict = "INCONCLUSIVE"
    metrics["verdict"] = verdict
    return metrics


def hash_file(path: str | pathlib.Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_source_manifest(
    source_paths: dict[str, Sequence[pathlib.Path]],
    output: str | pathlib.Path,
    spec: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = spec or load_protocol()
    records: dict[str, list[dict[str, Any]]] = {}
    for symbol, paths in sorted(source_paths.items()):
        seen = set()
        records[symbol] = []
        for raw_path in paths:
            path = pathlib.Path(raw_path)
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            records[symbol].append({
                "name": path.name, "bytes": path.stat().st_size,
                "sha256": hash_file(path),
            })
    manifest = {
        "protocol_sha256": protocol_sha256(),
        "evidence_class": spec["evidence_class"],
        "raw_redistribution": False,
        "source_contracts": {
            symbol: {
                "kaggle_slug": source["kaggle_slug"],
                "kaggle_version": source["kaggle_version"],
                "members": source["members"],
                "kaggle_license_label": source["kaggle_license_label"],
                "provenance": source["provenance"],
            }
            for symbol, source in spec["sources"].items()
            if symbol in ("qqq", "spy", "aapl")
        },
        "sources": records, "extra": extra or {},
    }
    out = pathlib.Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_exact_sessions(spec: dict[str, Any] | None = None,
                        root: pathlib.Path = ROOT) -> pd.DatetimeIndex:
    spec = spec or load_protocol()
    path = root / spec["sources"]["session_calendar"]["path"]
    frame = pd.read_parquet(path)
    index = pd.DatetimeIndex(pd.to_datetime(frame.index)).normalize()
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("locked session calendar must be unique and increasing")
    return index


def _read_member(raw_dir: pathlib.Path, member: str) -> tuple[pd.DataFrame, pathlib.Path]:
    candidates = list(raw_dir.rglob(member)) if raw_dir.exists() else []
    if len(candidates) > 1:
        raise ValueError(f"multiple extracted copies found for {member}")
    if candidates:
        path = candidates[0]
        if member.endswith(".parquet"):
            # The source Parquet retains literal bracketed column names.
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(
                path, usecols=lambda c: normalize_headers([c])[0] in OPTION_COLUMNS,
                low_memory=False,
            )
        return frame, path
    archives = sorted(raw_dir.glob("*.zip")) if raw_dir.exists() else []
    containing = []
    for archive in archives:
        with zipfile.ZipFile(archive) as zf:
            if member in zf.namelist():
                containing.append(archive)
    if len(containing) != 1:
        raise FileNotFoundError(
            f"expected exactly one extracted file or ZIP member {member} in {raw_dir}; "
            f"found {len(containing)}"
        )
    archive = containing[0]
    if member.endswith(".parquet"):
        with zipfile.ZipFile(archive) as zf:
            # Parquet needs a seekable buffer; yearly members are bounded.
            frame = pd.read_parquet(io.BytesIO(zf.read(member)))
    else:
        # Stream large QQQ/AAPL CSV members instead of materializing a second
        # 350-630 MB byte string before pandas parses them.
        with zipfile.ZipFile(archive) as zf, zf.open(member) as handle:
            frame = pd.read_csv(
                handle,
                usecols=lambda c: normalize_headers([c])[0] in OPTION_COLUMNS,
                low_memory=False,
            )
    return frame, archive


def ingest_sources(spec: dict[str, Any] | None = None,
                   root: pathlib.Path = ROOT) -> dict[str, Any]:
    """Normalize locally present private sources; never downloads anything."""
    spec = spec or load_protocol()
    sessions = load_exact_sessions(spec, root)
    daily_parts = []
    audits: dict[str, Any] = {}
    source_paths: dict[str, list[pathlib.Path]] = {}
    for symbol in ("qqq", "spy", "aapl"):
        source = spec["sources"][symbol]
        raw_dir = root / source["raw_dir"]
        symbol_daily = []
        symbol_audits = []
        paths = []
        raw_rows = 0
        for member in source["members"]:
            raw, material_path = _read_member(raw_dir, member)
            raw_rows += len(raw)
            normalized, audit = normalize_option_frame(raw, symbol.upper(), sessions, spec)
            symbol_daily.append(build_daily_surface(normalized, symbol.upper(), spec))
            symbol_audits.append({"member": member, **audit})
            paths.append(material_path)
        if "expected_rows" in source and raw_rows != int(source["expected_rows"]):
            raise ValueError(
                f"{symbol} raw row count {raw_rows} != frozen {source['expected_rows']}"
            )
        combined = pd.concat(symbol_daily, ignore_index=True)
        if combined.duplicated(["symbol", "quote_date"]).any():
            raise ValueError(f"{symbol} has duplicate daily surfaces across members")
        combined = combined.sort_values("quote_date")
        frozen_coverage = list(map(pd.Timestamp, source["validated_coverage"]))
        observed_coverage = [combined["quote_date"].min(), combined["quote_date"].max()]
        if observed_coverage != frozen_coverage:
            raise ValueError(
                f"{symbol} calendar-filtered coverage {observed_coverage} != "
                f"frozen {frozen_coverage}"
            )
        expected_sessions = sessions[
            (sessions >= observed_coverage[0]) & (sessions <= observed_coverage[1])
        ]
        observed_sessions = pd.DatetimeIndex(combined["quote_date"].unique())
        missing_sessions = expected_sessions.difference(observed_sessions)
        lagged = add_full_session_lag(combined, sessions)
        daily_parts.append(lagged)
        audits[symbol] = {
            "raw_rows": raw_rows, "members": symbol_audits,
            "daily_rows_before_lag": len(combined), "lagged_rows": len(lagged),
            "coverage_after_calendar_filter": [
                str(combined["quote_date"].min().date()),
                str(combined["quote_date"].max().date()),
            ],
            "expected_exchange_sessions_in_coverage": int(len(expected_sessions)),
            "observed_exchange_sessions": int(len(observed_sessions)),
            "missing_exchange_session_count": int(len(missing_sessions)),
            "missing_exchange_sessions": [str(date.date()) for date in missing_sessions],
        }
        if symbol == "aapl":
            audits[symbol]["split_audit"] = aapl_split_audit(combined, spec)
        source_paths[symbol] = paths
    output = pd.concat(daily_parts, ignore_index=True).sort_values(
        ["symbol", "origin"], kind="mergesort"
    )
    out_path = root / spec["outputs"]["daily_surface"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(out_path, index=False)
    manifest_path = root / spec["outputs"]["source_manifest"]
    manifest = write_source_manifest(
        source_paths, manifest_path, spec,
        extra={
            "parser_audits": audits,
            "daily_surface": {
                "name": out_path.name, "rows": len(output),
                "sha256": hash_file(out_path),
            },
        },
    )
    return {"rows": len(output), "audits": audits, "manifest": manifest}


def _render_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# Frozen private option-surface diagnostic", "",
        "> Diagnostic only. Raw-source provenance and redistribution rights are",
        "> unresolved. Gamma-weighted volume is **not dealer GEX**.", "",
        f"Protocol SHA-256: `{metrics['protocol_sha256']}`", "",
    ]
    for symbol, result in metrics["results"].items():
        lines += [f"## {symbol}", ""]
        for split, score in result.items():
            paired = score["paired"]
            lines.append(
                f"- `{split}`: n={score['n']}; augmented-minus-baseline QLIKE "
                f"{paired['mean_qlike_difference']:+.6g}; AUC delta "
                f"{paired['auc_difference']:+.4f}; verdict `{score['verdict']}`."
            )
        lines.append("")
    lines += [
        "AAPL earnings were omitted because the existing calendar does not prove",
        "that each announcement date and session was knowable at the origin.", "",
    ]
    return "\n".join(lines)


def run_study(spec: dict[str, Any] | None = None,
              root: pathlib.Path = ROOT) -> dict[str, Any]:
    spec = spec or load_protocol()
    surface_path = root / spec["outputs"]["daily_surface"]
    if not surface_path.exists():
        raise FileNotFoundError("normalized daily surface is missing; run ingest first")
    surface = pd.read_parquet(surface_path)
    forecasts = []
    results: dict[str, Any] = {}
    for symbol, part in surface.groupby("symbol", sort=True):
        design = build_forecast_design(part, spec)
        fcst = forecast_symbol(design, symbol, spec)
        forecasts.append(fcst)
        results[symbol] = {}
        splits = ("early", "late_confirmation") if symbol == "SPY" else ("all",)
        for split in splits:
            sample = fcst.loc[fcst["split"] == split]
            if not sample.empty:
                results[symbol][split] = summarize_split(sample, spec)
    if not forecasts:
        raise ValueError("surface artifact contains no symbols")
    forecast_frame = pd.concat(forecasts).sort_index(kind="mergesort")
    forecast_path = root / spec["outputs"]["forecasts"]
    forecast_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_frame.to_parquet(forecast_path)
    metrics = {
        "protocol_sha256": protocol_sha256(),
        "evidence_class": spec["evidence_class"], "raw_redistribution": False,
        "results": results, "spy": results.get("SPY", {}),
        "artifacts": {
            "daily_surface_sha256": hash_file(surface_path),
            "forecasts_sha256": hash_file(forecast_path),
        },
    }
    metrics_path = root / spec["outputs"]["metrics"]
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    report_path = root / spec["outputs"]["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(metrics))
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("ingest", "run", "all", "audit-inputs"))
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    args = parser.parse_args(argv)
    spec = load_protocol(args.protocol)
    if args.command in ("ingest", "all"):
        print(json.dumps(ingest_sources(spec), indent=2, sort_keys=True))
    if args.command in ("run", "all"):
        print(json.dumps(run_study(spec), indent=2, sort_keys=True))
    if args.command == "audit-inputs":
        found = {}
        for symbol in ("qqq", "spy", "aapl"):
            source = spec["sources"][symbol]
            raw_dir = ROOT / source["raw_dir"]
            found[symbol] = {
                "raw_dir_exists": raw_dir.exists(), "members": source["members"],
                "archives": [p.name for p in sorted(raw_dir.glob("*.zip"))]
                if raw_dir.exists() else [],
            }
        print(json.dumps(found, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
