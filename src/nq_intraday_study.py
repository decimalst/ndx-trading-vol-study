"""Frozen exploratory NQ one-minute intraday study.

The protocol and tests predate this implementation.  The source is a third-
party continuous series without contract identifiers or a documented roll
method.  Consequently the module detects and excludes suspicious stitch
neighborhoods, but never claims to identify actual rolls.

No command downloads data.  ``build`` requires the ignored raw CSV already to
exist at the path frozen in :mod:`nq_intraday_study.yaml`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import gamma
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "nq_intraday_study.yaml"
IMPLEMENTATION_PATH = Path(__file__).resolve()


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(path: Path | str = PROTOCOL_PATH) -> dict:
    with Path(path).open() as handle:
        protocol = yaml.safe_load(handle)
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: dict) -> None:
    if "exploratory" not in str(protocol.get("status", "")):
        raise ValueError("NQ study must remain diagnostic and exploratory")
    source = protocol["source"]
    if source.get("has_contract_identifier") or source.get("has_roll_methodology"):
        raise ValueError("contract identity/stitch methodology cannot be invented")
    fences = protocol["fences"]
    data_end = pd.Timestamp(fences["data_end"])
    clean = pd.Timestamp(fences["sealed_ndx_clean_start"])
    if data_end > pd.Timestamp("2025-10-17") or data_end >= clean:
        raise ValueError("data/clean fence must end no later than 2025-10-17")
    if not fences.get("forbid_rows_after_data_end_before_transforms"):
        raise ValueError("post-fence rows must be removed before transforms")
    if not fences.get("forbid_clean_origins") or not fences.get("no_forward_fill"):
        raise ValueError("clean-origin and no-fill fences are mandatory")
    session = protocol["session"]
    if (
        session.get("timezone") != "America/New_York"
        or session.get("rth_start_inclusive") != "09:30:00"
        or session.get("rth_end_exclusive") != "16:00:00"
        or int(session.get("expected_one_minute_bars", 0)) != 390
    ):
        raise ValueError("official U.S. equity-index RTH contract changed")
    sampling = protocol["sampling"]
    if int(sampling["endpoint_minutes"]) != 5 or int(
        sampling["expected_returns_per_complete_session"]
    ) != 78:
        raise ValueError("primary sampling must remain 78 five-minute returns")
    jump = protocol["jump_measure"]
    if not np.isclose(float(jump["alpha"]), 0.01) or not np.isclose(
        float(jump["z_threshold"]), 2.3263478740408408
    ):
        raise ValueError("one-percent BNS threshold is frozen")
    diagnostic = protocol["diagnostic"]
    if not diagnostic.get("no_candidate_selection"):
        raise ValueError("candidate selection is forbidden")
    if int(diagnostic["horizon_sessions"]) != 5 or int(
        diagnostic["phase_step_sessions"]
    ) != 5:
        raise ValueError("five-session target and phases are frozen")
    if list(diagnostic["augmented_features"])[-2:] != [
        "log_vxn_lag1",
        "log_skew_lag1",
    ]:
        raise ValueError("the only registered augmentation is lagged VXN plus SKEW")


def _canonical(column: str) -> str:
    return "_".join(str(column).strip().lower().replace("-", "_").split())


def localize_et(values: pd.Series | Iterable) -> pd.Series:
    """Parse ET wall times and fail on DST ambiguity/nonexistence."""
    series = values if isinstance(values, pd.Series) else pd.Series(list(values))
    try:
        parsed = pd.to_datetime(series, errors="raise")
        index = pd.DatetimeIndex(parsed)
        if index.tz is None:
            index = index.tz_localize(
                "America/New_York", ambiguous="raise", nonexistent="raise"
            )
        else:
            index = index.tz_convert("America/New_York")
    except Exception as exc:
        raise ValueError(f"invalid or DST-ambiguous ET timestamp: {exc}") from exc
    return pd.Series(index, index=series.index, name="timestamp")


def normalize_and_filter(raw: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    """Normalize schema, then fence dates before any price transformation."""
    validate_protocol(protocol)
    frame = raw.copy()
    frame.columns = [_canonical(column) for column in frame.columns]
    aliases = [_canonical(value) for value in protocol["source"]["timestamp_aliases"]]
    timestamp_column = next((name for name in aliases if name in frame), None)
    if timestamp_column is None:
        raise ValueError(f"NQ source has no timestamp alias among {aliases}")
    frame["timestamp"] = localize_et(frame[timestamp_column])

    # The upstream file extends beyond the research fence.  Only timestamps
    # are examined before this mask; post-fence prices never enter validation,
    # discontinuity detection, a rolling transform, or a summary.
    local_naive = frame["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    dates = local_naive.dt.normalize()
    start = pd.Timestamp(protocol["fences"]["data_start"])
    end = pd.Timestamp(protocol["fences"]["data_end"])
    frame = frame.loc[(dates >= start) & (dates <= end)].copy()
    local_naive = local_naive.loc[frame.index]

    required = [name for name in protocol["source"]["required_columns"] if name != "timestamp_et"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"NQ source missing columns {missing}")
    for name in required:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    if frame[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("NQ source contains missing prices")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("NQ prices must be positive")
    if (frame["volume"] < 0).any():
        raise ValueError("NQ volume must be non-negative")
    upper = frame[["open", "close", "low"]].max(axis=1)
    lower = frame[["open", "close", "high"]].min(axis=1)
    if (frame["high"] < upper).any() or (frame["low"] > lower).any():
        raise ValueError("NQ source has impossible OHLC geometry")

    local_naive = frame["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    seconds = (
        local_naive.dt.hour * 3600
        + local_naive.dt.minute * 60
        + local_naive.dt.second
    )
    start_seconds = 9 * 3600 + 30 * 60
    end_seconds = 16 * 3600
    keep = (
        (seconds >= start_seconds)
        & (seconds < end_seconds)
        & (local_naive.dt.dayofweek < 5)
    )
    frame = frame.loc[keep, ["timestamp", "open", "high", "low", "close", "volume"]]
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    if frame["timestamp"].duplicated().any():
        raise ValueError("NQ source contains duplicate RTH timestamps")
    local = frame["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    frame["session"] = local.dt.normalize()
    return frame


def session_quality(frame: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    expected_count = int(protocol["session"]["expected_one_minute_bars"])
    rows: list[dict] = []
    for session, group in frame.groupby("session", sort=True):
        group = group.sort_values("timestamp")
        local = group["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None)
        expected = pd.date_range(
            pd.Timestamp(session) + pd.Timedelta(hours=9, minutes=30),
            periods=expected_count,
            freq="1min",
        )
        unique = pd.DatetimeIndex(local).is_unique
        exact = unique and len(group) == expected_count and pd.DatetimeIndex(local).equals(expected)
        gaps = local.diff().dt.total_seconds().div(60.0).dropna()
        rows.append(
            {
                "session": pd.Timestamp(session),
                "bar_count": int(len(group)),
                "missing_bar_count": max(0, expected_count - int(len(group))),
                "max_gap_minutes": float(gaps.max()) if len(gaps) else np.nan,
                "complete_session": bool(exact),
            }
        )
    quality = pd.DataFrame(rows).set_index("session") if rows else pd.DataFrame()
    if (
        len(quality)
        and protocol["session"].get("reject_terminal_partial_session")
        and not bool(quality.iloc[-1]["complete_session"])
    ):
        raise ValueError("terminal partial session rejected")
    return quality


def sample_five_minute_returns(session: pd.DataFrame, protocol: dict) -> pd.Series:
    """Return 78 within-session returns; no previous-session close is accepted."""
    group = session.sort_values("timestamp")
    expected = int(protocol["session"]["expected_one_minute_bars"])
    width = int(protocol["sampling"]["endpoint_minutes"])
    if len(group) != expected:
        raise ValueError("five-minute sampling requires a complete 390-bar session")
    local = group["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    if len(local) > 1 and not np.all(local.diff().dropna() == pd.Timedelta(minutes=1)):
        raise ValueError("five-minute sampling requires consecutive one-minute bars")
    endpoints = group.iloc[width - 1 :: width]
    levels = np.r_[float(group.iloc[0]["open"]), endpoints["close"].to_numpy(dtype=float)]
    returns = np.diff(np.log(levels))
    out = pd.Series(returns, index=pd.DatetimeIndex(endpoints["timestamp"]), name="return_5m")
    expected_returns = int(protocol["sampling"]["expected_returns_per_complete_session"])
    if len(out) != expected_returns:
        raise RuntimeError("five-minute endpoint construction changed")
    return out


def bns_measures(returns: pd.Series | np.ndarray, protocol: dict) -> dict:
    values = np.asarray(returns, dtype=float)
    if len(values) < 3 or np.any(~np.isfinite(values)):
        raise ValueError("BNS measures require at least three finite returns")
    n = len(values)
    absolute = np.abs(values)
    rv = float(np.square(values).sum())
    bpv = float(
        (np.pi / 2.0)
        * n
        / (n - 1)
        * np.sum(absolute[1:] * absolute[:-1])
    )
    mu43 = float(2 ** (2 / 3) * gamma(7 / 6) / np.sqrt(np.pi))
    tq = float(
        mu43 ** -3
        * n**2
        / (n - 2)
        * np.sum((absolute[2:] * absolute[1:-1] * absolute[:-2]) ** (4 / 3))
    )
    if rv <= 0 or bpv <= 0:
        raise ValueError("BNS ratio requires positive RV and BPV")
    asymptotic = (np.pi / 2) ** 2 + np.pi - 5
    denominator = math.sqrt(asymptotic / n * max(1.0, tq / bpv**2))
    z = float((1.0 - bpv / rv) / denominator)
    threshold = float(protocol["jump_measure"]["z_threshold"])
    return {
        "rv": rv,
        "bpv": bpv,
        "tripower_quarticity": tq,
        "bns_z": z,
        "jump_variation": max(rv - bpv, 0.0),
        "jump_share": max(1.0 - bpv / rv, 0.0),
        "jump_significant": bool(z > threshold),
    }


def intraday_shape(returns: pd.Series | np.ndarray, protocol: dict) -> dict:
    values = np.asarray(returns, dtype=float)
    first = int(protocol["intraday_shape"]["first_hour_returns"])
    last = int(protocol["intraday_shape"]["last_hour_returns"])
    if len(values) <= first + last:
        raise ValueError("intraday shape requires a non-empty middle interval")
    squared = np.square(values)
    total = float(squared.sum())
    if total <= 0:
        raise ValueError("intraday shape requires positive RV")
    return {
        "first_hour_rv_share": float(squared[:first].sum() / total),
        "middle_rv_share": float(squared[first:-last].sum() / total),
        "last_hour_rv_share": float(squared[-last:].sum() / total),
    }


def infer_stitch_neighborhoods(panel: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    """Flag possible stitch/bad-print sessions without claiming true roll IDs."""
    frame = panel.sort_index()
    overnight = np.log(
        pd.to_numeric(frame["session_open"], errors="raise")
        / pd.to_numeric(frame["session_close"], errors="raise").shift(1)
    ).abs()
    within = pd.to_numeric(frame["max_abs_one_minute_return"], errors="raise").abs()
    trigger = (
        (overnight >= float(protocol["stitch_guard"]["overnight_absolute_log_gap"]))
        | (within >= float(protocol["stitch_guard"]["within_rth_one_minute_absolute_log_return"]))
    ).fillna(False)
    radius = int(protocol["stitch_guard"]["exclusion_radius_sessions"])
    excluded = trigger.copy()
    for offset in range(1, radius + 1):
        excluded = excluded | trigger.shift(offset, fill_value=False) | trigger.shift(
            -offset, fill_value=False
        )
    return pd.DataFrame(
        {
            "overnight_abs_log_gap": overnight,
            "stitch_trigger": trigger.astype(bool),
            "stitch_excluded": excluded.astype(bool),
        },
        index=frame.index,
    )


def build_daily_panel(frame: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    quality = session_quality(frame, protocol)
    rows: list[dict] = []
    groups = {pd.Timestamp(key): value for key, value in frame.groupby("session", sort=True)}
    for session in quality.index:
        group = groups[pd.Timestamp(session)].sort_values("timestamp")
        one_minute_levels = np.r_[float(group.iloc[0]["open"]), group["close"].to_numpy(float)]
        one_minute_returns = np.diff(np.log(one_minute_levels))
        row = {
            "session": pd.Timestamp(session),
            "session_open": float(group.iloc[0]["open"]),
            "session_close": float(group.iloc[-1]["close"]),
            "max_abs_one_minute_return": float(np.max(np.abs(one_minute_returns))),
            **quality.loc[session].to_dict(),
        }
        if bool(quality.loc[session, "complete_session"]):
            returns = sample_five_minute_returns(group, protocol)
            row.update(bns_measures(returns, protocol))
            row.update(intraday_shape(returns, protocol))
        else:
            for name in (
                "rv", "bpv", "tripower_quarticity", "bns_z", "jump_variation",
                "jump_share", "first_hour_rv_share", "middle_rv_share",
                "last_hour_rv_share",
            ):
                row[name] = np.nan
            row["jump_significant"] = False
        rows.append(row)
    daily = pd.DataFrame(rows).set_index("session").sort_index()
    stitch = infer_stitch_neighborhoods(daily, protocol)
    daily = daily.join(stitch)
    daily["quality_eligible"] = (
        daily["complete_session"].astype(bool) & ~daily["stitch_excluded"].astype(bool)
    )
    measure_columns = [
        "rv", "bpv", "tripower_quarticity", "bns_z", "jump_variation",
        "jump_share", "first_hour_rv_share", "middle_rv_share",
        "last_hour_rv_share",
    ]
    daily.loc[~daily["quality_eligible"], measure_columns] = np.nan
    daily.loc[~daily["quality_eligible"], "jump_significant"] = False
    daily.index.name = "session"
    return daily


def delay_cboe_close(
    source: pd.Series, sessions: Iterable, *, delay_sessions: int = 1
) -> pd.Series:
    if delay_sessions != 1:
        raise ValueError("Cboe closes must be delayed exactly one full session")
    values = pd.to_numeric(source.copy(), errors="coerce")
    index = pd.DatetimeIndex(pd.to_datetime(values.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    values.index = index.normalize()
    if values.index.duplicated().any():
        raise ValueError("Cboe input has duplicate dates")
    target = pd.DatetimeIndex(pd.to_datetime(list(sessions))).tz_localize(None).normalize()
    return values.sort_index().reindex(target).shift(delay_sessions)


def build_forward_target(daily: pd.DataFrame, *, horizon: int = 5) -> pd.DataFrame:
    frame = daily.sort_index()
    output = pd.DataFrame(index=frame.index)
    output["target_end"] = pd.NaT
    output["event"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    for position in range(max(0, len(frame) - int(horizon))):
        future = frame.iloc[position + 1 : position + 1 + int(horizon)]
        if len(future) != int(horizon) or not future["quality_eligible"].astype(bool).all():
            continue
        if future["jump_significant"].isna().any():
            continue
        origin = frame.index[position]
        output.loc[origin, "target_end"] = future.index[-1]
        output.loc[origin, "event"] = bool(future["jump_significant"].astype(bool).any())
    return output


def _close_series(path: Path | str) -> pd.Series:
    frame = pd.read_parquet(path)
    if "close" not in frame:
        raise ValueError(f"{path}: no close column")
    values = pd.to_numeric(frame["close"], errors="raise")
    if values.isna().any() or (values <= 0).any():
        raise ValueError(f"{path}: close must be finite and positive")
    return values


def build_model_frame(
    daily: pd.DataFrame, vxn: pd.Series, skew: pd.Series, protocol: dict
) -> pd.DataFrame:
    frame = daily.sort_index().copy()
    frame = frame.loc[: pd.Timestamp(protocol["fences"]["data_end"])]
    eligible_rv = frame["rv"].where(frame["quality_eligible"] & (frame["rv"] > 0))
    design = pd.DataFrame(index=frame.index)
    design["log_rv"] = np.log(eligible_rv)
    design["log_rv_5"] = np.log(eligible_rv.rolling(5, min_periods=5).mean())
    design["log_rv_22"] = np.log(eligible_rv.rolling(22, min_periods=22).mean())
    design["bns_z"] = frame["bns_z"].where(frame["quality_eligible"])
    design["jump_share_5"] = frame["jump_share"].where(
        frame["quality_eligible"]
    ).rolling(5, min_periods=5).mean()
    delay = int(protocol["inputs"]["cboe_close_delay_sessions"])
    complete_sessions = frame.index[frame["quality_eligible"].astype(bool)]
    vxn_lag = delay_cboe_close(
        vxn, complete_sessions, delay_sessions=delay
    ).reindex(frame.index)
    skew_lag = delay_cboe_close(
        skew, complete_sessions, delay_sessions=delay
    ).reindex(frame.index)
    design["log_vxn_lag1"] = np.log(vxn_lag.where(vxn_lag > 0))
    design["log_skew_lag1"] = np.log(skew_lag.where(skew_lag > 0))
    targets = build_forward_target(
        frame, horizon=int(protocol["diagnostic"]["horizon_sessions"])
    )
    return design.join(targets)


def annual_fold_rows(frame: pd.DataFrame, test_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_start = pd.Timestamp(f"{int(test_year)}-01-01")
    test_end = pd.Timestamp(f"{int(test_year)}-12-31")
    completed = pd.to_datetime(frame["target_end"], errors="coerce") < test_start
    train = frame.loc[(frame.index < test_start) & completed].copy()
    test = frame.loc[(frame.index >= test_start) & (frame.index <= test_end)].copy()
    return train, test


def _fit_probability(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str], protocol: dict
) -> np.ndarray:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[features].to_numpy(dtype=float))
    x_test = scaler.transform(test[features].to_numpy(dtype=float))
    model = LogisticRegression(
        penalty="l2",
        C=float(protocol["diagnostic"]["logistic_l2_c"]),
        solver="lbfgs",
        max_iter=int(protocol["diagnostic"]["logistic_max_iter"]),
        random_state=20260812,
    )
    model.fit(x_train, train["event"].astype(int).to_numpy())
    return model.predict_proba(x_test)[:, 1]


def run_annual_models(design: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    diagnostic = protocol["diagnostic"]
    baseline = list(diagnostic["baseline_features"])
    augmented = list(diagnostic["augmented_features"])
    required = list(dict.fromkeys([*augmented, "event", "target_end"]))
    outputs: list[pd.DataFrame] = []
    for year in range(int(diagnostic["first_test_year"]), int(diagnostic["final_test_year"]) + 1):
        train, test = annual_fold_rows(design, year)
        train = train.dropna(subset=required)
        test = test.dropna(subset=required)
        if len(train) < int(diagnostic["min_train_rows"]):
            continue
        if train["event"].astype(int).nunique() != 2 or test["event"].astype(int).nunique() != 2:
            continue
        part = test[["target_end", "event"]].copy()
        part["fold_year"] = year
        part["p_price_history"] = _fit_probability(train, test, baseline, protocol)
        part["p_augmented"] = _fit_probability(train, test, augmented, protocol)
        outputs.append(part)
    if not outputs:
        return pd.DataFrame(
            columns=["target_end", "event", "fold_year", "p_price_history", "p_augmented"],
            index=pd.DatetimeIndex([], name="origin"),
        )
    result = pd.concat(outputs).sort_index()
    result.index.name = "origin"
    return result


def _auc(y: np.ndarray, probability: np.ndarray) -> float:
    return float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else np.nan


def _top_lift(y: np.ndarray, probability: np.ndarray, fraction: float) -> float:
    base = float(np.mean(y))
    if base <= 0 or not len(y):
        return np.nan
    count = max(1, int(math.ceil(float(fraction) * len(y))))
    order = np.argsort(-probability, kind="stable")[:count]
    return float(np.mean(y[order]) / base)


def phase_ranking_metrics(
    forecasts: pd.DataFrame, *, phase_step: int = 5, top_fraction: float = 0.10
) -> pd.DataFrame:
    rows: list[dict] = []
    frame = forecasts.sort_index()
    for phase in range(int(phase_step)):
        sample = frame.iloc[phase::phase_step].dropna(
            subset=["event", "p_price_history", "p_augmented"]
        )
        y = sample["event"].astype(int).to_numpy()
        row = {"phase": phase, "n": int(len(sample)), "event_rate": float(np.mean(y)) if len(y) else np.nan}
        for model in ("price_history", "augmented"):
            probability = sample[f"p_{model}"].to_numpy(dtype=float)
            row[f"{model}_auc"] = _auc(y, probability) if len(y) else np.nan
            row[f"{model}_top_decile_lift"] = (
                _top_lift(y, probability, top_fraction) if len(y) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_metrics(forecasts: pd.DataFrame, phases: pd.DataFrame, protocol: dict) -> dict:
    def summary(column: str) -> dict:
        values = pd.to_numeric(phases[column], errors="coerce").dropna()
        return {
            "phase_mean": float(values.mean()) if len(values) else None,
            "phase_min": float(values.min()) if len(values) else None,
            "phase_max": float(values.max()) if len(values) else None,
        }

    return {
        "evidence_class": protocol["status"],
        "strict_verdict": "DIAGNOSTIC_ONLY",
        "unknown_contract_stitch": True,
        "n_forecasts": int(len(forecasts)),
        "event_rate": float(forecasts["event"].astype(float).mean()) if len(forecasts) else None,
        "price_history_auc": summary("price_history_auc"),
        "augmented_auc": summary("augmented_auc"),
        "price_history_top_decile_lift": summary("price_history_top_decile_lift"),
        "augmented_top_decile_lift": summary("augmented_top_decile_lift"),
    }


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary)
    temporary.replace(path)


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def assert_raw_is_ignored(path: Path) -> None:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", str(path)],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"raw input must be git-ignored before acquisition/build: {path}"
        )


def build_study(protocol: dict | None = None) -> pd.DataFrame:
    protocol = protocol or load_protocol()
    raw_path = ROOT / protocol["source"]["raw_path"]
    if not raw_path.exists():
        raise FileNotFoundError(
            f"raw Kaggle NQ file is absent (no downloader is provided): {raw_path}"
        )
    assert_raw_is_ignored(raw_path)
    raw = pd.read_csv(raw_path)
    normalized = normalize_and_filter(raw, protocol)
    daily = build_daily_panel(normalized, protocol)
    output_path = ROOT / protocol["outputs"]["daily_panel"]
    _atomic_parquet(daily, output_path)
    manifest = {
        "evidence_class": protocol["status"],
        "source": {
            "path": protocol["source"]["raw_path"],
            "sha256": sha256(raw_path),
            "raw_rows": int(len(raw)),
            "retained_rth_rows": int(len(normalized)),
        },
        "protocol": {"path": "nq_intraday_study.yaml", "sha256": sha256(PROTOCOL_PATH)},
        "implementation": {"path": "src/nq_intraday_study.py", "sha256": sha256(IMPLEMENTATION_PATH)},
        "daily_panel": {
            "path": protocol["outputs"]["daily_panel"],
            "sha256": sha256(output_path),
            "rows": int(len(daily)),
            "first_session": str(daily.index.min().date()) if len(daily) else None,
            "last_session": str(daily.index.max().date()) if len(daily) else None,
            "quality_eligible": int(daily["quality_eligible"].sum()) if len(daily) else 0,
            "stitch_excluded": int(daily["stitch_excluded"].sum()) if len(daily) else 0,
        },
    }
    _atomic_json(manifest, ROOT / protocol["outputs"]["manifest"])
    return daily


def run_diagnostic(protocol: dict | None = None) -> dict:
    protocol = protocol or load_protocol()
    daily_path = ROOT / protocol["outputs"]["daily_panel"]
    if not daily_path.exists():
        raise FileNotFoundError("build the frozen daily panel before scoring")
    daily = pd.read_parquet(daily_path)
    if len(daily) and pd.Timestamp(daily.index.max()) > pd.Timestamp(protocol["fences"]["data_end"]):
        raise RuntimeError("processed NQ panel crossed its data fence")
    vxn = _close_series(ROOT / protocol["inputs"]["vxn_path"])
    skew = _close_series(ROOT / protocol["inputs"]["skew_path"])
    design = build_model_frame(daily, vxn, skew, protocol)
    forecasts = run_annual_models(design, protocol)
    phases = phase_ranking_metrics(
        forecasts,
        phase_step=int(protocol["diagnostic"]["phase_step_sessions"]),
        top_fraction=float(protocol["diagnostic"]["top_fraction"]),
    )
    metrics = summarize_metrics(forecasts, phases, protocol)
    forecast_path = ROOT / protocol["outputs"]["forecasts"]
    phase_path = ROOT / protocol["outputs"]["phase_metrics"]
    metrics_path = ROOT / protocol["outputs"]["metrics"]
    _atomic_parquet(forecasts, forecast_path)
    _atomic_parquet(phases, phase_path)
    _atomic_json(metrics, metrics_path)

    manifest_path = ROOT / protocol["outputs"]["manifest"]
    manifest = json.loads(manifest_path.read_text())
    manifest["inputs"] = {
        "vxn": {"path": protocol["inputs"]["vxn_path"], "sha256": sha256(ROOT / protocol["inputs"]["vxn_path"])},
        "skew": {"path": protocol["inputs"]["skew_path"], "sha256": sha256(ROOT / protocol["inputs"]["skew_path"])},
    }
    manifest["diagnostic_outputs"] = {
        "forecasts": {"path": protocol["outputs"]["forecasts"], "sha256": sha256(forecast_path)},
        "phase_metrics": {"path": protocol["outputs"]["phase_metrics"], "sha256": sha256(phase_path)},
        "metrics": {"path": protocol["outputs"]["metrics"], "sha256": sha256(metrics_path)},
    }
    _atomic_json(manifest, manifest_path)
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "build", "run", "all"))
    args = parser.parse_args(argv)
    protocol = load_protocol()
    if args.command == "status":
        raw = ROOT / protocol["source"]["raw_path"]
        print(json.dumps({"raw_path": str(raw), "raw_exists": raw.exists(), "status": protocol["status"]}, indent=2))
    elif args.command == "build":
        panel = build_study(protocol)
        print(f"built {len(panel):,} NQ RTH sessions")
    elif args.command == "run":
        print(json.dumps(run_diagnostic(protocol), indent=2))
    else:
        build_study(protocol)
        print(json.dumps(run_diagnostic(protocol), indent=2))


if __name__ == "__main__":
    main()
