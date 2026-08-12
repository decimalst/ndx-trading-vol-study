"""Independent verifier for the frozen exploratory NQ intraday study.

This module deliberately does not import the study implementation.  It parses
the frozen protocol, rebuilds the RTH daily panel and the complete diagnostic
from source artifacts, then compares every persisted artifact and provenance
hash.  The upstream continuous series has no contract IDs or documented roll
method; successful verification therefore never upgrades the evidence beyond
diagnostic/exploratory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from scipy.special import gamma
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "nq_intraday_study.yaml"
IMPLEMENTATION_PATH = ROOT / "src" / "nq_intraday_study.py"
VERIFIER_PATH = Path(__file__).resolve()


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(path: Path | str = PROTOCOL_PATH) -> dict:
    with Path(path).open() as handle:
        protocol = yaml.safe_load(handle)
    _validate_frozen_protocol(protocol)
    return protocol


def _validate_frozen_protocol(protocol: dict) -> None:
    if "exploratory" not in str(protocol.get("status", "")):
        raise AssertionError("evidence class is not exploratory")
    if protocol["source"].get("has_contract_identifier"):
        raise AssertionError("source cannot be represented as contract identified")
    if protocol["source"].get("has_roll_methodology"):
        raise AssertionError("source cannot be represented as having a known stitch")
    data_end = pd.Timestamp(protocol["fences"]["data_end"])
    if data_end > pd.Timestamp("2025-10-17"):
        raise AssertionError("data fence moved beyond 2025-10-17")
    if data_end >= pd.Timestamp(protocol["fences"]["sealed_ndx_clean_start"]):
        raise AssertionError("NQ data fence overlaps the sealed NDX clean window")
    if protocol["session"]["timezone"] != "America/New_York":
        raise AssertionError("session timezone changed")
    if protocol["session"]["rth_start_inclusive"] != "09:30:00":
        raise AssertionError("RTH start changed")
    if protocol["session"]["rth_end_exclusive"] != "16:00:00":
        raise AssertionError("RTH end changed")
    if int(protocol["session"]["expected_one_minute_bars"]) != 390:
        raise AssertionError("complete-session bar count changed")
    if int(protocol["sampling"]["expected_returns_per_complete_session"]) != 78:
        raise AssertionError("five-minute return count changed")
    if not np.isclose(float(protocol["jump_measure"]["alpha"]), 0.01):
        raise AssertionError("jump-test alpha changed")
    if not np.isclose(
        float(protocol["jump_measure"]["z_threshold"]), 2.3263478740408408
    ):
        raise AssertionError("jump-test threshold changed")
    if int(protocol["diagnostic"]["horizon_sessions"]) != 5:
        raise AssertionError("forward horizon changed")
    if int(protocol["diagnostic"]["phase_step_sessions"]) != 5:
        raise AssertionError("phase count changed")


def preflight(protocol: dict | None = None) -> list[str]:
    """Return repo-relative required inputs/artifacts that are absent."""
    protocol = protocol or load_protocol()
    required = [
        protocol["source"]["raw_path"],
        protocol["inputs"]["vxn_path"],
        protocol["inputs"]["skew_path"],
        protocol["outputs"]["daily_panel"],
        protocol["outputs"]["forecasts"],
        protocol["outputs"]["phase_metrics"],
        protocol["outputs"]["metrics"],
        protocol["outputs"]["manifest"],
    ]
    return [path for path in required if not (ROOT / path).is_file()]


def _canonical(value: str) -> str:
    return "_".join(str(value).strip().lower().replace("-", "_").split())


def _strict_et(values: pd.Series) -> pd.Series:
    try:
        parsed = pd.to_datetime(values, errors="raise")
        index = pd.DatetimeIndex(parsed)
        if index.tz is None:
            index = index.tz_localize(
                "America/New_York", ambiguous="raise", nonexistent="raise"
            )
        else:
            index = index.tz_convert("America/New_York")
    except Exception as exc:
        raise AssertionError(f"invalid or DST-ambiguous ET timestamp: {exc}") from exc
    return pd.Series(index, index=values.index, name="timestamp")


def _read_fenced_rth(raw_path: Path, protocol: dict) -> tuple[pd.DataFrame, int]:
    raw = pd.read_csv(raw_path)
    raw_rows = len(raw)
    raw.columns = [_canonical(column) for column in raw.columns]
    aliases = [_canonical(column) for column in protocol["source"]["timestamp_aliases"]]
    timestamp_column = next((column for column in aliases if column in raw), None)
    if timestamp_column is None:
        raise AssertionError("raw source has no frozen timestamp alias")
    raw["timestamp"] = _strict_et(raw[timestamp_column])

    # Match the safety order independently: date fencing precedes every price
    # conversion, range check, return, rolling statistic, and stitch heuristic.
    local = raw["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    dates = local.dt.normalize()
    start = pd.Timestamp(protocol["fences"]["data_start"])
    end = pd.Timestamp(protocol["fences"]["data_end"])
    frame = raw.loc[(dates >= start) & (dates <= end)].copy()

    required = ["open", "high", "low", "close", "volume"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise AssertionError(f"raw source missing columns {missing}")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame[required].isna().any().any():
        raise AssertionError("raw source contains non-finite OHLCV")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise AssertionError("raw source contains non-positive prices")
    if (frame["volume"] < 0).any():
        raise AssertionError("raw source contains negative volume")
    if (
        frame["high"] < frame[["open", "close", "low"]].max(axis=1)
    ).any() or (
        frame["low"] > frame[["open", "close", "high"]].min(axis=1)
    ).any():
        raise AssertionError("raw source contains impossible OHLC geometry")

    local = frame["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    seconds = local.dt.hour * 3600 + local.dt.minute * 60 + local.dt.second
    rth = (
        (seconds >= 9 * 3600 + 30 * 60)
        & (seconds < 16 * 3600)
        & (local.dt.dayofweek < 5)
    )
    frame = frame.loc[
        rth, ["timestamp", "open", "high", "low", "close", "volume"]
    ].sort_values("timestamp")
    frame = frame.reset_index(drop=True)
    if frame["timestamp"].duplicated().any():
        raise AssertionError("raw source contains duplicate RTH timestamps")
    local = frame["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    frame["session"] = local.dt.normalize()
    return frame, raw_rows


def _bns(returns: np.ndarray, protocol: dict) -> dict:
    values = np.asarray(returns, dtype=float)
    n = len(values)
    if n != int(protocol["sampling"]["expected_returns_per_complete_session"]):
        raise AssertionError("complete session did not produce 78 returns")
    absolute = np.abs(values)
    rv = float(np.square(values).sum())
    bpv = float(
        np.pi / 2.0
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
        raise AssertionError("BNS ratio has non-positive RV or BPV")
    constant = (np.pi / 2) ** 2 + np.pi - 5
    z = float((1 - bpv / rv) / math.sqrt(constant / n * max(1.0, tq / bpv**2)))
    squared = np.square(values)
    first = int(protocol["intraday_shape"]["first_hour_returns"])
    last = int(protocol["intraday_shape"]["last_hour_returns"])
    return {
        "rv": rv,
        "bpv": bpv,
        "tripower_quarticity": tq,
        "bns_z": z,
        "jump_variation": max(rv - bpv, 0.0),
        "jump_share": max(1.0 - bpv / rv, 0.0),
        "jump_significant": bool(z > float(protocol["jump_measure"]["z_threshold"])),
        "first_hour_rv_share": float(squared[:first].sum() / rv),
        "middle_rv_share": float(squared[first:-last].sum() / rv),
        "last_hour_rv_share": float(squared[-last:].sum() / rv),
    }


def _rebuild_daily(frame: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    if frame.empty:
        raise AssertionError("no RTH rows survive the frozen date fence")
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
        complete = bool(
            len(group) == expected_count
            and pd.DatetimeIndex(local).is_unique
            and pd.DatetimeIndex(local).equals(expected)
        )
        gaps = local.diff().dt.total_seconds().div(60).dropna()
        one_minute_levels = np.r_[float(group.iloc[0]["open"]), group["close"].to_numpy(float)]
        one_minute_returns = np.diff(np.log(one_minute_levels))
        row = {
            "session": pd.Timestamp(session),
            "session_open": float(group.iloc[0]["open"]),
            "session_close": float(group.iloc[-1]["close"]),
            "max_abs_one_minute_return": float(np.max(np.abs(one_minute_returns))),
            "bar_count": int(len(group)),
            "missing_bar_count": max(0, expected_count - int(len(group))),
            "max_gap_minutes": float(gaps.max()) if len(gaps) else np.nan,
            "complete_session": complete,
        }
        if complete:
            width = int(protocol["sampling"]["endpoint_minutes"])
            endpoints = group.iloc[width - 1 :: width]["close"].to_numpy(float)
            returns = np.diff(np.log(np.r_[float(group.iloc[0]["open"]), endpoints]))
            row.update(_bns(returns, protocol))
        else:
            for column in (
                "rv",
                "bpv",
                "tripower_quarticity",
                "bns_z",
                "jump_variation",
                "jump_share",
                "first_hour_rv_share",
                "middle_rv_share",
                "last_hour_rv_share",
            ):
                row[column] = np.nan
            row["jump_significant"] = False
        rows.append(row)

    daily = pd.DataFrame(rows).set_index("session").sort_index()
    if (
        protocol["session"].get("reject_terminal_partial_session")
        and not bool(daily.iloc[-1]["complete_session"])
    ):
        raise AssertionError("terminal partial session was not rejected")

    overnight = np.log(daily["session_open"] / daily["session_close"].shift(1)).abs()
    trigger = (
        overnight >= float(protocol["stitch_guard"]["overnight_absolute_log_gap"])
    ) | (
        daily["max_abs_one_minute_return"].abs()
        >= float(protocol["stitch_guard"]["within_rth_one_minute_absolute_log_return"])
    )
    trigger = trigger.fillna(False)
    excluded = trigger.copy()
    for offset in range(1, int(protocol["stitch_guard"]["exclusion_radius_sessions"]) + 1):
        excluded |= trigger.shift(offset, fill_value=False)
        excluded |= trigger.shift(-offset, fill_value=False)
    daily["overnight_abs_log_gap"] = overnight
    daily["stitch_trigger"] = trigger.astype(bool)
    daily["stitch_excluded"] = excluded.astype(bool)
    daily["quality_eligible"] = (
        daily["complete_session"].astype(bool) & ~daily["stitch_excluded"]
    )
    measure_columns = [
        "rv",
        "bpv",
        "tripower_quarticity",
        "bns_z",
        "jump_variation",
        "jump_share",
        "first_hour_rv_share",
        "middle_rv_share",
        "last_hour_rv_share",
    ]
    daily.loc[~daily["quality_eligible"], measure_columns] = np.nan
    daily.loc[~daily["quality_eligible"], "jump_significant"] = False
    daily.index.name = "session"
    return daily


def _read_close(path: Path) -> pd.Series:
    frame = pd.read_parquet(path)
    if "close" not in frame:
        raise AssertionError(f"{path}: no close column")
    values = pd.to_numeric(frame["close"], errors="raise")
    if values.isna().any() or (values <= 0).any():
        raise AssertionError(f"{path}: close is not finite and positive")
    return values


def _lag_close(source: pd.Series, sessions: Iterable) -> pd.Series:
    values = pd.to_numeric(source.copy(), errors="coerce")
    index = pd.DatetimeIndex(pd.to_datetime(values.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    values.index = index.normalize()
    if values.index.duplicated().any():
        raise AssertionError("Cboe input has duplicate dates")
    target = pd.DatetimeIndex(pd.to_datetime(list(sessions))).tz_localize(None).normalize()
    return values.sort_index().reindex(target).shift(1)


def _targets(daily: pd.DataFrame, horizon: int) -> pd.DataFrame:
    frame = daily.sort_index()
    result = pd.DataFrame(index=frame.index)
    result["target_end"] = pd.NaT
    result["event"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    for position in range(max(0, len(frame) - horizon)):
        future = frame.iloc[position + 1 : position + 1 + horizon]
        if len(future) != horizon or not future["quality_eligible"].astype(bool).all():
            continue
        if future["jump_significant"].isna().any():
            continue
        origin = frame.index[position]
        result.loc[origin, "target_end"] = future.index[-1]
        result.loc[origin, "event"] = bool(future["jump_significant"].astype(bool).any())
    return result


def _design(daily: pd.DataFrame, vxn: pd.Series, skew: pd.Series, protocol: dict) -> pd.DataFrame:
    frame = daily.sort_index().loc[: pd.Timestamp(protocol["fences"]["data_end"])]
    eligible = frame["rv"].where(frame["quality_eligible"] & (frame["rv"] > 0))
    design = pd.DataFrame(index=frame.index)
    design["log_rv"] = np.log(eligible)
    design["log_rv_5"] = np.log(eligible.rolling(5, min_periods=5).mean())
    design["log_rv_22"] = np.log(eligible.rolling(22, min_periods=22).mean())
    design["bns_z"] = frame["bns_z"].where(frame["quality_eligible"])
    design["jump_share_5"] = frame["jump_share"].where(
        frame["quality_eligible"]
    ).rolling(5, min_periods=5).mean()
    complete_sessions = frame.index[frame["quality_eligible"].astype(bool)]
    vxn_lag = _lag_close(vxn, complete_sessions).reindex(frame.index)
    skew_lag = _lag_close(skew, complete_sessions).reindex(frame.index)
    design["log_vxn_lag1"] = np.log(vxn_lag.where(vxn_lag > 0))
    design["log_skew_lag1"] = np.log(skew_lag.where(skew_lag > 0))
    return design.join(_targets(frame, int(protocol["diagnostic"]["horizon_sessions"])))


def _probability(
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


def _forecasts(design: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    diagnostic = protocol["diagnostic"]
    baseline = list(diagnostic["baseline_features"])
    augmented = list(diagnostic["augmented_features"])
    required = list(dict.fromkeys([*augmented, "event", "target_end"]))
    outputs: list[pd.DataFrame] = []
    for year in range(
        int(diagnostic["first_test_year"]), int(diagnostic["final_test_year"]) + 1
    ):
        test_start = pd.Timestamp(f"{year}-01-01")
        test_end = pd.Timestamp(f"{year}-12-31")
        completed = pd.to_datetime(design["target_end"], errors="coerce") < test_start
        train = design.loc[(design.index < test_start) & completed].dropna(subset=required)
        test = design.loc[
            (design.index >= test_start) & (design.index <= test_end)
        ].dropna(subset=required)
        if len(train) < int(diagnostic["min_train_rows"]):
            continue
        if train["event"].astype(int).nunique() != 2:
            continue
        if test["event"].astype(int).nunique() != 2:
            continue
        part = test[["target_end", "event"]].copy()
        part["fold_year"] = year
        part["p_price_history"] = _probability(train, test, baseline, protocol)
        part["p_augmented"] = _probability(train, test, augmented, protocol)
        outputs.append(part)
    if not outputs:
        return pd.DataFrame(
            columns=["target_end", "event", "fold_year", "p_price_history", "p_augmented"],
            index=pd.DatetimeIndex([], name="origin"),
        )
    result = pd.concat(outputs).sort_index()
    result.index.name = "origin"
    return result


def _fold_diagnostics(design: pd.DataFrame, protocol: dict) -> list[dict]:
    """Explain frozen annual-fold gates without relaxing any after the run."""
    diagnostic = protocol["diagnostic"]
    required = list(
        dict.fromkeys([*diagnostic["augmented_features"], "event", "target_end"])
    )
    minimum = int(diagnostic["min_train_rows"])
    rows: list[dict] = []
    for year in range(
        int(diagnostic["first_test_year"]), int(diagnostic["final_test_year"]) + 1
    ):
        test_start = pd.Timestamp(f"{year}-01-01")
        test_end = pd.Timestamp(f"{year}-12-31")
        completed = pd.to_datetime(design["target_end"], errors="coerce") < test_start
        train = design.loc[(design.index < test_start) & completed].dropna(subset=required)
        test = design.loc[
            (design.index >= test_start) & (design.index <= test_end)
        ].dropna(subset=required)
        train_classes = int(train["event"].astype(int).nunique()) if len(train) else 0
        test_classes = int(test["event"].astype(int).nunique()) if len(test) else 0
        failed: list[str] = []
        if len(train) < minimum:
            failed.append("min_train_rows")
        if train_classes != 2:
            failed.append("train_requires_both_classes")
        if test_classes != 2:
            failed.append("test_requires_both_classes")
        rows.append(
            {
                "test_year": year,
                "common_origin_train_rows": int(len(train)),
                "common_origin_test_rows": int(len(test)),
                "train_event_rows": int(train["event"].astype(int).sum()) if len(train) else 0,
                "test_event_rows": int(test["event"].astype(int).sum()) if len(test) else 0,
                "failed_gates": failed,
                "evaluable": not failed,
            }
        )
    return rows


def _top_lift(y: np.ndarray, probability: np.ndarray, fraction: float) -> float:
    base = float(np.mean(y))
    if base <= 0 or not len(y):
        return np.nan
    count = max(1, int(math.ceil(fraction * len(y))))
    order = np.argsort(-probability, kind="stable")[:count]
    return float(np.mean(y[order]) / base)


def _phases(forecasts: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    rows: list[dict] = []
    step = int(protocol["diagnostic"]["phase_step_sessions"])
    fraction = float(protocol["diagnostic"]["top_fraction"])
    frame = forecasts.sort_index()
    for phase in range(step):
        sample = frame.iloc[phase::step].dropna(
            subset=["event", "p_price_history", "p_augmented"]
        )
        y = sample["event"].astype(int).to_numpy()
        row = {
            "phase": phase,
            "n": int(len(sample)),
            "event_rate": float(np.mean(y)) if len(y) else np.nan,
        }
        for model in ("price_history", "augmented"):
            probability = sample[f"p_{model}"].to_numpy(dtype=float)
            row[f"{model}_auc"] = (
                float(roc_auc_score(y, probability))
                if len(y) and len(np.unique(y)) == 2
                else np.nan
            )
            row[f"{model}_top_decile_lift"] = (
                _top_lift(y, probability, fraction) if len(y) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _metric_summary(forecasts: pd.DataFrame, phases: pd.DataFrame, protocol: dict) -> dict:
    def summarize(column: str) -> dict:
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
        "event_rate": (
            float(forecasts["event"].astype(float).mean()) if len(forecasts) else None
        ),
        "price_history_auc": summarize("price_history_auc"),
        "augmented_auc": summarize("augmented_auc"),
        "price_history_top_decile_lift": summarize("price_history_top_decile_lift"),
        "augmented_top_decile_lift": summarize("augmented_top_decile_lift"),
    }


def _nanosecond_index(index: pd.Index) -> pd.DatetimeIndex:
    normalized = pd.DatetimeIndex(pd.to_datetime(index, errors="raise"))
    if normalized.tz is not None:
        normalized = normalized.tz_localize(None)
    # Parquet may round-trip an empty index as datetime64[ms] while a newly
    # constructed empty DatetimeIndex uses datetime64[s] or [ns].  Normalize
    # the storage unit; there are no timestamp values to disagree about.
    return pd.DatetimeIndex(normalized.to_numpy(dtype="datetime64[ns]"))


def _assert_frame_equal(name: str, actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    actual = actual.copy()
    expected = expected.copy()
    actual.index = _nanosecond_index(actual.index)
    expected.index = _nanosecond_index(expected.index)
    if set(actual.columns) != set(expected.columns):
        raise AssertionError(
            f"{name} columns differ: actual={sorted(actual.columns)}, "
            f"expected={sorted(expected.columns)}"
        )
    try:
        pd.testing.assert_frame_equal(
            actual[expected.columns],
            expected,
            check_dtype=False,
            check_freq=False,
            check_names=False,
            rtol=1e-11,
            atol=1e-14,
        )
    except AssertionError as exc:
        raise AssertionError(f"{name} differs from independent reconstruction: {exc}") from exc


def _assert_json_close(actual: object, expected: object, path: str = "metrics") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError(f"{path}: keys or type differ")
        for key in expected:
            _assert_json_close(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, (float, np.floating)):
        if actual is None or not np.isclose(float(actual), float(expected), rtol=1e-12, atol=1e-14):
            raise AssertionError(f"{path}: {actual!r} != {expected!r}")
        return
    if actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def _assert_manifest_hashes(manifest: dict, protocol: dict) -> None:
    expected = {
        ("protocol", "sha256"): sha256(PROTOCOL_PATH),
        ("implementation", "sha256"): sha256(IMPLEMENTATION_PATH),
        ("source", "sha256"): sha256(ROOT / protocol["source"]["raw_path"]),
        ("daily_panel", "sha256"): sha256(ROOT / protocol["outputs"]["daily_panel"]),
        ("inputs", "vxn", "sha256"): sha256(ROOT / protocol["inputs"]["vxn_path"]),
        ("inputs", "skew", "sha256"): sha256(ROOT / protocol["inputs"]["skew_path"]),
        ("diagnostic_outputs", "forecasts", "sha256"): sha256(
            ROOT / protocol["outputs"]["forecasts"]
        ),
        ("diagnostic_outputs", "phase_metrics", "sha256"): sha256(
            ROOT / protocol["outputs"]["phase_metrics"]
        ),
        ("diagnostic_outputs", "metrics", "sha256"): sha256(
            ROOT / protocol["outputs"]["metrics"]
        ),
    }
    for keys, wanted in expected.items():
        value: object = manifest
        try:
            for key in keys:
                value = value[key]  # type: ignore[index]
        except (KeyError, TypeError) as exc:
            raise AssertionError(f"manifest is missing {'.'.join(keys)}") from exc
        if value != wanted:
            raise AssertionError(f"manifest hash mismatch for {'.'.join(keys)}")
    if manifest.get("evidence_class") != protocol["status"]:
        raise AssertionError("manifest evidence class differs from frozen protocol")


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def verify(protocol: dict | None = None) -> dict:
    protocol = protocol or load_protocol()
    missing = preflight(protocol)
    if missing:
        raise FileNotFoundError("missing required NQ artifacts: " + ", ".join(missing))

    manifest = json.loads((ROOT / protocol["outputs"]["manifest"]).read_text())
    _assert_manifest_hashes(manifest, protocol)

    rth, raw_rows = _read_fenced_rth(ROOT / protocol["source"]["raw_path"], protocol)
    rebuilt_daily = _rebuild_daily(rth, protocol)
    saved_daily = pd.read_parquet(ROOT / protocol["outputs"]["daily_panel"])
    _assert_frame_equal("daily panel", saved_daily, rebuilt_daily)
    data_end = pd.Timestamp(protocol["fences"]["data_end"])
    if len(saved_daily) and pd.Timestamp(saved_daily.index.max()) > data_end:
        raise AssertionError("saved daily panel crosses the frozen data fence")

    vxn = _read_close(ROOT / protocol["inputs"]["vxn_path"])
    skew = _read_close(ROOT / protocol["inputs"]["skew_path"])
    rebuilt_design = _design(rebuilt_daily, vxn, skew, protocol)
    rebuilt_forecasts = _forecasts(rebuilt_design, protocol)
    fold_diagnostics = _fold_diagnostics(rebuilt_design, protocol)
    saved_forecasts = pd.read_parquet(ROOT / protocol["outputs"]["forecasts"])
    _assert_frame_equal("forecasts", saved_forecasts, rebuilt_forecasts)
    if not len(rebuilt_forecasts) and any(row["evaluable"] for row in fold_diagnostics):
        raise AssertionError("forecast artifact is empty despite an evaluable frozen fold")
    if len(saved_forecasts):
        origins = pd.DatetimeIndex(pd.to_datetime(saved_forecasts.index)).tz_localize(None)
        target_end = pd.to_datetime(saved_forecasts["target_end"], errors="raise")
        if origins.max() > data_end or target_end.max() > data_end:
            raise AssertionError("forecast origin or outcome crosses the frozen data fence")
        for column in ("p_price_history", "p_augmented"):
            if not saved_forecasts[column].between(0, 1, inclusive="both").all():
                raise AssertionError(f"{column} contains an invalid probability")

    rebuilt_phases = _phases(rebuilt_forecasts, protocol)
    saved_phases = pd.read_parquet(ROOT / protocol["outputs"]["phase_metrics"])
    # Phase artifacts use a default integer index, unlike time-indexed panels.
    try:
        pd.testing.assert_frame_equal(
            saved_phases.reset_index(drop=True),
            rebuilt_phases.reset_index(drop=True),
            check_dtype=False,
            rtol=1e-12,
            atol=1e-14,
        )
    except AssertionError as exc:
        raise AssertionError(f"phase metrics differ from independent reconstruction: {exc}") from exc

    rebuilt_metrics = _metric_summary(rebuilt_forecasts, rebuilt_phases, protocol)
    saved_metrics = json.loads((ROOT / protocol["outputs"]["metrics"]).read_text())
    _assert_json_close(saved_metrics, rebuilt_metrics)

    no_evaluable_folds = not len(rebuilt_forecasts)
    report = {
        "status": (
            "VERIFIED_NO_EVALUABLE_FOLDS"
            if no_evaluable_folds
            else "VERIFIED_DIAGNOSTIC_ONLY"
        ),
        "evidence_class": protocol["status"],
        "independent_implementation_imported": False,
        "unknown_contract_stitch": True,
        "no_evaluable_folds": no_evaluable_folds,
        "frozen_min_train_rows": int(protocol["diagnostic"]["min_train_rows"]),
        "fold_diagnostics": fold_diagnostics,
        "raw_rows": int(raw_rows),
        "retained_rth_rows": int(len(rth)),
        "daily_sessions": int(len(rebuilt_daily)),
        "quality_eligible_sessions": int(rebuilt_daily["quality_eligible"].sum()),
        "stitch_excluded_sessions": int(rebuilt_daily["stitch_excluded"].sum()),
        "forecast_origins": int(len(rebuilt_forecasts)),
        "checks": [
            "frozen_protocol",
            "all_manifest_hashes",
            "strict_et_and_pretransform_fence",
            "official_rth_and_terminal_session",
            "daily_rv_bpv_tripower_bns_and_shape",
            "stitch_neighborhood_flags",
            "lagged_cboe_features_and_next_five_target",
            "annual_models_and_identical_origins",
            "five_phase_auc_and_top_decile_lift",
            "origin_and_outcome_data_fences",
        ],
        "hashes": {
            "protocol": sha256(PROTOCOL_PATH),
            "implementation": sha256(IMPLEMENTATION_PATH),
            "verifier": sha256(VERIFIER_PATH),
            "raw": sha256(ROOT / protocol["source"]["raw_path"]),
        },
    }
    _atomic_json(report, ROOT / protocol["outputs"]["verification"])
    return report


def status(protocol: dict | None = None) -> dict:
    protocol = protocol or load_protocol()
    missing = preflight(protocol)
    return {
        "status": "READY" if not missing else "BLOCKED_MISSING_ARTIFACTS",
        "evidence_class": protocol["status"],
        "missing": missing,
        "independent_verifier": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("status", "verify"), default="verify")
    args = parser.parse_args(argv)
    if args.command == "status":
        print(json.dumps(status(), indent=2))
        return 0
    try:
        print(json.dumps(verify(), indent=2))
        return 0
    except FileNotFoundError as exc:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_MISSING_ARTIFACTS",
                    "error": str(exc),
                    "missing": preflight(),
                },
                indent=2,
            )
        )
        return 2
    except Exception as exc:
        print(json.dumps({"status": "VERIFICATION_FAILED", "error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
