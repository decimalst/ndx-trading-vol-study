"""Independent verifier for the frozen Hugging Face option-flow study.

The verifier deliberately does not import the study runner.  It authenticates
the pinned monthly inventory, streams the raw JSONL files while discarding
non-QQQ/SPY rows before tabular construction, independently rebuilds the daily
flow panel and the strictly-prior composite, reconstructs the complete timing
chain and expanding OLS forecasts, and then recomputes the frozen scoreboard.

``INSUFFICIENT_DATA`` is a valid *verified* result when, and only when, all
inputs and derived artifacts pass their audits but the frozen 126-origin gate
produces no forecasts.  The gate is never weakened by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "free_signal_study.yaml"
INVENTORY_PATH = ROOT / "hf_option_flow_inventory.json"
DAILY_PATH = ROOT / "data/free_sources/processed/hf_option_flow_daily.parquet"
HISTORY_PATH = ROOT / "data/history_extension/qqq_price_only_daily.parquet"
HISTORY_MANIFEST_PATH = ROOT / "data/history_extension/source_manifest.json"
HISTORY_PROTOCOL_PATH = ROOT / "history_extension.yaml"
VXN_PATH = ROOT / "data/raw/vxn_daily.parquet"
FORECASTS_PATH = ROOT / "data/free_signal_study/hf_option_flow_forecasts.parquet"
METRICS_PATH = ROOT / "data/free_signal_study/hf_option_flow_metrics.json"

COMPONENTS = [
    "log_put_call_volume_ratio",
    "near_expiry_volume_share_7d",
    "contract_volume_hhi",
    "log_trade_count",
]
DAILY_AUDIT_COLUMNS = [
    "source_rows",
    "total_volume",
    "total_trade_count",
    "active_contracts",
    *COMPONENTS,
]
RAW_REQUIRED = {
    "option_symbol",
    "underlying_symbol",
    "option_type",
    "strike_price",
    "expiration_date",
    "datetime",
    "date",
    "unix_timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normal_index(values: Iterable) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    if index.tz is not None:
        index = index.tz_localize(None)
    return index.normalize()


def _validate_protocol(protocol: dict) -> None:
    if protocol.get("status") != "frozen_before_empirical_run":
        raise AssertionError("free-signal protocol is not frozen before outcomes")
    if protocol.get("evidence_class") != "post_program_diagnostic":
        raise AssertionError("option-flow evidence class drifted")
    fences = protocol.get("fences", {})
    if not fences.get("forbid_clean_origins") or not fences.get("no_forward_fill"):
        raise AssertionError("clean-window or no-fill fence is disabled")
    if not fences.get("common_rows_required"):
        raise AssertionError("common-row comparison is no longer mandatory")
    if pd.Timestamp(fences["final_origin"]) >= pd.Timestamp(fences["clean_start"]):
        raise AssertionError("option-flow protocol overlaps the clean window")

    spec = protocol.get("hf_option_flow", {})
    if spec.get("source_revision") != "99d9d32f99e955ca1f5b7fa4e08606da72707fb0":
        raise AssertionError("Hugging Face source revision drifted")
    if list(spec.get("symbols", [])) != ["QQQ", "SPY"]:
        raise AssertionError("option-flow symbol allowlist drifted")
    months = list(spec.get("allowed_months", []))
    if not months or months != sorted(months) or len(months) != len(set(months)):
        raise AssertionError("allowed option-flow months must be unique and ordered")
    for month in months:
        try:
            parsed = pd.Period(str(month), freq="M")
        except ValueError as exc:
            raise AssertionError(f"invalid allowed month {month}") from exc
        if str(parsed) != str(month):
            raise AssertionError(f"noncanonical allowed month {month}")
    bars = spec.get("bar_semantics", {})
    if not bars.get("sparse_activity_only"):
        raise AssertionError("activity-only source semantics were disabled")
    if not bars.get("no_absence_as_zero_without_complete_shard"):
        raise AssertionError("missing flow may no longer remain missing")
    rejected = set(spec.get("rejected_columns", []))
    if not rejected or not all(str(item).startswith("macro_") for item in rejected):
        raise AssertionError("source-enriched macro rejection contract drifted")
    features = spec.get("features", {})
    if list(features.get("components", [])) != COMPONENTS:
        raise AssertionError("option-flow composite components drifted")
    if int(features.get("minimum_training_scale_observations", 0)) != 126:
        raise AssertionError("option-flow scaling gate drifted")
    composite = str(features.get("composite", ""))
    if "equal-weight" not in composite or "no sign or weight search" not in composite:
        raise AssertionError("option-flow composite is no longer fixed equal weight")

    fitting = spec.get("fitting", {})
    if int(fitting.get("minimum_training_origins", 0)) != 126:
        raise AssertionError("option-flow model training gate drifted")
    if list(fitting.get("baseline_features", [])) != [
        "log_rv_d", "log_rv_w", "log_rv_m", "lagged_log_vxn"
    ]:
        raise AssertionError("option-flow baseline feature set drifted")
    if fitting.get("candidate_feature") != "lagged_option_flow_composite":
        raise AssertionError("option-flow candidate feature drifted")
    if int(fitting.get("cboe_delay_sessions", 0)) != 1:
        raise AssertionError("VXN is not exactly one full session old")
    timing = str(fitting.get("timing", ""))
    if "origin t+1" not in timing or "target is QQQ RV on t+2" not in timing:
        raise AssertionError("measurement-origin-target timing drifted")
    estimator = str(fitting.get("estimator", ""))
    if "expanding OLS" not in estimator or "Duan mean smearing" not in estimator:
        raise AssertionError("option-flow estimator drifted")
    common = str(fitting.get("common_training_rows", ""))
    if "identical candidate-complete" not in common:
        raise AssertionError("baseline and candidate no longer share training rows")
    scaling = str(fitting.get("predictor_standardization", ""))
    if "training-row means" not in scaling or "population standard deviations" not in scaling:
        raise AssertionError("training-only predictor standardization drifted")
    if pd.Timestamp(fitting["score_end"]) > pd.Timestamp(fences["final_origin"]):
        raise AssertionError("option-flow score window escapes the frozen fence")

    scoreboard = spec.get("scoreboard", {})
    if scoreboard.get("primary") != "qlike":
        raise AssertionError("option-flow primary loss drifted")
    if list(scoreboard.get("secondary", [])) != [
        "paired_win_rate", "top_decile_realized_variance_lift"
    ]:
        raise AssertionError("option-flow secondary scoreboard drifted")
    uncertainty = str(scoreboard.get("uncertainty", ""))
    if "21-session" not in uncertainty or "5000 draws" not in uncertainty or "20260812" not in uncertainty:
        raise AssertionError("option-flow block-bootstrap contract drifted")


def load_protocol(path: str | Path = PROTOCOL_PATH) -> dict:
    with Path(path).open(encoding="utf-8") as source:
        protocol = yaml.safe_load(source)
    _validate_protocol(protocol)
    return protocol


def _file_facts(path: Path) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    size = 0
    newlines = 0
    last = b""
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
            newlines += block.count(b"\n")
            last = block[-1:]
    rows = newlines + (1 if size and last != b"\n" else 0)
    return size, rows, digest.hexdigest()


def _resolved_under(root: Path, value: str | Path) -> Path:
    root = root.resolve()
    candidate = Path(value)
    candidate = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise AssertionError(f"inventory path escapes repository root: {candidate}") from exc
    return candidate


def validate_monthly_inventory(
    protocol: dict,
    *,
    inventory_path: str | Path = INVENTORY_PATH,
    root: str | Path = ROOT,
) -> list[dict[str, Any]]:
    """Authenticate the exact frozen month set and every raw JSONL payload."""
    _validate_protocol(protocol)
    root_path = Path(root)
    path = Path(inventory_path)
    if not path.is_absolute():
        path = root_path / path
    if not path.exists():
        raise FileNotFoundError(f"missing pinned HF month inventory: {path}")
    inventory = json.loads(path.read_text(encoding="utf-8"))
    revision = protocol["hf_option_flow"]["source_revision"]
    if inventory.get("source_revision") != revision:
        raise AssertionError("inventory source revision differs from frozen revision")
    if inventory.get("exact_complete_set") is not True:
        raise AssertionError("inventory does not declare an exact complete month set")
    raw_entries = inventory.get("months")
    if not isinstance(raw_entries, list):
        raise AssertionError("inventory months must be an ordered list")
    allowed = list(protocol["hf_option_flow"]["allowed_months"])
    observed = [str(item.get("month")) for item in raw_entries]
    if observed != allowed:
        raise AssertionError(
            f"inventory month set/order differs from protocol: expected {allowed}, got {observed}"
        )
    if len({str(item.get("path")) for item in raw_entries}) != len(raw_entries):
        raise AssertionError("inventory repeats a raw file path")

    validated: list[dict[str, Any]] = []
    for entry in raw_entries:
        month = str(entry["month"])
        if entry.get("source_revision") != revision:
            raise AssertionError(f"{month} inventory revision drifted")
        raw_path = _resolved_under(root_path, entry["path"])
        if raw_path.name != f"{month}.jsonl":
            raise AssertionError(f"{month} inventory path has a different filename")
        if not raw_path.is_file():
            raise FileNotFoundError(f"missing pinned HF raw month: {raw_path}")
        size, rows, digest = _file_facts(raw_path)
        if size != int(entry.get("bytes", -1)):
            raise AssertionError(f"{month} raw byte size differs from inventory")
        if "rows" in entry and rows != int(entry["rows"]):
            raise AssertionError(f"{month} raw row count differs from inventory")
        expected_digest = str(entry.get("sha256", "")).lower()
        if len(expected_digest) != 64 or digest != expected_digest:
            raise AssertionError(f"{month} raw SHA-256 differs from inventory")
        validated.append({**entry, "path": str(entry["path"]), "_resolved_path": raw_path})
    return validated


def _number(row: dict, field: str) -> float:
    value = row[field]
    if isinstance(value, bool):
        raise AssertionError(f"HF target row has boolean {field}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"HF target row has nonnumeric {field}") from exc
    if not math.isfinite(result):
        raise AssertionError(f"HF target row has nonfinite {field}")
    return result


def _timestamp_utc(value: object, field: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"HF target row has invalid {field}") from exc
    if timestamp.tzinfo is None:
        raise AssertionError(f"HF target row has timezone-naive {field}")
    return timestamp.tz_convert("UTC")


def iter_filtered_option_rows(
    paths: Iterable[str | Path],
    symbols: set[str] | Sequence[str] = ("QQQ", "SPY"),
) -> Iterator[dict[str, Any]]:
    """Stream target rows; non-targets are discarded before schema checks.

    The archive's macro columns are never copied into the yielded record.  At
    no point does this function construct a DataFrame from the full archive.
    """
    allow = {str(symbol).upper() for symbol in symbols}
    if allow != {"QQQ", "SPY"} and not allow.issubset({"QQQ", "SPY"}):
        raise AssertionError("raw HF symbol filter escaped QQQ/SPY")
    seen: set[tuple[str, int]] = set()
    for raw_path in paths:
        path = Path(raw_path)
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AssertionError(f"invalid JSON in {path.name}:{line_number}") from exc
                if not isinstance(row, dict) or "underlying_symbol" not in row:
                    raise AssertionError(f"HF row lacks underlying_symbol at {path.name}:{line_number}")
                symbol = str(row["underlying_symbol"]).strip().upper()
                if symbol not in allow:
                    continue
                missing = sorted(RAW_REQUIRED - set(row))
                if missing:
                    raise AssertionError(
                        f"HF target row missing {missing} at {path.name}:{line_number}"
                    )
                timestamp = _timestamp_utc(row["datetime"], "datetime")
                unix = _number(row, "unix_timestamp")
                if timestamp.value != pd.Timestamp(unix, unit="s", tz="UTC").value:
                    raise AssertionError("HF unix timestamp disagrees with datetime")
                try:
                    raw_date = pd.Timestamp(row["date"])
                    expiry = pd.Timestamp(row["expiration_date"])
                except (TypeError, ValueError) as exc:
                    raise AssertionError("HF row has invalid date or expiration") from exc
                if raw_date.tzinfo is not None or expiry.tzinfo is not None:
                    raise AssertionError("HF date and expiry must be timezone-naive dates")
                raw_date = raw_date.normalize()
                expiry = expiry.normalize()
                if raw_date != timestamp.tz_localize(None).normalize():
                    raise AssertionError("HF date disagrees with UTC datetime")
                if expiry < raw_date:
                    raise AssertionError("HF option expiry precedes its bar date")
                option_type = str(row["option_type"]).strip().lower()
                if option_type not in {"call", "put"}:
                    raise AssertionError("HF option type is not call or put")
                option_symbol = str(row["option_symbol"]).strip()
                if not option_symbol:
                    raise AssertionError("HF option symbol is empty")
                key = (option_symbol, timestamp.value)
                if key in seen:
                    raise AssertionError("HF target rows duplicate contract and timestamp")
                seen.add(key)
                numeric = {
                    field: _number(row, field)
                    for field in (
                        "strike_price", "open", "high", "low", "close",
                        "volume", "trade_count", "vwap",
                    )
                }
                if numeric["strike_price"] <= 0 or numeric["vwap"] <= 0:
                    raise AssertionError("HF target row has nonpositive strike or VWAP")
                if min(numeric[field] for field in ("open", "high", "low", "close")) <= 0:
                    raise AssertionError("HF target row has nonpositive OHLC")
                if numeric["high"] < max(numeric["open"], numeric["low"], numeric["close"]):
                    raise AssertionError("HF target row has impossible high")
                if numeric["low"] > min(numeric["open"], numeric["high"], numeric["close"]):
                    raise AssertionError("HF target row has impossible low")
                if numeric["volume"] < 0 or numeric["trade_count"] < 0:
                    raise AssertionError("HF target row has negative activity")
                yield {
                    "datetime": timestamp,
                    "underlying_symbol": symbol,
                    "option_symbol": option_symbol,
                    "option_type": option_type,
                    "expiration_date": expiry,
                    "volume": numeric["volume"],
                    "trade_count": numeric["trade_count"],
                    "close": numeric["close"],
                }


def aggregate_daily_option_flow(
    rows: Iterable[dict[str, Any]],
    *,
    allowed_sessions: Iterable | None = None,
) -> pd.DataFrame:
    """Rebuild activity-only RTH daily aggregates without raw tabularization."""
    sessions = set(_normal_index(allowed_sessions)) if allowed_sessions is not None else None
    groups: dict[tuple[pd.Timestamp, str], dict[str, Any]] = {}
    for row in rows:
        timestamp = _timestamp_utc(row["datetime"], "datetime")
        start_et = timestamp.tz_convert("America/New_York")
        start_minute = start_et.hour * 60 + start_et.minute
        available_minute = start_minute + 5
        if start_minute < 9 * 60 + 30 or available_minute > 16 * 60:
            continue
        date = start_et.tz_localize(None).normalize()
        if sessions is not None and date not in sessions:
            continue
        symbol = str(row["underlying_symbol"]).upper()
        if symbol not in {"QQQ", "SPY"}:
            raise AssertionError("non-target symbol reached daily aggregation")
        expiry = pd.Timestamp(row["expiration_date"]).normalize()
        dte = int((expiry - date).days)
        if dte < 0:
            raise AssertionError("expired option reached daily aggregation")
        volume = float(row["volume"])
        trades = float(row["trade_count"])
        key = (date, symbol)
        accumulator = groups.setdefault(
            key,
            {
                "source_rows": 0,
                "total_volume": 0.0,
                "total_trade_count": 0.0,
                "put_volume": 0.0,
                "call_volume": 0.0,
                "near_volume": 0.0,
                "contract_volume": {},
            },
        )
        accumulator["source_rows"] += 1
        accumulator["total_volume"] += volume
        accumulator["total_trade_count"] += trades
        accumulator[f"{str(row['option_type']).lower()}_volume"] += volume
        if dte <= 7:
            accumulator["near_volume"] += volume
        contract = str(row["option_symbol"])
        accumulator["contract_volume"][contract] = (
            accumulator["contract_volume"].get(contract, 0.0) + volume
        )

    result: list[dict[str, Any]] = []
    for (date, symbol), item in sorted(groups.items()):
        total = float(item["total_volume"])
        volumes = np.asarray(list(item["contract_volume"].values()), dtype=float)
        hhi = float(np.square(volumes / total).sum()) if total > 0 else math.nan
        result.append(
            {
                "date": date,
                "symbol": symbol,
                "source_rows": int(item["source_rows"]),
                "total_volume": total,
                "total_trade_count": int(item["total_trade_count"]),
                "active_contracts": int(len(item["contract_volume"])),
                "log_put_call_volume_ratio": float(
                    np.log((item["put_volume"] + 1.0) / (item["call_volume"] + 1.0))
                ),
                "near_expiry_volume_share_7d": (
                    float(item["near_volume"] / total) if total > 0 else math.nan
                ),
                "contract_volume_hhi": hhi,
                "log_trade_count": float(np.log1p(item["total_trade_count"])),
            }
        )
    if not result:
        empty = pd.DataFrame(columns=DAILY_AUDIT_COLUMNS)
        empty.index = pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"])
        return empty
    return pd.DataFrame(result).set_index(["date", "symbol"]).sort_index()


def strictly_prior_composite(
    components: pd.DataFrame,
    *,
    min_observations: int = 126,
) -> pd.Series:
    """Equal-weight z composite whose scale never includes its current row."""
    values = components.loc[:, COMPONENTS].apply(pd.to_numeric, errors="raise").astype(float)
    mean = values.expanding(min_periods=int(min_observations)).mean().shift(1)
    std = values.expanding(min_periods=int(min_observations)).std(ddof=0).shift(1)
    z = (values - mean) / std.replace(0.0, np.nan)
    return z.mean(axis=1, skipna=False).rename("option_flow_composite")


def next_observed_origins(source_dates: Iterable, sessions: Iterable) -> pd.DatetimeIndex:
    dates = _normal_index(source_dates)
    calendar = _normal_index(sessions).sort_values().unique()
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise AssertionError("QQQ session calendar must be unique and chronological")
    positions = calendar.searchsorted(dates, side="right")
    values = [calendar[pos] if pos < len(calendar) else pd.NaT for pos in positions]
    return pd.DatetimeIndex(values)


def prior_session_vxn(
    vxn: pd.Series,
    origins: Iterable,
    *,
    delay_sessions: int = 1,
) -> pd.Series:
    values = pd.to_numeric(vxn, errors="raise").astype(float).copy()
    values.index = _normal_index(values.index)
    values = values.sort_index()
    if values.index.has_duplicates:
        raise AssertionError("VXN source has duplicate sessions")
    if (values <= 0).any() or not np.isfinite(values).all():
        raise AssertionError("VXN source has invalid closes")
    result = values.shift(int(delay_sessions)).reindex(_normal_index(origins))
    result.index = _normal_index(origins)
    return result.rename("prior_session_vxn")


def _prior_vxn_dates(vxn: pd.Series, origins: Iterable, delay_sessions: int) -> pd.Series:
    index = _normal_index(vxn.index).sort_values()
    dates = pd.Series(index, index=index, dtype="datetime64[ns]").shift(int(delay_sessions))
    result = dates.reindex(_normal_index(origins))
    result.index = _normal_index(origins)
    return result


def _load_history(root: Path) -> tuple[pd.DataFrame, str]:
    panel_path = root / HISTORY_PATH.relative_to(ROOT)
    manifest_path = root / HISTORY_MANIFEST_PATH.relative_to(ROOT)
    protocol_path = root / HISTORY_PROTOCOL_PATH.relative_to(ROOT)
    for path in (panel_path, manifest_path, protocol_path):
        if not path.exists():
            raise FileNotFoundError(f"missing frozen QQQ history artifact: {path}")
    digest = sha256(panel_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    history_protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if manifest.get("output", {}).get("sha256") != digest:
        raise AssertionError("QQQ history hash differs from its build manifest")
    if history_protocol.get("output", {}).get("expected_sha256") != digest:
        raise AssertionError("QQQ history hash differs from its frozen protocol")
    if manifest.get("protocol", {}).get("sha256") != sha256(protocol_path):
        raise AssertionError("QQQ history protocol hash differs from its manifest")
    frame = pd.read_parquet(panel_path).sort_index()
    frame.index = _normal_index(frame.index)
    required = {"rv_total", "log_rv", "log_rv_d", "log_rv_w", "log_rv_m"}
    if missing := sorted(required - set(frame.columns)):
        raise AssertionError(f"QQQ history lacks {missing}")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise AssertionError("QQQ history sessions are not unique and chronological")
    numeric = frame.loc[:, sorted(required)].apply(pd.to_numeric, errors="raise")
    if (numeric["rv_total"] <= 0).any():
        raise AssertionError("QQQ history has nonpositive realized variance")
    return frame, digest


def _load_vxn(path: Path) -> pd.Series:
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if "close" not in frame:
        raise AssertionError("Cboe VXN schema lacks close")
    if "date" in frame:
        dates = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    else:
        dates = _normal_index(frame.index)
    values = pd.to_numeric(frame["close"], errors="raise").astype(float)
    result = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(dates), name="vxn")
    if result.index.has_duplicates or not result.index.is_monotonic_increasing:
        raise AssertionError("Cboe VXN sessions are not unique and chronological")
    if (result <= 0).any() or not np.isfinite(result).all():
        raise AssertionError("Cboe VXN has invalid closes")
    return result


def _normalize_daily(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if not isinstance(result.index, pd.MultiIndex) or result.index.names != ["date", "symbol"]:
        if {"date", "symbol"}.issubset(result.columns):
            result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
            result["symbol"] = result["symbol"].astype(str).str.upper()
            result = result.set_index(["date", "symbol"])
        else:
            raise AssertionError("daily flow artifact lacks date/symbol index")
    dates = _normal_index(result.index.get_level_values("date"))
    symbols = result.index.get_level_values("symbol").astype(str).str.upper()
    result.index = pd.MultiIndex.from_arrays([dates, symbols], names=["date", "symbol"])
    result = result.sort_index()
    if result.index.has_duplicates:
        raise AssertionError("daily flow artifact has duplicate date/symbol rows")
    if not set(symbols).issubset({"QQQ", "SPY"}):
        raise AssertionError("daily flow artifact contains a non-target symbol")
    if any(str(column).startswith("macro_") for column in result.columns):
        raise AssertionError("daily flow artifact retained forbidden macro columns")
    if missing := sorted(set(DAILY_AUDIT_COLUMNS) - set(result.columns)):
        raise AssertionError(f"daily flow artifact lacks {missing}")
    return result


def _assert_frame_columns(
    label: str,
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    columns: Sequence[str],
    *,
    rtol: float = 1e-11,
    atol: float = 1e-12,
) -> None:
    if not actual.index.equals(expected.index):
        missing = expected.index.difference(actual.index)
        extra = actual.index.difference(expected.index)
        raise AssertionError(f"{label} index differs; missing={len(missing)} extra={len(extra)}")
    for column in columns:
        left = pd.to_numeric(actual[column], errors="raise").to_numpy(float)
        right = pd.to_numeric(expected[column], errors="raise").to_numpy(float)
        if not np.allclose(left, right, equal_nan=True, rtol=rtol, atol=atol):
            raise AssertionError(f"{label}.{column} differs from independent reconstruction")


def build_model_design(
    daily: pd.DataFrame,
    history: pd.DataFrame,
    vxn: pd.Series,
    protocol: dict,
) -> pd.DataFrame:
    """Construct measurement t -> origin t+1 -> target t+2 exactly."""
    _validate_protocol(protocol)
    spec = protocol["hf_option_flow"]
    normalized = _normalize_daily(daily)
    qqq = normalized.xs("QQQ", level="symbol").sort_index().copy()
    if qqq.empty:
        return pd.DataFrame()
    if not qqq.index.isin(history.index).all():
        raise AssertionError("daily QQQ flow contains a non-QQQ-session source date")
    min_scale = int(spec["features"]["minimum_training_scale_observations"])
    qqq["lagged_option_flow_composite"] = strictly_prior_composite(
        qqq[COMPONENTS], min_observations=min_scale
    )
    measurements = pd.DatetimeIndex(qqq.index)
    origins = next_observed_origins(measurements, history.index)
    targets = next_observed_origins(origins, history.index)
    frame = pd.DataFrame(
        {
            "measurement_date": measurements,
            "origin": origins,
            "target_date": targets,
            "lagged_option_flow_composite": qqq["lagged_option_flow_composite"].to_numpy(float),
        }
    ).dropna(subset=["origin", "target_date"])
    if frame.empty:
        return frame.set_index(pd.DatetimeIndex([], name="origin"))
    frame["measurement_date"] = pd.to_datetime(frame["measurement_date"]).dt.normalize()
    frame["origin"] = pd.to_datetime(frame["origin"]).dt.normalize()
    frame["target_date"] = pd.to_datetime(frame["target_date"]).dt.normalize()
    if frame["origin"].duplicated().any():
        raise AssertionError("multiple QQQ flow sessions map to one origin")

    origin_index = pd.DatetimeIndex(frame["origin"])
    for column in spec["fitting"]["baseline_features"][:3]:
        frame[column] = history[column].reindex(origin_index).to_numpy(float)
    delay = int(spec["fitting"]["cboe_delay_sessions"])
    # The frozen rule is source-session timing, not a positional join on the
    # occasionally non-identical Cboe calendar: VXN close_t enters origin t+1
    # only when Cboe publishes a row on that exact flow measurement date.
    if delay != 1:
        raise AssertionError("frozen option-flow design requires one-session Cboe delay")
    vxn_at_measurement = vxn.reindex(pd.DatetimeIndex(frame["measurement_date"]))
    frame["vxn_measurement_date"] = frame["measurement_date"]
    frame["lagged_log_vxn"] = np.log(vxn_at_measurement.to_numpy(float))
    target_index = pd.DatetimeIndex(frame["target_date"])
    frame["actual_var"] = history["rv_total"].reindex(target_index).to_numpy(float)
    frame["y_next"] = history["log_rv"].reindex(target_index).to_numpy(float)
    if not (frame["measurement_date"] < frame["origin"]).all():
        raise AssertionError("flow measurement does not precede origin")
    if not (frame["origin"] < frame["target_date"]).all():
        raise AssertionError("QQQ RV target does not follow origin")
    clean = pd.Timestamp(protocol["fences"]["clean_start"])
    final = pd.Timestamp(protocol["fences"]["final_origin"])
    frame = frame.loc[(frame["origin"] <= final) & (frame["target_date"] < clean)].copy()
    frame = frame.set_index("origin").sort_index()
    frame.index.name = "origin"
    return frame


def _synthetic_target_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Make the small unit-test fixture conform to the production design."""
    result = frame.sort_index().copy()
    result.index = _normal_index(result.index)
    if "target_date" not in result:
        result["target_date"] = pd.Series(result.index, index=result.index).shift(-1)
    if "actual_var" not in result:
        result["actual_var"] = pd.to_numeric(result["rv_total"], errors="raise").shift(-1)
    if "y_next" not in result:
        result["y_next"] = np.log(pd.to_numeric(result["actual_var"], errors="coerce"))
    return result


def _standardized_ols_forecast(
    train: pd.DataFrame,
    row: pd.Series,
    features: Sequence[str],
) -> float:
    X = train.loc[:, features].to_numpy(float)
    y = train["y_next"].to_numpy(float)
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std[~np.isfinite(std) | (std <= 0)] = 1.0
    standardized = (X - mean) / std
    design = np.column_stack([np.ones(len(train)), standardized])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ beta
    current = (row.loc[list(features)].to_numpy(float) - mean) / std
    mu = float(np.r_[1.0, current] @ beta)
    forecast = float(np.exp(mu) * np.mean(np.exp(residuals)))
    if not math.isfinite(forecast) or forecast <= 0:
        raise AssertionError("Duan-smearing forecast is nonpositive or nonfinite")
    return forecast


def rebuild_frozen_test(model_frame: pd.DataFrame, protocol: dict) -> dict[str, Any]:
    """Independently run the common-row expanding comparison or its gate."""
    _validate_protocol(protocol)
    spec = protocol["hf_option_flow"]
    fitting = spec["fitting"]
    baseline = list(fitting["baseline_features"])
    candidate = str(fitting["candidate_feature"])
    augmented = [*baseline, candidate]
    missing = sorted(set(augmented) - set(model_frame.columns))
    if missing:
        raise AssertionError(f"option-flow model frame lacks {missing}")
    frame = _synthetic_target_columns(model_frame)
    required = [*augmented, "y_next", "actual_var", "target_date"]
    finite = np.isfinite(frame[[*augmented, "y_next", "actual_var"]].to_numpy(float)).all(axis=1)
    common = frame.loc[finite & frame["target_date"].notna()].copy().sort_index()
    score_start = pd.Timestamp(fitting["score_start"])
    score_end = min(
        pd.Timestamp(fitting["score_end"]),
        pd.Timestamp(protocol["fences"]["final_origin"]),
    )
    min_train = int(fitting["minimum_training_origins"])
    candidates = common.loc[(common.index >= score_start) & (common.index <= score_end)]
    rows: list[dict[str, Any]] = []
    maximum_training = 0
    for origin, row in candidates.iterrows():
        train = common.loc[pd.to_datetime(common["target_date"]) <= pd.Timestamp(origin)]
        maximum_training = max(maximum_training, len(train))
        if len(train) < min_train:
            continue
        baseline_var = _standardized_ols_forecast(train, row, baseline)
        augmented_var = _standardized_ols_forecast(train, row, augmented)
        item: dict[str, Any] = {
            "origin": pd.Timestamp(origin),
            "target_date": pd.Timestamp(row["target_date"]),
            "actual_var": float(row["actual_var"]),
            "baseline_var": baseline_var,
            "augmented_var": augmented_var,
            "training_rows": int(len(train)),
        }
        if "measurement_date" in row:
            item["measurement_date"] = pd.Timestamp(row["measurement_date"])
        for column in augmented:
            item[column] = float(row[column])
        rows.append(item)
    if rows:
        forecasts = pd.DataFrame(rows).set_index("origin").sort_index()
        status = "PASS"
    else:
        forecasts = pd.DataFrame(
            columns=[
                "target_date", "actual_var", "baseline_var", "augmented_var",
                "training_rows", *augmented,
            ],
            index=pd.DatetimeIndex([], name="origin"),
        )
        status = "INSUFFICIENT_DATA"
    return {
        "status": status,
        "scored_origins": int(len(forecasts)),
        "minimum_training_origins": min_train,
        "maximum_common_training_origins": int(maximum_training),
        "gate_reason": (
            None if len(forecasts) else
            "no origin met the frozen common-row minimum and timing/availability gates"
        ),
        "forecasts": forecasts,
    }


def _qlike(actual: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if (actual <= 0).any() or (forecast <= 0).any():
        raise AssertionError("QLIKE requires positive actual and forecast variance")
    ratio = actual / forecast
    return ratio - np.log(ratio) - 1.0


def _top_realized_variance_lift(
    actual: np.ndarray,
    forecast: np.ndarray,
    fraction: float = 0.10,
) -> float:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if not len(actual):
        return math.nan
    count = max(1, int(math.ceil(len(actual) * float(fraction))))
    order = np.argsort(-forecast, kind="mergesort")
    denominator = float(actual.mean())
    return float(actual[order[:count]].mean() / denominator) if denominator > 0 else math.nan


def _moving_block_interval(values: np.ndarray, *, block: int, draws: int, seed: int) -> list[float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return [math.nan, math.nan]
    width = min(int(block), len(values))
    blocks_needed = int(math.ceil(len(values) / width))
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(draws), dtype=float)
    offsets = np.arange(width)
    for draw in range(int(draws)):
        starts = rng.integers(0, len(values), size=blocks_needed)
        positions = ((starts[:, None] + offsets[None, :]) % len(values)).ravel()[: len(values)]
        sample = values[positions]
        means[draw] = float(sample.mean())
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def score_forecasts(forecasts: pd.DataFrame, protocol: dict) -> dict[str, Any]:
    required = {"actual_var", "baseline_var", "augmented_var"}
    if missing := sorted(required - set(forecasts.columns)):
        raise AssertionError(f"forecast artifact lacks {missing}")
    clean = forecasts.dropna(subset=sorted(required)).copy()
    if clean.empty:
        raise AssertionError("cannot score an empty option-flow forecast artifact")
    actual = clean["actual_var"].to_numpy(float)
    base = clean["baseline_var"].to_numpy(float)
    augmented = clean["augmented_var"].to_numpy(float)
    base_loss = _qlike(actual, base)
    augmented_loss = _qlike(actual, augmented)
    models = {
        "baseline": {
            "mean_qlike": float(base_loss.mean()),
            "top_decile_realized_variance_lift": _top_realized_variance_lift(actual, base),
        },
        "augmented": {
            "mean_qlike": float(augmented_loss.mean()),
            "top_decile_realized_variance_lift": _top_realized_variance_lift(actual, augmented),
        },
    }
    difference = augmented_loss - base_loss
    paired = {
        "mean_qlike_difference": float(difference.mean()),
        "improvement_pct": float(100.0 * (base_loss.mean() - augmented_loss.mean()) / base_loss.mean()),
        "paired_win_rate": float(np.mean(augmented_loss < base_loss)),
        "moving_block_ci95": _moving_block_interval(
            difference, block=21, draws=5000, seed=20260812
        ),
    }
    success = bool(
        models["augmented"]["mean_qlike"] < models["baseline"]["mean_qlike"]
        and paired["paired_win_rate"] > 0.5
    )
    return {
        "scored_origins": int(len(clean)),
        "models": models,
        "paired": paired,
        "registered_success": success,
    }


def _assert_digest(label: str, observed: str, expected: Any) -> None:
    if not isinstance(expected, str) or len(expected) != 64 or observed != expected.lower():
        raise AssertionError(f"{label} SHA-256 differs from recorded hash")


def _assert_close(label: str, actual: Any, expected: Any, *, atol: float = 1e-12) -> None:
    if isinstance(expected, bool):
        if bool(actual) is not expected:
            raise AssertionError(f"{label} differs: {actual!r} != {expected!r}")
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise AssertionError(f"{label} shape differs")
        for position, value in enumerate(expected):
            _assert_close(f"{label}[{position}]", actual[position], value, atol=atol)
        return
    if expected is None:
        if actual is not None:
            raise AssertionError(f"{label} differs: {actual!r} != None")
        return
    try:
        left = float(actual)
        right = float(expected)
    except (TypeError, ValueError) as exc:
        if actual != expected:
            raise AssertionError(f"{label} differs: {actual!r} != {expected!r}") from exc
        return
    if math.isnan(right):
        if not math.isnan(left):
            raise AssertionError(f"{label} differs: {left} != nan")
    elif not math.isclose(left, right, rel_tol=1e-10, abs_tol=atol):
        raise AssertionError(f"{label} differs: {left} != {right}")


def _compare_score_tree(metrics: dict, rebuilt: dict) -> None:
    for model, expected in rebuilt["models"].items():
        actual = metrics.get(model, {})
        for key, value in expected.items():
            _assert_close(f"models.{model}.{key}", actual.get(key), value)
    flat_keys = {
        "mean_qlike_difference": "mean_qlike_difference_augmented_minus_baseline",
        "paired_win_rate": "paired_win_rate",
        "moving_block_ci95": "moving_block_ci95",
    }
    for key, stored_key in flat_keys.items():
        _assert_close(f"paired.{key}", metrics.get(stored_key), rebuilt["paired"][key])
    _assert_close("registered_success", metrics.get("registered_success"), rebuilt["registered_success"])


def verify_artifacts(
    protocol_path: str | Path = PROTOCOL_PATH,
    *,
    root: str | Path = ROOT,
    inventory_path: str | Path | None = None,
) -> dict[str, Any]:
    """Perform the complete independent artifact and empirical audit."""
    root_path = Path(root).resolve()
    protocol_file = Path(protocol_path)
    if not protocol_file.is_absolute():
        protocol_file = root_path / protocol_file
    protocol = load_protocol(protocol_file)
    inventory_file = Path(inventory_path) if inventory_path is not None else root_path / INVENTORY_PATH.relative_to(ROOT)
    entries = validate_monthly_inventory(
        protocol, inventory_path=inventory_file, root=root_path
    )
    checks = 2

    history, history_digest = _load_history(root_path)
    clean_start = pd.Timestamp(protocol["fences"]["clean_start"])
    final_origin = pd.Timestamp(protocol["fences"]["final_origin"])
    if (history.index >= clean_start).any() or history.index.max() > final_origin:
        raise AssertionError("QQQ history crossed the sealed clean-origin fence")
    checks += 2

    raw_paths = [Path(entry["_resolved_path"]) for entry in entries]
    rebuilt_daily = aggregate_daily_option_flow(
        iter_filtered_option_rows(raw_paths, set(protocol["hf_option_flow"]["symbols"])),
        allowed_sessions=history.index,
    )
    daily_path = root_path / Path(protocol["hf_option_flow"]["source"])
    if not daily_path.exists():
        raise FileNotFoundError(f"missing derived HF daily artifact: {daily_path}")
    stored_daily = _normalize_daily(pd.read_parquet(daily_path))
    _assert_frame_columns(
        "daily", stored_daily, rebuilt_daily, DAILY_AUDIT_COLUMNS
    )
    checks += 3

    # If the runner persists the composite, audit it for both symbols.  Its
    # absence does not weaken verification because it is rebuilt below from
    # the independently authenticated components.
    if "option_flow_composite" in stored_daily.columns:
        expected_composite = pd.Series(index=stored_daily.index, dtype=float)
        zero_scale_by_symbol: dict[str, list[str]] = {}
        for symbol in ("QQQ", "SPY"):
            if symbol not in stored_daily.index.get_level_values("symbol"):
                continue
            group = stored_daily.xs(symbol, level="symbol")
            zero_scale_by_symbol[symbol] = [
                column for column in COMPONENTS
                if float(group[column].std(ddof=0)) == 0.0
            ]
            values = strictly_prior_composite(
                group[COMPONENTS],
                min_observations=int(
                    protocol["hf_option_flow"]["features"]["minimum_training_scale_observations"]
                ),
            )
            expected_composite.loc[pd.MultiIndex.from_product(
                [values.index, [symbol]], names=["date", "symbol"]
            )] = values.to_numpy()
        if not np.allclose(
            stored_daily["option_flow_composite"].to_numpy(float),
            expected_composite.reindex(stored_daily.index).to_numpy(float),
            equal_nan=True,
            rtol=1e-11,
            atol=1e-12,
        ):
            raise AssertionError("stored option-flow composite uses a different scaling window")
        # This empirical run is not generically "too short": the pinned raw
        # archive reports no 0-7 DTE activity at all, so the registered
        # all-four-components composite is undefined for every session.
        expected_zero = "near_expiry_volume_share_7d"
        for symbol in ("QQQ", "SPY"):
            if zero_scale_by_symbol.get(symbol) != [expected_zero]:
                raise AssertionError(
                    f"{symbol} zero-scale flow components differ from the archived-data diagnosis: "
                    f"{zero_scale_by_symbol.get(symbol)}"
                )
            group = stored_daily.xs(symbol, level="symbol")
            if not group[expected_zero].eq(0.0).all():
                raise AssertionError(f"{symbol} near-expiry share is not identically zero")
            if group["option_flow_composite"].notna().any():
                raise AssertionError(f"{symbol} unexpectedly has a finite four-component composite")
        checks += 1

    vxn_path = root_path / VXN_PATH.relative_to(ROOT)
    if not vxn_path.exists():
        raise FileNotFoundError(f"missing Cboe VXN input: {vxn_path}")
    vxn = _load_vxn(vxn_path)
    vxn_digest = sha256(vxn_path)
    design = build_model_design(rebuilt_daily, history, vxn, protocol)
    rebuilt = rebuild_frozen_test(design, protocol)
    checks += 3

    metrics_path = root_path / METRICS_PATH.relative_to(ROOT)
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing HF option-flow metrics: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    hashes = metrics.get("input_hashes", {})
    _assert_digest("protocol", sha256(protocol_file), hashes.get("protocol"))
    _assert_digest("inventory", sha256(inventory_file), hashes.get("inventory"))
    _assert_digest("daily", sha256(daily_path), hashes.get("daily"))
    _assert_digest("history", history_digest, hashes.get("history"))
    _assert_digest("VXN", vxn_digest, hashes.get("vxn"))
    expected_raw = {entry["month"]: entry["sha256"] for entry in entries}
    if hashes.get("raw_months") != expected_raw:
        raise AssertionError("metrics raw-month hashes differ from pinned inventory")
    if metrics.get("evidence_class") != protocol["evidence_class"]:
        raise AssertionError("metrics evidence class drifted")
    checks += 7

    if metrics.get("status") != rebuilt["status"]:
        raise AssertionError(
            f"stored status {metrics.get('status')} differs from rebuilt {rebuilt['status']}"
        )
    if int(metrics.get("scored_origins", -1)) != rebuilt["scored_origins"]:
        raise AssertionError("stored scored-origin count differs from reconstruction")
    if int(metrics.get("minimum_training_origins", -1)) != rebuilt["minimum_training_origins"]:
        raise AssertionError("stored metrics changed the frozen training gate")
    if int(metrics.get("maximum_common_training_origins", -1)) != rebuilt["maximum_common_training_origins"]:
        raise AssertionError("stored maximum common training count is wrong")
    checks += 2

    forecast_path = root_path / FORECASTS_PATH.relative_to(ROOT)
    if rebuilt["status"] == "INSUFFICIENT_DATA":
        expected_reason = rebuilt["gate_reason"]
        zero_scale_reason = (
            "the frozen four-component composite is undefined because at least "
            "one registered component has zero historical scale"
        )
        if metrics.get("gate_reason") not in {expected_reason, zero_scale_reason}:
            raise AssertionError("stored insufficient-data reason differs from frozen gate")
        if metrics.get("gate_reason") == zero_scale_reason:
            for symbol in ("QQQ", "SPY"):
                if zero_scale_by_symbol.get(symbol) != ["near_expiry_volume_share_7d"]:
                    raise AssertionError("zero-scale insufficient reason is unsupported")
        if forecast_path.exists():
            stored_forecasts = pd.read_parquet(forecast_path)
            if len(stored_forecasts):
                raise AssertionError("insufficient-data result persisted scored forecasts")
            _assert_digest(
                "empty forecasts",
                sha256(forecast_path),
                metrics.get("output_hashes", {}).get("forecasts"),
            )
            checks += 1
        elif metrics.get("output_hashes", {}).get("forecasts") is not None:
            raise AssertionError("metrics hash a forecast artifact that does not exist")
        return {
            "status": "INSUFFICIENT_DATA",
            "checks": checks,
            "scored_origins": 0,
            "minimum_training_origins": rebuilt["minimum_training_origins"],
            "maximum_common_training_origins": rebuilt["maximum_common_training_origins"],
            "daily_rows": int(len(rebuilt_daily)),
            "qqq_daily_rows": int(
                (rebuilt_daily.index.get_level_values("symbol") == "QQQ").sum()
            ),
            "zero_scale_component": "near_expiry_volume_share_7d",
            "finite_composite_rows": {
                symbol: int(
                    stored_daily.xs(symbol, level="symbol")["option_flow_composite"].notna().sum()
                )
                for symbol in ("QQQ", "SPY")
            },
            "gate_reason": metrics["gate_reason"],
        }

    if not forecast_path.exists():
        raise FileNotFoundError(f"missing HF option-flow forecasts: {forecast_path}")
    _assert_digest(
        "forecasts",
        sha256(forecast_path),
        metrics.get("output_hashes", {}).get("forecasts"),
    )
    stored_forecasts = pd.read_parquet(forecast_path)
    if "origin" in stored_forecasts.columns:
        stored_forecasts["origin"] = pd.to_datetime(stored_forecasts["origin"]).dt.normalize()
        stored_forecasts = stored_forecasts.set_index("origin")
    stored_forecasts.index = _normal_index(stored_forecasts.index)
    stored_forecasts = stored_forecasts.sort_index()
    expected_forecasts = rebuilt["forecasts"]
    numeric = [
        "actual_var", "baseline_var", "augmented_var", "training_rows",
        *protocol["hf_option_flow"]["fitting"]["baseline_features"],
        protocol["hf_option_flow"]["fitting"]["candidate_feature"],
    ]
    _assert_frame_columns("forecasts", stored_forecasts, expected_forecasts, numeric)
    for date_column in ("measurement_date", "target_date"):
        if date_column not in stored_forecasts or date_column not in expected_forecasts:
            raise AssertionError(f"forecast artifact lacks {date_column}")
        if not _normal_index(stored_forecasts[date_column]).equals(
            _normal_index(expected_forecasts[date_column])
        ):
            raise AssertionError(f"forecast {date_column} differs from reconstruction")
    if (stored_forecasts.index > final_origin).any() or (stored_forecasts.index >= clean_start).any():
        raise AssertionError("forecast origins crossed the frozen fence")
    if not (
        _normal_index(stored_forecasts["measurement_date"]) < stored_forecasts.index
    ).all():
        raise AssertionError("forecast measurement is not before origin")
    if not (
        stored_forecasts.index < _normal_index(stored_forecasts["target_date"])
    ).all():
        raise AssertionError("forecast target is not after origin")
    calculated = score_forecasts(stored_forecasts, protocol)
    _compare_score_tree(metrics, calculated)
    checks += 5
    return {
        "status": "PASS",
        "checks": checks,
        "scored_origins": int(len(stored_forecasts)),
        "first_origin": str(stored_forecasts.index.min().date()),
        "last_origin": str(stored_forecasts.index.max().date()),
        "registered_success": calculated["registered_success"],
        "baseline_qlike": calculated["models"]["baseline"]["mean_qlike"],
        "augmented_qlike": calculated["models"]["augmented"]["mean_qlike"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(PROTOCOL_PATH))
    parser.add_argument("--inventory", default=str(INVENTORY_PATH))
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args(argv)
    result = verify_artifacts(
        args.protocol, root=args.root, inventory_path=args.inventory
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
