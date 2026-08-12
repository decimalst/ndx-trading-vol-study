"""Frozen CFTC-positioning and Hugging Face option-flow diagnostics.

The source/acquisition contract is separate in ``free_data_sources.yaml``.
This module consumes only compact, hash-audited derived panels and refuses to
cross the repository's sealed clean-window boundary.
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

from .metrics import qlike
from .representation_study import (
    _fit_logistic,
    _predict_logistic,
    build_fold_targets,
    build_history_features,
    completed_training_origins,
    load_locked_history,
    roc_auc,
    top_decile_lift,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "free_signal_study.yaml"
OUTPUT_DIR = ROOT / "data" / "free_signal_study"
REPORT_DIR = ROOT / "reports" / "free_signal_study"
CFTC_FORECASTS_PATH = OUTPUT_DIR / "cftc_positioning_forecasts.parquet"
CFTC_METRICS_PATH = OUTPUT_DIR / "cftc_positioning_metrics.json"
CFTC_REPORT_PATH = REPORT_DIR / "cftc_positioning.md"
FLOW_FORECASTS_PATH = OUTPUT_DIR / "hf_option_flow_forecasts.parquet"
FLOW_METRICS_PATH = OUTPUT_DIR / "hf_option_flow_metrics.json"
FLOW_REPORT_PATH = REPORT_DIR / "hf_option_flow.md"
FLOW_INVENTORY_PATH = ROOT / "hf_option_flow_inventory.json"
FLOW_INGESTION_PATH = ROOT / "data" / "free_sources" / "processed" / "hf_option_flow_ingestion.json"
OPTION_FLOW_FIELDS = (
    "datetime", "underlying_symbol", "option_symbol", "option_type",
    "expiration_date", "volume", "trade_count", "close",
)


def load_protocol(path: Path = PROTOCOL_PATH) -> dict:
    protocol = yaml.safe_load(path.read_text())
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: dict) -> None:
    if protocol.get("status") != "frozen_before_empirical_run":
        raise ValueError("free-signal protocol must be frozen before outcomes")
    fences = protocol["fences"]
    if not fences.get("forbid_clean_origins") or not fences.get("no_forward_fill"):
        raise ValueError("clean-window and no-fill fences are mandatory")
    if pd.Timestamp(fences["final_origin"]) >= pd.Timestamp(fences["clean_start"]):
        raise ValueError("free-signal origins reach the sealed clean window")

    cftc = protocol["cftc_positioning"]
    if cftc.get("official_dataset_id") != "gpe5-46if":
        raise ValueError("CFTC source must remain the official TFF view")
    if cftc.get("contract_code") != "20974+" or cftc.get("report") != "TFF futures only":
        raise ValueError("CFTC study must use only consolidated futures-only NQ")
    if cftc.get("feature") != "leveraged_money_net_open_interest_share":
        raise ValueError("exactly one CFTC positioning feature is registered")
    ordinary = str(cftc["availability"].get("ordinary_rule", ""))
    if "10 calendar days" not in ordinary or "on or after" not in ordinary:
        raise ValueError("CFTC report dates may not be treated as availability dates")
    if len(cftc["availability"].get("excluded_report_date_ranges", ())) != 2:
        raise ValueError("known CFTC publication backlogs must remain excluded")
    if not cftc["fitting"].get("one_origin_per_release"):
        raise ValueError("weekly CFTC values may not be repeated across daily origins")
    if cftc["scoreboard"].get("primary") != "auc":
        raise ValueError("CFTC primary score must remain AUC")

    flow = protocol["hf_option_flow"]
    if flow.get("source_revision") != "99d9d32f99e955ca1f5b7fa4e08606da72707fb0":
        raise ValueError("Hugging Face revision drifted")
    expected_components = [
        "log_put_call_volume_ratio",
        "near_expiry_volume_share_7d",
        "contract_volume_hhi",
        "log_trade_count",
    ]
    if flow["features"].get("components") != expected_components:
        raise ValueError("option-flow composite components changed")
    composite = str(flow["features"].get("composite", ""))
    if "no sign or weight search" not in composite:
        raise ValueError("option-flow feature weights may not be selected")
    if not flow["bar_semantics"].get("sparse_activity_only"):
        raise ValueError("activity-only rows must remain explicit")
    if not flow["bar_semantics"].get("no_absence_as_zero_without_complete_shard"):
        raise ValueError("missing option-flow sessions may not become zero")
    if int(flow["fitting"].get("cboe_delay_sessions", 0)) != 1:
        raise ValueError("VXN must remain one full session old")
    if flow.get("inventory") != "hf_option_flow_inventory.json":
        raise ValueError("option-flow inventory path changed")
    if "next-session log RV" not in str(flow["fitting"].get("estimator", "")):
        raise ValueError("option-flow estimator is not frozen")
    if "identical candidate-complete" not in str(
        flow["fitting"].get("common_training_rows", "")
    ):
        raise ValueError("option-flow models must use identical training rows")
    if "flow and VXN close measured on session t" not in str(
        flow["fitting"].get("timing", "")
    ):
        raise ValueError("option-flow source/origin/target timing changed")
    if pd.Timestamp(flow["fitting"]["score_end"]) > pd.Timestamp(fences["final_origin"]):
        raise ValueError("option-flow score window escapes the protocol fence")


def _normal_index(values: Iterable) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(values))
    if index.tz is not None:
        index = index.tz_localize(None)
    return index.normalize()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_cftc_tff(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the official Socrata names without accepting other contracts."""
    renamed = {
        "report_date_as_yyyy_mm_dd": "report_date",
        "Report_Date_as_YYYY_MM_DD": "report_date",
        "cftc_contract_market_code": "contract_code",
        "CFTC_Contract_Market_Code": "contract_code",
        "open_interest_all": "open_interest",
        "Open_Interest_All": "open_interest",
        "lev_money_positions_long": "lev_long",
        "lev_money_positions_long_all": "lev_long",
        "Lev_Money_Positions_Long_All": "lev_long",
        "lev_money_positions_short": "lev_short",
        "lev_money_positions_short_all": "lev_short",
        "Lev_Money_Positions_Short_All": "lev_short",
    }
    data = frame.rename(columns={key: value for key, value in renamed.items() if key in frame}).copy()
    required = {"report_date", "contract_code", "open_interest", "lev_long", "lev_short"}
    if missing := required - set(data):
        raise ValueError(f"official CFTC panel missing fields: {sorted(missing)}")
    codes = data["contract_code"].astype(str).str.strip()
    if not codes.eq("20974+").all():
        raise ValueError("CFTC input contains a non-consolidated or overlapping contract")
    return data.loc[:, ["report_date", "lev_long", "lev_short", "open_interest"]]


def prepare_cftc_releases(
    reports: pd.DataFrame,
    sessions: Iterable,
    protocol: dict | None = None,
) -> pd.DataFrame:
    """Map Tuesday positions to a deliberately conservative usable origin.

    Each release contributes exactly one origin. Values are never carried to
    subsequent days. Known shutdown and ION-delay report dates are discarded.
    """
    protocol = protocol or load_protocol()
    spec = protocol["cftc_positioning"]
    required = {"report_date", "lev_long", "lev_short", "open_interest"}
    missing = required - set(reports)
    if missing:
        raise ValueError(f"CFTC input missing columns: {sorted(missing)}")
    frame = reports.loc[:, sorted(required)].copy()
    frame["report_date"] = pd.to_datetime(frame["report_date"], errors="raise").dt.normalize()
    if frame["report_date"].duplicated().any():
        raise ValueError("CFTC input has duplicate report dates")
    for column in ("lev_long", "lev_short", "open_interest"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if (frame["open_interest"] <= 0).any():
        raise ValueError("CFTC open interest must be positive")

    keep = pd.Series(True, index=frame.index)
    for start, end in spec["availability"]["excluded_report_date_ranges"]:
        keep &= ~frame["report_date"].between(pd.Timestamp(start), pd.Timestamp(end))
    frame = frame.loc[keep].sort_values("report_date").copy()
    frame["available_date"] = frame["report_date"] + pd.Timedelta(days=10)
    market_sessions = _normal_index(sessions).sort_values().unique()
    origins: list[pd.Timestamp] = []
    for available in frame["available_date"]:
        later = market_sessions[market_sessions >= available]
        origins.append(later[0] if len(later) else pd.NaT)
    frame["origin"] = origins
    frame = frame.dropna(subset=["origin"])
    if frame["origin"].duplicated().any():
        raise ValueError("multiple CFTC releases map to one scored origin")
    frame["lev_net_share"] = (
        frame["lev_long"] - frame["lev_short"]
    ) / frame["open_interest"]
    if not np.isfinite(frame["lev_net_share"]).all():
        raise ValueError("CFTC positioning share is not finite")
    return frame.reset_index(drop=True)


def aggregate_option_flow(
    bars: pd.DataFrame,
    *,
    decision_time: str = "16:00",
    observed_sessions: Iterable | None = None,
) -> pd.DataFrame:
    """Aggregate activity-only five-minute option bars into daily features.

    Source timestamps denote the interval start. A row becomes observable five
    minutes later. No absent contract/session is synthesized or forward-filled.
    """
    forbidden = [column for column in bars if str(column).startswith("macro_")]
    if forbidden:
        raise ValueError(f"source-enriched macro fields are forbidden: {forbidden}")
    required = {
        "datetime", "underlying_symbol", "option_symbol", "option_type",
        "expiration_date", "volume", "trade_count", "close",
    }
    missing = required - set(bars)
    if missing:
        raise ValueError(f"option-flow input missing columns: {sorted(missing)}")
    frame = bars.loc[:, list(required)].copy()
    dt = pd.to_datetime(frame["datetime"], utc=True, errors="raise")
    frame["bar_start_et"] = dt.dt.tz_convert("America/New_York")
    frame["available_et"] = frame["bar_start_et"] + pd.Timedelta(minutes=5)
    start_minutes = frame["bar_start_et"].dt.hour * 60 + frame["bar_start_et"].dt.minute
    available_minutes = frame["available_et"].dt.hour * 60 + frame["available_et"].dt.minute
    hour, minute = (int(value) for value in decision_time.split(":"))
    cutoff = hour * 60 + minute
    frame = frame.loc[(start_minutes >= 9 * 60 + 30) & (available_minutes <= cutoff)].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["date"] = frame["bar_start_et"].dt.tz_localize(None).dt.normalize()
    if observed_sessions is not None:
        sessions = _normal_index(observed_sessions).unique()
        frame = frame.loc[frame["date"].isin(sessions)].copy()
        if frame.empty:
            return pd.DataFrame()
    frame["expiration_date"] = pd.to_datetime(frame["expiration_date"], errors="raise").dt.normalize()
    frame["dte"] = (frame["expiration_date"] - frame["date"]).dt.days
    frame["volume"] = pd.to_numeric(frame["volume"], errors="raise")
    frame["trade_count"] = pd.to_numeric(frame["trade_count"], errors="raise")
    frame["close"] = pd.to_numeric(frame["close"], errors="raise")
    if (frame[["volume", "trade_count"]] < 0).any().any() or (frame["close"] <= 0).any():
        raise ValueError("option-flow bars contain invalid price or activity")
    if not np.equal(frame["trade_count"], np.floor(frame["trade_count"])).all():
        raise ValueError("option-flow trade_count must be an integer count")
    if (frame["dte"] < 0).any():
        raise ValueError("expired option appears in option-flow bars")
    kinds = frame["option_type"].astype(str).str.lower()
    if not kinds.isin(["call", "put"]).all():
        raise ValueError("option type must be call or put")
    frame["option_type"] = kinds

    rows: list[dict] = []
    for (date, symbol), group in frame.groupby(["date", "underlying_symbol"], sort=True):
        volume_by_contract = group.groupby("option_symbol", sort=False)["volume"].sum()
        total_volume = float(volume_by_contract.sum())
        put_volume = float(group.loc[group["option_type"] == "put", "volume"].sum())
        call_volume = float(group.loc[group["option_type"] == "call", "volume"].sum())
        near_volume = float(group.loc[group["dte"] <= 7, "volume"].sum())
        hhi = float(np.square(volume_by_contract / total_volume).sum()) if total_volume > 0 else np.nan
        rows.append({
            "date": date,
            "symbol": str(symbol),
            "total_volume": total_volume,
            "total_trade_count": int(group["trade_count"].sum()),
            "active_contracts": int(group["option_symbol"].nunique()),
            "log_put_call_volume_ratio": float(np.log((put_volume + 1.0) / (call_volume + 1.0))),
            "near_expiry_volume_share_7d": near_volume / total_volume if total_volume > 0 else np.nan,
            "contract_volume_hhi": hhi,
            "log_trade_count": float(np.log1p(group["trade_count"].sum())),
        })
    result = pd.DataFrame(rows).set_index(["date", "symbol"]).sort_index()
    return result


def training_scaled_composite(
    components: pd.DataFrame,
    *,
    min_observations: int = 126,
) -> pd.Series:
    """Equal-weight component z-score using only strictly prior rows."""
    values = components.astype(float).copy()
    mean = values.expanding(min_periods=int(min_observations)).mean().shift(1)
    std = values.expanding(min_periods=int(min_observations)).std(ddof=0).shift(1)
    z = (values - mean) / std.replace(0.0, np.nan)
    return z.mean(axis=1, skipna=False).rename("option_flow_composite")


def validate_option_flow_inventory(
    protocol: dict | None = None,
    *,
    inventory_path: Path = FLOW_INVENTORY_PATH,
    root: Path = ROOT,
    validate_hashes: bool = True,
) -> list[dict]:
    """Validate the exact immutable monthly set and, by default, every SHA.

    Full ingestion uses ``validate_hashes=False`` here because its streaming
    pass computes and checks the same digest while filtering.  This avoids a
    second 19-GB read without weakening the raw-byte contract.
    """
    protocol = protocol or load_protocol()
    spec = protocol["hf_option_flow"]
    inventory = json.loads(Path(inventory_path).read_text())
    revision = str(spec["source_revision"])
    if inventory.get("source_revision") != revision:
        raise ValueError("option-flow inventory revision differs from protocol")
    if inventory.get("exact_complete_set") is not True:
        raise ValueError("option-flow inventory must declare an exact complete set")
    entries = list(inventory.get("months", ()))
    months = [str(entry.get("month")) for entry in entries]
    allowed = list(spec["allowed_months"])
    if months != allowed or len(set(months)) != len(months):
        raise ValueError("option-flow inventory month set/order differs from protocol")
    result: list[dict] = []
    for entry in entries:
        month = str(entry["month"])
        if entry.get("source_revision") != revision:
            raise ValueError(f"{month}: source revision differs from protocol")
        expected_name = f"{month}.jsonl"
        relative = Path(str(entry["path"]))
        if relative.name != expected_name:
            raise ValueError(f"{month}: inventory path does not match month")
        path = Path(root) / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned option-flow shard: {path}")
        expected_bytes = int(entry["bytes"])
        if path.stat().st_size != expected_bytes:
            raise ValueError(
                f"{month}: byte size differs ({path.stat().st_size} != {expected_bytes})"
            )
        expected_sha = str(entry["sha256"]).lower()
        if len(expected_sha) != 64:
            raise ValueError(f"{month}: invalid pinned SHA-256")
        if validate_hashes:
            actual_sha = _sha256(path)
            if actual_sha != expected_sha:
                raise ValueError(f"{month}: raw SHA-256 differs from pinned inventory")
        resolved = dict(entry)
        resolved["resolved_path"] = path
        result.append(resolved)
    if Path(root).resolve() == ROOT.resolve():
        raw_root = ROOT / spec["raw_root"]
        actual_names = sorted(path.name for path in raw_root.glob("*.jsonl"))
        expected_names = [f"{month}.jsonl" for month in allowed]
        if actual_names != expected_names:
            raise ValueError("raw option-flow directory is not the exact allowed month set")
    return result


def _peek_underlying(line: bytes) -> str | None:
    """Extract the target discriminator without constructing a JSON object."""
    key = b'"underlying_symbol"'
    start = line.find(key)
    if start < 0:
        return None
    colon = line.find(b":", start + len(key))
    quote = line.find(b'"', colon + 1) if colon >= 0 else -1
    end = line.find(b'"', quote + 1) if quote >= 0 else -1
    if quote < 0 or end < 0:
        return None
    try:
        return line[quote + 1 : end].decode("ascii")
    except UnicodeDecodeError:
        return None


def _whitelist_option_row(record: dict) -> dict:
    missing = set(OPTION_FLOW_FIELDS) - set(record)
    if missing:
        raise ValueError(f"target option-flow row missing fields: {sorted(missing)}")
    # Source-enriched macro columns are intentionally discarded here. They
    # never enter a DataFrame or daily accumulator.
    return {column: record[column] for column in OPTION_FLOW_FIELDS}


def iter_filtered_option_rows(
    path: Path | str, symbols: set[str] | frozenset[str] = frozenset({"QQQ", "SPY"})
):
    """Yield whitelisted target rows one at a time from a JSONL shard.

    The underlying symbol is inspected in bytes first. Non-target lines are
    never decoded as JSON and never become records or frame rows.
    """
    allowed = {str(symbol).upper() for symbol in symbols}
    with Path(path).open("rb") as source:
        for line_number, line in enumerate(source, start=1):
            symbol = _peek_underlying(line)
            if symbol not in allowed:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid target JSON") from exc
            row = _whitelist_option_row(record)
            if str(row["underlying_symbol"]).upper() != symbol:
                raise ValueError(f"{path}:{line_number}: inconsistent underlying symbol")
            row["underlying_symbol"] = symbol
            yield row


def _clean_stream_row(row: dict, observed_sessions: set[pd.Timestamp]) -> dict | None:
    timestamp = pd.Timestamp(row["datetime"])
    if timestamp.tzinfo is None:
        raise ValueError("option-flow datetime must carry its UTC timezone")
    timestamp = timestamp.tz_convert("America/New_York")
    minute = timestamp.hour * 60 + timestamp.minute
    if timestamp.second or timestamp.microsecond or minute < 570 or minute > 955:
        return None
    # Five-minute interval starts are 09:30 + 5k through 15:55.
    if (minute - 570) % 5:
        raise ValueError("option-flow timestamp is not on the five-minute grid")
    date = timestamp.tz_localize(None).normalize()
    if date not in observed_sessions:
        return None
    expiration = pd.Timestamp(row["expiration_date"]).tz_localize(None).normalize()
    dte = int((expiration - date).days)
    if dte < 0:
        raise ValueError("expired option appears in option-flow source")
    option_type = str(row["option_type"]).lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option-flow type must be call or put")
    try:
        volume = float(row["volume"])
        trades = float(row["trade_count"])
        close = float(row["close"])
    except (TypeError, ValueError) as exc:
        raise ValueError("option-flow activity/close is non-numeric") from exc
    if not np.isfinite([volume, trades, close]).all():
        raise ValueError("option-flow activity/close is non-finite")
    if volume < 0 or trades < 0 or close <= 0:
        raise ValueError("option-flow activity/close is outside its valid range")
    if trades != math.floor(trades):
        raise ValueError("option-flow trade_count must be an integer count")
    return {
        "date": date,
        "symbol": str(row["underlying_symbol"]).upper(),
        "option_symbol": str(row["option_symbol"]),
        "option_type": option_type,
        "dte": dte,
        "volume": volume,
        "trade_count": trades,
    }


def _new_activity_bucket() -> dict:
    return {
        "total_volume": 0.0,
        "put_volume": 0.0,
        "call_volume": 0.0,
        "near_volume": 0.0,
        "total_trade_count": 0.0,
        "contract_volume": {},
        "accepted_rows": 0,
    }


def _update_activity_bucket(bucket: dict, row: dict) -> None:
    volume = float(row["volume"])
    bucket["total_volume"] += volume
    bucket[f"{row['option_type']}_volume"] += volume
    if int(row["dte"]) <= 7:
        bucket["near_volume"] += volume
    bucket["total_trade_count"] += float(row["trade_count"])
    contract = str(row["option_symbol"])
    bucket["contract_volume"][contract] = (
        bucket["contract_volume"].get(contract, 0.0) + volume
    )
    bucket["accepted_rows"] += 1


def _buckets_to_frame(buckets: dict[tuple[pd.Timestamp, str], dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for (date, symbol), bucket in sorted(buckets.items()):
        total = float(bucket["total_volume"])
        contracts = bucket["contract_volume"]
        hhi = (
            float(sum((float(volume) / total) ** 2 for volume in contracts.values()))
            if total > 0
            else np.nan
        )
        rows.append({
            "date": date,
            "symbol": symbol,
            "total_volume": total,
            "total_trade_count": int(bucket["total_trade_count"]),
            "active_contracts": int(len(contracts)),
            "source_rows": int(bucket["accepted_rows"]),
            "log_put_call_volume_ratio": float(
                np.log((bucket["put_volume"] + 1.0) / (bucket["call_volume"] + 1.0))
            ),
            "near_expiry_volume_share_7d": (
                float(bucket["near_volume"] / total) if total > 0 else np.nan
            ),
            "contract_volume_hhi": hhi,
            "log_trade_count": float(np.log1p(bucket["total_trade_count"])),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index(["date", "symbol"]).sort_index()


def stream_option_flow_shard(
    entry: dict,
    observed_sessions: Iterable,
    *,
    symbols: set[str] | frozenset[str] = frozenset({"QQQ", "SPY"}),
) -> tuple[pd.DataFrame, dict]:
    """Hash, filter, validate, and aggregate one shard in one bounded pass."""
    path = Path(entry["resolved_path"])
    sessions = set(_normal_index(observed_sessions))
    allowed = {str(symbol).upper() for symbol in symbols}
    digest = hashlib.sha256()
    buckets: dict[tuple[pd.Timestamp, str], dict] = {}
    raw_rows = target_rows = accepted_rows = omitted_non_session_or_time = 0
    with path.open("rb") as source:
        for line_number, line in enumerate(source, start=1):
            digest.update(line)
            raw_rows += 1
            symbol = _peek_underlying(line)
            if symbol not in allowed:
                continue
            target_rows += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid target JSON") from exc
            row = _whitelist_option_row(record)
            row["underlying_symbol"] = symbol
            clean = _clean_stream_row(row, sessions)
            if clean is None:
                omitted_non_session_or_time += 1
                continue
            if clean["date"].strftime("%Y-%m") != str(entry["month"]):
                raise ValueError(f"{path}:{line_number}: row is outside its pinned month")
            key = (clean["date"], clean["symbol"])
            bucket = buckets.setdefault(key, _new_activity_bucket())
            _update_activity_bucket(bucket, clean)
            accepted_rows += 1
    actual_sha = digest.hexdigest()
    if actual_sha != str(entry["sha256"]).lower():
        raise ValueError(f"{entry['month']}: raw SHA-256 differs from pinned inventory")
    audit = {
        "month": str(entry["month"]),
        "bytes": int(path.stat().st_size),
        "sha256": actual_sha,
        "raw_rows": int(raw_rows),
        "target_symbol_rows": int(target_rows),
        "accepted_rows": int(accepted_rows),
        "omitted_non_session_or_time": int(omitted_non_session_or_time),
        "daily_rows": int(len(buckets)),
    }
    return _buckets_to_frame(buckets), audit


def add_option_flow_composites(daily: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    if daily.empty:
        return daily.copy()
    components = list(protocol["hf_option_flow"]["features"]["components"])
    minimum = int(
        protocol["hf_option_flow"]["features"]["minimum_training_scale_observations"]
    )
    output = daily.sort_index().copy()
    output["option_flow_composite"] = np.nan
    for symbol in protocol["hf_option_flow"]["symbols"]:
        mask = output.index.get_level_values("symbol") == symbol
        if not mask.any():
            continue
        part = output.loc[mask, components]
        output.loc[mask, "option_flow_composite"] = training_scaled_composite(
            part, min_observations=minimum
        ).to_numpy()
    return output


def build_option_flow_design(
    daily: pd.DataFrame,
    history: pd.DataFrame,
    vxn: pd.Series,
    protocol: dict | None = None,
) -> pd.DataFrame:
    """Map measurement t to origin t+1 and target t+2, without any fill."""
    protocol = protocol or load_protocol()
    validate_protocol(protocol)
    required_history = {"rv_total", "log_rv_d", "log_rv_w", "log_rv_m"}
    if missing := required_history - set(history):
        raise ValueError(f"QQQ history missing columns: {sorted(missing)}")
    frame = history.loc[:, sorted(required_history)].copy().sort_index()
    frame.index = _normal_index(frame.index)
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("QQQ history index must be unique and increasing")
    final = pd.Timestamp(protocol["fences"]["final_origin"])
    frame = frame.loc[:final]
    sessions = frame.index
    flow = daily.xs("QQQ", level="symbol").sort_index()
    flow.index = _normal_index(flow.index)
    if flow.index.has_duplicates:
        raise ValueError("QQQ option-flow panel has duplicate source dates")
    if not flow.index.isin(sessions).all():
        raise ValueError("option-flow panel contains a non-QQQ source session")

    vxn_close = pd.to_numeric(vxn.copy(), errors="raise")
    vxn_close.index = _normal_index(vxn_close.index)
    if vxn_close.index.has_duplicates:
        raise ValueError("VXN close has duplicate dates")
    rows: list[dict] = []
    for measurement, values in flow.iterrows():
        source_position = int(sessions.searchsorted(measurement, side="right"))
        if source_position >= len(sessions):
            continue
        origin = sessions[source_position]
        target_position = source_position + 1
        if target_position >= len(sessions):
            continue
        target = sessions[target_position]
        if measurement not in vxn_close.index:
            continue
        actual = float(frame.loc[target, "rv_total"])
        vxn_value = float(vxn_close.loc[measurement])
        if actual <= 0 or vxn_value <= 0:
            raise ValueError("target variance and VXN must be positive")
        rows.append({
            "origin": origin,
            "measurement_date": measurement,
            "target_date": target,
            "actual_var": actual,
            "target_log_rv": float(np.log(actual)),
            "log_rv_d": float(frame.loc[origin, "log_rv_d"]),
            "log_rv_w": float(frame.loc[origin, "log_rv_w"]),
            "log_rv_m": float(frame.loc[origin, "log_rv_m"]),
            "lagged_log_vxn": float(np.log(vxn_value)),
            "lagged_option_flow_composite": float(values["option_flow_composite"]),
        })
    if not rows:
        return pd.DataFrame(columns=[
            "measurement_date", "target_date", "actual_var", "target_log_rv",
            *protocol["hf_option_flow"]["fitting"]["baseline_features"],
            protocol["hf_option_flow"]["fitting"]["candidate_feature"],
        ], index=pd.DatetimeIndex([], name="origin"))
    design = pd.DataFrame(rows).set_index("origin").sort_index()
    if design.index.has_duplicates:
        raise ValueError("multiple option-flow source dates mapped to one origin")
    if not (
        pd.to_datetime(design["measurement_date"]).to_numpy() < design.index.to_numpy()
    ).all():
        raise RuntimeError("option-flow source is not strictly before origin")
    if not (
        design.index.to_numpy() < pd.to_datetime(design["target_date"]).to_numpy()
    ).all():
        raise RuntimeError("option-flow target is not strictly after origin")
    return design


def _standardized_smeared_ols(
    train: pd.DataFrame, current: pd.Series, columns: list[str]
) -> float:
    X = train[columns].to_numpy(dtype=float)
    x = current[columns].to_numpy(dtype=float)
    mean = X.mean(axis=0)
    scale = X.std(axis=0, ddof=0)
    scale[~np.isfinite(scale) | (scale <= 0)] = 1.0
    Xz = (X - mean) / scale
    xz = (x - mean) / scale
    design = np.column_stack([np.ones(len(Xz)), Xz])
    beta, *_ = np.linalg.lstsq(
        design, train["target_log_rv"].to_numpy(dtype=float), rcond=None
    )
    residual = train["target_log_rv"].to_numpy(dtype=float) - design @ beta
    prediction = float(np.r_[1.0, xz] @ beta)
    return float(np.exp(prediction) * np.mean(np.exp(residual)))


def forecast_option_flow(
    design: pd.DataFrame, protocol: dict | None = None
) -> tuple[pd.DataFrame, dict]:
    """Expanding identical-row baseline/candidate forecasts at every origin."""
    protocol = protocol or load_protocol()
    spec = protocol["hf_option_flow"]
    fitting = spec["fitting"]
    baseline = list(fitting["baseline_features"])
    candidate = str(fitting["candidate_feature"])
    required = [
        "measurement_date", "target_date", "actual_var", "target_log_rv",
        *baseline, candidate,
    ]
    frame = design.sort_index().copy()
    if frame.index.has_duplicates:
        raise ValueError("option-flow design has duplicate origins")
    start = pd.Timestamp(fitting["score_start"])
    end = min(
        pd.Timestamp(fitting["score_end"]),
        pd.Timestamp(protocol["fences"]["final_origin"]),
    )
    minimum = int(fitting["minimum_training_origins"])
    rows: list[dict] = []
    candidate_origins = 0
    max_training = 0
    gate_counts = {"missing_current": 0, "minimum_training_origins": 0}
    for origin in frame.index[(frame.index >= start) & (frame.index <= end)]:
        current = frame.loc[origin]
        if current[required].isna().any():
            gate_counts["missing_current"] += 1
            continue
        if pd.Timestamp(current["target_date"]) > pd.Timestamp(protocol["fences"]["final_origin"]):
            continue
        candidate_origins += 1
        completed = pd.to_datetime(frame["target_date"], errors="coerce") <= origin
        prior = frame.index < origin
        train = frame.loc[prior & completed].dropna(subset=required)
        max_training = max(max_training, len(train))
        if len(train) < minimum:
            gate_counts["minimum_training_origins"] += 1
            continue
        baseline_var = _standardized_smeared_ols(train, current, baseline)
        augmented_var = _standardized_smeared_ols(
            train, current, [*baseline, candidate]
        )
        rows.append({
            "origin": origin,
            "measurement_date": pd.Timestamp(current["measurement_date"]),
            "target_date": pd.Timestamp(current["target_date"]),
            "actual_var": float(current["actual_var"]),
            "baseline_var": baseline_var,
            "augmented_var": augmented_var,
            "baseline_train_n": int(len(train)),
            "augmented_train_n": int(len(train)),
            "lagged_option_flow_composite": float(current[candidate]),
        })
    columns = [
        "measurement_date", "target_date", "actual_var", "baseline_var",
        "augmented_var", "baseline_train_n", "augmented_train_n",
        "lagged_option_flow_composite",
    ]
    forecasts = (
        pd.DataFrame(rows).set_index("origin").sort_index()
        if rows
        else pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="origin"))
    )
    diagnostics = {
        "status": "SCORED" if len(forecasts) else "INSUFFICIENT_DATA",
        "scored_origins": int(len(forecasts)),
        "candidate_complete_score_origins": int(candidate_origins),
        "minimum_training_origins": minimum,
        "maximum_common_training_origins": int(max_training),
        "gate_counts": gate_counts,
        "gate_reason": (
            None if len(forecasts)
            else "no origin met the frozen common-row minimum and timing/availability gates"
        ),
    }
    return forecasts, diagnostics


def _top_variance_lift(
    actual: np.ndarray, forecast: np.ndarray, fraction: float
) -> float:
    base = float(np.mean(actual))
    if not len(actual) or base <= 0:
        return np.nan
    count = max(1, int(math.ceil(len(actual) * fraction)))
    top = np.argsort(-forecast, kind="stable")[:count]
    return float(np.mean(actual[top]) / base)


def _moving_block_mean_interval(
    values: np.ndarray, *, block: int, draws: int, seed: int
) -> list[float]:
    data = np.asarray(values, dtype=float)
    if not len(data):
        return [np.nan, np.nan]
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(len(data) / block))
    offsets = np.arange(block)
    means = np.empty(draws, dtype=float)
    for draw in range(draws):
        starts = rng.integers(0, len(data), size=blocks)
        positions = ((starts[:, None] + offsets) % len(data)).ravel()[: len(data)]
        means[draw] = float(data[positions].mean())
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def option_flow_metrics(
    forecasts: pd.DataFrame, diagnostics: dict, protocol: dict
) -> dict:
    spec = protocol["hf_option_flow"]
    if forecasts.empty:
        return {
            **diagnostics,
            "evidence_class": protocol["evidence_class"],
            "registered_success": False,
            "verdict": "INSUFFICIENT_DATA",
        }
    actual = forecasts["actual_var"].to_numpy(dtype=float)
    baseline = forecasts["baseline_var"].to_numpy(dtype=float)
    augmented = forecasts["augmented_var"].to_numpy(dtype=float)
    baseline_loss = qlike(actual, baseline)
    augmented_loss = qlike(actual, augmented)
    forecasts["baseline_qlike"] = baseline_loss
    forecasts["augmented_qlike"] = augmented_loss
    difference = augmented_loss - baseline_loss
    fraction = 0.10
    metrics = {
        **diagnostics,
        "evidence_class": protocol["evidence_class"],
        "verdict": "PASS" if (
            augmented_loss.mean() < baseline_loss.mean()
            and np.mean(augmented_loss < baseline_loss) > 0.50
        ) else "FAIL",
        "baseline": {
            "mean_qlike": float(baseline_loss.mean()),
            "top_decile_realized_variance_lift": _top_variance_lift(
                actual, baseline, fraction
            ),
        },
        "augmented": {
            "mean_qlike": float(augmented_loss.mean()),
            "top_decile_realized_variance_lift": _top_variance_lift(
                actual, augmented, fraction
            ),
        },
        "paired_win_rate": float(np.mean(augmented_loss < baseline_loss)),
        "mean_qlike_difference_augmented_minus_baseline": float(difference.mean()),
        "moving_block_ci95": _moving_block_mean_interval(
            difference,
            block=21,
            draws=5000,
            seed=20260812,
        ),
        "bootstrap": {"block_sessions": 21, "draws": 5000, "seed": 20260812},
    }
    metrics["registered_success"] = metrics["verdict"] == "PASS"
    return metrics


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


def ingest_option_flow(protocol: dict | None = None) -> dict:
    """Stream every pinned shard once into a compact daily panel."""
    protocol = protocol or load_protocol()
    validate_protocol(protocol)
    spec = protocol["hf_option_flow"]
    history_path = ROOT / protocol["cftc_positioning"]["target"]["source"]
    history = pd.read_parquet(history_path)
    sessions = _normal_index(history.index)
    final = pd.Timestamp(protocol["fences"]["final_origin"])
    sessions = sessions[sessions <= final]
    entries = validate_option_flow_inventory(
        protocol, inventory_path=ROOT / spec["inventory"], root=ROOT,
        validate_hashes=False,
    )
    monthly_frames: list[pd.DataFrame] = []
    audits: list[dict] = []
    for entry in entries:
        frame, audit = stream_option_flow_shard(
            entry, sessions, symbols=set(spec["symbols"])
        )
        if not frame.empty:
            monthly_frames.append(frame)
        audits.append(audit)
        print(
            f"option flow {entry['month']}: {audit['raw_rows']:,} raw; "
            f"{audit['accepted_rows']:,} QQQ/SPY session rows; "
            f"{audit['daily_rows']:,} daily aggregates",
            flush=True,
        )
    if monthly_frames:
        daily = pd.concat(monthly_frames).sort_index()
        if daily.index.has_duplicates:
            raise RuntimeError("monthly option-flow shards overlap date/symbol keys")
    else:
        daily = pd.DataFrame()
    daily = add_option_flow_composites(daily, protocol)
    output_path = ROOT / spec["source"]
    _atomic_parquet(daily, output_path)
    symbol_summary: dict[str, dict] = {}
    if not daily.empty:
        for symbol in spec["symbols"]:
            mask = daily.index.get_level_values("symbol") == symbol
            part = daily.loc[mask]
            dates = part.index.get_level_values("date") if len(part) else []
            symbol_summary[symbol] = {
                "daily_rows": int(len(part)),
                "composite_rows": int(part["option_flow_composite"].notna().sum()),
                "first_date": str(min(dates).date()) if len(dates) else None,
                "last_date": str(max(dates).date()) if len(dates) else None,
            }
    audit_payload = {
        "status": "INGESTED",
        "evidence_class": protocol["evidence_class"],
        "source_revision": spec["source_revision"],
        "raw_redistribution": False,
        "non_target_rows_materialized": False,
        "absence_as_zero": False,
        "weekend_roll_forward": False,
        "months": audits,
        "symbols": symbol_summary,
        "hashes": {
            "protocol": _sha256(PROTOCOL_PATH),
            "inventory": _sha256(ROOT / spec["inventory"]),
            "history": _sha256(history_path),
            "daily": _sha256(output_path),
            "raw_months": {item["month"]: item["sha256"] for item in audits},
        },
    }
    _atomic_json(audit_payload, FLOW_INGESTION_PATH)
    return audit_payload


def _vxn_close(path: Path) -> pd.Series:
    frame = pd.read_parquet(path)
    if "close" not in frame:
        raise ValueError("VXN source has no close column")
    close = pd.to_numeric(frame["close"], errors="raise")
    if close.isna().any() or (close <= 0).any():
        raise ValueError("VXN close must be finite and positive")
    return close


def _render_option_flow_report(metrics: dict, ingestion: dict) -> str:
    lines = [
        "# Hugging Face option-flow diagnostic",
        "",
        "**Evidence class: post-program exploratory diagnostic.** The archive is activity-only,",
        "identifies its upstream bars only as various unnamed sources, and uses an `other` license.",
        "It cannot measure NBBO, spreads, open interest, implied volatility, or dealer gamma.",
        "",
        f"- Status: **{metrics['verdict']}**.",
        f"- Scored origins: {metrics['scored_origins']:,}.",
        f"- Frozen minimum common training origins: {metrics['minimum_training_origins']:,}.",
        f"- Maximum available common training origins: {metrics['maximum_common_training_origins']:,}.",
        "- Flow and VXN close from session t first enter the model at origin t+1; the target is QQQ RV at t+2.",
        "- Missing source sessions are absent, never fabricated as zero or carried forward.",
        "",
    ]
    if metrics["status"] == "INSUFFICIENT_DATA":
        lines += [
            "No registered comparison is reported because no origin met every frozen availability,",
            "common-row, and minimum-training gate. The gate was not relaxed after inspecting coverage.",
            "",
        ]
        zero_scale = metrics.get("source_quality", {}).get("zero_scale_components", {})
        for symbol, components in zero_scale.items():
            if components:
                lines.append(
                    f"- {symbol} zero-scale registered components: {', '.join(components)}."
                )
        if any(zero_scale.values()):
            lines += [
                "A zero-scale component has no training z-score. The frozen four-component",
                "equal-weight composite is therefore undefined; the implementation does not",
                "drop that component, assign it an invented zero score, or reweight the other three.",
                "",
            ]
    else:
        lines += [
            "| model | mean QLIKE | top-decile realized-variance lift |",
            "|---|---:|---:|",
            f"| HAR-IV baseline | {metrics['baseline']['mean_qlike']:.6f} | {metrics['baseline']['top_decile_realized_variance_lift']:.3f}x |",
            f"| + option-flow composite | {metrics['augmented']['mean_qlike']:.6f} | {metrics['augmented']['top_decile_realized_variance_lift']:.3f}x |",
            "",
            f"Paired win rate: {metrics['paired_win_rate']:.1%}. Registered gate: **{metrics['verdict']}**.",
            "",
        ]
    for symbol, summary in ingestion.get("symbols", {}).items():
        lines.append(
            f"- {symbol}: {summary['daily_rows']:,} observed activity days; "
            f"{summary['composite_rows']:,} have a strictly-prior scaled composite."
        )
    lines.append("")
    return "\n".join(lines)


def run_option_flow(protocol: dict | None = None) -> dict:
    protocol = protocol or load_protocol()
    validate_protocol(protocol)
    spec = protocol["hf_option_flow"]
    daily_path = ROOT / spec["source"]
    if not daily_path.is_file():
        raise FileNotFoundError(f"ingest the pinned option-flow shards first: {daily_path}")
    if not FLOW_INGESTION_PATH.is_file():
        raise FileNotFoundError(f"option-flow ingestion audit is absent: {FLOW_INGESTION_PATH}")
    ingestion = json.loads(FLOW_INGESTION_PATH.read_text())
    if ingestion.get("hashes", {}).get("daily") != _sha256(daily_path):
        raise RuntimeError("option-flow daily panel differs from its ingestion audit")
    history_path = ROOT / protocol["cftc_positioning"]["target"]["source"]
    vxn_path = ROOT / "data" / "raw" / "vxn_daily.parquet"
    history = pd.read_parquet(history_path)
    daily = pd.read_parquet(daily_path)
    design = build_option_flow_design(
        daily, history, _vxn_close(vxn_path), protocol
    )
    forecasts, diagnostics = forecast_option_flow(design, protocol)
    metrics = option_flow_metrics(forecasts, diagnostics, protocol)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(forecasts, FLOW_FORECASTS_PATH)
    metrics["input_hashes"] = {
        "protocol": _sha256(PROTOCOL_PATH),
        "inventory": _sha256(ROOT / spec["inventory"]),
        "daily": _sha256(daily_path),
        "history": _sha256(history_path),
        "vxn": _sha256(vxn_path),
        "raw_months": ingestion["hashes"]["raw_months"],
    }
    metrics["output_hashes"] = {"forecasts": _sha256(FLOW_FORECASTS_PATH)}
    components = list(spec["features"]["components"])
    component_scale: dict[str, dict] = {}
    zero_scale: dict[str, list[str]] = {}
    for symbol in spec["symbols"]:
        part = daily.xs(symbol, level="symbol") if (
            len(daily) and symbol in daily.index.get_level_values("symbol")
        ) else pd.DataFrame(columns=components)
        component_scale[symbol] = {}
        zero_scale[symbol] = []
        for component in components:
            values = pd.to_numeric(part[component], errors="coerce").dropna()
            scale = float(values.std(ddof=0)) if len(values) else np.nan
            component_scale[symbol][component] = {
                "finite_rows": int(len(values)),
                "unique_values": int(values.nunique()),
                "population_std": scale if np.isfinite(scale) else None,
                "minimum": float(values.min()) if len(values) else None,
                "maximum": float(values.max()) if len(values) else None,
            }
            if len(values) and (not np.isfinite(scale) or scale <= 0):
                zero_scale[symbol].append(component)
    if metrics["status"] == "INSUFFICIENT_DATA" and any(zero_scale.values()):
        metrics["gate_reason"] = (
            "the frozen four-component composite is undefined because at least "
            "one registered component has zero historical scale"
        )
    metrics["source_quality"] = {
        "upstream": "various unnamed sources",
        "license": "other",
        "activity_only": True,
        "no_gamma_or_open_interest_claim": True,
        "symbols": ingestion.get("symbols", {}),
        "component_scale": component_scale,
        "zero_scale_components": zero_scale,
    }
    _atomic_json(metrics, FLOW_METRICS_PATH)
    FLOW_REPORT_PATH.write_text(_render_option_flow_report(metrics, ingestion))
    return metrics


def _ranking_summary(frame: pd.DataFrame, model: str, top_fraction: float) -> dict:
    score = frame[f"p_{model}"]
    base_rate = float(frame["event"].mean())
    lift = top_decile_lift(frame["event"], score, top_fraction)
    return {
        "n": int(len(frame)),
        "positives": int(frame["event"].sum()),
        "base_rate": base_rate,
        "auc": roc_auc(frame["event"], score),
        "top_decile_lift": lift,
        "top_decile_event_rate": base_rate * lift,
    }


def join_origin_features(
    targets: pd.DataFrame,
    features: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Join model inputs without dropping valid non-events.

    Negative transition labels intentionally have ``trigger_date=NaT``.  Only
    missing model inputs make an origin unscorable; blanket ``dropna`` would
    silently turn the test set into an all-positive sample.
    """
    joined = targets.join(features[columns])
    return joined.dropna(subset=columns)


def run_cftc_positioning(protocol: dict | None = None) -> dict:
    """Score one CFTC release per week against direct RV-history controls."""
    protocol = protocol or load_protocol()
    validate_protocol(protocol)
    spec = protocol["cftc_positioning"]
    source_path = ROOT / spec["source"]
    if not source_path.exists():
        raise FileNotFoundError(f"build the CFTC panel first: {source_path}")
    source = pd.read_parquet(source_path)
    releases = prepare_cftc_releases(normalize_cftc_tff(source), load_locked_history({
        **yaml.safe_load((ROOT / "representation_study.yaml").read_text())
    }).index, protocol)

    representation_protocol = yaml.safe_load((ROOT / "representation_study.yaml").read_text())
    history = load_locked_history(representation_protocol)
    y = history["log_rv"].dropna().sort_index()
    features = build_history_features(y)
    horizon = int(spec["target"]["horizon_sessions"])
    quantile = float(spec["target"]["stress_quantile"])
    ridge = float(spec["fitting"]["ridge"])
    min_train = int(spec["fitting"]["minimum_training_releases"])
    first_year = int(spec["fitting"]["first_score_year"])
    final = pd.Timestamp(protocol["fences"]["final_origin"])
    columns = list(spec["fitting"]["baseline_features"])
    release_by_origin = releases.set_index("origin").sort_index()
    rows: list[pd.DataFrame] = []
    folds: list[dict] = []
    for year in range(first_year, final.year + 1):
        year_start = pd.Timestamp(f"{year}-01-01")
        year_end = min(pd.Timestamp(f"{year}-12-31"), final)
        prior_sessions = y.index[y.index < year_start]
        if not len(prior_sessions):
            continue
        cutoff = prior_sessions[-1]
        eligible_train_origins = completed_training_origins(y.index, cutoff, horizon)
        train_origins = release_by_origin.index[
            (release_by_origin.index < year_start)
            & release_by_origin.index.isin(eligible_train_origins)
        ]
        train_targets = build_fold_targets(
            y, cutoff=cutoff, origins=train_origins, horizon=horizon, quantile=quantile
        )
        if train_targets.empty:
            continue
        train_targets = train_targets.loc[train_targets["calm"]]
        train = features.loc[train_targets.index, columns].join(train_targets["event"])
        train["lev_net_share"] = release_by_origin.loc[train.index, "lev_net_share"]
        train = train.dropna()
        if len(train) < min_train or train["event"].nunique() != 2:
            continue
        mean = float(train["lev_net_share"].mean())
        std = float(train["lev_net_share"].std(ddof=0))
        if not np.isfinite(std) or std <= 0:
            raise RuntimeError("CFTC training feature has zero scale")
        train["cftc_lev_net_z"] = (train["lev_net_share"] - mean) / std
        baseline = _fit_logistic(train[columns], train["event"], ridge)
        augmented_columns = columns + ["cftc_lev_net_z"]
        augmented = _fit_logistic(train[augmented_columns], train["event"], ridge)

        origins = release_by_origin.index[
            (release_by_origin.index >= year_start) & (release_by_origin.index <= year_end)
        ]
        targets = build_fold_targets(
            y, cutoff=cutoff, origins=origins, horizon=horizon, quantile=quantile
        )
        if targets.empty:
            continue
        test = targets.loc[targets["calm"]].copy()
        test = join_origin_features(test, features, columns)
        if test.empty:
            continue
        test["lev_net_share"] = release_by_origin.loc[test.index, "lev_net_share"]
        test["cftc_lev_net_z"] = (test["lev_net_share"] - mean) / std
        test["p_baseline"] = _predict_logistic(baseline, test[columns])
        test["p_augmented"] = _predict_logistic(augmented, test[augmented_columns])
        test["fold_year"] = year
        test["training_cutoff"] = cutoff
        rows.append(test)
        folds.append({
            "year": year,
            "training_cutoff": str(cutoff.date()),
            "training_releases": int(len(train)),
            "feature_mean": mean,
            "feature_std": std,
        })
    if not rows:
        raise RuntimeError("CFTC positioning study produced no scored releases")
    scored = pd.concat(rows).sort_index()
    if (scored.index > final).any() or (scored.index >= pd.Timestamp(protocol["fences"]["clean_start"])).any():
        raise RuntimeError("CFTC origin crossed the frozen fence")
    top_fraction = float(spec["scoreboard"]["top_fraction"])
    summaries = {
        model: _ranking_summary(scored, model, top_fraction)
        for model in ("baseline", "augmented")
    }
    break_date = pd.Timestamp(spec["structural_break"]["date"])
    sensitivity: dict[str, dict] = {}
    for name, sample in {
        "pre_micro_inclusion": scored.loc[scored.index < break_date],
        "post_micro_inclusion": scored.loc[scored.index >= break_date],
    }.items():
        if len(sample) and sample["event"].nunique() == 2:
            sensitivity[name] = {
                model: _ranking_summary(sample, model, top_fraction)
                for model in ("baseline", "augmented")
            }
    metrics = {
        "evidence_class": protocol["evidence_class"],
        "source_sha256": _sha256(source_path),
        "first_origin": str(scored.index.min().date()),
        "last_origin": str(scored.index.max().date()),
        "models": summaries,
        "delta_auc": summaries["augmented"]["auc"] - summaries["baseline"]["auc"],
        "delta_top_decile_lift": summaries["augmented"]["top_decile_lift"] - summaries["baseline"]["top_decile_lift"],
        "registered_success": bool(
            summaries["augmented"]["auc"] > summaries["baseline"]["auc"]
            and summaries["augmented"]["top_decile_lift"] > summaries["baseline"]["top_decile_lift"]
        ),
        "structural_break_sensitivity": sensitivity,
        "folds": folds,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(CFTC_FORECASTS_PATH)
    CFTC_METRICS_PATH.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    CFTC_REPORT_PATH.write_text(_render_cftc_report(metrics))
    return metrics


def _render_cftc_report(metrics: dict) -> str:
    baseline = metrics["models"]["baseline"]
    augmented = metrics["models"]["augmented"]
    return "\n".join([
        "# CFTC Nasdaq positioning diagnostic",
        "",
        "This is a post-program, target-specific diagnostic on already inspected QQQ history. "
        "It scores one origin per conservatively delayed weekly release and never repeats a weekly value across daily rows.",
        "",
        f"- Origins: {baseline['n']:,} ({metrics['first_origin']} through {metrics['last_origin']}).",
        f"- Base event rate: {baseline['base_rate']:.2%} ({baseline['positives']} positives).",
        "",
        "| model | AUC | top-decile lift | top-decile event rate |",
        "|---|---:|---:|---:|",
        f"| RV-history benchmark | {baseline['auc']:.4f} | {baseline['top_decile_lift']:.3f}x | {baseline['top_decile_event_rate']:.2%} |",
        f"| + leveraged-money net/OI | {augmented['auc']:.4f} | {augmented['top_decile_lift']:.3f}x | {augmented['top_decile_event_rate']:.2%} |",
        "",
        f"Registered two-metric gate: **{'PASS' if metrics['registered_success'] else 'FAIL'}**. "
        f"Delta AUC {metrics['delta_auc']:+.4f}; delta lift {metrics['delta_top_decile_lift']:+.3f}x.",
        "",
        "Known federal-shutdown and 2023 ION-backlog report dates are excluded. Ordinary reports are delayed ten calendar days and then mapped to the first QQQ session on or after that date, deliberately sacrificing timeliness to avoid treating Tuesday positions as public Tuesday data. The May 2023 e-micro inclusion is reported as a fixed sensitivity, not fitted away.",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "validate", "run-cftc", "audit-flow-inputs", "ingest-flow",
            "run-flow", "all-flow",
        ],
    )
    args = parser.parse_args()
    if args.command == "validate":
        load_protocol()
        print("free signal protocol: PASS")
    elif args.command == "run-cftc":
        metrics = run_cftc_positioning()
        print(json.dumps(metrics, indent=2, sort_keys=True))
    elif args.command == "audit-flow-inputs":
        protocol = load_protocol()
        entries = validate_option_flow_inventory(protocol)
        print(json.dumps({
            "status": "PASS",
            "months": len(entries),
            "total_bytes": sum(int(item["bytes"]) for item in entries),
            "inventory_sha256": _sha256(FLOW_INVENTORY_PATH),
        }, indent=2, sort_keys=True))
    elif args.command == "ingest-flow":
        print(json.dumps(ingest_option_flow(), indent=2, sort_keys=True))
    elif args.command == "run-flow":
        print(json.dumps(run_option_flow(), indent=2, sort_keys=True))
    elif args.command == "all-flow":
        ingest_option_flow()
        print(json.dumps(run_option_flow(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
