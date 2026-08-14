"""Fail-closed acquisition contracts for free external research data.

The module plans and validates acquisitions. It never starts a download itself.
Raw files remain outside version control and become usable only after a
source-identity, byte-size, and SHA-256 manifest has been written and verified.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import shutil
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "free_data_sources.yaml"
_SHA256 = frozenset("0123456789abcdef")


def load_protocol(path: str | Path = PROTOCOL_PATH) -> dict:
    with Path(path).open(encoding="utf-8") as source:
        protocol = yaml.safe_load(source)
    validate_protocol(protocol)
    return protocol


def _valid_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and set(text) <= _SHA256


def _source_downloads(source: dict) -> list[dict]:
    if "downloads" in source:
        return list(source["downloads"])
    if "download" in source:
        return [source["download"]]
    return []


def validate_protocol(protocol: dict) -> None:
    if protocol.get("status") != "acquisition_only_no_download":
        raise ValueError("free-data protocol must remain acquisition_only_no_download")
    storage = protocol.get("storage", {})
    if storage.get("raw_root") != "data/free_sources/raw":
        raise ValueError("raw root must remain data/free_sources/raw")
    if storage.get("processed_root") != "data/free_sources/processed":
        raise ValueError("processed root must remain data/free_sources/processed")
    if storage["raw_root"] == storage["processed_root"]:
        raise ValueError("raw and processed roots must differ")
    if int(storage.get("minimum_free_reserve_bytes", 0)) <= 0:
        raise ValueError("minimum free reserve must be positive")
    if float(storage.get("download_safety_multiplier", 0)) < 1:
        raise ValueError("download safety multiplier must be at least one")
    policies = protocol.get("policies", {})
    if policies.get("raw_commit_policy") != "local_only_never_commit":
        raise ValueError("raw sources must remain local-only")
    if not policies.get("no_download_during_protocol_construction", False):
        raise ValueError("protocol construction must not authorize downloads")

    sources = protocol.get("sources", {})
    if not sources:
        raise ValueError("protocol contains no sources")
    seen_ids: set[str] = set()
    for name, source in sources.items():
        source_id = str(source.get("source_id", ""))
        if not source_id or source_id in seen_ids:
            raise ValueError(f"{name} needs a unique source identity")
        seen_ids.add(source_id)
        if not str(source.get("canonical_url", "")).startswith("https://"):
            raise ValueError(f"{name} needs a canonical HTTPS URL")
        for field in ("license_class", "provenance_class", "redistribution", "commit_policy"):
            if not source.get(field):
                raise ValueError(f"{name} needs {field}")
        if "raw_local_only" not in source["commit_policy"]:
            raise ValueError(f"{name} raw data must remain local-only")
        if source.get("immutable"):
            pin = source.get("revision", source.get("version"))
            if pin in (None, ""):
                raise ValueError(f"immutable source {name} lacks a revision/version pin")
        for hash_field in ("file_sha256", "pilot_file_sha256", "audited_sha256"):
            if hash_field in source and not _valid_sha256(source[hash_field]):
                raise ValueError(f"{name} {hash_field} is not a lowercase SHA-256")
        downloads = _source_downloads(source)
        if source.get("acquisition_enabled") and not downloads:
            raise ValueError(f"enabled source {name} has no acquisition command")
        if not source.get("acquisition_enabled") and downloads:
            raise ValueError(f"disabled source {name} must not expose a command")
        for download in downloads:
            output = Path(str(download.get("output", "")))
            if output.is_absolute() or ".." in output.parts or not output.parts:
                raise ValueError(f"{name} has an unsafe output path")
            command = str(download.get("command", ""))
            expected = f"{storage['raw_root']}/{output.as_posix()}"
            parent = f"{storage['raw_root']}/{output.parent.as_posix()}"
            targets_file = expected in command
            targets_hf_local_dir = (
                command.startswith("hf download ")
                and output.name in command
                and f"--local-dir {parent}" in command
            )
            if not (targets_file or targets_hf_local_dir):
                raise ValueError(f"{name} command must target the declared raw output")
            if "--unzip" in command:
                raise ValueError(f"{name} acquisition may not unzip before hash verification")
            expected_bytes = download.get("expected_bytes")
            if expected_bytes is not None and int(expected_bytes) <= 0:
                raise ValueError(f"{name} expected byte count must be positive")
            preflight_bytes = download.get("preflight_bytes", expected_bytes)
            if preflight_bytes is None or int(preflight_bytes) <= 0:
                raise ValueError(f"{name} needs a positive disk-preflight byte budget")

    options_schema = protocol.get("schemas", {}).get("optionsdx", {}).get("columns", [])
    if len(options_schema) != 33 or len(set(options_schema)) != 33:
        raise ValueError("OptionsDX schema must contain 33 unique columns")
    if any("OPEN_INTEREST" in column for column in options_schema):
        raise ValueError("OptionsDX source does not contain open interest")
    availability = protocol.get("availability", {})
    if int(availability.get("cftc_tff", {}).get("days", -1)) != 7:
        raise ValueError("CFTC availability must remain conservatively lagged seven days")
    if int(availability.get("cboe_daily", {}).get("sessions", -1)) != 1:
        raise ValueError("Cboe daily values must remain lagged one declared session")


def build_acquisition_plan(protocol: dict | None = None) -> list[dict]:
    protocol = load_protocol() if protocol is None else protocol
    validate_protocol(protocol)
    raw_root = protocol["storage"]["raw_root"]
    plan: list[dict] = []
    for name, source in protocol["sources"].items():
        if not source.get("acquisition_enabled", False):
            continue
        version = source.get("revision", source.get("version"))
        for download in _source_downloads(source):
            plan.append(
                {
                    "source": name,
                    "source_id": source["source_id"],
                    "source_version": str(version),
                    "output": f"{raw_root}/{download['output']}",
                    "expected_bytes": download.get("expected_bytes"),
                    "preflight_bytes": download.get(
                        "preflight_bytes", download.get("expected_bytes")
                    ),
                    "command": download["command"],
                    "commit_policy": source["commit_policy"],
                    "redistribution": source["redistribution"],
                }
            )
    return plan


def required_free_bytes(download: dict, storage: dict) -> int:
    expected = download.get("preflight_bytes", download.get("expected_bytes"))
    if expected is None:
        raise ValueError("disk preflight needs an expected or budgeted byte count")
    reserve = int(storage["minimum_free_reserve_bytes"])
    multiplier = float(storage["download_safety_multiplier"])
    return math.ceil(int(expected) * multiplier) + reserve


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def preflight_download(
    target: str | Path,
    raw_root: str | Path,
    download: dict,
    storage: dict,
    *,
    free_bytes: int | None = None,
) -> dict:
    target = Path(target)
    raw_root = Path(raw_root)
    if not _inside(target, raw_root):
        raise ValueError("download target must remain inside the raw root")
    required = required_free_bytes(download, storage)
    available = int(free_bytes) if free_bytes is not None else shutil.disk_usage(raw_root).free
    if available < required:
        raise OSError(f"insufficient free space: need {required}, found {available}")
    if target.exists():
        manifest_path = Path(str(target) + storage["manifest_suffix"])
        if not manifest_path.is_file():
            raise FileExistsError("existing raw file has no verified manifest; refusing overwrite")
        with manifest_path.open(encoding="utf-8") as source:
            manifest = json.load(source)
        _verify_file_fingerprint(target, manifest)
        raise FileExistsError("verified raw file already exists; refusing overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    return {
        "target": str(target),
        "required_free_bytes": required,
        "available_free_bytes": available,
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_text(value: str) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None or str(timestamp.tz_convert("UTC").tz) != "UTC":
        raise ValueError("retrieval timestamp must be timezone-aware UTC")
    return timestamp.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def build_raw_manifest(
    path: str | Path,
    source_id: str,
    source_version: str,
    retrieved_at_utc: str,
) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not source_id or not source_version:
        raise ValueError("manifest requires source identity and version")
    return {
        "manifest_version": 1,
        "source_id": source_id,
        "source_version": str(source_version),
        "retrieved_at_utc": _utc_text(retrieved_at_utc),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _verify_file_fingerprint(path: Path, manifest: dict) -> None:
    size = path.stat().st_size
    if int(manifest.get("bytes", -1)) != size:
        raise ValueError("raw file size differs from manifest")
    digest = sha256_file(path)
    if manifest.get("sha256") != digest:
        raise ValueError("raw file hash differs from manifest")


def verify_raw_manifest(
    path: str | Path,
    manifest: dict,
    source_id: str,
    source_version: str,
) -> None:
    if manifest.get("manifest_version") != 1:
        raise ValueError("unsupported manifest version")
    if manifest.get("source_id") != source_id:
        raise ValueError("manifest source identity mismatch")
    if str(manifest.get("source_version")) != str(source_version):
        raise ValueError("manifest source version mismatch")
    if not _valid_sha256(manifest.get("sha256", "")):
        raise ValueError("manifest SHA-256 is invalid")
    _utc_text(str(manifest.get("retrieved_at_utc", "")))
    _verify_file_fingerprint(Path(path), manifest)


def load_and_verify_manifest(
    path: str | Path,
    manifest_path: str | Path,
    source_id: str,
    source_version: str,
) -> dict:
    with Path(manifest_path).open(encoding="utf-8") as source:
        manifest = json.load(source)
    verify_raw_manifest(path, manifest, source_id, source_version)
    return manifest


def _strict_dates(values: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="raise")
    if parsed.isna().any():
        raise ValueError(f"{name} contains missing dates")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed.dt.normalize()


def _numeric(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if frame[column].isna().any():
            raise ValueError(f"{name} has missing {column}")


def _ohlc(frame: pd.DataFrame, name: str) -> None:
    columns = ["open", "high", "low", "close"]
    _numeric(frame, columns, name)
    if (frame[columns] <= 0).any().any():
        raise ValueError(f"{name} has nonpositive OHLC values")
    upper = frame[["open", "close", "low"]].max(axis=1)
    lower = frame[["open", "close", "high"]].min(axis=1)
    if (frame["high"] < upper).any() or (frame["low"] > lower).any():
        raise ValueError(f"{name} has impossible OHLC geometry")


def _unique_sorted_dates(frame: pd.DataFrame, date: str, name: str) -> None:
    if frame[date].duplicated().any():
        raise ValueError(f"{name} has duplicate dates")
    if not frame[date].is_monotonic_increasing:
        raise ValueError(f"{name} dates are not sorted")


def _same_datetimes(left: pd.Series, right: pd.Series) -> bool:
    """Compare timestamps by instant/date, independent of pandas storage units."""
    left_index = pd.DatetimeIndex(left)
    right_index = pd.DatetimeIndex(right)
    if left_index.tz is not None:
        left_index = left_index.tz_convert("UTC").tz_localize(None)
    if right_index.tz is not None:
        right_index = right_index.tz_convert("UTC").tz_localize(None)
    return np.array_equal(
        left_index.to_numpy(dtype="datetime64[ns]"),
        right_index.to_numpy(dtype="datetime64[ns]"),
    )


def parse_cboe_csv(text_or_buffer, schema: str) -> pd.DataFrame:
    frame = pd.read_csv(io.StringIO(text_or_buffer) if isinstance(text_or_buffer, str) else text_or_buffer)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if schema == "ohlc":
        required = ["date", "open", "high", "low", "close"]
        if frame.columns.tolist() != required:
            raise ValueError("Cboe OHLC schema changed")
        _ohlc(frame, "Cboe")
        values = frame[["open", "high", "low", "close"]].copy()
    elif schema == "close_only":
        if len(frame.columns) != 2 or frame.columns[0] != "date":
            raise ValueError("Cboe close-only schema changed")
        frame = frame.rename(columns={frame.columns[1]: "close"})
        _numeric(frame, ["close"], "Cboe")
        if (frame["close"] <= 0).any():
            raise ValueError("Cboe has nonpositive closes")
        values = frame[["close"]].copy()
    else:
        raise ValueError(f"unknown Cboe schema {schema}")
    frame["date"] = _strict_dates(frame["date"], "Cboe date")
    _unique_sorted_dates(frame, "date", "Cboe")
    values.index = pd.DatetimeIndex(frame["date"], name="date")
    return values


def parse_hf_spx_csv(text_or_buffer) -> pd.DataFrame:
    frame = pd.read_csv(io.StringIO(text_or_buffer) if isinstance(text_or_buffer, str) else text_or_buffer)
    frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
    required = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    if frame.columns.tolist() != required:
        raise ValueError("HF SPX schema changed")
    frame["date"] = _strict_dates(frame["date"], "HF SPX date")
    _unique_sorted_dates(frame, "date", "HF SPX")
    _ohlc(frame, "HF SPX")
    _numeric(frame, ["adj_close", "volume"], "HF SPX")
    if (frame["adj_close"] <= 0).any() or (frame["volume"] < 0).any():
        raise ValueError("HF SPX has invalid adjusted closes or volume")
    return frame.set_index("date")


_HF_OPTION_COLUMNS = {
    "option_symbol", "underlying_symbol", "option_type", "strike_price",
    "expiration_date", "datetime", "date", "unix_timestamp", "open", "high",
    "low", "close", "volume", "trade_count", "vwap",
}


def parse_hf_options_jsonl(text_or_buffer) -> pd.DataFrame:
    frame = pd.read_json(
        io.StringIO(text_or_buffer) if isinstance(text_or_buffer, str) else text_or_buffer,
        lines=True,
        convert_dates=False,
    )
    missing = sorted(_HF_OPTION_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"HF options schema missing {missing}")
    if frame.empty:
        raise ValueError("HF options input is empty")
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="raise")
    frame["date"] = _strict_dates(frame["date"], "HF options date")
    frame["expiration_date"] = _strict_dates(frame["expiration_date"], "HF options expiry")
    unix = pd.to_datetime(pd.to_numeric(frame["unix_timestamp"], errors="raise"), unit="s", utc=True)
    if not _same_datetimes(unix, frame["datetime"]):
        raise ValueError("HF options unix timestamp does not match datetime")
    utc_dates = frame["datetime"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    if not _same_datetimes(utc_dates, frame["date"]):
        raise ValueError("HF options date does not match UTC datetime")
    if (frame["expiration_date"] < frame["date"]).any():
        raise ValueError("HF options expiry precedes bar date")
    _ohlc(frame, "HF options")
    _numeric(frame, ["strike_price", "volume", "trade_count", "vwap"], "HF options")
    if (frame[["strike_price", "vwap"]] <= 0).any().any() or (
        frame[["volume", "trade_count"]] < 0
    ).any().any():
        raise ValueError("HF options has invalid price/count fields")
    if not frame["option_type"].isin(["call", "put"]).all():
        raise ValueError("HF options has invalid option type")
    if frame.duplicated(["option_symbol", "datetime"]).any():
        raise ValueError("HF options has duplicate contract timestamps")
    return frame


def _normalized_option_column(column: object) -> str:
    return str(column).lstrip("\ufeff").strip().strip("[]").strip()


def parse_optionsdx_csv(text_or_buffer, columns: list[str] | None = None) -> pd.DataFrame:
    columns = columns or load_protocol()["schemas"]["optionsdx"]["columns"]
    frame = pd.read_csv(
        io.StringIO(text_or_buffer) if isinstance(text_or_buffer, str) else text_or_buffer,
        skipinitialspace=True,
    )
    frame.columns = [_normalized_option_column(column) for column in frame.columns]
    if frame.columns.tolist() != columns:
        raise ValueError("OptionsDX schema changed")
    frame["QUOTE_READTIME"] = pd.to_datetime(frame["QUOTE_READTIME"], errors="raise")
    frame["QUOTE_DATE"] = _strict_dates(frame["QUOTE_DATE"], "OptionsDX QUOTE_DATE")
    frame["EXPIRE_DATE"] = _strict_dates(frame["EXPIRE_DATE"], "OptionsDX EXPIRE_DATE")
    if not _same_datetimes(frame["QUOTE_READTIME"].dt.normalize(), frame["QUOTE_DATE"]):
        raise ValueError("OptionsDX QUOTE_DATE disagrees with QUOTE_READTIME")
    unix = pd.to_datetime(pd.to_numeric(frame["QUOTE_UNIXTIME"], errors="raise"), unit="s", utc=True)
    unix_et = unix.dt.tz_convert("America/New_York").dt.tz_localize(None)
    if not _same_datetimes(unix_et.dt.normalize(), frame["QUOTE_DATE"]):
        raise ValueError("OptionsDX unix timestamp disagrees with QUOTE_DATE")
    read_delta = (unix_et - frame["QUOTE_READTIME"]).abs()
    if (read_delta > pd.Timedelta(minutes=2)).any():
        raise ValueError("OptionsDX unix timestamp disagrees with QUOTE_READTIME")
    expiry_unix = pd.to_datetime(
        pd.to_numeric(frame["EXPIRE_UNIX"], errors="raise"), unit="s", utc=True
    ).dt.tz_localize(None).dt.normalize()
    if not _same_datetimes(expiry_unix, frame["EXPIRE_DATE"]):
        raise ValueError("OptionsDX EXPIRE_UNIX disagrees with EXPIRE_DATE")
    if (frame["EXPIRE_DATE"] < frame["QUOTE_DATE"]).any():
        raise ValueError("OptionsDX expiry precedes quote")
    numeric = [
        column for column in columns
        if column not in {"QUOTE_READTIME", "QUOTE_DATE", "EXPIRE_DATE", "C_SIZE", "P_SIZE"}
    ]
    _numeric(frame, numeric, "OptionsDX")
    if (frame[["UNDERLYING_LAST", "STRIKE"]] <= 0).any().any():
        raise ValueError("OptionsDX has nonpositive underlying/strike")
    for side in ("C", "P"):
        if (frame[f"{side}_BID"] < 0).any() or (
            frame[f"{side}_ASK"] < frame[f"{side}_BID"]
        ).any():
            raise ValueError(f"OptionsDX has invalid {side} bid/ask")
    if (frame["DTE"] < 0).any():
        raise ValueError("OptionsDX has negative DTE")
    key = ["QUOTE_UNIXTIME", "EXPIRE_UNIX", "STRIKE"]
    if frame.duplicated(key).any():
        raise ValueError("OptionsDX has duplicate quote/expiry/strike rows")
    return frame


def parse_nq_1m_csv(text_or_buffer) -> pd.DataFrame:
    frame = pd.read_csv(io.StringIO(text_or_buffer) if isinstance(text_or_buffer, str) else text_or_buffer)
    expected = ["timestamp ET", "open", "high", "low", "close", "volume", "Vwap_RTH", "Vwap_ETH"]
    if frame.columns.tolist() != expected:
        raise ValueError("NQ 1-minute schema changed")
    timestamp = pd.to_datetime(frame.pop("timestamp ET"), format="%m/%d/%Y %H:%M", errors="raise")
    try:
        timestamp = timestamp.dt.tz_localize(
            "America/New_York", ambiguous="raise", nonexistent="raise"
        )
    except Exception as error:
        raise ValueError("NQ timestamps cannot be localized unambiguously") from error
    if timestamp.duplicated().any() or not timestamp.is_monotonic_increasing:
        raise ValueError("NQ timestamps must be unique and sorted")
    _ohlc(frame, "NQ")
    _numeric(frame, ["volume"], "NQ")
    if (frame["volume"] < 0).any():
        raise ValueError("NQ has negative volume")
    for column in ("Vwap_RTH", "Vwap_ETH"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if (frame[column].dropna() <= 0).any():
            raise ValueError(f"NQ has invalid {column}")
    frame.index = pd.DatetimeIndex(timestamp, name="timestamp")
    return frame


def parse_zenodo_tsla_csv(text_or_buffer, columns: list[str] | None = None) -> pd.DataFrame:
    columns = columns or load_protocol()["schemas"]["zenodo_tsla"]["columns"]
    frame = pd.read_csv(io.StringIO(text_or_buffer) if isinstance(text_or_buffer, str) else text_or_buffer)
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.columns.tolist() != columns:
        raise ValueError("Zenodo TSLA schema changed")
    frame["date"] = _strict_dates(frame["date"], "Zenodo TSLA date")
    frame["exdate"] = _strict_dates(frame["exdate"], "Zenodo TSLA expiry")
    if (frame["exdate"] < frame["date"]).any():
        raise ValueError("Zenodo TSLA expiry precedes quote")
    numeric = [column for column in columns if column not in {"date", "exdate"}]
    _numeric(frame, numeric, "Zenodo TSLA")
    if not frame["cp_flag"].isin([0, 1]).all():
        raise ValueError("Zenodo TSLA has invalid cp_flag")
    if (frame[["strike_price", "current_price"]] <= 0).any().any():
        raise ValueError("Zenodo TSLA has nonpositive strike/spot")
    if (frame["best_bid"] < 0).any() or (
        frame["best_offer"] < frame["best_bid"]
    ).any():
        raise ValueError("Zenodo TSLA has invalid bid/offer")
    if (frame[["volume", "open_interest"]] < 0).any().any():
        raise ValueError("Zenodo TSLA has invalid volume/open interest")
    key = ["date", "exdate", "cp_flag", "strike_price"]
    if frame.duplicated(key).any():
        raise ValueError("Zenodo TSLA has duplicate option observations")
    return frame


def _cftc_column(column: object) -> str:
    return str(column).strip().lower().replace("-", "_").replace(" ", "_")


def parse_cftc_tff_csv(text_or_buffer, allowed_contracts: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(
        io.StringIO(text_or_buffer) if isinstance(text_or_buffer, str) else text_or_buffer,
        dtype=str,
    )
    frame.columns = [_cftc_column(column) for column in frame.columns]
    aliases = {
        "report_date_as_yyyy_mm_dd": "report_date",
        "cftc_contract_market_code": "contract_code",
        "market_and_exchange_names": "market_name",
        "open_interest_all": "open_interest_all",
    }
    missing = sorted(set(aliases) - set(frame.columns))
    if missing:
        raise ValueError(f"CFTC schema missing {missing}")
    frame = frame.rename(columns=aliases)
    frame["report_date"] = _strict_dates(frame["report_date"], "CFTC report date")
    frame["contract_code"] = frame["contract_code"].astype(str).str.strip()
    unknown = set(frame["contract_code"]) - set(allowed_contracts)
    if unknown:
        raise ValueError(f"CFTC contains unexpected contract codes {sorted(unknown)}")
    _numeric(frame, ["open_interest_all"], "CFTC")
    if (frame["open_interest_all"] < 0).any():
        raise ValueError("CFTC has negative open interest")
    if frame.duplicated(["contract_code", "report_date"]).any():
        raise ValueError("CFTC has duplicate contract/report-date rows")
    if not frame.sort_values(["contract_code", "report_date"])[
        ["contract_code", "report_date"]
    ].reset_index(drop=True).equals(frame[["contract_code", "report_date"]].reset_index(drop=True)):
        raise ValueError("CFTC records are not sorted by contract/date")
    return frame


def apply_cftc_availability(
    frame: pd.DataFrame, date_column: str = "report_date", *, days: int = 7
) -> pd.DataFrame:
    if int(days) != 7:
        raise ValueError("CFTC research availability must use the frozen seven-day lag")
    out = frame.copy()
    out[date_column] = _strict_dates(out[date_column], "CFTC report date")
    out["available_date"] = out[date_column] + pd.Timedelta(days=7)
    return out


def apply_cboe_availability(
    frame: pd.DataFrame,
    date_column: str,
    declared_sessions: pd.DatetimeIndex,
) -> pd.DataFrame:
    out = frame.copy()
    out[date_column] = _strict_dates(out[date_column], "Cboe date")
    sessions = pd.DatetimeIndex(pd.to_datetime(declared_sessions))
    if sessions.tz is not None:
        sessions = sessions.tz_localize(None)
    sessions = sessions.normalize()
    if sessions.has_duplicates or not sessions.is_monotonic_increasing:
        raise ValueError("declared sessions must be unique and sorted")
    positions = sessions.searchsorted(pd.DatetimeIndex(out[date_column]), side="right")
    if (positions >= len(sessions)).any():
        raise ValueError("Cboe observation has no following session in declared calendar")
    out["available_date"] = sessions.take(positions)
    return out
