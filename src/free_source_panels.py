"""Build compact, timing-safe panels and audits for auxiliary free sources.

Raw exchange/uploader files stay in the ignored raw tree.  This module writes
only a local Cboe panel, a compact public-domain CFTC panel, and non-
reconstructive source audits.  It never imputes a source-validation failure.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from .free_data_sources import load_protocol, parse_hf_spx_csv, sha256_file

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "free_sources" / "raw"
PROCESSED = ROOT / "data" / "free_sources" / "processed"
LOCKED_HISTORY = ROOT / "data" / "history_extension" / "qqq_price_only_daily.parquet"
CBOE_PANEL = PROCESSED / "cboe_daily_local.parquet"
CFTC_209742_PANEL = PROCESSED / "cftc_nq_209742.parquet"
AUXILIARY_AUDIT = PROCESSED / "auxiliary_source_audit.json"
_PROTOCOL = load_protocol()
_HF_SPX_SOURCE = _PROTOCOL["sources"]["hf_spx"]
_ZENODO_SOURCE = _PROTOCOL["sources"]["zenodo_tsla_options"]
ZENODO_MD5 = str(_ZENODO_SOURCE["file_md5"])
ZENODO_COLUMNS = list(_PROTOCOL["schemas"]["zenodo_tsla"]["columns"])
ZENODO_KEY_COLUMNS = ["date", "exdate", "cp_flag", "strike_price"]

CBOE_SOURCES = {
    "VIX": ("VIX_History.csv", "ohlc"),
    "VXN": ("VXN_History.csv", "ohlc"),
    "VIX1D": ("VIX1D_History.csv", "ohlc"),
    "VIX9D": ("VIX9D_History.csv", "ohlc"),
    "SKEW": ("SKEW_History.csv", "close_only"),
    "VVIX": ("VVIX_History.csv", "close_only"),
}


def parse_cboe_close_with_audit(text_or_buffer, schema: str) -> tuple[pd.DataFrame, dict]:
    """Validate the requested close while quarantining inconsistent OHLC fields."""
    if hasattr(text_or_buffer, "read"):
        text = text_or_buffer.read()
    else:
        text = str(text_or_buffer)
    raw = pd.read_csv(io.StringIO(text))
    raw.columns = [str(column).strip().lower() for column in raw.columns]
    if schema == "ohlc":
        expected = ["date", "open", "high", "low", "close"]
        if raw.columns.tolist() != expected:
            raise ValueError("Cboe OHLC schema changed")
        value_columns = ["open", "high", "low", "close"]
    elif schema == "close_only":
        if len(raw.columns) != 2 or raw.columns[0] != "date":
            raise ValueError("Cboe close-only schema changed")
        raw = raw.rename(columns={raw.columns[1]: "close"})
        value_columns = ["close"]
    else:
        raise ValueError(f"unknown Cboe schema {schema}")
    raw["date"] = pd.to_datetime(raw["date"], errors="raise").dt.normalize()
    if raw["date"].isna().any():
        raise ValueError("Cboe date contains missing values")
    for column in value_columns:
        raw[column] = pd.to_numeric(raw[column], errors="raise")
    if raw["date"].duplicated().any() or not raw["date"].is_monotonic_increasing:
        raise ValueError("Cboe dates must be unique and sorted")
    if raw.empty:
        raise ValueError("Cboe input is empty")
    if raw["close"].isna().any() or not np.isfinite(raw["close"]).all() or (
        raw["close"] <= 0
    ).any():
        raise ValueError("Cboe close is nonpositive")
    if schema == "ohlc":
        auxiliary = raw[["open", "high", "low"]]
        bad = (
            auxiliary.isna().any(axis=1)
            | ~np.isfinite(auxiliary).all(axis=1)
            | (auxiliary <= 0).any(axis=1)
            | (raw["high"] < raw[["open", "close", "low"]].max(axis=1))
            | (raw["low"] > raw[["open", "close", "high"]].min(axis=1))
        )
    else:
        bad = pd.Series(False, index=raw.index)
    close = raw.set_index("date").loc[:, ["close"]]
    invalid_ohlc_rows = int(bad.sum())
    return close, {
        "status": (
            "close_validated_ohlc_quarantined"
            if invalid_ohlc_rows
            else "strict_schema_validated"
        ),
        "invalid_ohlc_rows": invalid_ohlc_rows,
    }


def _md5(path: Path) -> str:
    try:
        digest = hashlib.md5(usedforsecurity=False)
    except TypeError:  # Python/OpenSSL builds that do not expose this keyword.
        digest = hashlib.md5()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_cboe_close_panel(
    observations: Mapping[str, pd.DataFrame],
    sessions: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Map each official close to the next declared session, never same-day."""
    sessions = pd.DatetimeIndex(pd.to_datetime(sessions)).tz_localize(None).normalize()
    if sessions.empty or sessions.has_duplicates or not sessions.is_monotonic_increasing:
        raise ValueError("locked sessions must be nonempty, unique, and increasing")
    if not observations:
        raise ValueError("Cboe observations must be nonempty")
    rows: list[pd.DataFrame] = []
    for symbol, source in sorted(observations.items()):
        frame = source.copy()
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None).normalize()
        frame.index.name = "observation_date"
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError(f"{symbol} observations must be unique and sorted")
        eligible = frame.loc[(frame.index >= sessions[0]) & (frame.index < sessions[-1])]
        eligible = eligible.loc[eligible.index.isin(sessions)]
        positions = sessions.searchsorted(eligible.index, side="right")
        part = eligible.loc[:, ["close"]].reset_index()
        part["available_date"] = sessions.take(positions)
        part["series"] = symbol
        rows.append(part)
    panel = pd.concat(rows, ignore_index=True).loc[
        :, ["series", "observation_date", "available_date", "close"]
    ]
    panel = panel.sort_values(["available_date", "series"], kind="mergesort").reset_index(drop=True)
    if panel.duplicated(["series", "observation_date"]).any():
        raise ValueError("duplicate Cboe observation")
    if not (panel["observation_date"] < panel["available_date"]).all():
        raise ValueError("Cboe close was made available without a full-session lag")
    return panel


def audit_hf_spx(
    path: Path,
    *,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
    declared_rows: int | None = None,
) -> dict:
    raw = pd.read_csv(path)
    normalized = {str(column).strip().lower().replace(" ", "_"): column for column in raw.columns}
    zero_open = 0
    if "open" in normalized:
        zero_open = int((pd.to_numeric(raw[normalized["open"]], errors="coerce") <= 0).sum())
    observed_sha256 = sha256_file(path)
    record: dict = {
        "bytes": path.stat().st_size,
        "sha256": observed_sha256,
        "rows": int(len(raw)),
        "zero_open_rows": zero_open,
    }
    if declared_rows is not None:
        record.update({
            "declared_rows": int(declared_rows),
            "declared_rows_match": len(raw) == int(declared_rows),
        })
    identity_failures = []
    if expected_bytes is not None:
        bytes_match = path.stat().st_size == int(expected_bytes)
        record["pinned_bytes_match"] = bytes_match
        if not bytes_match:
            identity_failures.append("byte size differs from pinned source")
    if expected_sha256 is not None:
        sha256_match = observed_sha256 == expected_sha256
        record["pinned_sha256_match"] = sha256_match
        if not sha256_match:
            identity_failures.append("SHA-256 differs from pinned source")
    if identity_failures:
        record.update({
            "status": "quarantined_source_identity_failure",
            "failure": "; ".join(identity_failures),
        })
        return record
    try:
        with path.open() as source:
            parsed = parse_hf_spx_csv(source)
    except ValueError as error:
        record.update({
            "status": "quarantined_strict_ohlc_failure",
            "failure": str(error),
        })
        return record
    record.update({
        "status": "validated_private_research_only",
        "first_date": str(parsed.index.min().date()),
        "last_date": str(parsed.index.max().date()),
    })
    return record


def _zenodo_key_frame(chunk: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(chunk["date"], errors="raise").dt.normalize()
    exdates = pd.to_datetime(chunk["exdate"], errors="raise").dt.normalize()
    cp_flag = pd.to_numeric(chunk["cp_flag"], errors="raise")
    strike = pd.to_numeric(chunk["strike_price"], errors="raise")
    return pd.DataFrame({
        "date": dates,
        "exdate": exdates,
        "cp_flag": cp_flag,
        "strike_price": strike,
    })


def _exact_duplicate_for_hash_candidates(
    path: Path,
    candidate_hashes: np.ndarray,
    chunksize: int,
) -> bool:
    """Resolve rare 64-bit hash matches against exact option keys."""
    candidates = {int(value) for value in candidate_hashes}
    seen: dict[int, set[tuple[int, int, int, float]]] = {
        value: set() for value in candidates
    }
    with pd.read_csv(
        path, usecols=ZENODO_KEY_COLUMNS, chunksize=chunksize
    ) as chunks:
        for chunk in chunks:
            keys = _zenodo_key_frame(chunk)
            hashes = pd.util.hash_pandas_object(keys, index=False).to_numpy(np.uint64)
            for offset in np.flatnonzero(np.isin(hashes, candidate_hashes)):
                row = keys.iloc[int(offset)]
                value = int(hashes[int(offset)])
                key = (
                    int(pd.Timestamp(row["date"]).value),
                    int(pd.Timestamp(row["exdate"]).value),
                    int(row["cp_flag"]),
                    float(row["strike_price"]),
                )
                if key in seen[value]:
                    return True
                seen[value].add(key)
    return False


def audit_zenodo_tsla(
    path: Path,
    *,
    expected_md5: str | None = ZENODO_MD5,
    chunksize: int = 250_000,
) -> dict:
    """Strictly validate the large TSLA CSV in bounded-memory chunks."""
    observed_md5 = _md5(path)
    if expected_md5 is not None and observed_md5 != expected_md5:
        raise ValueError("Zenodo TSLA MD5 differs from the pinned record")
    count = 0
    first: pd.Timestamp | None = None
    last: pd.Timestamp | None = None
    quote_dates: set[pd.Timestamp] = set()
    expiries: set[pd.Timestamp] = set()
    key_hashes: list[np.ndarray] = []
    numeric = [column for column in ZENODO_COLUMNS if column not in {"date", "exdate"}]
    with pd.read_csv(path, chunksize=chunksize) as chunks:
        for chunk in chunks:
            chunk.columns = [str(column).strip() for column in chunk.columns]
            if chunk.columns.tolist() != ZENODO_COLUMNS:
                raise ValueError("Zenodo TSLA schema changed")
            dates = pd.to_datetime(chunk["date"], errors="raise").dt.normalize()
            exdates = pd.to_datetime(chunk["exdate"], errors="raise").dt.normalize()
            if (exdates < dates).any():
                raise ValueError("Zenodo TSLA expiry precedes quote")
            values = chunk[numeric].apply(pd.to_numeric, errors="raise")
            if values.isna().any().any():
                raise ValueError("Zenodo TSLA has missing numeric fields")
            if not values["cp_flag"].isin([0, 1]).all():
                raise ValueError("Zenodo TSLA has invalid cp_flag")
            if (values[["strike_price", "current_price"]] <= 0).any().any():
                raise ValueError("Zenodo TSLA has nonpositive strike/spot")
            if (values["best_bid"] < 0).any() or (
                values["best_offer"] < values["best_bid"]
            ).any():
                raise ValueError("Zenodo TSLA has invalid bid/offer")
            if (values[["volume", "open_interest"]] < 0).any().any():
                raise ValueError("Zenodo TSLA has invalid volume/open interest")
            keys = pd.DataFrame({
                "date": dates, "exdate": exdates,
                "cp_flag": values["cp_flag"], "strike_price": values["strike_price"],
            })
            key_hashes.append(
                pd.util.hash_pandas_object(keys, index=False).to_numpy(np.uint64)
            )
            count += len(chunk)
            chunk_first, chunk_last = dates.min(), dates.max()
            first = chunk_first if first is None else min(first, chunk_first)
            last = chunk_last if last is None else max(last, chunk_last)
            quote_dates.update(dates.unique())
            expiries.update(exdates.unique())
    if not count:
        raise ValueError("Zenodo TSLA input is empty")
    hashes = np.concatenate(key_hashes)
    repeated_hash = pd.Index(hashes).duplicated(keep=False)
    if repeated_hash.any():
        candidates = np.unique(hashes[repeated_hash])
        if _exact_duplicate_for_hash_candidates(path, candidates, chunksize):
            raise ValueError("Zenodo TSLA has duplicate option observations")
    return {
        "status": "validated_private_research_only",
        "bytes": path.stat().st_size,
        "md5": observed_md5,
        "md5_pin_verified": expected_md5 is not None,
        "sha256": sha256_file(path),
        "rows": int(count),
        "first_date": str(first.date()),
        "last_date": str(last.date()),
        "quote_dates": len(quote_dates),
        "expiries": len(expiries),
        "has_bid_offer": True,
        "has_open_interest": True,
        "schema_columns": ZENODO_COLUMNS,
    }


def build_cftc_209742(path: Path) -> tuple[pd.DataFrame, dict]:
    usecols = [
        "report_date_as_yyyy_mm_dd", "cftc_contract_market_code",
        "open_interest_all", "lev_money_positions_long", "lev_money_positions_short",
    ]
    frame = pd.read_csv(path, usecols=usecols, dtype={"cftc_contract_market_code": str})
    frame = frame.rename(columns={
        "report_date_as_yyyy_mm_dd": "report_date",
        "cftc_contract_market_code": "contract_code",
        "open_interest_all": "open_interest",
        "lev_money_positions_long": "lev_long",
        "lev_money_positions_short": "lev_short",
    })
    frame["report_date"] = pd.to_datetime(frame["report_date"], errors="raise").dt.normalize()
    frame["contract_code"] = frame["contract_code"].str.strip()
    if not frame["contract_code"].eq("209742").all():
        raise ValueError("pre-consolidation CFTC input contains another contract")
    for column in ("open_interest", "lev_long", "lev_short"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame.empty:
        raise ValueError("CFTC 209742 input is empty")
    position_columns = ["open_interest", "lev_long", "lev_short"]
    if frame[position_columns].isna().any().any() or not np.isfinite(
        frame[position_columns]
    ).all().all():
        raise ValueError("CFTC position fields contain missing or nonfinite values")
    if (frame["open_interest"] <= 0).any():
        raise ValueError("CFTC open interest must be positive")
    if (frame[["lev_long", "lev_short"]] < 0).any().any():
        raise ValueError("CFTC leveraged-money positions must be nonnegative")
    if (frame[position_columns] % 1 != 0).any().any():
        raise ValueError("CFTC position fields must be whole contracts")
    frame[position_columns] = frame[position_columns].astype("int64")
    frame = frame.sort_values("report_date", kind="mergesort").reset_index(drop=True)
    if frame["report_date"].duplicated().any():
        raise ValueError("duplicate CFTC report date")
    frame["available_date_generic"] = frame["report_date"] + pd.Timedelta(days=7)
    audit = {
        "status": "validated_official_public_domain",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": len(frame),
        "first_date": str(frame["report_date"].min().date()),
        "last_date": str(frame["report_date"].max().date()),
        "availability_note": "generic acquisition panel uses report_date+7d; predictive CFTC protocol independently uses +10d and blackout exclusions",
    }
    return frame, audit


def run() -> dict:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    sessions = pd.DatetimeIndex(pd.read_parquet(LOCKED_HISTORY).index)
    observations: dict[str, pd.DataFrame] = {}
    cboe_audit: dict[str, dict] = {}
    for symbol, (filename, schema) in CBOE_SOURCES.items():
        path = RAW / "cboe" / filename
        with path.open() as source:
            parsed, validation = parse_cboe_close_with_audit(source, schema)
        observations[symbol] = parsed
        in_range = parsed.index.to_series().between(sessions[0], sessions[-1], inclusive="left")
        withheld = int((in_range & ~parsed.index.to_series().isin(sessions)).sum())
        cboe_audit[symbol] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "rows": len(parsed),
            "first_date": str(parsed.index.min().date()),
            "last_date": str(parsed.index.max().date()),
            "withheld_outside_locked_calendar": withheld,
            **validation,
        }
    cboe_panel = build_cboe_close_panel(observations, sessions)
    cboe_panel.to_parquet(CBOE_PANEL, index=False)
    cftc_panel, cftc_audit = build_cftc_209742(RAW / "cftc" / "nq_tff_209742.csv")
    cftc_panel.to_parquet(CFTC_209742_PANEL, index=False)
    spx_audit = audit_hf_spx(
        RAW / "huggingface" / "misikoff_spx" / "^SPX.csv",
        expected_bytes=int(_HF_SPX_SOURCE["file_bytes"]),
        expected_sha256=str(_HF_SPX_SOURCE["file_sha256"]),
        declared_rows=int(_HF_SPX_SOURCE["rows"]),
    )
    zenodo_path = RAW / "zenodo" / "15496947" / "1_sorted_tsla.csv"
    try:
        zenodo_audit = audit_zenodo_tsla(zenodo_path)
    except ValueError as error:
        zenodo_audit = {
            "status": "quarantined_validation_failure",
            "bytes": zenodo_path.stat().st_size,
            "md5": _md5(zenodo_path),
            "sha256": sha256_file(zenodo_path),
            "failure": str(error),
        }
    audit = {
        "cboe": {
            "status": "validated_official_local_only",
            "availability": "next locked QQQ session",
            "panel_rows": len(cboe_panel),
            "panel_bytes": CBOE_PANEL.stat().st_size,
            "panel_sha256": sha256_file(CBOE_PANEL),
            "panel_commit_policy": "local_only_ignored",
            "sources": cboe_audit,
        },
        "cftc_209742": {
            **cftc_audit,
            "panel_bytes": CFTC_209742_PANEL.stat().st_size,
            "panel_sha256": sha256_file(CFTC_209742_PANEL),
            "panel_commit_policy": "processed_may_commit",
        },
        "hf_spx": spx_audit,
        "zenodo_tsla": zenodo_audit,
        "source_metadata_corrections": {
            "hf_spx": _HF_SPX_SOURCE.get("metadata_corrections", []),
            "zenodo_tsla": _ZENODO_SOURCE.get("metadata_corrections", []),
        },
    }
    AUXILIARY_AUDIT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run"])
    parser.parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
