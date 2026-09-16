"""Read authenticated, independently admitted commodity source snapshots."""

import hashlib
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd

from src.verify_commodity_implied_source import verify_history

SYMBOLS = ("OVX", "GVZ")
_START = date(2009, 1, 2)
_END = date(2025, 10, 20)


def _path(root, name):
    if (
        type(name) is not str
        or not name
        or Path(name).is_absolute()
        or ".." in Path(name).parts
    ):
        raise ValueError("Safe nonempty relative source path required")
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Source path must remain inside the supplied root")
    return path


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate parsed JSON key")
        result[key] = value
    return result


def _constant(token):
    raise ValueError("Nonfinite JSON constant is not an admitted observation")


def _date_preflight(payload):
    # JSON number tokens stay text: inspect every bounded record date in both
    # snapshots before the caller permits either series' numeric decoding.
    obj = json.loads(
        payload,
        parse_float=str,
        parse_int=str,
        parse_constant=_constant,
        object_pairs_hook=_object,
    )
    if type(obj) is not dict or type(obj.get("records")) is not list:
        raise ValueError("Admitted parsed source records required")
    previous = None
    for row in obj["records"]:
        if (
            type(row) is not dict
            or type(row.get("date")) is not str
            or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", row["date"]) is None
        ):
            raise ValueError("Literal bounded source record date required")
        observed = date.fromisoformat(row["date"])
        if not _START <= observed <= _END or (previous is not None and observed <= previous):
            raise ValueError("Ordered unique dates inside the admitted source window required")
        previous = observed


def _decode_parsed(payload):
    return json.loads(payload, parse_constant=_constant, object_pairs_hook=_object)


def read_commodity_sources(root, histories):
    """Return the two exact bounded histories without filling their outer calendar.

    Call only after prospective forecast registration. The caller must bind this
    inventory to the pinned completed admission certificate and all its evidence.
    Every raw/parsed byte hash and both parsed date envelopes pass before numeric
    JSON decoding. The independent source checker reconstructs both histories
    from the original bytes again before any data frame is returned.
    """
    if type(histories) is not dict or set(histories) != set(SYMBOLS):
        raise ValueError("Exact OVX/GVZ history inventory required")
    root = Path(root).resolve()
    locations = {}
    for symbol in SYMBOLS:
        entry = histories[symbol]
        if type(entry) is not dict or set(entry) != {
            "raw",
            "raw_sha256",
            "parsed",
            "parsed_sha256",
        }:
            raise ValueError("Exact raw and parsed snapshot pin fields required")
        for kind in ("raw", "parsed"):
            if (
                type(entry[kind + "_sha256"]) is not str
                or re.fullmatch(r"[0-9a-f]{64}", entry[kind + "_sha256"]) is None
            ):
                raise ValueError("Literal source SHA-256 required")
            locations[symbol, kind] = _path(root, entry[kind])
    if len(set(locations.values())) != 4:
        raise ValueError("Distinct original and parsed source snapshots required")
    snapshots = {key: path.read_bytes() for key, path in locations.items()}
    for (symbol, kind), payload in snapshots.items():
        if hashlib.sha256(payload).hexdigest() != histories[symbol][kind + "_sha256"]:
            raise ValueError("Source snapshot hash mismatch")
    for symbol in SYMBOLS:
        _date_preflight(snapshots[symbol, "parsed"])
    verified = {}
    for symbol in SYMBOLS:
        parsed = _decode_parsed(snapshots[symbol, "parsed"])
        verify_history(
            snapshots[symbol, "raw"], symbol, histories[symbol]["raw_sha256"], parsed
        )
        verified[symbol] = parsed["records"]
    series = {}
    for symbol, rows in verified.items():
        index = pd.DatetimeIndex([row["date"] for row in rows], name="date")
        series[symbol] = pd.Series([row["value"] for row in rows], index=index, dtype=float)
    return pd.DataFrame(series, columns=SYMBOLS).sort_index()
