"""Authenticate fixed market snapshots before bounded numerical decoding."""

from __future__ import annotations

import hashlib
import re
from io import BytesIO
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

SOURCE_COLUMNS = {
    "data/raw/daily_ohlc.parquet": ("open", "high", "low", "close", "adj close", "volume"),
    "data/raw/cross_asset_daily.parquet": ("hyg", "tlt", "gld", "uso", "uup"),
    "data/raw/vxn_daily.parquet": ("close",),
    "data/raw/short_dated_iv.parquet": ("vix", "vix9d"),
}
SOURCE_CEILING = pd.Timestamp("2025-10-20")


def _calendar(values):
    if not pd.api.types.is_datetime64_any_dtype(values.dtype):
        raise ValueError("Native source date column required")
    dates = pd.DatetimeIndex(values, name="date")
    if (
        dates.tz is not None
        or dates.hasnans
        or dates.has_duplicates
        or not dates.is_monotonic_increasing
        or not dates.equals(dates.normalize())
    ):
        raise ValueError("Unique ordered naive midnight source calendar required")
    return dates


def read_market_sources(root, pins):
    """Load only declared columns at/before the fixed ceiling from verified bytes.

    Call only after prospective experiment registration. The caller supplies
    frozen raw-file SHA256 values. No numerical source is decoded until all
    file hashes and all complete date columns have passed preflight. Full date
    metadata can extend beyond the ceiling; numerical output cannot. Numerical
    validity is separately enforced by the market-feature builder.
    """
    if (
        type(pins) is not dict
        or set(pins) != set(SOURCE_COLUMNS)
        or any(
            type(v) is not str or re.fullmatch(r"[0-9a-f]{64}", v) is None
            for v in pins.values()
        )
    ):
        raise ValueError("Exact source inventory and literal SHA256 hashes required")
    snapshots = {path: (Path(root) / path).read_bytes() for path in SOURCE_COLUMNS}
    for path, payload in snapshots.items():
        if hashlib.sha256(payload).hexdigest() != pins[path]:
            raise ValueError(f"Source hash mismatch: {path}")
    calendars = {}
    for path, payload in snapshots.items():
        frame = pq.read_table(BytesIO(payload), columns=["date"]).to_pandas(
            ignore_metadata=True
        )
        calendars[path] = _calendar(frame["date"])
    frames = {}
    for path, payload in snapshots.items():
        columns = list(SOURCE_COLUMNS[path])
        frame = pq.read_table(
            BytesIO(payload),
            columns=["date", *columns],
            filters=[("date", "<=", SOURCE_CEILING)],
        ).to_pandas(ignore_metadata=True)
        dates = _calendar(frame["date"])
        expected = calendars[path][calendars[path] <= SOURCE_CEILING]
        if not dates.equals(expected):
            raise ValueError("Bounded source rows disagree with date-only preflight")
        frame = frame.drop(columns="date")
        frame.index = dates
        frames[path] = frame.loc[:, columns]
    daily, cross, vxn, short = (frames[path] for path in SOURCE_COLUMNS)
    iv = vxn.rename(columns={"close": "vxn"}).join(short, how="outer").sort_index()
    return {"daily": daily, "cross": cross, "iv": iv}
