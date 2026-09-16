"""Six original joint-risk snapshots, authenticated before bounded decoding."""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.claims_release_pipeline import _dates, _iso
from src.index_hinge import OHLC, _daily_contract, _positive_market

SOURCES = {
    "qqq": "data/raw/daily_ohlc.parquet",
    "spx": "data/research_paths/spx_daily.parquet",
    "vxn": "data/free_sources/raw/cboe/VXN_History.csv",
    "vix": "data/free_sources/raw/cboe/VIX_History.csv",
    "vix9d": "data/free_sources/raw/cboe/VIX9D_History.csv",
    "vvix": "data/free_sources/raw/cboe/VVIX_History.csv",
}
CEILING = pd.Timestamp("2025-10-20")


def _metadata(payload, name):
    if name in ("qqq", "spx"):
        raw = None
        column = pq.read_table(io.BytesIO(payload), columns=["date"]).column("date")
        if not (pa.types.is_timestamp(column.type) or pa.types.is_date(column.type)):
            raise ValueError("Native parquet date metadata required")
        dates = column.to_pandas()
    else:
        field = "VVIX" if name == "vvix" else "CLOSE"
        raw = pd.read_csv(io.BytesIO(payload), usecols=["DATE", field], dtype=str)
        if not raw.DATE.str.fullmatch(r"[0-9]{2}/[0-9]{2}/[0-9]{4}").all():
            raise ValueError("Literal complete source date metadata required")
        dates = pd.to_datetime(raw.DATE, format="%m/%d/%Y", errors="raise")
    index = pd.DatetimeIndex(dates, name="date")
    if (
        not len(index)
        or index.hasnans
        or index.tz is not None
        or not index.is_unique
        or not index.is_monotonic_increasing
        or not index.equals(index.normalize())
    ):
        raise ValueError("Unique increasing native-midnight source dates required")
    return {"dates": index, "raw_strings": raw}


def _decode(payload, name, metadata, source_end):
    selected = metadata["dates"] <= source_end
    if name in ("qqq", "spx"):
        frame = pq.read_table(
            io.BytesIO(payload), columns=["date", *OHLC], filters=[("date", "<=", source_end)]
        ).to_pandas()
        if "date" in frame:
            frame = frame.set_index("date")
        frame.index = pd.DatetimeIndex(frame.index, name="date")
        if not frame.index.equals(metadata["dates"][selected]):
            raise ValueError("Bounded decoded dates differ from checked metadata")
        for column in OHLC:
            if frame[column].dtype.kind not in "iuf":
                raise ValueError("Real nonboolean source prices required")
        frame = frame.loc[:, list(OHLC)].astype(float)
        _daily_contract(frame)
        if frame.empty:
            raise ValueError("INSUFFICIENT_DATA: empty bounded OHLC source")
        return frame
    field = "VVIX" if name == "vvix" else "CLOSE"
    numeric = pd.to_numeric(metadata["raw_strings"].loc[selected, field], errors="raise")
    frame = pd.DataFrame({name: numeric.to_numpy(float)}, index=metadata["dates"][selected])
    _positive_market(frame, [name])
    if frame.empty:
        raise ValueError("INSUFFICIENT_DATA: empty bounded IV source")
    return frame


def read_sources(root, pins, *, source_end="2025-10-20"):
    """No caller should invoke this numerical reader before registration."""
    end = _iso(source_end)
    if end > CEILING:
        raise ValueError("Source ceiling cannot exceed October20,2025")
    if (
        type(pins) is not dict
        or set(pins) != set(SOURCES.values())
        or any(
            type(h) is not str or re.fullmatch(r"[0-9a-f]{64}", h) is None
            for h in pins.values()
        )
    ):
        raise ValueError("Exact six original source paths and SHA256 pins required")
    snapshots = {name: (Path(root) / path).read_bytes() for name, path in SOURCES.items()}
    for name, payload in snapshots.items():
        if hashlib.sha256(payload).hexdigest() != pins[SOURCES[name]]:
            raise ValueError("Source hash mismatch: " + SOURCES[name])
    metadata = {name: _metadata(payload, name) for name, payload in snapshots.items()}
    frames = {name: _decode(snapshots[name], name, metadata[name], end) for name in SOURCES}
    iv = pd.concat([frames[n] for n in ("vxn", "vix", "vix9d", "vvix")], axis=1).sort_index()
    _dates(iv.index, end)
    return (
        frames["qqq"],
        frames["spx"],
        iv,
        {
            "status": "VERIFIED_BOUNDED_SOURCE_BUFFERS",
            "source_end": source_end,
            "source_sha256": dict(pins),
            "source_metadata": {
                name: {
                    "raw_dates": len(metadata[name]["dates"]),
                    "bounded_dates": len(frames[name]),
                    "excluded_after_ceiling_dates": int(np.sum(metadata[name]["dates"] > end)),
                }
                for name in SOURCES
            },
            "numerical_columns": {
                "qqq": list(OHLC),
                "spx": list(OHLC),
                **{name: [name] for name in ("vxn", "vix", "vix9d", "vvix")},
            },
            "snapshot_semantics": "Every hash then every full date column precedes numerical decoding from the same checked bytes",
        },
    )
