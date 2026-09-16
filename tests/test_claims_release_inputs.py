"""Synthetic snapshot and date-fence contracts, written before the reader."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.claims_release_inputs import read_market_sources

SOURCES = {
    "data/raw/daily_ohlc.parquet": ["open", "high", "low", "close", "adj close", "volume"],
    "data/raw/cross_asset_daily.parquet": ["hyg", "tlt", "gld", "uso", "uup"],
    "data/raw/vxn_daily.parquet": ["close"],
    "data/raw/short_dated_iv.parquet": ["vix", "vix9d"],
}


class ClaimsReleaseInputsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pins = {}
        dates = pd.to_datetime(["2025-10-17", "2025-10-20", "2025-11-03"])
        self.frames = {}
        for path, columns in SOURCES.items():
            frame = pd.DataFrame({c: [10.0, 11.0, -999.0] for c in columns})
            frame.insert(0, "date", dates)
            frame["unused_protected_sentinel"] = [1.0, 1.0, np.inf]
            self.frames[path] = frame
            self.write(path, frame)

    def write(self, path, frame):
        file = self.root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), file)
        self.pins[path] = hashlib.sha256(file.read_bytes()).hexdigest()

    def test_only_declared_columns_and_bounded_dates_are_decoded(self):
        original = pq.read_table
        requests = []

        def traced(*args, **kwargs):
            requests.append(kwargs)
            return original(*args, **kwargs)

        with patch("src.claims_release_inputs.pq.read_table", side_effect=traced):
            result = read_market_sources(self.root, self.pins)
        self.assertEqual(set(result), {"daily", "cross", "iv"})
        for frame in result.values():
            self.assertEqual(
                list(frame.index), list(pd.to_datetime(["2025-10-17", "2025-10-20"]))
            )
            self.assertEqual(frame.index.name, "date")
            self.assertFalse((frame.to_numpy() == -999.0).any())
        self.assertEqual(list(result["iv"]), ["vxn", "vix", "vix9d"])
        numeric = [r for r in requests if r["columns"] != ["date"]]
        self.assertEqual(len(numeric), 4)
        for request in numeric:
            self.assertNotIn("unused_protected_sentinel", request["columns"])
            self.assertEqual(request["filters"], [("date", "<=", pd.Timestamp("2025-10-20"))])

    def test_every_hash_verified_before_any_parquet_decoding(self):
        path = next(reversed(SOURCES))
        self.pins[path] = "0" * 64
        with patch("src.claims_release_inputs.pq.read_table") as decoder:
            with self.assertRaisesRegex(ValueError, "hash"):
                read_market_sources(self.root, self.pins)
            decoder.assert_not_called()

    def test_exact_source_inventory_and_literal_hashes_required(self):
        for pins in (
            {},
            {**self.pins, "extra.parquet": "0" * 64},
            {**self.pins, next(iter(SOURCES)): True},
        ):
            with (
                self.subTest(pins=pins),
                patch("src.claims_release_inputs.pq.read_table") as decoder,
            ):
                with self.assertRaises(ValueError):
                    read_market_sources(self.root, pins)
                decoder.assert_not_called()

    def test_verified_bytes_are_the_bytes_decoded(self):
        original = pq.read_table
        calls = 0

        def mutate_disk_after_snapshot(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                for path in SOURCES:
                    (self.root / path).write_bytes(b"changed after hash")
            return original(*args, **kwargs)

        with patch(
            "src.claims_release_inputs.pq.read_table", side_effect=mutate_disk_after_snapshot
        ):
            result = read_market_sources(self.root, self.pins)
        self.assertEqual(result["daily"].iloc[0]["close"], 10.0)

    def test_date_only_preflight_rejects_bad_calendar_before_numeric_decode(self):
        path = next(iter(SOURCES))
        dates_cases = [
            pd.to_datetime(["2025-10-17", "2025-10-17", "2025-11-03"]),
            pd.to_datetime(["2025-10-20", "2025-10-17", "2025-11-03"]),
            pd.to_datetime(["2025-10-17 12:00", "2025-10-20 12:00", "2025-11-03 12:00"]),
            pd.to_datetime(["2025-10-17", None, "2025-11-03"]),
            pd.to_datetime(["2025-10-17", "2025-10-20", "2025-11-03"], utc=True),
        ]
        for dates in dates_cases:
            frame = self.frames[path].copy()
            frame["date"] = dates
            self.write(path, frame)
            original = pq.read_table
            requests = []

            def traced(*args, _requests=requests, _original=original, **kwargs):
                _requests.append(kwargs["columns"])
                return _original(*args, **kwargs)

            with (
                self.subTest(dates=str(dates)),
                patch("src.claims_release_inputs.pq.read_table", side_effect=traced),
                self.assertRaises(ValueError),
            ):
                read_market_sources(self.root, self.pins)
            self.assertTrue(all(columns == ["date"] for columns in requests))

    def test_all_source_calendars_preflight_before_any_numeric_decode(self):
        path = next(reversed(SOURCES))
        frame = self.frames[path].copy()
        frame.loc[2, "date"] = frame.loc[1, "date"]
        self.write(path, frame)
        original = pq.read_table
        requests = []

        def traced(*args, **kwargs):
            requests.append(kwargs["columns"])
            return original(*args, **kwargs)

        with (
            patch("src.claims_release_inputs.pq.read_table", side_effect=traced),
            self.assertRaises(ValueError),
        ):
            read_market_sources(self.root, self.pins)
        self.assertTrue(all(columns == ["date"] for columns in requests))

    def test_missing_dates_and_values_are_preserved_without_fill(self):
        path = "data/raw/short_dated_iv.parquet"
        frame = self.frames[path].iloc[[1, 2]].copy()
        frame.loc[1, "vix9d"] = np.nan
        self.write(path, frame)
        result = read_market_sources(self.root, self.pins)
        self.assertTrue(pd.isna(result["iv"].loc["2025-10-17", "vix"]))
        self.assertTrue(pd.isna(result["iv"].loc["2025-10-20", "vix9d"]))
        self.assertEqual(len(result["daily"]), 2)


if __name__ == "__main__":
    unittest.main()
