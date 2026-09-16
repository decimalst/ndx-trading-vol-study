"""Generated receipt-bound source integration tests before forecast admission."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src import commodity_implied_inputs as inputs
from src.commodity_implied_source import parse_cboe_history


class CommodityImpliedInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.histories = {}
        for symbol, rows in {
            "OVX": "2008-01-01,UNREAD\n2009-01-02,20\n2009-01-05,.\n2026-01-01,PROTECTED\n",
            "GVZ": "2009-01-02,30\n2009-01-06,40\n2026-01-01,PROTECTED\n",
        }.items():
            payload = (f"DATE,{symbol}\n" + rows).encode()
            signature = hashlib.sha256(payload).hexdigest()
            parsed = parse_cboe_history(payload, symbol, signature)
            raw_path, parsed_path = self.root / f"{symbol}.raw", self.root / f"{symbol}.json"
            raw_path.write_bytes(payload)
            parsed_path.write_text(json.dumps(parsed, allow_nan=False))
            self.histories[symbol] = {
                "raw": raw_path.name,
                "raw_sha256": signature,
                "parsed": parsed_path.name,
                "parsed_sha256": hashlib.sha256(parsed_path.read_bytes()).hexdigest(),
            }

    def change_parsed(self, symbol, mutation):
        path = self.root / self.histories[symbol]["parsed"]
        parsed = json.loads(path.read_text())
        mutation(parsed)
        path.write_text(json.dumps(parsed))
        self.histories[symbol]["parsed_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    def test_outer_calendar_preserves_absent_dates_and_explicit_missing(self):
        frame = inputs.read_commodity_sources(self.root, self.histories)
        self.assertEqual(list(frame.columns), ["OVX", "GVZ"])
        self.assertTrue(
            frame.index.equals(pd.to_datetime(["2009-01-02", "2009-01-05", "2009-01-06"]))
        )
        self.assertEqual(frame.iloc[0].tolist(), [20.0, 30.0])
        self.assertTrue(pd.isna(frame.loc["2009-01-05"]).all())
        self.assertTrue(pd.isna(frame.loc["2009-01-06", "OVX"]))
        self.assertEqual(frame.loc["2009-01-06", "GVZ"], 40.0)

    def test_every_byte_pin_before_any_json_numerical_decode(self):
        for symbol, field in [("OVX", "raw"), ("GVZ", "raw"), ("GVZ", "parsed")]:
            path = self.root / self.histories[symbol][field]
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with (
                self.subTest(symbol=symbol, field=field),
                mock.patch.object(inputs, "_decode_parsed") as decode,
            ):
                with self.assertRaises(ValueError):
                    inputs.read_commodity_sources(self.root, self.histories)
                decode.assert_not_called()
            path.write_bytes(original)

    def test_all_parsed_dates_preflight_before_any_numeric_decode(self):
        for bad in ["2026-01-01", "2008-12-31", "2009-1-06", "2009-02-30"]:
            original = (self.root / self.histories["GVZ"]["parsed"]).read_bytes()
            self.change_parsed(
                "GVZ", lambda p, bad=bad: p["records"][-1].update(date=bad, value=1e200)
            )
            with self.subTest(bad=bad), mock.patch.object(inputs, "_decode_parsed") as decode:
                with self.assertRaises(ValueError):
                    inputs.read_commodity_sources(self.root, self.histories)
                decode.assert_not_called()
            path = self.root / self.histories["GVZ"]["parsed"]
            path.write_bytes(original)
            self.histories["GVZ"]["parsed_sha256"] = hashlib.sha256(original).hexdigest()

    def test_duplicate_or_descending_parsed_dates_fail_preflight(self):
        self.change_parsed("GVZ", lambda p: p["records"][-1].update(date="2009-01-02"))
        with mock.patch.object(inputs, "_decode_parsed") as decode:
            with self.assertRaises(ValueError):
                inputs.read_commodity_sources(self.root, self.histories)
            decode.assert_not_called()

    def test_independent_reconstruction_rejects_rehashed_value_corruption(self):
        self.change_parsed("OVX", lambda p: p["records"][0].update(value=21.0))
        with self.assertRaises(ValueError):
            inputs.read_commodity_sources(self.root, self.histories)

    def test_independent_reconstruction_rejects_metadata_and_missingness(self):
        self.change_parsed("GVZ", lambda p: p["metadata"].update(after_end_rows=0))
        with self.assertRaises(ValueError):
            inputs.read_commodity_sources(self.root, self.histories)

    def test_exact_inventory_sha_types_and_safe_relative_paths(self):
        changes = [
            lambda h: h.pop("GVZ"),
            lambda h: h.update(VIX=h["GVZ"]),
            lambda h: h["GVZ"].update(parsed_sha256="wrong"),
            lambda h: h["GVZ"].update(extra=True),
            lambda h: h["GVZ"].update(raw="../outside"),
            lambda h: h["GVZ"].update(raw="/tmp/outside"),
        ]
        import copy

        for change in changes:
            histories = copy.deepcopy(self.histories)
            change(histories)
            with self.assertRaises(ValueError):
                inputs.read_commodity_sources(self.root, histories)

    def test_no_mutation_or_source_replacement(self):
        before = {p: p.read_bytes() for p in self.root.iterdir()}
        inputs.read_commodity_sources(self.root, self.histories)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.iterdir()})


if __name__ == "__main__":
    unittest.main()
