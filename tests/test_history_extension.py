"""Pre-run contracts for the 1999 QQQ price-only history extension."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import history_extension


def _protocol() -> dict:
    return {
        "status": "enabling_data_only",
        "source": {
            "qqq": {
                "expected_sha256": "a" * 64,
                "expected_first_date": "1999-03-10",
                "required_columns": [
                    "open", "high", "low", "close", "adj close", "volume"
                ],
            },
            "vxn": {
                "expected_sha256": "b" * 64,
                "free_file_first_date": "2009-09-14",
            },
        },
        "window": {
            "source_start": "1999-03-10",
            "origin_end": "2025-10-17",
            "clean_start": "2025-11-03",
            "forbid_clean_origins": True,
        },
        "transform": {
            "variance_floor": 1e-10,
            "har_week_sessions": 5,
            "har_month_sessions": 22,
        },
        "output": {"expected_sha256": "c" * 64},
        "implementation": {"expected_sha256": "d" * 64},
    }


def _toy_daily() -> pd.DataFrame:
    idx = pd.bdate_range("1999-03-10", periods=30)
    close = np.linspace(100.0, 106.0, len(idx))
    return pd.DataFrame(
        {
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj close": close * 0.8,
            "volume": np.arange(len(idx)) + 1_000,
        },
        index=idx,
    )


class ProtocolContract(unittest.TestCase):
    def test_clean_origins_are_forbidden(self):
        history_extension.validate_protocol(_protocol())
        bad = _protocol()
        bad["window"]["origin_end"] = bad["window"]["clean_start"]
        with self.assertRaisesRegex(ValueError, "clean"):
            history_extension.validate_protocol(bad)

    def test_source_hashes_are_mandatory(self):
        bad = _protocol()
        bad["source"]["qqq"]["expected_sha256"] = ""
        with self.assertRaisesRegex(ValueError, "sha256"):
            history_extension.validate_protocol(bad)

    def test_derived_panel_hash_is_mandatory(self):
        bad = _protocol()
        bad["output"]["expected_sha256"] = ""
        with self.assertRaisesRegex(ValueError, "output expected_sha256"):
            history_extension.validate_protocol(bad)

    def test_transform_implementation_hash_is_mandatory(self):
        bad = _protocol()
        bad["implementation"]["expected_sha256"] = ""
        with self.assertRaisesRegex(ValueError, "implementation expected_sha256"):
            history_extension.validate_protocol(bad)


class TransformationContract(unittest.TestCase):
    def test_gk_overnight_and_har_features_are_exact(self):
        daily = _toy_daily()
        got = history_extension.build_price_only_panel(daily, _protocol())
        d = daily.index[1]
        expected_gk = max(
            0.5 * np.log(daily.loc[d, "high"] / daily.loc[d, "low"]) ** 2
            - (2 * np.log(2.0) - 1)
            * np.log(daily.loc[d, "close"] / daily.loc[d, "open"]) ** 2,
            1e-10,
        )
        expected_overnight = np.log(
            daily.loc[d, "open"] / daily.iloc[0]["close"]
        ) ** 2
        self.assertAlmostEqual(got.loc[d, "rv_intraday"], expected_gk)
        self.assertAlmostEqual(got.loc[d, "var_overnight"], expected_overnight)
        self.assertAlmostEqual(
            got.loc[d, "rv_total"], expected_gk + expected_overnight
        )
        d22 = got.index[21]
        self.assertAlmostEqual(
            got.loc[d22, "log_rv_m"], got.loc[:d22, "log_rv"].tail(22).mean()
        )

    def test_post_fence_rows_cannot_change_panel(self):
        daily = _toy_daily()
        protocol = _protocol()
        protocol["window"]["origin_end"] = str(daily.index[20].date())
        protocol["window"]["clean_start"] = str(daily.index[25].date())
        before = history_extension.build_price_only_panel(daily, protocol)
        changed = daily.copy()
        changed.loc[changed.index > daily.index[20], ["open", "high", "low", "close"]] *= 10
        after = history_extension.build_price_only_panel(changed, protocol)
        pd.testing.assert_frame_equal(before, after)

    def test_panel_is_price_only_and_point_in_time(self):
        got = history_extension.build_price_only_panel(_toy_daily(), _protocol())
        self.assertEqual(
            got.columns.tolist(),
            [
                "rv_intraday", "var_overnight", "rv_total", "log_rv",
                "ret_cc", "log_rv_d", "log_rv_w", "log_rv_m",
            ],
        )
        self.assertNotIn("vxn", got.columns)
        self.assertFalse(got.index.duplicated().any())
        self.assertTrue(got.index.is_monotonic_increasing)


class SourceContract(unittest.TestCase):
    def test_bad_ohlc_geometry_is_rejected(self):
        daily = _toy_daily()
        daily.loc[daily.index[2], "high"] = daily.loc[daily.index[2], "low"] - 1
        with self.assertRaisesRegex(ValueError, "OHLC"):
            history_extension.validate_qqq_source(daily, _protocol()["source"]["qqq"])

    def test_manifest_records_input_and_output_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.bin"
            output = root / "output.bin"
            source.write_bytes(b"source")
            output.write_bytes(b"output")
            manifest = history_extension.build_manifest(
                source_path=source,
                output_path=output,
                source_rows=30,
                output_rows=29,
                source_first="1999-03-10",
                source_last="1999-04-20",
                output_first="1999-03-11",
                output_last="1999-04-20",
            )
            self.assertEqual(
                manifest["source"]["sha256"], hashlib.sha256(b"source").hexdigest()
            )
            self.assertEqual(
                manifest["output"]["sha256"], hashlib.sha256(b"output").hexdigest()
            )
            json.dumps(manifest)


class FrozenArtifactContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = history_extension.load_protocol()
        cls.root = Path(__file__).resolve().parents[1]

    def test_frozen_raw_hashes_and_boundaries_match(self):
        for name in ("qqq", "vxn"):
            spec = self.protocol["source"][name]
            path = self.root / spec["path"]
            self.assertEqual(history_extension._sha256(path), spec["expected_sha256"])
        qqq = pd.read_parquet(self.root / self.protocol["source"]["qqq"]["path"])
        vxn = pd.read_parquet(self.root / self.protocol["source"]["vxn"]["path"])
        self.assertEqual(pd.Timestamp(qqq.index.min()), pd.Timestamp("1999-03-10"))
        self.assertEqual(pd.Timestamp(vxn.index.min()), pd.Timestamp("2009-09-14"))

    def test_persisted_panel_recomputes_exactly_and_stays_preclean(self):
        qqq = pd.read_parquet(self.root / self.protocol["source"]["qqq"]["path"])
        expected = history_extension.build_price_only_panel(qqq, self.protocol)
        stored_path = self.root / self.protocol["output"]["panel"]
        stored = pd.read_parquet(stored_path)
        pd.testing.assert_frame_equal(stored, expected)
        self.assertLess(stored.index.max(), pd.Timestamp(self.protocol["window"]["clean_start"]))
        manifest = json.loads((self.root / self.protocol["output"]["manifest"]).read_text())
        self.assertEqual(
            history_extension._sha256(stored_path),
            self.protocol["output"]["expected_sha256"],
        )
        self.assertEqual(manifest["output"]["sha256"], history_extension._sha256(stored_path))
        self.assertEqual(
            manifest["protocol"]["sha256"],
            history_extension._sha256(self.root / "history_extension.yaml"),
        )
        self.assertEqual(
            manifest["implementation"]["sha256"],
            self.protocol["implementation"]["expected_sha256"],
        )
        self.assertFalse(manifest["vxn"]["joined_to_panel"])


if __name__ == "__main__":
    unittest.main()
