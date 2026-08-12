"""Contracts for the independent Hugging Face option-flow verifier.

These tests were written before ``src.verify_free_signal_option_flow``.  The
first run is intentionally expected to fail at import time; that failure is
part of the frozen tests-first record for this post-program diagnostic.
"""

from __future__ import annotations

import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import verify_free_signal_option_flow as verifier


def _option_row(
    *,
    symbol: str = "QQQ",
    timestamp: str = "2025-03-10T13:30:00Z",
    option_symbol: str = "QQQ250314P00500000",
    option_type: str = "put",
    expiration: str = "2025-03-14",
    volume: int = 20,
    trades: int = 4,
) -> dict:
    instant = pd.Timestamp(timestamp)
    return {
        "datetime": timestamp,
        "date": str(instant.tz_convert("UTC").date()),
        "unix_timestamp": int(instant.timestamp()),
        "underlying_symbol": symbol,
        "option_symbol": option_symbol,
        "option_type": option_type,
        "expiration_date": expiration,
        "strike_price": 500.0,
        "open": 2.0,
        "high": 2.1,
        "low": 1.9,
        "close": 2.0,
        "volume": volume,
        "trade_count": trades,
        "vwap": 2.0,
        # Present in the archive but forbidden from every model input.
        "macro_cpi": 300.0,
    }


class IndependenceContract(unittest.TestCase):
    def test_verifier_does_not_import_the_study_implementation(self):
        tree = ast.parse(Path(verifier.__file__).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        self.assertFalse(
            any(name.endswith("free_signal_study") for name in imported), imported
        )

    def test_protocol_freezes_timing_estimator_and_common_rows(self):
        protocol = verifier.load_protocol()
        fitting = protocol["hf_option_flow"]["fitting"]
        self.assertIn("origin t+1", fitting["timing"])
        self.assertIn("target is QQQ RV on t+2", fitting["timing"])
        self.assertIn("Duan mean smearing", fitting["estimator"])
        self.assertIn("identical candidate-complete", fitting["common_training_rows"])


class InventoryContract(unittest.TestCase):
    def test_inventory_pins_every_allowed_month_and_raw_hash(self):
        protocol = verifier.load_protocol()
        allowed = ["2024-01", "2024-02"]
        protocol["hf_option_flow"]["allowed_months"] = allowed
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            entries = []
            for month in allowed:
                path = raw / f"{month}.jsonl"
                path.write_text(json.dumps(_option_row()) + "\n", encoding="utf-8")
                entries.append(
                    {
                        "month": month,
                        "path": str(path.relative_to(root)),
                        "bytes": path.stat().st_size,
                        "rows": 1,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "source_revision": protocol["hf_option_flow"]["source_revision"],
                    }
                )
            inventory = {
                "source_revision": protocol["hf_option_flow"]["source_revision"],
                "exact_complete_set": True,
                "months": entries,
            }
            inventory_path = root / "inventory.json"
            inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

            got = verifier.validate_monthly_inventory(
                protocol, inventory_path=inventory_path, root=root
            )
            self.assertEqual([item["month"] for item in got], allowed)

            (raw / "2024-02.jsonl").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "byte size|SHA-256"):
                verifier.validate_monthly_inventory(
                    protocol, inventory_path=inventory_path, root=root
                )

    def test_inventory_cannot_silently_drop_an_allowed_month(self):
        protocol = verifier.load_protocol()
        protocol["hf_option_flow"]["allowed_months"] = ["2024-01", "2024-02"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory_path = root / "inventory.json"
            inventory_path.write_text(
                json.dumps(
                    {
                        "source_revision": protocol["hf_option_flow"]["source_revision"],
                        "exact_complete_set": True,
                        "months": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AssertionError, "month"):
                verifier.validate_monthly_inventory(
                    protocol, inventory_path=inventory_path, root=root
                )


class StreamingAndAggregationContract(unittest.TestCase):
    def test_non_target_row_is_filtered_before_full_schema_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "month.jsonl"
            non_target = {"underlying_symbol": "AAPL", "intentionally": "incomplete"}
            qqq = _option_row()
            path.write_text(
                json.dumps(non_target) + "\n" + json.dumps(qqq) + "\n",
                encoding="utf-8",
            )
            got = list(verifier.iter_filtered_option_rows([path], {"QQQ", "SPY"}))
            self.assertEqual(len(got), 1)
            self.assertEqual(got[0]["underlying_symbol"], "QQQ")
            self.assertNotIn("macro_cpi", got[0])

    def test_daily_aggregate_honors_bar_availability_and_components(self):
        rows = [
            _option_row(),
            _option_row(
                timestamp="2025-03-10T13:35:00Z",
                option_symbol="QQQ250314C00500000",
                option_type="call",
                volume=10,
                trades=2,
            ),
            _option_row(
                timestamp="2025-03-10T19:55:00Z",
                option_symbol="QQQ250321P00500000",
                expiration="2025-03-21",
                volume=5,
                trades=1,
            ),
            # Starts at 16:00 ET and is not available by the close.
            _option_row(
                timestamp="2025-03-10T20:00:00Z",
                option_symbol="QQQ250321C00500000",
                option_type="call",
                expiration="2025-03-21",
                volume=99,
                trades=99,
            ),
        ]
        got = verifier.aggregate_daily_option_flow(iter(rows))
        row = got.loc[(pd.Timestamp("2025-03-10"), "QQQ")]
        self.assertEqual(row["total_volume"], 35.0)
        self.assertEqual(row["total_trade_count"], 7)
        self.assertAlmostEqual(row["log_put_call_volume_ratio"], np.log(26.0 / 11.0))
        self.assertAlmostEqual(row["near_expiry_volume_share_7d"], 30.0 / 35.0)
        self.assertAlmostEqual(row["contract_volume_hhi"], (20 / 35) ** 2 + (10 / 35) ** 2 + (5 / 35) ** 2)
        self.assertAlmostEqual(row["log_trade_count"], np.log1p(7))


class TimingAndCompositeContract(unittest.TestCase):
    def test_locked_vxn_parquet_may_store_date_as_its_index(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vxn.parquet"
            pd.DataFrame(
                {"close": [20.0, 21.0]},
                index=pd.DatetimeIndex(["2025-03-03", "2025-03-04"], name="date"),
            ).to_parquet(path)
            got = verifier._load_vxn(path)
        self.assertEqual(got.index.name, "date")
        self.assertEqual(got.tolist(), [20.0, 21.0])

    def test_composite_uses_only_strictly_prior_component_rows(self):
        columns = verifier.COMPONENTS
        frame = pd.DataFrame(
            {
                columns[0]: [0.0, 1.0, 2.0, 999.0],
                columns[1]: [0.1, 0.2, 0.3, 999.0],
                columns[2]: [0.5, 0.4, 0.3, 999.0],
                columns[3]: [1.0, 2.0, 3.0, 999.0],
            },
            index=pd.date_range("2024-01-01", periods=4),
        )
        expected = verifier.strictly_prior_composite(frame.iloc[:3], min_observations=2)
        poisoned = frame.copy()
        poisoned.iloc[-1] = -999.0
        got = verifier.strictly_prior_composite(poisoned, min_observations=2).iloc[:3]
        np.testing.assert_allclose(got, expected, equal_nan=True)

    def test_source_date_maps_to_next_observed_qqq_session(self):
        sessions = pd.DatetimeIndex(["2025-03-07", "2025-03-10", "2025-03-12"])
        source_dates = pd.DatetimeIndex(["2025-03-07", "2025-03-10"])
        got = verifier.next_observed_origins(source_dates, sessions)
        self.assertEqual(got.tolist(), [pd.Timestamp("2025-03-10"), pd.Timestamp("2025-03-12")])

    def test_vxn_is_shifted_on_source_calendar_before_origin_reindex(self):
        sessions = pd.bdate_range("2025-03-03", periods=6)
        vxn = pd.Series(np.arange(10.0, 16.0), index=sessions)
        origins = sessions[[2, 4]]
        got = verifier.prior_session_vxn(vxn, origins, delay_sessions=1)
        np.testing.assert_array_equal(got.to_numpy(), [11.0, 13.0])


class GateContract(unittest.TestCase):
    def test_frozen_minimum_training_gate_returns_structured_insufficient_data(self):
        protocol = verifier.load_protocol()
        sessions = pd.bdate_range("2024-12-02", periods=40)
        model = pd.DataFrame(
            {
                "rv_total": np.linspace(0.01, 0.02, len(sessions)),
                "log_rv_d": np.linspace(-5.0, -4.0, len(sessions)),
                "log_rv_w": np.linspace(-5.0, -4.0, len(sessions)),
                "log_rv_m": np.linspace(-5.0, -4.0, len(sessions)),
                "lagged_log_vxn": 3.0,
                "lagged_option_flow_composite": 0.0,
            },
            index=sessions,
        )
        got = verifier.rebuild_frozen_test(model, protocol)
        self.assertEqual(got["status"], "INSUFFICIENT_DATA")
        self.assertEqual(got["scored_origins"], 0)
        self.assertEqual(
            got["minimum_training_origins"],
            protocol["hf_option_flow"]["fitting"]["minimum_training_origins"],
        )


class CurrentArtifactContract(unittest.TestCase):
    def test_current_artifacts_pass_or_verify_the_frozen_insufficient_gate(self):
        if not verifier.METRICS_PATH.exists():
            self.skipTest("HF option-flow empirical artifacts are not present yet")
        got = verifier.verify_artifacts()
        self.assertIn(got["status"], {"PASS", "INSUFFICIENT_DATA"})
        self.assertGreaterEqual(got["checks"], 12)
        if got["status"] == "INSUFFICIENT_DATA":
            self.assertIn("zero historical scale", got["gate_reason"])
            self.assertEqual(got["zero_scale_component"], "near_expiry_volume_share_7d")
            self.assertEqual(got["finite_composite_rows"], {"QQQ": 0, "SPY": 0})


if __name__ == "__main__":
    unittest.main()
