"""Pre-run contracts for the frozen CFTC and free option-flow diagnostics."""
from __future__ import annotations

import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import free_signal_study


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = yaml.safe_load(
            (free_signal_study.ROOT / "free_signal_study.yaml").read_text()
        )

    def test_protocol_is_frozen_and_clean_window_is_forbidden(self):
        free_signal_study.validate_protocol(self.protocol)
        self.assertTrue(self.protocol["fences"]["forbid_clean_origins"])
        self.assertLess(
            pd.Timestamp(self.protocol["fences"]["final_origin"]),
            pd.Timestamp(self.protocol["fences"]["clean_start"]),
        )

    def test_protocol_rejects_tuesday_as_availability(self):
        bad = yaml.safe_load(yaml.safe_dump(self.protocol))
        bad["cftc_positioning"]["availability"]["ordinary_rule"] = "report date"
        with self.assertRaises(ValueError):
            free_signal_study.validate_protocol(bad)

    def test_protocol_fixes_one_cftc_feature_and_one_flow_composite(self):
        self.assertEqual(
            self.protocol["cftc_positioning"]["feature"],
            "leveraged_money_net_open_interest_share",
        )
        self.assertEqual(len(self.protocol["hf_option_flow"]["features"]["components"]), 4)
        self.assertIn("no sign or weight search", self.protocol["hf_option_flow"]["features"]["composite"])


class CftcTimingTests(unittest.TestCase):
    def test_official_socrata_all_suffixes_normalize(self):
        raw = pd.DataFrame({
            "report_date_as_yyyy_mm_dd": ["2022-06-07T00:00:00.000"],
            "cftc_contract_market_code": ["20974+"],
            "open_interest_all": [100],
            "lev_money_positions_long_all": [30],
            "lev_money_positions_short_all": [10],
        })
        got = free_signal_study.normalize_cftc_tff(raw)
        self.assertEqual(got.columns.tolist(), ["report_date", "lev_long", "lev_short", "open_interest"])

    def test_availability_is_conservative_and_blackouts_are_removed(self):
        reports = pd.DataFrame(
            {
                "report_date": pd.to_datetime(["2018-12-18", "2022-06-07", "2023-01-31"]),
                "lev_long": [10, 20, 30],
                "lev_short": [5, 10, 15],
                "open_interest": [100, 100, 100],
            }
        )
        sessions = pd.bdate_range("2018-12-01", "2023-04-30")
        got = free_signal_study.prepare_cftc_releases(reports, sessions)
        self.assertEqual(got["report_date"].dt.strftime("%Y-%m-%d").tolist(), ["2022-06-07"])
        self.assertGreaterEqual(got.iloc[0]["origin"], pd.Timestamp("2022-06-17"))

    def test_position_share_has_no_future_fill(self):
        reports = pd.DataFrame(
            {
                "report_date": pd.to_datetime(["2022-06-07"]),
                "lev_long": [30],
                "lev_short": [10],
                "open_interest": [100],
            }
        )
        got = free_signal_study.prepare_cftc_releases(
            reports, pd.bdate_range("2022-06-01", "2022-06-30")
        )
        self.assertAlmostEqual(got.iloc[0]["lev_net_share"], 0.2)
        self.assertEqual(len(got), 1)

    def test_negative_targets_with_nat_trigger_are_retained(self):
        index = pd.DatetimeIndex(["2024-01-05", "2024-01-12"])
        targets = pd.DataFrame(
            {
                "event": [False, True],
                "trigger_date": [pd.NaT, pd.Timestamp("2024-01-16")],
            },
            index=index,
        )
        features = pd.DataFrame({"log_rv_d": [1.0, 2.0]}, index=index)

        got = free_signal_study.join_origin_features(
            targets, features, ["log_rv_d"]
        )

        self.assertEqual(got["event"].tolist(), [False, True])
        self.assertTrue(pd.isna(got.iloc[0]["trigger_date"]))

    def test_report_describes_the_registered_on_or_after_mapping(self):
        metrics = {
            "models": {
                "baseline": {
                    "n": 1, "base_rate": 0.0, "positives": 0,
                    "auc": np.nan, "top_decile_lift": np.nan,
                    "top_decile_event_rate": 0.0,
                },
                "augmented": {
                    "n": 1, "base_rate": 0.0, "positives": 0,
                    "auc": np.nan, "top_decile_lift": np.nan,
                    "top_decile_event_rate": 0.0,
                },
            },
            "first_origin": "2022-06-17", "last_origin": "2022-06-17",
            "registered_success": False, "delta_auc": 0.0,
            "delta_top_decile_lift": 0.0,
        }
        report = free_signal_study._render_cftc_report(metrics)
        self.assertIn("on or after", report)
        self.assertNotIn("strictly later", report)


class OptionFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = free_signal_study.load_protocol()

    def _bars(self):
        return pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    [
                        "2025-03-10T13:30:00Z", "2025-03-10T13:35:00Z",
                        "2025-03-10T19:55:00Z", "2025-03-10T20:00:00Z",
                    ], utc=True
                ),
                "underlying_symbol": ["QQQ"] * 4,
                "option_symbol": ["QQQ250314P00500000", "QQQ250314C00500000", "QQQ250321P00500000", "QQQ250321C00500000"],
                "option_type": ["put", "call", "put", "call"],
                "expiration_date": pd.to_datetime(["2025-03-14", "2025-03-14", "2025-03-21", "2025-03-21"]),
                "volume": [20, 10, 5, 99],
                "trade_count": [4, 2, 1, 99],
                "close": [2.0, 3.0, 1.0, 4.0],
            }
        )

    def test_bar_start_plus_five_minutes_enforces_close_boundary(self):
        got = free_signal_study.aggregate_option_flow(self._bars(), decision_time="16:00")
        row = got.iloc[0]
        # 16:00 ET bar start becomes available at 16:05 and is excluded.
        self.assertEqual(row["total_volume"], 35)
        self.assertEqual(row["total_trade_count"], 7)

    def test_macro_fields_are_rejected(self):
        bars = self._bars()
        bars["macro_cpi"] = 300.0
        with self.assertRaises(ValueError):
            free_signal_study.aggregate_option_flow(bars)

    def test_sparse_missing_session_is_not_filled(self):
        first = free_signal_study.aggregate_option_flow(self._bars())
        self.assertEqual(len(first), 1)
        self.assertNotIn(pd.Timestamp("2025-03-11"), first.index.get_level_values("date"))

    def test_training_zscore_is_invariant_to_future_poison(self):
        frame = pd.DataFrame(
            {
                "log_put_call_volume_ratio": [0.0, 1.0, 2.0, 999.0],
                "near_expiry_volume_share_7d": [0.1, 0.2, 0.3, 999.0],
                "contract_volume_hhi": [0.5, 0.4, 0.3, 999.0],
                "log_trade_count": [1.0, 2.0, 3.0, 999.0],
            },
            index=pd.date_range("2024-01-01", periods=4),
        )
        a = free_signal_study.training_scaled_composite(frame.iloc[:3], min_observations=2)
        poisoned = frame.copy()
        poisoned.iloc[-1] = -999.0
        b = free_signal_study.training_scaled_composite(poisoned.iloc[:3], min_observations=2)
        np.testing.assert_allclose(a, b, equal_nan=True)

    def test_zero_variance_registered_component_makes_composite_unavailable(self):
        frame = pd.DataFrame({
            "log_put_call_volume_ratio": np.arange(6.0),
            "near_expiry_volume_share_7d": 0.0,
            "contract_volume_hhi": np.linspace(0.1, 0.6, 6),
            "log_trade_count": np.linspace(1.0, 2.0, 6),
        })
        got = free_signal_study.training_scaled_composite(
            frame, min_observations=3
        )
        self.assertTrue(got.isna().all())

    def test_trade_count_is_not_silently_truncated(self):
        bars = self._bars().iloc[[0]].copy()
        bars["trade_count"] = 1.5
        with self.assertRaisesRegex(ValueError, "integer|trade_count"):
            free_signal_study.aggregate_option_flow(bars)

    def test_pinned_inventory_requires_every_month_and_exact_raw_sha(self):
        spec = yaml.safe_load(yaml.safe_dump(self.protocol))
        spec["hf_option_flow"]["allowed_months"] = ["2024-01", "2024-02"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            entries = []
            for month in spec["hf_option_flow"]["allowed_months"]:
                path = raw / f"{month}.jsonl"
                path.write_text(json.dumps({"underlying_symbol": "AAPL"}) + "\n")
                entries.append({
                    "month": month,
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "source_revision": spec["hf_option_flow"]["source_revision"],
                })
            inventory_path = root / "inventory.json"
            inventory_path.write_text(json.dumps({
                "source_revision": spec["hf_option_flow"]["source_revision"],
                "exact_complete_set": True,
                "months": entries,
            }))
            got = free_signal_study.validate_option_flow_inventory(
                spec, inventory_path=inventory_path, root=root
            )
            self.assertEqual([entry["month"] for entry in got], ["2024-01", "2024-02"])
            target = raw / "2024-02.jsonl"
            payload = bytearray(target.read_bytes())
            payload[-2] = ord("x") if payload[-2] != ord("x") else ord("y")
            target.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                free_signal_study.validate_option_flow_inventory(
                    spec, inventory_path=inventory_path, root=root
                )

    def test_stream_filters_non_targets_before_full_record_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "month.jsonl"
            rows = [
                {"underlying_symbol": "AAPL", "deliberately": "incomplete"},
                {
                    "datetime": "2025-03-10T13:30:00Z",
                    "underlying_symbol": "QQQ",
                    "option_symbol": "QQQ250314P00500000",
                    "option_type": "put",
                    "expiration_date": "2025-03-14",
                    "volume": 20,
                    "trade_count": 4,
                    "close": 2.0,
                    "macro_cpi": 300.0,
                },
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            iterator = free_signal_study.iter_filtered_option_rows(path, {"QQQ", "SPY"})
            self.assertTrue(inspect.isgenerator(iterator))
            got = list(iterator)
            self.assertEqual(len(got), 1)
            self.assertEqual(got[0]["underlying_symbol"], "QQQ")
            self.assertNotIn("macro_cpi", got[0])

    def test_weekend_activity_is_omitted_and_absent_session_is_not_zero(self):
        bars = pd.concat([
            self._bars().iloc[[0]],
            self._bars().iloc[[0]].assign(datetime=pd.Timestamp("2025-03-08T14:30:00Z")),
        ], ignore_index=True)
        sessions = pd.DatetimeIndex(["2025-03-07", "2025-03-10", "2025-03-11"])
        got = free_signal_study.aggregate_option_flow(
            bars, observed_sessions=sessions
        )
        dates = got.index.get_level_values("date")
        self.assertEqual(dates.tolist(), [pd.Timestamp("2025-03-10")])
        self.assertNotIn(pd.Timestamp("2025-03-11"), dates)

    def test_flow_and_vxn_t_enter_origin_t_plus_one_before_target_t_plus_two(self):
        sessions = pd.bdate_range("2025-03-03", periods=6)
        history = pd.DataFrame({
            "rv_total": np.linspace(0.01, 0.02, 6),
            "log_rv_d": np.linspace(-5.0, -4.5, 6),
            "log_rv_w": np.linspace(-5.1, -4.6, 6),
            "log_rv_m": np.linspace(-5.2, -4.7, 6),
        }, index=sessions)
        daily = pd.DataFrame({
            "option_flow_composite": [0.25],
        }, index=pd.MultiIndex.from_tuples(
            [(sessions[1], "QQQ")], names=["date", "symbol"]
        ))
        vxn = pd.Series(np.arange(10.0, 16.0), index=sessions)
        got = free_signal_study.build_option_flow_design(daily, history, vxn, self.protocol)
        row = got.loc[sessions[2]]
        self.assertEqual(row["measurement_date"], sessions[1])
        self.assertAlmostEqual(row["lagged_option_flow_composite"], 0.25)
        self.assertAlmostEqual(row["lagged_log_vxn"], np.log(vxn.loc[sessions[1]]))
        self.assertEqual(row["target_date"], sessions[3])
        self.assertAlmostEqual(row["actual_var"], history.loc[sessions[3], "rv_total"])

    def test_insufficient_training_is_reported_without_relaxing_gate(self):
        sessions = pd.bdate_range("2024-12-02", periods=40)
        design = pd.DataFrame({
            "measurement_date": sessions - pd.offsets.BDay(1),
            "target_date": sessions + pd.offsets.BDay(1),
            "actual_var": np.linspace(0.01, 0.02, len(sessions)),
            "target_log_rv": np.log(np.linspace(0.01, 0.02, len(sessions))),
            "log_rv_d": np.linspace(-5.0, -4.0, len(sessions)),
            "log_rv_w": np.linspace(-5.1, -4.1, len(sessions)),
            "log_rv_m": np.linspace(-5.2, -4.2, len(sessions)),
            "lagged_log_vxn": 3.0,
            "lagged_option_flow_composite": 0.0,
        }, index=sessions)
        forecasts, diagnostics = free_signal_study.forecast_option_flow(
            design, self.protocol
        )
        self.assertTrue(forecasts.empty)
        self.assertEqual(diagnostics["status"], "INSUFFICIENT_DATA")
        self.assertEqual(
            diagnostics["minimum_training_origins"],
            self.protocol["hf_option_flow"]["fitting"]["minimum_training_origins"],
        )

    def test_baseline_and_candidate_are_fit_and_scored_on_identical_rows(self):
        sessions = pd.bdate_range("2024-05-01", periods=230)
        values = np.linspace(0.01, 0.03, len(sessions))
        design = pd.DataFrame({
            "measurement_date": pd.Series(sessions, index=sessions).shift(1),
            "target_date": pd.Series(sessions, index=sessions).shift(-1),
            "actual_var": pd.Series(values, index=sessions).shift(-1),
            "target_log_rv": np.log(pd.Series(values, index=sessions).shift(-1)),
            "log_rv_d": np.log(values),
            "log_rv_w": np.log(values + 0.001),
            "log_rv_m": np.log(values + 0.002),
            "lagged_log_vxn": 3.0 + np.linspace(0.0, 0.1, len(sessions)),
            "lagged_option_flow_composite": np.sin(np.arange(len(sessions)) / 10),
        }, index=sessions)
        first_score = design.index[design.index >= pd.Timestamp("2025-01-02")][0]
        design.loc[first_score, "lagged_option_flow_composite"] = np.nan
        forecasts, diagnostics = free_signal_study.forecast_option_flow(design, self.protocol)
        self.assertNotIn(first_score, forecasts.index)
        self.assertGreater(len(forecasts), 0)
        self.assertTrue((forecasts["baseline_train_n"] == forecasts["augmented_train_n"]).all())
        self.assertEqual(diagnostics["status"], "SCORED")


if __name__ == "__main__":
    unittest.main()
