"""Pre-run contracts for the frozen SKEW-conditioned carry diagnostic."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import skew_carry


def _protocol() -> dict:
    return {
        "window": {
            "start": "2016-01-04",
            "end": "2025-10-17",
            "clean_start": "2025-11-03",
            "forbid_clean_origins": True,
        },
        "source": {
            "close_delay_sessions": 1,
            "no_forward_fill": True,
            "historical_anchor": {
                "date": "2018-08-13",
                "close": 159.03,
                "tolerance": 0.01,
            },
        },
        "repair": {
            "trailing_sessions": 5,
            "min_observations": 3,
            "high_quantile": 0.80,
            "missing_value_action": "flat",
        },
        "evaluation": {
            "step_sessions": 3,
            "all_phase_offsets": True,
            "known_adverse_origins": [
                "2020-02-18", "2020-02-19", "2020-02-21"
            ],
        },
    }


class ProtocolFenceContract(unittest.TestCase):
    def test_registered_window_stops_before_clean(self):
        skew_carry.validate_protocol(_protocol())

    def test_clean_overlap_is_rejected(self):
        cfg = _protocol()
        cfg["window"]["end"] = "2025-11-03"
        with self.assertRaisesRegex(ValueError, "clean"):
            skew_carry.validate_protocol(cfg)

    def test_only_fixed_80th_percentile_gate_is_allowed(self):
        cfg = _protocol()
        cfg["repair"]["high_quantile"] = 0.75
        with self.assertRaisesRegex(ValueError, "0.80"):
            skew_carry.validate_protocol(cfg)


class SourceContract(unittest.TestCase):
    SAMPLE = """DATE,SKEW
2018-08-13,159.03
2018-08-14,150.00
"""

    def test_cboe_parser_and_historical_anchor(self):
        got = skew_carry.parse_cboe_skew_csv(self.SAMPLE)
        self.assertEqual(got.index.name, "date")
        self.assertEqual(got.loc["2018-08-13", "close"], 159.03)
        skew_carry.validate_skew_history(got, _protocol())

    def test_rewritten_history_fails_closed(self):
        got = skew_carry.parse_cboe_skew_csv(
            self.SAMPLE.replace("159.03", "145.00")
        )
        with self.assertRaisesRegex(ValueError, "anchor"):
            skew_carry.validate_skew_history(got, _protocol())

    def test_duplicate_dates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            skew_carry.parse_cboe_skew_csv(self.SAMPLE + self.SAMPLE.splitlines()[1] + "\n")

    def test_study_coverage_requires_a_full_warmup_and_window_end(self):
        cfg = _protocol()
        cfg["repair"]["trailing_sessions"] = 5
        dates = pd.bdate_range("2015-12-24", "2025-10-17")
        frame = pd.DataFrame({"close": 120.0}, index=dates)
        skew_carry.validate_study_coverage(frame, cfg)
        with self.assertRaisesRegex(ValueError, "warmup"):
            skew_carry.validate_study_coverage(frame.loc["2016-01-01":], cfg)
        with self.assertRaisesRegex(ValueError, "end"):
            skew_carry.validate_study_coverage(frame.loc[:"2025-10-16"], cfg)


class AvailabilityContract(unittest.TestCase):
    def test_cboe_close_is_delayed_one_session_and_missing_is_not_filled(self):
        sessions = pd.bdate_range("2024-01-02", periods=5)
        raw = pd.Series([120.0, 130.0, 150.0, 140.0], index=sessions[[0, 1, 3, 4]])
        got = skew_carry.align_skew(raw, sessions, delay_sessions=1)
        self.assertTrue(np.isnan(got.iloc[0]))
        self.assertEqual(got.iloc[1], 120.0)
        self.assertEqual(got.iloc[2], 130.0)
        self.assertTrue(np.isnan(got.iloc[3]))

    def test_gate_threshold_excludes_the_value_being_judged(self):
        sessions = pd.bdate_range("2024-01-02", periods=7)
        raw = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 200.0, 105.0], index=sessions)
        gate = skew_carry.build_skew_gate(raw, sessions, _protocol())
        # At the final session, the judged value is raw 200 from one session
        # earlier; its threshold may use raw values only through 104.
        expected = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0]).quantile(0.8)
        self.assertEqual(gate.loc[sessions[-1], "skew_lagged"], 200.0)
        self.assertAlmostEqual(gate.loc[sessions[-1], "skew_threshold"], expected)
        self.assertFalse(bool(gate.loc[sessions[-1], "skew_allowed"]))

    def test_future_skew_cannot_change_past_gate(self):
        sessions = pd.bdate_range("2024-01-02", periods=10)
        raw = pd.Series(np.arange(100.0, 110.0), index=sessions)
        origin = sessions[-2]
        before = skew_carry.build_skew_gate(raw, sessions, _protocol()).loc[origin]
        changed = raw.copy()
        changed.loc[sessions[-1]] = 1_000_000.0
        after = skew_carry.build_skew_gate(changed, sessions, _protocol()).loc[origin]
        pd.testing.assert_series_equal(before, after)


class RuleContract(unittest.TestCase):
    def test_high_skew_vetoes_an_eligible_trade_and_missing_means_flat(self):
        idx = pd.bdate_range("2020-01-02", periods=3)
        trades = pd.DataFrame(
            {"richness": [1.0, 1.0, 1.0], "pnl_vol": [2.0, -10.0, 3.0]},
            index=idx,
        )
        gate = pd.DataFrame(
            {
                "skew_lagged": [120.0, 160.0, np.nan],
                "skew_threshold": [150.0, 150.0, np.nan],
                "skew_allowed": [True, False, False],
            },
            index=idx,
        )
        got = skew_carry.apply_rules(trades, gate, richness_threshold=0.5)
        self.assertEqual(got["richness_taken"].tolist(), [True, True, True])
        self.assertEqual(got["repaired_taken"].tolist(), [True, False, False])

    def test_non_overlapping_phase_evaluation_uses_every_fixed_offset(self):
        idx = pd.bdate_range("2020-01-02", periods=12)
        frame = pd.DataFrame(
            {
                "pnl_vol": np.arange(1.0, 13.0),
                "richness_taken": [True] * 12,
                "repaired_taken": [True, False, True] * 4,
            },
            index=idx,
        )
        got = skew_carry.evaluate_phases(frame, step=3, min_all=4, min_taken=2)
        self.assertEqual(got["phase"].tolist(), [0, 2])
        self.assertTrue((got["n_all"] == 4).all())


class ReportContract(unittest.TestCase):
    def test_regenerated_report_keeps_timing_and_descriptive_disclosures(self):
        idx = pd.bdate_range("2020-02-17", periods=4)
        frame = pd.DataFrame(
            {
                "vxn": [15.0, 16.0, 17.0, 18.0],
                "pnl_vol": [2.0, -25.0, 3.0, 1.0],
                "skew_lagged": [120.0, 160.0, 125.0, 130.0],
                "skew_threshold": [140.0, 140.0, 140.0, 140.0],
                "richness_taken": [False, True, True, False],
                "repaired_taken": [False, False, True, False],
            },
            index=idx,
        )
        phases = pd.DataFrame(
            {
                "all_n": [4], "richness_n": [2], "repaired_n": [1],
                "all_mean": [-4.75], "richness_mean": [-11.0], "repaired_mean": [3.0],
                "all_cvar5": [-25.0], "richness_cvar5": [-25.0], "repaired_cvar5": [3.0],
                "all_worst": [-25.0], "richness_worst": [-25.0], "repaired_worst": [3.0],
                "all_maxdd": [25.0], "richness_maxdd": [25.0], "repaired_maxdd": [0.0],
            }
        )
        metrics = {
            "known_adverse_origins": [str(idx[1].date())],
            "participation": 0.5,
            "checks": {"participation_at_least_70pct": False},
            "mechanism_pass": False,
        }
        with TemporaryDirectory() as directory:
            report = Path(directory) / "report.md"
            with patch.object(skew_carry, "REPORT_PATH", report):
                skew_carry._write_report(frame, phases, metrics, 0.386812814)
            text = report.read_text()
        self.assertIn("not a leakage-free strategy backtest", text)
        self.assertIn("Descriptive interpretation after the frozen verdict", text)
        self.assertIn("Mechanism verdict: **FAIL**", text)


if __name__ == "__main__":
    unittest.main()
