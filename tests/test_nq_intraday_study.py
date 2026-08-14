"""Pre-run contracts for the frozen exploratory NQ intraday study.

This file and ``nq_intraday_study.yaml`` were written before
``src.nq_intraday_study`` existed and before the raw Kaggle file was acquired.
"""

from __future__ import annotations

import ast
import copy
import inspect
import unittest

import numpy as np
import pandas as pd

from src import nq_intraday_study as study
from src import verify_nq_intraday_study as verifier


def protocol() -> dict:
    return study.load_protocol()


def minute_session(date: str, *, scale: float = 1.0, bars: int = 390) -> pd.DataFrame:
    index = pd.date_range(f"{date} 09:30:00", periods=bars, freq="1min")
    increments = np.tile(np.array([0.0002, -0.0001, 0.00015, -0.00005, 0.0001]), 78)
    increments = increments[:bars] * scale
    close = 15_000.0 * np.exp(np.cumsum(increments))
    open_ = np.r_[15_000.0, close[:-1]]
    high = np.maximum(open_, close) * 1.00001
    low = np.minimum(open_, close) * 0.99999
    return pd.DataFrame(
        {
            "timestamp ET": index,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 100,
        }
    )


class FrozenProtocolContract(unittest.TestCase):
    def test_protocol_is_exploratory_and_preclean(self):
        cfg = protocol()
        study.validate_protocol(cfg)
        self.assertIn("exploratory", cfg["status"])
        self.assertLess(
            pd.Timestamp(cfg["fences"]["data_end"]),
            pd.Timestamp(cfg["fences"]["sealed_ndx_clean_start"]),
        )
        self.assertFalse(cfg["source"]["has_contract_identifier"])
        self.assertTrue(cfg["diagnostic"]["no_candidate_selection"])

    def test_clean_overlap_and_relaxed_jump_threshold_fail_closed(self):
        clean = copy.deepcopy(protocol())
        clean["fences"]["data_end"] = clean["fences"]["sealed_ndx_clean_start"]
        with self.assertRaisesRegex(ValueError, "clean|fence"):
            study.validate_protocol(clean)
        loose = copy.deepcopy(protocol())
        loose["jump_measure"]["alpha"] = 0.05
        with self.assertRaisesRegex(ValueError, "one-percent|threshold"):
            study.validate_protocol(loose)

    def test_contract_and_roll_metadata_cannot_be_invented(self):
        changed = copy.deepcopy(protocol())
        changed["source"]["has_contract_identifier"] = True
        with self.assertRaisesRegex(ValueError, "contract|stitch"):
            study.validate_protocol(changed)


class TimestampAndSessionContract(unittest.TestCase):
    def test_dst_ambiguous_and_nonexistent_wall_times_raise(self):
        with self.assertRaises((ValueError, TypeError)):
            study.localize_et(pd.Series(["2024-11-03 01:30:00"]))
        with self.assertRaises((ValueError, TypeError)):
            study.localize_et(pd.Series(["2024-03-10 02:30:00"]))

    def test_rth_is_interval_start_0930_through_1559_and_weekdays_only(self):
        raw = pd.concat(
            [
                minute_session("2024-01-02"),
                pd.DataFrame(
                    {
                        "timestamp ET": pd.to_datetime(
                            ["2024-01-02 09:29", "2024-01-02 16:00", "2024-01-06 10:00"]
                        ),
                        "open": 1.0,
                        "high": 1.0,
                        "low": 1.0,
                        "close": 1.0,
                        "volume": 1,
                    }
                ),
            ],
            ignore_index=True,
        )
        got = study.normalize_and_filter(raw, protocol())
        self.assertEqual(len(got), 390)
        local = got["timestamp"].dt.tz_convert("America/New_York")
        self.assertEqual(local.iloc[0].strftime("%H:%M"), "09:30")
        self.assertEqual(local.iloc[-1].strftime("%H:%M"), "15:59")

    def test_terminal_partial_session_is_rejected(self):
        raw = pd.concat(
            [minute_session("2024-01-02"), minute_session("2024-01-03", bars=100)],
            ignore_index=True,
        )
        clean = study.normalize_and_filter(raw, protocol())
        with self.assertRaisesRegex(ValueError, "terminal partial"):
            study.session_quality(clean, protocol())

    def test_post_fence_values_cannot_change_processed_rth_rows(self):
        base = minute_session("2025-10-17")
        future = minute_session("2025-10-20")
        future[["open", "high", "low", "close"]] *= 100.0
        before = study.normalize_and_filter(base, protocol())
        after = study.normalize_and_filter(pd.concat([base, future]), protocol())
        pd.testing.assert_frame_equal(before, after)


class MeasureContract(unittest.TestCase):
    def test_five_minute_sampling_uses_open_anchor_and_has_78_returns(self):
        raw = study.normalize_and_filter(minute_session("2024-01-02"), protocol())
        returns = study.sample_five_minute_returns(raw, protocol())
        self.assertEqual(len(returns), 78)
        expected = np.log(raw.iloc[4]["close"] / raw.iloc[0]["open"])
        self.assertAlmostEqual(float(returns.iloc[0]), float(expected))

    def test_no_cross_session_return_enters_rv(self):
        first = minute_session("2024-01-02")
        second = minute_session("2024-01-03")
        second[["open", "high", "low", "close"]] *= 2.0
        raw = study.normalize_and_filter(pd.concat([first, second]), protocol())
        grouped = list(raw.groupby("session", sort=True))
        first_returns = study.sample_five_minute_returns(grouped[0][1], protocol())
        second_returns = study.sample_five_minute_returns(grouped[1][1], protocol())
        self.assertAlmostEqual(float(first_returns.iloc[0]), float(second_returns.iloc[0]))
        self.assertLess(float((second_returns**2).sum()), 0.01)

    def test_bpv_tripower_and_bns_formula_are_exact(self):
        returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.012, -0.008])
        got = study.bns_measures(returns, protocol())
        n = len(returns)
        absolute = returns.abs().to_numpy()
        rv = float(np.square(returns).sum())
        bpv = np.pi / 2 * n / (n - 1) * float(np.sum(absolute[1:] * absolute[:-1]))
        mu43 = 2 ** (2 / 3) * study.gamma(7 / 6) / np.sqrt(np.pi)
        tq = (
            mu43 ** -3
            * n**2
            / (n - 2)
            * float(np.sum((absolute[2:] * absolute[1:-1] * absolute[:-2]) ** (4 / 3)))
        )
        constant = (np.pi / 2) ** 2 + np.pi - 5
        z = (1 - bpv / rv) / np.sqrt(constant / n * max(1.0, tq / bpv**2))
        self.assertAlmostEqual(got["rv"], rv)
        self.assertAlmostEqual(got["bpv"], bpv)
        self.assertAlmostEqual(got["tripower_quarticity"], tq)
        self.assertAlmostEqual(got["bns_z"], z)
        self.assertEqual(got["jump_significant"], z > 2.3263478740408408)

    def test_intraday_shape_partitions_total_rv(self):
        returns = pd.Series(np.r_[np.repeat(0.02, 12), np.repeat(0.01, 54), np.repeat(0.03, 12)])
        shape = study.intraday_shape(returns, protocol())
        self.assertAlmostEqual(
            shape["first_hour_rv_share"]
            + shape["middle_rv_share"]
            + shape["last_hour_rv_share"],
            1.0,
        )
        self.assertGreater(shape["last_hour_rv_share"], shape["first_hour_rv_share"])

    def test_discontinuity_flags_expand_one_session_each_side(self):
        sessions = pd.date_range("2024-01-02", periods=5, freq="B")
        panel = pd.DataFrame(
            {
                "session_open": [100, 100, 120, 120, 120],
                "session_close": [100, 100, 120, 120, 120],
                "max_abs_one_minute_return": [0.001] * 5,
            },
            index=sessions,
        )
        flags = study.infer_stitch_neighborhoods(panel, protocol())
        self.assertEqual(flags["stitch_trigger"].tolist(), [False, False, True, False, False])
        self.assertEqual(flags["stitch_excluded"].tolist(), [False, True, True, True, False])


class TargetAndScoringContract(unittest.TestCase):
    def test_target_uses_exactly_next_five_and_requires_quality(self):
        index = pd.bdate_range("2024-01-02", periods=8)
        daily = pd.DataFrame(
            {"jump_significant": [False, False, True, False, False, False, False, False],
             "quality_eligible": True},
            index=index,
        )
        got = study.build_forward_target(daily, horizon=5)
        self.assertTrue(bool(got.loc[index[0], "event"]))
        self.assertEqual(got.loc[index[0], "target_end"], index[5])
        self.assertTrue(pd.isna(got.loc[index[3], "event"]))
        broken = daily.copy()
        broken.loc[index[4], "quality_eligible"] = False
        got_broken = study.build_forward_target(broken, horizon=5)
        self.assertTrue(pd.isna(got_broken.loc[index[0], "event"]))

    def test_cboe_inputs_are_exact_date_then_delayed_one_session(self):
        sessions = pd.bdate_range("2024-01-02", periods=4)
        source = pd.Series([10.0, 30.0, 40.0], index=sessions[[0, 2, 3]])
        got = study.delay_cboe_close(source, sessions, delay_sessions=1)
        self.assertTrue(np.isnan(got.iloc[0]))
        self.assertEqual(got.iloc[1], 10.0)
        self.assertTrue(np.isnan(got.iloc[2]))
        self.assertEqual(got.iloc[3], 30.0)

    def test_incomplete_session_does_not_consume_cboe_delay(self):
        sessions = pd.bdate_range("2024-01-02", periods=8)
        daily = pd.DataFrame(
            {
                "rv": 0.01,
                "bns_z": 0.0,
                "jump_share": 0.0,
                "jump_significant": False,
                "quality_eligible": [True, False, True, True, True, True, True, True],
            },
            index=sessions,
        )
        source = pd.Series(np.arange(10.0, 18.0), index=sessions)
        design = study.build_model_frame(daily, source, source + 100.0, protocol())
        self.assertTrue(np.isnan(design.loc[sessions[1], "log_vxn_lag1"]))
        self.assertAlmostEqual(design.loc[sessions[2], "log_vxn_lag1"], np.log(10.0))
        self.assertAlmostEqual(design.loc[sessions[3], "log_vxn_lag1"], np.log(12.0))

    def test_annual_fold_training_targets_complete_before_test_year(self):
        index = pd.bdate_range("2023-12-15", "2024-01-15")
        frame = pd.DataFrame(
            {"target_end": index + pd.offsets.BDay(5), "event": np.arange(len(index)) % 2},
            index=index,
        )
        train, test = study.annual_fold_rows(frame, 2024)
        self.assertTrue((train["target_end"] < pd.Timestamp("2024-01-01")).all())
        self.assertTrue((test.index.year == 2024).all())

    def test_phase_metrics_report_all_five_offsets(self):
        index = pd.bdate_range("2024-01-02", periods=100)
        event = np.tile([0, 1], 50)
        frame = pd.DataFrame(
            {"event": event, "p_price_history": np.linspace(0.01, 0.99, 100),
             "p_augmented": np.linspace(0.02, 0.98, 100)},
            index=index,
        )
        got = study.phase_ranking_metrics(frame, phase_step=5, top_fraction=0.10)
        self.assertEqual(got["phase"].tolist(), [0, 1, 2, 3, 4])
        self.assertTrue({"price_history_auc", "augmented_top_decile_lift"}.issubset(got))


class IndependentVerifierContract(unittest.TestCase):
    def test_verifier_does_not_import_study_implementation(self):
        tree = ast.parse(inspect.getsource(verifier))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertNotIn("src.nq_intraday_study", imported)
        self.assertNotIn("nq_intraday_study", imported)

    def test_verifier_preflight_identifies_absent_raw_and_outputs(self):
        cfg = protocol()
        missing = verifier.preflight(cfg)
        raw = cfg["source"]["raw_path"]
        if not (study.ROOT / raw).exists():
            self.assertIn(raw, missing)
        for name in ("daily_panel", "forecasts", "phase_metrics", "metrics", "manifest"):
            path = cfg["outputs"][name]
            if not (study.ROOT / path).exists():
                self.assertIn(path, missing)


if __name__ == "__main__":
    unittest.main()
