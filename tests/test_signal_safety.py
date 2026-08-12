"""Pre-written safety contract for the diagnostic-only signal study.

These tests were committed before src.signal_study existed and before any live
signal data was fetched or scored. Run with:
    python -m unittest tests.test_signal_safety
"""
from __future__ import annotations

import copy
import json
import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd

from src import signal_study


def _study_cfg() -> dict:
    return {
        "windows": {
            "discovery_start": "2016-01-04",
            "discovery_end": "2021-12-31",
            "confirmation_start": "2022-01-03",
            "confirmation_end": "2025-10-17",
        },
        "information_set": {
            "qqq_and_cross_asset_close_delay_sessions": 0,
            "cboe_daily_close_delay_sessions": 1,
            "no_forward_fill": True,
        },
        "cross_asset": {"scale_window": 20, "min_scale_observations": 5},
        "models": {
            "baseline": "safe_har_iv_lev",
            "candidates": ["term_slope", "cross_asset", "combined"],
        },
        "selection": {
            "tie_break_order": ["term_slope", "cross_asset", "combined"],
        },
        "fences": {"clean_start": "2025-11-03", "forbid_clean_origins": True},
    }


def _main_cfg() -> dict:
    return {
        "diagnostic_start": "2016-01-04",
        "diagnostic_end": "2025-10-17",
        "clean_start": "2025-11-03",
        "min_train_days": 30,
        "quantiles": [0.05, 0.5, 0.95],
    }


def _master(n: int = 120, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=n)
    ret = rng.normal(0.0, 0.012, n)
    rv = np.exp(-9.0 + 0.4 * rng.normal(size=n))
    return pd.DataFrame(
        {
            "rv_total": rv,
            "log_rv": np.log(rv),
            "ret_cc": ret,
            "vxn": 20.0 + rng.normal(size=n),
        },
        index=idx,
    )


class WindowFenceTests(unittest.TestCase):
    def test_registered_windows_are_disjoint_and_pre_clean(self) -> None:
        signal_study.validate_protocol(_study_cfg(), _main_cfg())

    def test_overlap_with_clean_is_rejected(self) -> None:
        cfg = _study_cfg()
        cfg["windows"]["confirmation_end"] = "2025-11-04"
        with self.assertRaises(ValueError):
            signal_study.validate_protocol(cfg, _main_cfg())

    def test_confirmation_origins_stop_at_diagnostic_end(self) -> None:
        idx = pd.bdate_range("2025-10-01", "2025-11-10")
        got = signal_study.stage_origins(idx, _study_cfg(), "confirmation")
        self.assertLessEqual(got.max(), pd.Timestamp("2025-10-17"))
        self.assertTrue((got < pd.Timestamp("2025-11-03")).all())


class AvailabilityTests(unittest.TestCase):
    def test_cboe_close_is_delayed_one_complete_session(self) -> None:
        idx = pd.bdate_range("2024-01-02", periods=4)
        raw = pd.Series([10.0, 20.0, 30.0, 40.0], index=idx)
        got = signal_study.align_daily_observations(raw, idx, delay_sessions=1)
        self.assertTrue(np.isnan(got.iloc[0]))
        self.assertEqual(got.iloc[1], 10.0)
        self.assertEqual(got.iloc[3], 30.0)

    def test_missing_daily_value_is_not_forward_filled(self) -> None:
        idx = pd.bdate_range("2024-01-02", periods=4)
        raw = pd.Series([10.0, 30.0, 40.0], index=idx[[0, 2, 3]])
        got = signal_study.align_daily_observations(raw, idx, delay_sessions=0)
        self.assertTrue(np.isnan(got.iloc[1]))

    def test_future_cross_asset_close_cannot_change_origin_feature(self) -> None:
        idx = pd.bdate_range("2023-01-02", periods=40)
        close = pd.DataFrame(
            {c: np.linspace(90 + i, 110 + i, len(idx))
             for i, c in enumerate(["hyg", "tlt", "gld", "uso", "uup"])},
            index=idx,
        )
        short_iv = pd.DataFrame({"vix9d": 18.0, "vix": 20.0}, index=idx)
        origin = idx[-2]
        before = signal_study.build_signal_features(
            idx, close, short_iv, _study_cfg()
        ).loc[origin]
        changed = close.copy()
        changed.loc[idx[-1], :] = 1_000_000.0
        after = signal_study.build_signal_features(
            idx, changed, short_iv, _study_cfg()
        ).loc[origin]
        pd.testing.assert_series_equal(before, after)


class WalkForwardTests(unittest.TestCase):
    def test_unknown_target_and_future_cannot_change_origin_forecast(self) -> None:
        master = _master()
        idx = master.index
        signals = pd.DataFrame(
            {"iv_term_slope": np.linspace(-0.2, 0.2, len(idx)),
             "xasset_stress": np.linspace(0.1, 0.8, len(idx))},
            index=idx,
        )
        origin = idx[80]
        first = signal_study.run_walk_forward(
            master, signals, pd.DatetimeIndex([origin]), _main_cfg(), "combined"
        )

        poisoned = master.copy()
        poisoned.loc[idx[81]:, "log_rv"] = 100.0
        poisoned.loc[idx[81]:, "rv_total"] = np.exp(100.0)
        second = signal_study.run_walk_forward(
            poisoned, signals, pd.DatetimeIndex([origin]), _main_cfg(), "combined"
        )
        pd.testing.assert_frame_equal(first, second)

    def test_same_date_cboe_vxn_close_cannot_change_origin_forecast(self) -> None:
        master = _master()
        idx = master.index
        signals = pd.DataFrame(
            {"iv_term_slope": np.linspace(-0.2, 0.2, len(idx)),
             "xasset_stress": np.linspace(0.1, 0.8, len(idx))},
            index=idx,
        )
        origin = idx[80]
        first = signal_study.run_walk_forward(
            master, signals, pd.DatetimeIndex([origin]), _main_cfg(),
            "safe_har_iv_lev"
        )
        changed = master.copy()
        changed.loc[origin, "vxn"] = 1_000_000.0
        second = signal_study.run_walk_forward(
            changed, signals, pd.DatetimeIndex([origin]), _main_cfg(),
            "safe_har_iv_lev"
        )
        pd.testing.assert_frame_equal(first, second)


class SelectionTests(unittest.TestCase):
    def test_no_discovery_improvement_locks_no_winner(self) -> None:
        idx = pd.bdate_range("2020-01-02", periods=3)
        losses = {
            "safe_har_iv_lev": pd.Series([0.2, 0.2, 0.2], index=idx),
            "term_slope": pd.Series([0.3, 0.2, 0.2], index=idx),
            "cross_asset": pd.Series([0.2, 0.2, 0.2], index=idx),
            "combined": pd.Series([0.4, 0.1, 0.2], index=idx),
        }
        winner, summary = signal_study.select_discovery_winner(losses, _study_cfg())
        self.assertIsNone(winner)
        self.assertEqual(summary["n_common"], 3)

    def test_selection_uses_one_common_origin_set(self) -> None:
        idx = pd.bdate_range("2020-01-02", periods=4)
        losses = {
            "safe_har_iv_lev": pd.Series([0.5, 0.5, 0.5, 0.5], index=idx),
            "term_slope": pd.Series([0.1, 0.1, 9.0], index=idx[:3]),
            "cross_asset": pd.Series([0.4, 0.4, 0.4], index=idx[1:]),
            "combined": pd.Series([0.45, 0.45, 0.45, 0.45], index=idx),
        }
        winner, summary = signal_study.select_discovery_winner(losses, _study_cfg())
        self.assertEqual(summary["n_common"], 2)
        self.assertEqual(winner, "cross_asset")


class DiscoveryLockTests(unittest.TestCase):
    def test_protocol_change_invalidates_discovery_lock(self) -> None:
        cfg = _study_cfg()
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "discovery_lock.json"
            signal_study.write_discovery_lock(
                path, cfg, winner="term_slope", discovery_scores={"term_slope": 0.3}
            )
            self.assertEqual(signal_study.read_discovery_lock(path, cfg)["winner"],
                             "term_slope")
            changed = copy.deepcopy(cfg)
            changed["windows"]["discovery_end"] = "2022-01-03"
            with self.assertRaises(ValueError):
                signal_study.read_discovery_lock(path, changed)

    def test_lock_file_is_plain_auditable_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "discovery_lock.json"
            signal_study.write_discovery_lock(
                path, _study_cfg(), winner=None, discovery_scores={}
            )
            payload = json.loads(path.read_text())
            self.assertIn("protocol_sha256", payload)
            self.assertIsNone(payload["winner"])


if __name__ == "__main__":
    unittest.main()
