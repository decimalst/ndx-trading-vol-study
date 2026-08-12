"""Pre-run contracts for HMM calibration and incremental-state repair."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src import regime_repair


def _protocol() -> dict:
    return {
        "training": {
            "oof_start": "2016-01-04",
            "oof_end": "2024-12-31",
            "hmm_fit_end": "2024-12-31",
            "no_holdout_refit": True,
            "min_oof_observations": 10,
        },
        "evaluation": {
            "start": "2025-11-03",
            "end": "2025-11-07",
            "required_data_end": "2025-11-14",
            "horizon_sessions": 2,
            "stress_quantile": 0.80,
            "score_only_calm_origins": True,
            "require_completed_targets": True,
        },
        "calibration": {"method": "platt", "ridge": 1e-6, "clip": 1e-6},
        "incremental_test": {
            "benchmark_features": ["log_rv_d", "log_rv_w", "log_rv_m"],
            "augmented_feature": "p_hmm_platt",
            "logistic_ridge": 1e-6,
            "same_training_rows_required": True,
        },
    }


class ProtocolContract(unittest.TestCase):
    def test_training_must_end_before_target_specific_holdout(self):
        regime_repair.validate_protocol(_protocol())
        cfg = _protocol()
        cfg["training"]["oof_end"] = cfg["evaluation"]["start"]
        with self.assertRaisesRegex(ValueError, "before"):
            regime_repair.validate_protocol(cfg)

    def test_holdout_refit_is_forbidden(self):
        cfg = _protocol()
        cfg["training"]["no_holdout_refit"] = False
        with self.assertRaisesRegex(ValueError, "refit"):
            regime_repair.validate_protocol(cfg)


class CalibrationContract(unittest.TestCase):
    def test_platt_fit_uses_logit_and_returns_bounded_probabilities(self):
        raw = pd.Series([0.01, 0.10, 0.30, 0.70, 0.90, 0.99] * 4)
        event = pd.Series([0, 0, 0, 1, 1, 1] * 4)
        model = regime_repair.fit_platt(raw, event, ridge=1e-6, clip=1e-6)
        got = regime_repair.predict_platt(model, raw)
        self.assertTrue(((got > 0) & (got < 1)).all())
        self.assertTrue(np.all(np.diff(got.iloc[:6]) >= 0))
        self.assertGreater(got.iloc[5], got.iloc[0])

    def test_holdout_labels_cannot_change_fitted_predictions(self):
        raw_train = pd.Series([0.02, 0.10, 0.25, 0.55, 0.80, 0.95] * 4)
        y_train = pd.Series([0, 0, 0, 1, 1, 1] * 4)
        raw_test = pd.Series([0.05, 0.40, 0.85])
        model = regime_repair.fit_platt(raw_train, y_train, ridge=1e-6, clip=1e-6)
        before = regime_repair.predict_platt(model, raw_test)
        _changed_holdout_labels = pd.Series([1, 1, 0])
        after = regime_repair.predict_platt(model, raw_test)
        pd.testing.assert_series_equal(before, after)


class TargetContract(unittest.TestCase):
    def test_only_completed_holdout_targets_are_returned(self):
        idx = pd.bdate_range("2025-11-03", periods=7)
        y = pd.Series(np.arange(7.0), index=idx)
        got = regime_repair.build_fixed_targets(
            y,
            origins=idx,
            threshold=2.5,
            horizon=2,
            evaluation_end=idx[-1],
        )
        self.assertEqual(got.index.max(), idx[-3])
        self.assertTrue((got["target_end"] <= idx[-1]).all())

    def test_same_row_comparison_rejects_misalignment(self):
        idx = pd.bdate_range("2025-11-03", periods=3)
        benchmark = pd.DataFrame({"event": [0, 1, 0]}, index=idx)
        augmented = benchmark.iloc[1:].copy()
        with self.assertRaisesRegex(ValueError, "same rows"):
            regime_repair.require_same_rows(benchmark, augmented)


if __name__ == "__main__":
    unittest.main()
