"""Pre-run contracts for the frozen QQQ transition-frame diagnostic."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src import regime_transition


def _protocol() -> dict:
    return {
        "window": {
            "start": "2016-01-04",
            "end": "2025-10-17",
            "clean_start": "2025-11-03",
            "forbid_clean_origins": True,
        },
        "target": {
            "horizon_sessions": 2,
            "stress_quantile": 0.80,
            "score_only_calm_origins": True,
        },
        "fitting": {
            "min_train_observations": 10,
            "hmm_states": 2,
            "hmm_max_iter": 100,
            "hmm_tolerance": 1e-7,
            "logistic_ridge": 1e-6,
        },
    }


class FenceContract(unittest.TestCase):
    def test_clean_overlap_is_rejected(self):
        regime_transition.validate_protocol(_protocol())
        cfg = _protocol()
        cfg["window"]["end"] = cfg["window"]["clean_start"]
        with self.assertRaisesRegex(ValueError, "clean"):
            regime_transition.validate_protocol(cfg)

    def test_target_end_must_precede_clean_start(self):
        idx = pd.bdate_range("2025-10-27", "2025-11-04")
        with self.assertRaisesRegex(ValueError, "target"):
            regime_transition.enforce_target_fence(
                pd.DatetimeIndex([pd.Timestamp("2025-10-30")]), idx, horizon=2,
                clean_start=pd.Timestamp("2025-11-03")
            )


class HMMContract(unittest.TestCase):
    def test_fitted_states_are_ordered_and_transition_rows_sum_to_one(self):
        rng = np.random.default_rng(7)
        y = np.r_[rng.normal(-10.0, 0.2, 120), rng.normal(-6.0, 0.3, 80)]
        params = regime_transition.fit_gaussian_hmm(y, max_iter=100, tolerance=1e-7)
        self.assertLess(params["means"][0], params["means"][1])
        np.testing.assert_allclose(params["transition"].sum(axis=1), 1.0)
        self.assertTrue((params["variances"] > 0).all())

    def test_forward_filter_is_future_invariant(self):
        params = {
            "initial": np.array([0.9, 0.1]),
            "transition": np.array([[0.95, 0.05], [0.20, 0.80]]),
            "means": np.array([-10.0, -6.0]),
            "variances": np.array([0.2, 0.3]),
        }
        y = np.array([-10.1, -9.9, -10.0, -6.2, -6.0])
        before = regime_transition.filter_gaussian_hmm(y, params)
        changed = y.copy()
        changed[-1] = 100.0
        after = regime_transition.filter_gaussian_hmm(changed, params)
        np.testing.assert_allclose(before[:-1], after[:-1], rtol=0, atol=0)

    def test_future_exceedance_probability_is_bounded_and_state_sensitive(self):
        params = {
            "initial": np.array([0.9, 0.1]),
            "transition": np.array([[0.95, 0.05], [0.20, 0.80]]),
            "means": np.array([-10.0, -6.0]),
            "variances": np.array([0.2, 0.3]),
        }
        low = regime_transition.future_exceedance_probability(
            np.array([1.0, 0.0]), params, threshold=-7.0, horizon=2
        )
        high = regime_transition.future_exceedance_probability(
            np.array([0.0, 1.0]), params, threshold=-7.0, horizon=2
        )
        self.assertTrue(0.0 <= low <= 1.0)
        self.assertTrue(0.0 <= high <= 1.0)
        self.assertGreater(high, low)


class TargetContract(unittest.TestCase):
    def test_fold_target_uses_training_threshold_and_completed_future_only(self):
        idx = pd.bdate_range("2015-12-21", periods=15)
        y = pd.Series(np.arange(15.0), index=idx)
        cutoff = idx[9]
        origins = idx[10:12]
        got = regime_transition.build_fold_targets(
            y, cutoff=cutoff, origins=origins, horizon=2, quantile=0.80
        )
        self.assertTrue((got["threshold"] == y.loc[:cutoff].quantile(0.80)).all())
        self.assertTrue(got["event"].all())
        eligible = regime_transition.completed_training_origins(idx, cutoff=cutoff, horizon=2)
        self.assertEqual(eligible.max(), idx[7])


if __name__ == "__main__":
    unittest.main()
