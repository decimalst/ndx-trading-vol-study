"""Pre-run contracts for the separate 99-control sparse-k=1 diagnostic.

This file and ``latent_k1_confirmation.yaml`` were written before running the
new empirical diagnostic. The earlier ten-control ladder remains untouched.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src import latent_k1_confirmation as study


class ProtocolContract(unittest.TestCase):
    def test_only_one_rung_and_ninety_nine_fixed_controls(self):
        protocol = study.load_protocol()
        self.assertEqual(protocol["probe"]["rung"], "sparse_k1_only")
        self.assertEqual(protocol["control_task"]["draws"], 99)
        self.assertEqual(study.control_seeds(protocol), list(range(4300, 4399)))
        self.assertEqual(protocol["control_task"]["familywise_correction"], "none_single_registered_test")

    def test_ninety_nine_controls_make_point_zero_one_attainable(self):
        got = study.exact_randomization_test(.80, np.linspace(.40, .70, 99), alpha=.05)
        self.assertAlmostEqual(got["minimum_attainable_p"], .01)
        self.assertAlmostEqual(got["exact_p"], .01)
        self.assertTrue(got["formal_evidence"])


class LeakageContract(unittest.TestCase):
    def test_selector_uses_training_labels_only(self):
        X_train = np.zeros((10, 4))
        y_train = np.array([0] * 5 + [1] * 5)
        X_train[y_train == 1, 3] = 4.0
        selected, _effect = study.select_top1(X_train, y_train)
        self.assertEqual(selected, 3)
        # Held-out values and labels are deliberately unavailable to the API.
        self.assertEqual(len(study.select_top1.__annotations__), 3)

    def test_control_selector_uses_each_controls_training_labels(self):
        X = np.zeros((12, 3))
        labels = np.column_stack([
            np.array([0] * 6 + [1] * 6),
            np.tile([0, 1], 6),
        ])
        X[6:, 0] = 3.0
        X[1::2, 2] = 5.0
        selected, _effects = study.select_top1_many(X, labels)
        np.testing.assert_array_equal(selected, [0, 2])

    def test_lagged_vxn_shifts_before_origin_reindex(self):
        sessions = pd.bdate_range("2020-01-02", periods=6)
        vxn = pd.Series(np.arange(10.0, 16.0), index=sessions)
        origins = sessions[[2, 4]]
        got = study.prior_session_vxn(vxn, origins)
        np.testing.assert_array_equal(got.to_numpy(), [11.0, 13.0])

    def test_coordinate_correlations_reject_training_overlap(self):
        train = pd.bdate_range("2020-01-02", periods=5)
        test = pd.bdate_range("2020-01-09", periods=5)
        frame = pd.DataFrame({"coordinate": range(5), "x": range(5)}, index=test)
        study.fold_correlations(frame, train_index=train, fold_year=2020, variables=["x"])
        with self.assertRaisesRegex(ValueError, "overlap"):
            study.fold_correlations(frame, train_index=test, fold_year=2020, variables=["x"])


# ArtifactContract moved to tests/env/test_local_artifacts.py on 2026-08-13.
# Its guard checked METRICS_PATH (committed) while verify_results() reaches
# latent_embedding_chunks/manifest.json (git-ignored), so it passed the guard
# and then hard-errored on any clean clone. It is an environment prerequisite,
# not a contract about this module.


if __name__ == "__main__":
    unittest.main()
