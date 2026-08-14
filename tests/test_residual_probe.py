"""Pre-run contracts for the residualized latent probe.

Written before src/residual_probe.py and before any residualized score exists.
`make residual-probe` runs this suite first; a failing fence blocks the run.

The contracts that matter here are the leakage ones. Residualizing a latent
coordinate against HAR features is a fit, and a fit on the wrong rows leaks the
held-out year into the very thing the probe claims is orthogonal to it. Three
separate fits happen per fold -- the residualization, the coordinate selection,
and the ridge -- and every one of them must see training rows only.
"""
from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROTOCOL = ROOT / "residual_probe.yaml"


class TestProtocolIsFrozenAndCoherent(unittest.TestCase):

    def setUp(self):
        self.p = yaml.safe_load(PROTOCOL.read_text())

    def test_protocol_exists_and_declares_itself_post_result(self):
        self.assertEqual(self.p["status"], "post_result_additive")
        self.assertIn("post-result", self.p["decision"].lower() + " "
                      + " ".join(self.p["must_report"]).lower())

    def test_residualization_features_are_the_benchmark_feature_set(self):
        self.assertEqual(self.p["residualization"]["features"],
                         ["log_rv_d", "log_rv_w", "log_rv_m"])

    def test_residualization_is_fit_on_training_rows_only(self):
        self.assertIn("TRAINING", self.p["residualization"]["fit_rows"])
        self.assertIn("forbidden", self.p["residualization"]["fit_rows"])

    def test_primary_metric_is_within_fold_not_pooled(self):
        self.assertIn("WITHIN-FOLD", self.p["scoring"]["primary_metric"])
        self.assertIn("NOT the registered statistic",
                      self.p["scoring"]["primary_metric"])

    def test_inference_unit_is_folds_not_fold_phase_cells(self):
        unit = self.p["scoring"]["unit_of_inference"]
        self.assertIn("24 annual folds", unit)
        self.assertIn("NOT independent", unit)

    def test_equivalence_margin_is_declared_before_running(self):
        self.assertEqual(self.p["inference"]["equivalence_margin_auc"], 0.01)
        self.assertTrue(self.p["inference"]["power_reporting_required"])
        self.assertIn("equivalent", self.p["inference"]["verdict"])
        self.assertIn("inconclusive", self.p["inference"]["verdict"])

    def test_criterion_has_no_resolution_floor_above_alpha(self):
        """The repository's signature defect, checked before running."""
        from scipy import stats
        n = int(self.p["sample"]["expected_folds"])
        best = stats.binomtest(n, n, 0.5).pvalue
        self.assertLess(best, self.p["inference"]["alpha"],
                        "the registered sign test cannot reach alpha at its own "
                        "resolution -- an unreachable gate")

    def test_clean_window_fence_is_registered(self):
        self.assertTrue(self.p["sample"]["last_origin_must_precede_clean_window"])
        self.assertEqual(self.p["sample"]["clean_window_start"], "2025-11-03")


class TestResidualizationMechanics(unittest.TestCase):
    """Properties the implementation must satisfy, tested on synthetic data."""

    # Signature settled before the module was written and before any empirical
    # result existed: residualizing rows you did not fit on needs both their
    # design rows and their own coordinate values.
    @staticmethod
    def _fit_apply(train_H, train_z, apply_H, apply_z=None):
        from src.residual_probe import residualize
        if apply_z is None:
            apply_z = train_z
        return residualize(train_H, train_z, apply_H, apply_z)

    def setUp(self):
        try:
            import src.residual_probe  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("src/residual_probe.py not written yet (pre-run state)")

    def test_residual_is_orthogonal_to_the_features_in_sample(self):
        rng = np.random.default_rng(0)
        H = np.column_stack([np.ones(400), rng.standard_normal((400, 3))])
        z = H @ np.array([1.0, 2.0, -1.0, 0.5]) + rng.standard_normal(400) * 0.3
        r = self._fit_apply(H, z, H)
        self.assertTrue(np.allclose(H.T @ r, 0, atol=1e-8),
                        "in-sample residual must be orthogonal to the design")

    def test_a_coordinate_that_is_pure_har_residualizes_to_noise(self):
        rng = np.random.default_rng(1)
        H = np.column_stack([np.ones(500), rng.standard_normal((500, 3))])
        z = H @ np.array([0.0, 1.0, 2.0, 3.0])          # exactly HAR, no extra
        r = self._fit_apply(H, z, H)
        self.assertLess(np.abs(r).max(), 1e-8)

    def test_a_coordinate_orthogonal_to_har_survives_residualization(self):
        rng = np.random.default_rng(2)
        H = np.column_stack([np.ones(500), rng.standard_normal((500, 3))])
        extra = rng.standard_normal(500)
        z = H @ np.array([1.0, 1.0, 1.0, 1.0]) + 5.0 * extra
        r = self._fit_apply(H, z, H)
        self.assertGreater(np.corrcoef(r, extra)[0, 1], 0.95)

    def test_coefficients_come_from_train_and_are_applied_to_test(self):
        """The leakage contract. Residuals on held-out rows must be computed
        with TRAINING coefficients, so a held-out row whose z is exactly its
        training-implied fit residualizes to ~0 even though it was never fit."""
        rng = np.random.default_rng(3)
        Htr = np.column_stack([np.ones(300), rng.standard_normal((300, 3))])
        beta = np.array([0.5, 1.0, -2.0, 0.25])
        ztr = Htr @ beta
        Hte = np.column_stack([np.ones(50), rng.standard_normal((50, 3))])
        zte = Hte @ beta                      # held-out rows obey the same law
        r_all = self._fit_apply(Htr, ztr, np.vstack([Htr, Hte]),
                                np.concatenate([ztr, zte]))
        self.assertEqual(len(r_all), 350)
        self.assertLess(np.abs(r_all[300:]).max(), 1e-6,
                        "held-out residuals must use TRAINING coefficients")

    def test_refitting_on_the_held_out_rows_would_be_detectable(self):
        """Guard against a future 'simplification' that fits on everything: if
        coefficients were refit on the pooled sample the held-out residual of a
        deliberately shifted block would be pulled toward zero."""
        rng = np.random.default_rng(6)
        Htr = np.column_stack([np.ones(300), rng.standard_normal((300, 3))])
        beta = np.array([0.0, 1.0, 0.0, 0.0])
        ztr = Htr @ beta + rng.standard_normal(300) * 0.01
        Hte = np.column_stack([np.ones(80), rng.standard_normal((80, 3))])
        zte = Hte @ beta + 7.0                # held-out block shifted by +7
        r = self._fit_apply(Htr, ztr, np.vstack([Htr, Hte]),
                            np.concatenate([ztr, zte]))
        self.assertGreater(r[300:].mean(), 6.0,
                           "training coefficients must not absorb a held-out shift")


class TestWithinFoldScoring(unittest.TestCase):

    def setUp(self):
        try:
            import src.residual_probe  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("src/residual_probe.py not written yet (pre-run state)")

    def test_within_fold_auc_ignores_between_fold_separation(self):
        """A fold-constant score must score 0.5 within fold, whatever it scores
        pooled. This is the whole point of the corrected metric."""
        from src.residual_probe import within_fold_auc
        rng = np.random.default_rng(4)
        rows = []
        for fold, rate in enumerate([0.05, 0.5, 0.9]):
            n = 200
            ev = (rng.random(n) < rate).astype(int)
            rows.append(pd.DataFrame({"fold_year": fold, "ranking_phase": 0,
                                      "event": ev, "s": float(rate)}))
        d = pd.concat(rows, ignore_index=True)
        auc, cells = within_fold_auc(d, "s")
        self.assertEqual(cells, 3)
        self.assertAlmostEqual(auc, 0.5, places=6)

    def test_within_fold_auc_rewards_genuine_within_fold_ranking(self):
        from src.residual_probe import within_fold_auc
        rng = np.random.default_rng(5)
        rows = []
        for fold, rate in enumerate([0.2, 0.5, 0.8]):
            n = 300
            ev = (rng.random(n) < rate).astype(int)
            s = ev + rng.standard_normal(n) * 0.4       # informative within fold
            rows.append(pd.DataFrame({"fold_year": fold, "ranking_phase": 0,
                                      "event": ev, "s": s}))
        d = pd.concat(rows, ignore_index=True)
        auc, _ = within_fold_auc(d, "s")
        self.assertGreater(auc, 0.85)

    def test_cells_without_both_classes_are_dropped_not_scored_as_half(self):
        from src.residual_probe import within_fold_auc
        d = pd.DataFrame({"fold_year": [0] * 10 + [1] * 10,
                          "ranking_phase": 0,
                          "event": [0] * 10 + [0, 1] * 5,
                          "s": np.arange(20, dtype=float)})
        _, cells = within_fold_auc(d, "s")
        self.assertEqual(cells, 1, "a single-class fold has no orderable pair")


class TestVerdictCannotLaunderAFailureToReject(unittest.TestCase):
    """The defect this repository has committed most often, fenced here."""

    def setUp(self):
        try:
            import src.residual_probe  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("src/residual_probe.py not written yet (pre-run state)")

    def test_non_significant_without_equivalence_is_inconclusive(self):
        from src.residual_probe import verdict
        self.assertEqual(verdict(p_paired=0.7, mean_delta=0.001,
                                 p_tost=0.4, alpha=0.05), "inconclusive")

    def test_equivalence_is_only_earned_from_the_equivalence_test(self):
        from src.residual_probe import verdict
        self.assertEqual(verdict(p_paired=0.7, mean_delta=0.0001,
                                 p_tost=0.01, alpha=0.05), "equivalent")

    def test_a_real_effect_is_reported_as_adding(self):
        from src.residual_probe import verdict
        self.assertEqual(verdict(p_paired=0.001, mean_delta=0.05,
                                 p_tost=0.9, alpha=0.05), "adds")

    def test_a_significant_negative_delta_is_not_called_adding(self):
        from src.residual_probe import verdict
        self.assertNotEqual(verdict(p_paired=0.001, mean_delta=-0.05,
                                    p_tost=0.9, alpha=0.05), "adds")


if __name__ == "__main__":
    unittest.main()
