"""Prewritten numerical contracts for staged event-cluster forecasts."""

import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import root
from scipy.special import expit

from src import event_cluster_models as model


def histories(n=240):
    from src import event_cluster_features as feature

    rng = np.random.default_rng(526)
    dates = pd.bdate_range("2010-01-04", periods=n)
    rows = [feature.cluster_summary(rng.integers(0, 2, 22).astype(float)) for _ in dates]
    return pd.DataFrame(rows, index=dates).loc[:, feature.FEATURES]


class StagedCluster(unittest.TestCase):
    def setUp(self):
        self.frame = histories()
        self.train, self.apply = self.frame.iloc[:220], self.frame.iloc[220:]
        self.y = pd.Series((np.arange(220) % 3 == 0).astype(float), index=self.train.index)
        self.eta = np.linspace(-1.5, 0.5, 220)
        self.query_eta = np.linspace(-2, 1, 20)
        self.base = expit(self.query_eta)

    def fit(self, train=None, apply=None):
        return model.fit_stages(
            self.train if train is None else train,
            self.y,
            self.apply if apply is None else apply,
            self.eta,
            self.query_eta,
            self.base,
        )

    def test_training_only_centering_fixed_scales_and_exact_constants(self):
        train = self.train.copy()
        train.iloc[:, 0] = 0.1
        x, q, z, qz, audit = model.transform(train, self.apply)
        np.testing.assert_array_equal(x[:, 1], np.zeros(len(train)))
        self.assertTrue(audit["nuisance_constant"][0])
        self.assertEqual(audit["scales"], [1.0] * 6)
        self.assertEqual(audit["memory_scale"], 1.0)
        changed = self.apply.iloc[::-1].copy()
        second = model.transform(train, changed)
        self.assertEqual(audit, second[-1])
        np.testing.assert_array_equal(x, second[0])
        np.testing.assert_array_equal(z, second[2])
        np.testing.assert_array_equal(q, second[1][::-1])
        np.testing.assert_array_equal(qz, second[3][::-1])

    def test_staged_optima_match_separate_equations(self):
        predictions, audit = self.fit()
        x, q, z, qz, _ = model.transform(self.train, self.apply)
        yy = self.y.to_numpy()

        def gradient(beta):
            penalty = beta.copy()
            penalty[0] = 0.0
            return x.T @ (expit(self.eta + x @ beta) - yy) / len(yy) + 0.02 * penalty

        solved = root(gradient, np.zeros(6), method="hybr", options={"xtol": 1e-10})
        self.assertTrue(solved.success)
        beta = np.array(audit["nuisance"]["beta"])
        np.testing.assert_allclose(beta, solved.x, rtol=1e-7, atol=1e-6)
        self.assertLessEqual(np.max(np.abs(gradient(beta))), 1e-8)
        offset = self.eta + x @ beta
        b = audit["cluster"]["coefficient"]
        g = np.mean(z * (expit(offset + b * z) - yy)) + 0.02 * b
        self.assertLessEqual(abs(g), 1e-8)
        np.testing.assert_array_equal(
            predictions["nuisance"], expit(self.query_eta + q @ beta)
        )
        np.testing.assert_array_equal(
            predictions["cluster"], expit(self.query_eta + q @ beta + b * qz)
        )
        json.dumps(audit, allow_nan=False)

    def test_application_changes_cannot_change_either_fit(self):
        _, first = self.fit()
        changed = self.apply.iloc[::-1].copy()
        _, second = self.fit(apply=changed)
        self.assertEqual(first, second)

    def test_constant_memory_retains_exact_parent_probabilities(self):
        train, apply = self.train.copy(), self.apply.copy()
        train.iloc[:, -1] = 0.125
        apply.iloc[:, -1] = 0.25
        predictions, audit = self.fit(train, apply)
        self.assertEqual(audit["cluster"]["coefficient"], 0.0)
        self.assertEqual(audit["cluster"]["status"], "EXACT_CONSTANT_INPUT")
        np.testing.assert_array_equal(predictions["cluster"], predictions["nuisance"])

    def test_scalar_analytic_stationarity_and_reflection(self):
        z = np.array([-0.2, 0.0, 0.4, 0.1])
        y = np.array([0.0, 1.0, 1.0, 0.0])
        eta = np.array([-2.0, 0.1, 1.0, -0.5])
        b, audit = model.fit_scalar(z, y, eta)
        reflected, _ = model.fit_scalar(z, 1 - y, -eta)
        self.assertAlmostEqual(b, -reflected, delta=2e-12)
        value, gradient = model.scalar_objective(b, z, y, eta)
        self.assertEqual(audit["objective"], value)
        self.assertEqual(audit["gradient"], gradient)
        self.assertLessEqual(abs(gradient), 1e-8)

    def test_extreme_logit_signed_residual_is_preserved(self):
        value, gradient = model.scalar_objective(
            0.0, np.array([1.0]), np.array([1.0]), np.array([40.0])
        )
        self.assertGreater(value, 0.0)
        self.assertEqual(gradient, -float(expit(-40.0)))
        with self.assertRaises(ValueError):
            model.scalar_objective(0.0, np.array([1.0]), np.array([1.0]), np.array([1000.0]))

    def test_scalar_rejects_false_success_and_no_fallback(self):
        z, y, eta = np.array([-0.1, 0.1]), np.array([0.0, 1.0]), np.zeros(2)
        fake = (20.0, SimpleNamespace(converged=True, iterations=1, function_calls=3))
        with patch.object(model, "brentq", return_value=fake) as solver:
            with self.assertRaises(ValueError):
                model.fit_scalar(z, y, eta)
            self.assertEqual(solver.call_count, 1)

    def test_failed_nuisance_prevents_scalar_fit(self):
        with (
            patch.object(model, "_newton", side_effect=ValueError("solver budget")),
            patch.object(model, "fit_scalar") as scalar,
        ):
            with self.assertRaises(ValueError):
                self.fit()
            scalar.assert_not_called()

    def test_schema_domains_and_training_labels_are_not_repaired(self):
        variants = [self.train.iloc[:, :-1], self.train.astype(str), self.train.copy()]
        variants[-1].iloc[0, 0] = np.nan
        for train in variants:
            with self.subTest(columns=len(train.columns)), self.assertRaises(ValueError):
                self.fit(train=train)
        changed = self.y.copy()
        changed.iloc[0] = 0.3
        with self.assertRaises(ValueError):
            model.fit_stages(
                self.train, changed, self.apply, self.eta, self.query_eta, self.base
            )
        with self.assertRaises(ValueError):
            model.fit_stages(
                self.train, self.y, self.apply, self.eta, self.query_eta, self.base + 0.1
            )

    def test_constant_nuisance_slope_canonical_zero(self):
        train = self.train.copy()
        train.iloc[:, 2] = 1.0
        _, audit = self.fit(train=train)
        self.assertEqual(audit["nuisance"]["beta"][3], 0.0)

    def test_no_adjacent_events_cannot_retune_varying_expectation(self):
        from src import event_cluster_features as feature

        rows = []
        for i in range(220):
            e = np.zeros(22)
            e[np.arange(1 + i % 9) * 2] = 1
            rows.append(feature.cluster_summary(e))
        full = pd.DataFrame(rows, index=self.train.index)
        self.assertGreater(full.expected_adjacency22.nunique(), 1)
        self.assertGreater(full.excess_adjacency22.nunique(), 1)
        train = full.loc[:, feature.FEATURES]
        self.assertTrue(train[feature.MEMORY].eq(0).all())
        predictions, audit = self.fit(train=train)
        self.assertEqual(audit["cluster"]["coefficient"], 0.0)
        np.testing.assert_array_equal(predictions["cluster"], predictions["nuisance"])

    def test_negative_raw_adjacency_is_invalid(self):
        apply = self.apply.copy()
        apply.iloc[0, -1] = -0.1
        with self.assertRaises(ValueError):
            self.fit(apply=apply)

    def test_zero_corrections_copy_parent_rounding_bits(self):
        from src import event_cluster_features as feature

        train = pd.DataFrame(0.0, index=self.train.index, columns=feature.FEATURES)
        apply = pd.DataFrame(0.0, index=self.apply.index, columns=feature.FEATURES)
        y = pd.Series((np.arange(len(train)) % 2).astype(float), index=train.index)
        parent = np.full(len(apply), 0.5 + 1e-13)
        predictions, audit = model.fit_stages(
            train, y, apply, np.zeros(len(train)), np.zeros(len(apply)), parent
        )
        self.assertEqual(audit["nuisance"]["beta"], [0.0] * 6)
        self.assertEqual(audit["cluster"]["coefficient"], 0.0)
        self.assertNotEqual(parent[0], 0.5)
        np.testing.assert_array_equal(predictions["nuisance"], parent)
        np.testing.assert_array_equal(predictions["cluster"], parent)

    def test_saved_baseline_geometry_is_replayed_without_fit(self):
        from src import range_alert_features as rf
        from src import range_alert_models as original
        from tests.test_range_alert_models import fixture

        f, t, _ = fixture()
        train, apply = f.iloc[30:1050], f.iloc[1050:1060]
        predictions, audit = original.fit_predict(train, t.loc[train.index, "y"], apply)
        eta, geometry = model.baseline_logits(apply, audit)
        expected, _, _, _, _ = rf.transform(train, apply)
        np.testing.assert_allclose(expit(eta), predictions["baseline"], rtol=1e-10, atol=1e-12)
        self.assertEqual(geometry.shape, (len(apply), 26))
        self.assertEqual(expected.shape[1], geometry.shape[1])
        bad = copy.deepcopy(audit)
        bad["baseline"]["scales"][1] = 0.0
        with self.assertRaises(ValueError):
            model.baseline_logits(apply, bad)


if __name__ == "__main__":
    unittest.main()
