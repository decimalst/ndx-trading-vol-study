"""Prewritten synthetic contracts for the model/memory estimator family.

These tests do not open market data. Reference estimates are reconstructed from
the objective equations rather than from the implementation's fit helpers.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_gamma_deviance
from sklearn.preprocessing import SplineTransformer

from src import model_memory_estimators as estimators

BASE = (
    "const", "lrv_d", "lrv_w", "lrv_m", "lev_d", "lev_w", "lev_m",
    "liv", "lvix", "term", "xasset_stress", "market_stress",
)
MODELS = {"baseline", "gamma", "adaptive_ols", "adaptive_gamma", "ridge_gamma", "spline_gamma"}
SPLINE_COLUMNS = ("lrv_d", "lrv_w", "lrv_m", "liv", "lvix", "term")


def fixture(n=320, m=9):
    rng = np.random.default_rng(908341)
    x = pd.DataFrame(rng.normal(size=(n, len(BASE))), columns=BASE)
    x["const"] = 1.0
    x["liv"] += 1.1 * x["lvix"]
    x["lrv_w"] += 0.5 * x["lrv_d"]
    x["lev_d"] *= 0.015
    x["market_stress"] += 2
    beta = np.array([-8, .15, .13, .09, -.2, -.03, -.02, .2, .14, .12, .04, .05])
    y = np.exp(x.to_numpy() @ beta + rng.normal(0, .4, n))
    query = x.iloc[:m].copy()
    query.index = pd.RangeIndex(n, n + m)
    return x, y, query, np.arange(n - 1, -1, -1, dtype=float)


class EstimatorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.y, cls.query, cls.ages = fixture()
        cls.fitted = estimators.fit_models(cls.x, cls.y, cls.query, cls.ages)

    def test_gamma_deviance_is_twice_qlike(self):
        y = np.array([.0001, .006, .031, .8])
        prediction = np.array([.0002, .01, .019, .7])
        weights = np.array([.2, 1, .7, .3])
        ratio = y / prediction
        expected = 2 * np.average(ratio - np.log(ratio) - 1, weights=weights)
        self.assertAlmostEqual(mean_gamma_deviance(y, prediction, sample_weight=weights), expected, places=13)

    def test_all_six_models_return_positive_finite_arrays_and_fit_metadata(self):
        self.assertEqual(set(self.fitted["predictions"]), MODELS)
        self.assertEqual(set(self.fitted["audit"]), MODELS)
        for model, pred in self.fitted["predictions"].items():
            self.assertEqual(np.asarray(pred).shape, (len(self.query),))
            self.assertTrue(np.isfinite(pred).all(), model)
            self.assertTrue((pred > 0).all(), model)
            audit = self.fitted["audit"][model]
            self.assertTrue(audit["converged"], model)
            self.assertEqual(audit["n_train"], len(self.x))
            self.assertLess(audit["gradient_inf_norm"], 2e-7, model)
            self.assertEqual(len(audit["coefficients"]), audit["n_features"])

    def test_age_weights_have_fixed_252_session_half_life(self):
        age = np.array([0., 126., 252., 504., 756.])
        np.testing.assert_allclose(estimators.age_weights(age), [1, np.sqrt(.5), .5, .25, .125], rtol=1e-14)
        for bad in ([-1, 0], [np.nan, 2], [np.inf], [[1, 2]]):
            with self.assertRaises(ValueError):
                estimators.age_weights(bad)

    def test_baseline_and_adaptive_ols_match_independent_exact_smearing(self):
        raw = self.x.to_numpy()
        for name, weights in (("baseline", np.ones(len(raw))),
                              ("adaptive_ols", np.exp2(-self.ages / 252))):
            weighted_x = raw * np.sqrt(weights)[:, None]
            weighted_y = np.log(self.y) * np.sqrt(weights)
            coefficients = np.linalg.lstsq(weighted_x, weighted_y, rcond=None)[0]
            residual = np.log(self.y) - raw @ coefficients
            smear = np.average(np.exp(residual), weights=weights)
            expected = np.exp(self.query.to_numpy() @ coefficients) * smear
            np.testing.assert_allclose(self.fitted["predictions"][name], expected, rtol=1e-11, atol=1e-14)
            self.assertAlmostEqual(self.fitted["audit"][name]["smearing_factor"], smear, places=12)

    def test_constant_variance_target_recovers_intercept_optimum(self):
        got = estimators.fit_models(self.x, np.full(len(self.x), .00027), self.query, self.ages)
        for name, prediction in got["predictions"].items():
            np.testing.assert_allclose(prediction, .00027, rtol=2e-7, atol=1e-12, err_msg=name)

    def test_scalers_and_estimates_cannot_use_apply_distribution(self):
        perturbed = self.query.copy()
        perturbed.iloc[1:, 1:] += 4
        got = estimators.fit_models(self.x, self.y, perturbed, self.ages)
        center = self.x.loc[:, BASE[1:]].mean().to_numpy()
        scale = self.x.loc[:, BASE[1:]].std(ddof=0).to_numpy()
        for name in MODELS:
            self.assertEqual(got["audit"][name], self.fitted["audit"][name], name)
            self.assertEqual(got["predictions"][name][0], self.fitted["predictions"][name][0], name)
            np.testing.assert_allclose(got["audit"][name]["input_center"], center, rtol=1e-14)
            np.testing.assert_allclose(got["audit"][name]["input_scale"], scale, rtol=1e-14)

    def test_gamma_score_equations_hold_with_intercept_and_optional_weights(self):
        got = estimators.fit_models(self.x, self.y, self.x, self.ages)
        z = (self.x.iloc[:, 1:].to_numpy() - self.x.iloc[:, 1:].mean().to_numpy())
        z /= self.x.iloc[:, 1:].std(ddof=0).to_numpy()
        for name, weights, alpha in (("gamma", np.ones(len(z)), 0.),
                                      ("adaptive_gamma", np.exp2(-self.ages / 252), 0.),
                                      ("ridge_gamma", np.ones(len(z)), .01)):
            score = weights * (1 - self.y / got["predictions"][name]) / weights.sum()
            beta = np.asarray(got["audit"][name]["coefficients"])
            gradient = np.r_[z.T @ score + alpha * beta, score.sum()]
            self.assertLess(np.max(np.abs(gradient)), 2e-7, name)
            audit = got["audit"][name]
            self.assertEqual(audit["alpha"], alpha)
            self.assertEqual(audit["max_iter"], 2000)
            self.assertEqual(audit["tol"], 1e-8)
            self.assertEqual(audit["solver"], "newton-cholesky")

    def test_linear_gamma_attains_exact_loglink_fixture_without_smearing(self):
        beta = np.array([-8, .13, .07, .09, -.4, .02, .01, .1, .11, .05, .06, .04])
        perfect_y = np.exp(self.x.to_numpy() @ beta)
        expected = np.exp(self.query.to_numpy() @ beta)
        got = estimators.fit_models(self.x, perfect_y, self.query, self.ages)
        for name in ("gamma", "adaptive_gamma"):
            np.testing.assert_allclose(got["predictions"][name], expected, rtol=5e-7, atol=1e-12)
            self.assertNotIn("smearing_factor", got["audit"][name])

    def test_spline_train_quantiles_and_train_basis_scaling_reconstruct_predictions(self):
        audit = self.fitted["audit"]["spline_gamma"]
        self.assertEqual(tuple(audit["spline_columns"]), SPLINE_COLUMNS)
        self.assertEqual(audit["alpha"], self.fitted["audit"]["ridge_gamma"]["alpha"])
        center = self.x.iloc[:, 1:].mean()
        scale = self.x.iloc[:, 1:].std(ddof=0)
        train = (self.x.iloc[:, 1:] - center) / scale
        query = (self.query.iloc[:, 1:] - center) / scale
        transformer = SplineTransformer(n_knots=4, degree=3, knots="quantile",
                                        include_bias=False, extrapolation="linear")
        spline_train = transformer.fit_transform(train.loc[:, SPLINE_COLUMNS])
        spline_query = transformer.transform(query.loc[:, SPLINE_COLUMNS])
        raw_columns = [c for c in BASE[1:] if c not in SPLINE_COLUMNS]
        design_train = np.column_stack([spline_train, train[raw_columns]])
        design_query = np.column_stack([spline_query, query[raw_columns]])
        expected_knots = np.column_stack([spline.t for spline in transformer.bsplines_])
        np.testing.assert_allclose(audit["spline_knots"], expected_knots, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(audit["design_center"], design_train.mean(axis=0), rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(audit["design_scale"], design_train.std(axis=0), rtol=1e-13, atol=1e-13)
        standardized = (design_query - design_train.mean(axis=0)) / design_train.std(axis=0)
        expected = np.exp(standardized @ np.asarray(audit["coefficients"]) + audit["intercept"])
        np.testing.assert_allclose(self.fitted["predictions"]["spline_gamma"], expected, rtol=1e-11)

    def test_zero_scale_nonconstant_features_raise_without_dropping(self):
        broken = self.x.copy()
        broken["lev_w"] = 0.0
        with self.assertRaisesRegex(ValueError, "zero.scale"):
            estimators.fit_models(broken, self.y, self.query, self.ages)

    def test_invalid_inputs_and_misaligned_columns_raise(self):
        for y in (np.zeros(len(self.x)), -self.y, np.full(len(self.x), np.nan), self.y[:-1]):
            with self.assertRaises(ValueError):
                estimators.fit_models(self.x, y, self.query, self.ages)
        for ages in (self.ages[:-1], -self.ages, np.full(len(self.x), np.nan)):
            with self.assertRaises(ValueError):
                estimators.fit_models(self.x, self.y, self.query, ages)
        for x, query in ((self.x.drop(columns="const"), self.query),
                          (self.x.assign(const=2), self.query),
                          (self.x, self.query.loc[:, list(reversed(BASE))]),
                          (self.x.assign(lrv_d=np.inf), self.query)):
            with self.assertRaises(ValueError):
                estimators.fit_models(x, self.y, query, self.ages)

    def test_convergence_warnings_fail_loudly(self):
        def fail_fit(*args, **kwargs):
            import warnings
            warnings.warn("Synthetic failed optimizer", ConvergenceWarning, stacklevel=2)
        with (patch.object(estimators.GammaRegressor, "fit", side_effect=fail_fit),
              self.assertRaises((RuntimeError, ConvergenceWarning))):
            estimators.fit_models(self.x, self.y, self.query, self.ages)


if __name__ == "__main__":
    unittest.main()
