"""Prewritten generated independent shape/calibration verification contracts."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy.optimize import minimize
from scipy.stats import norm, t

from src.verify_copula_shape import (
    independent_inverse_shape,
    independent_shape_coordinates,
    verify,
    verify_parameters,
)
from tests.test_verify_copula_calibration import synthetic_archive


class IndependentShapeMathTests(unittest.TestCase):
    def test_normalized_jacobian_and_independent_extreme_tail_inverse(self):
        p = {
            "location": [0.1, -0.2],
            "scale": [1.4, 1.3],
            "epsilon": [0.75, -0.75],
            "delta": [0.75, 0.75],
        }
        latent = np.array([[-4.8, 4.8], [4.8, -4.8], [0.2, -0.3]])
        z = independent_inverse_shape(latent, p)
        self.assertGreater(abs(z[1, 0]), 1e20)
        got = independent_shape_coordinates(z, np.ones_like(z), p)
        assert_allclose(got["normal"], latent, atol=2e-9, rtol=2e-10)
        z = np.array([[-1.2, 0.4], [0.2, 1.1]])
        h = np.array([[0.8, 0.4], [1.2, 0.7]])
        w = norm.ppf(t.cdf(z, 8))
        x = (w - p["location"]) / p["scale"]
        u = np.asarray(p["delta"]) * np.arcsinh(x) - p["epsilon"]
        v = np.sinh(u)
        expected = t.logpdf(z, 8) - 0.5 * np.log(0.75 * h) + norm.logpdf(v) - norm.logpdf(w)
        expected += (
            np.log(p["delta"])
            - np.log(p["scale"])
            + np.log(np.cosh(u))
            - 0.5 * np.log1p(x * x)
        )
        assert_allclose(
            independent_shape_coordinates(z, h, p)["log_marginal"], expected, atol=1e-10
        )

    def test_convex_certificate_matches_independent_slsqp_optimum(self):
        from src.copula_shape import fit_shape

        w = np.random.default_rng(3118).normal(size=(87, 2)) * [0.8, 1.2] + [0.2, -0.1]
        z = t.ppf(norm.cdf(w), 8)
        parameters = fit_shape(z)
        proof = verify_parameters(z, parameters)
        self.assertEqual(proof["status"], "VERIFIED")
        self.assertLessEqual(proof["maximum_box_gap"], 1e-7)
        self.assertLessEqual(proof["maximum_projected_gradient"], 1e-7)
        for j, audit in enumerate(parameters["optimizer_audits"]):
            x = (w[:, j] - parameters["location"][j]) / parameters["scale"][j]

            def objective(theta, x=x):
                u = theta[1] * np.arcsinh(x) - theta[0]
                return float(
                    np.mean(0.5 * np.sinh(u) ** 2 - np.log(np.cosh(u))) - np.log(theta[1])
                )

            result = minimize(
                objective,
                [0.0, 1.0],
                method="SLSQP",
                bounds=[(-0.75, 0.75), (0.75, 2.0)],
                options={"ftol": 1e-13, "maxiter": 500},
            )
            self.assertTrue(result.success)
            self.assertAlmostEqual(audit["objective"], result.fun, delta=1e-8)

    def test_affine_count_certificate_and_parameter_tampering_rejected(self):
        from src.copula_shape import fit_shape

        z = np.random.default_rng(712).standard_t(8, size=(51, 2))
        valid = fit_shape(z)
        mutations = [
            lambda x: x["location"].__setitem__(0, x["location"][0] + 0.01),
            lambda x: x.__setitem__("train_n", 52),
            lambda x: x["epsilon"].__setitem__(0, 0.8),
            lambda x: x["optimizer_audits"][0]["gradient"].__setitem__(0, 0.1),
            lambda x: x["optimizer_audits"][0]["hessian"][0].__setitem__(0, 0.0),
            lambda x: x["optimizer_audits"][0].__setitem__("lower_bound", -99.0),
        ]
        for mutation in mutations:
            altered = copy.deepcopy(valid)
            mutation(altered)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify_parameters(z, altered)

    def test_self_consistent_nonoptimal_audit_is_not_a_certificate(self):
        from src.copula_shape import fit_shape, objective

        z = np.random.default_rng(7261).standard_t(8, size=(79, 2))
        p = fit_shape(z)
        w = norm.ppf(t.cdf(z, 8))
        p["epsilon"][0], p["delta"][0] = 0.65, 0.8
        theta = np.array([0.65, 0.8])
        value, gradient, hessian = objective(
            theta, (w[:, 0] - p["location"][0]) / p["scale"][0]
        )
        gap = float(gradient @ (theta - np.where(gradient >= 0, [-0.75, 0.75], [0.75, 2.0])))
        p["optimizer_audits"][0] = {
            "theta": theta.tolist(),
            "objective": value,
            "gradient": gradient.tolist(),
            "hessian": hessian.tolist(),
            "box_gap": gap,
            "lower_bound": value - gap,
            "bounds": [[-0.75, 0.75], [0.75, 2.0]],
        }
        with self.assertRaises(ValueError):
            verify_parameters(z, p)


class IndependentShapeTimelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.copula_calibration import run_crossed
        from src.copula_shape import run_shape

        cls.archive, cls.calendar = synthetic_archive()
        cls.base = run_crossed(cls.archive, minimum_train=12)
        cls.produced = run_shape(
            cls.archive, cls.base["applications"], cls.base["panel"], minimum_train=12
        )

    def check(self, produced=None, archive=None):
        return verify(
            self.archive if archive is None else archive,
            self.base["applications"],
            self.base["panel"],
            self.produced if produced is None else produced,
            calendar=self.calendar,
            minimum_train=12,
        )

    def test_complete_same_dates_unscored_last_month_and_serialization(self):
        proof = self.check()
        self.assertEqual(proof["status"], "VERIFIED")
        self.assertEqual(proof["applications_verified"], len(self.base["applications"]))
        self.assertEqual(proof["scored_origins"], len(self.base["panel"]))
        self.assertEqual(
            proof["monthly_fits_verified"], self.base["applications"].fit_origin.nunique()
        )
        self.assertEqual(proof["contrasts_verified"], 6)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            changed = {}
            for name in ("applications", "panel"):
                self.produced[name].to_parquet(root / f"{name}.parquet", index=False)
                changed[name] = pd.read_parquet(root / f"{name}.parquet")
            (root / "fits.json").write_text(json.dumps(self.produced["fits"], allow_nan=False))
            changed["fits"] = json.loads((root / "fits.json").read_text())
            self.assertEqual(self.check(changed)["status"], "VERIFIED")

    def test_every_unscored_forecast_and_entire_month_are_required(self):
        for key in ("applications", "panel", "fits"):
            changed = copy.deepcopy(self.produced)
            if key == "fits":
                changed[key] = changed[key][:-1]
            else:
                changed[key] = changed[key].iloc[:-1].copy()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check(changed)
        changed = copy.deepcopy(self.produced)
        changed["applications"].loc[changed["applications"].index[-1], "rho_shape_t8"] += 0.01
        with self.assertRaises(ValueError):
            self.check(changed)

    def test_mature_issued_history_full_calendar_and_future_source_guard(self):
        changed = copy.deepcopy(self.produced)
        changed["fits"][0]["train_origins"][-1] = changed["fits"][0]["fit_origin"]
        with self.assertRaises(ValueError):
            self.check(changed)
        changed = self.archive.copy(deep=True)
        first = self.produced["fits"][0]
        donor = changed.index[changed.origin == pd.Timestamp(first["train_origins"][-1])][0]
        changed.loc[donor, "available_date"] = pd.Timestamp(first["fit_origin"])
        with self.assertRaises(ValueError):
            self.check(archive=changed)
        changed = self.archive.copy(deep=True)
        changed.loc[changed.index[-1], "origin"] = pd.Timestamp("2025-10-21")
        with self.assertRaises(ValueError):
            self.check(archive=changed)

    def test_baseline_values_cannot_be_changed_in_any_inherited_column(self):
        for key, column in (
            ("applications", "mu_qqq"),
            ("applications", "rho_cal_t8"),
            ("panel", "marginal_calibrated_qqq"),
            ("panel", "loss_orig_gaussian"),
        ):
            changed = copy.deepcopy(self.produced)
            changed[key].loc[0, column] += 1e-12
            with self.subTest(key=key, column=column), self.assertRaises(ValueError):
                self.check(changed)

    def test_density_pit_and_signed_interaction_tampering_rejected(self):
        for column in (
            "marginal_shape_qqq",
            "pit_shape_spx",
            "normal_shape_qqq",
            "loss_shape_gaussian",
            "d_qqq_shape",
            "d_gaussian_shape",
            "d_shape_interaction",
        ):
            changed = copy.deepcopy(self.produced)
            changed["panel"].loc[0, column] += 0.01
            with self.subTest(column=column), self.assertRaises(ValueError):
                self.check(changed)

    def test_dependence_certificate_and_unknown_schema_tampering_rejected(self):
        changed = copy.deepcopy(self.produced)
        changed["fits"][0]["dependence"]["t8"]["audit"]["objective"] += 0.1
        with self.assertRaises(ValueError):
            self.check(changed)
        changed = copy.deepcopy(self.produced)
        changed["panel"]["extra"] = 0.0
        with self.assertRaises(ValueError):
            self.check(changed)

    def test_inputs_not_mutated(self):
        archive = self.archive.copy(deep=True)
        apps = self.produced["applications"].copy(deep=True)
        fits = copy.deepcopy(self.produced["fits"])
        self.check()
        pd.testing.assert_frame_equal(archive, self.archive)
        pd.testing.assert_frame_equal(apps, self.produced["applications"])
        self.assertEqual(fits, self.produced["fits"])


if __name__ == "__main__":
    unittest.main()
