"""Generated independent copula and forecast contracts, written before producer use."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy.stats import multivariate_t, norm, t

from src import verify_joint_copula_forecasts as verifier


class IndependentCopulaMathTests(unittest.TestCase):
    def test_saved_forecast_absolute_tolerance_is_tight_in_variance_units(self):
        expected = pd.DataFrame({"h_qqq": [1e-7]})
        changed = pd.DataFrame({"h_qqq": [1e-7 + 2e-11]})
        with self.assertRaises(AssertionError):
            verifier._table(changed, expected, "applications")

    def test_saved_forecasts_retain_strict_domains_even_within_roundoff_tolerance(self):
        for name, actual, reference in (
            ("h_qqq", -1e-15, 1e-15),
            ("rho", 0.995 + 1e-10, 0.995),
        ):
            with self.subTest(name=name), self.assertRaises((ValueError, AssertionError)):
                verifier._table(
                    pd.DataFrame({name: [actual]}),
                    pd.DataFrame({name: [reference]}),
                    "applications",
                )

    def test_t8_density_matches_separate_multivariate_and_marginal_formula(self):
        z = np.array([[0.0, 0.0], [1.1, -0.7], [-4.0, 3.2], [9.0, 9.1]])
        for rho in (-0.995, -0.3, 0.0, 0.83, 0.995):
            expected = multivariate_t.logpdf(z, shape=[[1, rho], [rho, 1]], df=8)
            expected -= t.logpdf(z, 8).sum(axis=1)
            assert_allclose(
                verifier.independent_log_copula(z, rho, "t8"), expected, atol=2e-12, rtol=2e-12
            )

    def test_gaussian_density_and_zero_parameter_are_exact_independence(self):
        z = np.array([[0, 0], [1.1, -0.7], [-4, 3.2], [9, 9.1]], dtype=float)
        w = norm.ppf(t.cdf(z, 8))
        for rho in (-0.995, -0.3, 0.0, 0.83, 0.995):
            denom = 1 - rho**2
            expected = (
                -0.5 * np.log(denom)
                + (rho * w[:, 0] * w[:, 1] - rho**2 * np.sum(w**2, axis=1) / 2) / denom
            )
            assert_allclose(
                verifier.independent_log_copula(z, rho, "gaussian"),
                expected,
                atol=3e-10,
                rtol=2e-11,
            )
        np.testing.assert_array_equal(
            verifier.independent_log_copula(z, 0, "gaussian"), np.zeros(len(z))
        )
        self.assertNotEqual(float(verifier.independent_log_copula(z[:1], 0, "t8")[0]), 0)

    def test_tail_signs_and_extreme_finite_values_do_not_clip_cdf(self):
        z = np.array([[1e200, 1e200], [1e200, -1e200], [0, 1e-300], [-1e150, 3]])
        before = z.copy()
        for family in ("t8", "gaussian"):
            result = verifier.independent_log_copula(z, 0.8, family)
            self.assertTrue(np.isfinite(result).all())
            self.assertGreater(result[0], result[1])
            assert_allclose(
                result,
                verifier.independent_log_copula(-z, 0.8, family),
                atol=1e-10,
                rtol=1e-12,
            )
        np.testing.assert_array_equal(z, before)

    def test_copula_preserves_marginal_by_independent_quadrature(self):
        # Integrate c(u,v) dv using t8 quantiles; no bivariate CDF or producer call.
        nodes, weights = np.polynomial.legendre.leggauss(600)
        v, weights = (nodes + 1) / 2, weights / 2
        for family in ("t8", "gaussian"):
            for u in (0.2, 0.5, 0.8):
                z = np.column_stack([np.full(len(v), t.ppf(u, 8)), t.ppf(v, 8)])
                integral = weights @ np.exp(verifier.independent_log_copula(z, 0.55, family))
                self.assertAlmostEqual(integral, 1.0, delta=3e-5)

    def test_domain_shape_and_real_numeric_validation(self):
        valid = np.array([[1.0, 2.0], [0.0, -1.0]])
        cases = [
            ([], 0, "t8"),
            ([[1, 2, 3]], 0, "t8"),
            ([[1, np.nan]], 0, "t8"),
            ([[1, np.inf]], 0, "t8"),
            (valid, 0.996, "t8"),
            (valid, True, "t8"),
            (valid, np.nan, "gaussian"),
            (valid, 0, "student"),
            (valid.astype(str), 0, "t8"),
        ]
        for args in cases:
            with self.subTest(args=args), self.assertRaises((TypeError, ValueError)):
                verifier.independent_log_copula(*args)

    def test_gaussian_stationary_candidates_include_all_three_roots_and_endpoints(self):
        a = np.sqrt(0.05)
        w = np.array([[a, a], [a, -a], [-a, a], [-a, -a]])
        z = t.ppf(norm.cdf(w), 8)
        candidates = verifier._gaussian_candidates(z)
        assert_allclose(
            candidates, [-0.995, -np.sqrt(0.9), 0, np.sqrt(0.9), 0.995], atol=1e-10, rtol=1e-10
        )
        values = [
            -np.mean(verifier.independent_log_copula(z, r, "gaussian")) for r in candidates
        ]
        self.assertLess(values[1], values[2])
        self.assertAlmostEqual(values[1], values[3], places=12)

    def test_t8_objective_gradient_and_curvature_match_independent_differences(self):
        z = np.random.default_rng(112).standard_t(8, size=(31, 2))
        for rho in (-0.8, -0.1, 0.6):
            value, gradient, hessian = verifier._t8_reduced(z, rho)
            step = 1e-5

            def objective(r):
                aa = 8 * (1 - r * r) + (z * z).sum(axis=1) - 2 * r * z[:, 0] * z[:, 1]
                return 5 * np.log(aa).mean() - 4.5 * np.log1p(-r * r)

            self.assertAlmostEqual(value, objective(rho), places=12)
            self.assertAlmostEqual(
                gradient,
                (objective(rho + step) - objective(rho - step)) / (2 * step),
                delta=1e-7,
            )
            self.assertAlmostEqual(
                hessian,
                (objective(rho + step) - 2 * value + objective(rho - step)) / step**2,
                delta=1e-4,
            )

    def test_interval_lower_bound_covers_full_interval_including_negative_curvature(self):
        rng = np.random.default_rng(201)
        cases = [
            rng.standard_t(8, size=(29, 2)),
            np.zeros((7, 2)),
            np.array([[15, 15], [15, -15], [0.1, 0.1], [-0.1, 0.1]]),
        ]
        for z in cases:
            original = copy.deepcopy(z)
            for low, high in [(-0.995, 0.995), (-0.8, -0.1), (-0.03, 0.07), (0.91, 0.995)]:
                bound = verifier._t8_lower_bound(z, low, high)
                grid = [verifier._t8_reduced(z, r)[0] for r in np.linspace(low, high, 71)]
                self.assertLessEqual(bound, min(grid) + 1e-10)
            np.testing.assert_array_equal(z, original)


class DependenceCertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A missing checker must be an honest RED before importing a producer.
        cls.check = staticmethod(verifier.verify_dependence)
        from src.joint_copula_density import fit_dependence

        rng = np.random.default_rng(50427)
        cls.z = rng.standard_t(8, (73, 2))
        cls.z[:, 1] += 0.5 * cls.z[:, 0]
        cls.fits = {family: fit_dependence(cls.z, family) for family in ("t8", "gaussian")}

    def test_both_saved_certificates_independently_reconstruct(self):
        before = copy.deepcopy(self.fits)
        for family, (rho, audit) in self.fits.items():
            got = self.check(self.z, family, rho, audit)
            self.assertEqual(got["status"], "VERIFIED")
            self.assertLessEqual(got["global_value_gap"], 1e-8)
        self.assertEqual(self.fits, before)

    def test_wrong_objective_kkt_rho_training_and_unknown_schema_rejected(self):
        for family, (rho, original) in self.fits.items():
            changes = [
                ("objective", original["objective"] + 0.01),
                ("gradient", original["gradient"] + 0.01),
                ("rho", rho + 0.01),
                ("train_n", len(self.z) + 1),
                ("projected_gradient", 0.01),
                ("unexpected", 1),
            ]
            for key, value in changes:
                changed = copy.deepcopy(original)
                changed[key] = value
                with (
                    self.subTest(family=family, key=key),
                    self.assertRaises((ValueError, AssertionError)),
                ):
                    self.check(self.z, family, rho, changed)

    def test_interval_missing_overlap_inflated_bound_and_global_gap_rejected(self):
        rho, original = self.fits["t8"]
        for kind in ("missing", "overlap", "bound", "gradient", "gap"):
            changed = copy.deepcopy(original)
            leaves = changed["certificate"]["leaves"]
            if kind == "missing":
                leaves.pop(0)
            if kind == "overlap":
                leaves.append(copy.deepcopy(leaves[0]))
            if kind == "bound":
                leaves[0]["lower_bound"] += 0.1
            if kind == "gradient":
                leaves[0]["gradient"] += 0.1
            if kind == "gap":
                changed["certificate"]["gap"] = 0.01
            with self.subTest(kind=kind), self.assertRaises((ValueError, AssertionError)):
                self.check(self.z, "t8", rho, changed)

    def test_gaussian_omitted_stationary_candidate_and_root_corruption_rejected(self):
        rho, original = self.fits["gaussian"]
        changed = copy.deepcopy(original)
        changed["certificate"]["candidates"].pop()
        with self.assertRaises((ValueError, AssertionError)):
            self.check(self.z, "gaussian", rho, changed)
        changed = copy.deepcopy(original)
        changed["certificate"]["polynomial_roots"][0][0] += 0.1
        with self.assertRaises((ValueError, AssertionError)):
            self.check(self.z, "gaussian", rho, changed)


def generated_sources():
    rng = np.random.default_rng(50428)
    dates = pd.bdate_range("2014-01-02", periods=265, name="date")
    sources = []
    for scale in (0.009, 0.006):
        close = 100 * np.exp(rng.normal(0, scale, len(dates)).cumsum())
        opening = close * np.exp(rng.normal(0, scale / 2, len(dates)))
        sources.append(
            pd.DataFrame(
                {
                    "open": opening,
                    "close": close,
                    "high": np.maximum(opening, close) * 1.005,
                    "low": np.minimum(opening, close) / 1.005,
                },
                index=dates,
            )
        )
    iv = pd.DataFrame(
        {
            name: np.exp(rng.normal(3, 0.2, len(dates)))
            for name in ("vxn", "vix", "vix9d", "vvix")
        },
        index=dates,
    )
    sources[0] = sources[0].drop(dates[80])
    sources[0].loc["2014-12-02", "open"] = np.nan  # unknown first query's outcome
    config = {
        "origin_start": "2014-12-01",
        "origin_end": str(dates[-1].date()),
        "source_end": str(dates[-1].date()),
        "development": ["2014-12-01", "2014-12-31"],
        "evaluation": ["2015-01-01", str(dates[-1].date())],
        "minimum_train": 100,
    }
    return *sources, iv, config


class IssuedForecastVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verify = staticmethod(verifier.verify_forecasts)
        from src.joint_copula_pipeline import produce

        cls.q, cls.s, cls.iv, cls.config = generated_sources()
        cls.produced = produce(cls.q, cls.s, cls.iv, cls.config)

    def test_generated_all_issued_and_scored_forecasts_reconstruct(self):
        got = self.verify(self.q, self.s, self.iv, self.produced, self.config)
        self.assertEqual(got["status"], "VERIFIED")
        self.assertEqual(got["calendar_rows"], len(self.s))
        self.assertEqual(
            got["application_forecasts_verified"], len(self.produced["applications"])
        )
        self.assertEqual(got["forecasts_verified"], len(self.produced["panel"]))
        self.assertEqual(got["monthly_fits"], len(self.produced["fits"]))
        self.assertGreater(len(self.produced["applications"]), len(self.produced["panel"]))
        self.assertEqual(self.produced["fits"][0]["fit_origin"], "2014-12-01")
        self.assertTrue(
            self.produced["targets"].loc["2014-12-01", ["y_qqq", "y_spx"]].isna().any()
        )

    def test_serialized_six_parquet_and_fits_json_reconstruct_same_forecasts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reloaded = {}
            for name in (
                "features",
                "targets",
                "applications",
                "panel",
                "coverage",
                "schedules",
            ):
                self.produced[name].to_parquet(root / (name + ".parquet"))
                reloaded[name] = pd.read_parquet(root / (name + ".parquet"))
            path = root / "fits.json"
            path.write_text(json.dumps(self.produced["fits"], allow_nan=False))
            reloaded["fits"] = json.loads(path.read_text())
        got = self.verify(self.q, self.s, self.iv, reloaded, self.config)
        self.assertEqual(got["status"], "VERIFIED")
        self.assertEqual(got["forecasts_verified"], len(self.produced["panel"]))

    def test_exact_shared_marginals_and_independence_placeholder_on_unscored_origins(self):
        for name, value in (("h_qqq", None), ("rho", 1e-13)):
            altered = copy.deepcopy(self.produced)
            last = altered["applications"].index[-1]
            if name == "h_qqq":
                value = altered["applications"].loc[last, name] + 1e-14
            altered["applications"].loc[last, name] = value
            with self.subTest(name=name), self.assertRaises((ValueError, AssertionError)):
                self.verify(self.q, self.s, self.iv, altered, self.config)

    def test_unknown_query_and_unscored_forecast_cannot_be_dropped_or_tampered(self):
        for kind in ("drop", "prediction", "missing_arm"):
            altered = copy.deepcopy(self.produced)
            apps = altered["applications"]
            if kind == "drop":
                altered["applications"] = apps.iloc[:-3].copy()
            if kind == "prediction":
                apps.loc[apps.index[-1], "mu_qqq"] += 0.001
            if kind == "missing_arm":
                altered["applications"] = apps[apps.model != "independence"].copy()
            with self.subTest(kind=kind), self.assertRaises((ValueError, AssertionError)):
                self.verify(self.q, self.s, self.iv, altered, self.config)

    def test_feature_target_calendar_and_cutoff_corruption_rejected(self):
        for kind in ("feature", "target", "calendar", "cutoff", "coverage"):
            altered = copy.deepcopy(self.produced)
            if kind == "feature":
                altered["features"].iloc[200, 2] += 0.01
            if kind == "target":
                altered["targets"].iloc[200, 0] += 0.01
            if kind == "calendar":
                altered["features"] = altered["features"].iloc[1:].copy()
            if kind == "cutoff":
                altered["features"].loc[
                    altered["features"].index[200], "feature_cutoff_date"
                ] = self.s.index[200]
            if kind == "coverage":
                altered["coverage"] = altered["coverage"].iloc[:-1].copy()
            with self.subTest(kind=kind), self.assertRaises((ValueError, AssertionError)):
                self.verify(self.q, self.s, self.iv, altered, self.config)

    def test_fit_training_membership_maturity_and_schedule_are_reconstructed(self):
        for kind in ("future_train", "cutoff", "query", "n"):
            altered = copy.deepcopy(self.produced)
            fit = altered["fits"][0]
            if kind == "future_train":
                fit["train_origins"].append(fit["fit_origin"])
            if kind == "cutoff":
                fit["training_cutoff"] = fit["fit_origin"]
            if kind == "query":
                fit["application_origins"].pop(0)
            if kind == "n":
                fit["train_n"] += 1
            with self.subTest(kind=kind), self.assertRaises((ValueError, AssertionError)):
                self.verify(self.q, self.s, self.iv, altered, self.config)

    def test_shared_marginals_and_dependence_audits_cannot_be_changed(self):
        for kind in ("mean", "scale", "rho", "objective"):
            altered = copy.deepcopy(self.produced)
            audit = altered["fits"][0]["model_audit"]
            if kind == "mean":
                audit["moments"]["qqq"]["mean"]["beta"][0] += 0.001
            if kind == "scale":
                altered["applications"].loc[0, "h_qqq"] *= 1.01
            if kind == "rho":
                altered["applications"].loc[0, "rho"] += 0.01
            if kind == "objective":
                audit["dependence"]["t8_copula"]["audit"]["objective"] += 0.01
            with self.subTest(kind=kind), self.assertRaises((ValueError, AssertionError)):
                self.verify(self.q, self.s, self.iv, altered, self.config)

    def test_phase_maturity_and_original_calendar_offsets_cannot_be_relabelled(self):
        for kind in ("offset", "phase", "target_end"):
            altered = copy.deepcopy(self.produced)
            panel = altered["panel"]
            if kind == "offset":
                panel.loc[0, "offset"] = (int(panel.loc[0, "offset"]) + 1) % 5
            if kind == "phase":
                panel.loc[0, "phase"] = "evaluation"
            if kind == "target_end":
                panel.loc[0, "target_end"] = panel.loc[0, "origin"]
            with self.subTest(kind=kind), self.assertRaises((ValueError, AssertionError)):
                self.verify(self.q, self.s, self.iv, altered, self.config)

    def test_future_source_calendar_is_rejected_before_numeric_data(self):
        altered = self.iv.astype(object)
        altered.iloc[-1] = "NEVER_CONVERT_AFTER_CEILING"
        altered.index = altered.index[:-1].append(pd.DatetimeIndex(["2025-10-21"]))
        with self.assertRaisesRegex(ValueError, "date|calendar|ceiling"):
            self.verify(self.q, self.s, altered, self.produced, self.config)

    def test_insufficient_rows_and_measurement_failure_are_not_verified(self):
        config = copy.deepcopy(self.config)
        config["minimum_train"] = 1000
        with self.assertRaisesRegex((ValueError, AssertionError), "INSUFFICIENT_DATA"):
            self.verify(self.q, self.s, self.iv, self.produced, config)
        altered = self.s.copy()
        altered.iloc[0] = 100.0
        with self.assertRaisesRegex((ValueError, AssertionError), "MEASUREMENT|measurement"):
            self.verify(self.q, altered, self.iv, self.produced, self.config)


if __name__ == "__main__":
    unittest.main()
