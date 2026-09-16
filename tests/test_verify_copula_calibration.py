"""Prewritten generated checks for the crossed marginal/coplanar calibration study."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import mpmath as mp
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy.special import ndtr
from scipy.stats import multivariate_normal, multivariate_t, norm, t

from src.verify_copula_calibration import (
    _calibration,
    _from_normal,
    _row_scores,
    verify,
)


def synthetic_archive():
    calendar = pd.bdate_range("2015-12-31", "2016-05-03").as_unit("ns")
    origins = calendar[1:-1]
    n = len(origins)
    rng = np.random.default_rng(982)
    w = rng.normal(size=(n, 2))
    w[:, 1] = 0.55 * w[:, 0] + np.sqrt(1 - 0.55**2) * w[:, 1]
    w = w * [0.88, 1.17] + [0.13, -0.11]
    z = t.ppf(ndtr(w), 8)
    mu = np.column_stack([0.001 * np.sin(np.arange(n)), 0.002 * np.cos(np.arange(n))])
    h = np.column_stack([0.0002 + np.arange(n) * 1e-7, 0.0004 + np.arange(n) * 2e-7])
    y = mu + np.sqrt(0.75 * h) * z
    first = {
        month: origins[origins.to_period("M") == month][0]
        for month in origins.to_period("M").unique()
    }
    fits = pd.DatetimeIndex([first[d.to_period("M")] for d in origins])
    frame = pd.DataFrame(
        {
            "origin": origins,
            "available_date": calendar[2:],
            "target_end": calendar[2:],
            "issued": True,
            "phase": "development",
            "offset": np.arange(1, n + 1) % 5,
            "fit_origin": fits,
            "training_cutoff": calendar[calendar.get_indexer(fits) - 1],
            "mu_qqq": mu[:, 0],
            "mu_spx": mu[:, 1],
            "h_qqq": h[:, 0],
            "h_spx": h[:, 1],
            "y_qqq": y[:, 0],
            "y_spx": y[:, 1],
            "eligible_scored": True,
            "old_rho_t8": 0.42,
            "old_rho_gaussian": 0.38,
        }
    )
    frame.loc[frame.index[-1], ["y_qqq", "y_spx"]] = np.nan
    frame.loc[frame.index[-1], "eligible_scored"] = False
    return frame, calendar


class IndependentCalibrationMathTests(unittest.TestCase):
    def test_population_normal_score_parameters_and_identity(self):
        w = np.array([[-1.2, 0.4], [0.7, 1.1], [1.4, -0.8], [0.2, -0.3]])
        z = t.ppf(norm.cdf(w), 8)
        result = _calibration(z)
        assert_allclose(result["location"], w.mean(axis=0), atol=1e-10, rtol=1e-10)
        assert_allclose(result["scale"], w.std(axis=0, ddof=0), atol=1e-10, rtol=1e-10)
        self.assertEqual(result["train_n"], 4)
        self.assertFalse(np.allclose(result["scale"], w.std(axis=0, ddof=1)))

    def test_constant_scale_invalid_types_and_unrepresentable_tail_fail(self):
        for bad in (np.ones((4, 2)), [[0, np.inf], [1, 2]], [[False, True], [True, False]]):
            with self.subTest(bad=bad), self.assertRaises((ValueError, AssertionError)):
                _calibration(bad)
        with self.assertRaises((ValueError, AssertionError)):
            _from_normal(np.array([[110.0, 0.0]]))
        for scale in ([0, 1], [1e-12, 1], [-1, 1], [np.nan, 1]):
            with self.subTest(scale=scale), self.assertRaises((ValueError, AssertionError)):
                _row_scores(np.array([[0.3, 0.5]]), np.ones((1, 2)), [0, 0], scale, [0.3] * 4)

    def test_inverse_transform_uses_log_tails_without_clipping(self):
        w = np.array([[-50.0, 30.0], [-8.0, 8.0], [0.0, 0.3]])
        actual = _from_normal(w)
        self.assertTrue(np.isfinite(actual).all())
        self.assertEqual(actual[2, 0], 0.0)
        expected = np.sign(w[1:]) * t.isf(norm.sf(abs(w[1:])), 8)
        assert_allclose(actual[1:], expected, atol=2e-5, rtol=2e-5)
        with mp.workdps(100):
            for v, observed in zip(w[0], actual[0], strict=True):
                probability = mp.erfc(abs(mp.mpf(str(v))) / mp.sqrt(2)) / 2
                x = abs(mp.mpf(str(observed)))
                log_tail = mp.log(
                    mp.betainc(4, mp.mpf(".5"), 0, 8 / (8 + x * x), regularized=True) / 2
                )
                self.assertAlmostEqual(float(log_tail - mp.log(probability)), 0.0, delta=2e-9)

    def test_full_jacobian_joint_densities_and_all_seven_contrasts(self):
        z = np.array([[0.2, -0.8], [1.4, 2.0], [-3.0, -2.3]])
        h = np.array([[0.02, 0.03], [0.01, 0.04], [0.04, 0.07]])
        a, b = np.array([0.2, -0.15]), np.array([0.75, 1.1])
        rhos = [0.43, 0.47, 0.51, 0.57]
        result = _row_scores(z, h, a, b, rhos)
        w = norm.ppf(t.cdf(z, 8))
        v = (w - a) / b
        zcal = t.ppf(norm.cdf(v), 8)
        original = t.logpdf(z, 8) - np.log(np.sqrt(0.75 * h))
        calibrated = original + norm.logpdf(v) - norm.logpdf(w) - np.log(b)
        expected = {}
        for name, values, margin, r in zip(
            ("orig_gaussian", "orig_t8", "cal_gaussian", "cal_t8"),
            (w, z, v, zcal),
            (original, original, calibrated, calibrated),
            rhos,
            strict=True,
        ):
            if "gaussian" in name:
                copula = multivariate_normal.logpdf(
                    values, cov=[[1, r], [r, 1]]
                ) - norm.logpdf(values).sum(axis=1)
            else:
                copula = multivariate_t.logpdf(
                    values, shape=[[1, r], [r, 1]], df=8
                ) - t.logpdf(values, 8).sum(axis=1)
            expected["loss_" + name] = -margin.sum(axis=1) - copula
        for key, value in expected.items():
            assert_allclose(result[key], value, atol=1e-10, rtol=1e-8)
        for j, asset in enumerate(("qqq", "spx")):
            assert_allclose(
                result["marginal_calibrated_" + asset], calibrated[:, j], atol=1e-10
            )
            assert_allclose(result["pit_calibrated_" + asset], ndtr(v[:, j]), atol=1e-12)
            assert_allclose(
                result["d_" + asset + "_calibration"],
                original[:, j] - calibrated[:, j],
                atol=1e-10,
            )
        contrasts = {
            "d_original_gap": expected["loss_orig_t8"] - expected["loss_orig_gaussian"],
            "d_calibrated_gap": expected["loss_cal_t8"] - expected["loss_cal_gaussian"],
            "d_gaussian_calibration": expected["loss_cal_gaussian"]
            - expected["loss_orig_gaussian"],
            "d_t8_calibration": expected["loss_cal_t8"] - expected["loss_orig_t8"],
        }
        contrasts["d_interaction"] = (
            contrasts["d_calibrated_gap"] - contrasts["d_original_gap"]
        )
        for key, value in contrasts.items():
            assert_allclose(result[key], value, atol=1e-10, rtol=1e-8)

    def test_calibrated_marginal_is_normalized_and_units_cancel(self):
        nodes, weights = np.polynomial.legendre.leggauss(700)
        u, weights = (nodes + 1) / 2, weights / 2
        z = np.column_stack([t.ppf(u, 8), np.zeros(len(u))])
        h = np.full(z.shape, 4 / 3)
        base = _row_scores(z, h, [0.2, -0.1], [0.8, 0.9], [0.4] * 4)
        ratio = np.exp(base["marginal_calibrated_qqq"] - base["marginal_original_qqq"])
        self.assertAlmostEqual(float(weights @ ratio), 1.0, delta=2e-6)
        moved = _row_scores(z, h * 10000**2, [0.2, -0.1], [0.8, 0.9], [0.4] * 4)
        for key in base:
            if key.startswith("d_"):
                assert_allclose(base[key], moved[key], atol=1e-10, rtol=1e-8)
            elif key.startswith("loss_"):
                assert_allclose(moved[key] - base[key], 2 * np.log(10000), atol=1e-10)
        identity = _row_scores(z, h, [0, 0], [1, 1], [0.4] * 4)
        for key in (
            "d_interaction",
            "d_gaussian_calibration",
            "d_t8_calibration",
            "d_qqq_calibration",
            "d_spx_calibration",
        ):
            assert_allclose(identity[key], 0.0, atol=1e-9)


class IndependentCrossedChronologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.copula_calibration import run_crossed

        cls.archive, cls.calendar = synthetic_archive()
        cls.produced = run_crossed(cls.archive, minimum_train=12)

    def check(self, archive=None, produced=None, calendar=True):
        return verify(
            self.archive if archive is None else archive,
            self.produced if produced is None else produced,
            minimum_train=12,
            calendar=self.calendar if calendar else None,
        )

    def test_actual_producer_generated_replay_and_roundtrip(self):
        result = self.check()
        self.assertEqual(result["status"], "VERIFIED")
        self.assertEqual(result["contrasts_verified"], 7)
        self.assertGreater(result["monthly_fits_verified"], 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            roundtrip = {}
            for key, value in self.produced.items():
                if isinstance(value, pd.DataFrame):
                    value.to_parquet(path / (key + ".parquet"), index=False)
                    roundtrip[key] = pd.read_parquet(path / (key + ".parquet"))
                else:
                    roundtrip[key] = json.loads(json.dumps(value, allow_nan=False))
            self.assertEqual(self.check(produced=roundtrip)["status"], "VERIFIED")

    def test_exact_warmup_history_and_unscored_issuance(self):
        p = self.produced
        first = p["fits"][0]
        origin, cutoff = (
            pd.Timestamp(first["fit_origin"]),
            pd.Timestamp(first["training_cutoff"]),
        )
        earlier = self.archive[
            (self.archive.origin < origin) & (self.archive.available_date <= cutoff)
        ]
        self.assertEqual(
            first["train_origins"], earlier.origin.dt.strftime("%Y-%m-%d").tolist()
        )
        self.assertTrue(
            (p["coverage"].loc[p["coverage"].origin < origin, "status"] == "warmup").all()
        )
        self.assertIn(self.archive.origin.iloc[-1], set(p["applications"].origin))
        self.assertNotIn(self.archive.origin.iloc[-1], set(p["panel"].origin))
        self.assertEqual(self.check(calendar=False)["status"], "VERIFIED")

    def test_current_month_labels_do_not_change_its_calibration(self):
        from src.copula_calibration import run_crossed

        archive = self.archive.copy(deep=True)
        first = pd.Timestamp(self.produced["fits"][0]["fit_origin"])
        selected = archive.origin.dt.to_period("M") == first.to_period("M")
        archive.loc[selected, ["y_qqq", "y_spx"]] += 0.005
        changed = run_crossed(archive, minimum_train=12)
        self.assertEqual(changed["fits"][0], self.produced["fits"][0])
        self.assertEqual(self.check(archive, changed)["status"], "VERIFIED")

    def test_every_application_and_density_contrast_group_corruption_rejected(self):
        for key in ("a_qqq", "b_spx", "rho_cal_t8", "mu_qqq", "h_spx", "train_n"):
            changed = copy.deepcopy(self.produced)
            changed["applications"].loc[0, key] += 1 if key == "train_n" else 0.03
            with (
                self.subTest(application=key),
                self.assertRaises((ValueError, AssertionError)),
            ):
                self.check(produced=changed)
        for key in self.produced["panel"]:
            if key.startswith(("loss_", "d_", "marginal_", "pit_", "normal_")):
                changed = copy.deepcopy(self.produced)
                changed["panel"].loc[0, key] += 0.02
                with self.subTest(panel=key), self.assertRaises((ValueError, AssertionError)):
                    self.check(produced=changed)

    def test_cohort_certificate_provenance_and_coverage_corruptions_rejected(self):
        for corruption in (
            "train_origin",
            "calibration",
            "certificate",
            "fit_clock",
            "coverage",
            "missing_app",
            "extra_panel",
        ):
            changed = copy.deepcopy(self.produced)
            if corruption == "train_origin":
                changed["fits"][0]["train_origins"][0] = changed["fits"][0]["fit_origin"]
            elif corruption == "calibration":
                changed["fits"][0]["calibration"]["scale"][0] *= 1.01
            elif corruption == "certificate":
                changed["fits"][0]["dependence"]["cal_t8"]["audit"]["certificate"]["leaves"][
                    0
                ]["lower_bound"] += 0.01
            elif corruption == "fit_clock":
                changed["fits"][0]["training_cutoff"] = changed["fits"][0]["fit_origin"]
            elif corruption == "coverage":
                changed["coverage"].loc[0, "status"] = "scored"
            elif corruption == "missing_app":
                changed["applications"] = (
                    changed["applications"].iloc[1:].reset_index(drop=True)
                )
            else:
                changed["panel"] = pd.concat(
                    [changed["panel"], changed["panel"].iloc[:1]], ignore_index=True
                )
            with (
                self.subTest(corruption=corruption),
                self.assertRaises((ValueError, AssertionError)),
            ):
                self.check(produced=changed)

    def test_archive_clock_and_canonical_identity_corruption_rejected(self):
        for corruption in (
            "available",
            "cutoff",
            "unissued",
            "duplicate",
            "unsorted",
            "bad_variance",
            "offset",
        ):
            archive = self.archive.copy(deep=True)
            if corruption == "available":
                archive.loc[0, "available_date"] = archive.loc[0, "origin"]
            elif corruption == "cutoff":
                archive.loc[0, "training_cutoff"] = archive.loc[0, "fit_origin"]
            elif corruption == "unissued":
                archive.loc[0, "issued"] = False
            elif corruption == "duplicate":
                archive.loc[1, "origin"] = archive.loc[0, "origin"]
            elif corruption == "unsorted":
                archive = archive.iloc[::-1].reset_index(drop=True)
            elif corruption == "bad_variance":
                archive.loc[0, "h_qqq"] = 0.0
            else:
                archive.loc[0, "offset"] = (int(archive.loc[0, "offset"]) + 1) % 5
            with (
                self.subTest(corruption=corruption),
                self.assertRaises((ValueError, AssertionError)),
            ):
                self.check(archive=archive)


if __name__ == "__main__":
    unittest.main()
