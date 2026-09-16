"""Prewritten generated contracts for the new causal marginal calibration.

No historical source, forecast, residual, or score is opened by these tests.
The four-array contrast helper checks positional alignment; actual origin
identity remains a separate forecast-pipeline/verifier contract.
"""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad
from scipy.stats import norm, t

import src.copula_calibration as subject
from src.copula_calibration import (
    calibrate,
    contrasts,
    fit_calibration,
    identity_calibration,
    select_history,
)


def from_normal(w):
    return t.ppf(norm.cdf(np.asarray(w, dtype=float)), 8)


def calibration(location=(0.0, 0.0), scale=(1.0, 1.0), train_n=100):
    return {"location": list(location), "scale": list(scale), "train_n": train_n}


def archive_fixture():
    dates = pd.bdate_range("2015-12-21", periods=10)
    frame = pd.DataFrame(
        {
            "origin": dates[:8],
            "available_date": dates[1:9],
            "issued": True,
            "y_qqq": np.arange(8, dtype=float) / 100,
            "y_spx": np.arange(8, dtype=float) / 200,
            "mu_qqq": 0.001,
            "mu_spx": -0.001,
            "h_qqq": 0.0004,
            "h_spx": 0.0003,
        },
        index=np.arange(20, 28),
    )
    frame.loc[21, "issued"] = False
    frame.loc[22, "y_spx"] = np.nan
    return frame, dates[6], dates[5]


def crossed_archive():
    origins = pd.to_datetime(
        [
            "2016-01-04",
            "2016-01-05",
            "2016-01-29",
            "2016-02-01",
            "2016-02-02",
            "2016-02-29",
            "2016-03-01",
            "2016-03-02",
        ]
    )
    z = np.array(
        [
            [-0.8, 1.0],
            [0.5, -0.4],
            [1.2, 0.3],
            [-1.1, -0.8],
            [0.4, 0.6],
            [0.9, -0.5],
            [-0.2, 1.5],
            [0.6, -1.2],
        ]
    )
    frame = pd.DataFrame(
        {
            "origin": origins,
            "available_date": origins + pd.offsets.BDay(1),
            "target_end": origins + pd.offsets.BDay(1),
            "issued": True,
            "phase": "development",
            "offset": np.arange(8) % 5,
            "mu_qqq": 0.001,
            "mu_spx": -0.001,
            "h_qqq": 0.0004,
            "h_spx": 0.0003,
            "eligible_scored": True,
            "old_rho_t8": 0.8,
            "old_rho_gaussian": -0.7,
        }
    )
    frame["y_qqq"] = frame.mu_qqq + np.sqrt(0.75 * frame.h_qqq) * z[:, 0]
    frame["y_spx"] = frame.mu_spx + np.sqrt(0.75 * frame.h_spx) * z[:, 1]
    month_first = frame.groupby(frame.origin.dt.to_period("M")).origin.transform("min")
    frame["fit_origin"] = month_first
    frame["training_cutoff"] = month_first - pd.offsets.BDay(1)
    frame.loc[4, "y_spx"] = np.nan
    frame.loc[[4, 7], "eligible_scored"] = False
    return frame


def dependence_stub(z, family):
    return (0.25 if family == "gaussian" else 0.2), {
        "family": family,
        "train_n": len(z),
        "generated_timeline_stub": True,
    }


class CalibrationDensityTests(unittest.TestCase):
    def test_identity_preserves_original_coordinates_and_density_exactly(self):
        z = np.array([[-2.0, 0.0], [0.3, 1.2], [4.0, -0.7]])
        h = np.array([[0.2, 0.8], [1.0, 2.0], [0.5, 4.0]])
        identity = identity_calibration()
        self.assertEqual(identity, calibration(train_n=0))
        got = calibrate(z, h, identity)
        self.assertEqual(set(got), {"z", "normal", "log_marginal", "pit"})
        assert_array_equal(got["z"], z)
        assert_allclose(got["normal"], norm.ppf(t.cdf(z, 8)), atol=2e-13)
        assert_allclose(got["pit"], t.cdf(z, 8), atol=2e-15)
        assert_allclose(
            got["log_marginal"],
            t.logpdf(z, 8) - 0.5 * np.log(0.75 * h),
            atol=2e-13,
        )

    def test_affine_fit_recovers_known_distortion_and_population_scale(self):
        x = np.array([-1.5, -0.5, 0.5, 1.5]) / np.sqrt(1.25)
        clean = np.column_stack([x, x[::-1]])
        location, scale = np.array([0.35, -0.2]), np.array([1.3, 0.7])
        distorted = from_normal(location + scale * clean)
        fit = fit_calibration(distorted)
        self.assertEqual(set(fit), {"location", "scale", "train_n"})
        self.assertEqual(fit["train_n"], 4)
        assert_allclose(fit["location"], location, atol=2e-12)
        assert_allclose(fit["scale"], scale, atol=2e-12)
        got = calibrate(distorted, np.ones_like(distorted), fit)
        assert_allclose(got["normal"], clean, atol=2e-12)
        assert_allclose(got["z"], from_normal(clean), atol=2e-11)
        # ddof=1 would return sqrt(4/3) times the required scale.
        self.assertNotAlmostEqual(fit["scale"][0], scale[0] * np.sqrt(4 / 3))

    def test_change_of_variable_jacobian_matches_cdf_derivative(self):
        z = np.array([[-0.8, 0.4], [0.3, 1.1]])
        h = np.array([[0.6, 1.2], [0.7, 0.3]])
        spec = calibration((0.2, -0.4), (0.8, 1.3))
        w = norm.ppf(t.cdf(z, 8))
        v = (w - spec["location"]) / spec["scale"]
        expected = (
            t.logpdf(z, 8)
            - 0.5 * np.log(0.75 * h)
            + norm.logpdf(v)
            - norm.logpdf(w)
            - np.log(spec["scale"])
        )
        got = calibrate(z, h, spec)
        assert_allclose(got["log_marginal"], expected, atol=2e-12)
        step = 1e-5
        plus = norm.cdf((norm.ppf(t.cdf(z + step, 8)) - spec["location"]) / spec["scale"])
        minus = norm.cdf((norm.ppf(t.cdf(z - step, 8)) - spec["location"]) / spec["scale"])
        derivative_y = (plus - minus) / (2 * step * np.sqrt(0.75 * h))
        assert_allclose(np.exp(got["log_marginal"]), derivative_y, rtol=2e-8)

    def test_both_recalibrated_marginal_densities_integrate_to_one(self):
        spec = calibration((0.3, -0.45), (0.7, 1.3))
        # h=4/3 makes the original t8 scale one, so integrating dz is dy.
        for j in range(2):
            with self.subTest(asset=j):

                def density(value, asset=j):
                    result = calibrate(
                        np.array([[value, value]]), np.full((1, 2), 4 / 3), spec
                    )
                    return float(np.exp(result["log_marginal"][0, asset]))

                integral, error = quad(density, -np.inf, np.inf, epsabs=2e-8)
                self.assertLess(error, 2e-7)
                self.assertAlmostEqual(integral, 1.0, places=7)

    def test_log_tail_inverse_preserves_extreme_finite_coordinates_without_clipping(self):
        z = from_normal([[2.0, -2.0]])
        got = calibrate(z, np.ones((1, 2)), calibration((-8.0, 8.0)))
        assert_allclose(got["normal"], [[10.0, -10.0]], atol=2e-12)
        expected = t.isf(norm.sf(10.0), 8)
        assert_allclose(got["z"], [[expected, -expected]], rtol=2e-11)
        self.assertTrue(np.isfinite(got["log_marginal"]).all())
        # Ordinary Phi can round to an endpoint; the inverse must use log tails.
        self.assertEqual(got["pit"][0, 0], norm.cdf(10.0))

    def test_actual_tail_underflow_fails_instead_of_clipping(self):
        with self.assertRaises(ValueError):
            calibrate(np.zeros((1, 2)), np.ones((1, 2)), calibration((-50.0, 50.0)))

    def test_fit_rejects_insufficient_nonfinite_and_degenerate_samples(self):
        cases = [
            np.empty((0, 2)),
            np.zeros((1, 2)),
            np.zeros((4, 2)),
            np.array([[0.0, 0.0], [1.0, np.nan]]),
            np.array([[0.0, 0.0], [1.0, np.inf]]),
            np.ones((3, 3)),
            from_normal([[0.0, -1.0], [1e-13, 1.0]]),
        ]
        for values in cases:
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                fit_calibration(values)

    def test_transform_rejects_invalid_shapes_scales_and_variances(self):
        z, h = np.ones((2, 2)), np.ones((2, 2))
        for bad_h in (np.ones((2, 1)), np.zeros((2, 2)), -h, h * np.inf):
            with self.subTest(h=bad_h), self.assertRaises(ValueError):
                calibrate(z, bad_h, identity_calibration())
        for spec in (
            calibration(scale=(0.0, 1.0)),
            calibration(scale=(-1.0, 1.0)),
            calibration(scale=(np.nan, 1.0)),
            calibration(location=(np.inf, 0.0)),
            calibration(location=(0.0,)),
        ):
            with self.subTest(calibration=spec), self.assertRaises(ValueError):
                calibrate(z, h, spec)

    def test_fit_and_transform_do_not_mutate_input_arrays_or_audit(self):
        z = from_normal([[-0.9, 1.0], [-0.1, -0.7], [0.7, 0.4]])
        h = np.ones_like(z)
        z_before, h_before = z.copy(), h.copy()
        spec = fit_calibration(z)
        before = copy.deepcopy(spec)
        calibrate(z, h, spec)
        assert_array_equal(z, z_before)
        assert_array_equal(h, h_before)
        self.assertEqual(spec, before)


class IssuedHistoryTests(unittest.TestCase):
    def test_only_issued_complete_strictly_earlier_mature_rows_are_selected(self):
        frame, fit, cutoff = archive_fixture()
        got = select_history(frame, fit, cutoff, minimum_train=3)
        self.assertEqual(got.origin.tolist(), frame.loc[[20, 23, 24], "origin"].tolist())
        self.assertEqual(got.iloc[-1].available_date, cutoff)
        self.assertTrue((got.origin < fit).all())

    def test_future_and_unissued_outcome_perturbations_leave_history_and_fit_unchanged(self):
        frame, fit, cutoff = archive_fixture()
        before = frame.copy(deep=True)
        selected = select_history(frame, fit, cutoff, minimum_train=3)
        changed = frame.copy(deep=True)
        excluded = ~changed.origin.isin(selected.origin)
        changed.loc[excluded, ["y_qqq", "y_spx"]] = [123.0, -456.0]
        # The originally missing mature row remains missing; this is not new data.
        changed.loc[22, "y_spx"] = np.nan
        after = select_history(changed, fit, cutoff, minimum_train=3)
        pd.testing.assert_frame_equal(selected, after)

        def coordinates(rows):
            return (
                rows[["y_qqq", "y_spx"]].to_numpy() - rows[["mu_qqq", "mu_spx"]].to_numpy()
            ) / np.sqrt(0.75 * rows[["h_qqq", "h_spx"]].to_numpy())

        self.assertEqual(
            fit_calibration(coordinates(selected)), fit_calibration(coordinates(after))
        )
        pd.testing.assert_frame_equal(frame, before)

    def test_insufficient_history_and_ambiguous_order_fail(self):
        frame, fit, cutoff = archive_fixture()
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"):
            select_history(frame, fit, cutoff, minimum_train=4)
        for malformed in (frame.iloc[::-1], pd.concat([frame.iloc[:1], frame])):
            with self.subTest(rows=len(malformed)), self.assertRaises(ValueError):
                select_history(malformed, fit, cutoff, minimum_train=2)

    def test_observed_nonpositive_or_infinite_eligible_variance_is_not_dropped(self):
        frame, fit, cutoff = archive_fixture()
        for value in (0.0, -1.0, np.inf):
            bad = frame.copy(deep=True)
            bad.loc[23, "h_qqq"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                select_history(bad, fit, cutoff, minimum_train=2)


class CrossedContrastTests(unittest.TestCase):
    def test_all_seven_signed_contrasts_share_every_observation(self):
        losses = {
            "orig_t8": np.array([2.0, 5.0, -2.0]),
            "orig_gaussian": np.array([3.0, 4.0, -1.0]),
            "cal_t8": np.array([1.5, 2.0, -1.0]),
            "cal_gaussian": np.array([1.7, 1.5, 0.0]),
        }
        marginals = {
            "original": np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, -2.0]]),
            "calibrated": np.array([[0.7, 1.0], [2.0, 4.5], [-0.5, -3.0]]),
        }
        originals = copy.deepcopy((losses, marginals))
        got = contrasts(losses, marginals)
        expected = {
            "original_gap": [-1.0, 1.0, -1.0],
            "calibrated_gap": [-0.2, 0.5, -1.0],
            "interaction": [0.8, -0.5, 0.0],
            "gaussian_calibration": [-1.3, -2.5, 1.0],
            "t8_calibration": [-0.5, -3.0, 1.0],
            "qqq_calibration": [-0.3, -1.0, 0.5],
            "spx_calibration": [-1.0, 0.5, -1.0],
        }
        self.assertEqual(set(got), set(expected))
        for name, values in expected.items():
            assert_allclose(got[name], values, atol=1e-14)
        for current, old in zip((losses, marginals), originals, strict=True):
            for name in current:
                assert_array_equal(current[name], old[name])

    def test_identity_calibration_produces_zero_interaction_and_calibration_gaps(self):
        a, b = np.array([-3.0, 2.0]), np.array([-2.0, 3.0])
        got = contrasts(
            {"orig_t8": a, "orig_gaussian": b, "cal_t8": a, "cal_gaussian": b},
            {"original": np.ones((2, 2)), "calibrated": np.ones((2, 2))},
        )
        assert_array_equal(got["original_gap"], [-1.0, -1.0])
        assert_array_equal(got["calibrated_gap"], [-1.0, -1.0])
        for name in (
            "interaction",
            "gaussian_calibration",
            "t8_calibration",
            "qqq_calibration",
            "spx_calibration",
        ):
            assert_array_equal(got[name], [0.0, 0.0])

    def test_partial_or_nonfinite_crossed_cohorts_are_rejected_not_filtered(self):
        base = {
            name: np.ones(3) for name in ("orig_t8", "orig_gaussian", "cal_t8", "cal_gaussian")
        }
        marginal = {name: np.ones((3, 2)) for name in ("original", "calibrated")}
        malformed = []
        for value in (np.ones(2), np.array([1.0, np.nan, 2.0]), np.ones((3, 1))):
            bad = copy.deepcopy(base)
            bad["cal_t8"] = value
            malformed.append((bad, marginal))
        bad = copy.deepcopy(base)
        del bad["cal_gaussian"]
        malformed.append((bad, marginal))
        malformed.append((base, {"original": np.ones((3, 2)), "calibrated": np.ones((2, 2))}))
        malformed.append(
            (base, {"original": np.ones((3, 2)), "calibrated": np.full((3, 2), np.inf)})
        )
        for losses, margins in malformed:
            with self.subTest(losses=losses), self.assertRaises(ValueError):
                contrasts(losses, margins)


class CrossedTimelineTests(unittest.TestCase):
    def run_generated(self, frame):
        with patch.object(subject, "fit_dependence", side_effect=dependence_stub) as fitted:
            result = subject.run_crossed(frame, minimum_train=2)
        return result, fitted.call_args_list

    def test_monthly_mature_issued_history_warmup_and_same_date_crossing(self):
        frame = crossed_archive()
        before = frame.copy(deep=True)
        got, calls = self.run_generated(frame)
        self.assertEqual(set(got), {"applications", "panel", "fits", "coverage"})
        self.assertEqual(len(calls), 8)  # Both families in both marginal systems, twice.
        apps, panel, fits = got["applications"], got["panel"], got["fits"]
        self.assertEqual(apps.origin.tolist(), frame.origin.iloc[3:].tolist())
        self.assertEqual(panel.origin.tolist(), frame.origin.iloc[[3, 5, 6]].tolist())
        self.assertEqual(
            [f["train_origins"] for f in fits],
            [
                ["2016-01-04", "2016-01-05"],
                ["2016-01-04", "2016-01-05", "2016-01-29", "2016-02-01"],
            ],
        )
        self.assertEqual([f["train_n"] for f in fits], [2, 4])
        self.assertEqual(got["coverage"].origin.tolist(), frame.origin.tolist())
        self.assertEqual(
            got["coverage"].status.tolist(),
            [
                "warmup",
                "warmup",
                "warmup",
                "scored",
                "issued_unscored",
                "scored",
                "scored",
                "issued_unscored",
            ],
        )
        self.assertFalse({"y_qqq", "y_spx", "target_end", "available_date"} & set(apps))
        for name in ("loss_orig_gaussian", "loss_orig_t8", "loss_cal_gaussian", "loss_cal_t8"):
            self.assertTrue(np.isfinite(panel[name]).all())
        pd.testing.assert_frame_equal(frame, before)

    def test_unscored_missing_label_queries_still_require_valid_issued_marginals(self):
        for column, value in (
            ("h_qqq", 0.0),
            ("h_qqq", -0.1),
            ("h_spx", np.nan),
            ("h_spx", np.inf),
            ("mu_qqq", np.nan),
            ("mu_spx", np.inf),
        ):
            frame = crossed_archive()
            # This final issued row never enters a subsequent fit or scored panel.
            frame.loc[7, ["y_qqq", "y_spx"]] = np.nan
            frame.loc[7, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                self.run_generated(frame)

    def test_entire_month_forecasts_ignore_current_and_future_outcomes(self):
        frame = crossed_archive()
        original, _ = self.run_generated(frame)
        altered = frame.copy(deep=True)
        mask = altered.origin >= pd.Timestamp("2016-02-01")
        altered.loc[mask, ["y_qqq", "y_spx"]] += [0.2, -0.3]
        changed, _ = self.run_generated(altered)
        left = original["applications"].query("origin < '2016-03-01'").reset_index(drop=True)
        right = changed["applications"].query("origin < '2016-03-01'").reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right)
        self.assertEqual(original["fits"][0], changed["fits"][0])

    def test_base_marginals_are_preserved_and_old_dependence_is_refitted(self):
        frame = crossed_archive()
        got, _ = self.run_generated(frame)
        altered = frame.copy(deep=True)
        altered["old_rho_t8"] = -0.9
        altered["old_rho_gaussian"] = 0.9
        other, _ = self.run_generated(altered)
        pd.testing.assert_frame_equal(got["applications"], other["applications"])
        pd.testing.assert_frame_equal(got["panel"], other["panel"])
        for name in ("mu_qqq", "mu_spx", "h_qqq", "h_spx"):
            assert_array_equal(got["applications"][name], frame[name].iloc[3:])
        for prefix in ("orig", "cal"):
            assert_array_equal(got["applications"][f"rho_{prefix}_gaussian"], np.full(5, 0.25))
            assert_array_equal(got["applications"][f"rho_{prefix}_t8"], np.full(5, 0.2))

    def test_saved_crossed_losses_include_marginal_jacobian_and_signed_interaction(self):
        from scipy.stats import multivariate_normal, multivariate_t

        got, _ = self.run_generated(crossed_archive())
        for row in got["panel"].itertuples(index=False):
            mu = np.array([row.mu_qqq, row.mu_spx])
            h = np.array([row.h_qqq, row.h_spx])
            z = (np.array([row.y_qqq, row.y_spx]) - mu) / np.sqrt(0.75 * h)
            w = norm.ppf(t.cdf(z, 8))
            v = (w - [row.a_qqq, row.a_spx]) / [row.b_qqq, row.b_spx]
            cz = t.ppf(norm.cdf(v), 8)
            original = t.logpdf(z, 8) - 0.5 * np.log(0.75 * h)
            calibrated = (
                original + norm.logpdf(v) - norm.logpdf(w) - np.log([row.b_qqq, row.b_spx])
            )
            assert_allclose(
                [row.marginal_original_qqq, row.marginal_original_spx], original, atol=2e-10
            )
            assert_allclose(
                [row.marginal_calibrated_qqq, row.marginal_calibrated_spx],
                calibrated,
                atol=2e-10,
            )
            expected = {}
            for prefix, q, n, marginal in (
                ("orig", z, w, original),
                ("cal", cz, v, calibrated),
            ):
                for family in ("t8", "gaussian"):
                    rho = getattr(row, f"rho_{prefix}_{family}")
                    shape = [[1.0, rho], [rho, 1.0]]
                    if family == "t8":
                        copula = (
                            multivariate_t.logpdf(q, shape=shape, df=8) - t.logpdf(q, 8).sum()
                        )
                    else:
                        copula = (
                            multivariate_normal.logpdf(n, cov=shape) - norm.logpdf(n).sum()
                        )
                    expected[f"{prefix}_{family}"] = -float(marginal.sum() + copula)
                    self.assertAlmostEqual(
                        getattr(row, f"loss_{prefix}_{family}"),
                        expected[f"{prefix}_{family}"],
                        places=9,
                    )
            interaction = (expected["cal_t8"] - expected["cal_gaussian"]) - (
                expected["orig_t8"] - expected["orig_gaussian"]
            )
            self.assertAlmostEqual(row.d_interaction, interaction, places=9)
            assert_allclose(
                [row.pit_calibrated_qqq, row.pit_calibrated_spx], norm.cdf(v), atol=2e-13
            )


if __name__ == "__main__":
    unittest.main()
