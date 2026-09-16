"""Prewritten synthetic staged probabilities, chronology and rate-state tests."""

import json
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit

from src import range_alert_features as rf
from src import range_alert_models as model


def fixture(*, unscored_first_month=False, missing_feature_month=False):
    rng = np.random.default_rng(20260926)
    dates = pd.bdate_range("2010-01-04", periods=2100, name="date")
    f = pd.DataFrame(rng.normal(size=(len(dates), len(rf.RAW))), index=dates, columns=rf.RAW)
    f["const"] = 1.0
    for weekday in range(1, 5):
        f[f"entry_dow_{weekday}"] = (dates.dayofweek == weekday).astype(float)
    f["intraday"] *= 0.01
    f["intraday_sq"] = f.intraday**2
    f["overnight"] *= 0.01
    f["overnight_sq"] = f.overnight**2
    f["range_extremity"] = rng.uniform(size=len(dates))
    f["feature_cutoff_date"] = pd.Series(dates, index=dates).shift()
    future = pd.Series(dates, index=dates).shift(-1)
    y = (np.arange(len(dates)) % 3 == 0).astype(float)
    t = pd.DataFrame({"y": y, "target_end": future, "available_date": future}, index=dates)
    t.loc[dates[-1], "y"] = np.nan
    section = {
        "models": list(rf.MODELS),
        "raw": list(rf.RAW),
        "baseline": list(rf.BASE),
        "all_features": list(rf.ALL_FEATURES),
        "minimum_train": 1000,
        "minimum_train_per_class": 50,
        "minimum_phase_per_class": 30,
        "minimum_slice_per_class": 15,
        "minimum_phase_observations": 127,
        "origin_start": str(dates[1200].date()),
        "origin_end": str(dates[-2].date()),
        "latest_target": str(dates[-1].date()),
        "development": [str(dates[1200].date()), str(dates[1600].date())],
        "development_target_available_by": str(dates[1600].date()),
        "evaluation": [str(dates[1601].date()), str(dates[-2].date())],
        "evaluation_stability": [
            [str(dates[1601].date()), str(dates[1830].date())],
            [str(dates[1831].date()), str(dates[-2].date())],
        ],
    }
    if unscored_first_month:
        t.loc[dates.to_period("M") == dates[1200].to_period("M"), "y"] = np.nan
    if missing_feature_month:
        f.loc[dates.to_period("M") == dates[1300].to_period("M"), "skew"] = np.nan
    return f, t, section


class RangeAlertModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f, cls.t, cls.p = fixture(unscored_first_month=True, missing_feature_month=True)
        cls.panel, cls.fits, cls.states = model.forecast_panel(cls.f, cls.t, cls.p)

    def test_staged_scalar_matches_independent_derivative_root(self):
        f, t, _ = fixture()
        tr, app = f.iloc[1:1001], f.iloc[1001:1010]
        y = t.loc[tr.index, "y"]
        forecasts, audit = model.fit_predict(tr, y, app)
        x, query, z, query_z, transform = rf.transform(tr, app)
        beta = np.asarray(audit["baseline"]["beta"])
        eta = x.to_numpy() @ beta
        grad = x.to_numpy().T @ (expit(eta) - y.to_numpy()) / len(y)
        grad[1:] += 0.02 * beta[1:]
        self.assertLessEqual(np.max(np.abs(grad)), 1e-8 + 1e-12)
        derivative = lambda b: (
            np.mean(z.to_numpy() * (expit(eta + b * z.to_numpy()) - y.to_numpy())) + 0.02 * b
        )
        radius = (np.mean(np.abs(z)) + 1) / 0.02
        independent = brentq(derivative, -radius, radius, xtol=1e-12, rtol=1e-14)
        self.assertAlmostEqual(audit["location"]["b"], independent, delta=1e-6)
        np.testing.assert_array_equal(forecasts["baseline"], expit(query.to_numpy() @ beta))
        np.testing.assert_array_equal(
            forecasts["location"],
            expit(query.to_numpy() @ beta + audit["location"]["b"] * query_z.to_numpy()),
        )
        self.assertEqual(audit["transform"], transform)
        self.assertEqual(audit["location"]["scale"], 1.0)
        json.dumps(audit, allow_nan=False)

    def test_constant_nonzero_location_is_exactly_nested_and_retained(self):
        f, t, _ = fixture()
        f["range_extremity"] = 0.1
        p, audit = model.fit_predict(f.iloc[1:1001], t.y.iloc[1:1001], f.iloc[1001:1020])
        self.assertEqual(audit["location"]["b"], 0)
        self.assertEqual(audit["location"]["status"], "EXACT_CONSTANT_INPUT")
        np.testing.assert_array_equal(p["baseline"], p["location"])

    def test_application_values_and_labels_cannot_change_fit(self):
        f, t, _ = fixture()
        train, apply = f.iloc[1:1001], f.iloc[1001:1020]
        _, first = model.fit_predict(train, t.y.loc[train.index], apply)
        changed = apply.copy()
        changed["range_extremity"] = 1 - changed.range_extremity
        changed["I"] += 3
        _, second = model.fit_predict(train, t.y.loc[train.index], changed)
        self.assertEqual(first, second)

    def test_fit_rejects_label_misalignment_and_insufficient_class_support(self):
        f, t, _ = fixture()
        with self.assertRaises(ValueError):
            model.fit_predict(f.iloc[1:1001], t.y.iloc[2:1002], f.iloc[1001:1010])
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"):
            model.fit_predict(
                f.iloc[1:1001],
                pd.Series(np.zeros(1000), index=f.index[1:1001]),
                f.iloc[1001:1010],
            )

    def test_full_original_gradient_recomputed_after_claimed_solver_success(self):
        f, t, _ = fixture()
        fake = (
            np.zeros(26),
            {
                "objective": 0.0,
                "gradient": [0.0] * 26,
                "gradient_max_abs": 0.0,
                "iterations": 0,
                "backtracks": 0,
                "success": True,
                "start": [0.0] * 26,
            },
        )
        with patch.object(model, "_newton", return_value=fake), self.assertRaises(ValueError):
            model.fit_predict(f.iloc[1:1001], t.y.iloc[1:1001], f.iloc[1001:1010])

    def test_all_class_support_and_geometry_checked_before_any_fit(self):
        f, t, p = fixture()
        t.loc[t.index >= pd.Timestamp(p["evaluation"][0]), "y"] = 0.0
        t.loc[t.index[-1], "y"] = np.nan
        with (
            patch.object(
                model, "_newton", side_effect=AssertionError("optimizer must not run")
            ) as called,
            self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"),
        ):
            model.forecast_panel(f, t, p)
        called.assert_not_called()

    def test_late_application_geometry_failure_precedes_every_optimizer(self):
        f, t, p = fixture()
        f.loc[f.index[-2], "I"] = 1e308
        with (
            patch.object(
                model, "_newton", side_effect=AssertionError("no earlier fit")
            ) as called,
            self.assertRaises(ValueError),
        ):
            model.forecast_panel(f, t, p)
        called.assert_not_called()

    def test_feature_first_schedule_and_wholly_unscored_first_month(self):
        entries, labels, _ = model.application_calendar(self.f, self.t, self.p)
        self.assertEqual(pd.Timestamp(self.fits[0]["fit_origin"]), entries[0])
        month = entries[0].to_period("M")
        self.assertFalse((self.panel.origin.dt.to_period("M") == month).any())
        self.assertTrue((self.states.origin.dt.to_period("M") == month).any())
        self.assertEqual(len(self.fits), len(entries.to_period("M").unique()))
        self.assertEqual(len(self.panel), 3 * int(labels.loc[entries].sum()))
        self.assertEqual(len(self.states), len(entries))
        for fit in self.fits:
            apps = pd.to_datetime(fit["application_origins"])
            self.assertEqual(len(apps), fit["application_n"])
            self.assertEqual(set(fit["application_probabilities"]), {"baseline", "location"})
            self.assertTrue(
                all(len(v) == len(apps) for v in fit["application_probabilities"].values())
            )

    def test_unscored_application_probabilities_still_validated(self):
        original = model.fit_predict

        def bad(train, y, app):
            result, audit = original(train, y, app)
            if app.index[0].to_period("M") == self.f.index[1200].to_period("M"):
                result["location"][0] = np.nan
            return result, audit

        with (
            patch.object(model, "fit_predict", side_effect=bad),
            self.assertRaises(ValueError),
        ):
            model.forecast_panel(self.f, self.t, self.p)

    def test_training_requires_prior_cutoff_and_exact_maturity(self):
        f, t, p = fixture()
        fit = f.index[1200]
        mask = model.training_mask(f, t, fit, p["minimum_train"])
        self.assertFalse(mask.iloc[0])
        self.assertFalse(mask.iloc[1199])
        self.assertTrue(mask.iloc[1198])
        self.assertTrue((t.loc[mask, "available_date"] <= f.index[1199]).all())
        altered = f.copy()
        altered.loc[f.index[10], "feature_cutoff_date"] = f.index[10]
        with self.assertRaises(ValueError):
            model.training_mask(altered, t, fit, p["minimum_train"])

    def test_future_label_mutation_does_not_choose_first_fit_date(self):
        f, t, p = fixture()
        first, _, _ = model.application_calendar(f, t, p)
        changed = t.copy()
        changed.loc[first[:25], "y"] = np.nan
        second, _, _ = model.application_calendar(f, changed, p)
        self.assertTrue(first.equals(second))
        one = model.preflight(f, t, p)
        two = model.preflight(f, changed, p)
        self.assertEqual(one["fits"][0]["fit_origin"], two["fits"][0]["fit_origin"])

    def test_explicit_weight_sum_checks_seed_missing_aging_and_feature_gaps(self):
        reference = self.f.index
        seed = self.states.iloc[0]
        k0 = reference.get_loc(seed.seed_cutoff_date)
        self.assertEqual(seed.S, seed.seed_probability)
        self.assertEqual(seed.W, 1.0)
        self.assertEqual(seed.cumulative_updates, 0)
        self.assertEqual(seed.elapsed_sessions, 0)
        self.assertEqual(seed.latest_consumed_available, seed.seed_last_available)
        for row in self.states.iloc[[0, 5, 30, 150, -1]].itertuples():
            k = reference.get_loc(row.feature_cutoff_date)
            arrivals = [
                (j, float(self.t.y.iloc[j - 1]))
                for j in range(k0 + 1, k + 1)
                if pd.notna(self.t.y.iloc[j - 1])
            ]
            numerator = math.fsum(
                [seed.seed_probability * model.DECAY ** (k - k0)]
                + [model.LABEL_WEIGHT * model.DECAY ** (k - j) * y for j, y in arrivals]
            )
            denominator = math.fsum(
                [model.DECAY ** (k - k0)]
                + [model.LABEL_WEIGHT * model.DECAY ** (k - j) for j, _ in arrivals]
            )
            self.assertAlmostEqual(row.S, numerator, delta=2e-14)
            self.assertAlmostEqual(row.W, denominator, delta=2e-14)
            self.assertEqual(row.recent_frequency, row.S / row.W)
            self.assertEqual(row.cumulative_updates, len(arrivals))
            self.assertEqual(row.elapsed_sessions, k - k0)
        self.assertGreater(int(self.states.iloc[-1].cumulative_updates), len(self.states) - 25)

    def test_label_after_current_cutoff_cannot_change_earlier_states(self):
        changed = self.t.copy()
        origin = self.states.origin.iloc[100]
        cutoff = self.states.feature_cutoff_date.iloc[100]
        old = float(changed.loc[cutoff, "y"])
        changed.loc[cutoff, "y"] = 1 - old
        applications = pd.DatetimeIndex(self.states.origin)
        scored = pd.DatetimeIndex(self.panel.loc[self.panel.model == "baseline", "origin"])
        other = model._frequency_states(self.f, changed, applications, self.fits, scored)
        pd.testing.assert_frame_equal(
            self.states.loc[self.states.origin <= origin], other.loc[other.origin <= origin]
        )
        self.assertFalse(self.states.S.equals(other.S))

    def test_development_target_crossover_is_excluded_for_all_models(self):
        boundary = pd.Timestamp(self.p["development"][1])
        self.assertFalse(self.panel.origin.eq(boundary).any())
        dev = self.panel.loc[self.panel.phase == "development"]
        self.assertTrue((dev.available_date <= boundary).all())

    def test_complete_paired_panel_and_saved_scores(self):
        model.validate_panel(self.panel)
        self.assertEqual(tuple(self.panel.columns), model.PANEL_COLUMNS)
        self.assertEqual(set(self.panel.model), set(rf.MODELS))
        self.assertFalse(self.panel.duplicated(["origin", "model"]).any())
        np.testing.assert_array_equal(
            self.panel.loss, model.brier_loss(self.panel.probability, self.panel.y)
        )

    def test_probability_endpoints_valid_and_nonzero_score_underflow_rejected(self):
        np.testing.assert_array_equal(
            model.brier_loss(np.array([0.0, 1.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0, 1.0])),
            [0.0, 0.0, 1.0, 1.0],
        )
        with self.assertRaises(ValueError):
            model.brier_loss(np.array([1e-200]), np.array([0.0]))

    def test_tampered_panel_model_probability_score_and_metadata_rejected(self):
        for column, value in [
            ("probability", -0.1),
            ("loss", -0.1),
            ("train_n", 50),
            ("fit_cutoff_date", self.panel.origin.iloc[0]),
            ("model", "other"),
        ]:
            altered = self.panel.copy()
            altered.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                model.validate_panel(altered)
        with self.assertRaises(ValueError):
            model.validate_panel(self.panel.iloc[:-1])

    def test_panel_guard_requires_full_minimum_training_count(self):
        altered = self.panel.copy()
        altered["train_n"] = 500
        with self.assertRaises(ValueError):
            model.validate_panel(altered)

    def test_panel_numeric_metadata_cannot_be_boolean(self):
        altered = self.panel.copy()
        altered["horizon"] = True
        with self.assertRaises(ValueError):
            model.validate_panel(altered)

    def test_nonfinite_observed_features_and_labels_are_not_missing_row_repairs(self):
        for frame, column in [("features", "I"), ("targets", "y")]:
            f, t, p = fixture()
            (f if frame == "features" else t).loc[f.index[3], column] = np.inf
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                model.application_calendar(f, t, p)

    def test_preflight_counts_and_no_newton_calls(self):
        with patch.object(
            model, "_newton", side_effect=AssertionError("preflight has no fit")
        ) as called:
            result = model.preflight(self.f, self.t, self.p)
        called.assert_not_called()
        self.assertEqual(result["monthly_fits"], len(self.fits))
        self.assertEqual(result["common_application_origins"], len(self.states))
        self.assertEqual(result["common_scored_origins"], len(self.panel) // 3)
        self.assertEqual(len(result["phases"]), 2)
        self.assertEqual(len(result["phases"][1]["slices"]), 2)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
