"""Prewritten synthetic contracts for calibration of genuinely issued logits."""

import copy
import json
import math
import unittest
from decimal import Decimal, localcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.special import expit

from src import issued_calibration_models as model
from src import range_alert_models as original
from tests.test_range_alert_models import fixture


class ScalarCalibration(unittest.TestCase):
    def test_empty_history_is_exactly_nested(self):
        audit = model.fit_intercept(np.array([]), np.array([]), np.array([]))
        self.assertEqual(audit["intercept"], 0.0)
        self.assertEqual(audit["status"], "EMPTY_HISTORY")
        self.assertEqual(audit["objective"], 0.0)
        self.assertEqual(audit["gradient"], 0.0)
        json.dumps(audit, allow_nan=False)

    def test_analytic_balanced_history_is_exactly_zero(self):
        audit = model.fit_intercept(
            np.array([-2.0, 2.0]), np.array([1.0, 0.0]), np.array([0.2, 0.2])
        )
        self.assertEqual(audit["intercept"], 0.0)
        self.assertEqual(audit["status"], "BALANCED_AT_ZERO")

    def test_high_precision_root_and_original_stationarity(self):
        eta, y, w = (
            np.array([-2.5, 0.25, 3.0]),
            np.array([1.0, 0.0, 1.0]),
            np.array([0.15, 0.3, 0.05]),
        )
        audit = model.fit_intercept(eta, y, w)
        with localcontext() as ctx:
            ctx.prec = 70
            de = [Decimal(str(x)) for x in eta]
            dy = [Decimal(str(x)) for x in y]
            dw = [Decimal(str(x)) for x in w]
            lo, hi = Decimal(-100), Decimal(100)
            for _ in range(240):
                mid = (lo + hi) / 2
                g = (
                    sum(
                        ww * (1 / (1 + (-(ee + mid)).exp()) - yy)
                        for ee, yy, ww in zip(de, dy, dw)
                    )
                    + Decimal(".02") * mid
                )
                if g < 0:
                    lo = mid
                else:
                    hi = mid
            expected = float((lo + hi) / 2)
        self.assertAlmostEqual(audit["intercept"], expected, delta=2e-12)
        objective, gradient = model.calibration_objective(audit["intercept"], eta, y, w)
        self.assertEqual(audit["objective"], objective)
        self.assertEqual(audit["gradient"], gradient)
        self.assertLessEqual(abs(gradient), 1e-8)
        self.assertLess(audit["bracket_gradients"][0], 0)
        self.assertGreater(audit["bracket_gradients"][1], 0)

    def test_signed_residual_keeps_finite_endpoint_information(self):
        self.assertEqual(float(expit(40.0)), 1.0)
        objective, gradient = model.calibration_objective(
            0.0, np.array([40.0]), np.array([1.0]), np.array([0.1])
        )
        self.assertGreater(objective, 0)
        self.assertLess(gradient, 0)
        self.assertEqual(gradient, -0.1 * expit(-40.0))

    def test_reflection_and_unnormalized_information_mass(self):
        eta, y, w = np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([0.1, 0.2])
        a = model.fit_intercept(eta, y, w)["intercept"]
        reflected = model.fit_intercept(-eta, 1 - y, w)["intercept"]
        weak = model.fit_intercept(eta, y, 0.1 * w)["intercept"]
        self.assertAlmostEqual(a, -reflected, delta=2e-12)
        self.assertGreater(a, weak)
        self.assertGreater(weak, 0)

    def test_domains_and_weight_mass_are_not_repaired(self):
        for eta, y, w in [
            ([True], [1.0], [0.1]),
            (["1"], [1.0], [0.1]),
            ([np.inf], [1.0], [0.1]),
            ([1.0], [0.2], [0.1]),
            ([1.0], [1.0], [0.0]),
            ([1.0], [1.0], [1.1]),
            ([1.0], [1.0], [np.nan]),
            ([1.0, 2.0], [1.0], [0.1]),
        ]:
            with self.subTest(eta=eta, y=y, w=w), self.assertRaises(ValueError):
                model.fit_intercept(np.array(eta), np.array(y), np.array(w))

    def test_lost_positive_likelihood_or_weighted_contribution_rejects(self):
        for eta, w in [(1000.0, 0.1), (120.0, 1e-300)]:
            with self.subTest(eta=eta), self.assertRaises(ValueError):
                model.calibration_objective(
                    0.0, np.array([eta]), np.array([1.0]), np.array([w])
                )

    def test_no_false_solver_success_or_budget_fallback(self):
        eta, y, w = np.array([0.0]), np.array([1.0]), np.array([0.1])
        fake = (50.0, SimpleNamespace(converged=True, iterations=1, function_calls=3))
        with (
            patch.object(model, "brentq", return_value=fake) as solver,
            self.assertRaises(ValueError),
        ):
            model.fit_intercept(eta, y, w)
        self.assertEqual(solver.call_count, 1)
        with (
            patch.object(model, "brentq", side_effect=RuntimeError("budget")) as solver,
            self.assertRaises(ValueError),
        ):
            model.fit_intercept(eta, y, w)
        self.assertEqual(solver.call_count, 1)


class IssuedCalibrationPanel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f, cls.t, cls.config = fixture(
            unscored_first_month=True, missing_feature_month=True
        )
        cls.old_panel, cls.old_fits, cls.old_states = original.forecast_panel(
            cls.f, cls.t, cls.config
        )
        cls.panel, cls.states, cls.audit = model.forecast_panel(
            cls.f, cls.t, cls.old_panel, cls.old_fits, cls.old_states, cls.config
        )

    def run_panel(self, *, panel=None, fits=None, states=None, targets=None, features=None):
        return model.forecast_panel(
            self.f if features is None else features,
            self.t if targets is None else targets,
            self.old_panel if panel is None else panel,
            self.old_fits if fits is None else fits,
            self.old_states if states is None else states,
            self.config,
        )

    def test_exact_issued_controls_and_scored_cohort(self):
        for name in ("baseline", "recent_frequency"):
            pd.testing.assert_frame_equal(
                self.panel.loc[self.panel.model.eq(name)].reset_index(drop=True),
                self.old_panel.loc[self.old_panel.model.eq(name)].reset_index(drop=True),
                check_exact=True,
            )
        self.assertEqual(len(self.panel), len(self.old_panel))
        self.assertEqual(tuple(self.panel.columns), original.PANEL_COLUMNS)
        self.assertEqual(set(self.panel.model), {"baseline", "recent_frequency", "calibrated"})
        self.assertEqual(len(self.states), len(self.old_states))
        self.assertEqual(self.audit["new_monthly_fits"], 0)
        json.dumps(self.audit, allow_nan=False)

    def test_no_original_fit_is_called_and_inputs_are_unchanged(self):
        before = copy.deepcopy(
            (self.f, self.t, self.old_panel, self.old_fits, self.old_states)
        )
        with patch.object(
            original, "fit_predict", side_effect=AssertionError("refit forbidden")
        ):
            self.run_panel()
        for left, right in zip(
            before, (self.f, self.t, self.old_panel, self.old_fits, self.old_states)
        ):
            if isinstance(left, pd.DataFrame):
                pd.testing.assert_frame_equal(left, right, check_exact=True)
            else:
                self.assertEqual(left, right)

    def test_empty_seed_and_all_unscored_applications_are_retained(self):
        first = self.states.iloc[0]
        self.assertEqual(first.history_n, 0)
        self.assertEqual(first.intercept, 0)
        self.assertEqual(first.calibrated_probability, first.baseline_probability)
        self.assertEqual(
            first.feature_cutoff_date, pd.Timestamp(self.old_fits[0]["fit_cutoff_date"])
        )
        self.assertFalse(first.scored)
        self.assertTrue(pd.isna(first.latest_admitted_available))
        self.assertEqual(len(self.audit["issued_applications"]), len(self.old_states))

    def test_zero_correction_preserves_original_rounding_within_replay_tolerance(self):
        fits = copy.deepcopy(self.old_fits)
        original_probability = fits[0]["application_probabilities"]["baseline"][0]
        changed_probability = original_probability + 1e-13
        fits[0]["application_probabilities"]["baseline"][0] = changed_probability
        _, states, _ = self.run_panel(fits=fits)
        first = states.iloc[0]
        self.assertEqual(first.intercept, 0)
        self.assertEqual(first.baseline_probability, changed_probability)
        self.assertEqual(first.calibrated_probability, changed_probability)
        self.assertNotEqual(first.calibrated_probability, float(expit(first.baseline_logit)))
        corrected = states.loc[states.intercept.ne(0)]
        self.assertFalse(corrected.empty)
        np.testing.assert_array_equal(
            corrected.calibrated_probability,
            expit(corrected.baseline_logit.to_numpy() + corrected.intercept.to_numpy()),
        )

    def test_weights_match_full_calendar_explicit_age_and_no_refeeding(self):
        issued = {
            pd.Timestamp(row["origin"]): row for row in self.audit["issued_applications"]
        }
        seed = pd.Timestamp(self.audit["seed_cutoff_date"])
        for row in self.states.iloc[[0, 30, 200, -1]].itertuples():
            selected = self.t.loc[
                (self.t.available_date > seed)
                & (self.t.available_date <= row.feature_cutoff_date)
                & self.t.y.notna()
            ]
            selected = selected.loc[selected.index.isin(issued)]
            ages = self.f.index.get_loc(row.feature_cutoff_date) - self.f.index.get_indexer(
                pd.DatetimeIndex(selected.available_date)
            )
            weights = (1 - model.DECAY) * model.DECAY**ages
            self.assertEqual(row.history_n, len(selected))
            self.assertAlmostEqual(row.weight_sum, math.fsum(weights), delta=3e-14)
            self.assertTrue(selected.available_date.le(row.feature_cutoff_date).all())
        self.assertGreater(self.states.iloc[-1].missing_label_arrivals, 0)
        self.assertGreater(self.states.iloc[-1].no_forecast_arrivals, 0)

    def test_saved_application_and_geometry_mismatch_fail_before_new_fit(self):
        for kind in ("probability", "geometry", "missing_fit"):
            fits = copy.deepcopy(self.old_fits)
            if kind == "probability":
                fits[0]["application_probabilities"]["baseline"][0] = 0.123456
            elif kind == "geometry":
                fits[0]["model_audit"]["baseline"]["means"][1] += 1
            else:
                fits.pop(0)
            with (
                self.subTest(kind=kind),
                patch.object(model, "fit_intercept") as solve,
                self.assertRaises(ValueError),
            ):
                self.run_panel(fits=fits)
            solve.assert_not_called()

    def test_control_and_state_metadata_tampering_rejected(self):
        states = self.old_states.copy()
        states.loc[states.index[0], "source_fit_origin"] += pd.Timedelta(days=1)
        with self.assertRaises(ValueError):
            self.run_panel(states=states)
        panel = self.old_panel.copy()
        chosen = panel.model.eq("baseline")
        panel.loc[chosen, "probability"] = 0.123456
        panel.loc[chosen, "loss"] = original.brier_loss(
            panel.loc[chosen, "probability"].to_numpy(), panel.loc[chosen, "y"].to_numpy()
        )
        with self.assertRaises(ValueError):
            self.run_panel(panel=panel)

    def test_future_or_missing_calendar_maturity_is_not_repaired(self):
        targets = self.t.copy()
        targets.loc[targets.index[1400], "available_date"] += pd.Timedelta(days=1)
        with self.assertRaises(ValueError):
            self.run_panel(targets=targets)
        with self.assertRaises(ValueError):
            self.run_panel(features=self.f.iloc[1:])

    def test_saved_calibrated_probabilities_and_brier_replay_exactly(self):
        predicted = expit(
            self.states.baseline_logit.to_numpy() + self.states.intercept.to_numpy()
        )
        np.testing.assert_array_equal(predicted, self.states.calibrated_probability)
        scored = self.states.loc[self.states.scored].set_index("origin")
        panel = self.panel.loc[self.panel.model.eq("calibrated")].set_index("origin")
        np.testing.assert_array_equal(scored.calibrated_probability, panel.probability)
        np.testing.assert_array_equal(
            original.brier_loss(panel.probability.to_numpy(), panel.y.to_numpy()), panel.loss
        )
        bad = self.panel.copy()
        bad.loc[0, "loss"] += 0.01
        with self.assertRaises(ValueError):
            model.validate_panel(bad)


if __name__ == "__main__":
    unittest.main()
