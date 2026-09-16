"""Generated-source integration with the independent range-alert verifier."""

import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from src import range_alert_features as feature
from src import range_alert_models as model
from src import range_alert_search as search
from src import verify_range_alert as verifier
from tests.test_range_alert_models import fixture as schedule_fixture


def synthetic_prior():
    return [
        {
            "study": "generated_previous_trials",
            "candidate": str(number),
            "control": "baseline",
            "horizon": 1,
            "p_conservative": 1.0,
        }
        for number in range(127)
    ]


def raw_fixture():
    rng = np.random.default_rng(20260927)
    _, _, section = schedule_fixture(unscored_first_month=True)
    dates = pd.bdate_range("2010-01-04", periods=2100, name="date")
    overnight = rng.normal(0, 0.001, len(dates))
    intraday = rng.normal(0, 0.003, len(dates))
    close = 100 * np.exp(np.cumsum(overnight + intraday))
    opened = close / np.exp(intraday)
    width = np.where(np.arange(len(dates)) % 7 == 0, 0.04, 0.004)
    high = np.maximum(opened, close) * np.exp(width * rng.uniform(0.7, 1.2, len(dates)))
    low = np.minimum(opened, close) * np.exp(-width * rng.uniform(0.7, 1.2, len(dates)))
    daily = pd.DataFrame(
        {"open": opened, "high": high, "low": low, "close": close}, index=dates
    )
    iv = pd.DataFrame(
        {
            "vix": np.exp(3 + rng.normal(0, 0.1, len(dates))),
            "vix9d": np.exp(3 + rng.normal(0, 0.1, len(dates))),
            "vvix": np.exp(4.5 + rng.normal(0, 0.1, len(dates))),
            "skew": 110 + rng.normal(0, 3, len(dates)),
        },
        index=dates,
    )
    iv.loc[dates.to_period("M") == dates[1300].to_period("M"), "skew"] = np.nan
    f, t = feature.build_features(daily, iv)
    # Deliberately unavailable labels stress the model API's feature-first
    # schedule. This synthetic mask is not represented as another raw provider.
    t.loc[dates.to_period("M") == dates[1200].to_period("M"), "y"] = np.nan
    with open("range_alert.yaml") as stream:
        protocol = yaml.safe_load(stream)
    protocol["index"] = {**protocol["index"], **section}
    return f, t, protocol


class RangeAlertIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f, cls.t, cls.p = raw_fixture()
        cls.support = model.preflight(cls.f, cls.t, cls.p["index"])
        cls.panel, cls.fits, cls.states = model.forecast_panel(cls.f, cls.t, cls.p["index"])
        cls.proof = verifier.verify_forecasts(
            cls.f, cls.t, cls.panel, cls.fits, cls.states, cls.p
        )
        cls.inference_protocol = copy.deepcopy(cls.p)
        cls.inference_protocol["inference"]["bootstrap_draws"] = 199
        with (
            patch.object(search, "inherited") as inherited,
            patch.object(search.inference, "digest", return_value="a" * 64),
        ):
            cls.metrics = search.evaluate(
                cls.panel,
                cls.f.index,
                cls.inference_protocol,
                len(cls.fits),
                prior=synthetic_prior(),
            )
        inherited.assert_not_called()
        cls.metrics["common_application_origins"] = len(cls.states)

    def verify(self, *, panel=None, fits=None, states=None):
        return verifier.verify_forecasts(
            self.f,
            self.t,
            self.panel if panel is None else panel,
            self.fits if fits is None else fits,
            self.states if states is None else states,
            self.p,
        )

    def test_complete_generated_features_fits_states_and_independent_replay(self):
        independent = verifier.preflight(self.f, self.t, self.p["index"])
        self.assertEqual(self.support, independent)
        self.assertEqual(self.proof["forecasts_verified"], len(self.panel))
        self.assertEqual(self.proof["monthly_fits_verified"], len(self.fits))
        self.assertEqual(len(self.states), self.support["common_application_origins"])

    def test_wholly_unscored_initial_month_and_all_application_audits_survive(self):
        first = pd.Timestamp(self.fits[0]["fit_origin"])
        self.assertEqual(first, pd.Timestamp(self.p["index"]["origin_start"]))
        self.assertFalse((self.panel.origin.dt.to_period("M") == first.to_period("M")).any())
        self.assertTrue((self.states.origin.dt.to_period("M") == first.to_period("M")).any())
        for fit in self.fits:
            self.assertEqual(fit["application_n"], len(fit["application_origins"]))
            self.assertEqual(
                fit["application_n"], len(fit["application_probabilities"]["location"])
            )
        with self.assertRaises((ValueError, AssertionError)):
            self.verify(fits=self.fits[1:])

    def test_coefficient_and_support_tampering_rejected(self):
        for field in ["beta", "b", "support"]:
            fits = copy.deepcopy(self.fits)
            audit = fits[0]["model_audit"]
            if field == "beta":
                audit["baseline"]["beta"][1] += 0.05
            elif field == "b":
                audit["location"]["b"] += 0.05
            else:
                audit["support"]["events"] += 1
            with self.subTest(field=field), self.assertRaises((ValueError, AssertionError)):
                self.verify(fits=fits)

    def test_probability_and_unscored_application_tampering_rejected(self):
        panel = self.panel.copy()
        selected = panel.model.eq("location")
        row = panel.index[selected][0]
        panel.loc[row, "probability"] = 0.123456
        panel.loc[row, "loss"] = (panel.loc[row, "probability"] - panel.loc[row, "y"]) ** 2
        with self.assertRaises((ValueError, AssertionError)):
            self.verify(panel=panel)
        fits = copy.deepcopy(self.fits)
        fits[0]["application_probabilities"]["baseline"][0] = 0.123456
        with self.assertRaises((ValueError, AssertionError)):
            self.verify(fits=fits)

    def test_state_seed_and_future_information_tampering_rejected(self):
        for field, value in [
            ("S", 0.999),
            ("seed_probability", 0.999),
            ("cumulative_updates", 5000),
        ]:
            states = self.states.copy()
            states.loc[states.index[0], field] = value
            with self.subTest(field=field), self.assertRaises((ValueError, AssertionError)):
                self.verify(states=states)

    def test_future_labels_cannot_change_past_fits_or_rate_states(self):
        changed = self.t.copy()
        future_origin = self.f.index[1900]
        changed.loc[future_origin, "y"] = 1 - changed.loc[future_origin, "y"]
        _, fits, states = model.forecast_panel(self.f, changed, self.p["index"])
        maturity = self.t.loc[future_origin, "available_date"]
        before = [fit for fit in self.fits if pd.Timestamp(fit["fit_cutoff_date"]) < maturity]
        after = [fit for fit in fits if pd.Timestamp(fit["fit_cutoff_date"]) < maturity]
        self.assertEqual(before, after)
        pd.testing.assert_frame_equal(
            self.states.loc[self.states.feature_cutoff_date < maturity],
            states.loc[states.feature_cutoff_date < maturity],
        )

    def verify_metrics(self, metrics):
        with (
            TemporaryDirectory() as temporary,
            patch.object(verifier, "inherited_rows", return_value=synthetic_prior()),
        ):
            return verifier.verify_metrics(
                Path(temporary),
                self.panel,
                self.inference_protocol,
                metrics,
                len(self.fits),
                len(self.states),
            )

    def test_generated_scoring_independent_inference_and_all_inherited_trials(self):
        proof = self.verify_metrics(self.metrics)
        self.assertEqual(self.p["inference"]["bootstrap_draws"], 199999)
        self.assertEqual(proof["phase_comparisons_verified"], 4)
        self.assertEqual(proof["bootstrap_runs_verified"], 12)
        self.assertEqual(proof["bootstrap_draws_per_run"], 199)
        self.assertEqual(proof["new_hypotheses_verified"], 2)
        self.assertEqual(proof["cumulative_hypotheses_verified"], 129)
        self.assertEqual(proof["calibration_rows_verified"], 6)
        self.assertEqual(self.metrics["inherited_rows"], synthetic_prior())

    def test_inference_counts_family_and_uncertainty_tampering_rejected(self):
        for field in ("applications", "family", "prior", "bootstrap"):
            metrics = copy.deepcopy(self.metrics)
            if field == "applications":
                metrics["common_application_origins"] -= 1
            elif field == "family":
                metrics["cumulative_hypothesis_count"] -= 1
            elif field == "prior":
                metrics["inherited_rows"].pop()
            else:
                metrics["rows"][0]["phases"][0]["p_conservative"] = -1.0
            with self.subTest(field=field), self.assertRaises((ValueError, AssertionError)):
                self.verify_metrics(metrics)


if __name__ == "__main__":
    unittest.main()
