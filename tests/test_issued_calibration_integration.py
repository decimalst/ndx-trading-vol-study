"""Generated original issuance -> calibration -> independent proof and inference."""

import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import issued_calibration_models as model
from src import issued_calibration_search as search
from src import range_alert_models as original
from src import verify_issued_calibration as verify
from tests.test_range_alert_models import fixture


def synthetic_prior():
    return [
        {
            "study": "generated_previous_trials",
            "candidate": str(number),
            "control": "baseline",
            "horizon": 1,
            "p_conservative": 1.0,
        }
        for number in range(129)
    ]


class IssuedCalibrationIntegration(unittest.TestCase):
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
        cls.p = copy.deepcopy(search.CONTRACT)
        cls.p["index"] = {**cls.p["index"], **cls.config, "models": list(model.MODELS)}
        cls.p["inference"]["bootstrap_draws"] = 199

    def verify(self, *, panel=None, states=None, audit=None):
        return verify.verify_forecasts(
            self.f,
            self.t,
            self.old_panel,
            self.old_fits,
            self.old_states,
            self.panel if panel is None else panel,
            self.states if states is None else states,
            self.audit if audit is None else audit,
            self.p,
        )

    def test_every_original_issue_and_new_scalar_state_independently_reconstructed(self):
        proof = self.verify()
        self.assertEqual(proof["common_application_origins"], len(self.states))
        self.assertEqual(proof["common_scored_origins"], self.panel.origin.nunique())
        self.assertEqual(self.audit["replayed_monthly_fits"], len(self.old_fits))
        self.assertEqual(len(self.audit["calibrations"]), len(self.states))

    def test_unscored_mature_record_and_whole_feature_gap_have_different_policies(self):
        unscored_origin = pd.Timestamp(self.config["development"][1])
        available = self.t.loc[unscored_origin, "available_date"]
        issued = pd.DataFrame(self.audit["issued_applications"])
        issued.origin = pd.to_datetime(issued.origin)
        self.assertFalse(issued.set_index("origin").loc[unscored_origin, "scored"])
        arrival = pd.DataFrame(self.audit["arrival_audit"]).set_index("origin")
        self.assertEqual(arrival.loc[str(unscored_origin.date()), "status"], "ADMITTED")
        before = self.states.loc[self.states.feature_cutoff_date < available].iloc[-1]
        after = self.states.loc[self.states.feature_cutoff_date >= available].iloc[0]
        self.assertEqual(after.history_n, before.history_n + 1)
        self.assertEqual(after.latest_admitted_origin, unscored_origin)
        missing_month = self.f.index[
            self.f.index.to_period("M") == self.f.index[1300].to_period("M")
        ]
        self.assertFalse(issued.origin.isin(missing_month).any())
        for day in missing_month:
            self.assertEqual(arrival.loc[str(day.date()), "status"], "NO_ISSUED_FORECAST")

    def test_later_labels_or_features_cannot_change_past_fits_issued_logits_or_states(self):
        for mutation in ("future_label", "future_features"):
            f, t = self.f.copy(), self.t.copy()
            date = self.f.index[1900]
            if mutation == "future_label":
                t.loc[date, "y"] = 1 - t.loc[date, "y"]
                limit = t.loc[date, "available_date"]
            else:
                f.loc[date:, "I"] += 0.75
                limit = date
            old_panel, fits, old_states = original.forecast_panel(f, t, self.config)
            _, states, audit = model.forecast_panel(
                f, t, old_panel, fits, old_states, self.config
            )
            before = [
                one for one in self.old_fits if pd.Timestamp(one["fit_cutoff_date"]) < limit
            ]
            after = [one for one in fits if pd.Timestamp(one["fit_cutoff_date"]) < limit]
            # Application forecasts from a fit can change after the mutated
            # query date; fitted coefficients before its maturity cannot.
            with self.subTest(mutation=mutation):
                self.assertEqual(
                    [one["model_audit"] for one in before],
                    [one["model_audit"] for one in after],
                )
                self.assertNotEqual(
                    self.old_fits[-1]["model_audit"]["baseline"]["beta"],
                    fits[-1]["model_audit"]["baseline"]["beta"],
                )
                pd.testing.assert_frame_equal(
                    self.states.loc[
                        (self.states.feature_cutoff_date < limit) & (self.states.origin < date)
                    ],
                    states.loc[(states.feature_cutoff_date < limit) & (states.origin < date)],
                    check_exact=True,
                )
                original_issued = [
                    one
                    for one in self.audit["issued_applications"]
                    if pd.Timestamp(one["origin"]) < date
                ]
                changed_issued = [
                    one
                    for one in audit["issued_applications"]
                    if pd.Timestamp(one["origin"]) < date
                ]
                self.assertEqual(original_issued, changed_issued)

    def test_independent_rejection_of_scalar_history_and_arrival_tampering(self):
        for field in ("coefficient", "weight", "record", "arrival"):
            audit = copy.deepcopy(self.audit)
            if field == "coefficient":
                audit["calibrations"][20]["intercept"] += 0.02
            elif field == "weight":
                audit["calibrations"][20]["weight_sum"] += 0.01
            elif field == "record":
                audit["admitted_records"][0]["y"] = 1 - audit["admitted_records"][0]["y"]
            else:
                audit["arrival_audit"][0]["status"] = "ADMITTED"
            with self.subTest(field=field), self.assertRaises((AssertionError, ValueError)):
                self.verify(audit=audit)

    def test_independent_rejection_of_coherent_probability_or_state_change(self):
        panel = self.panel.copy()
        selected = panel.model.eq("calibrated")
        panel.loc[selected, "probability"] = 0.123456
        panel.loc[selected, "loss"] = original.brier_loss(
            panel.loc[selected, "probability"].to_numpy(), panel.loc[selected, "y"].to_numpy()
        )
        with self.assertRaises((AssertionError, ValueError)):
            self.verify(panel=panel)
        states = self.states.copy()
        states.loc[states.index[0], "no_forecast_arrivals"] += 1
        with self.assertRaises((AssertionError, ValueError)):
            self.verify(states=states)

    def scored_metrics(self):
        with (
            patch.object(search, "inherited") as inherited,
            patch.object(search.inference, "digest", return_value="a" * 64),
        ):
            metrics = search.evaluate(
                self.panel, self.f.index, self.p, 0, prior=synthetic_prior()
            )
        inherited.assert_not_called()
        metrics["common_application_origins"] = len(self.states)
        return metrics

    def verify_metrics(self, metrics):
        with (
            TemporaryDirectory() as temporary,
            patch.object(verify, "inherited_rows", return_value=synthetic_prior()),
        ):
            return verify.verify_metrics(
                Path(temporary), self.panel, self.p, metrics, len(self.states)
            )

    def test_real_generated_scoring_and_independent_131_family_inference(self):
        metrics = self.scored_metrics()
        proof = self.verify_metrics(metrics)
        self.assertEqual(search.CONTRACT["inference"]["bootstrap_draws"], 199999)
        self.assertEqual(proof["phase_comparisons_verified"], 4)
        self.assertEqual(proof["bootstrap_runs_verified"], 12)
        self.assertEqual(proof["bootstrap_draws_per_run"], 199)
        self.assertEqual(proof["cumulative_hypotheses_verified"], 131)
        self.assertEqual(metrics["new_monthly_fits"], 0)
        self.assertEqual(metrics["new_forecasts"], self.panel.origin.nunique())
        self.assertEqual(metrics["reused_control_forecasts"], 2 * self.panel.origin.nunique())
        self.assertEqual(metrics["inherited_rows"], synthetic_prior())

    def test_independent_family_count_and_phase_uncertainty_tampering_rejected(self):
        metrics = self.scored_metrics()
        for field in ("family", "application", "phase"):
            changed = copy.deepcopy(metrics)
            if field == "family":
                changed["inherited_rows"].pop()
            elif field == "application":
                changed["common_application_origins"] -= 1
            else:
                changed["rows"][0]["phases"][0]["p_conservative"] = -1.0
            with self.subTest(field=field), self.assertRaises((AssertionError, ValueError)):
                self.verify_metrics(changed)


if __name__ == "__main__":
    unittest.main()
