"""Pre-fit contracts for the measurement candidate family and causal fitting."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import measurement_search as study


class TestMeasurementSearch(unittest.TestCase):
    def fixture(self):
        dates = pd.bdate_range("2010-01-01", periods=170)
        rng = np.random.default_rng(29)
        features = pd.DataFrame(rng.normal(size=(170, len(study.mm.ALL_FEATURES))), index=dates,
                                columns=study.mm.ALL_FEATURES)
        features["const"] = 1.
        features["market_cutoff_date"] = pd.Series(dates, index=dates).shift()
        features["measurement_cutoff_date"] = pd.Series(dates, index=dates).shift(2)
        targets = pd.DataFrame({"y_qmle": rng.lognormal(-3, .5, 170),
                                "y_rv5": rng.lognormal(-3, .5, 170),
                                "y_rv15": rng.lognormal(-3, .5, 170),
                                "target_end": pd.Series(dates, index=dates).shift(-1),
                                "available_date": pd.Series(dates, index=dates).shift(-3)}, index=dates)
        targets.loc[dates[-1], ["y_qmle", "y_rv5", "y_rv15"]] = np.nan
        return features, targets

    def test_training_requires_all_three_mature_labels(self):
        features, targets = self.fixture()
        entry = features.index[100]
        selected = study.training_mask(features, targets, entry, 10)
        self.assertEqual(features.index[selected][-1], features.index[96])
        targets.loc[features.index[80], "y_rv5"] = np.nan
        selected = study.training_mask(features, targets, entry, 10)
        self.assertFalse(selected.iloc[80])
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"):
            study.training_mask(features, targets, entry, 100)

    def test_positive_alternate_values_do_not_change_primary_fits(self):
        features, targets = self.fixture()
        train, apply = features.iloc[:120], features.iloc[120:]
        first, audit = study.fit_predict(train, targets.iloc[:120], apply)
        mutated = targets.iloc[:120].copy()
        mutated["y_rv5"] *= 300
        mutated["y_rv15"] /= 40
        second, other_audit = study.fit_predict(train, mutated, apply)
        for model in study.mm.MODELS:
            np.testing.assert_array_equal(first[model], second[model])
        self.assertEqual(audit, other_audit)

    def test_target_unit_change_rescales_all_predictions(self):
        features, targets = self.fixture()
        first, _ = study.fit_predict(features.iloc[:120], targets.iloc[:120], features.iloc[120:])
        changed = targets.iloc[:120].copy()
        changed.loc[:, study.TARGETS] *= 10000
        second, _ = study.fit_predict(features.iloc[:120], changed, features.iloc[120:])
        for model in study.mm.MODELS:
            np.testing.assert_allclose(second[model], first[model]*10000, rtol=1e-10)

    def test_zero_scale_and_target_misalignment_reject_whole_family(self):
        features, targets = self.fixture()
        with self.assertRaises(ValueError):
            study.fit_predict(features.iloc[:120], targets.iloc[:120].iloc[::-1], features.iloc[120:])
        features["quality_memory"] = 0.
        with self.assertRaisesRegex(ValueError, "zero-scale"):
            study.fit_predict(features.iloc[:120], targets.iloc[:120], features.iloc[120:])

    def test_future_first_label_does_not_choose_monthly_refit(self):
        features, targets = self.fixture()
        dates = features.index
        config = {"origin_start": str(dates[80].date()), "origin_end": str(dates[150].date()),
                  "development": [str(dates[80].date()), str(dates[110].date())],
                  "evaluation": [str(dates[111].date()), str(dates[150].date())],
                  "development_target_available_by": str(dates[110].date()),
                  "latest_available": str(dates[-1].date()), "minimum_train": 10,
                  "models": list(study.mm.MODELS), "baseline": list(study.mm.BASE)}
        first_entry = dates[80]
        def fake_fit(train, target, application):
            return ({model: np.full(len(application), 1.) for model in study.mm.MODELS}, {})
        with patch.object(study, "fit_predict", side_effect=fake_fit):
            first, fits = study.forecast_panel(features, targets, config)
            targets.loc[first_entry, "y_rv15"] = np.nan
            second, other = study.forecast_panel(features, targets, config)
        self.assertEqual(fits[0]["fit_origin"], str(first_entry.date()))
        self.assertEqual([fit["fit_origin"] for fit in fits], [fit["fit_origin"] for fit in other])
        self.assertIn(first_entry, first.origin.to_list())
        self.assertNotIn(first_entry, second.origin.to_list())
        self.assertEqual(set(second.model), set(study.mm.MODELS))

    def test_all_fifteen_comparisons_and_failures_remain(self):
        self.assertEqual(len(study.CONTRASTS), 15)
        self.assertEqual(len(set(study.CONTRASTS)), 15)
        failed = study.failure_metrics(ValueError("INSUFFICIENT_DATA: gap"), "0"*64)
        self.assertEqual(len(failed["rows"]), 15)
        self.assertTrue(all(row["p_conservative"] == 1 for row in failed["rows"]))
        self.assertEqual(failed["cumulative_hypothesis_count"], 99)
        self.assertEqual(failed["leads"], [])

    def test_candidate_needs_each_primary_control_and_all_alternate_signs(self):
        rows = []
        for candidate, control, measure in study.CONTRASTS:
            rows.append({"candidate": candidate, "control": control, "measure": measure,
                         "p_holm_wave": .0001, "p_holm_cumulative": .01,
                         "phases": [{"name": "development", "delta": -.01, "n": 140},
                                    {"name": "evaluation", "delta": -.01, "n": 140,
                                     "stability": [{"delta": -.01}, {"delta": -.01}]}]})
        self.assertEqual(study.candidate_leads(rows), ["width", "quality"])
        changed = copy.deepcopy(rows)
        next(row for row in changed if row["candidate"] == "quality" and row["control"] == "width"
             and row["measure"] == "rv15")["phases"][0]["delta"] = 0.
        self.assertEqual(study.candidate_leads(changed), ["width"])
        changed = copy.deepcopy(rows)
        next(row for row in changed if row["candidate"] == "width" and row["measure"] == "qmle")["p_holm_wave"] = 1/600
        self.assertEqual(study.candidate_leads(changed), ["quality"])
        changed = copy.deepcopy(rows)
        for row in changed:
            if row["measure"] != "qmle":
                row["p_holm_wave"] = 1.
        self.assertEqual(study.candidate_leads(changed), ["width", "quality"])

    def test_incomplete_family_cannot_pass(self):
        with self.assertRaises(ValueError):
            study.candidate_leads([])
        self.assertLess(1/(99999+1), (1/600)/15/10)

    def test_each_measurement_scores_identical_frozen_predictions(self):
        dates = pd.bdate_range("2016-01-01", periods=280)
        panel = []
        for model, prediction in {"mean": 1., "baseline": 1.1, "width": 1.2, "quality": 1.3}.items():
            panel.append(pd.DataFrame({"origin": dates, "model": model, "horizon": 1,
                                       "prediction": prediction, "y_qmle": 1., "y_rv5": 2., "y_rv15": 3.,
                                       "target_end": dates, "available_date": dates, "fit_origin": dates[0],
                                       "market_cutoff_date": dates, "measurement_cutoff_date": dates}))
        panel = pd.concat(panel, ignore_index=True)
        p = {"index": {"development": [str(dates[0].date()), str(dates[139].date())],
                        "development_target_available_by": str(dates[139].date()),
                        "evaluation": [str(dates[140].date()), str(dates[-1].date())],
                        "evaluation_stability": [[str(dates[140].date()), str(dates[209].date())],
                                                 [str(dates[210].date()), str(dates[-1].date())]]},
             "inference": {"blocks": [21, 63, 126], "bootstrap_draws": 199, "seed": 53},
             "evidence_class": "synthetic"}
        with patch.object(study, "inherited", return_value=[{"p_conservative": 1.}]*84):
            metrics = study.evaluate(panel, dates, p)
        self.assertEqual(len(metrics["rows"]), 15)
        for measure, actual in {"qmle": 1., "rv5": 2., "rv15": 3.}.items():
            row = next(row for row in metrics["rows"] if row["candidate"] == "quality"
                       and row["control"] == "width" and row["measure"] == measure)
            expected = np.log(1.3)-np.log(1.2)+actual/1.3-actual/1.2
            for phase in row["phases"]:
                self.assertEqual(phase["n"], 140)
                self.assertAlmostEqual(phase["delta"], expected)


if __name__ == "__main__":
    unittest.main()
