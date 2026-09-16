"""Prewritten large synthetic schedule checks against the independent verifier.

These tests generate every feature and label in memory. They load no empirical
source, count, outcome, forecast, fit, or protocol artifact.
"""

import unittest
from copy import deepcopy

import numpy as np
import pandas as pd

from src import sign_memory_models as producer
from src import verify_sign_memory as verify


def synthetic_fixture(*, constant_memory=False, unscored_month=False):
    rng = np.random.default_rng(20260920)
    dates = pd.bdate_range("2010-01-04", periods=1500, name="date")
    features = pd.DataFrame(
        rng.normal(size=(len(dates), len(producer.ALL_FEATURES))),
        index=dates,
        columns=producer.ALL_FEATURES,
    )
    features["const"] = 1.0
    features["corr22"] = rng.uniform(-0.9, 0.9, len(dates))
    for column in producer.BOUNDED:
        features[column] = rng.uniform(0, 1, len(dates))
    features["qqq_pos22"] = 0.1
    features["excess22"] = 0.125 if constant_memory else rng.uniform(-0.4, 0.4, len(dates))
    features["feature_cutoff_date"] = pd.Series(dates, index=dates).shift(1)
    # Fixed generated labels are balanced without inspecting real event support.
    y = np.tile([0.0, 1.0], len(dates) // 2)
    rng.shuffle(y)
    future = pd.Series(dates, index=dates).shift(-1)
    targets = pd.DataFrame(
        {"y": y, "target_end": future, "available_date": future}, index=dates
    )
    targets.loc[dates[-1], "y"] = np.nan
    section = {
        "models": list(producer.MODELS),
        "all_features": list(producer.ALL_FEATURES),
        "minimum_train": 1000,
        "minimum_train_per_class": 50,
        "origin_start": str(dates[1370].date()),
        "origin_end": str(dates[-2].date()),
        "latest_target": str(dates[-1].date()),
        "development": [str(dates[1370].date()), str(dates[1434].date())],
        "development_target_available_by": str(dates[1434].date()),
        "evaluation": [str(dates[1435].date()), str(dates[-2].date())],
    }
    if unscored_month:
        month = dates[1370].to_period("M") + 1
        targets.loc[dates.to_period("M") == month, "y"] = np.nan
    panel, fits = producer.forecast_panel(features, targets, section)
    return features, targets, panel, fits, {"index": section}


class SignMemoryIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.complete = synthetic_fixture()
        cls.unscored = synthetic_fixture(unscored_month=True)
        cls.flat = synthetic_fixture(constant_memory=True)

    def test_full_1500_session_calendar_reconstructs_all_mature_monthly_fits(self):
        features, targets, panel, fits, protocol = self.complete
        result = verify.verify_forecasts(*self.complete)
        self.assertEqual(len(features), 1500)
        self.assertEqual(result["monthly_fits_verified"], len(fits))
        self.assertEqual(result["independent_convex_fits_verified"], 2 * len(fits))
        self.assertEqual(result["forecasts_verified"], len(panel))
        self.assertEqual(result["common_scored_origins"], panel.origin.nunique())
        self.assertEqual(
            result["common_application_origins"], sum(fit["application_n"] for fit in fits)
        )
        self.assertEqual(
            result["training_rows_reconstructed"], sum(fit["train_n"] for fit in fits)
        )
        self.assertTrue(all(fit["train_n"] >= 1000 for fit in fits))
        self.assertTrue(
            all(
                min(fit["model_audit"]["support"][name] for name in ["events", "nonevents"])
                >= 50
                for fit in fits
            )
        )
        for fit in fits:
            cutoff = pd.Timestamp(fit["fit_cutoff_date"])
            self.assertLessEqual(pd.Timestamp(fit["train_last_available"]), cutoff)
            zero_column = fit["model_audit"]["baseline"]["columns"].index("qqq_pos22")
            self.assertEqual(fit["model_audit"]["baseline"]["beta"][zero_column], 0.0)
        self.assertLessEqual(result["maximum_independent_baseline_gradient"], 1e-8 + 1e-12)
        self.assertLessEqual(result["maximum_independent_memory_gradient"], 1e-8 + 1e-12)
        boundary = pd.Timestamp(protocol["index"]["development"][1])
        self.assertTrue(targets.loc[boundary, "available_date"] > boundary)
        self.assertNotIn(boundary, set(panel.origin))

    def test_coherently_changed_probability_and_brier_cannot_replace_issued_fit(self):
        f, t, panel, fits, p = deepcopy(self.complete)
        row = panel.index[panel.model.eq("memory")][0]
        panel.loc[row, "probability"] += 0.01
        panel.loc[row, "loss"] = (panel.loc[row, "probability"] - panel.loc[row, "y"]) ** 2
        producer.validate_panel(panel)
        with self.assertRaisesRegex(AssertionError, "probability replay"):
            verify.verify_forecasts(f, t, panel, fits, p)

    def test_baseline_memory_support_and_center_audit_tampering_rejects(self):
        for fault in ["baseline", "memory", "support", "center", "class_gate"]:
            f, t, panel, fits, p = deepcopy(self.complete)
            audit = fits[0]["model_audit"]
            if fault == "baseline":
                audit["baseline"]["beta"][1] += 0.01
            elif fault == "memory":
                audit["memory"]["b"] += 0.01
            elif fault == "support":
                audit["support"]["nonevents"] += 1
            elif fault == "center":
                audit["memory"]["mean"] += 0.01
            else:
                entry = pd.Timestamp(fits[0]["fit_origin"])
                t.loc[t.index < entry, "y"] = 1.0
            with self.subTest(fault=fault), self.assertRaises((AssertionError, ValueError)):
                verify.verify_forecasts(f, t, panel, fits, p)

    def test_exact_calendar_labels_cohorts_and_monthly_metadata_required(self):
        for fault in [
            "target_calendar",
            "cohort",
            "fit_origin",
            "train_last_available",
            "application_count",
        ]:
            f, t, panel, fits, p = deepcopy(self.complete)
            if fault == "target_calendar":
                t.loc[t.index[100], "available_date"] = t.index[100]
            elif fault == "cohort":
                panel = panel.iloc[1:].copy()
            elif fault == "fit_origin":
                fits[0]["fit_origin"] = str(
                    pd.Timestamp(fits[0]["fit_origin"]) + pd.Timedelta(days=1)
                )
            elif fault == "train_last_available":
                panel.loc[0, "train_last_available"] = panel.loc[0, "origin"]
            else:
                fits[0]["application_n"] -= 1
            with self.subTest(fault=fault), self.assertRaises((AssertionError, ValueError)):
                verify.verify_forecasts(f, t, panel, fits, p)

    def test_exact_constant_memory_remains_all_months_and_all_forecasts(self):
        f, t, panel, fits, p = self.flat
        result = verify.verify_forecasts(f, t, panel, fits, p)
        self.assertEqual(result["constant_memory_fits_retained"], len(fits))
        wide = panel.pivot(index="origin", columns="model", values="probability")
        np.testing.assert_array_equal(wide.baseline, wide.memory)
        self.assertTrue(all(fit["model_audit"]["memory"]["b"] == 0 for fit in fits))
        self.assertEqual(set(panel.model), {"frequency", "baseline", "memory"})

    def test_entire_unscored_month_is_independently_refitted_and_cannot_be_removed(self):
        result = verify.verify_forecasts(*self.unscored)
        f, t, panel, fits, p = deepcopy(self.unscored)
        missing = [
            index
            for index, fit in enumerate(fits)
            if pd.Timestamp(fit["fit_origin"]) not in set(panel.fit_origin)
        ]
        self.assertEqual(len(missing), 1)
        self.assertGreater(
            result["common_application_origins"], result["common_scored_origins"]
        )
        self.assertEqual(result["monthly_fits_verified"], len(fits))
        for fault in ["remove", "memory"]:
            altered = deepcopy(fits)
            if fault == "remove":
                del altered[missing[0]]
            else:
                altered[missing[0]]["model_audit"]["memory"]["b"] += 0.02
            with self.subTest(fault=fault), self.assertRaises((AssertionError, ValueError)):
                verify.verify_forecasts(f, t, panel, altered, p)


if __name__ == "__main__":
    unittest.main()
