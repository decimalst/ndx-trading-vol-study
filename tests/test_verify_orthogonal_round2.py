"""Independent synthetic acceptance and corruption tests for the score verifier.

The valid fixture is assembled directly from definitions, without importing the
producer or using any verifier calculation to manufacture its expected inputs.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src import verify_orthogonal_round2 as verifier

BASELINE = [
    "const", "lrv_d", "lrv_w", "lrv_m", "lev_d", "lev_w", "lev_m",
    "liv", "lvix", "term", "xasset_stress", "market_stress",
]
CANDIDATES = ["vvix", "rv_dispersion", "stock_bond_corr", "close_pressure"]


def direct_fixture():
    """Generate all five arms, two horizons and monthly refits from scratch."""
    rng = np.random.default_rng(61083)
    dates = pd.bdate_range("2022-01-03", periods=110, name="date")
    features = pd.DataFrame(rng.normal(size=(len(dates), 16)), index=dates,
                            columns=BASELINE + CANDIDATES)
    features["const"] = 1.0
    features["rv_total"] = np.exp(-8 + rng.normal(0, 0.5, len(dates)))
    # One missing training input and one missing origin establish the common
    # complete-case population independently of any arm's own input columns.
    features.loc[dates[20], "liv"] = np.nan
    features.loc[dates[60], "vvix"] = np.nan
    protocol = {
        "horizons": [1, 5], "minimum_training_rows": 30,
        "origin_start": str(dates[45].date()), "origin_end": str(dates[-6].date()),
        "latest_target": str(dates[-1].date()), "sealed_start": "2023-01-01",
        "inference": {"stability_periods": [["2022-01-01", "2022-03-31"],
                                              ["2022-04-01", "2022-04-30"],
                                              ["2022-05-01", "2022-06-30"]]},
    }
    complete = np.isfinite(features[BASELINE + CANDIDATES]).all(axis=1).to_numpy()
    all_rows = []
    for horizon in protocol["horizons"]:
        target_y = np.full(len(dates), np.nan)
        target_end = [pd.NaT] * len(dates)
        for position in range(len(dates) - horizon):
            target_y[position] = np.mean(features["rv_total"].iloc[position + 1:position + horizon + 1])
            target_end[position] = dates[position + horizon]
        eligible = [i for i in range(45, len(dates) - 5) if complete[i]]
        months = {}
        for i in eligible:
            months.setdefault((dates[i].year, dates[i].month), []).append(i)
        for positions in months.values():
            fit_position, fit_origin = positions[0], dates[positions[0]]
            train_positions = [
                i for i in range(fit_position) if complete[i] and np.isfinite(target_y[i])
                and pd.notna(target_end[i]) and target_end[i] <= fit_origin
            ]
            if len(train_positions) < protocol["minimum_training_rows"]:
                continue
            log_target = np.log(target_y[train_positions])
            for model in ["baseline", *CANDIDATES]:
                columns = BASELINE + ([] if model == "baseline" else [model])
                training_design = features.iloc[train_positions][columns].to_numpy()
                coefficient = np.linalg.lstsq(training_design, log_target, rcond=None)[0]
                errors = log_target - training_design @ coefficient
                smear = sum(np.exp(error) for error in errors) / len(errors)
                for i in positions:
                    estimate = np.exp(features.iloc[i][columns].to_numpy() @ coefficient) * smear
                    all_rows.append({
                        "origin": dates[i], "horizon": horizon, "model": model,
                        "target_end": target_end[i], "y": target_y[i], "prediction": estimate,
                        "fit_origin": fit_origin, "train_n": len(train_positions),
                        "train_last_target": max(target_end[j] for j in train_positions),
                    })
    return features, pd.DataFrame(all_rows), protocol


class IndependentForecastVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.features, cls.forecasts, cls.protocol = direct_fixture()

    def verify(self, forecasts):
        return verifier.verify_forecasts(self.features, forecasts, self.protocol)

    def test_accepts_independent_direct_ols_and_exact_smearing_for_every_arm(self):
        result = self.verify(self.forecasts)
        self.assertEqual(result["independent_forecast_verification"], "PASS")
        self.assertEqual(result["n"], len(self.forecasts))
        self.assertEqual(result["monthly_model_fits"], 30)
        self.assertEqual(len(result["comparisons"]), 8)
        self.assertLess(result["max_absolute_error"], 1e-14)
        self.assertLess(result["max_relative_error"], 1e-10)
        expected_origins = self.forecasts.loc[self.forecasts["horizon"] == 1, "origin"].nunique()
        self.assertTrue(all(row["n"] == expected_origins for row in result["comparisons"]))
        missing_candidate_date = self.features.index[60]
        self.assertFalse((self.forecasts["origin"] == missing_candidate_date).any())

    def test_acceptance_is_independent_of_saved_forecast_order(self):
        shuffled = self.forecasts.sample(frac=1, random_state=42).reset_index(drop=True)
        self.assertEqual(self.verify(shuffled), self.verify(self.forecasts))

    def test_rejects_missing_single_arm_row_and_missing_entire_common_origin(self):
        with self.assertRaises(AssertionError):
            self.verify(self.forecasts.drop(index=self.forecasts.index[0]))
        first_origin = self.forecasts["origin"].min()
        with self.assertRaises(AssertionError):
            self.verify(self.forecasts.loc[self.forecasts["origin"] != first_origin])

    def test_rejects_duplicate_forecast_key(self):
        duplicated = pd.concat([self.forecasts, self.forecasts.iloc[[0]]], ignore_index=True)
        with self.assertRaises(AssertionError):
            self.verify(duplicated)

    def test_rejects_unknown_model_and_missing_model_family(self):
        wrong = self.forecasts.copy()
        wrong.loc[wrong.index[0], "model"] = "unregistered_signal"
        with self.assertRaises(AssertionError):
            self.verify(wrong)
        with self.assertRaises(AssertionError):
            self.verify(self.forecasts.loc[self.forecasts["model"] != "vvix"])

    def test_rejects_tampered_prediction_and_target_values(self):
        for column in ["prediction", "y"]:
            with self.subTest(column=column):
                wrong = self.forecasts.copy()
                wrong.loc[wrong.index[0], column] *= 1.1
                with self.assertRaises(AssertionError):
                    self.verify(wrong)

    def test_rejects_tampered_training_count_fit_date_and_target_timestamps(self):
        for column in ["train_n", "fit_origin", "train_last_target", "target_end"]:
            with self.subTest(column=column):
                wrong = self.forecasts.copy()
                increment = 1 if column == "train_n" else pd.Timedelta(days=1)
                wrong.loc[wrong.index[0], column] += increment
                with self.assertRaises(AssertionError):
                    self.verify(wrong)

    def test_rejects_invalid_positive_finite_contract(self):
        for value in [0.0, -1.0, np.nan, np.inf]:
            for column in ["prediction", "y"]:
                with self.subTest(value=value, column=column):
                    wrong = self.forecasts.copy()
                    wrong.loc[wrong.index[0], column] = value
                    with self.assertRaises(AssertionError):
                        self.verify(wrong)

    def test_rejects_foreign_horizon_origin_and_missing_required_column(self):
        wrong_horizon = self.forecasts.copy()
        wrong_horizon.loc[wrong_horizon.index[0], "horizon"] = 21
        with self.assertRaises(AssertionError):
            self.verify(wrong_horizon)
        wrong_origin = self.forecasts.copy()
        wrong_origin.loc[wrong_origin.index[0], "origin"] = pd.Timestamp("2023-01-03")
        with self.assertRaises(AssertionError):
            self.verify(wrong_origin)
        with self.assertRaises(AssertionError):
            self.verify(self.forecasts.drop(columns="train_last_target"))


if __name__ == "__main__":
    unittest.main()
