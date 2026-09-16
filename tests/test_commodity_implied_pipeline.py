"""Prospective invented-data tests for fixed commodity monthly scheduling."""

from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.commodity_implied_models import ALL, fit_models
from src.commodity_implied_pipeline import InsufficientDataError, build_panel

ARMS = ("market", "matched", "candidate")


def synthetic_inputs(calendar=None):
    """Invented already-built features; no source files or market values."""
    if calendar is None:
        calendar = pd.bdate_range("2011-01-03", "2016-03-10").as_unit("ns")
    rng = np.random.default_rng(917331)
    features = pd.DataFrame(
        rng.normal(size=(len(calendar), len(ALL))), index=calendar, columns=ALL
    )
    features["const"] = 1.0
    cutoff = pd.Series(calendar, index=calendar).shift(1)
    features["commodity_cutoff_date"] = cutoff.where(
        cutoff >= pd.Timestamp("2009-01-02")
    ).astype("datetime64[ns]")
    features.loc[features.commodity_cutoff_date.isna(), "lovx"] = np.nan
    y = np.exp(-7.0 + 0.12 * rng.normal(size=len(calendar)))
    end = pd.Series(calendar, index=calendar).shift(-5)
    y[end.isna()] = np.nan
    targets = pd.DataFrame({"y": y, "target_end": end}, index=calendar)
    config = {
        "origin_start": "2016-01-01",
        "origin_end": "2016-03-10",
        "development": ["2016-01-01", "2016-01-31"],
        "evaluation": ["2016-02-01", "2016-03-10"],
        "source_end": "2016-03-10",
        "minimum_train": 1000,
    }
    return features, targets, config


def mock_models(train, y, application):
    base = float(y.mean())
    return {
        "predictions": {
            arm: np.full(len(application), base * factor)
            for arm, factor in zip(ARMS, (1.0, 1.1, 1.2))
        },
        "fits": {
            arm: {"n_train": len(train), "n_application": len(application)} for arm in ARMS
        },
    }


class CommodityPipelineTests(unittest.TestCase):
    def run_mock(self, features, targets, config):
        with patch(
            "src.commodity_implied_pipeline.fit_models", side_effect=mock_models
        ) as model:
            result = build_panel(features, targets, config)
        return result, model

    def test_earliest_feature_complete_query_is_label_blind(self):
        features, targets, config = synthetic_inputs()
        january = features.index[
            (features.index >= "2016-01-01") & (features.index < "2016-02-01")
        ]
        features.loc[january[0], "lgvz"] = np.nan
        targets.loc[january[1], "y"] = np.nan
        result, model = self.run_mock(features, targets, config)
        first = result["fits"][0]
        self.assertEqual(first["fit_origin"], january[1].date().isoformat())
        self.assertEqual(model.call_args_list[0].args[2].index[0], january[1])
        row = result["coverage"].set_index("origin").loc[january[1]]
        self.assertEqual(row.status, "missing_target")
        self.assertEqual(row.fit_origin, january[1])

    def test_common_training_rows_and_strict_previous_session_maturity(self):
        features, targets, config = synthetic_inputs()
        query = features.index[features.index >= config["origin_start"]][0]
        p = features.index.get_loc(query)
        hole = features.index[p - 70]
        features.loc[hole, "lgvz"] = np.nan
        result, model = self.run_mock(features, targets, config)
        actual_train, actual_y, actual_app = model.call_args_list[0].args
        expected = features.index[1 : p - 5].difference(pd.DatetimeIndex([hole]))
        self.assertTrue(actual_train.index.equals(expected))
        self.assertTrue(actual_y.index.equals(expected))
        self.assertIn(features.index[p - 6], actual_train.index)
        self.assertNotIn(features.index[p - 5], actual_train.index)
        self.assertNotIn(hole, actual_train.index)
        self.assertEqual(
            result["fits"][0]["training_cutoff"], features.index[p - 1].date().isoformat()
        )
        self.assertEqual(
            result["fits"][0]["train_origins"], [x.date().isoformat() for x in expected]
        )
        self.assertEqual(
            result["fits"][0]["application_origins"],
            [x.date().isoformat() for x in actual_app.index],
        )

    def test_future_labels_do_not_change_first_fit_or_predictions(self):
        features, targets, config = synthetic_inputs()
        before, _ = self.run_mock(features, targets, config)
        poisoned = targets.copy(deep=True)
        query = pd.Timestamp(before["fits"][0]["fit_origin"])
        mask = poisoned.target_end >= query
        poisoned.loc[mask & poisoned.y.notna(), "y"] *= 123.0
        after, _ = self.run_mock(features, poisoned, config)
        first_month = before["applications"].origin.dt.month == 1
        pd.testing.assert_frame_equal(
            before["applications"][first_month], after["applications"][first_month]
        )
        self.assertEqual(before["fits"][0], after["fits"][0])

    def test_all_requested_origins_and_full_calendar_offsets_survive_holes(self):
        features, targets, config = synthetic_inputs()
        requested = features.index[features.index >= config["origin_start"]]
        holes = requested[[2, 7, 24]]
        features.loc[holes, ["uso_lrv22", "lovx"]] = np.nan
        result, _ = self.run_mock(features, targets, config)
        coverage = result["coverage"].set_index("origin")
        self.assertTrue(coverage.index.equals(requested))
        self.assertTrue((coverage.loc[holes, "missing_features"] == "uso_lrv22|lovx").all())
        self.assertTrue(coverage.loc[holes, "fit_origin"].isna().all())
        for origin, row in coverage.iterrows():
            self.assertEqual(row.offset, features.index.get_loc(origin) % 5)
        self.assertEqual(set(result["applications"].origin), set(requested) - set(holes))
        self.assertEqual(
            result["panel"].groupby("origin").model.apply(list).tolist(),
            [list(ARMS)] * result["panel"].origin.nunique(),
        )

    def test_unscored_applications_and_censoring_precedence(self):
        features, targets, config = synthetic_inputs()
        config["development"][1] = "2016-01-20"
        config["evaluation"][0] = "2016-02-03"
        targets.loc["2016-01-08", "y"] = np.nan
        features.loc["2016-01-12", "lovx"] = np.nan
        result, _ = self.run_mock(features, targets, config)
        coverage = result["coverage"].set_index("origin")
        expected = {
            "2016-01-08": "missing_target",
            "2016-01-12": "incomplete_features",
            "2016-01-19": "target_after_phase_cutoff",
            "2016-01-25": "outside_phase",
            "2016-03-10": "target_not_mature",
            "2016-02-10": "scored",
        }
        for origin, status in expected.items():
            self.assertEqual(coverage.loc[origin, "status"], status)
            self.assertEqual(coverage.loc[origin, "scored"], status == "scored")
            if status != "incomplete_features":
                self.assertIn(pd.Timestamp(origin), set(result["applications"].origin))
        self.assertEqual(len(result["panel"]), 3 * int(coverage.scored.sum()))

    def test_evaluation_target_may_end_after_evaluation_origin_end(self):
        features, targets, config = synthetic_inputs()
        config["origin_end"] = "2016-02-29"
        config["evaluation"][1] = "2016-02-29"
        result, _ = self.run_mock(features, targets, config)
        last = result["coverage"].set_index("origin").loc["2016-02-29"]
        self.assertEqual(last.target_end, pd.Timestamp("2016-03-07"))
        self.assertEqual(last.status, "scored")

    def test_empty_requested_and_all_incomplete_months_are_explicit(self):
        calendar = pd.bdate_range("2011-01-03", "2016-03-10").as_unit("ns")
        calendar = calendar[~((calendar >= "2016-02-01") & (calendar < "2016-03-01"))]
        features, targets, config = synthetic_inputs(calendar)
        features.loc["2016-01-01":"2016-01-31", "lgvz"] = np.nan
        result, model = self.run_mock(features, targets, config)
        schedule = result["schedules"].set_index("month")
        self.assertEqual(schedule.loc["2016-01", "status"], "no_complete_origin")
        self.assertEqual(schedule.loc["2016-02", "status"], "no_requested_origins")
        self.assertEqual(schedule.loc["2016-02", "requested_n"], 0)
        self.assertTrue(schedule.loc[["2016-01", "2016-02"], "train_n"].isna().all())
        self.assertEqual(model.call_count, 1)

    def test_insufficient_month_aborts_attempt_with_full_coverage(self):
        features, targets, config = synthetic_inputs()
        config["minimum_train"] = len(features)
        features.loc["2016-01-04", "lovx"] = np.nan
        with (
            patch(
                "src.commodity_implied_pipeline.fit_models", side_effect=mock_models
            ) as model,
            self.assertRaisesRegex(InsufficientDataError, "INSUFFICIENT_DATA") as caught,
        ):
            build_panel(features, targets, config)
        failure = caught.exception
        self.assertIsInstance(failure, ValueError)
        self.assertFalse(failure.coverage.scored.any())
        self.assertEqual(
            len(failure.coverage), int((features.index >= config["origin_start"]).sum())
        )
        self.assertEqual(
            failure.coverage.set_index("origin").loc["2016-01-04", "status"],
            "incomplete_features",
        )
        self.assertEqual(
            set(failure.coverage.loc[failure.coverage.feature_complete, "status"]),
            {"attempt_aborted"},
        )
        self.assertEqual(
            failure.schedules.status.tolist(),
            ["insufficient_training", "not_attempted", "not_attempted"],
        )
        self.assertFalse(hasattr(failure, "predictions"))
        model.assert_not_called()

    def test_unscored_forecast_validation_and_model_failure_are_fatal(self):
        features, targets, config = synthetic_inputs()
        config["origin_start"] = "2016-03-01"
        config["development"] = ["2016-03-01", "2016-03-02"]
        config["evaluation"] = ["2016-03-03", "2016-03-10"]
        for invalid in (0.0, -1.0, np.inf, np.nan):

            def invalid_last(train, y, application, invalid=invalid):
                result = mock_models(train, y, application)
                result["predictions"]["candidate"][-1] = invalid
                return result

            with (
                self.subTest(invalid=invalid),
                patch("src.commodity_implied_pipeline.fit_models", side_effect=invalid_last),
                self.assertRaises(ValueError),
            ):
                build_panel(features, targets, config)
        with (
            patch(
                "src.commodity_implied_pipeline.fit_models",
                side_effect=ValueError("rank failure"),
            ),
            self.assertRaisesRegex(ValueError, "rank failure"),
        ):
            build_panel(features, targets, config)

    def test_qlike_matches_literal_formula_and_overflow_aborts(self):
        features, targets, config = synthetic_inputs()
        result, _ = self.run_mock(features, targets, config)
        panel = result["panel"]
        ratio = panel.y.to_numpy() / panel.prediction.to_numpy()
        np.testing.assert_array_equal(panel.loss.to_numpy(), ratio - np.log(ratio) - 1.0)
        targets.loc["2016-01-06", "y"] = np.finfo(float).max

        def tiny(train, y, application):
            result = mock_models(train, y, application)
            result["predictions"] = {
                arm: np.full(len(application), np.finfo(float).tiny) for arm in ARMS
            }
            return result

        with (
            patch("src.commodity_implied_pipeline.fit_models", side_effect=tiny),
            self.assertRaises(ValueError),
        ):
            build_panel(features, targets, config)

    def test_exact_target_endpoint_and_prior_full_calendar_commodity_cutoff(self):
        features, targets, config = synthetic_inputs()
        for name in ("target_end", "commodity_cutoff_date"):
            f, t = features.copy(deep=True), targets.copy(deep=True)
            frame = t if name == "target_end" else f
            frame.loc["2016-01-08", name] = pd.Timestamp("2016-01-05")
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.run_mock(f, t, config)
        f = features.copy(deep=True)
        f.loc["2016-01-08", "commodity_cutoff_date"] = pd.NaT
        with self.assertRaises(ValueError):
            self.run_mock(f, targets, config)

    def test_pre_source_floor_cutoff_is_unknown_and_not_complete(self):
        calendar = pd.bdate_range("2008-12-29", "2009-04-10").as_unit("ns")
        features, targets, config = synthetic_inputs(calendar)
        config.update(
            origin_start="2009-01-02",
            origin_end="2009-04-10",
            development=["2009-01-02", "2009-02-27"],
            evaluation=["2009-03-02", "2009-04-10"],
            source_end="2009-04-10",
            minimum_train=1,
        )
        # Unknown commodity cutoff cannot masquerade as a fully observed row.
        features.loc["2009-01-02", "lovx"] = 0.5
        with self.assertRaises(ValueError):
            self.run_mock(features, targets, config)

    def test_numeric_and_calendar_validation(self):
        features, targets, config = synthetic_inputs()
        for field, value in (("lovx", np.inf), ("const", 2.0), ("lovx", True), ("lovx", "3")):
            f = features.copy(deep=True)
            if type(value) in (bool, str):
                f[field] = value
            else:
                f.loc["2016-01-08", field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.run_mock(f, targets, config)
        for value in (0.0, -1.0, np.inf):
            t = targets.copy(deep=True)
            t.loc["2016-01-08", "y"] = value
            with self.subTest(y=value), self.assertRaises(ValueError):
                self.run_mock(features, t, config)
        for f, t in (
            (features.iloc[::-1], targets.iloc[::-1]),
            (features, targets.iloc[:-1]),
            (
                features.set_axis(features.index.tz_localize("UTC")),
                targets.set_axis(targets.index.tz_localize("UTC")),
            ),
        ):
            with self.assertRaises(ValueError):
                self.run_mock(f, t, config)

    def test_config_defaults_and_bounded_exact_configuration(self):
        features, targets, config = synthetic_inputs()
        with patch("src.commodity_implied_pipeline.fit_models", side_effect=mock_models):
            result = build_panel(features, targets)
        self.assertEqual(result["schedules"].iloc[0].month, "2016-01")
        self.assertEqual(result["schedules"].iloc[-1].month, "2025-10")
        self.assertEqual(set(result["applications"].phase), {"development"})
        for updates in (
            {"minimum_train": True},
            {"minimum_train": 0},
            {"source_end": "2025-10-21"},
            {"origin_start": "2008-12-31"},
            {"evaluation": ["2016-01-20", "2016-03-10"]},
            {"minimum_train_releases": 1},
        ):
            bad = copy.deepcopy(config)
            bad.update(updates)
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                self.run_mock(features, targets, bad)

    def test_native_datetime_unit_transport_and_stable_output_schema(self):
        features, targets, config = synthetic_inputs()
        expected, _ = self.run_mock(features, targets, config)
        f, t = features.copy(deep=True), targets.copy(deep=True)
        f.index = f.index.as_unit("us")
        t.index = t.index.as_unit("us")
        f["commodity_cutoff_date"] = f.commodity_cutoff_date.astype("datetime64[us]")
        t["target_end"] = t.target_end.astype("datetime64[us]")
        actual, _ = self.run_mock(f, t, config)
        self.assertEqual(
            set(actual), {"applications", "panel", "coverage", "schedules", "fits"}
        )
        for name in ("applications", "panel", "coverage", "schedules"):
            pd.testing.assert_frame_equal(actual[name], expected[name])
            self.assertFalse(
                any(
                    "claim" in column or "release" in column for column in actual[name].columns
                )
            )
        self.assertEqual(actual["fits"], expected["fits"])
        self.assertEqual(str(actual["applications"].origin.dtype), "datetime64[ns]")
        self.assertEqual(str(actual["applications"].offset.dtype), "Int64")

    def test_real_models_receive_exact_reconstructed_cohorts_without_input_mutation(self):
        features, targets, config = synthetic_inputs()
        config.update(origin_end="2016-02-10", evaluation=["2016-02-01", "2016-02-10"])
        f_before, t_before = features.copy(deep=True), targets.copy(deep=True)
        result = build_panel(features, targets, config)
        for record in result["fits"]:
            query = pd.Timestamp(record["fit_origin"])
            position = features.index.get_loc(query)
            cutoff = features.index[position - 1]
            train = (
                features[list(ALL)].notna().all(axis=1)
                & (features.index < query)
                & targets.y.notna()
                & (targets.target_end <= cutoff)
            )
            application = (
                (features.index >= config["origin_start"])
                & (features.index <= config["origin_end"])
                & (features.index.to_period("M") == query.to_period("M"))
            )
            oracle = fit_models(
                features.loc[train], targets.loc[train, "y"], features.loc[application]
            )
            rows = result["applications"].loc[result["applications"].fit_origin == query]
            for arm in ARMS:
                np.testing.assert_allclose(
                    rows["pred_" + arm], oracle["predictions"][arm], rtol=1e-12, atol=0
                )
                self.assertEqual(record["model_audits"][arm]["n_train"], int(train.sum()))
            self.assertEqual(record["train_n"], int(train.sum()))
        pd.testing.assert_frame_equal(features, f_before)
        pd.testing.assert_frame_equal(targets, t_before)


if __name__ == "__main__":
    unittest.main()
