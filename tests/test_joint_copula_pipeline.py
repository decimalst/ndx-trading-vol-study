"""Generated full-calendar, measurement-first and issuance contracts."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import joint_copula_pipeline as pipeline
from tests.test_joint_risk_features import sample


def config(dates):
    return {
        "origin_start": str(dates[60].date()),
        "origin_end": str(dates[-1].date()),
        "source_end": str(dates[-1].date()),
        "development": [str(dates[60].date()), str(dates[100].date())],
        "evaluation": [str(dates[101].date()), str(dates[-1].date())],
        "minimum_train": 20,
    }


def fake_fit(train, targets, query):
    mu = np.tile(targets[["y_qqq", "y_spx"]].mean().to_numpy(), (len(query), 1))
    return {
        name: {
            "mu": mu.copy(),
            "h": np.full((len(query), 2), 0.001),
            "rho": np.full(len(query), rho),
        }
        for name, rho in (("t8_copula", 0.2), ("gaussian_copula", 0.1), ("independence", 0.0))
    }, {"generated": True}


class JointCopulaPipelineTests(unittest.TestCase):
    def test_full_reference_monthly_maturity_and_all_unscored_applications(self):
        q, s, iv = sample()
        with patch.object(pipeline, "fit_predict", side_effect=fake_fit):
            got = pipeline.produce(q, s, iv, config(s.index))
        self.assertEqual(
            set(got),
            {"features", "targets", "applications", "panel", "coverage", "schedules", "fits"},
        )
        self.assertTrue(got["features"].index.equals(s.index))
        self.assertEqual(len(got["coverage"]), len(s) - 60)
        self.assertEqual(len(got["applications"].query("origin == @s.index[-1]")), 3)
        self.assertFalse(got["coverage"].iloc[-1].scored)
        self.assertTrue(got["panel"].groupby("origin").size().eq(3).all())
        for fit in got["fits"]:
            first = s.index.get_loc(pd.Timestamp(fit["fit_origin"]))
            self.assertEqual(fit["training_cutoff"], str(s.index[first - 1].date()))
            expected = got["features"].loc[:, pipeline.ALL_FEATURES].notna().all(axis=1)
            expected &= got["targets"][["y_qqq", "y_spx"]].notna().all(axis=1)
            expected &= got["targets"].target_end <= s.index[first - 1]
            self.assertEqual(fit["train_origins"], [str(d.date()) for d in s.index[expected]])
        np.testing.assert_array_equal(
            got["panel"].offset, s.index.get_indexer(got["panel"].origin) % 5
        )
        self.assertTrue(
            (got["panel"].query("phase == 'development'").target_end <= s.index[100]).all()
        )

    def test_measurement_gate_sees_unusable_row_before_features_or_fit(self):
        q, s, iv = sample()
        s.loc[s.index[0], ["open", "high", "low", "close"]] = 100.0
        iv.iloc[:40] = np.nan
        observed = []
        with (
            patch.object(pipeline, "build_features") as build,
            patch.object(pipeline, "fit_predict") as fit,
            self.assertRaises(pipeline.PipelineExecutionError) as error,
        ):
            pipeline.produce(q, s, iv, config(s.index), measurement_callback=observed.append)
        build.assert_not_called()
        fit.assert_not_called()
        self.assertEqual(observed[0]["status"], "INSUFFICIENT_MEASUREMENT")
        self.assertEqual(error.exception.stage, "measurement")

    def test_future_source_dates_fail_before_measurement_arithmetic(self):
        q, s, iv = sample()
        iv.index = iv.index[:-1].append(pd.DatetimeIndex(["2025-10-21"]))
        with (
            patch.object(pipeline, "measurement_audit") as measure,
            self.assertRaises(ValueError),
        ):
            pipeline.produce(q, s, iv, config(s.index))
        measure.assert_not_called()

    def test_unknown_query_label_does_not_move_monthly_fit(self):
        q, s, iv = sample()
        with patch.object(pipeline, "fit_predict", side_effect=fake_fit):
            before = pipeline.produce(q, s, iv, config(s.index))
            q.loc[s.index[61], "open"] = np.nan
            after = pipeline.produce(q, s, iv, config(s.index))
        self.assertEqual(before["fits"][0]["fit_origin"], after["fits"][0]["fit_origin"])
        self.assertIn(s.index[60], set(after["applications"].origin))
        self.assertNotIn(s.index[60], set(after["panel"].origin))

    def test_future_sources_do_not_change_earlier_issued_forecasts(self):
        q, s, iv = sample()
        with patch.object(pipeline, "fit_predict", side_effect=fake_fit):
            before = pipeline.produce(q, s, iv, config(s.index))
            future = s.index[110]
            q.loc[future:, ["open", "high", "low", "close"]] *= 1.3
            after = pipeline.produce(q, s, iv, config(s.index))
        pd.testing.assert_frame_equal(
            before["applications"].query("origin < @future"),
            after["applications"].query("origin < @future"),
        )

    def test_fit_failure_and_insufficient_rows_preserve_attempt_diagnostics(self):
        q, s, iv = sample()
        calls = []

        def fail(*args):
            calls.append(1)
            if len(calls) == 2:
                raise ValueError("invented dependence failure")
            return fake_fit(*args)

        with (
            patch.object(pipeline, "fit_predict", side_effect=fail),
            self.assertRaises(pipeline.PipelineExecutionError) as caught,
        ):
            pipeline.produce(q, s, iv, config(s.index))
        self.assertGreater(len(caught.exception.produced["applications"]), 0)
        self.assertTrue(caught.exception.produced["panel"].empty)
        self.assertEqual(caught.exception.stage, "fit_predict")
        c = config(s.index)
        c["minimum_train"] = 1000
        with self.assertRaisesRegex(pipeline.PipelineExecutionError, "INSUFFICIENT_DATA"):
            pipeline.produce(q, s, iv, c)

    def test_feature_construction_and_alignment_failures_retain_available_outputs(self):
        q, s, iv = sample()
        with (
            patch.object(
                pipeline,
                "build_features",
                side_effect=RuntimeError("invented builder failure"),
            ),
            self.assertRaises(pipeline.PipelineExecutionError) as caught,
        ):
            pipeline.produce(q, s, iv, config(s.index))
        self.assertEqual(caught.exception.stage, "feature_construction")
        self.assertEqual(len(caught.exception.produced), 7)
        self.assertEqual(caught.exception.produced["fits"], [])
        f, t = pipeline.build_features(q, s, iv)
        broken = t.iloc[1:].copy()
        with (
            patch.object(pipeline, "build_features", return_value=(f, broken)),
            self.assertRaises(pipeline.PipelineExecutionError) as caught,
        ):
            pipeline.produce(q, s, iv, config(s.index))
        self.assertEqual(caught.exception.stage, "feature_alignment")
        pd.testing.assert_frame_equal(caught.exception.produced["features"], f)
        pd.testing.assert_frame_equal(caught.exception.produced["targets"], broken)

    def test_final_coverage_or_panel_failure_retains_all_issued_months(self):
        q, s, iv = sample()
        with patch.object(pipeline, "fit_predict", side_effect=fake_fit):
            expected = pipeline.produce(q, s, iv, config(s.index))
        frame = pipeline._frame
        for failure_columns in (pipeline.COVERAGE_COLUMNS, pipeline.PANEL_COLUMNS):

            def fail(rows, columns, failed_columns=failure_columns):
                if columns == failed_columns and len(rows):
                    raise RuntimeError("invented final assembly failure")
                return frame(rows, columns)

            with (
                self.subTest(columns=failure_columns),
                patch.object(pipeline, "fit_predict", side_effect=fake_fit),
                patch.object(pipeline, "_frame", side_effect=fail),
                self.assertRaises(pipeline.PipelineExecutionError) as caught,
            ):
                pipeline.produce(q, s, iv, config(s.index))
            partial = caught.exception.produced
            self.assertEqual(caught.exception.stage, "result_assembly")
            self.assertEqual(len(partial), 7)
            self.assertEqual(partial["fits"], expected["fits"])
            pd.testing.assert_frame_equal(partial["applications"], expected["applications"])
            self.assertTrue(partial["panel"].empty)
            self.assertFalse(partial["coverage"].scored.any())

    def test_missing_source_predecessor_and_source_immutability(self):
        q, s, iv = sample()
        q = q.drop(s.index[50])
        before = q.copy(deep=True)
        with patch.object(pipeline, "fit_predict", side_effect=fake_fit):
            got = pipeline.produce(q, s, iv, config(s.index))
        self.assertTrue(np.isnan(got["features"].loc[s.index[52], "qqq_lt_d"]))
        self.assertTrue(np.isfinite(got["targets"].loc[s.index[50], "y_qqq"]))
        pd.testing.assert_frame_equal(q, before)


if __name__ == "__main__":
    unittest.main()
