"""Prewritten independent peak-age contracts; generated observations only."""

import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from src import peak_age_verification as v


def market(n=360, unit="ms"):
    dates = pd.bdate_range("2010-01-04", periods=n).as_unit(unit)
    t = np.arange(n, dtype=float)
    close = 100 * np.exp(0.0002 * t + 0.02 * np.sin(t / 13) + 0.004 * np.cos(t / 3))
    opening = close * np.exp(0.001 * np.sin(t / 5))
    daily = pd.DataFrame(
        {
            "open": opening,
            "high": np.maximum(opening, close) * 1.003,
            "low": np.minimum(opening, close) / 1.004,
            "close": close,
        },
        index=dates,
    )
    iv = pd.DataFrame(
        {
            "vix": 20 + np.sin(t / 7),
            "vix9d": 18 + np.cos(t / 9),
            "vvix": 90 + 2 * np.sin(t / 11),
        },
        index=dates,
    )
    return daily, iv


def model_sample():
    rng = np.random.default_rng(84952)
    index = pd.bdate_range("2010-01-04", periods=1058)
    frame = pd.DataFrame(
        rng.normal(size=(len(index), len(v.COMMON))), index=index, columns=v.COMMON
    )
    frame["const"] = 1.0
    for day in range(1, 5):
        frame[f"entry_dow_{day}"] = (index.dayofweek == day).astype(float)
    frame["peak_age"] = rng.integers(0, 252, len(index)) / 251
    frame["drawdown"] = rng.uniform(0.001, 0.4, len(index))
    frame["drawdown_sq"] = frame.drawdown**2
    y = pd.Series(
        0.02 * frame.peak_age + 0.01 * frame.drawdown + rng.normal(0, 0.01, len(index)),
        index=index,
    )
    return frame.iloc[:1050].copy(), y.iloc[:1050].copy(), frame.iloc[1050:].copy()


class IndependentPrimitives(unittest.TestCase):
    def test_nonzero_compensated_mean_underflow_is_not_a_zero_branch(self):
        with self.assertRaises((ValueError, ArithmeticError)):
            v.mean_sum([np.nextafter(0.0, 1.0), 0.0])

    def test_high_precision_log_ratio_extremes_equal_and_nearby(self):
        self.assertEqual(v.decimal_log_ratio(7.0, 7.0), 0.0)
        for power in (-1000, 0, 1000):
            x = math.ldexp(1.0, power)
            y = np.nextafter(x, math.inf)
            self.assertGreater(v.decimal_log_ratio(y, x), 0.0)
            self.assertLess(v.decimal_log_ratio(x, y), 0.0)
            self.assertAlmostEqual(
                v.decimal_log_ratio(y, x), math.log1p((y - x) / x), places=28
            )
        smallest = np.nextafter(0.0, 1.0)
        largest = np.finfo(float).max
        self.assertTrue(math.isfinite(v.decimal_log_ratio(largest, smallest)))
        self.assertAlmostEqual(
            v.decimal_log_ratio(largest, smallest),
            -v.decimal_log_ratio(smallest, largest),
            places=12,
        )

    def test_log_ratio_rejects_invalid_observed_values(self):
        for x in (0.0, -1.0, float("nan"), float("inf"), True, "2", 1j):
            with self.subTest(x=x), self.assertRaises((ValueError, TypeError)):
                v.decimal_log_ratio(x, 1.0)

    def test_peak_latest_exact_tie_and_endpoints(self):
        x = np.full(252, 10.0)
        x[[0, 70, 200]] = 12.0
        answer = v.peak_summary(x)
        self.assertEqual(answer["peak_offset"], 200)
        self.assertEqual(answer["peak_age_sessions"], 51)
        self.assertEqual(answer["peak_age"], 51 / 251)
        self.assertAlmostEqual(answer["drawdown"], math.log(1.2))
        self.assertAlmostEqual(answer["window_return"], math.log(10 / 12), places=15)
        for x, age in (
            (np.arange(1, 253, dtype=float), 0),
            (np.arange(252, 0, -1, dtype=float), 251),
            (np.ones(252), 0),
        ):
            self.assertEqual(v.peak_summary(x)["peak_age_sessions"], age)
        x = np.ones(252)
        x[5] = np.nextafter(1.0, 2.0)
        self.assertEqual(v.peak_summary(x)["peak_offset"], 5)

    def test_peak_rejects_missing_wrong_length_and_invalid(self):
        for x in (
            np.ones(251),
            np.ones(253),
            np.ones((252, 1)),
            [True] * 252,
            [1.0] * 251 + [float("nan")],
            [1.0] * 251 + [0.0],
        ):
            with self.subTest(shape=np.shape(x)), self.assertRaises((ValueError, TypeError)):
                v.peak_summary(x)

    def test_moving_peak_changes_age_with_same_current_controls(self):
        daily, iv = market(253)
        daily.iloc[:188, daily.columns.get_loc("close")] = 100.0
        for column in ("open", "high", "low"):
            daily.loc[daily.index[:188], column] = {"open": 100.0, "high": 200.0, "low": 90.0}[
                column
            ]
        a, b = daily.copy(), daily.copy()
        a.iloc[60, a.columns.get_loc("close")] = 150.0
        b.iloc[130, b.columns.get_loc("close")] = 150.0
        fa, _, _ = v.independent_features(a, iv)
        fb, _, _ = v.independent_features(b, iv)
        columns = [*v.RAW, "drawdown", "drawdown_sq", "window_return"]
        pd.testing.assert_series_equal(fa.iloc[252][columns], fb.iloc[252][columns])
        self.assertNotEqual(fa.iloc[252].peak_age, fb.iloc[252].peak_age)

    def test_strict_calendar_window_target_and_feature_lag(self):
        daily, iv = market()
        features, targets, states = v.independent_features(daily, iv)
        self.assertTrue(features.peak_age.iloc[:252].isna().all())
        self.assertTrue(np.isfinite(features.peak_age.iloc[252:]).all())
        self.assertEqual(states.iloc[252].window_start_position, 0)
        self.assertEqual(states.iloc[252].window_end_position, 251)
        self.assertEqual(features.iloc[252].feature_cutoff_date, daily.index[251])
        self.assertEqual(targets.iloc[252].available_date, daily.index[273])
        self.assertAlmostEqual(
            targets.iloc[252].y,
            math.log(daily.close.iloc[273] / daily.close.iloc[252]),
            places=14,
        )
        self.assertTrue(targets.y.iloc[-21:].isna().all())
        changed = daily.copy()
        changed.iloc[252:, :] *= 2
        newer, _, _ = v.independent_features(changed, iv)
        pd.testing.assert_frame_equal(features.iloc[:253], newer.iloc[:253])
        missing = daily.copy()
        missing.iloc[200, missing.columns.get_loc("close")] = np.nan
        incomplete, _, audit = v.independent_features(missing, iv)
        self.assertTrue(incomplete.peak_age.iloc[252:].isna().all())
        self.assertFalse(audit.window_complete.iloc[252:].any())
        self.assertEqual(audit.iloc[252].window_start_position, 0)
        self.assertTrue(pd.isna(audit.iloc[252].peak_position))

    def test_ridge_augmented_solution_and_saved_mean_intercept(self):
        x = np.array([[1.0, -1.0], [1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        y = 2 + 3 * x[:, 1]
        query = np.array([[1.0, 2.0], [1.0, 0.0]])
        pred, audit = v.ridge_fit(x, y, query)
        np.testing.assert_allclose(audit["beta"], [2, 3 / 1.01], rtol=0, atol=1e-12)
        np.testing.assert_allclose(pred, [2 + 6 / 1.01, 2], rtol=0, atol=1e-12)
        self.assertLessEqual(audit["gradient_max_abs"], 1e-10)

    def test_ridge_strict_scale_and_target_alignment(self):
        for x, y, a in (
            ([[1.0, 1.0], [1.0, 1.0]], [1.0, 2.0], [[1.0, 1.0]]),
            ([[0.0, 1.0], [0.0, 2.0]], [1.0, 2.0], [[0.0, 1.0]]),
            ([[1.0, 1.0], [1.0, 2.0]], [1.0], [[1.0, 1.0]]),
        ):
            with self.assertRaises((ValueError, AssertionError)):
                v.ridge_fit(x, y, a)

    def test_scalar_augmented_solution_is_sequential_fixed_scale(self):
        ages = np.array([0.0, 0.2, 0.7, 1.0])
        offset = np.array([0.1, 0.2, -0.1, 0.4])
        y = np.array([0.4, -0.3, 0.8, 1.0])
        query = np.array([0.9, 0.1])
        base = np.array([0.2, -0.2])
        pred, audit = v.scalar_fit(ages, y, offset, query, base)
        centered = ages - math.fsum(ages) / len(ages)
        beta = np.dot(centered, y - offset) / (np.dot(centered, centered) + len(ages) * 0.01)
        self.assertAlmostEqual(audit["coefficient"], beta, places=12)
        np.testing.assert_allclose(pred, base + beta * (query - ages.mean()), atol=1e-12)
        self.assertEqual(audit["age_scale"], 1.0)
        self.assertLessEqual(audit["gradient_max_abs"], 1e-10)

    def test_scalar_zero_branches_copy_query_bits(self):
        base = np.array([-0.0, 1.25])
        for ages, y, status in (
            ([0.25] * 4, [1.0, 2.0, 3.0, 4.0], "EXACT_CONSTANT_INPUT"),
            ([0.0, 0.25, 0.75, 1.0], [1.0, 1.0, 1.0, 1.0], "BALANCED_AT_ZERO"),
        ):
            pred, audit = v.scalar_fit(ages, y, np.zeros(4), [0.0, 1.0], base)
            self.assertEqual(audit["status"], status)
            self.assertEqual(audit["coefficient"], 0.0)
            np.testing.assert_array_equal(pred.view(np.uint64), base.view(np.uint64))

    def test_nonzero_scalar_uses_declared_addition_even_at_zero_query_age(self):
        ages = np.array([0.0, 0.25, 0.75, 1.0])
        pred, audit = v.scalar_fit(ages, ages, np.zeros(4), [0.5], [-0.0])
        self.assertNotEqual(audit["coefficient"], 0.0)
        expected = np.array([-0.0]) + audit["coefficient"] * np.array([0.0])
        np.testing.assert_array_equal(pred.view(np.uint64), expected.view(np.uint64))

    def test_age_inside_nuisance_span_can_retune_ridge(self):
        ages = np.array([0.0, 0.25, 0.75, 1.0])
        design = np.column_stack([np.ones(4), ages])
        y = 3 * ages
        baseline, _ = v.ridge_fit(design, y, design)
        _, audit = v.scalar_fit(ages, y, baseline, ages, baseline)
        self.assertGreater(audit["coefficient"], 0.0)
        self.assertEqual(audit["status"], "FITTED")

    def test_scalar_rejects_bad_domains_and_shapes(self):
        for ages in ([0.0, 2.0], [False, True], [0.0, float("nan")], [[0.0], [1.0]]):
            with self.subTest(ages=ages), self.assertRaises((ValueError, AssertionError)):
                v.scalar_fit(ages, [1.0, 2.0], [0.0, 0.0], [0.5], [0.0])


class ProducerChallengeContracts(unittest.TestCase):
    def test_boolean_inside_zero_coefficient_list_is_not_numeric_evidence(self):
        from src import peak_age_models as producer

        train, y, apply = model_sample()
        y[:] = 0.0
        saved = producer.fit_predict(train, y, apply)
        saved["model_audit"]["baseline"]["beta"][0] = False
        with self.assertRaises((ValueError, AssertionError)):
            v.verify_models(train, y, apply, saved)

    def test_tiny_valid_drawdown_cannot_be_zeroed_within_float_tolerance(self):
        from src import peak_age_features as producer

        daily, iv = market(260)
        daily["close"] = 1.0
        daily["open"] = 1.0
        daily["high"] = 1.1
        daily["low"] = 0.9
        daily.iloc[100, daily.columns.get_loc("close")] = np.nextafter(1.0, 2.0)
        features, targets, states = producer.build_features(daily, iv)
        self.assertGreater(features.iloc[252].drawdown, 0.0)
        features.iloc[252, features.columns.get_loc("drawdown")] = 0.0
        features.iloc[252, features.columns.get_loc("drawdown_sq")] = 0.0
        with self.assertRaises((ValueError, AssertionError)):
            v.verify_features(daily, iv, features, targets, states)

    def test_generated_producer_features_match_independent_oracle_all_units(self):
        from src import peak_age_features as producer

        for unit in ("ms", "us", "ns"):
            daily, iv = market(unit=unit)
            saved = producer.build_features(daily, iv)
            with patch.object(
                producer, "build_features", side_effect=AssertionError("producer forbidden")
            ):
                proof = v.verify_features(daily, iv, *saved)
            self.assertEqual(proof["source_calendar_rows_verified"], len(daily))
            self.assertEqual(proof["complete_peak_windows_verified"], len(daily) - 252)

    def test_feature_or_target_coherent_tamper_rejected(self):
        from src import peak_age_features as producer

        daily, iv = market()
        original = producer.build_features(daily, iv)
        for mutation in ("age", "depth", "target", "peak", "unknown", "cutoff"):
            features, targets, states = [x.copy(deep=True) for x in original]
            row = features.index[280]
            if mutation == "age":
                features.loc[row, "peak_age"] += 1 / 251
            elif mutation == "depth":
                features.loc[row, "drawdown"] += 1e-4
            elif mutation == "target":
                targets.loc[row, "y"] += 1e-4
            elif mutation == "peak":
                states.loc[row, "peak_position"] -= 1
            elif mutation == "unknown":
                features.loc[row, "peak_age"] = np.nan
            else:
                features.loc[row, "feature_cutoff_date"] += pd.Timedelta(days=1)
            with (
                self.subTest(mutation=mutation),
                self.assertRaises((ValueError, AssertionError)),
            ):
                v.verify_features(daily, iv, features, targets, states)

    def test_saved_string_index_and_nondate_dtype_are_not_coerced(self):
        from src import peak_age_features as producer

        daily, iv = market()
        original = producer.build_features(daily, iv)
        for mutation in ("index", "age_dtype", "subday"):
            f, t, s = [x.copy(deep=True) for x in original]
            if mutation == "index":
                f.index = f.index.strftime("%Y-%m-%d")
            elif mutation == "age_dtype":
                f["peak_age"] = f.peak_age.astype("float32")
            else:
                f["feature_cutoff_date"] += pd.Timedelta(seconds=1)
            with (
                self.subTest(mutation=mutation),
                self.assertRaises((ValueError, AssertionError)),
            ):
                v.verify_features(daily, iv, f, t, s)

    def test_model_saved_coefficients_match_augmented_independent_solves(self):
        from src import peak_age_models as producer

        train, y, apply = model_sample()
        saved = producer.fit_predict(train, y, apply)
        with patch.object(
            producer, "fit_predict", side_effect=AssertionError("producer forbidden")
        ):
            predictions, proof = v.verify_models(train, y, apply, saved)
        self.assertEqual(proof["nuisance_fits_verified"], 2)
        self.assertEqual(proof["scalar_fits_verified"], 1)
        self.assertEqual(set(predictions), set(v.MODELS))
        for name in v.MODELS:
            np.testing.assert_allclose(
                predictions[name], saved["predictions"][name], rtol=1e-8, atol=1e-12
            )

    def test_saved_model_audit_prediction_settings_or_schema_tamper_rejected(self):
        from src import peak_age_models as producer

        train, y, apply = model_sample()
        original = producer.fit_predict(train, y, apply)
        for mutation in (
            "coefficient",
            "prediction",
            "scale",
            "alpha",
            "scalar",
            "center",
            "schema",
            "count",
        ):
            saved = copy.deepcopy(original)
            if mutation == "coefficient":
                saved["model_audit"]["depth"]["beta"][2] += 0.01
            elif mutation == "prediction":
                saved["predictions"]["peak_age"][0] += 0.01
            elif mutation == "scale":
                saved["model_audit"]["baseline"]["scales"][1] *= 2
            elif mutation == "alpha":
                saved["scalar_audit"]["alpha"] += 1e-13
            elif mutation == "scalar":
                saved["scalar_audit"]["coefficient"] += 0.01
            elif mutation == "center":
                saved["transform_audit"]["I_mean"] += 0.01
            elif mutation == "schema":
                saved["scalar_audit"]["extra"] = 0
            else:
                saved["scalar_audit"]["train_n"] = True
            with (
                self.subTest(mutation=mutation),
                self.assertRaises((ValueError, AssertionError)),
            ):
                v.verify_models(train, y, apply, saved)


def full_fixture(unit="ms"):
    from src import peak_age_features as feature
    from src import peak_age_pipeline as pipeline

    daily, iv = market(1420, unit)
    features, targets, states = feature.build_features(daily, iv)
    p = yaml.safe_load((Path(__file__).resolve().parents[1] / "peak_age.yaml").read_bytes())
    index = features.index
    text = lambda i: str(index[i].date())
    p["index"].update(
        origin_start=text(1300),
        origin_end=text(1419),
        source_end=text(1419),
        latest_target=text(1419),
        development=[text(1300), text(1359)],
        development_target_available_by=text(1359),
        evaluation=[text(1360), text(1419)],
        evaluation_stability=[[text(1360), text(1389)], [text(1390), text(1419)]],
    )
    panel, fits, applications, support = pipeline.build_panel(features, targets, states, p)
    return daily, iv, p, features, targets, states, panel, fits, applications, support


class FullPipelineContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = full_fixture()

    def test_all_transported_units_and_full_unscored_predictions_independent(self):
        from src import peak_age_features as feature
        from src import peak_age_models as model
        from src import peak_age_pipeline as pipeline

        for unit in ("ms", "us", "ns"):
            args = list(full_fixture(unit))
            with tempfile.TemporaryDirectory() as folder:
                for i in (0, 1, 3, 4, 5, 6, 8):
                    path = Path(folder) / f"table{i}.parquet"
                    args[i].to_parquet(path)
                    args[i] = pd.read_parquet(path)
                import json

                for i in (2, 7, 9):
                    args[i] = json.loads(json.dumps(args[i], allow_nan=False))
                with (
                    patch.object(
                        feature, "build_features", side_effect=AssertionError("no producer")
                    ),
                    patch.object(
                        model, "fit_predict", side_effect=AssertionError("no producer fit")
                    ),
                    patch.object(
                        pipeline,
                        "build_panel",
                        side_effect=AssertionError("no producer panel"),
                    ),
                ):
                    proof = v.verify_pipeline(*args)
            self.assertEqual(proof["forecasts_verified"], len(args[6]))
            self.assertEqual(proof["application_origins_verified"], len(args[8]))
            self.assertEqual(proof["new_nuisance_fits_verified"], 2 * len(args[7]))
            self.assertEqual(proof["new_scalar_fits_verified"], len(args[7]))
            self.assertGreater(args[9]["unscored_origins"], 0)

    def test_unscored_and_coherent_scored_prediction_changes_rejected(self):
        for mutation in ("unscored", "scored"):
            args = copy.deepcopy(list(self.fixture))
            panel, states = args[6], args[8]
            if mutation == "unscored":
                row = states.index[~states.origin.isin(panel.origin)][0]
                states.loc[row, "pred_peak_age"] += 0.001
            else:
                row = panel.index[panel.model.eq("peak_age")][0]
                origin = panel.loc[row, "origin"]
                panel.loc[row, "prediction"] += 0.001
                panel.loc[row, "loss"] = (
                    panel.loc[row, "y"] - panel.loc[row, "prediction"]
                ) ** 2
                states.loc[states.origin.eq(origin), "pred_peak_age"] += 0.001
            with (
                self.subTest(mutation=mutation),
                self.assertRaises((ValueError, AssertionError)),
            ):
                v.verify_pipeline(*args)

    def test_scored_target_and_application_copy_are_exact_inside_solver_tolerances(self):
        for name in ("y", "prediction"):
            args = copy.deepcopy(list(self.fixture))
            panel = args[6]
            row = panel.index[panel.model.eq("peak_age")][0]
            panel.loc[row, name] = np.nextafter(panel.loc[row, name], math.inf)
            panel.loc[row, "loss"] = (panel.loc[row, "y"] - panel.loc[row, "prediction"]) ** 2
            with self.subTest(name=name), self.assertRaises((ValueError, AssertionError)):
                v.verify_pipeline(*args)

    def test_every_cohort_fit_support_and_clock_identity_checked(self):
        for mutation in (
            "train_order",
            "fit_origin",
            "app_cohort",
            "support",
            "target",
            "state_date",
            "extra",
        ):
            args = copy.deepcopy(list(self.fixture))
            if mutation == "train_order":
                args[7][0]["train_origins"].reverse()
            elif mutation == "fit_origin":
                args[7][0]["fit_origin"] = "1999-01-01"
            elif mutation == "app_cohort":
                args[7][0]["application_origins"].pop()
            elif mutation == "support":
                args[9]["scored_origins"] += 1
            elif mutation == "target":
                args[4].iloc[500, 0] += 0.01
            elif mutation == "state_date":
                args[8].loc[0, "feature_cutoff_date"] += pd.Timedelta(days=1)
            else:
                args[7][0]["unused_hinge"] = {}
            with (
                self.subTest(mutation=mutation),
                self.assertRaises((ValueError, AssertionError)),
            ):
                v.verify_pipeline(*args)

    def test_saved_full_stage_coefficients_and_gradient_tamper_rejected(self):
        for mutation in ("beta", "scalar", "gradient", "alpha"):
            args = copy.deepcopy(list(self.fixture))
            fit = args[7][0]
            if mutation == "beta":
                fit["model_audit"]["depth"]["beta"][2] += 0.01
            elif mutation == "scalar":
                fit["scalar_audit"]["coefficient"] += 0.01
            elif mutation == "gradient":
                fit["scalar_audit"]["gradient_max_abs"] = -1e-16
            else:
                fit["model_audit"]["baseline"]["alpha"] += 1e-13
            with (
                self.subTest(mutation=mutation),
                self.assertRaises((ValueError, AssertionError)),
            ):
                v.verify_pipeline(*args)


if __name__ == "__main__":
    unittest.main()
