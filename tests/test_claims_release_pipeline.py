"""Generated-only prospective tests, written before the monthly producer."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.claims_release_models import ALL, fit_models
from src.claims_release_pipeline import build_panel


def fixture():
    calendar = pd.bdate_range("2018-01-01", "2018-05-11")
    rng = np.random.default_rng(23091)
    features = pd.DataFrame(rng.normal(size=(len(calendar), 19)), index=calendar, columns=ALL)
    features["const"] = 1.0
    nominal = pd.date_range("2017-12-21", "2018-05-17", freq="W-THU")
    releases = nominal - pd.to_timedelta(
        [int(i % 4 == 0) for i in range(len(nominal))], unit="D"
    )
    selected = np.searchsorted(releases, calendar, side="left") - 1
    features["claim_release_date"] = releases[selected]
    features["claim_reference_week"] = nominal[selected] - pd.Timedelta(days=5)
    age = (calendar - pd.DatetimeIndex(features.claim_release_date)).days
    features["claim_age"] = age / 7
    for day in range(1, 5):
        features[f"entry_dow_{day}"] = (calendar.dayofweek == day).astype(float)
    features["claim_status"] = "available"
    stale = age > 7
    features.loc[stale, ["claim_m4", "claim_age", "claim_x"]] = np.nan
    features.loc[stale, "claim_status"] = "stale_release"
    targets = pd.DataFrame(
        {
            "y": np.exp(-5 + 0.1 * features.lrv_d + rng.normal(0, 0.05, len(calendar))),
            "target_end": pd.Series(calendar, index=calendar).shift(-5),
        },
        index=calendar,
    )
    targets.loc[targets.target_end.isna(), "y"] = np.nan
    config = {
        "origin_start": "2018-03-01",
        "origin_end": "2018-05-11",
        "development": ["2018-03-01", "2018-03-30"],
        "evaluation": ["2018-04-02", "2018-05-11"],
        "source_end": "2018-05-11",
        "minimum_train": 20,
        "minimum_train_releases": 4,
    }
    return features, targets, config


def fake_fit(train, y, application):
    value = float(y.mean())
    return {
        "predictions": {
            arm: np.full(len(application), value * (1 + i / 10))
            for i, arm in enumerate(("market", "matched", "candidate"))
        },
        "fits": {"generated": {"train": [d.date().isoformat() for d in train.index]}},
    }


class ClaimsReleasePipelineTests(unittest.TestCase):
    def build_fake(self, features=None, targets=None, config=None):
        default = fixture()
        with patch("src.claims_release_pipeline.fit_models", side_effect=fake_fit) as fitted:
            result = build_panel(
                features if features is not None else default[0],
                targets if targets is not None else default[1],
                config if config is not None else default[2],
            )
        return result, fitted

    def test_first_complete_origin_precedes_query_label_filter(self):
        features, targets, config = fixture()
        features.loc["2018-03-01", "claim_x"] = np.nan
        targets.loc["2018-03-02", "y"] = np.nan
        result, fitted = self.build_fake(features, targets, config)
        self.assertEqual(result["schedules"].iloc[0].fit_origin, pd.Timestamp("2018-03-02"))
        self.assertEqual(fitted.call_args_list[0].args[2].index[0], pd.Timestamp("2018-03-02"))
        self.assertIn(pd.Timestamp("2018-03-02"), result["applications"].origin.tolist())
        self.assertNotIn(pd.Timestamp("2018-03-02"), result["panel"].origin.tolist())
        self.assertEqual(
            result["coverage"].set_index("origin").loc["2018-03-02", "status"],
            "missing_target",
        )

    def test_maturity_uses_previous_full_session_and_common_candidate_cohort(self):
        features, targets, config = fixture()
        features.loc["2018-02-28", "claim_x"] = np.nan
        features.loc["2018-02-20", "claim_x"] = np.nan
        result, fitted = self.build_fake(features, targets, config)
        train, y, application = fitted.call_args_list[0].args
        expected = features.index[
            (features.loc[:, list(ALL)].notna().all(axis=1))
            & (features.index < pd.Timestamp("2018-03-01"))
            & targets.y.notna()
            & (targets.target_end <= pd.Timestamp("2018-02-28"))
        ]
        pd.testing.assert_index_equal(train.index, expected)
        pd.testing.assert_index_equal(y.index, expected)
        self.assertIn(pd.Timestamp("2018-02-21"), train.index)
        self.assertNotIn(pd.Timestamp("2018-02-22"), train.index)
        self.assertNotIn(pd.Timestamp("2018-02-20"), train.index)
        self.assertEqual(result["fits"][0]["training_cutoff"], "2018-02-28")
        self.assertEqual(
            result["fits"][0]["train_releases"], train.claim_release_date.nunique()
        )
        self.assertEqual(
            result["fits"][0]["train_origins"], [d.date().isoformat() for d in expected]
        )
        self.assertEqual(
            result["fits"][0]["application_origins"],
            [d.date().isoformat() for d in application.index],
        )

    def test_future_label_permutation_or_unknown_cannot_change_schedule_or_previous_forecasts(
        self,
    ):
        features, targets, config = fixture()
        reference, _ = self.build_fake(features, targets, config)
        mask = targets.target_end > pd.Timestamp("2018-02-28")
        for unknown in (False, True):
            changed = targets.copy()
            changed.loc[mask, "y"] = (
                np.nan if unknown else targets.loc[mask, "y"].to_numpy()[::-1]
            )
            result, _ = self.build_fake(features, changed, config)
            assert_frame_equal(
                result["schedules"][
                    ["month", "fit_origin", "training_cutoff", "application_n"]
                ],
                reference["schedules"][
                    ["month", "fit_origin", "training_cutoff", "application_n"]
                ],
            )
            march = reference["applications"].origin.dt.month == 3
            assert_frame_equal(
                result["applications"].loc[march], reference["applications"].loc[march]
            )
            self.assertEqual(result["fits"][0], reference["fits"][0])

    def test_unscored_applications_and_phase_cutoffs_are_explicit(self):
        features, targets, config = fixture()
        config["origin_end"] = config["evaluation"][1] = "2018-05-08"
        result, _ = self.build_fake(features, targets, config)
        coverage = result["coverage"].set_index("origin")
        self.assertEqual(coverage.loc["2018-03-28", "status"], "target_after_phase_cutoff")
        self.assertEqual(coverage.loc["2018-05-04", "status"], "scored")
        self.assertEqual(coverage.loc["2018-05-08", "status"], "target_not_mature")
        self.assertIn(pd.Timestamp("2018-05-08"), result["applications"].origin.tolist())
        self.assertNotIn(pd.Timestamp("2018-05-08"), result["panel"].origin.tolist())
        self.assertNotIn("scored", result["applications"].columns)
        self.assertEqual(len(result["panel"]), 3 * int(coverage.scored.sum()))

    def test_full_calendar_offsets_missingness_and_order(self):
        features, targets, config = fixture()
        features.loc["2018-03-01", ["claim_x", "lrv_d"]] = np.nan
        result, _ = self.build_fake(features, targets, config)
        row = result["coverage"].set_index("origin").loc["2018-03-01"]
        self.assertEqual(row.missing_features, "lrv_d|claim_x")
        self.assertEqual(row.status, "incomplete_features")
        self.assertTrue(pd.isna(row.fit_origin))
        for name in ("applications", "coverage", "panel"):
            for row in result[name].itertuples():
                self.assertEqual(row.offset, features.index.get_loc(row.origin) % 5)
            self.assertTrue(result[name].origin.is_monotonic_increasing)
        for _, group in result["panel"].groupby("origin", sort=False):
            self.assertEqual(group.model.tolist(), ["market", "matched", "candidate"])

    def test_months_with_no_complete_or_requested_origins_remain_visible(self):
        features, targets, config = fixture()
        features.loc[features.index.month == 4, "claim_x"] = np.nan
        result, fitted = self.build_fake(features, targets, config)
        row = result["schedules"].set_index("month").loc["2018-04"]
        self.assertEqual(row.status, "no_complete_origin")
        self.assertEqual(row.requested_n, 21)
        self.assertEqual(row.application_n, 0)
        self.assertTrue(pd.isna(row.train_n) and pd.isna(row.train_releases))
        self.assertTrue(pd.isna(row.fit_origin))
        self.assertEqual(fitted.call_count, 2)
        retained = features.index.month != 4
        features, targets = features.loc[retained].copy(), targets.loc[retained].copy()
        targets["target_end"] = pd.Series(features.index, index=features.index).shift(-5)
        result, _ = self.build_fake(features, targets, config)
        row = result["schedules"].set_index("month").loc["2018-04"]
        self.assertEqual(row.status, "no_requested_origins")
        self.assertEqual(row.requested_n, 0)

    def test_training_and_release_support_fail_before_any_fit(self):
        for field in ("minimum_train", "minimum_train_releases"):
            features, targets, config = fixture()
            config[field] = 10000
            with patch("src.claims_release_pipeline.fit_models") as fitted:
                with self.assertRaisesRegex(ValueError, "support"):
                    build_panel(features, targets, config)
                fitted.assert_not_called()

    def test_registered_monthly_support_failure_retains_insufficient_data_classification(self):
        from src.claims_release_score import failure_metrics

        for field in ("minimum_train", "minimum_train_releases"):
            features, targets, config = fixture()
            config[field] = 10000
            with self.subTest(field=field):
                with self.assertRaises(ValueError) as failure:
                    build_panel(features, targets, config)
                metrics = failure_metrics(failure.exception, "a" * 64)
                self.assertEqual(len(metrics["rows"]), 2)
                self.assertTrue(
                    all(row["status"] == "INSUFFICIENT_DATA" for row in metrics["rows"])
                )
                self.assertTrue(all(row["p_conservative"] == 1.0 for row in metrics["rows"]))

    def test_model_error_aborts_and_malformed_or_unsafe_predictions_raise(self):
        features, targets, config = fixture()
        with (
            patch(
                "src.claims_release_pipeline.fit_models",
                side_effect=ValueError("rank failure"),
            ),
            self.assertRaisesRegex(ValueError, "rank failure"),
        ):
            build_panel(features, targets, config)
        for bad in (0.0, np.nan, np.inf, 1e-320):

            def unsafe(train, y, application, bad=bad):
                result = fake_fit(train, y, application)
                result["predictions"]["candidate"][:] = bad
                return result

            with (
                self.subTest(bad=bad),
                patch("src.claims_release_pipeline.fit_models", side_effect=unsafe),
                self.assertRaises(ValueError),
            ):
                build_panel(features, targets, config)

    def test_qlike_matches_formula_without_clipping(self):
        result, _ = self.build_fake()
        for row in result["panel"].itertuples():
            ratio = row.y / row.prediction
            self.assertAlmostEqual(row.loss, ratio - np.log(ratio) - 1, places=14)

    def test_input_calendars_and_target_end_mapping_fail_before_fit(self):
        for change in (
            "misalign",
            "target_end",
            "string_date",
            "timezone",
            "duplicate",
            "known_y_no_end",
        ):
            features, targets, config = fixture()
            if change == "misalign":
                targets = targets.iloc[1:]
            elif change == "target_end":
                targets.iloc[0, targets.columns.get_loc("target_end")] = features.index[4]
            elif change == "string_date":
                targets["target_end"] = targets.target_end.astype(str)
            elif change == "timezone":
                features.index = features.index.tz_localize("UTC")
                targets.index = targets.index.tz_localize("UTC")
            elif change == "duplicate":
                features.index = targets.index = features.index[:-1].append(
                    features.index[-2:-1]
                )
            else:
                targets.loc[targets.index[-1], "y"] = 1.0
            with (
                self.subTest(change=change),
                patch("src.claims_release_pipeline.fit_models") as fitted,
            ):
                with self.assertRaises(ValueError):
                    build_panel(features, targets, config)
                fitted.assert_not_called()

    def test_nonfinite_nonreal_and_invalid_claim_metadata_rejected(self):
        for change in (
            "inf",
            "bool",
            "negative_y",
            "const",
            "same_day_release",
            "string_release",
            "status",
        ):
            features, targets, config = fixture()
            if change == "inf":
                features.loc[features.index[0], "claim_x"] = np.inf
            elif change == "bool":
                features["claim_x"] = True
            elif change == "negative_y":
                targets.loc[targets.index[0], "y"] = -1.0
            elif change == "const":
                features.loc[features.index[0], "const"] = 2.0
            elif change == "same_day_release":
                features.loc["2018-03-01", "claim_release_date"] = pd.Timestamp("2018-03-01")
            elif change == "string_release":
                features["claim_release_date"] = features.claim_release_date.astype(str)
            else:
                features.loc["2018-03-01", "claim_status"] = "unresolved_correction"
            with (
                self.subTest(change=change),
                patch("src.claims_release_pipeline.fit_models") as fitted,
            ):
                with self.assertRaises(ValueError):
                    build_panel(features, targets, config)
                fitted.assert_not_called()

    def test_config_requires_typed_chronological_bounded_dates_and_floors(self):
        for field, value in (
            ("minimum_train", True),
            ("minimum_train_releases", 0),
            ("source_end", "2025-10-21"),
            ("origin_start", "2018-06-01"),
            ("evaluation", ["2018-03-29", "2018-05-11"]),
            ("origin_end", pd.Timestamp("2018-05-11")),
        ):
            features, targets, config = fixture()
            config[field] = value
            with (
                self.subTest(field=field),
                patch("src.claims_release_pipeline.fit_models") as fitted,
            ):
                with self.assertRaises(ValueError):
                    build_panel(features, targets, config)
                fitted.assert_not_called()

    def test_outside_phase_origin_and_fully_empty_application_months(self):
        features, targets, config = fixture()
        config["evaluation"][0] = "2018-04-04"
        result, _ = self.build_fake(features, targets, config)
        row = result["coverage"].set_index("origin").loc["2018-04-02"]
        self.assertEqual(row.phase, "outside_phase")
        self.assertEqual(row.status, "outside_phase")
        self.assertFalse(row.target_within_phase)
        self.assertIn(pd.Timestamp("2018-04-02"), result["applications"].origin.tolist())
        features["claim_x"] = np.nan
        result, fitted = self.build_fake(features, targets, config)
        self.assertTrue(result["applications"].empty and result["panel"].empty)
        self.assertEqual(result["fits"], [])
        self.assertEqual(len(result["schedules"]), 3)
        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(result["applications"].origin.dtype)
        )
        fitted.assert_not_called()

    def test_unmocked_monthly_models_match_explicit_mature_cohort_and_inputs_unchanged(self):
        features, targets, config = fixture()
        old_features, old_targets, old_config = (
            features.copy(deep=True),
            targets.copy(deep=True),
            copy.deepcopy(config),
        )
        result = build_panel(features, targets, config)
        self.assertEqual(
            set(result), {"applications", "panel", "fits", "coverage", "schedules"}
        )
        train = features.index[
            (features.loc[:, list(ALL)].notna().all(axis=1))
            & targets.y.notna()
            & (targets.target_end <= pd.Timestamp("2018-02-28"))
        ]
        application = features.index[
            (features.index.month == 3) & features.loc[:, list(ALL)].notna().all(axis=1)
        ]
        expected = fit_models(
            features.loc[train], targets.loc[train, "y"], features.loc[application]
        )
        march = result["applications"].loc[result["applications"].origin.dt.month == 3]
        for arm in ("market", "matched", "candidate"):
            np.testing.assert_allclose(
                march[f"pred_{arm}"], expected["predictions"][arm], rtol=1e-13
            )
        self.assertEqual(result["fits"][0]["model_audits"], expected["fits"])
        assert_frame_equal(features, old_features)
        assert_frame_equal(targets, old_targets)
        self.assertEqual(config, old_config)


if __name__ == "__main__":
    unittest.main()
