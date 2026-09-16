"""Prewritten synthetic civil-date, frozen-column, support and rank contracts."""

import unittest
from copy import deepcopy

import numpy as np
import pandas as pd

from src import calendar_variance_features as old
from src import civil_quarter_features as civil


def original(dates):
    dates = pd.DatetimeIndex(dates, name="date")
    frame = pd.DataFrame(
        np.arange(len(dates) * 18, dtype=float).reshape(len(dates), 18),
        index=dates,
        columns=old.ALL_FEATURES,
    )
    frame["const"] = 1.0
    frame["feature_cutoff_date"] = pd.Series(dates, index=dates).shift(1)
    return frame


def balanced(n=200):
    return pd.DataFrame(
        {column: (np.arange(n) % 2).astype(float) for column in civil.CIVIL_FEATURES}
    )


class CivilQuarterFeatures(unittest.TestCase):
    def test_fixed_names_counts_order_and_old_information_preserved(self):
        source = original(pd.bdate_range("2015-12-28", periods=12))
        source.loc[source.index[3], "cpi_plan"] = np.nan
        before = source.copy(deep=True)
        augmented = civil.augment_features(source)
        self.assertEqual(civil.OLD_FEATURES, old.ALL_FEATURES)
        self.assertEqual(civil.MONTHS, tuple(f"month_{month}" for month in range(2, 13)))
        self.assertEqual(civil.NUISANCE, civil.MONTHS + ("month_end5", "year_end5"))
        self.assertEqual(civil.MEMORY, "quarter_end5")
        self.assertEqual(civil.MODELS, ("mean", "baseline", "quarter"))
        self.assertEqual((len(civil.BASE), len(civil.ALL_FEATURES)), (31, 32))
        self.assertEqual(
            tuple(augmented.columns), civil.ALL_FEATURES + ("feature_cutoff_date",)
        )
        pd.testing.assert_frame_equal(source, before, check_exact=True)
        pd.testing.assert_frame_equal(
            augmented.loc[:, source.columns], source, check_exact=True
        )
        permuted = source.loc[:, source.columns[::-1]]
        pd.testing.assert_frame_equal(
            civil.augment_features(permuted), augmented, check_exact=True
        )

    def test_civil_next_weekday_edges_and_no_exchange_calendar(self):
        cases = {
            "2024-02-23": "2024-02-26",
            "2024-02-24": "2024-02-26",
            "2024-02-25": "2024-02-26",
            "2024-02-28": "2024-02-29",
            "2024-02-29": "2024-03-01",
            "2023-12-29": "2024-01-01",
            "2024-03-08": "2024-03-11",
            "2024-11-01": "2024-11-04",
        }
        for origin, expected in cases.items():
            with self.subTest(origin=origin):
                self.assertEqual(
                    civil.nominal_date(pd.Timestamp(origin)), pd.Timestamp(expected)
                )
        # January1 remains the nominal endpoint although it is a market holiday.
        self.assertEqual(civil.nominal_date(pd.Timestamp("2023-12-29")).day, 1)

    def test_final_five_civil_dates_boundaries_leap_year_and_december_exclusion(self):
        cases = {
            "2024-03-25": (3, 0, 0, 0),
            "2024-03-26": (3, 1, 0, 1),
            "2024-06-24": (6, 0, 0, 0),
            "2024-06-25": (6, 1, 0, 1),
            "2024-09-24": (9, 0, 0, 0),
            "2024-09-25": (9, 1, 0, 1),
            "2024-12-26": (12, 1, 1, 0),
            "2024-02-23": (2, 1, 0, 0),
            "2023-02-22": (2, 0, 0, 0),
            "2023-02-23": (2, 1, 0, 0),
            "2023-12-29": (1, 0, 0, 0),
        }
        source = original(sorted(cases))
        result = civil.augment_features(source)
        for day, (month, end, year, quarter) in cases.items():
            row = result.loc[day]
            self.assertEqual(
                tuple(row[["month_end5", "year_end5", "quarter_end5"]]), (end, year, quarter)
            )
            for m in range(2, 13):
                self.assertEqual(row[f"month_{m}"], float(m == month))

    def test_future_source_rows_and_prices_cannot_change_civil_predictors(self):
        source = original(pd.bdate_range("2024-03-01", periods=60))
        expected = civil.augment_features(source)
        prefix = source.iloc[:20].copy()
        prefix.loc[:, old.ALL_FEATURES[1:]] = -999.0
        actual = civil.augment_features(prefix)
        pd.testing.assert_frame_equal(
            actual.loc[:, civil.CIVIL_FEATURES],
            expected.loc[prefix.index, civil.CIVIL_FEATURES],
            check_exact=True,
        )
        changed = source.copy()
        changed.index = (
            changed.index[:20]
            .append(pd.bdate_range("2026-01-01", periods=40))
            .rename(source.index.name)
        )
        changed["feature_cutoff_date"] = pd.Series(changed.index, index=changed.index).shift(1)
        result = civil.augment_features(changed)
        pd.testing.assert_frame_equal(
            result.iloc[:20].loc[:, civil.CIVIL_FEATURES],
            expected.iloc[:20].loc[:, civil.CIVIL_FEATURES],
            check_exact=True,
            check_freq=False,
        )

    def test_invalid_schema_dates_and_cutoffs_reject_without_repair(self):
        source = original(pd.bdate_range("2024-01-01", periods=5))
        for fault in (
            "missing",
            "extra",
            "duplicate_column",
            "duplicate_date",
            "order",
            "timezone",
            "time",
            "cutoff",
        ):
            frame = deepcopy(source)
            if fault == "missing":
                frame = frame.drop(columns="cpi_plan")
            elif fault == "extra":
                frame["future_target"] = 1.0
            elif fault == "duplicate_column":
                frame = pd.concat([frame, frame[["lvix"]]], axis=1)
            elif fault == "duplicate_date":
                frame.index = frame.index[:-1].append(frame.index[-2:-1])
            elif fault == "order":
                frame = frame.iloc[::-1]
            elif fault == "timezone":
                frame.index = frame.index.tz_localize("UTC")
            elif fault == "time":
                frame.index = frame.index + pd.Timedelta(hours=1)
            else:
                frame.loc[frame.index[2], "feature_cutoff_date"] = frame.index[2]
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                civil.augment_features(frame)

    def test_each_support_threshold_and_both_binary_classes_are_literal(self):
        for scope, quarter_min, month_min in (
            ("train", 20, 20),
            ("phase", 30, 20),
            ("slice", 15, 10),
        ):
            limits = {
                **dict.fromkeys(civil.MONTHS, month_min),
                "month_end5": 20,
                "year_end5": 5,
                "quarter_end5": quarter_min,
            }
            for column, minimum in limits.items():
                for minority in (0.0, 1.0):
                    frame = balanced()
                    frame[column] = 1.0 - minority
                    frame.loc[: minimum - 1, column] = minority
                    before = frame.copy(deep=True)
                    audit = civil.civil_support(frame, scope)
                    self.assertEqual(audit["n"], len(frame))
                    self.assertEqual(audit["scope"], scope)
                    self.assertEqual(audit["groups"][column]["minimum_per_class"], minimum)
                    self.assertEqual(
                        min(audit["groups"][column]["ones"], audit["groups"][column]["zeros"]),
                        minimum,
                    )
                    pd.testing.assert_frame_equal(frame, before, check_exact=True)
                    frame.loc[minimum - 1, column] = 1.0 - minority
                    with (
                        self.subTest(scope=scope, column=column, minority=minority),
                        self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"),
                    ):
                        civil.civil_support(frame, scope)

    def test_unknown_nonbinary_and_invalid_support_scope_cannot_be_dropped(self):
        for value in (np.nan, np.inf, -1.0, 0.5, 2.0):
            frame = balanced()
            frame.loc[0, "quarter_end5"] = value
            with self.assertRaises(ValueError):
                civil.civil_support(frame, "train")
        with self.assertRaises(ValueError):
            civil.civil_support(balanced(), "evaluation")
        with self.assertRaises(ValueError):
            civil.civil_support(balanced().drop(columns="month_2"), "train")

    def test_civil_rank_is_full_fifteen_and_duplicate_interaction_rejects(self):
        frame = civil.augment_features(original(pd.bdate_range("2010-01-01", "2015-12-31")))
        audit = civil.require_civil_rank(frame)
        self.assertEqual(audit["columns"], ["const", *civil.CIVIL_FEATURES])
        self.assertEqual(audit["rank"], 15)
        self.assertEqual(audit["relative_threshold"], 1e-10)
        self.assertEqual(len(audit["singular_values"]), 15)
        self.assertGreater(min(audit["singular_values"]), audit["threshold"])
        bad = frame.copy()
        bad["quarter_end5"] = bad["year_end5"]
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"):
            civil.require_civil_rank(bad)
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"):
            civil.require_civil_rank(frame.iloc[:14])
        bad = frame.copy()
        bad.iloc[0, bad.columns.get_loc("quarter_end5")] = np.nan
        with self.assertRaises(ValueError):
            civil.require_civil_rank(bad)


if __name__ == "__main__":
    unittest.main()
