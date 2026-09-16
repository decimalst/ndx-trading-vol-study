"""Invented publication ledgers only; no source files or market values are read."""

import copy
import math
import unittest
from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import claims_release_features as features


def row(week, released, value, status="observed", **changes):
    result = {
        "reference_week": week,
        "release_date": released,
        "first_report_value": value,
        "status": status,
        "source_url": "https://example.invalid/generated" if released else None,
        "release_text_sha256": "b" * 64 if released else None,
        "source_comparison": {"kind": "generated", "alfred_date": "2025-10-20"},
    }
    result.update(changes)
    return result


def ledger():
    first = date(2017, 12, 23)
    return [
        row(
            (first + timedelta(weeks=i)).isoformat(),
            (first + timedelta(weeks=i, days=5)).isoformat(),
            2 ** (i + 3),
        )
        for i in range(8)
    ]


def calendar(*days):
    return pd.DatetimeIndex(days, name="origin")


class ClaimsReleaseFeatureTests(unittest.TestCase):
    def build(self, days=("2018-02-08", "2018-02-09"), rows=None):
        return features.build_claim_features(
            calendar(*days), ledger() if rows is None else rows
        )

    def test_exact_schema_prior_day_rule_and_arithmetic(self):
        result = self.build()
        self.assertEqual(
            list(result.columns),
            [
                "claim_m4",
                "claim_age",
                "entry_dow_1",
                "entry_dow_2",
                "entry_dow_3",
                "entry_dow_4",
                "claim_x",
                "claim_reference_week",
                "claim_release_date",
                "claim_age_days",
                "claim_status",
            ],
        )
        pd.testing.assert_index_equal(result.index, calendar("2018-02-08", "2018-02-09"))
        self.assertEqual(result.claim_status.tolist(), ["available", "available"])
        self.assertEqual(
            result.claim_release_date.tolist(),
            [pd.Timestamp("2018-02-01"), pd.Timestamp("2018-02-08")],
        )
        self.assertEqual(
            result.claim_reference_week.tolist(),
            [pd.Timestamp("2018-01-27"), pd.Timestamp("2018-02-03")],
        )
        np.testing.assert_allclose(
            result.claim_m4, [5.5 * math.log(2), 6.5 * math.log(2)], atol=1e-14
        )
        np.testing.assert_allclose(result.claim_x, [2.5 * math.log(2)] * 2, atol=1e-14)
        np.testing.assert_allclose(result.claim_age, [1, 1 / 7])
        self.assertEqual(result.claim_age_days.tolist(), [7, 1])

    def test_expiry_at_eight_days_without_future_release_calendar(self):
        result = self.build(("2018-02-08", "2018-02-09", "2018-02-12"), ledger()[:6])
        self.assertEqual(
            result.claim_status.tolist(), ["available", "stale_release", "stale_release"]
        )
        self.assertEqual(result.claim_age_days.tolist(), [7, 8, 11])
        self.assertTrue(
            result.loc["2018-02-09", ["claim_m4", "claim_age", "claim_x"]].isna().all()
        )
        self.assertEqual(result.loc["2018-02-09", "entry_dow_4"], 1)

    def test_wednesday_release_and_sparse_holiday_calendar_keep_every_origin(self):
        rows = ledger()
        rows[6]["release_date"] = "2018-02-07"
        dates = ("2018-02-07", "2018-02-09", "2018-02-12", "2018-02-14", "2018-02-15")
        result = self.build(dates, rows[:7])
        self.assertEqual(len(result), len(dates))
        self.assertEqual(result.claim_release_date.iloc[0], pd.Timestamp("2018-02-01"))
        self.assertEqual(result.claim_age_days.iloc[1:].tolist(), [2, 5, 7, 8])
        self.assertEqual(result.claim_status.iloc[-1], "stale_release")

    def test_conflicting_audit_values_never_replace_latest_unresolved_value(self):
        class Poison:
            def __float__(self):
                raise AssertionError("Audit value converted")

            def __int__(self):
                raise AssertionError("Audit value converted")

        rows = ledger()
        rows[6].update(
            first_report_value=None,
            status="unresolved_correction",
            source_comparison={
                "kind": "conflict",
                "dol_value": Poison(),
                "alfred_value": Poison(),
            },
        )
        result = self.build(rows=rows)
        self.assertEqual(
            result.claim_status.tolist(), ["available", "latest_unresolved_correction"]
        )
        self.assertEqual(result.claim_release_date.iloc[1], pd.Timestamp("2018-02-08"))
        self.assertTrue(result.iloc[1][["claim_m4", "claim_age", "claim_x"]].isna().all())

    def test_unresolved_week_breaks_subsequent_five_report_chain(self):
        rows = ledger()
        rows[5].update(first_report_value=None, status="unresolved_correction")
        result = self.build(("2018-02-09", "2018-02-16"), rows)
        self.assertEqual(result.claim_status.tolist(), ["prior_unresolved_correction"] * 2)
        self.assertTrue(result[["claim_m4", "claim_age", "claim_x"]].isna().all().all())

    def test_missing_reference_week_is_not_skipped(self):
        rows = ledger()
        del rows[4]
        result = self.build(("2018-02-09",), rows)
        self.assertEqual(result.claim_status.iloc[0], "missing_prior_week")
        self.assertTrue(np.isnan(result.claim_x.iloc[0]))

    def test_no_admitted_release_row_preserves_gap_without_invented_publication(self):
        rows = ledger()[:6] + [row("2018-02-03", None, None, "no_admitted_release")]
        result = self.build(("2018-02-08", "2018-02-09"), rows)
        self.assertEqual(result.claim_status.tolist(), ["available", "stale_release"])
        self.assertEqual(result.claim_release_date.iloc[1], pd.Timestamp("2018-02-01"))

    def test_each_prior_report_must_itself_be_published_before_origin(self):
        rows = ledger()
        rows[4]["release_date"] = "2018-02-09"
        result = self.build(("2018-02-09",), rows)
        self.assertEqual(result.claim_status.iloc[0], "prior_not_yet_released")

    def test_future_values_and_comparison_dates_do_not_change_earlier_features(self):
        rows = ledger()
        before = self.build(("2018-02-08", "2018-02-09"), rows)
        rows[-1]["first_report_value"] = 10**200
        for item in rows:
            item["source_comparison"] = {
                "kind": "unrelated",
                "alfred_date": "1901-01-01",
                "value": -999,
            }
        after = self.build(("2018-02-08", "2018-02-09"), rows)
        pd.testing.assert_frame_equal(before, after)

    def test_input_order_does_not_change_result_and_inputs_unchanged(self):
        rows = ledger()
        before = copy.deepcopy(rows)
        pd.testing.assert_frame_equal(self.build(rows=rows), self.build(rows=rows[::-1]))
        self.assertEqual(rows, before)

    def test_no_prior_release_and_empty_calendar_are_explicit(self):
        result = self.build(("2017-12-28",), ledger())
        self.assertEqual(result.claim_status.iloc[0], "no_prior_release")
        self.assertTrue(pd.isna(result.claim_release_date.iloc[0]))
        empty = features.build_claim_features(pd.DatetimeIndex([], name="origin"), ledger())
        self.assertEqual(len(empty), 0)
        self.assertEqual(list(empty.columns), list(result.columns))

    def test_calendar_validation_and_postcutoff_fail_before_feature_math(self):
        bad_calendars = [
            pd.Index(["2018-02-09"]),
            calendar("2018-02-09", "2018-02-08"),
            calendar("2018-02-09", "2018-02-09"),
            calendar("2018-02-09 12:00"),
            calendar("2025-10-21"),
            pd.DatetimeIndex([pd.NaT]),
            pd.date_range("2018-02-09", periods=1, tz="UTC"),
        ]
        with patch.object(
            features, "_log_value", side_effect=AssertionError("early logarithm")
        ) as log:
            for dates in bad_calendars:
                with self.subTest(dates=dates), self.assertRaises(ValueError):
                    features.build_claim_features(dates, ledger())
            rows = ledger() + [row("2025-10-18", "2025-10-23", 100)]
            with self.assertRaises(ValueError):
                self.build(rows=rows)
            log.assert_not_called()

    def test_duplicate_weeks_or_release_dates_are_ambiguous(self):
        for rows in (ledger() + [ledger()[0]], ledger()):
            if len(rows) == 8:
                rows[1]["release_date"] = rows[2]["release_date"]
            with self.assertRaises(ValueError):
                self.build(rows=rows)

    def test_invalid_ledger_values_dates_and_statuses_fail_closed(self):
        for field, value in [
            ("reference_week", "2018-02-04"),
            ("release_date", "2018-02-03"),
            ("release_date", "02/08/2018"),
            ("first_report_value", True),
            ("first_report_value", 512.0),
            ("first_report_value", 0),
            ("first_report_value", -1),
            ("first_report_value", None),
            ("status", "revised"),
            ("source_comparison", None),
        ]:
            rows = ledger()
            rows[6][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.build(rows=rows)
        rows = ledger()
        rows[6]["status"] = "unresolved_correction"
        with self.assertRaises(ValueError):
            self.build(rows=rows)


if __name__ == "__main__":
    unittest.main()
