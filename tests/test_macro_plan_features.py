"""Calendar-feature contracts before any association with market outcomes."""
import unittest

import numpy as np
import pandas as pd

from src.macro_plan_features import build_plan_features, nominal_window, validate_bls_plans


def plan(event, announced, planned):
    return {"event_type": event, "announced_at": pd.Timestamp(announced, tz="America/New_York").isoformat(),
            "planned_at": pd.Timestamp(planned, tz="America/New_York").isoformat(),
            "source_id": f"{event}-{announced}", "source_sha256": "a"*64}


def records():
    return [plan(event, "2017-03-10 08:30", "2017-04-14 08:30") for event in ("cpi", "nfp")]


def fed(year=2017, dates=None):
    full_dates = [str(date.date()) for date in pd.bdate_range(f"{year}-01-01", periods=250)[[10, 40, 70, 100, 130, 170, 220]]]
    full_dates.append(f"{year}-04-14")
    return [{"year": year, "announced_date": f"{year-1}-05-15", "final_dates": dates or full_dates,
             "source_id": f"annual-{year}", "source_sha256": "b"*64}]


class TestPlanFeatures(unittest.TestCase):
    def test_nominal_weekday_window_and_dst_elapsed_hours(self):
        a, b = nominal_window(pd.Timestamp("2020-03-06"))
        self.assertEqual(str(a), "2020-03-06 16:00:00-05:00")
        self.assertEqual(str(b), "2020-03-09 09:30:00-04:00")
        self.assertEqual((b-a).total_seconds()/3600, 64.5)
        a, b = nominal_window(pd.Timestamp("2020-10-30"))
        self.assertEqual((b-a).total_seconds()/3600, 66.5)

    def test_good_friday_remains_nominal_next_day_without_exchange_inference(self):
        _, end = nominal_window(pd.Timestamp("2017-04-13"))
        self.assertEqual(end.date().isoformat(), "2017-04-14")
        origins = pd.DatetimeIndex(["2017-04-13"])
        result, _ = build_plan_features(origins, pd.Series(pd.to_datetime(["2017-04-12"]), index=origins), records(), fed())
        self.assertEqual(result.iloc[0].cpi_plan, 1)
        self.assertEqual(result.iloc[0].nfp_plan, 1)
        self.assertEqual(result.iloc[0].fomc_plan, 1)

    def test_monday_holiday_is_deliberately_not_skipped(self):
        _, end = nominal_window(pd.Timestamp("2020-05-22"))
        self.assertEqual(end.date().isoformat(), "2020-05-25")
        plans = [plan(event, "2020-04-10 08:30", "2020-05-26 08:30") for event in ("cpi", "nfp")]
        origins = pd.DatetimeIndex(["2020-05-22"])
        result, _ = build_plan_features(origins, pd.Series(pd.to_datetime(["2020-05-21"]), index=origins), plans, fed(2020))
        self.assertEqual(result.iloc[0].cpi_plan, 0)

    def test_unknown_months_are_missing_not_non_events(self):
        origins = pd.DatetimeIndex(["2017-05-01"])
        result, audit = build_plan_features(origins, pd.Series(pd.to_datetime(["2017-04-28"]), index=origins), records(), fed())
        self.assertTrue(np.isnan(result.iloc[0].cpi_plan))
        self.assertEqual(audit[0]["cpi_missing_months"], ["2017-05"])

    def test_cross_month_window_requires_both_months_known(self):
        origins = pd.DatetimeIndex(["2017-03-31"])
        result, audit = build_plan_features(origins, pd.Series(pd.to_datetime(["2017-03-30"]), index=origins), records(), fed())
        self.assertTrue(np.isnan(result.iloc[0].cpi_plan))
        self.assertEqual(audit[0]["cpi_missing_months"], ["2017-03"])

    def test_source_publication_on_cutoff_date_is_not_yet_eligible(self):
        origins = pd.DatetimeIndex(["2017-04-13"])
        plans = [plan(event, "2017-04-12 08:30", "2017-04-14 08:30") for event in ("cpi", "nfp")]
        result, _ = build_plan_features(origins, pd.Series(pd.to_datetime(["2017-04-12"]), index=origins), plans, fed())
        self.assertTrue(np.isnan(result.iloc[0].cpi_plan))

    def test_later_cancellations_and_realized_event_dates_never_change_original_plan(self):
        origins = pd.DatetimeIndex(["2017-04-13"])
        cutoff = pd.Series(pd.to_datetime(["2017-04-12"]), index=origins)
        before, _ = build_plan_features(origins, cutoff, records(), fed())
        revised = [dict(record, canceled=True, actual_release="2099-01-01", revised_date="2017-04-21") for record in records()]
        after, _ = build_plan_features(origins, cutoff, revised, fed())
        pd.testing.assert_frame_equal(before, after)

    def test_future_sources_do_not_change_earlier_features(self):
        origins = pd.DatetimeIndex(["2017-04-13"])
        cutoff = pd.Series(pd.to_datetime(["2017-04-12"]), index=origins)
        before, _ = build_plan_features(origins, cutoff, records(), fed())
        future = [plan(event, "2017-04-14 08:30", "2017-05-12 08:30") for event in ("cpi", "nfp")]
        after, _ = build_plan_features(origins, cutoff, records()+future, fed())
        pd.testing.assert_frame_equal(before, after)

    def test_ambiguous_same_month_original_plans_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "month"):
            validate_bls_plans(records()+[plan("cpi", "2017-03-11 08:30", "2017-04-15 08:30")])

    def test_naive_timestamps_and_source_after_plan_are_rejected(self):
        bad = records()
        bad[0]["announced_at"] = "2017-03-10 08:30"
        with self.assertRaises(ValueError):
            validate_bls_plans(bad)
        with self.assertRaises(ValueError):
            validate_bls_plans([plan("cpi", "2017-04-15 08:30", "2017-04-14 08:30")])

    def test_fomc_requires_an_eligible_annual_schedule(self):
        origins = pd.DatetimeIndex(["2017-04-13"])
        result, _ = build_plan_features(origins, pd.Series(pd.to_datetime(["2017-04-12"]), index=origins), records(), [])
        self.assertTrue(np.isnan(result.iloc[0].fomc_plan))

    def test_truncated_annual_schedule_cannot_make_omitted_meetings_non_events(self):
        origins = pd.DatetimeIndex(["2017-04-13"])
        with self.assertRaisesRegex(ValueError, "annual"):
            build_plan_features(origins, pd.Series(pd.to_datetime(["2017-04-12"]), index=origins), records(), fed(dates=["2017-04-14"]))

    def test_fomc_uses_only_original_final_dates_and_no_emergency_injection(self):
        origins = pd.DatetimeIndex(["2017-04-12"])
        result, _ = build_plan_features(origins, pd.Series(pd.to_datetime(["2017-04-11"]), index=origins), records(), fed())
        self.assertEqual(result.iloc[0].fomc_plan, 0)


if __name__ == "__main__":
    unittest.main()
