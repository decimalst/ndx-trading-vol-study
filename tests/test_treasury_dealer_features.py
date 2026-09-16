"""Prewritten invented Treasury release-clock and prior-auction contracts."""

import unittest
from copy import deepcopy
from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import treasury_dealer_features as features


def known(name, day, *, tenor=5, dealer=20, total=100, offering=100):
    return {
        "event_id": name,
        "auction_date": day,
        "availability_after_date": day,
        "status": "KNOWN",
        "original_tenor_years": tenor,
        "reopening": False,
        "primary_dealer_accepted": dealer,
        "competitive_accepted": total,
        "offering_amount_usd": offering,
        "high_yield": "1.5",
        "bid_to_cover": "2.0",
    }


def unknown(name, day, *, tenor=None, status="UNKNOWN"):
    return {
        "event_id": name,
        "auction_date": day,
        "availability_after_date": None,
        "status": status,
        "original_tenor_years": tenor,
        "reopening": None,
        "primary_dealer_accepted": None,
        "competitive_accepted": None,
        "offering_amount_usd": None,
        "high_yield": None,
        "bid_to_cover": None,
    }


def history(tenor=5):
    return [
        known(
            f"d{tenor}_{i}",
            (date(2011, 1, 1) + timedelta(days=14 * i)).isoformat(),
            tenor=tenor,
            dealer=10 + i,
        )
        for i in range(12)
    ]


def calendar():
    return pd.bdate_range("2009-12-30", "2011-08-01")


def current():
    return known("current", "2011-07-01", dealer=50)


def event_audit(result, name="current"):
    return next(e for e in result["events"] if e["event_id"] == name)


class TreasuryDealerFeatureTests(unittest.TestCase):
    def test_unbounded_unknown_clock_keeps_every_later_origin_unknown(self):
        missing = unknown("unbounded", "2011-06-20")
        out = features.build_auction_features(calendar(), [missing])
        self.assertTrue((out["features"].loc["2011-06-20"] == 0).all())
        self.assertTrue(out["features"].loc["2011-06-21":].isna().all().all())
        combined = features.build_auction_features(
            calendar(), history() + [missing, current()]
        )
        self.assertTrue(combined["features"].loc["2011-07-04"].isna().all())
        self.assertTrue(combined["features"].loc["2011-07-05":].isna().all().all())

    def test_unknown_upper_bound_masks_through_last_possible_activation(self):
        event = unknown("bounded", "2011-06-20") | {"clock_upper_bound_date": "2011-06-24"}
        out = features.build_auction_features(calendar(), [event])
        self.assertTrue(out["features"].loc["2011-06-21":"2011-06-27"].isna().all().all())
        self.assertTrue((out["features"].loc["2011-06-28":] == 0).all().all())
        barrier = unknown("bounded", "2011-06-20", tenor=5) | {
            "clock_upper_bound_date": "2011-06-24"
        }
        combined = features.build_auction_features(
            calendar(), history() + [barrier, current()]
        )
        self.assertEqual(event_audit(combined)["mask_reason"], "UNKNOWN_PRIOR_EVENT")

    def test_exact_unknown_clock_masks_pending_interval_and_then_clears(self):
        for release, through in [("2011-06-20", "2011-06-21"), ("2011-06-24", "2011-06-27")]:
            event = unknown("exact", "2011-06-20")
            event["availability_after_date"] = release
            out = features.build_auction_features(calendar(), [event])
            self.assertTrue(out["features"].loc["2011-06-21":through].isna().all().all())
            later = out["features"].index > pd.Timestamp(through)
            self.assertTrue((out["features"].loc[later] == 0).all().all())

    def test_known_delayed_release_is_pending_until_its_single_actual_pulse(self):
        event = current()
        event["availability_after_date"] = "2011-07-08"
        out = features.build_auction_features(calendar(), history() + [event])
        self.assertTrue(out["features"].loc["2011-07-04":"2011-07-08"].isna().all().all())
        self.assertAlmostEqual(out["features"].loc["2011-07-11", "dealer_surprise"], 0.345)
        self.assertTrue((out["features"].loc["2011-07-12":] == 0).all().all())
        self.assertEqual(event_audit(out)["origin"], pd.Timestamp("2011-07-11"))

    def test_recent_unreleased_prior_is_never_replaced_by_older_donor(self):
        late = known("unreleased_prior", "2011-06-20", dealer=90)
        late["availability_after_date"] = "2011-07-10"
        out = features.build_auction_features(calendar(), history() + [late, current()])
        audit = event_audit(out)
        self.assertEqual(audit["mask_reason"], "UNRELEASED_PRIOR_EVENT")
        self.assertIn("unreleased_prior", audit["donor_ids"])
        self.assertNotIn("d5_0", audit["donor_ids"])
        self.assertTrue(out["features"].loc["2011-07-04"].isna().all())

    def test_fixed_columns_full_calendar_floor_and_no_event_zero(self):
        c = calendar()
        out = features.build_auction_features(c, [])
        self.assertEqual(set(out), {"features", "events"})
        self.assertEqual(
            list(out["features"]),
            [
                "auction_count",
                "tenor_3",
                "tenor_5",
                "tenor_7",
                "tenor_10",
                "tenor_30",
                "reopening_count",
                "log_offering_sum",
                "high_yield_sum",
                "bid_to_cover_sum",
                "prior_share_mean_sum",
                "dealer_surprise",
            ],
        )
        pd.testing.assert_index_equal(out["features"].index, c)
        self.assertTrue(out["features"].loc[:"2009-12-31"].isna().all().all())
        self.assertTrue((out["features"].loc["2010-01-01":] == 0).all().all())

    def test_strict_release_clock_exact_twelve_prior_mean_and_no_carry(self):
        out = features.build_auction_features(calendar(), history() + [current()])
        f = out["features"]
        self.assertTrue((f.loc["2011-07-01"] == 0).all())
        row = f.loc["2011-07-04"]
        self.assertEqual(row["auction_count"], 1)
        self.assertEqual(row["tenor_5"], 1)
        self.assertAlmostEqual(row["prior_share_mean_sum"], 0.155)
        self.assertAlmostEqual(row["dealer_surprise"], 0.345)
        self.assertAlmostEqual(row["log_offering_sum"], np.log(100))
        self.assertTrue((f.loc["2011-07-05"] == 0).all())
        self.assertEqual(event_audit(out)["origin"], pd.Timestamp("2011-07-04"))
        self.assertEqual(
            set(event_audit(out)["donor_ids"]), {x["event_id"] for x in history()}
        )

    def test_delayed_current_and_donor_are_unavailable_until_strictly_after_release(self):
        donors = history()
        late = known("late_prior", "2011-06-20", dealer=99)
        late["availability_after_date"] = "2011-07-10"
        event = current()
        event["availability_after_date"] = "2011-07-05"
        out = features.build_auction_features(calendar(), donors + [late, event])
        self.assertEqual(event_audit(out)["origin"], pd.Timestamp("2011-07-06"))
        self.assertIn("late_prior", event_audit(out)["donor_ids"])
        self.assertEqual(event_audit(out)["mask_reason"], "UNRELEASED_PRIOR_EVENT")
        self.assertTrue(out["features"].loc["2011-07-05"].isna().all())

    def test_same_origin_aggregates_events_and_sum_of_log_offerings(self):
        second = known("second", "2011-07-02", tenor=7, dealer=25, offering=200)
        second["reopening"] = True
        out = features.build_auction_features(
            calendar(), history() + history(7) + [current(), second]
        )
        row = out["features"].loc["2011-07-04"]
        self.assertEqual(row["auction_count"], 2)
        self.assertEqual(row["tenor_7"], 1)
        self.assertEqual(row["tenor_5"], 1)
        self.assertEqual(row["reopening_count"], 1)
        self.assertAlmostEqual(row["log_offering_sum"], np.log(100) + np.log(200))
        self.assertEqual(row["high_yield_sum"], 3)
        self.assertEqual(row["bid_to_cover_sum"], 4)
        self.assertAlmostEqual(row["prior_share_mean_sum"], 0.31)
        self.assertAlmostEqual(row["dealer_surprise"], 0.44)

    def test_cold_start_or_exact_same_day_unknown_masks_only_its_pulse(self):
        for events in [
            history()[:11] + [current()],
            history()
            + [
                current(),
                unknown("missing", "2011-07-02") | {"availability_after_date": "2011-07-02"},
            ],
        ]:
            out = features.build_auction_features(calendar(), events)
            f = out["features"]
            self.assertTrue(f.loc["2011-07-04"].isna().all())
            self.assertTrue((f.loc["2011-07-05"] == 0).all())
        out = features.build_auction_features(calendar(), [unknown("u", "2011-07-01")])
        a = event_audit(out, "u")
        self.assertEqual(a["origin"], pd.Timestamp("2011-07-04"))
        self.assertEqual(a["mask_reason"], "UNKNOWN_CURRENT_EVENT")

    def test_unknown_same_or_unresolved_tenor_blocks_but_older_other_and_excluded_do_not(self):
        for tenor in (5, None):
            out = features.build_auction_features(
                calendar(),
                history() + [unknown("barrier", "2011-06-20", tenor=tenor), current()],
            )
            self.assertTrue(out["features"].loc["2011-07-04"].isna().all())
            self.assertEqual(event_audit(out)["mask_reason"], "UNKNOWN_PRIOR_EVENT")
        for item in [
            unknown("old", "2010-12-31", tenor=5),
            unknown("other", "2011-06-20", tenor=7),
            unknown("excluded", "2011-06-20", status="EXCLUDED"),
        ]:
            if item["status"] == "UNKNOWN":
                item["availability_after_date"] = item["auction_date"]
            out = features.build_auction_features(calendar(), history() + [item, current()])
            self.assertAlmostEqual(out["features"].loc["2011-07-04", "dealer_surprise"], 0.345)

    def test_cutoff_tie_is_unknown_and_same_auction_day_never_donates(self):
        tied = history() + [known("tie", "2011-01-01"), current()]
        out = features.build_auction_features(calendar(), tied)
        self.assertEqual(event_audit(out)["mask_reason"], "AMBIGUOUS_PRIOR_CUTOFF")
        peer = known("same_day", "2011-07-01", dealer=99)
        out = features.build_auction_features(calendar(), history() + [peer, current()])
        self.assertNotIn("same_day", event_audit(out)["donor_ids"])
        self.assertAlmostEqual(event_audit(out)["prior_mean"], 0.155)

    def test_zero_dealer_share_is_real_and_two_year_tenor_is_omitted_category(self):
        ds = history(2)
        for e in ds:
            e["primary_dealer_accepted"] = 0
        e = known("zero", "2011-07-01", tenor=2, dealer=0)
        out = features.build_auction_features(calendar(), ds + [e])
        r = out["features"].loc["2011-07-04"]
        self.assertEqual(r["dealer_surprise"], 0)
        self.assertEqual(r["prior_share_mean_sum"], 0)
        self.assertEqual(r["auction_count"], 1)
        self.assertTrue(
            (r[["tenor_3", "tenor_5", "tenor_7", "tenor_10", "tenor_30"]] == 0).all()
        )

    def test_permutation_causal_prefix_and_inputs_are_preserved(self):
        events = history() + [current(), known("future", "2011-07-20")]
        before = deepcopy(events)
        c = calendar()
        full = features.build_auction_features(c, events)
        rev = features.build_auction_features(c, list(reversed(events)))
        pd.testing.assert_frame_equal(full["features"], rev["features"])
        self.assertEqual(full["events"], rev["events"])
        short = features.build_auction_features(c[c <= pd.Timestamp("2011-07-05")], events)
        pd.testing.assert_frame_equal(short["features"], full["features"].loc[:"2011-07-05"])
        self.assertEqual(events, before)

    def test_entire_date_envelope_precedes_financial_conversion(self):
        events = history() + [known("future", "2025-10-21")]
        with patch.object(
            features, "_financial", side_effect=AssertionError("early financial conversion")
        ) as convert:
            with self.assertRaises(ValueError):
                features.build_auction_features(calendar(), events)
            convert.assert_not_called()
        for day in ("2009-12-31", "2011-02-30", "2011-07-01T00:00:00"):
            with self.subTest(day=day), self.assertRaises(ValueError):
                features.build_auction_features(calendar(), [known("bad", day)])

    def test_schema_native_calendar_unknown_values_and_availability_fail_honestly(self):
        examples = []
        for key, value in [
            ("event_id", ""),
            ("availability_after_date", None),
            ("availability_after_date", "2011-06-30"),
            ("original_tenor_years", True),
            ("reopening", 1),
            ("primary_dealer_accepted", 101),
            ("competitive_accepted", 0),
            ("high_yield", "NaN"),
            ("bid_to_cover", "0"),
            ("offering_amount_usd", 0),
        ]:
            e = current()
            e[key] = value
            examples.append([e])
        u = unknown("u", "2011-07-01")
        u["primary_dealer_accepted"] = 99
        examples.append([u])
        examples.append([current(), current()])
        examples.append([current() | {"extra": 1}])
        for events in examples:
            with self.subTest(events=events), self.assertRaises(ValueError):
                features.build_auction_features(calendar(), events)
        for c in [
            calendar()[::-1],
            calendar().tz_localize("UTC"),
            pd.DatetimeIndex(["2011-07-01 12:00"]),
            pd.DatetimeIndex(["2025-10-21"]),
            pd.DatetimeIndex(["2011-07-01", "2011-07-01"]),
        ]:
            with self.assertRaises(ValueError):
                features.build_auction_features(c, [])


if __name__ == "__main__":
    unittest.main()
