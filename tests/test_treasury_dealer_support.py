"""Generated support contracts written before the source implementation."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.treasury_dealer_models import ALL
from src.treasury_dealer_support import audit_support

TENORS = (2, 3, 5, 7, 10, 30)
AUCTION = (
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
)


def synthetic_inputs():
    calendar = pd.bdate_range("2010-01-04", periods=200)
    features = pd.DataFrame(1.0, index=calendar, columns=ALL)
    features.loc[:, list(AUCTION)] = 0.0
    features.loc[calendar[0], list(ALL)] = np.nan
    features["treasury_cutoff_date"] = pd.Series(calendar, index=calendar).shift(1)
    targets = pd.DataFrame(
        {"y": 1.0, "target_end": pd.Series(calendar, index=calendar).shift(-5)}
    )
    targets.loc[calendar[-5:], "y"] = np.nan
    audits, tenors = [], {}
    for position in range(2, 200, 2):
        for tenor in TENORS:
            event_id = f"event-{position}-{tenor}"
            audits.append(
                {
                    "event_id": event_id,
                    "auction_date": calendar[position - 1].date().isoformat(),
                    "source_status": "KNOWN",
                    "origin": calendar[position],
                    "status": "SUPPORTED",
                    "origin_masked": False,
                    "mask_reason": None,
                    "donor_ids": [],
                    "share": "opaque-to-support",
                    "prior_mean": "opaque",
                }
            )
            tenors[event_id] = tenor
        features.loc[calendar[position], "auction_count"] = 6.0
        features.loc[calendar[position], [f"tenor_{t}" for t in TENORS[1:]]] = 1.0
    iso = lambda position: calendar[position].date().isoformat()
    config = {
        "forecast": {
            "origin_start": iso(60),
            "origin_end": iso(179),
            "source_end": iso(199),
            "development": [iso(60), iso(109)],
            "evaluation": [iso(110), iso(179)],
            "minimum_train": 20,
        },
        "stability": [[iso(110), iso(144)], [iso(145), iso(179)]],
        "support": {
            "phase_daily": 40,
            "slice_daily": 20,
            "offset_daily": 5,
            "training_activation_dates": 10,
            "phase_activation_dates": 10,
            "slice_activation_dates": 5,
            "phase_active_dates_per_offset": 1,
            "training_events_per_tenor": 2,
            "phase_events_per_tenor": 2,
            "slice_events_per_tenor": 1,
        },
    }
    return calendar, features, targets, audits, tenors, config


class TreasuryDealerSupportTests(unittest.TestCase):
    def setUp(self):
        self.calendar, self.features, self.targets, self.audits, self.tenors, self.config = (
            synthetic_inputs()
        )

    def audit(self, **changes):
        return audit_support(
            changes.get("calendar", self.calendar),
            changes.get("features", self.features),
            changes.get("targets", self.targets),
            changes.get("events", self.audits),
            event_tenors=changes.get("tenors", self.tenors),
            config=changes.get("config", self.config),
        )

    def test_literal_counts_common_maturity_and_full_calendar_offsets(self):
        result = self.audit()
        self.assertEqual(
            set(result),
            {"status", "passed", "monthly", "phases", "slices", "offsets", "discrepancies"},
        )
        self.assertEqual(
            (result["status"], result["passed"], result["discrepancies"]),
            ("SUPPORT_PASS", True, []),
        )
        first = result["monthly"][0]
        self.assertEqual(first["fit_origin"], self.calendar[60].date().isoformat())
        self.assertEqual(first["training_cutoff"], self.calendar[59].date().isoformat())
        self.assertEqual(
            (first["train_n"], first["train_activation_dates"], first["train_event_n"]),
            (54, 27, 162),
        )
        self.assertEqual(first["train_events_per_tenor"], {str(t): 27 for t in TENORS})
        for phase, daily, active in (("development", 45, 23), ("evaluation", 70, 35)):
            row = result["phases"][phase]
            self.assertEqual(
                (row["daily_n"], row["activation_dates"], row["event_n"]),
                (daily, active, active * 6),
            )
            self.assertEqual(row["events_per_tenor"], {str(t): active for t in TENORS})
            self.assertEqual(
                [row["origin_start"], row["origin_end"]], self.config["forecast"][phase]
            )
        self.assertEqual(
            [(r["daily_n"], r["activation_dates"]) for r in result["slices"]],
            [(35, 18), (35, 17)],
        )
        self.assertEqual([r["daily_n"] for r in result["offsets"]["development"]], [9] * 5)
        self.assertEqual(
            [r["activation_dates"] for r in result["offsets"]["development"]], [5, 4, 5, 4, 5]
        )
        self.assertEqual(
            [r["activation_dates"] for r in result["offsets"]["evaluation"]], [7] * 5
        )

    def test_zero_and_cancelling_surprises_count_dates_once_and_events_six_times(self):
        baseline = self.audit()
        self.features["dealer_surprise"] = np.arange(200) % 3 - 1.0
        self.features.loc[self.calendar[0], "dealer_surprise"] = np.nan
        self.assertEqual(self.audit(), baseline)

    def test_target_blind_schedule_keeps_unscored_first_query(self):
        baseline = self.audit()
        self.targets.loc[self.calendar[60], "y"] = np.nan
        result = self.audit()
        self.assertEqual(result["monthly"][0], baseline["monthly"][0])
        self.assertEqual(result["phases"]["development"]["daily_n"], 44)
        self.assertEqual(result["phases"]["development"]["activation_dates"], 22)

    def test_exact_prior_session_label_maturity_and_future_label_independence(self):
        baseline = self.audit()
        self.targets.loc[self.calendar[54], "y"] = np.nan
        result = self.audit()
        self.assertEqual(
            (result["monthly"][0]["train_n"], result["monthly"][0]["train_activation_dates"]),
            (53, 26),
        )
        self.targets.loc[self.calendar[110] :, "y"] = np.nan
        later = self.audit()
        self.assertEqual(later["monthly"][0], result["monthly"][0])
        self.assertEqual(
            [r["fit_origin"] for r in later["monthly"]],
            [r["fit_origin"] for r in baseline["monthly"]],
        )
        self.assertEqual(later["phases"]["evaluation"]["daily_n"], 0)
        self.assertFalse(later["passed"])

    def test_missing_common_market_feature_keeps_original_offset_and_event_mask(self):
        self.features.loc[self.calendar[70], "liv"] = np.nan
        result = self.audit()
        dev = result["phases"]["development"]
        self.assertEqual(
            (dev["daily_n"], dev["activation_dates"], dev["event_n"]), (44, 22, 132)
        )
        self.assertEqual(
            [r["daily_n"] for r in result["offsets"]["development"]], [8, 9, 9, 9, 9]
        )
        self.assertEqual(
            [r["activation_dates"] for r in result["offsets"]["development"]], [4, 4, 5, 4, 5]
        )

    def test_supported_event_on_masked_origin_is_not_counted(self):
        for event in self.audits:
            if event["origin"] == self.calendar[70]:
                event["origin_masked"] = True
        self.features.loc[self.calendar[70], list(AUCTION)] = np.nan
        result = self.audit()
        self.assertEqual(result["phases"]["development"]["activation_dates"], 22)
        self.assertEqual(
            result["phases"]["development"]["events_per_tenor"], {str(t): 22 for t in TENORS}
        )

    def test_masked_current_event_cannot_be_relabelled_certified_zero(self):
        for event in self.audits:
            if event["origin"] == self.calendar[70]:
                event["origin_masked"] = True
        self.features.loc[self.calendar[70], list(AUCTION)] = 0.0
        with self.assertRaisesRegex(ValueError, "[Mm]asked|[Uu]nsupported"):
            self.audit()

    def test_every_failure_retained_without_fitting_or_short_circuit(self):
        self.config["forecast"]["minimum_train"] = 55
        self.config["support"].update(
            training_activation_dates=28,
            training_events_per_tenor=28,
            phase_daily=46,
            phase_activation_dates=24,
            phase_events_per_tenor=24,
            slice_daily=36,
            slice_activation_dates=19,
            slice_events_per_tenor=19,
            offset_daily=10,
            phase_active_dates_per_offset=6,
        )
        with patch(
            "src.treasury_dealer_pipeline.fit_models",
            side_effect=AssertionError("No fitting in support"),
        ):
            result = self.audit()
        self.assertEqual(result["status"], "INSUFFICIENT_DATA")
        self.assertFalse(result["passed"])
        scopes = {r["scope"] for r in result["discrepancies"]}
        self.assertTrue(
            {
                "monthly:2010-03",
                "phase:development",
                "slice:0",
                "slice:1",
                "offset:development:0",
            }
            <= scopes
        )
        self.assertEqual(result["monthly"][0]["status"], "INSUFFICIENT_DATA")
        self.assertEqual(len(result["offsets"]["development"]), 5)
        self.assertTrue(
            all(
                set(r) == {"scope", "metric", "observed", "required"}
                for r in result["discrepancies"]
            )
        )

    def test_no_complete_month_retains_honest_no_fit_record(self):
        self.features.loc[self.calendar.month == 4, list(ALL)] = np.nan
        result = self.audit()
        row = next(r for r in result["monthly"] if r["month"] == "2010-04")
        self.assertEqual(row["status"], "NO_COMPLETE_ORIGIN")
        self.assertTrue(row["passed"])
        self.assertGreater(row["requested_n"], 0)
        self.assertEqual(row["application_n"], 0)
        for field in (
            "fit_origin",
            "training_cutoff",
            "train_n",
            "train_activation_dates",
            "train_event_n",
            "train_events_per_tenor",
        ):
            self.assertIsNone(row[field])

    def test_event_identity_membership_and_tenor_counts_cannot_be_fabricated(self):
        with self.assertRaises(ValueError):
            self.audit(events=self.audits + [self.audits[0]])
        with self.assertRaises(ValueError):
            self.audit(events=self.audits[:-1])
        wrong = dict(self.tenors)
        wrong["event-70-2"] = 3
        with self.assertRaises(ValueError):
            self.audit(tenors=wrong)
        self.features.loc[self.calendar[70], "auction_count"] = 5.0
        with self.assertRaises(ValueError):
            self.audit()

    def test_unknown_and_excluded_audits_are_retained_without_donating_support(self):
        for event in self.audits:
            if event["origin"] == self.calendar[70]:
                event["origin_masked"] = True
                if event["event_id"] == "event-70-2":
                    event.update(
                        source_status="UNKNOWN",
                        status="UNKNOWN",
                        mask_reason="UNKNOWN_CURRENT_EVENT",
                    )
                    self.tenors[event["event_id"]] = None
        self.audits.append(
            {
                "event_id": "excluded",
                "auction_date": "2010-02-01",
                "source_status": "EXCLUDED",
                "origin": pd.NaT,
                "status": "EXCLUDED",
                "origin_masked": False,
                "mask_reason": "EXCLUDED_EVENT",
            }
        )
        self.tenors["excluded"] = 10
        self.features.loc[self.calendar[70], list(AUCTION)] = np.nan
        self.assertEqual(self.audit()["phases"]["development"]["event_n"], 132)

    def test_invalid_calendar_target_cutoff_numeric_or_future_audit_fails(self):
        for kind in ("endpoint", "cutoff", "infinity", "fractional_count"):
            features, targets = self.features.copy(), self.targets.copy()
            if kind == "endpoint":
                targets.loc[self.calendar[20], "target_end"] = self.calendar[24]
            if kind == "cutoff":
                features.loc[self.calendar[20], "treasury_cutoff_date"] = self.calendar[20]
            if kind == "infinity":
                features.loc[self.calendar[20], "liv"] = np.inf
            if kind == "fractional_count":
                features.loc[self.calendar[20], "auction_count"] = 6.5
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.audit(features=features, targets=targets)
        with self.assertRaises(ValueError):
            self.audit(calendar=self.calendar.delete(20))
        bad = copy.deepcopy(self.audits)
        bad[-1]["auction_date"] = "2025-11-03"
        with self.assertRaises(ValueError):
            self.audit(events=bad)

    def test_default_production_floors_and_empty_requested_months(self):
        result = self.audit(config=None)
        self.assertEqual(len(result["monthly"]), 118)
        self.assertTrue(all(r["status"] == "NO_REQUESTED_ORIGINS" for r in result["monthly"]))
        requirements = {r["metric"]: r["required"] for r in result["discrepancies"]}
        self.assertEqual(requirements["phase_daily"], 505)
        self.assertEqual(requirements["phase_activation_dates"], 104)
        self.assertEqual(requirements["slice_daily"], 252)
        self.assertEqual(requirements["slice_activation_dates"], 52)
        self.assertEqual(requirements["offset_daily"], 63)
        self.assertEqual(requirements["phase_active_dates_per_offset"], 20)

    def test_config_floors_and_windows_must_be_typed_and_ordered(self):
        for field, value in (
            ("phase_daily", True),
            ("slice_daily", 0),
            ("training_events_per_tenor", 2.5),
        ):
            config = copy.deepcopy(self.config)
            config["support"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.audit(config=config)
        config = copy.deepcopy(self.config)
        config["stability"].reverse()
        with self.assertRaises(ValueError):
            self.audit(config=config)

    def test_input_preservation_and_date_units(self):
        features, targets, events, config = (
            self.features.copy(deep=True),
            self.targets.copy(deep=True),
            copy.deepcopy(self.audits),
            copy.deepcopy(self.config),
        )
        expected = self.audit()
        for unit in ("ms", "us", "ns"):
            f, t = self.features.copy(), self.targets.copy()
            f.index = f.index.as_unit(unit)
            t.index = t.index.as_unit(unit)
            self.assertEqual(
                self.audit(calendar=self.calendar.as_unit(unit), features=f, targets=t),
                expected,
            )
        pd.testing.assert_frame_equal(self.features, features)
        pd.testing.assert_frame_equal(self.targets, targets)
        self.assertEqual(self.audits, events)
        self.assertEqual(self.config, config)


if __name__ == "__main__":
    unittest.main()
