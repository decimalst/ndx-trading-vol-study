"""Prewritten independent Treasury reconstruction tests; all observations invented."""

import copy
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import verify_treasury_dealer_forecasts as verify


def known(name, day, tenor=5, dealer=20):
    return dict(
        event_id=name,
        auction_date=day,
        availability_after_date=day,
        status="KNOWN",
        original_tenor_years=tenor,
        reopening=False,
        primary_dealer_accepted=dealer,
        competitive_accepted=100,
        offering_amount_usd=1000,
        high_yield="1.50",
        bid_to_cover="2.25",
    )


def history(tenor=5):
    return [
        known(f"{tenor}_{i}", day.strftime("%Y-%m-%d"), tenor, 10 + i)
        for i, day in enumerate(pd.date_range("2011-01-01", periods=12, freq="14D"))
    ]


def unknown(day="2011-06-20", **updates):
    item = known("unknown", day)
    item.update(
        status="UNKNOWN",
        availability_after_date=None,
        original_tenor_years=None,
        reopening=None,
        primary_dealer_accepted=None,
        competitive_accepted=None,
        offering_amount_usd=None,
        high_yield=None,
        bid_to_cover=None,
    )
    return item | updates


def fixture():
    index = pd.bdate_range("2010-01-01", periods=850)
    rng = np.random.default_rng(390111)
    n = len(index)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    opened = close * np.exp(rng.normal(0, 0.008, n))
    daily = pd.DataFrame(
        {
            "open": opened,
            "high": np.maximum(opened, close) * np.exp(rng.uniform(0.002, 0.016, n)),
            "low": np.minimum(opened, close) * np.exp(-rng.uniform(0.002, 0.016, n)),
            "close": close,
            "adj close": close * np.exp(np.cumsum(rng.normal(0, 0.0001, n))),
            "volume": np.exp(rng.normal(14, 0.4, n)),
        },
        index=index,
    )
    cross = pd.DataFrame(
        80 * np.exp(np.cumsum(rng.normal(0, 0.012, (n, 5)), axis=0)),
        index=index,
        columns=("hyg", "tlt", "gld", "uso", "uup"),
    )
    iv = pd.DataFrame(
        np.exp(rng.normal(3, 0.2, (n, 3))), index=index, columns=("vxn", "vix", "vix9d")
    )
    events = []
    for day in index[:-1]:
        for _ in range(rng.poisson(0.95)):
            item = known(
                f"e{len(events)}",
                day.strftime("%Y-%m-%d"),
                int(rng.choice([2, 3, 5, 7, 10, 30])),
                int(rng.integers(1, 99)),
            )
            item.update(
                offering_amount_usd=int(rng.integers(1000, 10000)),
                high_yield=f"{rng.uniform(0.1, 5):.6f}",
                bid_to_cover=f"{rng.uniform(1.1, 4):.6f}",
                reopening=bool(rng.integers(2)),
            )
            events.append(item)
    config = dict(
        origin_start=index[-80].strftime("%Y-%m-%d"),
        origin_end=index[-1].strftime("%Y-%m-%d"),
        development=[index[-80].strftime("%Y-%m-%d"), index[-50].strftime("%Y-%m-%d")],
        evaluation=[index[-40].strftime("%Y-%m-%d"), index[-1].strftime("%Y-%m-%d")],
        source_end=index[-1].strftime("%Y-%m-%d"),
        minimum_train=300,
    )
    return dict(daily=daily, cross=cross, iv=iv), events, config


def produce(sources, events, config, fit=True):
    # Producer calls exist only in generated agreement/corruption fixtures.
    from src.claims_release_market import build_market_features, make_five_session_targets
    from src.treasury_dealer_features import build_auction_features
    from src.treasury_dealer_pipeline import build_panel
    from src.treasury_market_context import build_market_context

    market = build_market_features(**sources)
    context = build_market_context(sources["daily"].index, sources["cross"])
    auction = build_auction_features(sources["daily"].index, events)
    features = pd.concat([market, context, auction["features"]], axis=1)
    targets = make_five_session_targets(market.rv_total)
    return (
        features,
        targets,
        auction["events"],
        build_panel(features, targets, config) if fit else None,
    )


def support_config(config):
    a, b = config["evaluation"]
    middle = (pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) // 2).strftime("%Y-%m-%d")
    after = (pd.Timestamp(middle) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return {
        "forecast": config,
        "stability": [[a, middle], [after, b]],
        "support": dict(
            phase_daily=1,
            slice_daily=1,
            offset_daily=1,
            training_activation_dates=1,
            phase_activation_dates=1,
            slice_activation_dates=1,
            phase_active_dates_per_offset=1,
            training_events_per_tenor=1,
            phase_events_per_tenor=1,
            slice_events_per_tenor=1,
        ),
    }


class AuctionOracleTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.bdate_range("2009-12-30", "2011-08-01")
        self.current = known("current", "2011-07-01", dealer=50)

    def test_literal_twelve_donors_strict_clock_and_no_carry(self):
        result = verify._auction_features(self.index, history() + [self.current])
        row = result["features"].loc["2011-07-04"]
        self.assertAlmostEqual(row.prior_share_mean_sum, 0.155)
        self.assertAlmostEqual(row.dealer_surprise, 0.345)
        self.assertTrue((result["features"].loc["2011-07-01"] == 0).all())
        self.assertTrue((result["features"].loc["2011-07-05"] == 0).all())
        self.assertEqual(result["events"][-1]["donor_ids"], [e["event_id"] for e in history()])
        self.assertTrue(result["features"].loc[:"2009-12-31"].isna().all().all())

    def test_delayed_pending_and_membership_precedes_availability(self):
        late = known("late", "2011-06-20", dealer=99) | {
            "availability_after_date": "2011-07-10"
        }
        out = verify._auction_features(self.index, history() + [late, self.current])
        audit = next(a for a in out["events"] if a["event_id"] == "current")
        self.assertEqual(audit["mask_reason"], "UNRELEASED_PRIOR_EVENT")
        self.assertIn("late", audit["donor_ids"])
        self.assertNotIn("5_0", audit["donor_ids"])
        self.assertTrue(out["features"].loc["2011-06-21":"2011-07-08"].isna().all().all())

    def test_unknown_clock_interval_and_donor_barrier_are_separate(self):
        for bounds, through in [
            ({}, "2011-08-01"),
            ({"clock_upper_bound_date": "2011-06-24"}, "2011-06-27"),
            ({"availability_after_date": "2011-06-24"}, "2011-06-27"),
        ]:
            with self.subTest(bounds=bounds):
                out = verify._auction_features(
                    self.index, history() + [unknown(**bounds), self.current]
                )
                self.assertTrue(out["features"].loc["2011-06-21":through].isna().all().all())
                audit = next(a for a in out["events"] if a["event_id"] == "current")
                self.assertEqual(audit["mask_reason"], "UNKNOWN_PRIOR_EVENT")
                if bounds:
                    self.assertTrue((out["features"].loc["2011-06-28"] == 0).all())

    def test_same_day_exclusion_tied_cutoff_and_simultaneous_log_sum(self):
        same = known("same", "2011-07-01", dealer=30) | {"offering_amount_usd": 2000}
        out = verify._auction_features(self.index, history() + [self.current, same])
        self.assertAlmostEqual(
            out["features"].loc["2011-07-04", "log_offering_sum"],
            math.log(1000) + math.log(2000),
        )
        self.assertEqual(out["features"].loc["2011-07-04", "auction_count"], 2)
        self.assertNotIn(
            "same", next(a for a in out["events"] if a["event_id"] == "current")["donor_ids"]
        )
        tied = verify._auction_features(
            self.index, history() + [known("tie", "2011-01-01"), self.current]
        )
        self.assertEqual(tied["events"][-1]["mask_reason"], "AMBIGUOUS_PRIOR_CUTOFF")

    def test_true_zero_surprise_is_an_event_and_unknown_peer_masks_all(self):
        donors = history(2)
        for item in donors:
            item["primary_dealer_accepted"] = 0
        event = known("zero", "2011-07-01", 2, 0)
        out = verify._auction_features(self.index, donors + [event])
        self.assertEqual(out["features"].loc["2011-07-04", "dealer_surprise"], 0)
        self.assertEqual(out["features"].loc["2011-07-04", "auction_count"], 1)
        out = verify._auction_features(
            self.index,
            donors + [event, unknown("2011-07-01", availability_after_date="2011-07-01")],
        )
        self.assertTrue(out["features"].loc["2011-07-04"].isna().all())
        self.assertTrue(
            next(a for a in out["events"] if a["event_id"] == "zero")["origin_masked"]
        )

    def test_preflight_before_numeric_and_permutation_preserves_inputs(self):
        events = history() + [self.current]
        before = copy.deepcopy(events)
        one = verify._auction_features(self.index, events)
        other = verify._auction_features(self.index, list(reversed(events)))
        pd.testing.assert_frame_equal(one["features"], other["features"])
        self.assertEqual(one["events"], other["events"])
        self.assertEqual(events, before)
        bad = events + [known("future", "2025-10-21")]
        with patch.object(
            verify, "_financial", side_effect=AssertionError("decoded")
        ) as convert:
            with self.assertRaises(ValueError):
                verify._auction_features(self.index, bad)
            convert.assert_not_called()


class CompleteVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources, cls.events, cls.config = fixture()
        cls.features, cls.targets, cls.audit, cls.produced = produce(
            cls.sources, cls.events, cls.config
        )

    def check(self, **changes):
        args = dict(
            sources=self.sources,
            ledger_events=self.events,
            features=self.features,
            targets=self.targets,
            produced=self.produced,
            config=self.config,
        )
        return verify.verify_forecasts(**(args | changes))

    def test_generated_complete_agreement_all_arms_and_unscored_tail(self):
        result = self.check()
        self.assertEqual(result["status"], "VERIFIED")
        self.assertEqual(result["calendar_rows"], len(self.features))
        self.assertEqual(
            result["application_forecasts_verified"], 3 * len(self.produced["applications"])
        )
        self.assertGreater(
            result["application_forecasts_verified"], result["forecasts_verified"]
        )

    def test_all_feature_cells_targets_cutoff_and_calendar_checked(self):
        for key, column in [
            ("features", "dealer_surprise"),
            ("features", "prior_share_mean_sum"),
            ("features", "tlt_ret"),
            ("features", "weekday_1"),
            ("features", "rv_total"),
            ("targets", "y"),
            ("targets", "target_end"),
            ("features", "treasury_cutoff_date"),
        ]:
            frame = getattr(self, key).copy(deep=True)
            pos = frame.index[-20]
            frame.loc[pos, column] = frame.loc[pos, column] + (
                pd.Timedelta(days=1) if "date" in column or column == "target_end" else 0.001
            )
            with self.subTest(key=key, column=column), self.assertRaises(ValueError):
                self.check(**{key: frame})
        with self.assertRaises(ValueError):
            self.check(features=self.features.iloc[1:])

    def test_every_scored_and_unscored_output_is_checked(self):
        for name, column in [
            ("applications", "pred_candidate"),
            ("panel", "prediction"),
            ("coverage", "status"),
            ("coverage", "offset"),
            ("schedules", "training_cutoff"),
        ]:
            produced = copy.deepcopy(self.produced)
            i = len(produced[name]) - 1
            old = produced[name].loc[i, column]
            produced[name].loc[i, column] = (
                "scored"
                if column == "status"
                else old + pd.Timedelta(days=1)
                if column == "training_cutoff"
                else old + 1
            )
            with self.subTest(name=name, column=column), self.assertRaises(ValueError):
                self.check(produced=produced)
        for name in ("applications", "panel", "coverage", "schedules"):
            produced = copy.deepcopy(self.produced)
            produced[name] = produced[name].iloc[:-1]
            with self.subTest(drop=name), self.assertRaises(ValueError):
                self.check(produced=produced)

    def test_train_membership_and_separate_smearing_audits_checked(self):
        for field in ("train_origins", "training_cutoff", "model_audits"):
            produced = copy.deepcopy(self.produced)
            fit = produced["fits"][0]
            if field == "train_origins":
                fit[field] = fit[field][1:]
            elif field == "training_cutoff":
                fit[field] = fit["fit_origin"]
            else:
                fit[field]["candidate"]["log_smearing"] += 0.01
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(produced=produced)

    def test_future_calendar_rejected_before_market_or_financial_arithmetic(self):
        sources = copy.deepcopy(self.sources)
        sources["iv"].loc[pd.Timestamp("2025-10-21")] = [1, 1, 1]
        with patch.object(
            verify, "_market_targets", side_effect=AssertionError("market read")
        ) as market:
            with self.assertRaises(ValueError):
                self.check(sources=sources)
            market.assert_not_called()

    def test_target_missing_first_query_does_not_move_schedule(self):
        sources = copy.deepcopy(self.sources)
        first = pd.Timestamp(self.produced["fits"][0]["fit_origin"])
        nextday = sources["daily"].index[sources["daily"].index.get_loc(first) + 1]
        sources["daily"].loc[nextday, "high"] = np.nan
        features, targets, _, produced = produce(sources, self.events, self.config)
        self.assertTrue(pd.isna(targets.loc[first, "y"]))
        self.assertEqual(produced["fits"][0]["fit_origin"], first.strftime("%Y-%m-%d"))
        self.assertEqual(
            self.check(sources=sources, features=features, targets=targets, produced=produced)[
                "status"
            ],
            "VERIFIED",
        )

    def test_market_context_strict_full_calendar_delay_and_future_invariance(self):
        index = self.features.index
        cross = self.sources["cross"].drop(index[600])
        context = verify._market_context(index, cross)
        self.assertTrue(context.tlt_ret.iloc[601:603].isna().all())
        self.assertTrue(context.tlt_lrv22.iloc[601:624].isna().all())
        self.assertTrue(np.isfinite(context.tlt_lrv22.iloc[624]))
        i = 500
        returns = np.diff(np.log(cross.reindex(index).tlt.iloc[i - 23 : i].to_numpy()))
        self.assertAlmostEqual(
            context.tlt_lrv22.iloc[i], math.log(math.fsum(returns**2) / 22), places=12
        )
        changed = cross.copy()
        changed.loc[index[700] :, "tlt"] *= 2
        pd.testing.assert_frame_equal(
            context.iloc[:701], verify._market_context(index, changed).iloc[:701]
        )

    def test_independent_qr_matches_direct_unscaled_ols_and_exact_duan(self):
        rng = np.random.default_rng(451)
        train = pd.DataFrame(rng.normal(size=(160, 32)), columns=verify.ALL)
        train["const"] = 1.0
        query = train.iloc[:7].copy()
        y = pd.Series(np.exp(rng.normal(-9, 0.2, len(train))), index=train.index)
        forecasts, audits = verify._fit_expected(train, y, query)
        for arm, cols in verify.ARMS.items():
            x = train.loc[:, cols].to_numpy()
            a = query.loc[:, cols].to_numpy()
            beta = np.linalg.lstsq(x, np.log(y), rcond=None)[0]
            expected = np.exp(a @ beta) * np.exp(np.log(y) - x @ beta).mean()
            np.testing.assert_allclose(forecasts[arm], expected, rtol=1e-11, atol=1e-14)
            self.assertEqual(audits[arm]["rank"], len(cols))
        train["dealer_surprise"] = 1.0
        with self.assertRaises(ValueError):
            verify._fit_expected(train, y, query)

    def test_supported_and_failed_support_reconstructed_without_fits(self):
        from src.treasury_dealer_support import audit_support

        cfg = support_config(self.config)
        # Guarantee all six tenors in each invented slice, so this explicitly
        # exercises a passing support report as well as a whole-attempt failure.
        events = copy.deepcopy(self.events)
        for position in (-35, -15):
            day = self.features.index[position].strftime("%Y-%m-%d")
            for tenor in (2, 3, 5, 7, 10, 30):
                events.append(known(f"support_{position}_{tenor}", day, tenor))
        features, targets, audit, _ = produce(self.sources, events, self.config, fit=False)
        tenors = {e["event_id"]: e["original_tenor_years"] for e in events}
        for failing in (False, True):
            spec = copy.deepcopy(cfg)
            if failing:
                spec["support"]["training_activation_dates"] = 100000
            report = audit_support(
                features.index,
                features,
                targets,
                audit,
                event_tenors=tenors,
                config=spec,
            )
            with patch.object(
                verify, "_fit_expected", side_effect=AssertionError("fit before support")
            ):
                result = verify.verify_inputs_and_support(
                    self.sources,
                    events,
                    features,
                    targets,
                    audit,
                    report,
                    spec,
                )
            self.assertEqual(result["status"], "VERIFIED")
            self.assertEqual(result["support_passed"], report["passed"])
            self.assertEqual(report["passed"], not failing)
            report["monthly"][0]["train_n"] += 1
            with self.assertRaises(ValueError):
                verify.verify_inputs_and_support(
                    self.sources,
                    events,
                    features,
                    targets,
                    audit,
                    report,
                    spec,
                )

    def test_literal_support_counts_zero_pulses_and_original_offsets(self):
        config, intervals, floors = verify._support_config(support_config(self.config))
        features = self.features.copy()
        features["dealer_surprise"] = features["dealer_surprise"].where(
            features["dealer_surprise"].isna(), 0.0
        )
        report = verify._support_expected(
            features, self.targets, self.audit, self.events, config, intervals, floors
        )
        complete = np.isfinite(features.loc[:, verify.ALL].to_numpy()).all(axis=1)
        index = features.index
        for name in ("development", "evaluation"):
            start, end = config[name]
            fence = end if name == "development" else config["source_end"]
            positions = [
                i
                for i, day in enumerate(index)
                if start <= day <= end
                and complete[i]
                and np.isfinite(self.targets.y.iloc[i])
                and pd.notna(self.targets.target_end.iloc[i])
                and self.targets.target_end.iloc[i] <= fence
            ]
            active = [i for i in positions if features.auction_count.iloc[i] > 0]
            identities = [
                a
                for a in self.audit
                if a["status"] == "SUPPORTED"
                and not a["origin_masked"]
                and a["origin"] in index[active]
            ]
            self.assertEqual(report["phases"][name]["daily_n"], len(positions))
            self.assertEqual(report["phases"][name]["activation_dates"], len(active))
            self.assertEqual(report["phases"][name]["event_n"], len(identities))
            self.assertGreater(len(active), 0)
            for offset in range(5):
                row = report["offsets"][name][offset]
                self.assertEqual(row["daily_n"], sum(i % 5 == offset for i in positions))
                self.assertEqual(row["activation_dates"], sum(i % 5 == offset for i in active))

    def test_input_and_audit_preservation_and_datetime_transport(self):
        sources = {key: frame.copy(deep=True) for key, frame in self.sources.items()}
        events = copy.deepcopy(self.events)
        original_features = self.features.copy(deep=True)
        self.check(sources=sources, ledger_events=events)
        self.assertEqual(events, self.events)
        for key in sources:
            pd.testing.assert_frame_equal(sources[key], self.sources[key])
        pd.testing.assert_frame_equal(original_features, self.features)
        for unit in ("ms", "us"):
            transported = {
                key: frame.set_axis(frame.index.as_unit(unit))
                for key, frame in sources.items()
            }
            with self.subTest(unit=unit):
                self.assertEqual(self.check(sources=transported)["status"], "VERIFIED")

    def test_event_audit_donor_and_mask_tampering_rejected(self):
        from src.treasury_dealer_support import audit_support

        cfg = support_config(self.config)
        report = audit_support(
            self.features.index,
            self.features,
            self.targets,
            self.audit,
            event_tenors={e["event_id"]: e["original_tenor_years"] for e in self.events},
            config=cfg,
        )
        for field in ("donor_ids", "origin_masked", "origin"):
            audit = copy.deepcopy(self.audit)
            item = next(a for a in audit if a["status"] == "SUPPORTED")
            item[field] = (
                item[field][1:]
                if field == "donor_ids"
                else True
                if field == "origin_masked"
                else item[field] + pd.Timedelta(days=1)
            )
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify.verify_inputs_and_support(
                    self.sources, self.events, self.features, self.targets, audit, report, cfg
                )


if __name__ == "__main__":
    unittest.main()
