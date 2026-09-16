"""Wave 8 synthetic source/feature contracts, prewritten before implementation."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import calendar_variance_features as cf


def plan(event, announced="2017-03-10 08:30", planned="2017-04-14 08:30"):
    return {"event_type": event,
            "announced_at": pd.Timestamp(announced, tz="America/New_York").isoformat(),
            "planned_at": pd.Timestamp(planned, tz="America/New_York").isoformat(),
            "source_id": event+announced, "source_sha256": "a"*64}


def bls():
    return [plan(event) for event in ("cpi", "nfp")]


def fed(year=2017):
    dates = [f"{year}-{suffix}" for suffix in
             ("01-10", "02-20", "03-15", "04-14", "06-14", "07-26", "09-20", "12-13")]
    return [{"year": year, "announced_date": f"{year-1}-05-15", "final_dates": dates,
             "source_id": f"annual-{year}", "source_sha256": "b"*64}]


def one_origin(origin="2017-04-13", cutoff="2017-04-12"):
    dates = pd.DatetimeIndex([origin], name="date")
    return dates, pd.Series(pd.to_datetime([cutoff]), index=dates)


def sample():
    dates = pd.bdate_range("2014-01-02", periods=150, name="date").delete([12, 37])
    z = np.arange(len(dates), dtype=float)
    close = 100*np.exp(np.cumsum(.001+.008*np.sin(z*.37)))
    opening = close*np.exp(.004*np.cos(z*.23))
    daily = pd.DataFrame({"open": opening, "high": np.maximum(opening, close)*1.006,
                          "low": np.minimum(opening, close)*.994, "close": close,
                          "adj close": close*.7}, index=dates)
    iv = pd.DataFrame({"vix": 18+3*np.sin(z*.17), "vix9d": 16+2*np.cos(z*.21),
                       "vvix": 85+4*np.cos(z*.19)}, index=dates)
    plans = pd.DataFrame({"nominal_hours": np.where(dates.weekday == 4, 72., 24.),
                          "cpi_plan": (z % 21 == 0).astype(float),
                          "nfp_plan": (z % 23 == 0).astype(float),
                          "fomc_plan": (z % 30 == 0).astype(float)}, index=dates)
    return daily, iv, plans


def source_fixture(root):
    dates = pd.DatetimeIndex(["2017-04-12", "2017-04-13", "2025-10-20", "2025-11-03"], name="date")
    daily = pd.DataFrame({"open": [100., 101., 102., np.inf], "high": [102., 103., 104., np.inf],
                          "low": [99., 100., 101., np.inf], "close": [101., 102., 103., np.inf]}, index=dates)
    daily.to_parquet(root/"daily.parquet")
    paths = {"daily": "daily.parquet"}
    for name, field in [("vix", "CLOSE"), ("vix9d", "CLOSE"), ("vvix", "VVIX")]:
        paths[name] = name+".csv"
        (root/paths[name]).write_text(f"DATE,{field}\n04/12/2017,20\n04/13/2017,21\n10/20/2025,22\n11/03/2025,FORBIDDEN\n")
    return {"sources": paths, "index": {"source_end": "2025-10-20", "sealed_start": "2025-11-03"}}


class TestCalendarWindow(unittest.TestCase):
    def test_nominal_close_to_close_window_and_both_dst_directions(self):
        start, end = cf.nominal_window(pd.Timestamp("2020-03-06"))
        self.assertEqual(start.isoformat(), "2020-03-06T16:00:00-05:00")
        self.assertEqual(end.isoformat(), "2020-03-09T16:00:00-04:00")
        self.assertEqual((end-start).total_seconds()/3600, 71.)
        start, end = cf.nominal_window(pd.Timestamp("2020-10-30"))
        self.assertEqual((end-start).total_seconds()/3600, 73.)

    def test_holidays_and_early_closes_do_not_change_nominal_endpoints(self):
        start, end = cf.nominal_window(pd.Timestamp("2017-04-13"))
        self.assertEqual(str(end.date()), "2017-04-14")  # Good Friday remains nominal.
        start, end = cf.nominal_window(pd.Timestamp("2020-05-22"))
        self.assertEqual(str(end.date()), "2020-05-25")  # Monday holiday remains nominal.
        start, end = cf.nominal_window(pd.Timestamp("2020-11-27"))
        self.assertEqual(start.hour, 16)  # Declared policy keeps 16:00 on an early close.
        self.assertEqual(end.hour, 16)

    def test_window_requires_normalized_weekday_naive_dates(self):
        for bad in ["2020-03-07", "2020-03-06 12:00", "2020-03-06T00:00:00Z", pd.NaT]:
            with self.assertRaises(ValueError):
                cf.nominal_window(pd.Timestamp(bad))

    def test_open_left_closed_right_and_new_afternoon_endpoint(self):
        origins, cutoffs = one_origin()
        records = [plan("cpi", planned="2017-04-13 16:00"),
                   plan("nfp", planned="2017-04-14 16:00")]
        result, audit = cf.build_plan_features(origins, cutoffs, records, fed())
        self.assertEqual(result.iloc[0].cpi_plan, 0.)
        self.assertEqual(result.iloc[0].nfp_plan, 1.)
        self.assertEqual(result.iloc[0].fomc_plan, 1.)
        self.assertEqual(result.iloc[0].nominal_hours, 24.)
        self.assertEqual(audit[0]["nominal_end"], "2017-04-14T16:00:00-04:00")

    def test_exact_evidence_schema_and_original_source_identities(self):
        result, audit = cf.build_plan_features(*one_origin(), bls(), fed())
        self.assertEqual(tuple(result.columns), ("nominal_hours", "cpi_plan", "nfp_plan", "fomc_plan"))
        self.assertEqual(set(audit[0]), {"origin", "nominal_start", "nominal_end", "source_rule",
                         "cpi_missing_months", "cpi_source_ids", "cpi_source_hashes",
                         "nfp_missing_months", "nfp_source_ids", "nfp_source_hashes",
                         "fomc_source_id", "fomc_source_sha256"})
        self.assertEqual(audit[0]["cpi_source_ids"], [bls()[0]["source_id"]])
        self.assertEqual(audit[0]["cpi_source_hashes"], ["a"*64])

    def test_publication_date_must_strictly_precede_cutoff(self):
        records = [plan(event, "2017-04-12 00:01") for event in ("cpi", "nfp")]
        result, audit = cf.build_plan_features(*one_origin(), records, fed())
        self.assertTrue(result[["cpi_plan", "nfp_plan"]].isna().all().all())
        self.assertEqual(audit[0]["cpi_missing_months"], ["2017-04"])
        origins, _ = one_origin(cutoff="2017-04-11")
        with self.assertRaises(ValueError):
            cf.build_plan_features(origins, pd.Series(origins, index=origins), bls(), fed())

    def test_unknown_cutoff_and_missing_annual_schedule_remain_unknown(self):
        origins, cutoff = one_origin()
        result, _ = cf.build_plan_features(origins, pd.Series(pd.NaT, index=origins), bls(), fed())
        self.assertTrue(result.loc[:, ["cpi_plan", "nfp_plan", "fomc_plan"]].isna().all().all())
        result, _ = cf.build_plan_features(origins, cutoff, bls(), [])
        self.assertTrue(np.isnan(result.iloc[0].fomc_plan))

    def test_month_boundary_requires_both_bls_months_and_both_event_types(self):
        origins, cutoff = one_origin("2017-03-31", "2017-03-30")
        result, audit = cf.build_plan_features(origins, cutoff, bls(), fed())
        self.assertTrue(result[["cpi_plan", "nfp_plan"]].isna().all().all())
        self.assertEqual(audit[0]["cpi_missing_months"], ["2017-03"])
        march_cpi = plan("cpi", "2017-02-10 08:30", "2017-03-14 08:30")
        result, audit = cf.build_plan_features(origins, cutoff, bls()+[march_cpi], fed())
        self.assertEqual(result.iloc[0].cpi_plan, 0.)
        self.assertTrue(np.isnan(result.iloc[0].nfp_plan))

    def test_canceled_plans_and_future_sources_do_not_rewrite_original_flags(self):
        before, audit_before = cf.build_plan_features(*one_origin(), bls(), fed())
        changed = [dict(record, canceled=True, actual_release="2099-01-01") for record in bls()]
        changed += [plan(event, "2017-04-14 08:30", "2017-05-12 08:30") for event in ("cpi", "nfp")]
        after, audit_after = cf.build_plan_features(*one_origin(), changed, fed())
        pd.testing.assert_frame_equal(before, after)
        self.assertEqual(audit_before, audit_after)

    def test_ambiguous_original_month_or_incomplete_fomc_schedule_is_rejected(self):
        with self.assertRaises(ValueError):
            cf.build_plan_features(*one_origin(), bls()+[plan("cpi", "2017-03-11 08:30")], fed())
        annual = fed()
        annual[0]["final_dates"] = ["2017-04-14"]
        with self.assertRaises(ValueError):
            cf.build_plan_features(*one_origin(), bls(), annual)


class TestCalendarVarianceFeatures(unittest.TestCase):
    def test_fixed_feature_and_target_schema(self):
        base = ("const", "lrv_d", "lrv_w", "lrv_m", "neg_d", "neg_w", "neg_m", "lvix", "term", "lvvix",
                "entry_dow_1", "entry_dow_2", "entry_dow_3", "entry_dow_4", "nominal_hours")
        self.assertEqual(cf.BASE, base)
        self.assertEqual(cf.ADDITIONS, ("cpi_plan", "nfp_plan", "fomc_plan"))
        self.assertEqual(cf.ALL_FEATURES, base+cf.ADDITIONS)
        self.assertEqual(cf.MODELS, ("mean", "baseline", "calendar"))
        feature, target = cf.build_features(*sample())
        self.assertEqual(tuple(feature.columns), cf.ALL_FEATURES+("feature_cutoff_date",))
        self.assertEqual(tuple(target.columns), ("y", "target_end", "available_date"))

    def test_explicit_raw_risk_negative_histories_and_iv_lags(self):
        daily, iv, plans = sample()
        f, target = cf.build_features(daily, iv, plans)
        ret = np.log(daily.close/daily.close.shift())
        gk = np.maximum(.5*np.log(daily.high/daily.low)**2
                        -(2*np.log(2)-1)*np.log(daily.close/daily.open)**2, 1e-10)
        risk = gk+np.log(daily.open/daily.close.shift())**2
        negative = np.maximum(-ret, 0)
        t = 90
        for suffix, width in [("d", 1), ("w", 5), ("m", 22)]:
            self.assertAlmostEqual(f.iloc[t]["lrv_"+suffix], np.log(risk.iloc[t-width:t].mean()))
            self.assertAlmostEqual(f.iloc[t]["neg_"+suffix], negative.iloc[t-width:t].mean())
        self.assertAlmostEqual(f.iloc[t].lvix, np.log(iv.vix.iloc[t-1]))
        self.assertAlmostEqual(f.iloc[t].term, np.log(iv.vix9d.iloc[t-1]/iv.vix.iloc[t-1]))
        self.assertAlmostEqual(f.iloc[t].lvvix, np.log(iv.vvix.iloc[t-1]))
        self.assertAlmostEqual(target.iloc[t].y, risk.iloc[t+1])
        self.assertEqual(f.iloc[t].feature_cutoff_date, daily.index[t-1])

    def test_entry_calendar_is_current_and_last_target_is_unknown(self):
        daily, iv, plans = sample()
        f, target = cf.build_features(daily, iv, plans)
        for column in ("nominal_hours", *cf.ADDITIONS):
            pd.testing.assert_series_equal(f[column], plans[column])
        for weekday in range(1, 5):
            np.testing.assert_array_equal(f[f"entry_dow_{weekday}"], (daily.index.weekday == weekday).astype(float))
        self.assertEqual(target.iloc[40].target_end, daily.index[41])
        self.assertEqual(target.iloc[40].available_date, daily.index[41])
        self.assertTrue(target.iloc[-1].isna().all())

    def test_entry_and_future_price_iv_mutations_cannot_change_current_predictors(self):
        daily, iv, plans = sample()
        before, _ = cf.build_features(daily, iv, plans)
        changed, changed_iv = daily.copy(), iv.copy()
        changed.iloc[90:] *= 3
        changed_iv.iloc[90:] *= 2
        after, _ = cf.build_features(changed, changed_iv, plans)
        pd.testing.assert_frame_equal(before.iloc[:91], after.iloc[:91])

    def test_future_observed_calendar_only_changes_target_and_maturity(self):
        daily, iv, plans = sample()
        before, target_before = cf.build_features(daily, iv, plans)
        after, target_after = cf.build_features(daily.drop(daily.index[91]), iv, plans)
        pd.testing.assert_frame_equal(before.iloc[:91], after.iloc[:91])
        self.assertNotEqual(target_before.iloc[90].target_end, target_after.iloc[90].target_end)
        origins = daily.index[:91]
        cutoffs = pd.Series(origins, index=origins).shift()
        a, evidence_a = cf.build_plan_features(origins, cutoffs, [], [])
        b, evidence_b = cf.build_plan_features(daily.drop(daily.index[91]).index[:91], cutoffs, [], [])
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(evidence_a, evidence_b)

    def test_strict_windows_preserve_unknown_history_without_calendar_compression(self):
        daily, iv, plans = sample()
        daily.loc[daily.index[80], ["open", "high", "low", "close"]] = np.nan
        f, target = cf.build_features(daily, iv, plans)
        self.assertEqual(len(f), len(daily))
        self.assertTrue(f.lrv_m.iloc[:23].isna().all())
        self.assertTrue(f.lrv_w.iloc[81:87].isna().all())
        self.assertTrue(f.neg_w.iloc[81:87].isna().all())
        self.assertTrue(f.lrv_m.iloc[81:104].isna().all())
        self.assertTrue(f.neg_m.iloc[81:104].isna().all())
        self.assertTrue(np.isfinite(f.iloc[104].lrv_m))
        self.assertTrue(np.isfinite(f.iloc[104].neg_m))
        self.assertTrue(np.isnan(target.iloc[79].y))
        self.assertTrue(np.isnan(target.iloc[80].y))

    def test_missing_iv_or_plan_stays_unknown_without_forward_fill(self):
        daily, iv, plans = sample()
        iv = iv.drop(daily.index[80])
        plans = plans.drop(daily.index[82])
        f, _ = cf.build_features(daily, iv, plans)
        self.assertTrue(f.loc[daily.index[81], ["lvix", "term", "lvvix"]].isna().all())
        self.assertTrue(np.isfinite(f.loc[daily.index[82], "lvix"]))
        self.assertTrue(f.loc[daily.index[82], ["nominal_hours", *cf.ADDITIONS]].isna().all())
        self.assertEqual(len(f), len(daily))

    def test_flat_raw_prices_have_positive_declared_gk_floor(self):
        daily, iv, plans = sample()
        daily.loc[:, ["open", "high", "low", "close"]] = 100.
        f, target = cf.build_features(daily, iv, plans)
        self.assertEqual(target.iloc[90].y, 1e-10)
        self.assertAlmostEqual(f.iloc[90].lrv_m, np.log(1e-10))
        self.assertEqual(f.iloc[90].neg_m, 0.)

    def test_raw_price_units_and_adjusted_close_do_not_change_features_or_labels(self):
        daily, iv, plans = sample()
        before, target_before = cf.build_features(daily, iv, plans)
        daily.loc[:, ["open", "high", "low", "close"]] *= 7
        daily["adj close"] = np.nan
        after, target_after = cf.build_features(daily, iv, plans)
        np.testing.assert_allclose(before.loc[:, cf.ALL_FEATURES], after.loc[:, cf.ALL_FEATURES], atol=2e-13, equal_nan=True)
        np.testing.assert_allclose(target_before.y, target_after.y, atol=1e-15, equal_nan=True)

    def test_invalid_observed_prices_ohlc_or_calendar_values_are_rejected(self):
        daily, iv, plans = sample()
        for value in [0., -1., np.inf]:
            bad = daily.copy()
            bad.loc[bad.index[40], "close"] = value
            with self.assertRaises(ValueError):
                cf.build_features(bad, iv, plans)
        bad = daily.copy()
        bad.loc[bad.index[40], "high"] = .1
        with self.assertRaises(ValueError):
            cf.build_features(bad, iv, plans)
        for column, value in [("cpi_plan", .5), ("nominal_hours", 0.), ("nominal_hours", np.inf)]:
            bad = plans.copy()
            bad.loc[bad.index[40], column] = value
            with self.assertRaises(ValueError):
                cf.build_features(daily, iv, bad)
        with self.assertRaises(ValueError):
            cf.build_features(daily.iloc[::-1], iv, plans)


class TestCalendarSources(unittest.TestCase):
    def test_market_parser_fences_values_and_source_audit_preserves_plan_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol = source_fixture(root)
            with patch.object(cf.macro_search, "load_plan_records", return_value=(bls(), fed(), {"synthetic": True})) as loader:
                daily, iv, plans, audit = cf.load_sources(protocol, root)
            loader.assert_called_once_with(protocol)
            self.assertEqual(str(daily.index.max().date()), "2025-10-20")
            self.assertTrue(iv.index.max() <= pd.Timestamp("2025-10-20"))
            self.assertEqual(list(daily.columns), ["open", "high", "low", "close"])
            self.assertEqual(set(audit), {"market", "plans", "plan_availability",
                                         "plan_counts_full_bounded_calendar", "unknown_rows"})
            self.assertEqual(audit["plans"], {"synthetic": True})
            self.assertFalse(audit["market"]["numeric_post_cutoff_values_parsed"])
            self.assertEqual(audit["market"]["sources"]["vix"]["source_sha256"],
                             hashlib.sha256((root/"vix.csv").read_bytes()).hexdigest())
            self.assertEqual(audit["plan_availability"][1]["nominal_end"], "2017-04-14T16:00:00-04:00")
            self.assertEqual(audit["plan_counts_full_bounded_calendar"], {"cpi_plan": 1, "nfp_plan": 1, "fomc_plan": 1})
            self.assertEqual(audit["unknown_rows"], 2)
            self.assertTrue(plans.index.equals(daily.index))

    def test_source_fence_rejected_before_calendar_loader(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol = source_fixture(root)
            protocol["index"]["source_end"] = "2025-11-03"
            with patch.object(cf.macro_search, "load_plan_records") as loader, self.assertRaises(ValueError):
                cf.load_sources(protocol, root)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
