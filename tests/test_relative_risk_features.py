"""Prewritten wave 9 source, measurement-gate, and relative-risk contracts."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import relative_risk_features as rr


def sample(n=150):
    dates = pd.bdate_range("2014-01-02", periods=n, name="date").delete([12, 37])
    z = np.arange(len(dates), dtype=float)

    def daily(phase):
        close = 100*np.exp(np.cumsum(.001+.008*np.sin(z*.37+phase)))
        opening = close*np.exp(.004*np.cos(z*.23+phase))
        return pd.DataFrame({"open": opening, "high": np.maximum(opening, close)*1.006,
                             "low": np.minimum(opening, close)*.994, "close": close}, index=dates)

    spx, qqq = daily(0), daily(.4)
    qqq["adj close"] = qqq.close*.7
    prior = qqq.iloc[:2].copy()
    prior.index = pd.DatetimeIndex(["2013-12-30", "2013-12-31"], name="date")
    qqq = pd.concat([prior, qqq])
    iv = pd.DataFrame({"vxn": 20+2*np.cos(z*.21), "vix": 18+3*np.sin(z*.17),
                       "vix9d": 16+2*np.cos(z*.21), "vvix": 85+4*np.cos(z*.19)}, index=dates)
    return qqq, spx, iv


def raw_gk(frame):
    return .5*np.log(frame.high/frame.low)**2-(2*np.log(2)-1)*np.log(frame.close/frame.open)**2


def source_fixture(root):
    dates = pd.DatetimeIndex(["2017-04-12", "2017-04-13", "2025-10-20", "2025-11-03"], name="date")
    spx = pd.DataFrame({"open": [100., 101., 102., np.inf], "high": [102., 103., 104., np.inf],
                        "low": [99., 100., 101., np.inf], "close": [101., 102., 103., np.inf]}, index=dates)
    spx.to_parquet(root/"spx.parquet")
    qqq = spx.copy()
    qqq["adj close"] = np.inf
    earlier = qqq.iloc[[0]].copy()
    earlier.index = pd.DatetimeIndex(["2017-04-11"], name="date")
    pd.concat([earlier, qqq]).to_parquet(root/"qqq.parquet")
    paths = {"daily": "spx.parquet", "qqq": "qqq.parquet"}
    for name, field in [("vxn", "CLOSE"), ("vix", "CLOSE"), ("vix9d", "CLOSE"), ("vvix", "VVIX")]:
        paths[name] = name+".csv"
        (root/paths[name]).write_text(f"DATE,{field}\n04/12/2017,20\n04/13/2017,21\n10/20/2025,22\n11/03/2025,FORBIDDEN\n")
    return {"sources": paths, "index": {"source_end": "2025-10-20", "sealed_start": "2025-11-03"}}


class TestRelativeMeasurement(unittest.TestCase):
    def test_full_reference_measurement_audit_schema_and_pass(self):
        qqq, spx, _ = sample()
        audit = rr.measurement_audit(qqq, spx)
        self.assertEqual(set(audit), {"status", "gk_floor", "reference_rows", "per_asset"})
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["gk_floor"], 1e-10)
        self.assertEqual(audit["reference_rows"], len(spx))
        for name, data in [("qqq", qqq.reindex(spx.index)), ("spx", spx)]:
            entry = audit["per_asset"][name]
            self.assertEqual(set(entry), {"observed_complete_rows", "missing_ohlc_rows", "finite_gk_rows",
                             "floor_hit_rows", "nonpositive_gk_rows", "small_positive_gk_rows", "minimum_raw_gk"})
            self.assertEqual(entry["observed_complete_rows"], len(spx))
            self.assertEqual(entry["missing_ohlc_rows"], 0)
            self.assertEqual(entry["finite_gk_rows"], len(spx))
            self.assertEqual(entry["floor_hit_rows"], 0)
            self.assertAlmostEqual(entry["minimum_raw_gk"], raw_gk(data).min())
        rr.require_measurement(audit)

    def test_floor_hit_before_any_feature_complete_row_aborts_without_filtering(self):
        qqq, spx, iv = sample()
        spx.loc[spx.index[0], ["open", "high", "low", "close"]] = 100.
        iv.iloc[:40] = np.nan
        audit = rr.measurement_audit(qqq, spx)
        self.assertEqual(audit["status"], "INSUFFICIENT_MEASUREMENT")
        self.assertEqual(audit["per_asset"]["spx"]["floor_hit_rows"], 1)
        self.assertEqual(audit["per_asset"]["spx"]["nonpositive_gk_rows"], 1)
        self.assertEqual(audit["per_asset"]["spx"]["minimum_raw_gk"], 0.)
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"):
            rr.require_measurement(audit)
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"):
            rr.build_features(qqq, spx, iv)

    def test_small_positive_measurement_below_floor_is_distinct_from_nonpositive(self):
        qqq, spx, _ = sample()
        spx.loc[spx.index[30], ["open", "high", "low", "close"]] = 100.
        spx.loc[spx.index[30], "high"] *= np.exp(np.sqrt(1e-10/2))
        audit = rr.measurement_audit(qqq, spx)
        entry = audit["per_asset"]["spx"]
        self.assertEqual(entry["small_positive_gk_rows"], 1)
        self.assertEqual(entry["nonpositive_gk_rows"], 0)
        self.assertEqual(entry["floor_hit_rows"], 1)
        self.assertGreater(entry["minimum_raw_gk"], 0.)
        self.assertEqual(audit["status"], "INSUFFICIENT_MEASUREMENT")

    def test_missing_observations_are_reported_and_no_reference_row_is_removed(self):
        qqq, spx, _ = sample()
        qqq = qqq.drop(spx.index[50])
        spx.loc[spx.index[60], "high"] = np.nan
        audit = rr.measurement_audit(qqq, spx)
        self.assertEqual(audit["reference_rows"], len(spx))
        self.assertEqual(audit["status"], "PASS")
        for asset in ("qqq", "spx"):
            self.assertEqual(audit["per_asset"][asset]["missing_ohlc_rows"], 1)
            self.assertEqual(audit["per_asset"][asset]["finite_gk_rows"], len(spx)-1)

    def test_unpaired_earlier_qqq_row_is_not_a_reference_measurement(self):
        qqq, spx, _ = sample()
        qqq.loc[qqq.index[0], ["open", "high", "low", "close"]] = 100.
        self.assertEqual(rr.measurement_audit(qqq, spx)["status"], "PASS")

    def test_empty_observed_measurements_cannot_pass_by_vacuous_comparison(self):
        qqq, spx, _ = sample()
        qqq.loc[spx.index, ["open", "high", "low", "close"]] = np.nan
        audit = rr.measurement_audit(qqq, spx)
        self.assertIsNone(audit["per_asset"]["qqq"]["minimum_raw_gk"])
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"):
            rr.require_measurement(audit)

    def test_invalid_prices_ranges_and_nonfinite_computed_gk_are_rejected(self):
        qqq, spx, _ = sample()
        for value in [0., -1., np.inf]:
            bad = spx.copy()
            bad.loc[bad.index[50], "close"] = value
            with self.assertRaises(ValueError):
                rr.measurement_audit(qqq, bad)
        bad = spx.copy()
        bad.loc[bad.index[50], "high"] = .1
        with self.assertRaises(ValueError):
            rr.measurement_audit(qqq, bad)
        bad.loc[bad.index[50], ["open", "high", "low", "close"]] = [1., 1e308, 1e-308, 1.]
        with self.assertRaisesRegex(ValueError, "[Nn]onfinite"):
            rr.measurement_audit(qqq, bad)

    def test_require_measurement_rejects_inconsistent_pass_status(self):
        audit = rr.measurement_audit(*sample()[:2])
        audit["per_asset"]["qqq"]["floor_hit_rows"] = 1
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"):
            rr.require_measurement(audit)


class TestRelativeFeatures(unittest.TestCase):
    def test_exact_27_control_28_common_column_schema(self):
        baseline = ("const", *[f"{asset}_{kind}_{suffix}" for asset in ("qqq", "spx")
                                for kind in ("lg", "lt", "neg") for suffix in ("d", "w", "m")],
                    "lvxn", "lvix", "term", "lvvix", *[f"entry_dow_{i}" for i in range(1, 5)])
        self.assertEqual(rr.BASE, baseline)
        self.assertEqual(len(rr.BASE), 27)
        self.assertEqual(rr.ALL_FEATURES, baseline+("corr22",))
        self.assertEqual(rr.MODELS, ("mean", "baseline", "correlation"))
        f, target = rr.build_features(*sample())
        self.assertEqual(tuple(f.columns), rr.ALL_FEATURES+("feature_cutoff_date",))
        self.assertEqual(tuple(target.columns), ("y", "target_end", "available_date"))

    def test_exact_marginal_history_formulas_and_one_session_iv_lag(self):
        qqq, spx, iv = sample()
        f, _ = rr.build_features(qqq, spx, iv)
        t = 90
        for asset, data in [("qqq", qqq.reindex(spx.index)), ("spx", spx)]:
            gk = raw_gk(data)
            total = gk+np.log(data.open/data.close.shift())**2
            negative = np.maximum(-np.log(data.close/data.close.shift()), 0)
            for suffix, window in [("d", 1), ("w", 5), ("m", 22)]:
                self.assertAlmostEqual(f.iloc[t][f"{asset}_lg_{suffix}"], np.log(gk.iloc[t-window:t].mean()))
                self.assertAlmostEqual(f.iloc[t][f"{asset}_lt_{suffix}"], np.log(total.iloc[t-window:t].mean()))
                self.assertAlmostEqual(f.iloc[t][f"{asset}_neg_{suffix}"], negative.iloc[t-window:t].mean())
        for output, source in [("lvxn", "vxn"), ("lvix", "vix"), ("lvvix", "vvix")]:
            self.assertAlmostEqual(f.iloc[t][output], np.log(iv.iloc[t-1][source]))
        self.assertAlmostEqual(f.iloc[t].term, np.log(iv.vix9d.iloc[t-1]/iv.vix.iloc[t-1]))
        self.assertEqual(f.iloc[t].feature_cutoff_date, spx.index[t-1])

    def test_first_source_predecessor_boundary_is_unknown_for_both_total_histories(self):
        f, _ = rr.build_features(*sample())
        for asset in ("qqq", "spx"):
            self.assertTrue(np.isnan(f.iloc[1][asset+"_lt_d"]))
            self.assertTrue(np.isnan(f.iloc[1][asset+"_neg_d"]))
            self.assertTrue(np.isfinite(f.iloc[1][asset+"_lg_d"]))
            self.assertTrue(np.isfinite(f.iloc[22][asset+"_lg_m"]))
            self.assertTrue(np.isnan(f.iloc[22][asset+"_lt_m"]))
            self.assertTrue(np.isfinite(f.iloc[23][asset+"_lt_m"]))

    def test_source_predecessor_mismatch_does_not_manufacture_longer_raw_return(self):
        qqq, spx, iv = sample()
        extra = qqq.loc[[spx.index[11]]].copy()
        extra.index = pd.DatetimeIndex(["2014-01-20"], name="date")
        self.assertNotIn(extra.index[0], spx.index)
        changed = pd.concat([qqq, extra]).sort_index()
        f, _ = rr.build_features(changed, spx, iv)
        next_position = spx.index.searchsorted(extra.index[0])
        self.assertTrue(np.isnan(f.iloc[next_position+1].qqq_lt_d))
        self.assertTrue(np.isnan(f.iloc[next_position+1].qqq_neg_d))
        self.assertTrue(np.isfinite(f.iloc[next_position+1].qqq_lg_d))
        self.assertTrue(np.isfinite(f.iloc[next_position+1].spx_lt_d))
        self.assertTrue(f.qqq_lt_m.iloc[next_position+1:next_position+23].isna().all())
        self.assertTrue(np.isfinite(f.iloc[next_position+23].qqq_lt_m))

    def test_centered_two_pass_correlation_matches_exact_past_22_pairs(self):
        qqq, spx, iv = sample()
        f, _ = rr.build_features(qqq, spx, iv)
        qd = np.log(qqq.close/qqq.open).reindex(spx.index)
        sd = np.log(spx.close/spx.open)
        a, b = qd.iloc[68:90].to_numpy(), sd.iloc[68:90].to_numpy()
        a, b = a-a.mean(), b-b.mean()
        expected = a@b/(np.linalg.norm(a)*np.linalg.norm(b))
        self.assertAlmostEqual(f.iloc[90].corr22, expected)
        self.assertTrue(f.corr22.iloc[:22].isna().all())
        self.assertTrue(np.isfinite(f.iloc[22].corr22))

    def test_constant_day_return_correlation_is_unknown_without_fallback(self):
        qqq, spx, iv = sample()
        qqq["open"] = qqq.close
        f, _ = rr.build_features(qqq, spx, iv)
        self.assertTrue(f.corr22.isna().all())
        self.assertTrue(np.isfinite(f.iloc[90].qqq_lg_m))

    def test_nonzero_exact_constant_window_is_unknown_for_either_or_both_assets(self):
        dates = pd.bdate_range("2020-01-01", periods=30)
        constant = pd.Series(.1, index=dates)
        varied = pd.Series(np.linspace(.05, .15, len(dates)), index=dates)
        for a, b in [(constant, constant), (constant, varied), (varied, constant)]:
            with self.subTest(first_constant=a.equals(constant), second_constant=b.equals(constant)):
                self.assertTrue(rr._correlation22(a, b).isna().all())

    def test_missing_pair_stays_inside_both_history_and_correlation_windows(self):
        qqq, spx, iv = sample()
        qqq = qqq.drop(spx.index[80])
        f, target = rr.build_features(qqq, spx, iv)
        self.assertTrue(f.index.equals(spx.index))
        self.assertTrue(f.corr22.iloc[81:103].isna().all())
        self.assertTrue(f.qqq_lg_m.iloc[81:103].isna().all())
        self.assertTrue(f.qqq_lt_m.iloc[81:104].isna().all())
        self.assertTrue(f.qqq_neg_m.iloc[81:104].isna().all())
        self.assertTrue(np.isfinite(f.iloc[103].corr22))
        self.assertTrue(np.isfinite(f.iloc[104].qqq_lt_m))
        self.assertTrue(np.isnan(target.iloc[79].y))
        self.assertTrue(np.isfinite(target.iloc[80].y))

    def test_correlation_roundoff_tolerance_retains_small_overshoot_but_rejects_large(self):
        dates = pd.bdate_range("2020-01-01", periods=22)
        a = pd.Series(np.arange(22, dtype=float), index=dates)
        norm_product = np.linalg.norm(a-a.mean())**2
        with patch.object(rr.np, "dot", return_value=(1+5e-13)*norm_product):
            result = rr._correlation22(a, a)
        self.assertGreater(result.iloc[-1], 1.)
        with patch.object(rr.np, "dot", return_value=(1+2e-12)*norm_product), self.assertRaises(ValueError):
            rr._correlation22(a, a)

    def test_signed_target_zero_and_next_observed_close_maturity(self):
        _, spx, iv = sample()
        qqq = spx.copy()
        f, target = rr.build_features(qqq, spx, iv)
        self.assertTrue(target.y.iloc[:-1].eq(0.).all())
        self.assertEqual(target.iloc[36].target_end, spx.index[37])
        self.assertEqual(target.iloc[36].available_date, spx.index[37])
        self.assertTrue(target.iloc[-1].isna().all())
        qqq.loc[spx.index[91], "high"] *= 1.01
        _, changed = rr.build_features(qqq, spx, iv)
        self.assertGreater(changed.iloc[90].y, 0.)
        reverse, reverse_target = rr.build_features(spx, qqq, iv)
        self.assertTrue(reverse.index.equals(f.index))
        self.assertLess(reverse_target.iloc[90].y, 0.)

    def test_target_is_difference_of_logs_not_log_of_difference_or_ratio_of_means(self):
        qqq, spx, iv = sample()
        _, target = rr.build_features(qqq, spx, iv)
        expected = np.log(raw_gk(qqq).reindex(spx.index))-np.log(raw_gk(spx))
        np.testing.assert_allclose(target.y, expected.shift(-1), equal_nan=True, atol=1e-15)

    def test_entry_and_future_measurements_cannot_change_current_predictors(self):
        qqq, spx, iv = sample()
        before, _ = rr.build_features(qqq, spx, iv)
        changed_q, changed_s, changed_iv = qqq.copy(), spx.copy(), iv.copy()
        changed_q.loc[spx.index[90]:, ["open", "high", "low", "close"]] *= 3
        changed_q.loc[spx.index[90]:, "high"] *= 1.01
        changed_s.iloc[90:] *= 2
        changed_iv.iloc[90:] *= 1.5
        after, _ = rr.build_features(changed_q, changed_s, changed_iv)
        pd.testing.assert_frame_equal(before.iloc[:91], after.iloc[:91])

    def test_future_calendar_changes_target_date_but_not_current_predictors(self):
        qqq, spx, iv = sample()
        before, target_before = rr.build_features(qqq, spx, iv)
        after, target_after = rr.build_features(qqq, spx.drop(spx.index[91]), iv)
        pd.testing.assert_frame_equal(before.iloc[:91], after.iloc[:91])
        self.assertNotEqual(target_before.iloc[90].target_end, target_after.iloc[90].target_end)

    def test_price_units_and_unused_adjusted_close_leave_intraday_measurements_unchanged(self):
        qqq, spx, iv = sample()
        before, target_before = rr.build_features(qqq, spx, iv)
        qqq.loc[:, ["open", "high", "low", "close"]] *= 7
        spx.loc[:, ["open", "high", "low", "close"]] *= 3
        qqq["adj close"] = np.inf
        after, target_after = rr.build_features(qqq, spx, iv)
        np.testing.assert_allclose(before.loc[:, rr.ALL_FEATURES], after.loc[:, rr.ALL_FEATURES], atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(target_before.y, target_after.y, atol=1e-12, equal_nan=True)

    def test_missing_iv_is_not_filled_and_weekday_is_current_entry(self):
        qqq, spx, iv = sample()
        iv = iv.drop(spx.index[80])
        f, _ = rr.build_features(qqq, spx, iv)
        self.assertTrue(f.loc[spx.index[81], ["lvxn", "lvix", "term", "lvvix"]].isna().all())
        self.assertTrue(np.isfinite(f.loc[spx.index[82], "lvxn"]))
        for weekday in range(1, 5):
            np.testing.assert_array_equal(f[f"entry_dow_{weekday}"], (spx.index.weekday == weekday).astype(float))


class TestRelativeSources(unittest.TestCase):
    def test_bounded_qqq_and_vxn_values_are_parsed_only_after_date_fence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol = source_fixture(root)
            qqq, spx, iv, audit = rr.load_sources(protocol, root)
            self.assertEqual(str(qqq.index.min().date()), "2017-04-11")
            self.assertEqual(str(spx.index.min().date()), "2017-04-12")
            self.assertTrue(qqq.index.max() <= pd.Timestamp("2025-10-20"))
            self.assertEqual(tuple(qqq.columns), ("open", "high", "low", "close"))
            self.assertEqual(set(iv), {"vxn", "vix", "vix9d", "vvix"})
            self.assertFalse(audit["numeric_post_cutoff_values_parsed"])
            self.assertEqual(audit["sources"]["qqq"]["source_sha256"], hashlib.sha256((root/"qqq.parquet").read_bytes()).hexdigest())
            self.assertEqual(audit["sources"]["vxn"]["source_sha256"], hashlib.sha256((root/"vxn.csv").read_bytes()).hexdigest())
            self.assertNotIn("gap_interpretation", audit)
            self.assertEqual(audit["raw_columns"], list(rr.ALL_FEATURES))

    def test_fence_and_invalid_vxn_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol = source_fixture(root)
            protocol["index"]["source_end"] = "2025-11-03"
            with self.assertRaises(ValueError):
                rr.load_sources(protocol, root)
            protocol["index"]["source_end"] = "2025-10-20"
            (root/"vxn.csv").write_text("DATE,CLOSE\n04/12/2017,0\n")
            with self.assertRaises(ValueError):
                rr.load_sources(protocol, root)


if __name__ == "__main__":
    unittest.main()
