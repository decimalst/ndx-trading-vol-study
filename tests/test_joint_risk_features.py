"""Prewritten wave 10 paired-return source, measurement, and timing contracts."""
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import joint_risk_features as jf
from src import relative_risk_features as relative


def sample():
    dates = pd.bdate_range("2014-01-02", periods=150, name="date").delete([12, 37])
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


class TestJointMeasurement(unittest.TestCase):
    def test_old_gate_fields_preserved_and_new_signed_measurement_counts(self):
        qqq, spx, _ = sample()
        old = relative.measurement_audit(qqq, spx)
        audit = jf.measurement_audit(qqq, spx)
        self.assertEqual(set(audit), set(old))
        for key in ("status", "gk_floor", "reference_rows"):
            self.assertEqual(audit[key], old[key])
        for asset in ("qqq", "spx"):
            entry = audit["per_asset"][asset]
            self.assertEqual(set(entry), set(old["per_asset"][asset]) |
                             {"finite_intraday_return_rows", "zero_intraday_return_rows"})
            for key, value in old["per_asset"][asset].items():
                self.assertEqual(entry[key], value)
            self.assertEqual(entry["finite_intraday_return_rows"], len(spx))
            self.assertEqual(entry["zero_intraday_return_rows"], 0)
        jf.require_measurement(audit)

    def test_zero_signed_returns_are_valid_when_raw_gk_is_above_floor(self):
        qqq, spx, _ = sample()
        qqq["open"] = qqq.close
        audit = jf.measurement_audit(qqq, spx)
        self.assertEqual(audit["status"], "PASS")
        entry = audit["per_asset"]["qqq"]
        self.assertEqual(entry["zero_intraday_return_rows"], len(spx))
        self.assertEqual(entry["finite_intraday_return_rows"], len(spx))
        self.assertEqual(entry["floor_hit_rows"], 0)
        jf.require_measurement(audit)

    def test_floor_hit_outside_feature_complete_mask_stops_before_base_build(self):
        qqq, spx, iv = sample()
        spx.loc[spx.index[0], ["open", "high", "low", "close"]] = 100.
        iv.iloc[:40] = np.nan
        audit = jf.measurement_audit(qqq, spx)
        self.assertEqual(audit["status"], "INSUFFICIENT_MEASUREMENT")
        self.assertEqual(audit["per_asset"]["spx"]["floor_hit_rows"], 1)
        self.assertEqual(audit["per_asset"]["spx"]["zero_intraday_return_rows"], 1)
        with patch.object(relative, "build_features") as build:
            with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"):
                jf.build_features(qqq, spx, iv)
            build.assert_not_called()

    def test_missing_ohlc_rows_are_unknown_and_full_reference_is_retained(self):
        qqq, spx, _ = sample()
        qqq = qqq.drop(spx.index[50])
        spx.loc[spx.index[60], "high"] = np.nan
        audit = jf.measurement_audit(qqq, spx)
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["reference_rows"], len(spx))
        for asset in ("qqq", "spx"):
            self.assertEqual(audit["per_asset"][asset]["missing_ohlc_rows"], 1)
            self.assertEqual(audit["per_asset"][asset]["finite_intraday_return_rows"], len(spx)-1)

    def test_empty_complete_asset_cannot_pass(self):
        qqq, spx, _ = sample()
        qqq.loc[spx.index, ["open", "high", "low", "close"]] = np.nan
        audit = jf.measurement_audit(qqq, spx)
        self.assertEqual(audit["per_asset"]["qqq"]["finite_intraday_return_rows"], 0)
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"):
            jf.require_measurement(audit)

    def test_inconsistent_signed_gate_counts_are_not_accepted(self):
        audit = jf.measurement_audit(*sample()[:2])
        for field, value in [("finite_intraday_return_rows", 1), ("zero_intraday_return_rows", -1),
                             ("zero_intraday_return_rows", 9999)]:
            altered = copy.deepcopy(audit)
            altered["per_asset"]["qqq"][field] = value
            with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"):
                jf.require_measurement(altered)

    def test_nonfinite_signed_computation_is_rejected_even_if_old_gate_claims_pass(self):
        qqq, spx, _ = sample()
        old = relative.measurement_audit(qqq, spx)
        qqq.loc[spx.index[70], ["open", "high", "low", "close"]] = [1e-308, 1e308, 1e-308, 1e308]
        with patch.object(relative, "measurement_audit", return_value=old), self.assertRaisesRegex(ValueError, "[Nn]onfinite.*intraday"):
            jf.measurement_audit(qqq, spx)

    def test_pre_reference_qqq_rows_do_not_change_full_reference_gate(self):
        qqq, spx, _ = sample()
        qqq.loc[qqq.index[0], ["open", "high", "low", "close"]] = 100.
        self.assertEqual(jf.measurement_audit(qqq, spx)["status"], "PASS")

    def test_partial_high_low_gap_cannot_hide_nonfinite_observed_open_close_return(self):
        qqq, spx, iv = sample()
        qqq.loc[spx.index[70], ["open", "high", "close"]] = [1e-308, np.nan, 1e308]
        self.assertEqual(relative.measurement_audit(qqq, spx)["status"], "PASS")
        with self.assertRaisesRegex(ValueError, "[Nn]onfinite.*intraday"):
            jf.measurement_audit(qqq, spx)
        with patch.object(relative, "build_features") as build:
            with self.assertRaisesRegex(ValueError, "[Nn]onfinite.*intraday"):
                jf.build_features(qqq, spx, iv)
            build.assert_not_called()


class TestJointFeatures(unittest.TestCase):
    def test_exact_34_column_design_and_paired_target_schema(self):
        additions = tuple(f"{asset}_day_{suffix}" for asset in ("qqq", "spx") for suffix in ("d", "w", "m"))
        self.assertEqual(jf.BASE, relative.ALL_FEATURES)
        self.assertEqual(jf.ADDITIONS, additions)
        self.assertEqual(jf.ALL_FEATURES, relative.ALL_FEATURES+additions)
        self.assertEqual(len(jf.ALL_FEATURES), 34)
        self.assertEqual(jf.MODELS, ("constant_matrix", "constant_correlation", "dynamic_correlation"))
        f, target = jf.build_features(*sample())
        self.assertEqual(tuple(f), jf.ALL_FEATURES+("feature_cutoff_date",))
        self.assertEqual(tuple(target), ("y_qqq", "y_spx", "target_end", "available_date"))
        self.assertNotIn("corr22_centered_sq", f)

    def test_old_28_features_and_cutoff_are_preserved_exactly(self):
        fixture = sample()
        old, _ = relative.build_features(*fixture)
        new, _ = jf.build_features(*fixture)
        pd.testing.assert_frame_equal(new.loc[:, old.columns], old)

    def test_signed_day_means_are_strict_arithmetic_windows_lagged_one_session(self):
        qqq, spx, iv = sample()
        f, _ = jf.build_features(qqq, spx, iv)
        for asset, source in [("qqq", qqq), ("spx", spx)]:
            day = np.log(source.close/source.open).reindex(spx.index)
            for suffix, width in [("d", 1), ("w", 5), ("m", 22)]:
                expected = day.rolling(width, min_periods=width).mean().shift(1)
                np.testing.assert_allclose(f[f"{asset}_day_{suffix}"], expected, equal_nan=True, atol=1e-15)
                self.assertTrue(f[f"{asset}_day_{suffix}"].iloc[:width].isna().all())
                self.assertTrue(np.isfinite(f[f"{asset}_day_{suffix}"].iloc[width]))
        self.assertEqual(f.iloc[90].feature_cutoff_date, spx.index[89])

    def test_paired_targets_are_signed_next_actual_spx_session_returns(self):
        qqq, spx, iv = sample()
        _, target = jf.build_features(qqq, spx, iv)
        for asset, source in [("qqq", qqq), ("spx", spx)]:
            expected = np.log(source.close/source.open).reindex(spx.index).shift(-1)
            np.testing.assert_allclose(target[f"y_{asset}"], expected, equal_nan=True, atol=1e-15)
            self.assertTrue((target[f"y_{asset}"] > 0).any())
            self.assertTrue((target[f"y_{asset}"] < 0).any())
        self.assertEqual(target.iloc[11].target_end, spx.index[12])
        self.assertGreater((spx.index[12]-spx.index[11]).days, 1)
        pd.testing.assert_series_equal(target.target_end, target.available_date, check_names=False)
        self.assertTrue(target.iloc[-1].isna().all())

    def test_zero_targets_preserved_without_pseudovalues(self):
        qqq, spx, iv = sample()
        qqq["open"] = qqq.close
        f, target = jf.build_features(qqq, spx, iv)
        self.assertTrue(target.y_qqq.iloc[:-1].eq(0).all())
        self.assertTrue(f.qqq_day_m.iloc[22:].eq(0).all())
        self.assertTrue(f.corr22.isna().all())
        self.assertTrue(target.y_spx.iloc[:-1].notna().all())

    def test_missing_qqq_target_does_not_select_the_next_common_session(self):
        qqq, spx, iv = sample()
        qqq = qqq.drop(spx.index[80])
        f, target = jf.build_features(qqq, spx, iv)
        self.assertTrue(f.index.equals(spx.index))
        self.assertEqual(len(target), len(spx))
        self.assertTrue(np.isnan(target.iloc[79].y_qqq))
        self.assertTrue(np.isfinite(target.iloc[79].y_spx))
        self.assertEqual(target.iloc[79].target_end, spx.index[80])
        self.assertTrue(np.isfinite(target.iloc[80].y_qqq))
        self.assertTrue(f.qqq_day_m.iloc[81:103].isna().all())
        self.assertTrue(np.isfinite(f.iloc[103].qqq_day_m))

    def test_missing_open_stays_in_signed_history_and_target_but_high_only_does_not_mask_return(self):
        qqq, spx, iv = sample()
        qqq.loc[spx.index[80], "open"] = np.nan
        qqq.loc[spx.index[110], "high"] = np.nan
        f, target = jf.build_features(qqq, spx, iv)
        self.assertTrue(np.isnan(target.iloc[79].y_qqq))
        self.assertTrue(f.qqq_day_m.iloc[81:103].isna().all())
        self.assertTrue(np.isfinite(target.iloc[109].y_qqq))
        self.assertTrue(np.isfinite(f.iloc[111].qqq_day_d))
        self.assertTrue(np.isnan(f.iloc[111].qqq_lg_d))

    def test_source_predecessor_mismatch_preserves_same_day_return(self):
        qqq, spx, iv = sample()
        extra = qqq.loc[[spx.index[11]]].copy()
        extra.index = pd.DatetimeIndex(["2014-01-20"], name="date")
        changed = pd.concat([qqq, extra]).sort_index()
        before, target_before = jf.build_features(qqq, spx, iv)
        after, target_after = jf.build_features(changed, spx, iv)
        entry = spx.index.searchsorted(extra.index[0])+1
        self.assertTrue(np.isnan(after.iloc[entry].qqq_lt_d))
        self.assertTrue(np.isnan(after.iloc[entry].qqq_neg_d))
        pd.testing.assert_frame_equal(before.loc[:, jf.ADDITIONS], after.loc[:, jf.ADDITIONS])
        pd.testing.assert_frame_equal(target_before, target_after)

    def test_entry_and_future_prices_and_iv_cannot_change_current_predictors(self):
        qqq, spx, iv = sample()
        before, _ = jf.build_features(qqq, spx, iv)
        qqq.loc[spx.index[90]:, ["open", "high", "low", "close"]] *= 3
        qqq.loc[spx.index[90]:, "open"] *= 1.001
        spx.loc[spx.index[90]:, "high"] *= 1.01
        iv.iloc[90:] *= 1.5
        after, _ = jf.build_features(qqq, spx, iv)
        pd.testing.assert_frame_equal(before.iloc[:91], after.iloc[:91])

    def test_future_calendar_does_not_change_current_predictors(self):
        qqq, spx, iv = sample()
        before, old_target = jf.build_features(qqq, spx, iv)
        after, new_target = jf.build_features(qqq, spx.drop(spx.index[91]), iv)
        pd.testing.assert_frame_equal(before.iloc[:91], after.iloc[:91])
        self.assertNotEqual(old_target.iloc[90].target_end, new_target.iloc[90].target_end)

    def test_fixed_price_unit_changes_and_unused_adjusted_close_do_not_change_returns(self):
        qqq, spx, iv = sample()
        before, target_before = jf.build_features(qqq, spx, iv)
        qqq.loc[:, ["open", "high", "low", "close"]] *= 7
        spx.loc[:, ["open", "high", "low", "close"]] *= 3
        qqq["adj close"] = np.inf
        after, target_after = jf.build_features(qqq, spx, iv)
        np.testing.assert_allclose(before.loc[:, jf.ALL_FEATURES], after.loc[:, jf.ALL_FEATURES], equal_nan=True, atol=1e-12)
        np.testing.assert_allclose(target_before[["y_qqq", "y_spx"]], target_after[["y_qqq", "y_spx"]], equal_nan=True, atol=1e-12)

    def test_asset_permutation_swaps_returns_and_histories_and_keeps_pair_correlation(self):
        qqq, spx, iv = sample()
        qqq = qqq.reindex(spx.index)
        before, target_before = jf.build_features(qqq, spx, iv)
        after, target_after = jf.build_features(spx, qqq, iv)
        for suffix in ("d", "w", "m"):
            np.testing.assert_allclose(before[f"qqq_day_{suffix}"], after[f"spx_day_{suffix}"], equal_nan=True)
            np.testing.assert_allclose(before[f"spx_day_{suffix}"], after[f"qqq_day_{suffix}"], equal_nan=True)
        np.testing.assert_allclose(before.corr22, after.corr22, equal_nan=True)
        np.testing.assert_allclose(target_before.y_qqq, target_after.y_spx, equal_nan=True)
        np.testing.assert_allclose(target_before.y_spx, target_after.y_qqq, equal_nan=True)

    def test_invalid_order_duplicate_dates_and_prices_are_rejected(self):
        qqq, spx, iv = sample()
        for altered in [spx.iloc[::-1], pd.concat([spx.iloc[[0]], spx])]:
            with self.assertRaises(ValueError):
                jf.build_features(qqq, altered, iv)
        altered = spx.copy()
        altered.loc[spx.index[60], "close"] = 0.
        with self.assertRaises(ValueError):
            jf.build_features(qqq, altered, iv)


class TestJointSources(unittest.TestCase):
    def test_bound_before_numeric_parse_and_preserve_raw_source_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol = source_fixture(root)
            qqq, spx, iv, audit = jf.load_sources(protocol, root)
            old_q, old_s, old_iv, old_audit = relative.load_sources(protocol, root)
            pd.testing.assert_frame_equal(qqq, old_q)
            pd.testing.assert_frame_equal(spx, old_s)
            pd.testing.assert_frame_equal(iv, old_iv)
            self.assertEqual(audit["sources"], old_audit["sources"])
            self.assertEqual(audit["source_end"], "2025-10-20")
            self.assertEqual(audit["sealed_start"], "2025-11-03")
            self.assertFalse(audit["numeric_post_cutoff_values_parsed"])
            self.assertFalse(audit["historical_vintage_certified"])
            self.assertEqual(audit["raw_columns"], list(jf.ALL_FEATURES))
            self.assertIn("paired", audit["target_interpretation"])
            self.assertNotIn("gap_interpretation", audit)
            self.assertEqual(tuple(qqq), ("open", "high", "low", "close"))
            self.assertEqual(str(qqq.index.min().date()), "2017-04-11")
            self.assertEqual(str(spx.index.min().date()), "2017-04-12")
            for name, path in protocol["sources"].items():
                self.assertEqual(audit["sources"][name]["source_sha256"], hashlib.sha256((root/path).read_bytes()).hexdigest())

    def test_source_fence_and_observed_invalid_iv_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol = source_fixture(root)
            protocol["index"]["source_end"] = "2025-11-03"
            with self.assertRaises(ValueError):
                jf.load_sources(protocol, root)
            protocol["index"]["source_end"] = "2025-10-20"
            (root/"vxn.csv").write_text("DATE,CLOSE\n04/12/2017,0\n")
            with self.assertRaises(ValueError):
                jf.load_sources(protocol, root)


if __name__ == "__main__":
    unittest.main()
