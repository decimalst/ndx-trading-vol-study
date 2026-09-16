"""Source and timing contracts written before measurement-memory implementation."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import measurement_memory as memory


def sample(n=100):
    dates = pd.bdate_range("2015-01-02", periods=n, name="date")
    returns = np.resize(np.array([.01, -.02, .005, -.004, 0.]), n)
    daily = pd.DataFrame({"adj close": 100*np.exp(np.cumsum(returns))}, index=dates)
    iv = pd.DataFrame({"vix": 20+np.arange(n)*.1, "vix9d": 16+np.arange(n)*.05,
                       "vvix": 80+np.arange(n)*.2}, index=dates)
    vol = .1+np.arange(n)*.003
    table = pd.DataFrame({"qmle": vol, "rv5": 1.05*vol, "rv15": .97*vol,
                          "ci_width": .001+np.arange(n)*.0001}, index=dates)
    return daily, iv, table


def response(rows):
    dates = [row.split()[1] for row in rows]
    return ("\n".join(["SPY", "84398", "SPDR S & P 500 E T F TRUST", str(len(rows)),
                       dates[0], dates[-1], *rows])+"\n").encode()


class TestMeasurementMemory(unittest.TestCase):
    def test_fixed_feature_model_and_measurement_schema(self):
        self.assertEqual(memory.BASE, ("const", "lq_d", "lq_w", "lq_m", "lvix", "livshape", "lvvix",
                                       "neg_d", "neg_w", "neg_m", "entry_dow_1", "entry_dow_2",
                                       "entry_dow_3", "entry_dow_4"))
        self.assertEqual(memory.ALL_FEATURES, memory.BASE+("width", "quality_memory"))
        self.assertEqual(memory.MODELS, ("mean", "baseline", "width", "quality"))
        self.assertEqual(memory.MEASURES, ("qmle", "rv5", "rv15"))
        features, targets = memory.build_features(*sample())
        self.assertEqual(tuple(features.columns), memory.ALL_FEATURES+("market_cutoff_date", "measurement_cutoff_date"))
        self.assertEqual(tuple(targets.columns), ("y_qmle", "y_rv5", "y_rv15", "target_end", "available_date"))

    def test_measurement_log_means_and_quality_interaction_use_two_session_lag(self):
        daily, iv, table = sample()
        features, _ = memory.build_features(daily, iv, table)
        t, s = 50, 48
        for suffix, width in [("d", 1), ("w", 5), ("m", 22)]:
            expected = np.log(table.qmle.iloc[s-width+1:s+1].pow(2).mean())
            self.assertAlmostEqual(features.iloc[t]["lq_"+suffix], expected)
        width = np.log1p(table.ci_width.iloc[s]/table.qmle.iloc[s])
        self.assertAlmostEqual(features.iloc[t].width, width)
        self.assertAlmostEqual(features.iloc[t].quality_memory, width*(features.iloc[t].lq_d-features.iloc[t].lq_m))
        self.assertEqual(features.iloc[t].measurement_cutoff_date, daily.index[s])
        self.assertEqual(features.iloc[t].market_cutoff_date, daily.index[t-1])

    def test_market_controls_are_positive_negative_return_magnitudes_and_lagged_logs(self):
        daily, iv, table = sample()
        features, _ = memory.build_features(daily, iv, table)
        negative = (-np.log(daily["adj close"]).diff()).clip(lower=0)
        t = 50
        for suffix, width in [("d", 1), ("w", 5), ("m", 22)]:
            self.assertAlmostEqual(features.iloc[t]["neg_"+suffix], negative.iloc[t-width:t].mean())
        self.assertAlmostEqual(features.iloc[t].lvix, np.log(iv.vix.iloc[t-1]))
        self.assertAlmostEqual(features.iloc[t].livshape, np.log(iv.vix9d.iloc[t-1]/iv.vix.iloc[t-1]))
        self.assertAlmostEqual(features.iloc[t].lvvix, np.log(iv.vvix.iloc[t-1]))

    def test_three_native_squared_targets_and_two_session_maturity(self):
        daily, iv, table = sample()
        _, targets = memory.build_features(daily, iv, table)
        t = 40
        for name in memory.MEASURES:
            self.assertAlmostEqual(targets.iloc[t]["y_"+name], table.iloc[t+1][name]**2)
        self.assertEqual(targets.iloc[t].target_end, daily.index[t+1])
        self.assertEqual(targets.iloc[t].available_date, daily.index[t+3])
        self.assertTrue(targets.available_date.iloc[-3:].isna().all())
        self.assertTrue(pd.isna(targets.iloc[-1].target_end))

    def test_missing_source_date_is_not_skipped_in_features_or_next_target(self):
        daily, iv, table = sample()
        features, targets = memory.build_features(daily, iv, table.drop(table.index[40]))
        self.assertEqual(len(features), len(daily))
        self.assertTrue(np.isnan(features.iloc[42].lq_d))
        self.assertTrue(features.lq_w.iloc[42:47].isna().all())
        self.assertTrue(np.isfinite(features.iloc[47].lq_w))
        self.assertTrue(features.lq_m.iloc[42:64].isna().all())
        self.assertTrue(np.isfinite(features.iloc[64].lq_m))
        self.assertTrue(np.isnan(targets.iloc[39].y_qmle))
        self.assertEqual(targets.iloc[39].target_end, daily.index[40])
        self.assertEqual(targets.iloc[39].available_date, daily.index[42])

    def test_reference_calendar_holidays_control_lags_and_maturity(self):
        daily, iv, table = sample()
        keep = daily.index.delete([10, 20])
        features, targets = memory.build_features(daily.loc[keep], iv, table)
        self.assertEqual(features.iloc[25].measurement_cutoff_date, keep[23])
        self.assertEqual(features.iloc[25].market_cutoff_date, keep[24])
        self.assertEqual(targets.iloc[25].target_end, keep[26])
        self.assertEqual(targets.iloc[25].available_date, keep[28])

    def test_future_measurements_cannot_change_available_predictors(self):
        daily, iv, table = sample()
        before, _ = memory.build_features(daily, iv, table)
        changed = table.copy()
        changed.iloc[40:] *= 4
        after, _ = memory.build_features(daily, iv, changed)
        pd.testing.assert_frame_equal(before.iloc[:42], after.iloc[:42])
        self.assertNotEqual(before.iloc[42].lq_d, after.iloc[42].lq_d)

    def test_future_market_values_cannot_change_previous_session_controls(self):
        daily, iv, table = sample()
        before, _ = memory.build_features(daily, iv, table)
        changed, changed_iv = daily.copy(), iv.copy()
        changed.iloc[40:] *= 3
        changed_iv.iloc[40:] *= 2
        after, _ = memory.build_features(changed, changed_iv, table)
        pd.testing.assert_frame_equal(before.iloc[:41], after.iloc[:41])

    def test_removing_a_future_realized_session_cannot_change_current_features(self):
        daily, iv, table = sample()
        before, target_before = memory.build_features(daily, iv, table)
        after, target_after = memory.build_features(daily.drop(daily.index[41]), iv, table)
        pd.testing.assert_series_equal(before.loc[daily.index[40]], after.loc[daily.index[40]])
        self.assertNotEqual(target_before.loc[daily.index[40], "target_end"], target_after.loc[daily.index[40], "target_end"])

    def test_constant_native_unit_change_preserves_relative_width_and_memory(self):
        daily, iv, table = sample()
        before, target_before = memory.build_features(daily, iv, table)
        after, target_after = memory.build_features(daily, iv, table*7)
        np.testing.assert_allclose(after.loc[:, ["lq_d", "lq_w", "lq_m"]],
                                   before.loc[:, ["lq_d", "lq_w", "lq_m"]]+2*np.log(7), equal_nan=True)
        np.testing.assert_allclose(after.loc[:, ["width", "quality_memory"]], before.loc[:, ["width", "quality_memory"]], equal_nan=True, atol=1e-14)
        np.testing.assert_allclose(target_after.loc[:, ["y_"+x for x in memory.MEASURES]],
                                   target_before.loc[:, ["y_"+x for x in memory.MEASURES]]*49, equal_nan=True)

    def test_zero_width_is_valid_and_invalid_width_is_unknown(self):
        daily, iv, table = sample()
        table.loc[table.index[40], "ci_width"] = 0
        table.loc[table.index[41], "ci_width"] = -1
        table.loc[table.index[42], "ci_width"] = np.inf
        features, _ = memory.build_features(daily, iv, table)
        self.assertEqual(features.iloc[42].width, 0)
        self.assertEqual(features.iloc[42].quality_memory, 0)
        self.assertTrue(features.width.iloc[43:45].isna().all())

    def test_invalid_native_volatility_remains_unknown_not_zero_target(self):
        for bad in [0., -1., np.nan, np.inf]:
            with self.subTest(value=bad):
                daily, iv, table = sample()
                table.loc[table.index[40], "qmle"] = bad
                table.loc[table.index[41], "rv5"] = bad
                features, targets = memory.build_features(daily, iv, table)
                self.assertTrue(np.isnan(features.iloc[42].lq_d))
                self.assertTrue(np.isnan(targets.iloc[39].y_qmle))
                self.assertTrue(np.isnan(targets.iloc[40].y_rv5))
                self.assertTrue(np.isfinite(targets.iloc[40].y_qmle))

    def test_observed_extremes_are_never_chart_clipped(self):
        daily, iv, table = sample()
        table.loc[table.index[40], "qmle"] = 4
        features, targets = memory.build_features(daily, iv, table)
        self.assertEqual(targets.iloc[39].y_qmle, 16)
        self.assertAlmostEqual(features.iloc[42].lq_d, np.log(16))

    def test_missing_market_row_stays_inside_return_rolling_windows(self):
        daily, iv, table = sample()
        daily.loc[daily.index[40], "adj close"] = np.nan
        features, _ = memory.build_features(daily, iv, table)
        self.assertTrue(features.neg_w.iloc[41:47].isna().all())
        self.assertTrue(np.isfinite(features.iloc[47].neg_w))

    def test_invalid_calendar_or_observed_market_prices_are_rejected(self):
        daily, iv, table = sample()
        with self.assertRaises(ValueError):
            memory.build_features(daily.iloc[::-1], iv, table)
        daily.loc[daily.index[40], "adj close"] = 0
        with self.assertRaises(ValueError):
            memory.build_features(daily, iv, table)

    def test_source_hash_is_checked_before_the_frozen_parser(self):
        raw = response(["84398 20250102 4 2 0 5 6 7 3 .8 10 11"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"provider.txt"
            path.write_bytes(raw)
            with patch.object(memory, "parse_equity_response") as parser:
                with self.assertRaisesRegex(ValueError, "hash"):
                    memory.read_measurements(path)
                parser.assert_not_called()

    def test_field_mapping_and_after_cutoff_numerical_tokens_are_not_parsed(self):
        raw = response(["84398 20250102 4 2 0 5 6 7 3 .8 10 11",
                        "84398 20251103 forbidden numerical tokens"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"provider.txt"
            path.write_bytes(raw)
            with patch.object(memory, "SOURCE_SHA256", hashlib.sha256(raw).hexdigest()):
                table, audit = memory.read_measurements(path)
                with self.assertRaisesRegex(ValueError, "cutoff"):
                    memory.read_measurements(path, "2025-11-03")
        self.assertEqual(table.index.tolist(), [pd.Timestamp("2025-01-02")])
        self.assertEqual(table.iloc[0].to_dict(), {"qmle": 4., "rv5": 5., "rv15": 6., "ci_width": 0.})
        self.assertFalse(audit["numeric_post_cutoff_values_parsed"])
        self.assertEqual(audit["after_cutoff_rows"], 1)

    def test_source_invalid_values_remain_missing_and_are_counted(self):
        raw = response(["84398 20250102 0 2 -1 -2 inf 7 3 .8 10 11"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"provider.txt"
            path.write_bytes(raw)
            with patch.object(memory, "SOURCE_SHA256", hashlib.sha256(raw).hexdigest()):
                table, audit = memory.read_measurements(path)
        self.assertTrue(table.iloc[0].isna().all())
        self.assertEqual(audit["invalid_selected_fields"], {"qmle": 1, "rv5": 1, "rv15": 1, "ci_width": 1})

    def test_source_loader_bounds_before_numeric_conversion_and_preserves_calendar(self):
        raw = response(["84398 20150102 .2 2 .01 .21 .22 .23 3 .02 .24 .25",
                        "84398 20251103 forbidden numerical tokens"])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"risklab.txt").write_bytes(raw)
            dates = pd.DatetimeIndex(["2015-01-02", "2015-01-05", "2025-11-03"], name="date")
            pd.DataFrame({"adj close": [100., 101., np.inf]}, index=dates).to_parquet(root/"daily.parquet")
            for name, column, first in [("vix", "CLOSE", 20), ("vix9d", "CLOSE", 15), ("vvix", "VVIX", 80)]:
                (root/(name+".csv")).write_text(f"DATE,{column}\n01/02/2015,{first}\n01/05/2015,{first+1}\n11/03/2025,DO_NOT_PARSE\n")
            protocol = {"sources": {"risklab": "risklab.txt", "daily": "daily.parquet",
                                     "vix": "vix.csv", "vix9d": "vix9d.csv", "vvix": "vvix.csv"},
                        "index": {"source_end": "2025-10-20"}}
            with patch.object(memory, "SOURCE_SHA256", hashlib.sha256(raw).hexdigest()):
                daily, iv, table, audit = memory.load_sources(protocol, root=root)
        self.assertEqual(daily.index.tolist(), dates[:2].tolist())
        self.assertEqual(iv.index.tolist(), dates[:2].tolist())
        self.assertEqual(iv.iloc[0].to_dict(), {"vix": 20., "vix9d": 15., "vvix": 80.})
        self.assertEqual(table.index.tolist(), dates[:1].tolist())
        self.assertEqual(audit["risklab"]["source_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(audit["measurement_calendar"]["missing_source_dates"], ["2015-01-05"])


if __name__ == "__main__":
    unittest.main()
