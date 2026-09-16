"""Prewritten synthetic strict-sign, source, paired-window, and timing contracts."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import joint_risk_features as joint
from src import sign_memory_features as sf
from tests.test_joint_risk_features import sample, source_fixture


class SignObservations(unittest.TestCase):
    def test_all_nine_sign_pairs_and_signed_zero_truth_table(self):
        pairs = np.array([(a, b) for a in (-1., 0., 1.) for b in (-1., 0., 1.)])
        observed = sf.sign_observations(pairs[:, 0], pairs[:, 1])
        expected = np.array([
            [0, 1, 0, 1, 1], [0, 1, 0, 0, 0], [0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0],
            [1, 0, 0, 1, 0], [1, 0, 0, 0, 0], [1, 0, 1, 0, 1],
        ], float)
        np.testing.assert_array_equal(observed, expected)
        np.testing.assert_array_equal(sf.sign_observations([-0., 0.], [0., -0.]), 0.)

    def test_tiny_values_keep_sign_without_product_underflow(self):
        tiny = np.nextafter(0., 1.)
        actual = sf.sign_observations([tiny, -tiny, tiny, -tiny], [tiny, -tiny, -tiny, tiny])
        np.testing.assert_array_equal(actual[:, 4], [1, 1, 0, 0])

    def test_missing_pair_masks_all_indicators_and_infinity_rejects(self):
        actual = sf.sign_observations([np.nan, .1, np.nan, .1], [.1, np.nan, np.nan, .2])
        self.assertTrue(np.isnan(actual[:3]).all())
        np.testing.assert_array_equal(actual[3], [1, 0, 1, 0, 1])
        for a, b in [([np.inf], [np.nan]), ([.1], [-np.inf])]:
            with self.assertRaises(ValueError):
                sf.sign_observations(a, b)

    def test_shape_complex_and_non_numeric_inputs_rejected(self):
        for a, b in [([1.], [1., 2.]), ([[1.]], [[1.]]), ([1j], [1.]), (["up"], [1.])]:
            with self.assertRaises(ValueError):
                sf.sign_observations(a, b)

    def test_sign_symmetry_and_positive_return_unit_invariance(self):
        a, b = np.array([-.02, .03, .04, 0]), np.array([-.01, -.02, .03, .04])
        observed = sf.sign_observations(a, b)
        np.testing.assert_array_equal(sf.sign_observations(a*3, b*7), observed)
        swapped = sf.sign_observations(b, a)
        np.testing.assert_array_equal(swapped, observed[:, [2, 3, 0, 1, 4]])
        reversed_sign = sf.sign_observations(-a, -b)
        np.testing.assert_array_equal(reversed_sign, observed[:, [1, 0, 3, 2, 4]])


class SignFeatures(unittest.TestCase):
    def test_exact_feature_target_schema_and_preserved_34_controls(self):
        sources = sample()
        old, _ = joint.build_features(*sources)
        features, targets = sf.build_features(*sources)
        self.assertEqual(sf.OLD_FEATURES, joint.ALL_FEATURES)
        self.assertEqual(sf.BOUNDED, ("qqq_pos22", "qqq_neg22", "spx_pos22", "spx_neg22", "independent22"))
        self.assertEqual(sf.MEMORY, "excess22")
        self.assertEqual(sf.ALL_FEATURES, sf.OLD_FEATURES+sf.BOUNDED+(sf.MEMORY,))
        self.assertEqual(len(sf.ALL_FEATURES), 40)
        self.assertEqual(sf.MODELS, ("frequency", "baseline", "memory"))
        self.assertEqual(tuple(features), sf.ALL_FEATURES+("feature_cutoff_date",))
        self.assertEqual(tuple(targets), ("y", "target_end", "available_date"))
        pd.testing.assert_frame_equal(features.loc[:, old.columns], old)
        self.assertNotIn("corr22_centered_sq", features)

    def test_joint_window_frequencies_independence_and_excess_exact_lag(self):
        qqq, spx, iv = sample()
        features, _ = sf.build_features(qqq, spx, iv)
        q = np.log(qqq.close/qqq.open).reindex(spx.index).to_numpy()
        s = np.log(spx.close/spx.open).to_numpy()
        for position in [22, 23, 70, len(spx)-1]:
            a, b = q[position-22:position], s[position-22:position]
            fractions = [(a > 0).mean(), (a < 0).mean(), (b > 0).mean(), (b < 0).mean()]
            independence = fractions[0]*fractions[2]+fractions[1]*fractions[3]
            agreement = (((a > 0) & (b > 0)) | ((a < 0) & (b < 0))).mean()
            np.testing.assert_allclose(features.loc[spx.index[position], sf.BOUNDED].to_numpy(float), fractions+[independence], rtol=0, atol=1e-15)
            self.assertAlmostEqual(features.iloc[position].excess22, agreement-independence, places=15)
        self.assertTrue(features.loc[:, sf.BOUNDED+(sf.MEMORY,)].iloc[:22].isna().all().all())
        self.assertEqual(features.iloc[70].feature_cutoff_date, spx.index[69])

    def test_missing_one_asset_invalidates_every_paired_summary_for_full22(self):
        qqq, spx, iv = sample()
        qqq.loc[spx.index[60], "open"] = np.nan
        features, targets = sf.build_features(qqq, spx, iv)
        self.assertTrue(features.index.equals(spx.index))
        selected = features.loc[:, sf.BOUNDED+(sf.MEMORY,)]
        self.assertTrue(selected.iloc[61:83].isna().all().all())
        self.assertTrue(np.isfinite(selected.iloc[60]).all())
        self.assertTrue(np.isfinite(selected.iloc[83]).all())
        self.assertTrue(np.isnan(targets.iloc[59].y))
        self.assertEqual(targets.iloc[59].target_end, spx.index[60])
        self.assertTrue(np.isfinite(targets.iloc[60].y))

    def test_missing_target_does_not_skip_to_next_common_date(self):
        qqq, spx, iv = sample()
        qqq = qqq.drop(spx.index[80])
        features, targets = sf.build_features(qqq, spx, iv)
        self.assertEqual(len(features), len(spx))
        self.assertTrue(np.isnan(targets.iloc[79].y))
        self.assertEqual(targets.iloc[79].target_end, spx.index[80])
        self.assertTrue(np.isfinite(targets.iloc[80].y))
        self.assertTrue(targets.iloc[-1].isna().all())
        pd.testing.assert_series_equal(targets.target_end, targets.available_date, check_names=False)

    def test_zero_returns_are_nonagreement_and_high_only_gap_does_not_mask_sign(self):
        qqq, spx, iv = sample()
        qqq["open"] = qqq.close
        qqq.loc[spx.index[80], "high"] = np.nan
        features, targets = sf.build_features(qqq, spx, iv)
        self.assertTrue(targets.y.iloc[:-1].eq(0).all())
        for name in ("qqq_pos22", "qqq_neg22", "independent22", "excess22"):
            self.assertTrue(features[name].iloc[22:].eq(0).all())
        self.assertTrue(np.isnan(features.iloc[81].qqq_lg_d))
        self.assertTrue(np.isfinite(features.iloc[81].spx_pos22))

    def test_target_is_next_actual_session_agreement_not_entry_sign(self):
        qqq, spx, iv = sample()
        _, targets = sf.build_features(qqq, spx, iv)
        q = np.log(qqq.close/qqq.open).reindex(spx.index)
        s = np.log(spx.close/spx.open)
        expected = (((q > 0) & (s > 0)) | ((q < 0) & (s < 0))).astype(float).shift(-1)
        np.testing.assert_array_equal(targets.y, expected)
        self.assertEqual(targets.iloc[11].target_end, spx.index[12])
        self.assertGreater((spx.index[12]-spx.index[11]).days, 1)

    def test_current_and_future_changes_cannot_change_current_predictors(self):
        qqq, spx, iv = sample()
        before, old_target = sf.build_features(qqq, spx, iv)
        qqq.loc[spx.index[90]:, "open"] *= 1.001
        spx.loc[spx.index[90]:, "high"] *= 1.01
        iv.iloc[90:] *= 1.5
        after, _ = sf.build_features(qqq, spx, iv)
        pd.testing.assert_frame_equal(before.iloc[:91], after.iloc[:91])
        changed_calendar, new_target = sf.build_features(qqq, spx.drop(spx.index[91]), iv)
        pd.testing.assert_frame_equal(before.iloc[:91], changed_calendar.iloc[:91])
        self.assertNotEqual(old_target.iloc[90].target_end, new_target.iloc[90].target_end)

    def test_prior_source_date_mismatch_affects_old_histories_not_intraday_signs(self):
        qqq, spx, iv = sample()
        extra = qqq.loc[[spx.index[11]]].copy()
        extra.index = pd.DatetimeIndex(["2014-01-20"], name="date")
        before, y_before = sf.build_features(qqq, spx, iv)
        after, y_after = sf.build_features(pd.concat([qqq, extra]).sort_index(), spx, iv)
        pd.testing.assert_frame_equal(before.loc[:, sf.BOUNDED+(sf.MEMORY,)], after.loc[:, sf.BOUNDED+(sf.MEMORY,)])
        pd.testing.assert_frame_equal(y_before, y_after)
        position = spx.index.searchsorted(extra.index[0])+1
        self.assertTrue(np.isnan(after.iloc[position].qqq_lt_d))

    def test_price_units_and_asset_permutation_preserve_joint_signs(self):
        qqq, spx, iv = sample()
        qqq = qqq.reindex(spx.index)
        before, targets = sf.build_features(qqq, spx, iv)
        after, swapped_y = sf.build_features(spx, qqq, iv)
        pd.testing.assert_frame_equal(targets, swapped_y)
        for name in ("independent22", "excess22"):
            np.testing.assert_array_equal(before[name], after[name])
        np.testing.assert_array_equal(before.qqq_pos22, after.spx_pos22)
        qqq.loc[:, ["open", "high", "low", "close"]] *= 7
        spx.loc[:, ["open", "high", "low", "close"]] *= 3
        qqq["adj close"] = np.inf
        scaled, scaled_y = sf.build_features(qqq, spx, iv)
        pd.testing.assert_frame_equal(targets, scaled_y)
        pd.testing.assert_frame_equal(before.loc[:, sf.BOUNDED+(sf.MEMORY,)], scaled.loc[:, sf.BOUNDED+(sf.MEMORY,)])

    def test_full_source_gate_precedes_all_new_sign_construction(self):
        for mode in ("floor", "overflow"):
            qqq, spx, iv = sample()
            if mode == "floor":
                spx.loc[spx.index[0], ["open", "high", "low", "close"]] = 100.
                iv.iloc[:40] = np.nan
            else:
                qqq.loc[spx.index[70], ["open", "high", "close"]] = [1e-308, np.nan, 1e308]
            with patch.object(sf, "sign_observations") as signs:
                with self.assertRaises(ValueError):
                    sf.build_features(qqq, spx, iv)
                signs.assert_not_called()

    def test_loader_preserves_sources_hashes_fence_and_raw_only_semantics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol = source_fixture(root)
            q, s, iv, audit = sf.load_sources(protocol, root)
            old_q, old_s, old_iv, old_audit = joint.load_sources(protocol, root)
            for actual, expected in ((q, old_q), (s, old_s), (iv, old_iv)):
                pd.testing.assert_frame_equal(actual, expected)
            self.assertEqual(audit["sources"], old_audit["sources"])
            self.assertEqual(audit["measurement_gate"], old_audit["measurement_gate"])
            self.assertEqual(audit["raw_columns"], list(sf.ALL_FEATURES))
            self.assertIn("strict", audit["target_interpretation"])
            self.assertFalse(audit["numeric_post_cutoff_values_parsed"])
            self.assertFalse(audit["historical_vintage_certified"])
            self.assertEqual(audit["source_end"], "2025-10-20")
            for name, path in protocol["sources"].items():
                self.assertEqual(audit["sources"][name]["source_sha256"], hashlib.sha256((root/path).read_bytes()).hexdigest())
            protocol["index"]["source_end"] = "2025-11-03"
            with self.assertRaises(ValueError):
                sf.load_sources(protocol, root)


if __name__ == "__main__":
    unittest.main()
