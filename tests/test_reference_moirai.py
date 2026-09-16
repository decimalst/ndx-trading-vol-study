"""Pre-checkpoint contracts: synthetic inputs only, with no market-file reads."""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src import reference_moirai as moirai


class CausalWindowContracts(unittest.TestCase):
    def setUp(self):
        self.sessions = pd.bdate_range("2007-01-02", periods=1200).delete([19, 75, 113])
        self.rv = pd.Series(np.exp(-9 + np.arange(len(self.sessions)) / 10000),
                            index=self.sessions, name="rv_total")

    def test_window_has_exactly_512_actual_sessions_ending_at_origin(self):
        origin = self.sessions[700]
        windows, audit = moirai.causal_windows(self.rv, pd.DatetimeIndex([origin]))
        self.assertEqual(windows.shape, (1, 512))
        np.testing.assert_allclose(windows[0], np.log(self.rv.iloc[189:701]), atol=0)
        self.assertEqual(audit.iloc[0].context_start, self.sessions[189])
        self.assertEqual(audit.iloc[0].context_end, origin)
        self.assertEqual(audit.iloc[0].context_rows, 512)

    def test_origins_are_returned_in_order_without_target_shift(self):
        origins = self.sessions[[511, 600, 1000]]
        windows, audit = moirai.causal_windows(self.rv, origins)
        self.assertTrue(audit.index.equals(origins.rename("origin")))
        np.testing.assert_allclose(windows[:, -1], np.log(self.rv.loc[origins]), atol=0)
        # Forecast step one belongs to session after the retained origin; h5
        # spans five following sessions, never five context values.
        for origin in origins:
            p = self.sessions.get_loc(origin)
            self.assertEqual(self.sessions[p + 1:p + 6][0], self.sessions[p + 1])
            self.assertGreater(self.sessions[p + 1:p + 6][-1], audit.loc[origin, "context_end"])

    def test_future_value_poisoning_and_truncation_preserve_every_window(self):
        origins = self.sessions[[700, 701]]
        expected, _ = moirai.causal_windows(self.rv, origins)
        changed = self.rv.copy()
        changed.iloc[702:] = np.nan
        after, _ = moirai.causal_windows(changed, origins)
        prefix, _ = moirai.causal_windows(self.rv.iloc[:702], origins)
        np.testing.assert_array_equal(expected, after)
        np.testing.assert_array_equal(expected, prefix)

    def test_older_than_context_values_cannot_change_model_inputs(self):
        changed = self.rv.copy()
        changed.iloc[:189] = -100
        before, _ = moirai.causal_windows(self.rv, self.sessions[[700]])
        after, _ = moirai.causal_windows(changed, self.sessions[[700]])
        np.testing.assert_array_equal(before, after)

    def test_no_short_context_padding_or_missing_value_imputation(self):
        with self.assertRaises(ValueError):
            moirai.causal_windows(self.rv, self.sessions[[510]])
        for bad in [0., -1., np.nan, np.inf]:
            changed = self.rv.copy()
            changed.iloc[650] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                moirai.causal_windows(changed, self.sessions[[700]])

    def test_unknown_duplicate_or_unordered_origins_fail(self):
        for origins in [pd.DatetimeIndex(["2007-01-06"]), self.sessions[[701, 700]],
                        self.sessions[[700, 700]], pd.DatetimeIndex([])]:
            with self.subTest(origins=origins), self.assertRaises(ValueError):
                moirai.causal_windows(self.rv, origins)

    def test_duplicate_or_unsorted_source_sessions_fail(self):
        for series in [pd.concat([self.rv, self.rv.iloc[-1:]]), self.rv.iloc[::-1]]:
            with self.assertRaises(ValueError):
                moirai.causal_windows(series, self.sessions[[700]])

    def test_fixed_context_cannot_be_shortened(self):
        self.assertEqual(moirai.CONTEXT_LENGTH, 512)
        self.assertEqual(moirai.PREDICTION_LENGTH, 5)


class SummaryContracts(unittest.TestCase):
    def setUp(self):
        self.levels = np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9])
        self.output = np.zeros((2, 9, 5))
        self.output[0, 4] = np.log([1., 2., 3., 4., 5.])
        self.output[1, 4] = np.log([6., 7., 8., 9., 10.])

    def test_horizon_mapping_uses_step1_and_all_five_variance_scale_medians(self):
        result = moirai.quantile_summaries(self.output, self.levels)
        np.testing.assert_allclose(result[:, 0], np.log([1., 6.]), atol=1e-14)
        np.testing.assert_allclose(result[:, 1], np.log([3., 8.]), atol=1e-14)
        self.assertNotAlmostEqual(result[0, 1], np.log([1., 2., 3., 4., 5.]).mean())

    def test_median_selected_by_numeric_level_not_fixed_array_position(self):
        order = np.array([8, 4, 1, 0, 2, 7, 6, 5, 3])
        expected = moirai.quantile_summaries(self.output, self.levels)
        got = moirai.quantile_summaries(self.output[:, order], self.levels[order])
        np.testing.assert_array_equal(got, expected)

    def test_univariate_trailing_dimension_is_supported(self):
        np.testing.assert_array_equal(moirai.quantile_summaries(self.output, self.levels),
                                      moirai.quantile_summaries(self.output[..., None], self.levels))

    def test_stable_logmeanexp_does_not_overflow_or_underflow(self):
        for shift in [-10000., 10000.]:
            shifted = moirai.quantile_summaries(self.output + shift, self.levels)
            expected = moirai.quantile_summaries(self.output, self.levels) + shift
            np.testing.assert_allclose(shifted, expected, atol=1e-11)

    def test_nonmedian_quantiles_cannot_change_summary(self):
        changed = self.output.copy()
        changed[:, [0, 1, 2, 3, 5, 6, 7, 8]] = 100
        np.testing.assert_array_equal(moirai.quantile_summaries(changed, self.levels),
                                      moirai.quantile_summaries(self.output, self.levels))

    def test_absent_or_duplicate_median_wrong_horizon_and_nonfinite_fail(self):
        cases = [(self.output[:, :8], self.levels[self.levels != .5]),
                 (self.output, np.full(9, .5)), (self.output[:, :, :4], self.levels),
                 (np.full_like(self.output, np.nan), self.levels),
                 (np.zeros((2, 9, 5, 2)), self.levels)]
        for values, levels in cases:
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                moirai.quantile_summaries(values, levels)


class FrozenExtractionContracts(unittest.TestCase):
    def test_extraction_period_and_named_summary_contract(self):
        self.assertEqual(moirai.ORIGIN_START, "2010-01-04")
        self.assertEqual(moirai.ORIGIN_END, "2025-10-10")
        self.assertEqual(moirai.COLUMNS, ("moirai_log_h1", "moirai_log_h5"))
        self.assertEqual(len(moirai.CODE_REVISION), 40)
        self.assertEqual(len(moirai.MODEL_REVISION), 40)

    def test_hash_validation_rejects_a_changed_checkpoint(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "weights"
            path.write_bytes(b"fixed")
            moirai.require_hash(path, moirai.sha256(path))
            with self.assertRaises(ValueError):
                moirai.require_hash(path, "0" * 64)

    def test_batch_extraction_maps_fake_model_outputs_back_to_correct_origins(self):
        sessions = pd.bdate_range("2007-01-02", periods=600)
        rv = pd.Series(np.exp(np.arange(600) / 100), index=sessions)
        origins = sessions[[511, 530, 590]]
        captured = []
        def fake_predict(windows):
            captured.extend(windows.copy())
            # Distinct step medians expose both row swaps and h5 aggregation.
            return windows[:, -1, None, None] + np.log(np.arange(1, 6))[None, None, :] + np.zeros((len(windows), 9, 5))
        output, audit = moirai.extract_with_predictor(rv, origins, fake_predict,
                                                    np.arange(1, 10) / 10, batch_size=2)
        self.assertEqual(list(output.columns), list(moirai.COLUMNS))
        self.assertTrue(output.index.equals(origins.rename("origin")))
        np.testing.assert_allclose(output.iloc[:, 0], np.arange(600)[[511, 530, 590]] / 100)
        np.testing.assert_allclose(output.iloc[:, 1], np.arange(600)[[511, 530, 590]] / 100 + np.log(3))
        self.assertEqual(len(captured), 3)
        self.assertTrue((audit.context_end == audit.index).all())


if __name__ == "__main__":
    unittest.main()
