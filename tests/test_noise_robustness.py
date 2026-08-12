import pathlib
import unittest

import numpy as np
import pandas as pd

from src.noise_robustness import (
    MODEL_NAMES,
    aggregate_curves,
    array_sha256,
    bootstrap_pairwise,
    build_context_bank,
    cached_snapshot_path,
    fit_expanding_har,
    har_features,
    moving_block_indices,
    select_origins,
    verify_forecasts,
)
from src.representation_study import load_protocol, origin_noise_seed


class NoiseRobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()
        cls.noise = cls.protocol["noise_robustness"]

    def test_origin_seed_is_order_invariant(self):
        dates = pd.to_datetime(["2020-03-02", "2022-06-13", "2024-01-03"])
        forward = {d: origin_noise_seed(42, d) for d in dates}
        reverse = {d: origin_noise_seed(42, d) for d in dates[::-1]}
        self.assertEqual(forward, reverse)
        self.assertEqual(forward[dates[0]], origin_noise_seed(42, "2020-03-02"))

    def test_origins_follow_session_stride_and_require_next_target(self):
        index = pd.bdate_range("2015-12-01", periods=80)
        frame = pd.DataFrame({"log_rv": np.arange(80.0)}, index=index)
        cfg = {**self.noise, "sample_start": str(index[10].date()),
               "sample_end": str(index[-1].date()), "origin_stride_sessions": 20}
        origins = select_origins(frame, cfg)
        positions = [frame.index.get_loc(t) for t in origins]
        self.assertEqual(positions, [10, 30, 50, 70])
        self.assertTrue(all(p + 1 < len(frame) for p in positions))

    def test_context_bank_has_nested_common_corruptions(self):
        index = pd.bdate_range("2010-01-01", periods=1100)
        values = np.sin(np.arange(1100) / 19) - 9
        frame = pd.DataFrame({"log_rv": values}, index=index)
        origin = index[-2]
        cfg = {**self.noise, "context_length": 1024}
        bank, metadata = build_context_bank(frame, pd.DatetimeIndex([origin]), cfg)
        clean_g = bank[("gaussian", 0.0)][origin]
        clean_i = bank[("impulse", 0.0)][origin]
        np.testing.assert_array_equal(clean_g, clean_i)
        self.assertEqual(metadata.loc[origin, "target_date"], index[-1])
        sigma = np.std(clean_g)
        mask_05 = np.abs(bank[("impulse", .05)][origin] - clean_g) > sigma
        mask_10 = np.abs(bank[("impulse", .10)][origin] - clean_g) > sigma
        mask_20 = np.abs(bank[("impulse", .20)][origin] - clean_g) > sigma
        self.assertTrue(np.all(mask_05 <= mask_10))
        self.assertTrue(np.all(mask_10 <= mask_20))
        np.testing.assert_allclose(
            bank[("gaussian", .4)][origin] - clean_g,
            2 * (bank[("gaussian", .2)][origin] - clean_g),
        )

    def test_har_feature_is_log_of_mean_variance(self):
        x = np.linspace(-11, -7, 30)
        got = har_features(x)
        expected = [x[-1], np.log(np.exp(x[-5:]).mean()), np.log(np.exp(x[-22:]).mean())]
        np.testing.assert_allclose(got, expected)

    def test_expanding_har_does_not_consume_future_response(self):
        index = pd.bdate_range("2000-01-01", periods=180)
        y = pd.Series(-9 + np.sin(np.arange(180) / 13), index=index)
        origin = index[150]
        beta_a, residual_a = fit_expanding_har(y, origin)
        modified = y.copy()
        modified.iloc[151:] += 1000
        beta_b, residual_b = fit_expanding_har(modified, origin)
        np.testing.assert_array_equal(beta_a, beta_b)
        np.testing.assert_array_equal(residual_a, residual_b)
        changed_origin = y.copy()
        changed_origin.iloc[150] += 10
        beta_c, _ = fit_expanding_har(changed_origin, origin)
        self.assertFalse(np.array_equal(beta_a, beta_c))

    def test_context_hash_is_dtype_stable_and_value_sensitive(self):
        x = np.array([1, 2, 3], dtype=np.float32)
        self.assertEqual(array_sha256(x), array_sha256(x.astype(np.float64)))
        self.assertNotEqual(array_sha256(x), array_sha256(x + .1))

    def test_cached_snapshots_resolve_required_executable_files(self):
        # This is an environment prerequisite check, not model inference.
        chronos = cached_snapshot_path(
            "amazon/chronos-2", "29ec3766d36d6f73f0696f85560a422f50e8498c",
            ("config.json",),
        )
        tirex = cached_snapshot_path(
            "NX-AI/TiRex-2", "05e5b26db52bfb256f1ae1bdf785589850482de3",
            ("model-config.yaml", "model.ckpt"),
        )
        self.assertTrue((chronos / "config.json").exists())
        self.assertTrue((tirex / "model.ckpt").exists())

    def _toy_forecasts(self):
        rows = []
        taus = self.noise["quantiles"]
        origins = pd.bdate_range("2020-01-02", periods=12)
        for model_n, model in enumerate(MODEL_NAMES):
            for corruption, levels in (("gaussian", [0., .2, .4, .6, .8]),
                                       ("impulse", [0., .05, .1, .15, .2])):
                for intensity in levels:
                    for i, origin in enumerate(origins):
                        clean_loss = 1 + i / 100 + model_n / 10
                        loss = clean_loss * (1 + intensity * (model_n + 1))
                        q = {f"q{tau:.2f}": -10 + tau for tau in taus}
                        rows.append({
                            "origin": origin, "target_date": origin + pd.offsets.BDay(),
                            "corruption": corruption, "intensity": intensity,
                            "model": model, "actual": -9.5,
                            "context_sha256": (f"clean-{i}" if intensity == 0
                                               else f"{corruption}-{intensity}-{i}"),
                            "crps": loss, **q,
                        })
        return pd.DataFrame(rows)

    def test_relative_crps_uses_ratio_of_aggregate_losses(self):
        curves = aggregate_curves(self._toy_forecasts())
        row = curves[(curves.model == MODEL_NAMES[1]) &
                     (curves.corruption == "gaussian") &
                     (curves.intensity == .4)].iloc[0]
        self.assertAlmostEqual(row.relative_crps, 1.8)
        zero = curves[curves.intensity == 0]
        np.testing.assert_allclose(zero.relative_crps, 1.0)

    def test_score_is_explicitly_a_decile_grid_crps_approximation(self):
        from src import metrics
        self.assertIn("Approx CRPS", metrics.crps_from_quantiles.__doc__)
        report = pathlib.Path(
            "reports/representation_study/noise_robustness.md"
        ).read_text()
        self.assertIn("decile-grid CRPS approximation", report)

    def test_moving_block_indices_are_reproducible_and_contiguous(self):
        a = moving_block_indices(12, 2, 20, 420042)
        b = moving_block_indices(12, 2, 20, 420042)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (20, 12))
        self.assertTrue(np.all(a[:, 1::2] - a[:, 0::2] == 1))

    def test_bootstrap_is_paired_and_complete(self):
        inference = {"block_sampled_origins": 2, "bootstrap_draws": 100,
                     "seed": 420042, "confidence_level": .95}
        result = bootstrap_pairwise(self._toy_forecasts(), inference)
        self.assertEqual(len(result), 24)
        self.assertFalse(result.isna().any().any())
        # Model 2 degrades faster than model 1, hence A-B is negative.
        row = result[(result.corruption == "gaussian") &
                     (result.intensity == .4) &
                     (result.model_a == MODEL_NAMES[0]) &
                     (result.model_b == MODEL_NAMES[1])].iloc[0]
        self.assertAlmostEqual(row.relative_crps_difference_a_minus_b, -.4)

    def test_verifier_rejects_clean_window_and_accepts_common_grid(self):
        frame = self._toy_forecasts()
        verify_forecasts(frame, self.protocol)
        bad = frame.copy()
        bad.loc[0, "origin"] = pd.Timestamp(self.protocol["source"]["clean_start"])
        with self.assertRaisesRegex(ValueError, "clean-window"):
            verify_forecasts(bad, self.protocol)

    def test_verifier_rejects_model_specific_context(self):
        frame = self._toy_forecasts()
        mask = ((frame.model == MODEL_NAMES[0]) & (frame.corruption == "gaussian") &
                (frame.intensity == .2))
        frame.loc[mask, "context_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "common corrupted"):
            verify_forecasts(frame, self.protocol)


if __name__ == "__main__":
    unittest.main()
