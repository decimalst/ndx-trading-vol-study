"""Prewritten independent timing and latent-geometry contracts for wave two."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src import verify_international_volatility as verify


def foreign_fixture():
    rng = np.random.default_rng(71932)
    dates = pd.bdate_range("2008-10-01", periods=160)
    foreign = {symbol: pd.Series(np.exp(rng.normal(-8 + i * .1, .2, len(dates))), index=dates)
               for i, symbol in enumerate(verify.SYMBOLS)}
    return dates, foreign


class ForeignTimingTests(unittest.TestCase):
    def test_local_windows_are_computed_before_alignment_and_skip_only_absent_local_sessions(self):
        us, foreign = foreign_fixture()
        symbol = verify.SYMBOLS[0]
        foreign[symbol] = foreign[symbol].drop(us[70])
        values, _ = verify.foreign_design(foreign, us)
        query = us[74]
        local = foreign[symbol].loc[:us[73]].iloc[-5:]
        self.assertAlmostEqual(values.loc[query, f"{symbol}_5"], np.log(local.mean()), places=13)
        self.assertEqual(len(local), 5)
        self.assertNotIn(us[70], local.index)

    def test_latest_same_us_day_observation_is_unavailable_and_future_mutation_is_inert(self):
        us, foreign = foreign_fixture()
        values, audit = verify.foreign_design(foreign, us)
        query = us[80]
        changed = {symbol: data.copy() for symbol, data in foreign.items()}
        for data in changed.values():
            data.loc[query:] *= 1e6
        updated, _ = verify.foreign_design(changed, us)
        np.testing.assert_allclose(values.loc[query], updated.loc[query], rtol=0, atol=0)
        for symbol in verify.SYMBOLS:
            self.assertEqual(audit[symbol].loc[query, "source_date"], us[79])
            self.assertEqual(audit[symbol].loc[query, "extra_age"], 0)

    def test_staleness_allows_three_extra_us_sessions_and_rejects_four(self):
        us, foreign = foreign_fixture()
        end = us[70]
        truncated = {symbol: data.loc[:end] for symbol, data in foreign.items()}
        values, audit = verify.foreign_design(truncated, us)
        symbol = verify.SYMBOLS[0]
        for delta in range(1, 5):
            self.assertEqual(audit[symbol].loc[us[70 + delta], "extra_age"], delta - 1)
            self.assertTrue(np.isfinite(values.loc[us[70 + delta], f"{symbol}_1"]))
        self.assertEqual(audit[symbol].loc[us[75], "extra_age"], 4)
        self.assertTrue(np.isnan(values.loc[us[75], f"{symbol}_1"]))
        self.assertEqual(audit[symbol].loc[us[75], "source_date"], end)

    def test_invalid_local_observation_is_retained_as_missing_not_deleted(self):
        us, foreign = foreign_fixture()
        symbol = verify.SYMBOLS[0]
        foreign[symbol].loc[us[70]] = np.nan
        values, audit = verify.foreign_design(foreign, us)
        self.assertEqual(audit[symbol].loc[us[71], "source_date"], us[70])
        self.assertTrue(np.isnan(values.loc[us[71], f"{symbol}_1"]))
        self.assertTrue(np.isnan(values.loc[us[74], f"{symbol}_5"]))

    def test_regions_are_equal_means_of_three_log_features_with_no_skipping(self):
        us, foreign = foreign_fixture()
        values, _ = verify.foreign_design(foreign, us)
        regional = verify.regional_design(values)
        for region, symbols in (("asia", verify.SYMBOLS[:3]), ("europe", verify.SYMBOLS[3:])):
            for span in (1, 5, 22):
                expected = values[[f"{symbol}_{span}" for symbol in symbols]].sum(axis=1, min_count=3) / 3
                np.testing.assert_allclose(regional[f"{region}_{span}"], expected, equal_nan=True)
        values.loc[us[100], f"{verify.SYMBOLS[0]}_1"] = np.nan
        self.assertTrue(np.isnan(verify.regional_design(values).loc[us[100], "asia_1"]))


class LatentGeometryTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(90372)
        raw = rng.normal(size=(300, 3)) @ rng.normal(size=(3, 18)) + rng.normal(0, .1, (300, 18))
        return raw[:280], raw[280:]

    def test_covariance_eigen_reconstruction_matches_svd_projector(self):
        train, query = self.fixture()
        projected, applied, audit = verify.pca_design(train, query)
        centered = (train - train.mean(axis=0)) / train.std(axis=0)
        _, singular, right = np.linalg.svd(centered, full_matrices=False)
        expected = right[:3].T @ right[:3]
        components = np.asarray(audit["components"])
        np.testing.assert_allclose(components.T @ components, expected, rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(projected @ projected.T, centered @ expected @ centered.T, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(applied, ((query - train.mean(axis=0)) / train.std(axis=0)) @ components.T, rtol=1e-12)
        for component in components:
            self.assertGreater(component[np.argmax(np.abs(component))], 0)
        self.assertGreater(singular[2] ** 2 - singular[3] ** 2, 1e-8 * singular[0] ** 2)

    def test_pca_mapping_uses_no_query_distribution(self):
        train, query = self.fixture()
        _, applied, audit = verify.pca_design(train, query)
        changed = query.copy()
        changed[1:] += 100
        _, altered, revised = verify.pca_design(train, changed)
        for key in audit:
            np.testing.assert_allclose(audit[key], revised[key], rtol=0, atol=0)
        np.testing.assert_allclose(applied[0], altered[0], rtol=0, atol=0)

    def test_eigen_boundary_tie_and_zero_scale_are_rejected(self):
        symmetric = np.vstack([np.eye(18), -np.eye(18)])
        with self.assertRaisesRegex(ValueError, "eigengap"):
            verify.pca_design(symmetric, symmetric[:2])
        train, query = self.fixture()
        train[:, 0] = 1
        with self.assertRaisesRegex(ValueError, "scale"):
            verify.pca_design(train, query)

    def test_cumulative_holm_keeps_the_full72_denominator(self):
        adjusted = verify.holm([.0001] + [1.] * 71)
        self.assertAlmostEqual(adjusted[0], .0072)
        self.assertEqual(len(adjusted), 72)


if __name__ == "__main__":
    unittest.main()
