"""Prewritten synthetic contracts for full-calendar masked paired inference."""

import math
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import norm

from src.treasury_dealer_inference import masked_mean_inference


def literal_oracle(differences, mask, *, block, lag, draws, seed):
    """Independent explicit calendar-index resampling; no block-sum shortcut."""
    values = list(differences)
    selected = list(mask)
    total = len(values)
    count = sum(selected)
    mean = math.fsum(v for v, keep in zip(values, selected) if keep) / count
    proportion = count / total
    influence = [
        (value - mean) / proportion if keep else 0.0 for value, keep in zip(values, selected)
    ]
    effective_lag = min(lag, total - 1)
    variance = math.fsum(value * value for value in influence) / total
    for distance in range(1, effective_lag + 1):
        covariance = (
            math.fsum(influence[i] * influence[i - distance] for i in range(distance, total))
            / total
        )
        variance += 2 * (1 - distance / (effective_lag + 1)) * covariance
    if variance < 0:
        raise ValueError("Negative computed HAC variance in literal oracle")
    se = math.sqrt(variance / total)
    hac_p = float(2 * norm.sf(abs(mean) / se)) if se else float(mean == 0)
    rng = np.random.default_rng(seed + block)
    complete, remainder = divmod(total, block)
    starts = rng.integers(0, total, size=(draws, complete))
    tails = rng.integers(0, total, size=draws) if remainder else None
    means = []
    for draw in range(draws):
        positions = []
        for start in starts[draw]:
            positions.extend((int(start) + j) % total for j in range(block))
        if remainder:
            positions.extend((int(tails[draw]) + j) % total for j in range(remainder))
        assert len(positions) == total
        eligible = [i for i in positions if selected[i]]
        if not eligible:
            raise ValueError("zero-support bootstrap replicate")
        means.append(math.fsum(values[i] for i in eligible) / len(eligible))
    means = np.asarray(means)
    p = (1 + np.count_nonzero(np.abs(means - mean) >= abs(mean))) / (draws + 1)
    return {
        "mean": mean,
        "se": se,
        "hac_p": hac_p,
        "p": float(p),
        "ci95": np.quantile(means, [0.025, 0.975]),
    }


class TreasuryDealerInferenceTests(unittest.TestCase):
    def assert_oracle(self, result, oracle, block):
        self.assertAlmostEqual(result["mean"], oracle["mean"], places=13)
        self.assertAlmostEqual(result["hac"]["se"], oracle["se"], places=13)
        self.assertAlmostEqual(result["hac"]["p"], oracle["hac_p"], places=13)
        self.assertEqual(result["block_inference"][str(block)]["p"], oracle["p"])
        np.testing.assert_allclose(
            result["block_inference"][str(block)]["ci95"],
            oracle["ci95"],
            rtol=1e-12,
            atol=1e-13,
        )

    def test_masked_remainder_matches_independent_literal_oracle(self):
        values = np.array([2.0, -1.0, np.nan, 3.0, 0.0, -2.0, 1.0, 4.0, np.nan, -3.0, 1.0])
        mask = ~np.isnan(values)
        result = masked_mean_inference(
            values, mask, blocks=(3, 5), hac_lags=4, draws=257, seed=810
        )
        self.assertEqual(
            set(result),
            {"mean", "n", "full_calendar_n", "hac", "block_inference", "p_conservative"},
        )
        self.assertEqual(set(result["hac"]), {"se", "p", "ci95", "mde80_nominal"})
        self.assertEqual((result["n"], result["full_calendar_n"]), (9, 11))
        for block in (3, 5):
            oracle = literal_oracle(values, mask, block=block, lag=4, draws=257, seed=810)
            self.assert_oracle(result, oracle, block)
        self.assertEqual(
            result["p_conservative"],
            max(result["hac"]["p"], *(r["p"] for r in result["block_inference"].values())),
        )

    def test_multiple_draw_chunks_keep_full_then_tail_rng_order(self):
        values = np.arange(19, dtype=float) - 8.0
        mask = np.ones(19, dtype=bool)
        mask[[2, 7, 13]] = False
        result = masked_mean_inference(
            values, mask, blocks=(4,), hac_lags=2, draws=2057, seed=903
        )
        oracle = literal_oracle(values, mask, block=4, lag=2, draws=2057, seed=903)
        self.assert_oracle(result, oracle, 4)

    def test_all_observed_reduces_to_ordinary_calendar_mean_and_hac(self):
        values = np.array([1.0, -2.0, 4.0, 2.0, -1.0, 3.0, 0.0, -3.0])
        mask = np.ones(8, dtype=bool)
        result = masked_mean_inference(
            values, mask, blocks=(2,), hac_lags=3, draws=101, seed=50
        )
        oracle = literal_oracle(values, mask, block=2, lag=3, draws=101, seed=50)
        self.assert_oracle(result, oracle, 2)
        expected_ci = np.array(
            [oracle["mean"] - 1.96 * oracle["se"], oracle["mean"] + 1.96 * oracle["se"]]
        )
        np.testing.assert_allclose(result["hac"]["ci95"], expected_ci)
        self.assertAlmostEqual(
            result["hac"]["mde80_nominal"],
            float((norm.ppf(0.975) + norm.ppf(0.8)) * oracle["se"]),
        )

    def test_calendar_gaps_change_hac_without_changing_selected_sequence(self):
        adjacent_mask = np.zeros(12, dtype=bool)
        spaced_mask = np.zeros(12, dtype=bool)
        adjacent_mask[:4] = True
        spaced_mask[[0, 3, 6, 9]] = True
        adjacent = np.full(12, np.nan)
        spaced = np.full(12, np.nan)
        adjacent[adjacent_mask] = spaced[spaced_mask] = [2.0, 1.0, -1.0, -2.0]
        first = masked_mean_inference(
            adjacent, adjacent_mask, blocks=(12,), hac_lags=1, draws=13, seed=8
        )
        second = masked_mean_inference(
            spaced, spaced_mask, blocks=(12,), hac_lags=1, draws=13, seed=8
        )
        self.assertEqual(first["mean"], second["mean"])
        self.assertNotAlmostEqual(first["hac"]["se"], second["hac"]["se"])
        for values, mask, result in (
            (adjacent, adjacent_mask, first),
            (spaced, spaced_mask, second),
        ):
            oracle = literal_oracle(values, mask, block=12, lag=1, draws=13, seed=8)
            self.assert_oracle(result, oracle, 12)

    def test_false_mask_nan_and_large_numbers_never_enter_arithmetic(self):
        mask = np.array([True, True, False, True, True, False, True, True])
        values = np.array([1.0, -1.0, np.nan, 2.0, -2.0, np.nan, 3.0, 0.0])
        garbage = values.copy()
        garbage[~mask] = 1e300
        kwargs = {"blocks": (4,), "hac_lags": 2, "draws": 43, "seed": 33}
        self.assertEqual(
            masked_mean_inference(values, mask, **kwargs),
            masked_mean_inference(garbage, mask, **kwargs),
        )

    def test_zero_support_replicate_rejects_entire_call(self):
        values = np.arange(8, dtype=float)
        mask = np.array([True, True, False, False, False, False, False, False])
        with self.assertRaisesRegex(ValueError, "zero.support"):
            literal_oracle(values, mask, block=1, lag=1, draws=100, seed=11)
        with self.assertRaisesRegex(ValueError, "zero.support"):
            masked_mean_inference(values, mask, blocks=(8, 1), hac_lags=1, draws=100, seed=11)

    def test_inputs_unchanged_after_success_or_failure(self):
        values = np.array([1.0, -1.0, np.nan, 2.0, -2.0, 3.0])
        mask = np.array([True, True, False, True, True, True])
        before_values, before_mask = values.copy(), mask.copy()
        masked_mean_inference(values, mask, blocks=(3,), hac_lags=1, draws=17, seed=4)
        np.testing.assert_array_equal(values, before_values)
        np.testing.assert_array_equal(mask, before_mask)
        with self.assertRaises(ValueError):
            masked_mean_inference(values, mask, blocks=(0,), draws=17)
        np.testing.assert_array_equal(values, before_values)
        np.testing.assert_array_equal(mask, before_mask)

    def test_constant_zero_and_nonzero_have_declared_zero_variance_limits(self):
        mask = np.ones(8, dtype=bool)
        zero = masked_mean_inference(
            np.zeros(8), mask, blocks=(2,), hac_lags=99, draws=19, seed=5
        )
        self.assertEqual(zero["hac"]["se"], 0.0)
        self.assertEqual(zero["hac"]["p"], 1.0)
        self.assertEqual(zero["block_inference"]["2"], {"p": 1.0, "ci95": [0.0, 0.0]})
        nonzero = masked_mean_inference(
            np.full(8, 2.0), mask, blocks=(2,), hac_lags=99, draws=19, seed=5
        )
        self.assertEqual(nonzero["hac"]["se"], 0.0)
        self.assertEqual(nonzero["hac"]["p"], 0.0)
        self.assertEqual(nonzero["block_inference"]["2"], {"p": 0.05, "ci95": [2.0, 2.0]})

    def test_rejects_invalid_data_mask_and_support(self):
        good = np.array([1.0, 2.0, 3.0, 4.0])
        mask = np.ones(4, dtype=bool)
        cases = [
            (good, [1, 1, 1, 1]),
            (good, np.ones(3, dtype=bool)),
            (good.reshape(2, 2), mask),
            (good, mask.reshape(2, 2)),
            (good.astype(str), mask),
            (np.array([True, False, True, False]), mask),
            (good.astype(complex), mask),
            (good, np.zeros(4, dtype=bool)),
            (good, np.array([True, False, False, False])),
            (np.array([1.0, np.nan, 3.0, 4.0]), mask),
            (np.array([1.0, 2.0, np.inf, 4.0]), np.array([True, True, False, True])),
        ]
        for values, chosen in cases:
            with self.subTest(values=values, mask=chosen), self.assertRaises(ValueError):
                masked_mean_inference(values, chosen, blocks=(2,), draws=3)

    def test_rejects_invalid_bootstrap_and_hac_parameters(self):
        values, mask = np.arange(4, dtype=float), np.ones(4, dtype=bool)
        cases = [
            {"blocks": ()},
            {"blocks": (2, 2)},
            {"blocks": (0,)},
            {"blocks": (5,)},
            {"blocks": (True,)},
            {"blocks": (2.0,)},
            {"hac_lags": -1},
            {"hac_lags": True},
            {"hac_lags": 1.5},
            {"draws": 0},
            {"draws": True},
            {"draws": 1.5},
            {"seed": -1},
            {"seed": True},
            {"seed": 1.5},
        ]
        for changes in cases:
            kwargs = {"blocks": (2,), "draws": 3, **changes}
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                masked_mean_inference(values, mask, **kwargs)

    def test_finite_inputs_with_overflowing_influence_variance_fail(self):
        values = np.array([1e308, -1e308, 1e308, -1e308])
        with self.assertRaisesRegex(ValueError, "finite|overflow"):
            masked_mean_inference(values, np.ones(4, dtype=bool), blocks=(2,), draws=3)

    def test_negative_computed_hac_variance_is_rejected(self):
        # Inject finite covariance sums to exercise a numerical failure path.
        # A negative computed Bartlett variance must not become a zero SE.
        with (
            patch("src.treasury_dealer_inference.np.dot", side_effect=[1.0, -3.0]),
            self.assertRaisesRegex(ValueError, "[Nn]egative.*HAC"),
        ):
            masked_mean_inference(
                np.array([1.0, -1.0, 1.0, -1.0]),
                np.ones(4, dtype=bool),
                blocks=(2,),
                hac_lags=1,
                draws=3,
            )


if __name__ == "__main__":
    unittest.main()
