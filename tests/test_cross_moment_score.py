"""Prewritten contracts for frozen-forecast direct cross-moment scoring."""

from __future__ import annotations

import unittest
from decimal import Decimal, localcontext

import numpy as np
import pandas as pd

from src import cross_moment_score as score


def panel():
    rows = []
    origins = pd.to_datetime(["2016-01-04", "2016-01-05", "2020-01-02", "2020-01-03"])
    cutoffs = pd.to_datetime(["2015-12-31", "2016-01-04", "2019-12-31", "2020-01-02"])
    targets = pd.to_datetime(["2016-01-05", "2016-01-06", "2020-01-03", "2020-01-06"])
    for i, origin in enumerate(origins):
        for j, model in enumerate(
            ("constant_matrix", "constant_correlation", "dynamic_correlation")
        ):
            rows.append(
                dict(
                    origin=origin,
                    model=model,
                    horizon=1,
                    feature_cutoff_date=cutoffs[i],
                    target_end=targets[i],
                    available_date=targets[i],
                    y_qqq=[0.02, -0.01, 0.0, 0.005][i],
                    y_spx=[-0.01, 0.03, 0.005, 0.0][i],
                    mu_qqq=0.001,
                    mu_spx=-0.002,
                    h_qqq=0.0004 if j else 0.0003,
                    h_spx=0.0002 if j else 0.0005,
                    rho=[0.2, 0.4, -0.1][j],
                    loss=-10.0 + j,
                    fit_origin=origins[(i // 2) * 2],
                    fit_cutoff_date=cutoffs[(i // 2) * 2],
                    train_n=1200 + i // 2,
                    train_last_target=cutoffs[(i // 2) * 2],
                    train_last_available=cutoffs[(i // 2) * 2],
                    phase="development" if i < 2 else "evaluation",
                )
            )
    return pd.DataFrame(rows)


class ProductArithmeticTests(unittest.TestCase):
    def test_scalar_product_squared_error_identity_and_original_metadata_unchanged(self):
        original = panel()
        before = original.copy(deep=True)
        result = score.score_panel(original)
        pd.testing.assert_frame_equal(original, before)
        pd.testing.assert_frame_equal(result.loc[:, original.columns], original)
        self.assertEqual(
            list(result.columns[-3:]), ["realized_product", "forecast_product", "product_mse"]
        )
        for row in result.itertuples():
            y = (row.y_qqq - row.mu_qqq) * (row.y_spx - row.mu_spx)
            q = row.rho * np.sqrt(row.h_qqq) * np.sqrt(row.h_spx)
            self.assertAlmostEqual(row.realized_product, y, delta=1e-18)
            self.assertAlmostEqual(row.forecast_product, q, delta=1e-18)
            self.assertAlmostEqual(row.product_mse, (y - q) ** 2, delta=1e-20)

    def test_arbitrary_input_row_order_and_index_are_preserved(self):
        original = panel().sample(frac=1, random_state=6)
        original.index = np.arange(len(original)) * 3 + 7
        result = score.score_panel(original)
        pd.testing.assert_frame_equal(result.loc[:, original.columns], original)

    def test_assets_swap_and_extreme_representable_units_preserve_scores(self):
        original = panel()
        expected = score.score_panel(original)
        swapped = original.copy()
        for stem in ("y", "mu", "h"):
            swapped[[stem + "_qqq", stem + "_spx"]] = original[
                [stem + "_spx", stem + "_qqq"]
            ].to_numpy()
        np.testing.assert_allclose(
            score.score_panel(swapped).product_mse, expected.product_mse, rtol=2e-14
        )
        for units in ((100.0, 0.01), (3.0, 2.0), (1e150, 1e-150)):
            changed = original.copy()
            for asset, unit in zip(("qqq", "spx"), units, strict=True):
                changed["y_" + asset] *= unit
                changed["mu_" + asset] *= unit
                changed["h_" + asset] *= unit**2
            answer = score.score_panel(changed)
            np.testing.assert_allclose(
                answer.product_mse,
                expected.product_mse * (units[0] * units[1]) ** 2,
                rtol=2e-14,
            )

    def test_signed_and_exact_zero_realized_forecast_and_errors_are_valid(self):
        original = panel()
        original[["y_qqq", "y_spx", "mu_qqq", "mu_spx", "rho"]] = 0.0
        answer = score.score_panel(original)
        self.assertTrue(
            answer[["realized_product", "forecast_product", "product_mse"]]
            .eq(0)
            .all(axis=None)
        )
        signed = score.score_panel(panel())
        self.assertTrue((signed.realized_product < 0).any())
        self.assertTrue((signed.forecast_product < 0).any())

    def test_nonzero_realized_product_underflow_rejected_but_zero_operand_valid(self):
        original = panel()
        original[["mu_qqq", "mu_spx"]] = 0.0
        original[["y_qqq", "y_spx"]] = 1e-200
        with self.assertRaisesRegex(ValueError, "underflow|represent"):
            score.score_panel(original)
        original["y_qqq"] = 0.0
        self.assertTrue(score.score_panel(original).realized_product.eq(0).all())

    def test_forecast_product_and_squared_loss_underflow_rejected(self):
        original = panel()
        original[["y_qqq", "y_spx", "mu_qqq", "mu_spx"]] = 0.0
        original[["h_qqq", "h_spx"]] = 1e-200
        original.rho = 1e-200
        with self.assertRaisesRegex(ValueError, "underflow|represent"):
            score.score_panel(original)
        original.rho = 0.2
        with self.assertRaisesRegex(ValueError, "underflow|represent"):
            score.score_panel(original)

    def test_nonfinite_residual_product_and_loss_abort_without_subset(self):
        for values in ((1e308, -1e308), (1e200, 0.0), (1e100, 0.0)):
            original = panel()
            original[["y_qqq", "y_spx"]] = values[0]
            original[["mu_qqq", "mu_spx"]] = values[1]
            with self.assertRaises(ValueError):
                score.score_panel(original)

    def test_factored_gap_matches_direct_losses_and_high_precision_near_cancellation(self):
        qc = np.array([0.2, -0.3, 0.0])
        q0 = np.array([0.1, 0.4, 0.0])
        y = np.array([-0.1, 0.2, 0.0])
        np.testing.assert_allclose(
            score.paired_difference(qc, q0, y),
            (y - qc) ** 2 - (y - q0) ** 2,
            rtol=1e-14,
            atol=1e-16,
        )
        candidate = np.nextafter(1e-4, np.inf)
        control = 1e-4
        actual = 0.1
        with localcontext() as context:
            context.prec = 90
            a, b, c = map(Decimal.from_float, (candidate, control, actual))
            expected = float((a - c) ** 2 - (b - c) ** 2)
        observed = score.paired_difference(
            np.array([candidate]), np.array([control]), np.array([actual])
        )[0]
        self.assertNotEqual(observed, 0.0)
        self.assertAlmostEqual(observed / expected, 1.0, delta=1e-14)

    def test_equal_candidate_control_and_exact_error_cancellation_give_zero_gap(self):
        result = score.paired_difference(
            np.array([1e100, 0.1]), np.array([1e100, -0.1]), np.array([-1e100, 0.0])
        )
        np.testing.assert_array_equal(result, np.zeros(2))

    def test_factored_gap_rejects_nonfinite_misalignment_and_underflow(self):
        for q, q0, y in (
            ([1.0], [0.0, 1.0], [1.0]),
            ([np.nan], [0.0], [1.0]),
            ([1e-200], [0.0], [0.0]),
            ([1e308], [-1e308], [0.0]),
            ([1e308], [1e308], [-1e308]),
        ):
            with self.assertRaises(ValueError):
                score.paired_difference(np.array(q), np.array(q0), np.array(y))

    def test_paired_gap_units_change_by_fourth_power_return_factor(self):
        q, q0, y = np.array([0.02, -0.03]), np.array([-0.01, 0.04]), np.array([0.03, -0.02])
        baseline = score.paired_difference(q, q0, y)
        product_factor = 3.0 * 2.0
        changed = score.paired_difference(
            q * product_factor, q0 * product_factor, y * product_factor
        )
        np.testing.assert_allclose(changed, baseline * product_factor**2, rtol=1e-14)

    def test_coherence_allowance_does_not_overflow_on_finite_near_maximum_losses(self):
        difference = score.paired_difference(
            np.array([1e154]), np.array([-1e154]), np.array([0.0])
        )
        np.testing.assert_array_equal(difference, np.zeros(1))

    def test_nonzero_subnormal_factored_gap_is_retained_when_direct_gap_rounds_zero(self):
        candidate, control = 2e-162, 2.6e-162
        self.assertEqual(candidate**2, control**2)
        result = score.paired_difference(
            np.array([candidate]), np.array([control]), np.array([0.0])
        )
        self.assertEqual(result[0], -np.nextafter(0.0, 1.0))


class FrozenPanelContracts(unittest.TestCase):
    def test_complex_numeric_values_cannot_be_silently_cast_to_real(self):
        altered = panel()
        altered["y_qqq"] = altered.y_qqq.astype(complex) + 1j
        with self.assertRaises(ValueError):
            score.score_panel(altered)
        with self.assertRaises(ValueError):
            score.paired_difference(np.array([0.1 + 1j]), np.array([0.2]), np.array([0.3]))

    def test_exact_complete_schema_and_three_models_required(self):
        original = panel()
        variants = [
            original.drop(columns="loss"),
            original.assign(extra=1),
            original.iloc[:, ::-1],
            original.loc[original.model != "constant_matrix"],
            pd.concat([original, original.iloc[:1]], ignore_index=True),
        ]
        for altered in variants:
            with self.assertRaises(ValueError):
                score.score_panel(altered)

    def test_partial_model_cohort_and_target_mean_or_fit_mismatch_rejected(self):
        original = panel()
        with self.assertRaises(ValueError):
            score.score_panel(original.iloc[1:])
        for field, value in (
            ("y_qqq", 0.13),
            ("mu_spx", 0.2),
            ("train_n", 1210),
            ("fit_origin", pd.Timestamp("2016-01-05")),
        ):
            altered = original.copy()
            altered.loc[0, field] = value
            with self.assertRaises(ValueError):
                score.score_panel(altered)

    def test_conditional_diagonals_must_match_exactly(self):
        original = panel()
        original.loc[original.model == "dynamic_correlation", "h_qqq"] *= 1.0000000001
        with self.assertRaises(ValueError):
            score.score_panel(original)

    def test_date_fences_and_predictor_or_training_lookahead_rejected(self):
        for field, value in (
            ("feature_cutoff_date", pd.Timestamp("2016-01-04")),
            ("fit_cutoff_date", pd.Timestamp("2016-01-04")),
            ("train_last_available", pd.Timestamp("2016-01-04")),
            ("target_end", pd.Timestamp("2016-01-04")),
            ("available_date", pd.Timestamp("2025-11-03")),
            ("phase", "evaluation"),
            ("horizon", 2),
            ("train_n", 999),
        ):
            altered = panel()
            altered.loc[altered.origin.eq(pd.Timestamp("2016-01-04")), field] = value
            with self.assertRaises(ValueError):
                score.score_panel(altered)

    def test_phase_boundary_maturity_and_missing_or_intraday_timestamp_rejected(self):
        for mode in ("development_maturity", "missing", "intraday", "timezone"):
            altered = panel()
            if mode == "development_maturity":
                altered.loc[:2, ["target_end", "available_date"]] = pd.Timestamp("2020-01-02")
            elif mode == "missing":
                altered.loc[0, "fit_origin"] = pd.NaT
            elif mode == "intraday":
                altered.loc[0, "origin"] += pd.Timedelta(hours=1)
            else:
                altered["origin"] = altered.origin.dt.tz_localize("UTC")
            with self.assertRaises(ValueError):
                score.score_panel(altered)

    def test_numeric_domains_and_each_correlation_boundary(self):
        for field, value in (
            ("rho", np.nan),
            ("h_qqq", 0.0),
            ("h_spx", -1.0),
            ("y_qqq", np.inf),
            ("loss", np.nan),
            ("train_n", 1200.5),
        ):
            altered = panel()
            altered[field] = value
            with self.assertRaises(ValueError):
                score.score_panel(altered)
        for model, limit in (
            ("constant_matrix", 1 - 1e-6),
            ("constant_correlation", 0.995),
            ("dynamic_correlation", 0.995),
        ):
            altered = panel()
            altered.loc[altered.model == model, "rho"] = limit
            self.assertTrue(np.isfinite(score.score_panel(altered).product_mse).all())
            altered.loc[altered.model == model, "rho"] = np.nextafter(limit, np.inf)
            with self.assertRaises(ValueError):
                score.score_panel(altered)


if __name__ == "__main__":
    unittest.main()
