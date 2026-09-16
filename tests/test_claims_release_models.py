"""Generated fixed-design OLS tests; no historical cohorts or model runs."""

import copy
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from src import claims_release_models as models

MARKET = (
    "const",
    "lrv_d",
    "lrv_w",
    "lrv_m",
    "lev_d",
    "lev_w",
    "lev_m",
    "liv",
    "lvix",
    "term",
    "xasset_stress",
    "market_stress",
)
MATCHED = MARKET + (
    "claim_m4",
    "claim_age",
    "entry_dow_1",
    "entry_dow_2",
    "entry_dow_3",
    "entry_dow_4",
)
ALL = MATCHED + ("claim_x",)


def fixture(n=100, application_n=13):
    rng = np.random.default_rng(197321)
    train = pd.DataFrame(
        rng.normal(size=(n, len(ALL))),
        columns=ALL,
        index=pd.date_range("2018-01-01", periods=n, freq="B"),
    )
    apply = pd.DataFrame(
        rng.normal(size=(application_n, len(ALL))),
        columns=ALL,
        index=pd.date_range("2019-01-01", periods=application_n, freq="B"),
    )
    for frame in (train, apply):
        frame["const"] = 1.0
        frame["claim_age"] = rng.integers(1, 8, len(frame)) / 7
        for day in range(1, 5):
            frame[f"entry_dow_{day}"] = (frame.index.weekday == day).astype(float)
    logy = (
        -7
        + 0.4 * train.lrv_d
        + 0.7 * train.claim_m4
        - 0.6 * train.claim_x
        + rng.normal(scale=0.3, size=n)
    )
    return train, pd.Series(np.exp(logy), index=train.index, name="target"), apply


def oracle(train, y, apply, columns):
    x = train.loc[:, list(columns)].to_numpy(float)
    q = apply.loc[:, list(columns)].to_numpy(float)
    # Independent raw-design QR solution, not the producer's standardized SVD.
    orthogonal, triangular = np.linalg.qr(x, mode="reduced")
    beta = np.linalg.solve(triangular, orthogonal.T @ np.log(np.asarray(y)))
    residuals = np.log(np.asarray(y)) - x @ beta
    log_smear = float(logsumexp(residuals) - math.log(len(y)))
    return np.exp(q @ beta + log_smear), log_smear


class ClaimsReleaseModelTests(unittest.TestCase):
    def test_fixed_constants_and_three_independent_ols_smearing_oracles(self):
        self.assertEqual(models.MARKET, MARKET)
        self.assertEqual(models.MATCHED, MATCHED)
        self.assertEqual(models.ALL, ALL)
        train, y, apply = fixture()
        result = models.fit_models(train, y, apply)
        self.assertEqual(set(result), {"predictions", "fits"})
        self.assertEqual(set(result["predictions"]), {"market", "matched", "candidate"})
        for arm, columns in [("market", MARKET), ("matched", MATCHED), ("candidate", ALL)]:
            expected, log_smear = oracle(train, y, apply, columns)
            np.testing.assert_allclose(
                result["predictions"][arm], expected, rtol=1e-11, atol=1e-15
            )
            audit = result["fits"][arm]
            self.assertEqual(audit["feature_names"], list(columns))
            self.assertEqual(audit["rank"], len(columns))
            self.assertEqual(audit["n_train"], len(y))
            self.assertEqual(audit["n_application"], len(apply))
            self.assertAlmostEqual(audit["log_smearing"], log_smear, places=12)
            self.assertLess(audit["normal_equation_max_abs"], 1e-11)
        self.assertNotAlmostEqual(
            result["fits"]["market"]["log_smearing"],
            result["fits"]["candidate"]["log_smearing"],
        )

    def test_saved_named_audit_reconstructs_every_prediction(self):
        train, y, apply = fixture()
        result = models.fit_models(train, y, apply)
        for arm, fit in result["fits"].items():
            names = fit["feature_names"]
            design = apply.loc[:, names].to_numpy(float)
            for j, name in enumerate(names[1:], 1):
                design[:, j] = (design[:, j] - fit["feature_means"][name]) / fit[
                    "feature_scales"
                ][name]
            beta = np.array([fit["coefficients"][name] for name in names])
            expected = np.exp(design @ beta + fit["log_smearing"])
            np.testing.assert_allclose(expected, result["predictions"][arm], rtol=1e-13)
            self.assertAlmostEqual(
                fit["feature_means"].get("claim_age", 0),
                train.claim_age.mean() if "claim_age" in names else 0,
                places=14,
            )

    def test_query_values_do_not_influence_fits_or_scaling(self):
        train, y, apply = fixture()
        before = models.fit_models(train, y, apply)
        changed = apply.copy()
        changed["lrv_d"] += 1
        after = models.fit_models(train, y, changed)
        self.assertEqual(before["fits"], after["fits"])
        self.assertFalse(
            np.array_equal(
                before["predictions"]["candidate"], after["predictions"]["candidate"]
            )
        )

    def test_target_units_and_feature_affine_units_transport(self):
        train, y, apply = fixture()
        first = models.fit_models(train, y, apply)
        scaled_y = models.fit_models(train, y * 10000, apply)
        scaled_train, scaled_apply = train.copy(), apply.copy()
        for name in ("lrv_w", "claim_m4", "claim_x"):
            scaled_train[name] = 23 * scaled_train[name] - 51
            scaled_apply[name] = 23 * scaled_apply[name] - 51
        scaled_x = models.fit_models(scaled_train, y, scaled_apply)
        for arm in ("market", "matched", "candidate"):
            np.testing.assert_allclose(
                scaled_y["predictions"][arm], 10000 * first["predictions"][arm], rtol=1e-11
            )
            np.testing.assert_allclose(
                scaled_x["predictions"][arm], first["predictions"][arm], rtol=1e-11
            )

    def test_column_order_and_opaque_provenance_do_not_change_model(self):
        train, y, apply = fixture()
        expected = models.fit_models(train, y, apply)
        train2, apply2 = train.iloc[:, ::-1].copy(), apply.iloc[:, ::-1].copy()
        train2["source_comparison"] = [{"never_convert": object()} for _ in range(len(train2))]
        apply2["claim_status"] = "available"
        result = models.fit_models(train2, y, apply2)
        self.assertEqual(result["fits"], expected["fits"])
        for arm in result["predictions"]:
            np.testing.assert_array_equal(
                result["predictions"][arm], expected["predictions"][arm]
            )

    def test_inputs_are_not_modified_and_empty_application_is_retained(self):
        train, y, apply = fixture()
        originals = train.copy(deep=True), y.copy(deep=True), apply.copy(deep=True)
        models.fit_models(train, y, apply)
        pd.testing.assert_frame_equal(train, originals[0])
        pd.testing.assert_series_equal(y, originals[1])
        pd.testing.assert_frame_equal(apply, originals[2])
        result = models.fit_models(train, y, apply.iloc[:0])
        self.assertTrue(all(value.shape == (0,) for value in result["predictions"].values()))

    def test_all_arm_validation_precedes_any_fit(self):
        train, y, apply = fixture()
        for where in ("train", "apply"):
            bad_train, bad_apply = train.copy(), apply.copy()
            frame = bad_train if where == "train" else bad_apply
            frame.loc[frame.index[-1], "claim_x"] = np.nan
            with patch.object(
                models, "_fit_arm", side_effect=AssertionError("early fit")
            ) as fit:
                with self.subTest(where=where), self.assertRaises(ValueError):
                    models.fit_models(bad_train, y, bad_apply)
                fit.assert_not_called()

    def test_missing_duplicate_non_numeric_boolean_and_nonfinite_columns_fail(self):
        train, y, apply = fixture()
        bad_frames = [
            train.drop(columns="claim_x"),
            pd.concat([train, train[["claim_x"]]], axis=1),
        ]
        for value in (np.inf, -np.inf, "3", True, 1 + 2j):
            bad = train.copy()
            bad["claim_x"] = value
            bad_frames.append(bad)
        for bad in bad_frames:
            with self.subTest(columns=list(bad.columns)), self.assertRaises(ValueError):
                models.fit_models(bad, y, apply)
        bad = train.copy()
        bad["const"] = 2
        with self.assertRaises(ValueError):
            models.fit_models(bad, y, apply)

    def test_targets_require_positive_finite_numeric_one_dimensional_aligned_values(self):
        train, y, apply = fixture()
        targets = [
            y.iloc[::-1],
            y.iloc[:-1],
            np.zeros(len(y)),
            np.full(len(y), np.inf),
            np.ones((len(y), 1)),
            np.ones(len(y), dtype=bool),
            np.array(["1"] * len(y)),
        ]
        for target in targets:
            with self.subTest(type=type(target)), self.assertRaises(ValueError):
                models.fit_models(train, target, apply)

    def test_zero_scale_and_relative_rank_failure_never_drop_a_feature(self):
        train, y, apply = fixture()
        for replacement in (
            np.ones(len(train)),
            2 * train.lrv_d,
            train.lrv_d + 1e-14 * train.lrv_w,
        ):
            bad = train.copy()
            bad["claim_x"] = replacement
            with self.assertRaises(ValueError):
                models.fit_models(bad, y, apply)

    def test_stable_log_smearing_handles_extreme_residuals(self):
        self.assertAlmostEqual(
            models._log_mean_exp(np.array([-1000.0, 1000.0])), 1000 - math.log(2), places=12
        )

    def test_prediction_overflow_or_underflow_aborts_instead_of_clipping(self):
        train, y, apply = fixture()
        for sign in (-1, 1):
            bad = apply.copy()
            bad["lrv_d"] = sign * 1e100
            with self.subTest(sign=sign), self.assertRaises(ValueError):
                models.fit_models(train, y, bad)


if __name__ == "__main__":
    unittest.main()
