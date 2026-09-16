"""Prewritten generated fixed32-column OLS and independent QR contracts."""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.treasury_dealer_models import ALL, MARKET, MATCHED, fit_models

EXPECTED_MARKET = (
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
EXPECTED_MATCHED = EXPECTED_MARKET + (
    "tlt_ret",
    "tlt_r2",
    "tlt_lrv5",
    "tlt_lrv22",
    "weekday_1",
    "weekday_2",
    "weekday_3",
    "weekday_4",
    "auction_count",
    "tenor_3",
    "tenor_5",
    "tenor_7",
    "tenor_10",
    "tenor_30",
    "reopening_count",
    "log_offering_sum",
    "high_yield_sum",
    "bid_to_cover_sum",
    "prior_share_mean_sum",
)
EXPECTED_ALL = EXPECTED_MATCHED + ("dealer_surprise",)


def fixture():
    rng = np.random.default_rng(246732)
    index = pd.bdate_range("2010-01-04", periods=200)
    raw = rng.normal(size=(200, 32))
    raw[:, 0] = 1
    for ret, squared in ((12, 13),):
        raw[:, squared] = raw[:, ret] ** 2
    train = pd.DataFrame(raw, index=index, columns=EXPECTED_ALL)
    query = pd.DataFrame(rng.normal(size=(11, 32)), columns=EXPECTED_ALL)
    query["const"] = 1.0
    for ret, squared in (("tlt_ret", "tlt_r2"),):
        query[squared] = query[ret] ** 2
    y = pd.Series(
        np.exp(
            -6
            + 0.3 * train.lrv_d
            + 0.1 * train.dealer_surprise
            + rng.normal(0, 0.2, len(train))
        ),
        index=index,
    )
    return train, y, query


def qr_oracle(train, y, query, names):
    x, a = train.loc[:, names].to_numpy(), query.loc[:, names].to_numpy()
    q, r = np.linalg.qr(x, mode="reduced")
    beta = np.linalg.solve(r, q.T @ np.log(y.to_numpy()))
    residual = np.log(y.to_numpy()) - x @ beta
    return np.exp(a @ beta) * np.mean(np.exp(residual))


class TreasuryDealerModelTests(unittest.TestCase):
    def test_literal_order_and_all_three_arms_match_independent_unscaled_qr(self):
        self.assertEqual(MARKET, EXPECTED_MARKET)
        self.assertEqual(MATCHED, EXPECTED_MATCHED)
        self.assertEqual(ALL, EXPECTED_ALL)
        train, y, query = fixture()
        result = fit_models(train, y, query)
        self.assertEqual(set(result), {"predictions", "fits"})
        for arm, names in (("market", MARKET), ("matched", MATCHED), ("candidate", ALL)):
            np.testing.assert_allclose(
                result["predictions"][arm],
                qr_oracle(train, y, query, names),
                rtol=1e-11,
                atol=1e-14,
            )
            audit = result["fits"][arm]
            self.assertEqual(audit["feature_names"], list(names))
            self.assertEqual(audit["rank"], len(names))
            self.assertEqual(audit["n_train"], len(train))
            self.assertEqual(audit["n_application"], len(query))
            self.assertEqual(audit["rank_relative_cutoff"], 1e-12)

    def test_audits_reconstruct_scaling_smearing_and_predictions(self):
        train, y, query = fixture()
        result = fit_models(train, y, query)
        smears = []
        for arm in ("market", "matched", "candidate"):
            audit = result["fits"][arm]
            names = audit["feature_names"]
            raw = train[names].to_numpy()
            means = np.array([audit["feature_means"][n] for n in names[1:]])
            scales = np.array([audit["feature_scales"][n] for n in names[1:]])
            np.testing.assert_allclose(means, raw[:, 1:].mean(0))
            np.testing.assert_allclose(scales, raw[:, 1:].std(0, ddof=0))
            design = np.column_stack([np.ones(len(train)), (raw[:, 1:] - means) / scales])
            beta = np.array([audit["coefficients"][n] for n in names])
            residual = np.log(y) - design @ beta
            self.assertAlmostEqual(audit["log_smearing"], np.log(np.exp(residual).mean()))
            self.assertAlmostEqual(
                audit["normal_equation_max_abs"],
                np.max(np.abs(design.T @ residual / len(train))),
            )
            app = np.column_stack(
                [np.ones(len(query)), (query[names].to_numpy()[:, 1:] - means) / scales]
            )
            np.testing.assert_allclose(
                result["predictions"][arm],
                np.exp(app @ beta + audit["log_smearing"]),
                rtol=1e-13,
            )
            smears.append(audit["log_smearing"])
        self.assertEqual(len(set(smears)), 3)

    def test_application_rows_cannot_change_fit_or_other_predictions(self):
        train, y, query = fixture()
        first = fit_models(train, y, query)
        changed = query.copy()
        changed.iloc[-1, 1:] += 1
        second = fit_models(train, y, changed)
        self.assertEqual(first["fits"], second["fits"])
        for arm in first["predictions"]:
            np.testing.assert_array_equal(
                first["predictions"][arm][:-1], second["predictions"][arm][:-1]
            )

    def test_common32_completeness_rejects_bad_extra_even_for_market_arm(self):
        for defect in (
            "missing",
            "nan_train",
            "nan_query",
            "inf",
            "bool",
            "const",
            "duplicate",
            "target_index",
            "negative_target",
            "few_rows",
        ):
            train, y, query = fixture()
            if defect == "missing":
                train = train.drop(columns="dealer_surprise")
            elif defect == "nan_train":
                train.loc[train.index[0], "dealer_surprise"] = np.nan
            elif defect == "nan_query":
                query.loc[query.index[0], "dealer_surprise"] = np.nan
            elif defect == "inf":
                train.loc[train.index[0], "prior_share_mean_sum"] = np.inf
            elif defect == "bool":
                query["prior_share_mean_sum"] = False
            elif defect == "const":
                train.loc[train.index[0], "const"] = 2.0
            elif defect == "duplicate":
                train.index = train.index[:-1].append(train.index[-2:-1])
            elif defect == "target_index":
                y = y.iloc[::-1]
            elif defect == "negative_target":
                y.iloc[0] = -1
            else:
                train, y = train.iloc[:31], y.iloc[:31]
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                fit_models(train, y, query)

    def test_scale_and_rank_failures_do_not_drop_controls_or_change_solver(self):
        for defect in ("constant", "tiny_scale", "dependent"):
            train, y, query = fixture()
            if defect == "constant":
                train["dealer_surprise"] = 1.0
            elif defect == "tiny_scale":
                train["dealer_surprise"] *= 1e-14
            else:
                train["dealer_surprise"] = train.prior_share_mean_sum + train.tlt_ret
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                fit_models(train, y, query)

    def test_target_units_and_feature_affine_transport_preserve_model(self):
        train, y, query = fixture()
        expected = fit_models(train, y, query)
        train["tlt_r2"] = train.tlt_r2 * 100 + 20
        query["tlt_r2"] = query.tlt_r2 * 100 + 20
        actual = fit_models(train, y * 7, query)
        for arm in ("market", "matched", "candidate"):
            np.testing.assert_allclose(
                actual["predictions"][arm], expected["predictions"][arm] * 7, rtol=1e-11
            )

    def test_extreme_application_arithmetic_aborts_and_empty_application_valid(self):
        train, y, query = fixture()
        query.loc[0, "lrv_d"] = 1e300
        with self.assertRaises(ValueError):
            fit_models(train, y, query)
        result = fit_models(train, y, query.iloc[:0])
        self.assertTrue(all(len(p) == 0 for p in result["predictions"].values()))

    def test_metadata_and_inputs_preserved(self):
        train, y, query = fixture()
        train["treasury_cutoff_date"] = pd.NaT
        query["treasury_cutoff_date"] = pd.NaT
        old_train, old_y, old_query = (
            train.copy(deep=True),
            y.copy(deep=True),
            query.copy(deep=True),
        )
        fit_models(train, y, query)
        assert_frame_equal(train, old_train)
        assert_frame_equal(query, old_query)
        pd.testing.assert_series_equal(y, old_y)


if __name__ == "__main__":
    unittest.main()
