"""Prewritten causal frozen-coefficient replay and five-model cohort contracts."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from src import joint_risk_models as joint
from src import target_aligned_panel as panel
from src.cross_moment_score import score_panel
from tests.test_joint_risk_models import sample


def frozen_fit(train, target, application):
    columns = list(joint.MARGINAL_COLUMNS)
    n = len(columns)
    audits = {}
    for asset in ["qqq", "spx"]:
        audits[asset] = {
            "mean": {
                "columns": columns,
                "means": [0.0] * n,
                "scales": [1.0] * n,
                "beta": [0.0] * n,
                "train_n": len(train),
                "alpha": 0.01,
            },
            "variance": {
                "columns": columns,
                "means": [0.0] * (n - 1),
                "scales": [1.0] * (n - 1),
                "scaled_beta": [0.0] * n,
                "train_mean": 0.0001,
                "train_n": len(train),
                "alpha": 0.01,
            },
        }
    return {
        "forecasts": {
            name: {
                "mu": np.zeros((len(application), 2)),
                "h": np.full(
                    (len(application), 2), 0.0001 if name != "constant_matrix" else 0.0002
                ),
                "rho": np.full(len(application), 0.1),
            }
            for name in joint.MODELS
        },
        "audit": {
            "moments": audits,
            "transform": {"corr22_mean": float(train.corr22.mean())},
            "corr22_scale": float(train.corr22.std(ddof=0)),
            "residual_staging": "current_fit_training_residuals",
        },
    }


def fixture():
    f, t = sample(1340)
    dates = pd.bdate_range("2011-01-03", periods=len(f))
    f.index = t.index = dates
    f["feature_cutoff_date"] = pd.Series(dates, index=dates).shift(1)
    for c in ["target_end", "available_date"]:
        t[c] = pd.Series(dates, index=dates).shift(-1)
    f.loc[dates[0], "corr22"] = np.nan
    config = yaml.safe_load(panel.ROOT.joinpath("joint_risk.yaml").read_text())["index"]
    with patch.object(joint, "fit_predict", side_effect=frozen_fit):
        original, fits = joint.forecast_panel(f, t, config)
    return f, t, score_panel(original), fits, config


class TargetAlignedPanel(unittest.TestCase):
    def test_saved_coefficients_replayed_without_new_marginal_fit(self):
        f, t, _old, fits, _config = fixture()
        record = fits[0]
        entry = pd.Timestamp(record["fit_origin"])
        mask = joint.training_mask(f, t, entry)
        tr = f.loc[mask]
        app = f.loc[[entry]]
        result = panel.replay_frozen_moments(tr, t.loc[mask], app, record["model_audit"])
        np.testing.assert_array_equal(result["residual"], t.loc[mask, ["y_qqq", "y_spx"]])
        np.testing.assert_allclose(result["training_h"], 0.0001, rtol=1e-14, atol=0)
        np.testing.assert_array_equal(result["application_mu"], 0)
        np.testing.assert_allclose(result["application_h"], 0.0001, rtol=1e-14, atol=0)
        np.testing.assert_allclose(
            result["training_z"], (tr.corr22 - tr.corr22.mean()) / tr.corr22.std(ddof=0)
        )
        changed = app.copy()
        changed.iloc[0, 1] += 100
        other = panel.replay_frozen_moments(tr, t.loc[mask], changed, record["model_audit"])
        np.testing.assert_array_equal(result["residual"], other["residual"])
        np.testing.assert_array_equal(result["training_h"], other["training_h"])

    def test_replay_uses_stored_nonzero_coefficients_and_scaled_variance(self):
        f, t, _, fits, _ = fixture()
        entry = pd.Timestamp(fits[0]["fit_origin"])
        mask = joint.training_mask(f, t, entry)
        tr = f.loc[mask]
        a = copy.deepcopy(fits[0]["model_audit"])
        app = f.loc[[entry]]
        a["moments"]["qqq"]["mean"]["beta"][0] = 0.02
        a["moments"]["qqq"]["mean"]["beta"][1] = 0.001
        a["moments"]["spx"]["variance"]["scaled_beta"][0] = np.log(2)
        result = panel.replay_frozen_moments(tr, t.loc[mask], app, a)
        np.testing.assert_allclose(
            result["residual"][:, 0], t.loc[mask, "y_qqq"] - (0.02 + 0.001 * tr.iloc[:, 1])
        )
        np.testing.assert_allclose(result["training_h"][:, 1], 0.0002, rtol=1e-14, atol=0)

    def test_invalid_saved_geometry_is_not_refitted_or_repaired(self):
        f, t, _, fits, _ = fixture()
        entry = pd.Timestamp(fits[0]["fit_origin"])
        mask = joint.training_mask(f, t, entry)
        for fault in [
            "scale",
            "columns",
            "center",
            "corrscale",
            "staging",
            "n",
            "mean",
            "variance",
        ]:
            a = copy.deepcopy(fits[0]["model_audit"])
            if fault == "scale":
                a["moments"]["qqq"]["mean"]["scales"][1] = 0
            elif fault == "columns":
                a["moments"]["qqq"]["mean"]["columns"][1] = "wrong"
            elif fault == "center":
                a["transform"]["corr22_mean"] += 0.1
            elif fault == "corrscale":
                a["corr22_scale"] = 0
            elif fault == "staging":
                a["residual_staging"] = "future"
            elif fault == "n":
                a["moments"]["qqq"]["variance"]["train_n"] += 1
            elif fault == "mean":
                a["moments"]["qqq"]["mean"]["beta"][0] = np.inf
            else:
                a["moments"]["spx"]["variance"]["train_mean"] = 0
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                panel.replay_frozen_moments(f.loc[mask], t.loc[mask], f.loc[[entry]], a)

    def test_all_original_rows_exact_and_both_new_models_on_same_origins(self):
        f, t, old, fits, config = fixture()
        before = old.copy(deep=True)
        result, audits = panel.forecast_panel(f, t, old, fits, config)
        pd.testing.assert_frame_equal(old, before)
        pd.testing.assert_frame_equal(
            result.loc[result.model.isin(joint.MODELS)].reset_index(drop=True), old
        )
        self.assertEqual(set(result.model), set(panel.MODELS))
        self.assertEqual(len(result), len(old) // 3 * 5)
        self.assertEqual(len(audits), len(fits))
        for a, b in zip(audits, fits):
            self.assertEqual(a["train_n"], b["train_n"])
            self.assertEqual(a["application_n"], b["application_n"])
        base = old.loc[old.model == "constant_correlation"].set_index("origin")
        for name in panel.NEW_MODELS:
            rows = result.loc[result.model == name].set_index("origin")
            pd.testing.assert_frame_equal(
                rows[["mu_qqq", "mu_spx", "h_qqq", "h_spx"]],
                base[["mu_qqq", "mu_spx", "h_qqq", "h_spx"]],
            )

    def test_changed_frozen_metadata_missing_fit_and_wrong_issued_marginals_reject(self):
        for fault in [
            "n",
            "first",
            "last",
            "application",
            "missing",
            "mean",
            "h",
            "product",
            "missingrow",
        ]:
            f, t, old, fits, config = fixture()
            if fault in ["n", "first", "last", "application"]:
                key = {
                    "n": "train_n",
                    "first": "train_first_origin",
                    "last": "train_last_available",
                    "application": "application_n",
                }[fault]
                fits[0][key] = (
                    fits[0][key] + 1 if fault in ["n", "application"] else "2025-11-03"
                )
            elif fault == "missing":
                fits.pop()
            elif fault in ["mean", "h"]:
                column = "mu_qqq" if fault == "mean" else "h_qqq"
                old.loc[
                    old.model.ne("constant_matrix") if fault == "h" else old.model.notna(),
                    column,
                ] += 0.001
            elif fault == "product":
                old.loc[0, "forecast_product"] += 0.01
            else:
                old = old.iloc[3:].reset_index(drop=True)
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                panel.forecast_panel(f, t, old, fits, config)

    def test_future_labels_do_not_change_first_scalar_fit(self):
        f, t, old, fits, config = fixture()
        _, original = panel.forecast_panel(f, t, old, fits, config)
        entry = pd.Timestamp(fits[0]["fit_origin"])
        changed = t.copy()
        changed.loc[changed.index >= entry, ["y_qqq", "y_spx"]] *= 3
        # Run replay directly with the identical mature mask; future labels cannot enter it.
        mask = joint.training_mask(f, changed, entry)
        replay = panel.replay_frozen_moments(
            f.loc[mask], changed.loc[mask], f.loc[[entry]], fits[0]["model_audit"]
        )
        actual = panel.scalar.fit_cross_moment(
            replay["residual"], replay["training_h"], replay["training_z"]
        )
        self.assertEqual(actual, original[0]["model_audit"])

    def test_all_five_cohorts_and_products_validated(self):
        f, t, old, fits, config = fixture()
        result, _ = panel.forecast_panel(f, t, old, fits, config)
        for fault in ["rho", "mu", "h", "product", "missing", "loss"]:
            bad = result.copy()
            i = bad.index[bad.model == "aligned_dynamic"][0]
            if fault == "missing":
                bad = bad.drop(i)
            else:
                name = {
                    "rho": "rho",
                    "mu": "mu_spx",
                    "h": "h_spx",
                    "product": "product_mse",
                    "loss": "loss",
                }[fault]
                bad.loc[i, name] += 1
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                panel.validate_panel(bad)

    def test_entire_unscored_application_month_still_retains_both_scalar_fits(self):
        f, t, _old, _fits, config = fixture()
        t.loc[t.index >= "2016-02-01", ["y_qqq", "y_spx"]] = np.nan
        with patch.object(joint, "fit_predict", side_effect=frozen_fit):
            issued, fits = joint.forecast_panel(f, t, config)
        result, audits = panel.forecast_panel(f, t, score_panel(issued), fits, config)
        self.assertEqual(len(audits), len(fits))
        self.assertEqual(audits[-1]["fit_origin"], "2016-02-01")
        self.assertFalse(result.origin.ge("2016-02-01").any())
        self.assertGreater(audits[-1]["application_n"], 0)


if __name__ == "__main__":
    unittest.main()
