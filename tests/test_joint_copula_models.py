"""Generated shared-marginal contracts written before the new producer."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import joint_copula_models as model
from tests.test_joint_risk_models import sample


def dependency(z, family):
    return (0.2 if family == "t8" else 0.1), {
        "family": family,
        "n_train": len(z),
        "generated": True,
    }


class JointCopulaModelsTests(unittest.TestCase):
    def test_exact_shared_marginals_and_current_fit_t8_coordinates(self):
        f, t = sample()
        with patch.object(model, "fit_dependence", side_effect=dependency) as fit:
            predictions, audit = model.fit_predict(f.iloc[:120], t.iloc[:120], f.iloc[120:130])
        self.assertEqual(set(predictions), {"t8_copula", "gaussian_copula", "independence"})
        a = predictions["t8_copula"]
        for name in predictions:
            np.testing.assert_array_equal(predictions[name]["mu"], a["mu"])
            np.testing.assert_array_equal(predictions[name]["h"], a["h"])
        np.testing.assert_array_equal(predictions["independence"]["rho"], 0)
        self.assertEqual(audit["residual_staging"], "current_fit_training_residuals")
        transformed, _, _ = model.transform(f.iloc[:120], f.iloc[120:130])
        z = []
        for asset in ("qqq", "spx"):
            moment = audit["moments"][asset]
            mean = moment["mean"]
            x = transformed[mean["columns"]].to_numpy()
            mu = ((x - np.array(mean["means"])) / np.array(mean["scales"])) @ mean["beta"]
            variance = moment["variance"]
            design = np.c_[
                np.ones(len(x)), (x[:, 1:] - variance["means"]) / variance["scales"]
            ]
            h = np.exp(design @ variance["beta"])
            z.append((t["y_" + asset].iloc[:120].to_numpy() - mu) / np.sqrt(0.75 * h))
            self.assertLessEqual(mean["gradient_max_abs"], 1e-10)
            self.assertLessEqual(variance["gradient_max_abs"], 1e-8)
        for call in fit.call_args_list:
            np.testing.assert_allclose(
                call.args[0], np.column_stack(z), rtol=1e-12, atol=1e-12
            )
        self.assertEqual([c.args[1] for c in fit.call_args_list], ["t8", "gaussian"])

    def test_unit_changes_and_application_values_do_not_change_dependence_fit(self):
        f, t = sample()
        with patch.object(model, "fit_dependence", side_effect=dependency) as fit:
            before, audit = model.fit_predict(f.iloc[:120], t.iloc[:120], f.iloc[120:130])
            changed = t.copy()
            changed[["y_qqq", "y_spx"]] *= [10, 100]
            after, _ = model.fit_predict(f.iloc[:120], changed.iloc[:120], f.iloc[120:130])
            query = f.iloc[120:130].copy()
            query["corr22"] = 0.3
            _, other = model.fit_predict(f.iloc[:120], t.iloc[:120], query)
        self.assertEqual(audit, other)
        np.testing.assert_allclose(
            fit.call_args_list[0].args[0], fit.call_args_list[2].args[0], rtol=1e-7, atol=1e-9
        )
        for arm in before:
            np.testing.assert_allclose(
                after[arm]["mu"], before[arm]["mu"] * [10, 100], atol=1e-10
            )
            np.testing.assert_allclose(
                after[arm]["h"], before[arm]["h"] * [100, 10000], rtol=1e-7
            )

    def test_order_types_zero_scale_and_all_zero_risk_rejected(self):
        for defect in ("order", "bool", "infinity", "zero_scale", "zero_risk"):
            f, t = sample()
            if defect == "order":
                t = t.iloc[::-1]
            if defect == "bool":
                f["qqq_day_d"] = True
            if defect == "infinity":
                f.iloc[2, 2] = np.inf
            if defect == "zero_scale":
                f["qqq_day_d"] = 0.0
            if defect == "zero_risk":
                t[["y_qqq", "y_spx"]] = 0.0
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                model.fit_predict(f.iloc[:120], t.iloc[:120], f.iloc[120:130])

    def test_individual_signed_zero_return_is_valid_and_input_unchanged(self):
        f, t = sample()
        t.loc[t.index[20], ["y_qqq", "y_spx"]] = [0.0, -0.01]
        before = t.copy(deep=True)
        with patch.object(model, "fit_dependence", side_effect=dependency):
            predictions, _ = model.fit_predict(f.iloc[:120], t.iloc[:120], f.iloc[120:130])
        self.assertTrue(np.isfinite(predictions["t8_copula"]["h"]).all())
        pd.testing.assert_frame_equal(t, before)


if __name__ == "__main__":
    unittest.main()
