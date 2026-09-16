"""Pre-fit fixed protocol, paired loss and complete-family accounting tests."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from src import joint_risk_search as study
from src.joint_risk_features import MODELS


def protocol():
    return yaml.safe_load(study.PROTOCOL.read_text())


def passing_row(control="constant_correlation"):
    return {
        "candidate": "dynamic_correlation",
        "control": control,
        "score": "matrix_qlike",
        "p_holm_wave": 1e-5,
        "p_holm_cumulative": 0.001,
        "phases": [
            {
                "name": "development",
                "n": 200,
                "delta": -0.006,
                "stability": [],
            },
            {
                "name": "evaluation",
                "n": 300,
                "delta": -0.006,
                "stability": [{"delta": -0.002}, {"delta": -0.004}],
            },
        ],
    }


def panel():
    dates = (
        pd.bdate_range("2018-01-02", periods=140)
        .append(pd.bdate_range("2021-01-04", periods=140))
        .append(pd.bdate_range("2024-01-02", periods=140))
    )
    out = []
    for name in MODELS:
        out.append(
            pd.DataFrame(
                {
                    "origin": dates,
                    "model": name,
                    "horizon": 1,
                    "feature_cutoff_date": dates - pd.Timedelta(days=1),
                    "target_end": dates + pd.Timedelta(days=1),
                    "available_date": dates + pd.Timedelta(days=1),
                    "y_qqq": 0.02,
                    "y_spx": 0.01,
                    "mu_qqq": 0.0,
                    "mu_spx": 0.0,
                    "h_qqq": 0.001,
                    "h_spx": 0.001,
                    "rho": 0.5 if name == "dynamic_correlation" else 0.1,
                    "loss": 0.0,
                    "fit_origin": dates,
                    "fit_cutoff_date": dates - pd.Timedelta(days=1),
                    "train_n": 1000,
                    "train_last_target": dates - pd.Timedelta(days=2),
                    "train_last_available": dates - pd.Timedelta(days=2),
                    "phase": np.where(dates.year < 2020, "development", "evaluation"),
                }
            )
        )
    result = pd.concat(out, ignore_index=True)
    rho = result.rho.to_numpy()
    result["loss"] = (
        2 * np.log(0.001) + np.log1p(-rho * rho) + (0.5 - 0.4 * rho) / (1 - rho * rho)
    )
    return result, dates


class JointRiskSearch(unittest.TestCase):
    def test_literal_protocol(self):
        study.validate(protocol())
        self.assertEqual(study.WAVE_ALPHA, 0.05 / (10 * 11))
        self.assertEqual(len(study.CONTRASTS), 2)

    def test_core_contract_amendments_rejected(self):
        original = protocol()
        for group, key, value in [
            ("index", "minimum_train", 999),
            ("index", "market_lag", 0),
            ("index", "effect_threshold_absolute", 0.001),
            ("index", "penalty", 0.1),
            ("inference", "bootstrap_draws", 9999),
            ("inference", "hac_lags", 21),
            ("inference", "seed", 1),
            ("comparisons", "cumulative_hypotheses", 111),
            ("measurement", "gk_floor", 1e-8),
            ("index", "source_end", "2025-11-03"),
            ("correlation", "window", 21),
            ("correlation", "roundoff_tolerance", 1e-8),
            ("verification", "independent_variance_coefficient_absolute_tolerance", 1e-3),
            ("verification", "global_value_gap", 1e-5),
        ]:
            p = copy.deepcopy(original)
            p[group][key] = value
            with self.subTest(group=group, key=key), self.assertRaises(ValueError):
                study.validate(p)

    def test_paired_inference_uses_worst_dependence_method(self):
        p = protocol()
        p["inference"]["bootstrap_draws"] = 19
        a = np.linspace(0.1, 1.2, 140)
        out = study.paired_inference(a, a + 0.01, p, 41)
        self.assertAlmostEqual(out["delta"], -0.01)
        self.assertEqual(set(out["block_inference"]), {"21", "63", "126"})
        self.assertEqual(
            out["p_conservative"],
            max(out["hac126"]["p"], *(x["p"] for x in out["block_inference"].values())),
        )

    def test_inference_does_not_shrink_blocks_or_remove_nonfinite(self):
        for a, b in [
            (np.ones(126), np.ones(126)),
            (np.r_[np.ones(139), np.nan], np.ones(140)),
        ]:
            with self.assertRaises(ValueError):
                study.paired_inference(a, b, protocol(), 1)

    def test_all_gates_and_both_controls_required(self):
        a, b = passing_row(), passing_row("constant_matrix")
        self.assertEqual(study.candidate_leads([a, b]), ["dynamic_correlation"])
        b["phases"][1]["stability"][1]["delta"] = 0.00001
        self.assertEqual(study.candidate_leads([a, b]), [])
        with self.assertRaises(ValueError):
            study.candidate_leads([a])
        with self.assertRaises(ValueError):
            study.candidate_leads([a, a])

    def test_nominal_small_or_one_phase_gain_cannot_pass(self):
        for path, value in [
            ("effect", -0.004999),
            ("wave", study.WAVE_ALPHA),
            ("cumulative", 0.05),
        ]:
            row = passing_row()
            if path == "effect":
                row["phases"][0]["delta"] = value
            else:
                row["p_holm_" + path] = value
            self.assertFalse(study.passes(row))

    def test_failed_wave_retains_both_as_p_one(self):
        for error, status in [
            (ValueError("broken"), "INVALID_RUN"),
            (ValueError("INSUFFICIENT_DATA: scale"), "INSUFFICIENT_DATA"),
        ]:
            out = study.failure_metrics(error, "abc")
            self.assertEqual(
                (out["hypothesis_count"], out["cumulative_hypothesis_count"]), (2, 112)
            )
            self.assertEqual(out["status"], "UNEVALUABLE")
            self.assertEqual(len(out["rows"]), 2)
            self.assertTrue(
                all(r["p_conservative"] == 1 and r["status"] == status for r in out["rows"])
            )
            self.assertEqual(study.candidate_leads(out["rows"]), [])

    def test_inherited_count_is_full_enumerated_family(self):
        rows = study.inherited(protocol())
        self.assertEqual(len(rows), 110)
        self.assertEqual(
            sum(r["source"] == "reports/relative_risk/metrics.json" for r in rows), 2
        )

    def test_common_paired_metadata_and_finite_signed_forecasts_required(self):
        original, dates = panel()
        for fault in ["missing", "duplicate", "target", "nonfinite", "metadata"]:
            data = original.copy()
            ix = data.index[data.model.eq("dynamic_correlation")][0]
            if fault == "missing":
                data = data.drop(ix)
            elif fault == "duplicate":
                data = pd.concat([data, data.iloc[[ix]]], ignore_index=True)
            elif fault == "target":
                data.loc[ix, "y_spx"] = 0.02
            elif fault == "nonfinite":
                data.loc[ix, "rho"] = np.inf
            else:
                data.loc[ix, "train_n"] = 1001
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                study.evaluate(data, dates, protocol())

    def test_all_four_phase_scores_and_112_adjustment_preserved(self):
        data, dates = panel()

        def small_inference(a, b, p, seed):
            return {
                "n": len(a),
                "delta": float(np.mean(a - b)),
                "candidate_loss": float(np.mean(a)),
                "control_loss": float(np.mean(b)),
                "p_conservative": 0.01,
                "hac126": {},
                "ci95_envelope": [-0.2, 0.1],
                "block_inference": {},
            }

        with patch.object(study, "paired_inference", side_effect=small_inference) as checked:
            result = study.evaluate(data, dates, protocol())
        self.assertEqual(checked.call_count, 4)
        self.assertEqual(len(result["rows"]), 2)
        self.assertEqual(result["cumulative_hypothesis_count"], 112)
        self.assertEqual(result["leads"], [])
        for row in result["rows"]:
            self.assertEqual(len(row["phases"]), 2)
            self.assertEqual(len(row["phases"][1]["stability"]), 2)
            self.assertEqual(row["p_holm_wave"], 0.02)

    def test_negative_and_zero_raw_scores_are_valid(self):
        p = protocol()
        p["inference"]["bootstrap_draws"] = 19
        out = study.paired_inference(-np.ones(140), np.zeros(140), p, 1)
        self.assertEqual(out["delta"], -1.0)
        self.assertNotIn("gain_relative", out)

    def test_common_means_and_conditional_diagonals_enforced(self):
        original, dates = panel()
        for column in ["mu_qqq", "mu_spx", "h_qqq", "h_spx"]:
            data = original.copy()
            ix = data.index[data.model.eq("dynamic_correlation")][0]
            data.loc[ix, column] += 0.01
            with self.subTest(column=column), self.assertRaises(ValueError):
                study.evaluate(data, dates, protocol())

    def test_dependence_contract_cannot_be_relaxed(self):
        for key, value in [
            ("rho_max", 0.9999),
            ("parameter_bound", 8.0),
            ("slope_penalty", 0.0),
            ("global_value_gap", 1e-5),
            ("maximum_splits", 100000),
            ("projected_gradient_tolerance", 1e-4),
            ("minimum_correlation_eigenvalue", 0.0),
        ]:
            p = protocol()
            p["dependence"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                study.validate(p)


if __name__ == "__main__":
    unittest.main()
