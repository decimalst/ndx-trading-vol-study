"""Prewritten direct-product inference, frozen admission and family contracts."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from src import cross_moment_search as study
from tests.test_joint_risk_search import panel as joint_panel


def upstream_panel():
    panel, dates = joint_panel()
    first = panel.groupby(panel.origin.dt.to_period("M"))["origin"].transform("min")
    panel["fit_origin"] = first
    panel["fit_cutoff_date"] = first - pd.Timedelta(days=1)
    panel["train_last_target"] = first - pd.Timedelta(days=2)
    panel["train_last_available"] = first - pd.Timedelta(days=2)
    return panel, dates


def protocol():
    return yaml.safe_load(study.PROTOCOL.read_text())


def passing_row(control="constant_correlation"):
    return {
        "candidate": "dynamic_correlation",
        "control": control,
        "score": "product_mse",
        "p_holm_wave": 1e-5,
        "p_holm_cumulative": 0.001,
        "phases": [
            {"name": "development", "n": 200, "delta": -2e-10, "stability": []},
            {
                "name": "evaluation",
                "n": 300,
                "delta": -2e-10,
                "stability": [{"delta": -1e-10}, {"delta": -3e-10}],
            },
        ],
    }


class CrossMomentSearch(unittest.TestCase):
    def test_literal_protocol_and_both_hypotheses(self):
        study.validate(protocol())
        self.assertEqual(study.WAVE_ALPHA, 0.05 / (11 * 12))
        self.assertEqual(len(study.CONTRASTS), 2)

    def test_fixed_contract_cannot_be_relaxed(self):
        for group, key, value in [
            ("scoring", "effect_threshold_absolute", 1e-12),
            ("scoring", "new_model_fits", 1),
            ("scoring", "coherence_eps_multiplier", 10000),
            ("index", "market_lag", 0),
            ("index", "source_end", "2025-11-03"),
            ("upstream", "protocol_sha256", "bad"),
            ("upstream", "verification_sha256", "bad"),
            ("upstream", "forecasts_expected", 7383),
            ("inference", "inference_scale", 1.0),
            ("inference", "bootstrap_draws", 9999),
            ("inference", "seed", 1),
            ("inference", "hac_lags", 21),
            ("comparisons", "cumulative_hypotheses", 113),
            ("verification", "product_comparison_eps_multiplier", 10000),
        ]:
            p = copy.deepcopy(protocol())
            p[group][key] = value
            with self.subTest(group=group, key=key), self.assertRaises(ValueError):
                study.validate(p)

    def test_scaled_inference_and_original_units(self):
        p = protocol()
        p["inference"]["bootstrap_draws"] = 19
        a = np.linspace(2e-8, 4e-8, 140)
        b = a + 1e-10
        d = a - b
        result = study.paired_inference(a, b, d, p, 51)
        self.assertAlmostEqual(result["delta"] / 1e-10, -1.0, places=12)
        self.assertAlmostEqual(result["candidate_loss"], float(a.mean()))
        self.assertNotIn("gain_relative", result)
        self.assertEqual(set(result["block_inference"]), {"21", "63", "126"})
        self.assertEqual(
            result["p_conservative"],
            max(result["hac126"]["p"], *[v["p"] for v in result["block_inference"].values()]),
        )
        self.assertEqual(
            result["nominal_mde_effect_ratio"], result["hac126"]["mde80_nominal"] / 1e-10
        )

    def test_constant_units_conversion_preserves_probabilities(self):
        p = protocol()
        p["inference"]["bootstrap_draws"] = 39
        rng = np.random.default_rng(554)
        a = np.exp(rng.normal(size=140)) * 1e-8
        b = np.exp(rng.normal(size=140)) * 1e-8
        d = a - b
        original = study.paired_inference(a, b, d, p, 52)
        changed = copy.deepcopy(p)
        changed["inference"]["inference_scale"] *= 16
        other = study.paired_inference(a * 16, b * 16, d * 16, changed, 52)
        self.assertEqual(original["p_conservative"], other["p_conservative"])
        np.testing.assert_allclose(
            other["ci95_envelope"],
            np.asarray(original["ci95_envelope"]) * 16,
            rtol=1e-12,
            atol=0,
        )

    def test_zero_scores_and_equal_predictions_are_valid(self):
        p = protocol()
        p["inference"]["bootstrap_draws"] = 19
        result = study.paired_inference(np.zeros(140), np.zeros(140), np.zeros(140), p, 53)
        self.assertEqual(result["delta"], 0.0)
        self.assertEqual(result["p_conservative"], 1.0)
        self.assertEqual(result["hac126"]["mde80_nominal"], 0.0)

    def test_inconsistent_difference_and_invalid_arithmetic_rejected(self):
        for a, b, d in [
            (np.ones(126), np.ones(126), np.zeros(126)),
            (np.ones(140), np.ones(140), np.full(140, 1e-6)),
            (-np.ones(140), np.ones(140), -2 * np.ones(140)),
            (np.ones(140), np.ones(140), np.full(140, np.nan)),
            (
                np.linspace(1e-300, 2e-300, 140),
                np.zeros(140),
                np.linspace(1e-300, 2e-300, 140),
            ),
        ]:
            with self.subTest(first=a[0]), self.assertRaises(ValueError):
                study.paired_inference(a, b, d, protocol(), 54)

    def test_both_controls_effect_stability_and_corrections_required(self):
        a, b = passing_row(), passing_row("constant_matrix")
        self.assertEqual(study.candidate_leads([a, b]), ["dynamic_correlation"])
        b["phases"][1]["stability"][1]["delta"] = 0.0
        self.assertEqual(study.candidate_leads([a, b]), [])
        with self.assertRaises(ValueError):
            study.candidate_leads([a])
        for fault in ["effect", "wave", "cumulative"]:
            row = passing_row()
            if fault == "effect":
                row["phases"][0]["delta"] = -0.999e-10
            else:
                row["p_holm_" + fault] = study.WAVE_ALPHA if fault == "wave" else 0.05
            self.assertFalse(study.passes(row))

    def test_failed_wave_keeps_all_new_trials_without_changing_upstream(self):
        result = study.failure_metrics(ValueError("UPSTREAM: invalid"), "hash")
        self.assertEqual(
            (result["hypothesis_count"], result["cumulative_hypothesis_count"]), (2, 114)
        )
        self.assertEqual(result["leads"], [])
        self.assertTrue(
            all(
                r["p_conservative"] == r["p_holm_wave"] == r["p_holm_cumulative"] == 1
                for r in result["rows"]
            )
        )
        self.assertEqual(study.candidate_leads(result["rows"]), [])

    def test_full_inherited_family_includes_both_joint_contrasts(self):
        rows = study.inherited(protocol())
        self.assertEqual(len(rows), 112)
        self.assertEqual(
            sum(r["source"] == "reports/joint_risk/metrics.json" for r in rows), 2
        )

    def test_evaluate_rejects_changed_products_or_incomplete_cohorts(self):
        original, dates = upstream_panel()
        scored = study.cs.score_panel(original)
        for fault in ["loss", "product", "missing", "mean", "diagonal"]:
            bad = scored.copy()
            ix = bad.index[bad.model.eq("dynamic_correlation")][0]
            if fault == "loss":
                bad.loc[ix, "product_mse"] += 1e-10
            elif fault == "product":
                bad.loc[ix, "forecast_product"] += 1e-6
            elif fault == "missing":
                bad = bad.drop(ix)
            elif fault == "mean":
                bad.loc[ix, "mu_spx"] += 0.01
            else:
                bad.loc[ix, "h_qqq"] += 0.01
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                study.evaluate(bad, dates, protocol())

    def test_all_four_phase_scores_and_114_holm_family(self):
        original, dates = upstream_panel()
        scored = study.cs.score_panel(original)

        def stub(a, b, d, p, seed):
            return {
                "n": len(d),
                "delta": float(np.mean(d)),
                "candidate_loss": float(np.mean(a)),
                "control_loss": float(np.mean(b)),
                "p_conservative": 0.01,
                "hac126": {},
                "ci95_envelope": [-1e-10, 1e-10],
                "block_inference": {},
                "nominal_mde_effect_ratio": 1.0,
            }

        with patch.object(study, "paired_inference", side_effect=stub) as call:
            result = study.evaluate(scored, dates, protocol())
        self.assertEqual(call.call_count, 4)
        self.assertEqual(result["cumulative_hypothesis_count"], 114)
        self.assertEqual(result["leads"], [])
        self.assertEqual(len(result["rows"][0]["phases"][1]["stability"]), 2)
        self.assertEqual(result["new_model_fits"], 0)


if __name__ == "__main__":
    unittest.main()
