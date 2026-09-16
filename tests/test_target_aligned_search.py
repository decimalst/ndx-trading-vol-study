"""Prewritten full target-aligned family, fixed gates and inference contracts."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from src import joint_risk_density
from src import target_aligned_search as study
from tests.test_cross_moment_search import upstream_panel


def protocol():
    return yaml.safe_load(study.PROTOCOL.read_text())


def scored_panel():
    old, dates = upstream_panel()
    parts = [study.cs.score_panel(old)]
    base = old.loc[old.model == "constant_correlation"].copy()
    for name, rho in [("aligned_constant", 0.15), ("aligned_dynamic", 0.2)]:
        rows = base.copy()
        rows["model"] = name
        rows["rho"] = rho
        rows["loss"] = joint_risk_density.matrix_score(
            rows[["y_qqq", "y_spx"]].to_numpy() - rows[["mu_qqq", "mu_spx"]].to_numpy(),
            rows[["h_qqq", "h_spx"]].to_numpy(),
            rows.rho.to_numpy(),
        )
        parts.append(study.tp.append_products(rows))
    return pd.concat(parts, ignore_index=True).sort_values(["origin", "model"]).reset_index(
        drop=True
    ), dates


def passing(control="aligned_constant"):
    return {
        "candidate": "aligned_dynamic",
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


class TargetAlignedSearch(unittest.TestCase):
    def test_literal_protocol_and_all_three_registered_comparisons(self):
        study.validate(protocol())
        self.assertEqual(study.WAVE_ALPHA, 0.05 / (12 * 13))
        self.assertEqual(len(study.CONTRASTS), 3)

    def test_protocol_cannot_relax_counts_effect_fits_timing_or_numerics(self):
        changes = [
            ("fitting", "cap", 0.999),
            ("fitting", "gradient_eps_multiplier", 10000),
            ("fitting", "new_scalar_fits_expected", 2),
            ("scoring", "effect_threshold_absolute", 1e-12),
            ("index", "minimum_train", 999),
            ("index", "source_end", "2025-11-03"),
            ("upstream", "manifest_sha256", "bad"),
            ("upstream", "verification_sha256", "bad"),
            ("inference", "bootstrap_draws", 9999),
            ("inference", "seed", 0),
            ("comparisons", "new_hypotheses", 2),
            ("comparisons", "cumulative_hypotheses", 116),
            ("verification", "coefficient_absolute_tolerance", 0.1),
        ]
        for group, key, value in changes:
            p = copy.deepcopy(protocol())
            p[group][key] = value
            with self.subTest(group=group, key=key), self.assertRaises(ValueError):
                study.validate(p)

    def test_all_three_controls_are_required(self):
        rows = [
            passing(c) for c in ["aligned_constant", "constant_matrix", "dynamic_correlation"]
        ]
        self.assertEqual(study.candidate_leads(rows), ["aligned_dynamic"])
        rows[-1]["phases"][-1]["stability"][0]["delta"] = 0
        self.assertEqual(study.candidate_leads(rows), [])
        for bad in [rows[:2], rows + [rows[0]], rows[:2] + [rows[0]]]:
            with self.assertRaises(ValueError):
                study.candidate_leads(bad)

    def test_effect_both_phases_and_both_adjustments_are_strict(self):
        for fault in ["effect", "wave", "family"]:
            row = passing()
            if fault == "effect":
                row["phases"][0]["delta"] = -0.999e-10
            elif fault == "wave":
                row["p_holm_wave"] = study.WAVE_ALPHA
            else:
                row["p_holm_cumulative"] = 0.05
            self.assertFalse(study.passes(row))

    def test_failed_run_keeps_all_three_hypotheses_p_one(self):
        m = study.failure_metrics(ValueError("injected"), "hash")
        self.assertEqual(m["leads"], [])
        self.assertEqual((m["hypothesis_count"], m["cumulative_hypothesis_count"]), (3, 117))
        self.assertTrue(
            all(
                r["p_conservative"] == r["p_holm_wave"] == r["p_holm_cumulative"] == 1
                for r in m["rows"]
            )
        )

    def test_full_inherited_114_family_includes_previous_product_scores(self):
        rows = study.inherited(protocol())
        self.assertEqual(len(rows), 114)
        self.assertEqual(
            sum(r["source"] == "reports/cross_moment/metrics.json" for r in rows), 2
        )

    def test_six_phases_and_117_family_keep_generated_counts_separate(self):
        panel, dates = scored_panel()

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
            m = study.evaluate(panel, dates, protocol(), 118)
        self.assertEqual(call.call_count, 6)
        self.assertEqual(m["cumulative_hypothesis_count"], 117)
        self.assertEqual(m["leads"], [])
        self.assertEqual(m["new_forecasts"], 2 * len(dates))
        self.assertEqual(m["reused_forecasts"], 3 * len(dates))
        self.assertEqual(m["new_monthly_fits"], 118)
        self.assertEqual(m["new_scalar_fits"], 236)
        self.assertTrue(all(len(r["phases"][-1]["stability"]) == 2 for r in m["rows"]))

    def test_invalid_five_model_products_and_cohort_rejected_before_inference(self):
        p, dates = scored_panel()
        for fault in ["loss", "missing", "mean", "diagonal"]:
            bad = p.copy()
            i = bad.index[bad.model == "aligned_dynamic"][0]
            if fault == "missing":
                bad = bad.drop(i)
            else:
                bad.loc[
                    i, {"loss": "product_mse", "mean": "mu_spx", "diagonal": "h_spx"}[fault]
                ] += 0.01
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                study.evaluate(bad, dates, protocol(), 118)


if __name__ == "__main__":
    unittest.main()
