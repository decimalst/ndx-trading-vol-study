"""Prewritten support, proper-score, family and inference contracts."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from src import profiled_quarter_search as study


def protocol():
    return yaml.safe_load(study.PROTOCOL.read_text())


def panel():
    dates = pd.bdate_range("2015-12-01", "2025-10-20")
    old = pd.DataFrame(1.0, index=dates, columns=study.cf.OLD_FEATURES)
    old["feature_cutoff_date"] = pd.Series(dates, index=dates).shift(1)
    features = study.cf.augment_features(old)
    origins = dates[
        ((dates >= "2016-01-04") & (dates <= "2019-12-31"))
        | ((dates >= "2020-01-02") & (dates <= "2025-10-17"))
    ]
    nxt = pd.Series(dates, index=dates).shift(-1)
    origins = origins[
        (origins > "2019-12-31") | (nxt.loc[origins].to_numpy() <= np.datetime64("2019-12-31"))
    ]
    cutoff = pd.Series(dates, index=dates).shift(1)
    monthly = (
        pd.Series(origins, index=origins).groupby(origins.to_period("M")).transform("min")
    )
    y = 0.001 * (1 + np.arange(len(origins)) % 3)
    rows = []
    for name, h in [("mean", 0.002), ("baseline", 0.0018), ("quarter", 0.0019)]:
        rows.append(
            pd.DataFrame(
                {
                    "origin": origins,
                    "model": name,
                    "horizon": 1,
                    "feature_cutoff_date": cutoff.loc[origins].to_numpy(),
                    "target_end": nxt.loc[origins].to_numpy(),
                    "available_date": nxt.loc[origins].to_numpy(),
                    "y": y,
                    "prediction": h,
                    "fit_origin": monthly.to_numpy(),
                    "fit_cutoff_date": cutoff.loc[monthly].to_numpy(),
                    "train_n": 1000,
                    "train_last_target": cutoff.loc[monthly].to_numpy(),
                    "train_last_available": cutoff.loc[monthly].to_numpy(),
                    "phase": np.where(origins <= "2019-12-31", "development", "evaluation"),
                }
            )
        )
    return pd.concat(rows, ignore_index=True).sort_values(["origin", "model"]).reset_index(
        drop=True
    ), features


def passing(control="baseline"):
    return {
        "candidate": "quarter",
        "control": control,
        "score": "proper_variance",
        "p_holm_wave": 1e-5,
        "p_holm_cumulative": 0.001,
        "phases": [
            {"name": "development", "n": 200, "delta": -0.01, "stability": []},
            {
                "name": "evaluation",
                "n": 300,
                "delta": -0.01,
                "stability": [{"delta": -0.01}, {"delta": -0.01}],
            },
        ],
    }


class ProfiledQuarterSearch(unittest.TestCase):
    def test_unfinished_numerical_specification_cannot_be_registered(self):
        unfinished = copy.deepcopy(study.CONTRACT)
        unfinished["verification"]["profiled_solver"] = {
            "specification_status": "SYNTHETIC_DESIGN_PENDING_FINAL_CONSTANTS"
        }
        with (
            patch.object(study, "CONTRACT", unfinished),
            self.assertRaisesRegex(ValueError, "Unfinished"),
        ):
            study.validate(unfinished)

    def test_scientific_model_and_acceptance_gates_are_unchanged(self):
        previous = yaml.safe_load((study.ROOT / "civil_quarter_replay.yaml").read_bytes())
        current = protocol()
        for key in ["index", "target", "civil", "support", "fitting", "scoring"]:
            self.assertEqual(current[key], previous[key], key)
        for key in previous["verification"]:
            if key not in ["baseline_method", "baseline_initialization", "reconstruction"]:
                self.assertEqual(
                    current["verification"][key], previous["verification"][key], key
                )
        self.assertEqual(
            current["model_reuse"]["producer"], previous["model_reuse"]["producer"]
        )
        self.assertEqual(current["failed_verification"]["required_status"], "UNEVALUABLE")
        self.assertEqual(
            current["failed_verification"]["terminal_event"], "verification_failed"
        )
        self.assertEqual(
            current["comparisons"]["inherited_sources"][-1],
            "reports/civil_quarter_replay/metrics.json",
        )

    def test_fixed_contract_and_family(self):
        study.validate(protocol())
        self.assertEqual(study.WAVE_ALPHA, 0.05 / (17 * 18))
        self.assertEqual(
            study.CONTRASTS,
            (
                ("quarter", "baseline", "proper_variance"),
                ("quarter", "mean", "proper_variance"),
            ),
        )
        self.assertEqual(len(study.inherited(protocol())), 125)

    def test_no_relaxed_timing_support_models_numerics_or_family(self):
        for group, key, value in [
            ("index", "market_lag", 0),
            ("index", "minimum_train", 999),
            ("support", "minimum_phase_observations", 1),
            ("index", "source_end", "2025-11-03"),
            ("civil", "nominal_date", "Next observed price row"),
            ("scoring", "effect_threshold_absolute", 1e-10),
            ("fitting", "penalty", 0.1),
            ("comparisons", "cumulative_hypotheses", 122),
            ("support", "civil_rank_relative_tolerance", 1.0),
            ("support", "minimum_relative_residual_norm", 0.0),
            ("inference", "seed", 0),
            ("inference", "bootstrap_draws", 9999),
            ("upstream", "manifest_sha256", "bad"),
            ("failed_verification", "failure_sha256", "bad"),
            ("failed_verification", "generated_forecasts", 0),
            ("verification", "baseline_method", "trust-exact"),
        ]:
            p = copy.deepcopy(protocol())
            p[group][key] = value
            with self.subTest(group=group, key=key), self.assertRaises(ValueError):
                study.validate(p)

    def test_target_measurement_and_source_contract_are_literal(self):
        for group, key, value in [
            ("target", "formula", "Squared signed return"),
            ("fitting", "quarter", "Refit the entire baseline"),
            ("source_contract", "missing", "Zero fill unknown plans"),
            ("verification", "accepted_gradient_tolerance", 1.0),
        ]:
            changed = copy.deepcopy(protocol())
            changed[group][key] = value
            with self.subTest(group=group, key=key), self.assertRaises(ValueError):
                study.validate(changed)

    def test_unknown_or_missing_protocol_section_rejected(self):
        p = protocol()
        p["unregistered_alternate"] = True
        with self.assertRaises(ValueError):
            study.validate(p)
        for key in protocol():
            p = protocol()
            del p[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                study.validate(p)

    def test_model_and_evidence_geometry_are_exact_failed_attempt(self):
        import yaml

        original = yaml.safe_load((study.ROOT / "civil_quarter.yaml").read_bytes())
        new = protocol()
        for key in ["index", "civil", "support", "fitting", "scoring"]:
            self.assertEqual(new[key], original[key])
        self.assertEqual(new["inference"]["bootstrap_draws"], 99999)
        self.assertEqual(new["inference"]["seed"], 20260923)
        self.assertLess(1 / (new["inference"]["bootstrap_draws"] + 1), study.WAVE_ALPHA / 2)
        self.assertEqual(new["failed_attempt"]["required_status"], "UNEVALUABLE")

    def test_both_controls_both_effects_and_strict_p_and_stability(self):
        rows = [passing(c) for c in ["baseline", "mean"]]
        self.assertEqual(study.candidate_leads(rows), ["quarter"])
        for fault in ["effect", "wave", "family", "stability"]:
            bad = copy.deepcopy(rows)
            if fault == "effect":
                bad[0]["phases"][0]["delta"] = -0.004999
            if fault == "wave":
                bad[0]["p_holm_wave"] = study.WAVE_ALPHA
            if fault == "family":
                bad[0]["p_holm_cumulative"] = 0.05
            if fault == "stability":
                bad[0]["phases"][1]["stability"][0]["delta"] = 0
            self.assertEqual(study.candidate_leads(bad), [])
        for bad in [rows[:1], rows + [rows[0]], [rows[0], rows[0]]]:
            with self.assertRaises(ValueError):
                study.candidate_leads(bad)

    def test_all_unevaluable_hypotheses_retained(self):
        m = study.failure_metrics(ValueError("INSUFFICIENT_DATA: support"), "hash")
        self.assertEqual((m["hypothesis_count"], m["cumulative_hypothesis_count"]), (2, 127))
        self.assertEqual(m["leads"], [])
        self.assertTrue(
            all(
                r["p_conservative"] == r["p_holm_wave"] == r["p_holm_cumulative"] == 1
                for r in m["rows"]
            )
        )

    def test_civil_support_precedes_every_inference_call(self):
        rows, features = panel()
        features.loc[features.index <= "2019-12-31", "quarter_end5"] = 0.0
        with (
            patch.object(study, "paired_inference") as call,
            self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"),
        ):
            study.evaluate(rows, features, protocol(), 118)
        call.assert_not_called()

    def test_declared_training_and_phase_fences_precede_inference(self):
        original, calendar = panel()
        for fault in ["training_count", "phase", "development_label"]:
            bad = original.copy()
            mask = bad.origin.eq(pd.Timestamp("2019-12-30"))
            if fault == "training_count":
                bad.loc[mask, "train_n"] = 999
            elif fault == "phase":
                bad.loc[mask, "phase"] = "evaluation"
            else:
                bad.loc[mask, ["target_end", "available_date"]] = pd.Timestamp("2020-01-02")
            with (
                patch.object(study, "paired_inference") as call,
                self.assertRaises(ValueError),
            ):
                study.evaluate(bad, calendar, protocol(), 118)
            call.assert_not_called()

    def test_true_next_session_label_cannot_cross_development_boundary(self):
        original, calendar = panel()
        added = original.loc[original.origin.eq(pd.Timestamp("2019-12-30"))].copy()
        added["origin"] = pd.Timestamp("2019-12-31")
        added["feature_cutoff_date"] = pd.Timestamp("2019-12-30")
        added[["target_end", "available_date"]] = pd.Timestamp("2020-01-01")
        bad = pd.concat([original, added], ignore_index=True).sort_values(["origin", "model"])
        with (
            patch.object(study, "paired_inference") as call,
            self.assertRaisesRegex(ValueError, "development/evaluation fences"),
        ):
            study.evaluate(bad, calendar, protocol(), 118)
        call.assert_not_called()

    def test_four_phases_complete_accounting_and_support(self):
        rows, calendar = panel()

        def infer(a, b, d, p, seed):
            return {
                "n": len(d),
                "delta": float(np.mean(d)),
                "candidate_loss": float(np.mean(a)),
                "control_loss": float(np.mean(b)),
                "p_conservative": 0.5,
                "hac126": {},
                "ci95_envelope": [-0.1, 0.1],
                "block_inference": {},
                "nominal_mde_effect_ratio": 1.0,
            }

        with patch.object(study, "paired_inference", side_effect=infer) as call:
            m = study.evaluate(rows, calendar, protocol(), 118)
        self.assertEqual(call.call_count, 4)
        self.assertEqual((m["hypothesis_count"], m["cumulative_hypothesis_count"]), (2, 127))
        self.assertEqual(m["new_forecasts"], len(rows))
        self.assertEqual(m["new_monthly_fits"], 118)
        for phase in ["development", "evaluation"]:
            self.assertEqual(m["civil_support"][phase]["scope"], "phase")
            self.assertGreaterEqual(
                m["civil_support"][phase]["groups"]["quarter_end5"]["ones"], 30
            )
        self.assertTrue(all(len(r["phases"][-1]["stability"]) == 2 for r in m["rows"]))

    def test_native_negative_scores_and_fixed_variance_inference_units(self):
        y = np.linspace(0.001, 0.004, 160)
        candidate = np.full(160, 0.002)
        control = np.full(160, 0.0021)
        a = study.cs.proper_score(y, candidate)
        b = study.cs.proper_score(y, control)
        d = study.paired_difference(candidate, control, y)
        self.assertTrue((a < 0).all())
        hac = {"se": 0.001, "p": 0.5, "ci95": [-0.1, 0.1], "mde80_nominal": 0.01}

        def draws(values, block, n, seed):
            return np.full((3, 1), np.mean(values))

        with (
            patch.object(study.inference, "hac_summary", return_value=hac),
            patch.object(study.inference, "bootstrap_means", side_effect=draws),
        ):
            result = study.paired_inference(a, b, d, protocol(), 0)
        self.assertEqual(result["nominal_mde_effect_ratio"], 2.0)
        self.assertEqual(result["candidate_loss"], float(a.mean()))
        self.assertEqual(result["control_loss"], float(b.mean()))
        self.assertEqual(result["delta"], float(d.mean()))

    def test_inference_rejects_nonfinite_complex_and_misaligned_inputs(self):
        valid = np.linspace(-8.0, -7.0, 160)
        for a, b, d in [
            (valid + 1j, valid, np.zeros(160)),
            (valid, valid[:-1], np.zeros(160)),
            (valid, valid, np.full(160, np.nan)),
        ]:
            with self.subTest(case=(len(a), len(b))), self.assertRaises(ValueError):
                study.paired_inference(a, b, d, protocol(), 0)


if __name__ == "__main__":
    unittest.main()
