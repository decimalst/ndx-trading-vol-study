"""Prewritten support, proper-score, family and inference contracts."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from src import causal_pool_search as study


def protocol():
    return yaml.safe_load(study.PROTOCOL.read_text())


def panel():
    calendar = pd.bdate_range("2015-12-01", "2025-10-20")
    dates = calendar[
        ((calendar >= "2016-01-04") & (calendar <= "2019-12-31"))
        | ((calendar >= "2020-01-02") & (calendar <= "2025-10-17"))
    ]
    nxt = pd.Series(calendar, index=calendar).shift(-1)
    dates = dates[
        (dates > "2019-12-31") | (nxt.loc[dates].to_numpy() <= np.datetime64("2019-12-31"))
    ]
    cutoff = pd.Series(calendar, index=calendar).shift(1)
    monthly = pd.Series(dates, index=dates).groupby(dates.to_period("M")).transform("min")
    y = (np.arange(len(dates)) % 3 != 0).astype(float)
    rows = []
    for model, probability in [
        ("recent_frequency", 0.6),
        ("frozen_baseline", 0.65),
        ("pooled", 0.625),
    ]:
        rows.append(
            pd.DataFrame(
                {
                    "origin": dates,
                    "model": model,
                    "horizon": 1,
                    "feature_cutoff_date": cutoff.loc[dates].to_numpy(),
                    "target_end": nxt.loc[dates].to_numpy(),
                    "available_date": nxt.loc[dates].to_numpy(),
                    "y": y,
                    "probability": probability,
                    "loss": (y - probability) ** 2,
                    "fit_origin": monthly.to_numpy(),
                    "fit_cutoff_date": cutoff.loc[monthly].to_numpy(),
                    "train_n": 1000,
                    "train_last_target": cutoff.loc[monthly].to_numpy(),
                    "train_last_available": cutoff.loc[monthly].to_numpy(),
                    "phase": np.where(dates <= "2019-12-31", "development", "evaluation"),
                }
            )
        )
    return pd.concat(rows, ignore_index=True).sort_values(["origin", "model"]).reset_index(
        drop=True
    ), calendar


def passing(control="frozen_baseline"):
    return {
        "candidate": "pooled",
        "control": control,
        "score": "brier",
        "p_holm_wave": 1e-5,
        "p_holm_cumulative": 0.001,
        "phases": [
            {"name": "development", "n": 200, "delta": -0.001, "stability": []},
            {
                "name": "evaluation",
                "n": 300,
                "delta": -0.001,
                "stability": [{"delta": -0.001}, {"delta": -0.001}],
            },
        ],
    }


class CausalPoolSearch(unittest.TestCase):
    def test_fixed_contract_and_family(self):
        study.validate(protocol())
        self.assertEqual(study.WAVE_ALPHA, 0.05 / (14 * 15))
        self.assertEqual(
            study.CONTRASTS,
            (("pooled", "frozen_baseline", "brier"), ("pooled", "recent_frequency", "brier")),
        )
        self.assertEqual(len(study.inherited(protocol())), 119)

    def test_no_relaxed_timing_support_models_numerics_or_family(self):
        for group, key, value in [
            ("index", "market_lag", 0),
            ("index", "minimum_train", 999),
            ("index", "minimum_train_per_class", 1),
            ("index", "source_end", "2025-11-03"),
            ("pooling", "half_life_sessions", 5),
            ("scoring", "effect_threshold_absolute", 1e-10),
            ("pooling", "baseline_weight", 0.1),
            ("comparisons", "cumulative_hypotheses", 120),
            ("inference", "minimum_phase_per_class", 1),
            ("inference", "minimum_slice_per_class", 1),
            ("inference", "seed", 0),
            ("inference", "bootstrap_draws", 9999),
            ("upstream", "manifest_sha256", "bad"),
        ]:
            p = copy.deepcopy(protocol())
            p[group][key] = value
            with self.subTest(group=group, key=key), self.assertRaises(ValueError):
                study.validate(p)

    def test_target_measurement_and_source_contract_are_literal(self):
        for group, key, value in [
            ("target", "zero", "Count ties as agreement"),
            ("pooling", "seed", "Use full-phase frequency"),
            ("source_contract", "reference_calendar", "Intersect asset calendars"),
            ("verification", "state_relative_roundoff_multiplier", 64000),
        ]:
            changed = copy.deepcopy(protocol())
            changed[group][key] = value
            with self.subTest(group=group, key=key), self.assertRaises(ValueError):
                study.validate(changed)

    def test_application_origins_ignore_query_labels_and_keep_feature_gaps_explicit(self):
        calendar = pd.bdate_range("2016-01-01", periods=40)
        features = pd.DataFrame(1.0, index=calendar, columns=study.sf.ALL_FEATURES)
        features["feature_cutoff_date"] = pd.Series(calendar, index=calendar).shift(1)
        targets = pd.DataFrame({"y": np.nan}, index=calendar)
        p = protocol()["index"]
        expected = calendar[calendar >= "2016-01-04"]
        self.assertTrue(study.application_origins(features, targets, p).equals(expected))
        targets["y"] = 1.0
        self.assertTrue(study.application_origins(features, targets, p).equals(expected))
        features.loc[calendar[4], study.sf.ALL_FEATURES[1]] = np.nan
        self.assertTrue(
            study.application_origins(features, targets, p).equals(expected.drop(calendar[4]))
        )
        features.loc[calendar[5], "feature_cutoff_date"] = calendar[3]
        with self.assertRaisesRegex(ValueError, "previous-session"):
            study.application_origins(features, targets, p)

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

    def test_both_controls_both_effects_and_strict_p_and_stability(self):
        rows = [passing(c) for c in ["frozen_baseline", "recent_frequency"]]
        self.assertEqual(study.candidate_leads(rows), ["pooled"])
        for fault in ["effect", "wave", "family", "stability"]:
            bad = copy.deepcopy(rows)
            if fault == "effect":
                bad[0]["phases"][0]["delta"] = -0.000499
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
        self.assertEqual((m["hypothesis_count"], m["cumulative_hypothesis_count"]), (2, 121))
        self.assertEqual(m["leads"], [])
        self.assertTrue(
            all(
                r["p_conservative"] == r["p_holm_wave"] == r["p_holm_cumulative"] == 1
                for r in m["rows"]
            )
        )

    def test_brier_factored_difference_and_endpoint_truth(self):
        y = np.array([0.0, 1.0, 0.0, 1.0])
        a = np.array([0.0, 1.0, 0.2, 0.8])
        b = np.array([1.0, 0.0, 0.3, 0.7])
        np.testing.assert_allclose(
            study.paired_difference(a, b, y), (a - y) ** 2 - (b - y) ** 2, rtol=1e-14, atol=0
        )
        for bad in [np.array([-0.1, 0.5, 0.5, 0.5]), np.array([np.nan, 0.5, 0.5, 0.5])]:
            with self.assertRaises(ValueError):
                study.paired_difference(bad, b, y)
        with self.assertRaises(ValueError):
            study.paired_difference(a, b, y + 0.1)

    def test_support_checks_precede_every_inference_call(self):
        original, calendar = panel()
        for first, last in [("2016-01-04", "2019-12-31"), ("2020-01-02", "2022-12-31")]:
            bad = original.copy()
            select = bad.origin.between(first, last)
            bad.loc[select, "y"] = 1.0
            bad["loss"] = (bad.y - bad.probability) ** 2
            with (
                patch.object(study, "paired_inference") as call,
                self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"),
            ):
                study.evaluate(bad, calendar, protocol())
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
                study.evaluate(bad, calendar, protocol())
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
            study.evaluate(bad, calendar, protocol())
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
            m = study.evaluate(rows, calendar, protocol())
        self.assertEqual(call.call_count, 4)
        self.assertEqual((m["hypothesis_count"], m["cumulative_hypothesis_count"]), (2, 121))
        self.assertEqual(m["new_forecasts"], 2 * rows.origin.nunique())
        self.assertEqual(m["preserved_forecasts"], rows.origin.nunique())
        self.assertEqual(m["combined_forecasts"], len(rows))
        self.assertEqual(m["new_monthly_fits"], 0)
        for phase in ["development", "evaluation"]:
            calibration = m["calibration"][phase]["frozen_baseline"]
            observed = rows.loc[rows.phase.eq(phase) & rows.model.eq("frozen_baseline"), "y"]
            self.assertEqual(calibration["n"], len(observed))
            self.assertAlmostEqual(calibration["mean_probability"], 0.65)
            self.assertAlmostEqual(calibration["observed_frequency"], float(observed.mean()))
            self.assertAlmostEqual(
                calibration["calibration_gap"], 0.65 - float(observed.mean())
            )
            self.assertAlmostEqual(
                calibration["brier"], float(((observed - 0.65) ** 2).mean())
            )
        self.assertTrue(all(len(r["phases"][-1]["stability"]) == 2 for r in m["rows"]))

    def test_mde_uses_probability_effect_not_prior_product_effect(self):
        result = {"hac126": {"mde80_nominal": 0.001}, "nominal_mde_effect_ratio": 1e7}
        with patch.object(study.old_inference, "paired_inference", return_value=result):
            value = study.paired_inference([0.0], [0.0], [0.0], protocol(), 0)
        self.assertEqual(value["nominal_mde_effect_ratio"], 2.0)


if __name__ == "__main__":
    unittest.main()
