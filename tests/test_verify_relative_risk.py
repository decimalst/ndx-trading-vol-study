"""Independent relative-risk contracts written before verifier implementation."""
from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from src import verify_relative_risk as verify


def markets():
    rng = np.random.default_rng(91260915)
    dates = pd.bdate_range("2014-01-02", periods=340, name="date")
    assets = []
    for scale in (.009, .006):
        close = 100 * np.exp(rng.normal(0, scale, len(dates)).cumsum())
        opening = close * np.exp(rng.normal(0, scale / 2, len(dates)))
        assets.append(pd.DataFrame({"open": opening, "high": np.maximum(opening, close) * 1.004,
                                    "low": np.minimum(opening, close) / 1.004, "close": close}, index=dates))
    iv = pd.DataFrame({name: np.exp(rng.normal(3, .2, len(dates))) for name in ("vxn", "vix", "vix9d", "vvix")}, index=dates)
    return *assets, iv


def section():
    return {"origin_start": "2014-12-01", "origin_end": "2015-01-30", "latest_target": "2015-02-02",
            "development": ["2014-12-01", "2014-12-31"], "evaluation": ["2015-01-01", "2015-01-30"],
            "development_target_available_by": "2014-12-31", "minimum_train": 150,
            "evaluation_stability": [["2015-01-01", "2015-01-15"], ["2015-01-16", "2015-01-30"]],
            "models": list(verify.MODELS), "baseline": list(verify.BASE)}


class MeasurementContracts(unittest.TestCase):
    def test_raw_gk_has_no_floor_and_is_invariant_to_within_session_scaling(self):
        qqq, _, _ = markets()
        raw = verify.raw_gk(qqq)
        altered = qqq * np.linspace(.5, 1.5, len(qqq))[:, None]
        np.testing.assert_allclose(raw, verify.raw_gk(altered), rtol=1e-12, atol=1e-15)
        qqq.iloc[50] = 100.
        self.assertEqual(verify.raw_gk(qqq).iloc[50], 0.)

    def test_raw_gk_requires_valid_positive_ohlc(self):
        qqq, _, _ = markets()
        for name, value in (("open", 0.), ("close", np.inf), ("high", .01)):
            bad = qqq.copy()
            bad.loc[bad.index[50], name] = value
            with self.assertRaises(ValueError):
                verify.raw_gk(bad)

    def test_floor_gate_precedes_pairwise_or_feature_support_mask(self):
        qqq, spx, _ = markets()
        qqq.iloc[50] = 100.
        spx.iloc[50] = np.nan
        audit = verify.measurement_audit(qqq, spx)
        self.assertEqual(audit["status"], "INSUFFICIENT_MEASUREMENT")
        self.assertEqual(audit["per_asset"]["qqq"]["floor_hit_rows"], 1)
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_MEASUREMENT"):
            verify.require_measurement(audit)

    def test_floor_gate_includes_early_history_before_scored_period(self):
        qqq, spx, iv = markets()
        qqq.iloc[1] = 100.
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_MEASUREMENT"):
            verify.feature_target_tables(qqq, spx, iv)

    def test_missing_measurement_is_unknown_and_never_becomes_flat_risk(self):
        qqq, spx, _ = markets()
        qqq.iloc[50] = np.nan
        audit = verify.measurement_audit(qqq, spx)
        verify.require_measurement(audit)
        self.assertEqual(audit["per_asset"]["qqq"]["missing_ohlc_rows"], 1)
        self.assertEqual(audit["per_asset"]["qqq"]["floor_hit_rows"], 0)


class FeatureAndTargetContracts(unittest.TestCase):
    def test_next_matching_session_signed_logratio_and_maturity(self):
        qqq, spx, iv = markets()
        features, targets = verify.feature_target_tables(qqq, spx, iv)
        expected = np.log(verify.raw_gk(qqq).iloc[201]) - np.log(verify.raw_gk(spx).iloc[201])
        self.assertAlmostEqual(targets.y.iloc[200], expected, places=13)
        self.assertEqual(targets.target_end.iloc[200], spx.index[201])
        self.assertEqual(targets.available_date.iloc[200], spx.index[201])
        self.assertEqual(features.feature_cutoff_date.iloc[200], spx.index[199])
        self.assertTrue(np.isnan(targets.y.iloc[-1]))

    def test_equal_intraday_risk_gives_valid_zero_response(self):
        qqq, _, iv = markets()
        _, targets = verify.feature_target_tables(qqq, qqq.copy(), iv)
        np.testing.assert_array_equal(targets.y.iloc[:-1], 0.)

    def test_all_marginal_risk_and_negative_return_controls_end_at_previous_session(self):
        qqq, spx, iv = markets()
        features, _ = verify.feature_target_tables(qqq, spx, iv)
        for prefix, frame in (("qqq", qqq), ("spx", spx)):
            raw = verify.raw_gk(frame)
            total = raw + np.log(frame.open / frame.close.shift()).pow(2)
            negative = np.maximum(-np.log(frame.close / frame.close.shift()), 0)
            for suffix, width in (("d", 1), ("w", 5), ("m", 22)):
                window = slice(200-width, 200)
                self.assertAlmostEqual(features[f"{prefix}_lg_{suffix}"].iloc[200], np.log(raw.iloc[window].mean()), places=13)
                self.assertAlmostEqual(features[f"{prefix}_lt_{suffix}"].iloc[200], np.log(total.iloc[window].mean()), places=13)
                self.assertAlmostEqual(features[f"{prefix}_neg_{suffix}"].iloc[200], negative.iloc[window].mean(), places=13)

    def test_correlation_is_strict_centered_intraday_window_and_lagged(self):
        qqq, spx, iv = markets()
        features, _ = verify.feature_target_tables(qqq, spx, iv)
        qday = np.log(qqq.close / qqq.open).iloc[178:200].to_numpy()
        sday = np.log(spx.close / spx.open).iloc[178:200].to_numpy()
        expected = np.dot(qday-qday.mean(), sday-sday.mean()) / np.sqrt(np.sum((qday-qday.mean())**2) * np.sum((sday-sday.mean())**2))
        self.assertAlmostEqual(features.corr22.iloc[200], expected, places=13)
        qqq.loc[qqq.index[190], "open"] = np.nan
        changed, _ = verify.feature_target_tables(qqq, spx, iv)
        self.assertTrue(np.isnan(changed.corr22.iloc[200]))

    def test_future_values_cannot_change_current_features(self):
        qqq, spx, iv = markets()
        before, _ = verify.feature_target_tables(qqq, spx, iv)
        future_q, future_s, future_iv = qqq.copy(), spx.copy(), iv.copy()
        future_q.iloc[200:] *= 2
        future_s.iloc[200:] *= 3
        future_iv.iloc[200:] *= 4
        after, _ = verify.feature_target_tables(future_q, future_s, future_iv)
        pd.testing.assert_series_equal(before.iloc[200], after.iloc[200])

    def test_missing_qqq_date_preserves_reference_and_does_not_use_next_common_date(self):
        qqq, spx, iv = markets()
        missing = qqq.index[201]
        features, target = verify.feature_target_tables(qqq.drop(missing), spx, iv)
        self.assertTrue(features.index.equals(spx.index))
        self.assertEqual(target.target_end.iloc[200], missing)
        self.assertTrue(np.isnan(target.y.iloc[200]))
        self.assertTrue(np.isnan(features.qqq_lt_d.iloc[203]))

    def test_extra_qqq_source_session_breaks_previous_close_identity(self):
        qqq, spx, iv = markets()
        spx = spx.drop(spx.index[190])
        features, _ = verify.feature_target_tables(qqq, spx, iv)
        day = qqq.index[192]
        self.assertTrue(np.isnan(features.loc[day, "qqq_lt_d"]))
        self.assertTrue(np.isnan(features.loc[day, "qqq_neg_d"]))
        self.assertTrue(np.isfinite(features.loc[day, "qqq_lg_d"]))

    def test_zero_correlation_denominator_is_unknown_and_roundoff_is_not_clipped(self):
        self.assertTrue(np.isnan(verify.correlation_value(np.ones(22), np.arange(22.))))
        self.assertEqual(verify.checked_correlation(1+5e-13), 1+5e-13)
        with self.assertRaises(ValueError):
            verify.checked_correlation(1+2e-12)

    def test_nonbinary_constant_correlation_window_stays_unknown_despite_mean_roundoff(self):
        for constant in (.1, .2, .3):
            self.assertTrue(np.isnan(verify.correlation_value(np.full(22, constant), np.full(22, constant))))
            self.assertTrue(np.isnan(verify.correlation_value(np.full(22, constant), np.arange(22.))))


class FitContracts(unittest.TestCase):
    def test_full_synthetic_two_month_walkforward_and_coefficient_reconstruction(self):
        from src.relative_risk_models import forecast_panel
        qqq, spx, iv = markets()
        f, t = verify.feature_target_tables(qqq, spx, iv)
        forecasts, fits = forecast_panel(f, t, section())
        result = verify.verify_forecasts(f, t, forecasts, fits, {"index": section()})
        self.assertEqual(result["monthly_fits_verified"], 2)
        self.assertEqual(result["models_verified"], 3)
        bad = forecasts.copy()
        bad.loc[0, "prediction"] += .01
        with self.assertRaises(AssertionError):
            verify.verify_forecasts(f, t, bad, fits, {"index": section()})

    def test_phase_inference_uses_signed_target_mse_and_fixed_seeds(self):
        dates = pd.bdate_range("2017-01-02", periods=150)
        rng = np.random.default_rng(1286)
        target = rng.normal(size=len(dates))
        panel = pd.concat([pd.DataFrame({"origin": dates, "model": model, "y": target,
                                        "prediction": rng.normal(0, .2, len(dates)),
                                        "available_date": dates + pd.offsets.BDay()}) for model in verify.MODELS])
        protocol = {"index": {"development": ["2017-01-01", "2017-12-31"], "development_target_available_by": "2017-12-31"},
                    "inference": {"blocks": [21, 63, 126], "seed": 20260915, "bootstrap_draws": 99}}
        with patch.object(verify, "explicit_bootstrap_means", wraps=verify.explicit_bootstrap_means) as sampled:
            result = verify.phase_statistics(panel, "baseline", "development", 0, protocol)
        self.assertEqual([call.args[3] for call in sampled.call_args_list], [20260936, 20260978, 20261041])
        self.assertAlmostEqual(result["gain_relative"], 1-result["candidate_loss"]/result["control_loss"])
        self.assertEqual(result["p_conservative"], max(result["hac126"]["p"], *(row["p"] for row in result["block_inference"].values())))

    def test_subtraction_of_matched_component_ridge_forecasts_is_algebraically_redundant(self):
        rng = np.random.default_rng(998)
        x = np.c_[np.ones(200), rng.normal(size=(200, 4))]
        yq, ys = rng.normal(size=200), rng.normal(size=200)
        difference, _ = verify.ridge_prediction(x, yq-ys, x[:10])
        q, _ = verify.ridge_prediction(x, yq, x[:10])
        s, _ = verify.ridge_prediction(x, ys, x[:10])
        np.testing.assert_allclose(difference, q-s, rtol=1e-12, atol=1e-12)

    def test_meanloss_ridge_penalty_invariant_to_repeated_training_rows(self):
        rng = np.random.default_rng(973)
        x = np.c_[np.ones(150), rng.normal(size=(150, 3))]
        y = rng.normal(size=150)
        p, _ = verify.ridge_prediction(x, y, x[:6])
        repeated, _ = verify.ridge_prediction(np.tile(x, (2, 1)), np.tile(y, 2), x[:6])
        np.testing.assert_allclose(p, repeated, rtol=1e-12, atol=1e-12)

    def test_missing_query_label_never_moves_monthly_feature_application(self):
        qqq, spx, iv = markets()
        f, t = verify.feature_target_tables(qqq, spx, iv)
        applications, scored, _ = verify.eligible_entries(f, t, section())
        t.loc[applications[0], "y"] = np.nan
        after, changed, _ = verify.eligible_entries(f, t, section())
        self.assertTrue(applications.equals(after))
        self.assertEqual(len(scored)-len(changed), 1)

    def test_zero_and_negative_response_rows_are_eligible_and_labels_mature_before_fit(self):
        qqq, spx, iv = markets()
        f, t = verify.feature_target_tables(qqq, spx, iv)
        t.y = 0.
        t.iloc[-1, t.columns.get_loc("y")] = np.nan
        applications, scored, complete = verify.eligible_entries(f, t, section())
        self.assertGreater(len(scored), 0)
        entry = applications[0]
        selected = verify.training_mask(complete, t, entry, f.index)
        position = f.index.get_loc(entry)
        self.assertTrue(selected.iloc[position-2])
        self.assertFalse(selected.iloc[position-1])
        t.loc[scored, "y"] = -1.
        self.assertTrue(verify.eligible_entries(f, t, section())[1].equals(scored))

    def test_two_controls_and_one_candidate_with_fixed_effect_threshold(self):
        self.assertEqual(verify.MODELS, ("mean", "baseline", "correlation"))
        self.assertEqual(len(verify.BASE), 27)
        self.assertEqual(len(verify.ALL_FEATURES), 28)
        self.assertEqual(verify.WAVE_ALPHA, .05 / (9 * 10))
        phases = [{"gain_relative": .0025, "delta": -.1, "stability": []},
                  {"gain_relative": .003, "delta": -.2, "stability": [{"delta": -.1}, {"delta": -.1}]}]
        self.assertTrue(verify.effect_passes(phases))
        failed = deepcopy(phases)
        failed[1]["gain_relative"] = .002499
        self.assertFalse(verify.effect_passes(failed))


class PublicationContracts(unittest.TestCase):
    def test_protocol_protects_signed_objective_source_fence_and_numerical_tolerances(self):
        protocol = yaml.safe_load((verify.ROOT / "relative_risk.yaml").read_text())
        verify.validate_protocol(protocol)
        for block, key, value in (("index", "source_end", "2025-11-03"), ("measurement", "gk_floor", 1e-12),
                                   ("correlation", "window", 21), ("verification", "gradient_tolerance", 1e-8)):
            changed = deepcopy(protocol)
            changed[block][key] = value
            with self.assertRaises(AssertionError):
                verify.validate_protocol(changed)

    def test_source_reader_filters_before_parsing_future_numerics(self):
        qqq, spx, _ = markets()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qqq.iloc[:3].to_parquet(root / "qqq.parquet")
            spx.iloc[:3].to_parquet(root / "spx.parquet")
            sources = {"daily": "spx.parquet", "qqq": "qqq.parquet"}
            for name in ("vxn", "vix", "vix9d", "vvix"):
                field = "VVIX" if name == "vvix" else "CLOSE"
                sources[name] = name + ".csv"
                (root / sources[name]).write_text(f"DATE,{field}\n01/02/2014,20\n01/03/2014,21\n11/03/2025,protected-nonnumeric\n")
            protocol = {"sources": sources, "index": {"source_end": "2025-10-20", "sealed_start": "2025-11-03"}}
            q, s, iv, audit = verify.load_source_tables(root, protocol)
            self.assertEqual(len(iv), 2)
            self.assertEqual(len(q), 3)
            self.assertEqual(len(s), 3)
            self.assertFalse(audit["numeric_post_cutoff_values_parsed"])

    def test_measurement_failure_can_be_verified_without_forecast_artifacts(self):
        qqq, spx, _ = markets()
        qqq.iloc[1] = 100.
        measurement = verify.measurement_audit(qqq, spx)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report, output = root / "reports/relative_risk", root / "data/relative_risk"
            report.mkdir(parents=True)
            output.mkdir(parents=True)
            metrics = {"status": "UNEVALUABLE", "whole_wave_aborted": True, "leads": [], "hypothesis_count": 2,
                       "cumulative_hypothesis_count": 110, "protocol_sha256": "a"*64,
                       "rows": [{"study": "relative_risk", "candidate": a, "control": b, "score": "mse", "horizon": 1,
                                 "status": "INSUFFICIENT_DATA", "phases": [], "p_conservative": 1., "p_holm_wave": 1., "p_holm_cumulative": 1.}
                                for a, b in verify.COMPARISONS]}
            (report / "failure.json").write_text(json.dumps(metrics))
            prior = [{"source_row_index": number} for number in range(108)]
            ledger = [{"event": "inherited", **row} for row in prior]
            ledger += [{"event": "registered", "candidate": a, "control": b, "score": "mse", "horizon": 1, "protocol_sha256": "a"*64}
                       for a, b in verify.COMPARISONS]
            ledger += [{"event": "unevaluable", **row} for row in metrics["rows"]]
            (report / "trial_ledger.jsonl").write_text("\n".join(json.dumps(row) for row in ledger) + "\n")
            result = verify.verify_terminal_measurement(root, measurement, metrics, prior)
            self.assertEqual(result["status"], "VERIFIED_INSUFFICIENT_MEASUREMENT")
            (output / "features.parquet").write_text("Unexpected post-gate artifact")
            with self.assertRaises(AssertionError):
                verify.verify_terminal_measurement(root, measurement, metrics, prior)

    def test_verifier_failure_replaces_canonical_success_and_retains_diagnostic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "reports/relative_risk"
            report.mkdir(parents=True)
            original = {"leads": ["correlation"], "protocol_sha256": "a"*64, "rows": []}
            (report / "metrics.json").write_text(json.dumps(original))
            (report / "results.md").write_text("A lead")
            with (patch.object(verify, "verify", side_effect=AssertionError("synthetic mismatch")), self.assertRaises(AssertionError)):
                verify.verify_with_failure_guard(root)
            result = json.loads((report / "metrics.json").read_text())
            self.assertEqual(result["status"], "UNEVALUABLE")
            self.assertEqual(result["leads"], [])
            self.assertEqual(len(result["rows"]), 2)
            self.assertTrue(all(row["p_conservative"] == row["p_holm_wave"] == row["p_holm_cumulative"] == 1. for row in result["rows"]))
            self.assertEqual(json.loads((report / "unpublished_scored_metrics.json").read_text())["scored_metrics"], original)
            self.assertEqual(json.loads((report / "verification.json").read_text())["status"], "FAILED")


if __name__ == "__main__":
    unittest.main()
