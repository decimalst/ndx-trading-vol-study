"""Synthetic independent joint-risk contracts, written before implementation."""

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

from src import verify_joint_risk as verify


def markets():
    rng = np.random.default_rng(91360916)
    dates = pd.bdate_range("2014-01-02", periods=340, name="date")
    assets = []
    for scale in (0.009, 0.006):
        close = 100 * np.exp(rng.normal(0, scale, len(dates)).cumsum())
        opening = close * np.exp(rng.normal(0, scale / 2, len(dates)))
        assets.append(
            pd.DataFrame(
                {
                    "open": opening,
                    "high": np.maximum(opening, close) * 1.004,
                    "low": np.minimum(opening, close) / 1.004,
                    "close": close,
                },
                index=dates,
            )
        )
    iv = pd.DataFrame(
        {
            name: np.exp(rng.normal(3, 0.2, len(dates)))
            for name in ("vxn", "vix", "vix9d", "vvix")
        },
        index=dates,
    )
    return *assets, iv


def section():
    return {
        "origin_start": "2014-12-01",
        "origin_end": "2014-12-30",
        "latest_target": "2015-01-02",
        "development": ["2014-12-01", "2014-12-15"],
        "evaluation": ["2014-12-16", "2014-12-30"],
        "development_target_available_by": "2014-12-15",
        "minimum_train": 150,
        "evaluation_stability": [["2014-12-16", "2014-12-22"], ["2014-12-23", "2014-12-30"]],
        "models": list(verify.MODELS),
        "all_features": list(verify.ALL_FEATURES),
    }


class MatrixContracts(unittest.TestCase):
    def test_score_matches_direct_matrix_cholesky(self):
        e = np.array([[0.2, -0.3], [0.0, 0.0], [-2.0, -1.0]])
        h = np.array([[0.4, 0.3], [1.0, 2.0], [2.0, 3.0]])
        rho = np.array([0.7, -0.2, -0.8])
        expected = []
        for vector, diag, r in zip(e, h, rho):
            matrix = (
                np.diag(np.sqrt(diag))
                @ np.array([[1.0, r], [r, 1.0]])
                @ np.diag(np.sqrt(diag))
            )
            chol = np.linalg.cholesky(matrix)
            expected.append(
                2 * np.log(np.diag(chol)).sum()
                + np.linalg.norm(np.linalg.solve(chol, vector)) ** 2
            )
        np.testing.assert_allclose(
            verify.matrix_score(e, h, rho), expected, rtol=1e-13, atol=1e-13
        )

    def test_asset_order_and_return_units_preserve_paired_score_difference(self):
        e = np.array([[0.2, -0.1], [-0.7, 0.8]])
        h = np.array([[0.04, 0.03], [0.2, 0.4]])
        r0, r1 = np.array([0.4, -0.3]), np.array([0.8, 0.1])
        gap = verify.matrix_score(e, h, r1) - verify.matrix_score(e, h, r0)
        np.testing.assert_allclose(
            gap,
            verify.matrix_score(e[:, ::-1], h[:, ::-1], r1)
            - verify.matrix_score(e[:, ::-1], h[:, ::-1], r0),
            atol=1e-12,
        )
        scale = np.array([100.0, 0.01])
        np.testing.assert_allclose(
            gap,
            verify.matrix_score(e * scale, h * scale**2, r1)
            - verify.matrix_score(e * scale, h * scale**2, r0),
            atol=1e-12,
        )

    def test_zero_and_rank_one_targets_are_valid_without_jitter(self):
        answer = verify.matrix_score(
            np.array([[0.0, 0.0], [1.0, -1.0]]), np.ones((2, 2)), np.array([0.3, 0.4])
        )
        self.assertTrue(np.isfinite(answer).all())
        self.assertAlmostEqual(answer[0], np.log(1 - 0.3**2))

    def test_invalid_diagonal_and_singular_correlation_rejected(self):
        for diag, corr in (
            ([0.0, 1.0], 0.1),
            ([1.0, -1.0], 0.1),
            ([1.0, 1.0], 1.0),
            ([np.inf, 1.0], 0.2),
        ):
            with self.assertRaises(ValueError):
                verify.matrix_score(np.ones((1, 2)), np.array([diag]), np.array([corr]))

    def test_dependence_gradient_matches_central_difference(self):
        rng = np.random.default_rng(843)
        innovation, z = rng.normal(size=(60, 2)), rng.normal(size=60)
        for a in (-2.0, 0.3, 2.0):
            for b in (-1.0, 0.0, 0.6):
                value, gradient = verify.dependence_objective(b, innovation, z, a)
                step = 1e-5
                derivative = (
                    verify.dependence_objective(b + step, innovation, z, a)[0]
                    - verify.dependence_objective(b - step, innovation, z, a)[0]
                ) / (2 * step)
                self.assertTrue(np.isfinite(value))
                self.assertAlmostEqual(gradient, derivative, delta=2e-6 * (1 + abs(gradient)))

    def test_curvature_bound_dominates_synthetic_numerical_second_derivative(self):
        rng = np.random.default_rng(454)
        innovation, z = rng.normal(size=(30, 2)), rng.normal(size=30)
        bound = verify.curvature_bound(-0.6, 0.8, innovation, z, 0.3)
        for b in np.linspace(-0.6, 0.8, 31):
            step = 1e-5
            numerical = (
                verify.dependence_objective(b + step, innovation, z, 0.3)[1]
                - verify.dependence_objective(b - step, innovation, z, 0.3)[1]
            ) / (2 * step)
            self.assertLessEqual(abs(numerical), bound)

    def test_constant_solver_includes_multiple_stationary_points_and_endpoints(self):
        # A=.2,B=0 makes rho=0 a maximum and two nonzero minima.
        innovation = np.sqrt(0.1) * np.array(
            [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]
        )
        fit = verify.constant_optimum(innovation)
        self.assertAlmostEqual(abs(0.995 * np.tanh(fit["parameter"])), np.sqrt(0.8), places=9)
        self.assertGreaterEqual(len(fit["candidates"]), 5)
        self.assertAlmostEqual(fit["objective"], np.log(0.2) + 1, places=10)

    def test_projected_kkt_has_correct_boundary_signs(self):
        self.assertEqual(verify.projected_gradient(-4.0, 3.0), 0.0)
        self.assertEqual(verify.projected_gradient(4.0, -3.0), 0.0)
        self.assertEqual(verify.projected_gradient(-4.0, -3.0), -3.0)
        self.assertEqual(verify.projected_gradient(4.0, 3.0), 3.0)

    def test_shared_wrong_diagonals_can_reward_false_dynamic_correlation(self):
        signs = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
        innovation = np.vstack([np.sqrt(0.1) * signs, signs, signs, signs])
        z = np.r_[np.full(4, np.sqrt(3)), np.full(12, -1 / np.sqrt(3))]
        self.assertAlmostEqual(
            verify.constant_optimum(innovation)["parameter"], 0.0, places=10
        )
        zero = verify.dependence_objective(0.0, innovation, z, 0.0)[0]
        self.assertLess(verify.dependence_objective(0.01, innovation, z, 0.0)[0], zero)
        self.assertAlmostEqual(np.mean(innovation[:, 0] * innovation[:, 1]), 0.0)


class FeatureContracts(unittest.TestCase):
    def test_nonfinite_day_return_with_missing_high_is_rejected_at_measurement_gate(self):
        qqq, spx, _ = markets()
        qqq.loc[qqq.index[100], ["open", "close", "high", "low"]] = [
            1e-308,
            1e308,
            np.nan,
            1e-308,
        ]
        with np.errstate(over="ignore", invalid="ignore"), self.assertRaises(ValueError):
            verify.measurement_audit(qqq, spx)

    def test_next_paired_raw_intraday_return_targets_and_maturity(self):
        qqq, spx, iv = markets()
        f, t = verify.feature_target_tables(qqq, spx, iv)
        self.assertEqual(len(verify.ALL_FEATURES), 34)
        self.assertAlmostEqual(
            t.y_qqq.iloc[200], np.log(qqq.close.iloc[201] / qqq.open.iloc[201])
        )
        self.assertAlmostEqual(
            t.y_spx.iloc[200], np.log(spx.close.iloc[201] / spx.open.iloc[201])
        )
        self.assertEqual(t.available_date.iloc[200], spx.index[201])
        self.assertEqual(f.feature_cutoff_date.iloc[200], spx.index[199])

    def test_signed_intraday_histories_use_strict_previous_reference_sessions(self):
        qqq, spx, iv = markets()
        f, _ = verify.feature_target_tables(qqq, spx, iv)
        for name, frame in (("qqq", qqq), ("spx", spx)):
            values = np.log(frame.close / frame.open)
            for suffix, width in (("d", 1), ("w", 5), ("m", 22)):
                self.assertAlmostEqual(
                    f[f"{name}_day_{suffix}"].iloc[200], values.iloc[200 - width : 200].mean()
                )
        qqq.iloc[195] = np.nan
        changed, _ = verify.feature_target_tables(qqq, spx, iv)
        self.assertTrue(np.isnan(changed.qqq_day_w.iloc[200]))

    def test_future_mutation_cannot_change_present_features(self):
        qqq, spx, iv = markets()
        original, _ = verify.feature_target_tables(qqq, spx, iv)
        qqq.iloc[200:] *= 2
        spx.iloc[200:] *= 3
        iv.iloc[200:] *= 4
        changed, _ = verify.feature_target_tables(qqq, spx, iv)
        pd.testing.assert_series_equal(original.iloc[200], changed.iloc[200])

    def test_zero_day_return_is_valid_and_missing_paired_date_is_unknown(self):
        qqq, spx, iv = markets()
        qqq.loc[qqq.index[201], "close"] = qqq.open.iloc[201]
        _, t = verify.feature_target_tables(qqq, spx, iv)
        self.assertEqual(t.y_qqq.iloc[200], 0.0)
        _, changed = verify.feature_target_tables(qqq.drop(qqq.index[201]), spx, iv)
        self.assertTrue(np.isnan(changed.y_qqq.iloc[200]))
        self.assertEqual(changed.available_date.iloc[200], spx.index[201])

    def test_square_center_uses_training_only_and_zero_scale_aborts(self):
        qqq, spx, iv = markets()
        f, _ = verify.feature_target_tables(qqq, spx, iv)
        train, query = f.iloc[30:200], f.iloc[200:203].copy()
        a, b, audit = verify.transform(train, query)
        query.corr22 = 99.0
        aa, _, newer = verify.transform(train, query)
        pd.testing.assert_frame_equal(a, aa)
        self.assertEqual(audit, newer)
        np.testing.assert_allclose(
            b.corr22_centered_sq, (f.corr22.iloc[200:203] - train.corr22.mean()) ** 2
        )
        train = train.copy()
        train.corr22 = 0.1
        with self.assertRaises(ValueError):
            verify.transform(train, query)

    def test_label_missingness_does_not_shift_monthly_fit_date(self):
        qqq, spx, iv = markets()
        f, t = verify.feature_target_tables(qqq, spx, iv)
        apps, scored, _ = verify.eligible_entries(f, t, section())
        t.loc[apps[0], "y_spx"] = np.nan
        after, changed, _ = verify.eligible_entries(f, t, section())
        self.assertTrue(apps.equals(after))
        self.assertEqual(len(changed), len(scored) - 1)

    def test_mature_training_labels_are_joint_and_strictly_before_fit(self):
        qqq, spx, iv = markets()
        f, t = verify.feature_target_tables(qqq, spx, iv)
        entry = f.index[200]
        mask = verify.training_mask(f, t, entry)
        self.assertTrue(mask.iloc[198])
        self.assertFalse(mask.iloc[199])
        t.loc[t.index[100], "y_qqq"] = np.nan
        self.assertFalse(verify.training_mask(f, t, entry).iloc[100])


class PublicationContracts(unittest.TestCase):
    def test_nonfinite_json_metadata_preserves_bytes_and_failure_ledger(self):
        for raw in (
            '{"protocol_sha256": NaN, "rows": []}',
            '{"protocol_sha256": "a", "diagnostic": Infinity}',
        ):
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                report = root / "reports/joint_risk"
                report.mkdir(parents=True)
                (report / "metrics.json").write_text(raw)
                verify.invalidate_publication(root, ValueError("nonfinite prior JSON"))
                result = json.loads((report / "metrics.json").read_text())
                self.assertEqual(result["status"], "UNEVALUABLE")
                self.assertEqual(result["leads"], [])
                events = [
                    json.loads(line)
                    for line in (report / "trial_ledger.jsonl").read_text().splitlines()
                ]
                self.assertEqual(
                    [row["event"] for row in events],
                    ["verification_failed", "verification_failed"],
                )
                self.assertEqual((report / "unpublished_invalid_metrics.txt").read_text(), raw)

    def test_source_reader_filters_future_strings_and_matches_producer_audit(self):
        from src.joint_risk_features import load_sources

        qqq, spx, _ = markets()
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            qqq.to_parquet(root / "qqq.parquet")
            spx.to_parquet(root / "spx.parquet")
            sources = {"qqq": "qqq.parquet", "daily": "spx.parquet"}
            for field in ("vxn", "vix", "vix9d", "vvix"):
                sources[field] = field + ".csv"
                column = "VVIX" if field == "vvix" else "CLOSE"
                (root / sources[field]).write_text(
                    f"DATE,{column}\n01/02/2014,20\n01/03/2014,21\n11/03/2025,protected-string\n"
                )
            protocol = {
                "sources": sources,
                "index": {"source_end": "2025-10-20", "sealed_start": "2025-11-03"},
            }
            q, s, iv, audit = verify.load_source_tables(root, protocol)
            pq, ps, piv, paudit = load_sources(protocol, root)
            pd.testing.assert_frame_equal(q, pq)
            pd.testing.assert_frame_equal(s, ps)
            pd.testing.assert_frame_equal(iv, piv, check_like=True)
            self.assertEqual(audit, paudit)
            self.assertEqual(len(iv), 2)

    def test_measurement_failure_has_distinct_verified_status_and_no_model_artifacts(self):
        qqq, spx, _ = markets()
        qqq.iloc[1] = 100.0
        measurement = verify.measurement_audit(qqq, spx)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            report, output = root / "reports/joint_risk", root / "data/joint_risk"
            report.mkdir(parents=True)
            output.mkdir(parents=True)
            metrics = {
                "status": "UNEVALUABLE",
                "whole_wave_aborted": True,
                "leads": [],
                "hypothesis_count": 2,
                "cumulative_hypothesis_count": 112,
                "protocol_sha256": "a" * 64,
                "rows": [
                    {
                        "study": "joint_risk",
                        "candidate": a,
                        "control": b,
                        "score": "matrix_qlike",
                        "horizon": 1,
                        "status": "INSUFFICIENT_DATA",
                        "phases": [],
                        "p_conservative": 1.0,
                        "p_holm_wave": 1.0,
                        "p_holm_cumulative": 1.0,
                    }
                    for a, b in verify.COMPARISONS
                ],
            }
            prior = [{"source_row_index": i} for i in range(110)]
            ledger = [{"event": "inherited", **row} for row in prior]
            ledger += [
                {
                    "event": "registered",
                    "candidate": a,
                    "control": b,
                    "score": "matrix_qlike",
                    "horizon": 1,
                    "protocol_sha256": "a" * 64,
                }
                for a, b in verify.COMPARISONS
            ]
            ledger += [{"event": "unevaluable", **row} for row in metrics["rows"]]
            (report / "failure.json").write_text(json.dumps(metrics))
            (report / "trial_ledger.jsonl").write_text(
                "\n".join(json.dumps(row) for row in ledger) + "\n"
            )
            result = verify.verify_terminal_measurement(root, measurement, metrics, prior)
            self.assertEqual(result["status"], "VERIFIED_INSUFFICIENT_MEASUREMENT")
            (output / "fits.json").write_text("[]")
            with self.assertRaises(AssertionError):
                verify.verify_terminal_measurement(root, measurement, metrics, prior)

    def test_exact_protocol_and_mutation_rejection(self):
        protocol = yaml.safe_load((verify.ROOT / "joint_risk.yaml").read_text())
        verify.validate_protocol(protocol)
        for group, field, value in (
            ("index", "source_end", "2025-11-03"),
            ("index", "effect_threshold_absolute", 0.004),
            ("dependence", "rho_max", 0.999),
            ("dependence", "global_value_gap", 1e-6),
            ("comparisons", "cumulative_hypotheses", 110),
        ):
            bad = deepcopy(protocol)
            bad[group][field] = value
            with self.assertRaises(AssertionError):
                verify.validate_protocol(bad)

    def test_temporary_tree_full_publication_entrypoint_checks_paired_target_schema(self):
        qqq, spx, iv = markets()
        f, t = verify.feature_target_tables(qqq, spx, iv)
        protocol_text = (verify.ROOT / "joint_risk.yaml").read_text()
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            report = root / "reports/joint_risk"
            output = root / "data/joint_risk"
            report.mkdir(parents=True)
            output.mkdir(parents=True)
            (root / "joint_risk.yaml").write_text(protocol_text)
            signature = verify.digest(root / "joint_risk.yaml")
            manifest = {
                "protocol_sha256": signature,
                "code": {},
                "inputs": {},
                "preserved": {},
            }
            (report / "manifest.json").write_text(json.dumps(manifest))
            source = {"synthetic_only": True}
            (output / "source_audit.json").write_text(json.dumps(source))
            (output / "measurement_audit.json").write_text(
                json.dumps(verify.measurement_audit(qqq, spx))
            )
            f.to_parquet(output / "features.parquet")
            t.to_parquet(output / "targets.parquet")
            pd.DataFrame({"synthetic": [1.0]}).to_parquet(output / "forecasts.parquet")
            (output / "fits.json").write_text("[]")
            protocol = yaml.safe_load(protocol_text)
            (report / "metrics.json").write_text(
                json.dumps(
                    {
                        "protocol_sha256": signature,
                        "evidence_class": protocol["evidence_class"],
                        "inherited_rows": [],
                    }
                )
            )
            with (
                patch.object(verify, "verify_manifest_coverage"),
                patch.object(
                    verify, "load_source_tables", return_value=(qqq, spx, iv, source)
                ),
                patch.object(verify, "verify_forecasts", return_value={"synthetic": True}),
                patch.object(verify, "verify_metrics", return_value={}),
                patch.object(verify, "verify_ledger", return_value={}),
            ):
                result = verify.verify(root)
                self.assertEqual(result["status"], "VERIFIED")
                self.assertEqual(result["raw_feature_columns_verified"], 34)
                bad = t.rename(columns={"y_qqq": "y"})
                bad.to_parquet(output / "targets.parquet")
                with self.assertRaises(AssertionError):
                    verify.verify(root)

    def test_absolute_score_effect_threshold_cannot_be_replaced_by_percentage(self):
        good = [
            {"delta": -0.005, "stability": []},
            {"delta": -0.006, "stability": [{"delta": -0.001}, {"delta": -0.1}]},
        ]
        self.assertTrue(verify.effect_passes(good))
        good[0]["delta"] = -0.004999
        self.assertFalse(verify.effect_passes(good))

    def test_discrepancy_invalidates_both_comparisons_and_preserves_diagnostics(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            report = root / "reports/joint_risk"
            report.mkdir(parents=True)
            original = {
                "leads": ["dynamic_correlation"],
                "protocol_sha256": "a" * 64,
                "rows": [],
            }
            (report / "metrics.json").write_text(json.dumps(original))
            with (
                patch.object(
                    verify, "verify", side_effect=AssertionError("synthetic discrepancy")
                ),
                self.assertRaises(AssertionError),
            ):
                verify.verify_with_failure_guard(root)
            result = json.loads((report / "metrics.json").read_text())
            self.assertEqual(result["status"], "UNEVALUABLE")
            self.assertEqual(result["cumulative_hypothesis_count"], 112)
            self.assertEqual(result["leads"], [])
            self.assertEqual(len(result["rows"]), 2)
            self.assertTrue(
                all(
                    row["p_conservative"]
                    == row["p_holm_wave"]
                    == row["p_holm_cumulative"]
                    == 1
                    for row in result["rows"]
                )
            )
            self.assertEqual(
                json.loads((report / "unpublished_scored_metrics.json").read_text())[
                    "scored_metrics"
                ],
                original,
            )
            self.assertEqual(
                json.loads((report / "verification.json").read_text())["status"], "FAILED"
            )

    def test_malformed_canonical_metrics_cannot_leave_human_success(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            report = root / "reports/joint_risk"
            report.mkdir(parents=True)
            (report / "metrics.json").write_text("{bad bytes")
            (report / "results.md").write_text("A lead")
            verify.invalidate_publication(root, ValueError("bad metrics"))
            self.assertIn("UNEVALUABLE", (report / "results.md").read_text())
            self.assertEqual(
                (report / "unpublished_invalid_metrics.txt").read_text(), "{bad bytes"
            )


class SavedAuditContracts(unittest.TestCase):
    def test_phase_inference_retains_negative_raw_scores_and_fixed_seeds(self):
        from src.joint_risk_search import paired_inference

        rng = np.random.default_rng(710)
        dates = pd.bdate_range("2016-01-04", periods=150)
        y = rng.normal(0, 0.01, size=(150, 2))
        h = np.full((150, 2), 0.0001)
        frames = []
        for model, rho in zip(verify.MODELS, (0.1, 0.2, 0.3)):
            loss = verify.matrix_score(y, h, np.full(150, rho))
            frames.append(
                pd.DataFrame(
                    {
                        "origin": dates,
                        "model": model,
                        "loss": loss,
                        "available_date": dates + pd.offsets.BDay(),
                        "y_qqq": y[:, 0],
                        "y_spx": y[:, 1],
                        "mu_qqq": 0.0,
                        "mu_spx": 0.0,
                        "h_qqq": h[:, 0],
                        "h_spx": h[:, 1],
                        "rho": rho,
                    }
                )
            )
        panel = pd.concat(frames, ignore_index=True)
        protocol = {
            "index": {
                "development": ["2016-01-01", "2016-12-31"],
                "development_target_available_by": "2016-12-31",
            },
            "inference": {
                "blocks": [21, 63, 126],
                "hac_lags": 126,
                "seed": 20260916,
                "bootstrap_draws": 99,
            },
        }
        with patch.object(
            verify, "explicit_bootstrap_means", wraps=verify.explicit_bootstrap_means
        ) as sampled:
            result = verify.phase_statistics(
                panel, "constant_correlation", "development", 0, protocol
            )
        self.assertEqual(
            [call.args[3] for call in sampled.call_args_list], [20260937, 20260979, 20261042]
        )
        self.assertLess(result["control_loss"], 0.0)
        self.assertNotIn("gain_relative", result)
        producer = paired_inference(frames[2].loss, frames[1].loss, protocol, 20260916)
        for field in producer:
            verify.same_tree(
                result[field],
                producer[field],
                "Independent synthetic matrix inference " + field,
            )

    def test_plot_requires_verified_current_complete_family_and_accepts_negative_scores(self):
        from src.plot_joint_risk import validated_rows

        metrics = {
            "protocol_sha256": "a" * 64,
            "rows": [
                {
                    "candidate": a,
                    "control": b,
                    "score": "matrix_qlike",
                    "phases": [
                        {
                            "name": name,
                            "delta": -0.01,
                            "control_loss": -10.0,
                            "ci95_envelope": [-0.02, 0.01],
                        }
                        for name in ("evaluation", "development")
                    ],
                }
                for a, b in reversed(verify.COMPARISONS)
            ],
        }
        verification = {"status": "VERIFIED", "protocol_sha256": "a" * 64}
        rows = validated_rows(metrics, verification, "a" * 64)
        self.assertEqual(rows[0]["control"], "constant_correlation")
        self.assertEqual(rows[0]["phases"][0]["name"], "development")
        for mode in ("failed", "hash", "missing", "nonfinite"):
            bad, check = deepcopy(metrics), deepcopy(verification)
            if mode == "failed":
                check["status"] = "FAILED"
            elif mode == "hash":
                bad["protocol_sha256"] = "b" * 64
            elif mode == "missing":
                bad["rows"].pop()
            else:
                bad["rows"][0]["phases"][0]["delta"] = np.nan
            with self.assertRaises(ValueError):
                validated_rows(bad, check, "a" * 64)

    def test_complete_global_certificate_and_missing_or_optimistic_leaf_rejection(self):
        from src.joint_risk_density import fit_dependence

        rng = np.random.default_rng(389)
        innovations = rng.normal(size=(70, 2))
        z = rng.normal(size=70)
        constant = fit_dependence(innovations)
        candidate = fit_dependence(innovations, z, constant=constant["parameter"])
        audits = {"constant_correlation": constant, "dynamic_correlation": candidate}
        verified = verify.verify_dependence(innovations, z, audits)
        self.assertEqual(verified["scalar_fits_verified"], 2)
        for mode in ("missing_leaf", "optimistic_leaf", "changed_intercept"):
            bad = deepcopy(audits)
            if mode == "missing_leaf":
                bad["dynamic_correlation"]["certificate_intervals"].pop()
            elif mode == "optimistic_leaf":
                bad["dynamic_correlation"]["certificate_intervals"][0]["lower_bound"] += 0.01
            else:
                bad["dynamic_correlation"]["constant"] += 0.01
            with self.assertRaises(AssertionError):
                verify.verify_dependence(innovations, z, bad)
        bad = deepcopy(audits)
        bad["constant_correlation"]["global_lower_bound"] = constant["objective"] + 1e-12
        bad["constant_correlation"]["global_upper_bound"] = constant["objective"] + 2e-12
        bad["constant_correlation"]["global_value_gap"] = 1e-12
        with self.assertRaises(AssertionError):
            verify.verify_dependence(innovations, z, bad)

    def test_synthetic_staged_fit_forecast_reconstruction_and_tamper_rejection(self):
        from src.joint_risk_models import forecast_panel

        qqq, spx, iv = markets()
        f, t = verify.feature_target_tables(qqq, spx, iv)
        panel, fits = forecast_panel(f, t, section())
        result = verify.verify_forecasts(f, t, panel, fits, {"index": section()})
        self.assertEqual(result["monthly_fits_verified"], 1)
        self.assertEqual(result["models_verified"], 3)
        for column in ("mu_qqq", "h_spx", "rho", "loss"):
            changed = panel.copy()
            changed.loc[0, column] += 0.001
            with self.assertRaises(AssertionError):
                verify.verify_forecasts(f, t, changed, fits, {"index": section()})


if __name__ == "__main__":
    unittest.main()
