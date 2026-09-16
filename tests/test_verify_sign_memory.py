"""Prewritten independent strict-sign, causal-window and convex-fit contracts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.special import expit

from src import verify_sign_memory as verify


class SignContracts(unittest.TestCase):
    def test_strict_sign_truth_table_ties_missing_and_tiny_nonzero_values(self):
        q = np.array([1.0, -1.0, 1.0, -1.0, 0.0, 1.0, 0.0, np.nan, 1e-250, -1e-250])
        s = np.array([1.0, -1.0, -1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1e-250, 1e-250])
        expected = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan, 1.0, 0.0])
        np.testing.assert_equal(verify.strict_agreement(q, s), expected)
        np.testing.assert_equal(verify.strict_agreement(-q, -s), expected)
        np.testing.assert_equal(verify.strict_agreement(q * 3, s * 5), expected)
        with self.assertRaises(ValueError):
            verify.strict_agreement([np.inf], [1.0])

    def test_paired_local_window_retains_missing_observations_and_predecessor_lag(self):
        dates = pd.bdate_range("2010-01-04", periods=50)
        q = pd.Series(np.where(np.arange(50) % 3, 1.0, -1.0), index=dates)
        s = pd.Series(np.where(np.arange(50) % 2, 1.0, -1.0), index=dates)
        result = verify.sign_history(q, s)
        self.assertTrue(result.iloc[:22].isna().all(axis=None))
        window_q, window_s = q.iloc[:22], s.iloc[:22]
        posq, pos_s = (window_q > 0).mean(), (window_s > 0).mean()
        negq, neg_s = (window_q < 0).mean(), (window_s < 0).mean()
        expected = posq * pos_s + negq * neg_s
        self.assertEqual(result.iloc[22]["independent22"], expected)
        self.assertAlmostEqual(
            result.iloc[22]["excess22"],
            verify.strict_agreement(window_q, window_s).mean() - expected,
        )
        changed = q.copy()
        changed.iloc[25] = np.nan
        bad = verify.sign_history(changed, s)
        self.assertTrue(bad.iloc[26:48].isna().all(axis=None))
        self.assertTrue(bad.iloc[48].notna().all())
        changed = q.copy()
        changed.iloc[22:] *= -100
        pd.testing.assert_series_equal(
            verify.sign_history(changed, s).iloc[22], result.iloc[22]
        )


class LogisticContracts(unittest.TestCase):
    def sample(self, n=240):
        rng = np.random.default_rng(20260919)
        x = np.c_[np.ones(n), rng.normal(size=(n, 5))]
        y = (rng.random(n) < expit(x @ np.array([0.2, 0.4, -0.1, 0.0, 0.2, -0.1]))).astype(
            float
        )
        return x, y

    def test_gradient_hessian_against_central_differences(self):
        x, y = self.sample()
        beta = np.array([0.3, 0.2, -0.1, 0.4, 0.05, -0.3])
        value, g, h = verify.logistic_objective(beta, x, y)
        self.assertTrue(np.isfinite(value))
        self.assertGreater(np.linalg.eigvalsh(h).min(), 0)
        step = 1e-5
        for j in range(len(beta)):
            delta = np.eye(len(beta))[j] * step
            plus = verify.logistic_objective(beta + delta, x, y)
            minus = verify.logistic_objective(beta - delta, x, y)
            self.assertAlmostEqual(g[j], (plus[0] - minus[0]) / (2 * step), delta=1e-9)
            np.testing.assert_allclose(
                h[:, j], (plus[1] - minus[1]) / (2 * step), rtol=1e-7, atol=1e-10
            )

    def test_extreme_logits_use_finite_stable_nll_and_nonzero_positive_curvature(self):
        x = np.array([[1.0], [1.0]])
        value, g, h = verify.logistic_objective(np.array([40.0]), x, np.ones(2))
        self.assertGreater(value, 0.0)
        self.assertLess(g[0], 0.0)
        self.assertGreater(h[0, 0], 0.0)
        value, g, h = verify.logistic_objective(np.array([1000.0]), x, np.array([0.0, 1.0]))
        self.assertEqual(value, 500.0)
        self.assertTrue(np.isfinite(g).all())
        self.assertTrue(np.isfinite(h).all())

    def test_independent_trust_region_fit_has_stationary_convex_objective(self):
        x, y = self.sample()
        result = verify.independent_baseline_fit(x, y)
        value, g, _ = verify.logistic_objective(result["beta"], x, y)
        self.assertLessEqual(np.max(abs(g)), 1e-8 + 1e-12)
        self.assertLess(value, verify.logistic_objective(np.zeros(x.shape[1]), x, y)[0])

    def test_independent_memory_bracket_covers_stationary_unique_optimum_and_flat_case(self):
        x, y = self.sample()
        eta = x @ np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0])
        m = x[:, 1] / 10
        result = verify.independent_memory_fit(eta, y, m)
        value, g, h = verify.memory_objective(result["b"], eta, y, m)
        self.assertLessEqual(abs(g), 1e-8 + 1e-12)
        self.assertGreaterEqual(h, 0.02)
        self.assertLessEqual(value, verify.memory_objective(0.0, eta, y, m)[0])
        flat = verify.independent_memory_fit(eta, y, np.zeros(len(y)))
        self.assertEqual(flat["b"], 0.0)

    def test_memory_derivatives_match_differences_and_bad_probabilistic_inputs_reject(self):
        x, y = self.sample()
        eta = x[:, 1]
        m = x[:, 2] / 3
        b = 0.35
        step = 1e-5
        _value, g, h = verify.memory_objective(b, eta, y, m)
        plus = verify.memory_objective(b + step, eta, y, m)
        minus = verify.memory_objective(b - step, eta, y, m)
        self.assertAlmostEqual(g, (plus[0] - minus[0]) / (2 * step), delta=1e-9)
        self.assertAlmostEqual(h, (plus[1] - minus[1]) / (2 * step), delta=1e-9)
        with self.assertRaises(ValueError):
            verify.logistic_objective(np.zeros(x.shape[1]), x, np.ones(len(y)) * 0.5)
        with self.assertRaises(ValueError):
            verify.memory_objective(b, eta + 1j, y, m)


class TransformAndFailureContracts(unittest.TestCase):
    def frames(self):
        rng = np.random.default_rng(24)
        frame = pd.DataFrame(
            rng.normal(size=(130, len(verify.ALL_FEATURES))),
            columns=verify.ALL_FEATURES,
            index=pd.bdate_range("2010-01-04", periods=130),
        )
        frame["const"] = 1.0
        frame.loc[:, verify.BOUNDED] = rng.uniform(size=(130, 5))
        frame[verify.MEMORY] /= 10
        return frame.iloc[:100].copy(), frame.iloc[100:].copy()

    def test_training_only_geometry_retains_constant_bounded_and_memory(self):
        train, app = self.frames()
        train["qqq_pos22"] = 0.1
        train[verify.MEMORY] = 0.125
        tx, ax, tm, am, audit = verify.transform(train, app)
        self.assertEqual(tx.shape, (100, 40))
        self.assertEqual(ax.shape, (30, 40))
        np.testing.assert_equal(tx[:, verify.BASE_COLUMNS.index("qqq_pos22")], 0.0)
        np.testing.assert_equal(tm, 0.0)
        np.testing.assert_equal(am, app.excess22 - 0.125)
        changed = app * 100
        changed["const"] = 1.0
        tx2, _, tm2, _, audit2 = verify.transform(train, changed)
        np.testing.assert_equal(tx, tx2)
        np.testing.assert_equal(tm, tm2)
        self.assertEqual(audit, audit2)
        train["corr22"] = 0.1
        with self.assertRaises(ValueError):
            verify.transform(train, app)

    def test_maturity_selection_preserves_unscored_application_and_binary_unknown(self):
        train, app = self.frames()
        f = pd.concat([train, app])
        idx = f.index
        f["feature_cutoff_date"] = pd.Series(idx, index=idx).shift()
        future = pd.Series(idx, index=idx).shift(-1)
        t = pd.DataFrame(
            {"y": np.arange(len(f)) % 2, "target_end": future, "available_date": future},
            index=idx,
        )
        entry = idx[100]
        mask = verify.training_mask(f, t, entry)
        self.assertFalse(mask.loc[idx[99]])
        self.assertTrue(mask.loc[idx[98]])
        t.loc[idx[98], "y"] = np.nan
        self.assertFalse(verify.training_mask(f, t, entry).loc[idx[98]])
        section = {
            "origin_start": str(idx[100].date()),
            "origin_end": str(idx[-1].date()),
            "development": [str(idx[100].date()), str(idx[110].date())],
            "development_target_available_by": str(idx[110].date()),
            "evaluation": [str(idx[111].date()), str(idx[-1].date())],
            "latest_target": str(idx[-1].date()),
        }
        application, scored = verify.eligible_entries(f, t, section)
        self.assertIn(idx[-1], application)
        self.assertNotIn(idx[-1], scored)
        self.assertNotIn(idx[110], scored)

    def test_failure_guard_retains_both_trials_for_valid_malformed_and_nonfinite_json(self):
        for payload in [
            '{"leads":["memory"],"rows":[]}',
            "{broken",
            '{"value":NaN}',
            '{"value":Infinity}',
        ]:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                report = root / "reports/sign_memory"
                report.mkdir(parents=True)
                (report / "metrics.json").write_text(payload)
                prior = root / "reports/target_aligned"
                prior.mkdir()
                (prior / "proof").write_bytes(b"unchanged")
                with (
                    patch.object(verify, "verify", side_effect=AssertionError("fault")),
                    self.assertRaises(AssertionError),
                ):
                    verify.verify_with_failure_guard(root)
                result = json.loads((report / "metrics.json").read_text())
                self.assertEqual(result["status"], "UNEVALUABLE")
                self.assertEqual(result["leads"], [])
                self.assertEqual(len(result["rows"]), 2)
                self.assertTrue(
                    all(
                        r[k] == 1
                        for r in result["rows"]
                        for k in ["p_conservative", "p_holm_wave", "p_holm_cumulative"]
                    )
                )
                events = [
                    json.loads(line)
                    for line in (report / "trial_ledger.jsonl").read_text().splitlines()
                ]
                self.assertEqual([e["event"] for e in events], ["verification_failed"] * 2)
                self.assertEqual((prior / "proof").read_bytes(), b"unchanged")

    def test_protocol_rejects_changed_support_optimizer_and_memory_window(self):
        from copy import deepcopy

        import yaml

        p = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "sign_memory.yaml").read_text()
        )
        verify.validate_protocol(p)
        for group, key, value in [
            ("features", "window", 21),
            ("inference", "minimum_phase_per_class", 29),
            ("fitting", "penalty", 0.02),
        ]:
            bad = deepcopy(p)
            bad[group][key] = value
            with self.assertRaises(ValueError):
                verify.validate_protocol(bad)
        bad = deepcopy(p)
        bad["verification"]["independent_optimizer"]["baseline_maximum_iterations"] = 501
        with self.assertRaises(ValueError):
            verify.validate_protocol(bad)


class AdmissionContracts(unittest.TestCase):
    def fixture(self, root):
        import yaml

        oldreport = root / "reports/target_aligned"
        olddata = root / "data/target_aligned"
        report = root / "reports/sign_memory"
        for folder in [oldreport, olddata, report, root / "src"]:
            folder.mkdir(parents=True, exist_ok=True)
        (root / "src/verify_target_aligned.py").write_text("previous synthetic verifier")
        (root / "raw.bin").write_bytes(b"original")
        (olddata / "forecasts.parquet").write_bytes(b"original issued bytes")
        (root / "target_aligned.yaml").write_text("evidence_class: synthetic\n")
        oldp = verify.digest(root / "target_aligned.yaml")
        oldv = verify.digest(root / "src/verify_target_aligned.py")
        oldm = {
            "protocol_sha256": oldp,
            "code": {"src/verify_target_aligned.py": oldv},
            "inputs": {"raw.bin": verify.digest(root / "raw.bin")},
            "preserved": {},
        }
        (oldreport / "manifest.json").write_text(json.dumps(oldm))
        record = {"status": "VERIFIED", "protocol_sha256": oldp, "verifier_sha256": oldv}
        (oldreport / "verification.json").write_text(json.dumps(record))
        p = {
            "upstream": {
                "protocol": "target_aligned.yaml",
                "reports": "reports/target_aligned",
                "data": "data/target_aligned",
                "protocol_sha256": oldp,
                "verifier_sha256": oldv,
                "required_status": "VERIFIED",
                "manifest_sha256": verify.digest(oldreport / "manifest.json"),
                "verification_sha256": verify.digest(oldreport / "verification.json"),
            }
        }
        (root / "sign_memory.yaml").write_text(yaml.safe_dump(p))
        inputs = {**oldm["inputs"], "target_aligned.yaml": oldp}
        inputs.update(
            {
                str(path.relative_to(root)): verify.digest(path)
                for folder in [oldreport, olddata]
                for path in folder.rglob("*")
                if path.is_file()
            }
        )
        manifest = {
            "protocol_sha256": verify.digest(root / "sign_memory.yaml"),
            "code": oldm["code"],
            "inputs": inputs,
            "preserved": {},
        }
        (report / "manifest.json").write_text(json.dumps(manifest))
        return record

    def test_chained_admission_is_readonly_and_rejects_source_output_identity_tampering(self):
        for fault in [
            None,
            "raw.bin",
            "data/target_aligned/forecasts.parquet",
            "reports/target_aligned/verification.json",
        ]:
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                record = self.fixture(root)
                if fault:
                    with (root / fault).open("ab") as stream:
                        stream.write(b"changed")
                before = {
                    str(p.relative_to(root)): p.read_bytes()
                    for p in root.rglob("*")
                    if p.is_file()
                }
                with patch.object(verify, "reconstruct_previous", return_value=(record, {})):
                    if fault:
                        with self.assertRaises(AssertionError):
                            verify.validate_upstream(root)
                    else:
                        self.assertEqual(
                            verify.validate_upstream(root)["status"],
                            "UPSTREAM_VERIFIED_READ_ONLY",
                        )
                self.assertEqual(
                    before,
                    {
                        str(p.relative_to(root)): p.read_bytes()
                        for p in root.rglob("*")
                        if p.is_file()
                    },
                )

    def test_new_prior_failure_or_missing_prior_code_pin_rejects(self):
        for fault in ["missing", "added"]:
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                record = self.fixture(root)

                def reconstruct(*args, fault=fault, root=root, record=record):
                    if fault == "added":
                        (root / "reports/target_aligned/failure.json").write_text("{}")
                    return record, {}

                if fault == "missing":
                    path = root / "reports/sign_memory/manifest.json"
                    m = json.loads(path.read_text())
                    m["code"] = {}
                    path.write_text(json.dumps(m))
                with (
                    patch.object(verify, "reconstruct_previous", side_effect=reconstruct),
                    self.assertRaises(AssertionError),
                ):
                    verify.validate_upstream(root)


class FullForecastContracts(unittest.TestCase):
    def fixture(self, unscored=False):
        import yaml

        from src import sign_memory_models as producer
        from tests.test_sign_memory_models import config, sample

        f, t = sample()
        p = yaml.safe_load((verify.ROOT / "sign_memory.yaml").read_text())
        p["index"] = config(f.index)
        if unscored:
            start = pd.Timestamp(p["index"]["origin_start"])
            month = start.to_period("M") + 1
            t.loc[t.index.to_period("M") == month, "y"] = np.nan
        panel, fits = producer.forecast_panel(f, t, p["index"])
        return f, t, panel, fits, p

    def test_independent_optimizers_replay_all_stages_probabilities_and_mature_cohorts(self):
        args = self.fixture()
        result = verify.verify_forecasts(*args)
        self.assertEqual(result["monthly_fits_verified"], len(args[3]))
        self.assertEqual(result["forecasts_verified"], len(args[2]))
        self.assertEqual(result["independent_convex_fits_verified"], 2 * len(args[3]))

    def test_fit_support_geometry_probability_and_timing_tampering_rejects(self):
        from copy import deepcopy

        original = self.fixture()
        for fault in [
            "beta",
            "memory",
            "center",
            "gradient",
            "support",
            "timing",
            "probability",
            "missing",
        ]:
            f, t, panel, fits, p = deepcopy(original)
            audit = fits[0]["model_audit"]
            if fault == "beta":
                audit["baseline"]["beta"][1] += 0.01
            elif fault == "memory":
                audit["memory"]["b"] += 0.01
            elif fault == "center":
                audit["transform"]["corr22_mean"] += 0.01
            elif fault == "gradient":
                audit["baseline"]["gradient_max_abs"] = 1e-3
            elif fault == "support":
                audit["support"]["events"] += 1
            elif fault == "timing":
                panel.loc[0, "available_date"] = panel.loc[0, "origin"]
            elif fault == "probability":
                panel.loc[0, "probability"] += 0.01
            else:
                panel = panel.iloc[1:]
            with self.subTest(fault=fault), self.assertRaises((AssertionError, ValueError)):
                verify.verify_forecasts(f, t, panel, fits, p)

    def test_entire_unscored_month_is_still_fitted_and_checked(self):
        f, t, panel, fits, p = self.fixture(True)
        result = verify.verify_forecasts(f, t, panel, fits, p)
        self.assertGreater(
            result["common_application_origins"], result["common_scored_origins"]
        )
        unscored = next(
            fit for fit in fits if pd.Timestamp(fit["fit_origin"]) not in set(panel.fit_origin)
        )
        unscored["model_audit"]["memory"]["b"] += 0.02
        with self.assertRaises((AssertionError, ValueError)):
            verify.verify_forecasts(f, t, panel, fits, p)


class InferenceContracts(unittest.TestCase):
    def test_all_four_phase_inferences_calibration_and_121_trial_events(self):
        from copy import deepcopy

        from src import sign_memory_search as producer
        from tests.test_sign_memory_search import panel, protocol

        frame, calendar = panel()
        p = protocol()
        p["inference"]["bootstrap_draws"] = 39
        prior = [
            {
                "study": "synthetic",
                "candidate": str(n),
                "control": "baseline",
                "horizon": 1,
                "p_conservative": 1.0,
            }
            for n in range(117)
        ]
        with patch.object(producer, "inherited", return_value=prior):
            metrics = producer.evaluate(frame, calendar, p, 118)
        with patch.object(verify, "inherited_rows", return_value=prior):
            result = verify.verify_metrics(verify.ROOT, frame, p, metrics, 118)
        self.assertEqual(result["phase_comparisons_verified"], 4)
        self.assertEqual(result["bootstrap_runs_verified"], 12)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = root / "reports/sign_memory"
            report.mkdir(parents=True)
            ledger = [{"event": "inherited", **row} for row in prior]
            ledger += [
                {
                    "event": "registered",
                    "study": "sign_memory",
                    "candidate": a,
                    "control": b,
                    "score": "brier",
                    "horizon": 1,
                    "protocol_sha256": metrics["protocol_sha256"],
                }
                for a, b in verify.COMPARISONS
            ]
            ledger += [{"event": "evaluated", **row} for row in metrics["rows"]]
            (report / "trial_ledger.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in ledger)
            )
            self.assertEqual(
                verify.verify_ledger(root, metrics, prior, "evaluated"),
                {"inherited": 117, "registered": 2, "evaluated": 2},
            )
        for fault in ["support", "calibration", "delta", "mde", "hypotheses"]:
            bad = deepcopy(metrics)
            if fault == "support":
                bad["class_support"]["development"]["events"] += 1
            elif fault == "calibration":
                bad["calibration"]["development"]["baseline"]["brier"] += 0.01
            elif fault == "delta":
                bad["rows"][0]["phases"][0]["delta"] += 0.01
            elif fault == "mde":
                bad["rows"][0]["phases"][0]["nominal_mde_effect_ratio"] *= 100
            else:
                bad["rows"].pop()
            with (
                patch.object(verify, "inherited_rows", return_value=prior),
                self.assertRaises(AssertionError),
            ):
                verify.verify_metrics(verify.ROOT, frame, p, bad, 118)

    def test_invalid_probabilities_and_nonzero_square_underflow_rejected(self):
        with self.assertRaises(ValueError):
            verify.brier_loss(np.array([-1e-13]), np.array([0.0]))
        with self.assertRaises(ValueError):
            verify.brier_loss(np.array([1e-250]), np.array([0.0]))
        np.testing.assert_equal(
            verify.brier_loss(np.array([0.0, 1.0]), np.array([0.0, 1.0])), 0.0
        )

    def test_coherent_near_endpoint_panel_tamper_cannot_hide_inside_replay_tolerance(self):
        from tests.test_sign_memory_search import panel

        frame, _ = panel()
        frame.loc[0, "probability"] = -1e-13
        frame.loc[0, "loss"] = (frame.loc[0, "y"] + 1e-13) ** 2
        with self.assertRaises(ValueError):
            verify.validate_panel(frame)
        frame.loc[0, "probability"] = 1 + 1e-13
        frame.loc[0, "loss"] = (frame.loc[0, "y"] - frame.loc[0, "probability"]) ** 2
        with self.assertRaises(ValueError):
            verify.validate_panel(frame)


class PublicationAndSnapshotContracts(unittest.TestCase):
    def test_hash_checked_source_staging_reads_verified_bytes_and_keeps_original_audit_path(
        self,
    ):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "raw.csv"
            path.write_bytes(b"original raw bytes")
            p = {"sources": {"daily": "raw.csv"}}
            m = {"inputs": {"raw.csv": verify.digest(path)}}

            def load(staged, p):
                self.assertEqual((staged / "raw.csv").read_bytes(), b"original raw bytes")
                path.write_bytes(b"changed after snapshot")
                return (
                    None,
                    None,
                    None,
                    {
                        "sources": {
                            "daily": {
                                "source_path": str(staged / "raw.csv"),
                                "source_sha256": m["inputs"]["raw.csv"],
                            }
                        }
                    },
                )

            with patch.object(verify.original, "load_source_tables", side_effect=load):
                *_, audit = verify.load_source_tables(root, p, m)
            self.assertEqual(audit["sources"]["daily"]["source_path"], str(path))
            with self.assertRaises(AssertionError):
                verify.load_source_tables(root, p, m)

    def test_complete_temporary_tree_reconstructs_forecasts_and_publishes_only_new_verification(
        self,
    ):
        from contextlib import ExitStack
        from copy import deepcopy

        import yaml

        f, t, panel, fits, p = FullForecastContracts().fixture()
        for fault in [False, True]:
            with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
                root = Path(folder)
                report = root / "reports/sign_memory"
                out = root / "data/sign_memory"
                report.mkdir(parents=True)
                out.mkdir(parents=True)
                (root / "sign_memory.yaml").write_text(yaml.safe_dump(p))
                phash = verify.digest(root / "sign_memory.yaml")
                manifest = {
                    "protocol_sha256": phash,
                    "code": {},
                    "inputs": {},
                    "preserved": {},
                }
                (report / "manifest.json").write_text(json.dumps(manifest))
                old = root / "reports/target_aligned"
                old.mkdir()
                (old / "unchanged").write_text("prior bytes")
                proof = {"status": "UPSTREAM_VERIFIED_READ_ONLY", "prior_files_written": False}
                (out / "upstream_admission.json").write_text(json.dumps(proof))
                audit = {"sources": {}, "synthetic": True}
                measurement = {"status": "PASS"}
                # The registered runner publishes both source audits in REPORT.
                (report / "source_audit.json").write_text(json.dumps(audit))
                (report / "measurement_audit.json").write_text(json.dumps(measurement))
                f.to_parquet(out / "features.parquet")
                t.to_parquet(out / "targets.parquet")
                altered = deepcopy(panel)
                if fault:
                    altered.loc[0, "y"] = 1 - altered.loc[0, "y"]
                altered.to_parquet(out / "forecasts.parquet")
                (out / "fits.json").write_text(json.dumps(fits))
                metrics = {
                    "protocol_sha256": phash,
                    "evidence_class": p["evidence_class"],
                    "inherited_rows": [],
                }
                (report / "metrics.json").write_text(json.dumps(metrics))
                stack.enter_context(patch.object(verify, "validate_protocol"))
                stack.enter_context(patch.object(verify, "verify_manifest_coverage"))
                stack.enter_context(
                    patch.object(verify, "validate_upstream", return_value=proof)
                )
                stack.enter_context(
                    patch.object(
                        verify, "load_source_tables", return_value=(None, None, None, audit)
                    )
                )
                stack.enter_context(
                    patch.object(
                        verify.original, "measurement_audit", return_value=measurement
                    )
                )
                stack.enter_context(patch.object(verify.original, "require_measurement"))
                stack.enter_context(
                    patch.object(verify, "feature_target_tables", return_value=(f, t))
                )
                stack.enter_context(
                    patch.object(verify, "verify_metrics", return_value={"synthetic": True})
                )
                stack.enter_context(
                    patch.object(
                        verify,
                        "verify_ledger",
                        return_value={"inherited": 117, "registered": 2, "evaluated": 2},
                    )
                )
                if fault:
                    with self.assertRaises((AssertionError, ValueError)):
                        verify.verify_with_failure_guard(root)
                    self.assertEqual(
                        json.loads((report / "verification.json").read_text())["status"],
                        "FAILED",
                    )
                else:
                    result = verify.verify_with_failure_guard(root)
                    self.assertEqual(result["status"], "VERIFIED")
                    self.assertEqual(
                        result["forecast_reconstruction"]["forecasts_verified"], len(panel)
                    )
                self.assertEqual((old / "unchanged").read_text(), "prior bytes")


if __name__ == "__main__":
    unittest.main()
