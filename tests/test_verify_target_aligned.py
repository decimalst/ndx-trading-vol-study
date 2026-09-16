"""Prewritten synthetic contracts for fixed-marginal target alignment verification."""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import ExitStack
from copy import deepcopy
from decimal import Decimal, localcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from src import verify_target_aligned as verify


def admission_fixture(root):
    from tests.test_verify_cross_moment import panel

    oldreport = root / "reports/cross_moment"
    olddata = root / "data/cross_moment"
    report = root / "reports/target_aligned"
    joint = root / "data/joint_risk"
    for path in (oldreport, olddata, report, joint, root / "src"):
        path.mkdir(parents=True, exist_ok=True)
    (root / "src/verify_cross_moment.py").write_text("synthetic previous verifier")
    (root / "source.bin").write_bytes(b"original source")
    (root / "reports/earlier.txt").write_text("original preserved publication")
    issued = panel(2)
    issued.to_parquet(joint / "forecasts.parquet")
    scored = verify.previous.score_panel(issued)
    scored.to_parquet(olddata / "scored_forecasts.parquet")
    nested = {"status": "UPSTREAM_VERIFIED_READ_ONLY", "prior_files_written": False}
    (olddata / "upstream_admission.json").write_text(json.dumps(nested))
    oldp = {
        "upstream": {"forecasts": "data/joint_risk/forecasts.parquet"},
        "evidence_class": "synthetic",
    }
    (root / "cross_moment.yaml").write_text(yaml.safe_dump(oldp))
    phash = verify.previous.digest(root / "cross_moment.yaml")
    vhash = verify.previous.digest(root / "src/verify_cross_moment.py")
    oldm = {
        "protocol_sha256": phash,
        "code": {"src/verify_cross_moment.py": vhash},
        "inputs": {
            name: verify.previous.digest(root / name)
            for name in ("source.bin", "data/joint_risk/forecasts.parquet")
        },
        "preserved": {
            "reports/earlier.txt": verify.previous.digest(root / "reports/earlier.txt")
        },
    }
    (oldreport / "manifest.json").write_text(json.dumps(oldm))
    metrics = {"protocol_sha256": phash, "evidence_class": "synthetic", "inherited_rows": []}
    (oldreport / "metrics.json").write_text(json.dumps(metrics))
    (oldreport / "trial_ledger.jsonl").write_text("")
    record = {
        "status": "VERIFIED",
        "issued_forecasts_verified": 6,
        "common_scored_origins": 2,
        "primitive_score_cells_verified": 18,
        "paired_product_differences_verified": 4,
        "upstream_admission": {
            "status": "UPSTREAM_VERIFIED_READ_ONLY",
            "prior_files_written": False,
        },
        "new_model_fits": 0,
        "new_forecasts": 0,
        "inference": {"synthetic": True},
        "ledger_events_verified": {"inherited": 112, "registered": 2, "evaluated": 2},
        "protocol_sha256": phash,
        "verifier_sha256": vhash,
        "limitations": verify.PREVIOUS_LIMITATIONS,
    }
    (oldreport / "verification.json").write_text(json.dumps(record))
    p = {
        "upstream": {
            "protocol": "cross_moment.yaml",
            "reports": "reports/cross_moment",
            "data": "data/cross_moment",
            "forecasts": "data/cross_moment/scored_forecasts.parquet",
            "protocol_sha256": phash,
            "verifier_sha256": vhash,
            "manifest_sha256": verify.previous.digest(oldreport / "manifest.json"),
            "verification_sha256": verify.previous.digest(oldreport / "verification.json"),
            "required_status": "VERIFIED",
            "forecasts_expected": 6,
            "scored_origins_expected": 2,
        }
    }
    (root / "target_aligned.yaml").write_text(yaml.safe_dump(p))
    inputs = {**oldm["inputs"], "cross_moment.yaml": phash}
    inputs.update(
        {
            str(path.relative_to(root)): verify.previous.digest(path)
            for folder in (oldreport, olddata)
            for path in folder.rglob("*")
            if path.is_file()
        }
    )
    manifest = {
        "protocol_sha256": verify.previous.digest(root / "target_aligned.yaml"),
        "code": oldm["code"],
        "inputs": inputs,
        "preserved": oldm["preserved"],
    }
    (report / "manifest.json").write_text(json.dumps(manifest))
    return nested, record


def admission_mocks(stack, fixture):
    nested, record = fixture
    stack.enter_context(patch.object(verify.previous, "validate_protocol"))
    stack.enter_context(
        patch.object(verify.previous, "validate_upstream", return_value=nested)
    )
    stack.enter_context(
        patch.object(verify.previous, "verify_metrics", return_value=record["inference"])
    )
    stack.enter_context(
        patch.object(
            verify.previous, "verify_ledger", return_value=record["ledger_events_verified"]
        )
    )
    for name in ("verify", "verify_with_failure_guard", "verify_manifest_coverage"):
        stack.enter_context(
            patch.object(
                verify.previous,
                name,
                side_effect=AssertionError("Forbidden old writer/current-tree coverage"),
            )
        )


class AdmissionContracts(unittest.TestCase):
    def test_readonly_chained_admission_retains_exact_original_record(self):
        with tempfile.TemporaryDirectory() as name, ExitStack() as stack:
            root = Path(name)
            fixture = admission_fixture(root)
            admission_mocks(stack, fixture)
            before = {
                str(path.relative_to(root)): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
            }
            proof = verify.validate_upstream(root)
            self.assertEqual(proof["status"], "UPSTREAM_VERIFIED_READ_ONLY")
            self.assertEqual(
                before,
                {
                    str(path.relative_to(root)): path.read_bytes()
                    for path in root.rglob("*")
                    if path.is_file()
                },
            )

    def test_old_source_score_or_verified_record_tampering_rejected_without_old_writers(self):
        for name in (
            "source.bin",
            "data/cross_moment/scored_forecasts.parquet",
            "reports/cross_moment/verification.json",
        ):
            with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
                root = Path(folder)
                fixture = admission_fixture(root)
                admission_mocks(stack, fixture)
                with (root / name).open("ab") as stream:
                    stream.write(b"tamper")
                with self.assertRaises(AssertionError):
                    verify.validate_upstream(root)

    def test_missing_prior_code_pin_or_added_upstream_failure_during_admission_is_rejected(
        self,
    ):
        for fault in ("missing_code", "new_failure"):
            with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
                root = Path(folder)
                fixture = admission_fixture(root)
                admission_mocks(stack, fixture)
                if fault == "missing_code":
                    path = root / "reports/target_aligned/manifest.json"
                    m = json.loads(path.read_text())
                    m["code"] = {}
                    path.write_text(json.dumps(m))
                else:

                    def change(*args, root=root, fixture=fixture):
                        (root / "reports/cross_moment/failure.json").write_text("{}")
                        return fixture[0]

                    stack.enter_context(
                        patch.object(verify.previous, "validate_upstream", side_effect=change)
                    )
                with self.assertRaises(AssertionError):
                    verify.validate_upstream(root)


class ScalarContracts(unittest.TestCase):
    def test_saved_scalar_audit_is_independently_reconstructed_and_tampering_rejected(self):
        from src import target_aligned_models as producer

        e, h, z = self.fixture()
        audit = producer.fit_cross_moment(e, h, z)
        verify.verify_scalar_audit(e, h, z, audit)
        for field, value in (
            ("a", audit["a"] + 0.01),
            ("b", audit["b"] + 0.01),
            ("slope_status", "FLAT_OBJECTIVE"),
        ):
            bad = deepcopy(audit)
            bad[field] = value
            with self.assertRaises(AssertionError):
                verify.verify_scalar_audit(e, h, z, bad)
        for field in ("gradient", "objective", "denominator", "gradient_tolerance"):
            bad = deepcopy(audit)
            bad["dynamic"][field] += 1e-6
            with self.assertRaises(AssertionError):
                verify.verify_scalar_audit(e, h, z, bad)

    def test_checked_mean_does_not_convert_nonzero_total_to_exact_zero(self):
        with self.assertRaises(ValueError):
            verify._mean([np.nextafter(0.0, 1.0), 0.0])
        self.assertEqual(verify._mean([1.0, -1.0]), 0.0)

    def test_nonzero_coefficient_ratio_underflow_is_rejected_before_objective(self):
        with self.assertRaises(ValueError):
            verify._divide(np.nextafter(0.0, 1.0), 10.0)
        self.assertEqual(verify._divide(0.0, 10.0), 0.0)
        d = np.r_[10.0, np.full(9, 1e-20)]
        y = np.zeros(10)
        y[0] = np.nextafter(0.0, 1.0)
        with self.assertRaisesRegex(ValueError, "quotient"):
            verify.independent_scalar_fit(
                np.c_[y, np.ones(10)], np.c_[d, d], np.linspace(-1, 1, 10)
            )

    def test_fit_statistic_primitive_zero_and_sign_masks_are_exact_at_subnormal_scale(self):
        tiny = np.nextafter(0.0, 1.0)
        for actual, expected in ((0.0, tiny), (-tiny, tiny), (tiny, 0.0)):
            with self.assertRaises(AssertionError):
                verify.fit_statistic(actual, expected, "primitive norm")
        verify.fit_statistic(tiny, tiny, "equal subnormal")

    def test_signed_gradient_comparison_uses_the_fixed_two_term_roundoff_scale(self):
        verify.gradient_equal(1e-16, 0.0, 1e-14, "near cancellation")
        with self.assertRaises(AssertionError):
            verify.gradient_equal(2e-14, 0.0, 1e-14, "outside scale")
        with self.assertRaises(AssertionError):
            verify.gradient_equal(float("nan"), 0.0, 1e-14, "nonfinite")

    def fixture(self):
        z = np.linspace(-2, 2, 81)
        h = np.c_[np.linspace(0.1, 0.5, len(z)), np.linspace(0.3, 0.15, len(z))]
        d = np.sqrt(h[:, 0]) * np.sqrt(h[:, 1])
        product = d * (0.18 + 0.3 * np.tanh(z))
        residual = np.c_[np.ones(len(z)), product]
        return residual, h, z

    def test_constrained_constant_and_fixed_intercept_slope_minimize_uniform_loss(self):
        e, h, z = self.fixture()
        result = verify.independent_scalar_fit(e, h, z)
        d = np.sqrt(h[:, 0]) * np.sqrt(h[:, 1])
        y = e[:, 0] * e[:, 1]
        constant_loss = lambda a: np.mean((y - d * a) ** 2)
        best_constant = constant_loss(result["a"])
        self.assertTrue(
            all(
                best_constant <= constant_loss(a) + 1e-16
                for a in np.linspace(-0.995, 0.995, 4001)
            )
        )
        w = (0.995 - abs(result["a"])) * d * np.tanh(z)
        slope_loss = lambda b: np.mean((y - result["a"] * d - w * b) ** 2)
        self.assertTrue(
            all(
                slope_loss(result["b"]) <= slope_loss(b) + 1e-16
                for b in np.linspace(-1, 1, 4001)
            )
        )
        self.assertLessEqual(slope_loss(result["b"]), slope_loss(0) + 1e-16)

    def test_constant_coefficient_matches_high_precision_ratio_without_volatility_weights(
        self,
    ):
        e, h, z = self.fixture()
        result = verify.independent_scalar_fit(e, h, z)
        d = np.sqrt(h[:, 0]) * np.sqrt(h[:, 1])
        y = e[:, 0] * e[:, 1]
        with localcontext() as context:
            context.prec = 60
            numerator = sum(
                Decimal.from_float(float(a)) * Decimal.from_float(float(b))
                for a, b in zip(d, y)
            )
            denominator = sum(Decimal.from_float(float(a)) ** 2 for a in d)
            expected = float(numerator / denominator)
        self.assertAlmostEqual(result["a"], expected, places=14)
        self.assertGreater(abs(result["a"] - np.mean(y / d)), 1e-4)

    def test_asset_swap_and_common_return_unit_conversion_leave_coefficients_unchanged(self):
        e, h, z = self.fixture()
        one = verify.independent_scalar_fit(e, h, z)
        for changed_e, changed_h in (
            (e[:, ::-1], h[:, ::-1]),
            (e * np.array([5.0, 3.0]), h * np.array([25.0, 9.0])),
        ):
            other = verify.independent_scalar_fit(changed_e, changed_h, z)
            np.testing.assert_allclose(
                [other["a"], other["b"]], [one["a"], one["b"]], rtol=1e-13, atol=0
            )

    def test_saturated_constant_and_exact_zero_state_have_retained_canonical_flat_slope(self):
        for e, h, z in (
            (np.ones((10, 2)) * 2, np.ones((10, 2)), np.arange(10.0)),
            (np.c_[np.ones(10), np.linspace(-0.1, 0.1, 10)], np.ones((10, 2)), np.zeros(10)),
        ):
            result = verify.independent_scalar_fit(e, h, z)
            self.assertEqual(result["b"], 0.0)
            self.assertEqual(result["slope_status"], "FLAT_OBJECTIVE")
            self.assertLessEqual(abs(result["a"]), 0.995)

    def test_nonzero_norm_underflow_or_nonfinite_inputs_fail_without_flat_fallback(self):
        e, h, z = self.fixture()
        for altered in (
            (e, h, np.ones(len(z)) * 1e-300),
            (e, np.ones_like(h) * 1e-300, z),
            (e * np.inf, h, z),
        ):
            with self.assertRaises((ValueError, AssertionError)):
                verify.independent_scalar_fit(*altered)

    def test_zero_training_targets_remain_valid_and_zero_slope_nests_the_control(self):
        e, h, z = self.fixture()
        e[:, 1] = 0
        result = verify.independent_scalar_fit(e, h, z)
        self.assertEqual(result["a"], 0.0)
        self.assertEqual(result["b"], 0.0)
        self.assertNotEqual(result["slope_status"], "FLAT_OBJECTIVE")


class ReplayContracts(unittest.TestCase):
    def test_saved_mean_and_variance_geometry_replayed_without_any_refit(self):
        columns = verify.MARGINAL_COLUMNS
        x = pd.DataFrame(np.zeros((4, len(columns))), columns=columns)
        x["const"] = 1.0
        x.iloc[:, 1] = [-1.0, 0.0, 1.0, 2.0]
        mean_beta = np.zeros(len(columns))
        mean_beta[:2] = [0.1, 0.2]
        var_beta = np.zeros(len(columns))
        var_beta[:2] = [0.2, -0.3]
        audits = {
            "mean": {
                "columns": list(columns),
                "means": [0.0] * len(columns),
                "scales": [1.0] * len(columns),
                "beta": mean_beta.tolist(),
            },
            "variance": {
                "columns": list(columns),
                "means": [0.0] * (len(columns) - 1),
                "scales": [1.0] * (len(columns) - 1),
                "scaled_beta": var_beta.tolist(),
                "train_mean": 0.04,
            },
        }
        mu, h = verify.replay_moments(x, audits)
        np.testing.assert_allclose(mu, [-0.1, 0.1, 0.3, 0.5], rtol=1e-14, atol=1e-15)
        np.testing.assert_allclose(
            h, 0.04 * np.exp(0.2 - 0.3 * x.iloc[:, 1]), rtol=1e-14, atol=0
        )
        changed = deepcopy(audits)
        changed["variance"]["scales"][0] = 0.0
        with self.assertRaises((ValueError, AssertionError)):
            verify.replay_moments(x, changed)

    def test_application_mutation_cannot_change_training_transform_or_coefficients(self):
        rng = np.random.default_rng(20260918)
        train = pd.DataFrame(
            rng.normal(size=(80, len(verify.ALL_FEATURES))), columns=verify.ALL_FEATURES
        )
        train["const"] = 1.0
        app = train.iloc[:3].copy()
        changed = app.copy()
        changed.iloc[:, 1:] *= 1000
        tr, ap, _ = verify.original.transform(train, app)
        other, _, _ = verify.original.transform(train, changed)
        pd.testing.assert_frame_equal(tr, other)
        self.assertEqual(tuple(ap.columns), verify.MARGINAL_COLUMNS)


class WholeForecastContracts(unittest.TestCase):
    def fixture(self):
        from src import target_aligned_panel as producer
        from tests.test_target_aligned_panel import fixture

        f, t, old, oldfits, config = fixture()
        new, fits = producer.forecast_panel(f, t, old, oldfits, config)
        protocol = yaml.safe_load((verify.ROOT / "target_aligned.yaml").read_text())
        return f, t, old, oldfits, new, fits, protocol

    def test_original_masks_replayed_coefficients_all_scalar_fits_and_five_forecast_arms(self):
        args = self.fixture()
        result = verify.verify_forecasts(*args)
        self.assertEqual(result["monthly_fits_verified"], len(args[3]))
        self.assertEqual(result["scalar_fits_verified"], 2 * len(args[3]))
        self.assertEqual(result["combined_forecasts_verified"], len(args[4]))
        self.assertEqual(result["new_forecasts_verified"], len(args[2]) // 3 * 2)

    def test_scalar_metadata_original_row_new_diagonal_and_product_tampering_rejected(self):
        original = self.fixture()
        for fault in (
            "a",
            "hash",
            "train_n",
            "old",
            "mean",
            "rho",
            "product",
            "loss",
            "missing",
        ):
            f, t, old, oldfits, panel, fits, protocol = deepcopy(original)
            if fault in ("a", "hash", "train_n"):
                if fault == "a":
                    fits[0]["model_audit"]["a"] += 0.01
                elif fault == "hash":
                    fits[0]["upstream_fit_sha256"] = "changed"
                else:
                    fits[0]["train_n"] += 1
            else:
                idx = panel.index[panel.model.eq("aligned_dynamic")][0]
                if fault == "old":
                    idx = panel.index[panel.model.eq("constant_correlation")][0]
                    panel.loc[idx, "rho"] += 0.01
                elif fault == "missing":
                    panel = panel.drop(idx)
                else:
                    panel.loc[
                        idx,
                        {
                            "mean": "mu_spx",
                            "rho": "rho",
                            "product": "forecast_product",
                            "loss": "loss",
                        }[fault],
                    ] += 0.01
            with self.subTest(fault=fault), self.assertRaises((AssertionError, ValueError)):
                verify.verify_forecasts(f, t, old, oldfits, panel, fits, protocol)

    def test_training_maturity_excludes_newly_unavailable_label_without_changing_source_dates(
        self,
    ):
        args = self.fixture()
        f, t, old, oldfits, panel, fits, protocol = deepcopy(args)
        entry = pd.Timestamp(oldfits[0]["fit_origin"])
        mask = verify.original.training_mask(f, t, entry)
        last = f.index[mask][-1]
        t.loc[last, "available_date"] = entry
        with self.assertRaises(AssertionError):
            verify.verify_forecasts(f, t, old, oldfits, panel, fits, protocol)

    def test_independent_primitive_reconstruction_never_replaces_original_rows(self):
        panel = self.fixture()[4]
        selected = panel.model.eq("constant_matrix")
        panel.loc[selected, "forecast_product"] = np.nextafter(
            panel.loc[selected, "forecast_product"], np.inf
        )
        before = panel.loc[selected].copy()
        result = verify.score_panel(panel)
        pd.testing.assert_frame_equal(result.loc[selected], before, check_exact=True)


class InferenceContracts(unittest.TestCase):
    def test_six_independent_phase_analyses_all_bootstraps_and_120_ledger_rows(self):
        from src import target_aligned_search as producer
        from tests.test_target_aligned_search import scored_panel

        panel, dates = scored_panel()
        rng = np.random.default_rng(202609180)
        bydate = pd.Series(rng.normal(0, 0.01, len(dates)), index=dates)
        panel["y_qqq"] = panel.origin.map(bydate)
        for name in verify.MODELS:
            subset = panel.loc[panel.model == name]
            panel.loc[subset.index, "loss"] = producer.tp.density.matrix_score(
                subset[["y_qqq", "y_spx"]].to_numpy()
                - subset[["mu_qqq", "mu_spx"]].to_numpy(),
                subset[["h_qqq", "h_spx"]].to_numpy(),
                subset.rho.to_numpy(),
            )
        panel = producer.tp.append_products(panel.loc[:, verify.previous.INPUT_COLUMNS])
        p = yaml.safe_load((verify.ROOT / "target_aligned.yaml").read_text())
        p["inference"]["bootstrap_draws"] = 39
        prior = [
            {
                "study": "synthetic",
                "candidate": str(i),
                "control": "baseline",
                "horizon": 1,
                "p_conservative": 1.0,
            }
            for i in range(114)
        ]
        with patch.object(producer, "inherited", return_value=prior):
            metrics = producer.evaluate(panel, dates, p, 118)
        with patch.object(verify, "inherited_rows", return_value=prior):
            checked = verify.verify_metrics(verify.ROOT, verify.score_panel(panel), p, metrics)
        self.assertEqual(checked["phase_comparisons_verified"], 6)
        self.assertEqual(checked["bootstrap_runs_verified"], 18)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            report = root / "reports/target_aligned"
            report.mkdir(parents=True)
            ledger = [{"event": "inherited", **row} for row in prior]
            ledger += [
                {
                    "event": "registered",
                    "study": "target_aligned",
                    "candidate": a,
                    "control": b,
                    "horizon": 1,
                    "score": "product_mse",
                    "protocol_sha256": metrics["protocol_sha256"],
                }
                for a, b in verify.COMPARISONS
            ]
            ledger += [{"event": "evaluated", **row} for row in metrics["rows"]]
            path = report / "trial_ledger.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in ledger))
            self.assertEqual(
                verify.verify_ledger(root, metrics, prior, "evaluated"),
                {"inherited": 114, "registered": 3, "evaluated": 3},
            )
            ledger[-1] = deepcopy(ledger[-1])
            ledger[-1]["phases"][0]["delta"] += 1e-11
            path.write_text("".join(json.dumps(row) + "\n" for row in ledger))
            with self.assertRaises(AssertionError):
                verify.verify_ledger(root, metrics, prior, "evaluated")
        for fault in ("contrast", "probability", "effect", "counts"):
            bad = deepcopy(metrics)
            if fault == "contrast":
                bad["rows"].pop()
            elif fault == "probability":
                bad["rows"][0]["p_conservative"] += 1e-11
            elif fault == "effect":
                bad["rows"][0]["phases"][0]["delta"] += 1e-11
            else:
                bad["new_forecasts"] += 1
            with (
                patch.object(verify, "inherited_rows", return_value=prior),
                self.assertRaises(AssertionError),
            ):
                verify.verify_metrics(verify.ROOT, panel, p, bad)


class PublicationContracts(unittest.TestCase):
    def test_complete_temporary_tree_verification_replays_fits_and_writes_only_new_record(
        self,
    ):
        f, t, old, oldfits, panel, fits, p = WholeForecastContracts().fixture()
        for fault in (False, True):
            with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
                root = Path(folder)
                out = root / "data/target_aligned"
                report = root / "reports/target_aligned"
                out.mkdir(parents=True)
                report.mkdir(parents=True)
                inputs = {
                    p["upstream"]["features"]: f,
                    p["upstream"]["targets"]: t,
                    p["upstream"]["forecasts"]: old,
                }
                for name, frame in inputs.items():
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    frame.to_parquet(path)
                fitpath = root / p["upstream"]["fits"]
                fitpath.write_text(json.dumps(oldfits))
                inputs[p["upstream"]["fits"]] = oldfits
                pp = deepcopy(p)
                pp["upstream"]["forecasts_expected"] = len(old)
                pp["upstream"]["scored_origins_expected"] = len(old) // 3
                pp["fitting"].update(
                    new_monthly_fits_expected=len(fits),
                    new_scalar_fits_expected=2 * len(fits),
                    new_forecasts_expected=len(panel) - len(old),
                    combined_forecast_rows_expected=len(panel),
                )
                (root / "target_aligned.yaml").write_text(yaml.safe_dump(pp))
                manifest = {
                    "protocol_sha256": verify.digest(root / "target_aligned.yaml"),
                    "code": {},
                    "preserved": {},
                    "inputs": {name: verify.digest(root / name) for name in inputs},
                }
                (report / "manifest.json").write_text(json.dumps(manifest))
                proof = {"status": "UPSTREAM_VERIFIED_READ_ONLY", "prior_files_written": False}
                (out / "upstream_admission.json").write_text(json.dumps(proof))
                panel.to_parquet(out / "forecasts.parquet")
                saved = deepcopy(fits)
                if fault:
                    saved[0]["model_audit"]["b"] += 0.01
                (out / "fits.json").write_text(json.dumps(saved))
                metrics = {
                    "protocol_sha256": manifest["protocol_sha256"],
                    "evidence_class": pp["evidence_class"],
                    "inherited_rows": [],
                }
                (report / "metrics.json").write_text(json.dumps(metrics))
                for method in ("validate_protocol", "verify_manifest_coverage"):
                    stack.enter_context(patch.object(verify, method))
                stack.enter_context(
                    patch.object(verify, "validate_upstream", return_value=proof)
                )
                stack.enter_context(
                    patch.object(
                        verify, "verify_metrics", return_value={"new_hypotheses_verified": 3}
                    )
                )
                stack.enter_context(
                    patch.object(
                        verify,
                        "verify_ledger",
                        return_value={"inherited": 114, "registered": 3, "evaluated": 3},
                    )
                )
                before = {
                    str(path.relative_to(root)): path.read_bytes()
                    for path in root.rglob("*")
                    if path.is_file()
                }
                if fault:
                    with self.assertRaises(AssertionError):
                        verify.verify(root)
                else:
                    result = verify.verify(root)
                    self.assertEqual(result["status"], "VERIFIED")
                    self.assertEqual(
                        result["forecast_reconstruction"]["new_forecasts_verified"],
                        len(panel) - len(old),
                    )
                for name, value in before.items():
                    self.assertEqual((root / name).read_bytes(), value)

    def test_protocol_fixed_operative_fields_and_complete_counts_cannot_change(self):
        p = yaml.safe_load((verify.ROOT / "target_aligned.yaml").read_text())
        verify.validate_protocol(p)
        for group, key, value in [
            ("upstream", "verification_sha256", "bad"),
            ("scoring", "new_model_fits", 0),
            ("fitting", "cap", 0.999),
            ("fitting", "gradient_eps_multiplier", 2048),
            ("fitting", "new_forecasts_expected", 1),
            ("inference", "seed", 20260917),
            ("comparisons", "new_hypotheses", 2),
            ("verification", "coefficient_absolute_tolerance", 1e-5),
        ]:
            changed = deepcopy(p)
            changed[group][key] = value
            with self.assertRaises(AssertionError):
                verify.validate_protocol(changed)

    def test_guard_retains_all_three_trials_and_old_files_when_metrics_are_invalid(self):
        for prior in ("{bad", '{"x":NaN}', '{"x":Infinity}', '{"leads":["aligned_dynamic"]}'):
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                report = root / "reports/target_aligned"
                old = root / "reports/cross_moment"
                report.mkdir(parents=True)
                old.mkdir(parents=True)
                (report / "metrics.json").write_text(prior)
                (old / "verification.json").write_text("immutable prior verification")
                with (
                    patch.object(
                        verify, "verify", side_effect=AssertionError("synthetic mismatch")
                    ),
                    self.assertRaises(AssertionError),
                ):
                    verify.verify_with_failure_guard(root)
                metrics = json.loads((report / "metrics.json").read_text())
                self.assertEqual(metrics["status"], "UNEVALUABLE")
                self.assertEqual(metrics["leads"], [])
                self.assertEqual(metrics["cumulative_hypothesis_count"], 117)
                self.assertEqual(len(metrics["rows"]), 3)
                self.assertTrue(
                    all(
                        r["p_conservative"] == r["p_holm_wave"] == r["p_holm_cumulative"] == 1
                        for r in metrics["rows"]
                    )
                )
                rows = [
                    json.loads(line)
                    for line in (report / "trial_ledger.jsonl").read_text().splitlines()
                ]
                self.assertEqual([r["event"] for r in rows], ["verification_failed"] * 3)
                self.assertEqual(
                    (old / "verification.json").read_text(), "immutable prior verification"
                )

    def test_optional_backup_failure_occurs_after_all_terminal_rows(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            report = root / "reports/target_aligned"
            report.mkdir(parents=True)
            (report / "metrics.json").write_text('{"leads":["aligned_dynamic"]}')
            original = Path.write_text

            def fail(path, *args, **kwargs):
                if path.name == "unpublished_scored_metrics.json":
                    raise OSError("backup")
                return original(path, *args, **kwargs)

            with patch.object(Path, "write_text", fail), self.assertRaises(OSError):
                verify.invalidate_publication(root, AssertionError("mismatch"))
            self.assertEqual(len((report / "trial_ledger.jsonl").read_text().splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
