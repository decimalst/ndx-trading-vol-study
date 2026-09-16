"""Prewritten synthetic tests for reuse-only cross-moment verification."""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import ExitStack
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from src import verify_cross_moment as verify


def panel(n=150):
    dates = pd.bdate_range("2016-01-04", periods=n)
    rng = np.random.default_rng(29160917)
    y = rng.normal(0, 0.01, size=(n, 2))
    rows = []
    for model, rho in zip(verify.MODELS, (0.2, 0.3, 0.4)):
        rows.append(
            pd.DataFrame(
                {
                    "origin": dates,
                    "model": model,
                    "horizon": 1,
                    "feature_cutoff_date": dates - pd.offsets.BDay(),
                    "target_end": dates + pd.offsets.BDay(),
                    "available_date": dates + pd.offsets.BDay(),
                    "y_qqq": y[:, 0],
                    "y_spx": y[:, 1],
                    "mu_qqq": 0.0,
                    "mu_spx": 0.0,
                    "h_qqq": 0.0001,
                    "h_spx": 0.0002,
                    "rho": rho,
                    "loss": rng.normal(-14, 1, n),
                    "fit_origin": dates[0],
                    "fit_cutoff_date": dates[0] - pd.offsets.BDay(),
                    "train_n": 1254,
                    "train_last_target": dates[0] - pd.offsets.BDay(),
                    "train_last_available": dates[0] - pd.offsets.BDay(),
                    "phase": "development",
                }
            )
        )
    result = (
        pd.concat(rows, ignore_index=True)
        .sort_values(["origin", "model"])
        .reset_index(drop=True)
    )
    for _, group in result.groupby(result.origin.dt.to_period("M")):
        first = group.origin.min()
        result.loc[group.index, "fit_origin"] = first
        for column in ("fit_cutoff_date", "train_last_target", "train_last_available"):
            result.loc[group.index, column] = first - pd.offsets.BDay()
    return result


def upstream_fixture(root):
    """Tiny saved tree; expensive prior mathematics is mocked, pins are real."""
    report, data = root / "reports/joint_risk", root / "data/joint_risk"
    report.mkdir(parents=True)
    data.mkdir(parents=True)
    (root / "src").mkdir()
    (root / "src/verify_joint_risk.py").write_text("synthetic prior verifier")
    (root / "source.bin").write_bytes(b"synthetic raw source")
    (root / "joint_risk.yaml").write_text("index: {}\n")
    phash = verify.digest(root / "joint_risk.yaml")
    vhash = verify.digest(root / "src/verify_joint_risk.py")
    date = pd.DatetimeIndex(["2016-01-04"], name="date")
    f = pd.DataFrame(
        np.ones((1, len(verify.upstream.ALL_FEATURES))),
        index=date,
        columns=verify.upstream.ALL_FEATURES,
    )
    f["feature_cutoff_date"] = pd.Timestamp("2015-12-31")
    t = pd.DataFrame(
        {
            "y_qqq": [0.01],
            "y_spx": [-0.01],
            "target_end": pd.Timestamp("2016-01-05"),
            "available_date": pd.Timestamp("2016-01-05"),
        },
        index=date,
    )
    f.to_parquet(data / "features.parquet")
    t.to_parquet(data / "targets.parquet")
    panel(1).to_parquet(data / "forecasts.parquet")
    (data / "fits.json").write_text("[]")
    source, measured = {"synthetic": True}, {"status": "PASS"}
    (data / "source_audit.json").write_text(json.dumps(source))
    (data / "measurement_audit.json").write_text(json.dumps(measured))
    metrics = {"protocol_sha256": phash, "inherited_rows": [], "evidence_class": "synthetic"}
    (report / "metrics.json").write_text(json.dumps(metrics))
    (report / "trial_ledger.jsonl").write_text("")
    prior_manifest = {
        "protocol_sha256": phash,
        "code": {"src/verify_joint_risk.py": vhash},
        "inputs": {"source.bin": verify.digest(root / "source.bin")},
        "preserved": {},
    }
    (report / "manifest.json").write_text(json.dumps(prior_manifest))
    reconstructed = {
        "status": "VERIFIED",
        "raw_feature_rows_verified": 1,
        "raw_feature_columns_verified": 34,
        "measurement_audit": measured,
        "forecast_reconstruction": {
            "forecasts_verified": 3,
            "common_scored_origins": 1,
            "monthly_fits_verified": 1,
        },
        "inference": {"synthetic": True},
        "ledger_events_verified": {"inherited": 110, "registered": 2, "evaluated": 2},
        "protocol_sha256": phash,
        "verifier_sha256": vhash,
        "limitations": list(verify.UPSTREAM_LIMITATIONS),
    }
    (report / "verification.json").write_text(json.dumps(reconstructed))
    p = {
        "upstream": {
            "protocol": "joint_risk.yaml",
            "reports": "reports/joint_risk",
            "data": "data/joint_risk",
            "forecasts": "data/joint_risk/forecasts.parquet",
            "features": "data/joint_risk/features.parquet",
            "protocol_sha256": phash,
            "verifier_sha256": vhash,
            "required_status": "VERIFIED",
            "verification_sha256": verify.digest(report / "verification.json"),
            "manifest_sha256": verify.digest(report / "manifest.json"),
            "forecasts_expected": 3,
            "scored_origins_expected": 1,
            "monthly_fits_expected": 1,
        }
    }
    (root / "cross_moment.yaml").write_text(yaml.safe_dump(p))
    current = root / "reports/cross_moment"
    current.mkdir()
    inputs = [
        root / "source.bin",
        root / "joint_risk.yaml",
        *report.rglob("*"),
        *data.rglob("*"),
    ]
    manifest = {
        "protocol_sha256": verify.digest(root / "cross_moment.yaml"),
        "inputs": {
            str(path.relative_to(root)): verify.digest(path)
            for path in inputs
            if path.is_file()
        },
        "code": {},
        "preserved": {},
    }
    (current / "manifest.json").write_text(json.dumps(manifest))
    return f, t, source, measured, reconstructed


def upstream_mocks(stack, fixture):
    f, t, source, measured, record = fixture
    stack.enter_context(patch.object(verify.upstream, "validate_protocol"))
    stack.enter_context(patch.object(verify.upstream, "verify_manifest_coverage"))
    stack.enter_context(
        patch.object(
            verify.upstream, "load_source_tables", return_value=(None, None, None, source)
        )
    )
    stack.enter_context(
        patch.object(verify.upstream, "measurement_audit", return_value=measured)
    )
    stack.enter_context(patch.object(verify.upstream, "require_measurement"))
    stack.enter_context(
        patch.object(verify.upstream, "feature_target_tables", return_value=(f, t))
    )
    stack.enter_context(
        patch.object(
            verify.upstream, "verify_forecasts", return_value=record["forecast_reconstruction"]
        )
    )
    stack.enter_context(
        patch.object(verify.upstream, "verify_metrics", return_value=record["inference"])
    )
    stack.enter_context(
        patch.object(
            verify.upstream, "verify_ledger", return_value=record["ledger_events_verified"]
        )
    )
    stack.enter_context(
        patch.object(
            verify.upstream, "verify", side_effect=AssertionError("prior writer forbidden")
        )
    )
    stack.enter_context(
        patch.object(
            verify.upstream,
            "verify_with_failure_guard",
            side_effect=AssertionError("prior guard forbidden"),
        )
    )


class UpstreamContracts(unittest.TestCase):
    def test_issued_snapshot_is_verified_before_decode_and_never_reopens_path(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / "issued.parquet"
            expected = panel(2)
            expected.to_parquet(path)
            signature = verify.digest(path)
            original = pd.read_parquet

            def decode(buffer, *args, **kwargs):
                self.assertNotIsInstance(buffer, (str, Path))
                path.write_bytes(b"mutated after checked snapshot")
                return original(buffer, *args, **kwargs)

            with patch.object(pd, "read_parquet", side_effect=decode):
                actual = verify.read_issued_snapshot(root, "issued.parquet", signature)
            pd.testing.assert_frame_equal(actual, expected)
            with patch.object(pd, "read_parquet") as parse, self.assertRaises(AssertionError):
                verify.read_issued_snapshot(root, "issued.parquet", signature)
            parse.assert_not_called()

    def test_manifest_requires_entire_upstream_input_code_and_prior_artifact_inventory(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            upstream_fixture(root)
            p = yaml.safe_load((root / "cross_moment.yaml").read_text())
            required = (
                "src/cross_moment_score.py",
                "src/cross_moment_search.py",
                "src/verify_cross_moment.py",
                "tests/test_cross_moment_score.py",
                "tests/test_cross_moment_search.py",
                "tests/test_cross_moment_publication.py",
                "tests/test_verify_cross_moment.py",
            )
            for name in required:
                path = root / name
                path.parent.mkdir(exist_ok=True)
                path.write_text("synthetic code")
            (root / "reports/earlier.txt").write_text("immutable previous publication")
            path = root / "reports/cross_moment/manifest.json"
            manifest = json.loads(path.read_text())
            manifest["code"] = {
                str(path.relative_to(root)): verify.digest(path)
                for directory in ("src", "tests")
                for path in (root / directory).rglob("*.py")
            }
            manifest["preserved"] = {
                "reports/earlier.txt": verify.digest(root / "reports/earlier.txt")
            }
            verify.verify_manifest_coverage(root, p, manifest)
            for group in ("code", "inputs", "preserved"):
                changed = deepcopy(manifest)
                changed[group].pop(next(iter(changed[group])))
                with self.assertRaises(AssertionError):
                    verify.verify_manifest_coverage(root, p, changed)

    def test_new_upstream_failure_file_during_replay_is_not_ignored(self):
        with tempfile.TemporaryDirectory() as name, ExitStack() as stack:
            root = Path(name)
            fixture = upstream_fixture(root)
            upstream_mocks(stack, fixture)

            def change(*args):
                (root / "reports/joint_risk/failure.json").write_text("{}")
                return fixture[-1]["forecast_reconstruction"]

            with (
                patch.object(verify.upstream, "verify_forecasts", side_effect=change),
                self.assertRaises(AssertionError),
            ):
                verify.validate_upstream(root)

    def test_admission_reconstructs_prior_record_without_writing_any_prior_file(self):
        with tempfile.TemporaryDirectory() as name, ExitStack() as stack:
            root = Path(name)
            fixture = upstream_fixture(root)
            upstream_mocks(stack, fixture)
            before = {
                str(p.relative_to(root)): p.read_bytes()
                for p in root.rglob("*")
                if p.is_file()
            }
            proof = verify.validate_upstream(root)
            self.assertEqual(proof["status"], "UPSTREAM_VERIFIED_READ_ONLY")
            after = {
                str(p.relative_to(root)): p.read_bytes()
                for p in root.rglob("*")
                if p.is_file()
            }
            self.assertEqual(before, after)

    def test_source_forecast_and_verified_marker_tampering_rejected_before_replay(self):
        for filename in (
            "source.bin",
            "data/joint_risk/forecasts.parquet",
            "reports/joint_risk/verification.json",
        ):
            with tempfile.TemporaryDirectory() as name, ExitStack() as stack:
                root = Path(name)
                fixture = upstream_fixture(root)
                upstream_mocks(stack, fixture)
                with (root / filename).open("ab") as stream:
                    stream.write(b"changed")
                with (
                    patch.object(verify.upstream, "verify_forecasts") as replay,
                    self.assertRaises(AssertionError),
                ):
                    verify.validate_upstream(root)
                replay.assert_not_called()

    def test_missing_upstream_manifest_pin_rejected(self):
        with tempfile.TemporaryDirectory() as name, ExitStack() as stack:
            root = Path(name)
            fixture = upstream_fixture(root)
            upstream_mocks(stack, fixture)
            path = root / "reports/cross_moment/manifest.json"
            manifest = json.loads(path.read_text())
            del manifest["inputs"]["data/joint_risk/fits.json"]
            path.write_text(json.dumps(manifest))
            with self.assertRaises(AssertionError):
                verify.validate_upstream(root)

    def test_prior_reconstruction_failure_does_not_call_old_publication_guard(self):
        with tempfile.TemporaryDirectory() as name, ExitStack() as stack:
            root = Path(name)
            fixture = upstream_fixture(root)
            upstream_mocks(stack, fixture)
            path = root / "reports/joint_risk/verification.json"
            before = path.read_bytes()
            with (
                patch.object(
                    verify.upstream,
                    "verify_forecasts",
                    side_effect=AssertionError("synthetic forecast mismatch"),
                ),
                self.assertRaises(AssertionError),
            ):
                verify.validate_upstream(root)
            self.assertEqual(path.read_bytes(), before)

    def test_mid_admission_mutation_is_detected_by_final_hash_check(self):
        with tempfile.TemporaryDirectory() as name, ExitStack() as stack:
            root = Path(name)
            fixture = upstream_fixture(root)
            upstream_mocks(stack, fixture)

            def change(*args):
                (root / "source.bin").write_bytes(b"mid-audit mutation")
                return fixture[-1]["forecast_reconstruction"]

            with (
                patch.object(verify.upstream, "verify_forecasts", side_effect=change),
                self.assertRaises(AssertionError),
            ):
                verify.validate_upstream(root)


class ProductContracts(unittest.TestCase):
    def test_undefined_dates_daily_refits_and_nonmature_training_rejected(self):
        for fault in ("missing_date", "daily_fit", "training_maturity", "stale_fit"):
            bad = panel(30)
            if fault == "missing_date":
                bad["feature_cutoff_date"] = pd.NaT
            elif fault == "daily_fit":
                bad["fit_origin"] = bad.origin
            elif fault == "training_maturity":
                bad["train_last_target"] = bad.origin
            else:
                bad["fit_origin"] = pd.Timestamp("2015-12-01")
                bad["fit_cutoff_date"] = pd.Timestamp("2015-11-30")
            with self.subTest(fault=fault), self.assertRaises(AssertionError):
                verify.score_panel(bad)

    def test_complex_inputs_are_rejected_without_discarding_imaginary_parts(self):
        with self.assertRaises((ValueError, AssertionError)):
            verify.paired_difference(np.array([1 + 1j]), np.zeros(1), np.zeros(1))
        bad = panel(3)
        bad["rho"] = bad.rho.astype(complex) + 1j
        with self.assertRaises((ValueError, AssertionError)):
            verify.score_panel(bad)

    def test_subnormal_difference_is_retained_despite_equal_rounded_losses(self):
        actual = verify.paired_difference([2e-162], [2.6e-162], [0.0])
        self.assertEqual(actual[0], -np.nextafter(0.0, 1.0))
        verify.primitive_equal(actual, actual.copy(), "subnormal")
        with self.assertRaises(AssertionError):
            verify.primitive_equal([0.0], actual, "zero mask")
        with self.assertRaises(AssertionError):
            verify.primitive_equal([1e-20], [2e-20], "no absolute floor")

    def test_equal_forecasts_do_not_bypass_individual_loss_overflow(self):
        with self.assertRaises(ValueError):
            verify.paired_difference([1e308], [1e308], [-1e308])

    def test_signed_and_zero_products_and_forecasts_keep_original_rows(self):
        p = panel(3)
        p.loc[p.origin == p.origin.iloc[0], "y_qqq"] = 0.0
        result = verify.score_panel(p)
        pd.testing.assert_frame_equal(result.loc[:, p.columns], p)
        self.assertTrue(
            result.loc[result.origin == p.origin.iloc[0], "realized_product"].eq(0).all()
        )
        self.assertTrue((result.product_mse >= 0).all())
        for row in result.itertuples():
            self.assertAlmostEqual(
                row.realized_product, (row.y_qqq - row.mu_qqq) * (row.y_spx - row.mu_spx)
            )
            self.assertAlmostEqual(
                row.forecast_product, row.rho * np.sqrt(row.h_qqq) * np.sqrt(row.h_spx)
            )

    def test_stable_gap_matches_high_precision_under_cancellation(self):
        y, first, second = np.array([1.0]), np.array([1e-16]), np.array([0.0])
        exact = (Decimal.from_float(first[0]) - Decimal.from_float(second[0])) * (
            Decimal.from_float(first[0])
            + Decimal.from_float(second[0])
            - 2 * Decimal.from_float(y[0])
        )
        actual = verify.paired_difference(first, second, y)
        np.testing.assert_allclose(actual, [float(exact)], rtol=1e-15, atol=0)
        self.assertNotEqual(actual[0], (y - first)[0] ** 2 - (y - second)[0] ** 2)

    def test_nonzero_underflow_and_nonfinite_product_are_failures(self):
        for first, second in ((1e-300, 1e-300), (1e308, 1e308)):
            with self.assertRaises(ValueError):
                verify.checked_product(first, second)
        self.assertEqual(verify.checked_product(0.0, 1e-300), 0.0)

    def test_unequal_issued_means_conditional_diagonals_or_row_sets_rejected(self):
        p = panel(3)
        for column in ("mu_qqq", "h_spx", "available_date"):
            bad = p.copy()
            idx = bad.index[bad.model == "dynamic_correlation"][0]
            bad.loc[idx, column] = (
                pd.Timestamp("2025-11-03")
                if column == "available_date"
                else bad.loc[idx, column] + 0.01
            )
            with self.assertRaises(AssertionError):
                verify.score_panel(bad)
        with self.assertRaises(AssertionError):
            verify.score_panel(p.iloc[1:])

    def test_asset_exchange_and_unit_rescaling_of_direct_product_loss(self):
        p = panel(4)
        original = verify.score_panel(p)
        swapped = p.copy()
        for stem in ("y", "mu", "h"):
            swapped[[stem + "_qqq", stem + "_spx"]] = p[
                [stem + "_spx", stem + "_qqq"]
            ].to_numpy()
        np.testing.assert_allclose(
            verify.score_panel(swapped).product_mse, original.product_mse, rtol=1e-13, atol=0
        )
        scaled = p.copy()
        for asset, scale in (("qqq", 10.0), ("spx", 3.0)):
            for stem in ("y", "mu"):
                scaled[stem + "_" + asset] *= scale
            scaled["h_" + asset] *= scale**2
        np.testing.assert_allclose(
            verify.score_panel(scaled).product_mse,
            original.product_mse * 900,
            rtol=1e-13,
            atol=0,
        )


class PublicationContracts(unittest.TestCase):
    def test_complete_new_entrypoint_reads_issued_rows_and_writes_only_new_verification(self):
        for tamper in (False, True):
            with tempfile.TemporaryDirectory() as name, ExitStack() as stack:
                root = Path(name)
                upstream_fixture(root)
                p = yaml.safe_load((verify.ROOT / "cross_moment.yaml").read_text())
                p["upstream"]["forecasts_expected"] = 3
                p["upstream"]["scored_origins_expected"] = 1
                (root / "cross_moment.yaml").write_text(yaml.safe_dump(p))
                report = root / "reports/cross_moment"
                data = root / "data/cross_moment"
                data.mkdir()
                mpath = report / "manifest.json"
                manifest = json.loads(mpath.read_text())
                manifest["protocol_sha256"] = verify.digest(root / "cross_moment.yaml")
                mpath.write_text(json.dumps(manifest))
                proof = {"status": "UPSTREAM_VERIFIED_READ_ONLY", "new_model_fits": 0}
                (data / "upstream_admission.json").write_text(json.dumps(proof))
                scored = verify.score_panel(pd.read_parquet(root / p["upstream"]["forecasts"]))
                if tamper:
                    scored.loc[0, "forecast_product"] *= 2
                scored.to_parquet(data / "scored_forecasts.parquet")
                metrics = {
                    "protocol_sha256": manifest["protocol_sha256"],
                    "evidence_class": p["evidence_class"],
                    "inherited_rows": [],
                    "new_model_fits": 0,
                    "new_forecasts": 0,
                    "scored_issued_forecast_rows": 3,
                    "common_scored_origins": 1,
                }
                (report / "metrics.json").write_text(json.dumps(metrics))
                for method in ("validate_protocol", "verify_manifest_coverage"):
                    stack.enter_context(patch.object(verify, method))
                stack.enter_context(
                    patch.object(verify, "validate_upstream", return_value=proof)
                )
                stack.enter_context(
                    patch.object(
                        verify, "verify_metrics", return_value={"new_hypotheses_verified": 2}
                    )
                )
                stack.enter_context(
                    patch.object(
                        verify,
                        "verify_ledger",
                        return_value={"inherited": 112, "registered": 2, "evaluated": 2},
                    )
                )
                before = {
                    str(path.relative_to(root)): path.read_bytes()
                    for path in root.rglob("*")
                    if path.is_file()
                }
                if tamper:
                    with self.assertRaises(AssertionError):
                        verify.verify(root)
                else:
                    result = verify.verify(root)
                    self.assertEqual(result["status"], "VERIFIED")
                    self.assertEqual(result["issued_forecasts_verified"], 3)
                    self.assertEqual(result["new_model_fits"], 0)
                for path, original in before.items():
                    self.assertEqual((root / path).read_bytes(), original)

    def test_optional_diagnostic_backup_failure_keeps_both_terminal_ledger_events(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            report = root / "reports/cross_moment"
            report.mkdir(parents=True)
            (report / "metrics.json").write_text('{"leads":["dynamic_correlation"]}')
            original = Path.write_text

            def fail_backup(path, *args, **kwargs):
                if path.name == "unpublished_scored_metrics.json":
                    raise OSError("synthetic backup failure")
                return original(path, *args, **kwargs)

            with patch.object(Path, "write_text", fail_backup), self.assertRaises(OSError):
                verify.invalidate_publication(root, AssertionError("mismatch"))
            rows = [
                json.loads(line)
                for line in (report / "trial_ledger.jsonl").read_text().splitlines()
            ]
            self.assertEqual([r["event"] for r in rows], ["verification_failed"] * 2)
            self.assertEqual(json.loads((report / "metrics.json").read_text())["leads"], [])

    def test_guard_invalidates_only_new_wave_and_preserves_bad_json_and_ledger(self):
        for prior in (
            '{"protocol_sha256":NaN}',
            '{"diagnostic":Infinity}',
            "{bad",
            '{"leads":["dynamic_correlation"]}',
        ):
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                report = root / "reports/cross_moment"
                old = root / "reports/joint_risk"
                report.mkdir(parents=True)
                old.mkdir(parents=True)
                (report / "metrics.json").write_text(prior)
                (old / "verification.json").write_text("immutable old VERIFIED")
                with (
                    patch.object(
                        verify, "verify", side_effect=AssertionError("admission discrepancy")
                    ),
                    self.assertRaises(AssertionError),
                ):
                    verify.verify_with_failure_guard(root)
                result = json.loads((report / "metrics.json").read_text())
                self.assertEqual(result["status"], "UNEVALUABLE")
                self.assertEqual(result["leads"], [])
                self.assertEqual(result["cumulative_hypothesis_count"], 114)
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
                    (old / "verification.json").read_text(), "immutable old VERIFIED"
                )
                events = [
                    json.loads(line)
                    for line in (report / "trial_ledger.jsonl").read_text().splitlines()
                ]
                self.assertEqual(
                    [row["event"] for row in events],
                    ["verification_failed", "verification_failed"],
                )


class InferenceContracts(unittest.TestCase):
    def protocol(self):
        return yaml.safe_load((verify.ROOT / "cross_moment.yaml").read_text())

    def test_literal_numeric_protocol_and_relaxations(self):
        p = self.protocol()
        verify.validate_protocol(p)
        for group, key, value in [
            ("inference", "inference_scale", 1.0),
            ("scoring", "new_model_fits", 1),
            ("scoring", "effect_threshold_absolute", 1e-12),
            ("comparisons", "cumulative_hypotheses", 113),
            ("verification", "product_comparison_eps_multiplier", 1000),
            ("upstream", "verification_sha256", "bad"),
        ]:
            changed = deepcopy(p)
            changed[group][key] = value
            with self.assertRaises(AssertionError):
                verify.validate_protocol(changed)

    def test_scaled_inference_reconstructs_bootstrap_hac_and_restored_dimensions(self):
        p = self.protocol()
        p["inference"]["bootstrap_draws"] = 39
        scored = verify.score_panel(panel())
        result = verify.phase_statistics(scored, "constant_correlation", "development", 0, p)
        rows = {m: scored.loc[scored.model == m].set_index("origin") for m in verify.MODELS}
        d = verify.paired_difference(
            rows["dynamic_correlation"].forecast_product,
            rows["constant_correlation"].forecast_product,
            rows["dynamic_correlation"].realized_product,
        )
        normalized = d / 1e-10
        self.assertEqual(result["delta"], float(normalized.mean()) * 1e-10)
        expected = verify.independent_hac(normalized)
        self.assertEqual(result["hac126"]["p"], expected["p"])
        self.assertEqual(result["hac126"]["se"], expected["se"] * 1e-10)
        self.assertEqual(
            result["nominal_mde_effect_ratio"], result["hac126"]["mde80_nominal"] / 1e-10
        )
        for width in (21, 63, 126):
            means = verify.explicit_bootstrap_means(normalized, width, 39, 20260917 + width)[
                :, 0
            ]
            self.assertEqual(
                result["block_inference"][str(width)]["p"],
                (
                    1
                    + np.count_nonzero(
                        abs(means - normalized.mean()) >= abs(normalized.mean())
                    )
                )
                / 40,
            )
        self.assertNotIn("gain_relative", result)

    def test_normalized_nonconstant_underflow_rejected_and_exact_constants_allowed(self):
        with self.assertRaises(ValueError):
            verify.normalized_difference(np.array([1e-300, 2e-300]))
        with self.assertRaises(ValueError):
            verify.normalized_difference(np.array([1e308, 1e308]))
        np.testing.assert_array_equal(verify.normalized_difference(np.zeros(3)), np.zeros(3))
        np.testing.assert_array_equal(
            verify.normalized_difference(np.ones(3) * 1e-10), np.ones(3)
        )

    def test_tree_comparison_does_not_hide_tiny_effects_or_probability_changes(self):
        verify.compare_tree({"delta": 1e-20, "p": 0.5}, {"delta": 1e-20, "p": 0.5}, "equal")
        for actual in (
            {"delta": 2e-20, "p": 0.5},
            {"delta": 1e-20, "p": 0.5000000001},
            {"delta": float("nan"), "p": 0.5},
        ):
            with self.assertRaises(AssertionError):
                verify.compare_tree(actual, {"delta": 1e-20, "p": 0.5}, "strict")

    def synthetic_metrics(self):
        from src import cross_moment_search as producer
        from tests.test_joint_risk_search import panel as synthetic_upstream

        issued, dates = synthetic_upstream()
        for month, group in issued.groupby(issued.origin.dt.to_period("M")):
            first = group.origin.min()
            issued.loc[group.index, "fit_origin"] = first
            issued.loc[group.index, "fit_cutoff_date"] = first - pd.offsets.BDay()
            issued.loc[group.index, "train_last_target"] = first - pd.offsets.BDay()
            issued.loc[group.index, "train_last_available"] = first - pd.offsets.BDay()
        rng = np.random.default_rng(202609170)
        issued["y_qqq"] = np.tile(rng.normal(0, 0.01, len(dates)), 3)
        issued["y_spx"] = np.tile(rng.normal(0, 0.01, len(dates)), 3)
        scored = producer.cs.score_panel(issued)
        expected = verify.score_panel(issued)
        for column in verify.PRODUCT_COLUMNS:
            verify.primitive_equal(
                scored[column], expected[column], "Independent synthetic " + column
            )
        p = self.protocol()
        p["inference"]["bootstrap_draws"] = 39
        prior = [
            {
                "study": "synthetic",
                "candidate": f"prior{i}",
                "control": "baseline",
                "horizon": 1,
                "p_conservative": 1.0,
            }
            for i in range(112)
        ]
        with patch.object(producer, "inherited", return_value=prior):
            metrics = producer.evaluate(scored, dates, p)
        return p, metrics, expected, prior

    def test_independent_all_phase_scores_bootstraps_holm_and_literal_trial_ledger(self):
        p, metrics, scored, prior = self.synthetic_metrics()
        with patch.object(verify, "inherited_rows", return_value=prior):
            result = verify.verify_metrics(verify.ROOT, scored, p, metrics)
        self.assertEqual(result["bootstrap_runs_verified"], 12)
        self.assertEqual(result["cumulative_hypotheses_verified"], 114)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            report = root / "reports/cross_moment"
            report.mkdir(parents=True)
            events = [{"event": "inherited", **row} for row in prior]
            events += [
                {
                    "event": "registered",
                    "study": "cross_moment",
                    "candidate": a,
                    "control": b,
                    "score": "product_mse",
                    "horizon": 1,
                    "protocol_sha256": metrics["protocol_sha256"],
                }
                for a, b in verify.COMPARISONS
            ]
            events += [{"event": "evaluated", **row} for row in metrics["rows"]]
            path = report / "trial_ledger.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in events))
            self.assertEqual(
                verify.verify_ledger(root, metrics, prior, "evaluated"),
                {"inherited": 112, "registered": 2, "evaluated": 2},
            )
            events[-1] = deepcopy(events[-1])
            events[-1]["phases"][0]["delta"] += 1e-11
            path.write_text("".join(json.dumps(row) + "\n" for row in events))
            with self.assertRaises(AssertionError):
                verify.verify_ledger(root, metrics, prior, "evaluated")

    def test_metric_missing_contrast_probability_change_and_tiny_effect_tamper_rejected(self):
        p, metrics, scored, prior = self.synthetic_metrics()
        for fault in ("row", "probability", "effect"):
            changed = deepcopy(metrics)
            if fault == "row":
                changed["rows"].pop()
            elif fault == "probability":
                changed["rows"][0]["p_conservative"] += 1e-11
            else:
                changed["rows"][0]["phases"][0]["delta"] += 1e-11
            with (
                patch.object(verify, "inherited_rows", return_value=prior),
                self.assertRaises(AssertionError),
            ):
                verify.verify_metrics(verify.ROOT, scored, p, changed)


if __name__ == "__main__":
    unittest.main()
