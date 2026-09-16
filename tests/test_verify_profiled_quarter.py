"""Prewritten profiled verification and immutable failed-history contracts."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import verify_profiled_quarter as verify
from tests.test_verify_civil_quarter import original_fixture


def failed_verification_fixture(root):
    import yaml

    report = root / "reports/civil_quarter_replay"
    out = root / "data/civil_quarter_replay"
    report.mkdir(parents=True)
    out.mkdir(parents=True)
    oldp = copy.deepcopy(verify.previous.CONTRACT)
    protocol_path = root / "civil_quarter_replay.yaml"
    protocol_path.write_text(yaml.safe_dump(oldp))
    phash = verify.digest(protocol_path)
    pins = {}

    def save(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
        pins[name] = verify.digest(path)
        return pins[name]

    for i, name in enumerate(oldp["comparisons"]["inherited_sources"]):
        save(
            name,
            {
                "rows": [
                    {
                        "candidate": str(j),
                        "control": "baseline",
                        "horizon": 1,
                        "p_conservative": 1.0,
                    }
                    for j in range(123)
                ]
                if i == 0
                else []
            },
        )
    verifier_name = "src/verify_civil_quarter_replay.py"
    save(verifier_name, b"opaque frozen verifier")
    code = {verifier_name: pins.pop(verifier_name)}
    manifest = {"protocol_sha256": phash, "code": code, "inputs": dict(pins), "preserved": {}}
    save("reports/civil_quarter_replay/manifest.json", manifest)
    save(
        "reports/civil_quarter_replay/freeze_record.json",
        {"protocol_sha256": phash, "code": code, "prefit_design": {}},
    )
    pins.update(code)
    pins["civil_quarter_replay.yaml"] = phash
    prior = verify.previous.inherited_rows(root, oldp, pins=manifest["inputs"])
    error = "Independent verification failed: AssertionError: Independent baseline stationarity failed"
    rows = [
        {
            "study": "civil_quarter_replay",
            "candidate": a,
            "control": b,
            "score": "proper_variance",
            "horizon": 1,
            "phases": [],
            "p_conservative": 1.0,
            "p_holm_wave": 1.0,
            "p_holm_cumulative": 1.0,
            "status": "INVALID_RUN",
            "verdict": "UNEVALUABLE",
            "error": error,
        }
        for a, b in verify.COMPARISONS
    ]
    failure = {
        "status": "UNEVALUABLE",
        "whole_wave_aborted": True,
        "hypothesis_count": 2,
        "cumulative_hypothesis_count": 125,
        "leads": [],
        "protocol_sha256": phash,
        "rows": rows,
    }
    for name in ("metrics.json", "failure.json"):
        save("reports/civil_quarter_replay/" + name, failure)
    save(
        "reports/civil_quarter_replay/verification.json",
        {"status": "FAILED", "error": error, "protocol_sha256": phash},
    )
    save("reports/civil_quarter_replay/results.md", b"UNEVALUABLE")
    ledger = [{"event": "inherited", **r} for r in prior]
    ledger += [
        {
            "event": "registered",
            **{k: r[k] for k in ("study", "candidate", "control", "score", "horizon")},
            "protocol_sha256": phash,
        }
        for r in rows
    ]
    ledger += [
        {
            "event": "evaluated",
            "unread_diagnostic": {"opaque": "do not inspect", "number": 1e-99},
        }
        for _ in rows
    ]
    ledger += [{"event": "verification_failed", **r} for r in rows]
    save(
        "reports/civil_quarter_replay/trial_ledger.jsonl",
        ("\n".join(json.dumps(r) for r in ledger) + "\n").encode(),
    )
    outputs = {}
    for name in (
        "features.parquet",
        "targets.parquet",
        "forecasts.parquet",
        "fits.json",
        "source_closure.json",
        "support_audit.json",
        "upstream_admission.json",
    ):
        rel = "data/civil_quarter_replay/" + name
        outputs[rel] = save(rel, b"opaque unverified output bytes; never decode")
    save(
        "reports/civil_quarter_replay/numerical_failure_diagnosis.json",
        b"opaque training diagnostic; hash only",
    )
    save(
        "reports/civil_quarter_replay/unpublished_scored_metrics.json",
        b"opaque unpublished scores; hash only",
    )
    counts = {"inherited": 123, "registered": 2, "evaluated": 2, "verification_failed": 2}
    audit = {
        "status": "VERIFICATION_FAILURE_AND_PRESERVATION_AUDITED",
        "protocol_sha256": phash,
        "manifest_sha256": pins["reports/civil_quarter_replay/manifest.json"],
        "freeze_record_sha256": pins["reports/civil_quarter_replay/freeze_record.json"],
        "diagnostic_scores_read": False,
        "models_fitted": False,
        "verifier_rerun": False,
        "frozen_files_written": False,
        "canonical_publication": {
            "status": "UNEVALUABLE",
            "verification_status": "FAILED",
            "metrics_equal_failure_json": True,
            "terminal_rows_equal_canonical_metrics": True,
            "whole_wave_aborted": True,
            "hypotheses": 2,
            "cumulative_hypotheses": 125,
            "all_new_pvalues": 1.0,
            "phases_per_comparison": 0,
            "leads": [],
            "terminal_file_hashes": {
                name: pins[name]
                for name in pins
                if name.startswith("reports/civil_quarter_replay/")
                and name.rsplit("/", 1)[1]
                in (
                    "metrics.json",
                    "failure.json",
                    "verification.json",
                    "manifest.json",
                    "freeze_record.json",
                    "trial_ledger.jsonl",
                    "results.md",
                )
            },
        },
        "ledger": {
            "events": 129,
            "event_counts": counts,
            "order_verified": True,
            "inherited_and_evaluated_score_payloads_decoded": False,
        },
        "generated_outputs": {
            "files_sha256": outputs,
            "forecast_rows": 6297,
            "monthly_fit_records": 101,
            "independently_verified_forecasts": None,
            "independently_verified_monthly_fits": None,
        },
        "current_preservation": {
            "pin_mismatches": [],
            "end_pin_mismatches": [],
            "prefit_design_mismatches": [],
            "freeze_code_equals_manifest_code": True,
            "manifest_group_counts": {
                g: len(manifest[g]) for g in ("code", "inputs", "preserved")
            },
        },
    }
    save("reports/civil_quarter_replay/verification_failure_audit.json", audit)
    pub = {
        "status": "UNEVALUABLE_REPORT_AUDITED",
        "protocol_sha256": phash,
        "manifest_sha256": pins["reports/civil_quarter_replay/manifest.json"],
        "frozen_manifest_entries_checked": {
            g: len(manifest[g]) for g in ("code", "inputs", "preserved")
        },
        "prefit_code_and_design_pins_unchanged": True,
        "prior_failed_attempt_publication_unchanged": True,
        "ledger_events": 129,
        "ledger_event_counts": counts,
        "generated_unverified_forecasts": 6297,
        "generated_unverified_monthly_fits": 101,
        "cumulative_hypotheses": 125,
        "leads": [],
        "private_generated_artifact_hashes": {Path(k).name: v for k, v in outputs.items()},
        "report_artifact_hashes": {
            Path(k).name: v
            for k, v in pins.items()
            if k.startswith("reports/civil_quarter_replay/")
            and Path(k).name not in ("manifest.json", "freeze_record.json")
        },
    }
    save("reports/civil_quarter_replay/publication_audit.json", pub)
    info = {
        "protocol": "civil_quarter_replay.yaml",
        "reports": "reports/civil_quarter_replay",
        "data": "data/civil_quarter_replay",
        "protocol_sha256": phash,
        "verifier_sha256": code[verifier_name],
        "required_status": "UNEVALUABLE",
        "verification_status": "FAILED",
        "inherited_hypotheses": 123,
        "registered_hypotheses": 2,
        "cumulative_hypotheses": 125,
        "terminal_event": "verification_failed",
        "generated_forecasts": 6297,
        "generated_monthly_fits": 101,
        "unverified_output_hashes": outputs,
    }
    for key, name in (
        ("manifest", "manifest.json"),
        ("freeze_record", "freeze_record.json"),
        ("failure", "failure.json"),
        ("verification", "verification.json"),
        ("verification_failure_audit", "verification_failure_audit.json"),
        ("numerical_failure_diagnosis", "numerical_failure_diagnosis.json"),
        ("publication_audit", "publication_audit.json"),
        ("trial_ledger", "trial_ledger.jsonl"),
    ):
        info[key + "_sha256"] = pins["reports/civil_quarter_replay/" + name]
    return info, pins


class FailedVerificationContracts(unittest.TestCase):
    def test_reanchored_semantic_failure_tampering_is_rejected(self):
        cases = (
            (
                "failure.json",
                "failure_sha256",
                lambda value: value["rows"][0].update(p_conservative=0.2),
            ),
            (
                "verification.json",
                "verification_sha256",
                lambda value: value.update(status="VERIFIED"),
            ),
            (
                "verification_failure_audit.json",
                "verification_failure_audit_sha256",
                lambda value: value.update(diagnostic_scores_read=True),
            ),
        )
        for filename, anchor, change in cases:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                info, pins = failed_verification_fixture(root)
                name = info["reports"] + "/" + filename
                value = json.loads((root / name).read_bytes())
                change(value)
                (root / name).write_text(json.dumps(value))
                info[anchor] = pins[name] = verify.digest(root / name)
                if filename == "failure.json":
                    metrics = info["reports"] + "/metrics.json"
                    (root / metrics).write_text(json.dumps(value))
                    pins[metrics] = verify.digest(root / metrics)
                with self.subTest(filename=filename), self.assertRaises(AssertionError):
                    verify.verify_failed_verification(root, info, pins)

    def test_failed_verification_is_hash_admitted_without_decoding_unverified_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            info, pins = failed_verification_fixture(root)
            before = {
                str(p.relative_to(root)): p.read_bytes()
                for p in root.rglob("*")
                if p.is_file()
            }
            with (
                patch.object(
                    verify.pd,
                    "read_parquet",
                    side_effect=AssertionError("no unverified values"),
                ),
                patch.object(
                    verify.previous,
                    "verify",
                    side_effect=AssertionError("no failed verifier restart"),
                ),
            ):
                proof = verify.verify_failed_verification(root, info, pins)
            self.assertEqual(proof["status"], "FAILED_VERIFICATION_PRESERVED")
            self.assertEqual(
                proof["ledger_events_verified"],
                {"inherited": 123, "registered": 2, "evaluated": 2, "verification_failed": 2},
            )
            self.assertEqual(proof["all_new_pvalues"], 1)
            self.assertEqual(
                proof["unverified_output_hashes"], info["unverified_output_hashes"]
            )
            self.assertEqual(
                before,
                {
                    str(p.relative_to(root)): p.read_bytes()
                    for p in root.rglob("*")
                    if p.is_file()
                },
            )

    def test_changed_failure_artifact_or_unregistered_output_rejects(self):
        for relative in (
            "reports/civil_quarter_replay/failure.json",
            "reports/civil_quarter_replay/trial_ledger.jsonl",
            "reports/civil_quarter_replay/verification_failure_audit.json",
            "data/civil_quarter_replay/fits.json",
        ):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                info, pins = failed_verification_fixture(root)
                (root / relative).write_bytes(b"tampered")
                with (
                    self.subTest(path=relative),
                    self.assertRaises((AssertionError, ValueError)),
                ):
                    verify.verify_failed_verification(root, info, pins)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            info, pins = failed_verification_fixture(root)
            (root / "data/civil_quarter_replay/new_unregistered.txt").write_text("extra")
            with self.assertRaises(AssertionError):
                verify.verify_failed_verification(root, info, pins)


class ProfiledFitContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src import civil_quarter_models as producer

        frame = verify.augment_features(original_fixture())
        cls.training, cls.application = frame.iloc[:1400], frame.iloc[1400:]
        cls.y = np.exp(
            -9
            + 0.1 * cls.training.quarter_end5.to_numpy()
            + np.sin(np.arange(len(cls.training))) * 0.2
        )
        cls.predictions, cls.audits = producer.fit_predict(
            cls.training, cls.y, cls.application
        )

    def test_explicit_profiled_solve_preserves_full_objective_and_scalar_checks(self):
        with patch.object(
            verify.frozen,
            "independent_fit",
            side_effect=AssertionError("old solver forbidden"),
        ):
            replay, proof = verify.verify_fit(
                self.training, self.y, self.application, self.audits
            )
        self.assertEqual(proof["models_verified"], 3)
        self.assertIn("baseline_solver_audit", proof)
        self.assertLessEqual(proof["independent_baseline_gradient"], 1e-8 + 1e-12)
        self.assertLessEqual(proof["independent_quarter_gradient"], 1e-8 + 1e-12)
        for model in verify.MODELS:
            np.testing.assert_allclose(
                replay[model], self.predictions[model], rtol=1e-10, atol=0
            )

    def test_solver_success_cannot_override_original_full_gradient(self):
        with (
            patch.object(
                verify.profiled,
                "solve_baseline",
                return_value=(np.ones(31), {"success": True}),
            ),
            self.assertRaisesRegex(AssertionError, "stationarity"),
        ):
            verify.independent_fit(self.training, self.y, self.application)

    def test_saved_coefficients_geometry_scalar_and_support_still_reject_tampering(self):
        for fault in ("coefficient", "center", "quarter", "support"):
            audit = copy.deepcopy(self.audits)
            if fault == "coefficient":
                audit["baseline"]["scaled_beta"][2] += 0.1
            elif fault == "center":
                audit["baseline"]["means"][20] += 0.1
            elif fault == "quarter":
                audit["quarter"]["b"] += 0.1
            else:
                audit["quarter"]["support"]["groups"]["quarter_end5"]["ones"] += 1
            with self.subTest(fault=fault), self.assertRaises((ValueError, AssertionError)):
                verify.verify_fit(self.training, self.y, self.application, audit)


class PublicationContracts(unittest.TestCase):
    def test_new_failure_retains127family_and_preserves_failed_predecessors(self):
        for initial in ("{}", "{bad", '{"bad":NaN}', '{"bad":Infinity}'):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                report = root / "reports/profiled_quarter"
                report.mkdir(parents=True)
                (report / "metrics.json").write_text(initial)
                prior = []
                for folder in ("civil_quarter", "civil_quarter_replay"):
                    path = root / "reports" / folder / "failure.json"
                    path.parent.mkdir(parents=True)
                    path.write_text("original frozen failure")
                    prior.append(path)
                with (
                    patch.object(verify, "verify", side_effect=AssertionError("injected")),
                    self.assertRaises(AssertionError),
                ):
                    verify.verify_with_failure_guard(root)
                metrics = json.loads((report / "metrics.json").read_bytes())
                self.assertEqual(metrics["status"], "UNEVALUABLE")
                self.assertEqual(metrics["cumulative_hypothesis_count"], 127)
                self.assertEqual(metrics["hypothesis_count"], 2)
                self.assertEqual(metrics["leads"], [])
                self.assertTrue(
                    all(
                        row[key] == 1
                        for row in metrics["rows"]
                        for key in ("p_conservative", "p_holm_wave", "p_holm_cumulative")
                    )
                )
                self.assertTrue(
                    all(path.read_text() == "original frozen failure" for path in prior)
                )
                self.assertEqual(
                    len((report / "trial_ledger.jsonl").read_text().splitlines()), 2
                )


class EntryContracts(unittest.TestCase):
    def test_whole_entrypoint_reads_declared_artifacts_and_checks_final_protocol(self):
        import copy

        import yaml

        from tests.test_verify_civil_quarter import original_fixture

        for fault in (None, "support", "protocol"):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                report, out = (
                    root / "reports/profiled_quarter",
                    root / "data/profiled_quarter",
                )
                report.mkdir(parents=True)
                out.mkdir(parents=True)
                protocol = copy.deepcopy(verify.CONTRACT)
                (root / "profiled_quarter.yaml").write_text(yaml.safe_dump(protocol))
                phash = verify.digest(root / "profiled_quarter.yaml")
                original = original_fixture(80)
                features = verify.augment_features(original)
                dates = original.index
                maturity = pd.Series(dates, index=dates).shift(-1)
                target = pd.DataFrame(
                    {"y": 0.001, "target_end": maturity, "available_date": maturity},
                    index=dates,
                )
                target.loc[dates[-1], "y"] = np.nan
                pins = {}
                for name, frame in (
                    (protocol["feature_source"]["features"], original),
                    (protocol["feature_source"]["targets"], target),
                ):
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    frame.to_parquet(path)
                    pins[name] = verify.digest(path)
                old_protocol_name = protocol["feature_source"]["protocol"]
                old_protocol_path = root / old_protocol_name
                old_protocol_path.write_bytes((verify.ROOT / old_protocol_name).read_bytes())
                pins[old_protocol_name] = verify.digest(old_protocol_path)
                closure = {"files": {}, "counts": {"unique_files": 0}}
                (out / "source_closure.json").write_text(json.dumps(closure))
                (report / "manifest.json").write_text(
                    json.dumps(
                        {"protocol_sha256": phash, "code": {}, "inputs": pins, "preserved": {}}
                    )
                )
                proof = {
                    "status": "UPSTREAM_VERIFIED_READ_ONLY",
                    "prior_files_written": False,
                    "proofs": {
                        "failed_attempt": {"status": "FAILED_ATTEMPT_PRESERVED"},
                        "failed_verification": {"status": "FAILED_VERIFICATION_PRESERVED"},
                    },
                }
                support = {
                    "monthly_fits": 0,
                    "common_application_origins": 1,
                    "phases": [{"slices": []}, {"slices": [{}, {}]}],
                }
                (out / "upstream_admission.json").write_text(json.dumps(proof))
                (out / "support_audit.json").write_text(
                    json.dumps(
                        support if fault != "support" else {**support, "monthly_fits": 1}
                    )
                )
                features.to_parquet(out / "features.parquet")
                target.to_parquet(out / "targets.parquet")
                pd.DataFrame({"origin": [dates[30]] * 3}).to_parquet(out / "forecasts.parquet")
                (out / "fits.json").write_text("[]")
                (report / "metrics.json").write_text(
                    json.dumps(
                        {
                            "protocol_sha256": phash,
                            "evidence_class": protocol["evidence_class"],
                            "inherited_rows": [],
                        }
                    )
                )
                (report / "trial_ledger.jsonl").write_text("")

                def replay(*args, fault=fault, root=root):
                    if fault == "protocol":
                        with (root / "profiled_quarter.yaml").open("a") as stream:
                            stream.write("# injected mutation\n")
                    return {"forecasts_verified": 3}

                with (
                    patch.object(verify, "verify_manifest_coverage"),
                    patch.object(verify, "collect_sources", return_value=closure),
                    patch.object(verify, "validate_upstream", return_value=proof),
                    patch.object(verify, "preflight", return_value=support),
                    patch.object(verify, "verify_forecasts", side_effect=replay),
                    patch.object(verify, "verify_metrics", return_value={"leads": []}),
                    patch.object(
                        verify,
                        "verify_ledger",
                        return_value={"inherited": 125, "registered": 2, "evaluated": 2},
                    ),
                ):
                    if fault:
                        with self.assertRaises(AssertionError):
                            verify.verify_with_failure_guard(root)
                        self.assertEqual(
                            json.loads((report / "verification.json").read_text())["status"],
                            "FAILED",
                        )
                    else:
                        result = verify.verify(root)
                        self.assertEqual(result["status"], "VERIFIED")
                        self.assertEqual(result["primitive_proper_scores_verified"], 3)
                        self.assertEqual(result["paired_proper_differences_verified"], 2)


if __name__ == "__main__":
    unittest.main()
