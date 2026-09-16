"""Prewritten admission and whole-family failure contracts for wave28."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import copula_calibration_run as run


class RunContracts(unittest.TestCase):
    def test_old_terminal_report_hashes_resolve_in_old_report_directory(self):
        import json
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / run.OLD
            report.mkdir(parents=True)
            (report / "metrics.json").write_text("{}")
            (report / "freeze_record.json").write_text(
                json.dumps({"code": {}, "inputs": {}, "preserved": {}, "prefit": {}})
            )
            metric_hash = run.sha(report / "metrics.json")
            (report / "terminal.json").write_text(
                json.dumps(
                    {
                        "freeze_record_sha256": run.sha(report / "freeze_record.json"),
                        "metrics_sha256": metric_hash,
                        "output_hashes": {},
                        "report_artifact_hashes": {"metrics.json": metric_hash},
                    }
                )
            )
            with patch.object(run, "TERMINAL_SHA", run.sha(report / "terminal.json")):
                pins = run.source_pins(root)
            self.assertEqual(pins[str(run.OLD / "metrics.json")], metric_hash)
            self.assertNotIn("metrics.json", pins)

    def test_hash_pin_detects_mutated_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p = root / "saved"
            p.write_bytes(b"original")
            pins = {"saved": hashlib.sha256(p.read_bytes()).hexdigest()}
            run.verify_pins(root, pins)
            p.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                run.verify_pins(root, pins)

    def test_failure_retains_all_comparisons(self):
        prior = [{"p_conservative": 1.0, "id": i} for i in range(151)]
        result = run.failure_metrics(ValueError("certificate failure"), prior)
        self.assertEqual(result["status"], "UNEVALUABLE")
        self.assertEqual(result["inherited_rows"], prior)
        self.assertEqual(result["cumulative_hypothesis_count"], 158)
        self.assertEqual(len(result["rows"]), 7)
        self.assertTrue(all(x["p_conservative"] == 1.0 for x in result["rows"]))
        self.assertEqual(result["leads"], [])

    def fixture(self):
        calendar = pd.bdate_range("2020-01-01", periods=8)
        origins = calendar[1:6]
        records = []
        for model in ["t8_copula", "gaussian_copula", "independence"]:
            for origin in origins:
                records.append(
                    dict(
                        origin=origin,
                        model=model,
                        phase="evaluation",
                        offset=calendar.get_loc(origin) % 5,
                        fit_origin=origins[0],
                        training_cutoff=calendar[0],
                        feature_cutoff_date=calendar[calendar.get_loc(origin) - 1],
                        mu_qqq=0.001,
                        mu_spx=-0.001,
                        h_qqq=0.002,
                        h_spx=0.003,
                        rho=0.5 if model != "independence" else 0.0,
                    )
                )
        applications = pd.DataFrame(records)
        targets = pd.DataFrame(
            {
                "target_end": list(calendar[1:]) + [pd.NaT],
                "available_date": list(calendar[1:]) + [pd.NaT],
                "y_qqq": np.arange(8) / 1000,
                "y_spx": -np.arange(8) / 1000,
            },
            index=calendar,
        )
        coverage = pd.DataFrame({"origin": origins, "scored": True})
        return applications, targets, coverage, calendar

    def test_archive_shared_marginals_and_clock_are_bound(self):
        apps, targets, coverage, calendar = self.fixture()
        archive = run.assemble_archive(apps, targets, coverage, calendar)
        self.assertEqual(len(archive), 5)
        self.assertTrue(archive.issued.all())
        self.assertTrue(archive.eligible_scored.all())
        changed = apps.copy()
        changed.loc[changed.model == "gaussian_copula", "mu_spx"] += 0.001
        with self.assertRaises(ValueError):
            run.assemble_archive(changed, targets, coverage, calendar)
        changed = apps.copy()
        changed["training_cutoff"] = changed["fit_origin"]
        with self.assertRaises(ValueError):
            run.assemble_archive(changed, targets, coverage, calendar)

    def test_archive_retains_issued_but_phase_unscored_target(self):
        apps, targets, coverage, calendar = self.fixture()
        coverage.loc[coverage.index[-1], "scored"] = False
        archive = run.assemble_archive(apps, targets, coverage, calendar)
        self.assertEqual(len(archive), 5)
        self.assertFalse(archive.eligible_scored.iloc[-1])
        self.assertTrue(np.isfinite(archive.y_qqq.iloc[-1]))


class DurableBindingContracts(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        report, data = root / run.REPORT, root / run.DATA
        report.mkdir(parents=True)
        data.mkdir(parents=True)
        source = root / "original-source"
        source.write_bytes(b"frozen original source")
        prior = [{"id": i, "p_conservative": 1.0} for i in range(151)]
        frozen = {
            "pins": {"original-source": run.sha(source)},
            "inherited_rows": prior,
            "snapshot_sha256": hashlib.sha256(b"private snapshot").hexdigest(),
        }
        run.dump(report / "freeze.json", frozen)
        run.dump(
            report / "registration.json", {"freeze_sha256": run.sha(report / "freeze.json")}
        )
        anchors = {
            "frozen": frozen,
            "freeze_sha256": run.sha(report / "freeze.json"),
            "registration_sha256": run.sha(report / "registration.json"),
        }
        bound = {}
        for path, payload in (
            (data / "code_and_input_snapshot.zip", b"private snapshot"),
            (data / "panel.parquet", b"generated stand-in for previously verified parquet"),
            (report / "forecast_verification.json", b'{"status":"VERIFIED"}\n'),
            (report / "score_verification.json", b'{"status":"VERIFIED"}\n'),
        ):
            run.save_bound_bytes(root, path, payload, bound)
        metrics = {
            "status": "COMPLETED",
            "rows": [{"contrast": c, "p_conservative": 0.01} for c in run.CONTRASTS],
            "inherited_rows": prior,
            "hypothesis_count": 7,
            "cumulative_hypothesis_count": 158,
            "leads": [],
        }
        return root, report, data, bound, anchors, metrics

    def test_exclusive_first_save_and_checked_reload_preserve_expected_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, bound = root / "private" / "generated.json", {}
            payload = b'{"generated":true}\n'
            expected = hashlib.sha256(payload).hexdigest()
            self.assertEqual(run.save_bound_bytes(root, path, payload, bound), expected)
            self.assertEqual(bound, {"private/generated.json": expected})
            self.assertEqual(run.read_bound_bytes(root, path, bound), payload)
            with self.assertRaises((ValueError, FileExistsError)):
                run.save_bound_bytes(root, path, b"replacement", bound)
            self.assertEqual(path.read_bytes(), payload)
            path.write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                run.read_bound_bytes(root, path, bound)
            self.assertEqual(bound["private/generated.json"], expected)

    def test_mutation_during_first_write_cannot_be_adopted_as_expected_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, bound = root / "artifact", {}
            payload = b"intended serialized bytes"
            original_fsync = run.os.fsync

            def mutate(fd):
                path.write_bytes(b"changed during first write")
                original_fsync(fd)

            with (
                patch.object(run.os, "fsync", side_effect=mutate),
                self.assertRaises(ValueError),
            ):
                run.save_bound_bytes(root, path, payload, bound)
            self.assertEqual(bound["artifact"], hashlib.sha256(payload).hexdigest())
            self.assertNotEqual(bound["artifact"], run.sha(path))

    def test_successful_terminal_uses_verified_data_and_proof_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root, report, _data, bound, anchors, metrics = self.fixture(directory)
            prior_bound = dict(bound)
            terminal = run.commit_verified(root, report, metrics, bound, **anchors)
            self.assertEqual(terminal["status"], "COMPLETED_VERIFIED_MECHANISM_DIAGNOSTIC")
            self.assertEqual(terminal["freeze_sha256"], anchors["freeze_sha256"])
            for name, expected in prior_bound.items():
                mapping = (
                    terminal["output_hashes"]
                    if name.startswith(str(run.DATA))
                    else terminal["report_artifact_hashes"]
                )
                self.assertEqual(mapping[name], expected)
            self.assertEqual(
                terminal["metrics_sha256"], bound[str(run.REPORT / "metrics.json")]
            )
            self.assertEqual(json.loads((report / "metrics.json").read_bytes()), metrics)
            self.assertFalse((report / "failure.json").exists())
            run.verify_pins(root, bound)

    def test_mutation_during_metrics_write_invalidates_all_seven_and_preserves_provisional_metrics(
        self,
    ):
        for kind in ("data", "proof", "frozen_source"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root, report, data, bound, anchors, metrics = self.fixture(directory)
                panel_key = str(run.DATA / "panel.parquet")
                expected_panel = bound[panel_key]
                original_save = run.save_bound_bytes
                target = {
                    "data": data / "panel.parquet",
                    "proof": report / "forecast_verification.json",
                    "frozen_source": root / "original-source",
                }[kind]

                def mutate(root_arg, path, payload, pins, save=original_save, damaged=target):
                    result = save(root_arg, path, payload, pins)
                    if (
                        Path(path).name == "metrics.json"
                        and json.loads(payload)["status"] == "COMPLETED"
                    ):
                        damaged.write_bytes(b"late mutation")
                    return result

                with patch.object(run, "save_bound_bytes", side_effect=mutate):
                    terminal = run.commit_verified(root, report, metrics, bound, **anchors)
                self.assert_failed_family(report, terminal, metrics["inherited_rows"])
                self.assertEqual(
                    json.loads((report / "metrics.pre_invalidation.json").read_bytes()),
                    metrics,
                )
                self.assertEqual(
                    terminal["expected_unverified_output_hashes"][panel_key], expected_panel
                )
                self.assertEqual(target.read_bytes(), b"late mutation")

    def test_metrics_mutation_during_terminal_write_cannot_receive_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root, report, _data, bound, anchors, metrics = self.fixture(directory)
            original_save = run.save_bound_bytes
            changed = b'{"status":"COMPLETED","rows":[]}\n'

            def mutate(root_arg, path, payload, pins):
                result = original_save(root_arg, path, payload, pins)
                if Path(path).name == "terminal.json" and json.loads(payload)[
                    "status"
                ].startswith("COMPLETED"):
                    (report / "metrics.json").write_bytes(changed)
                return result

            with patch.object(run, "save_bound_bytes", side_effect=mutate):
                terminal = run.commit_verified(root, report, metrics, bound, **anchors)
            self.assert_failed_family(report, terminal, metrics["inherited_rows"])
            self.assertEqual((report / "metrics.pre_invalidation.json").read_bytes(), changed)
            self.assertEqual(
                json.loads((report / "terminal.pre_invalidation.json").read_bytes())["status"],
                "COMPLETED_VERIFIED_MECHANISM_DIAGNOSTIC",
            )

    def assert_failed_family(self, report, terminal, prior):
        self.assertEqual(terminal["status"], "UNEVALUABLE")
        self.assertEqual(terminal["output_hashes"], {})
        metrics = json.loads((report / "metrics.json").read_bytes())
        self.assertEqual(metrics["status"], "UNEVALUABLE")
        self.assertEqual(metrics["inherited_rows"], prior)
        self.assertEqual(metrics["cumulative_hypothesis_count"], 158)
        self.assertEqual(len(metrics["rows"]), 7)
        self.assertTrue(
            all(
                row["p_conservative"] == row["p_holm_wave"] == row["p_holm_cumulative"] == 1.0
                for row in metrics["rows"]
            )
        )
        self.assertEqual(terminal["metrics_sha256"], run.sha(report / "metrics.json"))
        self.assertEqual(json.loads((report / "terminal.json").read_bytes()), terminal)


if __name__ == "__main__":
    unittest.main()
