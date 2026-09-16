"""Generated-only transaction tests; no historical admission or model calls."""

import copy
import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import yaml

from src import event_cluster_search as runner


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(report):
    return [
        json.loads(line) for line in (report / "trial_ledger.jsonl").read_text().splitlines()
    ]


def metric_fixture():
    return {
        "status": "SCORED",
        "leads": [],
        "hypothesis_count": 3,
        "cumulative_hypothesis_count": 134,
        "rows": [
            {
                "study": "event_cluster",
                "candidate": candidate,
                "control": control,
                "score": score,
                "horizon": 1,
                "p_conservative": 1.0,
                "p_holm_wave": 1.0,
                "p_holm_cumulative": 1.0,
                "verdict": "DOES_NOT_QUALIFY",
                "phases": [
                    {"name": phase, "delta": -0.001} for phase in ("development", "evaluation")
                ],
            }
            for candidate, control, score in runner.CONTRASTS
        ],
    }


class TransactionContracts(unittest.TestCase):
    def fixture(self, root, stack):
        report = root / "reports/event_cluster"
        report.mkdir(parents=True)
        (root / "data").mkdir()
        protocol = root / "event_cluster.yaml"
        protocol.write_text(yaml.safe_dump(copy.deepcopy(runner.CONTRACT)))
        code = (
            "src/event_cluster_models.py",
            "src/old_model.py",
            "tests/test_event_cluster_publication.py",
            "tests/test_old_model.py",
        )
        for name in code:
            path = root / name
            path.parent.mkdir(exist_ok=True)
            path.write_text("# Generated transaction fixture, not a historical module.\n")
        design = report / "DESIGN.md"
        design.write_text("Synthetic pre-fit design\n")
        full_log = report / "full_repository_tests.txt"
        full_log.write_text("Synthetic full-suite evidence\n")
        previous = root / "reports/previous/results.md"
        previous.parent.mkdir()
        previous.write_text("Prior frozen result\n")
        (root / "previous.yaml").write_text("prior: unchanged\n")
        source = root / "input.json"
        source.write_text('{"synthetic_input":true}\n')
        prior_metrics = root / "reports/previous/metrics.json"
        prior_metrics.write_text('{"synthetic_prior":true}\n')
        freeze = {
            "protocol_sha256": digest(protocol),
            "code": {name: digest(root / name) for name in code},
            "prefit_design": {str(design.relative_to(root)): digest(design)},
            "checks": {"full_log_sha256": digest(full_log)},
        }
        (report / "freeze_record.json").write_text(json.dumps(freeze))
        inputs = {
            "input.json",
            "reports/previous/metrics.json",
            "reports/event_cluster/freeze_record.json",
            "reports/event_cluster/DESIGN.md",
        }
        prior = [
            {
                "candidate": f"previous_{i}",
                "control": "prior_control",
                "horizon": 1,
                "p_conservative": 1.0,
            }
            for i in range(131)
        ]
        for name, value in {
            "ROOT": root,
            "PROTOCOL": protocol,
            "REPORT": report,
            "OUT": root / "data/event_cluster",
        }.items():
            stack.enter_context(patch.object(runner, name, value))
        stack.enter_context(patch.object(runner, "input_paths", return_value=inputs))
        stack.enter_context(patch.object(runner, "inherited", return_value=prior))
        stack.enter_context(
            patch.object(
                runner.subprocess,
                "run",
                return_value=SimpleNamespace(
                    returncode=0, stdout="synthetic checks", stderr=""
                ),
            )
        )
        # Admission is always mocked; accidental continuation cannot touch old data.
        stack.enter_context(
            patch.object(
                runner, "validate_upstream", side_effect=ValueError("synthetic admission stop")
            )
        )
        return report, inputs

    def successful_pipeline(self, stack):
        index = pd.DatetimeIndex(["2016-01-04"], name="date")
        frame = pd.DataFrame({"origin": index, "value": [1.0]}, index=index)
        loaded = {
            "features": frame,
            "targets": frame,
            "forecasts": frame,
            "fits": [{}],
            "states": frame,
            "protocol": {"index": {}},
        }
        stack.enter_context(
            patch.object(
                runner, "validate_upstream", return_value=({"synthetic": True}, loaded)
            )
        )
        stack.enter_context(
            patch.object(
                runner.sm,
                "forecast_panel",
                return_value=(frame, [{}], frame, frame, {"synthetic_support": True}),
            )
        )
        stack.enter_context(patch.object(runner, "evaluate", return_value=metric_fixture()))

    def assert_failure(self, report, *, inherited=True):
        failure = json.loads((report / "metrics.json").read_bytes())
        self.assertEqual(failure, json.loads((report / "failure.json").read_bytes()))
        self.assertEqual(failure["status"], "UNEVALUABLE")
        self.assertTrue(failure["whole_wave_aborted"])
        self.assertEqual(failure["leads"], [])
        self.assertEqual(failure["hypothesis_count"], 3)
        self.assertEqual(failure["cumulative_hypothesis_count"], 134)
        self.assertEqual(
            [(row["candidate"], row["control"], row["score"]) for row in failure["rows"]],
            list(runner.CONTRASTS),
        )
        for row in failure["rows"]:
            self.assertEqual(row["phases"], [])
            for key in ("p_conservative", "p_holm_wave", "p_holm_cumulative"):
                self.assertEqual(row[key], 1.0)
        expected = (
            ["registered"] * 3
            + ["inherited"] * (131 if inherited else 0)
            + ["unevaluable"] * 3
        )
        self.assertEqual([row["event"] for row in records(report)], expected)
        self.assertEqual(failure["inherited_rows_available"], inherited)
        self.assertEqual(failure["inherited_rows_reconstructed"], 131 if inherited else 0)
        self.assertIn("UNEVALUABLE", (report / "results.md").read_text())
        return failure

    def test_three_registrations_exist_before_manifest_input_admission(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder)
            report, _ = self.fixture(root, stack)

            def stop(_protocol):
                self.assertEqual([row["event"] for row in records(report)], ["registered"] * 3)
                self.assertFalse((report / "manifest.json").exists())
                raise ValueError("manifest input failure")

            with (
                patch.object(runner, "input_paths", side_effect=stop),
                self.assertRaisesRegex(ValueError, "manifest input failure"),
            ):
                runner.run()
            self.assert_failure(report, inherited=False)

    def test_source_admission_failure_retains_all_three_before_any_fit(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder)
            report, _ = self.fixture(root, stack)

            def stop(*_args):
                self.assertTrue((report / "manifest.json").exists())
                self.assertEqual(
                    [row["event"] for row in records(report)][:3], ["registered"] * 3
                )
                self.assertEqual(len(records(report)), 134)
                raise ValueError("admission failed")

            with (
                patch.object(runner, "validate_upstream", side_effect=stop),
                patch.object(runner.sm, "forecast_panel") as fit,
            ):
                with self.assertRaisesRegex(ValueError, "admission failed"):
                    runner.run()
                fit.assert_not_called()
            self.assert_failure(report)
            self.assertFalse((runner.OUT / "forecasts.parquet").exists())

    def test_manifest_contains_every_new_and_prior_code_input_and_preserved_file(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder)
            report, inputs = self.fixture(root, stack)
            self.successful_pipeline(stack)
            runner.run()
            manifest = json.loads((report / "manifest.json").read_bytes())
            self.assertEqual(
                set(manifest["code"]),
                {
                    str(path.relative_to(root))
                    for directory in ("src", "tests")
                    for path in (root / directory).rglob("*.py")
                },
            )
            self.assertEqual(set(manifest["inputs"]), inputs)
            self.assertIn("reports/previous/results.md", manifest["preserved"])
            self.assertIn("previous.yaml", manifest["preserved"])
            for group in ("code", "inputs", "preserved"):
                for name, expected in manifest[group].items():
                    self.assertEqual(expected, digest(root / name))
            events = [row["event"] for row in records(report)]
            self.assertEqual(
                events, ["registered"] * 3 + ["inherited"] * 131 + ["evaluated"] * 3
            )
            self.assertFalse((report / "failure.json").exists())

    def test_pretest_failure_never_registers_or_admits(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            report, _ = self.fixture(Path(folder), stack)
            with (
                patch.object(
                    runner.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=1, stdout="failed", stderr=""),
                ),
                patch.object(runner, "validate_upstream") as admit,
            ):
                with self.assertRaisesRegex(RuntimeError, "Prewritten"):
                    runner.run()
                admit.assert_not_called()
            self.assertFalse((report / "trial_ledger.jsonl").exists())
            self.assertFalse((report / "manifest.json").exists())

    def test_pretest_mutation_of_frozen_evidence_blocks_registration(self):
        for mode in ("code", "added_code", "protocol", "design", "full_log"):
            with (
                self.subTest(mode=mode),
                tempfile.TemporaryDirectory() as folder,
                ExitStack() as stack,
            ):
                root = Path(folder)
                report, _ = self.fixture(root, stack)

                def mutate(*_args, root=root, report=report, mode=mode, **_kwargs):
                    paths = {
                        "code": root / "src/old_model.py",
                        "added_code": root / "src/unfrozen_new.py",
                        "protocol": runner.PROTOCOL,
                        "design": report / "DESIGN.md",
                        "full_log": report / "full_repository_tests.txt",
                    }
                    with paths[mode].open("a") as stream:
                        stream.write("\nchanged\n")
                    return SimpleNamespace(returncode=0, stdout="claimed success", stderr="")

                with (
                    patch.object(runner.subprocess, "run", side_effect=mutate),
                    self.assertRaises(ValueError),
                ):
                    runner.run()
                self.assertFalse((report / "trial_ledger.jsonl").exists())

    def test_full_original_history_support_failure_keeps_three_p1(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            report, _ = self.fixture(Path(folder), stack)
            self.successful_pipeline(stack)
            with (
                patch.object(
                    runner.sm,
                    "forecast_panel",
                    side_effect=ValueError("INSUFFICIENT_DATA: original history unknown"),
                ),
                patch.object(runner, "evaluate") as score,
            ):
                with self.assertRaisesRegex(ValueError, "original history unknown"):
                    runner.run()
                score.assert_not_called()
            failure = self.assert_failure(report)
            self.assertTrue(
                all(row["status"] == "INSUFFICIENT_DATA" for row in failure["rows"])
            )
            self.assertFalse((runner.OUT / "forecasts.parquet").exists())

    def test_missing_inherited_evidence_is_explicit_not_fabricated(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            report, _ = self.fixture(Path(folder), stack)
            with (
                patch.object(runner, "inherited", side_effect=ValueError("missing prior")),
                self.assertRaisesRegex(ValueError, "missing prior"),
            ):
                runner.run()
            self.assert_failure(report, inherited=False)

    def test_commit_rechecks_all_frozen_content_and_code_inventory(self):
        for mode in (
            "code",
            "input",
            "preserved",
            "protocol",
            "design",
            "full_log",
            "added_code",
        ):
            with (
                self.subTest(mode=mode),
                tempfile.TemporaryDirectory() as folder,
                ExitStack() as stack,
            ):
                root = Path(folder)
                report, _ = self.fixture(root, stack)
                self.successful_pipeline(stack)

                def mutate(*_args, root=root, report=report, mode=mode, **_kwargs):
                    paths = {
                        "code": root / "src/event_cluster_models.py",
                        "input": root / "input.json",
                        "preserved": root / "reports/previous/results.md",
                        "protocol": runner.PROTOCOL,
                        "design": report / "DESIGN.md",
                        "full_log": report / "full_repository_tests.txt",
                        "added_code": root / "src/added_during_run.py",
                    }
                    with paths[mode].open("a") as stream:
                        stream.write("\nchanged\n")
                    return metric_fixture()

                with (
                    patch.object(runner, "evaluate", side_effect=mutate),
                    self.assertRaises(ValueError),
                ):
                    runner.run()
                self.assert_failure(report)
                diagnostic = json.loads(
                    (report / "unpublished_scored_metrics.json").read_bytes()
                )
                self.assertTrue(diagnostic["not_for_inherited_inference_or_promotion"])

    def test_new_upstream_failure_marker_blocks_commit_at_both_late_stages(self):
        for upstream in ("range_alert", "issued_calibration"):
            for stage in ("scoring", "report"):
                with (
                    self.subTest(upstream=upstream, stage=stage),
                    tempfile.TemporaryDirectory() as folder,
                    ExitStack() as stack,
                ):
                    root = Path(folder)
                    report, _ = self.fixture(root, stack)
                    self.successful_pipeline(stack)
                    original_report = runner.report

                    def mark(root=root, upstream=upstream):
                        path = root / f"reports/{upstream}/failure.json"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text('{"status":"FAILED"}')

                    def scoring(*_args, mark=mark, **_kwargs):
                        mark()
                        return metric_fixture()

                    def reporting(metrics, original_report=original_report, mark=mark):
                        original_report(metrics)
                        mark()

                    method, action = (
                        ("evaluate", scoring) if stage == "scoring" else ("report", reporting)
                    )
                    with (
                        patch.object(runner, method, side_effect=action),
                        self.assertRaisesRegex(ValueError, "upstream failure"),
                    ):
                        runner.run()
                    self.assert_failure(report)
                    self.assertTrue((report / "unpublished_scored_metrics.json").exists())

    def test_partial_registration_retained_and_terminal_three_reconstructed(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            report, _ = self.fixture(Path(folder), stack)

            def partial(rows):
                (report / "trial_ledger.jsonl").write_text(json.dumps(rows[0]) + "\n{partial")
                raise OSError("partial registration")

            with (
                patch.object(runner, "ledger", side_effect=partial),
                self.assertRaisesRegex(OSError, "partial registration"),
            ):
                runner.run()
            self.assert_failure(report, inherited=False)
            self.assertTrue(
                (report / "interrupted_trial_ledger.jsonl").read_text().endswith("{partial")
            )

    def test_report_and_partial_evaluated_write_failures_preserve_scored_diagnostic(self):
        for stage in ("report", "evaluated"):
            with (
                self.subTest(stage=stage),
                tempfile.TemporaryDirectory() as folder,
                ExitStack() as stack,
            ):
                report, _ = self.fixture(Path(folder), stack)
                self.successful_pipeline(stack)
                if stage == "report":
                    stack.enter_context(
                        patch.object(runner, "report", side_effect=OSError("report failed"))
                    )
                else:
                    original_ledger = runner.ledger

                    def partial(rows, report=report, original_ledger=original_ledger):
                        if rows[0]["event"] == "evaluated":
                            with (report / "trial_ledger.jsonl").open("a") as stream:
                                stream.write(json.dumps(rows[0]) + "\n{partial")
                            raise OSError("evaluated failed")
                        original_ledger(rows)

                    stack.enter_context(patch.object(runner, "ledger", side_effect=partial))
                with self.assertRaisesRegex(OSError, stage + " failed"):
                    runner.run()
                self.assert_failure(report)
                self.assertTrue((report / "unpublished_scored_metrics.json").exists())
                self.assertTrue((report / "interrupted_trial_ledger.jsonl").exists())

    def test_existing_registration_or_private_output_never_overwritten(self):
        for existing in ("manifest.json", "trial_ledger.jsonl", "failure.json", "private"):
            with (
                self.subTest(existing=existing),
                tempfile.TemporaryDirectory() as folder,
                ExitStack() as stack,
            ):
                root = Path(folder)
                report, _ = self.fixture(root, stack)
                if existing == "private":
                    runner.OUT.mkdir()
                    path = runner.OUT / "forecasts.parquet"
                else:
                    path = report / existing
                path.write_bytes(b"unchanged prior")
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    runner.run()
                self.assertEqual(path.read_bytes(), b"unchanged prior")

    def test_atomic_nonfinite_serialization_cannot_corrupt_existing_ledger(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(runner, "REPORT", Path(folder)),
        ):
            runner.ledger([{"event": "registered"}])
            path = Path(folder) / "trial_ledger.jsonl"
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                runner.ledger([{"p": float("nan")}])
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(list(Path(folder).glob(".trial-ledger-*")))


if __name__ == "__main__":
    unittest.main()
