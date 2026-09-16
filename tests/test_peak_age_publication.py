"""Prewritten temporary-data publication contracts; no historical admission."""

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

from src import peak_age_search as runner

REQUIRED_PINS = tuple("reports/event_cluster/" + name for name in
                      ("failure.json", "metrics.json", "verification.json")) + tuple(
    "reports/event_cluster_replay/" + name for name in
    ("metrics.json", "verification.json", "publication_audit.json"))
SUCCESSFUL_OR_CURRENT = ("index_hinge", "range_alert", "issued_calibration",
                         "event_cluster_replay", "peak_age")
DATA_OUTPUTS = {"upstream_admission.json", "features.parquet", "targets.parquet",
                "feature_states.parquet", "forecasts.parquet", "fits.json",
                "application_states.parquet", "support_audit.json"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(report):
    return [json.loads(line) for line in (report / "trial_ledger.jsonl").read_text().splitlines()]


def metric_fixture():
    return {
        "status": "SCORED", "leads": [], "hypothesis_count": 3,
        "cumulative_hypothesis_count": 140,
        "rows": [{"study": "peak_age", "candidate": candidate, "control": control,
                  "horizon": horizon, "p_conservative": 1., "p_holm_wave": 1.,
                  "p_holm_cumulative": 1., "verdict": "DOES_NOT_QUALIFY",
                  "phases": [{"name": phase, "delta": -.001} for phase in
                             ("development", "evaluation")]}
                 for candidate, control, horizon in runner.CONTRASTS],
    }


class PeakAgePublicationContracts(unittest.TestCase):
    def fixture(self, root, stack):
        report = root / "reports/peak_age"
        report.mkdir(parents=True)
        (root / "data").mkdir()
        protocol = root / "peak_age.yaml"
        protocol.write_text(yaml.safe_dump(copy.deepcopy(runner.CONTRACT)))
        code = ("src/peak_age_models.py", "src/old_model.py",
                "tests/test_peak_age_publication.py", "tests/test_old_model.py")
        for name in code:
            path = root / name
            path.parent.mkdir(exist_ok=True)
            path.write_text("# Generated transaction fixture; no historical program.\n")
        design = report / "DESIGN.md"
        design.write_text("Generated pre-fit design\n")
        full_log = report / "full_repository_tests.txt"
        full_log.write_text("Generated full-suite evidence\n")
        prior_report = root / "reports/previous/results.md"
        prior_report.parent.mkdir()
        prior_report.write_text("Prior frozen result fixture\n")
        (root / "previous.yaml").write_text("prior: unchanged\n")
        (root / "input.json").write_text('{"synthetic_input":true}\n')
        (root / "reports/previous/metrics.json").write_text('{"synthetic_prior":true}\n')
        for name in REQUIRED_PINS:
            path = root / name
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps({"synthetic_required_record": name}) + "\n")
        freeze = {"protocol_sha256": digest(protocol),
                  "code": {name: digest(root / name) for name in code},
                  "prefit_design": {str(design.relative_to(root)): digest(design)},
                  "checks": {"full_log_sha256": digest(full_log)}}
        (report / "freeze_record.json").write_text(json.dumps(freeze))
        inputs = {"input.json", "reports/previous/metrics.json",
                  "reports/peak_age/freeze_record.json", "reports/peak_age/DESIGN.md", *REQUIRED_PINS}
        prior = [{"candidate": f"previous_{i}", "control": "prior_control", "horizon": 1,
                  "p_conservative": 1.} for i in range(137)]
        for name, value in {"ROOT": root, "PROTOCOL": protocol, "REPORT": report,
                            "OUT": root / "data/peak_age"}.items():
            stack.enter_context(patch.object(runner, name, value))
        stack.enter_context(patch.object(runner, "input_paths", return_value=inputs))
        stack.enter_context(patch.object(runner, "inherited", return_value=prior))
        stack.enter_context(patch.object(runner.subprocess, "run", return_value=SimpleNamespace(
            returncode=0, stdout="generated checks", stderr="")))
        # Always replace real admission. Accidental continuation cannot open raw data.
        stack.enter_context(patch.object(runner, "validate_upstream", side_effect=ValueError("generated admission stop")))
        return report, inputs

    def successful_pipeline(self, stack):
        index = pd.DatetimeIndex(["2016-01-04", "2016-01-05"], name="date")
        frame = pd.DataFrame({"origin": index, "value": [1., 2.]}, index=index)
        panel = pd.DataFrame({"origin": [index[0]] * 4, "model": ["mean", "baseline", "depth", "peak_age"],
                              "horizon": [21] * 4, "phase": ["development"] * 4,
                              "prediction": [0.] * 4, "y": [.01] * 4, "loss": [.0001] * 4})
        fits = [{"horizon": 21, "fit_origin": "2016-01-04", "feature_cutoff_date": "2015-12-31",
                 "train_n": 1000, "train_origins": [], "application_origins": [str(d.date()) for d in index]}]
        states = pd.DataFrame({"origin": index, "fit_origin": [index[0]] * 2,
                               "feature_cutoff_date": pd.to_datetime(["2015-12-31", "2016-01-04"]),
                               "train_n": [1000] * 2, "pred_mean": [0., 0.], "pred_baseline": [0., 0.],
                               "pred_depth": [0., 0.], "pred_peak_age": [0., 0.]}, index=index)
        support = {"monthly_schedules": 1, "common_application_origins": 2, "synthetic": True}
        stack.enter_context(patch.object(runner, "validate_upstream", return_value=({"synthetic": True}, frame, frame)))
        stack.enter_context(patch.object(runner, "produce", return_value=(frame, frame, frame, panel, fits, states, support)))
        stack.enter_context(patch.object(runner, "evaluate", return_value=metric_fixture()))

        def generated_report(_metrics):
            (runner.REPORT / "results.md").write_text("Generated scored report\n")

        stack.enter_context(patch.object(runner, "report", side_effect=generated_report))

    def assert_failure(self, report, inherited=True):
        failure = json.loads((report / "metrics.json").read_bytes())
        self.assertEqual(failure, json.loads((report / "failure.json").read_bytes()))
        self.assertEqual(failure["status"], "UNEVALUABLE")
        self.assertTrue(failure["whole_wave_aborted"])
        self.assertEqual(failure["leads"], [])
        self.assertEqual(failure["hypothesis_count"], 3)
        self.assertEqual(failure["cumulative_hypothesis_count"], 140)
        self.assertEqual([(r["candidate"], r["control"], r["horizon"]) for r in failure["rows"]],
                         list(runner.CONTRASTS))
        for row in failure["rows"]:
            self.assertEqual(row["phases"], [])
            for key in ["p_conservative", "p_holm_wave", "p_holm_cumulative"]:
                self.assertEqual(row[key], 1.)
        events = ["registered"] * 3 + ["inherited"] * (137 if inherited else 0) + ["unevaluable"] * 3
        self.assertEqual([r["event"] for r in records(report)], events)
        self.assertEqual(failure["inherited_rows_available"], inherited)
        self.assertEqual(failure["inherited_rows_reconstructed"], 137 if inherited else 0)
        self.assertIn("UNEVALUABLE", (report / "results.md").read_text())
        return failure

    def test_run_entrypoint_is_implemented(self):
        self.assertTrue(callable(runner.run))

    def test_registration_precedes_even_manifest_input_enumeration(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            report, _ = self.fixture(Path(folder), stack)

            def stop(_protocol):
                self.assertEqual([r["event"] for r in records(report)], ["registered"] * 3)
                self.assertFalse((report / "manifest.json").exists())
                raise ValueError("input enumeration failed")

            with patch.object(runner, "input_paths", side_effect=stop), self.assertRaisesRegex(ValueError, "input enumeration failed"):
                runner.run()
            self.assert_failure(report, inherited=False)

    def test_admission_failure_has_three_registrations_and137_inherited_before_fit(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            report, _ = self.fixture(Path(folder), stack)

            def stop(*_args):
                self.assertTrue((report / "manifest.json").exists())
                self.assertEqual([r["event"] for r in records(report)], ["registered"] * 3 + ["inherited"] * 137)
                raise ValueError("admission failed")

            with patch.object(runner, "validate_upstream", side_effect=stop), patch.object(runner, "produce") as produce:
                with self.assertRaisesRegex(ValueError, "admission failed"):
                    runner.run()
                produce.assert_not_called()
            self.assert_failure(report)
            self.assertFalse((runner.OUT / "forecasts.parquet").exists())

    def test_success_has_full_inventory_exactten_outputs143events_and_unscored_applications(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder)
            report, inputs = self.fixture(root, stack)
            self.successful_pipeline(stack)
            runner.run()
            manifest = json.loads((report / "manifest.json").read_bytes())
            expected_code = {str(p.relative_to(root)) for directory in ["src", "tests"]
                             for p in (root / directory).rglob("*.py")}
            self.assertEqual(set(manifest["code"]), expected_code)
            self.assertEqual(set(manifest["inputs"]), inputs)
            self.assertIn("previous.yaml", manifest["preserved"])
            self.assertIn("reports/previous/results.md", manifest["preserved"])
            for group in ["code", "inputs", "preserved"]:
                for name, expected in manifest[group].items():
                    self.assertEqual(digest(root / name), expected)
            self.assertEqual({p.name for p in runner.OUT.iterdir()}, DATA_OUTPUTS)
            self.assertEqual(len(DATA_OUTPUTS) + 2, 10)
            applications = pd.read_parquet(runner.OUT / "application_states.parquet")
            scored = pd.read_parquet(runner.OUT / "forecasts.parquet")
            self.assertEqual(len(applications), 2)
            self.assertEqual(scored.origin.nunique(), 1)
            self.assertEqual([r["event"] for r in records(report)], ["registered"] * 3 + ["inherited"] * 137 + ["evaluated"] * 3)
            self.assertEqual(len(records(report)), 143)
            self.assertFalse((report / "failure.json").exists())

    def test_pretest_failure_has_no_registration_or_source_admission(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            report, _ = self.fixture(Path(folder), stack)
            with patch.object(runner.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout="failed", stderr="")), \
                    patch.object(runner, "validate_upstream") as admit:
                with self.assertRaisesRegex(RuntimeError, "Prewritten"):
                    runner.run()
                admit.assert_not_called()
            self.assertFalse((report / "trial_ledger.jsonl").exists())
            self.assertFalse((report / "manifest.json").exists())

    def test_pretests_recheck_frozen_code_inventory_protocol_design_and_full_suite_evidence(self):
        for mode in ["code", "added_code", "protocol", "design", "full_log"]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
                root = Path(folder)
                report, _ = self.fixture(root, stack)

                def mutate(*_args, root=root, report=report, mode=mode, **_kwargs):
                    paths = {"code": root / "src/old_model.py", "added_code": root / "src/unfrozen.py",
                             "protocol": runner.PROTOCOL, "design": report / "DESIGN.md",
                             "full_log": report / "full_repository_tests.txt"}
                    with paths[mode].open("a") as stream:
                        stream.write("\nchanged\n")
                    return SimpleNamespace(returncode=0, stdout="claimed pass", stderr="")

                with patch.object(runner.subprocess, "run", side_effect=mutate), self.assertRaises(ValueError):
                    runner.run()
                self.assertFalse((report / "trial_ledger.jsonl").exists())

    def test_produce_support_or_fit_failure_retains_all_three_without_scoring(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            report, _ = self.fixture(Path(folder), stack)
            self.successful_pipeline(stack)
            with patch.object(runner, "produce", side_effect=ValueError("INSUFFICIENT_DATA: generated support")), \
                    patch.object(runner, "evaluate") as score:
                with self.assertRaisesRegex(ValueError, "generated support"):
                    runner.run()
                score.assert_not_called()
            self.assert_failure(report)
            self.assertFalse((runner.OUT / "forecasts.parquet").exists())

    def test_score_failure_keeps_written_diagnostics_and_canonical_p1(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            report, _ = self.fixture(Path(folder), stack)
            self.successful_pipeline(stack)
            with patch.object(runner, "evaluate", side_effect=ValueError("score failed")), self.assertRaisesRegex(ValueError, "score failed"):
                runner.run()
            self.assert_failure(report)
            self.assertTrue((runner.OUT / "application_states.parquet").exists())
            self.assertFalse((report / "unpublished_scored_metrics.json").exists())

    def test_missing_inherited_evidence_is_explicit_and_never_fabricated(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            report, _ = self.fixture(Path(folder), stack)
            with patch.object(runner, "inherited", side_effect=ValueError("missing inherited")), self.assertRaisesRegex(ValueError, "missing inherited"):
                runner.run()
            self.assert_failure(report, inherited=False)

    def test_commit_rechecks_all_frozen_files_and_entire_code_inventory(self):
        for mode in ["code", "input", "preserved", "protocol", "design", "full_log", "added_code"]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
                root = Path(folder)
                report, _ = self.fixture(root, stack)
                self.successful_pipeline(stack)

                def mutate(*_args, root=root, report=report, mode=mode, **_kwargs):
                    paths = {"code": root / "src/old_model.py", "input": root / "input.json",
                             "preserved": root / "reports/previous/results.md", "protocol": runner.PROTOCOL,
                             "design": report / "DESIGN.md", "full_log": report / "full_repository_tests.txt",
                             "added_code": root / "src/unfrozen_after_fit.py"}
                    with paths[mode].open("a") as stream:
                        stream.write("\nchanged\n")
                    return metric_fixture()

                with patch.object(runner, "evaluate", side_effect=mutate), self.assertRaises(ValueError):
                    runner.run()
                self.assert_failure(report)
                saved = json.loads((report / "unpublished_scored_metrics.json").read_bytes())
                self.assertTrue(saved["not_for_inherited_inference_or_promotion"])

    def test_required_failed20_and_latest21_pins_rechecked_after_general_hash_loop(self):
        for name in REQUIRED_PINS:
            for action in ["mutate", "remove"]:
                with self.subTest(name=name, action=action), tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
                    root = Path(folder)
                    report, _ = self.fixture(root, stack)
                    self.successful_pipeline(stack)
                    original_digest, original_report = runner.inference.digest, runner.report
                    state = {"armed": False, "mutated": False}

                    def staging(metrics, original_report=original_report, state=state):
                        original_report(metrics)
                        state["armed"] = True

                    def late_digest(path, original_digest=original_digest, state=state,
                                    root=root, name=name, action=action):
                        value = original_digest(path)
                        if state["armed"] and not state["mutated"] and Path(path) == root / "reports/previous/results.md":
                            target = root / name
                            if action == "remove":
                                target.unlink()
                            else:
                                target.write_bytes(b"late invalidation of required prior pin")
                            state["mutated"] = True
                        return value

                    with patch.object(runner, "report", side_effect=staging), \
                            patch.object(runner.inference, "digest", side_effect=late_digest), self.assertRaises(ValueError):
                        runner.run()
                    self.assertTrue(state["mutated"])
                    self.assert_failure(report)

    def test_successful_ancestor_or_current_markers_block_score_and_report_boundaries(self):
        for ancestor in SUCCESSFUL_OR_CURRENT:
            for stage in ["scoring", "report"]:
                with self.subTest(ancestor=ancestor, stage=stage), tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
                    root = Path(folder)
                    report, _ = self.fixture(root, stack)
                    self.successful_pipeline(stack)
                    original_report = runner.report

                    def mark(root=root, ancestor=ancestor):
                        path = root / f"reports/{ancestor}/failure.json"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text('{"status":"FAILED"}')

                    def scoring(*_args, mark=mark, **_kwargs):
                        mark()
                        return metric_fixture()

                    def reporting(metrics, original_report=original_report, mark=mark):
                        original_report(metrics)
                        mark()

                    method, action = ("evaluate", scoring) if stage == "scoring" else ("report", reporting)
                    with patch.object(runner, method, side_effect=action), self.assertRaises(ValueError):
                        runner.run()
                    self.assert_failure(report)
                    self.assertTrue((report / "unpublished_scored_metrics.json").exists())

    def test_final_required_pin_hash_is_bracketed_by_fresh_marker_absence_checks(self):
        for ancestor in SUCCESSFUL_OR_CURRENT:
            for dangling in [False, True]:
                with self.subTest(ancestor=ancestor, dangling=dangling), tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
                    root = Path(folder)
                    report, _ = self.fixture(root, stack)
                    self.successful_pipeline(stack)
                    original_bound, original_digest = runner.require_bound_upstream, runner.inference.digest
                    state = {"in_final": False, "mutated": False}

                    def final_bound(pins, state=state, original_bound=original_bound):
                        state["in_final"] = True
                        return original_bound(pins)

                    def late_digest(path, original_digest=original_digest, state=state,
                                    root=root, ancestor=ancestor, dangling=dangling):
                        value = original_digest(path)
                        if state["in_final"] and not state["mutated"]:
                            marker = root / f"reports/{ancestor}/failure.json"
                            marker.parent.mkdir(parents=True, exist_ok=True)
                            if dangling:
                                marker.symlink_to(root / "nonexistent_failure_target")
                            else:
                                marker.write_text('{"late_failure":true}')
                            state["mutated"] = True
                        return value

                    with patch.object(runner, "require_bound_upstream", side_effect=final_bound), \
                            patch.object(runner.inference, "digest", side_effect=late_digest), self.assertRaises(ValueError):
                        runner.run()
                    self.assertTrue(state["mutated"])
                    self.assert_failure(report)

    def test_missing_required_bound_pin_cannot_be_omitted_from_manifest(self):
        for name in REQUIRED_PINS:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
                report, inputs = self.fixture(Path(folder), stack)
                self.successful_pipeline(stack)
                with patch.object(runner, "input_paths", return_value=inputs - {name}), self.assertRaises(ValueError):
                    runner.run()
                self.assert_failure(report)

    def test_late_current_failure_symlink_never_overwrites_existing_sentinel(self):
        """Additional pre-empirical coverage; no claim this preceded run()."""
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder)
            report, _ = self.fixture(root, stack)
            self.successful_pipeline(stack)
            sentinel = root / "existing_sentinel.json"
            sentinel.write_bytes(b"preserve exact existing sentinel bytes\n")
            original = sentinel.read_bytes()
            original_digest, original_bound = runner.inference.digest, runner.require_bound_upstream
            state = {"in_final": False, "mutated": False}

            def bound(pins):
                state["in_final"] = True
                return original_bound(pins)

            def late_digest(path):
                value = original_digest(path)
                if state["in_final"] and not state["mutated"]:
                    (report / "failure.json").symlink_to(sentinel)
                    state["mutated"] = True
                return value

            with patch.object(runner, "require_bound_upstream", side_effect=bound), \
                    patch.object(runner.inference, "digest", side_effect=late_digest), self.assertRaises(ValueError):
                runner.run()
            self.assertTrue(state["mutated"])
            self.assertEqual(sentinel.read_bytes(), original)
            self.assertFalse((report / "failure.json").is_symlink())
            self.assertTrue((report / "failure.json").is_file())
            self.assert_failure(report)

    def test_partial_registration_bytes_are_preserved_and_all_three_failures_reconstructed(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            report, _ = self.fixture(Path(folder), stack)

            def partial(rows):
                (report / "trial_ledger.jsonl").write_text(json.dumps(rows[0]) + "\n{partial")
                raise OSError("partial registration")

            with patch.object(runner, "ledger", side_effect=partial), self.assertRaisesRegex(OSError, "partial registration"):
                runner.run()
            self.assert_failure(report, inherited=False)
            self.assertTrue((report / "interrupted_trial_ledger.jsonl").read_text().endswith("{partial"))

    def test_report_or_partial_evaluated_write_failure_keeps_scored_backup(self):
        for stage in ["report", "evaluated"]:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
                report, _ = self.fixture(Path(folder), stack)
                self.successful_pipeline(stack)
                if stage == "report":
                    stack.enter_context(patch.object(runner, "report", side_effect=OSError("report failed")))
                else:
                    original_ledger = runner.ledger

                    def partial(rows, report=report, original_ledger=original_ledger):
                        if rows[0]["event"] == "evaluated":
                            with (report / "trial_ledger.jsonl").open("a") as stream:
                                stream.write(json.dumps(rows[0]) + "\n{partial")
                            raise OSError("evaluated failed")
                        return original_ledger(rows)

                    stack.enter_context(patch.object(runner, "ledger", side_effect=partial))
                with self.assertRaisesRegex(OSError, stage + " failed"):
                    runner.run()
                self.assert_failure(report)
                self.assertTrue((report / "interrupted_trial_ledger.jsonl").exists())
                self.assertTrue((report / "unpublished_scored_metrics.json").exists())

    def test_existing_registration_proof_metrics_private_outputs_or_dangling_markers_never_overwritten(self):
        for existing in ["manifest.json", "trial_ledger.jsonl", "failure.json", "metrics.json",
                         "verification.json", "publication_audit.json", "private", "dangling"]:
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
                report, _ = self.fixture(Path(folder), stack)
                if existing == "private":
                    runner.OUT.mkdir()
                    path = runner.OUT / "forecasts.parquet"
                else:
                    path = report / ("failure.json" if existing == "dangling" else existing)
                if existing == "dangling":
                    path.symlink_to(report / "absent_target")
                else:
                    path.write_bytes(b"unchanged prior attempt")
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    runner.run()
                if existing == "dangling":
                    self.assertTrue(path.is_symlink())
                else:
                    self.assertEqual(path.read_bytes(), b"unchanged prior attempt")

    def test_atomic_nonfinite_ledger_serialization_preserves_exact_existing_bytes(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(runner, "REPORT", Path(folder)):
            runner.ledger([{"event": "registered"}])
            path = Path(folder) / "trial_ledger.jsonl"
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                runner.ledger([{"p_conservative": float("nan")}])
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(list(Path(folder).glob(".trial-ledger-*")))

    def test_required_old_failed_family_remains_byte_exact_and_is_not_globally_blocked(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder)
            report, _ = self.fixture(root, stack)
            self.successful_pipeline(stack)
            original = {name: (root / name).read_bytes() for name in REQUIRED_PINS}
            runner.run()
            self.assertFalse((report / "failure.json").exists())
            for name, value in original.items():
                self.assertEqual((root / name).read_bytes(), value)

    def test_scoring_follows_complete_production_and_uses_identical_declared_inputs(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            self.fixture(Path(folder), stack)
            self.successful_pipeline(stack)
            original_produce = runner.produce
            order = []

            def produce(*args):
                order.append("produced")
                return original_produce(*args)

            def score(_panel, _calendar, _protocol, monthly_schedules, application_n, prior):
                self.assertEqual(order, ["produced"])
                self.assertEqual(len(prior), 137)
                self.assertEqual(monthly_schedules, 1)
                self.assertEqual(application_n, 2)
                order.append("scored")
                return metric_fixture()

            with patch.object(runner, "produce", side_effect=produce), patch.object(runner, "evaluate", side_effect=score):
                runner.run()
            self.assertEqual(order, ["produced", "scored"])

    def test_incomplete_or_duplicate_contrast_results_cannot_publish_a_partial_family(self):
        for mode in ["missing", "duplicate", "foreign"]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
                report, _ = self.fixture(Path(folder), stack)
                self.successful_pipeline(stack)
                metrics = metric_fixture()
                if mode == "missing":
                    metrics["rows"] = metrics["rows"][:-1]
                elif mode == "duplicate":
                    metrics["rows"][1] = copy.deepcopy(metrics["rows"][0])
                else:
                    metrics["rows"][0]["control"] = "unregistered_control"
                with patch.object(runner, "evaluate", return_value=metrics), self.assertRaises(ValueError):
                    runner.run()
                self.assert_failure(report)

    def test_pre_run_evidence_symlink_cannot_overwrite_existing_sentinel(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder)
            report, _ = self.fixture(root, stack)
            sentinel = root / "pretest_sentinel.txt"
            sentinel.write_bytes(b"existing pretest sentinel remains exact\n")
            original = sentinel.read_bytes()
            destination = report / "pre_run_checks.txt"

            def checked(*_args, **_kwargs):
                destination.symlink_to(sentinel)
                return SimpleNamespace(returncode=0, stdout="generated checks", stderr="")

            with patch.object(runner.subprocess, "run", side_effect=checked), \
                    self.assertRaisesRegex(ValueError, "generated admission stop"):
                runner.run()
            self.assertEqual(sentinel.read_bytes(), original)
            self.assertFalse(destination.is_symlink())
            self.assertEqual(destination.read_text(), "generated checks")
            self.assert_failure(report)

    def test_success_and_failure_report_symlinks_cannot_overwrite_existing_sentinel(self):
        for mode in ["success", "failure"]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
                root = Path(folder)
                report, _ = self.fixture(root, stack)
                actual_report = runner.report
                self.successful_pipeline(stack)
                sentinel = root / "report_sentinel.txt"
                sentinel.write_bytes(b"existing report sentinel remains exact\n")
                original = sentinel.read_bytes()
                destination = report / "results.md"
                stack.enter_context(patch.object(runner, "report", side_effect=actual_report))

                def scoring(*_args, destination=destination, sentinel=sentinel, **_kwargs):
                    destination.symlink_to(sentinel)
                    result = metric_fixture()
                    for row in result["rows"]:
                        for phase in row["phases"]:
                            phase["gain_relative"] = .01
                    return result

                def failing(*_args, destination=destination, sentinel=sentinel, **_kwargs):
                    destination.symlink_to(sentinel)
                    raise ValueError("generated prose failure")

                if mode == "success":
                    with patch.object(runner, "evaluate", side_effect=scoring):
                        runner.run()
                else:
                    with patch.object(runner, "produce", side_effect=failing), \
                            self.assertRaisesRegex(ValueError, "generated prose failure"):
                        runner.run()
                self.assertEqual(sentinel.read_bytes(), original)
                self.assertFalse(destination.is_symlink())
                self.assertTrue(destination.is_file())
                if mode == "failure":
                    self.assert_failure(report)
                else:
                    self.assertIn("Trailing peak age", destination.read_text())

    def test_all_five_parquet_output_symlinks_preserve_existing_sentinels(self):
        names = ["features", "targets", "feature_states", "forecasts", "application_states"]
        for name in names:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
                root = Path(folder)
                report, _ = self.fixture(root, stack)
                self.successful_pipeline(stack)
                original_produce = runner.produce
                sentinel = root / "parquet_sentinel.txt"
                sentinel.write_bytes(b"existing parquet sentinel remains exact\n")
                original = sentinel.read_bytes()
                destination = runner.OUT / (name + ".parquet")

                def produce(*args, destination=destination, sentinel=sentinel, original_produce=original_produce):
                    destination.symlink_to(sentinel)
                    return original_produce(*args)

                with patch.object(runner, "produce", side_effect=produce):
                    runner.run()
                self.assertEqual(sentinel.read_bytes(), original)
                self.assertFalse(destination.is_symlink())
                self.assertGreater(len(pd.read_parquet(destination)), 0)
                self.assertEqual(len(records(report)), 143)
                self.assertFalse((report / "failure.json").exists())

    def test_interrupted_ledger_backup_symlink_preserves_existing_sentinel_and_recovery(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder)
            report, _ = self.fixture(root, stack)
            self.successful_pipeline(stack)
            sentinel = root / "ledger_backup_sentinel.txt"
            sentinel.write_bytes(b"existing ledger-backup sentinel remains exact\n")
            original = sentinel.read_bytes()
            destination = report / "interrupted_trial_ledger.jsonl"

            def failing(*_args, **_kwargs):
                destination.symlink_to(sentinel)
                raise ValueError("generated backup failure")

            with patch.object(runner, "produce", side_effect=failing), \
                    self.assertRaisesRegex(ValueError, "generated backup failure"):
                runner.run()
            self.assertEqual(sentinel.read_bytes(), original)
            self.assertFalse(destination.is_symlink())
            self.assertTrue(destination.is_file())
            interrupted = [json.loads(line) for line in destination.read_text().splitlines()]
            self.assertEqual([row["event"] for row in interrupted], ["registered"] * 3 + ["inherited"] * 137)
            self.assert_failure(report)


if __name__ == "__main__":
    unittest.main()
