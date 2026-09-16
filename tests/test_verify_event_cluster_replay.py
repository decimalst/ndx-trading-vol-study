"""Prewritten wave21 entry and inference contracts; generated inputs only."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from src import verify_event_cluster as frozen
from src import verify_event_cluster_replay as v


def inference_fixture():
    from tests.test_verify_event_cluster import inference_fixture as original

    panel, p, metrics, prior = original()
    p.update(study_id="event_cluster_replay_wave21", wave=21)
    p["comparisons"].update(inherited_hypotheses=134, cumulative_hypotheses=137)
    p["inference"]["wave_alpha"] = 0.05 / (21 * 22)
    prior += [
        {
            "study": "event_cluster", "candidate": a, "control": b,
            "horizon": 1, "score": "brier", "p_conservative": 1.0,
            "source": "reports/event_cluster/metrics.json", "source_sha256": "b" * 64,
            "source_row_index": i,
        }
        for i, (a, b) in enumerate(v.COMPARISONS)
    ]
    for name in ("new_monthly_fits", "new_nuisance_fits", "new_cluster_fits",
                 "original_baseline_fits", "new_forecasts", "reused_control_forecasts",
                 "combined_forecasts"):
        metrics.pop(name)
    metrics.update(
        newly_generated_forecasts=0, new_producer_fits=0,
        retained_candidate_scored_forecasts=len(panel) // 2,
        original_control_forecasts=len(panel) // 2, combined_retained_forecasts=len(panel),
        retained_monthly_schedules=21, independent_stage_fits_verified=42,
        cumulative_hypothesis_count=137, inherited_rows=prior,
    )
    probabilities = [row["p_conservative"] for row in metrics["rows"]]
    cumulative = frozen.holm([r["p_conservative"] for r in prior] + probabilities)[-3:]
    for row, adjusted in zip(metrics["rows"], cumulative, strict=True):
        row["study"] = "event_cluster_replay"
        row["p_holm_cumulative"] = float(adjusted)
        passed = (
            all(s["delta"] <= -0.0005 for s in row["phases"])
            and all(s["delta"] < 0 for s in row["phases"][1]["stability"])
            and row["p_holm_wave"] < 0.05 / 462 and adjusted < 0.05
        )
        row["verdict"] = "PASSES_ALL_GATES" if passed else "DOES_NOT_QUALIFY"
    metrics["leads"] = ["cluster"] if all(
        r["verdict"] == "PASSES_ALL_GATES" for r in metrics["rows"]
    ) else []
    return panel, p, metrics, prior


def aligned_source_fixture(root):
    import pandas as pd

    from tests.test_event_cluster_replay_admission import Fixture

    f = Fixture(root)
    # The shared fixture is metadata-only; give this entry fixture the exact
    # declared generated sizes before forming its final documentary hashes.
    for name, count in (("forecasts", 4), ("states", 2), ("memory", 3)):
        f.frame("data/event_cluster/" + name + ".parquet", pd.DataFrame({"synthetic": range(count)}))
    f.seal()
    return f


class InferenceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel, cls.protocol, cls.metrics, cls.prior = inference_fixture()

    def verify(self, metrics=None):
        with patch.object(v, "inherited_rows", return_value=self.prior):
            return v.verify_metrics(
                None, self.panel, self.protocol, self.metrics if metrics is None else metrics,
                420, 21, admitted_inputs={},
            )

    def test_literal_prospective_protocol_and_frozen_inference(self):
        p = yaml.safe_load(Path("event_cluster_replay.yaml").read_bytes())
        v.validate_protocol(p)
        self.assertIs(v.phase_statistics, frozen.phase_statistics)
        self.assertEqual(v.WAVE_ALPHA, 0.05 / (21 * 22))
        self.assertEqual(p["inference"]["bootstrap_draws"], 399999)
        p["wave"] = True
        with self.assertRaises((ValueError, AssertionError)):
            v.validate_protocol(p)

    def test_all_six_phases_eighteen_bootstraps_and137_family(self):
        proof = self.verify()
        self.assertEqual(proof["new_hypotheses_verified"], 3)
        self.assertEqual(proof["cumulative_hypotheses_verified"], 137)
        self.assertEqual(proof["phase_comparisons_verified"], 6)
        self.assertEqual(proof["bootstrap_runs_verified"], 18)
        self.assertEqual(proof["calibration_rows_verified"], 8)

    def test_counts_support_coeffless_identity_uncertainty_mutations_reject(self):
        for field in ("producer", "boolcount", "retained", "family", "prior", "interval", "lead"):
            m = copy.deepcopy(self.metrics)
            if field == "producer":
                m["new_producer_fits"] = 1
            elif field == "boolcount":
                m["newly_generated_forecasts"] = False
            elif field == "retained":
                m["retained_candidate_scored_forecasts"] -= 1
            elif field == "family":
                m["cumulative_hypothesis_count"] = 134
            elif field == "prior":
                m["inherited_rows"][-1]["p_conservative"] = 0.1
            elif field == "interval":
                m["rows"][0]["phases"][0]["ci95_envelope"][0] += 0.001
            else:
                m["leads"] = ["cluster"] if not m["leads"] else []
            with self.subTest(field=field), self.assertRaises((ValueError, AssertionError)):
                self.verify(m)

    def test_stricter_wave21_gate_is_used_not_wave20(self):
        m = copy.deepcopy(self.metrics)
        for row in m["rows"]:
            for phase in row["phases"]:
                phase["p_conservative"] = 0.000038
                phase["delta"] = -0.001
                for s in phase["stability"]:
                    s["delta"] = -0.001
            row.update(p_conservative=0.000038, p_holm_wave=0.000114,
                       p_holm_cumulative=0.000038 * 137, verdict="DOES_NOT_QUALIFY")
        m["leads"] = []
        lookup = {row["control"]: row["phases"] for row in m["rows"]}
        with patch.object(v, "phase_statistics", side_effect=lambda _, c, __, i, ___: lookup[c][i]):
            self.verify(m)
            bad = copy.deepcopy(m)
            bad["leads"] = ["cluster"]
            for row in bad["rows"]:
                row["verdict"] = "PASSES_ALL_GATES"
            with self.assertRaises((ValueError, AssertionError)):
                self.verify(bad)

    def test_exact140_event_ledger_and_hash_before_decode(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            folder = root / v.REPORT
            folder.mkdir(parents=True)
            registered = [
                {"event": "registered", "study": "event_cluster_replay", "candidate": a,
                 "control": b, "score": "brier", "horizon": 1,
                 "protocol_sha256": self.metrics["protocol_sha256"]}
                for a, b in v.COMPARISONS
            ]
            rows = registered + [{"event": "inherited", **r} for r in self.prior]
            rows += [{"event": "evaluated", **r} for r in self.metrics["rows"]]
            path = folder / "trial_ledger.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            signature = v.digest(path)
            result = v.verify_ledger(root, self.metrics, self.prior, "evaluated", signature=signature)
            self.assertEqual(result, {"inherited": 134, "registered": 3, "evaluated": 3})
            path.write_text(path.read_text() + "{}\n")
            with self.assertRaises((ValueError, AssertionError)):
                v.verify_ledger(root, self.metrics, self.prior, "evaluated", signature=signature)


class AdmissionAndFailureContracts(unittest.TestCase):
    def test_real_generated_failed_source_closure_and_checked_retained_buffers(self):
        from src import event_cluster_replay_admission as source

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            f = aligned_source_fixture(root)
            pins = source.collect_input_pins(root, f.expected)
            expected, original, retained = source.admit_upstream(root, f.expected, pins)
            proof, loaded, saved = v.admit_upstream(root, f.expected, pins)
            self.assertEqual(proof, expected)
            self.assertEqual(set(saved), {"forecasts", "fits", "states", "memory", "support", "upstream_admission"})
            self.assertFalse(proof["retained_outputs_numerically_verified"])
            self.assertFalse(proof["retained_outputs_transformed"])
            self.assertEqual(loaded["protocol"], original["protocol"])
            for name in ("forecasts", "states", "memory"):
                self.assertTrue(saved[name].equals(retained[name]))
            missing = dict(pins)
            missing.pop(next(iter(f.expected)))
            with self.assertRaises((ValueError, AssertionError)):
                v.admit_upstream(root, f.expected, missing)

    def test_changed_required_failure_or_late_original_marker_rejects_admission(self):
        from src import event_cluster_replay_admission as source

        for mutation in ("required_failure", "range_alert", "issued_calibration"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                f = aligned_source_fixture(root)
                pins = source.collect_input_pins(root, f.expected)
                admit = source.admit_upstream
                def inject(*args, admit=admit, mutation=mutation, root=root, **kwargs):
                    answer = admit(*args, **kwargs)
                    name = "event_cluster" if mutation == "required_failure" else mutation
                    (root / f"reports/{name}/failure.json").write_text("{}")
                    return answer
                with (patch.object(source, "admit_upstream", side_effect=inject),
                      self.assertRaises((ValueError, AssertionError))):
                    v.admit_upstream(root, f.expected, pins)

    def test_inherited_failed_three_are_canonical_p1_and_hashed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            old = {"rows": [{"candidate": str(i), "horizon": 1, "p_conservative": 1.0}
                            for i in range(131)]}
            failed = {"status": "UNEVALUABLE", "whole_wave_aborted": True, "leads": [],
                      "hypothesis_count": 3, "cumulative_hypothesis_count": 134,
                      "rows": [{"study": "event_cluster", "candidate": a, "control": b,
                                "horizon": 1, "score": "brier", "p_conservative": 1.0,
                                "p_holm_wave": 1.0, "p_holm_cumulative": 1.0,
                                "verdict": "UNEVALUABLE"} for a, b in v.COMPARISONS]}
            names = ["prior.json", "reports/event_cluster/metrics.json"]
            for name, data in zip(names, [old, failed], strict=True):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data))
            pins = {name: v.digest(root / name) for name in names}
            p = {"comparisons": {"inherited_sources": names}}
            rows = v.inherited_rows(root, p, pins=pins)
            self.assertEqual(len(rows), 134)
            self.assertEqual([r["source_row_index"] for r in rows[-3:]], [0, 1, 2])
            failed["rows"][1]["p_conservative"] = 0.1
            (root / names[-1]).write_text(json.dumps(failed))
            with self.assertRaises((ValueError, AssertionError)):
                v.inherited_rows(root, p, pins=pins)
            pins[names[-1]] = v.digest(root / names[-1])
            with self.assertRaises((ValueError, AssertionError)):
                v.inherited_rows(root, p, pins=pins)

    def test_snapshot_hash_checked_before_json_decode(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "bad.json").write_bytes(b"{malformed")
            with self.assertRaisesRegex((ValueError, AssertionError), "changed|hash|pin"):
                v.read_json_snapshot(root, "bad.json", "0" * 64)

    def test_failed20_marker_is_retained_but_original18_19_markers_block(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ("event_cluster", "range_alert", "issued_calibration"):
                (root / "reports" / name).mkdir(parents=True)
            old = root / "reports/event_cluster/failure.json"
            old.write_text('{"status":"UNEVALUABLE"}')
            v.require_upstream_success(root)
            for study in ("range_alert", "issued_calibration"):
                marker = root / "reports" / study / "failure.json"
                marker.write_text("{}")
                with self.assertRaises((ValueError, AssertionError)):
                    v.require_upstream_success(root)
                marker.unlink()
            self.assertEqual(old.read_text(), '{"status":"UNEVALUABLE"}')

    def test_guard_preserves_old_bytes_and_invalidates_three_even_nonfinite_json(self):
        for original in (b'{"protocol_sha256":NaN}', b'{"nested":{"x":Infinity}}', b'{broken',
                         b'{"protocol_sha256":"synthetic","leads":["cluster"]}'):
            with self.subTest(original=original), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                report = root / v.REPORT
                report.mkdir(parents=True)
                old = root / "reports/event_cluster/failure.json"
                old.parent.mkdir(parents=True)
                old.write_bytes(b"old failure immutable")
                (report / "metrics.json").write_bytes(original)
                (report / "trial_ledger.jsonl").write_bytes(b"broken unterminated record")
                with (patch.object(v, "verify", side_effect=ValueError("synthetic failure")),
                      self.assertRaisesRegex(ValueError, "synthetic failure")):
                    v.verify_with_failure_guard(root)
                result = json.loads((report / "metrics.json").read_bytes())
                self.assertEqual(result["status"], "UNEVALUABLE")
                self.assertEqual(result["cumulative_hypothesis_count"], 137)
                self.assertEqual(result["leads"], [])
                self.assertEqual(len(result["rows"]), 3)
                self.assertTrue(all(r[key] == 1 for r in result["rows"]
                                    for key in ("p_conservative", "p_holm_wave", "p_holm_cumulative")))
                self.assertEqual(json.loads((report / "verification.json").read_text())["status"], "FAILED")
                lines = (report / "trial_ledger.jsonl").read_text().splitlines()
                self.assertEqual(lines[0], "broken unterminated record")
                self.assertEqual([json.loads(s)["event"] for s in lines[1:]], ["verification_failed"] * 3)
                self.assertEqual(old.read_bytes(), b"old failure immutable")
                if original.startswith(b'{"protocol_sha256":"synthetic"'):
                    self.assertEqual(json.loads((report / "unpublished_scored_metrics.json").read_text())["status"],
                                     "UNPUBLISHED_DIAGNOSTIC_ONLY")
                else:
                    self.assertEqual((report / "unpublished_invalid_metrics.txt").read_bytes(), original)


def reconstruction_fixture():
    from src import verify_event_cluster as old

    nuisance = {
        "method": "hybr", "start": [0.0] * 6, "xtol": 1e-10, "maxfev": 2000,
        "factor": 1, "success": True, "status": 1, "message": "synthetic converged",
        "function_evaluations": 12, "jacobian_evaluations": 2,
        "coefficients": [0.01] * 6, "objective": 0.5,
        "gradient": [1e-9] * 6, "gradient_max_abs": 1e-9,
    }
    scalar = {
        "coefficient": 0.01, "status": "FITTED", "alpha": 0.01, "objective": 0.4,
        "gradient": 1e-10, "gradient_max_abs": 1e-10, "bracket": [-50.0, 50.0],
        "bracket_gradients": [-1.0, 1.0], "gradient_at_zero": -0.001,
        "iterations": 38, "function_calls": 40, "success": True,
        "method": "bisect", "xtol": 1e-12, "rtol": 1e-14, "maxiter": 200,
    }
    fields = ["origin", "feature_cutoff_date", "source_fit_origin",
              *[c for c in old.DATE_COLUMNS if c != "feature_cutoff_date"]]
    return {
        "forecasts_verified": 4, "retained_candidate_scored_forecasts": 2,
        "original_control_forecasts": 2, "newly_generated_forecasts": 0,
        "new_producer_fits": 0, "retained_monthly_schedules": 1,
        "independent_stage_fits_verified": 2, "original_monthly_fits_replayed": 1,
        "common_scored_origins": 1, "common_application_origins": 2,
        "application_states_verified": 2, "memory_rows_verified": 250,
        "monthly_stage_audits": [{"fit_origin": "2010-01-04", "audit": {
            "train_n": 220, "application_n": 2,
            "nuisance_recomputed_gradient_max_abs": 1e-9,
            "cluster_recomputed_gradient_max_abs": 1e-10,
            "independent_nuisance": nuisance, "independent_cluster": scalar,
        }}],
        "date_comparison_audit": {
            "policy": "lossless_naive_midnight_state_dates_v1", "comparison_unit": "ns",
            "allowed_units": ["ms", "us", "ns"], "columns": fields, "rows": 2,
            "index_policy": "frozen_exact", "other_columns_policy": "frozen_exact",
            "fields": [{"column": c, "actual_unit": "us", "expected_unit": "ms",
                        "nat_rows": 0, "lossless": True, "equal_instants": True} for c in fields],
            "input_frames_unchanged": True,
        },
    }


class ReconstructionAuditContracts(unittest.TestCase):
    def test_solver_settings_and_zero_starts_are_exact_not_numeric_tolerances(self):
        expected = reconstruction_fixture()
        for mutation in ("start", "ridge", "root_tolerance"):
            saved = copy.deepcopy(expected)
            audit = saved["monthly_stage_audits"][0]["audit"]
            if mutation == "start":
                audit["independent_nuisance"]["start"][0] = 1e-13
            elif mutation == "ridge":
                audit["independent_cluster"]["alpha"] += 1e-13
            else:
                audit["independent_nuisance"]["xtol"] += 1e-13
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, AssertionError)):
                v.compare_reconstruction_audit(saved, expected)

    def test_solver_diagnostics_are_admissible_not_byte_identity(self):
        expected = reconstruction_fixture()
        saved = copy.deepcopy(expected)
        item = saved["monthly_stage_audits"][0]["audit"]["independent_nuisance"]
        item.update(function_evaluations=14, jacobian_evaluations=3, message="other convergence wording")
        v.compare_reconstruction_audit(saved, expected)
        self.assertEqual(expected["monthly_stage_audits"][0]["audit"]["independent_nuisance"]["function_evaluations"], 12)

    def test_coefficients_gradients_domains_dates_and_count_types_reject(self):
        expected = reconstruction_fixture()
        for mutation in ("coefficient", "objective", "negative_gradient", "nan", "calls",
                         "count", "date", "schema", "nonconverged"):
            saved = copy.deepcopy(expected)
            a = saved["monthly_stage_audits"][0]["audit"]["independent_nuisance"]
            if mutation == "coefficient":
                a["coefficients"][0] += 0.01
            elif mutation == "objective":
                a["objective"] += 0.01
            elif mutation == "negative_gradient":
                a["gradient_max_abs"] = -1e-16
            elif mutation == "nan":
                a["gradient"][0] = float("nan")
            elif mutation == "calls":
                a["function_evaluations"] = True
            elif mutation == "count":
                saved["new_producer_fits"] = False
            elif mutation == "date":
                saved["date_comparison_audit"]["fields"][0]["lossless"] = False
            elif mutation == "schema":
                a["extra"] = 1
            else:
                a["success"] = False
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, AssertionError)):
                v.compare_reconstruction_audit(saved, expected)


class CompleteEntryContracts(unittest.TestCase):
    def tree(self, root):
        from pandas import DataFrame

        report, out = root / v.REPORT, root / v.OUT
        report.mkdir(parents=True)
        out.mkdir(parents=True)
        (root / "src").mkdir()
        (root / "tests").mkdir()
        (root / "src/synthetic.py").write_text("# generated entry fixture\n")
        for study in ("range_alert", "issued_calibration", "event_cluster"):
            (root / "reports" / study).mkdir(parents=True)
        (root / "event_cluster.yaml").write_text("synthetic original protocol")
        old_signature = v.digest(root / "event_cluster.yaml")
        old_failure = {"status": "UNEVALUABLE", "whole_wave_aborted": True, "leads": [],
                       "protocol_sha256": old_signature, "rows": [{
                           "study": "event_cluster", "candidate": a, "control": b,
                           "horizon": 1, "score": "brier", "verdict": "UNEVALUABLE",
                           "p_conservative": 1.0, "p_holm_wave": 1.0, "p_holm_cumulative": 1.0,
                       } for a, b in v.COMPARISONS]}
        for name in ("failure.json", "metrics.json"):
            (root / "reports/event_cluster" / name).write_text(json.dumps(old_failure))
        (root / "reports/event_cluster/verification.json").write_text(json.dumps({
            "status": "FAILED", "protocol_sha256": old_signature,
            "error": "Independent verification failed: AssertionError: every saved application state "
                     "differs from independent reconstruction",
        }))
        closure = {"status": "PINNED_FAILED_WAVE20_WITH_VERIFIED_WAVE19_WAVE18",
                   "anchors": {"event_cluster.yaml": old_signature},
                   "retained_output_hashes": {},
                   "files": {name: v.digest(root / name) for name in (
                       "reports/event_cluster/failure.json", "reports/event_cluster/metrics.json",
                       "reports/event_cluster/verification.json", "event_cluster.yaml")}}
        p = copy.deepcopy(v.CONTRACT)
        p["upstream"]["anchors"] = {"event_cluster.yaml": v.digest(root / "event_cluster.yaml")}
        p["comparisons"]["inherited_sources"] = []
        (root / "event_cluster_replay.yaml").write_text(yaml.safe_dump(p))
        signature = v.digest(root / "event_cluster_replay.yaml")
        (report / "full_repository_tests.txt").write_text("generated prefit log")
        (report / "ENTRY_DESIGN.md").write_text("generated design")
        code = {"src/synthetic.py": v.digest(root / "src/synthetic.py")}
        designs = {v.REPORT + "/ENTRY_DESIGN.md": v.digest(report / "ENTRY_DESIGN.md")}
        freeze = {"protocol_sha256": signature, "code": code, "prefit_design": designs,
                  "checks": {"full_log_sha256": v.digest(report / "full_repository_tests.txt")}}
        (report / "freeze_record.json").write_text(json.dumps(freeze))
        inputs = {**closure["files"], **designs,
                  v.REPORT + "/freeze_record.json": v.digest(report / "freeze_record.json")}
        manifest = {"protocol_sha256": signature, "code": code, "inputs": inputs, "preserved": {}}
        (report / "manifest.json").write_text(json.dumps(manifest))
        checked = reconstruction_fixture()
        (out / "upstream_admission.json").write_text(json.dumps(closure))
        (out / "reconstruction_audit.json").write_text(json.dumps(checked))
        metrics = {"protocol_sha256": signature, "evidence_class": p["evidence_class"],
                   "rows": [], "inherited_rows": []}
        (report / "metrics.json").write_text(json.dumps(metrics))
        (report / "trial_ledger.jsonl").write_text('{"event":"generated"}\n')
        loaded = {name: DataFrame() for name in ("features", "targets", "forecasts", "states")}
        loaded.update(fits=[], protocol={"index": p["original_index"]})
        retained = {"forecasts": DataFrame({"synthetic": range(4)}),
                    "states": DataFrame({"synthetic": range(2)}), "memory": DataFrame(),
                    "fits": [{}], "support": {}, "upstream_admission": {}}
        return closure, loaded, retained, checked

    def run_tree(self, root, tree, mutate=None):
        from contextlib import ExitStack

        closure, loaded, retained, checked = tree
        with ExitStack() as stack:
            stack.enter_context(patch.object(v, "validate_protocol"))
            stack.enter_context(patch.object(v, "collect_closure", return_value=closure))
            stack.enter_context(patch.object(v, "admit_upstream", return_value=(closure, loaded, retained)))
            helper = stack.enter_context(patch.object(v, "verify_pipeline", return_value=copy.deepcopy(checked)))
            stack.enter_context(patch.object(v, "verify_metrics", return_value={"synthetic": True}, side_effect=mutate))
            stack.enter_context(patch.object(v, "verify_ledger", return_value={"inherited": 134, "registered": 3, "evaluated": 3}))
            stack.enter_context(patch("src.event_cluster_pipeline.forecast_panel", side_effect=AssertionError("forbidden producer")))
            stack.enter_context(patch("src.event_cluster_models.fit_stages", side_effect=AssertionError("forbidden fit")))
            result = v.verify(root)
            args = helper.call_args.args
            self.assertEqual(len(args), 11)
            self.assertIs(args[6], retained["forecasts"])
            self.assertIs(args[7], retained["fits"])
            self.assertIs(args[8], retained["states"])
            return result

    def test_all_four_snapshots_and_no_producer_or_old_output_writes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            tree = self.tree(root)
            old = (root / "reports/event_cluster/failure.json").read_bytes()
            result = self.run_tree(root, tree)
            self.assertEqual(result["status"], "VERIFIED")
            self.assertEqual(len(result["verified_output_hashes"]), 4)
            self.assertEqual(result["forecast_reconstruction"]["new_producer_fits"], 0)
            self.assertEqual(result["forecast_reconstruction"]["newly_generated_forecasts"], 0)
            self.assertEqual(old, (root / "reports/event_cluster/failure.json").read_bytes())
            for name, signature in result["verified_output_hashes"].items():
                self.assertEqual(v.digest(root / name), signature)
            self.assertEqual({p.name for p in (root / v.OUT).iterdir()},
                             {"upstream_admission.json", "reconstruction_audit.json"})

    def test_late_mutations_or_failure_markers_block_commit(self):
        for mutation in ("output", "source", "newcode", "testslog", "protocol",
                         "range_alert", "issued_calibration", "event_cluster_replay"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                tree = self.tree(root)
                def mutate(*_, mutation=mutation, root=root, **__):
                    paths = {
                        "output": v.OUT + "/reconstruction_audit.json",
                        "source": "event_cluster.yaml", "newcode": "src/late.py",
                        "testslog": v.REPORT + "/full_repository_tests.txt",
                        "protocol": "event_cluster_replay.yaml",
                    }
                    name = paths.get(mutation, f"reports/{mutation}/failure.json")
                    (root / name).write_text("changed")
                    return {"synthetic": True}
                with self.assertRaises((ValueError, AssertionError)):
                    self.run_tree(root, tree, mutate)
                self.assertFalse((root / v.REPORT / "verification.json").exists())

    def test_saved_reconstruction_coefficient_tamper_fails_before_inference(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            tree = self.tree(root)
            bad = copy.deepcopy(tree[-1])
            bad["monthly_stage_audits"][0]["audit"]["independent_cluster"]["coefficient"] += 0.01
            (root / v.OUT / "reconstruction_audit.json").write_text(json.dumps(bad))
            with self.assertRaises((ValueError, AssertionError)):
                self.run_tree(root, tree)


    def test_exact_old_failure_marker_rechecked_after_last_closure_hash_loop(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            tree = self.tree(root)
            original = v.pins_checked
            calls = []
            def mutate_after_loop(root_path, pins):
                answer = original(root_path, pins)
                if pins is tree[0]["files"]:
                    calls.append(True)
                    (root / "reports/event_cluster/failure.json").write_text("changed after last hash")
                return answer
            with (patch.object(v, "pins_checked", side_effect=mutate_after_loop),
                  self.assertRaises((ValueError, AssertionError))):
                self.run_tree(root, tree)
            self.assertEqual(len(calls), 1)
            self.assertFalse((root / v.REPORT / "verification.json").exists())


if __name__ == "__main__":
    unittest.main()
