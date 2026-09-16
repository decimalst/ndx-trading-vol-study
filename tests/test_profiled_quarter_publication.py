"""Prewritten immutable inputs and whole-family publication failures."""

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src import profiled_quarter_search as study


class ProfiledQuarterPublication(unittest.TestCase):
    def test_protocol_changed_during_pretests_prevents_registration(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            p = root / "profiled_quarter.yaml"
            p.write_text("wave: 17\n")

            def checks(*args, **kwargs):
                p.write_text("wave: 18\n")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            for k, v in {
                "ROOT": root,
                "REPORT": root / "report",
                "OUT": root / "data",
                "PROTOCOL": p,
                "validate": lambda p: None,
            }.items():
                stack.enter_context(patch.object(study, k, v))
            stack.enter_context(patch.object(study.subprocess, "run", side_effect=checks))
            with (
                patch.object(study, "input_paths") as inputs,
                self.assertRaisesRegex(ValueError, "protocol changed during pretests"),
            ):
                study.run()
            inputs.assert_not_called()
            self.assertFalse((root / "report/manifest.json").exists())

    def test_input_bytes_checked_before_decode_and_one_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(study, "ROOT", Path(tmp)):
            root = Path(tmp)
            payload = io.BytesIO()
            pd.DataFrame({"x": [1.0]}).to_parquet(payload)
            (root / "one.parquet").write_bytes(payload.getvalue())
            p = {"feature_source": dict.fromkeys(["features", "targets"], "one.parquet")}
            signature = hashlib.sha256(payload.getvalue()).hexdigest()
            original = pd.read_parquet

            def decode(snapshot, *args, **kwargs):
                self.assertIsInstance(snapshot, io.BytesIO)
                (root / "one.parquet").write_bytes(b"changed after snapshot")
                return original(snapshot, *args, **kwargs)

            with patch.object(study.pd, "read_parquet", side_effect=decode):
                frames = study.load_pinned_inputs(p, {"inputs": {"one.parquet": signature}})
            self.assertEqual([f.iloc[0, 0] for f in frames], [1.0, 1.0])
            with (
                patch.object(study.pd, "read_parquet") as decoder,
                self.assertRaisesRegex(ValueError, "Pinned upstream input changed"),
            ):
                study.load_pinned_inputs(p, {"inputs": {"one.parquet": signature}})
            decoder.assert_not_called()

    def test_inherited_json_pin_is_checked_before_decode(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(study, "ROOT", Path(tmp)):
            root = Path(tmp)
            source = root / "old.json"
            rows = [
                {"candidate": str(i), "horizon": 1, "p_conservative": 0.5} for i in range(125)
            ]
            payload = json.dumps({"rows": rows}).encode()
            source.write_bytes(payload)
            config = {"comparisons": {"inherited_sources": ["old.json"]}}
            signature = hashlib.sha256(payload).hexdigest()
            with (
                patch.object(study.json, "loads") as decode,
                self.assertRaisesRegex(ValueError, "Pinned inherited metrics changed"),
            ):
                study.inherited(config, {"old.json": "wrong"})
            decode.assert_not_called()
            actual_loads = json.loads

            def replace_after_snapshot(blob):
                source.write_bytes(b"replaced after snapshot")
                return actual_loads(blob)

            with patch.object(study.json, "loads", side_effect=replace_after_snapshot):
                result = study.inherited(config, {"old.json": signature})
            self.assertEqual(len(result), 125)
            self.assertTrue(all(row["source_sha256"] == signature for row in result))

    def test_registration_includes_failed_inventory_and_omitted_sources_without_collection(
        self,
    ):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(study, "ROOT", Path(folder)),
        ):
            root = Path(folder)
            p = {
                "comparisons": {"inherited_sources": ["old_metrics.json"]},
                "prospectus": {"path": "prospectus.md"},
                "verification": {"profiled_solver": {"design_path": "solver_design.md"}},
                "source_closure": {"omitted_dependencies": {"outside/doc.txt": "bound"}},
            }
            for key in ["upstream", "feature_source", "failed_attempt", "failed_verification"]:
                reports = root / key / "reports"
                data = root / key / "data"
                reports.mkdir(parents=True)
                data.mkdir()
                payload = json.dumps({"inputs": {key + "/raw.dat": "bound"}}).encode()
                (reports / "manifest.json").write_bytes(payload)
                (reports / "terminal.json").write_text("{}")
                (data / "artifact.json").write_text("{}")
                p[key] = {
                    "reports": key + "/reports",
                    "data": key + "/data",
                    "protocol": key + ".yaml",
                    "manifest_sha256": hashlib.sha256(payload).hexdigest(),
                }
            with patch.object(study.source, "collect_sources") as collector:
                paths = study.input_paths(p)
            collector.assert_not_called()
            self.assertIn("outside/doc.txt", paths)
            self.assertIn("solver_design.md", paths)
            self.assertIn("failed_attempt/raw.dat", paths)
            self.assertIn("failed_attempt/reports/terminal.json", paths)
            self.assertIn("failed_attempt/data/artifact.json", paths)
            self.assertIn("failed_attempt.yaml", paths)
            self.assertIn("failed_verification/raw.dat", paths)
            self.assertIn("failed_verification/reports/terminal.json", paths)
            self.assertIn("failed_verification/data/artifact.json", paths)

    def test_closure_protocol_and_extra_pins_checked_before_metadata_collection(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(study, "ROOT", Path(folder)),
        ):
            root = Path(folder)
            payload = b"sources: {}\n"
            document = b"bound source"
            (root / "old.yaml").write_bytes(payload)
            (root / "extra.txt").write_bytes(document)
            signature = hashlib.sha256(payload).hexdigest()
            doc_hash = hashlib.sha256(document).hexdigest()
            p = {
                "feature_source": {"protocol": "old.yaml", "protocol_sha256": signature},
                "source_closure": {"omitted_dependencies": {"extra.txt": doc_hash}},
            }
            pins = {"inputs": {"old.yaml": signature, "extra.txt": doc_hash}}
            collected = {"files": {"extra.txt": doc_hash}}
            with patch.object(
                study.source, "collect_sources", return_value=collected
            ) as collector:
                self.assertEqual(study.prepare_source_closure(p, pins), collected)
            collector.assert_called_once()
            (root / "old.yaml").write_bytes(b"changed: metadata")
            with (
                patch.object(study.source, "collect_sources") as collector,
                patch.object(study.yaml, "safe_load") as decoder,
                self.assertRaises(ValueError),
            ):
                study.prepare_source_closure(p, pins)
            collector.assert_not_called()
            decoder.assert_not_called()
            (root / "old.yaml").write_bytes(payload)
            (root / "extra.txt").write_bytes(b"changed source")
            with (
                patch.object(study.source, "collect_sources") as collector,
                self.assertRaises(ValueError),
            ):
                study.prepare_source_closure(p, pins)
            collector.assert_not_called()

    def exercise(self, fault):
        prior = [
            {
                "study": "synthetic",
                "candidate": str(i),
                "control": "baseline",
                "horizon": 1,
                "p_conservative": 1.0,
            }
            for i in range(125)
        ]
        scored = {
            "rows": [
                {
                    "study": "profiled_quarter",
                    "candidate": a,
                    "control": b,
                    "score": s,
                    "horizon": 1,
                    "p_conservative": 1e-5,
                    "phases": [],
                }
                for a, b, s in study.CONTRASTS
            ],
            "leads": ["quarter"],
            "hypothesis_count": 2,
            "cumulative_hypothesis_count": 127,
            "new_monthly_fits": 0,
            "new_forecasts": 2,
            "combined_forecasts": 3,
            "preserved_forecasts": 1,
            "common_scored_origins": 1,
        }
        if fault == "nonfinite":
            scored["rows"][0]["p_conservative"] = float("nan")
        original_ledger = study.ledger
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            report = root / "report"
            output = root / "data"
            p = root / "new.yaml"
            p.write_text(json.dumps({"feature_source": {"features": "raw.bin"}, "index": {}}))
            (root / "raw.bin").write_bytes(b"upstream unchanged")
            (root / "prior.txt").write_bytes(b"prior unchanged")

            def admit(_):
                if fault == "admission":
                    raise ValueError("admission failure")
                if fault == "source":
                    (root / "raw.bin").write_bytes(b"changed")
                return {"status": "UPSTREAM_VERIFIED_READ_ONLY"}

            def load(config, manifest):
                for name, expected in manifest["inputs"].items():
                    if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                        raise ValueError("Pinned upstream input changed")
                frame = pd.DataFrame({"x": [1.0]})
                return frame, frame

            def render(_):
                (report / "results.md").write_text("provisional")
                if fault == "report":
                    raise OSError("report failure")

            def score(*_, **__):
                if fault == "protocol":
                    p.write_text(p.read_text() + "\n")
                return scored

            def record(rows):
                if rows and rows[0]["event"] == fault:
                    original_ledger(rows[:1])
                    raise OSError("partial " + fault)
                original_ledger(rows)

            for k, v in {
                "ROOT": root,
                "REPORT": report,
                "OUT": output,
                "PROTOCOL": p,
                "validate": lambda _: None,
                "input_paths": lambda _: {"raw.bin", "prior.txt"},
                "inherited": lambda *args: prior,
                "validate_upstream": admit,
                "prepare_source_closure": lambda *args: (
                    (_ for _ in ()).throw(ValueError("closure failure"))
                    if fault == "closure"
                    else {"files": {}}
                ),
                "load_pinned_inputs": load,
                "evaluate": score,
                "report": render,
                "ledger": record,
                "version": lambda _: "synthetic",
            }.items():
                stack.enter_context(patch.object(study, k, v))
            stack.enter_context(
                patch.object(
                    study.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
                )
            )
            stack.enter_context(patch.object(study.np, "show_config", return_value=None))
            stack.enter_context(
                patch.object(study.cf, "augment_features", side_effect=lambda frame: frame)
            )
            stack.enter_context(
                patch.object(
                    study.cm,
                    "preflight",
                    side_effect=ValueError("INSUFFICIENT_DATA: civil support")
                    if fault == "support"
                    else None,
                    return_value={"status": "PASS"},
                )
            )
            builder = stack.enter_context(
                patch.object(
                    study.cm,
                    "forecast_panel",
                    return_value=(pd.DataFrame({"x": [1]}), [{"application_n": 1}]),
                )
            )
            stack.enter_context(
                patch.object(
                    pd.DataFrame,
                    "to_parquet",
                    new=lambda f, path: Path(path).write_bytes(b"synthetic"),
                )
            )
            with self.assertRaises((ValueError, OSError)):
                study.run()
            m = json.loads((report / "metrics.json").read_text())
            self.assertEqual(m, json.loads((report / "failure.json").read_text()))
            self.assertEqual(
                (
                    m["status"],
                    m["leads"],
                    m["hypothesis_count"],
                    m["cumulative_hypothesis_count"],
                ),
                ("UNEVALUABLE", [], 2, 127),
            )
            self.assertTrue(
                all(
                    r["p_conservative"] == r["p_holm_wave"] == r["p_holm_cumulative"] == 1
                    for r in m["rows"]
                )
            )
            events = [
                json.loads(s) for s in (report / "trial_ledger.jsonl").read_text().splitlines()
            ]
            self.assertEqual(sum(r["event"] == "inherited" for r in events), 125)
            self.assertEqual([r["event"] for r in events[-2:]], ["unevaluable"] * 2)
            self.assertEqual((root / "prior.txt").read_bytes(), b"prior unchanged")
            if fault in ["admission", "source", "registered", "support", "closure"]:
                builder.assert_not_called()
                self.assertFalse((output / "forecasts.parquet").exists())
            else:
                backup = json.loads((report / "unpublished_scored_metrics.json").read_text())
                self.assertTrue(backup["not_for_inherited_inference_or_promotion"])
                if fault == "nonfinite":
                    self.assertIn("nan", backup["nonserializable_metrics_repr"])

    def test_failed_source_closure_prevents_all_optimization(self):
        self.exercise("closure")

    def test_support_failure_before_any_optimization(self):
        self.exercise("support")

    def test_failed_admission(self):
        self.exercise("admission")

    def test_source_changed_after_admission(self):
        self.exercise("source")

    def test_partial_registration(self):
        self.exercise("registered")

    def test_partial_evaluation(self):
        self.exercise("evaluated")

    def test_failed_report(self):
        self.exercise("report")

    def test_protocol_mutation(self):
        self.exercise("protocol")

    def test_nonfinite_metrics_keep_terminal_ledger(self):
        self.exercise("nonfinite")


if __name__ == "__main__":
    unittest.main()
