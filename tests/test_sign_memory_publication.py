"""Prewritten immutable source admission and whole-family failure accounting."""

import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src import sign_memory_search as study


class SignMemoryPublication(unittest.TestCase):
    def test_protocol_changed_during_pretests_prevents_registration(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            p = root / "sign_memory.yaml"
            p.write_text("wave: 13\n")

            def checks(*args, **kwargs):
                p.write_text("wave: 14\n")
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

    def test_staged_source_decodes_exact_admitted_bytes_and_original_identity(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(study, "ROOT", Path(tmp)):
            root = Path(tmp)
            name = "data/raw/synthetic.bin"
            source = root / name
            source.parent.mkdir(parents=True)
            source.write_bytes(b"first admitted payload")
            p = {"sources": {"daily": name}}
            signature = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest = {"inputs": {name: signature}}

            def loader(config, staged):
                source.write_bytes(b"changed real path after snapshot")
                self.assertEqual((staged / name).read_bytes(), b"first admitted payload")
                return (
                    1,
                    2,
                    3,
                    {
                        "sources": {
                            "daily": {
                                "source_path": str(staged / name),
                                "source_sha256": signature,
                            }
                        }
                    },
                )

            with patch.object(study.sf, "load_sources", side_effect=loader):
                *_, audit = study.load_pinned_sources(p, manifest)
            self.assertEqual(audit["sources"]["daily"]["source_path"], str(source))
            with self.assertRaisesRegex(ValueError, "Pinned raw source changed"):
                study.load_pinned_sources(p, manifest)

    def exercise(self, fault):
        prior = [
            {
                "study": "synthetic",
                "candidate": str(i),
                "control": "baseline",
                "horizon": 1,
                "p_conservative": 1.0,
            }
            for i in range(117)
        ]
        scored = {
            "rows": [
                {
                    "study": "sign_memory",
                    "candidate": a,
                    "control": b,
                    "score": s,
                    "horizon": 1,
                    "p_conservative": 1e-5,
                    "phases": [],
                }
                for a, b, s in study.CONTRASTS
            ],
            "leads": ["memory"],
            "hypothesis_count": 2,
            "cumulative_hypothesis_count": 119,
            "new_monthly_fits": 1,
            "new_forecasts": 3,
        }
        if fault == "nonfinite":
            scored["rows"][0]["p_conservative"] = float("nan")
        original_ledger = study.ledger
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            report = root / "report"
            output = root / "data"
            p = root / "new.yaml"
            p.write_text(json.dumps({"sources": {"daily": "raw.bin"}, "index": {}}))
            (root / "raw.bin").write_bytes(b"raw source unchanged")
            old = root / "prior.txt"
            old.write_bytes(b"prior artifact unchanged")

            def admit(_):
                if fault == "admission":
                    raise ValueError("admission failure")
                if fault == "source":
                    (root / "raw.bin").write_bytes(b"tampered after admission")
                return {"status": "UPSTREAM_VERIFIED_READ_ONLY"}

            def render(_):
                (report / "results.md").write_text("provisional")
                if fault == "report":
                    raise OSError("report failure")

            def score(*_):
                if fault == "protocol":
                    p.write_text(p.read_text() + "\n")
                return scored

            def record(rows):
                if rows and rows[0]["event"] == fault:
                    original_ledger(rows[:1])
                    raise OSError("partial " + fault)
                original_ledger(rows)

            def loader(config, staged):
                return (
                    None,
                    None,
                    None,
                    {"sources": {"daily": {"source_path": str(staged / "raw.bin")}}},
                )

            def save(frame, path, *args, **kwargs):
                Path(path).write_bytes(b"synthetic parquet")

            for k, v in {
                "ROOT": root,
                "REPORT": report,
                "OUT": output,
                "PROTOCOL": p,
                "validate": lambda _: None,
                "input_paths": lambda _: {"raw.bin", "prior.txt"},
                "inherited": lambda _: prior,
                "validate_upstream": admit,
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
            stack.enter_context(patch.object(study.sf, "load_sources", side_effect=loader))
            stack.enter_context(
                patch.object(study.sf, "measurement_audit", return_value={"status": "PASS"})
            )
            stack.enter_context(
                patch.object(study.sf, "require_measurement", return_value=None)
            )
            stack.enter_context(
                patch.object(
                    study.sf,
                    "build_features",
                    return_value=(pd.DataFrame({"x": [1]}), pd.DataFrame({"y": [1]})),
                )
            )
            fitter = stack.enter_context(
                patch.object(
                    study.sm, "forecast_panel", return_value=(pd.DataFrame({"x": [1]}), [{}])
                )
            )
            stack.enter_context(patch.object(pd.DataFrame, "to_parquet", new=save))
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
                ("UNEVALUABLE", [], 2, 119),
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
            self.assertEqual(sum(r["event"] == "inherited" for r in events), 117)
            self.assertEqual([r["event"] for r in events[-2:]], ["unevaluable"] * 2)
            self.assertEqual(old.read_bytes(), b"prior artifact unchanged")
            if fault in ["admission", "source", "registered"]:
                fitter.assert_not_called()
                self.assertFalse((output / "forecasts.parquet").exists())
            else:
                backup = json.loads((report / "unpublished_scored_metrics.json").read_text())
                self.assertTrue(backup["not_for_inherited_inference_or_promotion"])
                if fault == "nonfinite":
                    self.assertIn("nan", backup["nonserializable_metrics_repr"])

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
