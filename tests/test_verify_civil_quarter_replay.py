"""Prewritten replay dependency, failed-attempt and publication contracts."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import verify_civil_quarter_replay as verify


def closure_fixture(root):
    sources = {
        "cpi": "metadata/cpi.json",
        "nfp": "metadata/nfp.json",
        "fomc": "metadata/fomc.csv",
        "fomc_coverage": "metadata/fomc.json",
    }
    protocol = {
        "sources": sources,
        "calendar": {
            "source_publication_start": "2010-01-01",
            "source_publication_end": "2025-10-20",
        },
    }
    bodies = {
        "outside/cpi.txt": b"original CPI text",
        "outside/nfp.txt": b"original payroll text",
        "outside/fomc.txt": b"original annual statement",
        verify.CAPTURE_MANIFEST: b'{"documentary_only":true}',
    }
    for name, body in bodies.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    (root / "metadata").mkdir()
    cpi = {
        "records": [
            {
                "source_url": "https://www.bls.gov/news.release/archives/cpi_01092020.htm",
                "parse_status": "VERIFIED_EXPLICIT_PLAN",
                "snapshot_path": "outside/cpi.txt",
                "snapshot_sha256": verify.digest(root / "outside/cpi.txt"),
            },
            {
                "source_url": "https://www.bls.gov/news.release/archives/cpi_02132020.htm",
                "parse_status": "ORIGINAL_VINTAGE_UNCERTAIN",
                "snapshot_path": "outside/cpi.txt",
                "snapshot_sha256": verify.digest(root / "outside/cpi.txt"),
            },
            {
                "source_url": "https://www.bls.gov/news.release/archives/cpi_10242025.htm",
                "parse_status": "OUTSIDE_PUBLICATION_FENCE",
                "snapshot_path": "../must-not-read",
                "snapshot_sha256": "bad",
            },
        ]
    }
    nfp = {
        "records": [
            {
                "source_url": "https://www.bls.gov/news.release/archives/empsit_01102020.htm",
                "parse_status": "EXPLICIT_ORIGINAL_PLAN",
                "snapshot_path": "outside/nfp.txt",
                "snapshot_sha256": verify.digest(root / "outside/nfp.txt"),
            }
        ],
        "artifact_sha256": {
            verify.CAPTURE_MANIFEST: verify.digest(root / verify.CAPTURE_MANIFEST)
        },
    }
    (root / sources["cpi"]).write_text(json.dumps(cpi))
    (root / sources["nfp"]).write_text(json.dumps(nfp))
    signature = verify.digest(root / "outside/fomc.txt")
    header = (
        "annual_schedule_year,source_url,source_extraction_path,source_extraction_sha256\n"
    )
    rows = "".join(
        f"2020,https://www.federalreserve.gov/example,outside/fomc.txt,{signature}\n"
        for _ in range(8)
    )
    (root / sources["fomc"]).write_text(header + rows)
    (root / sources["fomc_coverage"]).write_text(
        json.dumps(
            [
                {
                    "annual_schedule_year": 2020,
                    "coverage_start_date": "2020-01-01",
                    "coverage_end_date": "2020-12-31",
                    "plan_count": 8,
                    "source_url": "https://www.federalreserve.gov/example",
                    "source_extraction_sha256": signature,
                }
            ]
        )
    )
    pins = {name: verify.digest(root / name) for name in sources.values()}
    return protocol, pins


def failed_fixture(root):
    import yaml

    report = root / "reports/civil_quarter"
    report.mkdir(parents=True)
    (root / "data/civil_quarter").mkdir(parents=True)
    oldp = copy.deepcopy(verify.frozen.CONTRACT)
    (root / "civil_quarter.yaml").write_text(yaml.safe_dump(oldp))
    protocol_hash = verify.digest(root / "civil_quarter.yaml")
    pins = {}
    for index, name in enumerate(oldp["comparisons"]["inherited_sources"]):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = (
            [
                {
                    "candidate": str(i),
                    "control": "baseline",
                    "horizon": 1,
                    "p_conservative": 1.0,
                }
                for i in range(121)
            ]
            if index == 0
            else []
        )
        path.write_text(json.dumps({"rows": rows}))
        pins[name] = verify.digest(path)
    manifest = {
        "protocol_sha256": protocol_hash,
        "code": {},
        "inputs": dict(pins),
        "preserved": {},
    }
    (report / "manifest.json").write_text(json.dumps(manifest))
    freeze = {"protocol_sha256": protocol_hash, "code": {}, "prefit_design": {}}
    (report / "freeze_record.json").write_text(json.dumps(freeze))
    prior = verify.frozen.inherited_rows(root, oldp, pins=pins)
    failure = {
        "status": "UNEVALUABLE",
        "whole_wave_aborted": True,
        "leads": [],
        "hypothesis_count": 2,
        "cumulative_hypothesis_count": 123,
        "protocol_sha256": protocol_hash,
        "rows": [
            {
                "study": "civil_quarter",
                "candidate": a,
                "control": b,
                "score": "proper_variance",
                "horizon": 1,
                "p_conservative": 1.0,
                "p_holm_wave": 1.0,
                "p_holm_cumulative": 1.0,
                "phases": [],
                "status": "INVALID_RUN",
                "error": "Missing staged source",
            }
            for a, b in verify.COMPARISONS
        ],
    }
    for name in ("metrics.json", "failure.json"):
        (report / name).write_text(json.dumps(failure))
    (report / "results.md").write_text("UNEVALUABLE")
    ledger = (
        [{"event": "inherited", **row} for row in prior]
        + [
            {
                "event": "registered",
                "study": "civil_quarter",
                "candidate": a,
                "control": b,
                "score": "proper_variance",
                "horizon": 1,
                "protocol_sha256": protocol_hash,
            }
            for a, b in verify.COMPARISONS
        ]
        + [{"event": "unevaluable", **row} for row in failure["rows"]]
    )
    (report / "trial_ledger.jsonl").write_text(
        "\n".join(json.dumps(row) for row in ledger) + "\n"
    )
    terminal = {
        name: verify.digest(report / name)
        for name in (
            "metrics.json",
            "failure.json",
            "trial_ledger.jsonl",
            "results.md",
            "manifest.json",
            "freeze_record.json",
        )
    }
    audit = {
        "status": "REPRODUCED_ADMISSION_FAILURE",
        "protocol_sha256": protocol_hash,
        "entrypoint_read_only": True,
        "producer_restarted": False,
        "verification_guard_called": False,
        "frozen_specification_changed": False,
        "exception": {"type": "FileNotFoundError"},
        "new_data_outputs": [],
        "new_forecasts": 0,
        "new_monthly_fits": 0,
        "terminal_publication": {
            "terminal_files_sha256_before": terminal,
            "terminal_files_sha256_after": terminal,
        },
    }
    (report / "admission_failure_audit.json").write_text(json.dumps(audit))
    publication = {
        "status": "UNEVALUABLE_REPORT_AUDITED",
        "protocol_sha256": protocol_hash,
        "manifest_sha256": verify.digest(report / "manifest.json"),
        "independent_admission_failure_audit_sha256": verify.digest(
            report / "admission_failure_audit.json"
        ),
        "new_forecasts": 0,
        "new_monthly_fits": 0,
        "ledger_events": 125,
        "ledger_event_counts": {"inherited": 121, "registered": 2, "unevaluable": 2},
        "leads": [],
        "current_wave_new_hypotheses": 2,
        "cumulative_hypotheses": 123,
        "prefit_code_and_design_pins_unchanged": True,
        "frozen_manifest_entries_checked": {
            g: len(manifest[g]) for g in ("code", "inputs", "preserved")
        },
        "report_artifact_hashes": {},
        "count_source_verification_hashes": {},
    }
    (report / "publication_audit.json").write_text(json.dumps(publication))
    info = {
        "protocol": "civil_quarter.yaml",
        "reports": "reports/civil_quarter",
        "data": "data/civil_quarter",
        "protocol_sha256": protocol_hash,
        "manifest_sha256": verify.digest(report / "manifest.json"),
        "freeze_record_sha256": verify.digest(report / "freeze_record.json"),
        "failure_sha256": verify.digest(report / "failure.json"),
        "admission_failure_audit_sha256": verify.digest(
            report / "admission_failure_audit.json"
        ),
        "publication_audit_sha256": verify.digest(report / "publication_audit.json"),
        "trial_ledger_sha256": verify.digest(report / "trial_ledger.jsonl"),
        "required_status": "UNEVALUABLE",
        "inherited_hypotheses": 121,
        "registered_hypotheses": 2,
        "cumulative_hypotheses": 123,
        "terminal_event": "unevaluable",
        "new_forecasts": 0,
        "new_monthly_fits": 0,
    }
    pins.update(
        {
            str(path.relative_to(root)): verify.digest(path)
            for path in report.rglob("*")
            if path.is_file()
        }
    )
    pins["civil_quarter.yaml"] = protocol_hash
    return info, pins


class ClosureContracts(unittest.TestCase):
    def test_independent_canonical_inventory_matches_separate_synthetic_producer(self):
        from src import civil_quarter_sources as producer

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p, pins = closure_fixture(root)
            self.assertEqual(
                verify.collect_sources(root, p, pins), producer.collect_sources(root, p, pins)
            )

    def test_symlink_escape_rejected_and_confined_symlink_remains_hash_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "root"
            root.mkdir()
            closure_fixture(root)
            (root / "inside_alias").symlink_to(root / "outside/cpi.txt")
            self.assertEqual(
                verify.snapshot_sources(
                    root, {"inside_alias": verify.digest(root / "outside/cpi.txt")}
                )["inside_alias"],
                b"original CPI text",
            )
            (base / "external.txt").write_text("outside root")
            (root / "escape").symlink_to(base / "external.txt")
            with self.assertRaises(ValueError):
                verify.snapshot_sources(root, {"escape": verify.digest(base / "external.txt")})

    def test_active_runtime_and_documentary_closure_is_complete_and_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p, pins = closure_fixture(root)
            before = {
                str(path.relative_to(root)): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
            }
            result = verify.collect_sources(root, p, pins)
            self.assertEqual(set(result["files"]), set(before))
            self.assertEqual(
                result["counts"],
                {
                    "metadata_files": 4,
                    "runtime_files": 3,
                    "documentary_files": 1,
                    "unique_files": 8,
                    "bounded_bls_records": 3,
                    "fomc_reference_rows": 8,
                    "excluded_bls_records": 1,
                },
            )
            self.assertEqual(len(result["references"]), 16)
            self.assertEqual(result["files"], dict(sorted(result["files"].items())))
            self.assertEqual(
                before,
                {
                    str(path.relative_to(root)): path.read_bytes()
                    for path in root.rglob("*")
                    if path.is_file()
                },
            )

    def test_missing_or_changed_indirect_source_and_metadata_pin_reject(self):
        for fault in ("metadata", "missing", "content", "registered_conflict"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                p, pins = closure_fixture(root)
                if fault == "metadata":
                    (root / p["sources"]["cpi"]).write_text("{}")
                elif fault == "missing":
                    (root / "outside/cpi.txt").unlink()
                elif fault == "content":
                    (root / "outside/cpi.txt").write_text("altered")
                else:
                    pins["outside/cpi.txt"] = "0" * 64
                with self.assertRaises((AssertionError, ValueError, FileNotFoundError)):
                    verify.collect_sources(root, p, pins)

    def test_unsafe_paths_malformed_hashes_duplicates_and_conflicting_references_reject(self):
        for fault in (
            "traversal",
            "absolute",
            "backslash",
            "uppercase_hash",
            "duplicate_json",
            "conflict",
        ):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                p, pins = closure_fixture(root)
                name = p["sources"]["cpi"]
                data = json.loads((root / name).read_bytes())
                if fault in ("traversal", "absolute", "backslash"):
                    data["records"][0]["snapshot_path"] = {
                        "traversal": "outside/../outside/cpi.txt",
                        "absolute": str(root / "outside/cpi.txt"),
                        "backslash": "outside\\cpi.txt",
                    }[fault]
                elif fault == "uppercase_hash":
                    data["records"][0]["snapshot_sha256"] = "A" * 64
                elif fault == "conflict":
                    data["records"][1]["snapshot_sha256"] = "0" * 64
                if fault == "duplicate_json":
                    (root / name).write_text('{"records":[],"records":[]}')
                else:
                    (root / name).write_text(json.dumps(data))
                pins[name] = verify.digest(root / name)
                with self.assertRaises((AssertionError, ValueError)):
                    verify.collect_sources(root, p, pins)

    def test_fomc_coverage_and_final_documentary_binding_cannot_be_substituted(self):
        for fault in ("coverage_count", "coverage_year", "capture_hash"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                p, pins = closure_fixture(root)
                name = p["sources"]["nfp" if fault == "capture_hash" else "fomc_coverage"]
                value = json.loads((root / name).read_bytes())
                if fault == "capture_hash":
                    value["artifact_sha256"][verify.CAPTURE_MANIFEST] = "0" * 64
                elif fault == "coverage_count":
                    value[0]["plan_count"] = 7
                else:
                    value[0]["annual_schedule_year"] = 2021
                (root / name).write_text(json.dumps(value))
                pins[name] = verify.digest(root / name)
                with self.assertRaises((AssertionError, ValueError)):
                    verify.collect_sources(root, p, pins)

    def test_snapshot_bytes_are_hash_checked_before_decode_and_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p, pins = closure_fixture(root)
            closure = verify.collect_sources(root, p, pins)
            payloads = verify.snapshot_sources(root, closure["files"])
            self.assertEqual(set(payloads), set(closure["files"]))
            (root / "outside/cpi.txt").write_text("changed after collection")
            with self.assertRaises(AssertionError):
                verify.snapshot_sources(root, closure["files"])


class FailureContracts(unittest.TestCase):
    def test_failed_prior_proof_rejects_terminal_tamper_and_any_new_old_output(self):
        for fault in (None, "metrics", "ledger", "audit", "output"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                info, pins = failed_fixture(root)
                if fault == "output":
                    (root / "data/civil_quarter/forecasts.parquet").write_text(
                        "forbidden late output"
                    )
                elif fault:
                    name = {
                        "metrics": "metrics.json",
                        "ledger": "trial_ledger.jsonl",
                        "audit": "admission_failure_audit.json",
                    }[fault]
                    with (root / "reports/civil_quarter" / name).open("a") as stream:
                        stream.write("altered")
                before = {
                    str(path.relative_to(root)): path.read_bytes()
                    for path in root.rglob("*")
                    if path.is_file()
                }
                if fault:
                    with self.assertRaises((AssertionError, ValueError)):
                        verify.verify_failed_attempt(root, info, pins)
                else:
                    result = verify.verify_failed_attempt(root, info, pins)
                    self.assertEqual(result["status"], "FAILED_ATTEMPT_PRESERVED")
                    self.assertEqual(
                        result["ledger_events_verified"],
                        {"inherited": 121, "registered": 2, "unevaluable": 2},
                    )
                    self.assertEqual(result["new_forecasts"], 0)
                self.assertEqual(
                    before,
                    {
                        str(path.relative_to(root)): path.read_bytes()
                        for path in root.rglob("*")
                        if path.is_file()
                    },
                )

    def test_new_failure_retains_both_hypotheses_and_preserves_original_attempt(self):
        for text in ('{"rows":[]}', "{bad", '{"bad":NaN}', '{"bad":Infinity}'):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                new = root / "reports/civil_quarter_replay"
                old = root / "reports/civil_quarter"
                new.mkdir(parents=True)
                old.mkdir(parents=True)
                (new / "metrics.json").write_text(text)
                (old / "failure.json").write_text("original failure")
                with (
                    patch.object(verify, "verify", side_effect=AssertionError("injected")),
                    self.assertRaises(AssertionError),
                ):
                    verify.verify_with_failure_guard(root)
                result = json.loads((new / "metrics.json").read_bytes())
                self.assertEqual(
                    (result["hypothesis_count"], result["cumulative_hypothesis_count"]),
                    (2, 125),
                )
                self.assertEqual(result["leads"], [])
                self.assertEqual(len(result["rows"]), 2)
                for row in result["rows"]:
                    self.assertEqual(
                        [
                            row[k]
                            for k in ("p_conservative", "p_holm_wave", "p_holm_cumulative")
                        ],
                        [1, 1, 1],
                    )
                self.assertEqual((old / "failure.json").read_text(), "original failure")
                self.assertEqual(len((new / "trial_ledger.jsonl").read_text().splitlines()), 2)


class EntryContracts(unittest.TestCase):
    def test_old_calendar_complete_proof_stages_transitive_sources_and_preserves_schema(self):
        import yaml

        from src.verify_macro_overnight import scan_source_admissions
        from tests.test_civil_quarter_sources import fixture
        from tests.test_verify_civil_quarter import original_fixture

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            oldp, metadata_pins = fixture(root)
            oldp.update(comparisons={"inherited_sources": []}, evidence_class="synthetic")
            closure = verify.collect_sources(root, oldp, metadata_pins)
            pins = dict(closure["files"])
            report, out = root / "reports/calendar_variance", root / "data/calendar_variance"
            report.mkdir(parents=True)
            out.mkdir(parents=True)

            def write(name, value):
                path = root / name
                path.write_text(json.dumps(value))
                pins[name] = verify.digest(path)

            (root / "calendar_variance.yaml").write_text(yaml.safe_dump(oldp))
            phash = verify.digest(root / "calendar_variance.yaml")
            pins["calendar_variance.yaml"] = phash
            write("reports/calendar_variance/manifest.json", {"inputs": metadata_pins})
            info = {
                "protocol": "calendar_variance.yaml",
                "protocol_sha256": phash,
                "reports": "reports/calendar_variance",
                "data": "data/calendar_variance",
                "manifest_sha256": pins["reports/calendar_variance/manifest.json"],
                "verifier_sha256": "a" * 64,
            }
            features = original_fixture(8)
            dates = features.index
            maturity = pd.Series(dates, index=dates).shift(-1)
            targets = pd.DataFrame(
                {"y": 0.001, "target_end": maturity, "available_date": maturity}, index=dates
            )
            targets.loc[dates[-1], "y"] = np.nan
            panel = pd.DataFrame({"origin": [dates[2]]})
            for name, table in (
                ("features", features),
                ("targets", targets),
                ("forecasts", panel),
            ):
                path = out / (name + ".parquet")
                table.to_parquet(path)
                pins[str(path.relative_to(root))] = verify.digest(path)
            source_audit = {
                "plans": {"synthetic": True},
                "plan_availability": ["synthetic"],
                "market": {"source_path": str(root / "NEVER_READ_MARKET.parquet")},
            }
            write("data/calendar_variance/source_audit.json", source_audit)
            write("data/calendar_variance/fits.json", [])
            prior = [{"candidate": f"old_{i}", "p_conservative": 1.0} for i in range(106)]
            rows = [
                {"candidate": c, "control": b, "score": "qlike", "horizon": 1}
                for c, b in verify.calendar_old.COMPARISONS
            ]
            metrics = {
                "protocol_sha256": phash,
                "evidence_class": "synthetic",
                "inherited_rows": prior,
                "rows": rows,
            }
            write("reports/calendar_variance/metrics.json", metrics)
            ledger = [{"event": "inherited", **row} for row in prior]
            ledger += [
                {"event": "registered", **row, "protocol_sha256": phash} for row in rows
            ]
            ledger += [{"event": "evaluated", **row} for row in rows]
            ledger_path = report / "trial_ledger.jsonl"
            ledger_path.write_text("".join(json.dumps(row) + "\n" for row in ledger))
            pins[str(ledger_path.relative_to(root))] = verify.digest(ledger_path)
            before = {
                str(path.relative_to(root)): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
            }

            def reconstruction(stage, protocol):
                self.assertNotEqual(stage, root)
                scanned = scan_source_admissions(stage, protocol)
                self.assertEqual(scanned["bounded_source_sections_scanned"], 2)
                self.assertEqual(scanned["mismatches"], [])
                for name in closure["files"]:
                    self.assertEqual((stage / name).read_bytes(), before[name])
                self.assertFalse((stage / "NEVER_READ_MARKET.parquet").exists())
                audit = copy.deepcopy(source_audit)
                audit["market"]["source_path"] = str(stage / "NEVER_READ_MARKET.parquet")
                return features, targets, audit

            forecast_proof = {"forecasts_verified": 1, "monthly_fits_verified": 0}
            inference_proof = {"new_hypotheses_verified": 2, "leads": []}
            with (
                patch.object(verify.calendar_old, "validate_protocol"),
                patch.object(verify.calendar_old, "reconstruct", side_effect=reconstruction),
                patch.object(
                    verify.calendar_old, "verify_forecasts", return_value=forecast_proof
                ),
                patch.object(
                    verify.calendar_old, "verify_metrics", return_value=inference_proof
                ),
            ):
                proof = verify.reconstruct_calendar(root, info, pins, closure)
            self.assertEqual(
                set(proof),
                {
                    "status",
                    "protocol_sha256",
                    "verifier_sha256",
                    "raw_feature_rows_verified",
                    "raw_feature_columns_verified",
                    "source_availability_rows_verified",
                    "sources",
                    "forecast_reconstruction",
                    "inference",
                    "ledger_events_verified",
                    "limitations",
                },
            )
            self.assertEqual(proof["status"], "VERIFIED")
            self.assertEqual(proof["forecast_reconstruction"], forecast_proof)
            self.assertEqual(proof["inference"], inference_proof)
            self.assertEqual(
                proof["ledger_events_verified"],
                {"inherited": 106, "registered": 2, "evaluated": 2},
            )
            self.assertEqual(
                before,
                {
                    str(path.relative_to(root)): path.read_bytes()
                    for path in root.rglob("*")
                    if path.is_file()
                },
            )

    def test_whole_entrypoint_reads_declared_artifacts_and_checks_final_protocol(self):
        import copy

        import yaml

        from tests.test_verify_civil_quarter import original_fixture

        for fault in (None, "support", "protocol"):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                report, out = (
                    root / "reports/civil_quarter_replay",
                    root / "data/civil_quarter_replay",
                )
                report.mkdir(parents=True)
                out.mkdir(parents=True)
                protocol = copy.deepcopy(verify.CONTRACT)
                (root / "civil_quarter_replay.yaml").write_text(yaml.safe_dump(protocol))
                phash = verify.digest(root / "civil_quarter_replay.yaml")
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
                    "proofs": {"failed_attempt": {"status": "FAILED_ATTEMPT_PRESERVED"}},
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
                        with (root / "civil_quarter_replay.yaml").open("a") as stream:
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
                        return_value={"inherited": 123, "registered": 2, "evaluated": 2},
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
