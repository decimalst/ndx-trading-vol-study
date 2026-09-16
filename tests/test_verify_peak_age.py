"""Prewritten independent wave22 entry/inference contracts, generated data only."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from src import verify_index_hinge as old
from src import verify_peak_age as v
from tests.test_peak_age_admission import Fixture as SourceFixture


def inference_fixture():
    p = yaml.safe_load(Path("peak_age.yaml").read_bytes())
    p["inference"]["bootstrap_draws"] = 19
    calendar = pd.bdate_range("2015-01-01", "2025-10-20")
    dates = calendar[(calendar >= "2016-01-04") & (calendar <= "2025-10-17")]
    dates = dates[(dates <= "2019-12-31") | (dates >= "2020-01-02")]
    positions = calendar.get_indexer(dates)
    dates = dates[(positions + 21) < len(calendar)]
    positions = calendar.get_indexer(dates)
    end = calendar[positions + 21]
    keep = (dates > "2019-12-31") | (end <= "2019-12-31")
    dates, end, positions = dates[keep], end[keep], positions[keep]
    rng = np.random.default_rng(22)
    y = rng.normal(0, 0.02, len(dates))
    rows = []
    for name, scale in [
        ("mean", 0.025),
        ("baseline", 0.02),
        ("depth", 0.015),
        ("peak_age", 0.001),
    ]:
        prediction = y + rng.normal(0, scale, len(dates))
        rows.append(
            pd.DataFrame(
                {
                    "origin": dates,
                    "fit_origin": dates,
                    "feature_cutoff_date": calendar[positions - 1],
                    "target_end": end,
                    "available_date": end,
                    "horizon": 21,
                    "phase": np.where(dates <= "2019-12-31", "development", "evaluation"),
                    "model": name,
                    "prediction": prediction,
                    "y": y,
                    "loss": (y - prediction) ** 2,
                    "train_n": 1000,
                }
            )
        )
    panel = (
        pd.concat(rows, ignore_index=True)
        .sort_values(["origin", "model"])
        .reset_index(drop=True)
    )
    prior = [
        {
            "study": "synthetic",
            "candidate": "c",
            "control": "b",
            "horizon": 1,
            "p_conservative": 1.0,
            "source": "prior.json",
            "source_sha256": "a" * 64,
            "source_row_index": i,
        }
        for i in range(137)
    ]
    result = []
    for candidate, control, h in v.COMPARISONS:
        phases = [
            v.phase_statistics(panel, control, h, phase, code, p, calendar)
            for code, phase in enumerate(["development", "evaluation"])
        ]
        result.append(
            {
                "study": "peak_age",
                "candidate": candidate,
                "control": control,
                "horizon": h,
                "score": "mse",
                "phases": phases,
                "p_conservative": max(x["p_conservative"] for x in phases),
            }
        )
    wave = v.holm([r["p_conservative"] for r in result])
    cumulative = v.holm([1.0] * 137 + [r["p_conservative"] for r in result])[-3:]
    for row, a, b in zip(result, wave, cumulative, strict=True):
        row.update(
            p_holm_wave=float(a), p_holm_cumulative=float(b), verdict="DOES_NOT_QUALIFY"
        )
    n = panel.origin.nunique()
    k = dates.to_period("M").nunique()
    a = n + 7
    metrics = {
        "rows": result,
        "inherited_rows": prior,
        "leads": [],
        "hypothesis_count": 3,
        "cumulative_hypothesis_count": 140,
        "protocol_sha256": "a" * 64,
        "evidence_class": p["evidence_class"],
        "newly_generated_forecasts": 4 * n,
        "common_scored_origins": n,
        "common_application_origins": a,
        "full_application_predictions": 4 * a,
        "monthly_schedules": k,
        "ridge_models_fitted": 2 * k,
        "scalar_models_fitted": k,
        "training_means_computed": k,
    }
    return panel, p, metrics, prior, calendar, a, k


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = inference_fixture()

    def check(self, metrics=None, panel=None):
        original, p, m, prior, calendar, a, k = self.fixture
        with patch.object(v, "inherited_rows", return_value=prior):
            return v.verify_metrics(
                None,
                original if panel is None else panel,
                p,
                m if metrics is None else metrics,
                calendar,
                a,
                k,
                admitted_inputs={},
            )

    def test_all_three_contrasts_six_phases_126_offsets_and140_family(self):
        proof = self.check()
        self.assertEqual(proof["new_hypotheses_verified"], 3)
        self.assertEqual(proof["cumulative_hypotheses_verified"], 140)
        self.assertEqual(proof["phase_comparisons_verified"], 6)
        self.assertEqual(proof["bootstrap_runs_verified"], 18)
        self.assertEqual(proof["nonoverlap_offsets_verified"], 126)
        self.assertEqual(proof["leads"], [])

    def test_literal_contract_all_leaf_mutations_rejected(self):
        original = yaml.safe_load(Path("peak_age.yaml").read_bytes())
        v.validate_protocol(original)

        def leaves(value, path=()):
            if isinstance(value, dict):
                for key, item in value.items():
                    yield from leaves(item, path + (key,))
            elif isinstance(value, list):
                for key, item in enumerate(value):
                    yield from leaves(item, path + (key,))
            else:
                yield path, value

        for path, value in leaves(original):
            p = copy.deepcopy(original)
            part = p
            for key in path[:-1]:
                part = part[key]
            part[path[-1]] = (
                False
                if type(value) is int
                else value + 1
                if type(value) is float
                else str(value) + " changed"
            )
            with self.subTest(path=path), self.assertRaises((ValueError, AssertionError)):
                v.validate_protocol(p)
        self.assertEqual(v.WAVE_ALPHA, 0.05 / (22 * 23))

    def test_frozen_independent_bootstrap_hac_primitives_and_candidate_adaptation(self):
        self.assertIs(v.explicit_bootstrap_means, old.explicit_bootstrap_means)
        self.assertIs(v.independent_hac, old.independent_hac)
        panel, p, _, _, calendar, _, _ = self.fixture
        # Equivalent generated panel with a renamed candidate must reproduce the
        # frozen independent return-inference routine exactly.
        changed = panel.copy()
        changed.loc[changed.model == "peak_age", "model"] = "hinge"
        expected = old.phase_statistics(changed, "depth", 21, "evaluation", 1, p, calendar)
        got = v.phase_statistics(panel, "depth", 21, "evaluation", 1, p, calendar)
        v.same_tree(got, expected, "exact frozen inference arithmetic")

    def test_changed_score_phase_offset_count_or_integer_metadata_rejects(self):
        _, _, original, _, _, _, _ = self.fixture
        for kind in [
            "score",
            "count",
            "boolcount",
            "offset",
            "interval",
            "family",
            "subset",
            "lead",
        ]:
            m = copy.deepcopy(original)
            if kind == "score":
                m["rows"][0]["phases"][0]["delta"] += 0.01
            elif kind == "count":
                m["full_application_predictions"] -= 4
            elif kind == "boolcount":
                m["scalar_models_fitted"] = True
            elif kind == "offset":
                m["rows"][1]["phases"][1]["nonoverlap_phases"].pop()
            elif kind == "interval":
                m["rows"][0]["phases"][0]["ci95_envelope"][0] += 0.1
            elif kind == "family":
                m["cumulative_hypothesis_count"] = 137
            elif kind == "subset":
                m["rows"].pop()
            else:
                m["leads"] = [21]
            with self.subTest(kind=kind), self.assertRaises((ValueError, AssertionError)):
                self.check(m)

    def test_four_arm_target_timing_loss_and_order_tampering_rejected(self):
        original = self.fixture[0]
        for kind in [
            "target",
            "loss",
            "prediction",
            "missing",
            "duplicate",
            "horizon",
            "order",
            "clock",
        ]:
            panel = original.copy()
            if kind == "target":
                panel.loc[0, "y"] += 0.1
            elif kind == "loss":
                panel.loc[0, "loss"] += 0.1
            elif kind == "prediction":
                panel.loc[0, "prediction"] += 0.1
            elif kind == "missing":
                panel = panel.iloc[1:].reset_index(drop=True)
            elif kind == "duplicate":
                panel = pd.concat([panel, panel.iloc[:1]], ignore_index=True)
            elif kind == "horizon":
                panel.loc[0, "horizon"] = 63
            elif kind == "order":
                panel = panel.iloc[::-1].reset_index(drop=True)
            else:
                panel.loc[0, "feature_cutoff_date"] = panel.loc[0, "origin"]
            with self.subTest(kind=kind), self.assertRaises((ValueError, AssertionError)):
                self.check(panel=panel)

    def test_common_clock_and_minimum_training_faults_reject_without_score_change(self):
        original = self.fixture[0]
        chosen = original.loc[original.origin >= "2020-06-01", "origin"].iloc[0]
        for kind in ("feature_clock", "target_clock", "train_minimum"):
            panel = original.copy()
            mask = panel.origin.eq(chosen)
            if kind == "feature_clock":
                panel.loc[mask, "feature_cutoff_date"] -= pd.Timedelta(days=1)
            elif kind == "target_clock":
                for column in ("target_end", "available_date"):
                    panel.loc[mask, column] += pd.Timedelta(days=1)
            else:
                panel.loc[mask, "train_n"] = 999
            with self.subTest(kind=kind), self.assertRaises((ValueError, AssertionError)):
                self.check(panel=panel)

    def test_effect_gate_requires_both_slices_every_offset_and_strict_negative(self):
        phases = copy.deepcopy(self.fixture[2]["rows"][0]["phases"])
        self.assertTrue(v.effect_passes(phases, 21))
        for kind in ["phase", "gain", "slice", "zero", "offset", "empty"]:
            changed = copy.deepcopy(phases)
            if kind == "phase":
                changed.pop()
            elif kind == "gain":
                changed[0]["gain_relative"] = 0.00249
            elif kind == "slice":
                changed[1]["stability"][0]["delta"] = 0.0
            elif kind == "zero":
                changed[0]["delta"] = 0.0
            elif kind == "offset":
                changed[0]["nonoverlap_phases"][0]["delta"] = 0.0
            else:
                changed[0]["nonoverlap_phases"][0]["n"] = 0
            with self.subTest(kind=kind):
                self.assertFalse(v.effect_passes(changed, 21))

    def test_phase_505_boundary_and_full_calendar_offset_anchor(self):
        panel, p, _, _, calendar, _, _ = self.fixture
        dates = panel.loc[panel.phase == "development", "origin"].drop_duplicates().iloc[:505]
        selected = panel[panel.origin.isin(dates)].copy()
        got = v.phase_statistics(selected, "baseline", 21, "development", 0, p, calendar)
        self.assertEqual(got["n"], 505)
        shifted = calendar[1:]
        other = v.phase_statistics(selected, "baseline", 21, "development", 0, p, shifted)
        self.assertNotEqual(got["nonoverlap_phases"], other["nonoverlap_phases"])
        with self.assertRaises((ValueError, AssertionError)):
            v.phase_statistics(
                selected[selected.origin != dates.iloc[0]],
                "baseline",
                21,
                "development",
                0,
                p,
                calendar,
            )

    def test_wave22_strict_significance_and_all_three_controls(self):
        _, _, original, _, _, _, _ = self.fixture
        m = copy.deepcopy(original)
        for row in m["rows"]:
            for phase in row["phases"]:
                phase["p_conservative"] = 0.000034
            row.update(
                p_conservative=0.000034, p_holm_wave=0.000102, p_holm_cumulative=0.00476
            )
        lookup = {r["control"]: r["phases"] for r in m["rows"]}
        with patch.object(
            v,
            "phase_statistics",
            side_effect=lambda _, control, h, phase, code, p, cal: lookup[control][code],
        ):
            self.check(m)
            for row in m["rows"]:
                row["verdict"] = "COMPARISON_GATE_PASS"
            m["leads"] = [21]
            with self.assertRaises((ValueError, AssertionError)):
                self.check(m)


class EnvelopeAndFailureTests(unittest.TestCase):
    def test_generated_source_admission_independently_rehashes_without_old_tables(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            f = SourceFixture(root)
            pins = v.collect_closure(root, f.expected)["files"]
            audit, daily, iv = v.admit_upstream(root, f.expected, pins)
            self.assertEqual(len(daily), 2)
            self.assertEqual(list(iv), ["vix", "vix9d", "vvix"])
            self.assertEqual(audit["files"], pins)
            self.assertFalse(audit["prior_forecast_tables_decoded"])
            missing = dict(pins)
            missing.pop("data/research_paths/spx_daily.parquet")
            with self.assertRaises((ValueError, AssertionError)):
                v.admit_upstream(root, f.expected, missing)

    def test_exact143_ledger_order_and_predecode_hash(self):
        _, _, m, prior, _, _, _ = inference_fixture()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            report = root / v.REPORT
            report.mkdir(parents=True)
            events = [
                {
                    "event": "registered",
                    "study": "peak_age",
                    "candidate": x,
                    "control": c,
                    "horizon": h,
                    "score": "mse",
                    "protocol_sha256": m["protocol_sha256"],
                }
                for x, c, h in v.COMPARISONS
            ]
            events += [{"event": "inherited", **r} for r in prior] + [
                {"event": "evaluated", **r} for r in m["rows"]
            ]
            path = report / "trial_ledger.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in events))
            signature = v.digest(path)
            self.assertEqual(
                v.verify_ledger(root, m, prior, "evaluated", signature=signature),
                {"registered": 3, "inherited": 137, "evaluated": 3},
            )
            events[0], events[3] = events[3], events[0]
            path.write_text("".join(json.dumps(r) + "\n" for r in events))
            with self.assertRaises((ValueError, AssertionError)):
                v.verify_ledger(root, m, prior, "evaluated", signature=signature)
            with self.assertRaises((ValueError, AssertionError)):
                v.verify_ledger(root, m, prior, "evaluated")

    def test_three_failed_prior_families_require_canonical_p_one(self):
        for study, count in [
            ("civil_quarter", 2),
            ("civil_quarter_replay", 2),
            ("event_cluster", 3),
        ]:
            record = {
                "status": "UNEVALUABLE",
                "whole_wave_aborted": True,
                "leads": [],
                "rows": [
                    {"p_conservative": 1.0, "p_holm_wave": 1.0, "p_holm_cumulative": 1.0}
                    for _ in range(count)
                ],
            }
            v.require_failed_metrics(study, record)
            for key in ["p_conservative", "p_holm_wave", "p_holm_cumulative"]:
                bad = copy.deepcopy(record)
                bad["rows"][0][key] = 0.1
                with self.assertRaises((ValueError, AssertionError)):
                    v.require_failed_metrics(study, bad)

    def test_failure_guard_keeps_allthree_pone_and_old_bytes_and_score_diagnostics(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            report = root / v.REPORT
            report.mkdir(parents=True)
            old = root / "reports/event_cluster/failure.json"
            old.parent.mkdir(parents=True)
            old.write_bytes(b"original failed record")
            metrics = {
                "protocol_sha256": "b" * 64,
                "rows": [{"diagnostic": "unverified"}],
                "leads": [21],
            }
            original = (json.dumps(metrics) + "\n").encode()
            (report / "metrics.json").write_bytes(original)
            ledger = b'{"event":"registered"}\n'
            (report / "trial_ledger.jsonl").write_bytes(ledger)
            with (
                patch.object(
                    v, "verify", side_effect=ValueError("injected independent check")
                ),
                self.assertRaises(ValueError),
            ):
                v.verify_with_failure_guard(root)
            failed = json.loads((report / "metrics.json").read_text())
            self.assertEqual(failed["cumulative_hypothesis_count"], 140)
            self.assertEqual(len(failed["rows"]), 3)
            self.assertEqual(failed["leads"], [])
            self.assertTrue(all(r["p_conservative"] == 1 for r in failed["rows"]))
            self.assertEqual(
                (report / "metrics.json").read_bytes(), (report / "failure.json").read_bytes()
            )
            self.assertEqual(old.read_bytes(), b"original failed record")
            self.assertTrue((report / "trial_ledger.jsonl").read_bytes().startswith(ledger))
            self.assertEqual(
                json.loads((report / "unpublished_scored_metrics.json").read_text())[
                    "scored_metrics"
                ],
                metrics,
            )

    def test_failure_outputs_replace_symlink_leaves_without_reading_or_writing_targets(self):
        for name in (
            "metrics.json",
            "failure.json",
            "verification.json",
            "results.md",
            "trial_ledger.jsonl",
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                report = root / v.REPORT
                report.mkdir(parents=True)
                sentinel = root / "frozen_prior_record.json"
                sentinel.write_bytes(b'"untouchable prior record"\n')
                before = sentinel.read_bytes()
                (report / "metrics.json").write_text(
                    json.dumps({"protocol_sha256": "a" * 64, "rows": [], "leads": []})
                )
                (report / "trial_ledger.jsonl").write_bytes(b'{"event":"registered"}\n')
                leaf = report / name
                leaf.unlink(missing_ok=True)
                leaf.symlink_to(sentinel)
                result = v.invalidate_publication(
                    root, ValueError("injected current output link")
                )
                self.assertEqual(sentinel.read_bytes(), before)
                self.assertFalse(leaf.is_symlink())
                self.assertEqual(result["cumulative_hypothesis_count"], 140)
                if name == "metrics.json":
                    self.assertIsNone(result["protocol_sha256"])
                    self.assertFalse((report / "unpublished_scored_metrics.json").exists())
                if name == "trial_ledger.jsonl":
                    events = [json.loads(line) for line in leaf.read_bytes().splitlines()]
                    self.assertEqual(len(events), 3)
                    self.assertTrue(
                        all(row["event"] == "verification_failed" for row in events)
                    )

    def test_failure_report_directory_links_rejected_before_mutation(self):
        for component in ("reports", "reports/peak_age"):
            with self.subTest(component=component), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                target = root / "frozen_prior_directory"
                destination = target / "peak_age" if component == "reports" else target
                destination.mkdir(parents=True)
                (destination / "metrics.json").write_bytes(
                    b"untouchable prior directory record"
                )
                before = {
                    str(p.relative_to(target)): p.read_bytes()
                    for p in target.rglob("*")
                    if p.is_file()
                }
                link = root / component
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(target, target_is_directory=True)
                with self.assertRaises(ValueError):
                    v.invalidate_publication(
                        root, ValueError("injected report directory link")
                    )
                after = {
                    str(p.relative_to(target)): p.read_bytes()
                    for p in target.rglob("*")
                    if p.is_file()
                }
                self.assertEqual(after, before)

    def test_diagnostic_backup_links_cannot_create_or_change_targets(self):
        for invalid in (False, True):
            for dangling in (False, True):
                with (
                    self.subTest(invalid=invalid, dangling=dangling),
                    tempfile.TemporaryDirectory() as d,
                ):
                    root = Path(d)
                    report = root / v.REPORT
                    report.mkdir(parents=True)
                    sentinel = root / "frozen_backup_target.json"
                    if not dangling:
                        sentinel.write_bytes(b"unchanged prior backup")
                    before = sentinel.read_bytes() if sentinel.exists() else None
                    metrics = (
                        b"not JSON"
                        if invalid
                        else json.dumps(
                            {"protocol_sha256": "a" * 64, "rows": [], "leads": []}
                        ).encode()
                    )
                    (report / "metrics.json").write_bytes(metrics)
                    name = (
                        "unpublished_invalid_metrics.txt"
                        if invalid
                        else "unpublished_scored_metrics.json"
                    )
                    leaf = report / name
                    leaf.symlink_to(sentinel)
                    v.invalidate_publication(root, ValueError("injected backup link"))
                    self.assertEqual(
                        sentinel.read_bytes() if sentinel.exists() else None, before
                    )
                    self.assertFalse(leaf.is_symlink())
                    if invalid:
                        self.assertEqual(leaf.read_bytes(), metrics)
                    else:
                        self.assertEqual(
                            json.loads(leaf.read_text())["scored_metrics"], json.loads(metrics)
                        )

    def test_final_required_failure_trio_and_success_markers_checked(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            f = SourceFixture(root)
            proof = v.collect_closure(root, f.expected)
            original = v.pins_checked

            def changing(r, pins):
                result = original(r, pins)
                (root / "reports/event_cluster/failure.json").unlink(missing_ok=True)
                return result

            with (
                patch.object(v, "pins_checked", side_effect=changing),
                self.assertRaises((ValueError, AssertionError, FileNotFoundError)),
            ):
                v.collect_closure(root, f.expected)
            self.assertEqual(proof["counts"]["preceding_cumulative_hypotheses"], 137)


def complete_entry_fixture(root):
    f = SourceFixture(root)
    protocol = copy.deepcopy(v.CONTRACT)
    protocol["upstream"]["anchors"] = f.expected
    f.put(v.PROTOCOL, yaml.safe_dump(protocol, sort_keys=False).encode())
    f.put("src/verify_peak_age.py", Path(v.__file__).read_bytes())
    pins = v.collect_closure(root, f.expected)["files"]
    audit, daily, _ = v.admit_upstream(root, f.expected, pins)
    panel, _, metrics, prior, _, a, k = inference_fixture()
    metrics["protocol_sha256"] = f.hash(v.PROTOCOL)
    f.put(v.OUT + "/upstream_admission.json", audit)
    for name in ("features", "targets", "feature_states"):
        f.frame(
            v.OUT + "/" + name + ".parquet",
            pd.DataFrame({"synthetic": [1.0]}, index=daily.index[:1]),
        )
    f.frame(v.OUT + "/forecasts.parquet", panel)
    f.frame(
        v.OUT + "/application_states.parquet",
        pd.DataFrame({"origin": pd.bdate_range("2016-01-04", periods=a)}),
    )
    f.put(v.OUT + "/fits.json", [{"synthetic_fit": i} for i in range(k)])
    f.put(v.OUT + "/support_audit.json", {"synthetic": True})
    f.put(v.REPORT + "/metrics.json", metrics)
    events = [
        {
            "event": "registered",
            "study": "peak_age",
            "candidate": x,
            "control": c,
            "horizon": h,
            "score": "mse",
            "protocol_sha256": f.hash(v.PROTOCOL),
        }
        for x, c, h in v.COMPARISONS
    ]
    events += [{"event": "inherited", **r} for r in prior] + [
        {"event": "evaluated", **r} for r in metrics["rows"]
    ]
    f.put(
        v.REPORT + "/trial_ledger.jsonl",
        ("".join(json.dumps(r) + "\n" for r in events)).encode(),
    )
    f.put(v.REPORT + "/ENTRY_DESIGN.md", b"synthetic prospective entry design")
    f.put(v.REPORT + "/full_repository_tests.txt", b"Ran 1 test in 0.100s\n\nOK\n")
    f.put(v.REPORT + "/pre_run_checks.txt", b"Ran 1 test in 0.100s\n\nOK\n")
    code = {
        str(p.relative_to(root)): v.digest(p)
        for folder in ("src", "tests")
        for p in (root / folder).rglob("*.py")
    }
    freeze = {
        "protocol_sha256": f.hash(v.PROTOCOL),
        "code": code,
        "prefit_design": f.pins(v.REPORT + "/ENTRY_DESIGN.md"),
        "checks": {
            "full_repository_tests": 1,
            "full_log_sha256": f.hash(v.REPORT + "/full_repository_tests.txt"),
        },
    }
    f.put(v.REPORT + "/freeze_record.json", freeze)
    inputs = {**pins, **f.pins(v.REPORT + "/freeze_record.json"), **freeze["prefit_design"]}
    # Actual canonical inherited files already live in the closure in real runs.
    # The full-entry fixture bypasses only numerical reconstruction/inference;
    # use its generated prior-family source list for exact closure accounting.
    inherited = protocol["comparisons"]["inherited_sources"]
    for name in inherited:
        if name not in inputs:
            f.put(name, {"rows": []})
            inputs[name] = f.hash(name)
    preserved = {
        str(p.relative_to(root)): v.digest(p)
        for p in [*root.glob("*.yaml"), *(root / "reports").rglob("*")]
        if p.is_file()
        and p != root / v.PROTOCOL
        and root / v.REPORT not in p.parents
        and str(p.relative_to(root)) not in inputs
    }
    manifest = {
        "protocol_sha256": f.hash(v.PROTOCOL),
        "code": code,
        "inputs": inputs,
        "preserved": preserved,
    }
    f.put(v.REPORT + "/manifest.json", manifest)
    return (
        f,
        protocol,
        {
            "forecasts_verified": len(panel),
            "common_scored_origins": int(panel.origin.nunique()),
            "primitive_squared_losses_verified": len(panel),
            "application_origins_verified": a,
            "application_predictions_verified": 4 * a,
            "unscored_origins_verified": a - int(panel.origin.nunique()),
            "monthly_fits_verified": k,
            "new_nuisance_fits_verified": 2 * k,
            "new_scalar_fits_verified": k,
            "training_mean_fits_verified": k,
        },
        panel,
        metrics,
    )


class CompleteEntryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.f, self.protocol, self.reconstruction, self.panel, self.metrics = (
            complete_entry_fixture(self.root)
        )

    def run_entry(self, stage=None):
        def numerical(*args):
            self.assertEqual(len(args), 10)
            self.assertEqual(args[2], self.protocol)
            self.assertEqual(len(args[6]), len(self.panel))
            self.assertEqual(len(args[7]), self.reconstruction["monthly_fits_verified"])
            return self.reconstruction

        def inference(*args, **kwargs):
            self.assertEqual(args[2], self.protocol)
            self.assertEqual(args[3], self.metrics)
            if stage:
                stage()
            return {
                "new_hypotheses_verified": 3,
                "cumulative_hypotheses_verified": 140,
                "leads": [],
            }

        with (
            patch.object(v, "CONTRACT", self.protocol),
            patch.object(v, "verify_pipeline", side_effect=numerical),
            patch.object(v, "verify_metrics", side_effect=inference),
        ):
            return v.verify(self.root)

    def test_complete_entry_ten_snapshots_protocol_source_and_pipeline_wiring(self):
        proof = self.run_entry()
        self.assertEqual(proof["status"], "VERIFIED")
        self.assertEqual(
            proof["manifest_sha256"], v.digest(self.root / v.REPORT / "manifest.json")
        )
        self.assertEqual(set(proof["verified_output_hashes"]), set(v.OUTPUT_PATHS))
        self.assertEqual(proof["artifact_hashes_checked"]["outputs"], 10)
        self.assertEqual(proof["primitive_squared_losses_verified"], len(self.panel))
        self.assertEqual(
            proof["ledger_events_verified"],
            {"registered": 3, "inherited": 137, "evaluated": 3},
        )
        for name, signature in proof["verified_output_hashes"].items():
            self.assertEqual(signature, v.digest(self.root / name))

    def test_successful_proof_replaces_symlink_without_overwriting_prior_target(self):
        sentinel = self.root / "frozen_proof_target.json"
        sentinel.write_bytes(b"unchanged prior proof")
        proof_path = self.root / v.REPORT / "verification.json"
        proof_path.symlink_to(sentinel)
        proof = self.run_entry()
        self.assertEqual(proof["status"], "VERIFIED")
        self.assertEqual(sentinel.read_bytes(), b"unchanged prior proof")
        self.assertFalse(proof_path.is_symlink())
        self.assertEqual(json.loads(proof_path.read_text())["status"], "VERIFIED")

    def test_output_mutation_during_inference_rejects_publication(self):
        def mutation():
            path = self.root / v.OUT / "application_states.parquet"
            path.write_bytes(path.read_bytes() + b"late output mutation")

        with self.assertRaises((ValueError, AssertionError)):
            self.run_entry(mutation)
        self.assertFalse((self.root / v.REPORT / "verification.json").exists())

    def test_upstream_source_or_new_failure_mutation_after_solves_rejects(self):
        for name in (
            "reports/event_cluster_replay/failure.json",
            "reports/peak_age/failure.json",
        ):

            def mutation(name=name):
                (self.root / name).write_bytes(b"{}")

            with self.subTest(name=name), self.assertRaises((ValueError, AssertionError)):
                self.run_entry(mutation)
            (self.root / name).unlink()

    def test_required_old_failure_removed_after_solves_rejects(self):
        with self.assertRaises((ValueError, AssertionError, FileNotFoundError)):
            self.run_entry(lambda: (self.root / "reports/event_cluster/failure.json").unlink())

    def test_protocol_or_preserved_file_mutation_after_solves_rejects(self):
        path = self.root / v.PROTOCOL
        with self.assertRaises((ValueError, AssertionError)):
            self.run_entry(lambda: path.write_bytes(path.read_bytes() + b"\n# changed"))

    def test_incomplete_manifest_or_output_symlink_reject_before_solves(self):
        manifest = self.f.documents[v.REPORT + "/manifest.json"]
        manifest["code"].pop("src/verify_peak_age.py")
        self.f.put(v.REPORT + "/manifest.json", manifest)
        with (
            patch.object(v, "CONTRACT", self.protocol),
            patch.object(v, "verify_pipeline") as solved,
        ):
            with self.assertRaises((ValueError, AssertionError)):
                v.verify(self.root)
            solved.assert_not_called()

    def test_failure_guard_diagnostics_do_not_destroy_registered_ledger_bytes(self):
        report = self.root / v.REPORT
        original = (report / "trial_ledger.jsonl").read_bytes()
        with (
            patch.object(v, "verify", side_effect=ValueError("generated late audit failure")),
            self.assertRaises(ValueError),
        ):
            v.verify_with_failure_guard(self.root)
        updated = (report / "trial_ledger.jsonl").read_bytes()
        self.assertTrue(updated.startswith(original))
        self.assertEqual(len(updated.splitlines()), 146)
        self.assertEqual(
            json.loads((report / "failure.json").read_text())["cumulative_hypothesis_count"],
            140,
        )


if __name__ == "__main__":
    unittest.main()
