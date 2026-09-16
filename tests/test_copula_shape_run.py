"""Prewritten generated-only registration, scoring and durable-publication tests."""

import copy
import hashlib
import json
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src import copula_shape_run as run

CONTRASTS = (
    "qqq_shape",
    "spx_shape",
    "gaussian_shape",
    "t8_shape",
    "shape_gap",
    "shape_interaction",
)


def prior_rows(n=158):
    return [
        {
            "study": "invented",
            "id": i,
            "p_conservative": 0.5,
            "source": "generated.json",
            "source_sha256": "a" * 64,
            "source_row_index": i,
        }
        for i in range(n)
    ]


def independent_holm(values):
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    adjusted, largest = [None] * len(values), 0.0
    for rank, index in enumerate(order):
        largest = max(largest, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, largest)
    return adjusted


def score_fixture():
    calendar = pd.bdate_range("2019-12-02", "2020-02-14")
    development = calendar[(calendar >= "2019-12-02") & (calendar <= "2019-12-31")]
    evaluation = calendar[(calendar >= "2020-01-01") & (calendar <= "2020-02-14")]
    origins = development[2:-1].delete(7).append(evaluation[:-1].delete(9))
    panel = pd.DataFrame(
        {
            "origin": origins,
            "phase": np.where(origins.year == 2019, "development", "evaluation"),
            "offset": calendar.get_indexer(origins) % 5,
        }
    )
    for j, name in enumerate(CONTRASTS):
        values = -0.02 - 0.001 * j + 0.002 * np.sin(np.arange(len(panel)))
        if j == 2:
            values[panel.phase.eq("evaluation")] *= -1
        panel["d_" + name] = values
    for asset in ("qqq", "spx"):
        for margin in ("original", "calibrated", "shape"):
            panel[f"normal_{margin}_{asset}"] = np.sin(np.arange(len(panel)))
            panel[f"pit_{margin}_{asset}"] = np.linspace(0.01, 0.99, len(panel))
            panel[f"marginal_{margin}_{asset}"] = 2.0 + 0.01 * np.cos(np.arange(len(panel)))
    protocol = {
        "forecast": {
            "development": ["2019-12-02", "2019-12-31"],
            "evaluation": ["2020-01-01", "2020-02-14"],
            "stability": [["2020-01-01", "2020-01-22"], ["2020-01-23", "2020-02-14"]],
        },
        "support": {"phase_daily": 10, "offset_daily": 1, "slice_daily": 5},
        "inference": {
            "blocks": [2, 3],
            "hac_lags": 2,
            "bootstrap_draws": 19,
            "seed": 20260914,
            "phase_codes": {"development": 1, "evaluation": 2},
        },
        "comparisons": {"wave_alpha": 0.05 / (29 * 30), "cumulative_alpha": 0.05},
    }
    return panel, calendar, protocol


def fake_inference(values, mask, **kwargs):
    mean = float(np.mean(values[mask]))
    p = 1e-8 if kwargs["seed"] == 20270914 else 2e-8
    return {
        "mean": mean,
        "n": int(mask.sum()),
        "full_calendar_n": len(mask),
        "hac": {"p": p, "ci95": [mean - 0.001, mean + 0.001]},
        "block_inference": {
            str(b): {"p": p, "ci95": [mean - 0.002, mean + 0.002]} for b in kwargs["blocks"]
        },
        "p_conservative": p,
    }


class ShapeRunContracts(unittest.TestCase):
    def entry_fixture(self, directory):
        from tests.test_copula_spread_backtest import synthetic_inputs

        root = Path(directory)
        (root / run.REPORT).mkdir(parents=True)
        (root / run.OLD_DATA).mkdir(parents=True)
        risks, realized, calendar = synthetic_inputs()
        panel = realized.copy()
        panel["phase"] = ["development", "development", "evaluation", "evaluation"]
        for name in ["archive.parquet", "applications.parquet", "panel.parquet"]:
            panel.to_parquet(root / run.OLD_DATA / name, index=False)
        (root / run.OLD_DATA / "fits.json").write_text("[]\n")
        feature_path = root / run.CALENDAR_FILE
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(index=calendar.rename("date")).to_parquet(feature_path)
        (root / "copula_shape.yaml").write_text("generated: true\n")
        names = [
            str(run.OLD_DATA / n)
            for n in ["archive.parquet", "applications.parquet", "panel.parquet", "fits.json"]
        ]
        names += [run.CALENDAR_FILE, "copula_shape.yaml"]
        pins = {name: run.sha(root / name) for name in names}
        frozen = {
            "pins": pins,
            "inherited_rows": prior_rows(),
            "protocol_sha256": pins["copula_shape.yaml"],
        }
        run.dump(root / run.REPORT / "freeze.json", frozen)
        run.dump(
            root / run.REPORT / "registration.json",
            {
                "freeze_sha256": run.sha(root / run.REPORT / "freeze.json"),
                "contrasts": list(CONTRASTS),
                "inherited": 158,
                "new": 6,
                "cumulative": 164,
            },
        )
        risks["integration_gate_ambiguous"] = False
        risks["p_both_breach"] = 0.01
        risks["p_both_full"] = 0.005
        risks["var97_5"] = 0.1
        produced = {"applications": panel.copy(), "panel": panel.copy(), "fits": []}
        metrics = {
            "status": "COMPLETED",
            "rows": [{"contrast": c, "p_conservative": 0.1} for c in CONTRASTS],
            "inherited_rows": prior_rows(),
            "new_hypotheses": 6,
            "cumulative_hypotheses": 164,
            "leads": [],
        }
        return root, pins, risks, produced, metrics

    def test_bound_bytes_guard_paths_first_write_and_expected_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root, pins = Path(directory), {}
            payload = b"generated immutable bytes"
            run.save_bound_bytes(root, Path("private/item"), payload, pins)
            self.assertEqual(pins, {"private/item": hashlib.sha256(payload).hexdigest()})
            self.assertEqual(run.read_bound_bytes(root, "private/item", pins), payload)
            for name in ["../outside", "private/../../outside"]:
                with self.assertRaises(ValueError):
                    run.read_bound_bytes(root, name, pins)
            with self.assertRaises((ValueError, FileExistsError)):
                run.save_bound_bytes(root, "private/item", b"different", pins)
            (root / "private/item").write_bytes(b"mutated")
            with self.assertRaises(ValueError):
                run.read_bound_bytes(root, "private/item", pins)

    def test_source_pins_preserves_full_predecessor_records_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / run.OLD
            old.mkdir(parents=True)
            source = root / "generated-source"
            source.write_bytes(b"source")
            inherited = prior_rows(151)
            rows = [
                {
                    "contrast": f"old_{j}",
                    "p_conservative": 0.2,
                    "p_holm_wave": 0.4,
                    "p_holm_cumulative": 0.8,
                    "phases": [{"name": "evaluation", "n": 17}],
                    "adjusted_difference_detected": False,
                    "extra": "preserve me",
                }
                for j in range(7)
            ]
            metrics = {
                "status": "COMPLETED",
                "inherited_rows": inherited,
                "rows": rows,
                "hypothesis_count": 7,
                "cumulative_hypothesis_count": 158,
            }
            run.dump(old / "metrics.json", metrics)
            run.dump(old / "freeze.json", {"pins": {"generated-source": run.sha(source)}})
            run.dump(old / "registration.json", {"new": 7})
            terminal = {
                "status": "COMPLETED_VERIFIED_MECHANISM_DIAGNOSTIC",
                "freeze_sha256": run.sha(old / "freeze.json"),
                "metrics_sha256": run.sha(old / "metrics.json"),
                "registration_sha256": run.sha(old / "registration.json"),
                "output_hashes": {},
                "report_artifact_hashes": {
                    str(run.OLD / "metrics.json"): run.sha(old / "metrics.json")
                },
            }
            run.dump(old / "terminal.json", terminal)
            with patch.object(run, "OLD_TERMINAL", run.sha(old / "terminal.json")):
                pins, actual = run.source_pins(root)
            self.assertEqual(actual[:151], inherited)
            self.assertEqual(len(actual), 158)
            for j, original in enumerate(rows):
                for key, value in original.items():
                    self.assertEqual(actual[151 + j][key], value)
                self.assertEqual(actual[151 + j]["source_row_index"], j)
                self.assertEqual(actual[151 + j]["source_sha256"], terminal["metrics_sha256"])
            self.assertEqual(pins["generated-source"], run.sha(source))
            source.write_bytes(b"changed")
            with (
                patch.object(run, "OLD_TERMINAL", run.sha(old / "terminal.json")),
                self.assertRaises(ValueError),
            ):
                run.source_pins(root)

    def test_freeze_requires_pass_before_registration_and_refuses_duplicate_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ["src", "tests", "docs", "reports/copula_shape"]:
                (root / folder).mkdir(parents=True)
            for name in [
                "src/generated.py",
                "copula_shape.yaml",
                ".gitignore",
                "Makefile",
                "docs/PROSPECTIVE_BENCHMARK.md",
            ]:
                (root / name).write_text("# generated fixture\n")
            receipt = {
                "status": "PASS",
                "tests_selected": 1,
                "tests_run": 1,
                "failures": 0,
                "errors": 0,
                "skipped": 0,
                "code_unchanged_during_tests": True,
                "code_sha256": {"src/generated.py": run.sha(root / "src/generated.py")},
            }
            (root / "reports/copula_shape/PREFIT.log").write_text("Ran 1 test in 0.001s\nOK\n")
            receipt["log_sha256"] = run.sha(root / "reports/copula_shape/PREFIT.log")
            path = root / "reports/copula_shape/PREFIT.json"
            bad = copy.deepcopy(receipt)
            bad["failures"] = 1
            run.dump(path, bad)
            with (
                patch.object(run, "source_pins", return_value=({}, prior_rows())),
                self.assertRaises(ValueError),
            ):
                run.freeze(root)
            self.assertFalse((root / run.REPORT / "registration.json").exists())
            path.write_text(json.dumps(receipt))
            with patch.object(run, "source_pins", return_value=({}, prior_rows())):
                result = run.freeze(root)
            self.assertEqual(result["status"], "FROZEN_REGISTERED")
            frozen = json.loads((root / run.REPORT / "freeze.json").read_text())
            registered = json.loads((root / run.REPORT / "registration.json").read_text())
            self.assertEqual(
                registered["freeze_sha256"], run.sha(root / run.REPORT / "freeze.json")
            )
            self.assertEqual(registered["contrasts"], list(CONTRASTS))
            self.assertEqual(frozen["inherited_rows"], prior_rows())
            with self.assertRaises(ValueError):
                run.freeze(root)

    def test_six_signs_full_phase_masks_common_streams_and_holm164(self):
        panel, calendar, protocol = score_fixture()
        recorder = Mock(side_effect=fake_inference)
        result = run.evaluate(panel, calendar, prior_rows(), protocol, inference=recorder)
        self.assertEqual([r["contrast"] for r in result["rows"]], list(CONTRASTS))
        self.assertEqual(recorder.call_count, 12)
        expected_wave = independent_holm([2e-8] * 6)
        expected_cum = independent_holm([0.5] * 158 + [2e-8] * 6)[-6:]
        for j, row in enumerate(result["rows"]):
            self.assertEqual(row["p_conservative"], 2e-8)
            self.assertEqual(row["p_holm_wave"], expected_wave[j])
            self.assertEqual(row["p_holm_cumulative"], expected_cum[j])
            self.assertEqual(row["adjusted_difference_detected"], j != 2)
            for k, phase in enumerate(["development", "evaluation"]):
                call = recorder.call_args_list[2 * j + k]
                values, mask = call.args
                lo, hi = protocol["forecast"][phase]
                full = calendar[(calendar >= lo) & (calendar <= hi)]
                part = panel.loc[panel.phase == phase].set_index("origin")
                np.testing.assert_array_equal(mask, full.isin(part.index))
                np.testing.assert_allclose(
                    values[mask], part.loc[full[mask], "d_" + CONTRASTS[j]]
                )
                self.assertTrue(np.isnan(values[~mask]).all())
                self.assertEqual(call.kwargs["seed"], 20260914 + (k + 1) * 10000)
                self.assertGreater(len(values), mask.sum())
        self.assertEqual(result["inherited_rows"], prior_rows())
        self.assertEqual(result["cumulative_hypotheses"], 164)

    def test_all_support_floors_precede_any_resampling(self):
        panel, calendar, protocol = score_fixture()
        for field in ["phase_daily", "offset_daily", "slice_daily"]:
            p = copy.deepcopy(protocol)
            p["support"][field] = 10000
            inference = Mock(side_effect=fake_inference)
            with self.subTest(field=field), self.assertRaises(ValueError):
                run.evaluate(panel, calendar, prior_rows(), p, inference=inference)
            inference.assert_not_called()

    def fixture(self, directory):
        root = Path(directory)
        report = root / run.REPORT
        report.mkdir(parents=True)
        source = root / "generated-input"
        source.write_bytes(b"frozen")
        pins = {"generated-input": run.sha(source)}
        prior = prior_rows()
        run.dump(report / "freeze.json", {"pins": pins, "inherited_rows": prior})
        run.dump(
            report / "registration.json", {"freeze_sha256": run.sha(report / "freeze.json")}
        )
        anchors = {
            str(run.REPORT / name): run.sha(report / name)
            for name in ["freeze.json", "registration.json"]
        }
        bound = {}
        for path, payload in [
            (run.DATA / "panel.parquet", b"generated verified panel stand-in"),
            (run.REPORT / "forecast_verification.json", b'{"status":"VERIFIED"}\n'),
            (run.REPORT / "started.json", b'{"generated":true}\n'),
        ]:
            run.save_bound_bytes(root, path, payload, bound)
        metrics = {
            "status": "COMPLETED",
            "rows": [{"contrast": c, "p_conservative": 0.1} for c in CONTRASTS],
            "inherited_rows": prior,
            "new_hypotheses": 6,
            "cumulative_hypotheses": 164,
            "leads": [],
        }
        return root, pins, anchors, bound, metrics

    def assert_failed(self, root, prior):
        report = root / run.REPORT
        failure = json.loads((report / "failure.json").read_text())
        terminal = json.loads((report / "terminal.json").read_text())
        self.assertEqual(failure["status"], "UNEVALUABLE")
        self.assertEqual(terminal["status"], "UNEVALUABLE")
        self.assertEqual(failure["inherited_rows"], prior)
        self.assertEqual([r["contrast"] for r in failure["rows"]], list(CONTRASTS))
        self.assertTrue(all(r["p_conservative"] == 1 for r in failure["rows"]))
        self.assertEqual(failure["cumulative_hypotheses"], 164)

    def test_verified_commit_binds_original_output_hashes_and_started_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root, pins, anchors, bound, metrics = self.fixture(directory)
            originals = dict(bound)
            terminal = run.commit_verified(root, metrics, bound, pins=pins, anchors=anchors)
            self.assertEqual(
                terminal["status"], "COMPLETED_VERIFIED_SHAPE_AND_SPREAD_DIAGNOSTIC"
            )
            for name, expected in originals.items():
                self.assertEqual(terminal["output_hashes"][name], expected)
            self.assertEqual(terminal["anchors"], anchors)
            self.assertFalse((root / run.REPORT / "failure.json").exists())
            run.verify_pins(root, bound)

    def test_late_metrics_write_mutations_invalidate_whole_six_family(self):
        for kind in ["source", "data", "proof", "freeze"]:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root, pins, anchors, bound, metrics = self.fixture(directory)
                target = {
                    "source": root / "generated-input",
                    "data": root / run.DATA / "panel.parquet",
                    "proof": root / run.REPORT / "forecast_verification.json",
                    "freeze": root / run.REPORT / "freeze.json",
                }[kind]
                original = run.save_bound_bytes

                def mutate(root_arg, path, payload, mapping, original=original, target=target):
                    result = original(root_arg, path, payload, mapping)
                    if (
                        Path(path).name == "metrics.json"
                        and json.loads(payload)["status"] == "COMPLETED"
                    ):
                        target.write_bytes(b"late mutation")
                    return result

                with (
                    patch.object(run, "save_bound_bytes", side_effect=mutate),
                    self.assertRaises(ValueError),
                ):
                    run.commit_verified(root, metrics, bound, pins=pins, anchors=anchors)
                self.assert_failed(root, metrics["inherited_rows"])
                self.assertEqual(target.read_bytes(), b"late mutation")
                self.assertTrue((root / run.REPORT / "metrics.pre_invalidation.json").exists())

    def test_post_terminal_mutation_cannot_leave_completed_terminal(self):
        for kind in ["metrics", "source"]:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root, pins, anchors, bound, metrics = self.fixture(directory)
                target = (
                    root / run.REPORT / "metrics.json"
                    if kind == "metrics"
                    else root / "generated-input"
                )
                original = run.save_bound_bytes

                def mutate(root_arg, path, payload, mapping, original=original, target=target):
                    result = original(root_arg, path, payload, mapping)
                    if Path(path).name == "terminal.json" and json.loads(payload)[
                        "status"
                    ].startswith("COMPLETED"):
                        target.write_bytes(b"post-terminal mutation")
                    return result

                with (
                    patch.object(run, "save_bound_bytes", side_effect=mutate),
                    self.assertRaises(ValueError),
                ):
                    run.commit_verified(root, metrics, bound, pins=pins, anchors=anchors)
                self.assert_failed(root, metrics["inherited_rows"])
                retained = root / run.REPORT / "terminal.pre_invalidation.json"
                self.assertTrue(retained.exists())
                self.assertTrue(
                    json.loads(retained.read_text())["status"].startswith("COMPLETED")
                )

    def test_actual_entry_rejects_freeze_change_before_any_model_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root, pins, _risks, _produced, _metrics = self.entry_fixture(directory)
            original = run.verify_pins
            mutated = False

            def change_after_check(root_arg, mapping):
                nonlocal mutated
                result = original(root_arg, mapping)
                if not mutated and mapping == pins:
                    payload = json.loads((root / run.REPORT / "freeze.json").read_text())
                    payload["late_unregistered_change"] = True
                    (root / run.REPORT / "freeze.json").write_text(json.dumps(payload))
                    mutated = True
                return result

            model = Mock(side_effect=RuntimeError("MODEL_WORK_MUST_NOT_START"))
            with (
                patch.object(run, "verify_pins", side_effect=change_after_check),
                patch.object(run, "run_shape", model),
                self.assertRaises(ValueError),
            ):
                run.run(root)
            self.assertTrue(mutated)
            model.assert_not_called()

    def test_actual_entry_reloads_outputs_projects_risks_and_binds_started(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _pins, risks, produced, metrics = self.entry_fixture(directory)
            diagnostics = [
                {"differences": {"p_any_breach": 0.001, "mean_debit": 0.001, "es97_5": 0.001}}
            ]
            with ExitStack() as stack:
                stack.enter_context(patch.object(run, "run_shape", return_value=produced))
                stack.enter_context(patch.object(run, "evaluate", return_value=metrics))
                stack.enter_context(
                    patch.object(
                        run, "verify_score_inference", return_value={"status": "VERIFIED"}
                    )
                )
                stack.enter_context(
                    patch(
                        "src.verify_copula_shape.verify", return_value={"status": "VERIFIED"}
                    )
                )
                spread_verifier = types.ModuleType("src.verify_copula_spreads")
                spread_verifier.verify = Mock(return_value={"status": "VERIFIED"})
                stack.enter_context(
                    patch.dict(sys.modules, {"src.verify_copula_spreads": spread_verifier})
                )
                stack.enter_context(
                    patch(
                        "src.copula_spread_forecasts.build_risks",
                        return_value=(risks, diagnostics),
                    )
                )
                terminal = run.run(root)
            self.assertEqual(
                terminal["status"], "COMPLETED_VERIFIED_SHAPE_AND_SPREAD_DIAGNOSTIC"
            )
            started = str(run.REPORT / "started.json")
            self.assertIn(started, terminal["output_hashes"])
            self.assertEqual(terminal["output_hashes"][started], run.sha(root / started))
            saved = pd.read_parquet(root / run.DATA / "spread_risks.parquet")
            self.assertIn("integration_gate_ambiguous", saved)
            self.assertEqual(
                len(pd.read_parquet(root / run.DATA / "spread_positions.parquet")), 4 * 9 * 7
            )
            for name, expected in terminal["output_hashes"].items():
                self.assertEqual(run.sha(root / name), expected)
            with self.assertRaises(ValueError):
                run.run(root)


if __name__ == "__main__":
    unittest.main()
