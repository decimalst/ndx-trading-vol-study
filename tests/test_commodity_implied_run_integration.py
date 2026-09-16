"""Generated temporary-root orchestration tests; all numerical stages are mocked."""

import contextlib
import copy
import hashlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import yaml

from src import commodity_implied_search as search
from src.commodity_implied_pipeline import InsufficientDataError


def metric_fixture():
    return {
        "status": "COMPLETED",
        "hypothesis_count": 2,
        "cumulative_hypothesis_count": 144,
        "leads": [],
        "rows": [
            {
                "study": "commodity_implied",
                "candidate": "candidate",
                "control": control,
                "horizon": 5,
                "score": "qlike",
                "p_conservative": 1.0,
                "p_holm_wave": 1.0,
                "p_holm_cumulative": 1.0,
            }
            for control in ("matched", "market")
        ],
    }


@contextlib.contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
        root = Path(directory)
        report, out = root / "reports/new", root / "data/new"
        report.mkdir(parents=True)
        original = {}
        for name in ("data/raw/market.csv", "data/raw/OVX.response", "data/raw/GVZ.response"):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            original[name] = ("generated raw sentinel: " + name + "\n").encode()
            path.write_bytes(original[name])
        protocol = {
            "outputs": {"reports": "reports/new", "data": "data/new"},
            "source_pins": {
                name: hashlib.sha256(value).hexdigest() for name, value in original.items()
            },
            "commodity": {
                "histories": {
                    "OVX": {"raw": "data/raw/OVX.response"},
                    "GVZ": {"raw": "data/raw/GVZ.response"},
                }
            },
            "forecast": {"generated_test_only": True},
        }
        protocol_path = root / "commodity_implied.yaml"
        protocol_path.write_text(yaml.safe_dump(protocol))
        signature = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
        frozen = {
            "status": "FROZEN_BEFORE_COMMODITY_MARKET_COHORT_OR_FITS",
            "protocol_sha256": signature,
            "inputs": protocol["source_pins"],
            "code": {},
        }
        (report / "freeze_record.json").write_text(json.dumps(frozen))
        prior = [
            {
                "study": "generated",
                "candidate": "candidate",
                "control": f"control{i}",
                "horizon": 5,
                "score": "qlike",
                "source": "reports/generated/metrics.json",
                "source_sha256": "b" * 64,
                "source_row_index": i,
                "p_conservative": 1.0,
            }
            for i in range(142)
        ]
        calendar = pd.bdate_range("2014-02-03", periods=8)
        sources = {
            "daily": pd.DataFrame({"generated": range(8)}, index=calendar),
            "cross": pd.DataFrame({"generated": range(8)}, index=calendar),
            "iv": pd.DataFrame({"generated": range(8)}, index=calendar),
        }
        commodity_iv = pd.DataFrame({"OVX": 20.0, "GVZ": 15.0}, index=calendar)
        market = pd.DataFrame({"const": 1.0, "rv_total": 0.001}, index=calendar)
        commodity = pd.DataFrame({"lovx": -8.0, "lgvz": -9.0}, index=calendar)
        targets = pd.DataFrame(
            {"y": 0.001, "target_end": pd.Series(calendar, index=calendar).shift(-5)},
            index=calendar,
        )
        produced = {
            "applications": pd.DataFrame(
                {
                    "origin": calendar,
                    "pred_market": 0.001,
                    "pred_matched": 0.0011,
                    "pred_candidate": 0.0012,
                }
            ),
            "panel": pd.DataFrame(
                {
                    "origin": calendar[:3].repeat(3),
                    "model": ["market", "matched", "candidate"] * 3,
                    "prediction": 0.001,
                }
            ),
            "coverage": pd.DataFrame(
                {
                    "origin": calendar,
                    "scored": [True] * 3 + [False] * 5,
                    "status": ["scored"] * 3 + ["target_not_mature"] * 5,
                }
            ),
            "schedules": pd.DataFrame(
                {"month": ["2014-02"], "status": ["fitted"], "train_n": [1000]}
            ),
            "fits": [{"month": "2014-02", "generated_fit": True}],
        }
        calls, synced = [], []
        real_fsync = os.fsync

        def durable(fd):
            real_fsync(fd)
            synced.append(os.fstat(fd).st_ino)

        def events():
            return [
                json.loads(line)
                for line in (report / "trial_ledger.jsonl").read_text().splitlines()
            ]

        def assert_registered():
            ledger = report / "trial_ledger.jsonl"
            rows = events()
            assert len(rows) == 144
            assert ledger.stat().st_ino in synced
            assert [r["event"] for r in rows[:2]] == ["registered", "registered"]
            assert [r["control"] for r in rows[:2]] == ["matched", "market"]
            assert all(r["protocol_sha256"] == signature for r in rows[:2])
            assert [{k: v for k, v in r.items() if k != "event"} for r in rows[2:]] == prior

        def stage(name, result, registration=False):
            def call(*args, **kwargs):
                if registration:
                    assert_registered()
                calls.append(name)
                return result

            return call

        mocks = {}
        specifications = {
            "market_source": ("src.claims_release_inputs.read_market_sources", sources, True),
            "commodity_source": (
                "src.commodity_implied_inputs.read_commodity_sources",
                commodity_iv,
                True,
            ),
            "market_features": (
                "src.claims_release_market.build_market_features",
                market,
                False,
            ),
            "commodity_features": (
                "src.commodity_implied_features.build_commodity_features",
                commodity,
                False,
            ),
            "targets": ("src.claims_release_market.make_five_session_targets", targets, False),
            "models": ("src.commodity_implied_pipeline.build_panel", produced, False),
            "forecast_verification": (
                "src.verify_commodity_implied_forecasts.verify_forecasts",
                {"status": "VERIFIED", "generated": True},
                False,
            ),
            "score": ("src.commodity_implied_search.evaluate", metric_fixture(), False),
        }
        for name, (target, result, registration) in specifications.items():
            mocks[name] = stack.enter_context(
                patch(target, side_effect=stage(name, result, registration))
            )
        # Numerical score verification is mocked at its import boundary. The
        # independent verifier's own contracts separately validate its math.
        score_module = types.ModuleType("src.verify_commodity_implied_scores")
        score_module.verify_scores = MagicMock(
            side_effect=stage("score_verification", {"status": "VERIFIED", "generated": True})
        )
        module_name = score_module.__name__
        previous_module = sys.modules.get(module_name)
        sys.modules[module_name] = score_module
        if previous_module is None:
            stack.callback(sys.modules.pop, module_name, None)
        else:
            stack.callback(sys.modules.__setitem__, module_name, previous_module)
        mocks["score_verification"] = score_module.verify_scores
        mocks["source_pins"] = stack.enter_context(
            patch.object(search, "_source_pins", return_value=(protocol["source_pins"], prior))
        )
        mocks["freeze"] = stack.enter_context(patch.object(search, "_verify_freeze"))
        stack.enter_context(patch.object(search.os, "fsync", side_effect=durable))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        yield types.SimpleNamespace(
            root=root,
            report=report,
            out=out,
            original=original,
            protocol=protocol,
            frozen=frozen,
            prior=prior,
            sources=sources,
            market=market,
            commodity=commodity,
            targets=targets,
            produced=produced,
            mocks=mocks,
            calls=calls,
            events=events,
        )


class CommodityRunIntegrationTests(unittest.TestCase):
    def assert_preserved(self, case):
        for name, expected in case.original.items():
            self.assertEqual((case.root / name).read_bytes(), expected)

    def assert_whole_failure(self, case, result, status="INVALID_RUN"):
        self.assertEqual(result["status"], "UNEVALUABLE")
        self.assertTrue(result["whole_wave_aborted"])
        self.assertEqual(result["hypothesis_count"], 2)
        self.assertEqual(result["cumulative_hypothesis_count"], 144)
        self.assertEqual(result["leads"], [])
        self.assertEqual(result["inherited_rows"], case.prior)
        self.assertEqual([r["control"] for r in result["rows"]], ["matched", "market"])
        for row in result["rows"]:
            self.assertEqual(row["status"], status)
            self.assertEqual(
                [row[k] for k in ("p_conservative", "p_holm_wave", "p_holm_cumulative")],
                [1.0, 1.0, 1.0],
            )
        events = case.events()
        self.assertEqual(len(events), 146)
        self.assertEqual([r["event"] for r in events[-2:]], ["evaluated", "evaluated"])
        self.assertTrue((case.report / "failure.json").is_file())
        self.assertEqual(json.loads((case.report / "metrics.json").read_text()), result)
        self.assertEqual(
            json.loads((case.report / "terminal.json").read_text())["status"], "UNEVALUABLE"
        )
        self.assert_preserved(case)

    def test_durable_whole_family_registration_precedes_both_source_readers(self):
        with fixture() as case:
            result = search.run(case.root)
            self.assertEqual(result["status"], "COMPLETED")
            self.assertEqual(case.calls[:2], ["market_source", "commodity_source"])
            self.assertEqual(len(case.events()), 146)
            self.assertEqual(result["inherited_rows"], case.prior)
            self.assert_preserved(case)

    def test_all_stages_and_persisted_outputs_link_to_the_completed_proof(self):
        with fixture() as case:
            result = search.run(case.root)
            self.assertEqual(
                case.calls,
                [
                    "market_source",
                    "commodity_source",
                    "market_features",
                    "commodity_features",
                    "targets",
                    "models",
                    "forecast_verification",
                    "score",
                    "score_verification",
                ],
            )
            for name in (
                "features",
                "targets",
                "applications",
                "panel",
                "coverage",
                "schedules",
            ):
                path = case.out / (name + ".parquet")
                self.assertTrue(path.is_file())
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                expected = (
                    case.market.join(case.commodity)
                    if name == "features"
                    else (case.targets if name == "targets" else case.produced[name])
                )
                pd.testing.assert_frame_equal(
                    pd.read_parquet(path), expected, check_freq=False
                )
            self.assertEqual(
                json.loads((case.out / "fits.json").read_text()), case.produced["fits"]
            )
            proof = json.loads((case.report / "verification.json").read_text())
            self.assertEqual(proof["status"], "VERIFIED")
            self.assertEqual(len(proof["outputs"]), 7)
            for name, signature in proof["outputs"].items():
                self.assertEqual(
                    hashlib.sha256((case.root / name).read_bytes()).hexdigest(), signature
                )
            terminal = json.loads((case.report / "terminal.json").read_text())
            self.assertEqual(
                terminal["metrics_sha256"],
                hashlib.sha256((case.report / "metrics.json").read_bytes()).hexdigest(),
            )
            self.assertEqual(
                terminal["trial_ledger_sha256"],
                hashlib.sha256((case.report / "trial_ledger.jsonl").read_bytes()).hexdigest(),
            )
            self.assertEqual(result["cumulative_hypothesis_count"], 144)
            self.assertFalse((case.report / "failure.json").exists())
            self.assert_preserved(case)

    def test_missing_or_mismatched_freeze_stops_before_source_reads_and_registration(self):
        for defect in ("missing", "mismatch"):
            with self.subTest(defect=defect), fixture() as case:
                if defect == "missing":
                    (case.report / "freeze_record.json").unlink()
                else:
                    case.mocks["freeze"].side_effect = ValueError("generated freeze mismatch")
                with self.assertRaises((ValueError, FileNotFoundError)):
                    search.run(case.root)
                case.mocks["market_source"].assert_not_called()
                case.mocks["commodity_source"].assert_not_called()
                case.mocks["source_pins"].assert_not_called()
                self.assertFalse((case.report / "trial_ledger.jsonl").exists())
                self.assert_preserved(case)

    def test_failed_source_authentication_prevents_registration_and_numerical_reads(self):
        with fixture() as case:
            case.mocks["source_pins"].side_effect = ValueError("generated source pin mismatch")
            with self.assertRaises(ValueError):
                search.run(case.root)
            case.mocks["market_source"].assert_not_called()
            case.mocks["commodity_source"].assert_not_called()
            self.assertFalse((case.report / "trial_ledger.jsonl").exists())
            self.assert_preserved(case)

    def test_existing_output_or_registration_is_never_restarted(self):
        for existing in ("output", "ledger"):
            with self.subTest(existing=existing), fixture() as case:
                path = (
                    case.out / "previous.txt"
                    if existing == "output"
                    else case.report / "trial_ledger.jsonl"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"existing attempt sentinel\n")
                with self.assertRaises(ValueError):
                    search.run(case.root)
                self.assertEqual(path.read_bytes(), b"existing attempt sentinel\n")
                case.mocks["market_source"].assert_not_called()
                case.mocks["commodity_source"].assert_not_called()
                self.assert_preserved(case)

    def test_either_source_reader_failure_retains_both_attempts_at_one(self):
        for name in ("market_source", "commodity_source"):
            with self.subTest(name=name), fixture() as case:
                case.mocks[name].side_effect = ValueError("generated source decode failure")
                result = search.run(case.root)
                self.assert_whole_failure(case, result)
                case.mocks["market_features"].assert_not_called()
                case.mocks["models"].assert_not_called()
                case.mocks["forecast_verification"].assert_not_called()

    def test_feature_and_target_failures_keep_whole_family_and_stop_models(self):
        for name in ("market_features", "commodity_features", "targets"):
            with self.subTest(name=name), fixture() as case:
                case.mocks[name].side_effect = ValueError(
                    "generated feature arithmetic failure"
                )
                result = search.run(case.root)
                self.assert_whole_failure(case, result)
                case.mocks["models"].assert_not_called()
                case.mocks["score"].assert_not_called()

    def test_model_failure_keeps_whole_family_and_saved_feature_target_stage(self):
        with fixture() as case:
            case.mocks["models"].side_effect = ValueError("generated rank failure")
            result = search.run(case.root)
            self.assert_whole_failure(case, result)
            self.assertEqual(
                {p.name for p in case.out.iterdir()}, {"features.parquet", "targets.parquet"}
            )
            case.mocks["forecast_verification"].assert_not_called()
            case.mocks["score"].assert_not_called()

    def test_insufficient_training_persists_complete_failure_coverage_and_schedules(self):
        with fixture() as case:
            coverage = pd.DataFrame(
                {
                    "origin": case.market.index,
                    "feature_complete": [False] + [True] * 7,
                    "status": ["incomplete_features"] + ["attempt_aborted"] * 7,
                    "scored": False,
                }
            )
            schedules = pd.DataFrame(
                {
                    "month": ["2014-02", "2014-03"],
                    "status": ["insufficient_training", "not_attempted"],
                }
            )
            case.mocks["models"].side_effect = InsufficientDataError(
                "INSUFFICIENT_DATA: generated scheduled month", coverage, schedules
            )
            result = search.run(case.root)
            self.assert_whole_failure(case, result, "INSUFFICIENT_DATA")
            pd.testing.assert_frame_equal(
                pd.read_parquet(case.out / "coverage.parquet"), coverage
            )
            pd.testing.assert_frame_equal(
                pd.read_parquet(case.out / "schedules.parquet"), schedules
            )
            self.assertFalse((case.out / "panel.parquet").exists())
            self.assertFalse((case.out / "fits.json").exists())
            case.mocks["forecast_verification"].assert_not_called()

    def test_raised_or_negative_verification_invalidates_whole_family(self):
        for name in ("forecast_verification", "score_verification"):
            for failure in ("raised", "status"):
                with self.subTest(name=name, failure=failure), fixture() as case:
                    if failure == "raised":
                        case.mocks[name].side_effect = ValueError(
                            "generated independent mismatch"
                        )
                    else:
                        case.mocks[name].side_effect = None
                        case.mocks[name].return_value = {"status": "REJECTED"}
                    result = search.run(case.root)
                    self.assert_whole_failure(case, result)
                    self.assertFalse((case.report / "verification.json").exists())
                    if name == "forecast_verification":
                        case.mocks["score"].assert_not_called()

    def test_scoring_failure_keeps_valid_outputs_without_success_proof(self):
        with fixture() as case:
            case.mocks["score"].side_effect = ValueError(
                "INSUFFICIENT_DATA: generated phase support"
            )
            result = search.run(case.root)
            self.assert_whole_failure(case, result, "INSUFFICIENT_DATA")
            self.assertTrue((case.out / "fits.json").exists())
            self.assertTrue((case.out / "panel.parquet").exists())
            self.assertFalse((case.report / "verification.json").exists())
            case.mocks["score_verification"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
