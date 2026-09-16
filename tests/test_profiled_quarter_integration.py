"""Prewritten generated pipeline for the prospective profiled verifier.

Every market value and target is synthetic. Frozen calendar/model primitives are
reused, while the new independent baseline solve and full verification are real.
Only bootstrap repetitions are reduced to199 for this integration fixture.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from src import civil_quarter_features as civil
from src import civil_quarter_models as model
from src import civil_quarter_sources as source
from src import profiled_quarter_search as search
from src import verify_profiled_quarter as verify
from src.verify_macro_overnight import scan_source_admissions
from tests import test_civil_quarter_integration as legacy
from tests.test_civil_quarter_sources import fixture as source_fixture

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    original, targets, old_protocol, unknown = legacy.fixture()
    protocol = yaml.safe_load((ROOT / "profiled_quarter.yaml").read_bytes())
    protocol["index"] = old_protocol["index"]
    protocol["inference"]["bootstrap_draws"] = 199
    return original, targets, protocol, unknown


def synthetic_prior():
    rows = [
        {
            "study": "synthetic_prior",
            "candidate": str(i),
            "control": "baseline",
            "horizon": 1,
            "p_conservative": 1.0,
        }
        for i in range(121)
    ]
    for study in ("civil_quarter", "civil_quarter_replay"):
        rows.extend(
            {
                "study": study,
                "candidate": "quarter",
                "control": control,
                "score": "proper_variance",
                "horizon": 1,
                "p_conservative": 1.0,
            }
            for control in ("baseline", "mean")
        )
    return rows


class ProfiledSourceStagingIntegration(unittest.TestCase):
    def test_checked_closure_outside_ledger_directory_reaches_real_frozen_scanner(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            tempfile.TemporaryDirectory() as temporary,
        ):
            root, stage = Path(folder), Path(temporary)
            protocol, pins = source_fixture(root)
            producer = source.collect_sources(root, protocol, pins)
            independently = verify.collect_sources(root, protocol, pins)
            self.assertEqual(producer, independently)
            registered = {**pins, **producer["files"]}
            snapshots = verify.snapshot_sources(root, registered)
            (root / "data/exemplars/cpi.txt").write_bytes(b"changed after snapshot")
            for name, payload in snapshots.items():
                destination = stage / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
            audit = scan_source_admissions(stage, protocol)
            self.assertEqual(audit["mismatches"], [])
            self.assertEqual(audit["bounded_source_sections_scanned"], 2)
            self.assertIn("data/exemplars/cpi.txt", independently["runtime_files"])
            with self.assertRaises((AssertionError, ValueError)):
                verify.snapshot_sources(root, registered)


class ProfiledPipelineIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original, cls.targets, cls.protocol, cls.unknown_month = fixture()
        cls.features = civil.augment_features(cls.original)
        with patch.object(model, "fit_predict") as fitting:
            cls.support = model.preflight(cls.features, cls.targets, cls.protocol["index"])
        fitting.assert_not_called()
        cls.independent_support = verify.preflight(cls.features, cls.targets, cls.protocol)
        panel, fits = model.forecast_panel(cls.features, cls.targets, cls.protocol["index"])
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            panel.to_parquet(folder / "forecasts.parquet")
            (folder / "fits.json").write_text(json.dumps(fits, allow_nan=False))
            cls.panel = pd.read_parquet(folder / "forecasts.parquet")
            cls.fits = json.loads((folder / "fits.json").read_bytes())
        cls.proof = verify.verify_forecasts(
            cls.features, cls.targets, cls.panel, cls.fits, cls.protocol
        )
        with (
            patch.object(search, "inherited") as inherited,
            patch.object(search.inference, "digest", return_value="a" * 64),
        ):
            cls.metrics = search.evaluate(
                cls.panel, cls.features, cls.protocol, len(cls.fits), prior=synthetic_prior()
            )
        inherited.assert_not_called()
        cls.metrics["common_application_origins"] = cls.support["common_application_origins"]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(verify, "inherited_rows", return_value=synthetic_prior()),
        ):
            cls.metric_proof = verify.verify_metrics(
                Path(directory),
                cls.panel,
                cls.features,
                cls.protocol,
                cls.metrics,
                len(cls.fits),
                cls.support["common_application_origins"],
            )

    def assert_nested_close(self, actual, expected):
        if isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_nested_close(actual[key], expected[key])
        elif isinstance(expected, list):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected, strict=True):
                self.assert_nested_close(left, right)
        elif isinstance(expected, float):
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
        else:
            self.assertEqual(actual, expected)

    def test_profiled_baseline_reconstruction_retains_every_native_forecast_and_fit(self):
        self.assert_nested_close(self.support, self.independent_support)
        self.assertEqual(self.proof["forecasts_verified"], len(self.panel))
        self.assertEqual(self.proof["monthly_fits_verified"], len(self.fits))
        self.assertEqual(
            self.proof["common_application_origins"],
            self.support["common_application_origins"],
        )
        self.assertEqual(
            self.proof["common_scored_origins"], self.support["common_scored_origins"]
        )
        audits = self.proof["baseline_solver_audits"]
        self.assertEqual(
            [item["fit_origin"] for item in audits], [item["fit_origin"] for item in self.fits]
        )
        self.assertEqual(len(audits), len(self.fits))
        for item in audits:
            audit = item["audit"]
            self.assertTrue(np.equal(audit["start_slopes"], 0).all())
            self.assertLessEqual(
                audit["unprofiled_gradient_max_abs"],
                self.protocol["verification"]["accepted_gradient_tolerance"],
            )
        self.assertTrue(self.panel.groupby("origin").size().eq(3).all())
        pd.testing.assert_frame_equal(
            self.features.loc[:, self.original.columns], self.original, check_exact=True
        )

    def test_unscored_first_month_is_fitted_and_unknown_source_month_is_excluded(self):
        first = pd.Timestamp(self.protocol["index"]["origin_start"])
        self.assertEqual(self.fits[0]["fit_origin"], str(first.date()))
        self.assertGreater(self.fits[0]["application_n"], 0)
        self.assertFalse(self.panel.origin.dt.to_period("M").eq(first.to_period("M")).any())
        self.assertFalse(self.panel.origin.dt.to_period("M").eq(self.unknown_month).any())
        unknown = self.features.index.to_period("M") == self.unknown_month
        self.assertTrue(self.features.loc[unknown, "cpi_plan"].isna().all())
        self.assertTrue(self.targets.loc[unknown, "y"].notna().all())
        self.assertTrue(
            all(
                pd.Timestamp(item["fit_origin"]).to_period("M") != self.unknown_month
                for item in self.fits
            )
        )
        self.assertGreater(
            self.support["common_application_origins"], self.support["common_scored_origins"]
        )
        self.assertNotIn(
            pd.Timestamp(self.protocol["index"]["development"][1]), set(self.panel.origin)
        )

    def test_both_failed_predecessors_remain_in_full_127_family(self):
        self.assertEqual(self.metric_proof["new_hypotheses_verified"], 2)
        self.assertEqual(self.metric_proof["cumulative_hypotheses_verified"], 127)
        self.assertEqual(self.metrics["cumulative_hypothesis_count"], 127)
        self.assertEqual(len(self.metrics["inherited_rows"]), 125)
        self.assertEqual(search.WAVE_ALPHA, 0.05 / (17 * 18))
        for study in ("civil_quarter", "civil_quarter_replay"):
            kept = [row for row in self.metrics["inherited_rows"] if row["study"] == study]
            self.assertEqual(
                [(row["candidate"], row["control"]) for row in kept],
                [("quarter", "baseline"), ("quarter", "mean")],
            )
            self.assertTrue(all(row["p_conservative"] == 1 for row in kept))
        self.assertEqual(self.metric_proof["phase_comparisons_verified"], 4)
        self.assertEqual(self.metric_proof["bootstrap_runs_verified"], 12)
        self.assertEqual(self.metrics["leads"], self.metric_proof["leads"])
        for removed_study in ("civil_quarter", "civil_quarter_replay"):
            prior = [row for row in synthetic_prior() if row["study"] != removed_study]
            with self.subTest(removed_study=removed_study), self.assertRaises(ValueError):
                search.evaluate(
                    self.panel, self.features, self.protocol, len(self.fits), prior=prior
                )

    def test_independent_coefficient_and_unscored_fit_faults_reject(self):
        for fault in ("coefficient", "missing_unscored_fit"):
            fits = deepcopy(self.fits)
            if fault == "coefficient":
                fits[0]["model_audit"]["baseline"]["scaled_beta"][1] += 0.01
            else:
                fits.pop(0)
            with self.subTest(fault=fault), self.assertRaises((AssertionError, ValueError)):
                verify.verify_forecasts(
                    self.features, self.targets, self.panel, fits, self.protocol
                )

    def test_failed_verification_invalidates_entire_family_and_preserves_both_prior_failures(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "reports/profiled_quarter"
            report.mkdir(parents=True)
            (report / "metrics.json").write_text(json.dumps(self.metrics, allow_nan=False))
            events = [{"event": "inherited", **row} for row in synthetic_prior()]
            events += [
                {
                    "event": "registered",
                    "study": "profiled_quarter",
                    "candidate": "quarter",
                    "control": control,
                    "score": "proper_variance",
                    "horizon": 1,
                    "protocol_sha256": "a" * 64,
                }
                for control in ("baseline", "mean")
            ]
            events += [{"event": "evaluated", **row} for row in self.metrics["rows"]]
            (report / "trial_ledger.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in events)
            )
            previous = {}
            for study in ("civil_quarter", "civil_quarter_replay"):
                path = root / "reports" / study / "metrics.json"
                path.parent.mkdir(parents=True)
                payload = json.dumps(
                    {
                        "status": "UNEVALUABLE",
                        "rows": [row for row in synthetic_prior() if row["study"] == study],
                    }
                ).encode()
                path.write_bytes(payload)
                previous[path] = hashlib.sha256(payload).hexdigest()
            with (
                patch.object(
                    verify,
                    "verify",
                    side_effect=AssertionError("synthetic independent full gradient failure"),
                ),
                self.assertRaises(AssertionError),
            ):
                verify.verify_with_failure_guard(root)
            failed = json.loads((report / "metrics.json").read_bytes())
            self.assertEqual(failed, json.loads((report / "failure.json").read_bytes()))
            self.assertEqual(
                (
                    failed["status"],
                    failed["hypothesis_count"],
                    failed["cumulative_hypothesis_count"],
                    failed["leads"],
                ),
                ("UNEVALUABLE", 2, 127, []),
            )
            self.assertTrue(
                all(
                    row["phases"] == []
                    and row["p_conservative"]
                    == row["p_holm_wave"]
                    == row["p_holm_cumulative"]
                    == 1
                    for row in failed["rows"]
                )
            )
            history = [
                json.loads(line)
                for line in (report / "trial_ledger.jsonl").read_bytes().splitlines()
            ]
            self.assertEqual(len(history), 131)
            self.assertEqual(
                [row["event"] for row in history[-2:]], ["verification_failed"] * 2
            )
            self.assertEqual(
                [{k: v for k, v in row.items() if k != "event"} for row in history[-2:]],
                failed["rows"],
            )
            for path, signature in previous.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), signature)


if __name__ == "__main__":
    unittest.main()
