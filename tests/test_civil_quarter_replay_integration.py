"""Generated replay integration; no empirical values or inherited results are read.

The frozen wave15 fixture supplies all market values and labels synthetically.
All monthly optimization and independent replay are real. Only bootstrap draws
are reduced to199 for this synthetic integration, leaving every model, support,
phase, block and numerical tolerance unchanged.
"""

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
from src import civil_quarter_replay_search as search
from src import plot_civil_quarter_replay as plot
from src import verify_civil_quarter_replay as verify
from tests import test_civil_quarter_integration as legacy

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    old, targets, old_protocol, unknown_month = legacy.fixture()
    protocol = yaml.safe_load((ROOT / "civil_quarter_replay.yaml").read_bytes())
    protocol["index"] = old_protocol["index"]
    protocol["inference"]["bootstrap_draws"] = 199
    return old, targets, protocol, unknown_month


def synthetic_prior():
    return [
        {
            "study": "generated_previous_trials",
            "candidate": str(number),
            "control": "baseline",
            "horizon": 1,
            "p_conservative": 1.0,
        }
        for number in range(123)
    ]


class ReplayIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old, cls.targets, cls.protocol, cls.unknown_month = fixture()
        cls.features = civil.augment_features(cls.old)
        with patch.object(model, "fit_predict") as optimization:
            cls.support = model.preflight(cls.features, cls.targets, cls.protocol["index"])
        optimization.assert_not_called()
        cls.independent_support = verify.preflight(cls.features, cls.targets, cls.protocol)
        panel, fits = model.forecast_panel(cls.features, cls.targets, cls.protocol["index"])
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            panel.to_parquet(directory / "forecasts.parquet")
            (directory / "fits.json").write_text(json.dumps(fits, allow_nan=False))
            cls.panel = pd.read_parquet(directory / "forecasts.parquet")
            cls.fits = json.loads((directory / "fits.json").read_bytes())
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
            tempfile.TemporaryDirectory() as temporary,
            patch.object(verify, "inherited_rows", return_value=synthetic_prior()),
        ):
            cls.metric_proof = verify.verify_metrics(
                Path(temporary),
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

    def test_independent_all_fit_support_geometry_and_native_forecast_replay(self):
        self.assert_nested_close(self.support, self.independent_support)
        self.assertEqual(self.proof["monthly_fits_verified"], len(self.fits))
        self.assertEqual(self.proof["forecasts_verified"], len(self.panel))
        self.assertEqual(
            self.proof["common_application_origins"],
            self.support["common_application_origins"],
        )
        self.assertEqual(
            self.proof["common_scored_origins"], self.support["common_scored_origins"]
        )
        self.assertTrue(self.panel.groupby("origin").size().eq(3).all())
        pd.testing.assert_frame_equal(
            self.features.loc[:, self.old.columns], self.old, check_exact=True
        )

    def test_unknown_source_month_and_wholly_unscored_first_month_are_distinct(self):
        first = pd.Timestamp(self.protocol["index"]["origin_start"])
        self.assertEqual(self.fits[0]["fit_origin"], str(first.date()))
        self.assertGreater(self.fits[0]["application_n"], 0)
        self.assertFalse(self.panel.origin.dt.to_period("M").eq(first.to_period("M")).any())
        self.assertFalse(self.panel.origin.dt.to_period("M").eq(self.unknown_month).any())
        self.assertTrue(
            self.targets.loc[self.targets.index.to_period("M") == self.unknown_month, "y"]
            .notna()
            .all()
        )
        self.assertTrue(
            self.features.loc[
                self.features.index.to_period("M") == self.unknown_month, "cpi_plan"
            ]
            .isna()
            .all()
        )
        self.assertTrue(
            all(
                pd.Timestamp(one["fit_origin"]).to_period("M") != self.unknown_month
                for one in self.fits
            )
        )
        self.assertGreater(
            self.support["common_application_origins"], self.support["common_scored_origins"]
        )
        self.assertNotIn(
            pd.Timestamp(self.protocol["index"]["development"][1]), set(self.panel.origin)
        )

    def test_real_synthetic_four_phase_inference_and_full_new_family(self):
        self.assertEqual(self.metric_proof["new_hypotheses_verified"], 2)
        self.assertEqual(self.metric_proof["phase_comparisons_verified"], 4)
        self.assertEqual(self.metric_proof["bootstrap_runs_verified"], 12)
        self.assertEqual(self.metric_proof["bootstrap_draws_per_run"], 199)
        self.assertEqual(self.metric_proof["cumulative_hypotheses_verified"], 125)
        self.assertEqual(self.metrics["cumulative_hypothesis_count"], 125)
        self.assertEqual(len(self.metrics["inherited_rows"]), 123)
        self.assertEqual(search.WAVE_ALPHA, 0.05 / (16 * 17))
        for row in self.metrics["rows"]:
            self.assertEqual(row["study"], "civil_quarter_replay")
            self.assertEqual(len(row["phases"]), 2)
            self.assertTrue(
                all(
                    set(phase["block_inference"]) == {"21", "63", "126"}
                    for phase in row["phases"]
                )
            )
        self.assertEqual(self.metrics["leads"], self.metric_proof["leads"])

    def test_actual_synthetic_metrics_render_only_with_same_verified_counts(self):
        verification = {
            "status": "VERIFIED",
            "protocol_sha256": "a" * 64,
            "forecast_reconstruction": self.proof,
        }
        rows = plot.validated_rows(self.metrics, verification, "a" * 64)
        self.assertEqual(len(rows), 2)
        for count in (
            "forecasts_verified",
            "monthly_fits_verified",
            "common_scored_origins",
            "common_application_origins",
        ):
            bad = deepcopy(verification)
            bad["forecast_reconstruction"][count] += 1
            with self.assertRaises(ValueError):
                plot.validated_rows(self.metrics, bad, "a" * 64)
        with tempfile.TemporaryDirectory() as directory:
            paths = plot.render_verified(self.metrics, verification, "a" * 64, directory)
            self.assertTrue(Path(paths["png"]).read_bytes().startswith(b"\x89PNG"))
            self.assertTrue(Path(paths["pdf"]).read_bytes().startswith(b"%PDF"))

    def test_independent_replay_rejects_coefficients_support_metadata_and_missing_unscored_fit(
        self,
    ):
        for fault in ("prediction", "coefficient", "support", "fit_metadata", "unscored_fit"):
            panel, fits = self.panel.copy(), deepcopy(self.fits)
            if fault == "prediction":
                panel.loc[0, "prediction"] *= 1.01
            elif fault == "coefficient":
                fits[0]["model_audit"]["baseline"]["scaled_beta"][1] += 0.01
            elif fault == "support":
                fits[0]["model_audit"]["quarter"]["support"]["groups"]["quarter_end5"][
                    "ones"
                ] += 1
            elif fault == "fit_metadata":
                fits[0]["train_n"] += 1
            else:
                fits.pop(0)
            with self.subTest(fault=fault), self.assertRaises((AssertionError, ValueError)):
                verify.verify_forecasts(
                    self.features, self.targets, panel, fits, self.protocol
                )

    def test_complete_failed_predecessor_slots_cannot_be_dropped(self):
        with self.assertRaises(ValueError):
            search.evaluate(
                self.panel,
                self.features,
                self.protocol,
                len(self.fits),
                prior=synthetic_prior()[:-2],
            )
        corrupted = deepcopy(self.metrics)
        corrupted["cumulative_hypothesis_count"] = 123
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(verify, "inherited_rows", return_value=synthetic_prior()),
            self.assertRaises(AssertionError),
        ):
            verify.verify_metrics(
                Path(temporary),
                self.panel,
                self.features,
                self.protocol,
                corrupted,
                len(self.fits),
                self.support["common_application_origins"],
            )


if __name__ == "__main__":
    unittest.main()
