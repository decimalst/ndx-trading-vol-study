"""Synthetic civil augmentation through support, fitting, replay and publication.

Only generated dates/old features/positive labels enter these tests. Historical
source tables, upstream admission and inherited empirical p-values are never
read. A fixed null replaces bootstrap computation at the scorer seam; actual
baseline/scalar fitting and independent numerical replay are exercised.
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
from src import civil_quarter_search as search
from src import plot_civil_quarter as plot
from src import verify_civil_quarter as verify


def fixture():
    dates = pd.bdate_range("2008-01-02", periods=4200, name="date")
    rng = np.random.default_rng(151515)
    old = pd.DataFrame(
        rng.normal(size=(len(dates), len(civil.OLD_FEATURES))),
        index=dates,
        columns=civil.OLD_FEATURES,
    )
    old["const"] = 1.0
    old["feature_cutoff_date"] = pd.Series(dates, index=dates).shift(1)
    first, boundary, split, end = dates[[1300, 2500, 3350, -2]]
    unknown_month = first.to_period("M") + 3
    missing_source = dates.to_period("M") == unknown_month
    old.loc[missing_source, "cpi_plan"] = np.nan
    # Source values need not have an economic interpretation in this synthetic
    # numerical test, but plan unknowns must survive every model's common mask.
    positive = np.exp(-9.0 + 0.12 * old.lrv_d.to_numpy() + 0.08 * rng.normal(size=len(dates)))
    future = pd.Series(dates, index=dates).shift(-1)
    targets = pd.DataFrame(
        {"y": positive, "target_end": future, "available_date": future}, index=dates
    )
    targets.loc[dates.to_period("M") == first.to_period("M"), "y"] = np.nan
    targets.loc[dates[-1], "y"] = np.nan
    protocol = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "civil_quarter.yaml").read_bytes()
    )
    protocol["index"].update(
        origin_start=str(first.date()),
        origin_end=str(end.date()),
        latest_target=str(dates[-1].date()),
        development=[str(first.date()), str(boundary.date())],
        development_target_available_by=str(boundary.date()),
        evaluation=[str(dates[2501].date()), str(end.date())],
        evaluation_stability=[
            [str(dates[2501].date()), str(split.date())],
            [str(dates[3351].date()), str(end.date())],
        ],
    )
    return old, targets, protocol, unknown_month


def synthetic_prior():
    return [
        {
            "study": "synthetic_prior",
            "candidate": str(i),
            "control": "baseline",
            "horizon": 1,
            "p_conservative": 1.0,
        }
        for i in range(121)
    ]


def null_inference(candidate, control, difference, protocol, seed):
    return {
        "n": len(difference),
        "delta": float(np.mean(difference)),
        "candidate_loss": float(np.mean(candidate)),
        "control_loss": float(np.mean(control)),
        "p_conservative": 1.0,
        "ci95_envelope": [-1.0, 1.0],
        "hac126": {"mde80_nominal": 0.01},
        "block_inference": {},
        "nominal_mde_effect_ratio": 2.0,
    }


class CivilQuarterPipeline(unittest.TestCase):
    def assert_nested_close(self, actual, expected):
        if isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_nested_close(actual[key], expected[key])
        elif isinstance(expected, list):
            self.assertEqual(len(actual), len(expected))
            for a, e in zip(actual, expected, strict=True):
                self.assert_nested_close(a, e)
        elif isinstance(expected, float):
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
        else:
            self.assertEqual(actual, expected)

    def test_generated_pipeline_keeps_source_unknowns_unscored_fits_and_all_three_controls(
        self,
    ):
        old, targets, protocol, unknown_month = fixture()
        original = old.copy(deep=True)
        features = civil.augment_features(old)
        pd.testing.assert_frame_equal(features.loc[:, old.columns], original, check_exact=True)
        with patch.object(model, "fit_predict") as optimization:
            support = model.preflight(features, targets, protocol["index"])
        optimization.assert_not_called()
        independent_support = verify.preflight(features, targets, protocol)
        self.assert_nested_close(support, independent_support)
        json.dumps(support, allow_nan=False)
        first = pd.Timestamp(protocol["index"]["origin_start"])
        self.assertEqual(support["fits"][0]["fit_origin"], str(first.date()))
        self.assertGreater(support["fits"][0]["application_n"], 0)
        for one in support["fits"]:
            self.assertNotEqual(pd.Timestamp(one["fit_origin"]).to_period("M"), unknown_month)
        panel, fits = model.forecast_panel(features, targets, protocol["index"])
        self.assertEqual(len(fits), support["monthly_fits"])
        self.assertEqual(fits[0]["fit_origin"], str(first.date()))
        self.assertFalse(panel.origin.dt.to_period("M").eq(first.to_period("M")).any())
        self.assertFalse(panel.origin.dt.to_period("M").eq(unknown_month).any())
        self.assertTrue(
            targets.loc[targets.index.to_period("M") == unknown_month, "y"].notna().all()
        )
        self.assertTrue(
            features.loc[features.index.to_period("M") == unknown_month, "cpi_plan"]
            .isna()
            .all()
        )
        self.assertTrue(panel.groupby("origin").size().eq(3).all())
        self.assertGreater(
            support["common_application_origins"], support["common_scored_origins"]
        )
        boundary = pd.Timestamp(protocol["index"]["development"][1])
        self.assertNotIn(boundary, set(panel.origin))
        self.assertTrue(
            panel.loc[panel.phase.eq("development"), "available_date"].le(boundary).all()
        )
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            panel.to_parquet(folder / "forecasts.parquet")
            (folder / "fits.json").write_text(json.dumps(fits, allow_nan=False))
            saved_panel = pd.read_parquet(folder / "forecasts.parquet")
            saved_fits = json.loads((folder / "fits.json").read_bytes())
            proof = verify.verify_forecasts(
                features, targets, saved_panel, saved_fits, protocol
            )
        self.assertEqual(proof["monthly_fits_verified"], support["monthly_fits"])
        self.assertEqual(
            proof["common_application_origins"], support["common_application_origins"]
        )
        self.assertEqual(proof["common_scored_origins"], support["common_scored_origins"])
        with (
            patch.object(search, "inherited") as inherited,
            patch.object(search, "paired_inference", side_effect=null_inference) as infer,
            patch.object(search.inference, "digest", return_value="a" * 64),
        ):
            metrics = search.evaluate(
                saved_panel, features, protocol, len(fits), prior=synthetic_prior()
            )
        inherited.assert_not_called()
        self.assertEqual(infer.call_count, 4)
        metrics["common_application_origins"] = support["common_application_origins"]
        verification = {
            "status": "VERIFIED",
            "protocol_sha256": "a" * 64,
            "forecast_reconstruction": proof,
        }
        self.assertEqual(len(plot.validated_rows(metrics, verification, "a" * 64)), 2)
        self.assertEqual(metrics["new_forecasts"], 3 * support["common_scored_origins"])
        self.assertEqual(metrics["new_monthly_fits"], support["monthly_fits"])
        self.assertEqual(metrics["cumulative_hypothesis_count"], 123)
        self.assertEqual(metrics["leads"], [])
        pd.testing.assert_frame_equal(old, original, check_exact=True)

    def test_bad_prediction_in_wholly_unscored_first_month_cannot_disappear(self):
        old, targets, protocol, _ = fixture()
        features = civil.augment_features(old)

        def bad_fit(training, actual, applications):
            predictions = {name: np.full(len(applications), 0.001) for name in civil.MODELS}
            predictions["quarter"][0] = np.nan
            return predictions, {}

        with (
            patch.object(model, "fit_predict", side_effect=bad_fit) as fitted,
            self.assertRaises(ValueError),
        ):
            model.forecast_panel(features, targets, protocol["index"])
        self.assertEqual(fitted.call_count, 1)

    def test_complete_month_source_unknowns_are_excluded_from_every_training_basis(self):
        old, targets, protocol, unknown_month = fixture()
        features = civil.augment_features(old)
        fit = old.index[old.index.to_period("M") == unknown_month + 1][0]
        mask = model.training_mask(features, targets, fit, protocol["index"]["minimum_train"])
        unknown = features.index.to_period("M") == unknown_month
        self.assertFalse(mask.loc[unknown].any())
        self.assertTrue(targets.loc[unknown, "y"].notna().all())
        self.assertTrue(features.loc[unknown, "month_end5"].notna().all())
        cutoff = features.loc[fit, "feature_cutoff_date"]
        self.assertTrue(targets.loc[mask, "available_date"].le(cutoff).all())
        self.assertGreaterEqual(int(mask.sum()), 1000)

    def test_evaluator_rejects_unknown_old_plan_before_any_inference(self):
        from tests.test_civil_quarter_search import panel as scored_fixture

        panel, features = scored_fixture()
        protocol = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "civil_quarter.yaml").read_bytes()
        )
        features.loc[panel.origin.iloc[0], "cpi_plan"] = np.nan
        with (
            patch.object(
                search, "paired_inference", side_effect=RuntimeError("inference reached")
            ) as infer,
            self.assertRaises(ValueError),
        ):
            search.evaluate(panel, features, protocol, 118, prior=synthetic_prior())
        infer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
