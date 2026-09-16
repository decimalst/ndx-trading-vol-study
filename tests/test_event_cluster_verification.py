"""Independent pipeline contracts, written before verifier implementation."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import event_cluster_features as feature
from src import event_cluster_models as models
from src import event_cluster_pipeline as pipeline
from src import event_cluster_verification as verification
from src import range_alert_models as original
from src import verify_event_cluster as independent
from tests.test_event_cluster_pipeline import fixture


class VerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f, cls.t, cls.config = fixture()
        cls.old_panel, cls.old_fits, cls.old_states = original.forecast_panel(
            cls.f, cls.t, cls.config
        )
        cls.outputs = pipeline.forecast_panel(
            cls.f, cls.t, cls.old_panel, cls.old_fits, cls.old_states, cls.config
        )
        cls.checked = verification.verify_pipeline(
            cls.f, cls.t, cls.old_panel, cls.old_fits, cls.old_states, cls.config, *cls.outputs
        )

    def verify(self, outputs=None, **changes):
        return verification.verify_pipeline(
            changes.get("features", self.f),
            changes.get("targets", self.t),
            changes.get("upstream_panel", self.old_panel),
            changes.get("upstream_fits", self.old_fits),
            changes.get("upstream_states", self.old_states),
            self.config,
            *(self.outputs if outputs is None else outputs),
        )

    def test_complete_pipeline_and_unscored_application_counts(self):
        panel, fits, states, memory, _ = self.outputs
        result = self.checked
        self.assertEqual(result["forecasts_verified"], len(panel))
        self.assertEqual(result["new_forecasts"], len(panel) // 2)
        self.assertEqual(result["reused_control_forecasts"], len(panel) // 2)
        self.assertEqual(result["new_monthly_fits"], len(fits))
        self.assertEqual(result["independent_stage_fits_verified"], 2 * len(fits))
        self.assertEqual(result["common_application_origins"], len(states))
        self.assertEqual(result["memory_rows_verified"], len(memory))
        self.assertEqual(len(result["monthly_stage_audits"]), len(fits))
        self.assertFalse(states.iloc[-1].scored)
        self.assertEqual(result["common_scored_origins"], int(states.scored.sum()))

    def test_new_producer_calculations_and_old_optimizer_never_called(self):
        with (
            patch.object(
                feature, "build_memory", side_effect=AssertionError("producer memory")
            ),
            patch.object(
                models, "baseline_logits", side_effect=AssertionError("producer offsets")
            ),
            patch.object(models, "fit_stages", side_effect=AssertionError("producer model")),
            patch.object(
                pipeline, "forecast_panel", side_effect=AssertionError("producer pipeline")
            ),
            patch.object(original, "fit_predict", side_effect=AssertionError("old optimizer")),
        ):
            self.verify()

    def test_full_memory_value_date_and_unknown_mask_before_independent_fits(self):
        for field, row, value in (
            (independent.MEMORY, 100, 0.1234),
            ("excess_adjacency_numerator", 100, 999.0),
            ("window_last_available", 100, self.f.index[100]),
            (independent.MEMORY, 0, 0.0),
        ):
            out = copy.deepcopy(self.outputs)
            out[3].loc[self.f.index[row], field] = value
            with (
                self.subTest(field=field),
                patch.object(independent, "verify_stages") as solver,
            ):
                with self.assertRaises((AssertionError, ValueError)):
                    self.verify(out)
                solver.assert_not_called()

    def test_support_history_counts_and_old_geometry_are_exact(self):
        for kind in ("history", "count", "geometry"):
            out = copy.deepcopy(self.outputs)
            if kind == "history":
                out[4]["fits"][-1]["training_history_rows"] -= 1
            elif kind == "count":
                out[4]["common_scored_origins"] = True
            else:
                out[4]["original_support"]["fits"][0]["transform"]["means"][1] += 1
            with self.subTest(kind=kind), patch.object(independent, "verify_stages") as solver:
                with self.assertRaises((AssertionError, ValueError)):
                    self.verify(out)
                solver.assert_not_called()

    def test_training_and_application_metadata_cannot_be_changed(self):
        for kind in ("count", "last_available", "source_fit", "applications"):
            out = copy.deepcopy(self.outputs)
            fit = out[1][-1]
            if kind == "count":
                fit["train_n"] -= 1
            elif kind == "last_available":
                fit["train_last_available"] = fit["fit_origin"]
            elif kind == "source_fit":
                fit["source_fit_origin"] = out[1][0]["fit_origin"]
            else:
                fit["application_origins"].pop()
            with self.subTest(kind=kind), patch.object(independent, "verify_stages") as solver:
                with self.assertRaises((AssertionError, ValueError)):
                    self.verify(out)
                solver.assert_not_called()

    def test_saved_new_coefficients_and_stationarity_are_independent(self):
        for kind in ("nuisance", "cluster"):
            out = copy.deepcopy(self.outputs)
            if kind == "nuisance":
                out[1][0]["model_audit"]["nuisance"]["beta"][0] += 0.01
            else:
                out[1][0]["model_audit"]["cluster"]["coefficient"] += 0.1
            with self.subTest(kind=kind), self.assertRaises((AssertionError, ValueError)):
                self.verify(out)

    def test_all_unscored_state_fields_remain_bound(self):
        for field, value in (
            ("cluster_probability", 0.123),
            ("scored", True),
            ("event_count22", 900.0),
            ("feature_cutoff_date", pd.Timestamp(self.config["origin_end"])),
        ):
            out = copy.deepcopy(self.outputs)
            out[2].loc[out[2].index[-1], field] = value
            with (
                self.subTest(field=field),
                patch.object(independent, "verify_stages") as solver,
            ):
                with self.assertRaises((AssertionError, ValueError)):
                    self.verify(out)
                solver.assert_not_called()

    def test_panel_controls_scores_and_cohort_cannot_be_replaced(self):
        for kind in ("old_probability", "new_loss", "missing", "target"):
            out = copy.deepcopy(self.outputs)
            panel = out[0]
            if kind == "old_probability":
                i = panel.model.eq("baseline").idxmax()
                panel.loc[i, "probability"] = np.nextafter(panel.loc[i, "probability"], 1)
                panel.loc[i, "loss"] = (panel.loc[i, "probability"] - panel.loc[i, "y"]) ** 2
            elif kind == "new_loss":
                panel.loc[panel.model.eq("cluster").idxmax(), "loss"] += 1e-5
            elif kind == "target":
                panel.loc[panel.model.eq("cluster").idxmax(), "target_end"] += pd.Timedelta(
                    days=1
                )
            else:
                out = (panel.iloc[1:].copy(), *out[1:])
            with self.subTest(kind=kind), patch.object(independent, "verify_stages") as solver:
                with self.assertRaises((AssertionError, ValueError)):
                    self.verify(out)
                solver.assert_not_called()

    def test_coherent_finite_unscored_probability_tamper_needs_stage_replay(self):
        out = copy.deepcopy(self.outputs)
        p = out[1][-1]["application_probabilities"]["cluster"][-1] + 0.01
        out[1][-1]["application_probabilities"]["cluster"][-1] = p
        out[2].loc[out[2].index[-1], "cluster_probability"] = p
        with self.assertRaises((AssertionError, ValueError)):
            self.verify(out)

    def test_missing_original_application_or_fit_rejects(self):
        for change in (
            {"upstream_fits": self.old_fits[:-1]},
            {"upstream_states": self.old_states.iloc[:-1]},
        ):
            with (
                self.subTest(change=next(iter(change))),
                patch.object(independent, "verify_stages") as solver,
            ):
                with self.assertRaises((AssertionError, ValueError)):
                    self.verify(**change)
                solver.assert_not_called()


if __name__ == "__main__":
    unittest.main()
