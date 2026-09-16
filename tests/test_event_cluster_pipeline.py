"""Synthetic immutable-cohort orchestration tests, prewritten before producer."""

import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import event_cluster_features as feature
from src import event_cluster_models as models
from src import event_cluster_pipeline as pipeline
from src import range_alert_features as original_feature
from src import range_alert_models as original
from tests.test_range_alert_models import fixture as original_fixture


def fixture():
    f, t, p = original_fixture()
    # A real source has a preceding feature warmup. Preserve all date/label rows
    # and declare this absence before creating the old synthetic fitted records.
    f.loc[f.index[:23], original_feature.RAW[1]] = np.nan
    rng = np.random.default_rng(20261020)
    t["y"] = (rng.uniform(size=len(t)) < 0.3).astype(float)
    t.loc[t.index[-1], "y"] = np.nan
    t.loc[pd.Timestamp(p["origin_end"]), "y"] = np.nan
    return f, t, p


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f, cls.t, cls.p = fixture()
        cls.old_panel, cls.old_fits, cls.old_states = original.forecast_panel(
            cls.f, cls.t, cls.p
        )
        cls.result = pipeline.forecast_panel(
            cls.f, cls.t, cls.old_panel, cls.old_fits, cls.old_states, cls.p
        )

    def run_pipeline(self, *, f=None, t=None, old_panel=None, old_fits=None, old_states=None):
        return pipeline.forecast_panel(
            self.f if f is None else f,
            self.t if t is None else t,
            self.old_panel if old_panel is None else old_panel,
            self.old_fits if old_fits is None else old_fits,
            self.old_states if old_states is None else old_states,
            self.p,
        )

    def test_exact_four_models_original_controls_and_full_schema(self):
        panel, fits, states, memory, support = self.result
        self.assertEqual(tuple(panel.columns), original.PANEL_COLUMNS)
        self.assertEqual(set(panel.model), set(pipeline.MODELS))
        for name in ("baseline", "recent_frequency"):
            pd.testing.assert_frame_equal(
                panel.loc[panel.model.eq(name)].reset_index(drop=True),
                self.old_panel.loc[self.old_panel.model.eq(name)].reset_index(drop=True),
                check_exact=True,
            )
        self.assertEqual(len(panel), 4 * self.old_panel.origin.nunique())
        self.assertEqual(len(fits), len(self.old_fits))
        self.assertEqual(tuple(states.columns), pipeline.STATE_COLUMNS)
        self.assertTrue(memory.index.equals(self.f.index))
        self.assertEqual(tuple(memory.columns), feature.MEMORY_COLUMNS)
        self.assertEqual(support["monthly_fits"], len(fits))
        self.assertEqual(support["common_application_origins"], len(states))
        json.dumps({"fits": fits, "support": support}, allow_nan=False)

    def test_final_unscored_and_phase_boundary_application_retained(self):
        panel, fits, states, _, _ = self.result
        final = states.iloc[-1]
        self.assertEqual(final.origin, pd.Timestamp(self.p["origin_end"]))
        self.assertFalse(final.scored)
        self.assertNotIn(final.origin, set(panel.origin))
        self.assertTrue(
            np.isfinite([final.nuisance_probability, final.cluster_probability]).all()
        )
        self.assertEqual(fits[-1]["application_origins"][-1], self.p["origin_end"])
        self.assertEqual(set(fits[-1]["application_probabilities"]), set(pipeline.MODELS))
        boundary = states.loc[states.origin.eq(pd.Timestamp(self.p["development"][1]))]
        self.assertEqual(len(boundary), 1)
        self.assertFalse(boundary.iloc[0].scored)

    def test_no_old_optimizer_and_no_input_mutation(self):
        before = copy.deepcopy(
            (self.f, self.t, self.old_panel, self.old_fits, self.old_states)
        )
        with patch.object(
            original, "fit_predict", side_effect=AssertionError("old refit forbidden")
        ):
            self.run_pipeline()
        for a, b in zip(
            before,
            (self.f, self.t, self.old_panel, self.old_fits, self.old_states),
            strict=True,
        ):
            if isinstance(a, pd.DataFrame):
                pd.testing.assert_frame_equal(a, b, check_exact=True)
            else:
                self.assertEqual(a, b)

    def test_any_late_application_history_gap_blocks_all_new_fits(self):
        memory = feature.build_memory(self.t)
        memory.loc[pd.Timestamp(self.p["origin_end"]), feature.MEMORY] = np.nan
        with (
            patch.object(feature, "build_memory", return_value=memory),
            patch.object(models, "fit_stages") as fit,
        ):
            with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA|history"):
                self.run_pipeline()
            fit.assert_not_called()

    def test_any_historical_training_gap_blocks_all_new_fits_without_mask(self):
        memory = feature.build_memory(self.t)
        memory.loc[self.f.index[400], feature.NUISANCE[0]] = np.nan
        with (
            patch.object(feature, "build_memory", return_value=memory),
            patch.object(models, "fit_stages") as fit,
        ):
            with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA|history"):
                self.run_pipeline()
            fit.assert_not_called()

    def test_old_warmup_is_not_silently_removed_by_new_history(self):
        f, t, p = fixture()
        f.loc[f.index[:23], original_feature.RAW[1]] = 0.0
        old_panel, old_fits, old_states = original.forecast_panel(f, t, p)
        with patch.object(models, "fit_stages") as fit:
            with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA|history"):
                pipeline.forecast_panel(f, t, old_panel, old_fits, old_states, p)
            fit.assert_not_called()

    def test_current_month_baseline_offset_and_exact_training_cohort(self):
        seen = []
        original_fit = models.fit_stages

        def fit(train, y, apply, train_eta, query_eta, probability):
            position = len(seen)
            source = self.old_fits[position]
            mask = original.training_mask(
                self.f, self.t, source["fit_origin"], self.p["minimum_train"]
            )
            self.assertTrue(train.index.equals(self.f.index[mask]))
            self.assertTrue(y.index.equals(train.index))
            expected, _ = models.baseline_logits(self.f.loc[mask], source["model_audit"])
            np.testing.assert_array_equal(train_eta, expected)
            self.assertEqual(len(apply), source["application_n"])
            self.assertTrue(
                (
                    self.t.loc[y.index, "available_date"]
                    <= pd.Timestamp(source["fit_cutoff_date"])
                ).all()
            )
            seen.append(source["fit_origin"])
            return original_fit(train, y, apply, train_eta, query_eta, probability)

        with patch.object(models, "fit_stages", side_effect=fit):
            self.run_pipeline()
        self.assertEqual(seen, [x["fit_origin"] for x in self.old_fits])

    def test_future_label_change_cannot_change_prior_application_state(self):
        changed = self.t.copy()
        cut = self.f.index[-8]
        known = (changed.index >= cut) & changed.y.notna()
        changed.loc[known, "y"] = 1 - changed.loc[known, "y"]
        old_panel, old_fits, old_states = original.forecast_panel(self.f, changed, self.p)
        _, _, states, memory, _ = self.run_pipeline(
            t=changed, old_panel=old_panel, old_fits=old_fits, old_states=old_states
        )
        original_states = self.result[2]
        eligible = original_states.origin <= cut
        pd.testing.assert_frame_equal(
            states.loc[eligible], original_states.loc[eligible], check_exact=True
        )
        pd.testing.assert_frame_equal(
            memory.loc[:cut], self.result[3].loc[:cut], check_exact=True
        )

    def test_invalid_probability_in_final_unscored_application_rejects(self):
        original_fit = models.fit_stages

        def invalid(*args):
            probabilities, audit = original_fit(*args)
            if args[2].index[-1] == pd.Timestamp(self.p["origin_end"]):
                probabilities["cluster"][-1] = np.nan
            return probabilities, audit

        with (
            patch.object(models, "fit_stages", side_effect=invalid),
            self.assertRaises(ValueError),
        ):
            self.run_pipeline()

    def test_both_new_arms_get_original_panel_validation(self):
        for name in ("nuisance", "cluster"):
            changed = self.result[0].copy()
            changed.loc[changed.model.eq(name).idxmax(), "probability"] = np.nan
            with self.assertRaises(ValueError):
                pipeline.validate_panel(changed)
        changed = self.result[0].iloc[1:].copy()
        with self.assertRaises(ValueError):
            pipeline.validate_panel(changed)


if __name__ == "__main__":
    unittest.main()
