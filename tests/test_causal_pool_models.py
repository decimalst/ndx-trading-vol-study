"""Prewritten synthetic causal pooling, immutable cohort and arithmetic tests."""

import unittest
from copy import deepcopy

import numpy as np
import pandas as pd

from src import causal_pool_models as model
from src import sign_memory_models as original


def fixture(
    *,
    missing_labels=(),
    omit_applications=(),
    unscored_first=False,
    unscored_month=False,
    seed_gap=0,
):
    reference = pd.bdate_range("2010-01-04", periods=1500, name="date")
    future = pd.Series(reference, index=reference).shift(-1)
    y = (np.arange(len(reference)) % 3 != 0).astype(float)
    targets = pd.DataFrame(
        {"y": y, "target_end": future, "available_date": future}, index=reference
    )
    targets.loc[reference[-1], "y"] = np.nan
    for position in missing_labels:
        targets.loc[reference[position], "y"] = np.nan
    positions = np.arange(1370, 1499)
    applications = reference[positions[~np.isin(positions, omit_applications)]]
    if unscored_first:
        targets.loc[applications[0], "y"] = np.nan
    if unscored_month:
        month = applications[0].to_period("M") + 1
        targets.loc[reference.to_period("M") == month, "y"] = np.nan
    section = {
        "models": ["frozen_baseline", "recent_frequency", "pooled"],
        "minimum_train": 1000,
        "minimum_train_per_class": 50,
        "origin_start": str(applications[0].date()),
        "origin_end": str(applications[-1].date()),
        "latest_target": str(reference[-1].date()),
        "development": [str(applications[0].date()), str(reference[1434].date())],
        "development_target_available_by": str(reference[1434].date()),
        "evaluation": [str(reference[1435].date()), str(applications[-1].date())],
    }
    fits, rows = [], []
    for number, month in enumerate(applications.to_period("M").unique()):
        app = applications[applications.to_period("M") == month]
        entry = app[0]
        position = reference.get_loc(entry)
        cutoff = reference[position - 1]
        last_allowed = reference[position - 1 - (seed_gap if number == 0 else 0)]
        train = (
            (targets.index < entry)
            & (targets.available_date <= last_allowed)
            & targets.y.notna()
        )
        train.iloc[0] = False
        n = int(train.sum())
        frequency = float(targets.loc[train, "y"].mean())
        last_available = targets.loc[train, "available_date"].max()
        fit = {
            "fit_origin": str(entry.date()),
            "fit_cutoff_date": str(cutoff.date()),
            "train_n": n,
            "train_first_origin": str(reference[train][0].date()),
            "train_last_origin": str(reference[train][-1].date()),
            "train_last_target": str(last_available.date()),
            "train_last_available": str(last_available.date()),
            "application_n": len(app),
            "model_audit": {
                "train_n": n,
                "application_n": len(app),
                "frequency": {
                    "probability": frequency,
                    "train_n": n,
                    "application_n": len(app),
                },
            },
        }
        fits.append(fit)
        keep = targets.loc[app, "y"].notna() & (
            targets.loc[app, "available_date"] <= reference[-1]
        )
        keep &= (app > reference[1434]) | (
            targets.loc[app, "available_date"] <= reference[1434]
        )
        scored = app[keep]
        actual = targets.loc[scored, "y"].to_numpy()
        if not len(scored):
            continue
        for name in original.MODELS:
            probability = (
                np.full(len(scored), frequency)
                if name == "frequency"
                else 0.55 + 0.08 * np.sin(reference.get_indexer(scored) / 13)
            )
            rows.append(
                pd.DataFrame(
                    {
                        "origin": scored,
                        "model": name,
                        "horizon": 1,
                        "feature_cutoff_date": reference[reference.get_indexer(scored) - 1],
                        "target_end": targets.loc[scored, "target_end"].to_numpy(),
                        "available_date": targets.loc[scored, "available_date"].to_numpy(),
                        "y": actual,
                        "probability": probability,
                        "loss": original.brier_loss(probability, actual),
                        "fit_origin": entry,
                        "fit_cutoff_date": cutoff,
                        "train_n": n,
                        "train_last_target": last_available,
                        "train_last_available": last_available,
                        "phase": np.where(
                            scored <= reference[1434], "development", "evaluation"
                        ),
                    }
                )
            )
    panel = (
        pd.concat(rows, ignore_index=True)
        .sort_values(["origin", "model"])
        .reset_index(drop=True)
    )
    original.validate_panel(panel)
    return panel, targets, reference, fits, section, applications


def run(args):
    return model.pool_panel(*args[:5], application_origins=args[5])


def change_label(args, position, value):
    panel, targets, reference, fits, section, applications = deepcopy(args)
    date = reference[position]
    targets.loc[date, "y"] = value
    selected = panel.origin.eq(date)
    panel.loc[selected, "y"] = value
    panel.loc[selected, "loss"] = (panel.loc[selected, "probability"] - value) ** 2
    return panel, targets, reference, fits, section, applications


class CausalPoolModels(unittest.TestCase):
    def test_initial_prior_mass_exactly_one_and_no_cold_start_deletion(self):
        args = fixture()
        panel, states = run(args)
        seed = args[3][0]
        first = states.iloc[0]
        self.assertEqual(tuple(panel), model.PANEL_COLUMNS)
        self.assertEqual(tuple(states), model.STATE_COLUMNS)
        self.assertEqual(first.origin, args[5][0])
        self.assertEqual(first.S, seed["model_audit"]["frequency"]["probability"])
        self.assertEqual(first.W, 1)
        self.assertEqual(first.recent_frequency, first.S)
        self.assertEqual(first.elapsed_sessions, 0)
        self.assertEqual(first.cumulative_updates, 0)
        self.assertEqual(first.feature_cutoff_date, pd.Timestamp(seed["fit_cutoff_date"]))
        self.assertIn(args[5][0], set(panel.origin))

    def test_first_update_has_one_minus_decay_weight_and_two_session_maturity(self):
        args = fixture()
        _, states = run(args)
        reference, targets = args[2], args[1]
        initial = states.iloc[0]
        first_position = reference.get_loc(initial.origin)
        y = targets.loc[reference[first_position - 1], "y"]
        second = states.iloc[1]
        expected = model.DECAY * initial.S + (1 - model.DECAY) * y
        self.assertEqual(second.S, expected)
        self.assertEqual(second.W, 1.0)
        self.assertEqual(second.cumulative_updates, 1)
        self.assertEqual(second.latest_consumed_available, initial.origin)
        wrong = (model.DECAY * initial.S + y) / (model.DECAY + 1)
        self.assertNotAlmostEqual(second.recent_frequency, wrong, places=5)

    def test_states_match_independent_explicit_full_calendar_weights(self):
        args = fixture(missing_labels=(1372, 1380, 1400), omit_applications=(1382, 1383))
        _, states = run(args)
        targets, reference, fits = args[1], args[2], args[3]
        seed_cutoff = pd.Timestamp(fits[0]["fit_cutoff_date"])
        seed_position = reference.get_loc(seed_cutoff)
        f0 = fits[0]["model_audit"]["frequency"]["probability"]
        for state in states.itertuples(index=False):
            position = reference.get_loc(state.feature_cutoff_date)
            elapsed = position - seed_position
            valid = targets.loc[
                (targets.available_date > seed_cutoff)
                & (targets.available_date <= state.feature_cutoff_date)
                & targets.y.notna()
            ]
            ages = position - reference.get_indexer(pd.DatetimeIndex(valid.available_date))
            weights = (1 - model.DECAY) * model.DECAY**ages
            numerator = f0 * model.DECAY**elapsed + np.dot(weights, valid.y.to_numpy())
            denominator = model.DECAY**elapsed + weights.sum()
            np.testing.assert_allclose(
                [state.S, state.W], [numerator, denominator], rtol=1e-13, atol=0
            )
            self.assertEqual(state.elapsed_sessions, elapsed)
            self.assertEqual(state.cumulative_updates, len(valid))
            self.assertEqual(state.recent_frequency, state.S / state.W)

    def test_seed_consumed_availability_is_distinct_from_state_cutoff(self):
        args = fixture(seed_gap=3)
        _, states = run(args)
        first = states.iloc[0]
        seed = args[3][0]
        self.assertEqual(first.seed_last_available, pd.Timestamp(seed["train_last_available"]))
        self.assertEqual(first.latest_consumed_available, first.seed_last_available)
        self.assertLess(first.latest_consumed_available, first.feature_cutoff_date)
        self.assertEqual(states.iloc[1].cumulative_updates, 1)

    def test_preseed_labels_are_never_refed(self):
        args = fixture(seed_gap=3)
        _, before = run(args)
        altered = deepcopy(args)
        cutoff = pd.Timestamp(args[3][0]["fit_cutoff_date"])
        selected = altered[1].available_date <= cutoff
        altered[1].loc[selected, "y"] = 1 - altered[1].loc[selected, "y"]
        _, after = run(altered)
        pd.testing.assert_frame_equal(before, after)

    def test_future_labels_cannot_change_prior_cutoff_state(self):
        args = fixture()
        _, before = run(args)
        origin_position = 1390
        changed_position = origin_position - 1
        old = args[1].iloc[changed_position].y
        altered = change_label(args, changed_position, 1 - old)
        _, after = run(altered)
        retained = before.origin <= args[2][origin_position]
        pd.testing.assert_frame_equal(before.loc[retained], after.loc[retained])
        later = before.origin == args[2][origin_position + 1]
        self.assertFalse(
            np.array_equal(
                before.loc[later, "recent_frequency"], after.loc[later, "recent_frequency"]
            )
        )

    def test_missing_labels_decay_mass_without_becoming_zero_observations(self):
        args = fixture(missing_labels=(1369, 1370))
        _, states = run(args)
        f0 = states.iloc[0].S
        self.assertEqual(states.iloc[1].S, model.DECAY * f0)
        self.assertEqual(states.iloc[1].W, model.DECAY)
        self.assertEqual(states.iloc[2].cumulative_updates, 0)
        self.assertAlmostEqual(states.iloc[2].recent_frequency, f0, places=15)
        self.assertEqual(states.iloc[3].cumulative_updates, 1)
        self.assertLess(states.iloc[3].W, 1)

    def test_zero_binary_label_adds_denominator_weight(self):
        args = fixture()
        changed = change_label(args, 1369, 0.0)
        _, states = run(changed)
        self.assertEqual(states.iloc[1].S, model.DECAY * states.iloc[0].S)
        self.assertEqual(states.iloc[1].W, 1.0)
        self.assertEqual(states.iloc[1].cumulative_updates, 1)

    def test_labels_from_feature_gap_and_development_boundary_are_consumed(self):
        args = fixture(omit_applications=(1390,))
        _, before = run(args)
        for position in (1390, 1434):
            self.assertNotIn(args[2][position], set(args[0].origin))
            altered = change_label(args, position, 1 - args[1].iloc[position].y)
            _, after = run(altered)
            applicable = before.origin >= args[2][position + 2]
            self.assertFalse(
                np.array_equal(
                    before.loc[applicable, "recent_frequency"],
                    after.loc[applicable, "recent_frequency"],
                )
            )
            pd.testing.assert_frame_equal(before.loc[~applicable], after.loc[~applicable])

    def test_unscored_first_application_seeds_before_first_scored_origin(self):
        args = fixture(unscored_first=True)
        panel, states = run(args)
        self.assertEqual(states.iloc[0].origin, args[5][0])
        self.assertFalse(states.iloc[0].scored)
        self.assertGreater(panel.origin.min(), states.iloc[0].origin)
        self.assertEqual(states.iloc[0].elapsed_sessions, 0)
        self.assertEqual(states.iloc[1].elapsed_sessions, 1)

    def test_entire_unscored_month_retains_every_application_state(self):
        args = fixture(unscored_month=True)
        panel, states = run(args)
        self.assertEqual(len(states), len(args[5]))
        month = args[5][0].to_period("M") + 1
        self.assertFalse((panel.origin.dt.to_period("M") == month).any())
        self.assertTrue((states.origin.dt.to_period("M") == month).any())
        self.assertTrue(
            states.loc[states.origin.dt.to_period("M") == month, "scored"].eq(False).all()
        )
        self.assertTrue(states.seed_fit_origin.eq(states.iloc[0].seed_fit_origin).all())
        self.assertTrue(states.cumulative_updates.is_monotonic_increasing)

    def test_original_baseline_metadata_probability_loss_and_inputs_preserved(self):
        args = fixture()
        prior = deepcopy(args)
        panel, states = run(args)
        actual = panel.loc[panel.model == "frozen_baseline"].copy()
        actual["model"] = "baseline"
        expected = args[0].loc[args[0].model == "baseline"]
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True), expected.reset_index(drop=True)
        )
        self.assertTrue(panel.groupby("origin").size().eq(3).all())
        self.assertEqual(states.scored.sum(), len(expected))
        for before, after in zip(prior[:3], args[:3], strict=True):
            if isinstance(before, pd.DataFrame):
                pd.testing.assert_frame_equal(before, after)
            else:
                pd.testing.assert_index_equal(before, after)
        self.assertEqual(prior[3:5], args[3:5])

    def test_pool_is_exact_separate_half_products_and_endpoints_are_valid(self):
        np.testing.assert_array_equal(
            model.convex_pool(np.array([0.0, 1.0, 0.0, 1.0]), np.array([0.0, 1.0, 1.0, 0.0])),
            [0.0, 1.0, 0.5, 0.5],
        )
        args = fixture()
        panel, states = run(args)
        wide = panel.pivot(index="origin", columns="model", values="probability")
        np.testing.assert_array_equal(
            wide.pooled, 0.5 * wide.frozen_baseline + 0.5 * wide.recent_frequency
        )
        np.testing.assert_array_equal(
            wide.recent_frequency,
            states.set_index("origin").loc[wide.index, "recent_frequency"],
        )

    def test_nonzero_underflow_and_bad_probability_inputs_reject(self):
        tiny = np.nextafter(0.0, 1.0)
        with self.assertRaises(ValueError):
            model.convex_pool(np.array([tiny]), np.array([0.5]))
        for bad in [np.nan, np.inf, -0.01, 1.01, 1j]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                model.convex_pool(np.array([bad]), np.array([0.5]))
        with self.assertRaises(ValueError):
            model.checked_divide(tiny, 2.0)
        self.assertEqual(model.checked_divide(tiny, 1.0), tiny)
        self.assertEqual(model.checked_divide(0.0, 2.0), 0.0)

    def test_invalid_full_calendar_and_label_alignment_reject(self):
        for fault in [
            "reference",
            "order",
            "available",
            "target",
            "binary",
            "infinite",
            "terminal",
        ]:
            args = list(deepcopy(fixture()))
            if fault == "reference":
                args[2] = args[2].delete(100)
            elif fault == "order":
                args[1] = args[1].iloc[::-1]
            elif fault == "available":
                args[1].loc[args[2][100], "available_date"] = args[2][100]
            elif fault == "target":
                args[1].loc[args[2][100], "target_end"] = args[2][102]
            elif fault == "binary":
                args[1].loc[args[2][100], "y"] = 0.5
            elif fault == "infinite":
                args[1].loc[args[2][100], "y"] = np.inf
            else:
                args[1].loc[args[2][-1], "y"] = 0.0
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                run(args)

    def test_exact_application_index_fit_counts_and_first_cutoffs_required(self):
        for fault in [
            "missing",
            "duplicate",
            "fit_count",
            "first_fit",
            "cutoff",
            "seed_date",
            "frequency",
        ]:
            args = list(deepcopy(fixture(unscored_first=True)))
            if fault == "missing":
                args[5] = args[5][1:]
            elif fault == "duplicate":
                args[5] = args[5].insert(0, args[5][0])
            elif fault == "fit_count":
                args[3][0]["application_n"] += 1
            elif fault == "first_fit":
                args[3][0]["fit_origin"] = str(args[5][1].date())
            elif fault == "cutoff":
                args[3][0]["fit_cutoff_date"] = args[3][0]["fit_origin"]
            elif fault == "seed_date":
                args[3][0]["train_last_available"] = args[3][0]["fit_origin"]
            else:
                args[3][0]["model_audit"]["frequency"]["probability"] = 1.0
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                run(args)

    def test_original_panel_tampering_rejects_before_new_state(self):
        for fault in ["cohort", "label", "frequency", "loss", "fit_metadata"]:
            args = list(deepcopy(fixture()))
            if fault == "cohort":
                args[0] = args[0].iloc[1:].copy()
            elif fault == "label":
                args[0].loc[0, "y"] = 1 - args[0].loc[0, "y"]
            elif fault == "frequency":
                selected = args[0].model.eq("frequency")
                args[0].loc[selected, "probability"] = 0.4
                args[0].loc[selected, "loss"] = (args[0].loc[selected, "y"] - 0.4) ** 2
            elif fault == "loss":
                args[0].loc[0, "loss"] += 0.01
            else:
                args[3][0]["train_n"] += 1
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                run(args)

    def test_output_validator_rejects_unknown_models_and_changed_brier(self):
        panel, _ = run(fixture())
        model.validate_panel(panel)
        for fault in ["model", "loss", "missing"]:
            altered = panel.copy()
            if fault == "model":
                altered.loc[0, "model"] = "other"
            elif fault == "loss":
                altered.loc[0, "loss"] += 0.01
            else:
                altered = altered.iloc[1:]
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                model.validate_panel(altered)


if __name__ == "__main__":
    unittest.main()
