"""Prewritten generated-only scheduling, maturity and complete-application contracts."""

import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from src import peak_age_pipeline as pipeline


def fixture(unit="ns"):
    p = yaml.safe_load((Path(__file__).resolve().parents[1] / "peak_age.yaml").read_text())
    dates = pd.bdate_range("2000-01-03", periods=245).as_unit(unit)
    rng = np.random.default_rng(731)
    f = pd.DataFrame(
        rng.uniform(0.1, 0.8, (len(dates), len(p["index"]["common"]))),
        index=dates,
        columns=p["index"]["common"],
    )
    f["const"] = 1.0
    f["drawdown_sq"] = f.drawdown**2
    f["feature_cutoff_date"] = pd.Series(dates, index=dates).shift(1)
    f.loc[dates[0], p["index"]["common"]] = np.nan
    t = pd.DataFrame(
        {
            "y": rng.normal(0, 0.03, len(dates)),
            "target_end": pd.Series(dates, index=dates).shift(-21),
            "available_date": pd.Series(dates, index=dates).shift(-21),
        },
        index=dates,
    )
    t.loc[t.target_end.isna(), "y"] = np.nan
    s = pd.DataFrame({"feature_cutoff_date": f.feature_cutoff_date}, index=dates)
    i = p["index"]
    i.update(
        origin_start=str(dates[70].date()),
        origin_end=str(dates[-1].date()),
        source_end=str(dates[-1].date()),
        latest_target=str(dates[-1].date()),
        development=[str(dates[70].date()), str(dates[139].date())],
        development_target_available_by=str(dates[139].date()),
        evaluation=[str(dates[140].date()), str(dates[-1].date())],
        evaluation_stability=[
            [str(dates[140].date()), str(dates[190].date())],
            [str(dates[191].date()), str(dates[-1].date())],
        ],
        minimum_train=40,
    )
    return f, t, s, p


def fake_fit(train, y, apply):
    return {
        "predictions": {
            name: np.full(len(apply), float(y.mean()) + offset * 0.0001)
            for offset, name in enumerate(pipeline.MODELS)
        },
        "model_audit": {},
        "transform_audit": {},
        "scalar_audit": {},
    }


class PipelineContracts(unittest.TestCase):
    def run_panel(self, f, t, s, p):
        with patch.object(pipeline.models, "fit_predict", side_effect=fake_fit):
            return pipeline.build_panel(f, t, s, p)

    def test_full_applications_precede_scoring_and_all_controls_match(self):
        f, t, s, p = fixture()
        panel, fits, states, support = self.run_panel(f, t, s, p)
        self.assertEqual(set(panel.model), set(pipeline.MODELS))
        self.assertEqual(set(panel.horizon), {21})
        self.assertEqual(len(states), 175)
        self.assertEqual(support["application_origins"], 175)
        self.assertEqual(support["unscored_origins"], 42)
        self.assertEqual(len(panel), 4 * 133)
        self.assertEqual(list(states.columns), list(pipeline.STATE_COLUMNS))
        self.assertEqual(list(panel.columns), list(pipeline.PANEL_COLUMNS))
        self.assertEqual(len(fits), len(states.origin.dt.to_period("M").unique()))
        for fit in fits:
            origin = pd.Timestamp(fit["fit_origin"])
            cutoff = pd.Timestamp(fit["feature_cutoff_date"])
            expected = f.index[
                (f.index < origin)
                & (t.available_date <= cutoff)
                & t.y.notna()
                & f.feature_cutoff_date.notna()
            ]
            self.assertEqual(fit["train_origins"], [str(x.date()) for x in expected])
            app = states.loc[states.fit_origin == origin, "origin"]
            self.assertEqual(fit["application_origins"], [str(x.date()) for x in app])
            self.assertEqual(origin, app.iloc[0])
        pipeline.validate_panel(panel)

    def test_training_availability_inclusive_boundary(self):
        f, t, _s, _p = fixture()
        entry = f.index[70]
        mask = pipeline.training_mask(f, t, entry, 40)
        self.assertTrue(mask.iloc[48])
        self.assertFalse(mask.iloc[49])
        self.assertTrue(
            (t.loc[mask, "available_date"] <= f.loc[entry, "feature_cutoff_date"]).all()
        )
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"):
            pipeline.training_mask(f, t, entry, 49)

    def test_query_missing_label_never_moves_fit_date(self):
        f, t, s, p = fixture()
        original = self.run_panel(f, t, s, p)
        changed = t.copy()
        changed.loc[f.index[70], "y"] = np.nan
        panel, fits, states, _ = self.run_panel(f, changed, s, p)
        self.assertEqual(fits[0]["fit_origin"], original[1][0]["fit_origin"])
        self.assertEqual(fits[0]["train_origins"], original[1][0]["train_origins"])
        a = states.loc[states.fit_origin == states.fit_origin.iloc[0]].reset_index(drop=True)
        b = (
            original[2]
            .loc[original[2].fit_origin == original[2].fit_origin.iloc[0]]
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(a, b)
        self.assertNotIn(f.index[70], set(panel.origin))
        self.assertIn(f.index[70], set(states.origin))

    def test_feature_incomplete_origin_never_enters_any_arm(self):
        f, t, s, p = fixture()
        f.loc[f.index[70], "peak_age"] = np.nan
        panel, fits, states, _ = self.run_panel(f, t, s, p)
        self.assertNotIn(f.index[70], set(states.origin))
        self.assertNotIn(f.index[70], set(panel.origin))
        self.assertEqual(fits[0]["fit_origin"], str(f.index[71].date()))
        self.assertTrue(
            all(str(f.index[70].date()) not in one["train_origins"] for one in fits)
        )

    def test_future_label_mutation_preserves_earlier_issuance(self):
        f, t, s, p = fixture()
        before = self.run_panel(f, t, s, p)
        t.loc[t.index[70:], "y"] += 10
        after = self.run_panel(f, t, s, p)
        first = before[2].fit_origin.iloc[0]
        pd.testing.assert_frame_equal(
            before[2].loc[before[2].fit_origin == first],
            after[2].loc[after[2].fit_origin == first],
        )

    def test_unscored_final_month_still_has_fit_and_predictions(self):
        f, t, s, p = fixture()
        t.loc[t.index[-35:], "y"] = np.nan
        panel, fits, states, support = self.run_panel(f, t, s, p)
        last_month = f.index[-1].to_period("M")
        self.assertTrue(states.origin.dt.to_period("M").eq(last_month).any())
        self.assertFalse(panel.origin.dt.to_period("M").eq(last_month).any())
        self.assertEqual(pd.Timestamp(fits[-1]["fit_origin"]).to_period("M"), last_month)
        self.assertTrue(support["unscored_origin_dates"])

    def test_ms_us_ns_transport_keeps_same_origin_decisions(self):
        outcomes = []
        for unit in ("ms", "us", "ns"):
            panel, fits, _states, support = self.run_panel(*fixture(unit))
            outcomes.append(([str(x.date()) for x in panel.origin], fits, support))
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[1], outcomes[2])

    def test_changed_date_clocks_and_misalignment_reject(self):
        for kind in ("cutoff", "end", "available", "target_index", "state_index", "duplicate"):
            f, t, s, p = fixture()
            if kind == "cutoff":
                f.loc[f.index[70], "feature_cutoff_date"] = f.index[70]
            elif kind == "end":
                t.loc[t.index[70], "target_end"] = t.index[92]
            elif kind == "available":
                t.loc[t.index[70], "available_date"] = t.index[92]
            elif kind == "target_index":
                t = t.iloc[::-1]
            elif kind == "state_index":
                s = s.iloc[::-1]
            else:
                f.index = list(f.index[:-1]) + [f.index[-2]]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.run_panel(f, t, s, p)

    def test_panel_rejects_partial_family_wrong_losses_and_pairing(self):
        panel, _, _, _ = self.run_panel(*fixture())
        for kind in (
            "missing",
            "duplicate",
            "label",
            "loss",
            "horizon",
            "nonfinite",
            "train_n",
        ):
            q = panel.copy()
            if kind == "missing":
                q = q.iloc[1:]
            elif kind == "duplicate":
                q = pd.concat([q, q.iloc[:1]])
            elif kind == "label":
                q.loc[0, "y"] += 0.1
            elif kind == "loss":
                q.loc[0, "loss"] += 0.1
            elif kind == "horizon":
                q.loc[0, "horizon"] = 63
            elif kind == "nonfinite":
                q.loc[0, "prediction"] = np.inf
            else:
                q.loc[0, "train_n"] += 1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                pipeline.validate_panel(q)

    def test_fixed_models_and_horizon_reject(self):
        for key, value in [
            ("models", ["mean", "baseline", "depth"]),
            ("horizons", [21, 63]),
            ("market_lag", 0),
        ]:
            f, t, s, p = fixture()
            p["index"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.run_panel(f, t, s, p)

    def test_input_frames_remain_unchanged(self):
        f, t, s, p = fixture()
        snapshots = [x.copy(deep=True) for x in (f, t, s)]
        self.run_panel(f, t, s, p)
        for x, y in zip((f, t, s), snapshots, strict=True):
            pd.testing.assert_frame_equal(x, y)


if __name__ == "__main__":
    unittest.main()
