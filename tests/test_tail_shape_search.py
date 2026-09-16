"""Prewritten timing, support, inference and family gates for wave seven."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from src import tail_shape_features as features
from src import tail_shape_search as study


def fixture(n=1800):
    dates = pd.bdate_range("2010-01-04", periods=n)
    rng = np.random.default_rng(781)
    f = pd.DataFrame(
        rng.normal(size=(n, len(features.RAW))), index=dates, columns=features.RAW
    )
    f["const"] = 1.0
    f["normalization_mean"] = 0.0
    f["normalization_scale"] = 1.0
    f["feature_cutoff_date"] = pd.Series(dates, index=dates).shift(1)
    ends = pd.Series(dates, index=dates).shift(-1)
    u = rng.normal(size=n)
    t = pd.DataFrame(
        {
            "y": u,
            "raw_return": u,
            "event": (u < -1.5).astype(float),
            "target_end": ends,
            "available_date": ends,
        },
        index=dates,
    )
    t.loc[ends.isna(), ["y", "event", "raw_return"]] = np.nan
    return f, t


class TestTailShapeSearch(unittest.TestCase):
    def protocol(self):
        return yaml.safe_load(study.PROTOCOL.read_text())

    def test_fixed_protocol_and_resolution(self):
        p = self.protocol()
        study.validate(p)
        self.assertEqual(len(study.CONTRASTS), 3)
        self.assertLess(1 / (p["inference"]["bootstrap_draws"] + 1), p["wave_alpha"] / 30)
        for group, key, value in [
            ("shape", "degrees_of_freedom", 5.0),
            ("index", "event_threshold", -2.0),
            ("index", "minimum_train_events", 20),
            ("inference", "blocks", [63, 126]),
            ("index", "effect_brier_absolute", 0.0001),
            ("comparisons", "cumulative_hypotheses", 105),
        ]:
            bad = copy.deepcopy(p)
            bad[group][key] = value
            with self.assertRaises(ValueError):
                study.validate(bad)

    def test_training_excludes_unmatured_and_future_mutations(self):
        f, t = fixture()
        entry = f.index[1200]
        mask = study.training_mask(f, t, entry, self.protocol()["index"])
        self.assertEqual(mask.sum(), 1199)
        self.assertEqual(f.index[mask][-1], f.index[1198])
        changed = t.copy()
        future = changed.available_date > f.loc[entry, "feature_cutoff_date"]
        changed.loc[future, ["y", "raw_return"]] = 10.0
        changed.loc[future, "event"] = 0.0
        pd.testing.assert_series_equal(
            mask, study.training_mask(f, changed, entry, self.protocol()["index"])
        )

    def test_missing_feature_makes_common_row_ineligible(self):
        f, t = fixture()
        f.loc[f.index[40], "skew"] = np.nan
        self.assertFalse(
            study.training_mask(f, t, f.index[1200], self.protocol()["index"]).iloc[40]
        )

    def test_minimum_train_class_support_is_required(self):
        f, t = fixture()
        t["y"] = 0.0
        t["raw_return"] = 0.0
        t["event"] = 0.0
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"):
            study.training_mask(f, t, f.index[1200], self.protocol()["index"])

    def test_calendar_maturity_and_event_values_are_not_silently_repaired(self):
        f, t = fixture()
        t.loc[f.index[20], "available_date"] = f.index[20]
        with self.assertRaises(ValueError):
            study.training_mask(f, t, f.index[1200], self.protocol()["index"])
        f, t = fixture()
        t.loc[f.index[20], "event"] = 0.4
        with self.assertRaises(ValueError):
            study.training_mask(f, t, f.index[1200], self.protocol()["index"])

    def test_entry_refits_independent_of_future_missing_label(self):
        f, t = fixture(2200)
        start, end = f.index[1200], f.index[2000]
        cfg = self.protocol()["index"]
        boundary = f.index[1550]
        cfg.update(
            origin_start=str(start.date()),
            origin_end=str(end.date()),
            latest_target=str(f.index[-1].date()),
            development=[str(start.date()), str(boundary.date())],
            development_target_available_by=str(boundary.date()),
            evaluation=[str(f.index[1551].date()), str(end.date())],
            evaluation_stability=[
                [str(f.index[1551].date()), str(f.index[1800].date())],
                [str(f.index[1801].date()), str(end.date())],
            ],
            minimum_phase_events=1,
            minimum_phase_nonevents=1,
            minimum_slice_events=1,
            minimum_slice_nonevents=1,
        )
        entries, ready = study.eligible_entries(f, t, cfg)
        first = entries[~entries.to_period("M").duplicated()]
        t.loc[first, "y"] = np.nan
        t.loc[first, "event"] = np.nan
        t.loc[first, "raw_return"] = np.nan
        changed, after = study.eligible_entries(f, t, cfg)
        pd.testing.assert_index_equal(entries, changed)
        self.assertLess(after.sum(), ready.sum())
        self.assertTrue((t.loc[ready & (t.index <= boundary), "target_end"] <= boundary).all())

    def test_paired_inference_absolute_nll_can_be_negative(self):
        rng = np.random.default_rng(719)
        control = rng.normal(-1.0, 0.1, 200)
        candidate = control - 0.005
        p = self.protocol()
        p["inference"]["bootstrap_draws"] = 99
        out = study.paired_inference(candidate, control, p, 44)
        self.assertAlmostEqual(out["delta"], -0.005)
        self.assertNotIn("gain_relative", out)
        self.assertEqual(
            out["hac126"], study.inference.hac_summary(candidate - control, lags=126)
        )
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"):
            study.paired_inference(candidate[:126], control[:126], p, 44)

    def test_three_comparisons_all_required(self):
        rows = []
        for candidate, control, score in study.CONTRASTS:
            phases = [
                {
                    "name": name,
                    "n": 500,
                    "delta": -0.01,
                    "stability": [{"delta": -0.01}, {"delta": -0.01}]
                    if name == "evaluation"
                    else [],
                }
                for name in ["development", "evaluation"]
            ]
            rows.append(
                {
                    "candidate": candidate,
                    "control": control,
                    "score": score,
                    "horizon": 1,
                    "phases": phases,
                    "p_holm_wave": 1e-6,
                    "p_holm_cumulative": 1e-6,
                }
            )
        self.assertEqual(study.candidate_leads(rows), ["skew_shape"])
        rows[-1]["phases"][0]["delta"] = -0.004
        self.assertEqual(study.candidate_leads(rows), [])
        with self.assertRaises(ValueError):
            study.candidate_leads(rows[:-1])

    def test_failure_preserves_every_comparison(self):
        result = study.failure_metrics(ValueError("INSUFFICIENT_DATA: class support"), "hash")
        self.assertEqual(len(result["rows"]), 3)
        self.assertEqual(result["cumulative_hypothesis_count"], 106)
        self.assertTrue(all(row["p_conservative"] == 1 for row in result["rows"]))
        self.assertEqual(study.candidate_leads(result["rows"]), [])

    def test_inherited_source_identities_are_preserved(self):
        rows = study.inherited(self.protocol())
        self.assertEqual(len(rows), 103)
        self.assertEqual(len({(r["source"], r["source_row_index"]) for r in rows}), 103)
        self.assertEqual(sum("measure" in r for r in rows), 15)

    def test_full_synthetic_forecasts_preserve_moments_and_score_all_controls(self):
        f, t = fixture(2000)
        t["y"] *= 2
        t["raw_return"] *= 2
        t["event"] = (t.y < -1.5).astype(float).where(t.y.notna())
        p = self.protocol()
        cfg = p["index"]
        start, boundary, end = f.index[1200], f.index[1525], f.index[1850]
        cfg.update(
            origin_start=str(start.date()),
            origin_end=str(end.date()),
            latest_target=str(f.index[-1].date()),
            development=[str(start.date()), str(boundary.date())],
            development_target_available_by=str(boundary.date()),
            evaluation=[str(f.index[1526].date()), str(end.date())],
            evaluation_stability=[
                [str(f.index[1526].date()), str(f.index[1680].date())],
                [str(f.index[1681].date()), str(end.date())],
            ],
        )
        panel, fits = study.forecast_panel(f, t, cfg)
        entries, ready = study.eligible_entries(f, t, cfg)
        expected = entries[~entries.to_period("M").duplicated()]
        self.assertEqual([x["fit_origin"] for x in fits], [str(x.date()) for x in expected])
        self.assertEqual(len(panel), int(ready.sum()) * 3)
        self.assertTrue((panel.train_last_available <= panel.fit_cutoff_date).all())
        self.assertTrue(
            (panel.loc[panel.phase == "development", "target_end"] <= boundary).all()
        )
        a = panel.loc[panel.model == "constant_shape"].set_index("origin")
        b = panel.loc[panel.model == "skew_shape"].set_index("origin")
        for name in ["mu", "variance"]:
            pd.testing.assert_series_equal(a[name], b[name])
        p["inference"]["bootstrap_draws"] = 99
        with patch.object(study, "inherited", return_value=[{"p_conservative": 1.0}] * 103):
            metrics = study.evaluate(panel, f.index, p)
            self.assertEqual(len(metrics["rows"]), 3)
            for row in metrics["rows"]:
                for phase in row["phases"]:
                    self.assertEqual(phase["n"], phase["event_support"]["n"])
            for phase in metrics["calibration"].values():
                for model in phase.values():
                    self.assertGreater(sum(bin["n"] for bin in model["bins"]), 0)
            broken = panel.copy()
            position = broken.index[broken.model == "skew_shape"][0]
            broken.loc[position, "mu"] += 0.1
            with self.assertRaisesRegex(ValueError, "shared conditional moments"):
                study.evaluate(broken, f.index, p)


if __name__ == "__main__":
    unittest.main()
