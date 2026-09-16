"""Synthetic snapshot -> filter -> independent proof -> scorer -> plot contracts.

Every label, probability and date table here is generated in memory. No market
file, issued empirical probability, inherited result or live admission is read.
The numerical inference engine has separate tests; this joins its runner seam
with fixed synthetic null inference so no bootstrap is needed for these checks.
"""

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import causal_pool_models as model
from src import causal_pool_search as search
from src import plot_causal_pool as plot
from src import sign_memory_models as old_model
from src import verify_causal_pool as verify


def synthetic_inputs():
    protocol = deepcopy(search.CONTRACT)
    config = protocol["index"]
    dates = pd.bdate_range("2010-01-04", "2025-10-20", name="date")
    future = pd.Series(dates, index=dates).shift(-1)
    labels = (np.arange(len(dates)) % 5 < 3).astype(float)
    targets = pd.DataFrame(
        {"y": labels, "target_end": future, "available_date": future}, index=dates
    )
    targets.iloc[-1, targets.columns.get_loc("y")] = np.nan
    # The first monthly application cohort exists, but has no scored forecasts.
    targets.loc["2016-01-01":"2016-01-31", "y"] = np.nan
    features = pd.DataFrame(1.0, index=dates, columns=search.sf.ALL_FEATURES)
    features["feature_cutoff_date"] = pd.Series(dates, index=dates).shift(1)
    features.loc["2018-07-04":"2018-07-05", "corr22"] = np.nan
    applications = search.application_origins(features, targets, config)
    fits, parts = [], []
    for month in applications.to_period("M").unique():
        app = applications[applications.to_period("M") == month]
        entry = app[0]
        cutoff = features.loc[entry, "feature_cutoff_date"]
        training = (
            np.isfinite(features.loc[:, search.sf.ALL_FEATURES]).all(axis=1)
            & features.feature_cutoff_date.notna()
            & targets.y.notna()
            & (dates < entry)
            & (targets.available_date <= cutoff)
        )
        n = int(training.sum())
        frequency = float(targets.loc[training, "y"].mean())
        last_available = targets.loc[training, "available_date"].max()
        fit = {
            "fit_origin": str(entry.date()), "fit_cutoff_date": str(cutoff.date()),
            "train_n": n, "train_first_origin": str(dates[training][0].date()),
            "train_last_origin": str(dates[training][-1].date()),
            "train_last_target": str(last_available.date()),
            "train_last_available": str(last_available.date()),
            "application_n": len(app),
            "model_audit": {
                "train_n": n, "application_n": len(app),
                "frequency": {"probability": frequency, "train_n": n,
                              "application_n": len(app)},
            },
        }
        fits.append(fit)
        keep = targets.loc[app, "y"].notna()
        keep &= targets.loc[app, "available_date"] <= pd.Timestamp(config["latest_target"])
        keep &= (app > pd.Timestamp(config["development"][1])) | (
            targets.loc[app, "available_date"] <= pd.Timestamp(config["development"][1])
        )
        scored = app[keep]
        if len(scored) == 0:
            continue
        actual = targets.loc[scored, "y"].to_numpy()
        for name in old_model.MODELS:
            probability = (
                np.full(len(scored), frequency) if name == "frequency"
                else 0.55 + 0.04 * np.sin(dates.get_indexer(scored) / 17.0)
            )
            parts.append(pd.DataFrame({
                "origin": scored, "model": name, "horizon": 1,
                "feature_cutoff_date": features.loc[scored, "feature_cutoff_date"].to_numpy(),
                "target_end": targets.loc[scored, "target_end"].to_numpy(),
                "available_date": targets.loc[scored, "available_date"].to_numpy(),
                "y": actual, "probability": probability,
                "loss": old_model.brier_loss(probability, actual),
                "fit_origin": entry, "fit_cutoff_date": cutoff, "train_n": n,
                "train_last_target": last_available, "train_last_available": last_available,
                "phase": np.where(scored <= config["development"][1], "development", "evaluation"),
            }))
    issued = pd.concat(parts, ignore_index=True).sort_values(["origin", "model"]).reset_index(drop=True)
    old_model.validate_panel(issued)
    return features, targets, issued, fits, protocol


def null_inference(candidate, control, difference, protocol, seed):
    return {
        "n": len(difference), "delta": float(np.mean(difference)),
        "candidate_loss": float(np.mean(candidate)), "control_loss": float(np.mean(control)),
        "p_conservative": 1.0, "ci95_envelope": [-0.25, 0.25],
        "hac126": {"mde80_nominal": 0.001}, "block_inference": {},
        "nominal_mde_effect_ratio": 2.0,
    }


def synthetic_prior():
    return [{"study": "synthetic_prior", "candidate": str(i), "control": "baseline",
             "horizon": 1, "p_conservative": 1.0} for i in range(119)]


class CausalPoolPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = synthetic_inputs()

    def evaluate(self, panel, calendar, protocol):
        with patch.object(search, "inherited") as inherited, patch.object(
            search, "paired_inference", side_effect=null_inference
        ) as infer, patch.object(search.inference, "digest", return_value="a" * 64):
            metrics = search.evaluate(panel, calendar, protocol, prior=synthetic_prior())
        self.assertEqual(infer.call_count, 4)
        inherited.assert_not_called()
        return metrics

    def test_snapshots_roundtrip_full_filter_independent_proof_scoring_and_plot_admission(self):
        features, targets, issued, fits, protocol = deepcopy(self.source)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = dict(features="features.parquet", targets="targets.parquet",
                         forecasts="issued.parquet", fits="fits.json")
            for key, frame in zip(("features", "targets", "forecasts"),
                                  (features, targets, issued), strict=True):
                frame.to_parquet(root / names[key])
            (root / names["fits"]).write_text(json.dumps(fits))
            pins = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names.values()}
            with patch.object(search, "ROOT", root):
                f, t, old, audit = search.load_pinned_inputs({"upstream": names}, {"inputs": pins})
            applications = search.application_origins(f, t, protocol["index"])
            panel, states = model.pool_panel(old, t, f.index, audit, protocol["index"],
                                             application_origins=applications)
            panel.to_parquet(root / "panel.parquet")
            states.to_parquet(root / "states.parquet")
            panel = pd.read_parquet(root / "panel.parquet")
            states = pd.read_parquet(root / "states.parquet")
            proof = verify.verify_forecasts(f, t, old, audit, panel, states, protocol)
            metrics = self.evaluate(panel, f.index, protocol)
            metrics["common_application_origins"] = len(states)
            verified = {"status": "VERIFIED", "protocol_sha256": "a" * 64,
                        "forecast_reconstruction": proof}
            rows = plot.validated_rows(metrics, verified, "a" * 64)
            self.assertEqual(len(rows), 2)
            self.assertEqual(metrics["new_monthly_fits"], 0)
            self.assertEqual(metrics["new_forecasts"], 2 * metrics["preserved_forecasts"])
            self.assertEqual(metrics["combined_forecasts"], proof["forecasts_verified"])
            self.assertEqual(metrics["common_scored_origins"], proof["common_scored_origins"])
            self.assertEqual(metrics["leads"], [])
            self.assertEqual(len(metrics["inherited_rows"]), 119)
            self.assertEqual(metrics["cumulative_hypothesis_count"], 121)
            self.assertGreater(proof["common_application_origins"], proof["common_scored_origins"])
            self.assertEqual(states.origin.min(), pd.Timestamp("2016-01-04"))
            self.assertEqual(panel.origin.min(), pd.Timestamp("2016-02-01"))
            january = states.origin.dt.to_period("M").eq(pd.Period("2016-01"))
            self.assertTrue(january.any())
            self.assertFalse(states.loc[january, "scored"].any())
            self.assertTrue(states.seed_fit_origin.eq(pd.Timestamp("2016-01-04")).all())
            self.assertEqual(states.iloc[0].elapsed_sessions, 0)
            self.assertGreater(proof["full_calendar_states_reconstructed"], len(states))
            self.assertEqual(pins, {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                    for name in names.values()})

    def test_boundary_label_is_unscored_but_arrives_on_full_clock_without_reset(self):
        features, targets, issued, fits, protocol = deepcopy(self.source)
        applications = search.application_origins(features, targets, protocol["index"])
        panel, states = model.pool_panel(issued, targets, features.index, fits, protocol["index"],
                                         application_origins=applications)
        proof = verify.verify_forecasts(features, targets, issued, fits, panel, states, protocol)
        by_origin = states.set_index("origin")
        self.assertNotIn(pd.Timestamp("2019-12-31"), set(panel.origin))
        self.assertFalse(by_origin.loc["2019-12-31", "scored"])
        self.assertEqual(targets.loc["2019-12-31", "available_date"], pd.Timestamp("2020-01-01"))
        # This synthetic full calendar deliberately includes a civil holiday:
        # only its given reference order controls maturity, never a holiday guess.
        next_state = by_origin.loc["2020-01-02"]
        previous_state = by_origin.loc["2019-12-31"]
        self.assertEqual(next_state.latest_consumed_available, pd.Timestamp("2020-01-01"))
        self.assertEqual(next_state.cumulative_updates - previous_state.cumulative_updates, 2)
        self.assertEqual(next_state.elapsed_sessions - previous_state.elapsed_sessions, 2)
        self.assertEqual(next_state.seed_fit_origin, states.iloc[0].seed_fit_origin)
        metrics = self.evaluate(panel, features.index, protocol)
        self.assertEqual(sum(phase["n"] for phase in metrics["rows"][0]["phases"]),
                         proof["common_scored_origins"])

    def test_dropped_unscored_state_blocks_independent_admission_before_inference(self):
        features, targets, issued, fits, protocol = deepcopy(self.source)
        applications = search.application_origins(features, targets, protocol["index"])
        panel, states = model.pool_panel(issued, targets, features.index, fits, protocol["index"],
                                         application_origins=applications)
        bad = states.loc[states.scored].reset_index(drop=True)
        with patch.object(search, "paired_inference") as infer, self.assertRaises((AssertionError, ValueError)):
            verify.verify_forecasts(features, targets, issued, fits, panel, bad, protocol)
            search.evaluate(panel, features.index, protocol)
        infer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
