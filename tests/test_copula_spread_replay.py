"""Prewritten contracts for the separately declared probability-only replay."""

from __future__ import annotations

import tempfile
import unittest
from itertools import product
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.copula_spread_backtest import DISTANCES, MODELS, STRUCTURES, run_backtest
from src.copula_spread_replay import probability_backtest
from src.verify_copula_spreads import verify


def independent_public_accounting(risks, realized, public, calendar):
    """Check reloaded public tables with the unchanged independent accountant.

    The internal legacy field is a mathematical support bound, never an ES
    estimate. This adapter exists only for checking the old exact schema.
    """
    internal_risks = risks[
        [
            "origin",
            "target_end",
            "feature_cutoff_date",
            "phase",
            "model",
            "structure",
            "distance",
            "width",
            "p_any_breach",
            "mean_debit",
        ]
    ].copy()
    internal_risks["es97_5"] = 1.0
    internal = {
        name: public[name].copy() for name in ("positions", "path", "summary", "outcomes")
    }
    for name in ("positions", "path"):
        if "max_terminal_debit" in internal[name]:
            if not internal[name].max_terminal_debit.eq(1.0).all():
                raise ValueError("universal bounded-payoff support label changed")
            internal[name] = internal[name].drop(columns="max_terminal_debit")
        internal[name]["es97_5"] = np.where(internal[name].model == "always_sell", np.nan, 1.0)
    return verify(internal_risks, realized, internal, calendar)


class ProbabilityReplayContracts(unittest.TestCase):
    def test_duplicate_entry_and_failed_gate_precede_source_access(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from src import copula_spread_replay as replay

        with TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / replay.REPORT
            report.mkdir(parents=True)
            (report / "PREFIT.json").write_text(
                json.dumps(
                    {
                        "status": "FAIL",
                        "tests_run": 1,
                        "failures": 1,
                        "errors": 0,
                        "skipped": 0,
                    }
                )
            )
            with (
                patch.object(
                    replay,
                    "source_pins",
                    side_effect=AssertionError("source must remain unread"),
                ),
                self.assertRaises(ValueError),
            ):
                replay.freeze(root)
            (report / "started.json").write_text("{}")
            with self.assertRaises(ValueError):
                replay.run(root)

    @classmethod
    def setUpClass(cls):
        cls.calendar = pd.bdate_range("2019-12-23", periods=15)
        origins = cls.calendar[[1, 2, 9, 10]]
        rows = []
        for origin in origins:
            position = cls.calendar.get_loc(origin)
            for structure, distance, (j, model) in product(
                STRUCTURES, DISTANCES, enumerate(MODELS)
            ):
                rows.append(
                    {
                        "origin": origin,
                        "target_end": cls.calendar[position + 1],
                        "feature_cutoff_date": cls.calendar[position - 1],
                        "phase": "development" if origin.year == 2019 else "evaluation",
                        "model": model,
                        "structure": structure,
                        "distance": distance,
                        "width": 0.01,
                        "p_any_breach": [0.1, 0.10001, 0.02, 0.03, 0.2, 0.07][j],
                        "mean_debit": 0.04,
                        "es97_5": 0.2,
                        "var97_5": 0.15,
                    }
                )
        cls.risks = pd.DataFrame(rows)
        cls.realized = pd.DataFrame(
            {
                "origin": origins,
                "target_end": [cls.calendar[cls.calendar.get_loc(x) + 1] for x in origins],
                "y_qqq": [-0.04, 0.03, 0.0, -0.015],
                "y_spx": [0.02, -0.025, -0.05, 0.03],
            }
        )

    def test_preserves_probabilities_means_policy_cases_and_inputs(self):
        before = self.risks.copy(deep=True)
        realized_before = self.realized.copy(deep=True)
        result = probability_backtest(self.risks, self.realized, self.calendar)
        pd.testing.assert_frame_equal(self.risks, before)
        pd.testing.assert_frame_equal(self.realized, realized_before)
        legacy = run_backtest(self.risks.drop(columns="var97_5"), self.realized, self.calendar)
        actual = result["positions"].drop(columns="max_terminal_debit")
        pd.testing.assert_frame_equal(actual, legacy["positions"].drop(columns="es97_5"))
        self.assertEqual(len(result["outcomes"]), 4 * 9)
        self.assertEqual(len(result["path"]), 4 * 9 * 7 * 3)
        self.assertEqual(result["verification"]["status"], "VERIFIED")
        self.assertTrue(
            result["positions"].loc[result["positions"].model == "orig_gaussian", "sell"].all()
        )
        self.assertFalse(
            result["positions"].loc[result["positions"].model == "orig_t8", "sell"].any()
        )

    def test_invalid_qmc_tail_outputs_never_affect_accounts(self):
        good = probability_backtest(self.risks, self.realized, self.calendar)
        failed = self.risks.copy()
        failed["es97_5"] = np.nan
        failed["var97_5"] = np.inf
        failed["sample_mean_debit"] = -1e50
        failed["qmc_status"] = "FAILED"
        ignored = probability_backtest(failed, self.realized, self.calendar)
        absent = probability_backtest(
            self.risks.drop(columns=["es97_5", "var97_5"]), self.realized, self.calendar
        )
        for name in ("positions", "path", "summary", "outcomes"):
            pd.testing.assert_frame_equal(good[name], ignored[name])
            pd.testing.assert_frame_equal(good[name], absent[name])

    def test_public_outputs_remove_tail_estimates_and_saved_accounting_checks(self):
        result = probability_backtest(self.risks, self.realized, self.calendar)
        self.assertTrue(result["positions"].max_terminal_debit.eq(1.0).all())
        for name in ("positions", "path", "summary", "outcomes"):
            self.assertNotIn("es97_5", result[name].columns)
            self.assertNotIn("var97_5", result[name].columns)
        with tempfile.TemporaryDirectory() as directory:
            reloaded = {}
            for name in ("positions", "path", "summary", "outcomes"):
                filename = Path(directory) / f"{name}.parquet"
                result[name].to_parquet(filename, index=False)
                reloaded[name] = pd.read_parquet(filename)
            receipt = independent_public_accounting(
                self.risks, self.realized, reloaded, self.calendar
            )
            self.assertEqual(receipt["status"], "VERIFIED")
            reloaded["path"].loc[0, "net_pnl"] += 0.01
            with self.assertRaises(ValueError):
                independent_public_accounting(
                    self.risks, self.realized, reloaded, self.calendar
                )

    def test_probabilities_and_cohort_corruption_still_fail(self):
        for value in (np.nan, -0.01, 1.01):
            bad = self.risks.copy()
            bad.loc[0, "p_any_breach"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                probability_backtest(bad, self.realized, self.calendar)
        with self.assertRaises(ValueError):
            probability_backtest(self.risks.iloc[1:], self.realized, self.calendar)
        bad = self.risks.copy()
        bad.loc[0, "mean_debit"] = np.nan
        with self.assertRaises(ValueError):
            probability_backtest(bad, self.realized, self.calendar)

    def test_replay_cannot_refit_resample_or_cross_source_ceiling(self):
        with (
            patch(
                "src.copula_spread_forecasts.build_risks",
                side_effect=AssertionError("resampling"),
            ),
            patch("src.copula_shape.fit_shape", side_effect=AssertionError("refitting")),
            patch(
                "src.copula_spread_forecasts.base_draws",
                side_effect=AssertionError("new draws"),
            ),
        ):
            probability_backtest(self.risks, self.realized, self.calendar)
        future_risk, future_realized = self.risks.copy(), self.realized.copy()
        for frame in (future_risk, future_realized):
            for column in set(frame.columns) & {"origin", "target_end", "feature_cutoff_date"}:
                frame[column] = frame[column] + pd.DateOffset(years=20)
        with self.assertRaises(ValueError):
            probability_backtest(
                future_risk, future_realized, self.calendar + pd.DateOffset(years=20)
            )


if __name__ == "__main__":
    unittest.main()
