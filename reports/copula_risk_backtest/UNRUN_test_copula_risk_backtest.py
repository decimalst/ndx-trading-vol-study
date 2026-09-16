"""Prewritten, invented-data contracts for a leveraged intraday proxy engine."""

import copy
import unittest
from decimal import Decimal

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.copula_risk_backtest import ES_COLUMNS, run_backtest, size_positions


def synthetic_inputs():
    """No source files: two phase fragments and known portfolio returns."""
    calendar = pd.bdate_range("2019-12-20", "2020-01-10")
    origins = pd.DatetimeIndex(["2019-12-23", "2019-12-24", "2020-01-02", "2020-01-03"])
    positions = calendar.get_indexer(origins)
    risks = pd.DataFrame(
        {
            "origin": origins,
            "feature_cutoff_date": calendar[positions - 1],
            "target_end": calendar[positions + 1],
            "phase": ["development", "development", "evaluation", "evaluation"],
            "portfolio_vol": [0.01, 0.02, 0.005, 0.01],
            "es_orig_gaussian": [0.02, 0.01, 0.04, 0.02],
            "es_orig_t8": [0.04, 0.02, 0.02, 0.01],
            "es_cal_gaussian": [0.01, 0.04, 0.02, 0.01],
            "es_cal_t8": [0.03, 0.05, 0.01, 0.02],
            "es_shape_gaussian": [0.05, 0.01, 0.02, 0.02],
            "es_shape_t8": [0.005, 0.04, 0.01, 0.005],
        }
    )
    realized = risks.loc[:, ["origin", "target_end"]].copy()
    realized["y_qqq"] = np.log1p([0.10, -0.10, 0.04, -0.08])
    realized["y_spx"] = np.log1p([0.02, -0.02, 0.02, -0.02])
    return risks, realized, calendar


def account(produced, strategy="fixed_1x", cost=5.0, phase="development"):
    p = produced["path"]
    return p.loc[(p.strategy == strategy) & (p.cost_bps == cost) & (p.phase == phase)]


class CopulaRiskBacktestTests(unittest.TestCase):
    def test_closed_form_sizing_cap_and_exact_zero_risk(self):
        risks, _, calendar = synthetic_inputs()
        risks.loc[0, "es_orig_t8"] = 0.0
        risks.loc[1, "portfolio_vol"] = 0.0
        positions = size_positions(risks, calendar)
        self.assertEqual(len(positions), 36)
        self.assertEqual(positions.strategy.nunique(), 9)
        for row in positions.itertuples():
            source = risks.loc[risks.origin == row.origin].iloc[0]
            if row.strategy == "fixed_1x":
                expected = 1.0
            elif row.strategy == "fixed_2x":
                expected = 2.0
            else:
                risk = source.portfolio_vol if row.strategy == "vol_target" else source[row.strategy]
                budget = 0.1 / np.sqrt(252) if row.strategy == "vol_target" else 0.01
                expected = 2.0 if risk == 0 else min(2.0, budget / risk)
            self.assertAlmostEqual(row.gross_exposure, expected, places=14)
            self.assertEqual(row.weight_qqq, row.gross_exposure / 2)
            self.assertEqual(row.weight_spx, row.gross_exposure / 2)
        self.assertTrue(np.isfinite(positions.gross_exposure).all())

    def test_log_to_simple_marked_exit_cost_decimal_oracle(self):
        risks, realized, calendar = synthetic_inputs()
        out = run_backtest(risks, realized, calendar)
        for cost in [0, 5, 10]:
            row = account(out, "fixed_2x", cost).iloc[0]
            g, r, c = Decimal(2), Decimal(".06"), Decimal(cost) / Decimal(10000)
            entry, exit_cost = g * c, g * (1 + r) * c
            expected = g * r - entry - exit_cost
            self.assertAlmostEqual(row.basket_return, float(r), places=14)
            self.assertAlmostEqual(row.entry_cost_return, float(entry), places=14)
            self.assertAlmostEqual(row.exit_cost_return, float(exit_cost), places=14)
            self.assertAlmostEqual(row.net_return, float(expected), places=14)
            self.assertAlmostEqual(row.equity, float(1 + expected), places=14)
        # This is not a weighted log return or an approximation to exit turnover.
        self.assertNotAlmostEqual(account(out).iloc[0].basket_return, realized.iloc[0][["y_qqq", "y_spx"]].mean())

    def test_flat_each_session_pays_both_legs_and_costs_are_monotone(self):
        risks, realized, calendar = synthetic_inputs()
        realized.loc[:, ["y_qqq", "y_spx"]] = 0.0
        out = run_backtest(risks, realized, calendar)
        for cost in [0, 5, 10]:
            rows = account(out, "fixed_2x", cost)
            np.testing.assert_allclose(rows.entry_cost_return, 2 * cost / 10000)
            np.testing.assert_allclose(rows.exit_cost_return, 2 * cost / 10000)
            self.assertAlmostEqual(rows.iloc[-1].equity, (1 - 4 * cost / 10000) ** 2)
        self.assertGreater(account(out, cost=0).iloc[-1].equity, account(out, cost=5).iloc[-1].equity)
        self.assertGreater(account(out, cost=5).iloc[-1].equity, account(out, cost=10).iloc[-1].equity)

    def test_compounding_initial_peak_and_phase_reset(self):
        risks, realized, calendar = synthetic_inputs()
        out = run_backtest(risks, realized, calendar)
        dev = account(out, cost=0)
        eva = account(out, cost=0, phase="evaluation")
        np.testing.assert_allclose(dev.equity, [1.06, 1.06 * 0.94])
        np.testing.assert_allclose(dev.drawdown, [0.0, -0.06], atol=1e-14)
        self.assertEqual(eva.iloc[0].starting_equity, 1.0)
        self.assertAlmostEqual(eva.iloc[-1].equity, 1.03 * 0.95)
        summary = out["summary"]
        self.assertEqual(len(summary), 54)
        row = summary.loc[(summary.phase == "development") & (summary.strategy == "fixed_1x") & (summary.cost_bps == 0)].iloc[0]
        self.assertEqual(row.sessions, 2)
        self.assertAlmostEqual(row.total_return, 1.06 * 0.94 - 1)
        self.assertAlmostEqual(row.max_drawdown, -0.06)

    def test_attribution_identity_for_more_and_less_than_one_times(self):
        risks, realized, calendar = synthetic_inputs()
        p = run_backtest(risks, realized, calendar)["path"]
        identity = p.saved_loss_return - p.missed_upside_return + p.extra_upside_return - p.extra_loss_return + p.cost_saving_return
        np.testing.assert_allclose(identity, p.relative_net_return, atol=1e-14)
        for name in ["saved_loss_return", "missed_upside_return", "extra_upside_return", "extra_loss_return"]:
            self.assertTrue((p[name] >= 0).all())
        one = p.loc[p.strategy == "fixed_1x"]
        np.testing.assert_array_equal(one.relative_net_return, 0.0)

    def test_target_blind_positions_immutability_and_underestimated_risk_stress(self):
        risks, realized, calendar = synthetic_inputs()
        before_risks, before_realized = risks.copy(deep=True), realized.copy(deep=True)
        a = run_backtest(risks, realized, calendar)
        changed = realized.copy()
        changed["y_qqq"] *= -2
        changed["y_spx"] *= -2
        b = run_backtest(risks, changed, calendar)
        assert_frame_equal(a["positions"], b["positions"])
        assert_frame_equal(risks, before_risks)
        assert_frame_equal(realized, before_realized)
        understated = risks.copy()
        understated.loc[:, list(ES_COLUMNS)] /= 2
        stress = size_positions(understated, calendar)
        baseline = size_positions(risks, calendar)
        self.assertTrue((stress.gross_exposure >= baseline.gross_exposure).all())
        self.assertLessEqual(stress.gross_exposure.max(), 2.0)

    def test_ruin_preserves_negative_equity_and_absorbs_later_rows(self):
        risks, realized, calendar = synthetic_inputs()
        realized.loc[0, ["y_qqq", "y_spx"]] = np.log(0.4)
        realized.loc[1, ["y_qqq", "y_spx"]] = np.log(2.0)
        out = run_backtest(risks, realized, calendar)
        rows = account(out, "fixed_2x", 0)
        self.assertAlmostEqual(rows.iloc[0].equity, -0.2)
        self.assertTrue(rows.iloc[0].ruined)
        self.assertEqual(rows.iloc[1].executed_gross, 0.0)
        self.assertEqual(rows.iloc[1].net_pnl, 0.0)
        self.assertAlmostEqual(rows.iloc[1].equity, -0.2)
        self.assertEqual(rows.iloc[1].status, "inactive_after_ruin")
        self.assertAlmostEqual(rows.iloc[0].drawdown, -1.2)
        self.assertEqual(account(out, "fixed_2x", 0, "evaluation").iloc[0].starting_equity, 1.0)
        self.assertTrue((out["positions"].loc[out["positions"].strategy == "fixed_2x", "gross_exposure"] == 2).all())

    def test_exact_calendar_clocks_and_complete_return_join(self):
        risks, realized, calendar = synthetic_inputs()
        bad = risks.copy()
        bad.loc[0, "feature_cutoff_date"] = bad.loc[0, "origin"]
        with self.assertRaises(ValueError):
            size_positions(bad, calendar)
        bad = risks.copy()
        bad.loc[0, "target_end"] = calendar[0]
        with self.assertRaises(ValueError):
            size_positions(bad, calendar)
        for outcome in [realized.iloc[:-1], pd.concat([realized, realized.iloc[[0]]]), realized.iloc[::-1]]:
            with self.subTest(outcome=len(outcome)), self.assertRaises(ValueError):
                run_backtest(risks, outcome, calendar)
        bad = realized.copy()
        bad.loc[0, "target_end"] = bad.loc[0, "origin"]
        with self.assertRaises(ValueError):
            run_backtest(risks, bad, calendar)
        with self.assertRaises(ValueError):
            size_positions(risks, calendar.delete(1))

    def test_invalid_numeric_values_and_phase_or_instrument_aliases_fail(self):
        risks, realized, calendar = synthetic_inputs()
        for value in [np.nan, np.inf, -1.0, True, "0.03"]:
            bad = risks.copy()
            bad["es_orig_t8"] = bad["es_orig_t8"].astype(object)
            bad.loc[0, "es_orig_t8"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                size_positions(bad, calendar)
        for value in [np.nan, np.inf, 1000.0, -1000.0]:
            bad = realized.copy()
            bad.loc[0, "y_qqq"] = value
            with self.subTest(return_value=value), self.assertRaises(ValueError):
                run_backtest(risks, bad, calendar)
        with self.assertRaises(ValueError):
            run_backtest(risks, realized.rename(columns={"y_spx": "y_spy"}), calendar)
        bad = risks.copy()
        bad.loc[0, "phase"] = "evaluation"
        with self.assertRaises(ValueError):
            size_positions(bad, calendar)
        with self.assertRaises(ValueError):
            size_positions(risks.drop(columns="es_shape_t8"), calendar)

    def test_schema_native_dates_duplicate_columns_and_config_validation(self):
        risks, realized, calendar = synthetic_inputs()
        for bad in [risks.iloc[::-1], pd.concat([risks, risks.iloc[[0]]]), pd.concat([risks, risks[["portfolio_vol"]]], axis=1)]:
            with self.assertRaises(ValueError):
                size_positions(bad, calendar)
        bad = risks.copy()
        bad["origin"] = bad.origin.astype(str)
        with self.assertRaises(ValueError):
            size_positions(bad, calendar)
        for config in [{"unknown": 1}, {"source_end": "2025-11-03"}, {"cost_bps": [5, 0]}, {"es_budget": True}, {"gross_cap": 3}]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                run_backtest(risks, realized, calendar, config=config)
        config = {"es_budget": 0.02}
        before = copy.deepcopy(config)
        run_backtest(risks, realized, calendar, config=config)
        self.assertEqual(config, before)


if __name__ == "__main__":
    unittest.main()
