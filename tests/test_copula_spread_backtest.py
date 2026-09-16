"""Prewritten synthetic contracts; no option quotes or historical returns."""

import unittest
from decimal import Decimal

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.copula_spread_backtest import run_backtest, terminal_debit

MODELS = ("orig_gaussian", "orig_t8", "cal_gaussian", "cal_t8", "shape_gaussian", "shape_t8")


def synthetic_inputs():
    calendar = pd.bdate_range("2019-12-20", "2020-01-10")
    origins = pd.DatetimeIndex(["2019-12-23", "2019-12-24", "2020-01-02", "2020-01-03"])
    positions = calendar.get_indexer(origins)
    rows = []
    for j, origin in enumerate(origins):
        for structure in ("put", "call", "condor"):
            for distance in (0.01, 0.02, 0.03):
                for k, model in enumerate(MODELS):
                    rows.append(
                        dict(
                            origin=origin,
                            target_end=calendar[positions[j] + 1],
                            feature_cutoff_date=calendar[positions[j] - 1],
                            phase="development" if j < 2 else "evaluation",
                            model=model,
                            structure=structure,
                            distance=distance,
                            width=0.01,
                            p_any_breach=(0.1 if k == 0 else 0.09 if k == 1 else 0.11),
                            mean_debit=0.04,
                            es97_5=0.7,
                        )
                    )
    risks = pd.DataFrame(rows)
    realized = pd.DataFrame(
        {
            "origin": origins,
            "target_end": calendar[positions + 1],
            "y_qqq": np.log([0.975, 1.04, 1.0, 0.95]),
            "y_spx": np.log([1.0, 1.04, 1.0, 1.0]),
        }
    )
    return risks, realized, calendar


def account(
    out, model="always_sell", credit=0.1, phase="development", structure="put", distance=0.02
):
    p = out["path"]
    return p.loc[
        (p.model == model)
        & (p.credit_fraction == credit)
        & (p.phase == phase)
        & (p.structure == structure)
        & (p.distance == distance)
    ]


class CopulaSpreadBacktestTests(unittest.TestCase):
    def test_piecewise_debit_matches_independent_dollar_intrinsic(self):
        prices = np.array([[0.95, 0.97], [0.975, 0.98], [1.0, 1.025], [1.03, 1.05]])
        for structure in ("put", "call", "condor"):
            puts = np.maximum(0.98 - prices, 0) - np.maximum(0.97 - prices, 0)
            calls = np.maximum(prices - 1.02, 0) - np.maximum(prices - 1.03, 0)
            expected = {"put": puts, "call": calls, "condor": puts + calls}[structure] / 0.01
            np.testing.assert_allclose(
                terminal_debit(np.log(prices), structure), expected, atol=2e-14
            )

    def test_extreme_logs_safe_without_exponential_overflow(self):
        y = np.array([-np.finfo(float).max, -1000.0, 0.0, 1000.0, np.finfo(float).max])
        with np.errstate(all="raise"):
            np.testing.assert_array_equal(terminal_debit(y, "put"), [1, 1, 0, 0, 0])
            np.testing.assert_array_equal(terminal_debit(y, "call"), [0, 0, 0, 1, 1])
            np.testing.assert_array_equal(terminal_debit(y, "condor"), [1, 1, 0, 1, 1])
        self.assertEqual(terminal_debit(np.array(0.0), "condor").shape, ())

    def test_boundary_touch_and_condor_single_width_maximum(self):
        for distance in (0.01, 0.02, 0.03):
            y = np.log([1 - distance - 0.01, 1 - distance, 1 + distance, 1 + distance + 0.01])
            np.testing.assert_array_equal(terminal_debit(y, "condor", distance), [1, 0, 0, 1])
            self.assertLessEqual(
                terminal_debit(np.linspace(-10, 10, 1001), "condor", distance).max(), 1.0
            )

    def test_credit_cashflow_decimal_oracle_equal_maximum_reserve(self):
        risks, realized, calendar = synthetic_inputs()
        out = run_backtest(risks, realized, calendar)
        # QQQ half width, SPX zero => average normalized intrinsic debit .25.
        for credit in (0.05, 0.1, 0.2):
            row = account(out, credit=credit).iloc[0]
            budget, debit, fee = Decimal(".02"), Decimal(".25"), Decimal(".02")
            c = Decimal(str(credit))
            self.assertAlmostEqual(row.portfolio_debit, float(debit), places=13)
            self.assertAlmostEqual(row.credit_return, float(budget * c), places=14)
            self.assertAlmostEqual(row.liability_return, float(budget * debit), places=14)
            self.assertAlmostEqual(row.fee_return, float(budget * fee), places=14)
            self.assertAlmostEqual(
                row.net_return, float(budget * (c - fee - debit)), places=13
            )
            self.assertEqual(row.reserve_qqq, 0.01)
            self.assertEqual(row.reserve_spx, 0.01)
        self.assertEqual(len(out["positions"]), 4 * 9 * 7)
        self.assertEqual(len(out["path"]), 4 * 9 * 7 * 3)

    def test_inclusive_filter_cash_and_no_mean_or_es_sizing(self):
        risks, realized, calendar = synthetic_inputs()
        out = run_backtest(risks, realized, calendar)
        self.assertTrue(account(out, "orig_gaussian").sell.all())
        self.assertTrue(account(out, "orig_t8").sell.all())
        blocked = account(out, "shape_t8")
        self.assertFalse(blocked.sell.any())
        np.testing.assert_array_equal(blocked.net_return, 0.0)
        np.testing.assert_array_equal(blocked.equity, 1.0)
        changed = risks.copy()
        changed["mean_debit"] = 0.01
        changed["es97_5"] = 0.9
        a, b = out["positions"], run_backtest(changed, realized, calendar)["positions"]
        np.testing.assert_array_equal(a.sell, b.sell)
        np.testing.assert_array_equal(a.gross_reserve, b.gross_reserve)

    def test_target_blind_decisions_input_immutability_and_attribution(self):
        risks, realized, calendar = synthetic_inputs()
        r0, y0 = risks.copy(deep=True), realized.copy(deep=True)
        out = run_backtest(risks, realized, calendar)
        changed = realized.copy()
        changed.loc[:, ["y_qqq", "y_spx"]] *= -3
        other = run_backtest(risks, changed, calendar)
        assert_frame_equal(out["positions"], other["positions"])
        assert_frame_equal(risks, r0)
        assert_frame_equal(realized, y0)
        p = out["path"]
        np.testing.assert_allclose(
            p.avoided_liability_return - p.missed_credit_return + p.saved_fee_return,
            p.relative_net_return,
            atol=1e-14,
        )

    def test_compounding_drawdown_and_phase_resets(self):
        risks, realized, calendar = synthetic_inputs()
        out = run_backtest(risks, realized, calendar)
        dev, eva = account(out), account(out, phase="evaluation")
        first = 1 + 0.02 * (0.1 - 0.02 - 0.25)
        second = first * (1 + 0.02 * (0.1 - 0.02))
        np.testing.assert_allclose(dev.equity, [first, second], atol=1e-14)
        self.assertAlmostEqual(dev.iloc[0].drawdown, first - 1)
        self.assertEqual(eva.iloc[0].starting_equity, 1.0)
        summary = out["summary"]
        self.assertEqual(len(summary), 2 * 9 * 7 * 3)
        s = summary.loc[(summary.model == "shape_t8") & (summary.phase == "development")].iloc[
            0
        ]
        self.assertEqual(s.sold_sessions, 0)
        self.assertEqual(s.coverage, 0.0)
        self.assertTrue(pd.isna(s.sold_mean_debit))

    def test_breach_and_full_loss_are_joint_events_not_mean_debit(self):
        risks, realized, calendar = synthetic_inputs()
        outcomes = run_backtest(risks, realized, calendar)["outcomes"]
        puts = outcomes.loc[(outcomes.structure == "put") & (outcomes.distance == 0.02)]
        self.assertTrue(puts.iloc[0].any_breach)
        self.assertFalse(puts.iloc[0].both_breach)
        self.assertFalse(puts.iloc[0].any_full_loss)
        self.assertTrue(puts.iloc[3].any_full_loss)
        self.assertFalse(puts.iloc[3].both_full_loss)
        self.assertAlmostEqual(puts.iloc[3].portfolio_debit, 0.5)

    def test_strict_clocks_complete_models_cases_and_return_join(self):
        risks, realized, calendar = synthetic_inputs()
        for bad in [
            risks.iloc[1:],
            pd.concat([risks, risks.iloc[[0]]]),
            risks.drop(columns="es97_5"),
        ]:
            with self.assertRaises(ValueError):
                run_backtest(bad, realized, calendar)
        bad = risks.copy()
        bad.loc[0, "feature_cutoff_date"] = bad.loc[0, "origin"]
        with self.assertRaises(ValueError):
            run_backtest(bad, realized, calendar)
        for bad in [
            realized.iloc[:-1],
            pd.concat([realized, realized.iloc[[0]]]),
            realized.rename(columns={"y_spx": "y_spy"}),
        ]:
            with self.assertRaises(ValueError):
                run_backtest(risks, bad, calendar)
        bad = realized.copy()
        bad.loc[0, "target_end"] = bad.loc[0, "origin"]
        with self.assertRaises(ValueError):
            run_backtest(risks, bad, calendar)

    def test_invalid_payoff_parameters_risks_dates_and_no_silent_clipping(self):
        for y in [
            np.array([True]),
            np.array(["0.01"]),
            np.array([np.nan]),
            np.array([np.inf]),
        ]:
            with self.assertRaises(ValueError):
                terminal_debit(y, "put")
        for kwargs in [
            dict(structure="straddle"),
            dict(structure="put", width=0),
            dict(structure="put", distance=0.999),
            dict(structure="put", width=True),
        ]:
            with self.assertRaises(ValueError):
                terminal_debit(np.array([0.0]), **kwargs)
        risks, realized, calendar = synthetic_inputs()
        for value in [-0.01, 1.01, np.nan, np.inf]:
            bad = risks.copy()
            bad.loc[0, "p_any_breach"] = value
            with self.assertRaises(ValueError):
                run_backtest(bad, realized, calendar)
        bad = risks.copy()
        bad["origin"] = bad.origin.astype(str)
        with self.assertRaises(ValueError):
            run_backtest(bad, realized, calendar)
        with self.assertRaises(ValueError):
            run_backtest(risks, realized, calendar, config={"source_end": "2025-11-03"})


if __name__ == "__main__":
    unittest.main()
