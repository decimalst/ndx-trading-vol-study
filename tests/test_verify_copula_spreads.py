"""Prewritten synthetic saved-accounting verification contracts."""
from __future__ import annotations

import copy
import unittest
from itertools import product

import numpy as np
import pandas as pd

from src.copula_spread_backtest import DISTANCES, MODELS, STRUCTURES, run_backtest
from src.verify_copula_spreads import verify


class IndependentSpreadAccountingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calendar = pd.bdate_range("2019-12-23", periods=15)
        origins = cls.calendar[[1, 2, 3, 9, 10, 11]]
        rows = []
        for i, origin in enumerate(origins):
            p = cls.calendar.get_loc(origin)
            for structure, distance, (j, model) in product(
                    STRUCTURES, DISTANCES, enumerate(MODELS)):
                rows.append({"origin": origin, "target_end": cls.calendar[p + 1],
                             "feature_cutoff_date": cls.calendar[p - 1],
                             "phase": "development" if origin.year == 2019 else "evaluation",
                             "model": model, "structure": structure, "distance": distance,
                             "width": .01, "p_any_breach": [0., .1, .10001, .2, .08, .05][j],
                             "mean_debit": .04, "es97_5": .2})
        cls.risks = pd.DataFrame(rows)
        cls.realized = pd.DataFrame({
            "origin": origins, "target_end": [cls.calendar[cls.calendar.get_loc(x) + 1]
                                                for x in origins],
            "y_qqq": [-.025, .005, 1000., -.05, .04, 0.],
            "y_spx": [.01, -.035, -1000., .045, -.004, 0.],
        })
        cls.backtest = run_backtest(cls.risks, cls.realized, cls.calendar)

    def test_independent_reconstruction_accepts_complete_accounting(self):
        result = verify(self.risks, self.realized, self.backtest, self.calendar)
        self.assertEqual(result["status"], "VERIFIED")
        self.assertEqual(result["origins"], 6)
        self.assertEqual(result["positions"], 6 * 9 * 7)
        self.assertEqual(result["paths"], 6 * 9 * 7 * 3)
        self.assertEqual(result["summaries"], 2 * 9 * 7 * 3)

    def test_rejects_decision_threshold_or_split_reserve_corruption(self):
        for column in ("sell", "reserve_qqq", "gross_reserve"):
            bad = copy.deepcopy(self.backtest)
            frame = bad["positions"]
            frame.loc[0, column] = False if column == "sell" else .03
            with self.subTest(column=column), self.assertRaises(ValueError):
                verify(self.risks, self.realized, bad, self.calendar)

    def test_rejects_payoff_events_and_all_path_accounting_corruption(self):
        for table, column in (("outcomes", "debit_qqq"), ("outcomes", "any_breach"),
                              ("path", "net_return"), ("path", "net_pnl"),
                              ("path", "equity"), ("path", "drawdown"),
                              ("path", "fee_return"), ("path", "avoided_liability_return"),
                              ("path", "missed_credit_return"), ("path", "relative_net_return")):
            bad = copy.deepcopy(self.backtest)
            value = bad[table].loc[0, column]
            bad[table].loc[0, column] = not bool(value) if isinstance(value, (bool, np.bool_)) else value + .01
            with self.subTest(table=table, column=column), self.assertRaises(ValueError):
                verify(self.risks, self.realized, bad, self.calendar)

    def test_rejects_summary_accounting_and_missing_cases_or_rows(self):
        for column in ("sold_sessions", "ending_equity", "max_drawdown", "total_credit_pnl",
                       "net_pnl", "avoided_liability_return_sum", "sold_mean_debit"):
            bad = copy.deepcopy(self.backtest)
            bad["summary"].loc[0, column] += 1
            with self.subTest(column=column), self.assertRaises(ValueError):
                verify(self.risks, self.realized, bad, self.calendar)
        for table in self.backtest:
            bad = copy.deepcopy(self.backtest)
            bad[table] = bad[table].iloc[1:].copy()
            with self.subTest(table=table), self.assertRaises(ValueError):
                verify(self.risks, self.realized, bad, self.calendar)

    def test_rejects_cohort_clock_scope_and_nonfinite_input_changes(self):
        bad_risk = self.risks.iloc[1:].copy()
        with self.assertRaises(ValueError):
            verify(bad_risk, self.realized, self.backtest, self.calendar)
        bad_return = self.realized.copy()
        bad_return.loc[0, "target_end"] = self.calendar[4]
        with self.assertRaises(ValueError):
            verify(self.risks, bad_return, self.backtest, self.calendar)
        bad_return = self.realized.copy()
        bad_return.loc[0, "y_qqq"] = np.nan
        with self.assertRaises(ValueError):
            verify(self.risks, bad_return, self.backtest, self.calendar)
        bad_risk = self.risks.copy()
        bad_risk.loc[0, "feature_cutoff_date"] = bad_risk.loc[0, "origin"]
        with self.assertRaises(ValueError):
            verify(bad_risk, self.realized, self.backtest, self.calendar)


if __name__ == "__main__":
    unittest.main()
