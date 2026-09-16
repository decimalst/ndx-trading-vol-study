"""Prewritten synthetic contracts for an independent hindsight-geometry audit.

The reference liabilities below are hand specified, and no producing search or
earlier accounting module is imported by this test module.
"""
from __future__ import annotations

from itertools import product
import unittest

import numpy as np
import pandas as pd

from src.verify_spread_geometry_hindsight import verify_search


POLICIES = ("always_sell",) + tuple("fixed_2pct_" + name for name in (
    "orig_gaussian", "orig_t8", "cal_gaussian", "cal_t8", "shape_gaussian", "shape_t8"))
STRUCTURES = ("put", "call", "condor")
GROUP = ["phase", "structure", "policy", "scenario"]
TIE_FIELDS = ("tie_distance_min_bps", "tie_distance_max_bps",
              "tie_width_min_bps", "tie_width_max_bps")


def fixture(quiet=False, distances=(100,), widths=(100,)):
    origins = pd.to_datetime(["2019-12-26", "2019-12-27", "2020-01-02", "2020-01-03"])
    returns = pd.DataFrame({
        "origin": origins, "target_end": origins + pd.Timedelta(days=1),
        "phase": ["development", "development", "evaluation", "evaluation"],
        "y_qqq": np.log([1., .98, .97, 1.]), "y_spx": np.log([1., 1.02, 1., 1.]),
    })
    if quiet:
        returns[["y_qqq", "y_spx"]] = 0.
    masks = {
        policy: np.ones(4, bool) if policy in POLICIES[:2] else
        np.array([True, False, True, False]) if policy == POLICIES[2] else np.zeros(4, bool)
        for policy in POLICIES
    }
    selections = pd.DataFrame([
        {"origin": origin, "structure": structure, "policy": policy, "sell": bool(masks[policy][i])}
        for i, origin in enumerate(origins) for structure, policy in product(STRUCTURES, POLICIES)
    ])
    geometry, accounts = [], []
    # For 1%/1% wings the second session fully loses one put and one call;
    # the third session fully loses one put. These are hand specified outcomes.
    known = {"put": [0., .5, .5, 0.], "call": [0., .5, 0., 0.],
             "condor": [0., 1., .5, 0.]}
    for phase, indices in (("development", np.array([0, 1])),
                           ("evaluation", np.array([2, 3])), ("pooled", np.arange(4))):
        for structure, policy, distance, width in product(STRUCTURES, POLICIES, distances, widths):
            debit = np.zeros(4) if quiet else np.array(known[structure])
            any_event = debit > 0
            both_event = debit == 1
            active = masks[policy][indices]
            taken = debit[indices][active]
            mean, maximum = (float(taken.mean()), float(taken.max())) if len(taken) else (np.nan, np.nan)
            key = dict(phase=phase, structure=structure, policy=policy,
                       distance_bps=distance, width_bps=width)
            geometry.append({**key, "sessions": len(indices), "sold_sessions": len(taken),
                             "coverage": float(active.mean()),
                             "any_breach_count": int(any_event[indices][active].sum()),
                             "both_breach_count": int(both_event[indices][active].sum()),
                             "any_full_count": int(any_event[indices][active].sum()),
                             "both_full_count": int(both_event[indices][active].sum()),
                             "mean_debit": mean, "max_debit": maximum,
                             "mean_debit_open_bps": mean * width,
                             "average_cost_credit_fraction": mean + .02,
                             "average_cost_credit_open_bps": (mean + .02) * width})
            for scenario, credit in (("width_05", .05), ("width_10", .10),
                                      ("width_20", .20), ("open_5bp", 5 / width)):
                daily = np.where(active, .02 * (credit - .02 - debit[indices]), 0.)
                # Sequential capital reference differs from the verifier's reconstruction.
                capital, high, drawdown = 1., 1., 0.
                for ret in daily:
                    capital *= 1 + ret
                    high = max(high, capital)
                    drawdown = min(drawdown, capital / high - 1)
                accounts.append({**key, "scenario": scenario, "credit_fraction": credit,
                                 "ending_equity": capital, "total_log_growth": float(np.log1p(daily).sum()),
                                 "total_return": capital - 1, "max_drawdown": drawdown,
                                 "worst_return": float(daily.min()), "sessions": len(indices),
                                 "sold_sessions": len(taken)})
    accounts = pd.DataFrame(accounts)
    winners = []
    for _, group in accounts.groupby(GROUP, sort=False):
        if group.sold_sessions.iloc[0] == 0:
            row = group.iloc[0].to_dict()
            for field in set(row) - set(GROUP) - {"sessions", "sold_sessions"}:
                row[field] = np.nan
            row.update(status="NO_TRADES", tie_count=0)
            row.update({field: np.nan for field in TIE_FIELDS})
        else:
            tied = group[np.abs(group.total_log_growth - group.total_log_growth.max()) <= 1e-10]
            row = tied.sort_values(["distance_bps", "width_bps"]).iloc[0].to_dict()
            row.update(status="HINDSIGHT_SELECTED", tie_count=len(tied),
                       tie_distance_min_bps=tied.distance_bps.min(),
                       tie_distance_max_bps=tied.distance_bps.max(),
                       tie_width_min_bps=tied.width_bps.min(), tie_width_max_bps=tied.width_bps.max())
        winners.append(row)
    return returns, selections, {"geometry": pd.DataFrame(geometry), "accounts": accounts,
                                  "winners": pd.DataFrame(winners)}


class IndependentGeometryVerifierContracts(unittest.TestCase):
    def test_hand_worked_portfolio_intrinsic_accounts_and_phase_pooling(self):
        returns, selections, outputs = fixture()
        result = verify_search(returns, selections, outputs, [100], [100])
        self.assertEqual(result["status"], "VERIFIED")
        self.assertEqual(result["geometry_rows"], 63)
        self.assertEqual(result["account_rows"], 252)
        self.assertEqual(result["winner_rows"], 252)

    def test_all_far_geometries_remain_and_ties_choose_nearest_narrowest(self):
        returns, selections, outputs = fixture(quiet=True, distances=(100, 200), widths=(50, 100))
        result = verify_search(returns, selections, outputs, [100, 200], [50, 100])
        self.assertEqual(result["geometry_rows"], 252)
        bad = {key: frame.copy(deep=True) for key, frame in outputs.items()}
        row = bad["winners"].query("status == 'HINDSIGHT_SELECTED' and scenario == 'width_10'").index[0]
        bad["winners"].loc[row, "distance_bps"] = 200
        with self.assertRaises(ValueError):
            verify_search(returns, selections, bad, [100, 200], [50, 100])

    def test_detects_every_geometry_measure_and_account_tampering(self):
        returns, selections, outputs = fixture()
        for table in ("geometry", "accounts"):
            for column in outputs[table].select_dtypes(include=[np.number]).columns:
                bad = {key: frame.copy(deep=True) for key, frame in outputs.items()}
                bad[table].loc[0, column] += 1 if bad[table][column].dtype.kind in "iu" else .125
                with self.subTest(table=table, column=column), self.assertRaises(ValueError):
                    verify_search(returns, selections, bad, [100], [100])

    def test_no_trade_winner_cannot_masquerade_as_optimized_zero_risk(self):
        returns, selections, outputs = fixture()
        row = outputs["winners"].query("status == 'NO_TRADES'").index[0]
        for field, value in (("status", "HINDSIGHT_SELECTED"), ("width_bps", 100.),
                             ("ending_equity", 1.), ("tie_count", 1)):
            bad = {key: frame.copy(deep=True) for key, frame in outputs.items()}
            bad["winners"].loc[row, field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_search(returns, selections, bad, [100], [100])

    def test_fractional_discrete_fields_cannot_hide_inside_numeric_tolerance(self):
        returns, selections, outputs = fixture()
        for table, column in (("geometry", "sessions"), ("geometry", "any_full_count"),
                              ("accounts", "sold_sessions"), ("winners", "tie_count"),
                              ("winners", "distance_bps"), ("winners", "tie_width_max_bps")):
            bad = {key: frame.copy(deep=True) for key, frame in outputs.items()}
            bad[table][column] = bad[table][column].astype(float)
            bad[table].loc[0, column] += 1e-12
            with self.subTest(table=table, column=column), self.assertRaises(ValueError):
                verify_search(returns, selections, bad, [100], [100])

    def test_rejects_missing_duplicate_nonfinite_or_extra_output(self):
        returns, selections, outputs = fixture()
        for table in outputs:
            for mutation in ("missing", "duplicate", "extra"):
                bad = {key: frame.copy(deep=True) for key, frame in outputs.items()}
                if mutation == "missing":
                    bad[table] = bad[table].iloc[1:].copy()
                elif mutation == "duplicate":
                    bad[table] = pd.concat([bad[table], bad[table].iloc[[0]]], ignore_index=True)
                else:
                    bad[table]["silently_added"] = 0.
                with self.subTest(table=table, mutation=mutation), self.assertRaises(ValueError):
                    verify_search(returns, selections, bad, [100], [100])
        bad = {key: frame.copy(deep=True) for key, frame in outputs.items()}
        bad["accounts"].loc[0, "ending_equity"] = np.inf
        with self.assertRaises(ValueError):
            verify_search(returns, selections, bad, [100], [100])

    def test_rejects_invalid_grid_cohort_clock_and_selection(self):
        returns, selections, outputs = fixture()
        for distances, widths in (([True], [100]), ([100, 100], [100]), ([200, 100], [100]),
                                  ([100.], [100]), ([0], [100]), ([100], [5]), ([9990], [10])):
            with self.subTest(distances=distances, widths=widths), self.assertRaises(ValueError):
                verify_search(returns, selections, outputs, distances, widths)
        for field, value in (("y_qqq", np.nan), ("phase", "pooled"),
                             ("target_end", returns.origin.iloc[0])):
            bad = returns.copy(deep=True)
            bad.loc[0, field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_search(bad, selections, outputs, [100], [100])
        bad = selections.copy(deep=True)
        bad.loc[0, "sell"] = False
        with self.assertRaises(ValueError):
            verify_search(returns, bad, outputs, [100], [100])
        with self.assertRaises(ValueError):
            verify_search(returns, selections.iloc[1:], outputs, [100], [100])

    def test_extreme_finite_returns_have_bounded_intrinsic_without_overflow(self):
        returns, selections, outputs = fixture()
        returns.loc[1, ["y_qqq", "y_spx"]] = [-1000., 1000.]
        returns.loc[2, "y_qqq"] = -1000.
        with np.errstate(over="raise", invalid="raise"):
            result = verify_search(returns, selections, outputs, [100], [100])
        self.assertEqual(result["status"], "VERIFIED")

    def test_short_strike_equality_is_not_a_breach_or_intrinsic_loss(self):
        returns, selections, outputs = fixture(quiet=True)
        returns["y_qqq"] = np.log(.99)
        returns["y_spx"] = np.log(1.01)
        result = verify_search(returns, selections, outputs, [100], [100])
        self.assertEqual(result["status"], "VERIFIED")


if __name__ == "__main__":
    unittest.main()
