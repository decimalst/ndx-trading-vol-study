"""Prewritten synthetic checks for an explicitly hindsight-only geometry search.

No empirical returns, fitted forecasts, option premiums or external data are
read. Intrinsic payoffs and account paths are reconstructed independently.
"""

import unittest
from decimal import Decimal, localcontext

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.spread_geometry_hindsight import (
    DEFAULT_DISTANCE_BPS,
    DEFAULT_WIDTH_BPS,
    POLICIES,
    STRUCTURES,
    run_search,
    select_winners,
)


MODELS = ("orig_gaussian", "orig_t8", "cal_gaussian", "cal_t8", "shape_gaussian", "shape_t8")
EXPECTED_POLICIES = ("always_sell",) + tuple("fixed_2pct_" + model for model in MODELS)
DISTANCES = (100, 200, 300)
WIDTHS = (25, 100, 200)
SCENARIOS = ("width_05", "width_10", "width_20", "open_5bp")
KEYS = ["phase", "structure", "policy", "distance_bps", "width_bps"]


def synthetic_inputs():
    origins = pd.DatetimeIndex([
        "2019-12-23", "2019-12-24", "2019-12-26", "2019-12-27",
        "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07",
    ])
    target_end = pd.DatetimeIndex([
        "2019-12-24", "2019-12-26", "2019-12-27", "2019-12-30",
        "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08",
    ])
    prices = np.array([
        [.975, 1.0], [1.035, 1.045], [.95, .96], [1.0, 1.0],
        [1.02, .98], [1.03, .97], [1.025, .975], [1.1, .90],
    ])
    returns = pd.DataFrame({
        "origin": origins,
        "target_end": target_end,
        "phase": ["development"] * 4 + ["evaluation"] * 4,
        "y_qqq": np.log(prices[:, 0]),
        "y_spx": np.log(prices[:, 1]),
    })
    rows = []
    for j, origin in enumerate(origins):
        for k, structure in enumerate(("put", "call", "condor")):
            masks = [True, j % 2 == 0, j % 3 != 0, False, True, j % 2 == 1, (j + k) % 3 != 1]
            for policy, sell in zip(EXPECTED_POLICIES, masks):
                rows.append(dict(origin=origin, structure=structure, policy=policy, sell=bool(sell)))
    return returns, pd.DataFrame(rows)


def oracle(returns, selections, phase, structure, policy, distance_bps, width_bps):
    chosen = returns if phase == "pooled" else returns.loc[returns.phase == phase]
    chosen = chosen.sort_values("origin")
    flags = selections.loc[(selections.structure == structure) & (selections.policy == policy)]
    mask = chosen.origin.map(flags.set_index("origin").sell).to_numpy(dtype=bool)
    y = chosen[["y_qqq", "y_spx"]].to_numpy()
    price = np.exp(y)
    distance, width = distance_bps / 10000, width_bps / 10000
    put = np.maximum(1 - distance - price, 0) - np.maximum(1 - distance - width - price, 0)
    call = np.maximum(price - (1 + distance), 0) - np.maximum(price - (1 + distance + width), 0)
    liability = {"put": put, "call": call, "condor": put + call}[structure] / width
    # Compare original log prices at exact boundaries, avoiding exp roundoff.
    low = y < np.log(1 - distance)
    high = y > np.log(1 + distance)
    low_full = y <= np.log(1 - distance - width)
    high_full = y >= np.log(1 + distance + width)
    breaches = {"put": low, "call": high, "condor": low | high}[structure]
    full = {"put": low_full, "call": high_full, "condor": low_full | high_full}[structure]
    return mask, liability.mean(axis=1), breaches, full


def pick(frame, **values):
    mask = np.ones(len(frame), dtype=bool)
    for name, value in values.items():
        mask &= frame[name].eq(value).to_numpy()
    rows = frame.loc[mask]
    if len(rows) != 1:
        raise AssertionError(f"Expected one row, found {len(rows)} for {values}")
    return rows.iloc[0]


class SpreadGeometryHindsightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.returns, cls.selections = synthetic_inputs()
        cls.out = run_search(cls.returns, cls.selections, DISTANCES, WIDTHS)

    def test_defaults_and_complete_crossed_grid(self):
        self.assertEqual(tuple(POLICIES), EXPECTED_POLICIES)
        self.assertEqual(tuple(STRUCTURES), ("put", "call", "condor"))
        self.assertEqual(tuple(DEFAULT_DISTANCE_BPS), tuple(range(25, 501, 25)))
        self.assertEqual(tuple(DEFAULT_WIDTH_BPS), (10, 25, 50, 75, 100, 150, 200, 300))
        self.assertEqual(set(self.out), {"geometry", "accounts", "winners"})
        cases = 3 * 3 * 7 * len(DISTANCES) * len(WIDTHS)
        self.assertEqual(len(self.out["geometry"]), cases)
        self.assertEqual(len(self.out["accounts"]), cases * 4)
        self.assertEqual(len(self.out["winners"]), 3 * 3 * 7 * 4)
        self.assertFalse(self.out["geometry"].duplicated(KEYS).any())
        self.assertFalse(self.out["accounts"].duplicated(KEYS + ["scenario"]).any())
        self.assertEqual(set(self.out["accounts"].scenario), set(SCENARIOS))

    def test_every_geometry_matches_independent_intrinsic_and_event_oracle(self):
        required = set(KEYS + ["sessions", "sold_sessions", "coverage", "any_breach_count", "both_breach_count", "any_full_count", "both_full_count", "mean_debit", "max_debit", "mean_debit_open_bps", "average_cost_credit_fraction", "average_cost_credit_open_bps"])
        self.assertTrue(required <= set(self.out["geometry"].columns))
        for row in self.out["geometry"].itertuples(index=False):
            with self.subTest(phase=row.phase, structure=row.structure, policy=row.policy, d=row.distance_bps, w=row.width_bps):
                mask, debit, breach, full = oracle(self.returns, self.selections, row.phase, row.structure, row.policy, row.distance_bps, row.width_bps)
                self.assertEqual(row.sessions, len(mask))
                self.assertEqual(row.sold_sessions, int(mask.sum()))
                self.assertAlmostEqual(row.coverage, mask.mean(), places=15)
                self.assertEqual(row.any_breach_count, int(breach[mask].any(axis=1).sum()))
                self.assertEqual(row.both_breach_count, int(breach[mask].all(axis=1).sum()))
                self.assertEqual(row.any_full_count, int(full[mask].any(axis=1).sum()))
                self.assertEqual(row.both_full_count, int(full[mask].all(axis=1).sum()))
                if mask.any():
                    self.assertAlmostEqual(row.mean_debit, debit[mask].mean(), places=11)
                    self.assertAlmostEqual(row.max_debit, debit[mask].max(), places=11)
                    self.assertAlmostEqual(row.mean_debit_open_bps, debit[mask].mean() * row.width_bps, places=9)
                    self.assertAlmostEqual(row.average_cost_credit_fraction, debit[mask].mean() + .02, places=11)
                    self.assertAlmostEqual(row.average_cost_credit_open_bps, (debit[mask].mean() + .02) * row.width_bps, places=9)
                else:
                    for field in ("mean_debit", "max_debit", "mean_debit_open_bps", "average_cost_credit_fraction", "average_cost_credit_open_bps"):
                        self.assertTrue(pd.isna(getattr(row, field)))

    def test_exact_short_and_long_touches_and_single_width_condor(self):
        returns, selections = synthetic_inputs()
        # Each phase includes exact short touches and exact long touches.
        returns["y_qqq"] = np.log([.98, .97, 1.02, 1.03] * 2)
        returns["y_spx"] = np.log([1.02, 1.03, .98, .97] * 2)
        out = run_search(returns, selections, (200,), (100,))["geometry"]
        condor = pick(out, phase="development", structure="condor", policy="always_sell")
        self.assertEqual(condor.any_breach_count, 2)
        self.assertEqual(condor.both_breach_count, 2)
        self.assertEqual(condor.any_full_count, 2)
        self.assertEqual(condor.both_full_count, 2)
        self.assertAlmostEqual(condor.mean_debit, .5, places=13)
        self.assertEqual(condor.max_debit, 1.0)
        for structure in ("put", "call"):
            row = pick(out, phase="development", structure=structure, policy="always_sell")
            self.assertEqual(row.any_breach_count, 2)
            self.assertEqual(row.both_breach_count, 0)
            self.assertAlmostEqual(row.max_debit, .5, places=13)

    def test_extreme_finite_log_returns_cannot_overflow_or_exceed_liability(self):
        returns, selections = synthetic_inputs()
        returns["y_qqq"] = [-1e300, 1e300] * 4
        returns["y_spx"] = [1e300, -1e300] * 4
        with np.errstate(all="raise"):
            out = run_search(returns, selections, (200,), (100,))
        row = pick(out["geometry"], phase="pooled", structure="condor", policy="always_sell")
        self.assertEqual(row.mean_debit, 1.0)
        self.assertEqual(row.max_debit, 1.0)
        self.assertEqual(row.both_full_count, 8)
        self.assertTrue(np.isfinite(out["accounts"].ending_equity).all())

    def test_masks_are_fixed_across_geometry_and_independent_of_targets(self):
        before_returns = self.returns.copy(deep=True)
        before_selections = self.selections.copy(deep=True)
        changed = self.returns.copy()
        changed[["y_qqq", "y_spx"]] *= -4
        other = run_search(changed, self.selections, DISTANCES, WIDTHS)
        fields = KEYS + ["sessions", "sold_sessions", "coverage"]
        assert_frame_equal(self.out["geometry"][fields], other["geometry"][fields])
        for _, group in self.out["geometry"].groupby(["phase", "structure", "policy"]):
            self.assertEqual(group.sold_sessions.nunique(), 1)
            self.assertEqual(group.coverage.nunique(), 1)
        assert_frame_equal(self.returns, before_returns)
        assert_frame_equal(self.selections, before_selections)

    def test_width_does_not_change_breaches_and_liability_monotonicity(self):
        for _, group in self.out["geometry"].groupby(["phase", "structure", "policy", "distance_bps"]):
            group = group.sort_values("width_bps")
            self.assertEqual(group.any_breach_count.nunique(), 1)
            self.assertEqual(group.both_breach_count.nunique(), 1)
            if group.sold_sessions.iloc[0]:
                self.assertTrue((np.diff(group.mean_debit) <= 1e-12).all())
                self.assertTrue((np.diff(group.mean_debit_open_bps) >= -1e-10).all())
                self.assertTrue((np.diff(group.any_full_count) <= 0).all())
        for _, group in self.out["geometry"].groupby(["phase", "structure", "policy", "width_bps"]):
            group = group.sort_values("distance_bps")
            self.assertTrue((np.diff(group.any_breach_count) <= 0).all())
            if group.sold_sessions.iloc[0]:
                self.assertTrue((np.diff(group.mean_debit) <= 1e-12).all())

    def test_decimal_compounding_equal_reserve_credit_conversion_and_drawdown(self):
        required = set(KEYS + ["scenario", "credit_fraction", "ending_equity", "total_log_growth", "total_return", "max_drawdown", "worst_return", "sessions", "sold_sessions"])
        self.assertTrue(required <= set(self.out["accounts"].columns))
        # These synthetic .975 prices give quarter-width portfolio liability
        # with 100-bp wings, and exactly bounded outcomes for other days.
        for policy in ("always_sell", "fixed_2pct_orig_gaussian", "fixed_2pct_cal_gaussian"):
            for phase in ("development", "evaluation", "pooled"):
                for width in WIDTHS:
                    mask, debit, _, _ = oracle(self.returns, self.selections, phase, "put", policy, 200, width)
                    for scenario in SCENARIOS:
                        credit = {"width_05": Decimal(".05"), "width_10": Decimal(".10"), "width_20": Decimal(".20"), "open_5bp": Decimal(5) / Decimal(width)}[scenario]
                        row = pick(self.out["accounts"], phase=phase, structure="put", policy=policy, distance_bps=200, width_bps=width, scenario=scenario)
                        with localcontext() as ctx:
                            ctx.prec = 40
                            equity = Decimal(1)
                            peak = Decimal(1)
                            dd = Decimal(0)
                            net_returns = []
                            for sell, liability in zip(mask, debit):
                                # Round independently computed intrinsic to
                                # remove binary noise at known rational cases.
                                loss = Decimal(str(round(float(liability), 12)))
                                net = Decimal(".02") * (credit - Decimal(".02") - loss) if sell else Decimal(0)
                                net_returns.append(float(net))
                                equity *= 1 + net
                                peak = max(peak, equity)
                                dd = min(dd, equity / peak - 1)
                            self.assertAlmostEqual(row.credit_fraction, float(credit), places=14)
                            self.assertAlmostEqual(row.ending_equity, float(equity), places=11)
                            self.assertAlmostEqual(row.total_return, float(equity - 1), places=11)
                            self.assertAlmostEqual(row.total_log_growth, float(equity.ln()), places=11)
                            self.assertAlmostEqual(row.max_drawdown, float(dd), places=11)
                            self.assertAlmostEqual(row.worst_return, min(net_returns), places=11)
                        self.assertEqual(row.sessions, len(mask))
                        self.assertEqual(row.sold_sessions, mask.sum())
        for scenario in SCENARIOS:
            values = {phase: pick(self.out["accounts"], phase=phase, structure="put", policy="always_sell", distance_bps=200, width_bps=100, scenario=scenario).ending_equity for phase in ("development", "evaluation", "pooled")}
            self.assertAlmostEqual(values["pooled"], values["development"] * values["evaluation"], places=13)

    def test_no_trade_accounts_flat_and_winners_retain_no_trade_groups(self):
        accounts = self.out["accounts"].loc[self.out["accounts"].policy == "fixed_2pct_cal_gaussian"]
        self.assertTrue(accounts.ending_equity.eq(1).all())
        for column in ("total_log_growth", "total_return", "max_drawdown", "worst_return", "sold_sessions"):
            self.assertTrue(accounts[column].eq(0).all())
        winners = self.out["winners"].loc[self.out["winners"].policy == "fixed_2pct_cal_gaussian"]
        self.assertEqual(len(winners), 3 * 3 * 4)
        self.assertTrue(winners.status.eq("NO_TRADES").all())
        self.assertTrue(winners.tie_count.eq(0).all())
        for field in ("distance_bps", "width_bps", "credit_fraction", "ending_equity", "total_log_growth", "total_return", "max_drawdown", "worst_return", "tie_distance_min_bps", "tie_distance_max_bps", "tie_width_min_bps", "tie_width_max_bps"):
            self.assertTrue(winners[field].isna().all())

    def test_winner_ties_are_against_maximum_and_choose_closest_then_narrowest(self):
        accounts = self.out["accounts"].copy(deep=True)
        mask = (accounts.phase == "development") & (accounts.structure == "put") & (accounts.policy == "always_sell") & (accounts.scenario == "width_10")
        selected = accounts.loc[mask].copy()
        selected["total_log_growth"] = -1.0
        # Closest distance only counts when within tolerance of the maximum.
        # 300/25 is maximal, 100/100 and 100/200 tie within tolerance.
        values = {(100, 25): 1 - 1.5e-10, (100, 100): 1 - .9e-10, (100, 200): 1 - .5e-10, (300, 25): 1.0}
        for (distance, width), value in values.items():
            selected.loc[(selected.distance_bps == distance) & (selected.width_bps == width), "total_log_growth"] = value
        frozen = selected.copy(deep=True)
        winners = select_winners(selected)
        self.assertEqual(len(winners), 1)
        row = winners.iloc[0]
        self.assertEqual(row.status, "HINDSIGHT_SELECTED")
        self.assertEqual(row.distance_bps, 100)
        self.assertEqual(row.width_bps, 100)
        self.assertEqual(row.tie_count, 3)
        self.assertEqual(row.tie_distance_min_bps, 100)
        self.assertEqual(row.tie_distance_max_bps, 300)
        self.assertEqual(row.tie_width_min_bps, 25)
        self.assertEqual(row.tie_width_max_bps, 200)
        assert_frame_equal(selected, frozen)
        assert_frame_equal(winners, select_winners(selected.sample(frac=1, random_state=17)))

    def test_generated_winners_reconstruct_and_exclude_zero_trade_candidates(self):
        assert_frame_equal(self.out["winners"], select_winners(self.out["accounts"]))
        for _, group in self.out["accounts"].groupby(["phase", "structure", "policy", "scenario"], sort=False):
            first = group.iloc[0]
            row = pick(self.out["winners"], phase=first.phase, structure=first.structure, policy=first.policy, scenario=first.scenario)
            eligible = group.loc[group.sold_sessions > 0]
            if eligible.empty:
                continue
            best = eligible.total_log_growth.max()
            tied = eligible.loc[(eligible.total_log_growth - best).abs() <= 1e-10].sort_values(["distance_bps", "width_bps"])
            self.assertEqual(row.distance_bps, tied.iloc[0].distance_bps)
            self.assertEqual(row.width_bps, tied.iloc[0].width_bps)
            self.assertEqual(row.tie_count, len(tied))

    def test_strict_return_schema_dates_phase_and_numeric_validation(self):
        bad_cases = [
            self.returns.drop(columns="y_spx"),
            self.returns.assign(extra=0),
            self.returns.iloc[::-1],
            pd.concat([self.returns, self.returns.iloc[[0]]], ignore_index=True),
            self.returns.iloc[:0],
            self.returns.loc[self.returns.phase == "development"],
        ]
        for column, value in [("y_qqq", np.nan), ("y_spx", np.inf), ("y_spx", True), ("y_qqq", "0.1"), ("phase", "pooled"), ("phase", "future")]:
            bad = self.returns.copy()
            bad[column] = bad[column].astype(object)
            bad.loc[0, column] = value
            bad_cases.append(bad)
        for column in ("origin", "target_end"):
            bad = self.returns.copy()
            bad[column] = bad[column].dt.strftime("%Y-%m-%d")
            bad_cases.append(bad)
            bad = self.returns.copy()
            bad[column] = bad[column].dt.tz_localize("UTC")
            bad_cases.append(bad)
            bad = self.returns.copy()
            bad.loc[0, column] += pd.Timedelta(hours=1)
            bad_cases.append(bad)
            bad = self.returns.copy()
            bad.loc[0, column] = pd.NaT
            bad_cases.append(bad)
        bad = self.returns.copy()
        bad.loc[0, "target_end"] = bad.loc[0, "origin"]
        bad_cases.append(bad)
        bad = self.returns.copy()
        bad.loc[7, "target_end"] = pd.Timestamp("2025-10-21")
        bad_cases.append(bad)
        for index, bad in enumerate(bad_cases):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    run_search(bad, self.selections, (200,), (100,))

    def test_strict_complete_selections_and_actual_boolean_masks(self):
        bad_cases = [self.selections.iloc[1:], pd.concat([self.selections, self.selections.iloc[[0]]]), self.selections.assign(extra=0), self.selections.drop(columns="sell")]
        for column, value in [("policy", "future_model"), ("structure", "straddle"), ("sell", 1), ("sell", "True"), ("sell", None)]:
            bad = self.selections.copy()
            bad[column] = bad[column].astype(object)
            bad.loc[0, column] = value
            bad_cases.append(bad)
        bad = self.selections.copy()
        bad.loc[0, "sell"] = False
        bad_cases.append(bad)
        bad = self.selections.copy()
        bad["origin"] = bad.origin.dt.strftime("%Y-%m-%d")
        bad_cases.append(bad)
        bad = self.selections.copy()
        bad.loc[0, "origin"] = pd.Timestamp("2019-12-20")
        bad_cases.append(bad)
        for index, bad in enumerate(bad_cases):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    run_search(self.returns, bad, (200,), (100,))

    def test_grid_requires_finite_unique_positive_integer_basis_points(self):
        for grid in [(), (0,), (-1,), (True,), (100.5,), (np.nan,), (np.inf,), ("100",), (100, 100), (200, 100)]:
            for parameter in ("distance_bps", "width_bps"):
                arguments = dict(distance_bps=(200,), width_bps=(100,))
                arguments[parameter] = grid
                with self.subTest(parameter=parameter, grid=grid):
                    with self.assertRaises(ValueError):
                        run_search(self.returns, self.selections, **arguments)
        for width in (1, 5):
            with self.assertRaises(ValueError):
                run_search(self.returns, self.selections, (200,), (width,))
        with self.assertRaises(ValueError):
            run_search(self.returns, self.selections, (9900,), (100,))


if __name__ == "__main__":
    unittest.main()
