"""Explicit hindsight geometry search with fixed inherited day-selection masks.

No option quotes, new forecasts, model fits, inference or source readers here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.copula_spread_backtest import terminal_debit

MODELS = ("orig_gaussian", "orig_t8", "cal_gaussian", "cal_t8", "shape_gaussian", "shape_t8")
POLICIES = ("always_sell",) + tuple("fixed_2pct_" + model for model in MODELS)
STRUCTURES = ("put", "call", "condor")
DEFAULT_DISTANCE_BPS = tuple(range(25, 501, 25))
DEFAULT_WIDTH_BPS = (10, 25, 50, 75, 100, 150, 200, 300)
KEYS = ["phase", "structure", "policy", "distance_bps", "width_bps"]
GROUP = ["phase", "structure", "policy", "scenario"]
ACCOUNT_COLUMNS = KEYS + ["scenario", "credit_fraction", "ending_equity", "total_log_growth",
                         "total_return", "max_drawdown", "worst_return", "sessions", "sold_sessions"]
TIE_FIELDS = ["tie_distance_min_bps", "tie_distance_max_bps", "tie_width_min_bps", "tie_width_max_bps"]


def _grid(values, minimum):
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ValueError("Nonempty increasing integer-bps grid required") from exc
    if (not result or any(isinstance(x, (bool, np.bool_)) or
                         not isinstance(x, (int, np.integer)) or x < minimum for x in result)
            or tuple(sorted(set(result))) != result):
        raise ValueError("Nonempty increasing integer-bps grid required")
    return tuple(int(x) for x in result)


def _schema(frame, columns):
    if (not isinstance(frame, pd.DataFrame) or frame.empty or
            len(frame.columns) != len(columns) or set(frame.columns) != set(columns)):
        raise ValueError("Exact nonempty input schema required")


def _dates(series):
    if getattr(series.dtype, "kind", None) != "M":
        raise ValueError("Native dates required")
    dates = pd.DatetimeIndex(series)
    if (dates.tz is not None or dates.hasnans or not dates.equals(dates.normalize()) or
            (dates > pd.Timestamp("2025-10-20")).any()):
        raise ValueError("Naive finite midnight dates within historical ceiling required")
    return dates


def _inputs(returns, selections):
    _schema(returns, ["origin", "target_end", "phase", "y_qqq", "y_spx"])
    _schema(selections, ["origin", "structure", "policy", "sell"])
    origins, targets = _dates(returns.origin), _dates(returns.target_end)
    if (not origins.is_unique or not origins.is_monotonic_increasing or
            not targets.is_unique or not targets.is_monotonic_increasing or (targets <= origins).any()):
        raise ValueError("Unique increasing origins and later targets required")
    if set(returns.phase) != {"development", "evaluation"}:
        raise ValueError("Both original phases required")
    development = returns.phase.eq("development").to_numpy()
    if ((targets[development] > pd.Timestamp("2019-12-31")).any() or
            (origins[~development] < pd.Timestamp("2020-01-01")).any()):
        raise ValueError("Original phase boundary required")
    for name in ["y_qqq", "y_spx"]:
        if getattr(returns[name].dtype, "kind", None) not in "iuf":
            raise ValueError("Finite real numeric returns required")
    y = returns[["y_qqq", "y_spx"]].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Finite real numeric returns required")
    _dates(selections.origin)
    if selections.sell.dtype != np.dtype(bool):
        raise ValueError("Actual boolean selection masks required")
    if (set(selections.policy) != set(POLICIES) or set(selections.structure) != set(STRUCTURES)
            or selections.duplicated(["origin", "structure", "policy"]).any()
            or len(selections) != len(returns) * len(POLICIES) * len(STRUCTURES)
            or set(selections.origin) != set(returns.origin)
            or not selections.loc[selections.policy == "always_sell", "sell"].all()):
        raise ValueError("Exact complete frozen selection cohort required")
    masks = {}
    for structure in STRUCTURES:
        wide = selections[selections.structure == structure].pivot(index="origin", columns="policy", values="sell")
        wide = wide.reindex(index=origins, columns=POLICIES)
        if wide.isna().any().any():
            raise ValueError("Missing policy/structure/origin selection")
        masks[structure] = wide.to_numpy(dtype=bool)
    periods = {"development": development, "evaluation": ~development,
               "pooled": np.ones(len(returns), dtype=bool)}
    return y, masks, periods


def select_winners(accounts):
    """Select in hindsight, exposing exact tolerance ties and empty groups."""
    if (not isinstance(accounts, pd.DataFrame) or accounts.empty or
            not set(ACCOUNT_COLUMNS).issubset(accounts.columns) or
            accounts.duplicated(KEYS + ["scenario"]).any()):
        raise ValueError("Unique complete account records required")
    rows = []
    for _, group in accounts.groupby(GROUP, sort=True):
        if group.sessions.nunique() != 1:
            raise ValueError("Same scheduled cohort required within a ranking")
        eligible = group[group.sold_sessions > 0]
        if eligible.empty:
            first = group.iloc[0]
            row = dict.fromkeys(ACCOUNT_COLUMNS, np.nan)
            row.update({name: first[name] for name in GROUP})
            row.update(sessions=int(first.sessions), sold_sessions=0, status="NO_TRADES", tie_count=0)
            row.update(dict.fromkeys(TIE_FIELDS, np.nan))
        else:
            if not np.isfinite(eligible.total_log_growth.to_numpy(float)).all():
                raise ValueError("Finite ranking objective required")
            top = eligible.total_log_growth.max()
            tied = eligible[(eligible.total_log_growth - top).abs() <= 1e-10]
            tied = tied.sort_values(["distance_bps", "width_bps"])
            row = tied.iloc[0][ACCOUNT_COLUMNS].to_dict()
            row.update(status="HINDSIGHT_SELECTED", tie_count=len(tied),
                       tie_distance_min_bps=int(tied.distance_bps.min()),
                       tie_distance_max_bps=int(tied.distance_bps.max()),
                       tie_width_min_bps=int(tied.width_bps.min()),
                       tie_width_max_bps=int(tied.width_bps.max()))
        rows.append(row)
    return pd.DataFrame(rows, columns=ACCOUNT_COLUMNS + ["status", "tie_count"] + TIE_FIELDS)


def run_search(returns, selections, distance_bps=None, width_bps=None):
    distances = _grid(DEFAULT_DISTANCE_BPS if distance_bps is None else distance_bps, 1)
    widths = _grid(DEFAULT_WIDTH_BPS if width_bps is None else width_bps, 6)
    if max(distances) + max(widths) >= 10000:
        raise ValueError("Positive put long strikes required")
    y, masks, periods = _inputs(returns, selections)
    geometry, accounts = [], []
    for structure in STRUCTURES:
        for d_bps in distances:
            d = d_bps / 10000
            breach = np.zeros_like(y, dtype=bool)
            if structure in ("put", "condor"):
                breach |= y < np.log(1-d)
            if structure in ("call", "condor"):
                breach |= y > np.log(1+d)
            for w_bps in widths:
                w = w_bps / 10000
                debit = terminal_debit(y, structure, d, w).mean(axis=1)
                full = np.zeros_like(y, dtype=bool)
                if structure in ("put", "condor"):
                    full |= y <= np.log(1-d-w)
                if structure in ("call", "condor"):
                    full |= y >= np.log(1+d+w)
                for phase, phase_mask in periods.items():
                    liability = debit[phase_mask]
                    n = int(phase_mask.sum())
                    for policy_index, policy in enumerate(POLICIES):
                        sell = masks[structure][phase_mask, policy_index]
                        count = int(sell.sum())
                        selected = phase_mask.copy()
                        selected[phase_mask] = sell
                        mean = float(liability[sell].mean()) if count else np.nan
                        maximum = float(liability[sell].max()) if count else np.nan
                        base = dict(phase=phase, structure=structure, policy=policy,
                                    distance_bps=d_bps, width_bps=w_bps)
                        geometry.append(dict(
                            **base, sessions=n, sold_sessions=count, coverage=count/n,
                            any_breach_count=int(breach[selected].any(axis=1).sum()),
                            both_breach_count=int(breach[selected].all(axis=1).sum()),
                            any_full_count=int(full[selected].any(axis=1).sum()),
                            both_full_count=int(full[selected].all(axis=1).sum()),
                            mean_debit=mean, max_debit=maximum, mean_debit_open_bps=mean*w_bps,
                            average_cost_credit_fraction=mean+.02,
                            average_cost_credit_open_bps=(mean+.02)*w_bps))
                        for scenario, credit in [("width_05", .05), ("width_10", .10),
                                                 ("width_20", .20), ("open_5bp", 5/w_bps)]:
                            net = np.where(sell, .02*(credit-.02-liability), 0.)
                            logs = np.cumsum(np.log1p(net))
                            peak = np.maximum.accumulate(np.r_[0., logs])[1:]
                            growth = float(logs[-1])
                            accounts.append(dict(
                                **base, scenario=scenario, credit_fraction=credit,
                                ending_equity=float(np.exp(growth)), total_log_growth=growth,
                                total_return=float(np.expm1(growth)),
                                max_drawdown=float(np.expm1(logs-peak).min()),
                                worst_return=float(net.min()), sessions=n, sold_sessions=count))
    geom = pd.DataFrame(geometry).sort_values(KEYS).reset_index(drop=True)
    account = pd.DataFrame(accounts, columns=ACCOUNT_COLUMNS).sort_values(KEYS+["scenario"]).reset_index(drop=True)
    return {"geometry": geom, "accounts": account, "winners": select_winners(account)}
