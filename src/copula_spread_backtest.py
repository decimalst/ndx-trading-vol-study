"""Bounded terminal spread liabilities and explicitly hypothetical credit accounts.

QQQ ETF / SPX index forecasts are underlying proxies, not option quotes or fills.
No source reader, model fit or historical parameter selection belongs here.
"""

from __future__ import annotations

import copy
import re
from itertools import product

import numpy as np
import pandas as pd

MODELS = (
    "orig_gaussian",
    "orig_t8",
    "cal_gaussian",
    "cal_t8",
    "shape_gaussian",
    "shape_t8",
)
STRUCTURES = ("put", "call", "condor")
DISTANCES = (0.01, 0.02, 0.03)
WIDTH = 0.01
RISK_COLUMNS = (
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
    "es97_5",
)
REALIZED_COLUMNS = ("origin", "target_end", "y_qqq", "y_spx")
DEFAULT_CONFIG = {
    "breach_threshold": 0.10,
    "max_liability_budget": 0.02,
    "credit_fractions": [0.05, 0.10, 0.20],
    "fee_fraction": 0.02,
    "development": ["2016-01-04", "2019-12-31"],
    "evaluation": ["2020-01-01", "2025-10-20"],
    "source_end": "2025-10-20",
}


def _number(value, name, low, high, *, low_open=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"Real numeric {name} required")
    value = float(value)
    if not np.isfinite(value) or not (
        low < value <= high if low_open else low <= value <= high
    ):
        raise ValueError(f"Invalid {name}")
    return value


def _geometry(structure, distance, width):
    if structure not in STRUCTURES:
        raise ValueError("Structure must be put, call or condor")
    d = _number(distance, "distance", 0, 1)
    w = _number(width, "width", 0, 1, low_open=True)
    if d + w >= 1:
        raise ValueError("Positive put long strike required")
    return np.log(1 - d), np.log(1 - d - w), np.log(1 + d), np.log(1 + d + w)


def _returns(values):
    array = np.asarray(values)
    if array.dtype.kind not in "iuf" or not array.size:
        raise ValueError("Nonempty real nonboolean log returns required")
    array = array.astype(float, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("Finite log returns required")
    return array


def terminal_debit(log_returns, structure, distance=0.02, width=0.01):
    """Per-asset intrinsic liability / wing width, same shape as log_returns.

    Exponentials are evaluated only between the two finite log strikes. The
    ratio of expm1 differences avoids cancellation near the short strike and
    equals dollar intrinsic divided by the strike interval. A condor's mutually
    exclusive wings share one width of maximum liability, not two.
    """
    y = _returns(log_returns)
    ps, pl, cs, cl = _geometry(structure, distance, width)
    debit = np.zeros_like(y, dtype=float)
    if structure in ("put", "condor"):
        debit[y <= pl] = 1.0
        inside = (y > pl) & (y < ps)
        debit[inside] = np.expm1(y[inside] - ps) / np.expm1(pl - ps)
    if structure in ("call", "condor"):
        debit[y >= cl] = 1.0
        inside = (y > cs) & (y < cl)
        debit[inside] = np.expm1(y[inside] - cs) / np.expm1(cl - cs)
    if not np.isfinite(debit).all() or (debit < 0).any() or (debit > 1).any():
        raise ValueError("Invalid terminal liability arithmetic")
    return debit


def _events(y, structure, distance, width):
    ps, pl, cs, cl = _geometry(structure, distance, width)
    breach, full = np.zeros_like(y, dtype=bool), np.zeros_like(y, dtype=bool)
    if structure in ("put", "condor"):
        breach |= y < ps
        full |= y <= pl
    if structure in ("call", "condor"):
        breach |= y > cs
        full |= y >= cl
    return breach, full


def _iso(value):
    if type(value) is not str or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        raise ValueError("Literal ISO date required")
    try:
        return pd.Timestamp(value)
    except Exception as error:
        raise ValueError("Valid ISO date required") from error


def _config(config):
    if config is not None and (type(config) is not dict or set(config) - set(DEFAULT_CONFIG)):
        raise ValueError("Known configuration fields required")
    c = copy.deepcopy(DEFAULT_CONFIG)
    c.update(copy.deepcopy(config or {}))
    for name in ("breach_threshold", "fee_fraction"):
        c[name] = _number(c[name], name, 0, 1)
    c["max_liability_budget"] = _number(
        c["max_liability_budget"], "liability budget", 0, 1, low_open=True
    )
    if type(c["credit_fractions"]) is not list or not c["credit_fractions"]:
        raise ValueError("Nonempty credit scenarios required")
    credits = [_number(x, "credit fraction", 0, 1) for x in c["credit_fractions"]]
    if credits != sorted(set(credits)):
        raise ValueError("Unique increasing credit scenarios required")
    c["credit_fractions"] = credits
    c["source_end"] = _iso(c["source_end"])
    for phase in ("development", "evaluation"):
        if type(c[phase]) is not list or len(c[phase]) != 2:
            raise ValueError("Two phase bounds required")
        c[phase] = [_iso(x) for x in c[phase]]
    if not (
        c["development"][0]
        <= c["development"][1]
        < c["evaluation"][0]
        <= c["evaluation"][1]
        <= c["source_end"]
        <= pd.Timestamp("2025-10-20")
    ):
        raise ValueError("Ordered phases within fixed source ceiling required")
    return c


def _dates(values, ceiling):
    if getattr(values.dtype, "kind", None) != "M":
        raise ValueError("Native timezone-free dates required")
    dates = pd.DatetimeIndex(values)
    if (
        dates.hasnans
        or dates.tz is not None
        or not dates.equals(dates.normalize())
        or (dates > ceiling).any()
    ):
        raise ValueError("Finite midnight dates within source ceiling required")
    return dates.astype("datetime64[ns]")


def _frame(frame, columns, ceiling):
    if (
        not isinstance(frame, pd.DataFrame)
        or frame.empty
        or not frame.columns.is_unique
        or set(frame.columns) != set(columns)
    ):
        raise ValueError("Exact nonempty frame schema required")
    out = frame.loc[:, columns].copy(deep=True)
    for name in set(columns) & {"origin", "target_end", "feature_cutoff_date"}:
        out[name] = _dates(out[name], ceiling)
    return out


def _numeric(frame, columns):
    for name in columns:
        if frame[name].dtype.kind not in "iuf":
            raise ValueError("Real nonboolean numeric column required: " + name)
        if not np.isfinite(frame[name].to_numpy(float)).all():
            raise ValueError("Finite numeric column required: " + name)


def _risks(risks, calendar, c):
    if not isinstance(calendar, pd.DatetimeIndex):
        raise ValueError("Native full reference calendar required")
    cal = _dates(calendar, c["source_end"])
    if not cal.is_unique or not cal.is_monotonic_increasing or len(cal) < 3:
        raise ValueError("Unique increasing full calendar required")
    r = _frame(risks, RISK_COLUMNS, c["source_end"])
    _numeric(r, ["distance", "width", "p_any_breach", "mean_debit", "es97_5"])
    for name in ("p_any_breach", "mean_debit", "es97_5"):
        if not r[name].between(0, 1).all():
            raise ValueError("Unit-bounded predictive risk required")
    if (r["mean_debit"] > r["es97_5"] + 1e-12).any():
        raise ValueError("Tail expected loss cannot be smaller than mean loss")
    keys = ["origin", "structure", "distance", "model"]
    if r.duplicated(keys).any() or not r.origin.is_monotonic_increasing:
        raise ValueError("Unique chronological risk keys required")
    expected = set(product(STRUCTURES, DISTANCES, MODELS))
    for origin, group in r.groupby("origin", sort=True):
        if set(zip(group.structure, group.distance, group.model)) != expected:
            raise ValueError("Complete nine cases and six model forecasts required")
        if not (group.width == WIDTH).all():
            raise ValueError("Fixed one-percent wing width required")
        for name in ("target_end", "feature_cutoff_date", "phase"):
            if group[name].nunique(dropna=False) != 1:
                raise ValueError("Common clock/phase across all model cases required")
        p = cal.get_indexer([origin])[0]
        if p < 1 or p + 1 >= len(cal):
            raise ValueError("Origin must have preceding and following reference sessions")
        row = group.iloc[0]
        if row.feature_cutoff_date != cal[p - 1] or row.target_end != cal[p + 1]:
            raise ValueError("Exact prior-feature and next-target session clock required")
        if row.phase not in ("development", "evaluation"):
            raise ValueError("Recognized phase required")
        lo, hi = c[row.phase]
        if not (lo <= origin <= row.target_end <= hi):
            raise ValueError("Origin and expiration proxy must stay within their phase")
    return r


def _decisions(r, c):
    rows = []
    keys = ["origin", "structure", "distance"]
    for _, group in r.groupby(keys, sort=True):
        first = group.iloc[0]
        base = {
            name: first[name]
            for name in (
                "origin",
                "target_end",
                "feature_cutoff_date",
                "phase",
                "structure",
                "distance",
                "width",
            )
        }
        for model in ("always_sell", *MODELS):
            if model == "always_sell":
                sell, probability, mean, es = True, None, None, None
            else:
                source = group.loc[group.model == model].iloc[0]
                probability, mean, es = (
                    float(source.p_any_breach),
                    float(source.mean_debit),
                    float(source.es97_5),
                )
                sell = probability <= c["breach_threshold"]
            reserve = c["max_liability_budget"] if sell else 0.0
            rows.append(
                {
                    **base,
                    "model": model,
                    "p_any_breach": probability,
                    "mean_debit": mean,
                    "es97_5": es,
                    "sell": bool(sell),
                    "gross_reserve": reserve,
                    "reserve_qqq": reserve / 2,
                    "reserve_spx": reserve / 2,
                }
            )
    return pd.DataFrame(rows)


def run_backtest(risks, realized, calendar, config=None):
    """Seven safety policies, all cases and hypothetical credits, with no tuning.

    Counterfactual accounts reset independently for each phase, structure,
    distance, model and credit. Positions are decided before reading outcomes.
    The caller freezes source provenance and the actual configuration.
    """
    c = _config(config)
    r = _risks(risks, calendar, c)
    positions = _decisions(r, c)
    y = _frame(realized, REALIZED_COLUMNS, c["source_end"])
    _numeric(y, ["y_qqq", "y_spx"])
    origins = pd.DatetimeIndex(r.origin.drop_duplicates())
    if (
        not y.origin.is_unique
        or not y.origin.is_monotonic_increasing
        or not pd.DatetimeIndex(y.origin).equals(origins)
    ):
        raise ValueError("Exact ordered realized origin coverage required")
    metadata = r.drop_duplicates("origin").set_index("origin")
    if not np.array_equal(
        y.target_end.to_numpy(), metadata.loc[origins, "target_end"].to_numpy()
    ):
        raise ValueError("Realized target clock mismatch")
    returns = y[["y_qqq", "y_spx"]].to_numpy(float)
    outcomes = []
    for structure, distance in product(STRUCTURES, DISTANCES):
        debit = terminal_debit(returns, structure, distance, WIDTH)
        breach, full = _events(returns, structure, distance, WIDTH)
        for j, origin in enumerate(origins):
            outcomes.append(
                {
                    "origin": origin,
                    "target_end": y.iloc[j].target_end,
                    "phase": metadata.loc[origin, "phase"],
                    "structure": structure,
                    "distance": distance,
                    "width": WIDTH,
                    "debit_qqq": float(debit[j, 0]),
                    "debit_spx": float(debit[j, 1]),
                    "portfolio_debit": float(debit[j].mean()),
                    "any_breach": bool(breach[j].any()),
                    "both_breach": bool(breach[j].all()),
                    "any_full_loss": bool(full[j].any()),
                    "both_full_loss": bool(full[j].all()),
                }
            )
    outcomes = (
        pd.DataFrame(outcomes)
        .sort_values(["origin", "structure", "distance"])
        .reset_index(drop=True)
    )
    lookup = outcomes.set_index(["origin", "structure", "distance"]).to_dict("index")
    path, summaries = [], []
    for (phase, structure, distance, model), group in positions.groupby(
        ["phase", "structure", "distance", "model"], sort=True
    ):
        for credit in c["credit_fractions"]:
            equity, peak, ruined = 1.0, 1.0, False
            account_rows = []
            for pos in group.sort_values("origin").to_dict("records"):
                outcome = lookup[(pos["origin"], structure, distance)]
                sell = bool(pos["sell"] and not ruined)
                reserve = c["max_liability_budget"] if sell else 0.0
                debit = outcome["portfolio_debit"]
                credit_return, liability_return, fee_return = (
                    reserve * credit,
                    reserve * debit,
                    reserve * c["fee_fraction"],
                )
                net = credit_return - liability_return - fee_return
                start = equity
                pnl = start * net if not ruined else 0.0
                equity += pnl
                if not np.isfinite(equity):
                    raise ValueError("Nonfinite capital accounting")
                status = "inactive_after_ruin" if ruined else "sold" if sell else "blocked"
                ruined = bool(ruined or equity <= 0)
                peak = max(peak, equity)
                budget = c["max_liability_budget"]
                baseline_net = budget * (credit - c["fee_fraction"] - debit)
                row = {
                    **pos,
                    **outcome,
                    "credit_fraction": credit,
                    "fee_fraction": c["fee_fraction"],
                    "planned_sell": pos["sell"],
                    "sell": sell,
                    "executed_reserve": reserve,
                    "reserve_qqq": reserve / 2,
                    "reserve_spx": reserve / 2,
                    "starting_equity": start,
                    "credit_return": credit_return,
                    "liability_return": liability_return,
                    "fee_return": fee_return,
                    "net_return": net,
                    "credit_pnl": start * credit_return if sell else 0.0,
                    "liability_pnl": start * liability_return if sell else 0.0,
                    "fee_pnl": start * fee_return if sell else 0.0,
                    "net_pnl": pnl,
                    "equity": equity,
                    "running_peak": peak,
                    "drawdown": equity / peak - 1,
                    "avoided_liability_return": (budget - reserve) * debit,
                    "missed_credit_return": (budget - reserve) * credit,
                    "saved_fee_return": (budget - reserve) * c["fee_fraction"],
                    "always_sell_counterfactual_return": baseline_net,
                    "relative_net_return": net - baseline_net,
                    "ruined": ruined,
                    "status": status,
                }
                path.append(row)
                account_rows.append(row)
            a = pd.DataFrame(account_rows)
            sold = a.loc[a.sell]
            summaries.append(
                {
                    "phase": phase,
                    "structure": structure,
                    "distance": distance,
                    "width": WIDTH,
                    "model": model,
                    "credit_fraction": credit,
                    "fee_fraction": c["fee_fraction"],
                    "sessions": len(a),
                    "sold_sessions": len(sold),
                    "coverage": len(sold) / len(a),
                    "first_origin": a.origin.iloc[0],
                    "last_origin": a.origin.iloc[-1],
                    "first_target": a.target_end.iloc[0],
                    "last_target": a.target_end.iloc[-1],
                    "ending_equity": float(equity),
                    "total_return": float(equity - 1),
                    "max_drawdown": float(a.drawdown.min()),
                    "worst_net_return": float(a.net_return.min()),
                    "total_credit_pnl": float(a.credit_pnl.sum()),
                    "total_liability_pnl": float(a.liability_pnl.sum()),
                    "total_fee_pnl": float(a.fee_pnl.sum()),
                    "net_pnl": float(a.net_pnl.sum()),
                    "sold_mean_debit": float(sold.portfolio_debit.mean())
                    if len(sold)
                    else None,
                    "sold_any_breach_rate": float(sold.any_breach.mean())
                    if len(sold)
                    else None,
                    "sold_both_breach_rate": float(sold.both_breach.mean())
                    if len(sold)
                    else None,
                    "sold_any_full_loss_rate": float(sold.any_full_loss.mean())
                    if len(sold)
                    else None,
                    "sold_both_full_loss_rate": float(sold.both_full_loss.mean())
                    if len(sold)
                    else None,
                    "avoided_liability_return_sum": float(a.avoided_liability_return.sum()),
                    "missed_credit_return_sum": float(a.missed_credit_return.sum()),
                    "saved_fee_return_sum": float(a.saved_fee_return.sum()),
                    "ruined": ruined,
                    "interpretation": "Hypothetical-credit terminal-liability proxy; not observed option profit",
                }
            )
    return {
        "positions": positions,
        "path": pd.DataFrame(path),
        "summary": pd.DataFrame(summaries),
        "outcomes": outcomes,
    }
