"""Independent intrinsic-value, capital, and hindsight-selection reconstruction.

This verifier imports no search producer, forecast model, or earlier accounting
engine. It validates arithmetic and cohort completeness, not premiums, fills,
calibration, statistical significance, or the deployability of hindsight picks.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd


_STRUCTURES = ("put", "call", "condor")
_POLICIES = ("always_sell",) + tuple("fixed_2pct_" + name for name in (
    "orig_gaussian", "orig_t8", "cal_gaussian", "cal_t8", "shape_gaussian", "shape_t8"))
_KEY = ["phase", "structure", "policy", "distance_bps", "width_bps"]
_WINNER_GROUP = ["phase", "structure", "policy", "scenario"]
_TIE_FIELDS = ("tie_distance_min_bps", "tie_distance_max_bps",
               "tie_width_min_bps", "tie_width_max_bps")
_DISCRETE = {"sessions", "sold_sessions", "any_breach_count", "both_breach_count",
             "any_full_count", "both_full_count", "distance_bps", "width_bps",
             "tie_count", *_TIE_FIELDS}


def _frame(frame, columns, label):
    if (not isinstance(frame, pd.DataFrame) or frame.empty
            or not frame.columns.is_unique or set(frame.columns) != set(columns)):
        raise ValueError(f"{label}: exact nonempty frame schema required")
    return frame.copy()


def _dates(values):
    if getattr(values.dtype, "kind", None) != "M":
        raise ValueError("native dates required")
    dates = pd.DatetimeIndex(values)
    if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
        raise ValueError("finite timezone-free midnight dates required")
    if (dates > pd.Timestamp("2025-10-20")).any():
        raise ValueError("historical source ceiling exceeded")
    return dates


def _grid(values, label):
    try:
        items = list(values)
    except TypeError as exc:
        raise ValueError(f"{label}: grid must be a sequence") from exc
    if (not items or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer))
                         for x in items) or any(a >= b for a, b in zip(items, items[1:]))):
        raise ValueError(f"{label}: nonempty unique increasing integer grid required")
    return items


def _inputs(returns, selections, distances, widths):
    distances, widths = _grid(distances, "distance"), _grid(widths, "width")
    if min(distances) <= 0 or min(widths) <= 5 or max(distances) + max(widths) >= 10000:
        raise ValueError("invalid distance/width geometry")
    y = _frame(returns, ("origin", "target_end", "phase", "y_qqq", "y_spx"), "returns")
    s = _frame(selections, ("origin", "structure", "policy", "sell"), "selections")
    origins, targets = _dates(y.origin), _dates(y.target_end)
    _dates(s.origin)
    if (not origins.is_unique or not origins.is_monotonic_increasing or not targets.is_unique
            or not targets.is_monotonic_increasing or not (targets > origins).all()):
        raise ValueError("unique chronological origins and following targets required")
    if set(y.phase) != {"development", "evaluation"}:
        raise ValueError("both historical phases required")
    valid_phase = ((y.phase.eq("development") & (y.target_end <= pd.Timestamp("2019-12-31")))
                   | (y.phase.eq("evaluation") & (y.origin >= pd.Timestamp("2020-01-01"))))
    if not valid_phase.all():
        raise ValueError("phase chronology mismatch")
    for column in ("y_qqq", "y_spx"):
        if y[column].dtype.kind not in "iuf" or not np.isfinite(y[column]).all():
            raise ValueError("finite real log returns required")
    if s.sell.dtype.kind != "b" or s.sell.isna().any():
        raise ValueError("boolean frozen selections required")
    expected = pd.MultiIndex.from_product([origins, _STRUCTURES, _POLICIES],
                                          names=["origin", "structure", "policy"])
    observed = pd.MultiIndex.from_frame(s[["origin", "structure", "policy"]])
    if observed.has_duplicates or not observed.sort_values().equals(expected.sort_values()):
        raise ValueError("complete origin/structure/policy selection cohort required")
    if not s.loc[s.policy.eq("always_sell"), "sell"].all():
        raise ValueError("always-sell control must always sell")
    masks = {key: frame.set_index("origin").loc[origins, "sell"].to_numpy(bool)
             for key, frame in s.groupby(["structure", "policy"], sort=False)}
    phases = [(phase, np.flatnonzero(y.phase.eq(phase).to_numpy()))
              for phase in ("development", "evaluation")]
    phases.append(("pooled", np.arange(len(y))))
    return y[["y_qqq", "y_spx"]].to_numpy(float), masks, phases, distances, widths


def _intrinsic(log_returns, structure, distance_bps, width_bps):
    d, w = distance_bps / 10000., width_bps / 10000.
    debit = np.zeros_like(log_returns)
    breach = np.zeros(log_returns.shape, bool)
    full = np.zeros(log_returns.shape, bool)
    if structure in ("put", "condor"):
        short, long = 1 - d, 1 - d - w
        at_max = log_returns <= np.log(long)
        inside = (log_returns > np.log(long)) & (log_returns < np.log(short))
        debit[at_max] = 1.
        debit[inside] = (short - np.exp(log_returns[inside])) / w
        breach |= log_returns < np.log(short)
        full |= at_max
    if structure in ("call", "condor"):
        short, long = 1 + d, 1 + d + w
        at_max = log_returns >= np.log(long)
        inside = (log_returns > np.log(short)) & (log_returns < np.log(long))
        debit[at_max] = 1.
        debit[inside] = (np.exp(log_returns[inside]) - short) / w
        breach |= log_returns > np.log(short)
        full |= at_max
    return debit.mean(axis=1), breach.any(axis=1), breach.all(axis=1), full.any(axis=1), full.all(axis=1)


def _reconstruct(log_returns, masks, phases, distances, widths):
    geometry, accounts = [], []
    for structure, distance, width in product(_STRUCTURES, distances, widths):
        debit, any_breach, both_breach, any_full, both_full = _intrinsic(log_returns, structure, distance, width)
        for phase, idx in phases:
            for policy in _POLICIES:
                sell = masks[(structure, policy)][idx]
                selected = idx[sell]
                count = len(selected)
                mean = float(np.mean(debit[selected])) if count else np.nan
                maximum = float(np.max(debit[selected])) if count else np.nan
                key = dict(phase=phase, structure=structure, policy=policy,
                           distance_bps=distance, width_bps=width)
                geometry.append({**key, "sessions": len(idx), "sold_sessions": count,
                                 "coverage": count / len(idx),
                                 "any_breach_count": int(np.count_nonzero(any_breach[selected])),
                                 "both_breach_count": int(np.count_nonzero(both_breach[selected])),
                                 "any_full_count": int(np.count_nonzero(any_full[selected])),
                                 "both_full_count": int(np.count_nonzero(both_full[selected])),
                                 "mean_debit": mean, "max_debit": maximum,
                                 "mean_debit_open_bps": mean * width,
                                 "average_cost_credit_fraction": mean + .02,
                                 "average_cost_credit_open_bps": (mean + .02) * width})
                for scenario, credit in (("width_05", .05), ("width_10", .10),
                                          ("width_20", .20), ("open_5bp", 5 / width)):
                    daily = np.zeros(len(idx))
                    daily[sell] = .02 * (credit - .02 - debit[selected])
                    # Include initial equity so a first-session loss is a drawdown.
                    capital = np.r_[1., np.cumprod(1 + daily)]
                    dd = capital / np.maximum.accumulate(capital) - 1
                    accounts.append({**key, "scenario": scenario, "credit_fraction": credit,
                                     "ending_equity": float(capital[-1]),
                                     "total_log_growth": float(np.sum(np.log1p(daily))),
                                     "total_return": float(capital[-1] - 1),
                                     "max_drawdown": float(np.min(dd)),
                                     "worst_return": float(np.min(daily)),
                                     "sessions": len(idx), "sold_sessions": count})
    accounts = pd.DataFrame(accounts)
    winners = []
    for _, group in accounts.groupby(_WINNER_GROUP, sort=False):
        eligible = group.loc[group.sold_sessions.gt(0)]
        if eligible.empty:
            first = group.iloc[0]
            winner = {column: first[column] if column in _WINNER_GROUP + ["sessions", "sold_sessions"]
                      else np.nan for column in accounts.columns}
            winner.update(status="NO_TRADES", tie_count=0)
            winner.update({field: np.nan for field in _TIE_FIELDS})
        else:
            best = eligible.total_log_growth.max()
            tied = eligible.loc[np.abs(eligible.total_log_growth - best) <= 1e-10]
            chosen = tied.sort_values(["distance_bps", "width_bps"]).iloc[0]
            winner = chosen.to_dict()
            winner.update(status="HINDSIGHT_SELECTED", tie_count=len(tied),
                          tie_distance_min_bps=int(tied.distance_bps.min()),
                          tie_distance_max_bps=int(tied.distance_bps.max()),
                          tie_width_min_bps=int(tied.width_bps.min()),
                          tie_width_max_bps=int(tied.width_bps.max()))
        winners.append(winner)
    return {"geometry": pd.DataFrame(geometry), "accounts": accounts, "winners": pd.DataFrame(winners)}


def _compare(actual, expected, keys, label):
    if (not isinstance(actual, pd.DataFrame) or not actual.columns.is_unique
            or set(actual.columns) != set(expected.columns) or len(actual) != len(expected)
            or actual.duplicated(keys).any()):
        raise ValueError(f"{label}: row count, columns, or unique keys differ")
    left, right = (frame.sort_values(keys).reset_index(drop=True) for frame in (actual, expected))
    for column in right.columns:
        a, e = left[column], right[column]
        if e.dtype.kind in "iuf":
            if column in _DISCRETE:
                valid = (a.dtype.kind in "iuf"
                         and np.array_equal(a.to_numpy(float), e.to_numpy(float), equal_nan=True))
            else:
                valid = (a.dtype.kind in "iuf"
                         and np.allclose(a.to_numpy(float), e.to_numpy(float),
                                         atol=1e-11, rtol=1e-9, equal_nan=True))
        else:
            valid = np.array_equal(a.to_numpy(), e.to_numpy())
        if not valid:
            raise ValueError(f"{label}: independent reconstruction differs in {column}")


def verify_search(returns, selections, outputs, distance_bps, width_bps):
    """Reconstruct every grid case and reject missing, changed, or extra outputs."""
    if not isinstance(outputs, dict) or set(outputs) != {"geometry", "accounts", "winners"}:
        raise ValueError("exact three-table output mapping required")
    inputs = _inputs(returns, selections, distance_bps, width_bps)
    expected = _reconstruct(*inputs)
    for label, keys in (("geometry", _KEY), ("accounts", _KEY + ["scenario"]),
                        ("winners", _WINNER_GROUP)):
        _compare(outputs[label], expected[label], keys, label)
    return {"status": "VERIFIED", "origins": len(returns), "policies": len(_POLICIES),
            "geometries_per_structure": len(inputs[-2]) * len(inputs[-1]),
            "geometry_rows": len(expected["geometry"]), "account_rows": len(expected["accounts"]),
            "winner_rows": len(expected["winners"]),
            "scope": "Independent intrinsic, event counts, all hypothetical capital accounts, and hindsight selections; no forecast calibration or actual fills."}
