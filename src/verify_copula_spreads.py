"""Independent saved terminal-liability and hypothetical-account reconstruction.

No imports from the forecast or backtest producer. No fitting or resampling.
Vectorized cumulative products check capital accounts independently of the
producer's sequential update loop. Predictive probabilities are inputs here;
this certificate does not validate their statistical calibration or real fills.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

MODELS = ("orig_gaussian", "orig_t8", "cal_gaussian", "cal_t8", "shape_gaussian", "shape_t8")
STRUCTURES = ("put", "call", "condor")
DISTANCES = (.01, .02, .03)
RISK_FIELDS = ("origin", "target_end", "feature_cutoff_date", "phase", "model", "structure",
               "distance", "width", "p_any_breach", "mean_debit", "es97_5")
REALIZED_FIELDS = ("origin", "target_end", "y_qqq", "y_spx")
CASE = ["origin", "structure", "distance"]
ACCOUNT = ["phase", "structure", "distance", "model", "credit_fraction"]


def _frame(frame, columns):
    if (not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique
            or set(frame.columns) != set(columns)):
        raise ValueError("nonempty exact frame schema required")
    return frame.copy()


def _numeric(frame, columns):
    for column in columns:
        if frame[column].dtype.kind not in "iuf" or not np.isfinite(frame[column]).all():
            raise ValueError(f"finite real numeric values required: {column}")


def _dates(values):
    if getattr(values.dtype, "kind", None) != "M":
        raise ValueError("native dates required")
    dates = pd.DatetimeIndex(values)
    if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
        raise ValueError("finite timezone-free midnight dates required")
    if (dates > pd.Timestamp("2025-10-20")).any():
        raise ValueError("historical source ceiling exceeded")
    return dates


def _compare(actual, expected, keys, label):
    if (not isinstance(actual, pd.DataFrame) or not actual.columns.is_unique
            or set(actual.columns) != set(expected.columns) or len(actual) != len(expected)
            or actual.duplicated(keys).any()):
        raise ValueError(f"{label} schema, row count or key mismatch")
    a = actual.sort_values(keys).reset_index(drop=True)
    e = expected.sort_values(keys).reset_index(drop=True)
    for column in e.columns:
        left, right = a[column], e[column]
        if right.dtype.kind == "M":
            valid = left.dtype.kind == "M" and np.array_equal(left, right)
        elif right.dtype.kind == "b":
            valid = left.dtype.kind == "b" and np.array_equal(left, right)
        elif right.dtype.kind in "iuf":
            valid = left.dtype.kind in "iuf" and np.allclose(
                left.to_numpy(float), right.to_numpy(float), atol=1e-11, rtol=1e-9,
                equal_nan=True)
        else:
            valid = np.array_equal(left.to_numpy(), right.to_numpy())
        if not valid:
            raise ValueError(f"{label} independent reconstruction differs: {column}")


def _inputs(risks, realized, calendar):
    r, y = _frame(risks, RISK_FIELDS), _frame(realized, REALIZED_FIELDS)
    if not isinstance(calendar, pd.DatetimeIndex):
        raise ValueError("full reference calendar required")
    cal = _dates(calendar)
    if not cal.is_unique or not cal.is_monotonic_increasing:
        raise ValueError("unique increasing reference calendar required")
    for frame in (r, y):
        for field in set(frame.columns) & {"origin", "target_end", "feature_cutoff_date"}:
            _dates(frame[field])
    _numeric(r, ("distance", "width", "p_any_breach", "mean_debit", "es97_5"))
    _numeric(y, ("y_qqq", "y_spx"))
    if not r.origin.is_monotonic_increasing or not y.origin.is_monotonic_increasing:
        raise ValueError("chronological input cohorts required")
    if r.duplicated([*CASE, "model"]).any() or y.origin.duplicated().any():
        raise ValueError("duplicate input keys")
    for field in ("p_any_breach", "mean_debit", "es97_5"):
        if not r[field].between(0, 1).all():
            raise ValueError("risk outside unit interval")
    if (r.width != .01).any() or (r.mean_debit > r.es97_5 + 1e-12).any():
        raise ValueError("fixed width or mean/tail ordering violated")
    origins = pd.DatetimeIndex(r.origin.drop_duplicates())
    expected_keys = pd.MultiIndex.from_product(
        [origins, STRUCTURES, DISTANCES, MODELS], names=[*CASE, "model"])
    observed_keys = pd.MultiIndex.from_frame(r[[*CASE, "model"]])
    if not observed_keys.sort_values().equals(expected_keys.sort_values()):
        raise ValueError("all nine structures/distances and six models required per origin")
    if not pd.DatetimeIndex(y.origin).equals(origins):
        raise ValueError("realized and forecast origin cohorts differ")
    grouped = r.groupby("origin", sort=False)
    for column in ("target_end", "feature_cutoff_date", "phase"):
        if (grouped[column].nunique(dropna=False) != 1).any():
            raise ValueError("model cases disagree on dates or phase")
    metadata = r.drop_duplicates("origin").set_index("origin").loc[origins]
    positions = cal.get_indexer(origins)
    if (positions < 1).any() or (positions + 1 >= len(cal)).any():
        raise ValueError("origin lacks preceding or following full session")
    if (not np.array_equal(metadata.feature_cutoff_date, cal[positions - 1])
            or not np.array_equal(metadata.target_end, cal[positions + 1])
            or not np.array_equal(metadata.target_end, y.target_end)):
        raise ValueError("prior feature/next target clock mismatch")
    development = metadata.phase == "development"
    evaluation = metadata.phase == "evaluation"
    valid = ((development & (origins >= pd.Timestamp("2016-01-04"))
              & (metadata.target_end <= pd.Timestamp("2019-12-31")))
             | (evaluation & (origins >= pd.Timestamp("2020-01-01"))
                & (metadata.target_end <= pd.Timestamp("2025-10-20"))))
    if not valid.all():
        raise ValueError("registered phase bounds violated")
    return r, y, metadata


def _outcomes(y, metadata):
    returns = y[["y_qqq", "y_spx"]].to_numpy(float)
    frames = []
    for structure, distance in product(STRUCTURES, DISTANCES):
        debit = np.zeros_like(returns)
        breach, full = np.zeros_like(returns, bool), np.zeros_like(returns, bool)
        if structure in ("put", "condor"):
            short, long = 1-distance, 1-distance-.01
            # Exponentiate only prices inside the wing: extreme finite log returns stay bounded.
            prices = np.exp(np.clip(returns, np.log(long), np.log(short)))
            debit += np.clip((short - prices) / .01, 0, 1)
            breach |= returns < np.log(short)
            full |= returns <= np.log(long)
        if structure in ("call", "condor"):
            short, long = 1+distance, 1+distance+.01
            prices = np.exp(np.clip(returns, np.log(short), np.log(long)))
            debit += np.clip((prices - short) / .01, 0, 1)
            breach |= returns > np.log(short)
            full |= returns >= np.log(long)
        frame = y[["origin", "target_end"]].copy()
        frame["phase"] = metadata.phase.to_numpy()
        frame["structure"], frame["distance"], frame["width"] = structure, distance, .01
        frame["debit_qqq"], frame["debit_spx"] = debit[:, 0], debit[:, 1]
        frame["portfolio_debit"] = debit.sum(axis=1) / 2
        frame["any_breach"], frame["both_breach"] = breach.any(axis=1), breach.all(axis=1)
        frame["any_full_loss"], frame["both_full_loss"] = full.any(axis=1), full.all(axis=1)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _positions(r):
    baseline = r.drop_duplicates(CASE).copy()
    baseline["model"] = "always_sell"
    baseline[["p_any_breach", "mean_debit", "es97_5"]] = np.nan
    positions = pd.concat([r, baseline], ignore_index=True)
    positions["sell"] = ((positions.model == "always_sell") | (positions.p_any_breach <= .10))
    positions["gross_reserve"] = .02 * positions.sell
    positions["reserve_qqq"] = .01 * positions.sell
    positions["reserve_spx"] = .01 * positions.sell
    return positions


def _paths(positions, outcomes):
    common = ["origin", "target_end", "phase", "structure", "distance", "width"]
    path = positions.merge(outcomes, on=common, validate="many_to_one").merge(
        pd.DataFrame({"credit_fraction": [.05, .10, .20]}), how="cross")
    path = path.sort_values([*ACCOUNT, "origin"]).reset_index(drop=True)
    path["fee_fraction"] = .02
    path["planned_sell"] = path.sell
    path["executed_reserve"] = .02 * path.sell
    path["credit_return"] = path.executed_reserve * path.credit_fraction
    path["liability_return"] = path.executed_reserve * path.portfolio_debit
    path["fee_return"] = path.executed_reserve * .02
    path["net_return"] = path.executed_reserve * (path.credit_fraction - path.portfolio_debit - .02)
    # Maximum reserve is 2%, debit is bounded by one: no account can ruin here.
    path["growth"] = 1 + path.net_return
    path["equity"] = path.groupby(ACCOUNT, sort=False).growth.cumprod()
    path["starting_equity"] = path.groupby(ACCOUNT, sort=False).equity.shift(fill_value=1.)
    if not np.isfinite(path.equity).all() or (path.equity <= 0).any():
        raise ValueError("bounded account reconstruction has invalid capital")
    path["running_peak"] = np.maximum(1., path.groupby(ACCOUNT, sort=False).equity.cummax())
    path["drawdown"] = path.equity / path.running_peak - 1
    for component in ("credit", "liability", "fee", "net"):
        path[f"{component}_pnl"] = path.starting_equity * path[f"{component}_return"]
    unused = .02 - path.executed_reserve
    path["avoided_liability_return"] = unused * path.portfolio_debit
    path["missed_credit_return"] = unused * path.credit_fraction
    path["saved_fee_return"] = unused * .02
    path["always_sell_counterfactual_return"] = .02 * (path.credit_fraction - .02 - path.portfolio_debit)
    path["relative_net_return"] = path.avoided_liability_return - path.missed_credit_return + path.saved_fee_return
    if not np.allclose(path.relative_net_return, path.net_return - path.always_sell_counterfactual_return,
                       atol=1e-14, rtol=1e-12):
        raise ValueError("avoided liability / missed credit decomposition failed")
    path["ruined"] = False
    path["status"] = np.where(path.sell, "sold", "blocked")
    return path.drop(columns="growth")


def _summaries(path):
    working = path.copy()
    conditional = {
        "sold_mean_debit": "portfolio_debit", "sold_any_breach_rate": "any_breach",
        "sold_both_breach_rate": "both_breach", "sold_any_full_loss_rate": "any_full_loss",
        "sold_both_full_loss_rate": "both_full_loss",
    }
    for name, field in conditional.items():
        working[name] = working[field].astype(float).where(working.sell)
    aggregation = {
        "sessions": ("origin", "size"), "sold_sessions": ("sell", "sum"),
        "coverage": ("sell", "mean"), "first_origin": ("origin", "first"),
        "last_origin": ("origin", "last"), "first_target": ("target_end", "first"),
        "last_target": ("target_end", "last"), "ending_equity": ("equity", "last"),
        "max_drawdown": ("drawdown", "min"), "worst_net_return": ("net_return", "min"),
        "total_credit_pnl": ("credit_pnl", "sum"), "total_liability_pnl": ("liability_pnl", "sum"),
        "total_fee_pnl": ("fee_pnl", "sum"), "net_pnl": ("net_pnl", "sum"),
        "avoided_liability_return_sum": ("avoided_liability_return", "sum"),
        "missed_credit_return_sum": ("missed_credit_return", "sum"),
        "saved_fee_return_sum": ("saved_fee_return", "sum"),
        **{name: (name, "mean") for name in conditional},
    }
    summary = working.groupby(ACCOUNT, sort=False).agg(**aggregation).reset_index()
    summary["width"], summary["fee_fraction"] = .01, .02
    summary["total_return"] = summary.ending_equity - 1
    summary["ruined"] = False
    summary["interpretation"] = "Hypothetical-credit terminal-liability proxy; not observed option profit"
    if not np.allclose(summary.net_pnl, summary.total_return, atol=1e-11, rtol=1e-9):
        raise ValueError("capital change differs from accumulated net P&L")
    return summary


def verify(risks, realized, backtest, calendar):
    """Reconstruct all four saved tables under the fixed default spread protocol."""
    if not isinstance(backtest, dict) or set(backtest) != {"positions", "outcomes", "path", "summary"}:
        raise ValueError("complete saved backtest tables required")
    r, y, metadata = _inputs(risks, realized, calendar)
    outcomes = _outcomes(y, metadata)
    _compare(backtest["outcomes"], outcomes, CASE, "outcomes")
    positions = _positions(r)
    _compare(backtest["positions"], positions, [*CASE, "model"], "positions")
    path = _paths(positions, outcomes)
    _compare(backtest["path"], path, [*ACCOUNT, "origin"], "path")
    summary = _summaries(path)
    _compare(backtest["summary"], summary, ACCOUNT, "summary")
    return {"status": "VERIFIED", "origins": len(y), "cases_per_origin": 9,
            "policies": 7, "credit_scenarios": 3, "positions": len(positions),
            "outcomes": len(outcomes), "paths": len(path), "summaries": len(summary),
            "scope": "Independent saved payoff, decision, account and summary reconstruction; hypothetical credits only"}
