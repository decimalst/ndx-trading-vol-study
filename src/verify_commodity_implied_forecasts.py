"""Independent commodity features, common calendar cohorts, QR fits and outputs."""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from src.verify_claims_release_forecasts import _compare, _market_targets

MARKET = (
    "const",
    "lrv_d",
    "lrv_w",
    "lrv_m",
    "lev_d",
    "lev_w",
    "lev_m",
    "liv",
    "lvix",
    "term",
    "xasset_stress",
    "market_stress",
)
HISTORY = (
    "uso_ret",
    "uso_r2",
    "uso_lrv5",
    "uso_lrv22",
    "gld_ret",
    "gld_r2",
    "gld_lrv5",
    "gld_lrv22",
)
ALL = MARKET + HISTORY + ("lovx", "lgvz")
ARMS = {"market": MARKET, "matched": MARKET + HISTORY, "candidate": ALL}
_FLOOR, _CEILING = pd.Timestamp("2009-01-02"), pd.Timestamp("2025-10-20")
_CROSS = ("hyg", "tlt", "gld", "uso", "uup")
_APPLICATION = (
    "origin",
    "fit_origin",
    "training_cutoff",
    "train_n",
    "commodity_cutoff_date",
    "offset",
    "phase",
    "pred_market",
    "pred_matched",
    "pred_candidate",
)
_PANEL = (
    "origin",
    "model",
    "prediction",
    "y",
    "loss",
    "target_end",
    "phase",
    "commodity_cutoff_date",
    "offset",
    "fit_origin",
    "training_cutoff",
    "train_n",
)
_COVERAGE = (
    "origin",
    "phase",
    "feature_complete",
    "missing_features",
    "commodity_cutoff_date",
    "offset",
    "target_end",
    "target_observed",
    "target_within_phase",
    "scored",
    "status",
    "fit_origin",
    "training_cutoff",
)
_SCHEDULE = (
    "month",
    "status",
    "fit_origin",
    "training_cutoff",
    "requested_n",
    "application_n",
    "train_n",
)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _calendar(index, ceiling=_CEILING, empty=True):
    _need(isinstance(index, pd.DatetimeIndex), "Native calendar required")
    _need(
        (empty or len(index) > 0)
        and index.tz is None
        and not index.hasnans
        and index.is_unique
        and index.is_monotonic_increasing,
        "Invalid calendar identity",
    )
    _need(index.equals(index.normalize()), "Midnight calendar required")
    try:
        index.as_unit("ns")
    except (ValueError, OverflowError) as error:
        raise ValueError("Nanosecond-representable dates required") from error
    _need(
        bool((index <= ceiling).all()) and bool((index <= _CEILING).all()),
        "Calendar exceeds source ceiling",
    )


def _schema(frame, names, ceiling=_CEILING, empty=True):
    _need(
        isinstance(frame, pd.DataFrame)
        and frame.columns.is_unique
        and set(frame.columns) == set(names),
        "Exact input schema required",
    )
    _calendar(frame.index, ceiling, empty)


def _iso(value):
    _need(
        type(value) is str and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value) is not None,
        "Literal ISO config date required",
    )
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid config date") from error
    _need(result <= _CEILING, "Config exceeds fixed source ceiling")
    return result


def _config(config):
    if config is None:
        config = {
            "origin_start": "2016-01-01",
            "origin_end": "2025-10-20",
            "development": ["2016-01-01", "2019-12-31"],
            "evaluation": ["2020-01-01", "2025-10-20"],
            "source_end": "2025-10-20",
            "minimum_train": 1000,
        }
    _need(
        type(config) is dict
        and set(config)
        == {
            "origin_start",
            "origin_end",
            "development",
            "evaluation",
            "source_end",
            "minimum_train",
        },
        "Exact scheduling configuration required",
    )
    result = {
        name: _iso(config[name]) for name in ("origin_start", "origin_end", "source_end")
    }
    for phase in ("development", "evaluation"):
        _need(
            type(config[phase]) is list and len(config[phase]) == 2, "Phase date pair required"
        )
        result[phase] = tuple(_iso(value) for value in config[phase])
    a, b = result["development"]
    c, d = result["evaluation"]
    _need(
        _FLOOR
        <= result["origin_start"]
        <= a
        <= b
        < c
        <= d
        <= result["origin_end"]
        <= result["source_end"],
        "Ordered bounded phases and origin window required",
    )
    _need(
        type(config["minimum_train"]) is int and config["minimum_train"] > 0,
        "Positive exact training floor required",
    )
    result["minimum_train"] = config["minimum_train"]
    return result


def _real_dtype(series):
    dtype = series.dtype
    _need(
        (pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype))
        and not pd.api.types.is_bool_dtype(dtype),
        "Real nonboolean numeric columns required",
    )


def _square(value):
    if math.isnan(value):
        return math.nan
    _need(math.isfinite(value), "Nonfinite commodity return")
    try:
        with np.errstate(over="raise", under="raise", invalid="raise"):
            result = float(np.float64(value) * np.float64(value))
    except FloatingPointError as error:
        raise ValueError("Nonzero square cannot be represented") from error
    _need(
        math.isfinite(result) and (value == 0 or result > 0),
        "Invalid squared-return arithmetic",
    )
    return result


def _commodity_features(index, cross, commodity_iv):
    """Independent scalar history reconstruction on the unchanged QQQ calendar."""
    _calendar(index)
    _schema(cross, _CROSS)
    _schema(commodity_iv, ("OVX", "GVZ"))
    aligned = []
    for frame, names in ((cross, ("uso", "gld")), (commodity_iv, ("OVX", "GVZ"))):
        for name in names:
            _real_dtype(frame[name])
        table = frame.loc[:, names].reindex(index).copy()
        table.loc[index < _FLOOR] = np.nan
        numbers = table.to_numpy(dtype=float, na_value=np.nan)
        _need(
            not np.isinf(numbers).any() and bool(np.all(np.isnan(numbers) | (numbers > 0))),
            "Positive finite commodity observations required",
        )
        aligned.append(numbers)
    prices, implied = aligned
    n = len(index)
    values = np.full((n, 10), np.nan)
    for asset in range(2):
        returns = np.full(n, np.nan)
        squared = np.full(n, np.nan)
        for i in range(2, n):
            now, before = prices[i - 1, asset], prices[i - 2, asset]
            if math.isfinite(now) and math.isfinite(before):
                returns[i] = math.log(now) - math.log(before)
                squared[i] = _square(float(returns[i]))
        values[:, 4 * asset] = returns
        values[:, 4 * asset + 1] = squared
        for i in range(n):
            for column, window in ((2, 5), (3, 22)):
                if i + 1 < window:
                    continue
                history = squared[i + 1 - window : i + 1]
                if not np.isfinite(history).all():
                    continue
                try:
                    total = math.fsum(history)
                    with np.errstate(over="raise", under="raise", invalid="raise"):
                        mean = float(np.float64(total) / window)
                except (OverflowError, FloatingPointError) as error:
                    raise ValueError("Invalid strict variance mean arithmetic") from error
                _need(
                    math.isfinite(mean) and (total == 0 or mean > 0),
                    "Invalid strict variance mean",
                )
                if mean > 0:
                    values[i, 4 * asset + column] = math.log(mean)
    for i in range(1, n):
        for column in range(2):
            number = implied[i - 1, column]
            if math.isfinite(number):
                values[i, 8 + column] = 2 * (math.log(number) - math.log(100)) - math.log(252)
    _need(not np.isinf(values).any(), "Invalid commodity feature arithmetic")
    frame = pd.DataFrame(values, index=index, columns=HISTORY + ("lovx", "lgvz"))
    frame["commodity_cutoff_date"] = (
        pd.Series(
            [pd.NaT] + [day if day >= _FLOOR else pd.NaT for day in index[:-1]],
            index=index,
            dtype="datetime64[ns]",
        )
        if n
        else pd.Series(index=index, dtype="datetime64[ns]")
    )
    return frame


def _fit_expected(train, y, application):
    """Three independent QR fits with arm-specific residual smearing and audits."""
    for frame in (train, application):
        _need(
            isinstance(frame, pd.DataFrame)
            and frame.index.is_unique
            and frame.columns.is_unique
            and set(ALL) <= set(frame.columns),
            "Unique common22 design required",
        )
        for name in ALL:
            _real_dtype(frame[name])
        values = frame.loc[:, ALL].to_numpy(dtype=float, na_value=np.nan)
        _need(
            np.isfinite(values).all() and bool((values[:, 0] == 1).all()),
            "Finite common22 rows and unit constant required",
        )
    if isinstance(y, pd.Series):
        _need(y.index.equals(train.index), "Target index mismatch")
        _real_dtype(y)
    target = np.asarray(y)
    _need(
        target.dtype.kind in "fiu"
        and target.ndim == 1
        and len(target) == len(train)
        and len(train) >= len(ALL),
        "Positive aligned target vector and supported design required",
    )
    target = target.astype(float)
    _need(
        np.isfinite(target).all() and bool((target > 0).all()),
        "Positive finite target required",
    )
    logged = np.log(target)
    predictions, audits = {}, {}
    for arm, columns in ARMS.items():
        raw, query = (
            frame.loc[:, columns].to_numpy(dtype=float) for frame in (train, application)
        )
        means = np.array([math.fsum(raw[:, j]) / len(raw) for j in range(1, len(columns))])
        scales = np.sqrt(np.mean((raw[:, 1:] - means) ** 2, axis=0))
        _need(
            np.isfinite(means).all()
            and np.isfinite(scales).all()
            and bool((scales > 1e-12).all()),
            "Fixed feature scale failure",
        )
        design = np.column_stack((np.ones(len(raw)), (raw[:, 1:] - means) / scales))
        query_design = np.column_stack((np.ones(len(query)), (query[:, 1:] - means) / scales))
        _need(
            np.isfinite(design).all() and np.isfinite(query_design).all(),
            "Invalid scaled design",
        )
        try:
            singular = np.linalg.svd(design, compute_uv=False)
            rank = int(np.count_nonzero(singular > singular[0] * 1e-12))
            _need(rank == len(columns), "Fixed model rank failure")
            q, r = np.linalg.qr(design, mode="reduced")
            beta = np.linalg.solve(r, q.T @ logged)
        except np.linalg.LinAlgError as error:
            raise ValueError("Independent QR fit failed") from error
        residual = logged - design @ beta
        top = float(residual.max())
        smear = top + math.log(
            math.fsum(math.exp(float(v) - top) for v in residual) / len(raw)
        )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            forecast = np.exp(query_design @ beta + smear)
        _need(
            np.isfinite(beta).all()
            and math.isfinite(smear)
            and np.isfinite(forecast).all()
            and bool((forecast > 0).all()),
            "Invalid forecast or smearing arithmetic",
        )
        predictions[arm] = forecast
        audits[arm] = {
            "feature_names": list(columns),
            "coefficients": dict(zip(columns, map(float, beta))),
            "feature_means": dict(zip(columns[1:], map(float, means))),
            "feature_scales": dict(zip(columns[1:], map(float, scales))),
            "singular_values": list(map(float, singular)),
            "rank": rank,
            "rank_relative_cutoff": 1e-12,
            "log_smearing": smear,
            "normal_equation_max_abs": float(np.abs(design.T @ residual / len(raw)).max()),
            "n_train": len(raw),
            "n_application": len(query),
        }
    return predictions, audits


def _frame(actual, rows, columns, name):
    _need(
        isinstance(actual, pd.DataFrame)
        and tuple(actual.columns) == columns
        and len(actual) == len(rows),
        "Frame schema/count mismatch: " + name,
    )
    for i, row in enumerate(rows):
        for column in columns:
            _compare(actual.iloc[i][column], row[column], f"{name}[{i}].{column}")


def verify_forecasts(daily, cross, iv, commodity_iv, config, produced):
    """Check every successful application, fit, score, schedule and coverage row."""
    cfg = _config(config)
    for table, names in (
        (daily, ("open", "high", "low", "close", "adj close", "volume")),
        (cross, _CROSS),
        (iv, ("vxn", "vix", "vix9d")),
        (commodity_iv, ("OVX", "GVZ")),
    ):
        _schema(table, names, cfg["source_end"], empty=table is not daily)
    _need(
        type(produced) is dict
        and set(produced) == {"applications", "panel", "coverage", "schedules", "fits"},
        "Exact produced payload required",
    )
    market, targets = _market_targets(daily, cross, iv)
    commodity = _commodity_features(daily.index, cross, commodity_iv)
    features = pd.concat([market.loc[:, MARKET], commodity], axis=1)
    matrix = features.loc[:, ALL].to_numpy(float)
    _need(not np.isinf(matrix).any(), "Invalid reconstructed feature arithmetic")
    complete = np.isfinite(matrix).all(axis=1)
    index = daily.index
    cutoff_dates = commodity.commodity_cutoff_date
    _need(
        not bool(np.any(complete & cutoff_dates.isna())),
        "Complete row without commodity cutoff",
    )
    requested = [
        i for i, day in enumerate(index) if cfg["origin_start"] <= day <= cfg["origin_end"]
    ]

    def phase(day):
        for name in ("development", "evaluation"):
            if cfg[name][0] <= day <= cfg[name][1]:
                return name
        return "outside_phase"

    fits, schedules, applications, by_position = [], [], [], {}
    for month in pd.period_range(cfg["origin_start"], cfg["origin_end"], freq="M"):
        positions = [i for i in requested if index[i].to_period("M") == month]
        query_positions = [i for i in positions if complete[i]]
        schedule = {
            "month": str(month),
            "status": "no_requested_origins" if not positions else "no_complete_origin",
            "fit_origin": pd.NaT,
            "training_cutoff": pd.NaT,
            "requested_n": len(positions),
            "application_n": len(query_positions),
            "train_n": None,
        }
        if query_positions:
            first = query_positions[0]
            _need(first > 0, "Previous full-session cutoff required")
            origin, cutoff = index[first], index[first - 1]
            train_positions = [
                i
                for i in range(first)
                if complete[i]
                and pd.notna(targets.target_end.iloc[i])
                and targets.target_end.iloc[i] <= cutoff
                and math.isfinite(targets.y.iloc[i])
                and targets.y.iloc[i] > 0
            ]
            _need(
                len(train_positions) >= cfg["minimum_train"],
                "INSUFFICIENT_DATA: monthly common training floor",
            )
            prediction, audit = _fit_expected(
                features.iloc[train_positions],
                targets.y.iloc[train_positions],
                features.iloc[query_positions],
            )
            fits.append(
                {
                    "month": str(month),
                    "fit_origin": origin.strftime("%Y-%m-%d"),
                    "training_cutoff": cutoff.strftime("%Y-%m-%d"),
                    "train_origins": [index[i].strftime("%Y-%m-%d") for i in train_positions],
                    "application_origins": [
                        index[i].strftime("%Y-%m-%d") for i in query_positions
                    ],
                    "train_n": len(train_positions),
                    "model_audits": audit,
                }
            )
            schedule.update(
                status="fitted",
                fit_origin=origin,
                training_cutoff=cutoff,
                train_n=len(train_positions),
            )
            for j, i in enumerate(query_positions):
                app = {
                    "origin": index[i],
                    "fit_origin": origin,
                    "training_cutoff": cutoff,
                    "train_n": len(train_positions),
                    "commodity_cutoff_date": cutoff_dates.iloc[i],
                    "offset": i % 5,
                    "phase": phase(index[i]),
                    **{"pred_" + arm: float(prediction[arm][j]) for arm in ARMS},
                }
                applications.append(app)
                by_position[i] = app
        schedules.append(schedule)
    coverage, panel = [], []
    for i in requested:
        day, assigned = index[i], phase(index[i])
        endpoint, value = targets.target_end.iloc[i], targets.y.iloc[i]
        observed = bool(math.isfinite(value) and value > 0)
        fence = cfg["development"][1] if assigned == "development" else cfg["source_end"]
        within = bool(assigned != "outside_phase" and pd.notna(endpoint) and endpoint <= fence)
        if not complete[i]:
            status = "incomplete_features"
        elif assigned == "outside_phase":
            status = "outside_phase"
        elif pd.isna(endpoint):
            status = "target_not_mature"
        elif not observed:
            status = "missing_target"
        elif not within:
            status = "target_after_phase_cutoff"
        else:
            status = "scored"
        app = by_position.get(i)
        coverage.append(
            {
                "origin": day,
                "phase": assigned,
                "feature_complete": bool(complete[i]),
                "missing_features": "|".join(
                    name for j, name in enumerate(ALL) if not math.isfinite(matrix[i, j])
                ),
                "commodity_cutoff_date": cutoff_dates.iloc[i],
                "offset": i % 5,
                "target_end": endpoint,
                "target_observed": observed,
                "target_within_phase": within,
                "scored": status == "scored",
                "status": status,
                "fit_origin": pd.NaT if app is None else app["fit_origin"],
                "training_cutoff": pd.NaT if app is None else app["training_cutoff"],
            }
        )
        if status == "scored":
            for arm in ARMS:
                forecast = app["pred_" + arm]
                ratio = value / forecast
                _need(math.isfinite(ratio) and ratio > 0, "Invalid positive QLIKE ratio")
                loss = ratio - math.log(ratio) - 1
                _need(math.isfinite(loss), "Invalid QLIKE loss")
                panel.append(
                    {
                        "origin": day,
                        "model": arm,
                        "prediction": forecast,
                        "y": float(value),
                        "loss": loss,
                        "target_end": endpoint,
                        "phase": assigned,
                        "commodity_cutoff_date": cutoff_dates.iloc[i],
                        "offset": i % 5,
                        "fit_origin": app["fit_origin"],
                        "training_cutoff": app["training_cutoff"],
                        "train_n": app["train_n"],
                    }
                )
    _compare(produced["fits"], fits, "fits")
    for name, rows, columns in (
        ("applications", applications, _APPLICATION),
        ("panel", panel, _PANEL),
        ("coverage", coverage, _COVERAGE),
        ("schedules", schedules, _SCHEDULE),
    ):
        _frame(produced[name], rows, columns, name)
    return {
        "status": "VERIFIED",
        "application_origins": len(applications),
        "application_forecasts_verified": 3 * len(applications),
        "scored_origins": len(panel) // 3,
        "forecasts_verified": len(panel),
        "monthly_fits": len(fits),
        "model_fits": 3 * len(fits),
        "coverage_origins": len(coverage),
        "calendar_rows": len(index),
    }
