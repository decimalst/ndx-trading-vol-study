"""Independent calendar, feature, QR-fit and forecast reconstruction.

Only standard numerical libraries are used. Source authentication and statistical
inference are separate boundaries; no producer routine establishes expectations.
"""

from __future__ import annotations

import bisect
import math

import numpy as np
import pandas as pd

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
EXTRA = ("claim_m4", "claim_age", "entry_dow_1", "entry_dow_2", "entry_dow_3", "entry_dow_4")
ALL = MARKET + EXTRA + ("claim_x",)
ARMS = {"market": MARKET, "matched": MARKET + EXTRA, "candidate": ALL}
_CEILING = pd.Timestamp("2025-10-20")
_APPLICATION_COLUMNS = (
    "origin",
    "fit_origin",
    "training_cutoff",
    "train_n",
    "train_releases",
    "claim_reference_week",
    "claim_release_date",
    "offset",
    "phase",
    "pred_market",
    "pred_matched",
    "pred_candidate",
)
_PANEL_COLUMNS = (
    "origin",
    "model",
    "prediction",
    "y",
    "loss",
    "target_end",
    "phase",
    "claim_reference_week",
    "claim_release_date",
    "offset",
    "fit_origin",
    "training_cutoff",
    "train_n",
    "train_releases",
)
_COVERAGE_COLUMNS = (
    "origin",
    "phase",
    "feature_complete",
    "missing_features",
    "claim_status",
    "claim_reference_week",
    "claim_release_date",
    "offset",
    "target_end",
    "target_observed",
    "target_within_phase",
    "scored",
    "status",
    "fit_origin",
    "training_cutoff",
)
_SCHEDULE_COLUMNS = (
    "month",
    "status",
    "fit_origin",
    "training_cutoff",
    "requested_n",
    "application_n",
    "train_n",
    "train_releases",
)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _calendar(index, allow_empty=False, ceiling=_CEILING):
    _need(isinstance(index, pd.DatetimeIndex), "Native date index required")
    _need(
        (allow_empty or len(index) > 0)
        and index.tz is None
        and not index.hasnans
        and index.is_unique
        and index.is_monotonic_increasing,
        "Invalid calendar identity",
    )
    _need(
        index.equals(index.normalize()) and bool((index <= ceiling).all()),
        "Calendar exceeds date/time fence",
    )


def _iso(value):
    _need(type(value) is str and len(value) == 10, "Literal ISO date required")
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Valid date required") from error
    _need(
        result.tz is None
        and result == result.normalize()
        and result.strftime("%Y-%m-%d") == value
        and result <= _CEILING,
        "Literal bounded date required",
    )
    return result


def _numeric_frame(frame, columns, allow_empty=False):
    _need(
        isinstance(frame, pd.DataFrame)
        and frame.columns.is_unique
        and set(frame.columns) == set(columns),
        "Exact input columns required",
    )
    _calendar(frame.index, allow_empty=allow_empty)
    for name in columns:
        dtype = frame[name].dtype
        _need(
            pd.api.types.is_numeric_dtype(dtype)
            and not pd.api.types.is_bool_dtype(dtype)
            and not pd.api.types.is_complex_dtype(dtype),
            "Real numeric dtype required",
        )
    values = frame.loc[:, columns].to_numpy(dtype=float, na_value=np.nan)
    _need(not np.isinf(values).any(), "Infinite source value")
    return values


def _logratio(numerator, denominator):
    if not (math.isfinite(numerator) and math.isfinite(denominator)):
        return math.nan
    return math.log(numerator) - math.log(denominator)


def _prior_z(values):
    result = np.full(len(values), np.nan)
    for i, current in enumerate(values):
        past = values[max(0, i - 252) : i]
        past = past[np.isfinite(past)]
        if math.isfinite(current) and len(past) >= 126:
            center = math.fsum(past) / len(past)
            scale = math.sqrt(
                math.fsum((float(v) - center) ** 2 for v in past) / (len(past) - 1)
            )
            if scale > 0:
                result[i] = (current - center) / scale
    return result


def _market_targets(daily, cross, iv):
    """Loop-based independent market history and exact five-position target."""
    raw = _numeric_frame(daily, ("open", "high", "low", "close", "adj close", "volume"))
    auxiliary = _numeric_frame(cross, ("hyg", "tlt", "gld", "uso", "uup"), allow_empty=True)
    implied = _numeric_frame(iv, ("vxn", "vix", "vix9d"), allow_empty=True)
    _need(
        bool(np.all(np.isnan(raw[:, :5]) | (raw[:, :5] > 0)))
        and bool(np.all(np.isnan(raw[:, 5]) | (raw[:, 5] >= 0))),
        "Invalid daily prices/volume",
    )
    _need(
        bool(np.all(np.isnan(auxiliary) | (auxiliary > 0)))
        and bool(np.all(np.isnan(implied) | (implied > 0))),
        "Nonpositive auxiliary price",
    )
    for lower, upper in ((2, 0), (2, 3), (2, 1), (0, 1), (3, 1)):
        valid = np.isfinite(raw[:, lower]) & np.isfinite(raw[:, upper])
        _need(
            bool((raw[valid, lower] <= raw[valid, upper]).all()), "Inconsistent OHLC geometry"
        )
    index, n = daily.index, len(daily)
    x = np.full((n, 13), np.nan)
    x[:, 0] = 1.0
    variance, overnight, returns = (np.full(n, np.nan) for _ in range(3))
    cross_map = {day: auxiliary[i] for i, day in enumerate(cross.index)}
    iv_map = {day: implied[i] for i, day in enumerate(iv.index)}
    cross_returns = np.full((n, 5), np.nan)
    for i in range(n):
        opened, high, low, closed, adjusted, _ = raw[i]
        if i:
            overnight[i] = _logratio(opened, raw[i - 1, 3]) ** 2
            returns[i] = _logratio(adjusted, raw[i - 1, 4])
        if np.isfinite(raw[i, :4]).all() and math.isfinite(overnight[i]):
            intraday = (
                0.5 * _logratio(high, low) ** 2
                - (2 * math.log(2) - 1) * _logratio(closed, opened) ** 2
            )
            variance[i] = max(intraday, 1e-10) + overnight[i]
        if i:
            before, now = cross_map.get(index[i - 1]), cross_map.get(index[i])
            if before is not None and now is not None:
                cross_returns[i] = [_logratio(a, b) for a, b in zip(now, before)]
            delayed = iv_map.get(index[i - 1])
            if delayed is not None:
                for col, value in ((7, delayed[0]), (8, delayed[1])):
                    if math.isfinite(value):
                        x[i, col] = math.log(value)
                x[i, 9] = _logratio(delayed[2], delayed[1])
        for col, window in ((1, 1), (2, 5), (3, 22)):
            if i + 1 >= window:
                v = variance[i + 1 - window : i + 1]
                r = returns[i + 1 - window : i + 1]
                if np.isfinite(v).all():
                    x[i, col] = math.log(math.fsum(v) / window)
                if np.isfinite(r).all():
                    x[i, col + 3] = min(math.fsum(r) / window, 0.0)
    zcross = np.column_stack([_prior_z(cross_returns[:, j]) for j in range(5)])
    log_volume = np.array(
        [math.log(v) if math.isfinite(v) and v > 0 else math.nan for v in raw[:, 5]]
    )
    share = np.array(
        [
            min(max(o / v, 0.0), 1.0) if math.isfinite(v) else math.nan
            for o, v in zip(overnight, variance)
        ]
    )
    zv, zo = _prior_z(log_volume), _prior_z(share)
    for i in range(n):
        if np.isfinite(zcross[i]).all():
            x[i, 10] = math.sqrt(math.fsum(zcross[i] ** 2) / 5)
        if math.isfinite(zv[i]) and math.isfinite(zo[i]):
            x[i, 11] = math.sqrt((max(zv[i], 0.0) ** 2 + max(zo[i], 0.0) ** 2) / 2)
    x[:, 12] = variance
    _need(not np.isinf(x).any(), "Invalid feature arithmetic")
    targets, ends = np.full(n, np.nan), [pd.NaT] * n
    for i in range(n - 5):
        ends[i] = index[i + 5]
        future = variance[i + 1 : i + 6]
        if np.isfinite(future).all():
            targets[i] = math.fsum(future) / 5
    market = pd.DataFrame(x, columns=MARKET + ("rv_total",), index=index)
    target = pd.DataFrame(
        {"y": targets, "target_end": pd.Series(ends, index=index, dtype="datetime64[ns]")},
        index=index,
    )
    return market, target


def _claim_frame(index, ledger_rows):
    _calendar(index)
    _need(type(ledger_rows) is list, "Ledger rows required")
    by_week, dated = {}, []
    for row in ledger_rows:
        _need(
            type(row) is dict
            and {"reference_week", "release_date", "first_report_value", "status"} <= set(row),
            "Required ledger fields missing",
        )
        week = _iso(row["reference_week"])
        _need(
            week.dayofweek == 5 and week not in by_week, "Unique reference Saturdays required"
        )
        status, value = row["status"], row["first_report_value"]
        _need(
            status in ("observed", "unresolved_correction", "no_admitted_release"),
            "Unknown ledger status",
        )
        if status == "observed":
            _need(type(value) is int and value > 0, "Positive admitted integer required")
        else:
            _need(value is None, "Unknown first value cannot be substituted")
        released = None if row["release_date"] is None else _iso(row["release_date"])
        _need(
            (released is None and status == "no_admitted_release")
            or (released is not None and released > week and status != "no_admitted_release"),
            "Invalid ledger release identity",
        )
        by_week[week] = (released, value, status)
        if released is not None:
            dated.append((released, week))
    dated.sort()
    clocks = [item[0] for item in dated]
    _need(len(set(clocks)) == len(clocks), "Duplicate known release date")
    rows = []
    for origin in index:
        values = {
            "claim_m4": np.nan,
            "claim_age": np.nan,
            **{f"entry_dow_{d}": float(origin.dayofweek == d) for d in range(1, 5)},
            "claim_x": np.nan,
            "claim_reference_week": pd.NaT,
            "claim_release_date": pd.NaT,
            "claim_age_days": np.nan,
            "claim_status": "no_prior_release",
        }
        selected = bisect.bisect_left(clocks, origin) - 1
        if selected >= 0:
            released, week = dated[selected]
            age = (origin - released).days
            values.update(
                claim_reference_week=week,
                claim_release_date=released,
                claim_age_days=float(age),
            )
            _, count, status = by_week[week]
            reason = "available"
            if age > 7:
                reason = "stale_release"
            elif status != "observed":
                reason = "latest_" + status
            prior_values = []
            if reason == "available":
                for lag in (1, 2, 3, 4):
                    prior = by_week.get(week - pd.Timedelta(days=lag * 7))
                    if prior is None:
                        reason = "missing_prior_week"
                    elif prior[0] is None:
                        reason = "prior_" + prior[2]
                    elif prior[0] >= origin:
                        reason = "prior_not_yet_released"
                    elif prior[2] != "observed":
                        reason = "prior_" + prior[2]
                    else:
                        prior_values.append(prior[1])
                        continue
                    break
            values["claim_status"] = reason
            if reason == "available":
                center = math.fsum(math.log(v) for v in prior_values) / 4
                values.update(
                    claim_m4=center, claim_age=age / 7, claim_x=math.log(count) - center
                )
        rows.append(values)
    return pd.DataFrame(rows, index=index)


def _fit_expected(train, y, application):
    _need(
        isinstance(train, pd.DataFrame) and isinstance(application, pd.DataFrame),
        "Design frames required",
    )
    for frame in (train, application):
        _need(
            frame.columns.is_unique and set(ALL) <= set(frame.columns),
            "Complete declared design required",
        )
        for name in ALL:
            dtype = frame[name].dtype
            _need(
                pd.api.types.is_numeric_dtype(dtype)
                and not pd.api.types.is_bool_dtype(dtype)
                and not pd.api.types.is_complex_dtype(dtype),
                "Real numeric design required",
            )
        matrix = frame.loc[:, ALL].to_numpy(dtype=float, na_value=np.nan)
        _need(
            np.isfinite(matrix).all() and (matrix[:, 0] == 1).all(),
            "Finite common design and exact constant required",
        )
    if isinstance(y, pd.Series):
        _need(y.index.equals(train.index), "Target alignment mismatch")
    target = np.asarray(y, dtype=float)
    _need(
        target.ndim == 1
        and len(target) == len(train)
        and len(train) > len(ALL)
        and np.isfinite(target).all()
        and (target > 0).all(),
        "Mature positive target vector required",
    )
    logged = np.log(target)
    predictions, audits = {}, {}
    for arm, columns in ARMS.items():
        raw = train.loc[:, columns].to_numpy(float)
        query = application.loc[:, columns].to_numpy(float)
        means = np.array([math.fsum(raw[:, j]) / len(raw) for j in range(1, len(columns))])
        scales = np.sqrt(np.mean((raw[:, 1:] - means) ** 2, axis=0))
        _need(
            np.isfinite(scales).all() and (scales > 1e-12).all(),
            "Declared feature scale failure",
        )
        design = np.column_stack((np.ones(len(raw)), (raw[:, 1:] - means) / scales))
        query_design = np.column_stack((np.ones(len(query)), (query[:, 1:] - means) / scales))
        singular = np.linalg.svd(design, compute_uv=False)
        rank = int(np.count_nonzero(singular > singular[0] * 1e-12))
        _need(rank == len(columns), "Declared design rank failure")
        q, r = np.linalg.qr(design, mode="reduced")
        try:
            beta = np.linalg.solve(r, q.T @ logged)
        except np.linalg.LinAlgError as error:
            raise ValueError("QR solve failed") from error
        residual = logged - design @ beta
        top = float(residual.max())
        smear = top + math.log(
            math.fsum(math.exp(float(e) - top) for e in residual) / len(residual)
        )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            forecast = np.exp(query_design @ beta + smear)
        _need(
            np.isfinite(beta).all()
            and math.isfinite(smear)
            and np.isfinite(forecast).all()
            and (forecast > 0).all(),
            "Invalid forecast/smearing arithmetic",
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


def _compare(actual, expected, path):
    if type(expected) is dict:
        _need(
            type(actual) is dict and set(actual) == set(expected), "Schema mismatch: " + path
        )
        for key, value in expected.items():
            _compare(actual[key], value, path + "." + key)
    elif type(expected) is list:
        _need(
            type(actual) is list and len(actual) == len(expected),
            "List identity mismatch: " + path,
        )
        for i, value in enumerate(expected):
            _compare(actual[i], value, f"{path}[{i}]")
    elif isinstance(expected, (bool, np.bool_)):
        _need(
            isinstance(actual, (bool, np.bool_)) and actual == expected,
            "Boolean mismatch: " + path,
        )
    elif isinstance(expected, (pd.Timestamp, np.datetime64)):
        _need(
            isinstance(actual, (pd.Timestamp, np.datetime64)) and actual == expected,
            "Date mismatch: " + path,
        )
    elif expected is None or pd.isna(expected):
        _need(actual is None or bool(pd.isna(actual)), "Missingness mismatch: " + path)
    elif isinstance(expected, str):
        _need(type(actual) is str and actual == expected, "Identity mismatch: " + path)
    elif isinstance(expected, (int, np.integer)):
        _need(
            isinstance(actual, (int, np.integer))
            and not isinstance(actual, (bool, np.bool_))
            and actual == expected,
            "Count mismatch: " + path,
        )
    else:
        _need(
            isinstance(actual, (float, int, np.floating, np.integer))
            and not isinstance(actual, (bool, np.bool_))
            and math.isfinite(actual),
            "Finite number required: " + path,
        )
        atol, rtol = (
            (1e-7, 1e-7)
            if ".coefficients." in path
            else (
                (1e-12, 1e-7) if ".pred_" in path or ".prediction" in path else (1e-12, 1e-9)
            )
        )
        _need(
            math.isclose(float(actual), float(expected), rel_tol=rtol, abs_tol=atol),
            "Numerical mismatch: " + path,
        )


def _frame(actual, rows, columns, name):
    _need(
        isinstance(actual, pd.DataFrame)
        and tuple(actual.columns) == columns
        and len(actual) == len(rows),
        "Frame schema/count mismatch: " + name,
    )
    for i, row in enumerate(rows):
        for col in columns:
            value = actual.iloc[i][col]
            expected = row[col]
            if col in ("train_n", "train_releases") and expected is not None:
                _need(
                    not isinstance(value, (bool, np.bool_))
                    and pd.notna(value)
                    and value == expected,
                    "Frame count mismatch: " + name,
                )
            else:
                _compare(value, expected, f"{name}[{i}].{col}")


def verify_forecasts(daily, cross, iv, ledger_rows, config, produced):
    """Reconstruct all applications, fits, target scoring and calendar coverage."""
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
            "minimum_train_releases",
        },
        "Exact scheduling config required",
    )
    origin_start, origin_end, source_end = (
        _iso(config[k]) for k in ("origin_start", "origin_end", "source_end")
    )
    _need(
        origin_start <= origin_end <= source_end, "Ordered source/application fences required"
    )
    phase_bounds = {}
    for name in ("development", "evaluation"):
        bounds = config[name]
        _need(type(bounds) is list and len(bounds) == 2, "Phase pair required")
        a, b = map(_iso, bounds)
        _need(a <= b <= source_end, "Invalid phase bounds")
        phase_bounds[name] = (a, b)
    _need(phase_bounds["development"][1] < phase_bounds["evaluation"][0], "Overlapping phases")
    for name in ("minimum_train", "minimum_train_releases"):
        _need(
            type(config[name]) is int and config[name] > 0, "Positive support floor required"
        )
    for table in (daily, cross, iv):
        _calendar(table.index, allow_empty=table is not daily, ceiling=source_end)
    _need(
        type(produced) is dict
        and set(produced) == {"applications", "panel", "coverage", "schedules", "fits"},
        "Exact produced payload required",
    )
    market, targets = _market_targets(daily, cross, iv)
    claims = _claim_frame(daily.index, ledger_rows)
    features = pd.concat((market.loc[:, MARKET], claims), axis=1)
    matrix = features.loc[:, ALL].to_numpy(float)
    complete = np.isfinite(matrix).all(axis=1)
    index = daily.index
    requested_positions = [
        i for i, day in enumerate(index) if origin_start <= day <= origin_end
    ]

    def phase(day):
        for name, (a, b) in phase_bounds.items():
            if a <= day <= b:
                return name
        return "outside_phase"

    fits, applications, schedules, app_by_position = [], [], [], {}
    for period in pd.period_range(origin_start, origin_end, freq="M"):
        positions = [i for i in requested_positions if index[i].to_period("M") == period]
        apply_positions = [i for i in positions if complete[i]]
        schedule = {
            "month": str(period),
            "status": "no_requested_origins" if not positions else "no_complete_origin",
            "fit_origin": pd.NaT,
            "training_cutoff": pd.NaT,
            "requested_n": len(positions),
            "application_n": len(apply_positions),
            "train_n": None,
            "train_releases": None,
        }
        if apply_positions:
            fit_position = apply_positions[0]
            _need(fit_position > 0, "Prior observed training cutoff required")
            fit_origin, cutoff = index[fit_position], index[fit_position - 1]
            train_positions = [
                i
                for i in range(fit_position)
                if complete[i]
                and pd.notna(targets.target_end.iloc[i])
                and targets.target_end.iloc[i] <= cutoff
                and math.isfinite(targets.y.iloc[i])
                and targets.y.iloc[i] > 0
            ]
            release_count = len(set(claims.claim_release_date.iloc[train_positions]))
            _need(
                len(train_positions) >= config["minimum_train"]
                and release_count >= config["minimum_train_releases"],
                "Monthly common training support failure",
            )
            prediction, audit = _fit_expected(
                features.iloc[train_positions],
                targets.y.iloc[train_positions],
                features.iloc[apply_positions],
            )
            fits.append(
                {
                    "month": str(period),
                    "fit_origin": fit_origin.strftime("%Y-%m-%d"),
                    "training_cutoff": cutoff.strftime("%Y-%m-%d"),
                    "train_origins": [index[i].strftime("%Y-%m-%d") for i in train_positions],
                    "application_origins": [
                        index[i].strftime("%Y-%m-%d") for i in apply_positions
                    ],
                    "train_n": len(train_positions),
                    "train_releases": release_count,
                    "model_audits": audit,
                }
            )
            schedule.update(
                status="fitted",
                fit_origin=fit_origin,
                training_cutoff=cutoff,
                train_n=len(train_positions),
                train_releases=release_count,
            )
            for j, i in enumerate(apply_positions):
                app = {
                    "origin": index[i],
                    "fit_origin": fit_origin,
                    "training_cutoff": cutoff,
                    "train_n": len(train_positions),
                    "train_releases": release_count,
                    "claim_reference_week": claims.claim_reference_week.iloc[i],
                    "claim_release_date": claims.claim_release_date.iloc[i],
                    "offset": i % 5,
                    "phase": phase(index[i]),
                    **{"pred_" + arm: float(prediction[arm][j]) for arm in ARMS},
                }
                applications.append(app)
                app_by_position[i] = app
        schedules.append(schedule)
    coverage, panel = [], []
    for i in requested_positions:
        origin, assigned = index[i], phase(index[i])
        endpoint, value = targets.target_end.iloc[i], targets.y.iloc[i]
        observed = bool(math.isfinite(value) and value > 0)
        within = bool(
            assigned != "outside_phase"
            and pd.notna(endpoint)
            and endpoint
            <= (phase_bounds["development"][1] if assigned == "development" else source_end)
        )
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
        app = app_by_position.get(i)
        coverage.append(
            {
                "origin": origin,
                "phase": assigned,
                "feature_complete": bool(complete[i]),
                "missing_features": "|".join(
                    name for j, name in enumerate(ALL) if not math.isfinite(matrix[i, j])
                ),
                "claim_status": claims.claim_status.iloc[i],
                "claim_reference_week": claims.claim_reference_week.iloc[i],
                "claim_release_date": claims.claim_release_date.iloc[i],
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
                _need(math.isfinite(ratio) and ratio > 0, "Invalid QLIKE ratio")
                loss = ratio - math.log(ratio) - 1
                _need(math.isfinite(loss), "Invalid QLIKE loss")
                panel.append(
                    {
                        "origin": origin,
                        "model": arm,
                        "prediction": forecast,
                        "y": float(value),
                        "loss": loss,
                        "target_end": endpoint,
                        "phase": assigned,
                        "claim_reference_week": app["claim_reference_week"],
                        "claim_release_date": app["claim_release_date"],
                        "offset": i % 5,
                        "fit_origin": app["fit_origin"],
                        "training_cutoff": app["training_cutoff"],
                        "train_n": app["train_n"],
                        "train_releases": app["train_releases"],
                    }
                )
    _compare(produced["fits"], fits, "fits")
    for key, rows, columns in (
        ("applications", applications, _APPLICATION_COLUMNS),
        ("panel", panel, _PANEL_COLUMNS),
        ("coverage", coverage, _COVERAGE_COLUMNS),
        ("schedules", schedules, _SCHEDULE_COLUMNS),
    ):
        _frame(produced[key], rows, columns, key)
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
