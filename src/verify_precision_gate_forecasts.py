"""Independent source, expert-issuance and convex precision-gate verification."""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
from scipy.optimize import minimize

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
ARMS = ("base", "adaptive", "constant", "contextual")
_SHARED = (
    "fit_origin",
    "training_cutoff",
    "train_n",
    "gate_n",
    "gate_status",
    "state",
    "offset",
    "phase",
)
_APPLICATION = ("origin", *_SHARED, *("pred_" + name for name in ARMS))
_PANEL = ("origin", "model", "prediction", "y", "loss", "target_end", *_SHARED)
_COVERAGE = (
    "origin",
    "phase",
    "feature_complete",
    "missing_features",
    "state",
    "offset",
    "target_end",
    "target_observed",
    "target_within_phase",
    "issued",
    "scored",
    "status",
    "fit_origin",
    "training_cutoff",
    "gate_status",
)
_SCHEDULE = (
    "month",
    "status",
    "fit_origin",
    "training_cutoff",
    "requested_n",
    "feature_complete_n",
    "application_n",
    "train_n",
    "gate_n",
    "gate_status",
)
_CEILING = pd.Timestamp("2025-10-20")
_FLOOR = pd.Timestamp("2010-01-04")
_CROSS = ("hyg", "tlt", "gld", "uso", "uup")


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


def _real_dtype(series):
    dtype = series.dtype
    _need(
        (pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype))
        and not pd.api.types.is_bool_dtype(dtype),
        "Real nonboolean numeric columns required",
    )


def _gate_data(base, adaptive, target, state):
    arrays = []
    for i, value in enumerate((base, adaptive, target, state)):
        raw = np.asarray(value)
        _need(raw.ndim == 1 and raw.dtype.kind in "fiu", "Real aligned gate vectors required")
        try:
            converted = raw.astype(float, copy=True)
        except (ValueError, OverflowError) as error:
            raise ValueError("Unrepresentable gate observations") from error
        _need(
            np.isfinite(converted).all() and (i == 3 or bool((converted > 0).all())),
            "Finite positive gate observations required",
        )
        arrays.append(converted)
    _need(len({len(x) for x in arrays}) == 1, "Aligned gate observations required")
    b, a, y, s = arrays
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        relative_y, relative_a = y / b, b / a
    _need(
        np.isfinite(relative_y).all()
        and (relative_y > 0).all()
        and np.isfinite(relative_a).all()
        and (relative_a > 0).all(),
        "Positive representable gate ratios required",
    )
    z = np.tanh(s)
    return b, relative_y, relative_a, np.column_stack(((1 - z) / 2, (1 + z) / 2))


def _coefficients(value):
    raw = np.asarray(value)
    _need(
        raw.shape == (2,) and raw.dtype.kind in "fiu", "Two numeric gate coefficients required"
    )
    result = raw.astype(float)
    _need(
        np.isfinite(result).all() and ((result >= 0) & (result <= 1)).all(),
        "Gate coefficients outside box",
    )
    return result


def _gate_moments(coefficients, relative_y, ratio, design):
    c = _coefficients(coefficients)
    weight = design @ c
    _need(
        np.isfinite(weight).all() and ((weight >= 0) & (weight <= 1)).all(),
        "Invalid convex mixture weights",
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        precision = (1 - weight) + weight * ratio
    _need(np.isfinite(precision).all() and (precision > 0).all(), "Invalid relative precision")
    _need(len(precision) > 0, "Nonempty gate fit required")
    try:
        terms = relative_y * precision - np.log(precision)
        value = math.fsum(float(v) for v in terms) / len(terms) + 0.005 * math.fsum(
            float(v * v) for v in c
        )
        derivative = (relative_y - 1 / precision) * (ratio - 1)
        gradient = np.array(
            [
                math.fsum(float(d * x) for d, x in zip(derivative, design[:, j])) / len(terms)
                + 0.01 * c[j]
                for j in range(2)
            ]
        )
    except (OverflowError, ValueError, FloatingPointError) as error:
        raise ValueError("Invalid gate objective arithmetic") from error
    _need(
        math.isfinite(value) and np.isfinite(gradient).all(),
        "Finite gate objective/gradient required",
    )
    return value, gradient


def _gate_prediction(base, adaptive, state, coefficients):
    b, _, ratio, design = _gate_data(base, adaptive, base, state)
    c = _coefficients(coefficients)
    weight = design @ c
    _need(
        np.isfinite(weight).all() and ((weight >= 0) & (weight <= 1)).all(),
        "Invalid precision weight",
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        precision = (1 - weight) + weight * ratio
        result = b / precision
    _need(
        np.isfinite(precision).all()
        and (precision > 0).all()
        and np.isfinite(result).all()
        and (result > 0).all(),
        "Positive finite mixture prediction required",
    )
    return result


def _verify_gate(base, adaptive, y, state, saved, constant=False):
    """Certify saved coefficients with literal moments and a separate SLSQP solve."""
    _need(type(constant) is bool, "Exact constant flag required")
    b, relative_y, ratio, design = _gate_data(base, adaptive, y, state)
    _need(len(b) > 0, "Nonempty gate training records required")
    keys = {
        "coefficients",
        "constant",
        "objective",
        "gradient",
        "optimization_gradient",
        "projected_gradient_max_abs",
        "n_train",
        "n_iterations",
        "solver",
        "success",
    }
    _need(type(saved) is dict and set(saved) == keys, "Exact fitted gate audit required")
    _need(
        type(saved["success"]) is bool and saved["success"] and saved["solver"] == "L-BFGS-B",
        "Successful declared producer solver required",
    )
    _need(
        type(saved["n_iterations"]) is int and 0 <= saved["n_iterations"] <= 1000,
        "Producer iteration count outside declared range",
    )
    _compare(saved["constant"], constant, "gate.constant")
    _compare(saved["n_train"], len(b), "gate.n_train")
    c = _coefficients(saved["coefficients"])
    _need(not constant or c[0] == c[1], "Constant gate must be exactly nested")
    objective, full_gradient = _gate_moments(c, relative_y, ratio, design)
    theta = c[:1] if constant else c
    gradient = np.array([math.fsum(full_gradient)]) if constant else full_gradient
    projected = theta - np.minimum(1, np.maximum(0, theta - gradient))
    norm = float(np.abs(projected).max())
    _need(norm <= 1e-8, "Saved gate violates projected stationarity")
    _compare(saved["objective"], objective, "gate.objective")
    _compare(saved["gradient"], full_gradient.tolist(), "gate.gradient")
    _compare(saved["optimization_gradient"], gradient.tolist(), "gate.optimization_gradient")
    _compare(saved["projected_gradient_max_abs"], norm, "gate.projected_gradient_max_abs")

    def moments(theta):
        coeff = np.repeat(theta, 2) if constant else theta
        value, grad = _gate_moments(coeff, relative_y, ratio, design)
        return value, np.array([math.fsum(grad)]) if constant else grad

    independent = minimize(
        moments,
        np.full(1 if constant else 2, 0.5),
        method="SLSQP",
        jac=True,
        bounds=[(0.0, 1.0)] * (1 if constant else 2),
        options={"ftol": 1e-14, "maxiter": 2000},
    )
    _need(independent.success, "Independent SLSQP gate certification failed")
    optimum = _coefficients(np.repeat(independent.x, 2) if constant else independent.x)
    value, derivative = moments(independent.x)
    residual = independent.x - np.minimum(1, np.maximum(0, independent.x - derivative))
    _need(
        np.isfinite(residual).all() and np.abs(residual).max() <= 1e-7,
        "Independent gate optimum fails stationarity",
    )
    _need(
        np.allclose(c, optimum, atol=5e-6, rtol=1e-7),
        "Saved gate differs from independent optimum",
    )
    _need(
        math.isclose(objective, value, abs_tol=1e-10, rel_tol=1e-9),
        "Independent gate objective mismatch",
    )
    return c.copy()


def _expert_expected(train, y, application, train_positions, cutoff_position, half_life=None):
    """Unweighted/weighted QR with full-calendar weights and exact Duan smearing."""
    for frame in (train, application):
        _need(
            isinstance(frame, pd.DataFrame)
            and frame.columns.is_unique
            and frame.index.is_unique
            and set(MARKET) <= set(frame),
            "Common12 expert design required",
        )
        for name in MARKET:
            _real_dtype(frame[name])
        numbers = frame.loc[:, MARKET].to_numpy(dtype=float, na_value=np.nan)
        _need(
            np.isfinite(numbers).all() and (numbers[:, 0] == 1).all(),
            "Finite expert design and exact unit constant required",
        )
    if isinstance(y, pd.Series):
        _need(y.index.equals(train.index), "Expert target alignment required")
    target = np.asarray(y)
    _need(
        target.ndim == 1
        and target.dtype.kind in "fiu"
        and len(target) == len(train)
        and len(train) >= 12,
        "Aligned expert target vector required",
    )
    target = target.astype(float)
    _need(np.isfinite(target).all() and (target > 0).all(), "Positive expert target required")
    positions = np.asarray(train_positions)
    _need(
        positions.ndim == 1
        and positions.dtype.kind in "iu"
        and len(positions) == len(train)
        and (positions >= 0).all()
        and all(int(a) < int(b) for a, b in zip(positions[:-1], positions[1:])),
        "Unique ascending original calendar positions required",
    )
    _need(
        type(cutoff_position) is int
        or (
            isinstance(cutoff_position, np.integer)
            and not isinstance(cutoff_position, np.bool_)
        ),
        "Exact cutoff position required",
    )
    _need(bool((positions <= cutoff_position).all()), "Training position after cutoff")
    if half_life is None:
        weights = np.ones(len(train))
    else:
        _need(
            type(half_life) in (int, float) and math.isfinite(half_life) and half_life > 0,
            "Positive half life required",
        )
        weights = np.array(
            [math.exp2(-(int(cutoff_position) - int(i)) / half_life) for i in positions]
        )
    _need(
        np.isfinite(weights).all() and (weights > 0).all(),
        "Strictly positive representable historical weights required",
    )
    total = math.fsum(weights)
    raw, query = [frame.loc[:, MARKET].to_numpy(float) for frame in (train, application)]
    means = np.array(
        [
            math.fsum(float(w * v) for w, v in zip(weights, raw[:, j])) / total
            for j in range(1, 12)
        ]
    )
    scales = np.array(
        [
            math.sqrt(
                math.fsum(
                    float(w * (v - means[j - 1]) ** 2) for w, v in zip(weights, raw[:, j])
                )
                / total
            )
            for j in range(1, 12)
        ]
    )
    _need(
        np.isfinite(means).all() and np.isfinite(scales).all() and (scales > 1e-12).all(),
        "Expert fixed feature scale failure",
    )
    x = np.column_stack((np.ones(len(train)), (raw[:, 1:] - means) / scales))
    query_x = np.column_stack((np.ones(len(query)), (query[:, 1:] - means) / scales))
    design = np.sqrt(weights)[:, None] * x
    logged = np.log(target)
    singular = np.linalg.svd(design, compute_uv=False)
    rank = int(np.count_nonzero(singular > singular[0] * 1e-12))
    _need(rank == 12, "Expert fixed design rank failure")
    q, r = np.linalg.qr(design, mode="reduced")
    beta = np.linalg.solve(r, q.T @ (np.sqrt(weights) * logged))
    residual = logged - x @ beta
    top = float(residual.max())
    smear = top + math.log(
        math.fsum(float(w) * math.exp(float(v) - top) for w, v in zip(weights, residual))
        / total
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        prediction = np.exp(query_x @ beta + smear)
    _need(
        np.isfinite(prediction).all()
        and (prediction > 0).all()
        and np.isfinite(beta).all()
        and math.isfinite(smear),
        "Positive finite expert forecasts required",
    )
    audit = dict(
        feature_names=list(MARKET),
        coefficients=dict(zip(MARKET, map(float, beta))),
        feature_means=dict(zip(MARKET[1:], map(float, means))),
        feature_scales=dict(zip(MARKET[1:], map(float, scales))),
        singular_values=list(map(float, singular)),
        rank=rank,
        rank_relative_cutoff=1e-12,
        log_smearing=smear,
        normal_equation_max_abs=float(np.abs(x.T @ (weights * residual) / total).max()),
        weights=weights.tolist(),
        weight_sum=total,
        n_train=len(train),
        n_application=len(application),
    )
    return prediction, audit


def _config(config):
    if config is None:
        config = dict(
            issuance_start="2010-01-04",
            origin_start="2016-01-01",
            origin_end="2025-10-20",
            source_end="2025-10-20",
            development=["2016-01-01", "2019-12-31"],
            evaluation=["2020-01-01", "2025-10-20"],
            minimum_train=1000,
            adaptive_half_life=252,
            gate_window=1260,
            gate_minimum_train=252,
        )
    date_keys = ("issuance_start", "origin_start", "origin_end", "source_end")
    integer_keys = ("minimum_train", "adaptive_half_life", "gate_window", "gate_minimum_train")
    _need(
        type(config) is dict
        and set(config) == set(date_keys + integer_keys + ("development", "evaluation")),
        "Exact precision-gate configuration required",
    )
    result = {key: _iso(config[key]) for key in date_keys}
    for phase in ("development", "evaluation"):
        _need(
            type(config[phase]) is list and len(config[phase]) == 2,
            "Literal phase date pair required",
        )
        result[phase] = tuple(map(_iso, config[phase]))
    a, b = result["development"]
    c, d = result["evaluation"]
    _need(
        _FLOOR
        <= result["issuance_start"]
        <= result["origin_start"]
        <= a
        <= b
        < c
        <= d
        <= result["origin_end"]
        <= result["source_end"],
        "Ordered issuance and scoring dates required",
    )
    for key in integer_keys:
        _need(
            type(config[key]) is int and config[key] > 0,
            "Positive exact configuration integers required",
        )
        result[key] = config[key]
    return result


def _inputs(sources, features, targets, cfg):
    _need(
        type(sources) is dict and set(sources) == {"daily", "cross", "iv"},
        "Exact raw market source mapping required",
    )
    for key, names in (
        ("daily", ("open", "high", "low", "close", "adj close", "volume")),
        ("cross", _CROSS),
        ("iv", ("vxn", "vix", "vix9d")),
    ):
        _schema(sources[key], names, cfg["source_end"], empty=key != "daily")
    _schema(features, MARKET + ("rv_total",), cfg["source_end"], empty=False)
    _schema(targets, ("y", "target_end"), cfg["source_end"], empty=False)
    _need(
        features.index.equals(sources["daily"].index) and targets.index.equals(features.index),
        "Full source calendar cannot be compressed",
    )
    market, wanted_targets = _market_targets(**sources)
    for actual, wanted, label in (
        (features, market, "features"),
        (targets, wanted_targets, "targets"),
    ):
        for col in wanted:
            if col != "target_end":
                _real_dtype(actual[col])
            for i in range(len(wanted)):
                _compare(actual[col].iloc[i], wanted[col].iloc[i], f"{label}[{i}].{col}")
    return market, wanted_targets


def _frame(actual, rows, columns, label):
    _need(
        isinstance(actual, pd.DataFrame)
        and tuple(actual.columns) == columns
        and len(actual) == len(rows),
        "Output frame schema/count mismatch: " + label,
    )
    for i, row in enumerate(rows):
        for col in columns:
            _compare(actual.iloc[i][col], row[col], f"{label}[{i}].{col}")


def verify_forecasts(sources, features, targets, produced, config=None):
    """Reconstruct all issued expert/gate records, including warmup and unscored rows."""
    cfg = _config(config)
    features, targets = _inputs(sources, features, targets, cfg)
    _need(
        type(produced) is dict
        and set(produced) == {"applications", "panel", "coverage", "schedules", "fits"},
        "Exact complete produced payload required",
    )
    _need(type(produced["fits"]) is list, "Ordered fit audit list required")
    index = features.index
    matrix = features.loc[:, MARKET].to_numpy(float)
    complete = np.isfinite(matrix).all(axis=1)
    state = features.lrv_d.to_numpy() - features.lrv_m.to_numpy()
    _need(not np.isinf(state).any(), "Unrepresentable gate state")
    requested = [
        i for i, day in enumerate(index) if cfg["issuance_start"] <= day <= cfg["origin_end"]
    ]

    def phase(day):
        for name in ("development", "evaluation"):
            if cfg[name][0] <= day <= cfg[name][1]:
                return name
        return "outside_phase"

    def dates(positions):
        return [index[i].strftime("%Y-%m-%d") for i in positions]

    schedules, applications, fits, issued = [], [], [], {}
    gate_solves = 0
    for month in pd.period_range(cfg["issuance_start"], cfg["origin_end"], freq="M"):
        positions = [i for i in requested if index[i].to_period("M") == month]
        queries = [i for i in positions if complete[i]]
        schedule = dict(
            month=str(month),
            status="no_complete_origin" if positions else "no_requested_origins",
            fit_origin=pd.NaT,
            training_cutoff=pd.NaT,
            requested_n=len(positions),
            feature_complete_n=len(queries),
            application_n=0,
            train_n=None,
            gate_n=None,
            gate_status="not_issued",
        )
        schedules.append(schedule)
        if not queries:
            continue
        first = queries[0]
        origin, cutoff = index[first], index[first - 1] if first else pd.NaT
        train = [
            i
            for i in range(first)
            if complete[i]
            and math.isfinite(targets.y.iloc[i])
            and pd.notna(targets.target_end.iloc[i])
            and pd.notna(cutoff)
            and targets.target_end.iloc[i] <= cutoff
        ]
        schedule.update(fit_origin=origin, training_cutoff=cutoff, train_n=len(train))
        fit = dict(
            month=str(month),
            status="not_attempted",
            fit_origin=origin.strftime("%Y-%m-%d"),
            training_cutoff=cutoff.strftime("%Y-%m-%d") if pd.notna(cutoff) else None,
            train_origins=dates(train),
            train_positions=train,
            train_n=len(train),
            planned_application_origins=dates(queries),
            application_origins=[],
            gate_train_origins=[],
            gate_train_positions=[],
            gate_n=0,
            gate_status="not_issued",
            expert_audits=None,
            gate_audits=None,
        )
        fits.append(fit)
        if pd.isna(cutoff) or len(train) < cfg["minimum_train"]:
            _need(
                origin < cfg["origin_start"],
                "INSUFFICIENT_DATA: expert support fails in scoring period",
            )
            fit["status"] = schedule["status"] = "expert_warmup"
            continue
        predictions, expert_audits = {}, {}
        for arm, half_life in (("base", None), ("adaptive", cfg["adaptive_half_life"])):
            predictions[arm], expert_audits[arm] = _expert_expected(
                features.iloc[train],
                targets.y.iloc[train],
                features.iloc[queries],
                np.array(train, dtype=int),
                first - 1,
                half_life,
            )
        gate_positions = [
            i
            for i in issued
            if first - cfg["gate_window"] <= i < first
            and math.isfinite(targets.y.iloc[i])
            and pd.notna(targets.target_end.iloc[i])
            and targets.target_end.iloc[i] <= cutoff
        ]
        gate_n = len(gate_positions)
        gate_status = "fitted" if gate_n >= cfg["gate_minimum_train"] else "cold_start"
        gate_audits = {}
        _need(len(produced["fits"]) >= len(fits), "Missing monthly expert/gate audit")
        saved = produced["fits"][len(fits) - 1]
        _need(
            type(saved) is dict
            and type(saved.get("gate_audits")) is dict
            and set(saved["gate_audits"]) == {"constant", "contextual"},
            "Both gate audit arms required",
        )
        for arm, constant in (("constant", True), ("contextual", False)):
            if gate_status == "cold_start":
                gate = dict(
                    status="cold_start",
                    coefficients=[0.0, 0.0],
                    constant=constant,
                    n_train=gate_n,
                )
                coefficients = np.zeros(2)
            else:
                gate = saved["gate_audits"][arm]
                _need(
                    type(gate) is dict and gate.get("status") == "fitted",
                    "Fitted issued-history gate audit required",
                )
                coefficients = _verify_gate(
                    np.array([issued[i]["pred_base"] for i in gate_positions]),
                    np.array([issued[i]["pred_adaptive"] for i in gate_positions]),
                    targets.y.iloc[gate_positions].to_numpy(),
                    np.array([issued[i]["state"] for i in gate_positions]),
                    {key: value for key, value in gate.items() if key != "status"},
                    constant=constant,
                )
                gate_solves += 1
            predictions[arm] = _gate_prediction(
                predictions["base"], predictions["adaptive"], state[queries], coefficients
            )
            gate_audits[arm] = gate
        schedule.update(
            status="fitted", application_n=len(queries), gate_n=gate_n, gate_status=gate_status
        )
        fit.update(
            status="fitted",
            application_origins=dates(queries),
            gate_train_origins=dates(gate_positions),
            gate_train_positions=gate_positions,
            gate_n=gate_n,
            gate_status=gate_status,
            expert_audits=expert_audits,
            gate_audits=gate_audits,
        )
        for j, i in enumerate(queries):
            app = dict(
                origin=index[i],
                fit_origin=origin,
                training_cutoff=cutoff,
                train_n=len(train),
                gate_n=gate_n,
                gate_status=gate_status,
                state=float(state[i]),
                offset=i % 5,
                phase=phase(index[i]),
                **{"pred_" + arm: float(predictions[arm][j]) for arm in ARMS},
            )
            applications.append(app)
            issued[i] = app
    by_month = {row["month"]: row for row in schedules}
    coverage, panel = [], []
    for i in requested:
        day, assigned = index[i], phase(index[i])
        endpoint, value = targets.target_end.iloc[i], targets.y.iloc[i]
        observed = math.isfinite(value)
        within = bool(
            assigned != "outside_phase" and pd.notna(endpoint) and endpoint <= cfg[assigned][1]
        )
        app = issued.get(i)
        if not complete[i]:
            status = "incomplete_features"
        elif by_month[str(day.to_period("M"))]["status"] == "expert_warmup":
            status = "expert_warmup"
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
        coverage.append(
            dict(
                origin=day,
                phase=assigned,
                feature_complete=bool(complete[i]),
                missing_features="|".join(
                    name for j, name in enumerate(MARKET) if not math.isfinite(matrix[i, j])
                ),
                state=float(state[i]),
                offset=i % 5,
                target_end=endpoint,
                target_observed=observed,
                target_within_phase=within,
                issued=app is not None,
                scored=status == "scored",
                status=status,
                fit_origin=app["fit_origin"] if app else pd.NaT,
                training_cutoff=app["training_cutoff"] if app else pd.NaT,
                gate_status=app["gate_status"] if app else "not_issued",
            )
        )
        if status == "scored":
            _need(app is not None, "Scored origin without issued forecast")
            for arm in ARMS:
                prediction = app["pred_" + arm]
                ratio = value / prediction
                _need(math.isfinite(ratio) and ratio > 0, "Invalid positive QLIKE ratio")
                loss = ratio - math.log(ratio) - 1
                _need(math.isfinite(loss), "Nonfinite QLIKE arithmetic")
                panel.append(
                    dict(
                        origin=day,
                        model=arm,
                        prediction=prediction,
                        y=float(value),
                        loss=loss,
                        target_end=endpoint,
                        **{key: app[key] for key in _SHARED},
                    )
                )
    _compare(produced["fits"], fits, "fits")
    for name, rows, columns in (
        ("applications", applications, _APPLICATION),
        ("panel", panel, _PANEL),
        ("coverage", coverage, _COVERAGE),
        ("schedules", schedules, _SCHEDULE),
    ):
        _frame(produced[name], rows, columns, name)
    fitted = sum(fit["status"] == "fitted" for fit in fits)
    return dict(
        status="VERIFIED",
        calendar_rows=len(index),
        coverage_origins=len(coverage),
        application_origins=len(applications),
        application_forecasts_verified=4 * len(applications),
        scored_origins=len(panel) // 4,
        forecasts_verified=len(panel),
        monthly_fits=fitted,
        expert_fits=2 * fitted,
        warmup_months=len(fits) - fitted,
        gate_fits=gate_solves,
        cold_start_months=sum(fit["gate_status"] == "cold_start" for fit in fits),
        gate_solver="Independent SLSQP; saved coefficient stationarity and objective reconstructed",
        limits=[
            "Producer optimizer iteration counts are checked as bounded metadata, not independently reproduced.",
            "Source byte authentication and original market oracle certification are separate; historically accessed later periods are not untouched confirmation.",
        ],
    )
