"""Fixed monthly experts and gates trained only on previously issued forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.claims_release_pipeline import _dates, _iso, _numeric, _qlike
from src.precision_gate import fit_gate, predict_gate
from src.precision_gate_models import MARKET, fit_experts

ARMS = ("base", "adaptive", "constant", "contextual")
SHARED = (
    "fit_origin",
    "training_cutoff",
    "train_n",
    "gate_n",
    "gate_status",
    "state",
    "offset",
    "phase",
)
APPLICATION_COLUMNS = ("origin", *SHARED, *(f"pred_{arm}" for arm in ARMS))
PANEL_COLUMNS = ("origin", "model", "prediction", "y", "loss", "target_end", *SHARED)
COVERAGE_COLUMNS = (
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
SCHEDULE_COLUMNS = (
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


class PipelineExecutionError(ValueError):
    """Whole attempt failed; partial issuance and diagnostics remain explicit."""

    def __init__(self, message, produced, *, month=None, stage=None):
        super().__init__(message)
        self.month = month
        self.stage = stage
        self.produced = produced
        self.coverage = produced["coverage"]
        self.schedules = produced["schedules"]


class InsufficientDataError(PipelineExecutionError):
    """Insufficient expert support, with the same retained diagnostic payload."""


def _configuration(config):
    if config is None:
        config = {
            "issuance_start": "2010-01-04",
            "origin_start": "2016-01-01",
            "origin_end": "2025-10-20",
            "source_end": "2025-10-20",
            "development": ["2016-01-01", "2019-12-31"],
            "evaluation": ["2020-01-01", "2025-10-20"],
            "minimum_train": 1000,
            "adaptive_half_life": 252,
            "gate_window": 1260,
            "gate_minimum_train": 252,
        }
    date_names = ("issuance_start", "origin_start", "origin_end", "source_end")
    integers = ("minimum_train", "adaptive_half_life", "gate_window", "gate_minimum_train")
    if type(config) is not dict or set(config) != {
        *date_names,
        *integers,
        "development",
        "evaluation",
    }:
        raise ValueError("Exact precision-gate pipeline configuration required")
    parsed = {key: _iso(config[key]) for key in date_names}
    for phase in ("development", "evaluation"):
        if type(config[phase]) is not list or len(config[phase]) != 2:
            raise ValueError("Two literal ISO dates per phase required")
        parsed[phase] = tuple(_iso(value) for value in config[phase])
    if not (
        pd.Timestamp("2010-01-04")
        <= parsed["issuance_start"]
        <= parsed["origin_start"]
        <= parsed["development"][0]
        <= parsed["development"][1]
        < parsed["evaluation"][0]
        <= parsed["evaluation"][1]
        <= parsed["origin_end"]
        <= parsed["source_end"]
        <= pd.Timestamp("2025-10-20")
    ):
        raise ValueError("Ordered bounded issuance and scoring phases required")
    for key in integers:
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError("Positive literal integer pipeline settings required")
        parsed[key] = config[key]
    return parsed


def _inputs(features, targets, config):
    if (
        not isinstance(features, pd.DataFrame)
        or not isinstance(targets, pd.DataFrame)
        or not features.columns.is_unique
        or not targets.columns.is_unique
        or not set(MARKET) <= set(features.columns)
        or set(targets.columns) != {"y", "target_end"}
        or not isinstance(features.index, pd.DatetimeIndex)
        or not isinstance(targets.index, pd.DatetimeIndex)
        or not features.index.equals(targets.index)
    ):
        raise ValueError("Common12 features, exact targets and aligned full calendar required")
    calendar = _dates(features.index, config["source_end"])
    ends = _dates(targets.target_end, config["source_end"])
    if (
        not len(calendar)
        or calendar.hasnans
        or not calendar.is_unique
        or not calendar.is_monotonic_increasing
        or not ends.equals(pd.DatetimeIndex(pd.Series(calendar).shift(-5)))
    ):
        raise ValueError(
            "Unique full calendar and exact fifth-session target endpoints required"
        )
    matrix = np.column_stack([_numeric(features[name]) for name in MARKET])
    if np.any(np.isfinite(matrix[:, 0]) & (matrix[:, 0] != 1)):
        raise ValueError("Observed intercept must equal one")
    complete = np.isfinite(matrix).all(axis=1)
    y = _numeric(targets.y)
    if np.any(np.isfinite(y) & ((y <= 0) | ends.isna())):
        raise ValueError("Observed positive target requires full endpoint")
    try:
        with np.errstate(over="raise", invalid="raise"):
            state = matrix[:, MARKET.index("lrv_d")] - matrix[:, MARKET.index("lrv_m")]
    except FloatingPointError as error:
        raise ValueError("Nonfinite fixed gate state arithmetic") from error
    return calendar, matrix, complete, y, ends, state


def _phase(origin, config):
    for phase in ("development", "evaluation"):
        if config[phase][0] <= origin <= config[phase][1]:
            return phase
    return "outside_phase"


def _frame(rows, columns):
    frame = pd.DataFrame(rows, columns=columns)
    for name in {"origin", "fit_origin", "training_cutoff", "target_end"} & set(columns):
        frame[name] = pd.to_datetime(frame[name]).astype("datetime64[ns]")
    for name in {
        "requested_n",
        "feature_complete_n",
        "application_n",
        "train_n",
        "gate_n",
        "offset",
    } & set(columns):
        frame[name] = pd.array(frame[name], dtype="Int64")
    for name in {
        "feature_complete",
        "target_observed",
        "target_within_phase",
        "issued",
        "scored",
    } & set(columns):
        frame[name] = frame[name].astype(bool)
    for name in {"prediction", "y", "loss", "state", *(f"pred_{arm}" for arm in ARMS)} & set(
        columns
    ):
        frame[name] = frame[name].astype(float)
    return frame


def _coverage(inputs, requested, config, application_map, schedules, aborted=False):
    calendar, matrix, complete, y, ends, state = inputs
    month_map = {r["month"]: r for r in schedules}
    rows = []
    for position in np.flatnonzero(requested):
        origin = calendar[position]
        phase = _phase(origin, config)
        observed = bool(np.isfinite(y[position]))
        within = (
            phase != "outside_phase"
            and pd.notna(ends[position])
            and ends[position] <= config[phase][1]
        )
        app = application_map.get(position)
        schedule = month_map[str(origin.to_period("M"))]
        if not complete[position]:
            status = "incomplete_features"
        elif aborted:
            status = "attempt_aborted"
        elif schedule["status"] == "expert_warmup":
            status = "expert_warmup"
        elif phase == "outside_phase":
            status = "outside_phase"
        elif pd.isna(ends[position]):
            status = "target_not_mature"
        elif not observed:
            status = "missing_target"
        elif not within:
            status = "target_after_phase_cutoff"
        else:
            status = "scored"
        rows.append(
            {
                "origin": origin,
                "phase": phase,
                "feature_complete": bool(complete[position]),
                "missing_features": "|".join(
                    n for n, v in zip(MARKET, matrix[position]) if np.isnan(v)
                ),
                "state": state[position],
                "offset": int(position % 5),
                "target_end": ends[position],
                "target_observed": observed,
                "target_within_phase": bool(within),
                "issued": app is not None,
                "scored": status == "scored",
                "status": status,
                "fit_origin": app["fit_origin"] if app else pd.NaT,
                "training_cutoff": app["training_cutoff"] if app else pd.NaT,
                "gate_status": app["gate_status"] if app else "not_issued",
            }
        )
    return _frame(rows, COVERAGE_COLUMNS)


def build_panel(features, targets, config=None):
    config = _configuration(config)
    inputs = _inputs(features, targets, config)
    calendar, _, complete, y, ends, state = inputs
    requested = (calendar >= config["issuance_start"]) & (calendar <= config["origin_end"])
    schedules, planned = [], []
    months = calendar.to_period("M")
    for month in pd.period_range(config["issuance_start"], config["origin_end"], freq="M"):
        positions = np.flatnonzero(requested & (months == month))
        query = positions[complete[positions]]
        first = int(query[0]) if len(query) else None
        schedules.append(
            {
                "month": str(month),
                "status": "not_attempted"
                if first is not None
                else "no_complete_origin"
                if len(positions)
                else "no_requested_origins",
                "fit_origin": calendar[first] if first is not None else pd.NaT,
                "training_cutoff": calendar[first - 1]
                if first is not None and first > 0
                else pd.NaT,
                "requested_n": len(positions),
                "feature_complete_n": len(query),
                "application_n": 0,
                "train_n": None,
                "gate_n": None,
                "gate_status": "not_issued",
            }
        )
        planned.append(query)
    applications, fits, application_map = [], [], {}

    def result(aborted=False):
        return {
            "applications": _frame(applications, APPLICATION_COLUMNS),
            "panel": _frame([], PANEL_COLUMNS),
            "coverage": _coverage(
                inputs, requested, config, application_map, schedules, aborted
            ),
            "schedules": _frame(schedules, SCHEDULE_COLUMNS),
            "fits": fits,
        }

    def dates(positions):
        return [day.date().isoformat() for day in calendar[positions]]

    for schedule, query in zip(schedules, planned):
        if not len(query):
            continue
        first = int(query[0])
        origin, cutoff = schedule["fit_origin"], schedule["training_cutoff"]
        train = np.flatnonzero(
            complete & (calendar < origin) & np.isfinite(y) & (ends <= cutoff)
        )
        schedule["train_n"] = len(train)
        audit = {
            "month": schedule["month"],
            "status": "not_attempted",
            "fit_origin": origin.date().isoformat(),
            "training_cutoff": cutoff.date().isoformat() if pd.notna(cutoff) else None,
            "train_origins": dates(train),
            "train_positions": train.tolist(),
            "train_n": len(train),
            "planned_application_origins": dates(query),
            "application_origins": [],
            "gate_train_origins": [],
            "gate_train_positions": [],
            "gate_n": 0,
            "gate_status": "not_issued",
            "expert_audits": None,
            "gate_audits": None,
        }
        fits.append(audit)
        if pd.isna(cutoff) or len(train) < config["minimum_train"]:
            status = (
                "expert_warmup" if origin < config["origin_start"] else "insufficient_training"
            )
            schedule["status"] = audit["status"] = status
            if status == "expert_warmup":
                continue
            raise InsufficientDataError(
                f"INSUFFICIENT_DATA: Expert training at {origin.date()} has {len(train)} common mature rows",
                result(aborted=True),
            )
        try:
            stage = "expert_fit"
            expert = fit_experts(
                features.iloc[train],
                targets.y.iloc[train],
                features.iloc[query],
                train_positions=train,
                cutoff_position=first - 1,
                adaptive_half_life=config["adaptive_half_life"],
            )
            audit["expert_audits"] = expert["fits"]
            stage = "gate_membership"
            gate_positions = np.array(
                [
                    p
                    for p in application_map
                    if first - config["gate_window"] <= p < first
                    and np.isfinite(y[p])
                    and ends[p] <= cutoff
                ],
                dtype=int,
            )
            gate_n = len(gate_positions)
            gate_status = "fitted" if gate_n >= config["gate_minimum_train"] else "cold_start"
            audit.update(
                gate_train_origins=dates(gate_positions),
                gate_train_positions=gate_positions.tolist(),
                gate_n=gate_n,
                gate_status=gate_status,
                gate_audits={},
            )
            schedule.update(gate_n=gate_n, gate_status=gate_status)
            gate_audits, predictions = audit["gate_audits"], dict(expert["predictions"])
            for arm, constant in (("constant", True), ("contextual", False)):
                stage = arm + "_gate_fit"
                if gate_status == "cold_start":
                    gate = {
                        "status": "cold_start",
                        "coefficients": [0.0, 0.0],
                        "constant": constant,
                        "n_train": gate_n,
                    }
                else:
                    records = [application_map[p] for p in gate_positions]
                    gate = {
                        "status": "fitted",
                        **fit_gate(
                            np.array([r["pred_base"] for r in records]),
                            np.array([r["pred_adaptive"] for r in records]),
                            y[gate_positions],
                            np.array([r["state"] for r in records]),
                            constant=constant,
                        ),
                    }
                gate_audits[arm] = gate
                stage = arm + "_gate_prediction"
                predictions[arm] = predict_gate(
                    predictions["base"],
                    predictions["adaptive"],
                    state[query],
                    gate["coefficients"],
                )
            stage = "prediction_validation"
            for arm in ARMS:
                values = np.asarray(predictions[arm])
                if (
                    values.shape != (len(query),)
                    or not np.isfinite(values).all()
                    or np.any(values <= 0)
                ):
                    raise ValueError(
                        "Finite positive aligned predictions required for every arm"
                    )
            schedule.update(
                status="fitted",
                application_n=len(query),
                gate_n=gate_n,
                gate_status=gate_status,
            )
            audit.update(
                status="fitted",
                application_origins=dates(query),
                gate_train_origins=dates(gate_positions),
                gate_train_positions=gate_positions.tolist(),
                gate_n=gate_n,
                gate_status=gate_status,
                expert_audits=expert["fits"],
                gate_audits=gate_audits,
            )
            stage = "application_issuance"
            for i, position in enumerate(query):
                row = {
                    "origin": calendar[position],
                    "fit_origin": origin,
                    "training_cutoff": cutoff,
                    "train_n": len(train),
                    "gate_n": gate_n,
                    "gate_status": gate_status,
                    "state": float(state[position]),
                    "offset": int(position % 5),
                    "phase": _phase(calendar[position], config),
                    **{f"pred_{arm}": float(predictions[arm][i]) for arm in ARMS},
                }
                applications.append(row)
                application_map[int(position)] = row
        except Exception as error:
            schedule["status"] = audit["status"] = "execution_failed"
            raise PipelineExecutionError(
                f"Pipeline failure at {schedule['month']} during {stage}: {error}",
                result(aborted=True),
                month=schedule["month"],
                stage=stage,
            ) from error
    output = result()
    panel = []
    try:
        for row in output["coverage"].loc[output["coverage"].scored].itertuples(index=False):
            position = calendar.get_loc(row.origin)
            app = application_map[position]
            for arm in ARMS:
                prediction = app[f"pred_{arm}"]
                panel.append(
                    {
                        "origin": row.origin,
                        "model": arm,
                        "prediction": prediction,
                        "y": float(y[position]),
                        "loss": _qlike(y[position], prediction),
                        "target_end": row.target_end,
                        **{name: app[name] for name in SHARED},
                    }
                )
        output["panel"] = _frame(panel, PANEL_COLUMNS)
    except Exception as error:
        raise PipelineExecutionError(
            f"Pipeline failure while constructing scored panel: {error}",
            result(aborted=True),
            stage="scored_panel",
        ) from error
    return output
