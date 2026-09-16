"""Fixed monthly commodity forecasts on one unchanged observed calendar."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.claims_release_pipeline import _dates, _iso, _numeric, _prediction_arrays, _qlike
from src.commodity_implied_models import ALL, fit_models

SOURCE_FLOOR = pd.Timestamp("2009-01-02")
ARMS = ("market", "matched", "candidate")
APPLICATION_COLUMNS = (
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
PANEL_COLUMNS = (
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
COVERAGE_COLUMNS = (
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
SCHEDULE_COLUMNS = (
    "month",
    "status",
    "fit_origin",
    "training_cutoff",
    "requested_n",
    "application_n",
    "train_n",
)
DATE_COLUMNS = {
    "origin",
    "fit_origin",
    "training_cutoff",
    "commodity_cutoff_date",
    "target_end",
}


class InsufficientDataError(ValueError):
    """A failed whole attempt with coverage and schedules, without forecasts."""

    def __init__(self, message, coverage, schedules):
        super().__init__(message)
        self.coverage = coverage
        self.schedules = schedules


def _configuration(config):
    if config is None:
        config = {
            "origin_start": "2016-01-01",
            "origin_end": "2025-10-20",
            "development": ["2016-01-01", "2019-12-31"],
            "evaluation": ["2020-01-01", "2025-10-20"],
            "source_end": "2025-10-20",
            "minimum_train": 1000,
        }
    fields = {
        "origin_start",
        "origin_end",
        "development",
        "evaluation",
        "source_end",
        "minimum_train",
    }
    if type(config) is not dict or set(config) != fields:
        raise ValueError("Exact commodity pipeline configuration fields required")
    parsed = {
        name: _iso(config[name]) for name in ("origin_start", "origin_end", "source_end")
    }
    for phase in ("development", "evaluation"):
        if type(config[phase]) is not list or len(config[phase]) != 2:
            raise ValueError("Two literal ISO dates required per phase")
        parsed[phase] = tuple(_iso(value) for value in config[phase])
    if not (
        SOURCE_FLOOR <= parsed["origin_start"] <= parsed["origin_end"] <= parsed["source_end"]
        and parsed["origin_start"]
        <= parsed["development"][0]
        <= parsed["development"][1]
        < parsed["evaluation"][0]
        <= parsed["evaluation"][1]
        <= parsed["origin_end"]
    ):
        raise ValueError(
            "Ordered nonoverlapping bounded phases within the origin window required"
        )
    if type(config["minimum_train"]) is not int or config["minimum_train"] <= 0:
        raise ValueError("Positive exact integer training floor required")
    parsed["minimum_train"] = config["minimum_train"]
    return parsed


def _inputs(features, targets, config):
    required = set(ALL) | {"commodity_cutoff_date"}
    if (
        not isinstance(features, pd.DataFrame)
        or not isinstance(targets, pd.DataFrame)
        or not features.columns.is_unique
        or not targets.columns.is_unique
        or not required <= set(features.columns)
        or set(targets.columns) != {"y", "target_end"}
        or not isinstance(features.index, pd.DatetimeIndex)
        or not isinstance(targets.index, pd.DatetimeIndex)
        or not features.index.equals(targets.index)
    ):
        raise ValueError("Required frame columns and exactly aligned full calendars required")
    calendar = _dates(features.index, config["source_end"])
    if (
        not len(calendar)
        or calendar.hasnans
        or not calendar.is_unique
        or not calendar.is_monotonic_increasing
    ):
        raise ValueError("Nonempty unique increasing full observed calendar required")
    matrix = np.column_stack([_numeric(features[name]) for name in ALL])
    if np.any(np.isfinite(matrix[:, 0]) & (matrix[:, 0] != 1.0)):
        raise ValueError("Observed intercept must be exactly one")
    complete = np.isfinite(matrix).all(axis=1)
    y = _numeric(targets["y"])
    if np.any(np.isfinite(y) & (y <= 0)):
        raise ValueError("Observed targets must be finite and positive")
    ends = _dates(targets["target_end"], config["source_end"])
    expected_ends = pd.DatetimeIndex(pd.Series(calendar).shift(-5))
    if not ends.equals(expected_ends) or np.any(np.isfinite(y) & ends.isna()):
        raise ValueError(
            "Exact fifth full-calendar target endpoint and no label without endpoint required"
        )
    cutoffs = _dates(features["commodity_cutoff_date"], config["source_end"])
    previous = pd.Series(calendar).shift(1)
    expected_cutoffs = pd.DatetimeIndex(previous.where(previous >= SOURCE_FLOOR))
    if not cutoffs.equals(expected_cutoffs):
        raise ValueError(
            "Commodity cutoff must be the prior full-calendar session after the source floor"
        )
    if np.any(complete & cutoffs.isna()):
        raise ValueError("Complete common22 features require known commodity cutoff metadata")
    return calendar, matrix, complete, y, ends, cutoffs


def _phase(origin, config):
    for name in ("development", "evaluation"):
        if config[name][0] <= origin <= config[name][1]:
            return name
    return "outside_phase"


def _frame(rows, columns):
    frame = pd.DataFrame(rows, columns=columns)
    for name in DATE_COLUMNS & set(columns):
        frame[name] = pd.to_datetime(frame[name]).astype("datetime64[ns]")
    for name in {"train_n", "application_n", "requested_n", "offset"} & set(columns):
        frame[name] = pd.array(frame[name], dtype="Int64")
    for name in {"feature_complete", "target_observed", "target_within_phase", "scored"} & set(
        columns
    ):
        frame[name] = frame[name].astype(bool)
    for name in {
        "prediction",
        "y",
        "loss",
        "pred_market",
        "pred_matched",
        "pred_candidate",
    } & set(columns):
        frame[name] = frame[name].astype(float)
    return frame


def _coverage(inputs, requested, config, application_by_position, aborted=False):
    calendar, matrix, complete, y, ends, cutoffs = inputs
    rows = []
    for position in np.flatnonzero(requested):
        origin = calendar[position]
        phase = _phase(origin, config)
        observed = bool(np.isfinite(y[position]))
        fence = config["development"][1] if phase == "development" else config["source_end"]
        within = (
            phase != "outside_phase"
            and not pd.isna(ends[position])
            and ends[position] <= fence
        )
        if not complete[position]:
            status = "incomplete_features"
        elif aborted:
            status = "attempt_aborted"
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
        application = application_by_position.get(position)
        rows.append(
            {
                "origin": origin,
                "phase": phase,
                "feature_complete": bool(complete[position]),
                "missing_features": "|".join(
                    name for name, value in zip(ALL, matrix[position]) if np.isnan(value)
                ),
                "commodity_cutoff_date": cutoffs[position],
                "offset": int(position % 5),
                "target_end": ends[position],
                "target_observed": observed,
                "target_within_phase": bool(within),
                "scored": status == "scored",
                "status": status,
                "fit_origin": application["fit_origin"] if application is not None else pd.NaT,
                "training_cutoff": application["training_cutoff"]
                if application is not None
                else pd.NaT,
            }
        )
    return _frame(rows, COVERAGE_COLUMNS)


def build_panel(
    features: pd.DataFrame, targets: pd.DataFrame, config: dict | None = None
) -> dict:
    """Build all fixed daily applications; any unsupported monthly fit aborts."""
    config = _configuration(config)
    inputs = _inputs(features, targets, config)
    calendar, _, complete, y, ends, cutoffs = inputs
    requested = (calendar >= config["origin_start"]) & (calendar <= config["origin_end"])
    schedules, planned = [], []
    months = calendar.to_period("M")
    # All query dates and application sets are determined from features alone.
    for month in pd.period_range(config["origin_start"], config["origin_end"], freq="M"):
        positions = np.flatnonzero(requested & (months == month))
        application_positions = positions[complete[positions]]
        first = int(application_positions[0]) if len(application_positions) else None
        schedule = {
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
            "application_n": len(application_positions),
            "train_n": None,
        }
        schedules.append(schedule)
        planned.append(application_positions)
    applications, fits = [], []
    application_by_position = {}
    for schedule, application_positions in zip(schedules, planned):
        if not len(application_positions):
            continue
        fit_origin, cutoff = schedule["fit_origin"], schedule["training_cutoff"]
        train_positions = np.flatnonzero(
            complete & (calendar < fit_origin) & np.isfinite(y) & (ends <= cutoff)
        )
        train_n = len(train_positions)
        schedule["train_n"] = train_n
        if pd.isna(cutoff) or train_n < config["minimum_train"]:
            schedule["status"] = "insufficient_training"
            raise InsufficientDataError(
                f"INSUFFICIENT_DATA: Insufficient training support at {fit_origin.date()}: {train_n} common rows",
                _coverage(inputs, requested, config, {}, aborted=True),
                _frame(schedules, SCHEDULE_COLUMNS),
            )
        fitted = fit_models(
            features.iloc[train_positions],
            targets["y"].iloc[train_positions],
            features.iloc[application_positions],
        )
        predictions = _prediction_arrays(fitted, len(application_positions))
        if set(fitted["fits"]) != set(ARMS):
            raise ValueError("Separate audits for all three fixed arms required")
        schedule["status"] = "fitted"
        fits.append(
            {
                "month": schedule["month"],
                "fit_origin": fit_origin.date().isoformat(),
                "training_cutoff": cutoff.date().isoformat(),
                "train_origins": [
                    date.date().isoformat() for date in calendar[train_positions]
                ],
                "application_origins": [
                    date.date().isoformat() for date in calendar[application_positions]
                ],
                "train_n": train_n,
                "model_audits": fitted["fits"],
            }
        )
        for local, position in enumerate(application_positions):
            row = {
                "origin": calendar[position],
                "fit_origin": fit_origin,
                "training_cutoff": cutoff,
                "train_n": train_n,
                "commodity_cutoff_date": cutoffs[position],
                "offset": int(position % 5),
                "phase": _phase(calendar[position], config),
                **{f"pred_{arm}": float(predictions[arm][local]) for arm in ARMS},
            }
            applications.append(row)
            application_by_position[position] = row
    coverage = _coverage(inputs, requested, config, application_by_position)
    panel = []
    for row in coverage.loc[coverage.scored].itertuples(index=False):
        position = calendar.get_loc(row.origin)
        application = application_by_position[position]
        for arm in ARMS:
            prediction = application[f"pred_{arm}"]
            panel.append(
                {
                    "origin": row.origin,
                    "model": arm,
                    "prediction": prediction,
                    "y": float(y[position]),
                    "loss": _qlike(y[position], prediction),
                    "target_end": row.target_end,
                    "phase": row.phase,
                    **{
                        name: application[name]
                        for name in (
                            "commodity_cutoff_date",
                            "offset",
                            "fit_origin",
                            "training_cutoff",
                            "train_n",
                        )
                    },
                }
            )
    return {
        "applications": _frame(applications, APPLICATION_COLUMNS),
        "panel": _frame(panel, PANEL_COLUMNS),
        "coverage": coverage,
        "schedules": _frame(schedules, SCHEDULE_COLUMNS),
        "fits": fits,
    }
