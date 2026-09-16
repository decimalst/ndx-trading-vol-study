"""Fixed monthly common-cohort claims forecasts on the complete observed calendar."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.claims_release_models import ALL, fit_models

SOURCE_CEILING = pd.Timestamp("2025-10-20")
ARMS = ("market", "matched", "candidate")
APPLICATION_COLUMNS = (
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
PANEL_COLUMNS = (
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
COVERAGE_COLUMNS = (
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
SCHEDULE_COLUMNS = (
    "month",
    "status",
    "fit_origin",
    "training_cutoff",
    "requested_n",
    "application_n",
    "train_n",
    "train_releases",
)
DATE_COLUMNS = {
    "origin",
    "fit_origin",
    "training_cutoff",
    "claim_reference_week",
    "claim_release_date",
    "target_end",
}


def _iso(value):
    if type(value) is not str or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value) is None:
        raise ValueError("Literal ISO configuration date required")
    try:
        result = pd.Timestamp(value).as_unit("ns")
    except (ValueError, OverflowError) as error:
        raise ValueError("Valid bounded configuration date required") from error
    if result > SOURCE_CEILING:
        raise ValueError("Configuration date exceeds fixed source ceiling")
    return result


def _configuration(config):
    fields = {
        "origin_start",
        "origin_end",
        "development",
        "evaluation",
        "source_end",
        "minimum_train",
        "minimum_train_releases",
    }
    if type(config) is not dict or set(config) != fields:
        raise ValueError("Exact pipeline configuration fields required")
    parsed = {
        name: _iso(config[name]) for name in ("origin_start", "origin_end", "source_end")
    }
    for phase in ("development", "evaluation"):
        if type(config[phase]) is not list or len(config[phase]) != 2:
            raise ValueError("Two literal ISO dates required per phase")
        parsed[phase] = tuple(_iso(value) for value in config[phase])
    if not (
        parsed["origin_start"] <= parsed["origin_end"] <= parsed["source_end"]
        and parsed["origin_start"]
        <= parsed["development"][0]
        <= parsed["development"][1]
        < parsed["evaluation"][0]
        <= parsed["evaluation"][1]
        <= parsed["origin_end"]
    ):
        raise ValueError(
            "Ordered nonoverlapping phases within the origin/source window required"
        )
    for name in ("minimum_train", "minimum_train_releases"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError("Positive exact integer support floors required")
        parsed[name] = config[name]
    return parsed


def _dates(values, ceiling):
    if not pd.api.types.is_datetime64_any_dtype(values.dtype):
        raise ValueError("Native datetime columns required")
    index = pd.DatetimeIndex(values)
    if index.tz is not None or not index.equals(index.normalize()):
        raise ValueError("Naive midnight dates required")
    try:
        index = index.as_unit("ns")
    except (ValueError, OverflowError) as error:
        raise ValueError("Nanosecond-representable calendar dates required") from error
    if np.any(index > ceiling):
        raise ValueError("Input date exceeds source ceiling")
    return index


def _numeric(series):
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype) or not (
        pd.api.types.is_float_dtype(dtype) or pd.api.types.is_integer_dtype(dtype)
    ):
        raise ValueError("Real numeric, nonboolean input columns required")
    values = series.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
    if np.isinf(values).any():
        raise ValueError("Infinite numerical inputs are invalid, not missing")
    return values


def _inputs(features, targets, config):
    required = set(ALL) | {"claim_reference_week", "claim_release_date", "claim_status"}
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
        raise ValueError("Nonempty unique ordered full observed calendar required")
    matrix = np.column_stack([_numeric(features[name]) for name in ALL])
    if np.any(np.isfinite(matrix[:, 0]) & (matrix[:, 0] != 1.0)):
        raise ValueError("Observed intercept must be exactly one")
    complete = np.isfinite(matrix).all(axis=1)
    y = _numeric(targets["y"])
    if np.any(np.isfinite(y) & (y <= 0)):
        raise ValueError("Observed targets must be finite and positive")
    target_end = _dates(targets["target_end"], config["source_end"])
    expected = pd.DatetimeIndex(pd.Series(calendar).shift(-5))
    if not target_end.equals(expected) or np.any(np.isfinite(y) & target_end.isna()):
        raise ValueError(
            "Target endpoint must be the fifth full-calendar session; no label without endpoint"
        )
    weeks = _dates(features["claim_reference_week"], config["source_end"])
    releases = _dates(features["claim_release_date"], config["source_end"])
    statuses = features["claim_status"].to_numpy(copy=True)
    if not all(type(value) is str and bool(value) for value in statuses):
        raise ValueError("Explicit claim-status strings required")
    known = ~releases.isna()
    if np.any(weeks.isna() != releases.isna()) or np.any(
        known & ((weeks.dayofweek != 5) | (weeks >= releases) | (releases >= calendar))
    ):
        raise ValueError(
            "Paired Saturday reference weeks and strictly prior release dates required"
        )
    if np.any(complete & (~known | (statuses != "available"))):
        raise ValueError("Complete common features require an available dated claim state")
    return calendar, matrix, complete, y, target_end, weeks, releases, statuses


def _phase(origin, config):
    for name in ("development", "evaluation"):
        if config[name][0] <= origin <= config[name][1]:
            return name
    return "outside_phase"


def _prediction_arrays(result, n):
    if (
        type(result) is not dict
        or set(result) != {"predictions", "fits"}
        or type(result["predictions"]) is not dict
        or set(result["predictions"]) != set(ARMS)
        or type(result["fits"]) is not dict
    ):
        raise ValueError("All three model predictions and fit audits required")
    predictions = {}
    for arm in ARMS:
        values = np.asarray(result["predictions"][arm])
        if values.shape != (n,) or values.dtype.kind not in "fiu":
            raise ValueError("One real prediction per complete application required")
        values = values.astype(np.float64, copy=True)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(
                "Every scored and unscored prediction must be finite and positive"
            )
        predictions[arm] = values
    return predictions


def _qlike(y, prediction):
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
            ratio = np.float64(y) / np.float64(prediction)
            loss = ratio - np.log(ratio) - 1.0
    except FloatingPointError as error:
        raise ValueError(
            "Invalid QLIKE arithmetic; no clipping or fallback allowed"
        ) from error
    if not np.isfinite(loss):
        raise ValueError("Nonfinite QLIKE loss")
    return float(loss)


def _frame(rows, columns):
    result = pd.DataFrame(rows, columns=columns)
    for name in DATE_COLUMNS & set(columns):
        result[name] = pd.to_datetime(result[name]).astype("datetime64[ns]")
    for name in {"train_n", "train_releases", "application_n", "requested_n", "offset"} & set(
        columns
    ):
        result[name] = pd.array(result[name], dtype="Int64")
    for name in {"feature_complete", "target_observed", "target_within_phase", "scored"} & set(
        columns
    ):
        result[name] = result[name].astype(bool)
    for name in {
        "prediction",
        "y",
        "loss",
        "pred_market",
        "pred_matched",
        "pred_candidate",
    } & set(columns):
        result[name] = result[name].astype(float)
    return result


def build_panel(features: pd.DataFrame, targets: pd.DataFrame, config: dict) -> dict:
    """Build expanding monthly forecasts; any support or arithmetic failure aborts."""
    config = _configuration(config)
    calendar, matrix, complete, y, ends, weeks, releases, statuses = _inputs(
        features, targets, config
    )
    requested = (calendar >= config["origin_start"]) & (calendar <= config["origin_end"])
    applications, schedules, fits = [], [], []
    application_by_position = {}
    for month in pd.period_range(config["origin_start"], config["origin_end"], freq="M"):
        positions = np.flatnonzero(requested & (calendar.to_period("M") == month))
        application_positions = positions[complete[positions]]
        schedule = {
            "month": str(month),
            "status": "no_requested_origins" if not len(positions) else "no_complete_origin",
            "fit_origin": pd.NaT,
            "training_cutoff": pd.NaT,
            "requested_n": len(positions),
            "application_n": len(application_positions),
            "train_n": None,
            "train_releases": None,
        }
        if not len(application_positions):
            schedules.append(schedule)
            continue
        # The schedule is fixed from features before any query-label selection.
        first = int(application_positions[0])
        fit_origin = calendar[first]
        if first == 0:
            raise ValueError(
                "INSUFFICIENT_DATA: Insufficient training support: fit has no previous full session"
            )
        cutoff = calendar[first - 1]
        train_positions = np.flatnonzero(
            complete & (calendar < fit_origin) & np.isfinite(y) & (ends <= cutoff)
        )
        train_n = len(train_positions)
        train_releases = len(releases[train_positions].unique())
        if (
            train_n < config["minimum_train"]
            or train_releases < config["minimum_train_releases"]
        ):
            raise ValueError(
                f"INSUFFICIENT_DATA: Insufficient training support at {fit_origin.date()}: {train_n} rows, {train_releases} releases"
            )
        fitted = fit_models(
            features.iloc[train_positions],
            targets["y"].iloc[train_positions],
            features.iloc[application_positions],
        )
        predictions = _prediction_arrays(fitted, len(application_positions))
        schedule.update(
            status="fitted",
            fit_origin=fit_origin,
            training_cutoff=cutoff,
            train_n=train_n,
            train_releases=train_releases,
        )
        schedules.append(schedule)
        fits.append(
            {
                "month": str(month),
                "fit_origin": fit_origin.date().isoformat(),
                "training_cutoff": cutoff.date().isoformat(),
                "train_origins": [
                    date.date().isoformat() for date in calendar[train_positions]
                ],
                "application_origins": [
                    date.date().isoformat() for date in calendar[application_positions]
                ],
                "train_n": train_n,
                "train_releases": train_releases,
                "model_audits": fitted["fits"],
            }
        )
        for local, position in enumerate(application_positions):
            row = {
                "origin": calendar[position],
                "fit_origin": fit_origin,
                "training_cutoff": cutoff,
                "train_n": train_n,
                "train_releases": train_releases,
                "claim_reference_week": weeks[position],
                "claim_release_date": releases[position],
                "offset": int(position % 5),
                "phase": _phase(calendar[position], config),
                **{f"pred_{arm}": float(predictions[arm][local]) for arm in ARMS},
            }
            applications.append(row)
            application_by_position[position] = row
    coverage, panel = [], []
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
        row = {
            "origin": origin,
            "phase": phase,
            "feature_complete": bool(complete[position]),
            "missing_features": "|".join(
                name for name, value in zip(ALL, matrix[position]) if np.isnan(value)
            ),
            "claim_status": statuses[position],
            "claim_reference_week": weeks[position],
            "claim_release_date": releases[position],
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
        coverage.append(row)
        if status == "scored":
            for arm in ARMS:
                prediction = application[f"pred_{arm}"]
                panel.append(
                    {
                        "origin": origin,
                        "model": arm,
                        "prediction": prediction,
                        "y": float(y[position]),
                        "loss": _qlike(y[position], prediction),
                        "target_end": ends[position],
                        "phase": phase,
                        **{
                            name: application[name]
                            for name in (
                                "claim_reference_week",
                                "claim_release_date",
                                "offset",
                                "fit_origin",
                                "training_cutoff",
                                "train_n",
                                "train_releases",
                            )
                        },
                    }
                )
    return {
        "applications": _frame(applications, APPLICATION_COLUMNS),
        "panel": _frame(panel, PANEL_COLUMNS),
        "coverage": _frame(coverage, COVERAGE_COLUMNS),
        "schedules": _frame(schedules, SCHEDULE_COLUMNS),
        "fits": fits,
    }
