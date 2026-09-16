"""Bounded, measurement-first joint-copula issuance on the full SPX calendar."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.claims_release_pipeline import _dates, _iso, _numeric
from src.joint_copula_models import MODELS, TARGETS, fit_predict
from src.joint_risk_features import (
    ALL_FEATURES,
    build_features,
    measurement_audit,
    require_measurement,
)

APPLICATION_COLUMNS = (
    "origin",
    "model",
    "horizon",
    "feature_cutoff_date",
    "mu_qqq",
    "mu_spx",
    "h_qqq",
    "h_spx",
    "rho",
    "fit_origin",
    "training_cutoff",
    "train_n",
    "phase",
    "offset",
)
PANEL_COLUMNS = (*APPLICATION_COLUMNS, "target_end", "available_date", "y_qqq", "y_spx")
COVERAGE_COLUMNS = (
    "origin",
    "phase",
    "feature_complete",
    "missing_features",
    "feature_cutoff_date",
    "offset",
    "target_end",
    "available_date",
    "target_observed",
    "target_within_phase",
    "issued",
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
    "feature_complete_n",
    "application_n",
    "train_n",
)


class PipelineExecutionError(ValueError):
    def __init__(self, message, produced, *, month=None, stage=None, measurement=None):
        super().__init__(message)
        self.produced, self.month, self.stage = produced, month, stage
        self.measurement_audit = measurement


def _config(config):
    if config is None:
        config = {
            "origin_start": "2016-01-04",
            "origin_end": "2025-10-17",
            "source_end": "2025-10-20",
            "development": ["2016-01-04", "2019-12-31"],
            "evaluation": ["2020-01-01", "2025-10-20"],
            "minimum_train": 1000,
        }
    if type(config) is not dict or set(config) != {
        "origin_start",
        "origin_end",
        "source_end",
        "development",
        "evaluation",
        "minimum_train",
    }:
        raise ValueError("Exact six pipeline configuration fields required")
    parsed = {k: _iso(config[k]) for k in ("origin_start", "origin_end", "source_end")}
    for p in ("development", "evaluation"):
        if type(config[p]) is not list or len(config[p]) != 2:
            raise ValueError("Two literal phase dates required")
        parsed[p] = tuple(_iso(x) for x in config[p])
    if not (
        parsed["origin_start"]
        <= parsed["development"][0]
        <= parsed["development"][1]
        < parsed["evaluation"][0]
        <= parsed["origin_end"]
        <= parsed["evaluation"][1]
        <= parsed["source_end"]
        <= pd.Timestamp("2025-10-20")
    ):
        raise ValueError("Ordered bounded origin and phase dates required")
    if type(config["minimum_train"]) is not int or config["minimum_train"] < 2:
        raise ValueError("Exact training floor at least two required")
    parsed["minimum_train"] = config["minimum_train"]
    return parsed


def _frame(rows, columns):
    out = pd.DataFrame(rows, columns=columns)
    for name in {
        "origin",
        "feature_cutoff_date",
        "target_end",
        "available_date",
        "fit_origin",
        "training_cutoff",
    } & set(columns):
        out[name] = pd.to_datetime(out[name]).astype("datetime64[ns]")
    for name in {
        "horizon",
        "offset",
        "train_n",
        "requested_n",
        "feature_complete_n",
        "application_n",
    } & set(columns):
        out[name] = pd.array(out[name], dtype="Int64")
    for name in {
        "feature_complete",
        "target_observed",
        "target_within_phase",
        "issued",
        "scored",
    } & set(columns):
        out[name] = out[name].astype(bool)
    for name in {"mu_qqq", "mu_spx", "h_qqq", "h_spx", "rho", "y_qqq", "y_spx"} & set(columns):
        out[name] = out[name].astype(float)
    return out


def _phase(date, config):
    for phase in ("development", "evaluation"):
        if config[phase][0] <= date <= config[phase][1]:
            return phase
    return "outside_phase"


def produce(qqq, spx, iv, config=None, *, measurement_callback=None):
    config = _config(config)
    for source in (qqq, spx, iv):
        if not isinstance(source, pd.DataFrame) or not isinstance(
            source.index, pd.DatetimeIndex
        ):
            raise ValueError("Native source calendars required")
        dates = _dates(source.index, config["source_end"])
        if dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
            raise ValueError("Unique increasing bounded source calendars required")
    if not len(spx):
        raise ValueError("INSUFFICIENT_DATA: empty SPX calendar")
    if measurement_callback is not None and not callable(measurement_callback):
        raise ValueError("Measurement callback must be callable")
    output = {
        "features": pd.DataFrame(),
        "targets": pd.DataFrame(),
        "applications": _frame([], APPLICATION_COLUMNS),
        "panel": _frame([], PANEL_COLUMNS),
        "coverage": _frame([], COVERAGE_COLUMNS),
        "schedules": _frame([], SCHEDULE_COLUMNS),
        "fits": [],
    }
    measured = None
    try:
        measured = measurement_audit(qqq, spx)
        if measurement_callback is not None:
            measurement_callback(measured)
        require_measurement(measured)
    except Exception as error:
        raise PipelineExecutionError(
            str(error), output, stage="measurement", measurement=measured
        ) from error
    stage = "feature_construction"
    current_month = None
    try:
        features, targets = build_features(qqq, spx, iv)
        output.update(features=features, targets=targets)
        stage = "feature_alignment"
        calendar = spx.index
        if not features.index.equals(calendar) or not targets.index.equals(calendar):
            raise ValueError("Feature/target builder changed full reference calendar")
        previous = pd.DatetimeIndex(pd.Series(calendar).shift(1))
        following = pd.DatetimeIndex(pd.Series(calendar).shift(-1))
        for values, expected in (
            (features.feature_cutoff_date, previous),
            (targets.target_end, following),
            (targets.available_date, following),
        ):
            if not _dates(values, config["source_end"]).equals(expected):
                raise ValueError(
                    "Exact prior feature and next-session maturity dates required"
                )
        matrix = np.column_stack([_numeric(features[name]) for name in ALL_FEATURES])
        y = np.column_stack([_numeric(targets[name]) for name in TARGETS])
        if np.any(np.isfinite(y).any(axis=1) & following.isna()):
            raise ValueError("Finite target requires next-session endpoint")
        complete = np.isfinite(matrix).all(axis=1)
        known = np.isfinite(y).all(axis=1)
        requested = (calendar >= config["origin_start"]) & (calendar <= config["origin_end"])
        stage = "monthly_schedule"
        plans, schedules, applications, fits, issued = [], [], [], [], {}
        output["fits"] = fits
        for month in pd.period_range(config["origin_start"], config["origin_end"], freq="M"):
            positions = np.flatnonzero(requested & (calendar.to_period("M") == month))
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
                    "training_cutoff": previous[first] if first is not None else pd.NaT,
                    "requested_n": len(positions),
                    "feature_complete_n": len(query),
                    "application_n": 0,
                    "train_n": None,
                }
            )
            plans.append(query)

        output["schedules"] = _frame(schedules, SCHEDULE_COLUMNS)

        def result(aborted=False):
            rows = []
            for position in np.flatnonzero(requested):
                origin = calendar[position]
                phase = _phase(origin, config)
                within = (
                    phase != "outside_phase"
                    and pd.notna(following[position])
                    and following[position] <= config[phase][1]
                )
                app = issued.get(position)
                if not complete[position]:
                    status = "incomplete_features"
                elif aborted:
                    status = "attempt_aborted"
                elif phase == "outside_phase":
                    status = "outside_phase"
                elif pd.isna(following[position]):
                    status = "target_not_mature"
                elif not known[position]:
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
                            n for n, v in zip(ALL_FEATURES, matrix[position]) if np.isnan(v)
                        ),
                        "feature_cutoff_date": previous[position],
                        "offset": int(position % 5),
                        "target_end": following[position],
                        "available_date": following[position],
                        "target_observed": bool(known[position]),
                        "target_within_phase": bool(within),
                        "issued": app is not None,
                        "scored": status == "scored",
                        "status": status,
                        "fit_origin": app[0]["fit_origin"] if app else pd.NaT,
                        "training_cutoff": app[0]["training_cutoff"] if app else pd.NaT,
                    }
                )
            return {
                **output,
                "applications": _frame(applications, APPLICATION_COLUMNS),
                "panel": _frame([], PANEL_COLUMNS),
                "coverage": _frame(rows, COVERAGE_COLUMNS),
                "schedules": _frame(schedules, SCHEDULE_COLUMNS),
                "fits": fits,
            }

        def dates(positions):
            return [str(d.date()) for d in calendar[positions]]

        for schedule, query in zip(schedules, plans):
            if not len(query):
                continue
            current_month = schedule["month"]
            stage = "training_support"
            first = int(query[0])
            origin = calendar[first]
            cutoff = previous[first]
            train = np.flatnonzero(
                complete & known & (calendar < origin) & (following <= cutoff)
            )
            schedule["train_n"] = len(train)
            audit = {
                "month": schedule["month"],
                "status": "not_attempted",
                "fit_origin": str(origin.date()),
                "training_cutoff": str(cutoff.date()) if pd.notna(cutoff) else None,
                "train_origins": dates(train),
                "train_positions": train.tolist(),
                "train_n": len(train),
                "planned_application_origins": dates(query),
                "application_origins": [],
                "model_audit": None,
            }
            fits.append(audit)
            if pd.isna(cutoff) or len(train) < config["minimum_train"]:
                schedule["status"] = audit["status"] = "insufficient_training"
                raise PipelineExecutionError(
                    f"INSUFFICIENT_DATA: {len(train)} common mature pairs at {origin.date()}",
                    result(True),
                    month=schedule["month"],
                    stage="training_support",
                    measurement=measured,
                )
            stage = "fit_predict"
            try:
                predictions, model_audit = fit_predict(
                    features.iloc[train], targets.iloc[train], features.iloc[query]
                )
                audit["model_audit"] = model_audit
                stage = "prediction_validation"
                if set(predictions) != set(MODELS):
                    raise ValueError("Every declared density arm required")
                reference = predictions[MODELS[0]]
                for name, p in predictions.items():
                    if (
                        set(p) != {"mu", "h", "rho"}
                        or np.shape(p["mu"]) != (len(query), 2)
                        or np.shape(p["h"]) != (len(query), 2)
                        or np.shape(p["rho"]) != (len(query),)
                        or not all(np.isfinite(p[k]).all() for k in p)
                        or np.any(p["h"] <= 0)
                        or np.any(np.abs(p["rho"]) > 0.995)
                        or not np.array_equal(p["mu"], reference["mu"])
                        or not np.array_equal(p["h"], reference["h"])
                        or (name == "independence" and np.any(p["rho"] != 0))
                    ):
                        raise ValueError(
                            "Finite exact shared marginals and bounded dependence required"
                        )
                stage = "application_issuance"
                for local, position in enumerate(query):
                    rowset = []
                    for name in MODELS:
                        p = predictions[name]
                        row = {
                            "origin": calendar[position],
                            "model": name,
                            "horizon": 1,
                            "feature_cutoff_date": previous[position],
                            "mu_qqq": float(p["mu"][local, 0]),
                            "mu_spx": float(p["mu"][local, 1]),
                            "h_qqq": float(p["h"][local, 0]),
                            "h_spx": float(p["h"][local, 1]),
                            "rho": float(p["rho"][local]),
                            "fit_origin": origin,
                            "training_cutoff": cutoff,
                            "train_n": len(train),
                            "phase": _phase(calendar[position], config),
                            "offset": int(position % 5),
                        }
                        rowset.append(row)
                        applications.append(row)
                    issued[int(position)] = rowset
                schedule.update(status="fitted", application_n=len(query))
                audit.update(status="fitted", application_origins=dates(query))
                output["applications"] = _frame(applications, APPLICATION_COLUMNS)
                output["schedules"] = _frame(schedules, SCHEDULE_COLUMNS)
            except Exception as error:
                schedule["status"] = audit["status"] = "execution_failed"
                raise PipelineExecutionError(
                    f"{schedule['month']} {stage}: {error}",
                    result(True),
                    month=schedule["month"],
                    stage=stage,
                    measurement=measured,
                ) from error
        stage = "result_assembly"
        current_month = None
        out = result()
        output.update(out)
        panel = []
        for row in out["coverage"].loc[out["coverage"].scored].itertuples(index=False):
            position = calendar.get_loc(row.origin)
            for app in issued[position]:
                panel.append(
                    {
                        **app,
                        "target_end": row.target_end,
                        "available_date": row.available_date,
                        "y_qqq": float(y[position, 0]),
                        "y_spx": float(y[position, 1]),
                    }
                )
        out["panel"] = _frame(panel, PANEL_COLUMNS)
        return out
    except PipelineExecutionError:
        raise
    except Exception as error:
        # Preserve completed snapshots even if the final table builder itself fails.
        # An aborted attempt never exposes a completed scored panel.
        output["panel"] = output["panel"].iloc[:0].copy()
        if not output["coverage"].empty:
            output["coverage"] = output["coverage"].copy()
            output["coverage"].loc[:, "scored"] = False
            complete_rows = output["coverage"]["feature_complete"]
            output["coverage"].loc[complete_rows, "status"] = "attempt_aborted"
        raise PipelineExecutionError(
            f"{stage}: {error}",
            output,
            month=current_month,
            stage=stage,
            measurement=measured,
        ) from error
