"""Direct cross-moment scores for an intact, already verified forecast panel.

This module neither reads sources nor fits models. The caller must pin the full
wave10 panel bytes and its successful independent verification. Forecast-only
inequalities cannot independently reconstruct the source session calendar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INPUT_COLUMNS = (
    "origin",
    "model",
    "horizon",
    "feature_cutoff_date",
    "target_end",
    "available_date",
    "y_qqq",
    "y_spx",
    "mu_qqq",
    "mu_spx",
    "h_qqq",
    "h_spx",
    "rho",
    "loss",
    "fit_origin",
    "fit_cutoff_date",
    "train_n",
    "train_last_target",
    "train_last_available",
    "phase",
)
MODELS = ("constant_matrix", "constant_correlation", "dynamic_correlation")
ADDED_COLUMNS = ("realized_product", "forecast_product", "product_mse")
DATE_COLUMNS = (
    "origin",
    "feature_cutoff_date",
    "target_end",
    "available_date",
    "fit_origin",
    "fit_cutoff_date",
    "train_last_target",
    "train_last_available",
)
NUMERIC_COLUMNS = (
    "horizon",
    "y_qqq",
    "y_spx",
    "mu_qqq",
    "mu_spx",
    "h_qqq",
    "h_spx",
    "rho",
    "loss",
    "train_n",
)
COHERENCE_MULTIPLIER = 64.0


def _finite(values, label):
    if np.iscomplexobj(values):
        raise ValueError("Real numeric " + label + " required")
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite " + label + "; no clipping or omission")
    return array


def _product(left, right, label):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.multiply(left, right)
    if not np.isfinite(result).all():
        raise ValueError("Unrepresentable " + label + "; nonfinite multiplication")
    if np.any((left != 0) & (right != 0) & (result == 0)):
        raise ValueError("Unrepresentable " + label + "; nonzero multiplication underflow")
    return result


def _subtract(left, right, label):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.subtract(left, right)
    return _finite(result, label)


def _squared_error(forecast, actual):
    error = _subtract(forecast, actual, "product forecast error")
    return error, _product(error, error, "product squared error")


def _coherence_allowance(*terms):
    """Scale-aware normal rounding or downward ULP allowances; zero stays zero."""
    allowance = np.zeros_like(terms[0])
    for term in terms:
        magnitude = np.abs(term)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            downward_ulp = magnitude - np.nextafter(magnitude, 0.0)
            unit = np.maximum(np.finfo(float).eps * magnitude, downward_ulp)
            allowance += COHERENCE_MULTIPLIER * unit
    return _finite(allowance, "paired score coherence allowance")


def validate_panel(panel):
    """Check every row/model/metadata before calculating any new product."""
    if (
        not isinstance(panel, pd.DataFrame)
        or panel.empty
        or tuple(panel.columns) != INPUT_COLUMNS
    ):
        raise ValueError("Exact nonempty twenty-column issued forecast schema required")
    if (
        set(panel.model) != set(MODELS)
        or panel.duplicated(["origin", "model", "horizon"]).any()
    ):
        raise ValueError("Exactly three distinct issued models per origin required")
    for name in DATE_COLUMNS:
        if not pd.api.types.is_datetime64_any_dtype(panel[name].dtype):
            raise ValueError("Date-only datetime metadata required: " + name)
        dates = pd.DatetimeIndex(panel[name])
        if dates.tz is not None or dates.hasnans or not dates.equals(dates.normalize()):
            raise ValueError(
                "Finite timezone-naive normalized date metadata required: " + name
            )
    for name in NUMERIC_COLUMNS:
        if panel[name].dtype.kind not in "iuf":
            raise ValueError("Original numeric forecast fields required: " + name)
        _finite(panel[name].to_numpy(), name)
    if (
        not panel.horizon.eq(1).all()
        or not (panel.train_n.ge(1000) & panel.train_n.eq(np.floor(panel.train_n))).all()
    ):
        raise ValueError("Horizon one and at least1000 integer training rows required")
    if not (panel.h_qqq.gt(0) & panel.h_spx.gt(0)).all():
        raise ValueError("Finite strictly positive issued marginal moments required")
    conditional = panel.model.ne("constant_matrix")
    if (
        panel.loc[conditional, "rho"].abs().gt(0.995).any()
        or panel.loc[~conditional, "rho"].abs().gt(1 - 1e-6).any()
    ):
        raise ValueError("Issued correlation exceeds its registered model bound")
    development = panel.origin.between("2016-01-04", "2019-12-31")
    evaluation = panel.origin.between("2020-01-02", "2025-10-17")
    if not (development | evaluation).all() or not np.array_equal(
        panel.phase.to_numpy(), np.where(development, "development", "evaluation")
    ):
        raise ValueError("Fixed development/evaluation origin and phase fences required")
    if (
        not panel.target_end.equals(panel.available_date)
        or not panel.train_last_target.equals(panel.train_last_available)
        or not (panel.available_date <= pd.Timestamp("2025-10-20")).all()
        or not (panel.loc[development, "available_date"] <= pd.Timestamp("2019-12-31")).all()
    ):
        raise ValueError("Exact paired label maturity and fixed phase fences required")
    if not (
        panel.feature_cutoff_date.lt(panel.origin)
        & panel.origin.lt(panel.target_end)
        & panel.fit_cutoff_date.lt(panel.fit_origin)
        & panel.fit_origin.le(panel.origin)
        & panel.fit_cutoff_date.le(panel.feature_cutoff_date)
        & panel.train_last_available.le(panel.fit_cutoff_date)
    ).all():
        raise ValueError("Issued predictor/training cutoff or label chronology is invalid")
    if not panel.fit_origin.dt.to_period("M").equals(panel.origin.dt.to_period("M")):
        raise ValueError("Monthly fit must belong to its forecast origin month")
    first = panel.loc[panel.model == MODELS[0]].set_index("origin").sort_index()
    shared = [
        name
        for name in INPUT_COLUMNS
        if name not in ("origin", "model", "h_qqq", "h_spx", "rho", "loss")
    ]
    for model in MODELS[1:]:
        rows = panel.loc[panel.model == model].set_index("origin").sort_index()
        if not rows.index.equals(first.index) or not rows.loc[:, shared].equals(
            first.loc[:, shared]
        ):
            raise ValueError(
                "Exact paired model cohorts, targets, means and metadata required"
            )
    base = panel.loc[panel.model == "constant_correlation"].set_index("origin").sort_index()
    candidate = (
        panel.loc[panel.model == "dynamic_correlation"].set_index("origin").sort_index()
    )
    if not base[["h_qqq", "h_spx"]].equals(candidate[["h_qqq", "h_spx"]]):
        raise ValueError("Conditional models must share exact issued diagonal forecasts")
    fit_fields = [
        "fit_origin",
        "fit_cutoff_date",
        "train_n",
        "train_last_target",
        "train_last_available",
    ]
    monthly = first.groupby(first.index.to_period("M"), sort=False)
    if any(group[fit_fields].nunique(dropna=False).gt(1).any() for _, group in monthly):
        raise ValueError("One fixed mature training fit per application month required")


def score_panel(panel):
    """Append signed products and product MSE without reordering or filtering."""
    validate_panel(panel)
    left = _subtract(panel.y_qqq.to_numpy(), panel.mu_qqq.to_numpy(), "QQQ issued residual")
    right = _subtract(panel.y_spx.to_numpy(), panel.mu_spx.to_numpy(), "SPX issued residual")
    realized = _product(left, right, "realized residual cross product")
    geometric = _product(
        np.sqrt(panel.h_qqq.to_numpy()),
        np.sqrt(panel.h_spx.to_numpy()),
        "geometric marginal moment",
    )
    forecast = _product(geometric, panel.rho.to_numpy(), "forecast cross product")
    _, loss = _squared_error(forecast, realized)
    result = panel.copy(deep=True)
    result[ADDED_COLUMNS[0]] = realized
    result[ADDED_COLUMNS[1]] = forecast
    result[ADDED_COLUMNS[2]] = loss
    return result


def paired_difference(candidateq, controlq, Y):
    """Factored signed MSE gap, with both individual losses admissible first."""
    candidate, control, actual = (
        _finite(values, name)
        for values, name in (
            (candidateq, "candidate cross product"),
            (controlq, "control cross product"),
            (Y, "realized cross product"),
        )
    )
    if (
        candidate.ndim != 1
        or not len(candidate)
        or control.shape != candidate.shape
        or actual.shape != candidate.shape
    ):
        raise ValueError("Finite aligned nonempty one-dimensional products required")
    candidate_error, candidate_loss = _squared_error(candidate, actual)
    control_error, control_loss = _squared_error(control, actual)
    first = _subtract(candidate, control, "paired forecast difference")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        second = candidate_error + control_error
    second = _finite(second, "paired error sum")
    difference = _product(first, second, "paired product-loss difference")
    direct = _subtract(candidate_loss, control_loss, "direct product-loss difference")
    allowance = _coherence_allowance(candidate_loss, control_loss, difference)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        discrepancy = np.abs(difference - direct)
    if np.any(discrepancy > allowance):
        raise ValueError(
            "Factored and direct product-loss differences are numerically incoherent"
        )
    return difference
