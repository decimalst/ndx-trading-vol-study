"""Causal scalar calibration of immutable, genuinely issued baseline logits.

No baseline is fitted here. All source tables and original fits are supplied,
and every original application replays before any new scalar fit is attempted.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit

from . import range_alert_features as feature
from . import range_alert_models as original
from .cross_moment_score import _product

MODELS = ("baseline", "recent_frequency", "calibrated")
PANEL_COLUMNS = original.PANEL_COLUMNS
HALF_LIFE = 63
DECAY = 2.0 ** (-1.0 / HALF_LIFE)
LABEL_WEIGHT = 1.0 - DECAY
ALPHA = 0.01
XTOL = 1e-12
RTOL = 1e-14
MAX_ITERATIONS = 200
GRADIENT_TOLERANCE = 1e-8
REPLAY_RTOL = 1e-10
REPLAY_ATOL = 1e-12
STATE_COLUMNS = (
    "origin",
    "feature_cutoff_date",
    "source_fit_origin",
    "seed_cutoff_date",
    "elapsed_sessions",
    "history_n",
    "weight_sum",
    "latest_admitted_origin",
    "latest_admitted_available",
    "missing_label_arrivals",
    "no_forecast_arrivals",
    "intercept",
    "baseline_logit",
    "baseline_probability",
    "calibrated_probability",
    "scored",
    "status",
    "objective",
    "gradient",
    "gradient_max_abs",
)


def _finite(values, name):
    array = np.asarray(values)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"Real numeric {name} required")
    array = array.astype(float)
    if not np.isfinite(array).all():
        raise ValueError(f"Finite {name} required")
    return array


def _scalar(value, name):
    value = _finite(value, name)
    if value.ndim:
        raise ValueError(f"Scalar {name} required")
    return float(value)


def _sum(values, name):
    values = _finite(values, name)
    try:
        result = math.fsum(values.ravel())
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"Finite compensated {name} required") from exc
    return _scalar(result, name)


def _history(eta, y, weights):
    eta, y, weights = (
        _finite(eta, "issued logits"),
        _finite(y, "mature binary labels"),
        _finite(weights, "record weights"),
    )
    if eta.ndim != 1 or y.shape != eta.shape or weights.shape != eta.shape:
        raise ValueError("Exactly aligned one-dimensional calibration records required")
    if not np.isin(y, [0.0, 1.0]).all() or np.any(weights <= 0):
        raise ValueError("Binary labels and strictly positive record weights required")
    mass = _sum(weights, "unnormalized weight mass")
    if mass > 1:
        raise ValueError("At most one arrival per full reference close; mass cannot exceed1")
    return eta, y, weights, mass


def calibration_objective(intercept, eta, y, weights):
    """Signed binary NLL plus fixed ridge, and its cancellation-safe derivative."""
    a = _scalar(intercept, "intercept")
    eta, y, weights, _ = _history(eta, y, weights)
    with np.errstate(all="ignore"):
        logits = _finite(eta + a, "corrected historical logits")
        losses = np.logaddexp(0.0, (1 - 2 * y) * logits)
        residual = np.where(y == 1, -expit(-logits), expit(logits))
    if (
        not np.isfinite(losses).all()
        or np.any(losses <= 0)
        or not np.isfinite(residual).all()
        or np.any(residual == 0)
    ):
        raise ValueError("Positive finite signed loss/residual required; no underflow repair")
    likelihood = _sum(
        _product(weights, losses, "weighted binary losses"), "weighted likelihood"
    )
    error = _sum(_product(weights, residual, "weighted signed residuals"), "weighted gradient")
    penalty = float(_product(ALPHA, _product(a, a, "intercept square"), "intercept penalty"))
    ridge_gradient = float(_product(2 * ALPHA, a, "ridge gradient"))
    value = _sum(np.array([likelihood, penalty]), "penalized objective")
    gradient = _sum(np.array([error, ridge_gradient]), "full scalar gradient")
    if value < 0:
        raise ValueError("Nonnegative scalar objective required")
    return value, gradient


def fit_intercept(eta, y, weights):
    """One fixed bracketed derivative solve, including exact empty/balanced zero."""
    eta, y, weights, mass = _history(eta, y, weights)
    _, gradient0 = calibration_objective(0.0, eta, y, weights)
    radius = (mass + 1.0) / (2 * ALPHA)
    left, right = -radius, radius
    lower = calibration_objective(left, eta, y, weights)[1]
    upper = calibration_objective(right, eta, y, weights)[1]
    if not lower < 0 < upper:
        raise ValueError("Strict finite derivative bracket failed")
    iterations, calls, status = 0, 0, "EMPTY_HISTORY" if not len(eta) else "BALANCED_AT_ZERO"
    if not len(eta) or gradient0 == 0:
        coefficient = 0.0
    else:
        try:
            coefficient, result = brentq(
                lambda a: calibration_objective(a, eta, y, weights)[1],
                left,
                right,
                xtol=XTOL,
                rtol=RTOL,
                maxiter=MAX_ITERATIONS,
                full_output=True,
                disp=False,
            )
        except (RuntimeError, ValueError) as exc:
            raise ValueError(
                "Calibration derivative solve failed; no retry or fallback"
            ) from exc
        if (
            result.converged is not True
            or isinstance(result.iterations, bool)
            or not isinstance(result.iterations, (int, np.integer))
            or not 0 <= result.iterations <= MAX_ITERATIONS
            or isinstance(result.function_calls, bool)
            or not isinstance(result.function_calls, (int, np.integer))
            or result.function_calls < 2
        ):
            raise ValueError("Fixed converged scalar solver audit required")
        iterations, calls, status = (
            int(result.iterations),
            int(result.function_calls),
            "FITTED",
        )
    coefficient = _scalar(coefficient, "fitted intercept")
    value, gradient = calibration_objective(coefficient, eta, y, weights)
    if not left <= coefficient <= right or abs(gradient) > GRADIENT_TOLERANCE:
        raise ValueError("Original full scalar gradient gate failed")
    return {
        "intercept": coefficient,
        "alpha": ALPHA,
        "history_n": len(eta),
        "weight_sum": mass,
        "objective": value,
        "gradient": gradient,
        "gradient_max_abs": abs(gradient),
        "gradient_at_zero": gradient0,
        "bracket": [left, right],
        "bracket_gradients": [lower, upper],
        "iterations": iterations,
        "function_calls": calls,
        "converged": True,
        "status": status,
        "method": "brentq",
        "xtol": XTOL,
        "rtol": RTOL,
        "maxiter": MAX_ITERATIONS,
        "curvature_lower": 2 * ALPHA,
        "curvature_upper": 2 * ALPHA + 0.25 * mass,
    }


def _replay_application(app, audit, saved_probability):
    baseline, geometry = audit["baseline"], audit["transform"]
    if (
        baseline["columns"] != list(feature.BASE)
        or geometry["columns"] != list(feature.BASE)
        or baseline["means"] != geometry["means"]
        or baseline["scales"] != geometry["scales"]
    ):
        raise ValueError("Original saved baseline geometry/order differs")
    means, scales, beta = (
        _finite(baseline["means"], "saved means"),
        _finite(baseline["scales"], "saved scales"),
        _finite(baseline["beta"], "saved coefficients"),
    )
    if (
        means.shape != (26,)
        or scales.shape != (26,)
        or beta.shape != (26,)
        or means[0] != 0
        or scales[0] != 1
        or np.any(scales[1:] <= 1e-12)
    ):
        raise ValueError("Exact26-column finite original geometry required")
    raw = app.loc[:, feature.RAW].copy()
    for column in ("I", "R", "skew"):
        center = _scalar(geometry["curvature_means"][column], "original curvature mean")
        difference = _finite(raw[column].to_numpy() - center, "original centered feature")
        raw[column + "_square"] = _product(difference, difference, "original curvature square")
    # Preserve the original dataframe evaluation and column layout. No new
    # centering, scale, coefficient, optimizer or hindsight calibration enters.
    numerator = raw.loc[:, feature.BASE] - pd.Series(means, index=feature.BASE)
    with np.errstate(all="ignore"):
        design = numerator / pd.Series(scales, index=feature.BASE)
    values = _finite(design, "saved standardized application design")
    if np.any((numerator.to_numpy() != 0) & (values == 0)):
        raise ValueError("Nonzero saved-coordinate division underflow")
    _product(values, beta, "saved coefficient products")
    with np.errstate(all="ignore"):
        eta = _finite(values @ beta, "issued application logits")
    probability = original._probabilities(saved_probability, len(app))
    replayed = expit(eta)
    if not np.allclose(replayed, probability, rtol=REPLAY_RTOL, atol=REPLAY_ATOL):
        raise ValueError("Original issued baseline probability does not replay")
    return eta, probability


def replay_issued(features, targets, upstream_panel, upstream_fits, upstream_states, config):
    """Admit all original application records and both controls before calibration."""
    support = original.preflight(features, targets, config)
    applications, labels, dev_end = original.application_calendar(features, targets, config)
    scored = applications[labels.loc[applications].to_numpy()]
    original.validate_panel(upstream_panel)
    if not isinstance(upstream_fits, list) or len(upstream_fits) != support["monthly_fits"]:
        raise ValueError("Every original monthly fit must be present")
    base = upstream_panel.loc[upstream_panel.model.eq("baseline")].sort_values("origin")
    if not pd.DatetimeIndex(base.origin).equals(scored):
        raise ValueError("Original scored cohort differs from full application/label contract")
    original_expected = {
        "feature_cutoff_date": features.loc[scored, "feature_cutoff_date"].to_numpy(),
        "target_end": targets.loc[scored, "target_end"].to_numpy(),
        "available_date": targets.loc[scored, "available_date"].to_numpy(),
        "y": targets.loc[scored, "y"].to_numpy(),
        "phase": np.where(scored <= dev_end, "development", "evaluation"),
    }
    for name, expected in original_expected.items():
        if not np.array_equal(base[name], expected):
            raise ValueError(f"Original forecast {name} differs from source contract")
    issued, scored_set = [], set(scored)
    for expected, fit in zip(support["fits"], upstream_fits, strict=True):
        audit = fit["model_audit"]
        for key, value in expected.items():
            if (audit[key] if key in ("support", "transform") else fit[key]) != value:
                raise ValueError(
                    f"Original monthly {key} differs from unchanged mature geometry"
                )
        entry = pd.Timestamp(fit["fit_origin"])
        dates = applications[applications.to_period("M") == entry.to_period("M")]
        if fit["application_origins"] != [str(date.date()) for date in dates]:
            raise ValueError("Every originally issued application must be retained in order")
        if audit["application_n"] != len(dates) or audit["train_n"] != fit["train_n"]:
            raise ValueError("Original fitted application/training counts differ")
        eta, saved = _replay_application(
            features.loc[dates], audit, fit["application_probabilities"]["baseline"]
        )
        part = base.loc[base.origin.isin(dates)]
        positions = dates.get_indexer(pd.DatetimeIndex(part.origin))
        if not np.array_equal(part.probability, saved[positions]):
            raise ValueError(
                "Original scored baseline differs from original application issue"
            )
        for column in (
            "fit_origin",
            "fit_cutoff_date",
            "train_last_target",
            "train_last_available",
        ):
            if not part[column].eq(pd.Timestamp(fit[column])).all():
                raise ValueError("Original scored monthly provenance differs")
        if not part.train_n.eq(fit["train_n"]).all():
            raise ValueError("Original scored training count differs")
        for date, logit, probability in zip(dates, eta, saved, strict=True):
            issued.append(
                {
                    "origin": str(date.date()),
                    "feature_cutoff_date": str(
                        features.loc[date, "feature_cutoff_date"].date()
                    ),
                    "source_fit_origin": fit["fit_origin"],
                    "baseline_logit": float(logit),
                    "baseline_probability": float(probability),
                    "scored": date in scored_set,
                }
            )
    if (
        not isinstance(upstream_states, pd.DataFrame)
        or tuple(upstream_states.columns) != original.STATE_COLUMNS
    ):
        raise ValueError("Exact original event-rate state schema required")
    expected_states = original._frequency_states(
        features, targets, applications, upstream_fits, scored
    )
    if not upstream_states.equals(expected_states):
        raise ValueError("Original full-application frequency state differs")
    frequency = upstream_panel.loc[upstream_panel.model.eq("recent_frequency")].sort_values(
        "origin"
    )
    if not np.array_equal(
        frequency.probability, expected_states.loc[expected_states.scored, "recent_frequency"]
    ):
        raise ValueError("Original event-rate forecast differs from its original state")
    return issued, support


def _calibrate(issued, targets, reference):
    applications = pd.DatetimeIndex([record["origin"] for record in issued])
    indexed = {pd.Timestamp(record["origin"]): record for record in issued}
    seed = pd.Timestamp(issued[0]["feature_cutoff_date"])
    seed_position = reference.get_loc(seed)
    cursor = seed_position
    weights, historical_eta, historical_y = np.array([]), [], []
    admitted, arrivals, states, fits = [], [], [], []
    missing, absent = 0, 0
    latest_origin, latest_available = pd.NaT, pd.NaT
    for origin in applications:
        cutoff_position = reference.get_loc(origin) - 1
        while cursor < cutoff_position:
            cursor += 1
            weights = _product(DECAY, weights, "full-calendar aged calibration weights")
            source_origin, available = reference[cursor - 1], reference[cursor]
            one = indexed.get(source_origin)
            if one is None:
                absent += 1
                status = "NO_ISSUED_FORECAST"
            elif pd.isna(targets.loc[source_origin, "y"]):
                missing += 1
                status = "UNKNOWN_LABEL"
            else:
                label = float(targets.loc[source_origin, "y"])
                weights = np.append(weights, LABEL_WEIGHT)
                historical_eta.append(one["baseline_logit"])
                historical_y.append(label)
                admitted.append(
                    {
                        "origin": str(source_origin.date()),
                        "available_date": str(available.date()),
                        "baseline_logit": one["baseline_logit"],
                        "y": label,
                    }
                )
                latest_origin, latest_available = source_origin, available
                status = "ADMITTED"
            arrivals.append(
                {
                    "origin": str(source_origin.date()),
                    "available_date": str(available.date()),
                    "status": status,
                }
            )
        audit = fit_intercept(np.array(historical_eta), np.array(historical_y), weights)
        one = indexed[origin]
        eta, probability, coefficient = (
            one["baseline_logit"],
            one["baseline_probability"],
            audit["intercept"],
        )
        corrected_eta = _scalar(eta + coefficient, "corrected application logit")
        # Fixed identity branch preserves genuine issued rounding at exact zero.
        corrected_probability = (
            probability if coefficient == 0 else float(expit(corrected_eta))
        )
        original._probabilities(np.array([corrected_probability]), 1)
        states.append(
            {
                "origin": origin,
                "feature_cutoff_date": reference[cutoff_position],
                "source_fit_origin": pd.Timestamp(one["source_fit_origin"]),
                "seed_cutoff_date": seed,
                "elapsed_sessions": cutoff_position - seed_position,
                "history_n": len(historical_eta),
                "weight_sum": audit["weight_sum"],
                "latest_admitted_origin": latest_origin,
                "latest_admitted_available": latest_available,
                "missing_label_arrivals": missing,
                "no_forecast_arrivals": absent,
                "intercept": coefficient,
                "baseline_logit": eta,
                "baseline_probability": probability,
                "calibrated_probability": corrected_probability,
                "scored": one["scored"],
                "status": audit["status"],
                "objective": audit["objective"],
                "gradient": audit["gradient"],
                "gradient_max_abs": audit["gradient_max_abs"],
            }
        )
        fits.append({"origin": str(origin.date()), **audit})
    return pd.DataFrame(states, columns=STATE_COLUMNS), {
        "seed_origin": issued[0]["origin"],
        "seed_cutoff_date": str(seed.date()),
        "issued_applications": issued,
        "admitted_records": admitted,
        "arrival_audit": arrivals,
        "calibrations": fits,
    }


def validate_panel(panel):
    if (
        not isinstance(panel, pd.DataFrame)
        or tuple(panel.columns) != PANEL_COLUMNS
        or set(panel.model) != set(MODELS)
    ):
        raise ValueError("Exact three-model issued-calibration panel required")
    renamed = panel.copy()
    renamed["model"] = renamed.model.replace({"calibrated": "location"})
    original.validate_panel(renamed)
    return panel


def forecast_panel(
    features, targets, upstream_panel, upstream_fits, upstream_states, original_index_config
):
    """Preserve issued controls, calibrate all applications, then select old labels."""
    issued, support = replay_issued(
        features,
        targets,
        upstream_panel,
        upstream_fits,
        upstream_states,
        original_index_config,
    )
    states, audit = _calibrate(issued, targets, features.index)
    controls = upstream_panel.loc[
        upstream_panel.model.isin(["baseline", "recent_frequency"])
    ].copy()
    candidate = controls.loc[controls.model.eq("baseline")].sort_values("origin").copy()
    candidate["model"] = "calibrated"
    candidate["probability"] = (
        states.set_index("origin").loc[candidate.origin, "calibrated_probability"].to_numpy()
    )
    candidate["loss"] = original.brier_loss(
        candidate.probability.to_numpy(), candidate.y.to_numpy()
    )
    combined = (
        pd.concat([controls, candidate], ignore_index=True)
        .loc[:, PANEL_COLUMNS]
        .sort_values(["origin", "model"])
        .reset_index(drop=True)
    )
    validate_panel(combined)
    audit.update(
        common_application_origins=len(states),
        common_scored_origins=len(candidate),
        replayed_monthly_fits=len(upstream_fits),
        new_monthly_fits=0,
        support=support,
    )
    return combined, states, audit
