"""Independent SAS likelihood, convex-certificate and issued-archive replay.

No shape producer is imported. Frozen independent t8 log-tail inversion,
normal-coordinate and dependence-certificate oracles are explicitly reused.
Source byte authentication and prior baseline verification remain external.
"""

import math

import numpy as np
import pandas as pd
from scipy.special import ndtr

from src.verify_copula_calibration import (
    APP_SOURCE,
    T_CONSTANT,
    _archive,
    _array,
    _calibration,
    _compare,
    _from_normal,
    _index,
    _pairs,
    _standardized,
)
from src.verify_joint_copula_forecasts import verify_dependence
from src.verify_joint_copula_scores import (
    _independent_logcopula,
    _log_one_plus_square,
    _normal_coordinates,
)

ASSETS = ("qqq", "spx")
FAMILIES = ("gaussian", "t8")
PARAMETER_KEYS = {"location", "scale", "epsilon", "delta", "train_n", "optimizer_audits"}
APP_NEW = (
    "epsilon_qqq",
    "epsilon_spx",
    "delta_qqq",
    "delta_spx",
    "rho_shape_gaussian",
    "rho_shape_t8",
)
PANEL_NEW = APP_NEW + (
    "marginal_shape_qqq",
    "marginal_shape_spx",
    "pit_shape_qqq",
    "pit_shape_spx",
    "normal_shape_qqq",
    "normal_shape_spx",
    "d_qqq_shape",
    "d_spx_shape",
    "loss_shape_gaussian",
    "loss_shape_t8",
    "d_gaussian_shape",
    "d_t8_shape",
    "d_shape_gap",
    "d_shape_interaction",
)
ATOL, RTOL = 1e-10, 1e-8
CERTIFICATE_TOLERANCE = 1e-7


def _parameters(parameters):
    if not isinstance(parameters, dict):
        raise ValueError("Explicit shape parameter dictionary required")
    values = tuple(
        _array(parameters[key], (2,)) for key in ("location", "scale", "epsilon", "delta")
    )
    _, scale, epsilon, delta = values
    if (
        (scale <= 1e-12).any()
        or (abs(epsilon) > 0.75).any()
        or (delta < 0.75).any()
        or (delta > 2).any()
    ):
        raise ValueError("Parameters violate fixed affine floor or shape box")
    return values


def independent_shape_coordinates(z, h, parameters):
    """Normalized density and transformed PIT, independently reconstructed."""
    z = _pairs(z)
    h = _array(h, z.shape)
    if (h <= 0).any():
        raise ValueError("Strictly positive original variances required")
    location, scale, epsilon, delta = _parameters(parameters)
    w = _normal_coordinates(z)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            x = (w - location) / scale
            u = np.arcsinh(x) * delta - epsilon
            v = np.sinh(u)
            log_jacobian = (
                np.log(delta / scale)
                + np.logaddexp(u, -u)
                - math.log(2)
                - np.log(np.hypot(1.0, x))
            )
            original = (
                T_CONSTANT - 4.5 * _log_one_plus_square(z) - 0.5 * (np.log(h) + math.log(0.75))
            )
            density = original - 0.5 * (v * v - w * w) + log_jacobian
        except FloatingPointError as error:
            raise ValueError("Unrepresentable independent shape density") from error
    transformed = _from_normal(v)
    if not np.isfinite(density).all():
        raise ValueError("Finite normalized shape log density required")
    return {"z": transformed, "normal": v, "log_marginal": density, "pit": ndtr(v)}


def independent_inverse_shape(normal, parameters):
    normal = _pairs(normal)
    location, scale, epsilon, delta = _parameters(parameters)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            base_normal = location + scale * np.sinh((np.arcsinh(normal) + epsilon) / delta)
        except FloatingPointError as error:
            raise ValueError("Unrepresentable inverse shape coordinates") from error
    return _from_normal(base_normal)


def _objective_certificate(x, theta):
    """Direct hyperbolic differentiation; convexity gives a global box bound."""
    epsilon, delta = theta
    r = np.arcsinh(x)
    u = delta * r - epsilon
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            sinh_u, cosh_u = np.sinh(u), np.cosh(u)
            log_cosh = np.logaddexp(u, -u) - math.log(2)
            value = float(np.mean(0.5 * sinh_u**2 - log_cosh) - math.log(delta))
            first = sinh_u * cosh_u - np.tanh(u)
            second = np.cosh(2 * u) - 1 / cosh_u**2
            design = np.column_stack([-np.ones(len(x)), r])
            gradient = np.mean(first[:, None] * design, axis=0)
            gradient[1] -= 1 / delta
            hessian = (design.T * second) @ design / len(x)
            hessian[1, 1] += 1 / delta**2
        except FloatingPointError as error:
            raise ValueError("Nonfinite independent convex certificate") from error
    if not all(np.isfinite(a).all() for a in (value, gradient, hessian)):
        raise ValueError("Finite independent objective geometry required")
    lower, upper = np.array([-0.75, 0.75]), np.array([0.75, 2.0])
    corner = np.where(gradient >= 0, lower, upper)
    gap = float(gradient @ (theta - corner))
    projected = gradient.copy()
    projected[(theta <= lower + 1e-12) & (gradient > 0)] = 0
    projected[(theta >= upper - 1e-12) & (gradient < 0)] = 0
    residual = float(np.max(abs(projected)))
    return (
        {
            "theta": theta.tolist(),
            "objective": value,
            "gradient": gradient.tolist(),
            "hessian": hessian.tolist(),
            "box_gap": max(gap, 0.0),
            "lower_bound": value - max(gap, 0.0),
            "bounds": [[-0.75, 0.75], [0.75, 2.0]],
        },
        gap,
        residual,
    )


def verify_parameters(z, parameters):
    if type(parameters) is not dict or set(parameters) != PARAMETER_KEYS:
        raise ValueError("Exact fitted shape schema required")
    z = _pairs(z)
    location, scale, epsilon, delta = _parameters(parameters)
    independent_affine = _calibration(z)
    _compare(
        {key: parameters[key] for key in independent_affine},
        independent_affine,
        "affine_archive",
    )
    audits = parameters["optimizer_audits"]
    if type(audits) is not list or len(audits) != 2:
        raise ValueError("Both complete convex shape certificates required")
    w = _normal_coordinates(z)
    maximum_gap = maximum_residual = 0.0
    for j, audit in enumerate(audits):
        x = (w[:, j] - location[j]) / scale[j]
        theta = np.array([epsilon[j], delta[j]])
        expected, gap, residual = _objective_certificate(x, theta)
        if gap < -1e-12 or gap > CERTIFICATE_TOLERANCE or residual > CERTIFICATE_TOLERANCE:
            raise ValueError("Shape estimate lacks the fixed convex/KKT certificate")
        _compare(audit, expected, f"shape_audit:{j}")
        maximum_gap = max(maximum_gap, gap)
        maximum_residual = max(maximum_residual, residual)
    return {
        "status": "VERIFIED",
        "train_n": len(z),
        "maximum_box_gap": maximum_gap,
        "maximum_projected_gradient": maximum_residual,
    }


def _frame(value, label):
    if not isinstance(value, pd.DataFrame) or not value.columns.is_unique or value.empty:
        raise ValueError("Complete nonempty dataframe required: " + label)
    if "origin" not in value:
        raise ValueError("Origin field required: " + label)
    origin = _index(value.origin)
    if not origin.is_unique or not origin.is_monotonic_increasing:
        raise ValueError("Unique ordered origins required: " + label)
    return value.reset_index(drop=True).copy(deep=True)


def _exact_columns(actual, expected, columns, label):
    if len(actual) != len(expected) or not set(columns) <= set(actual):
        raise ValueError("Same-date complete inherited rows required: " + label)
    for name in columns:
        a, b = actual[name].reset_index(drop=True), expected[name].reset_index(drop=True)
        if pd.api.types.is_datetime64_any_dtype(b.dtype):
            if not pd.api.types.is_datetime64_any_dtype(a.dtype) or not _index(
                a, missing=True
            ).equals(_index(b, missing=True)):
                raise ValueError("Inherited date changed: " + label + "." + name)
        else:
            try:
                pd.testing.assert_series_equal(
                    a, b, check_dtype=False, check_names=False, check_exact=True
                )
            except AssertionError as error:
                raise ValueError("Inherited column changed: " + label + "." + name) from error


def _numeric_columns(actual, expected, columns, label):
    for name in columns:
        values = _array(actual[name])
        wanted = _array(expected[name])
        if values.shape != wanted.shape or not np.allclose(
            values, wanted, atol=ATOL, rtol=RTOL
        ):
            raise ValueError("Independent saved output differs: " + label + "." + name)


def _score_rows(frame, parameters, rhos):
    z, h = _standardized(frame)
    transformed = independent_shape_coordinates(z, h, parameters)
    expected = {}
    for j, asset in enumerate(ASSETS):
        for field, key in (("marginal", "log_marginal"), ("pit", "pit"), ("normal", "normal")):
            expected[f"{field}_shape_{asset}"] = transformed[key][:, j]
        expected[f"d_{asset}_shape"] = (
            frame[f"marginal_calibrated_{asset}"].to_numpy(float)
            - transformed["log_marginal"][:, j]
        )
    for family in FAMILIES:
        logcopula = _independent_logcopula(transformed["z"], rhos[family], family)
        loss = -transformed["log_marginal"].sum(axis=1) - logcopula
        expected[f"loss_shape_{family}"] = loss
        expected[f"d_{family}_shape"] = loss - frame[f"loss_cal_{family}"].to_numpy(float)
    expected["d_shape_gap"] = expected["loss_shape_t8"] - expected["loss_shape_gaussian"]
    expected["d_shape_interaction"] = expected["d_shape_gap"] - (
        frame.loss_cal_t8.to_numpy(float) - frame.loss_cal_gaussian.to_numpy(float)
    )
    return expected


def verify(
    archive, baseline_applications, baseline_panel, produced, calendar=None, minimum_train=252
):
    """Replay all supported months, including issued forecasts without outcomes."""
    if type(minimum_train) is not int or minimum_train < 2:
        raise ValueError("Exact training floor of at least two required")
    a = _archive(archive, calendar)
    base_app = _frame(baseline_applications, "baseline_applications")
    base_panel = _frame(baseline_panel, "baseline_panel")
    if (
        type(produced) is not dict
        or set(produced) != {"applications", "panel", "fits"}
        or type(produced["fits"]) is not list
    ):
        raise ValueError("Exact three-output shape result required")
    actual_app = _frame(produced["applications"], "applications")
    actual_panel = _frame(produced["panel"], "panel")
    if set(actual_app) != set(base_app) | set(APP_NEW) or set(base_app) & set(APP_NEW):
        raise ValueError("Exact shape application additions required")
    if set(actual_panel) != set(base_panel) | set(PANEL_NEW) or set(base_panel) & set(
        PANEL_NEW
    ):
        raise ValueError("Exact shape scored additions required")
    _exact_columns(actual_app, base_app, base_app.columns, "applications")
    _exact_columns(actual_panel, base_panel, base_panel.columns, "panel")
    wanted_apps, wanted_panels = [], []
    fit_number = 0
    max_gap = max_projected = max_dependence_gap = 0.0
    eligible_seen = False
    for _, query in a.groupby(a.origin.dt.to_period("M"), sort=True):
        fit_origin, cutoff = query.fit_origin.iloc[0], query.training_cutoff.iloc[0]
        history = a.loc[
            (a.origin < fit_origin)
            & (a.available_date <= cutoff)
            & (a.target_end <= cutoff)
            & a.issued
            & a[["y_qqq", "y_spx"]].notna().all(axis=1)
        ]
        if len(history) < minimum_train:
            if eligible_seen:
                raise ValueError("INSUFFICIENT_DATA after the first supported month")
            continue
        eligible_seen = True
        if fit_number >= len(produced["fits"]):
            raise ValueError("Missing supported monthly shape fit")
        fit = produced["fits"][fit_number]
        if type(fit) is not dict or set(fit) != {
            "fit_origin",
            "training_cutoff",
            "train_origins",
            "parameters",
            "dependence",
        }:
            raise ValueError("Exact monthly shape fit schema required")
        metadata = {
            "fit_origin": str(fit_origin.date()),
            "training_cutoff": str(cutoff.date()),
            "train_origins": history.origin.dt.strftime("%Y-%m-%d").tolist(),
        }
        _compare({k: fit[k] for k in metadata}, metadata, "shape_history")
        base_query = base_app.loc[base_app.fit_origin == fit_origin].reset_index(drop=True)
        _exact_columns(
            base_query,
            query.reset_index(drop=True),
            APP_SOURCE,
            "baseline_archive_application",
        )
        if not np.array_equal(
            _array(base_query.train_n), np.full(len(query), len(history), dtype=float)
        ):
            raise ValueError("Frozen baseline has a different mature archive count")
        z, h = _standardized(history)
        parameters = fit["parameters"]
        proof = verify_parameters(z, parameters)
        max_gap = max(max_gap, proof["maximum_box_gap"])
        max_projected = max(max_projected, proof["maximum_projected_gradient"])
        transformed = independent_shape_coordinates(z, h, parameters)
        rhos = {}
        if type(fit["dependence"]) is not dict or set(fit["dependence"]) != set(FAMILIES):
            raise ValueError("Both shape dependence fits on the same archive required")
        for family in FAMILIES:
            item = fit["dependence"][family]
            if type(item) is not dict or set(item) != {"rho", "audit"}:
                raise ValueError(
                    "Explicit dependence parameter and numerical certificate required"
                )
            rho = _array(item["rho"], ()).item()
            try:
                dependence = verify_dependence(transformed["z"], family, rho, item["audit"])
            except AssertionError as error:
                raise ValueError("Independent dependence certificate rejected") from error
            max_dependence_gap = max(max_dependence_gap, dependence["global_value_gap"])
            rhos[family] = rho
        app = base_query.copy(deep=True)
        for j, asset in enumerate(ASSETS):
            for name, parameter in (("a", "location"), ("b", "scale")):
                if not np.allclose(
                    _array(app[f"{name}_{asset}"]),
                    parameters[parameter][j],
                    atol=ATOL,
                    rtol=RTOL,
                ):
                    raise ValueError(
                        "Frozen affine control differs from the shape fitting archive"
                    )
            app[f"epsilon_{asset}"] = parameters["epsilon"][j]
            app[f"delta_{asset}"] = parameters["delta"][j]
        for family, rho in rhos.items():
            app[f"rho_shape_{family}"] = rho
        wanted_apps.append(app)
        scored_query = query.loc[query.eligible_scored].reset_index(drop=True)
        base_scored = base_panel.loc[base_panel.fit_origin == fit_origin].reset_index(
            drop=True
        )
        _exact_columns(
            base_scored,
            scored_query,
            (*APP_SOURCE, "target_end", "available_date", "y_qqq", "y_spx"),
            "baseline_archive_panel",
        )
        if len(base_scored):
            scored = base_scored.copy(deep=True)
            selected_app = app.set_index("origin").loc[scored.origin]
            for name in APP_NEW:
                scored[name] = selected_app[name].to_numpy()
            for name, values in _score_rows(scored, parameters, rhos).items():
                scored[name] = values
            wanted_panels.append(scored)
        fit_number += 1
    if fit_number != len(produced["fits"]) or not wanted_apps or not wanted_panels:
        raise ValueError("Exact nonempty supported fitting and scoring universe required")
    expected_app = pd.concat(wanted_apps, ignore_index=True)
    expected_panel = pd.concat(wanted_panels, ignore_index=True)
    _exact_columns(base_app, expected_app, base_app.columns, "complete_baseline_applications")
    _exact_columns(base_panel, expected_panel, base_panel.columns, "complete_baseline_panel")
    _numeric_columns(actual_app, expected_app, APP_NEW, "applications")
    _numeric_columns(actual_panel, expected_panel, PANEL_NEW, "panel")
    return {
        "status": "VERIFIED",
        "monthly_fits_verified": fit_number,
        "shape_fits_verified": fit_number * 2,
        "dependence_fits_verified": fit_number * 2,
        "applications_verified": len(expected_app),
        "scored_origins": len(expected_panel),
        "contrasts_verified": 6,
        "maximum_box_gap": max_gap,
        "maximum_projected_gradient": max_projected,
        "largest_independent_dependence_gap": max_dependence_gap,
        "clock_verification": "full_reference_calendar"
        if calendar is not None
        else "inherited_original_calendar_validation",
        "scope": "Saved shape density and forecasts; source authentication and prior baseline verification remain external; no economic payoff or live execution validation",
    }
