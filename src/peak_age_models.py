"""Four common-cohort ridge/peak-age return arms, using supplied tables only."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .peak_age_features import BASE, COMMON, _dates

DEPTH = BASE + ("drawdown", "drawdown_sq", "window_return")
MODELS = ("mean", "baseline", "depth", "peak_age")
ALPHA = .01
MIN_TRAIN = 1000
GRADIENT_TOLERANCE = 1e-10


def _finite(value, label):
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite " + label)
    return array


def _scalar(value, label):
    array = _finite(value, label)
    if array.ndim != 0:
        raise ValueError("Scalar " + label + " required")
    return float(array)


def _product(left, right, label):
    a, b = _finite(left, label), _finite(right, label)
    with np.errstate(all="ignore"):
        value = _finite(a * b, label)
    if ((a != 0) & (b != 0) & (value == 0)).any():
        raise ValueError("Nonzero product underflow in " + label)
    return value


def _divide(numerator, denominator, label):
    a, b = _finite(numerator, label), _finite(denominator, label)
    with np.errstate(all="ignore"):
        result = _finite(a / b, label)
    if ((a != 0) & (result == 0)).any():
        raise ValueError("Nonzero division underflow in " + label)
    return result


def _sum(values, label):
    try:
        return _scalar(math.fsum(_finite(values, label).ravel()), label)
    except (OverflowError, ArithmeticError) as exc:
        raise ValueError("Invalid finite sum in " + label) from exc


def _mean_square(values, label):
    return _scalar(_divide(_sum(_product(values, values, label), label), len(values), label), label)


def _design(frame):
    if not isinstance(frame, pd.DataFrame) or not len(frame):
        raise ValueError("Nonempty feature DataFrame required")
    _dates(frame.index)
    selected = frame.loc[:, COMMON].astype(float).copy()
    _finite(selected.to_numpy(), "complete common features")
    if not selected.const.eq(1).all():
        raise ValueError("Unit intercept required")
    if ((selected.peak_age < 0) | (selected.peak_age > 1)).any():
        raise ValueError("Age must be within zero and one")
    if (selected[["drawdown", "drawdown_sq"]] < 0).any().any():
        raise ValueError("Nonnegative depth coordinates required")
    return selected


def transform(train, apply):
    """Original curvature centers, retaining exactly sixteen/nineteen slopes."""
    tr, ap = _design(train), _design(apply)
    if len(tr) < 2:
        raise ValueError("At least two training rows required")
    audit = {"train_n": len(tr)}
    for name in ["I", "R"]:
        with np.errstate(all="ignore"):
            center = _scalar(tr[name].mean(), "training curvature center")
        audit[name + "_mean"] = center
        for frame in [tr, ap]:
            delta = _finite(frame[name].to_numpy() - center, "curvature difference")
            frame[name + "_square"] = _product(delta, delta, "curvature square")
    return tr.loc[:, DEPTH], ap.loc[:, DEPTH], audit


def scalar_objective(coefficient, centered_age, y, offset):
    """Specified MSE+.01*b² and its independently checkable half derivative."""
    b = _scalar(coefficient, "scalar coefficient")
    a, target, f0 = (_finite(centered_age, "centered age"), _finite(y, "scalar target"),
                      _finite(offset, "frozen current depth predictions"))
    if a.ndim != 1 or len(a) < 2 or a.shape != target.shape or a.shape != f0.shape:
        raise ValueError("Aligned nonempty scalar stage arrays required")
    correction = _product(b, a, "scalar correction")
    residual = _finite(f0 + correction - target, "scalar residual")
    penalty = _scalar(_product(ALPHA, _product(b, b, "scalar coefficient square"), "scalar penalty"), "penalty")
    objective = _scalar(_mean_square(residual, "scalar squared errors") + penalty, "scalar objective")
    gradient = _scalar(_divide(_sum(_product(a, residual, "scalar gradient products"), "scalar gradient sum"),
                               len(a), "scalar gradient mean")
                       + _product(ALPHA, b, "scalar penalty gradient"), "scalar half gradient")
    return objective, gradient


def fit_scalar(age, y, offset):
    """Fit the sequential scalar stage against current-fit in-sample residuals."""
    age, target, f0 = _finite(age, "raw age"), _finite(y, "target"), _finite(offset, "depth offset")
    if (age.ndim != 1 or len(age) < 2 or age.shape != target.shape or age.shape != f0.shape
            or ((age < 0) | (age > 1)).any()):
        raise ValueError("Aligned bounded raw ages and target/offset arrays required")
    constant = bool(np.all(age == age[0]))
    center = float(age[0]) if constant else _scalar(_divide(_sum(age, "age sum"), len(age), "age center"), "age center")
    a = _finite(age - center, "centered age")
    residual = _finite(target - f0, "current-fit depth residual")
    numerator = _scalar(_divide(_sum(_product(a, residual, "scalar numerator products"), "scalar numerator sum"),
                               len(a), "scalar numerator mean"), "scalar numerator")
    second_moment = _mean_square(a, "centered age second moment")
    denominator = _scalar(second_moment + ALPHA, "scalar denominator")
    if constant:
        coefficient, status = 0.0, "EXACT_CONSTANT_INPUT"
    elif numerator == 0:
        coefficient, status = 0.0, "BALANCED_AT_ZERO"
    else:
        coefficient = _scalar(_divide(numerator, denominator, "scalar coefficient"), "scalar coefficient")
        status = "FITTED"
    objective, gradient = scalar_objective(coefficient, a, target, f0)
    if abs(gradient) > GRADIENT_TOLERANCE:
        raise ValueError("Scalar half-gradient gate failed")
    audit = {"coefficient": coefficient, "age_center": center, "age_scale": 1.0, "alpha": ALPHA,
             "status": status, "train_n": len(age), "numerator": numerator,
             "second_moment": second_moment, "denominator": denominator,
             "objective": objective, "gradient": gradient, "gradient_max_abs": abs(gradient)}
    return coefficient, audit


def _dot(design, coefficient, label):
    _product(design, coefficient, label + " products")
    with np.errstate(all="ignore"):
        return _finite(design @ coefficient, label)


def fit_predict(train_features, y, apply_features):
    """Fit all four arms on exactly the supplied complete, mature common rows.

    The caller owns monthly scheduling and cutoff maturity. All declared
    nuisance scales must exceed1e-12; age remains an unstandardized scalar.
    """
    if len(train_features) < MIN_TRAIN:
        raise ValueError("INSUFFICIENT_DATA: at least1000 common mature rows required")
    if isinstance(y, pd.Series) and not y.index.equals(train_features.index):
        raise ValueError("Training target order must exactly match common feature rows")
    target = _finite(y, "training target")
    if target.ndim != 1 or len(target) != len(train_features):
        raise ValueError("Aligned one-dimensional training targets required")
    tr, ap, transform_audit = transform(train_features, apply_features)
    all_raw = tr.loc[:, DEPTH[1:]].to_numpy(float)
    with np.errstate(all="ignore"):
        means = _finite(all_raw.mean(axis=0), "nuisance means")
        delta = _finite(all_raw - means, "nuisance centered values")
        _product(delta, delta, "nuisance variance products")
        scales = _finite(all_raw.std(axis=0, ddof=0), "nuisance scales")
    if (scales <= 1e-12).any():
        raise ValueError("INSUFFICIENT_DATA: every declared nuisance scale must exceed1e-12")
    query_raw = ap.loc[:, DEPTH[1:]].to_numpy(float)
    z_all = _divide(delta, scales, "standardized training design")
    query_all = _divide(_finite(query_raw - means, "centered application design"), scales, "standardized application design")
    with np.errstate(all="ignore"):
        target_mean = _scalar(target.mean(), "training target mean")
    centered = _finite(target - target_mean, "centered training target")
    mean_residual = _finite(target_mean - target, "mean residual")
    mean_gradient = _scalar(mean_residual.mean(), "mean intercept gradient")
    predictions = {"mean": np.full(len(ap), target_mean)}
    audits = {"mean": {"columns": ["const"], "means": [0.0], "scales": [1.0], "beta": [target_mean],
                        "alpha": 0.0, "train_n": len(tr), "objective": _mean_square(mean_residual, "mean loss"),
                        "gradient": [mean_gradient], "gradient_max_abs": abs(mean_gradient)}}
    fitted_depth = None
    for name, columns in [("baseline", BASE), ("depth", DEPTH)]:
        count = len(columns) - 1
        z, query = z_all[:, :count], query_all[:, :count]
        for column in range(count):
            _product(z[:, column, None], z, "ridge Gram products")
        _product(z, centered[:, None], "ridge RHS products")
        with np.errstate(all="ignore"):
            gram = _finite(z.T @ z / len(z) + ALPHA * np.eye(count), "ridge Gram system")
            rhs = _finite(z.T @ centered / len(z), "ridge RHS system")
        try:
            coefficient = _finite(np.linalg.solve(gram, rhs), "ridge coefficients")
        except np.linalg.LinAlgError as exc:
            raise ValueError("Fixed normal-equation ridge solve failed") from exc
        fitted = _finite(target_mean + _dot(z, coefficient, "ridge training prediction"), "ridge fitted values")
        prediction = _finite(target_mean + _dot(query, coefficient, "ridge application prediction"), "ridge predictions")
        residual = _finite(fitted - target, "ridge residuals")
        _product(z, residual[:, None], "ridge gradient products")
        gradient = _finite(np.r_[residual.mean(), z.T @ residual / len(z) + _product(ALPHA, coefficient, "ridge penalty gradient")],
                           "full ridge half gradient")
        maximum = float(np.max(np.abs(gradient)))
        if maximum > GRADIENT_TOLERANCE:
            raise ValueError("Ridge full half-gradient gate failed")
        penalty = _scalar(_product(ALPHA, _sum(_product(coefficient, coefficient, "ridge coefficient squares"), "ridge squared norm"),
                                   "ridge penalty"), "ridge penalty")
        objective = _scalar(_mean_square(residual, "ridge squared errors") + penalty, "ridge objective")
        predictions[name] = prediction
        audits[name] = {"columns": list(columns), "means": [0., *means[:count].tolist()],
                        "scales": [1., *scales[:count].tolist()], "beta": [target_mean, *coefficient.tolist()],
                        "alpha": ALPHA, "train_n": len(z), "objective": objective,
                        "gradient": gradient.tolist(), "gradient_max_abs": maximum}
        if name == "depth":
            fitted_depth = fitted
    coefficient, scalar_audit = fit_scalar(train_features.peak_age.to_numpy(float), target, fitted_depth)
    if coefficient == 0.0:
        predictions["peak_age"] = predictions["depth"].copy()
    else:
        query_age = _finite(apply_features.peak_age.to_numpy(float) - scalar_audit["age_center"], "centered query age")
        correction = _product(coefficient, query_age, "peak-age query correction")
        predictions["peak_age"] = _finite(predictions["depth"] + correction, "peak-age prediction")
    return {"predictions": predictions, "model_audit": audits,
            "transform_audit": transform_audit, "scalar_audit": scalar_audit}
