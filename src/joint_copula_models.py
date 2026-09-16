"""Shared frozen marginal estimators with two target-aligned copula fits."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.joint_risk_features import ALL_FEATURES
from src.joint_risk_models import MARGINAL_COLUMNS, _mean_fit, _serializable, transform
from src.macro_second_moment import fit_second_moment

MODELS = ("t8_copula", "gaussian_copula", "independence")
TARGETS = ("y_qqq", "y_spx")


def fit_dependence(z, family):
    from src.joint_copula_density import fit_dependence as fit

    return fit(z, family)


def _validate(frame, columns):
    if (
        not isinstance(frame, pd.DataFrame)
        or not frame.index.is_unique
        or not frame.columns.is_unique
        or not set(columns) <= set(frame)
    ):
        raise ValueError("Unique complete typed model frame required")
    for name in columns:
        if (
            frame[name].dtype.kind not in "iuf"
            or not np.isfinite(frame[name].to_numpy(float)).all()
        ):
            raise ValueError("Finite nonboolean real model values required")


def fit_predict(train, targets, apply):
    _validate(train, ALL_FEATURES)
    _validate(apply, ALL_FEATURES)
    _validate(targets, TARGETS)
    if not train.index.equals(targets.index):
        raise ValueError("Identical common training and target ordering required")
    y = targets.loc[:, list(TARGETS)].to_numpy(float)
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
            tr, ap, transformation = transform(train, apply)
            x, query = tr.iloc[:, 1:].to_numpy(float), ap.iloc[:, 1:].to_numpy(float)
            moments, coordinates, means, variances = {}, [], [], []
            for j, asset in enumerate(("qqq", "spx")):
                fitted, prediction, mean_audit = _mean_fit(x, y[:, j], query)
                residual = y[:, j] - fitted
                square = residual**2
                if np.any((residual != 0) & (square == 0)):
                    raise ValueError("Nonzero residual-square underflow; no fallback")
                variance_audit = fit_second_moment(x, square, np.concatenate([x, query]))
                all_h = variance_audit.pop("prediction")
                scale = np.sqrt(0.75 * all_h[: len(x)])
                if not np.isfinite(scale).all() or np.any(scale <= 0):
                    raise ValueError("Invalid t8 marginal scale")
                z = residual / scale
                if not np.isfinite(z).all() or np.any((residual != 0) & (z == 0)):
                    raise ValueError("Invalid t8 dependence coordinates")
                coordinates.append(z)
                means.append(prediction)
                variances.append(all_h[len(x) :])
                moments[asset] = {
                    "mean": mean_audit,
                    "variance": {
                        "columns": list(MARGINAL_COLUMNS),
                        **_serializable(variance_audit),
                    },
                }
            z, mu, h = (
                np.column_stack(coordinates),
                np.column_stack(means),
                np.column_stack(variances),
            )
            if not np.isfinite(mu).all() or not np.isfinite(h).all() or np.any(h <= 0):
                raise ValueError("Finite shared locations and positive variances required")
            predictions, dependence = {}, {}
            for name, family in (("t8_copula", "t8"), ("gaussian_copula", "gaussian")):
                rho, audit = fit_dependence(z, family)
                if type(rho) is bool or not np.isfinite(rho) or abs(rho) > 0.995:
                    raise ValueError("Dependence parameter outside fixed correlation domain")
                dependence[name] = {"rho": float(rho), "audit": audit}
                predictions[name] = {
                    "mu": mu.copy(),
                    "h": h.copy(),
                    "rho": np.full(len(apply), rho),
                }
            predictions["independence"] = {
                "mu": mu.copy(),
                "h": h.copy(),
                "rho": np.zeros(len(apply)),
            }
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise ValueError(
            "Invalid fixed marginal/dependence arithmetic; no fallback"
        ) from error
    return predictions, {
        "moments": moments,
        "transform": transformation,
        "residual_staging": "current_fit_training_residuals",
        "dependence": dependence,
    }
