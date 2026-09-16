"""Train-only sinh-arcsinh correction after the frozen affine PIT calibration.

The two shape parameters have a convex objective conditional on affine estimates.
Normalized log-return densities do not imply finite exponential moments.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, hyp2f1, log_ndtr, ndtr
from scipy.stats import t

from src.copula_calibration import ASSETS, _coordinates, fit_calibration, select_history
from src.joint_copula_density import fit_dependence, log_copula, marginal_logpdf, normal_scores

BOUNDS = ((-0.75, 0.75), (0.75, 2.0))
FIT_GAP = 1e-7
SHAPE_CELLS = ("shape_gaussian", "shape_t8")
CONTRASTS = (
    "qqq_shape",
    "spx_shape",
    "gaussian_shape",
    "t8_shape",
    "shape_gap",
    "shape_interaction",
)


def from_normal(values):
    """Vector inverse t8 probability coordinates, including very small log tails."""
    v = np.asarray(values, dtype=float)
    if not np.isfinite(v).all():
        raise ValueError("Finite normal coordinates required")
    a = np.abs(v)
    result = np.empty_like(v)
    central = a <= 8
    result[central] = t.isf(ndtr(-a[central]), 8)
    if np.any(~central):
        target = log_ndtr(-a[~central])
        constant = -np.log(2.0) - np.log(4.0) - betaln(4.0, 0.5)
        logx = (target - constant) / 4.0
        for _ in range(12):
            x = np.exp(logx)
            logx = (target - constant - np.log(hyp2f1(4.0, 0.5, 5.0, x))) / 4.0
        logz = 0.5 * (np.log(8.0) + np.log1p(-np.exp(logx)) - logx)
        if np.any(logz >= np.log(np.finfo(float).max)):
            raise ValueError("Inverse coordinate exceeds numerical domain")
        result[~central] = np.exp(logz)
    result *= np.sign(v)
    if not np.isfinite(result).all() or not np.allclose(
        normal_scores(result), v, atol=2e-9, rtol=2e-10
    ):
        raise ValueError("Inverse probability coordinate failed roundtrip")
    return result


def objective(theta, x):
    """Parameter-dependent mean NLL, gradient, and Hessian; constants omitted."""
    epsilon, delta = np.asarray(theta, dtype=float)
    x = np.asarray(x, dtype=float)
    if delta <= 0 or not np.isfinite(x).all():
        raise ValueError("Positive tail parameter and finite training values required")
    r = np.arcsinh(x)
    u = delta * r - epsilon
    v = np.sinh(u)
    value = 0.5 * np.mean(v * v - np.log1p(v * v)) - np.log(delta)
    q = v * v * np.tanh(u)
    k = v * v * (2.0 + 1.0 / (1.0 + v * v))
    gradient = np.array([-q.mean(), (q * r).mean() - 1.0 / delta])
    hessian = np.array(
        [[k.mean(), -(k * r).mean()], [-(k * r).mean(), (k * r * r).mean() + 1.0 / delta**2]]
    )
    if not all(np.isfinite(a).all() for a in (value, gradient, hessian)):
        raise ValueError("Unrepresentable shape objective")
    return float(value), gradient, hessian


def fit_shape(z):
    base = fit_calibration(z)
    w = normal_scores(z)
    a, b = np.asarray(base["location"]), np.asarray(base["scale"])
    epsilons, deltas, audits = [], [], []
    for j in range(2):
        x = (w[:, j] - a[j]) / b[j]
        fun = lambda theta, x=x: objective(theta, x)[:2]
        result = minimize(
            fun,
            [0.0, 1.0],
            method="L-BFGS-B",
            jac=True,
            bounds=BOUNDS,
            options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 1000, "maxls": 50},
        )
        theta = result.x
        # Fixed Newton polishing, with projected steps and Armijo backtracking.
        for _ in range(30):
            value, gradient, hessian = objective(theta, x)
            lower, upper = np.array(BOUNDS).T
            free = ~(
                ((theta <= lower + 1e-12) & (gradient > 0))
                | ((theta >= upper - 1e-12) & (gradient < 0))
            )
            if not free.any():
                break
            direction = np.zeros(2)
            direction[free] = np.linalg.solve(hessian[np.ix_(free, free)], gradient[free])
            updated = False
            for k in range(30):
                candidate = np.clip(theta - 2.0 ** (-k) * direction, lower, upper)
                difference = candidate - theta
                if (
                    objective(candidate, x)[0]
                    <= value + 1e-4 * gradient.dot(difference) + 1e-15
                ):
                    theta = candidate
                    updated = True
                    break
            if not updated or np.max(np.abs(difference)) < 1e-13:
                break
        value, gradient, hessian = objective(theta, x)
        corner = np.where(gradient >= 0, lower, upper)
        gap = float(gradient.dot(theta - corner))
        if not np.isfinite(gap) or gap < -1e-12 or gap > FIT_GAP:
            raise ValueError("Shape fit failed convex box certificate")
        audits.append(
            {
                "theta": theta.tolist(),
                "objective": value,
                "gradient": gradient.tolist(),
                "hessian": hessian.tolist(),
                "box_gap": max(0.0, gap),
                "lower_bound": value - max(0.0, gap),
                "bounds": [list(b) for b in BOUNDS],
            }
        )
        epsilons.append(float(theta[0]))
        deltas.append(float(theta[1]))
    return {**base, "epsilon": epsilons, "delta": deltas, "optimizer_audits": audits}


def _parameters(parameters):
    a, b, epsilon, delta = (
        np.asarray(parameters[k], float) for k in ("location", "scale", "epsilon", "delta")
    )
    if any(v.shape != (2,) or not np.isfinite(v).all() for v in (a, b, epsilon, delta)):
        raise ValueError("Finite two-asset shape parameters required")
    if np.any(b <= 1e-12) or np.any(delta <= 0):
        raise ValueError("Positive affine and shape scales required")
    if np.any(np.abs(epsilon) > 0.75) or np.any(delta < 0.75) or np.any(delta > 2.0):
        raise ValueError("Shape parameters outside fixed bounds")
    return a, b, epsilon, delta


def shape_coordinates(z, h, parameters):
    z, h = np.asarray(z, float), np.asarray(h, float)
    if z.ndim != 2 or z.shape[1] != 2 or h.shape != z.shape:
        raise ValueError("Aligned paired coordinates and variances required")
    a, b, epsilon, delta = _parameters(parameters)
    w = normal_scores(z)
    if (
        np.array_equal(a, np.zeros(2))
        and np.array_equal(b, np.ones(2))
        and np.array_equal(epsilon, np.zeros(2))
        and np.array_equal(delta, np.ones(2))
    ):
        return {
            "z": z.copy(),
            "normal": w,
            "log_marginal": marginal_logpdf(z, h),
            "pit": ndtr(w),
        }
    x = (w - a) / b
    u = delta * np.arcsinh(x) - epsilon
    v = np.sinh(u)
    logjac = (
        np.log(delta) - np.log(b) + np.logaddexp(u, -u) - np.log(2.0) - 0.5 * np.log1p(x * x)
    )
    density = marginal_logpdf(z, h) - 0.5 * (v * v - w * w) + logjac
    if not np.isfinite(density).all():
        raise ValueError("Nonfinite normalized shape density")
    return {"z": from_normal(v), "normal": v, "log_marginal": density, "pit": ndtr(v)}


def inverse_shape(normal, parameters):
    a, b, epsilon, delta = _parameters(parameters)
    v = np.asarray(normal, float)
    if v.ndim != 2 or v.shape[1] != 2:
        raise ValueError("Paired normal scores required")
    return from_normal(a + b * np.sinh((np.arcsinh(v) + epsilon) / delta))


def run_shape(archive, baseline_applications, baseline_panel, minimum_train=252):
    """Reuse authenticated identity/affine forecasts, fit only the new shape cells."""
    applications, fits = [], []
    for fit_origin, query in baseline_applications.groupby("fit_origin", sort=True):
        cutoff = pd.Timestamp(query.training_cutoff.iloc[0])
        history = select_history(archive, fit_origin, cutoff, minimum_train)
        if (history.target_end > cutoff).any():
            raise ValueError("Immature target in calibration history")
        z, h = _coordinates(history)
        parameters = fit_shape(z)
        changed = shape_coordinates(z, h, parameters)
        dependence = {}
        app = query.copy()
        for j, asset in enumerate(ASSETS):
            if not np.allclose(
                app[f"a_{asset}"], parameters["location"][j], atol=1e-13, rtol=0
            ) or not np.allclose(
                app[f"b_{asset}"], parameters["scale"][j], atol=1e-13, rtol=0
            ):
                raise ValueError("Affine control disagrees with same issued-error archive")
            app[f"epsilon_{asset}"] = parameters["epsilon"][j]
            app[f"delta_{asset}"] = parameters["delta"][j]
        for family in ("gaussian", "t8"):
            rho, audit = fit_dependence(changed["z"], family)
            dependence[family] = {"rho": float(rho), "audit": audit}
            app[f"rho_shape_{family}"] = rho
        fits.append(
            {
                "fit_origin": str(pd.Timestamp(fit_origin).date()),
                "training_cutoff": str(cutoff.date()),
                "train_origins": [str(pd.Timestamp(d).date()) for d in history.origin],
                "parameters": parameters,
                "dependence": dependence,
            }
        )
        applications.append(app)
    app = pd.concat(applications, ignore_index=True)
    new_columns = [
        "origin",
        "epsilon_qqq",
        "epsilon_spx",
        "delta_qqq",
        "delta_spx",
        "rho_shape_gaussian",
        "rho_shape_t8",
    ]
    panel = baseline_panel.merge(app[new_columns], on="origin", validate="one_to_one")
    parts = []
    by_fit = {pd.Timestamp(f["fit_origin"]): f for f in fits}
    for fit_origin, part in panel.groupby("fit_origin", sort=True):
        fit = by_fit[pd.Timestamp(fit_origin)]
        part = part.copy()
        z, h = _coordinates(part)
        transformed = shape_coordinates(z, h, fit["parameters"])
        for j, asset in enumerate(ASSETS):
            for label, key in (
                ("marginal", "log_marginal"),
                ("pit", "pit"),
                ("normal", "normal"),
            ):
                part[f"{label}_shape_{asset}"] = transformed[key][:, j]
            part[f"d_{asset}_shape"] = (
                part[f"marginal_calibrated_{asset}"] - transformed["log_marginal"][:, j]
            )
        for family in ("gaussian", "t8"):
            loss = -transformed["log_marginal"].sum(axis=1) - log_copula(
                transformed["z"], part[f"rho_shape_{family}"].to_numpy(), family
            )
            part[f"loss_shape_{family}"] = loss
            part[f"d_{family}_shape"] = loss - part[f"loss_cal_{family}"]
        part["d_shape_gap"] = part.loss_shape_t8 - part.loss_shape_gaussian
        part["d_shape_interaction"] = part.d_shape_gap - (
            part.loss_cal_t8 - part.loss_cal_gaussian
        )
        parts.append(part)
    return {"applications": app, "panel": pd.concat(parts, ignore_index=True), "fits": fits}
