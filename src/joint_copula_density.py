"""Fixed t8 marginals and two globally certified scalar dependence fits.

The certificate is a floating-point numerical bound, not an exact-arithmetic
proof. Every terminal interval is retained for independent reconstruction.
"""

import heapq

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import betaln, gammaln, hyp2f1, ndtri_exp, stdtr

NU = 8.0
RHO_MAX = 0.995
GAP = 1e-8
MAX_SPLITS = 32768
ROUNDING = 256.0
TIE = 1e-12
ROOT_TOL = 1e-10
KKT = 1e-7
T_LOG_CONSTANT = gammaln(4.5) - gammaln(4.0) - 0.5 * np.log(8.0 * np.pi)


def _pairs(values):
    z = np.asarray(values, dtype=float)
    if z.ndim != 2 or z.shape[1] != 2 or not len(z) or not np.isfinite(z).all():
        raise ValueError("Finite nonempty paired t8 standardized returns required")
    return z


def _rho(values, n):
    rho = np.asarray(values, dtype=float)
    if rho.ndim == 0:
        rho = np.full(n, float(rho))
    if rho.shape != (n,) or not np.isfinite(rho).all() or (np.abs(rho) > RHO_MAX).any():
        raise ValueError("Correlation outside fixed closed domain")
    return rho


def normal_scores(values):
    """Phi inverse of F_t8(z), from the smaller log tail without clipping."""
    z = np.asarray(values, dtype=float)
    if not np.isfinite(z).all():
        raise ValueError("Finite standardized values required")
    absolute = np.abs(z)
    central = absolute <= 1.0
    tail = np.empty_like(z)
    tail[central] = np.log(stdtr(NU, -absolute[central]))
    with np.errstate(divide="ignore", under="ignore"):
        logx = np.log(NU) - np.logaddexp(np.log(NU), 2 * np.log(absolute[~central]))
        x = np.exp(logx)
    # I_x(4,1/2)=x^4/(4 B(4,1/2))*2F1(4,1/2;5;x).
    tail[~central] = (
        -np.log(2.0)
        + 4 * logx
        - np.log(4.0)
        - betaln(4.0, 0.5)
        + np.log(hyp2f1(4.0, 0.5, 5.0, x))
    )
    value = -np.sign(z) * ndtri_exp(tail)
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite log-tail normal transform")
    return value


def _t_logpdf(z):
    with np.errstate(divide="ignore"):
        logterm = np.logaddexp(np.log(NU), 2 * np.log(np.abs(z))) - np.log(NU)
    return T_LOG_CONSTANT - 4.5 * logterm


def marginal_logpdf(z, h):
    z, h = np.asarray(z, dtype=float), np.asarray(h, dtype=float)
    if (
        z.shape != h.shape
        or not np.isfinite(z).all()
        or not np.isfinite(h).all()
        or (h <= 0).any()
    ):
        raise ValueError("Aligned finite t8 values and positive variances required")
    return _t_logpdf(z) - 0.5 * (np.log(h) + np.log(0.75))


def _scaled(z):
    scale = np.maximum(1.0, np.abs(z).max(axis=1))
    q = z / scale[:, None]
    with np.errstate(under="ignore"):
        c = NU * np.exp(-2 * np.log(scale))
    return (q * q).sum(axis=1), q[:, 0] * q[:, 1], c, np.log(scale)


def log_copula(z, rho, family):
    z = _pairs(z)
    rho = _rho(rho, len(z))
    d = 1.0 - rho * rho
    if family == "independence":
        if np.any(rho != 0.0):
            raise ValueError("Independence requires a zero placeholder parameter")
        return np.zeros(len(z))
    if family == "gaussian":
        w = normal_scores(z)
        result = (
            -0.5 * np.log(d)
            - 0.5 * (rho * rho * (w * w).sum(axis=1) - 2 * rho * w[:, 0] * w[:, 1]) / d
        )
    elif family == "t8":
        a, b, c, logs = _scaled(z)
        W = c * d + a - 2 * rho * b
        if not np.isfinite(W).all() or (W <= 0).any():
            raise ValueError("Invalid positive t8 quadratic")
        logterm = 2 * logs + np.log(W) - np.log(NU) - np.log(d)
        joint = -np.log(2 * np.pi) - 0.5 * np.log(d) - 5 * logterm
        result = joint - _t_logpdf(z).sum(axis=1)
    else:
        raise ValueError("Unknown dependence family")
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite copula log density")
    return result


def _prepared(z, family):
    if family == "gaussian":
        w = normal_scores(z)
        return {
            "A": float(np.mean((w * w).sum(axis=1))),
            "B": float(np.mean(w[:, 0] * w[:, 1])),
        }
    if family != "t8":
        raise ValueError("Only the two dependence families have fitted parameters")
    a, b, c, _ = _scaled(z)
    return {
        "a": a,
        "b": b,
        "c": c,
        "logW0": np.log(c + a),
        "base": -float(np.mean(log_copula(z, 0.0, "t8"))),
    }


def _value_gradient(rho, p, family):
    d = 1.0 - rho * rho
    if family == "gaussian":
        A, B = p["A"], p["B"]
        value = 0.5 * np.log(d) + 0.5 * (rho * rho * A - 2 * rho * B) / d
        gradient = (rho**3 - B * rho * rho + (A - 1) * rho - B) / (d * d)
    else:
        W = p["c"] * d + p["a"] - 2 * rho * p["b"]
        value = p["base"] + 5 * float(np.mean(np.log(W) - p["logW0"])) - 4.5 * np.log(d)
        gradient = 5 * float(np.mean((-2 * p["c"] * rho - 2 * p["b"]) / W)) + 9 * rho / d
    if not np.isfinite([value, gradient]).all():
        raise ValueError("Nonfinite dependence objective or gradient")
    return float(value), float(gradient)


def objective_gradient(z, rho, family):
    z = _pairs(z)
    if np.asarray(rho).ndim != 0:
        raise ValueError("Scalar objective parameter required")
    r = float(_rho(rho, len(z))[0])
    return _value_gradient(r, _prepared(z, family), family)


def _projected(rho, gradient):
    if rho == -RHO_MAX:
        return max(0.0, -gradient)
    if rho == RHO_MAX:
        return max(0.0, gradient)
    return abs(gradient)


def _leaf(left, right, p):
    mid, half = 0.5 * (left + right), 0.5 * (right - left)
    value, gradient = _value_gradient(mid, p, "t8")
    c, a, b = p["c"], p["a"], p["b"]
    Wmin = np.minimum(
        c * (1 - left * left) + a - 2 * left * b, c * (1 - right * right) + a - 2 * right * b
    )
    derivative = np.maximum(np.abs(-2 * c * left - 2 * b), np.abs(-2 * c * right - 2 * b))
    nearest = 0.0 if left <= 0 <= right else min(abs(left), abs(right))
    curvature = (
        -5 * float(np.mean(2 * c / Wmin + (derivative / Wmin) ** 2))
        + 9 * (1 + nearest * nearest) / (1 - nearest * nearest) ** 2
    )
    steps = [-half, half]
    if curvature > 0:
        steps.append(float(np.clip(-gradient / curvature, -half, half)))
    correction = min(gradient * s + 0.5 * curvature * s * s for s in steps)
    inflation = (
        ROUNDING
        * np.finfo(float).eps
        * (1 + abs(value) + abs(gradient) * half + 0.5 * abs(curvature) * half * half)
    )
    lower = value + correction - inflation
    if not np.isfinite([curvature, lower]).all():
        raise ValueError("Nonfinite interval certification bound")
    return {
        "left": left,
        "right": right,
        "midpoint": mid,
        "value": value,
        "gradient": gradient,
        "curvature_lower": curvature,
        "lower_bound": float(lower),
    }


def fit_dependence(z, family, *, max_splits=MAX_SPLITS):
    """Fit the single prescribed parameter; budget exhaustion is a failure."""
    z = _pairs(z)
    p = _prepared(z, family)
    if type(max_splits) is not int or not 0 <= max_splits <= MAX_SPLITS:
        raise ValueError("Invalid fixed certification budget")
    candidates = []

    def consider(rho):
        value, gradient = _value_gradient(float(rho), p, family)
        candidates.append((value, float(rho), gradient))

    for rho in (-RHO_MAX, 0.0, RHO_MAX):
        consider(rho)

    def best():
        value = min(row[0] for row in candidates)
        return min(
            (row for row in candidates if row[0] <= value + TIE), key=lambda row: row[1]
        )

    if family == "gaussian":
        polynomial = [1.0, -p["B"], p["A"] - 1.0, -p["B"]]
        roots = np.roots(polynomial)
        for root in roots:
            if abs(root.imag) <= ROOT_TOL and -RHO_MAX <= root.real <= RHO_MAX:
                if abs(np.polyval(polynomial, root.real)) > ROOT_TOL:
                    raise ValueError("Gaussian stationary root failed residual check")
                consider(float(root.real))
        value, rho, gradient = best()
        lower = min(row[0] for row in candidates) - ROUNDING * np.finfo(float).eps * (
            1 + abs(value)
        )
        certificate = {
            "method": "all_real_stationary_roots_and_endpoints",
            "polynomial_roots": [[float(r.real), float(r.imag)] for r in roots],
            "polynomial": polynomial,
            "candidates": [
                {"rho": r, "objective": v}
                for v, r, _ in sorted(candidates, key=lambda row: row[1])
            ],
            "upper_bound": value,
            "lower_bound": lower,
            "gap": max(0.0, value - lower),
        }
    else:
        optimum = minimize(
            lambda x: _value_gradient(float(x[0]), p, family),
            np.array([0.0]),
            method="L-BFGS-B",
            jac=True,
            bounds=[(-RHO_MAX, RHO_MAX)],
            options={"maxiter": 1000, "maxls": 50, "ftol": 1e-14, "gtol": 1e-9},
        )
        if not optimum.success or not np.isfinite(optimum.x).all():
            raise ValueError("Fixed dependence optimizer failed")
        consider(float(optimum.x[0]))
        initial = _leaf(-RHO_MAX, RHO_MAX, p)
        heap, serial, splits = [(initial["lower_bound"], 0, initial)], 0, 0
        while True:
            value, rho, gradient = best()
            if value - heap[0][0] <= GAP:
                break
            if splits >= max_splits:
                raise ValueError("Global numerical certificate budget exhausted")
            _, _, leaf = heapq.heappop(heap)
            left_bound, m, r = leaf["left"], leaf["midpoint"], leaf["right"]
            if m in (left_bound, r):
                raise ValueError("Global certification interval cannot be split")
            gl = _value_gradient(left_bound, p, family)[1]
            gr = _value_gradient(r, p, family)[1]
            if gl * gr < 0:
                stationary = brentq(
                    lambda x: _value_gradient(x, p, family)[1],
                    left_bound,
                    r,
                    xtol=5e-15,
                    rtol=4 * np.finfo(float).eps,
                    maxiter=200,
                )
                consider(stationary)
            consider(m)
            for left, right in ((left_bound, m), (m, r)):
                child = _leaf(left, right, p)
                serial += 1
                heapq.heappush(heap, (child["lower_bound"], serial, child))
            splits += 1
        value, rho, gradient = best()
        # Refine a near-stationary interior winner in its containing leaf.
        if abs(rho) < RHO_MAX and _projected(rho, gradient) > KKT:
            containing = [
                node[2] for node in heap if node[2]["left"] <= rho <= node[2]["right"]
            ]
            for leaf in containing:
                left_bound, r = leaf["left"], leaf["right"]
                if (
                    _value_gradient(left_bound, p, family)[1]
                    * _value_gradient(r, p, family)[1]
                    < 0
                ):
                    candidate = brentq(
                        lambda x: _value_gradient(x, p, family)[1],
                        left_bound,
                        r,
                        xtol=5e-15,
                        rtol=4 * np.finfo(float).eps,
                        maxiter=200,
                    )
                    refined_value, refined_gradient = _value_gradient(candidate, p, family)
                    if refined_value <= value + TIE:
                        value, rho, gradient = refined_value, candidate, refined_gradient
        lower = min(node[0] for node in heap)
        certificate = {
            "method": "interval_taylor_lower_bound",
            "splits": splits,
            "max_splits": max_splits,
            "upper_bound": value,
            "lower_bound": lower,
            "gap": max(0.0, value - lower),
            "leaves": sorted((node[2] for node in heap), key=lambda row: row["left"]),
            "local_optimizer": {
                "success": bool(optimum.success),
                "message": str(optimum.message),
                "rho": float(optimum.x[0]),
            },
        }
    certificate.update({"gap_tolerance": GAP, "roundoff_multiplier": ROUNDING})
    if certificate["gap"] > GAP or _projected(rho, gradient) > KKT:
        raise ValueError("Global certificate or projected gradient failed")
    audit = {
        "status": "CERTIFIED_GLOBAL_NUMERICAL_OPTIMUM",
        "family": family,
        "rho": float(rho),
        "objective": float(value),
        "gradient": float(gradient),
        "projected_gradient": float(_projected(rho, gradient)),
        "train_n": len(z),
        "domain": [-RHO_MAX, RHO_MAX],
        "certificate": certificate,
    }
    return float(rho), audit
