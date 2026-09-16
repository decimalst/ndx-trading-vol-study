"""Independent prospective event-adjacency construction and numerical checks.

Independent features, fit geometry, optimizers, scores and inference. Shared
source-admission metadata is explicitly rehashed; no producer mathematical
routine establishes reconstructed values. Empirical execution requires the
complete registered protocol, source closure, frozen inventory and output record.
"""

from __future__ import annotations

import io
import json
import math
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import bisect, root
from scipy.special import expit

from . import event_cluster_admission as shared_admission
from . import verify_issued_calibration as previous
from . import verify_range_alert as old

OLD_RAW = tuple(old.ALL_FEATURES)
OLD_BASE = tuple(old.BASE)
NUISANCE = (
    "event_fraction22",
    "event_fraction22_sq",
    "last_event",
    "expected_adjacency22",
    "linear_recency22",
)
MEMORY = "adjacency_fraction22"
FEATURES = (*NUISANCE, MEMORY)
AUDIT_FIELDS = (
    "event_count22",
    "adjacent_pairs22",
    "expected_adjacency_numerator",
    "linear_recency_numerator",
    "excess_adjacency_numerator",
)
SUMMARY_COLUMNS = (*FEATURES, "excess_adjacency22", *AUDIT_FIELDS)
DATE_COLUMNS = (
    "feature_cutoff_date",
    "window_first_available",
    "window_last_available",
    "window_first_origin",
    "window_last_origin",
)
MEMORY_COLUMNS = (*SUMMARY_COLUMNS, *DATE_COLUMNS)
ALPHA = 0.01
GRADIENT_TOLERANCE = 1e-8 + 1e-12
REPLAY_RTOL, REPLAY_ATOL = 1e-10, 1e-12
COEFFICIENT_RTOL, COEFFICIENT_ATOL = 1e-7, 1e-6


def real(value, label="values"):
    try:
        a = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Real numeric {label} required") from exc
    if a.dtype.kind not in "iuf" or not np.isfinite(a).all():
        raise ValueError(f"Finite real {label} required")
    return a.astype(float)


def scalar(value, label="scalar"):
    a = real(value, label)
    if a.shape != ():
        raise ValueError(f"Scalar {label} required")
    return float(a)


def binary(value, *, unknown=False):
    if isinstance(value, (list, tuple)) and any(
        isinstance(v, (bool, np.bool_)) for v in value
    ):
        raise ValueError("Boolean observations are not numeric binary labels")
    a = np.asarray(value)
    if a.ndim != 1 or not len(a) or a.dtype.kind not in "iuf" or np.isinf(a).any():
        raise ValueError("Nonempty one-dimensional real binary labels required")
    observed = ~np.isnan(a)
    if not unknown and not observed.all():
        raise ValueError("Every required binary label must be observed")
    if not np.isin(a[observed], [0, 1]).all():
        raise ValueError("Observed labels must be exactly zero or one")
    return a.astype(float)


def product(a, b, label="product"):
    a, b = real(a, label), real(b, label)
    with np.errstate(all="ignore"):
        result = a * b
    result = real(result, label)
    if ((a != 0) & (b != 0) & (result == 0)).any():
        raise ValueError(f"Nonzero multiplication underflow in {label}")
    return result


def total(value):
    try:
        answer = math.fsum(float(v) for v in real(value, "summands").ravel())
    except (OverflowError, ValueError) as exc:
        raise ValueError("Finite compensated sum required") from exc
    return scalar(answer, "sum")


def close(actual, expected, label, *, rtol=REPLAY_RTOL, atol=REPLAY_ATOL):
    a, b = real(actual, label), real(expected, label)
    if a.shape != b.shape or not np.allclose(a, b, rtol=rtol, atol=atol):
        raise ValueError(f"{label} does not independently reconstruct")


def same_tree(actual, expected, label):
    try:
        a = json.dumps(actual, sort_keys=True, allow_nan=False)
        b = json.dumps(expected, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Strict finite JSON {label} required") from exc
    if a != b:
        raise ValueError(f"{label} identity changed")


def independent_summary(events):
    e = binary(events)
    if len(e) != 22:
        raise ValueError("Exactly22 uncompressed binary arrivals required")
    ones = tuple(int(v) for v in e)
    n = sum(ones)
    last = ones[21]
    c = sum(ones[j - 1] * ones[j] for j in range(1, 22))
    g_num = (n - last) * (n - 1)
    r_num = sum((2 * j - 21) * ones[j] for j in range(22))
    m_num = 21 * c - g_num
    values = (
        n / 22,
        n * n / 484,
        float(last),
        g_num / 441,
        r_num / 462,
        c / 21,
        m_num / 441,
        n,
        c,
        g_num,
        r_num,
        m_num,
    )
    return dict(zip(SUMMARY_COLUMNS, values, strict=True))


def reconstruct_memory(targets):
    if not isinstance(targets, pd.DataFrame) or tuple(targets.columns) != (
        "y",
        "target_end",
        "available_date",
    ):
        raise ValueError("Exact original target table schema required")
    dates = targets.index
    if (
        not isinstance(dates, pd.DatetimeIndex)
        or not len(dates)
        or dates.has_duplicates
        or dates.hasnans
        or dates.tz is not None
        or not dates.is_monotonic_increasing
        or not dates.equals(dates.normalize())
        or dates[-1] > pd.Timestamp("2025-10-20")
    ):
        raise ValueError("Full bounded unique naive reference dates required")
    values = binary(targets.y.to_numpy(), unknown=True)
    following = pd.Series(dates, index=dates).shift(-1)
    for column in ("target_end", "available_date"):
        if not pd.api.types.is_datetime64_dtype(targets[column].dtype) or not targets[
            column
        ].equals(following.rename(column)):
            raise ValueError("Each label must mature at its next actual reference close")
    if not np.isnan(values[-1]):
        raise ValueError("Terminal label must be unknown")
    numeric = np.full((len(dates), len(SUMMARY_COLUMNS)), np.nan)
    structural = {
        name: pd.Series(pd.NaT, index=dates, dtype=dates.dtype) for name in DATE_COLUMNS
    }
    structural["feature_cutoff_date"] = pd.Series(dates, index=dates).shift(1)
    # Select literal availability positions, not the last22 observed records.
    for i in range(23, len(dates)):
        positions = np.arange(i - 23, i - 1)
        arrivals = following.iloc[positions]
        if not arrivals.index.equals(dates[positions]) or not np.array_equal(
            arrivals.to_numpy(), dates[i - 22 : i].to_numpy()
        ):
            raise ValueError("Exact22 full-clock mature arrivals required")
        for key, position in zip(
            DATE_COLUMNS[1:], (i - 22, i - 1, i - 23, i - 2), strict=True
        ):
            structural[key].iloc[i] = dates[position]
        history = values[positions]
        if np.isfinite(history).all():
            numeric[i] = list(independent_summary(history).values())
    frame = pd.DataFrame(numeric, index=dates, columns=SUMMARY_COLUMNS)
    for name in DATE_COLUMNS:
        frame[name] = structural[name]
    return frame.loc[:, MEMORY_COLUMNS]


def require_original_histories(memory, training_dates, application_dates):
    if (
        not isinstance(memory, pd.DataFrame)
        or tuple(memory.columns) != MEMORY_COLUMNS
        or memory.index.has_duplicates
    ):
        raise ValueError("Exact full memory table required")
    for label, selected in (("training", training_dates), ("application", application_dates)):
        selected = pd.Index(selected)
        if (
            not len(selected)
            or selected.has_duplicates
            or not selected.isin(memory.index).all()
        ):
            raise ValueError(f"Exact original nonempty {label} cohort required")
        rows = memory.loc[selected]
        real(rows.loc[:, FEATURES].to_numpy(), f"every original {label} history")
        if rows.loc[:, DATE_COLUMNS].isna().any().any():
            raise ValueError(f"Incomplete original {label} history; cohort cannot shrink")


def _history_frame(frame):
    if (
        not isinstance(frame, pd.DataFrame)
        or tuple(frame.columns) != FEATURES
        or not len(frame)
        or frame.index.has_duplicates
    ):
        raise ValueError("Exact nonempty six-coordinate history table required")
    a = real(frame.to_numpy(), "history coordinates")
    if (a[:, :4] < 0).any() or (a[:, :4] > 1).any() or (np.abs(a[:, 4:]) > 1).any():
        raise ValueError("Original bounded history coordinate domains required")
    return a


def independent_transform(train, apply):
    values, query = _history_frame(train), _history_frame(apply)
    if len(values) < 2:
        raise ValueError("At least two original training rows required")
    constant = np.array([bool(np.all(values[:, j] == values[0, j])) for j in range(6)])
    means = np.array(
        [values[0, j] if constant[j] else total(values[:, j]) / len(values) for j in range(6)]
    )
    centered, app = (
        real(values - means, "training center"),
        real(query - means, "application center"),
    )
    x, q = (
        np.column_stack((np.ones(len(values)), centered[:, :5])),
        np.column_stack((np.ones(len(query)), app[:, :5])),
    )
    audit = {
        "columns": ["const", *NUISANCE],
        "means": [0.0, *means[:5].tolist()],
        "scales": [1.0] * 6,
        "nuisance_constant": constant[:5].tolist(),
        "memory_mean": float(means[5]),
        "memory_scale": 1.0,
        "memory_constant": bool(constant[5]),
        "train_n": len(values),
    }
    return x, q, centered[:, 5], app[:, 5], audit


def logit(design, beta):
    x, b = real(design, "design"), real(beta, "coefficients")
    if x.ndim != 2 or b.shape != (x.shape[1],):
        raise ValueError("Aligned matrix and coefficients required")
    product(x, b, "coefficient/design products")
    with np.errstate(all="ignore"):
        result = x @ b
    return real(result, "logits")


def independent_baseline_logits(frame, audit):
    base, geometry = audit["baseline"], audit["transform"]
    same_tree(base["columns"], list(OLD_BASE), "old baseline columns")
    same_tree(geometry["columns"], list(OLD_BASE), "old transformation columns")
    for field in ("means", "scales"):
        same_tree(base[field], geometry[field], f"old {field}")
    means, scales, beta = (real(base[field], field) for field in ("means", "scales", "beta"))
    if (
        means.shape != (26,)
        or scales.shape != (26,)
        or beta.shape != (26,)
        or means[0] != 0
        or scales[0] != 1
        or (scales[1:] <= 1e-12).any()
    ):
        raise ValueError("Original26 dimensional fixed geometry required")
    raw = frame.loc[:, OLD_RAW].copy()
    real(raw.to_numpy(), "old raw values")
    for name in ("I", "R", "skew"):
        center = scalar(geometry["curvature_means"][name], "curvature center")
        difference = real(raw[name].to_numpy() - center, "centered curvature")
        raw[name + "_square"] = product(difference, difference, "original curvature square")
    numerator = real(raw.loc[:, OLD_BASE].to_numpy() - means, "centered old coordinates")
    with np.errstate(all="ignore"):
        design = numerator / scales
    design = real(design, "old standardized coordinates")
    if ((numerator != 0) & (design == 0)).any():
        raise ValueError("Nonzero original-coordinate division underflow")
    table = pd.DataFrame(design, index=frame.index, columns=OLD_BASE)
    return logit(design, beta), table


def nuisance_objective(beta, design, y, offset):
    b, x, labels, eta0 = (
        real(beta, "nuisance coefficients"),
        real(design, "nuisance design"),
        binary(y),
        real(offset, "nuisance offset"),
    )
    if (
        b.shape != (6,)
        or x.shape != (len(labels), 6)
        or eta0.shape != labels.shape
        or not np.all(x[:, 0] == 1)
    ):
        raise ValueError("Exact six-coordinate offset logistic problem required")
    # The registered nuisance objective has the original finite-value policy.
    # Underflow guards for unused scalar likelihood diagnostics do not apply.
    with np.errstate(all="ignore"):
        eta = real(eta0 + x @ b, "nuisance logits")
        terms = np.logaddexp(0, (1 - 2 * labels) * eta)
        p, opposite = expit(eta), expit(-eta)
        residual = np.where(labels == 1, -opposite, p)
        curvature = p * opposite
        penalty = b.copy()
        penalty[0] = 0
        value = total(terms) / len(labels) + ALPHA * float(penalty @ penalty)
        gradient = (
            np.array([total(x[:, j] * residual) / len(labels) for j in range(6)])
            + 2 * ALPHA * penalty
        )
        jac = x.T @ (curvature[:, None] * x) / len(labels) + np.diag([0.0, *([2 * ALPHA] * 5)])
    value = scalar(value, "nuisance objective")
    if value < 0:
        raise ValueError("Nuisance objective must be nonnegative")
    return value, real(gradient, "full nuisance gradient"), real(jac, "nuisance Jacobian")


def independent_nuisance_fit(x, y, eta):
    labels = binary(y)
    if min(np.count_nonzero(labels), np.count_nonzero(labels == 0)) < 50:
        raise ValueError("Original training class support requires50 each")
    nuisance_objective(np.zeros(6), x, labels, eta)
    try:
        solved = root(
            lambda b: nuisance_objective(b, x, labels, eta)[1],
            np.zeros(6),
            jac=lambda b: nuisance_objective(b, x, labels, eta)[2],
            method="hybr",
            options={"xtol": 1e-10, "maxfev": 2000, "factor": 1},
        )
    except Exception as exc:
        raise ValueError("Single independent nuisance root failed; no fallback") from exc
    beta = real(solved.x, "independent nuisance coefficients")
    value, gradient, _ = nuisance_objective(beta, x, labels, eta)
    maximum = float(np.max(np.abs(gradient)))
    if solved.success is not True or maximum > GRADIENT_TOLERANCE:
        raise ValueError(
            f"Independent nuisance stationarity failed: {solved.message}; gradient={maximum}"
        )
    return beta, {
        "method": "hybr",
        "start": [0.0] * 6,
        "xtol": 1e-10,
        "maxfev": 2000,
        "factor": 1,
        "success": True,
        "status": int(solved.status),
        "message": str(solved.message),
        "function_evaluations": int(solved.nfev),
        "jacobian_evaluations": int(solved.njev),
        "coefficients": beta.tolist(),
        "objective": value,
        "gradient": gradient.tolist(),
        "gradient_max_abs": maximum,
    }


def scalar_objective(coefficient, z, y, offset):
    b, memory, labels, eta0 = (
        scalar(coefficient, "scalar coefficient"),
        real(z, "centered adjacency"),
        binary(y),
        real(offset, "frozen nuisance offset"),
    )
    if (
        memory.shape != labels.shape
        or eta0.shape != labels.shape
        or (np.abs(memory) > 2).any()
    ):
        raise ValueError("Aligned bounded scalar problem required")
    eta = real(eta0 + product(b, memory, "scalar correction"), "scalar logits")
    with np.errstate(all="ignore"):
        terms = real(np.logaddexp(0, (1 - 2 * labels) * eta), "scalar signed softplus")
        residual = real(
            np.where(labels == 1, -expit(-eta), expit(eta)), "scalar signed residual"
        )
    if (terms == 0).any() or (residual == 0).any():
        raise ValueError("Finite-logit scalar likelihood/residual underflow")
    value = total(terms) / len(labels) + float(
        product(ALPHA, product(b, b, "scalar square"), "scalar penalty")
    )
    gradient = total(product(memory, residual, "scalar derivative")) / len(labels) + float(
        product(2 * ALPHA, b, "scalar ridge gradient")
    )
    return scalar(value, "scalar objective"), scalar(gradient, "scalar gradient")


def independent_scalar_fit(z, y, offset):
    memory, labels, eta = real(z, "centered adjacency"), binary(y), real(offset, "offset")
    _, g0 = scalar_objective(0.0, memory, labels, eta)
    radius = (total(np.abs(memory)) / len(labels) + 1) / (2 * ALPHA)
    bracket = [-radius, radius]
    gradients = [scalar_objective(b, memory, labels, eta)[1] for b in bracket]
    if not gradients[0] < 0 < gradients[1]:
        raise ValueError("Independent guaranteed scalar bracket failed")
    iterations, calls = 0, 0
    if np.all(memory == 0):
        b, status = 0.0, "EXACT_CONSTANT_INPUT"
    elif g0 == 0:
        b, status = 0.0, "BALANCED_AT_ZERO"
    else:
        try:
            b, solved = bisect(
                lambda c: scalar_objective(c, memory, labels, eta)[1],
                *bracket,
                xtol=1e-12,
                rtol=1e-14,
                maxiter=200,
                full_output=True,
            )
        except Exception as exc:
            raise ValueError(
                "Single independent scalar bisection failed; no fallback"
            ) from exc
        if (
            solved.converged is not True
            or not 0 <= solved.iterations <= 200
            or not 2 <= solved.function_calls <= 202
        ):
            raise ValueError("Independent scalar convergence record invalid")
        iterations, calls, status = (
            int(solved.iterations),
            int(solved.function_calls),
            "FITTED",
        )
    objective, gradient = scalar_objective(b, memory, labels, eta)
    if not -radius <= b <= radius or abs(gradient) > GRADIENT_TOLERANCE:
        raise ValueError("Independent full scalar stationarity failed")
    return float(b), {
        "coefficient": float(b),
        "status": status,
        "alpha": ALPHA,
        "objective": objective,
        "gradient": gradient,
        "gradient_max_abs": abs(gradient),
        "bracket": bracket,
        "bracket_gradients": gradients,
        "gradient_at_zero": g0,
        "iterations": iterations,
        "function_calls": calls,
        "success": True,
        "method": "bisect",
        "xtol": 1e-12,
        "rtol": 1e-14,
        "maxiter": 200,
    }


def probabilities(value, length, label):
    p = real(value, label)
    if p.shape != (length,) or (p < 0).any() or (p > 1).any():
        raise ValueError(f"Strict probability domain and shape required: {label}")
    return p


def integer(value, lower, upper, label):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not lower <= value <= upper
    ):
        raise ValueError(f"Fixed integer {label} required")


def verify_stages(
    train, y, apply, train_eta, apply_eta, baseline_probability, predictions, audit
):
    """Reconstruct saved supplied-table stages using independent optimizers."""
    if not isinstance(y, pd.Series) or not y.index.equals(train.index):
        raise ValueError("Exact mature training label alignment required")
    labels = binary(y.to_numpy())
    events = int(np.count_nonzero(labels))
    if min(events, len(labels) - events) < 50:
        raise ValueError("At least50 training labels per class required")
    same_tree(
        audit["support"],
        {"n": len(labels), "events": events, "nonevents": len(labels) - events},
        "training support",
    )
    if (
        audit.get("baseline_frozen") is not True
        or audit.get("training_offset_kind") != "current_month_saved_fit_in_sample"
    ):
        raise ValueError("Original monthly baseline must remain a frozen in-sample offset")
    x, q, z, qz, geometry = independent_transform(train, apply)
    same_tree(audit["transform"], geometry, "training-only geometry")
    eta, query = real(train_eta, "training offsets"), real(apply_eta, "query offsets")
    if eta.shape != labels.shape or query.shape != (len(apply),):
        raise ValueError("Original offset dimensions changed")
    parent = probabilities(baseline_probability, len(apply), "old baseline probability")
    close(parent, expit(query), "old baseline replay")
    saved = audit["nuisance"]
    beta = real(saved["beta"], "saved nuisance beta")
    if scalar(saved["alpha"]) != ALPHA or saved.get("success") is not True:
        raise ValueError("Successful fixed-ridge nuisance fit required")
    value, gradient, _ = nuisance_objective(beta, x, labels, eta)
    maximum = float(np.max(np.abs(gradient)))
    if not 0 <= scalar(saved["gradient_max_abs"]) <= 1e-8 or maximum > GRADIENT_TOLERANCE:
        raise ValueError("Saved original full nuisance gradient gate failed")
    close(saved["objective"], value, "nuisance objective")
    close(saved["gradient"], gradient, "full nuisance gradient")
    close(saved["gradient_max_abs"], maximum, "nuisance gradient maximum")
    same_tree(saved["start"], [0.0] * 6, "nuisance zero start")
    integer(saved["iterations"], 0, 200, "Newton iterations")
    integer(saved["backtracks"], 0, 59 * saved["iterations"], "Armijo backtracks")
    for k, constant in enumerate(geometry["nuisance_constant"], start=1):
        if constant and beta[k] != 0:
            raise ValueError("Constant nuisance coordinate must retain zero slope")
    independent_beta, independent_nuisance = independent_nuisance_fit(x, labels, eta)
    close(
        beta,
        independent_beta,
        "independent nuisance coefficients",
        rtol=COEFFICIENT_RTOL,
        atol=COEFFICIENT_ATOL,
    )
    nuisance_train = real(eta + logit(x, beta), "saved nuisance training logits")
    correction = logit(q, beta)
    nuisance_query = real(query + correction, "saved nuisance query logits")
    p_nuisance = np.where(correction == 0, parent, expit(nuisance_query))
    fitted = audit["cluster"]
    b = scalar(fitted["coefficient"], "saved cluster coefficient")
    independent_b, independent_scalar = independent_scalar_fit(z, labels, nuisance_train)
    close(
        b,
        independent_b,
        "independent cluster coefficient",
        rtol=COEFFICIENT_RTOL,
        atol=COEFFICIENT_ATOL,
    )
    scalar_value, scalar_gradient = scalar_objective(b, z, labels, nuisance_train)
    if (
        not 0 <= scalar(fitted["gradient_max_abs"]) <= 1e-8
        or abs(scalar_gradient) > GRADIENT_TOLERANCE
    ):
        raise ValueError("Saved original full cluster gradient gate failed")
    for name, expected in (
        ("objective", scalar_value),
        ("gradient", scalar_gradient),
        ("gradient_max_abs", abs(scalar_gradient)),
    ):
        close(fitted[name], expected, "saved scalar " + name)
    for name in (
        "alpha",
        "bracket",
        "bracket_gradients",
        "gradient_at_zero",
        "xtol",
        "rtol",
        "maxiter",
    ):
        close(fitted[name], independent_scalar[name], "scalar " + name)
    if (
        fitted.get("success") is not True
        or fitted.get("method") != "brentq"
        or fitted.get("status") != independent_scalar["status"]
    ):
        raise ValueError("Declared scalar method/status changed")
    bracket = independent_scalar["bracket"]
    if not bracket[0] <= b <= bracket[1]:
        raise ValueError("Saved coefficient outside original bracket")
    if independent_scalar["status"] == "FITTED":
        integer(fitted["iterations"], 0, 200, "Brent iterations")
        integer(fitted["function_calls"], 2, 202, "Brent calls")
    else:
        integer(fitted["iterations"], 0, 0, "flat/balanced iterations")
        integer(fitted["function_calls"], 0, 0, "flat/balanced calls")
        if b != 0:
            raise ValueError("Canonical zero cluster coefficient required")
    delta = product(b, qz, "application cluster correction")
    p_cluster = np.where(
        delta == 0, p_nuisance, expit(real(nuisance_query + delta, "cluster query logits"))
    )
    if set(predictions) != {"nuisance", "cluster"}:
        raise ValueError("Exact new prediction arms required")
    for model, expected, zero, original_probability in (
        ("nuisance", p_nuisance, correction == 0, parent),
        (
            "cluster",
            p_cluster,
            delta == 0,
            probabilities(predictions["nuisance"], len(apply), "saved nuisance parent"),
        ),
    ):
        actual = probabilities(predictions[model], len(apply), model)
        close(actual, expected, "saved " + model + " probabilities")
        if not np.array_equal(actual[zero], original_probability[zero]):
            raise ValueError("Exact zero-correction parent probability identity failed")
    return {
        "train_n": len(train),
        "application_n": len(apply),
        "nuisance_recomputed_gradient_max_abs": maximum,
        "cluster_recomputed_gradient_max_abs": abs(scalar_gradient),
        "independent_nuisance": independent_nuisance,
        "independent_cluster": independent_scalar,
    }


ROOT = Path(__file__).resolve().parents[1]
REPORT = "reports/event_cluster"
OUT = "data/event_cluster"
MODELS = ("baseline", "recent_frequency", "nuisance", "cluster")
COMPARISONS = (
    ("cluster", "baseline"),
    ("cluster", "nuisance"),
    ("cluster", "recent_frequency"),
)
PANEL_COLUMNS = old.PANEL_COLUMNS
EFFECT = 0.0005
WAVE_ALPHA = 0.05 / (20 * 21)
compare_tree = previous.compare_tree
holm = previous.holm
support = old.support
brier_loss = old.brier_loss
paired_difference = old.paired_difference
normalized_difference = old.normalized_difference
explicit_bootstrap_means = old.explicit_bootstrap_means
independent_hac = old.independent_hac
_finite = scalar
strict_json = previous.strict_json
source_path = previous.source_path
hash_mapping = previous.hash_mapping
identity = same_tree

CONTRACT = {
    "study_id": "event_cluster_wave20",
    "specified_on": "2026-09-07",
    "status": "specified_before_new_event_history_or_fits",
    "wave": 20,
    "evidence_class": "exploratory_reused_history_archival_SPX_daily_risk_event",
    "objective": "Test within-series adjacent mature risk events beyond fixed market forecast and "
    "matched event count/recency controls",
    "original_index": {
        "asset": "SPX",
        "source_end": "2025-10-20",
        "sealed_start": "2025-11-03",
        "origin_start": "2016-01-04",
        "origin_end": "2025-10-17",
        "latest_target": "2025-10-20",
        "development": ["2016-01-04", "2019-12-31"],
        "development_target_available_by": "2019-12-31",
        "evaluation": ["2020-01-02", "2025-10-17"],
        "evaluation_stability": [["2020-01-02", "2022-12-31"], ["2023-01-01", "2025-10-17"]],
        "horizons": [1],
        "market_lag": 1,
        "minimum_train": 1000,
        "models": ["baseline", "recent_frequency", "location"],
        "minimum_train_per_class": 50,
        "minimum_phase_per_class": 30,
        "minimum_slice_per_class": 15,
        "minimum_phase_observations": 127,
        "baseline": [
            "const",
            "I",
            "R",
            "ret_d",
            "ret_w",
            "ret_m",
            "ret_q",
            "lr_d",
            "lr_w",
            "term",
            "lvvix",
            "entry_dow_1",
            "entry_dow_2",
            "entry_dow_3",
            "entry_dow_4",
            "neg_d",
            "neg_w",
            "neg_m",
            "skew",
            "I_square",
            "R_square",
            "skew_square",
            "intraday",
            "intraday_sq",
            "overnight",
            "overnight_sq",
        ],
        "all_features": [
            "const",
            "I",
            "R",
            "ret_d",
            "ret_w",
            "ret_m",
            "ret_q",
            "lr_d",
            "lr_w",
            "term",
            "lvvix",
            "entry_dow_1",
            "entry_dow_2",
            "entry_dow_3",
            "entry_dow_4",
            "neg_d",
            "neg_w",
            "neg_m",
            "skew",
            "intraday",
            "intraday_sq",
            "overnight",
            "overnight_sq",
            "range_extremity",
        ],
    },
    "history": {
        "window": 22,
        "ordered_by": "available_date on full original SPX calendar, oldest first, "
        "through previous-close cutoff",
        "origin_position_rule": "at i>=23 use y.iloc[i-23:i-1], whose available positions "
        "are i-22..i-1",
        "labels": "all known original target labels, including "
        "unissued/unscored/feature-incomplete origins; not historical forecast "
        "errors",
        "unknown": "every original training/application window must contain22 known "
        "labels; otherwise whole-trial failure before fitting; never fill, "
        "compress or drop original rows",
        "nuisance": [
            "event_fraction22",
            "event_fraction22_sq",
            "last_event",
            "expected_adjacency22",
            "linear_recency22",
        ],
        "nuisance_formula": [
            "N/22",
            "N*N/484",
            "last_event",
            "(N-last_event)*(N-1)/441",
            "sum((2*i-23)*e_i for i=1..22)/462",
        ],
        "candidate": "adjacency_fraction22",
        "candidate_formula": "C/21; C=sum(e_i*e_(i-1) for i=2..22)",
        "diagnostic": "excess_adjacency22=(21*C-(N-last_event)*(N-1))/441; never enters "
        "fit or promotion",
        "arithmetic": "exact Python integer numerators followed by single fixed division; "
        "N*N/484 is not square of rounded N/22",
        "centering": "training-only math.fsum/n; exactconstant uses firstvalue; "
        "fixedscale1 and no rank/column deletion",
    },
    "models": ["baseline", "recent_frequency", "nuisance", "cluster"],
    "model": {
        "baseline": "original wave18 saved26-coefficient monthly fit; no refitting; "
        "original issuedquery probabilities strictlyreplayed and retained",
        "training_offset": "apply that same savedmonthlyfit to its original "
        "maturetrainingrows; in-sample currentfit offsets, not "
        "historicalissuances",
        "nuisance": "mean Bernoulli NLL with frozenbaselineoffset +freeintercept +five "
        "centeredboundedslopes, .01sum(slope^2)",
        "candidate": "mean Bernoulli NLL with frozennuisanceoffset +one centeredC/21 slope, "
        ".01coefficient^2; no newintercept",
        "ridge": 0.01,
        "producer_nuisance": {
            "method": "frozen sign_memory_models._newton",
            "start": "six literalzeros",
            "maxiter": 200,
            "max_backtracks": 60,
            "armijo": 0.0001,
            "gradient_tolerance": 1e-08,
        },
        "producer_scalar": {
            "method": "brentq",
            "bracket": "+/- (mean(abs(centered_candidate))+1)/.02",
            "xtol": 1e-12,
            "rtol": 1e-14,
            "maxiter": 200,
            "gradient_tolerance": 1e-08,
        },
        "constant": "center exactconstants exactly; constantnuisances "
        "retainzerocoefficients; constantadjacency retainsscalarzero and "
        "nuisanceprobability",
        "probability": "copy checkedparent probability when addedquerycorrection "
        "exactlyzero; otherwise expit(parentlogit+correction); no clipping",
        "scalar_arithmetic": "finitefloat64; signedsoftplus and signedresidual remain "
        "nonzero for finite logits; checked nonzero products "
        "rejectunderflow; compensated scalar reductions",
        "nuisance_arithmetic": "frozen logistic_state numerical contract; finite objective, "
        "fullgradient and Hessian; no optional unusedcurvature "
        "admission gate",
        "fallback": "none; no empiricalrepair or tolerance change",
    },
    "comparisons": {
        "new_hypotheses": 3,
        "inherited_hypotheses": 131,
        "cumulative_hypotheses": 134,
        "inherited_sources": [
            "reports/orthogonal_round2/metrics.json",
            "reports/model_memory_study/combined_metrics.json",
            "reports/iterative_signal_search/metrics.json",
            "reports/international_volatility/metrics.json",
            "reports/overnight_index/metrics.json",
            "reports/macro_overnight/metrics.json",
            "reports/measurement_memory/metrics.json",
            "reports/index_hinge/metrics.json",
            "reports/tail_shape/metrics.json",
            "reports/calendar_variance/metrics.json",
            "reports/relative_risk/metrics.json",
            "reports/joint_risk/metrics.json",
            "reports/cross_moment/metrics.json",
            "reports/target_aligned/metrics.json",
            "reports/sign_memory/metrics.json",
            "reports/causal_pool/metrics.json",
            "reports/civil_quarter/metrics.json",
            "reports/civil_quarter_replay/metrics.json",
            "reports/profiled_quarter/metrics.json",
            "reports/range_alert/metrics.json",
            "reports/issued_calibration/metrics.json",
        ],
        "contrasts": [
            ["cluster", "baseline", "brier"],
            ["cluster", "nuisance", "brier"],
            ["cluster", "recent_frequency", "brier"],
        ],
        "gate": "allthreecontrols must improve in bothphases by>=.0005 Brier, "
        "bothfixedlateslices negative, bothmultiplicity gates; "
        "nootherpromotion contrast",
    },
    "inference": {
        "loss": "brier",
        "effect_absolute": 0.0005,
        "blocks": [21, 63, 126],
        "hac_lags": 126,
        "bootstrap_draws": 399999,
        "seed": 20260926,
        "seed_rule": "seed+phase_code*10000+block; development0,evaluation1; same "
        "drawdesign acrosscontrols",
        "wave_alpha": 0.00011904761904761905,
        "cumulative_alpha": 0.05,
        "p_phase": "maximum two-sided centered-null circularblock bootstrap and "
        "BartlettHAC126p",
        "p_hypothesis": "maximumdevelopment/evaluationp",
        "multiplicity": "Holm3 versus wave_alpha and cumulativeHolm134 versus.05",
        "resolution": "1/400000 below one tenth minimumHolm3rawwavecutoff1/25200",
        "support": {
            "minimum_train": 1000,
            "train_per_class": 50,
            "phase_observations": 127,
            "phase_per_class": 30,
            "slice_per_class": 15,
        },
        "failure": "anysource,history,support,arithmetic,solver,forecast,inference,verificationorpublication "
        "failure keepsall3UNEVALUABLEp1; retainpriorfailures",
        "calibration": "descriptivein-the-large meanprob,eventrate,gap,Brier "
        "perphase/model; no fittedcalibrationbins",
        "power": "nominal HAC80percentMDE/.0005 descriptive; not equivalence or "
        "post-hocselection",
    },
    "verification": {
        "nuisance": {
            "method": "root_hybr",
            "start": "six literalzeros",
            "xtol": 1e-10,
            "maxfev": 2000,
            "factor": 1.0,
            "analytic_jacobian": True,
            "full_gradient_tolerance": 1.0001e-08,
        },
        "scalar": {
            "method": "bisect",
            "bracket": "+/- (mean(abs(centered_candidate))+1)/.02",
            "xtol": 1e-12,
            "rtol": 1e-14,
            "maxiter": 200,
            "full_gradient_tolerance": 1.0001e-08,
        },
        "coefficient_rtol": 1e-07,
        "coefficient_atol": 1e-06,
        "replay_rtol": 1e-10,
        "replay_atol": 1e-12,
        "inference_rtol": 1e-09,
        "inference_atol": 1e-14,
        "conditional_scalar": "independentlysolve using saved independentlyvalidated "
        "nuisancebeta, preservingactualfrozenoffsetobjective",
        "scope": "fullnewhistory, originalcohortsandissuedcontrols, "
        "allnewmonthlystages/applicationpredictionsincludingunscored, "
        "scores,inference,ledger,sourceclosure",
    },
    "upstream": {
        "anchors": {
            "issued_calibration.yaml": "f4c03c25718439432c176aba36ce3bc572776b67a5ff80f906ddc00cd85b2051",
            "reports/issued_calibration/manifest.json": "5d824aad69991ea06cc35916be2e9852d9f5613c17bafe7b18e46047d93bdf6a",
            "reports/issued_calibration/freeze_record.json": "d12459d404d4bb7cfab43a351764b1c29353d0e2dbfbec6af689ffc411ba0c71",
            "reports/issued_calibration/verification.json": "5de5f8ed0f4a7757c475e0557b9798fe04d8fea33714a5c68437fbb181d48d14",
            "reports/issued_calibration/publication_audit.json": "17275a47b0c8738e3177e905025869fd69523cfc4cdcca45d4fd0cbd49d45be8",
            "reports/issued_calibration/publication_review.json": "481565b24889457d8bf87ee824aac1284e688418a02fe7b4715aeee935e96003",
            "reports/early_session_feasibility/preservation_audit.json": "feb526675c97a52bc13f372dd50024093d9fc6a7fe3580859fd20f7ab585dc41",
        },
        "admission": "fullwave19successandpublicationclosure plusoriginalwave18tables "
        "andzero-hypothesisearlysessiondocumentation; checkedbytes "
        "beforedecode",
    },
    "outputs": {"data": "data/event_cluster", "reports": "reports/event_cluster"},
}


def phase_statistics(panel, control, phase, code, protocol):
    section = protocol["original_index"]
    first, last = section[phase]
    selected = panel.loc[(panel.origin >= first) & (panel.origin <= last)]
    if phase == "development":
        selected = selected.loc[
            selected.available_date <= section["development_target_available_by"]
        ]
    groups = {
        name: selected.loc[selected.model == name].set_index("origin").sort_index()
        for name in MODELS
    }
    rows, benchmark = groups["cluster"], groups[control]
    if any(not part.index.equals(rows.index) for part in groups.values()) or len(rows) < 127:
        raise AssertionError(
            "Complete original cohort and at least127 phase observations required"
        )
    difference = paired_difference(rows.probability, benchmark.probability, rows.y)
    values = normalized_difference(difference)
    mean = float(values.mean())
    delta = _finite(mean * 1.0)
    hac = independent_hac(values, maxlags=126)
    blocks = {}
    for width in protocol["inference"]["blocks"]:
        seed = protocol["inference"]["seed"] + code * 10000 + width
        samples = explicit_bootstrap_means(
            values, width, protocol["inference"]["bootstrap_draws"], seed
        )[:, 0]
        if not np.isfinite(samples).all():
            raise ValueError("Nonfinite independent normalized bootstrap samples")
        probability = float(
            (1 + np.count_nonzero(abs(samples - mean) >= abs(mean))) / (len(samples) + 1)
        )
        blocks[str(width)] = {
            "p": probability,
            "ci95": [_finite(value * 1.0) for value in np.quantile(samples, [0.025, 0.975])],
        }
    hac = {
        "se": _finite(hac["se"] * 1.0),
        "p": _finite(hac["p"]),
        "ci95": [_finite(value * 1.0) for value in hac["ci95"]],
        "mde80_nominal": _finite(hac["mde80_nominal"] * 1.0),
    }
    intervals = [hac["ci95"], *(row["ci95"] for row in blocks.values())]
    result = {
        "name": phase,
        "first_origin": str(rows.index[0].date()),
        "last_origin": str(rows.index[-1].date()),
        "n": len(rows),
        "class_support": support(rows.y, 30),
        "delta": delta,
        "candidate_loss": _finite(rows.loss.mean()),
        "control_loss": _finite(benchmark.loss.mean()),
        "block_inference": blocks,
        "hac126": hac,
        "ci95_envelope": [
            min(pair[0] for pair in intervals),
            max(pair[1] for pair in intervals),
        ],
        "p_conservative": max(hac["p"], *(row["p"] for row in blocks.values())),
        "nominal_mde_effect_ratio": _finite(hac["mde80_nominal"] / EFFECT),
        "annual": [],
        "stability": [],
        "nonoverlap_phases": [{"phase": 0, "n": len(rows), "delta": delta}],
    }
    for year in sorted(set(rows.index.year)):
        mask = rows.index.year == year
        result["annual"].append(
            {
                "year": int(year),
                "n": int(mask.sum()),
                "delta": _finite(values[mask].mean() * 1.0),
            }
        )
    if phase == "evaluation":
        for start, end in section["evaluation_stability"]:
            mask = (rows.index >= start) & (rows.index <= end)
            if not mask.any():
                raise AssertionError("Fixed evaluation slice empty")
            result["stability"].append(
                {
                    "start": start,
                    "end": end,
                    "n": int(mask.sum()),
                    "delta": _finite(values[mask].mean() * 1.0),
                }
            )
    return result


def inherited_rows(root, protocol, *, pins=None):
    root = Path(root)
    if pins is None:
        pins = json.loads((root / "reports/event_cluster/manifest.json").read_bytes())[
            "inputs"
        ]
    output = []
    for name in protocol["comparisons"]["inherited_sources"]:
        signature = pins[name]
        decoded = read_json_snapshot(root, name, signature)
        for number, row in enumerate(decoded["rows"]):
            output.append(
                {
                    "study": row.get("study", Path(name).parent.name or Path(name).stem),
                    "candidate": row["candidate"],
                    "control": row.get("control", "baseline"),
                    "horizon": row["horizon"],
                    "p_conservative": row["p_conservative"],
                    "source": name,
                    "source_sha256": signature,
                    "source_row_index": number,
                    **{key: row[key] for key in ("measure", "score") if key in row},
                }
            )
    if len(output) != 131:
        raise AssertionError("All131 inherited comparisons must remain identified")
    for row in output:
        pvalue = row["p_conservative"]
        if (
            isinstance(pvalue, bool)
            or not isinstance(pvalue, (int, float))
            or not math.isfinite(pvalue)
            or not 0 <= pvalue <= 1
        ):
            raise ValueError("Finite inherited p-values in [0,1] required")
    return output


def verify_metrics(
    root, panel, protocol, metrics, application_n, monthly_fits, *, admitted_inputs=None
):
    validate_panel(panel)
    if type(monthly_fits) is not int or monthly_fits <= 0:
        raise ValueError("Positive literal verified monthly fit count required")
    n = panel.origin.nunique()
    expected_counts = {
        "new_monthly_fits": monthly_fits,
        "new_nuisance_fits": monthly_fits,
        "new_cluster_fits": monthly_fits,
        "original_baseline_fits": 0,
        "new_forecasts": 2 * n,
        "reused_control_forecasts": 2 * n,
        "combined_forecasts": len(panel),
        "common_application_origins": application_n,
        "common_scored_origins": n,
        "hypothesis_count": 3,
        "cumulative_hypothesis_count": 134,
    }
    same_tree(
        {key: metrics[key] for key in expected_counts},
        expected_counts,
        "literal output/family counts",
    )
    if (
        metrics["new_monthly_fits"] != monthly_fits
        or metrics["new_nuisance_fits"] != monthly_fits
        or metrics["new_cluster_fits"] != monthly_fits
        or metrics["original_baseline_fits"] != 0
        or metrics["new_forecasts"] != 2 * n
        or metrics["reused_control_forecasts"] != 2 * n
        or metrics["combined_forecasts"] != len(panel)
        or metrics["common_application_origins"] != application_n
        or metrics["common_scored_origins"] != n
    ):
        raise AssertionError(
            "Exact new/reused forecast and unchanged original-baseline fit counts differ"
        )
    section = protocol["original_index"]
    counts = {}
    calibration = {}
    phase_union = np.zeros(len(panel), dtype=bool)
    for name in ("development", "evaluation"):
        first, last = section[name]
        mask = panel.origin.between(first, last)
        phase_union |= mask
        if not panel.loc[mask, "phase"].eq(name).all():
            raise AssertionError("Literal phase membership differs")
        base = panel.loc[mask & panel.model.eq("recent_frequency")]
        counts[name] = support(base.y, 30)
        if len(base) < 127:
            raise AssertionError("Literal phase bandwidth unsupported")
        calibration[name] = {}
        for model in MODELS:
            rows = panel.loc[mask & panel.model.eq(model)]
            frequency = float(rows.y.mean())
            probability = float(rows.probability.mean())
            calibration[name][model] = {
                "n": len(rows),
                "observed_frequency": frequency,
                "mean_probability": probability,
                "calibration_gap": probability - frequency,
                "brier": float(rows.loss.mean()),
            }
    if (
        not phase_union.all()
        or not panel.horizon.eq(1).all()
        or (panel.train_n < section["minimum_train"]).any()
        or (
            panel.loc[panel.phase.eq("development"), "available_date"]
            > section["development_target_available_by"]
        ).any()
        or (panel.available_date > section["latest_target"]).any()
    ):
        raise AssertionError("Literal inference sample support/date fences differ")
    counts["evaluation_slices"] = []
    for start, end in section["evaluation_stability"]:
        values = panel.loc[
            panel.origin.between(start, end) & panel.model.eq("recent_frequency"), "y"
        ]
        counts["evaluation_slices"].append({"start": start, "end": end, **support(values, 15)})
    same_tree(metrics["class_support"], counts, "Declared phase and slice class support")
    compare_tree(metrics["calibration"], calibration, "Descriptive calibration in the large")
    rows = metrics["rows"]
    if [(row["candidate"], row["control"]) for row in rows] != list(COMPARISONS):
        raise AssertionError("All three fixed Brier controls required")
    probabilities = []
    effects = []
    for row in rows:
        if row["study"] != "event_cluster" or row["horizon"] != 1 or row["score"] != "brier":
            raise AssertionError("Event-cluster Brier comparison identity differs")
        phases = [
            phase_statistics(panel, row["control"], phase, code, protocol)
            for code, phase in enumerate(("development", "evaluation"))
        ]
        compare_tree(row["phases"], phases, "Independent Brier phase inference")
        probability = max(phase["p_conservative"] for phase in phases)
        compare_tree(row["p_conservative"], probability, "Conjunction of both phases", "p")
        probabilities.append(probability)
        effects.append(
            all(phase["delta"] <= -EFFECT for phase in phases)
            and len(phases[1]["stability"]) == 2
            and all(one["delta"] < 0 for one in phases[1]["stability"])
        )
    prior = inherited_rows(root, protocol, pins=admitted_inputs)
    same_tree(metrics["inherited_rows"], prior, "All inherited comparisons retained")
    wave = holm(probabilities)
    cumulative = holm([row["p_conservative"] for row in prior] + probabilities)[-3:]
    passed = []
    for number, row in enumerate(rows):
        compare_tree(row["p_holm_wave"], wave[number], "Three-hypothesis wave correction", "p")
        compare_tree(
            row["p_holm_cumulative"],
            cumulative[number],
            "134-hypothesis cumulative correction",
            "p",
        )
        one = bool(effects[number] and wave[number] < WAVE_ALPHA and cumulative[number] < 0.05)
        if row["verdict"] != ("PASSES_ALL_GATES" if one else "DOES_NOT_QUALIFY"):
            raise AssertionError("Fixed statistical/effect/stability gate differs")
        passed.append(one)
    leads = ["cluster"] if all(passed) else []
    if (
        metrics["leads"] != leads
        or metrics["hypothesis_count"] != 3
        or metrics["cumulative_hypothesis_count"] != 134
    ):
        raise AssertionError("Complete candidate and hypothesis family required")
    return {
        "new_hypotheses_verified": 3,
        "cumulative_hypotheses_verified": 134,
        "phase_comparisons_verified": 6,
        "bootstrap_runs_verified": 18,
        "bootstrap_draws_per_run": protocol["inference"]["bootstrap_draws"],
        "calibration_rows_verified": 8,
        "class_support": counts,
        "leads": leads,
    }


def verify_ledger(root, metrics, prior, final_event, *, signature=None):
    report = Path(root) / "reports/event_cluster"
    payload = (report / "trial_ledger.jsonl").read_bytes()
    if signature is not None and sha256(payload).hexdigest() != signature:
        raise AssertionError("Trial ledger bytes changed before decoding")
    ledger = [strict_json(line) for line in payload.splitlines() if line]
    for event, expected in (("inherited", prior), (final_event, metrics["rows"])):
        actual = [
            {key: value for key, value in row.items() if key != "event"}
            for row in ledger
            if row["event"] == event
        ]
        same_tree(actual, expected, "Independent " + event + " trial ledger")
    registered = [row for row in ledger if row["event"] == "registered"]
    if (
        len(ledger) != 137
        or len(registered) != 3
        or len(prior) != 131
        or [(row["candidate"], row["control"]) for row in registered] != list(COMPARISONS)
        or any(
            row["protocol_sha256"] != metrics["protocol_sha256"]
            or row["study"] != "event_cluster"
            or row["horizon"] != 1
            or row["score"] != "brier"
            for row in registered
        )
    ):
        raise AssertionError("Complete131inherited+3registered+3terminal ledger required")
    return {"inherited": 131, "registered": 3, final_event: 3}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def validate_protocol(protocol):
    same_tree(protocol, CONTRACT, "literal typed event-cluster protocol")


def read_snapshot(root_path, name, expected):
    payload = source_path(root_path, name).read_bytes()
    if sha256(payload).hexdigest() != expected:
        raise ValueError("Pinned bytes changed before decoding: " + name)
    return payload


def read_json_snapshot(root_path, name, expected, *, object_required=True):
    return strict_json(
        read_snapshot(root_path, name, expected), object_required=object_required
    )


def pins_checked(root_path, pins):
    for name, expected in hash_mapping(pins).items():
        read_snapshot(root_path, name, expected)
    return len(pins)


def require_upstream_success(root_path):
    for study in ("range_alert", "issued_calibration"):
        path = Path(root_path) / f"reports/{study}/failure.json"
        if path.exists() or path.is_symlink():
            raise ValueError(
                "Canonical upstream failure blocks admission or verification commit: " + study
            )


def collect_closure(root_path, expected):
    """Shared metadata traversal, with separately checked literal anchors/bytes.

    Sharing this metadata-only admission implementation is explicit. It supplies
    no new feature, optimizer, prediction, score or inference calculations.
    """
    require_upstream_success(root_path)
    for name, signature in hash_mapping(expected).items():
        read_snapshot(root_path, name, signature)
    collected = shared_admission._collect(root_path, expected)
    proof = collected.audit
    same_tree(proof["anchors"], expected, "exact wave19/documentary anchors")
    if (
        proof["status"] != "PINNED_WAVE19_WITH_WAVE18_RECORDS"
        or proof["historical_models_refitted"] is not False
        or proof["source_values_reparsed"] is not False
    ):
        raise ValueError("Declared structural-only original admission required")
    pins_checked(root_path, proof["files"])
    record_name = "reports/issued_calibration/verification.json"
    record = read_json_snapshot(root_path, record_name, expected[record_name])
    if record.get("status") != "VERIFIED":
        raise ValueError("Original issued-calibration success identity required")
    same_tree(
        record["verified_output_hashes"],
        proof["verified_output_hashes"],
        "exact six prior verified output identities",
    )
    if record["inference"]["cumulative_hypotheses_verified"] != 131:
        raise ValueError("Original131 family identity required")
    require_upstream_success(root_path)
    return proof


def admit_upstream(root_path, expected, registered_pins):
    proof = collect_closure(root_path, expected)
    registered = hash_mapping(registered_pins)
    for name, signature in proof["files"].items():
        if registered.get(name) != signature:
            raise ValueError("Original closure missing from registered input pins: " + name)
    actual, loaded = shared_admission.admit_upstream(
        root_path, expected=expected, registered_pins=registered
    )
    same_tree(actual, proof, "unchanged shared metadata admission")
    pins_checked(root_path, proof["files"])
    require_upstream_success(root_path)
    return proof, loaded


def validate_panel(panel):
    if (
        not isinstance(panel, pd.DataFrame)
        or panel.empty
        or tuple(panel.columns) != PANEL_COLUMNS
        or panel.duplicated(["origin", "model"]).any()
        or set(panel.model) != set(MODELS)
    ):
        raise ValueError("Exact four-model complete panel schema required")
    shared = [c for c in PANEL_COLUMNS if c not in ("model", "probability", "loss")]
    reference = (
        panel.loc[panel.model == MODELS[0], shared]
        .sort_values("origin")
        .reset_index(drop=True)
    )
    for model in MODELS[1:]:
        actual = (
            panel.loc[panel.model == model, shared]
            .sort_values("origin")
            .reset_index(drop=True)
        )
        if not reference.equals(actual):
            raise ValueError("Exact cross-model labels, cohorts and timing required")
    for column in (
        "origin",
        "feature_cutoff_date",
        "target_end",
        "available_date",
        "fit_origin",
        "fit_cutoff_date",
        "train_last_target",
        "train_last_available",
    ):
        dates = pd.DatetimeIndex(panel[column])
        if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
            raise ValueError("Finite normalized source/calendar dates required")
    probabilities(panel.probability, len(panel), "issued panel probabilities")
    loss = real(panel.loss, "issued Brier loss")
    if (
        (loss < 0).any()
        or (loss > 1).any()
        or not np.array_equal(brier_loss(panel.probability, panel.y), loss)
    ):
        raise ValueError("Exact issued Brier score and domain required")
    if (
        not panel.horizon.eq(1).all()
        or not panel.feature_cutoff_date.lt(panel.origin).all()
        or not panel.target_end.gt(panel.origin).all()
        or not panel.target_end.eq(panel.available_date).all()
        or not panel.fit_cutoff_date.le(panel.feature_cutoff_date).all()
        or not panel.fit_origin.le(panel.origin).all()
        or not panel.train_last_target.le(panel.fit_cutoff_date).all()
        or not panel.train_last_available.le(panel.fit_cutoff_date).all()
    ):
        raise ValueError("Causal original source/maturity ordering required")


def verify_manifest_coverage(root_path, protocol, manifest, closure):
    root_path = Path(root_path)
    code = {
        str(path.relative_to(root_path))
        for folder in ("src", "tests")
        for path in (root_path / folder).rglob("*.py")
    }
    inputs = set(closure["files"]) | set(protocol["comparisons"]["inherited_sources"])
    name = REPORT + "/freeze_record.json"
    frozen = read_json_snapshot(root_path, name, manifest["inputs"][name])
    inputs.add(name)
    inputs.update(frozen["prefit_design"])
    same_tree(frozen["protocol_sha256"], manifest["protocol_sha256"], "new prefit protocol")
    same_tree(frozen["code"], manifest["code"], "new prefit entire code inventory")
    pins_checked(root_path, frozen["prefit_design"])
    read_snapshot(
        root_path, REPORT + "/full_repository_tests.txt", frozen["checks"]["full_log_sha256"]
    )
    preserved = {
        str(path.relative_to(root_path))
        for path in [*root_path.glob("*.yaml"), *(root_path / "reports").rglob("*")]
        if path.is_file()
        and path != root_path / "event_cluster.yaml"
        and root_path / REPORT not in path.parents
        and str(path.relative_to(root_path)) not in inputs
    }
    if (
        set(manifest["code"]) != code
        or set(manifest["inputs"]) != inputs
        or set(manifest["preserved"]) != preserved
        or inputs & preserved
    ):
        raise ValueError("Complete registered source/code/preservation inventory required")
    for name, signature in closure["files"].items():
        if manifest["inputs"].get(name) != signature:
            raise ValueError("Upstream closure differs from registered input pin: " + name)


def verify_pipeline(*args):
    # Runtime import avoids a cycle: the independent helper uses our primitives.
    from .event_cluster_verification import verify_pipeline as reconstruct

    return reconstruct(*args)


def verify(root=ROOT):
    root = Path(root)
    protocol_path = root / "event_cluster.yaml"
    payload = protocol_path.read_bytes()
    protocol_hash = sha256(payload).hexdigest()
    protocol = yaml.safe_load(payload)
    validate_protocol(protocol)
    report, out = root / REPORT, root / OUT
    manifest_payload = (report / "manifest.json").read_bytes()
    manifest_hash = sha256(manifest_payload).hexdigest()
    manifest = strict_json(manifest_payload)
    same_tree(manifest["protocol_sha256"], protocol_hash, "new registered protocol hash")
    for group in ("code", "inputs", "preserved"):
        pins_checked(root, manifest[group])
    closure = collect_closure(root, protocol["upstream"]["anchors"])
    verify_manifest_coverage(root, protocol, manifest, closure)
    output_names = [
        OUT + "/" + name
        for name in (
            "upstream_admission.json",
            "forecasts.parquet",
            "states.parquet",
            "memory.parquet",
            "fits.json",
            "support_audit.json",
        )
    ]
    output_names += [REPORT + "/metrics.json", REPORT + "/trial_ledger.jsonl"]
    # Every decoder consumes exactly the byte buffer whose hash we preserve.
    buffers = {name: source_path(root, name).read_bytes() for name in output_names}
    snapshots = {name: sha256(data).hexdigest() for name, data in buffers.items()}
    metrics = strict_json(buffers[REPORT + "/metrics.json"])
    if (
        (report / "failure.json").exists()
        or (report / "failure.json").is_symlink()
        or metrics.get("status") == "UNEVALUABLE"
        or metrics.get("whole_wave_aborted") is True
        or metrics["protocol_sha256"] != protocol_hash
        or metrics["evidence_class"] != protocol["evidence_class"]
    ):
        raise ValueError("Only an intact new scored publication may be verified")
    proof, loaded = admit_upstream(root, protocol["upstream"]["anchors"], manifest["inputs"])
    same_tree(
        strict_json(buffers[OUT + "/upstream_admission.json"]),
        proof,
        "saved complete upstream admission",
    )
    same_tree(
        protocol["original_index"],
        loaded["protocol"]["index"],
        "unchanged original source/index contract",
    )
    panel, states, memory = [
        pd.read_parquet(io.BytesIO(buffers[OUT + "/" + name + ".parquet"]))
        for name in ("forecasts", "states", "memory")
    ]
    fits = strict_json(buffers[OUT + "/fits.json"], object_required=False)
    if type(fits) is not list:
        raise ValueError("Original scheduled fit list required")
    support_audit = strict_json(buffers[OUT + "/support_audit.json"])
    checked = verify_pipeline(
        loaded["features"],
        loaded["targets"],
        loaded["forecasts"],
        loaded["fits"],
        loaded["states"],
        protocol["original_index"],
        panel,
        fits,
        states,
        memory,
        support_audit,
    )
    result = {
        "status": "VERIFIED",
        "protocol_sha256": protocol_hash,
        "verifier_sha256": digest(Path(__file__)),
        "upstream_admission": {
            "status": proof["status"],
            "historical_models_refitted": False,
            "source_values_reparsed": False,
            "metadata_implementation": "shared checked source traversal with independent anchor and full-file rehashes",
        },
        "forecast_reconstruction": checked,
        "primitive_brier_scores_verified": len(panel),
        "paired_brier_differences_verified": 3 * checked["common_scored_origins"],
        "inference": verify_metrics(
            root,
            panel,
            protocol,
            metrics,
            len(states),
            len(fits),
            admitted_inputs=manifest["inputs"],
        ),
        "ledger_events_verified": verify_ledger(
            root,
            metrics,
            metrics["inherited_rows"],
            "evaluated",
            signature=snapshots[REPORT + "/trial_ledger.jsonl"],
        ),
        "verified_output_hashes": snapshots,
        "artifact_hashes_checked": {
            **{group: len(manifest[group]) for group in ("code", "inputs", "preserved")},
            "outputs": len(snapshots),
        },
        "limitations": [
            "Model-relative adjacency of mature archived SPX risk-alert labels; no claim of conditional orthogonality, a new exogenous channel or causal regime discovery.",
            "Overlapping daily GK-plus-overnight threshold references can themselves induce event dependence. The target is an archival proxy, not high-frequency realized variance.",
            "The original monthly baseline is replayed on its own mature training rows for current-fit in-sample offsets; historical query controls and cohorts remain unchanged.",
            "Centered raw adjacency alone enters the candidate. Conditional-permutation excess adjacency remains descriptive; nuisance coefficients are frozen for the scalar stage.",
            "Every original training/application history must be complete. Unknown labels cannot be filled, compressed, dropped or used to alter scheduled fits.",
            "Selection followed repeated inspection of reused historical studies; passing all three Brier gates is neither untouched validation nor trading profitability.",
        ],
    }
    for group in ("code", "inputs", "preserved"):
        pins_checked(root, manifest[group])
    pins_checked(root, snapshots)
    verify_manifest_coverage(root, protocol, manifest, closure)
    if (
        digest(protocol_path) != protocol_hash
        or digest(report / "manifest.json") != manifest_hash
    ):
        raise ValueError("Protocol/manifest changed during independent verification")
    require_upstream_success(root)
    if (report / "failure.json").exists() or (report / "failure.json").is_symlink():
        raise ValueError("New failure marker appeared before verification commit")
    (report / "verification.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def invalidate_publication(root, error):
    report = Path(root) / REPORT
    report.mkdir(parents=True, exist_ok=True)
    path = report / "metrics.json"
    original_bytes = path.read_bytes() if path.exists() else None
    prior, invalid = None, None
    if original_bytes is not None:
        try:
            prior = strict_json(original_bytes)
        except (UnicodeDecodeError, ValueError):
            invalid = original_bytes
    protocol_hash = prior.get("protocol_sha256") if prior else None
    message = "Independent verification failed: " + type(error).__name__ + ": " + str(error)
    failure = {
        "status": "UNEVALUABLE",
        "whole_wave_aborted": True,
        "leads": [],
        "hypothesis_count": 3,
        "cumulative_hypothesis_count": 134,
        "protocol_sha256": protocol_hash,
        "rows": [
            {
                "study": "event_cluster",
                "candidate": candidate,
                "control": control,
                "score": "brier",
                "horizon": 1,
                "phases": [],
                "p_conservative": 1.0,
                "p_holm_wave": 1.0,
                "p_holm_cumulative": 1.0,
                "status": "INVALID_RUN",
                "verdict": "UNEVALUABLE",
                "error": message,
            }
            for candidate, control in COMPARISONS
        ],
    }
    payload = json.dumps(failure, indent=2, allow_nan=False) + "\n"
    path.write_text(payload)
    (report / "failure.json").write_text(payload)
    (report / "verification.json").write_text(
        json.dumps(
            {"status": "FAILED", "error": message, "protocol_sha256": protocol_hash}, indent=2
        )
        + "\n"
    )
    (report / "results.md").write_text(
        "# SPX event-adjacency risk-alert forecasts\n\nUNEVALUABLE: independent verification failed. All three comparisons retained with p=1; no lead.\n\n"
        + message
        + "\n"
    )
    # Guaranteed canonical family/ledger invalidation precedes optional backups.
    ledger_path = report / "trial_ledger.jsonl"
    old_ledger = ledger_path.read_bytes() if ledger_path.exists() else b""
    with ledger_path.open("a") as stream:
        if old_ledger and not old_ledger.endswith(b"\n"):
            stream.write("\n")
        for row in failure["rows"]:
            stream.write(
                json.dumps(
                    {"event": "verification_failed", **row}, sort_keys=True, allow_nan=False
                )
                + "\n"
            )
    backup = report / "unpublished_scored_metrics.json"
    if prior is not None and prior.get("status") != "UNEVALUABLE" and not backup.exists():
        backup.write_text(
            json.dumps(
                {
                    "status": "UNPUBLISHED_DIAGNOSTIC_ONLY",
                    "not_for_inherited_inference_or_promotion": True,
                    "scored_metrics": prior,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
    invalid_backup = report / "unpublished_invalid_metrics.txt"
    if invalid is not None and not invalid_backup.exists():
        invalid_backup.write_bytes(invalid)
    return failure


def verify_with_failure_guard(root=ROOT):
    try:
        return verify(root)
    except Exception as error:
        invalidate_publication(root, error)
        raise


if __name__ == "__main__":
    print(json.dumps(verify_with_failure_guard(), indent=2, allow_nan=False))
