"""Independent strict-sign memory verification; prior verifiers are read-only.

No wave13 feature, model, scoring or runner implementation is imported here.
"""

from __future__ import annotations

import json
import tempfile
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import brentq, minimize
from scipy.special import expit

from . import verify_joint_risk as original
from . import verify_target_aligned as previous
from .verify_cross_moment import (
    _finite,
    compare_tree,
    digest,
    explicit_bootstrap_means,
    holm,
    independent_hac,
    paired_difference,
    same_tree,
)

ROOT = Path(__file__).resolve().parents[1]
OLD_FEATURES = original.ALL_FEATURES
BOUNDED = ("qqq_pos22", "qqq_neg22", "spx_pos22", "spx_neg22", "independent22")
MEMORY = "excess22"
ALL_FEATURES = OLD_FEATURES + BOUNDED + (MEMORY,)
EMPIRICAL = OLD_FEATURES[1:] + ("corr22_centered_sq",)
BASE_COLUMNS = ("const",) + EMPIRICAL + BOUNDED
MODELS = ("frequency", "baseline", "memory")
COMPARISONS = (("memory", "baseline"), ("memory", "frequency"))
WAVE_ALPHA = 0.05 / (13 * 14)
EFFECT = 0.0005
GRADIENT_TOLERANCE = 1e-8 + 1e-12
PANEL_COLUMNS = (
    "origin",
    "model",
    "horizon",
    "feature_cutoff_date",
    "target_end",
    "available_date",
    "y",
    "probability",
    "loss",
    "fit_origin",
    "fit_cutoff_date",
    "train_n",
    "train_last_target",
    "train_last_available",
    "phase",
)


def real(value, *, missing=False):
    if np.iscomplexobj(value):
        raise ValueError("Real input required")
    result = np.asarray(value, dtype=float)
    if np.isinf(result).any() or (not missing and np.isnan(result).any()):
        raise ValueError("Finite input required")
    return result


def binary(value):
    result = real(value)
    if result.ndim != 1 or not len(result) or not np.isin(result, (0.0, 1.0)).all():
        raise ValueError("Nonempty one-dimensional binary labels required")
    return result


def strict_agreement(qqq, spx):
    q, s = real(qqq, missing=True), real(spx, missing=True)
    if q.ndim != 1 or q.shape != s.shape:
        raise ValueError("Aligned one-dimensional paired returns required")
    value = (((q > 0) & (s > 0)) | ((q < 0) & (s < 0))).astype(float)
    value[np.isnan(q) | np.isnan(s)] = np.nan
    return value


def sign_history(qqq, spx):
    if not qqq.index.equals(spx.index):
        raise ValueError("Identical full reference calendars required")
    agreement = strict_agreement(qqq, spx)
    q, s = qqq.to_numpy(), spx.to_numpy()
    values = np.c_[q > 0, q < 0, s > 0, s < 0, agreement].astype(float)
    values[np.isnan(q) | np.isnan(s)] = np.nan
    fractions = (
        pd.DataFrame(values, index=qqq.index).rolling(22, min_periods=22).mean().shift()
    )
    out = pd.DataFrame(index=qqq.index)
    for number, name in enumerate(BOUNDED[:4]):
        out[name] = fractions[number]
    out["independent22"] = fractions[0] * fractions[2] + fractions[1] * fractions[3]
    out[MEMORY] = fractions[4] - out["independent22"]
    return out


def feature_target_tables(qqq, spx, iv):
    old, _ = original.feature_target_tables(qqq, spx, iv)
    q, s = qqq.reindex(spx.index), spx
    qday, sday = np.log(q.close / q.open), np.log(s.close / s.open)
    history = sign_history(qday, sday)
    features = pd.concat(
        [old.loc[:, OLD_FEATURES], history, old[["feature_cutoff_date"]]], axis=1
    )
    future = pd.Series(strict_agreement(qday, sday), index=spx.index).shift(-1)
    maturity = pd.Series(spx.index, index=spx.index).shift(-1)
    return features, pd.DataFrame(
        {"y": future, "target_end": maturity, "available_date": maturity}
    )


def logistic_objective(beta, x, y):
    beta, x, y = real(beta), real(x), binary(y)
    if x.ndim != 2 or beta.shape != (x.shape[1],) or len(x) != len(y):
        raise ValueError("Aligned logistic geometry required")
    eta = real(x @ beta)
    signed = (1 - 2 * y) * eta
    terms = np.logaddexp(0.0, signed)
    score = np.where(y == 0, expit(eta), -expit(-eta))
    curvature = expit(eta) * expit(-eta)
    penalty = beta.copy()
    penalty[0] = 0.0
    objective = float(terms.mean() + 0.01 * (penalty @ penalty))
    gradient = x.T @ score / len(y) + 0.02 * penalty
    hessian = x.T @ (curvature[:, None] * x) / len(y)
    hessian += np.diag(np.r_[0.0, np.full(len(beta) - 1, 0.02)])
    real([objective])
    real(gradient)
    real(hessian)
    return objective, gradient, hessian


def independent_baseline_fit(x, y):
    x, y = real(x), binary(y)
    if not 0 < y.mean() < 1:
        raise ValueError("Both classes needed for finite intercept optimum")
    result = minimize(
        lambda b: logistic_objective(b, x, y)[0],
        np.zeros(x.shape[1]),
        jac=lambda b: logistic_objective(b, x, y)[1],
        hess=lambda b: logistic_objective(b, x, y)[2],
        method="trust-exact",
        options={"gtol": 1e-10, "maxiter": 500},
    )
    value, gradient, _ = logistic_objective(result.x, x, y)
    if np.max(abs(gradient)) > GRADIENT_TOLERANCE:
        raise ValueError("Independent baseline stationary-gradient tolerance failed")
    return {
        "beta": real(result.x),
        "objective": value,
        "gradient_max_abs": float(np.max(abs(gradient))),
        "success": bool(result.success),
        "status": int(result.status),
        "iterations": int(result.nit),
    }


def memory_objective(b, eta, y, memory):
    b = float(real(b))
    eta, y, memory = real(eta), binary(y), real(memory)
    if eta.ndim != 1 or eta.shape != y.shape or memory.shape != y.shape:
        raise ValueError("Aligned frozen logits and centered memory required")
    prediction = real(eta + b * memory)
    score = np.where(y == 0, expit(prediction), -expit(-prediction))
    value = float(np.logaddexp(0.0, (1 - 2 * y) * prediction).mean() + 0.01 * b * b)
    gradient = float(np.mean(memory * score) + 0.02 * b)
    curvature = float(np.mean(memory * memory * expit(prediction) * expit(-prediction)) + 0.02)
    real([value, gradient, curvature])
    return value, gradient, curvature


def independent_memory_fit(eta, y, memory):
    eta, y, memory = real(eta), binary(y), real(memory)
    radius = float((np.mean(abs(memory)) + 1.0) / 0.02)
    if (memory == 0).all():
        coefficient = 0.0
        iterations = 0
    else:
        coefficient, result = brentq(
            lambda b: memory_objective(b, eta, y, memory)[1],
            -radius,
            radius,
            xtol=1e-12,
            rtol=1e-14,
            maxiter=200,
            full_output=True,
        )
        if not result.converged:
            raise ValueError("Independent scalar root did not converge")
        iterations = int(result.iterations)
    value, gradient, _ = memory_objective(coefficient, eta, y, memory)
    if abs(gradient) > GRADIENT_TOLERANCE:
        raise ValueError("Independent scalar stationary-gradient tolerance failed")
    return {
        "b": float(coefficient),
        "objective": value,
        "gradient_max_abs": abs(gradient),
        "bracket": [-radius, radius],
        "iterations": iterations,
    }


def transform(train, application):
    if len(train) < 2 or not train.const.eq(1).all() or not application.const.eq(1).all():
        raise ValueError("Unit intercept and nonempty training geometry required")
    real(train.loc[:, ALL_FEATURES])
    real(application.loc[:, ALL_FEATURES])
    center = float(train.corr22.mean())
    frames = []
    for source in (train, application):
        part = source.loc[:, OLD_FEATURES[1:]].copy()
        part["corr22_centered_sq"] = (source.corr22 - center) ** 2
        frames.append(part.to_numpy())
    means = frames[0].mean(axis=0)
    scales = frames[0].std(axis=0, ddof=0)
    if not np.isfinite(scales).all() or (scales <= 1e-12).any():
        raise ValueError("INSUFFICIENT_DATA: positive empirical population scale required")
    bounded = train.loc[:, BOUNDED].to_numpy()
    constant = (bounded == bounded[0]).all(axis=0)
    bmean = np.where(constant, bounded[0], bounded.mean(axis=0))
    memory = train[MEMORY].to_numpy()
    flat = bool((memory == memory[0]).all())
    mmean = float(memory[0] if flat else memory.mean())
    tr = np.c_[np.ones(len(train)), (frames[0] - means) / scales, bounded - bmean]
    ap = np.c_[
        np.ones(len(application)),
        (frames[1] - means) / scales,
        application.loc[:, BOUNDED].to_numpy() - bmean,
    ]
    tm = memory - mmean
    am = application[MEMORY].to_numpy() - mmean
    for value in (tr, ap, tm, am):
        real(value)
    audit = {
        "columns": list(BASE_COLUMNS),
        "empirical_columns": list(EMPIRICAL),
        "corr22_mean": center,
        "empirical_means": means.tolist(),
        "empirical_scales": scales.tolist(),
        "bounded_columns": list(BOUNDED),
        "bounded_means": bmean.tolist(),
        "bounded_constant": constant.tolist(),
        "bounded_scale": 1.0,
        "memory_mean": mmean,
        "memory_constant": flat,
        "memory_scale": 1.0,
    }
    return tr, ap, tm, am, audit


def eligible_entries(features, targets, section):
    if not features.index.equals(targets.index):
        raise AssertionError("Identical full calendars required")
    complete = (
        np.isfinite(features.loc[:, ALL_FEATURES]).all(axis=1)
        & features.feature_cutoff_date.notna()
    )
    permitted = pd.Series(False, index=features.index)
    for name in ("development", "evaluation"):
        first, last = section[name]
        permitted |= (features.index >= first) & (features.index <= last)
    application = features.index[
        complete
        & permitted
        & (features.index >= section["origin_start"])
        & (features.index <= section["origin_end"])
    ]
    known = (
        targets.y.isin([0.0, 1.0])
        & targets.available_date.notna()
        & (targets.available_date <= section["latest_target"])
    )
    known &= (features.index > section["development"][1]) | (
        targets.available_date <= section["development_target_available_by"]
    )
    return application, application[known.loc[application]]


def training_mask(features, targets, entry):
    if not features.index.equals(targets.index):
        raise AssertionError("Identical full calendars required")
    position = features.index.get_loc(entry)
    if position < 1:
        raise ValueError("Predecessor session required")
    return (
        np.isfinite(features.loc[:, ALL_FEATURES]).all(axis=1)
        & features.feature_cutoff_date.notna()
        & targets.y.isin([0.0, 1.0])
        & (features.index < entry)
        & (targets.available_date <= features.index[position - 1])
    )


def close(actual, expected, label, rtol=1e-10, atol=1e-12):
    a, e = real(actual), real(expected)
    if a.shape != e.shape or not np.allclose(a, e, rtol=rtol, atol=atol):
        raise AssertionError(label)


def support(y, minimum):
    values = binary(y)
    events = int((values == 1).sum())
    nonevents = len(values) - events
    if min(events, nonevents) < minimum:
        raise ValueError("INSUFFICIENT_DATA: fixed class support")
    return {"n": len(values), "events": events, "nonevents": nonevents}


def verify_fit(train, targets, app, audit):
    x, query, m, querym, geometry = transform(train, app)
    y = binary(targets.y)
    expected_support = support(y, 50)
    means = np.r_[0.0, geometry["empirical_means"], geometry["bounded_means"]]
    scales = np.r_[1.0, geometry["empirical_scales"], np.ones(len(BOUNDED))]
    expected_transform = {
        "columns": list(BASE_COLUMNS),
        "means": means.tolist(),
        "scales": scales.tolist(),
        "corr22_mean": geometry["corr22_mean"],
        "bounded_constant_columns": [
            name
            for name, constant in zip(BOUNDED, geometry["bounded_constant"], strict=True)
            if constant
        ],
    }
    same_tree(audit["transform"], expected_transform, "Independent training-only geometry")
    if (
        audit["train_n"] != len(train)
        or audit["application_n"] != len(app)
        or audit["support"] != expected_support
    ):
        raise AssertionError("Exact common mature train/support/application audit differs")
    frequency = float(y.mean())
    same_tree(
        audit["frequency"],
        {"probability": frequency, "train_n": len(y), "application_n": len(app)},
        "Same-sample frequency control",
    )
    base, mem = audit["baseline"], audit["memory"]
    for one in (base, mem):
        if (
            one["alpha"] != 0.01
            or one["train_n"] != len(y)
            or one["application_n"] != len(app)
            or one["success"] is not True
            or not isinstance(one["iterations"], int)
            or not 0 <= one["iterations"] <= 200
            or not isinstance(one["backtracks"], int)
            or not 0 <= one["backtracks"] <= 59 * one["iterations"]
            or not 0 <= float(real(one["gradient_max_abs"])) <= 1e-8
        ):
            raise AssertionError("Saved fixed numerical fit contract differs")
    if base["columns"] != list(BASE_COLUMNS):
        raise AssertionError("Declared baseline coefficient order differs")
    close(base["means"], means, "Saved baseline means")
    close(base["scales"], scales, "Saved baseline scales")
    start = np.zeros(len(BASE_COLUMNS))
    start[0] = np.log(frequency / (1 - frequency))
    close(base["start"], start, "Fixed producer initialization", rtol=0, atol=0)
    beta = real(base["beta"])
    value, g, _ = logistic_objective(beta, x, y)
    close(base["objective"], value, "Saved normalized Bernoulli objective")
    close(
        base["gradient"],
        g,
        "Saved independently recomputed full gradient",
        rtol=1e-7,
        atol=1e-12,
    )
    close(
        base["gradient_max_abs"], np.max(abs(g)), "Saved gradient norm", rtol=1e-7, atol=1e-12
    )
    if np.max(abs(g)) > GRADIENT_TOLERANCE:
        raise AssertionError("Saved baseline is not stationary")
    for column in expected_transform["bounded_constant_columns"]:
        if beta[BASE_COLUMNS.index(column)] != 0:
            raise AssertionError(
                "Exact constant bounded input coefficient must remain canonical zero"
            )
    independent = independent_baseline_fit(x, y)
    close(
        beta, independent["beta"], "Independent convex baseline optimum", rtol=1e-7, atol=1e-6
    )
    eta, appeta = real(x @ beta), real(query @ beta)
    independent_p = expit(query @ independent["beta"])
    close(
        expit(appeta), independent_p, "Independent baseline probability", rtol=1e-7, atol=1e-6
    )
    if (
        mem["column"] != MEMORY
        or mem["scale"] != 1.0
        or mem["baseline_frozen"] is not True
        or mem["status"]
        != ("EXACT_CONSTANT_INPUT" if geometry["memory_constant"] else "FITTED")
    ):
        raise AssertionError("Frozen baseline offset or constant-memory convention differs")
    close(mem["mean"], geometry["memory_mean"], "Training-only memory mean")
    close(mem["start"], [0.0], "Fixed scalar initialization", rtol=0, atol=0)
    b = float(real(mem["b"]))
    if geometry["memory_constant"] and b != 0:
        raise AssertionError("Exact constant memory coefficient must be zero")
    mvalue, mg, _ = memory_objective(b, eta, y, m)
    close(mem["objective"], mvalue, "Saved scalar normalized Bernoulli objective")
    close(mem["gradient"], [mg], "Independent saved scalar gradient", rtol=1e-7, atol=1e-12)
    close(
        mem["gradient_max_abs"], abs(mg), "Saved scalar gradient norm", rtol=1e-7, atol=1e-12
    )
    if abs(mg) > GRADIENT_TOLERANCE:
        raise AssertionError("Saved scalar memory is not stationary")
    scalar = independent_memory_fit(eta, y, m)
    close(b, scalar["b"], "Independent monotone scalar root", rtol=1e-7, atol=1e-6)
    issued = expit(real(appeta + b * querym))
    close(
        issued,
        expit(real(appeta + scalar["b"] * querym)),
        "Independent scalar probability",
        rtol=1e-7,
        atol=1e-6,
    )
    # Replay with the saved geometry as well, so audit rounding never changes
    # the authoritative emitted probability used for the proper score.
    raw = app.loc[:, OLD_FEATURES].copy()
    raw["corr22_centered_sq"] = (app.corr22 - audit["transform"]["corr22_mean"]) ** 2
    for column in BOUNDED:
        raw[column] = app[column]
    saved_query = (raw.loc[:, BASE_COLUMNS].to_numpy() - real(base["means"])) / real(
        base["scales"]
    )
    saved_eta = real(saved_query @ beta)
    saved_m = app[MEMORY].to_numpy() - mem["mean"]
    probabilities = {
        "frequency": np.full(len(app), frequency),
        "baseline": expit(saved_eta),
        "memory": expit(real(saved_eta + b * saved_m)),
    }
    close(probabilities["baseline"], expit(appeta), "Independent geometry probability replay")
    close(probabilities["memory"], issued, "Independent memory geometry replay")
    for values in probabilities.values():
        real(values)
        if ((values < 0) | (values > 1)).any():
            raise AssertionError(
                "Finite application probabilities required even without labels"
            )
    return probabilities, {
        "constant_memory": geometry["memory_constant"],
        "baseline_status": independent["status"],
        "independent_baseline_gradient": independent["gradient_max_abs"],
        "independent_memory_gradient": scalar["gradient_max_abs"],
    }


def verify_forecasts(features, targets, panel, fits, protocol):
    if (
        not features.index.equals(targets.index)
        or not features.index.is_unique
        or not features.index.is_monotonic_increasing
        or tuple(features.columns) != ALL_FEATURES + ("feature_cutoff_date",)
        or tuple(targets.columns) != ("y", "target_end", "available_date")
    ):
        raise AssertionError("Exact full source/target schema and ordered dates required")
    dates = features.index
    series = pd.Series(dates, index=dates)
    for frame, column, shift in (
        (features, "feature_cutoff_date", 1),
        (targets, "target_end", -1),
        (targets, "available_date", -1),
    ):
        if not frame[column].equals(series.shift(shift).rename(column)):
            raise AssertionError("Full-calendar one-session timing differs")
    if targets.y.notna().any():
        binary(targets.loc[targets.y.notna(), "y"])
    if (
        tuple(panel.columns) != PANEL_COLUMNS
        or panel.empty
        or panel.duplicated(["origin", "model"]).any()
        or set(panel.model) != set(MODELS)
    ):
        raise AssertionError("Three complete model cohorts and exact panel schema required")
    validate_panel(panel)
    section = protocol["index"]
    applications, scored = eligible_entries(features, targets, section)
    groups = {
        name: panel.loc[panel.model == name].set_index("origin").sort_index()
        for name in MODELS
    }
    if any(not frame.index.equals(scored) for frame in groups.values()):
        raise AssertionError("Full scoreable common cohort differs")
    months = applications.to_period("M").unique()
    if len(fits) != len(months):
        raise AssertionError(
            "Every monthly application fit required, including unscored months"
        )
    flat = 0
    trainrows = 0
    maxbase = 0.0
    maxmemory = 0.0
    for number, month in enumerate(months):
        app = applications[applications.to_period("M") == month]
        entry = app[0]
        mask = training_mask(features, targets, entry)
        n = int(mask.sum())
        if n < section["minimum_train"]:
            raise AssertionError("Fixed minimum mature sample failed")
        cutoff = dates[dates.get_loc(entry) - 1]
        lasttarget = targets.loc[mask, "target_end"].max()
        lastavailable = targets.loc[mask, "available_date"].max()
        metadata = {
            "fit_origin": str(entry.date()),
            "fit_cutoff_date": str(cutoff.date()),
            "train_n": n,
            "train_first_origin": str(dates[mask][0].date()),
            "train_last_origin": str(dates[mask][-1].date()),
            "train_last_target": str(lasttarget.date()),
            "train_last_available": str(lastavailable.date()),
            "application_n": len(app),
        }
        saved = fits[number]
        if {k: v for k, v in saved.items() if k != "model_audit"} != metadata:
            raise AssertionError("Monthly fitting metadata differs")
        probabilities, checked = verify_fit(
            features.loc[mask], targets.loc[mask], features.loc[app], saved["model_audit"]
        )
        flat += int(checked["constant_memory"])
        trainrows += n
        maxbase = max(maxbase, checked["independent_baseline_gradient"])
        maxmemory = max(maxmemory, checked["independent_memory_gradient"])
        keep = app.isin(scored)
        chosen = app[keep]
        for name in MODELS:
            frame = groups[name].loc[chosen]
            expected = {
                "horizon": np.ones(len(chosen), int),
                "feature_cutoff_date": features.loc[chosen, "feature_cutoff_date"].to_numpy(),
                "target_end": targets.loc[chosen, "target_end"].to_numpy(),
                "available_date": targets.loc[chosen, "available_date"].to_numpy(),
                "y": targets.loc[chosen, "y"].to_numpy(),
                "fit_origin": np.full(len(chosen), entry.to_datetime64()),
                "fit_cutoff_date": np.full(len(chosen), cutoff.to_datetime64()),
                "train_n": np.full(len(chosen), n),
                "train_last_target": np.full(len(chosen), lasttarget.to_datetime64()),
                "train_last_available": np.full(len(chosen), lastavailable.to_datetime64()),
                "phase": np.where(
                    chosen <= pd.Timestamp(section["development"][1]),
                    "development",
                    "evaluation",
                ),
            }
            for column, values in expected.items():
                if not np.array_equal(frame[column].to_numpy(), values):
                    raise AssertionError(
                        "Forecast " + column + " differs from full calendar/mature sample"
                    )
            close(
                frame.probability,
                probabilities[name][keep],
                "Saved-coefficient issued probability replay",
            )
            real(frame.loss)
            actual = real(frame.y) - real(frame.probability)
            loss = np.array(
                [previous.previous.checked_product(value, value) for value in actual]
            )
            if not np.array_equal(loss, frame.loss.to_numpy()):
                raise AssertionError("Exact issued Brier score differs")
    return {
        "monthly_fits_verified": len(fits),
        "independent_convex_fits_verified": 2 * len(fits),
        "forecasts_verified": len(panel),
        "common_scored_origins": len(scored),
        "common_application_origins": len(applications),
        "constant_memory_fits_retained": flat,
        "training_rows_reconstructed": trainrows,
        "maximum_independent_baseline_gradient": maxbase,
        "maximum_independent_memory_gradient": maxmemory,
    }


def brier_loss(probability, y):
    p, y = real(probability), binary(y)
    if p.shape != y.shape or ((p < 0) | (p > 1)).any():
        raise ValueError("Aligned valid probabilities required")
    residual = p - y
    return np.array([previous.previous.checked_product(value, value) for value in residual])


def validate_panel(panel):
    if (
        panel.empty
        or tuple(panel.columns) != PANEL_COLUMNS
        or panel.duplicated(["origin", "model"]).any()
        or set(panel.model) != set(MODELS)
    ):
        raise AssertionError("Three full model cohorts and exact schema required")
    shared = [
        column for column in PANEL_COLUMNS if column not in ("model", "probability", "loss")
    ]
    first = (
        panel.loc[panel.model == MODELS[0], shared]
        .sort_values("origin")
        .reset_index(drop=True)
    )
    for name in MODELS[1:]:
        if not first.equals(
            panel.loc[panel.model == name, shared].sort_values("origin").reset_index(drop=True)
        ):
            raise AssertionError("Exact cross-model metadata and labels required")
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
            raise AssertionError("Normalized finite dates required")
    if not np.array_equal(brier_loss(panel.probability, panel.y), real(panel.loss)):
        raise AssertionError("Exact issued Brier score required")


def normalized_difference(difference):
    values = real(difference)
    if values.ndim != 1 or not len(values):
        raise ValueError("Nonempty paired differences required")
    center = float(values.mean())
    deviation = values - center
    norm = float(deviation @ deviation)
    if not np.isfinite([center, norm]).all() or (
        not (values == values[0]).all() and norm == 0
    ):
        raise ValueError("Nonfinite or underflowed centered difference norm")
    return values


def verify_metrics(root, panel, protocol, metrics, monthly_fits):
    validate_panel(panel)
    if (
        metrics["new_monthly_fits"] != monthly_fits
        or metrics["new_forecasts"] != len(panel)
        or metrics["common_scored_origins"] != panel.origin.nunique()
    ):
        raise AssertionError("New full fit/forecast cohort counts differ")
    section = protocol["index"]
    counts = {}
    calibration = {}
    phase_union = np.zeros(len(panel), dtype=bool)
    for name in ("development", "evaluation"):
        first, last = section[name]
        mask = panel.origin.between(first, last)
        phase_union |= mask
        if not panel.loc[mask, "phase"].eq(name).all():
            raise AssertionError("Literal phase membership differs")
        base = panel.loc[mask & panel.model.eq("frequency")]
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
        values = panel.loc[panel.origin.between(start, end) & panel.model.eq("frequency"), "y"]
        counts["evaluation_slices"].append({"start": start, "end": end, **support(values, 15)})
    same_tree(metrics["class_support"], counts, "Declared phase and slice class support")
    compare_tree(metrics["calibration"], calibration, "Descriptive calibration in the large")
    rows = metrics["rows"]
    if [(row["candidate"], row["control"]) for row in rows] != list(COMPARISONS):
        raise AssertionError("Both fixed Brier controls required")
    probabilities = []
    effects = []
    for row in rows:
        if row["study"] != "sign_memory" or row["horizon"] != 1 or row["score"] != "brier":
            raise AssertionError("Strict-sign Brier comparison identity differs")
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
    prior = inherited_rows(root, protocol)
    same_tree(metrics["inherited_rows"], prior, "All inherited comparisons retained")
    wave = holm(probabilities)
    cumulative = holm([row["p_conservative"] for row in prior] + probabilities)[-2:]
    passed = []
    for number, row in enumerate(rows):
        compare_tree(row["p_holm_wave"], wave[number], "Two-hypothesis wave correction", "p")
        compare_tree(
            row["p_holm_cumulative"],
            cumulative[number],
            "119-hypothesis cumulative correction",
            "p",
        )
        one = bool(effects[number] and wave[number] < WAVE_ALPHA and cumulative[number] < 0.05)
        if row["verdict"] != ("PASSES_ALL_GATES" if one else "DOES_NOT_QUALIFY"):
            raise AssertionError("Fixed statistical/effect/stability gate differs")
        passed.append(one)
    leads = ["memory"] if all(passed) else []
    if (
        metrics["leads"] != leads
        or metrics["hypothesis_count"] != 2
        or metrics["cumulative_hypothesis_count"] != 119
    ):
        raise AssertionError("Complete candidate and hypothesis family required")
    return {
        "new_hypotheses_verified": 2,
        "cumulative_hypotheses_verified": 119,
        "phase_comparisons_verified": 4,
        "bootstrap_runs_verified": 12,
        "bootstrap_draws_per_run": protocol["inference"]["bootstrap_draws"],
        "calibration_rows_verified": 6,
        "class_support": counts,
        "leads": leads,
    }


PREVIOUS_LIMITATIONS = [
    "Two new scalar fits per original monthly fold; shared marginal coefficients replayed without refitting and all original issued rows preserved.",
    "Current-fit training residuals are in-sample at the monthly fit cutoff; only their conditional expected product equals covariance plus mean-error product.",
    "The staged bounded affine model changes both target alignment and dependence functional form, so success cannot isolate a loss-function change or establish true correlation dynamics.",
    "Reused archival QQQ ETF/SPX index daily products, no measured high-frequency covariance or trading-profit claim; nominal MDE is diagnostic and nonqualification is not equivalence.",
]


def reconstruct_previous(root, info, pins):
    oldp = yaml.safe_load((root / info["protocol"]).read_text())
    previous.validate_protocol(oldp)
    oldreport, olddata = root / info["reports"], root / info["data"]
    nested = previous.validate_upstream(root)
    same_tree(
        json.loads((olddata / "upstream_admission.json").read_text()),
        nested,
        "Original read-only admission proof",
    )
    data = {}
    for key in ("features", "targets", "forecasts"):
        name = oldp["upstream"][key]
        data[key] = previous.previous.read_issued_snapshot(root, name, pins[name])
    name = oldp["upstream"]["fits"]
    oldfits = previous.read_json_snapshot(root, name, pins[name])
    name = str((olddata / "forecasts.parquet").relative_to(root))
    panel = previous.previous.read_issued_snapshot(root, name, pins[name])
    name = str((olddata / "fits.json").relative_to(root))
    fits = previous.read_json_snapshot(root, name, pins[name])
    reconstruction = previous.verify_forecasts(
        data["features"], data["targets"], data["forecasts"], oldfits, panel, fits, oldp
    )
    name = str((oldreport / "metrics.json").relative_to(root))
    metrics = previous.read_json_snapshot(root, name, pins[name])
    if (
        metrics.get("status") == "UNEVALUABLE"
        or metrics["protocol_sha256"] != info["protocol_sha256"]
        or metrics["evidence_class"] != oldp["evidence_class"]
    ):
        raise AssertionError("Original publication identity or success differs")
    counts = oldp["fitting"]
    for actual, key in (
        (len(panel), "combined_forecast_rows_expected"),
        (len(fits), "new_monthly_fits_expected"),
        (2 * len(fits), "new_scalar_fits_expected"),
        (len(panel) - len(data["forecasts"]), "new_forecasts_expected"),
    ):
        if actual != counts[key]:
            raise AssertionError("Original full registered fit/forecast cohort required")
    result = {
        "status": "VERIFIED",
        "upstream_admission": {"status": nested["status"], "prior_files_written": False},
        "forecast_reconstruction": reconstruction,
        "primitive_score_cells_verified": len(panel) * 3,
        "paired_product_differences_verified": int(panel.origin.nunique()) * 3,
        "inference": previous.verify_metrics(root, previous.score_panel(panel), oldp, metrics),
        "ledger_events_verified": previous.verify_ledger(
            root, metrics, metrics["inherited_rows"], "evaluated"
        ),
        "protocol_sha256": info["protocol_sha256"],
        "verifier_sha256": info["verifier_sha256"],
        "limitations": PREVIOUS_LIMITATIONS,
    }
    return result, nested


def validate_upstream(root=ROOT):
    root = Path(root)
    protocol_path = root / "sign_memory.yaml"
    protocol = yaml.safe_load(protocol_path.read_text())
    info = protocol["upstream"]
    captured = json.loads((root / "reports/sign_memory/manifest.json").read_text())
    if digest(protocol_path) != captured["protocol_sha256"]:
        raise AssertionError("New manifest must precede upstream admission")
    oldreport, olddata = root / info["reports"], root / info["data"]

    def inventory():
        return {
            str(path.relative_to(root))
            for folder in (oldreport, olddata)
            for path in folder.rglob("*")
            if path.is_file()
        } | {info["protocol"]}

    required = inventory()
    if not required.issubset(captured["inputs"]) or (oldreport / "failure.json").exists():
        raise AssertionError("Every original successful publication/output must be pinned")
    for path, key in (
        (root / info["protocol"], "protocol_sha256"),
        (oldreport / "manifest.json", "manifest_sha256"),
        (oldreport / "verification.json", "verification_sha256"),
    ):
        if digest(path) != info[key]:
            raise AssertionError("Original wave12 anchor identity differs")
    oldm = json.loads((oldreport / "manifest.json").read_text())
    record = json.loads((oldreport / "verification.json").read_text())
    json.dumps(record, allow_nan=False)
    if (
        info["required_status"] != "VERIFIED"
        or record["status"] != "VERIFIED"
        or record["protocol_sha256"] != info["protocol_sha256"]
        or record["verifier_sha256"] != info["verifier_sha256"]
        or oldm["protocol_sha256"] != info["protocol_sha256"]
        or oldm["code"]["src/verify_target_aligned.py"] != info["verifier_sha256"]
    ):
        raise AssertionError("Original verification and registration identity differs")
    pins = {}
    for group in ("code", "inputs", "preserved"):
        previous.previous._pins(root, captured[group])
        for name, signature in captured[group].items():
            if name in pins and pins[name] != signature:
                raise AssertionError("Conflicting current pins")
            pins[name] = signature
    if not set(oldm["code"]).issubset(captured["code"]) or not set(oldm["inputs"]).issubset(
        captured["inputs"]
    ):
        raise AssertionError("Original code/input identities must remain registered")
    for group in ("code", "inputs", "preserved"):
        previous.previous._pins(root, oldm[group])
        if any(pins.get(name) != signature for name, signature in oldm[group].items()):
            raise AssertionError(
                "Original preserved artifact absent from current registration"
            )
    rebuilt, nested = reconstruct_previous(root, info, pins)
    compare_tree(record, rebuilt, "Entire original wave12 VERIFIED record")
    if inventory() != required or digest(protocol_path) != captured["protocol_sha256"]:
        raise AssertionError("Original inventory or new protocol changed during admission")
    for group in ("code", "inputs", "preserved"):
        previous.previous._pins(root, captured[group])
    return {
        "status": "UPSTREAM_VERIFIED_READ_ONLY",
        "prior_files_written": False,
        "previous_manifest_entries_verified": sum(
            len(oldm[g]) for g in ("code", "inputs", "preserved")
        ),
        "pinned_previous_artifacts": len(required),
        "input_hashes": captured["inputs"],
        "original_verification": rebuilt,
        "nested_target_aligned_admission": nested,
        "upstream_protocol_sha256": info["protocol_sha256"],
        "upstream_manifest_sha256": info["manifest_sha256"],
        "upstream_verification_sha256": info["verification_sha256"],
    }


def verify(root=ROOT):
    root = Path(root)
    protocol_path = root / "sign_memory.yaml"
    protocol = yaml.safe_load(protocol_path.read_text())
    validate_protocol(protocol)
    report, out = root / "reports/sign_memory", root / "data/sign_memory"
    manifest = json.loads((report / "manifest.json").read_text())
    if digest(protocol_path) != manifest["protocol_sha256"]:
        raise AssertionError("Frozen new protocol changed")
    verify_manifest_coverage(root, protocol, manifest)
    for group in ("code", "inputs", "preserved"):
        previous.previous._pins(root, manifest[group])
    output_paths = [
        out / name
        for name in (
            "upstream_admission.json",
            "features.parquet",
            "targets.parquet",
            "forecasts.parquet",
            "fits.json",
        )
    ]
    output_paths += [
        report / name
        for name in ("source_audit.json", "measurement_audit.json", "metrics.json")
    ]
    if (report / "trial_ledger.jsonl").exists():
        output_paths.append(report / "trial_ledger.jsonl")
    snapshot = {str(path.relative_to(root)): digest(path) for path in output_paths}
    proof = validate_upstream(root)

    def saved_json(path):
        name = str(path.relative_to(root))
        return previous.read_json_snapshot(root, name, snapshot[name])

    def saved_parquet(path):
        name = str(path.relative_to(root))
        return previous.previous.read_issued_snapshot(root, name, snapshot[name])

    same_tree(
        saved_json(out / "upstream_admission.json"),
        proof,
        "Entire previous read-only admission",
    )
    qqq, spx, iv, audit = load_source_tables(root, protocol, manifest)
    same_tree(
        saved_json(report / "source_audit.json"),
        audit,
        "Independent immutable bounded-source audit",
    )
    measurement = original.measurement_audit(qqq, spx)
    same_tree(
        saved_json(report / "measurement_audit.json"),
        measurement,
        "Whole-source GK and observed-return gate",
    )
    original.require_measurement(measurement)
    features, targets = feature_target_tables(qqq, spx, iv)
    for name, expected, numeric in (
        ("features", features, ALL_FEATURES),
        ("targets", targets, ("y",)),
    ):
        actual = saved_parquet(out / (name + ".parquet"))
        if tuple(actual.columns) != tuple(expected.columns) or not actual.index.equals(
            expected.index
        ):
            raise AssertionError("Independent raw sign/source table schema differs")
        a = real(actual.loc[:, numeric], missing=True)
        e = real(expected.loc[:, numeric], missing=True)
        if not np.array_equal(np.isnan(a), np.isnan(e)):
            raise AssertionError("Exact paired-window/source missingness differs")
        if name == "targets":
            if not np.array_equal(a, e, equal_nan=True):
                raise AssertionError("Strict binary sign labels differ")
        elif not np.allclose(a, e, rtol=1e-10, atol=1e-12, equal_nan=True):
            raise AssertionError("Independent full-reference features differ")
        for column in set(expected) - set(numeric):
            if not actual[column].equals(expected[column]):
                raise AssertionError("Exact one-session source/target maturity differs")
    panel = saved_parquet(out / "forecasts.parquet")
    fits = saved_json(out / "fits.json")
    metrics = saved_json(report / "metrics.json")
    if (
        (report / "failure.json").exists()
        or metrics.get("status") == "UNEVALUABLE"
        or metrics["protocol_sha256"] != manifest["protocol_sha256"]
        or metrics["evidence_class"] != protocol["evidence_class"]
    ):
        raise AssertionError("New score publication failed or its registered identity differs")
    result = {
        "status": "VERIFIED",
        "upstream_admission": {"status": proof["status"], "prior_files_written": False},
        "raw_feature_rows_verified": len(features),
        "raw_feature_columns_verified": len(ALL_FEATURES),
        "measurement_audit": measurement,
        "forecast_reconstruction": verify_forecasts(features, targets, panel, fits, protocol),
        "primitive_brier_scores_verified": len(panel),
        "paired_brier_differences_verified": int(panel.origin.nunique()) * 2,
        "inference": verify_metrics(root, panel, protocol, metrics, len(fits)),
        "ledger_events_verified": verify_ledger(
            root, metrics, metrics["inherited_rows"], "evaluated"
        ),
        "protocol_sha256": manifest["protocol_sha256"],
        "verifier_sha256": digest(Path(__file__)),
        "limitations": [
            "Strict same-sign probability includes exact-zero returns in the complement and retains missing paired returns as unknown; no volatility-magnitude or covariance target.",
            "The excess-agreement increment is a staged logistic model with frozen baseline logits; regularized training NLL and evaluation Brier share a probability target but need not improve together under misspecification.",
            "Reused archival QQQ ETF/SPX index history, unverified historical source vintages and synchronized auction definitions, and early back-calculated VIX9D remain exploratory limitations.",
            "Calibration in the large and nominal minimum-detectable effect are descriptive diagnostics; nonqualification is not equivalence and no execution or profit claim is made.",
        ],
    }
    for group in ("code", "inputs", "preserved"):
        previous.previous._pins(root, manifest[group])
    previous.previous._pins(root, snapshot)
    verify_manifest_coverage(root, protocol, manifest)
    if digest(protocol_path) != manifest["protocol_sha256"]:
        raise AssertionError("Protocol changed during verification")
    (report / "verification.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def load_source_tables(root, protocol, manifest):
    root = Path(root)
    names = set(protocol["sources"].values())
    names.update(
        name + ".manifest.json"
        for name in list(names)
        if name + ".manifest.json" in manifest["inputs"]
    )
    with tempfile.TemporaryDirectory(prefix="independent-sign-source-") as directory:
        staged = Path(directory)
        for name in sorted(names):
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise AssertionError("Relative registered source paths required")
            payload = (root / name).read_bytes()
            if sha256(payload).hexdigest() != manifest["inputs"][name]:
                raise AssertionError("Raw source changed before immutable decode")
            target = staged / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        qqq, spx, iv, audit = original.load_source_tables(staged, protocol)
    for key, source in audit["sources"].items():
        source["source_path"] = str(root / protocol["sources"][key])
    audit["raw_columns"] = list(ALL_FEATURES)
    audit["target_interpretation"] = (
        "strict same-sign next observed SPX session raw log(close/open) returns for QQQ ETF and SPX price index; any exact zero is nonagreement, missing pair unknown; directional agreement, not volatility magnitude or covariance"
    )
    return qqq, spx, iv, audit


CONTRACT = {
    "sources": {
        "daily": "data/research_paths/spx_daily.parquet",
        "vix": "data/free_sources/raw/cboe/VIX_History.csv",
        "vix9d": "data/free_sources/raw/cboe/VIX9D_History.csv",
        "vvix": "data/free_sources/raw/cboe/VVIX_History.csv",
        "qqq": "data/raw/daily_ohlc.parquet",
        "vxn": "data/free_sources/raw/cboe/VXN_History.csv",
    },
    "index": {
        "asset": "QQQ_ETF_and_SPX_price_index",
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
        "models": ["frequency", "baseline", "memory"],
        "all_features": [
            "const",
            "qqq_lg_d",
            "qqq_lg_w",
            "qqq_lg_m",
            "qqq_lt_d",
            "qqq_lt_w",
            "qqq_lt_m",
            "qqq_neg_d",
            "qqq_neg_w",
            "qqq_neg_m",
            "spx_lg_d",
            "spx_lg_w",
            "spx_lg_m",
            "spx_lt_d",
            "spx_lt_w",
            "spx_lt_m",
            "spx_neg_d",
            "spx_neg_w",
            "spx_neg_m",
            "lvxn",
            "lvix",
            "term",
            "lvvix",
            "entry_dow_1",
            "entry_dow_2",
            "entry_dow_3",
            "entry_dow_4",
            "corr22",
            "qqq_day_d",
            "qqq_day_w",
            "qqq_day_m",
            "spx_day_d",
            "spx_day_w",
            "spx_day_m",
            "qqq_pos22",
            "qqq_neg22",
            "spx_pos22",
            "spx_neg22",
            "independent22",
            "excess22",
        ],
        "minimum_train_per_class": 50,
    },
    "measurement": {
        "gk_floor": 1e-10,
        "formula": "0.5*log(high/low)^2-(2*log(2)-1)*log(close/open)^2",
        "admission_pool": "For each asset separately, every observed complete OHLC row after "
        "alignment to full bounded SPX reference calendar; before ANY "
        "correlation, feature, target or sample mask",
        "gate": "Every complete observed raw GK strictly greater than1e-10; every observed "
        "open/close pair must yield finite signed log(close/open), including high/low gaps. "
        "Audit precedes any feature/target/sample mask.",
        "failure": "Save measurement audit before requiring PASS; no feature/target construction or "
        "fits on floor failure; retain both registered hypotheses as UNEVALUABLEp1",
        "missing": "Absent or partially missing OHLC is counted separately and remains unknown; "
        "never converted to zero or floored",
        "invalid": "Invalid observed prices/ranges or nonfinite computed GK fail wholewave. Any "
        "observed open/close pair producing nonfinite signed log return fails even when "
        "high/low is missing; missing open/close stays unknown.",
        "zero_target": "Any exact zero raw intraday return is retained as strict nonagreement; any "
        "missing pair remains unknown",
        "diagnostics": "Full inherited GK/signed return gate before any feature/target mask; binary "
        "counts only after pretests and registration",
        "claims": "No standalone index direction, volatility magnitude, covariance, copula "
        "identification or profit claim",
    },
    "source_contract": {
        "instruments": "QQQ ETF vendor raw OHLC and Yahoo ^GSPC SPX price index; no audited "
        "exact Nasdaq100 OHLC source is claimed",
        "fields": "open,high,low,close only; no adjusted close, volume, inferred distributions "
        "or corporate-action substitution",
        "reference_calendar": "Retain every bounded observed SPX date; reindex QQQ without "
        "first intersecting calendars; no next-common-date labels",
        "previous_close": "Each asset raw overnight and close return requires its actual source "
        "predecessor date equal to preceding SPX reference date; otherwise "
        "unknown",
        "vintage": "Archival Yahoo and Cboe extracts; revisions, synchronized auctions and "
        "exact historical publication latency unverified",
        "vix9d": "Back-calculated January2011-October2013 training history; forecasts begin2016",
        "iv_match": "VXN underlying Nasdaq100 and SPX/VIX controls; no implied covariance or "
        "exact synchronous index/ETF auction data claimed",
        "missing": "Missing observed quotes or absent exact-date sources remain unknown through "
        "strict rolling windows; no filling or compressed rolling",
        "limitations": "Intraday ratios cancel common within-session units, but ETF tracking, "
        "index construction and vendor conventions remain; raw intersession "
        "controls retain distribution effects",
        "immutable_read": "Copy each six declared raw file and its documentary manifest from a "
        "single hash-checked byte snapshot into an isolated temporary "
        "directory at same relative path; frozen loader bounds dates before "
        "numerical parsing; source audits retain original registered "
        "path/hash; no source writes",
    },
    "correlation": {
        "window": 22,
        "returns": "raw log(close/open) per matched SPX date, common within-session units cancel",
        "calculation": "Centered two-pass Pearson covariance divided by centered norms over "
        "exactly22 complete paired returns; ends at previous SPX session",
        "missing": "Any missing pair in window or exact zero centered denominator makes correlation "
        "unknown for all models",
        "roundoff_tolerance": 1e-12,
        "bounds": "Retain finite rho including tiny overshoot with abs(rho)<=1+1e-12; reject larger "
        "violation; no clipping, epsilon denominator, alternate window or Fisher "
        "transform",
    },
    "study_id": "sign_memory_wave13",
    "wave": 13,
    "wave_alpha": 0.0002747252747252747,
    "evidence_class": "exploratory_reused_history_archival_QQQ_SPX_raw_sign_agreement",
    "upstream": {
        "protocol": "target_aligned.yaml",
        "reports": "reports/target_aligned",
        "data": "data/target_aligned",
        "protocol_sha256": "ca11ec1c90f8d83867b6dce4efaaac4f0e5ed11e23143e2ab1f28d78a9f136b3",
        "manifest_sha256": "996e9f8f081a69a66dad9c4c611feb6775fc69c5ad9e3630230d55ad97b426d9",
        "verification_sha256": "dfdc8224a8b7455622e6a16e2671bc70fe01a9a2600e78f39814ad824510fbd0",
        "verifier_sha256": "93e8ac7f88624fec0933600a85062397c9b0b2306afe87d0b4e3ee4af0f751d2",
        "required_status": "VERIFIED",
        "admission": "Reconstruct original VERIFIED record via frozen read-only functions; all prior "
        "pins and full outputs included in new manifest; historical inventory anchored by "
        "original manifest hash, not current-tree old coverage",
    },
    "target": {
        "event": "Both next observed SPX-session raw log(close/open) returns strictly positive, or both "
        "strictly negative",
        "zero": "Either or both exactly zero gives class0; complement includes opposite directions and "
        "ties",
        "missing": "Any missing return gives unknown, never class0; reject infinity or invalid observed "
        "source",
        "sign_arithmetic": "Direct comparisons, never multiplication of returns or epsilon sign "
        "threshold",
        "positive_rescaling": "Within-session positive price unit changes preserve each raw return; no "
        "total-return or executable auction equivalence",
        "fit_schedule": "Monthly first feature-complete origin before future query-label filtering; "
        "expanding training labels mature by prior SPX close; retain unscored "
        "applications and folds",
    },
    "features": {
        "window": 22,
        "bounded": ["qqq_pos22", "qqq_neg22", "spx_pos22", "spx_neg22", "independent22"],
        "memory": "excess22",
        "formula": "independent22=P_Q+*P_S+ + P_Q-*P_S-; "
        "excess22=joint_strict_agreement22-independent22",
        "history": "All five component fractions require identical complete paired 22-session window "
        "ending t-1 on full SPX calendar; no compressed dates or filling",
        "interpretation": "Empirical excess agreement memory beyond marginal signs; not an unbiased "
        "dependence estimator or copula parameter",
    },
    "fitting": {
        "penalty": 0.01,
        "maximum_iterations": 200,
        "maximum_backtracks": 60,
        "armijo": 0.0001,
        "gradient_tolerance": 1e-08,
        "scale_minimum": 1e-12,
        "baseline": "Unpenalized intercept and normalized logistic NLL plus .01 squared slopes; "
        "original33nonintercept+training-centered corr22 square population standardized, "
        "five bounded controls centered with fixedscale1",
        "memory": "Freeze baseline logit; add b*(excess22-training_mean) with .01*b², no intercept or "
        "joint augmented refit",
        "constant_bounded": "Predeclared exact all-equal test centers column exactly to zero, retains "
        "column with canonical penalized coefficient zero; fixedscale1",
        "constant_memory": "Predeclared exact all-equal excess22 yields centered memory zero and "
        "canonical b0 with retained model/fold",
        "frequency": "Mean binary training label on exact same complete mature sample",
        "arithmetic": "Finite real inputs, logits, derivatives, objectives, coefficients and "
        "probabilities; stable logaddexp/expit; numeric probability endpoints allowed; no "
        "posthoc probability clipping or optimizer fallback",
        "optimizer": "Deterministic damped Newton with positive Hessian and Armijo from fixed "
        "initialization; audit objective/fullgradient and independent stationary convex "
        "optimum before interpretation",
        "optimization_claim": "Each stage convex with unique slope optimum due positive ridge; staged "
        "pair is not the jointly optimized augmented model",
        "initialization": "Baseline intercept logit(training frequency), all slopes0; memoryb0; no "
        "restarts",
        "stable_logistic": "NLL mean(logaddexp(0,(1-2y)*eta)); score expit(eta) fory0 and -expit(-eta) "
        "fory1; curvature expit(eta)*expit(-eta)",
    },
    "scoring": {
        "loss": "brier",
        "effect_threshold_absolute": 0.0005,
        "paired_difference": "(p_candidate-p_control)*((p_candidate-y)+(p_control-y)), after finite "
        "individual squared loss validation",
        "arithmetic": "Binary finite y and finite probabilities in[0,1]; loss in[0,1]; exactzero errors "
        "valid; nonzero squares/products underflowing tozero reject; "
        "coherence64epsilon/downwardULP no arbitrary absolute floor",
        "functional": "Conditional strict-agreement probability; expected Brier pi(1-pi)+(p-pi)^2; "
        "bounded loss needs no return fourth moments",
        "effect_reference": "Absolute squared-probability error decrease .0005 bothphases; fixed "
        "statistical reference, not profit or direct probability-point improvement",
    },
    "comparisons": {
        "new_hypotheses": 2,
        "inherited_hypotheses": 117,
        "cumulative_hypotheses": 119,
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
        ],
        "controls": ["baseline", "frequency"],
        "contrasts": [["memory", "baseline", "brier"], ["memory", "frequency", "brier"]],
        "candidate_gate": "Both registered contrasts must pass all fixed "
        "phase/effect/stability/multiplicity gates",
    },
    "inference": {
        "blocks": [21, 63, 126],
        "hac_lags": 126,
        "minimum_phase_observations": 127,
        "minimum_phase_per_class": 30,
        "minimum_slice_per_class": 15,
        "bootstrap_draws": 99999,
        "seed": 20260919,
        "inference_scale": 1.0,
        "seed_rule": "seed+phase_code*10000+block; development0,evaluation1",
        "p_phase": "Maximum two-sided centered-null circular-block bootstrap p and Bartlett HAC126 p",
        "p_hypothesis": "Maximum development/evaluation p",
        "multiplicity": "Holm2 at.05/(13*14), separately cumulativeHolm119 at.05",
        "gate": "Bothphase absolute Brier decrease>=.0005 against BOTHcontrols; negative differences "
        "in bothfixed evaluation slices; waveHolm<.05/182,cumulativeHolm<.05",
        "support": "Atleast50events/50nonevents perfit;30/class perphase;15/class perfixed "
        "evaluationslice; any insufficient support aborts wholefamily, no droppedfolds",
        "power": "Nominal HAC80percent MDE diagnostic only, divided by fixed .0005 effect; no "
        "postscore power gate or equivalence claim",
        "resolution": "Minimum p1/100000 below one tenth strictest two-comparison raw wave "
        "cutoff1/7280",
        "failure_rule": "All2new hypotheses UNEVALUABLEp1 after any "
        "admission/measurement/support/fit/scoring/verification/publication failure; "
        "preserve all diagnostics and old artifacts; no outcome-dependent repair",
        "calibration": "Descriptive calibration in the large only: perphase/model n, observed binary "
        "frequency, mean issued probability, probability-minus-frequency gap and "
        "Brier. No bins, calibration fitting, hypothesis, selection or promotion gate; "
        "not a full conditional calibration claim.",
    },
    "verification": {
        "coefficient_relative_tolerance": 1e-07,
        "coefficient_absolute_tolerance": 1e-06,
        "independent_probability_relative_tolerance": 1e-07,
        "independent_probability_absolute_tolerance": 1e-06,
        "saved_probability_relative_tolerance": 1e-10,
        "saved_probability_absolute_tolerance": 1e-12,
        "objective_relative_tolerance": 1e-10,
        "objective_absolute_tolerance": 1e-12,
        "saved_gradient_tolerance": 1e-08,
        "gradient_roundoff_allowance": 1e-12,
        "independent_gradient_tolerance": 1e-10,
        "inference_relative_tolerance": 1e-09,
        "inference_absolute_tolerance": 1e-14,
        "reconstruction": "Independent raw source, feature, "
        "sign/tie/rolling/maturity/support/transformation/convex objective, fit, "
        "probability, score, fourphase inference and complete119family ledger; "
        "no new producer import",
        "independent_optimizer": {
            "baseline_method": "trust-exact",
            "baseline_initialization": "all_zero",
            "baseline_gradient_target": 1e-10,
            "baseline_maximum_iterations": 500,
            "memory_method": "brentq",
            "memory_bracket": "symmetric plus/minus (mean(abs(centered_memory))+1)/0.02",
            "memory_absolute_tolerance": 1e-12,
            "memory_relative_tolerance": 1e-14,
            "memory_maximum_iterations": 200,
            "accepted_gradient_tolerance": 1.0001e-08,
            "acceptance": "Require finite objectives, parameters and "
            "probabilities plus independently recomputed full "
            "gradient within the accepted tolerance. Record "
            "optimizer status; a tighter target being unmet "
            "does not override the declared accepted gradient "
            "criterion. Compare independent "
            "coefficients/probabilities at their separate "
            "fixed tolerances; reconstruct primary scores from "
            "stricter saved-coefficient replay.",
            "arithmetic": "Independently evaluate stable signed-label "
            "logaddexp NLL, signed logistic score and "
            "expit(eta)*expit(-eta) curvature. Finite "
            "probability endpoints are valid; no probability "
            "clipping or optimization retries.",
        },
    },
    "outputs": {
        "data": "data/sign_memory",
        "reports": "reports/sign_memory",
        "features": "data/sign_memory/features.parquet",
        "targets": "data/sign_memory/targets.parquet",
        "forecasts": "data/sign_memory/forecasts.parquet",
        "fits": "data/sign_memory/fits.json",
        "admission": "data/sign_memory/upstream_admission.json",
    },
}


def validate_protocol(p):
    if (
        p.get("wave") != 13
        or p.get("wave_alpha") != WAVE_ALPHA
        or any(p.get(key) != value for key, value in CONTRACT.items())
    ):
        raise ValueError("Frozen strict-sign protocol differs")


def verify_manifest_coverage(root, protocol, manifest):
    root = Path(root)
    info = protocol["upstream"]
    old = json.loads((root / info["reports"] / "manifest.json").read_text())
    code = {
        str(path.relative_to(root))
        for folder in ("src", "tests")
        for path in (root / folder).rglob("*.py")
    } | set(old["code"])
    inputs = set(old["inputs"]) | {info["protocol"]}
    for folder in (info["reports"], info["data"]):
        inputs.update(
            str(path.relative_to(root))
            for path in (root / folder).rglob("*")
            if path.is_file()
        )
    preserved = {
        str(path.relative_to(root))
        for path in [*root.glob("*.yaml"), *(root / "reports").rglob("*")]
        if path.is_file()
        and path != root / "sign_memory.yaml"
        and root / "reports/sign_memory" not in path.parents
        and str(path.relative_to(root)) not in inputs
    }
    if (
        set(manifest["code"]) != code
        or set(manifest["inputs"]) != inputs
        or set(manifest["preserved"]) != preserved
    ):
        raise AssertionError(
            "All original and new code, upstream bytes and prior publications must be frozen"
        )
    if set(manifest["inputs"]) & set(manifest["preserved"]):
        raise AssertionError(
            "Manifest inputs and preserved publications must not be counted twice"
        )


def invalidate_publication(root, error):
    report = Path(root) / "reports/sign_memory"
    report.mkdir(parents=True, exist_ok=True)
    path = report / "metrics.json"
    original_bytes = path.read_bytes() if path.exists() else None
    previous, invalid_bytes = None, None
    if original_bytes is not None:
        try:
            decoded = json.loads(original_bytes)
            if isinstance(decoded, dict):
                # JSON's permissive parser accepts NaN/Infinity. These cannot
                # enter either the canonical failure record or its JSON backup.
                json.dumps(decoded, allow_nan=False)
                previous = decoded
            else:
                invalid_bytes = original_bytes
        except (UnicodeDecodeError, ValueError):
            invalid_bytes = original_bytes
    protocol_hash = previous.get("protocol_sha256") if previous else None
    message = "Independent verification failed: " + type(error).__name__ + ": " + str(error)
    failure = {
        "status": "UNEVALUABLE",
        "whole_wave_aborted": True,
        "leads": [],
        "hypothesis_count": 2,
        "cumulative_hypothesis_count": 119,
        "protocol_sha256": protocol_hash,
        "rows": [
            {
                "study": "sign_memory",
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
    serialized = json.dumps(failure, indent=2, allow_nan=False) + "\n"
    path.write_text(serialized)
    (report / "failure.json").write_text(serialized)
    (report / "results.md").write_text(
        "# QQQ–SPX strict directional-agreement memory\n\nUNEVALUABLE: independent verification failed. Both comparisons retained with p=1; no lead.\n\n"
        + message
        + "\n"
    )
    (report / "verification.json").write_text(
        json.dumps(
            {"status": "FAILED", "error": message, "protocol_sha256": protocol_hash}, indent=2
        )
        + "\n"
    )
    with (report / "trial_ledger.jsonl").open("a") as stream:
        for row in failure["rows"]:
            stream.write(
                json.dumps(
                    {"event": "verification_failed", **row}, sort_keys=True, allow_nan=False
                )
                + "\n"
            )
    backup = report / "unpublished_scored_metrics.json"
    if (
        previous is not None
        and previous.get("status") != "UNEVALUABLE"
        and not backup.exists()
    ):
        backup.write_text(
            json.dumps(
                {
                    "status": "UNPUBLISHED_DIAGNOSTIC_ONLY",
                    "not_for_inherited_inference_or_promotion": True,
                    "scored_metrics": previous,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
    invalid_backup = report / "unpublished_invalid_metrics.txt"
    if invalid_bytes is not None and not invalid_backup.exists():
        invalid_backup.write_bytes(invalid_bytes)
    return failure


def verify_with_failure_guard(root=ROOT):
    try:
        return verify(root)
    except Exception as error:
        invalidate_publication(root, error)
        raise


def phase_statistics(panel, control, phase, code, protocol):
    section = protocol["index"]
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
    rows, benchmark = groups["memory"], groups[control]
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


def inherited_rows(root, protocol):
    output = []
    for source in protocol["comparisons"]["inherited_sources"]:
        for number, row in enumerate(json.loads((root / source).read_text())["rows"]):
            output.append(
                {
                    "study": row.get("study", Path(source).parent.name or Path(source).stem),
                    "candidate": row["candidate"],
                    "control": row.get("control", "baseline"),
                    "horizon": row["horizon"],
                    "p_conservative": row["p_conservative"],
                    "source": source,
                    "source_sha256": digest(root / source),
                    "source_row_index": number,
                    **{name: row[name] for name in ("measure", "score") if name in row},
                }
            )
    if len(output) != 117:
        raise AssertionError("All117 inherited comparisons must remain identified")
    return output


def verify_ledger(root, metrics, prior, final_event):
    report = Path(root) / "reports/sign_memory"
    ledger = [
        json.loads(line)
        for line in (report / "trial_ledger.jsonl").read_text().splitlines()
        if line
    ]
    for event, expected in (("inherited", prior), (final_event, metrics["rows"])):
        actual = [
            {key: value for key, value in row.items() if key != "event"}
            for row in ledger
            if row["event"] == event
        ]
        compare_tree(actual, expected, "Independent " + event + " trial ledger")
    registered = [row for row in ledger if row["event"] == "registered"]
    if (
        len(ledger) != 121
        or len(registered) != 2
        or len(prior) != 117
        or [(row["candidate"], row["control"]) for row in registered] != list(COMPARISONS)
        or any(
            row["protocol_sha256"] != metrics["protocol_sha256"]
            or row["study"] != "sign_memory"
            or row["horizon"] != 1
            or row["score"] != "brier"
            for row in registered
        )
    ):
        raise AssertionError("Complete117inherited+2registered+2terminal ledger required")
    return {"inherited": 117, "registered": 2, final_event: 2}


if __name__ == "__main__":
    print(json.dumps(verify_with_failure_guard(), indent=2, allow_nan=False), flush=True)
