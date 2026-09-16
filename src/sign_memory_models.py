"""Fixed staged ridge-logistic forecasts of strict paired sign agreement.

This module never loads sources. Calendar admission precedes label selection for
every monthly application cohort; the memory stage freezes the baseline logit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from .cross_moment_score import _squared_error
from .index_hinge import _dates
from .sign_memory_features import ALL_FEATURES, BOUNDED, MEMORY, MODELS, OLD_FEATURES

ALPHA = 0.01
MAX_ITERATIONS = 200
MAX_BACKTRACKS = 60
ARMIJO = 1e-4
GRADIENT_TOLERANCE = 1e-8
SCALE_MINIMUM = 1e-12
MINIMUM_PER_CLASS = 50
BASELINE_COLUMNS = (*OLD_FEATURES, "corr22_centered_sq", *BOUNDED)
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


def _finite(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"Finite real {name} required")
    out = np.asarray(value, dtype=float)
    if not np.isfinite(out).all():
        raise ValueError(f"Finite real {name} required")
    return out


def _binary(value):
    y = _finite(value, "binary labels")
    if y.ndim != 1 or not len(y) or not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("Nonempty one-dimensional binary labels required")
    return y


def _support(y, minimum=MINIMUM_PER_CLASS):
    y = _binary(y)
    events = int(np.count_nonzero(y == 1))
    nonevents = len(y) - events
    if events < minimum or nonevents < minimum:
        raise ValueError(
            f"INSUFFICIENT_DATA: binary support {events} events/{nonevents} nonevents"
        )
    return {"n": len(y), "events": events, "nonevents": nonevents}


def logistic_state(beta, design, y, *, offset=None, penalize_intercept=False):
    """Normalized NLL, full gradient and Hessian, in fixed coefficient units."""
    beta, design, y = _finite(beta, "coefficients"), _finite(design, "design"), _binary(y)
    if beta.ndim != 1 or design.shape != (len(y), len(beta)) or not len(beta):
        raise ValueError("Aligned logistic dimensions required")
    if offset is None:
        offset = np.zeros(len(y))
    else:
        offset = _finite(offset, "offset")
        if offset.shape != y.shape:
            raise ValueError("Offset and labels must align")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        eta = design @ beta + offset
        if not np.isfinite(eta).all():
            raise ValueError("Finite logistic logits required")
        positive, negative = expit(eta), expit(-eta)
        error = np.where(y == 1, -negative, positive)
        curvature = positive * negative
        penalty = beta.copy()
        if not penalize_intercept:
            penalty[0] = 0.0
        objective = float(
            np.mean(np.logaddexp(0, (1 - 2 * y) * eta)) + ALPHA * (penalty @ penalty)
        )
        gradient = design.T @ error / len(y) + 2 * ALPHA * penalty
        hessian = design.T @ (curvature[:, None] * design) / len(y)
        diagonal_penalty = np.full(len(beta), 2 * ALPHA)
        if not penalize_intercept:
            diagonal_penalty[0] = 0.0
        hessian += np.diag(diagonal_penalty)
    if (
        not np.isfinite(objective)
        or objective < 0
        or not np.isfinite(gradient).all()
        or not np.isfinite(hessian).all()
    ):
        raise ValueError("Nonfinite logistic objective or derivatives")
    return objective, gradient, hessian


def _newton(design, y, start, *, offset=None, penalize_intercept=False):
    beta = _finite(start, "initial coefficients").copy()
    initial = beta.tolist()
    backtracks = 0
    for iteration in range(MAX_ITERATIONS + 1):
        objective, gradient, hessian = logistic_state(
            beta, design, y, offset=offset, penalize_intercept=penalize_intercept
        )
        gradient_max = float(np.max(np.abs(gradient)))
        if gradient_max <= GRADIENT_TOLERANCE:
            return beta, {
                "objective": objective,
                "gradient": gradient.tolist(),
                "gradient_max_abs": gradient_max,
                "iterations": iteration,
                "backtracks": backtracks,
                "success": True,
                "start": initial,
            }
        if iteration == MAX_ITERATIONS:
            break
        try:
            np.linalg.cholesky(hessian)
            direction = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Newton Hessian/linear solve failed; no fallback") from exc
        decrease = float(gradient @ direction)
        if not np.isfinite(direction).all() or not np.isfinite(decrease) or decrease <= 0:
            raise ValueError("Newton direction must have finite positive descent")
        accepted = False
        for halvings in range(MAX_BACKTRACKS):
            fraction = 2.0 ** (-halvings)
            trial = beta - fraction * direction
            trial_objective, _, _ = logistic_state(
                trial, design, y, offset=offset, penalize_intercept=penalize_intercept
            )
            if trial_objective <= objective - ARMIJO * fraction * decrease:
                beta = trial
                backtracks += halvings
                accepted = True
                break
        if not accepted:
            raise ValueError("Newton Armijo budget exhausted; no fallback")
    raise ValueError("Logistic fit did not converge within fixed Newton budget")


def transform(train, apply):
    """Training-centered corr-square and fixed versus empirical scaling."""
    if train.columns.has_duplicates or apply.columns.has_duplicates:
        raise ValueError("Unique feature names required")
    tr = train.loc[:, ALL_FEATURES].copy()
    ap = apply.loc[:, ALL_FEATURES].copy()
    values, query = _finite(tr, "training features"), _finite(ap, "application features")
    if len(tr) < 2 or not np.all(values[:, 0] == 1) or not np.all(query[:, 0] == 1):
        raise ValueError("Finite common designs with unit intercept required")
    tr, ap = tr.astype(float), ap.astype(float)
    center = float(tr["corr22"].mean())
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for frame in [tr, ap]:
            frame["corr22_centered_sq"] = (frame["corr22"] - center) ** 2
        empirical = (*OLD_FEATURES[1:], "corr22_centered_sq")
        means = tr.loc[:, empirical].mean().to_numpy()
        scales = tr.loc[:, empirical].std(ddof=0).to_numpy()
    if (
        not np.isfinite(means).all()
        or not np.isfinite(scales).all()
        or (scales <= SCALE_MINIMUM).any()
    ):
        raise ValueError("INSUFFICIENT_DATA: zero-scale existing feature; no fallback")
    bounded_means, constants = [], []
    for name in BOUNDED:
        column = tr[name].to_numpy()
        constant = bool(np.all(column == column[0]))
        bounded_means.append(float(column[0]) if constant else float(column.mean()))
        if constant:
            constants.append(name)
    means = np.r_[0.0, means, bounded_means]
    scales = np.r_[1.0, scales, np.ones(len(BOUNDED))]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        x = (tr.loc[:, BASELINE_COLUMNS] - means) / scales
        app = (ap.loc[:, BASELINE_COLUMNS] - means) / scales
    _finite(x, "transformed training design")
    _finite(app, "transformed application design")
    return (
        x,
        app,
        {
            "columns": list(BASELINE_COLUMNS),
            "means": means.tolist(),
            "scales": scales.tolist(),
            "corr22_mean": center,
            "bounded_constant_columns": constants,
        },
    )


def _probability(eta):
    eta = _finite(eta, "issued logits")
    p = expit(eta)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Finite issued probabilities in [0,1] required")
    return p


def fit_predict(train_features, train_targets, apply_features):
    if not train_features.index.equals(train_targets.index):
        raise ValueError("Identical ordered training feature/label rows required")
    y = _binary(train_targets["y"].to_numpy())
    support = _support(y)
    tr, app, transform_audit = transform(train_features, apply_features)
    x, query = tr.to_numpy(), app.to_numpy()
    frequency = float(y.mean())
    start = np.zeros(x.shape[1])
    start[0] = np.log(frequency / (1 - frequency))
    beta, baseline_solver = _newton(x, y, start)
    train_eta, app_eta = x @ beta, query @ beta
    _finite(train_eta, "training logits")
    _finite(app_eta, "application logits")
    memory_raw = train_features[MEMORY].to_numpy(float)
    constant_memory = bool(np.all(memory_raw == memory_raw[0]))
    memory_mean = float(memory_raw[0]) if constant_memory else float(memory_raw.mean())
    m = memory_raw - memory_mean
    query_m = apply_features[MEMORY].to_numpy(float) - memory_mean
    _finite(m, "centered training memory")
    _finite(query_m, "centered application memory")
    coefficient, memory_solver = _newton(
        m[:, None], y, np.zeros(1), offset=train_eta, penalize_intercept=True
    )
    b = float(coefficient[0])
    if constant_memory and b != 0:
        raise ValueError("Exact constant memory must have canonical zero slope")
    forecasts = {
        "frequency": np.full(len(query), frequency),
        "baseline": _probability(app_eta),
        "memory": _probability(app_eta + b * query_m),
    }
    audit = {
        "train_n": len(y),
        "application_n": len(query),
        "support": support,
        "transform": transform_audit,
        "frequency": {
            "probability": frequency,
            "train_n": len(y),
            "application_n": len(query),
        },
        "baseline": {
            "columns": transform_audit["columns"],
            "means": transform_audit["means"],
            "scales": transform_audit["scales"],
            "beta": beta.tolist(),
            "alpha": ALPHA,
            "train_n": len(y),
            "application_n": len(query),
            **baseline_solver,
        },
        "memory": {
            "column": MEMORY,
            "mean": memory_mean,
            "scale": 1.0,
            "b": b,
            "alpha": ALPHA,
            "train_n": len(y),
            "application_n": len(query),
            "baseline_frozen": True,
            "status": "EXACT_CONSTANT_INPUT" if constant_memory else "FITTED",
            **memory_solver,
        },
    }
    return {"forecasts": forecasts, "audit": audit}


def _alignment(features, targets):
    _dates(features.index)
    if not features.index.equals(targets.index):
        raise ValueError("Features and labels need identical full entry calendars")
    if features.columns.has_duplicates or targets.columns.has_duplicates:
        raise ValueError("Unique feature and target names required")
    dates = pd.Series(features.index, index=features.index)
    for frame, name, shift in [
        (features, "feature_cutoff_date", 1),
        (targets, "target_end", -1),
        (targets, "available_date", -1),
    ]:
        if not frame[name].equals(dates.shift(shift).rename(name)):
            raise ValueError(
                "Exact previous-session features and next-session labels required"
            )
    known = targets["y"].notna()
    if known.any():
        _binary(targets.loc[known, "y"].to_numpy())
        if targets.loc[known, "target_end"].isna().any():
            raise ValueError("Known labels must have a next-session date")


def training_mask(
    features, targets, fit_entry, min_train=1000, minimum_per_class=MINIMUM_PER_CLASS
):
    _alignment(features, targets)
    fit_entry = pd.Timestamp(fit_entry)
    if (
        fit_entry not in features.index
        or min_train < 2
        or minimum_per_class != MINIMUM_PER_CLASS
    ):
        raise ValueError("Invalid fit origin or fixed training support contract")
    cutoff = features.loc[fit_entry, "feature_cutoff_date"]
    if pd.isna(cutoff):
        raise ValueError("INSUFFICIENT_DATA: no prior-session cutoff")
    mask = (
        np.isfinite(features.loc[:, ALL_FEATURES]).all(axis=1)
        & features["feature_cutoff_date"].notna()
        & targets["y"].notna()
        & (features.index < fit_entry)
        & (targets["available_date"] <= cutoff)
    )
    if int(mask.sum()) < min_train:
        raise ValueError(f"INSUFFICIENT_DATA: {int(mask.sum())} complete mature training rows")
    _support(targets.loc[mask, "y"].to_numpy(), minimum_per_class)
    return mask


def brier_loss(probability, y):
    p, y = _finite(probability, "probabilities"), _binary(y)
    if p.shape != y.shape or ((p < 0) | (p > 1)).any():
        raise ValueError("Aligned probabilities in [0,1] required")
    _, loss = _squared_error(p, y)
    if (loss < 0).any() or (loss > 1).any():
        raise ValueError("Brier loss must be finite in [0,1]")
    return loss


def validate_panel(panel):
    if (
        not isinstance(panel, pd.DataFrame)
        or panel.empty
        or tuple(panel.columns) != PANEL_COLUMNS
    ):
        raise ValueError("Exact nonempty sign-probability panel schema required")
    if panel.duplicated(["origin", "model"]).any() or set(panel.model) != set(MODELS):
        raise ValueError("Exactly one row per origin/model required")
    date_columns = [
        "origin",
        "feature_cutoff_date",
        "target_end",
        "available_date",
        "fit_origin",
        "fit_cutoff_date",
        "train_last_target",
        "train_last_available",
    ]
    for name in date_columns:
        dates = pd.DatetimeIndex(panel[name])
        if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
            raise ValueError("Normalized finite date metadata required")
    if (
        not panel.horizon.eq(1).all()
        or not panel.phase.isin(["development", "evaluation"]).all()
        or not (panel.feature_cutoff_date < panel.origin).all()
        or not (panel.origin < panel.target_end).all()
        or not panel.target_end.equals(panel.available_date.rename("target_end"))
        or not (panel.fit_cutoff_date < panel.fit_origin).all()
        or not (panel.fit_origin <= panel.origin).all()
        or not (panel.fit_cutoff_date <= panel.feature_cutoff_date).all()
        or not panel.fit_origin.dt.to_period("M").equals(panel.origin.dt.to_period("M"))
        or not (panel.train_last_available <= panel.fit_cutoff_date).all()
        or not panel.train_last_target.equals(
            panel.train_last_available.rename("train_last_target")
        )
        or not (panel.target_end <= pd.Timestamp("2025-10-20")).all()
        or not (panel.origin <= pd.Timestamp("2025-10-17")).all()
    ):
        raise ValueError("Invalid horizon, phase, maturity or bounded date metadata")
    counts = _finite(panel.train_n, "training counts")
    if (counts < 2 * MINIMUM_PER_CLASS).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError("Integral supported training counts required")
    fit_metadata = [
        "fit_origin",
        "fit_cutoff_date",
        "train_n",
        "train_last_target",
        "train_last_available",
    ]
    for _, month in panel.groupby(panel.origin.dt.to_period("M")):
        if not month[fit_metadata].nunique(dropna=False).eq(1).all():
            raise ValueError("Monthly fit metadata must be constant across applications")
    shared = [name for name in PANEL_COLUMNS if name not in ["model", "probability", "loss"]]
    first = (
        panel.loc[panel.model == MODELS[0], shared]
        .sort_values("origin")
        .reset_index(drop=True)
    )
    for name in MODELS[1:]:
        other = (
            panel.loc[panel.model == name, shared].sort_values("origin").reset_index(drop=True)
        )
        if not first.equals(other):
            raise ValueError("All three models need identical cohorts, labels and metadata")
    actual = brier_loss(panel.probability.to_numpy(), panel.y.to_numpy())
    if not np.array_equal(actual, panel.loss.to_numpy()):
        raise ValueError("Saved Brier loss differs from exact issued probability score")
    return panel


def forecast_panel(features, targets, config):
    _alignment(features, targets)
    if tuple(config["models"]) != MODELS or tuple(config["all_features"]) != ALL_FEATURES:
        raise ValueError("Fixed sign-memory feature/model family differs")
    minimum_per_class = config["minimum_train_per_class"]
    if minimum_per_class != MINIMUM_PER_CLASS:
        raise ValueError("Fixed minimum class support differs")
    start, end, latest = map(
        pd.Timestamp, [config["origin_start"], config["origin_end"], config["latest_target"]]
    )
    dev_start, dev_end = map(pd.Timestamp, config["development"])
    ev_start, ev_end = map(pd.Timestamp, config["evaluation"])
    if (
        not start <= dev_start <= dev_end < ev_start <= ev_end <= end < latest
        or config["development_target_available_by"] != config["development"][1]
        or latest > pd.Timestamp("2025-10-20")
        or end > pd.Timestamp("2025-10-17")
    ):
        raise ValueError("Invalid fixed historical phases or numerical fence")
    dates = features.index
    phases = ((dates >= dev_start) & (dates <= dev_end)) | (
        (dates >= ev_start) & (dates <= ev_end)
    )
    ready = (
        np.isfinite(features.loc[:, ALL_FEATURES]).all(axis=1)
        & features.feature_cutoff_date.notna()
    )
    entries = dates[ready & phases & (dates >= start) & (dates <= end)]
    if not len(entries):
        raise ValueError("INSUFFICIENT_DATA: no common complete application features")
    labels = (
        targets.y.notna()
        & (targets.available_date <= latest)
        & ((dates > dev_end) | (targets.available_date <= dev_end))
    )
    rows, fits = [], []
    for month in entries.to_period("M").unique():
        application_dates = entries[entries.to_period("M") == month]
        fit_entry = application_dates[0]
        mask = training_mask(
            features, targets, fit_entry, config["minimum_train"], minimum_per_class
        )
        result = fit_predict(
            features.loc[mask], targets.loc[mask], features.loc[application_dates]
        )
        cutoff = features.loc[fit_entry, "feature_cutoff_date"]
        last_target = targets.loc[mask, "target_end"].max()
        last_available = targets.loc[mask, "available_date"].max()
        fits.append(
            {
                "fit_origin": str(fit_entry.date()),
                "fit_cutoff_date": str(cutoff.date()),
                "train_n": int(mask.sum()),
                "train_first_origin": str(features.index[mask][0].date()),
                "train_last_origin": str(features.index[mask][-1].date()),
                "train_last_target": str(last_target.date()),
                "train_last_available": str(last_available.date()),
                "application_n": len(application_dates),
                "model_audit": result["audit"],
            }
        )
        if set(result["forecasts"]) != set(MODELS):
            raise ValueError("Every scheduled fit must issue all three models")
        probabilities = {}
        for name in MODELS:
            value = _finite(result["forecasts"][name], "application probabilities")
            if value.shape != (len(application_dates),) or ((value < 0) | (value > 1)).any():
                raise ValueError("All scheduled applications need valid probabilities")
            probabilities[name] = value
        keep = labels.loc[application_dates].to_numpy()
        scored = application_dates[keep]
        if not len(scored):
            continue
        actual = targets.loc[scored, "y"].to_numpy()
        for name in MODELS:
            prediction = probabilities[name][keep]
            rows.append(
                pd.DataFrame(
                    {
                        "origin": scored,
                        "model": name,
                        "horizon": 1,
                        "feature_cutoff_date": features.loc[
                            scored, "feature_cutoff_date"
                        ].to_numpy(),
                        "target_end": targets.loc[scored, "target_end"].to_numpy(),
                        "available_date": targets.loc[scored, "available_date"].to_numpy(),
                        "y": actual,
                        "probability": prediction,
                        "loss": brier_loss(prediction, actual),
                        "fit_origin": fit_entry,
                        "fit_cutoff_date": cutoff,
                        "train_n": int(mask.sum()),
                        "train_last_target": last_target,
                        "train_last_available": last_available,
                        "phase": np.where(scored <= dev_end, "development", "evaluation"),
                    }
                )
            )
    if not rows:
        raise ValueError("INSUFFICIENT_DATA: no scoreable common binary labels")
    panel = (
        pd.concat(rows, ignore_index=True)
        .sort_values(["origin", "model"])
        .reset_index(drop=True)
    )
    validate_panel(panel)
    return panel, fits
