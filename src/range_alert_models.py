"""Staged range-alert probabilities and a continuous causal event-rate control.

Only supplied tables are consumed. All fold, geometry and event support checks
precede optimization. New application dates never depend on future labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from . import causal_pool_models as rate
from . import range_alert_features as rf
from . import sign_memory_models as logistic
from .cross_moment_score import _product
from .index_hinge import _dates

MODELS = rf.MODELS
PANEL_COLUMNS = logistic.PANEL_COLUMNS
STATE_COLUMNS = rate.STATE_COLUMNS
ALPHA = logistic.ALPHA
GRADIENT_TOLERANCE = logistic.GRADIENT_TOLERANCE
MINIMUM_PER_CLASS = 50
PHASE_PER_CLASS = 30
SLICE_PER_CLASS = 15
MINIMUM_PHASE_OBSERVATIONS = 127
HALF_LIFE = rate.HALF_LIFE
DECAY = rate.DECAY
LABEL_WEIGHT = rate.LABEL_WEIGHT
_newton = logistic._newton
logistic_state = logistic.logistic_state


def _numeric(values, label, *, unknown=False):
    values = np.asarray(values)
    if values.dtype.kind not in "iuf":
        raise ValueError(f"Real numerical {label} required")
    values = values.astype(float)
    if np.isinf(values).any() or (not unknown and np.isnan(values).any()):
        raise ValueError(f"Finite {label} required; no row repair")
    return values


def _binary(values):
    values = _numeric(values, "binary labels")
    if values.ndim != 1 or not len(values) or not np.isin(values, [0.0, 1.0]).all():
        raise ValueError("Nonempty binary vector required")
    return values


def _support(values, minimum):
    values = _binary(values)
    events = int(np.count_nonzero(values == 1))
    nonevents = len(values) - events
    if events < minimum or nonevents < minimum:
        raise ValueError(
            f"INSUFFICIENT_DATA: {events} events/{nonevents} nonevents; need {minimum} each"
        )
    return {"n": len(values), "events": events, "nonevents": nonevents}


def _probabilities(value, expected=None):
    value = _numeric(value, "probabilities")
    if (
        value.ndim != 1
        or (expected is not None and value.shape != (expected,))
        or ((value < 0) | (value > 1)).any()
    ):
        raise ValueError("Aligned probabilities in [0,1] required")
    return value


def _logit(design, beta):
    _product(design, beta, "coefficient/design products")
    with np.errstate(all="ignore"):
        eta = design @ beta
    return _numeric(eta, "logits")


def _check_solver(beta, audit, x, y, *, offset=None, penalize_intercept=False):
    beta = _numeric(beta, "solver coefficients")
    if beta.shape != (x.shape[1],):
        raise ValueError("Exact fitted coefficient shape required")
    value, gradient, _ = logistic_state(
        beta, x, y, offset=offset, penalize_intercept=penalize_intercept
    )
    maximum = float(np.max(np.abs(gradient)))
    if (
        maximum > GRADIENT_TOLERANCE
        or audit.get("success") is not True
        or audit.get("objective") != value
        or audit.get("gradient_max_abs") != maximum
        or not np.array_equal(np.asarray(audit.get("gradient")), gradient)
        or isinstance(audit.get("iterations"), bool)
        or not isinstance(audit.get("iterations"), (int, np.integer))
        or not 0 <= audit["iterations"] <= logistic.MAX_ITERATIONS
    ):
        raise ValueError("Original full logistic gradient or solver audit failed")
    return beta


def fit_predict(train_features, y, apply_features):
    """Fit the baseline and one frozen-logit location correction on common rows."""
    if not isinstance(y, pd.Series) or not train_features.index.equals(y.index):
        raise ValueError("Exactly aligned training feature/label rows required")
    labels = _binary(y.to_numpy())
    support = _support(labels, MINIMUM_PER_CLASS)
    tr, app, centered, app_centered, transform = rf.transform(train_features, apply_features)
    x, query = _numeric(tr, "training design"), _numeric(app, "application design")
    z, query_z = (
        _numeric(centered, "centered location"),
        _numeric(app_centered, "query location"),
    )
    frequency = float(labels.mean())
    start = np.zeros(x.shape[1])
    start[0] = np.log(frequency / (1 - frequency))
    beta, baseline_solver = _newton(x, labels, start)
    beta = _check_solver(beta, baseline_solver, x, labels)
    train_eta, app_eta = _logit(x, beta), _logit(query, beta)
    coefficient, location_solver = _newton(
        z[:, None], labels, np.zeros(1), offset=train_eta, penalize_intercept=True
    )
    coefficient = _check_solver(
        coefficient,
        location_solver,
        z[:, None],
        labels,
        offset=train_eta,
        penalize_intercept=True,
    )
    b = float(coefficient[0])
    constant = transform["range_extremity_constant"]
    if constant and (b != 0 or not np.array_equal(z, np.zeros(len(z)))):
        raise ValueError("Exact constant location must retain canonical zero correction")
    # The original scalar objective supplies the audit; these checks preserve
    # the issued-logit arithmetic before evaluating endpoint-valid expit.
    _numeric(
        train_eta + _product(b, z, "training location correction"), "corrected training logits"
    )
    location_eta = _numeric(
        app_eta + _product(b, query_z, "application location correction"),
        "corrected application logits",
    )
    forecasts = {
        "baseline": _probabilities(expit(app_eta), len(app)),
        "location": _probabilities(expit(location_eta), len(app)),
    }
    common = {"train_n": len(labels), "application_n": len(app)}
    audit = {
        **common,
        "support": support,
        "transform": transform,
        "frequency": {"probability": frequency, **common},
        "baseline": {
            "columns": list(rf.BASE),
            "means": transform["means"],
            "scales": transform["scales"],
            "beta": beta.tolist(),
            "alpha": ALPHA,
            **common,
            **baseline_solver,
        },
        "location": {
            "column": "range_extremity",
            "mean": transform["range_extremity_mean"],
            "scale": 1.0,
            "b": b,
            "alpha": ALPHA,
            **common,
            "baseline_frozen": True,
            "status": "EXACT_CONSTANT_INPUT" if constant else "FITTED",
            **location_solver,
        },
    }
    return forecasts, audit


def _alignment(features, targets):
    if not isinstance(features, pd.DataFrame) or not isinstance(targets, pd.DataFrame):
        raise ValueError("Full feature and target tables required")
    _dates(features.index)
    if not len(features) or features.index[-1] > pd.Timestamp("2025-10-20"):
        raise ValueError("Nonempty bounded reference calendar required")
    if (
        not features.index.equals(targets.index)
        or tuple(features.columns) != (*rf.RAW, "feature_cutoff_date")
        or tuple(targets.columns) != ("y", "target_end", "available_date")
        or features.columns.has_duplicates
        or targets.columns.has_duplicates
    ):
        raise ValueError("Exact common full-calendar schemas required")
    raw = _numeric(features.loc[:, rf.RAW], "raw features", unknown=True)
    const = raw[:, 0]
    if not np.all(const[np.isfinite(const)] == 1):
        raise ValueError("Literal known intercept required")
    extremity = _numeric(features.range_extremity, "range extremity", unknown=True)
    if ((extremity < 0) | (extremity > 1)).any():
        raise ValueError("Known range extremity must lie in [0,1]")
    dates = pd.Series(features.index, index=features.index)
    for frame, column, shift in [
        (features, "feature_cutoff_date", 1),
        (targets, "target_end", -1),
        (targets, "available_date", -1),
    ]:
        if not frame[column].equals(dates.shift(shift).rename(column)):
            raise ValueError(
                "Exact previous-session cutoff and next-session maturity required"
            )
    _numeric(targets.y, "full labels", unknown=True)
    known = targets.y.notna()
    if known.any():
        _binary(targets.loc[known, "y"])
        if targets.loc[known, "available_date"].isna().any():
            raise ValueError("Known label needs next-session availability")


def training_mask(features, targets, fit_entry, min_train=1000):
    _alignment(features, targets)
    fit_entry = pd.Timestamp(fit_entry)
    if (
        fit_entry not in features.index
        or isinstance(min_train, bool)
        or not isinstance(min_train, (int, np.integer))
        or min_train < 1000
    ):
        raise ValueError("Observed fit entry and at least1000 minimum training rows required")
    cutoff = features.loc[fit_entry, "feature_cutoff_date"]
    if pd.isna(cutoff):
        raise ValueError("INSUFFICIENT_DATA: unknown prior-session cutoff")
    mask = (
        features.loc[:, rf.RAW].notna().all(axis=1)
        & features.feature_cutoff_date.notna()
        & targets.y.notna()
        & (features.index < fit_entry)
        & (targets.available_date <= cutoff)
    )
    if int(mask.sum()) < min_train:
        raise ValueError(f"INSUFFICIENT_DATA: {int(mask.sum())} common mature labels")
    _support(targets.loc[mask, "y"], MINIMUM_PER_CLASS)
    return mask


def application_calendar(features, targets, config):
    """Return feature-first applications, full-calendar label mask and dev end."""
    _alignment(features, targets)
    if (
        tuple(config["models"]) != MODELS
        or tuple(config["baseline"]) != rf.BASE
        or tuple(config["all_features"]) != rf.ALL_FEATURES
    ):
        raise ValueError("Fixed model and feature family differs")
    fixed = {
        "minimum_train_per_class": MINIMUM_PER_CLASS,
        "minimum_phase_per_class": PHASE_PER_CLASS,
        "minimum_slice_per_class": SLICE_PER_CLASS,
        "minimum_phase_observations": MINIMUM_PHASE_OBSERVATIONS,
    }
    if any(config.get(key, value) != value for key, value in fixed.items()):
        raise ValueError("Fixed class or phase support contract differs")
    start, end, latest = map(
        pd.Timestamp, [config["origin_start"], config["origin_end"], config["latest_target"]]
    )
    dev_start, dev_end = map(pd.Timestamp, config["development"])
    ev_start, ev_end = map(pd.Timestamp, config["evaluation"])
    bounds = [start, end, latest, dev_start, dev_end, ev_start, ev_end]
    if any(
        pd.isna(date) or date.tz is not None or date != date.normalize() for date in bounds
    ):
        raise ValueError("Normalized finite phase dates required")
    if (
        not start <= dev_start <= dev_end < ev_start <= ev_end <= end < latest
        or pd.Timestamp(config["development_target_available_by"]) != dev_end
        or end > pd.Timestamp("2025-10-17")
        or latest > pd.Timestamp("2025-10-20")
    ):
        raise ValueError("Invalid historical phase or target fence")
    dates = features.index
    phase = ((dates >= dev_start) & (dates <= dev_end)) | (
        (dates >= ev_start) & (dates <= ev_end)
    )
    ready = features.loc[:, rf.RAW].notna().all(axis=1) & features.feature_cutoff_date.notna()
    entries = dates[ready & phase & (dates >= start) & (dates <= end)]
    if not len(entries):
        raise ValueError("INSUFFICIENT_DATA: no common feature-complete applications")
    labels = (
        targets.y.notna()
        & (targets.available_date <= latest)
        & ((dates > dev_end) | (targets.available_date <= dev_end))
    )
    return entries, labels, dev_end


def _metadata(features, targets, applications, mask):
    fit_entry = applications[0]
    return {
        "fit_origin": str(fit_entry.date()),
        "fit_cutoff_date": str(features.loc[fit_entry, "feature_cutoff_date"].date()),
        "train_n": int(mask.sum()),
        "train_first_origin": str(features.index[mask][0].date()),
        "train_last_origin": str(features.index[mask][-1].date()),
        "train_last_target": str(targets.loc[mask, "target_end"].max().date()),
        "train_last_available": str(targets.loc[mask, "available_date"].max().date()),
        "application_n": len(applications),
    }


def preflight(features, targets, config):
    """Validate every fold and phase support before any optimizer can run."""
    entries, labels, _ = application_calendar(features, targets, config)
    scored = entries[labels.loc[entries].to_numpy()]
    fits = []
    for month in entries.to_period("M").unique():
        applications = entries[entries.to_period("M") == month]
        mask = training_mask(features, targets, applications[0], config["minimum_train"])
        _, _, _, _, transform = rf.transform(features.loc[mask], features.loc[applications])
        fits.append(
            {
                **_metadata(features, targets, applications, mask),
                "support": _support(targets.loc[mask, "y"], MINIMUM_PER_CLASS),
                "transform": transform,
            }
        )
    phases = []
    for name in ["development", "evaluation"]:
        start, end = map(pd.Timestamp, config[name])
        dates = scored[(scored >= start) & (scored <= end)]
        if len(dates) < MINIMUM_PHASE_OBSERVATIONS:
            raise ValueError(f"INSUFFICIENT_DATA: {name} needs127 observations")
        phase = {
            "name": name,
            "n": len(dates),
            "first_origin": str(dates[0].date()),
            "last_origin": str(dates[-1].date()),
            "support": _support(targets.loc[dates, "y"], PHASE_PER_CLASS),
            "slices": [],
        }
        if name == "evaluation":
            intervals = config["evaluation_stability"]
            if len(intervals) != 2:
                raise ValueError("Two fixed evaluation intervals required")
            covered = []
            for lower, upper in intervals:
                lower_date, upper_date = pd.Timestamp(lower), pd.Timestamp(upper)
                if lower_date > upper_date:
                    raise ValueError("Ordered evaluation slice required")
                subset = dates[(dates >= lower_date) & (dates <= upper_date)]
                covered.extend(subset)
                if not len(subset):
                    raise ValueError("INSUFFICIENT_DATA: empty evaluation slice")
                phase["slices"].append(
                    {
                        "start": lower,
                        "end": upper,
                        "n": len(subset),
                        "support": _support(targets.loc[subset, "y"], SLICE_PER_CLASS),
                    }
                )
            if not pd.DatetimeIndex(covered).equals(pd.DatetimeIndex(dates.to_numpy())):
                raise ValueError(
                    "Evaluation slices must partition all scored evaluation dates"
                )
        phases.append(phase)
    return {
        "common_application_origins": len(entries),
        "common_scored_origins": len(scored),
        "monthly_fits": len(fits),
        "fits": fits,
        "phases": phases,
    }


def brier_loss(probability, y):
    probability = _probabilities(probability)
    return logistic.brier_loss(probability, _binary(y))


def validate_panel(panel):
    if (
        not isinstance(panel, pd.DataFrame)
        or tuple(panel.columns) != PANEL_COLUMNS
        or set(panel.model) != set(MODELS)
    ):
        raise ValueError("Exact three-model range-alert panel required")
    _probabilities(panel.probability.to_numpy())
    _binary(panel.y.to_numpy())
    _numeric(panel.loss.to_numpy(), "saved Brier loss")
    _numeric(panel.horizon.to_numpy(), "horizons")
    counts = _numeric(panel.train_n.to_numpy(), "training counts")
    if (counts < 1000).any():
        raise ValueError("At least1000 training rows required for every issued model")
    renamed = panel.copy()
    renamed["model"] = renamed.model.map(
        {"baseline": "baseline", "recent_frequency": "frequency", "location": "memory"}
    )
    logistic.validate_panel(renamed)
    return panel


def _frequency_states(features, targets, applications, fits, scored):
    reference = features.index
    seed = fits[0]
    seed_fit = pd.Timestamp(seed["fit_origin"])
    seed_cutoff = pd.Timestamp(seed["fit_cutoff_date"])
    seed_available = pd.Timestamp(seed["train_last_available"])
    seed_probability = float(seed["model_audit"]["frequency"]["probability"])
    seed_position = reference.get_loc(seed_cutoff)
    numerator, denominator, cursor = seed_probability, 1.0, seed_position
    latest_available, updates = seed_available, 0
    source_fits = {
        pd.Timestamp(origin): pd.Timestamp(fit["fit_origin"])
        for fit in fits
        for origin in fit["application_origins"]
    }
    scored_set = set(scored)
    rows = []
    for origin in applications:
        cutoff_position = reference.get_loc(origin) - 1
        while cursor < cutoff_position:
            cursor += 1
            numerator = float(_product(DECAY, numerator, "decayed frequency numerator"))
            denominator = float(_product(DECAY, denominator, "decayed frequency mass"))
            label = targets.y.iloc[cursor - 1]
            if not pd.isna(label):
                contribution = float(
                    _product(LABEL_WEIGHT, label, "new binary event contribution")
                )
                numerator = float(
                    _numeric(numerator + contribution, "updated frequency numerator")
                )
                denominator = float(
                    _numeric(denominator + LABEL_WEIGHT, "updated frequency mass")
                )
                latest_available = reference[cursor]
                updates += 1
            rate._state_probability(numerator, denominator)
        probability = rate._state_probability(numerator, denominator)
        rows.append(
            {
                "origin": origin,
                "feature_cutoff_date": reference[cutoff_position],
                "source_fit_origin": source_fits[origin],
                "seed_fit_origin": seed_fit,
                "seed_cutoff_date": seed_cutoff,
                "seed_last_available": seed_available,
                "seed_probability": seed_probability,
                "seed_train_n": seed["train_n"],
                "S": numerator,
                "W": denominator,
                "recent_frequency": probability,
                "latest_consumed_available": latest_available,
                "cumulative_updates": updates,
                "elapsed_sessions": cutoff_position - seed_position,
                "scored": origin in scored_set,
            }
        )
    return pd.DataFrame(rows, columns=STATE_COLUMNS)


def forecast_panel(features, targets, config):
    preflight(features, targets, config)
    entries, labels, dev_end = application_calendar(features, targets, config)
    fits, rows = [], []
    for month in entries.to_period("M").unique():
        applications = entries[entries.to_period("M") == month]
        mask = training_mask(features, targets, applications[0], config["minimum_train"])
        predictions, audit = fit_predict(
            features.loc[mask], targets.loc[mask, "y"], features.loc[applications]
        )
        if set(predictions) != {"baseline", "location"}:
            raise ValueError("Both conditional probability arms required at every fit")
        predictions = {
            name: _probabilities(values, len(applications))
            for name, values in predictions.items()
        }
        metadata = _metadata(features, targets, applications, mask)
        fits.append(
            {
                **metadata,
                "model_audit": audit,
                "application_origins": [str(date.date()) for date in applications],
                "application_probabilities": {
                    name: values.tolist() for name, values in predictions.items()
                },
            }
        )
        keep = labels.loc[applications].to_numpy()
        scored = applications[keep]
        if not len(scored):
            continue
        actual = targets.loc[scored, "y"].to_numpy()
        for name, probability in predictions.items():
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
                        "probability": probability[keep],
                        "loss": brier_loss(probability[keep], actual),
                        "fit_origin": pd.Timestamp(metadata["fit_origin"]),
                        "fit_cutoff_date": pd.Timestamp(metadata["fit_cutoff_date"]),
                        "train_n": metadata["train_n"],
                        "train_last_target": pd.Timestamp(metadata["train_last_target"]),
                        "train_last_available": pd.Timestamp(metadata["train_last_available"]),
                        "phase": np.where(scored <= dev_end, "development", "evaluation"),
                    }
                )
            )
    if not rows:
        raise ValueError("INSUFFICIENT_DATA: no scored conditional probabilities")
    scored_entries = entries[labels.loc[entries].to_numpy()]
    states = _frequency_states(features, targets, entries, fits, scored_entries)
    conditional = pd.concat(rows, ignore_index=True)
    baseline = conditional.loc[conditional.model == "baseline"].sort_values("origin").copy()
    baseline["model"] = "recent_frequency"
    baseline["probability"] = (
        states.set_index("origin").loc[scored_entries, "recent_frequency"].to_numpy()
    )
    baseline["loss"] = brier_loss(baseline.probability.to_numpy(), baseline.y.to_numpy())
    panel = (
        pd.concat([conditional, baseline], ignore_index=True)
        .loc[:, PANEL_COLUMNS]
        .sort_values(["origin", "model"])
        .reset_index(drop=True)
    )
    validate_panel(panel)
    return panel, fits, states
