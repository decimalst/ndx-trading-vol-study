"""Causal event-rate pooling of immutable, already issued binary forecasts.

Only supplied tables are consumed. No source loading, baseline fitting, or
retrospective calibration occurs here. Application dates precede label masks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import sign_memory_models as original
from .cross_moment_score import _finite, _product
from .index_hinge import _dates

MODELS = ("frozen_baseline", "recent_frequency", "pooled")
PANEL_COLUMNS = original.PANEL_COLUMNS
HALF_LIFE = 63
DECAY = 2.0 ** (-1.0 / HALF_LIFE)
LABEL_WEIGHT = 1.0 - DECAY
POOL_WEIGHT = 0.5
STATE_COLUMNS = (
    "origin",
    "feature_cutoff_date",
    "source_fit_origin",
    "seed_fit_origin",
    "seed_cutoff_date",
    "seed_last_available",
    "seed_probability",
    "seed_train_n",
    "S",
    "W",
    "recent_frequency",
    "latest_consumed_available",
    "cumulative_updates",
    "elapsed_sessions",
    "scored",
)


def checked_divide(numerator, denominator):
    numerator = _finite(numerator, "division numerator")
    denominator = _finite(denominator, "division denominator")
    if np.any(denominator == 0):
        raise ValueError("Zero denominator is not an admissible state")
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        result = numerator / denominator
    _finite(result, "division result")
    if np.any((numerator != 0) & (result == 0)):
        raise ValueError("Nonzero quotient underflow; no zero replacement")
    return result


def _probability(values):
    values = _finite(values, "probabilities")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Probabilities must be in [0,1]")
    return values


def convex_pool(baseline, frequency):
    baseline, frequency = _probability(baseline), _probability(frequency)
    if baseline.shape != frequency.shape:
        raise ValueError("Aligned probabilities required for pooling")
    left = _product(POOL_WEIGHT, baseline, "half-weight baseline probability")
    right = _product(POOL_WEIGHT, frequency, "half-weight frequency probability")
    return _probability(left + right)


def _state_probability(numerator, denominator):
    numerator, denominator = (
        float(_finite(numerator, "state numerator")),
        float(_finite(denominator, "state denominator")),
    )
    if not 0 <= numerator <= denominator <= 1 or denominator == 0:
        raise ValueError("State must satisfy 0 <= S <= W <= 1 and W > 0")
    return float(_probability(checked_divide(numerator, denominator)))


def validate_panel(panel):
    """Check paired metadata/Brier plus the exact fixed convex-pool identity."""
    if (
        not isinstance(panel, pd.DataFrame)
        or tuple(panel.columns) != PANEL_COLUMNS
        or set(panel.model) != set(MODELS)
    ):
        raise ValueError("Exact three-model causal-pool panel required")
    renamed = panel.copy()
    renamed["model"] = renamed.model.map(
        {"frozen_baseline": "baseline", "recent_frequency": "frequency", "pooled": "memory"}
    )
    original.validate_panel(renamed)
    by_model = {
        name: panel.loc[panel.model == name].set_index("origin").sort_index()
        for name in MODELS
    }
    pooled = convex_pool(
        by_model["frozen_baseline"].probability.to_numpy(),
        by_model["recent_frequency"].probability.to_numpy(),
    )
    if not np.array_equal(pooled, by_model["pooled"].probability.to_numpy()):
        raise ValueError("Issued pool differs from exact separate half-weight sum")
    return panel


def _integer(value, minimum, label):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"Integral {label} at least {minimum} required")
    return int(value)


def _date(value, reference, label):
    date = pd.Timestamp(value)
    if (
        pd.isna(date)
        or date.tz is not None
        or date != date.normalize()
        or date not in reference
    ):
        raise ValueError(f"Observed normalized {label} required")
    return date


def _validate_inputs(panel, targets, reference, fits, config, applications):
    _dates(reference)
    _dates(applications)
    if not len(reference) or reference[-1] > pd.Timestamp("2025-10-20"):
        raise ValueError("Nonempty bounded reference calendar required")
    if (
        not isinstance(targets, pd.DataFrame)
        or tuple(targets.columns) != ("y", "target_end", "available_date")
        or not targets.index.equals(reference)
    ):
        raise ValueError("Exact full-calendar target table required")
    dates = pd.Series(reference, index=reference)
    for name in ["target_end", "available_date"]:
        if not targets[name].equals(dates.shift(-1).rename(name)):
            raise ValueError("Targets must mature at exactly the next reference close")
    known = targets.y.notna()
    if known.any():
        values = _finite(targets.loc[known, "y"], "known binary labels")
        if (
            not np.isin(values, [0.0, 1.0]).all()
            or targets.loc[known, "available_date"].isna().any()
        ):
            raise ValueError("Known binary labels require finite next-session availability")
    original.validate_panel(panel)
    if tuple(config["models"]) != MODELS:
        raise ValueError("Fixed three-model family differs")
    start, end, latest = map(
        pd.Timestamp, [config["origin_start"], config["origin_end"], config["latest_target"]]
    )
    dev_start, dev_end = map(pd.Timestamp, config["development"])
    ev_start, ev_end = map(pd.Timestamp, config["evaluation"])
    if (
        not start <= dev_start <= dev_end < ev_start <= ev_end <= end < latest
        or config["development_target_available_by"] != config["development"][1]
        or end > pd.Timestamp("2025-10-17")
        or latest > pd.Timestamp("2025-10-20")
    ):
        raise ValueError("Fixed historical phases and maturity fences required")
    if (
        not len(applications)
        or not applications.isin(reference).all()
        or applications[0] == reference[0]
    ):
        raise ValueError("Nonempty observed applications need prior-session cutoffs")
    allowed = ((applications >= dev_start) & (applications <= dev_end)) | (
        (applications >= ev_start) & (applications <= ev_end)
    )
    if not allowed.all() or applications[0] < start or applications[-1] > end:
        raise ValueError("Applications must remain in the original phase windows")
    eligible = targets.loc[applications, "y"].notna() & (
        targets.loc[applications, "available_date"] <= latest
    )
    eligible &= (applications > dev_end) | (
        targets.loc[applications, "available_date"] <= dev_end
    )
    scored = applications[eligible]
    baseline = (
        panel.loc[panel.model == "baseline"]
        .copy()
        .sort_values("origin")
        .reset_index(drop=True)
    )
    if not pd.DatetimeIndex(baseline.origin).equals(scored):
        raise ValueError(
            "Original scored cohort must match all eligible original applications"
        )
    positions = reference.get_indexer(scored)
    expected = {
        "feature_cutoff_date": reference[positions - 1].to_numpy(),
        "target_end": targets.loc[scored, "target_end"].to_numpy(),
        "available_date": targets.loc[scored, "available_date"].to_numpy(),
        "y": targets.loc[scored, "y"].to_numpy(),
        "phase": np.where(scored <= dev_end, "development", "evaluation"),
    }
    for name, value in expected.items():
        if not np.array_equal(baseline[name].to_numpy(), value):
            raise ValueError(f"Original forecast {name} differs from full-calendar targets")
    months = applications.to_period("M").unique()
    if not isinstance(fits, list) or len(fits) != len(months):
        raise ValueError("Every original monthly application fit must be retained")
    minimum = _integer(config["minimum_train"], 100, "minimum training count")
    if config["minimum_train_per_class"] != 50:
        raise ValueError("Original fixed class-support requirement differs")
    source_fits = {}
    for month, fit in zip(months, fits, strict=True):
        app = applications[applications.to_period("M") == month]
        entry = _date(fit["fit_origin"], reference, "fit origin")
        cutoff = _date(fit["fit_cutoff_date"], reference, "fit cutoff")
        if entry != app[0] or cutoff != reference[reference.get_loc(entry) - 1]:
            raise ValueError("Original first monthly application/cutoff differs")
        count = _integer(fit["application_n"], 1, "application count")
        n = _integer(fit["train_n"], minimum, "original training count")
        first_origin = _date(fit["train_first_origin"], reference, "first training origin")
        last_origin = _date(fit["train_last_origin"], reference, "last training origin")
        last_target = _date(fit["train_last_target"], reference, "last training target")
        last_available = _date(
            fit["train_last_available"], reference, "last training availability"
        )
        if (
            count != len(app)
            or not first_origin <= last_origin < entry
            or last_available != last_target
            or last_available > cutoff
            or targets.loc[last_origin, "available_date"] != last_available
            or n > reference.get_loc(entry) - 1
        ):
            raise ValueError("Original monthly training/application metadata differs")
        audit = fit["model_audit"]
        frequency_audit = audit["frequency"]
        probability = float(_probability(frequency_audit["probability"]))
        if not 0 < probability < 1:
            raise ValueError("Seed/source frequency must retain both original classes")
        if (
            audit["train_n"] != n
            or audit["application_n"] != count
            or frequency_audit["train_n"] != n
            or frequency_audit["application_n"] != count
        ):
            raise ValueError("Original frequency fitting metadata differs")
        expected_fit = {
            "fit_origin": entry,
            "fit_cutoff_date": cutoff,
            "train_n": n,
            "train_last_target": last_target,
            "train_last_available": last_available,
        }
        old_month = panel.loc[panel.origin.dt.to_period("M") == month]
        for name, value in expected_fit.items():
            if not old_month[name].eq(value).all():
                raise ValueError(f"Original issued {name} and fitting audit differ")
        if (
            not old_month.loc[old_month.model == "frequency", "probability"]
            .eq(probability)
            .all()
        ):
            raise ValueError(
                "Original issued frequency differs from frozen fitting probability"
            )
        for origin in app:
            source_fits[origin] = entry
    return baseline, scored, source_fits


def pool_panel(
    original_forecasts, full_targets, reference, fits, index_config, *, application_origins
):
    """Return three paired scored arms and one state row per original application.

    The caller supplies the independently admitted original application dates;
    scored rows alone cannot identify an application with an unknown label.
    """
    baseline, scored, source_fits = _validate_inputs(
        original_forecasts, full_targets, reference, fits, index_config, application_origins
    )
    seed = fits[0]
    seed_fit_origin = pd.Timestamp(seed["fit_origin"])
    seed_cutoff = pd.Timestamp(seed["fit_cutoff_date"])
    seed_available = pd.Timestamp(seed["train_last_available"])
    seed_probability = float(seed["model_audit"]["frequency"]["probability"])
    seed_position = reference.get_loc(seed_cutoff)
    cursor = seed_position
    numerator, denominator = seed_probability, 1.0
    latest_available = seed_available
    updates = 0
    labels = full_targets.y.to_numpy()
    scored_set = set(scored)
    rows = []
    for origin in application_origins:
        cutoff_position = reference.get_loc(origin) - 1
        while cursor < cutoff_position:
            cursor += 1
            numerator = float(_product(DECAY, numerator, "decayed event numerator"))
            denominator = float(_product(DECAY, denominator, "decayed event denominator"))
            # Full-calendar target j matures at reference close j+1. Never
            # restrict this arrival stream to previously scored applications.
            label = labels[cursor - 1]
            if not pd.isna(label):
                contribution = float(_product(LABEL_WEIGHT, label, "new finite binary label"))
                numerator = float(_finite(numerator + contribution, "updated event numerator"))
                denominator = float(
                    _finite(denominator + LABEL_WEIGHT, "updated event denominator")
                )
                latest_available = reference[cursor]
                updates += 1
            _state_probability(numerator, denominator)
        probability = _state_probability(numerator, denominator)
        rows.append(
            {
                "origin": origin,
                "feature_cutoff_date": reference[cutoff_position],
                "source_fit_origin": source_fits[origin],
                "seed_fit_origin": seed_fit_origin,
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
    states = pd.DataFrame(rows, columns=STATE_COLUMNS)
    recent = states.set_index("origin").loc[scored, "recent_frequency"].to_numpy()
    issued = baseline.probability.to_numpy()
    probabilities = {
        "frozen_baseline": issued,
        "recent_frequency": recent,
        "pooled": convex_pool(issued, recent),
    }
    panels = []
    for name in MODELS:
        one = baseline.copy()
        one["model"] = name
        if name != "frozen_baseline":
            one["probability"] = probabilities[name]
            one["loss"] = original.brier_loss(probabilities[name], one.y.to_numpy())
        panels.append(one)
    panel = (
        pd.concat(panels, ignore_index=True)
        .sort_values(["origin", "model"])
        .reset_index(drop=True)
    )
    validate_panel(panel)
    return panel, states
