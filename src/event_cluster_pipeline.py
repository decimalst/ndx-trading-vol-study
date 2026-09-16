"""Pure-table event-cluster orchestration on the exact original alert cohort."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import event_cluster_features as feature
from . import event_cluster_models as models
from . import issued_calibration_models as issued_model
from . import range_alert_models as original

MODELS = ("baseline", "recent_frequency", "nuisance", "cluster")
PANEL_COLUMNS = original.PANEL_COLUMNS
STATE_COLUMNS = (
    "origin",
    "feature_cutoff_date",
    "source_fit_origin",
    "baseline_logit",
    "baseline_probability",
    "recent_frequency",
    "nuisance_probability",
    "cluster_probability",
    "scored",
) + tuple(name for name in feature.MEMORY_COLUMNS if name != "feature_cutoff_date")


def validate_panel(panel):
    """Check both new arms against the same unchanged original two controls."""
    if (
        not isinstance(panel, pd.DataFrame)
        or tuple(panel.columns) != PANEL_COLUMNS
        or set(panel.model) != set(MODELS)
        or panel.duplicated(["origin", "model"]).any()
        or not panel.groupby("origin", sort=False).size().eq(4).all()
    ):
        raise ValueError("Exact complete four-model event-cluster panel required")
    for candidate in ("nuisance", "cluster"):
        selected = panel.loc[
            panel.model.isin(["baseline", "recent_frequency", candidate])
        ].copy()
        selected["model"] = selected.model.replace({candidate: "location"})
        original.validate_panel(selected)
    return panel


def _prepare(features, targets, upstream_panel, upstream_fits, upstream_states, config):
    issued, original_support = issued_model.replay_issued(
        features, targets, upstream_panel, upstream_fits, upstream_states, config
    )
    memory = feature.build_memory(targets)
    if (
        not memory.index.equals(features.index)
        or tuple(memory.columns) != feature.MEMORY_COLUMNS
        or not memory.feature_cutoff_date.equals(features.feature_cutoff_date)
    ):
        raise ValueError("Exact full-calendar memory and preceding cutoff required")
    issued_by_date = {pd.Timestamp(row["origin"]): row for row in issued}
    if len(issued_by_date) != len(issued):
        raise ValueError("Unique original application records required")
    frequency = upstream_states.set_index("origin")["recent_frequency"]
    jobs, history_fits = [], []
    # This complete pass must finish before the first new optimizer. A single
    # unknown history invalidates the unchanged cohort; it never defines a mask.
    for source in upstream_fits:
        dates = pd.DatetimeIndex(source["application_origins"])
        entry = pd.Timestamp(source["fit_origin"])
        mask = original.training_mask(features, targets, entry, config["minimum_train"])
        train = memory.loc[mask, feature.FEATURES]
        apply = memory.loc[dates, feature.FEATURES]
        for frame in (train, apply):
            values = frame.to_numpy()
            if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
                raise ValueError(
                    "INSUFFICIENT_DATA: original cohort has unknown event history"
                )
        # All bounded transformations and saved training-offset computations
        # are validated during preflight, without fitting a new baseline.
        models.transform(train, apply)
        train_eta, _ = models.baseline_logits(features.loc[mask], source["model_audit"])
        query_eta = np.array([issued_by_date[d]["baseline_logit"] for d in dates])
        baseline = np.array([issued_by_date[d]["baseline_probability"] for d in dates])
        recent = original._probabilities(frequency.loc[dates].to_numpy(), len(dates))
        metadata = original._metadata(features, targets, dates, mask)
        for key, value in metadata.items():
            if source[key] != value:
                raise ValueError("Original training/application cohort changed: " + key)
        history_fits.append(
            {
                **metadata,
                "source_fit_origin": source["fit_origin"],
                "history_columns": list(feature.FEATURES),
                "training_history_rows": len(train),
                "application_history_rows": len(apply),
            }
        )
        jobs.append(
            {
                "train": train,
                "y": targets.loc[mask, "y"],
                "apply": apply,
                "train_eta": train_eta,
                "query_eta": query_eta,
                "baseline": baseline,
                "recent": recent,
                "metadata": metadata,
                "source": source,
            }
        )
    support = {
        "status": "COMPLETE_ORIGINAL_COHORT_WITH_EVENT_HISTORY",
        "common_application_origins": original_support["common_application_origins"],
        "common_scored_origins": original_support["common_scored_origins"],
        "monthly_fits": original_support["monthly_fits"],
        "history_columns": list(feature.FEATURES),
        "original_support": original_support,
        "fits": history_fits,
        "unknown_history_policy": "whole_trial_failure_no_row_mask",
    }
    return jobs, issued_by_date, memory, support


def forecast_panel(
    features, targets, upstream_panel, upstream_fits, upstream_states, original_index_config
):
    """Return panel, new fits, all application states, full memory and support.

    Inputs are supplied immutable original tables. Training offsets replay the
    current month's saved baseline fit on its exact mature training cohort.
    Query offsets and saved probabilities come from original issued records.
    """
    jobs, issued, memory, support = _prepare(
        features,
        targets,
        upstream_panel,
        upstream_fits,
        upstream_states,
        original_index_config,
    )
    controls = upstream_panel.loc[
        upstream_panel.model.isin(["baseline", "recent_frequency"])
    ].copy()
    new_rows, fits, states = [], [], []
    for job in jobs:
        predictions, audit = models.fit_stages(
            job["train"],
            job["y"],
            job["apply"],
            job["train_eta"],
            job["query_eta"],
            job["baseline"],
        )
        if type(predictions) is not dict or set(predictions) != {"nuisance", "cluster"}:
            raise ValueError("Every application requires both new model probabilities")
        dates = job["apply"].index
        probabilities = {
            "baseline": job["baseline"],
            "recent_frequency": job["recent"],
            **{
                name: original._probabilities(p, len(dates)) for name, p in predictions.items()
            },
        }
        metadata = job["metadata"]
        source = job["source"]
        fits.append(
            {
                **metadata,
                "source_fit_origin": source["fit_origin"],
                "application_origins": list(source["application_origins"]),
                "application_probabilities": {
                    name: p.tolist() for name, p in probabilities.items()
                },
                "model_audit": audit,
            }
        )
        # Validate/save every application first. Query-label selection never
        # controls fitting, memory admission, saved probabilities or fit records.
        for position, date in enumerate(dates):
            state = {
                "origin": date,
                "feature_cutoff_date": pd.Timestamp(issued[date]["feature_cutoff_date"]),
                "source_fit_origin": pd.Timestamp(source["fit_origin"]),
                "baseline_logit": float(job["query_eta"][position]),
                "baseline_probability": float(probabilities["baseline"][position]),
                "recent_frequency": float(probabilities["recent_frequency"][position]),
                "nuisance_probability": float(probabilities["nuisance"][position]),
                "cluster_probability": float(probabilities["cluster"][position]),
                "scored": issued[date]["scored"],
            }
            state.update(
                {
                    name: memory.loc[date, name]
                    for name in feature.MEMORY_COLUMNS
                    if name != "feature_cutoff_date"
                }
            )
            states.append(state)
        template = controls.loc[controls.model.eq("baseline") & controls.origin.isin(dates)]
        positions = dates.get_indexer(pd.DatetimeIndex(template.origin))
        if (positions < 0).any():
            raise ValueError("Scored row was not an original application")
        for name in ("nuisance", "cluster"):
            if not len(template):
                continue
            row = template.copy()
            row["model"] = name
            row["probability"] = probabilities[name][positions]
            row["loss"] = original.brier_loss(row.probability.to_numpy(), row.y.to_numpy())
            new_rows.append(row)
    if not new_rows:
        raise ValueError("INSUFFICIENT_DATA: no scored new forecasts")
    panel = pd.concat([controls, *new_rows], ignore_index=True).loc[:, PANEL_COLUMNS]
    panel = panel.sort_values(["origin", "model"]).reset_index(drop=True)
    validate_panel(panel)
    states = pd.DataFrame(states, columns=STATE_COLUMNS)
    if (
        len(states) != support["common_application_origins"]
        or len(fits) != support["monthly_fits"]
        or int(states.scored.sum()) != support["common_scored_origins"]
        or len(panel) != 4 * support["common_scored_origins"]
    ):
        raise ValueError("Original complete application/scored counts changed")
    return panel, fits, states, memory, support
