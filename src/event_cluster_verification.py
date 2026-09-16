"""Independent supplied-table replay of every event-cluster pipeline output.

Only independent new primitives and frozen independent old-record helpers are
used. This module loads no files, writes no proof and refits no original model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import verify_event_cluster as stage
from . import verify_issued_calibration as issued
from . import verify_range_alert as old

MODELS = ("baseline", "recent_frequency", "nuisance", "cluster")
PANEL_COLUMNS = old.PANEL_COLUMNS
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
) + tuple(name for name in stage.MEMORY_COLUMNS if name != "feature_cutoff_date")


def _frame(actual, expected, label):
    if not isinstance(actual, pd.DataFrame):
        raise ValueError(label + " must be a DataFrame")
    try:
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    except AssertionError as error:
        raise AssertionError(label + " differs from independent reconstruction") from error


def _panel(panel, upstream, scored):
    if (
        not isinstance(panel, pd.DataFrame)
        or tuple(panel.columns) != PANEL_COLUMNS
        or set(panel.model) != set(MODELS)
        or panel.duplicated(["origin", "model"]).any()
        or len(panel) != 4 * len(scored)
    ):
        raise AssertionError("Exact four-model scored panel required")
    for name in MODELS:
        rows = panel.loc[panel.model.eq(name)].sort_values("origin").reset_index(drop=True)
        if not pd.DatetimeIndex(rows.origin).equals(scored):
            raise AssertionError("Complete original scored cohort required for " + name)
        if name in ("baseline", "recent_frequency"):
            expected = (
                upstream.loc[upstream.model.eq(name)]
                .sort_values("origin")
                .reset_index(drop=True)
            )
            _frame(rows, expected, "unchanged original " + name + " rows")
    for name in ("nuisance", "cluster"):
        selected = panel.loc[panel.model.isin(["baseline", "recent_frequency", name])].copy()
        selected["model"] = selected.model.replace({name: "location"})
        old.validate_panel(selected)


def verify_pipeline(
    features,
    targets,
    upstream_panel,
    upstream_fits,
    upstream_states,
    original_index_config,
    panel,
    newfits,
    newstates,
    memory,
    support,
):
    """Verify exact histories/cohorts/records, then independently solve each stage."""
    records, oldsupport = issued.replay_issued(
        features,
        targets,
        upstream_panel,
        upstream_fits,
        upstream_states,
        original_index_config,
    )
    applications, scored = old.eligible_entries(features, targets, original_index_config)
    rebuilt = stage.reconstruct_memory(targets)
    _frame(memory, rebuilt, "full memory including unknowns and structural dates")
    if not rebuilt.feature_cutoff_date.equals(features.feature_cutoff_date):
        raise AssertionError("Exact preceding-session source cutoff required")
    _panel(panel, upstream_panel, scored)
    if type(newfits) is not list or len(newfits) != len(upstream_fits):
        raise AssertionError("Every original application month needs one new staged fit")
    if (
        not isinstance(newstates, pd.DataFrame)
        or tuple(newstates.columns) != STATE_COLUMNS
        or not pd.DatetimeIndex(newstates.origin).equals(applications)
    ):
        raise AssertionError("All original applications and exact new state schema required")
    issued_records = {pd.Timestamp(r["origin"]): r for r in records}
    rate = upstream_states.set_index("origin")["recent_frequency"]
    jobs, support_fits, expected_states = [], [], []
    # Structural checks for all months precede any new independent optimizer.
    for source, saved in zip(upstream_fits, newfits, strict=True):
        entry = pd.Timestamp(source["fit_origin"])
        dates = applications[applications.to_period("M") == entry.to_period("M")]
        mask = old.training_mask(features, targets, entry)
        stage.require_original_histories(rebuilt, features.index[mask], dates)
        metadata = old.metadata(features, targets, dates, mask)
        expected_keys = set(metadata) | {
            "source_fit_origin",
            "application_origins",
            "application_probabilities",
            "model_audit",
        }
        if type(saved) is not dict or set(saved) != expected_keys:
            raise AssertionError("Exact new monthly fit schema required")
        for name, value in {
            **metadata,
            "source_fit_origin": source["fit_origin"],
            "application_origins": [str(d.date()) for d in dates],
        }.items():
            stage.same_tree(saved[name], value, "new monthly " + name)
        predictions = saved["application_probabilities"]
        if type(predictions) is not dict or set(predictions) != set(MODELS):
            raise AssertionError("All four full-application probability arrays required")
        predictions = {
            name: stage.probabilities(values, len(dates), name)
            for name, values in predictions.items()
        }
        eta = np.array([issued_records[d]["baseline_logit"] for d in dates])
        baseline = np.array([issued_records[d]["baseline_probability"] for d in dates])
        frequency = rate.loc[dates].to_numpy()
        for name, expected in (("baseline", baseline), ("recent_frequency", frequency)):
            if not np.array_equal(predictions[name], expected):
                raise AssertionError("Original application probability bits changed: " + name)
        train_eta, _ = stage.independent_baseline_logits(
            features.loc[mask], source["model_audit"]
        )
        train, apply = rebuilt.loc[mask, stage.FEATURES], rebuilt.loc[dates, stage.FEATURES]
        stage.independent_transform(train, apply)
        support_fits.append(
            {
                **metadata,
                "source_fit_origin": source["fit_origin"],
                "history_columns": list(stage.FEATURES),
                "training_history_rows": len(train),
                "application_history_rows": len(apply),
            }
        )
        chosen = dates[dates.isin(scored)]
        positions = dates.get_indexer(chosen)
        for name in ("nuisance", "cluster"):
            row = panel.loc[panel.model.eq(name)].set_index("origin").loc[chosen]
            if not np.array_equal(row.probability.to_numpy(), predictions[name][positions]):
                raise AssertionError("New scored/application probability bits differ")
            losses = (
                old.brier_loss(
                    predictions[name][positions], targets.loc[chosen, "y"].to_numpy()
                )
                if len(chosen)
                else np.array([])
            )
            if not np.array_equal(row.loss.to_numpy(), losses):
                raise AssertionError("Exact independent Brier losses required")
        for position, date in enumerate(dates):
            state = {
                "origin": date,
                "feature_cutoff_date": features.loc[date, "feature_cutoff_date"],
                "source_fit_origin": entry,
                "baseline_logit": float(eta[position]),
                "baseline_probability": float(baseline[position]),
                "recent_frequency": float(frequency[position]),
                "nuisance_probability": float(predictions["nuisance"][position]),
                "cluster_probability": float(predictions["cluster"][position]),
                "scored": date in scored,
            }
            state.update(
                {
                    name: rebuilt.loc[date, name]
                    for name in stage.MEMORY_COLUMNS
                    if name != "feature_cutoff_date"
                }
            )
            expected_states.append(state)
        jobs.append(
            {
                "fit_origin": source["fit_origin"],
                "train": train,
                "y": targets.loc[mask, "y"],
                "apply": apply,
                "train_eta": train_eta,
                "apply_eta": eta,
                "baseline": baseline,
                "predictions": {name: predictions[name] for name in ("nuisance", "cluster")},
                "audit": saved["model_audit"],
            }
        )
    expected_support = {
        "status": "COMPLETE_ORIGINAL_COHORT_WITH_EVENT_HISTORY",
        "common_application_origins": len(applications),
        "common_scored_origins": len(scored),
        "monthly_fits": len(upstream_fits),
        "history_columns": list(stage.FEATURES),
        "original_support": oldsupport,
        "fits": support_fits,
        "unknown_history_policy": "whole_trial_failure_no_row_mask",
    }
    stage.same_tree(support, expected_support, "complete new and original support")
    expected = pd.DataFrame(expected_states, columns=STATE_COLUMNS)
    stage.close(
        newstates.baseline_logit.to_numpy(),
        expected.baseline_logit.to_numpy(),
        "independent issued state logits",
    )
    _frame(
        newstates.drop(columns="baseline_logit"),
        expected.drop(columns="baseline_logit"),
        "every saved application state",
    )
    proofs = []
    for job in jobs:
        proof = stage.verify_stages(
            job["train"],
            job["y"],
            job["apply"],
            job["train_eta"],
            job["apply_eta"],
            job["baseline"],
            job["predictions"],
            job["audit"],
        )
        proofs.append({"fit_origin": job["fit_origin"], "audit": proof})
    return {
        "forecasts_verified": len(panel),
        "new_forecasts": 2 * len(scored),
        "reused_control_forecasts": 2 * len(scored),
        "new_monthly_fits": len(newfits),
        "original_monthly_fits_replayed": len(upstream_fits),
        "independent_stage_fits_verified": 2 * len(newfits),
        "common_scored_origins": len(scored),
        "common_application_origins": len(applications),
        "application_states_verified": len(expected),
        "memory_rows_verified": len(rebuilt),
        "monthly_stage_audits": proofs,
    }
