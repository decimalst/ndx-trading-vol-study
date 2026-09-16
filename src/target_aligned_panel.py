"""Causal scalar fits on fixed, previously issued mean and variance geometry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import cross_moment_score as scores
from . import joint_risk_density as density
from . import joint_risk_models as joint
from . import target_aligned_models as scalar

ROOT = Path(__file__).resolve().parents[1]
NEW_MODELS = ("aligned_constant", "aligned_dynamic")
MODELS = (*scores.MODELS, *NEW_MODELS)
COLUMNS = scores.INPUT_COLUMNS + scores.ADDED_COLUMNS


def _array(value, shape, label):
    if np.iscomplexobj(value):
        raise ValueError("Real frozen " + label + " required")
    value = np.asarray(value, float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError("Invalid frozen " + label)
    return value


def _same(actual, expected, label):
    if not np.allclose(actual, expected, rtol=1e-10, atol=1e-12, equal_nan=False):
        raise ValueError("Frozen " + label + " mismatch")


def replay_frozen_moments(training, targets, application, audit):
    if not training.index.equals(targets.index):
        raise ValueError("Training targets must have exact original row order")
    if audit["residual_staging"] != "current_fit_training_residuals":
        raise ValueError("Original current-fit training residual convention required")
    tr, ap, transformation = joint.transform(training, application)
    _same(
        audit["transform"]["corr22_mean"],
        transformation["corr22_mean"],
        "training corr22 center",
    )
    scale = float(audit["corr22_scale"])
    computed = float(tr.corr22.std(ddof=0))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("Positive original corr22 training scale required")
    _same(scale, computed, "training corr22 scale")
    combined = pd.concat([tr, ap]).to_numpy(float)
    n = len(training)
    width = len(joint.MARGINAL_COLUMNS)
    residual = []
    training_h = []
    application_mu = []
    application_h = []
    for asset in ["qqq", "spx"]:
        mean, variance = audit["moments"][asset]["mean"], audit["moments"][asset]["variance"]
        for record in [mean, variance]:
            if (
                tuple(record["columns"]) != joint.MARGINAL_COLUMNS
                or record["train_n"] != n
                or record["alpha"] != 0.01
            ):
                raise ValueError(
                    "Original full marginal coefficient schema and same training rows required"
                )
        centers = _array(mean["means"], (width,), "mean centers")
        scales = _array(mean["scales"], (width,), "mean scales")
        beta = _array(mean["beta"], (width,), "mean coefficients")
        if centers[0] != 0 or scales[0] != 1 or (scales <= 1e-12).any():
            raise ValueError("Original positive mean scales/intercept geometry required")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            mu = ((combined - centers) / scales) @ beta
        mu = scores._finite(mu, "replayed shared mean")
        varcenters = _array(variance["means"], (width - 1,), "variance centers")
        varscales = _array(variance["scales"], (width - 1,), "variance scales")
        varbeta = _array(variance["scaled_beta"], (width,), "scaled variance coefficients")
        unit = float(variance["train_mean"])
        if not np.isfinite(unit) or unit <= 0 or (varscales <= 1e-12).any():
            raise ValueError("Original positive variance geometry and training unit required")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            design = np.c_[np.ones(len(combined)), (combined[:, 1:] - varcenters) / varscales]
            h = np.exp(np.log(unit) + design @ varbeta)
        scores._finite(design, "replayed variance design")
        scores._finite(h, "replayed shared variance")
        if (h <= 0).any():
            raise ValueError("Positive original marginal moment required; no underflow repair")
        y = _array(targets["y_" + asset].to_numpy(), (n,), "signed training target")
        residual.append(scores._subtract(y, mu[:n], "current-fit training residual"))
        training_h.append(h[:n])
        application_mu.append(mu[n:])
        application_h.append(h[n:])
    return {
        "residual": np.column_stack(residual),
        "training_h": np.column_stack(training_h),
        "application_mu": np.column_stack(application_mu),
        "application_h": np.column_stack(application_h),
        "training_z": scores._finite(
            (tr.corr22.to_numpy() - transformation["corr22_mean"]) / scale,
            "training dependence state",
        ),
        "application_z": scores._finite(
            (ap.corr22.to_numpy() - transformation["corr22_mean"]) / scale,
            "application dependence state",
        ),
    }


def append_products(frame):
    result = frame.copy(deep=True)
    residual = np.column_stack(
        [
            scores._subtract(
                result["y_" + a].to_numpy(), result["mu_" + a].to_numpy(), "issued residual"
            )
            for a in ["qqq", "spx"]
        ]
    )
    realized = scores._product(residual[:, 0], residual[:, 1], "realized product")
    geometric = scores._product(
        np.sqrt(result.h_qqq.to_numpy()),
        np.sqrt(result.h_spx.to_numpy()),
        "geometric marginal moment",
    )
    forecast = scores._product(geometric, result.rho.to_numpy(), "forecast product")
    error = scores._subtract(forecast, realized, "product error")
    result["realized_product"] = realized
    result["forecast_product"] = forecast
    result["product_mse"] = scores._product(error, error, "product squared error")
    return result


def validate_panel(panel):
    if (
        tuple(panel.columns) != COLUMNS
        or set(panel.model) != set(MODELS)
        or panel.duplicated(["origin", "model", "horizon"]).any()
    ):
        raise ValueError("Exact five-model original plus product schema required")
    old = panel.loc[panel.model.isin(scores.MODELS)].reset_index(drop=True)
    checked = scores.score_panel(old.loc[:, scores.INPUT_COLUMNS])
    if not old.equals(checked):
        raise ValueError("Original model cohort/product rows differ")
    base = old.loc[old.model == "constant_correlation"].set_index("origin").sort_index()
    shared = [
        name
        for name in COLUMNS
        if name not in ["origin", "model", "rho", "loss", "forecast_product", "product_mse"]
    ]
    for name in NEW_MODELS:
        rows = panel.loc[panel.model == name].set_index("origin").sort_index()
        if not rows.index.equals(base.index) or not rows.loc[:, shared].equals(
            base.loc[:, shared]
        ):
            raise ValueError(
                "New models must retain original labels, means, diagonals and metadata exactly"
            )
        if not np.isfinite(rows.rho).all() or rows.rho.abs().gt(0.995).any():
            raise ValueError("Bounded finite new dependence forecasts required")
        recalculated = append_products(rows.reset_index().loc[:, scores.INPUT_COLUMNS])
        for column in scores.ADDED_COLUMNS:
            if not np.array_equal(rows[column].to_numpy(), recalculated[column].to_numpy()):
                raise ValueError("Invalid cached new product scores")
        matrix = density.matrix_score(
            rows[["y_qqq", "y_spx"]].to_numpy() - rows[["mu_qqq", "mu_spx"]].to_numpy(),
            rows[["h_qqq", "h_spx"]].to_numpy(),
            rows.rho.to_numpy(),
        )
        if not np.array_equal(rows.loss.to_numpy(), matrix):
            raise ValueError("New auxiliary matrix validity field differs")


def forecast_panel(features, targets, original, frozen_fits, config):
    joint._alignment(features, targets)
    if tuple(original.columns) != COLUMNS:
        raise ValueError("Original issued product panel required")
    expected_old = scores.score_panel(original.loc[:, scores.INPUT_COLUMNS])
    if not expected_old.equals(original):
        raise ValueError("Original product values differ")
    dates = features.index
    ready = np.isfinite(features.loc[:, joint.ALL_FEATURES]).all(axis=1)
    phases = np.zeros(len(dates), bool)
    for phase in ["development", "evaluation"]:
        first, last = config[phase]
        phases |= (dates >= first) & (dates <= last)
    entries = dates[
        ready & phases & (dates >= config["origin_start"]) & (dates <= config["origin_end"])
    ]
    labels = (
        np.isfinite(targets[joint.TARGETS]).all(axis=1)
        & (targets.available_date <= config["latest_target"])
        & (
            (dates > config["development"][1])
            | (targets.available_date <= config["development_target_available_by"])
        )
    )
    scored = entries[labels.loc[entries].to_numpy()]
    original_dates = pd.DatetimeIndex(
        original.loc[original.model == "constant_correlation", "origin"]
    )
    if not original_dates.equals(scored):
        raise ValueError("Entire original source-derived scored cohort required")
    months = entries.to_period("M").unique()
    if len(frozen_fits) != len(months):
        raise ValueError("Every original monthly fit required")
    base = original.loc[original.model == "constant_correlation"].set_index("origin")
    newrows = []
    records = []
    for number, month in enumerate(months):
        application = entries[entries.to_period("M") == month]
        entry = application[0]
        old = frozen_fits[number]
        mask = joint.training_mask(features, targets, entry, config["minimum_train"])
        last = targets.loc[mask, "target_end"].max()
        metadata = {
            "fit_origin": str(entry.date()),
            "fit_cutoff_date": str(features.loc[entry, "feature_cutoff_date"].date()),
            "train_n": int(mask.sum()),
            "train_first_origin": str(dates[mask][0].date()),
            "train_last_origin": str(dates[mask][-1].date()),
            "train_last_target": str(last.date()),
            "train_last_available": str(last.date()),
            "application_n": len(application),
        }
        if any(old[k] != v for k, v in metadata.items()):
            raise ValueError(
                "Original monthly fit metadata or completed training cohort differs"
            )
        replay = replay_frozen_moments(
            features.loc[mask],
            targets.loc[mask],
            features.loc[application],
            old["model_audit"],
        )
        fit = scalar.fit_cross_moment(
            replay["residual"], replay["training_h"], replay["training_z"]
        )
        prediction = scalar.predict_cross_moment(
            replay["application_h"], replay["application_z"], fit
        )
        records.append(
            {
                **metadata,
                "upstream_fit_index": number,
                "upstream_fit_sha256": hashlib.sha256(
                    json.dumps(old, sort_keys=True, allow_nan=False).encode()
                ).hexdigest(),
                "model_audit": fit,
            }
        )
        keep = labels.loc[application].to_numpy()
        chosen = application[keep]
        if not len(chosen):
            continue
        original_rows = (
            base.loc[chosen].rename_axis("origin").reset_index().loc[:, scores.INPUT_COLUMNS]
        )
        _same(
            replay["application_mu"][keep],
            original_rows[["mu_qqq", "mu_spx"]].to_numpy(),
            "issued application means",
        )
        _same(
            replay["application_h"][keep],
            original_rows[["h_qqq", "h_spx"]].to_numpy(),
            "issued application diagonals",
        )
        for name in NEW_MODELS:
            rows = original_rows.copy(deep=True)
            rows["model"] = name
            rows["rho"] = prediction[name][keep]
            rows["loss"] = density.matrix_score(
                rows[["y_qqq", "y_spx"]].to_numpy() - rows[["mu_qqq", "mu_spx"]].to_numpy(),
                rows[["h_qqq", "h_spx"]].to_numpy(),
                rows.rho.to_numpy(),
            )
            newrows.append(append_products(rows))
    if not newrows:
        raise ValueError("INSUFFICIENT_DATA: no new issued forecast rows")
    result = (
        pd.concat([original, *newrows], ignore_index=True)
        .sort_values(["origin", "model"])
        .reset_index(drop=True)
    )
    validate_panel(result)
    return result, records
