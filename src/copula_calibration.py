"""Causal marginal recalibration crossed with separately fitted copulas.

This diagnostic reuses issued base marginal forecasts, not their in-sample errors.
Every complete joint density includes the probability-calibration Jacobian.
"""

import numpy as np
import pandas as pd
from scipy.special import log_ndtr, ndtr
from scipy.stats import t

from src.joint_copula_density import (
    fit_dependence,
    log_copula,
    marginal_logpdf,
    normal_scores,
)

CELLS = ("orig_gaussian", "orig_t8", "cal_gaussian", "cal_t8")
CONTRASTS = (
    "original_gap",
    "calibrated_gap",
    "interaction",
    "gaussian_calibration",
    "t8_calibration",
    "qqq_calibration",
    "spx_calibration",
)
ASSETS = ("qqq", "spx")
SCALE_FLOOR = 1e-12


def _pairs(values):
    z = np.asarray(values, dtype=float)
    if z.ndim != 2 or z.shape[1] != 2 or not len(z) or not np.isfinite(z).all():
        raise ValueError("Finite nonempty pairs required")
    return z


def identity_calibration():
    return {"location": [0.0, 0.0], "scale": [1.0, 1.0], "train_n": 0}


def fit_calibration(z):
    z = _pairs(z)
    if len(z) < 2:
        raise ValueError("At least two historical residual pairs required")
    w = normal_scores(z)
    a = w.mean(axis=0)
    b = np.sqrt(np.mean((w - a) ** 2, axis=0))
    if not np.isfinite(a).all() or not np.isfinite(b).all() or (b <= SCALE_FLOOR).any():
        raise ValueError("Calibration scale outside fixed numerical domain")
    return {"location": a.tolist(), "scale": b.tolist(), "train_n": len(z)}


def calibrate(z, h, calibration):
    z = _pairs(z)
    h = np.asarray(h, dtype=float)
    if h.shape != z.shape or not np.isfinite(h).all() or (h <= 0).any():
        raise ValueError("Finite positive aligned original variances required")
    try:
        a, b = (np.asarray(calibration[k], dtype=float) for k in ("location", "scale"))
    except (KeyError, TypeError) as exc:
        raise ValueError("Explicit calibration location and scale required") from exc
    if (
        a.shape != (2,)
        or b.shape != (2,)
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or (b <= SCALE_FLOOR).any()
    ):
        raise ValueError("Finite calibration with scale above fixed floor required")
    w = normal_scores(z)
    v = (w - a) / b
    if not np.isfinite(v).all():
        raise ValueError("Nonfinite calibrated normal scores")
    if np.array_equal(a, np.zeros(2)) and np.array_equal(b, np.ones(2)):
        zcal = z.copy()
    else:
        logtail = log_ndtr(-np.abs(v))
        tail = np.exp(logtail)
        if not np.isfinite(tail).all() or (tail == 0).any():
            raise ValueError("Unrepresentable calibrated inverse tail; no clipping")
        zcal = np.sign(v) * t.isf(tail, 8)
        if not np.isfinite(zcal).all():
            raise ValueError("Nonfinite calibrated inverse tail")
        if not np.allclose(normal_scores(zcal), v, atol=1e-10, rtol=1e-10):
            raise ValueError("Calibrated inverse tail failed roundtrip")
    log_marginal = marginal_logpdf(z, h) - 0.5 * (v * v - w * w) - np.log(b)
    if not np.isfinite(log_marginal).all():
        raise ValueError("Nonfinite normalized marginal log density")
    return {"z": zcal, "normal": v, "log_marginal": log_marginal, "pit": ndtr(v)}


def select_history(archive, fit_origin, training_cutoff, minimum_train=252):
    if type(minimum_train) is not int or minimum_train < 2:
        raise ValueError("Exact minimum training count of at least two required")
    required = {"origin", "available_date", "issued"} | {
        f"{kind}_{asset}" for kind in ("mu", "h", "y") for asset in ASSETS
    }
    if (
        not isinstance(archive, pd.DataFrame)
        or not required <= set(archive)
        or not archive.columns.is_unique
    ):
        raise ValueError("Complete unique archive columns required")
    origins = pd.DatetimeIndex(archive.origin)
    if origins.hasnans or not origins.is_unique or not origins.is_monotonic_increasing:
        raise ValueError("Unique increasing archive origins required")
    fit_origin, training_cutoff = pd.Timestamp(fit_origin), pd.Timestamp(training_cutoff)
    if pd.isna(fit_origin) or pd.isna(training_cutoff) or training_cutoff >= fit_origin:
        raise ValueError("Strictly prior training cutoff required")
    if archive.issued.dtype.kind != "b":
        raise ValueError("Boolean historical issuance required")
    available = pd.DatetimeIndex(archive.available_date)
    selected = archive.loc[
        (origins < fit_origin) & (available <= training_cutoff) & archive.issued
    ].copy()
    numerical = selected[
        [f"{kind}_{asset}" for kind in ("mu", "h", "y") for asset in ASSETS]
    ].to_numpy(float)
    if np.isinf(numerical).any():
        raise ValueError("Infinite observed history is invalid")
    # Missing outcomes are unavailable. An issued finite-outcome forecast must be valid.
    known = selected[["y_qqq", "y_spx"]].notna().all(axis=1)
    selected = selected.loc[known].copy()
    base = selected[["mu_qqq", "mu_spx", "h_qqq", "h_spx"]].to_numpy(float)
    if (
        not np.isfinite(base).all()
        or (selected[["h_qqq", "h_spx"]].to_numpy(float) <= 0).any()
    ):
        raise ValueError("Invalid observed historical marginal forecast")
    if len(selected) < minimum_train:
        raise ValueError("INSUFFICIENT_DATA: mature issued residual history")
    return selected


def contrasts(losses, marginal_losses):
    if set(losses) != set(CELLS) or set(marginal_losses) != {"original", "calibrated"}:
        raise ValueError("All four joint cells and both marginal systems required")
    losses = {k: np.asarray(v, dtype=float) for k, v in losses.items()}
    n = len(losses[CELLS[0]]) if losses[CELLS[0]].ndim == 1 else 0
    if not n or any(v.shape != (n,) or not np.isfinite(v).all() for v in losses.values()):
        raise ValueError("Finite nonempty aligned joint loss arrays required")
    marginal_losses = {k: np.asarray(v, dtype=float) for k, v in marginal_losses.items()}
    if any(v.shape != (n, 2) or not np.isfinite(v).all() for v in marginal_losses.values()):
        raise ValueError("Finite aligned two-asset marginal losses required")
    raw = losses["orig_t8"] - losses["orig_gaussian"]
    cal = losses["cal_t8"] - losses["cal_gaussian"]
    marg = marginal_losses["calibrated"] - marginal_losses["original"]
    return dict(
        zip(
            CONTRASTS,
            (
                raw,
                cal,
                cal - raw,
                losses["cal_gaussian"] - losses["orig_gaussian"],
                losses["cal_t8"] - losses["orig_t8"],
                marg[:, 0],
                marg[:, 1],
            ),
            strict=True,
        )
    )


def _coordinates(frame):
    y = frame[["y_qqq", "y_spx"]].to_numpy(float)
    mu = frame[["mu_qqq", "mu_spx"]].to_numpy(float)
    h = frame[["h_qqq", "h_spx"]].to_numpy(float)
    if (
        not np.isfinite(y).all()
        or not np.isfinite(mu).all()
        or not np.isfinite(h).all()
        or (h <= 0).any()
    ):
        raise ValueError("Finite observed targets and valid issued marginals required")
    z = (y - mu) / np.sqrt(0.75 * h)
    return _pairs(z), h


def run_crossed(archive, minimum_train=252):
    """Fit on mature issued errors and issue fixed parameters for each whole month."""
    if not isinstance(archive, pd.DataFrame) or archive.empty or not archive.columns.is_unique:
        raise ValueError("Nonempty canonical issued archive required")
    required = {
        "origin",
        "available_date",
        "target_end",
        "issued",
        "phase",
        "offset",
        "fit_origin",
        "training_cutoff",
        "eligible_scored",
    } | {f"{kind}_{asset}" for kind in ("mu", "h", "y") for asset in ASSETS}
    if not required <= set(archive):
        raise ValueError("Incomplete canonical issued archive")
    origins = pd.DatetimeIndex(archive.origin)
    if (
        origins.hasnans
        or not origins.is_unique
        or not origins.is_monotonic_increasing
        or archive.issued.dtype.kind != "b"
        or not archive.issued.all()
    ):
        raise ValueError("Unique sorted canonical issued origins required")
    if archive.eligible_scored.dtype.kind != "b":
        raise ValueError("Boolean original scoring eligibility required")
    applications, fits, statuses = [], [], {}
    started = False
    base_columns = [
        "origin",
        "phase",
        "offset",
        "mu_qqq",
        "mu_spx",
        "h_qqq",
        "h_spx",
        "fit_origin",
        "training_cutoff",
    ]
    for _, query in archive.groupby(origins.to_period("M"), sort=True):
        fit = pd.Timestamp(query.fit_origin.iloc[0])
        cutoff = pd.Timestamp(query.training_cutoff.iloc[0])
        if (
            not query.fit_origin.eq(fit).all()
            or not query.training_cutoff.eq(cutoff).all()
            or fit != query.origin.iloc[0]
            or cutoff >= fit
        ):
            raise ValueError("Exact original first-issued monthly schedule required")
        try:
            history = select_history(archive, fit, cutoff, minimum_train)
        except ValueError as error:
            if "INSUFFICIENT_DATA" not in str(error) or started:
                raise
            statuses.update(dict.fromkeys(query.origin, "warmup"))
            continue
        started = True
        query_base = query[["mu_qqq", "mu_spx", "h_qqq", "h_spx"]].to_numpy(float)
        if not np.isfinite(query_base).all() or (query_base[:, 2:] <= 0).any():
            raise ValueError("Every issued query requires finite means and positive variances")
        z, h = _coordinates(history)
        calibration = fit_calibration(z)
        changed = calibrate(z, h, calibration)
        dependence = {}
        for model in CELLS:
            coords = z if model.startswith("orig_") else changed["z"]
            family = model.split("_", 1)[1]
            rho, audit = fit_dependence(coords, family)
            if not np.isfinite(rho) or abs(rho) > 0.995:
                raise ValueError("Invalid fitted dependence parameter")
            dependence[model] = {"rho": float(rho), "audit": audit}
        fits.append(
            {
                "fit_origin": str(fit.date()),
                "training_cutoff": str(cutoff.date()),
                "train_origins": [str(pd.Timestamp(d).date()) for d in history.origin],
                "train_n": len(history),
                "calibration": calibration,
                "dependence": dependence,
            }
        )
        # No query outcomes are consulted to issue these four complete distributions.
        app = query[base_columns].copy()
        app["train_n"] = len(history)
        for j, asset in enumerate(ASSETS):
            app[f"a_{asset}"] = calibration["location"][j]
            app[f"b_{asset}"] = calibration["scale"][j]
        for model in CELLS:
            app[f"rho_{model}"] = dependence[model]["rho"]
        applications.append(app)
        statuses.update(
            {
                row.origin: "scored" if row.eligible_scored else "issued_unscored"
                for row in query.itertuples(index=False)
            }
        )
    if not applications:
        raise ValueError("INSUFFICIENT_DATA: no supported original monthly fit")
    applications = pd.concat(applications, ignore_index=True)
    actual = archive.loc[
        archive.eligible_scored, ["origin", "target_end", "available_date", "y_qqq", "y_spx"]
    ]
    panel = (
        applications.merge(actual, on="origin", how="inner", validate="one_to_one")
        .sort_values("origin")
        .reset_index(drop=True)
    )
    if panel.empty:
        raise ValueError("INSUFFICIENT_DATA: no common scored origins")
    panels = []
    for _, part in panel.groupby("fit_origin", sort=True):
        part = part.copy()
        z, h = _coordinates(part)
        calibration = {
            "location": [float(part[f"a_{a}"].iloc[0]) for a in ASSETS],
            "scale": [float(part[f"b_{a}"].iloc[0]) for a in ASSETS],
        }
        original = calibrate(z, h, identity_calibration())
        calibrated = calibrate(z, h, calibration)
        margins = {"original": original, "calibrated": calibrated}
        for label, transformed in margins.items():
            for j, asset in enumerate(ASSETS):
                part[f"marginal_{label}_{asset}"] = transformed["log_marginal"][:, j]
                part[f"pit_{label}_{asset}"] = transformed["pit"][:, j]
                part[f"normal_{label}_{asset}"] = transformed["normal"][:, j]
        losses = {}
        for model in CELLS:
            transformed = original if model.startswith("orig_") else calibrated
            density = transformed["log_marginal"].sum(axis=1) + log_copula(
                transformed["z"], part[f"rho_{model}"].to_numpy(float), model.split("_", 1)[1]
            )
            losses[model] = -density
            part[f"loss_{model}"] = -density
        differences = contrasts(
            losses, {name: -v["log_marginal"] for name, v in margins.items()}
        )
        for name, difference in differences.items():
            part[f"d_{name}"] = difference
        panels.append(part)
    return {
        "applications": applications,
        "panel": pd.concat(panels, ignore_index=True)
        .sort_values("origin")
        .reset_index(drop=True),
        "fits": fits,
        "coverage": pd.DataFrame(
            {"origin": archive.origin, "status": [statuses[d] for d in archive.origin]}
        ),
    }
