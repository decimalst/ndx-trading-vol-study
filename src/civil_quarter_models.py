"""Fixed mixed-geometry positive model and one frozen-baseline civil increment.

Every support and identification failure aborts the scheduled comparison. The
only optimization is the prespecified zero-start Newton/Armijo sequence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import civil_quarter_features as cf
from .index_hinge import _dates

BASE = cf.BASE
ALL_FEATURES = cf.ALL_FEATURES
MODELS = cf.MODELS
ALPHA = 0.01
MAX_ITERATIONS = 200
MAX_BACKTRACKS = 60
ARMIJO = 1e-4
GRADIENT_TOLERANCE = 1e-8
SCALE_MINIMUM = 1e-12
PANEL_COLUMNS = (
    "origin",
    "model",
    "horizon",
    "feature_cutoff_date",
    "target_end",
    "available_date",
    "y",
    "prediction",
    "fit_origin",
    "fit_cutoff_date",
    "train_n",
    "train_last_target",
    "train_last_available",
    "phase",
)


def _finite(value, name):
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f"Real numeric {name} required")
    value = np.asarray(raw, dtype=float)
    if not np.isfinite(value).all():
        raise ValueError(f"Finite {name} required")
    return value


def _positive(value, name):
    value = _finite(value, name)
    if (value <= 0).any():
        raise ValueError(f"Strictly positive {name} required")
    return value


def _product(left, right, name):
    left, right = _finite(left, name), _finite(right, name)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = left * right
    _finite(result, name)
    if ((left != 0) & (right != 0) & (result == 0)).any():
        raise ValueError(f"Unsupported nonzero multiplication underflow in {name}")
    return result


def _quotient(numerator, denominator, name):
    numerator, denominator = _finite(numerator, name), _finite(denominator, name)
    if (denominator == 0).any():
        raise ValueError(f"Zero denominator in {name}")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = numerator / denominator
    _finite(result, name)
    if ((numerator != 0) & (result == 0)).any():
        raise ValueError(f"Unsupported nonzero quotient underflow in {name}")
    return result


def _mean(values, name):
    values = _finite(values, name)
    if not values.size:
        raise ValueError(f"Nonempty {name} required")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        total = values.sum()
    return float(_quotient(total, values.size, name))


def _exp(log_value, name):
    log_value = _finite(log_value, name)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        value = np.exp(log_value)
    return _positive(value, name)


def _design_product(design, coefficient):
    # Check the explicit primitive products, then preserve the fixed BLAS
    # reduction used by the admitted baseline/native forecast expression.
    _product(design, coefficient, "design/coefficient products")
    with np.errstate(over="ignore", invalid="ignore"):
        result = design @ coefficient
    return _finite(result, "linear predictor")


def objective(beta, design, scaled_y):
    """Normalized proper variance objective, full gradient and Hessian."""
    beta = _finite(beta, "coefficients")
    design = _finite(design, "design")
    scaled_y = _positive(scaled_y, "normalized targets")
    if (
        design.ndim != 2
        or beta.shape != (design.shape[1],)
        or scaled_y.shape != (len(design),)
        or not len(design)
        or not design.shape[1]
        or not np.equal(design[:, 0], 1).all()
    ):
        raise ValueError("Aligned nonempty design with unit intercept required")
    eta = _design_product(design, beta)
    ratio = _exp(np.log(scaled_y) - eta, "positive target/prediction ratio")
    penalty = _product(beta[1:], beta[1:], "squared slopes")
    value = _mean(eta + ratio, "objective terms") + float(
        _product(ALPHA, penalty.sum(), "slope penalty")
    )
    score = 1.0 - ratio
    _product(design, score[:, None], "gradient products")
    gradient = _quotient(design.T @ score, len(design), "mean gradient")
    gradient[1:] += _product(2 * ALPHA, beta[1:], "penalty gradient")
    weighted = _product(design, ratio[:, None], "Hessian weights")
    hessian = _quotient(design.T @ weighted, len(design), "mean Hessian")
    hessian[1:, 1:] += 2 * ALPHA * np.eye(len(beta) - 1)
    _finite(value, "objective")
    _finite(gradient, "gradient")
    _finite(hessian, "Hessian")
    return float(value), gradient, hessian


def quarter_objective(b, eta0, scaled_y, z):
    """One strictly convex coefficient with the entire baseline frozen."""
    b = _finite(b, "quarter coefficient")
    eta0, z = _finite(eta0, "baseline eta"), _finite(z, "quarter increment")
    scaled_y = _positive(scaled_y, "normalized targets")
    if (
        b.ndim
        or eta0.ndim != 1
        or z.shape != eta0.shape
        or scaled_y.shape != eta0.shape
        or not len(z)
    ):
        raise ValueError("Aligned nonempty scalar quarter objective required")
    eta = _finite(eta0 + _product(b, z, "quarter offset"), "quarter eta")
    ratio = _exp(np.log(scaled_y) - eta, "positive target/prediction ratio")
    value = _mean(eta + ratio, "quarter objective terms") + float(
        _product(ALPHA, _product(b, b, "squared quarter slope"), "quarter penalty")
    )
    gradient = _mean(_product(z, 1 - ratio, "quarter scores"), "quarter gradient") + float(
        _product(2 * ALPHA, b, "quarter penalty gradient")
    )
    hessian = (
        _mean(
            _product(
                _product(z, z, "squared quarter increment"), ratio, "quarter curvature terms"
            ),
            "quarter curvature",
        )
        + 2 * ALPHA
    )
    _finite([value, gradient, hessian], "quarter objective/derivatives")
    if hessian < 2 * ALPHA:
        raise ValueError("Quarter curvature lost its strict convexity bound")
    return float(value), float(gradient), float(hessian)


def _newton(evaluate, size):
    beta = np.zeros(size)
    backtracks, rejected = 0, 0
    for iteration in range(MAX_ITERATIONS):
        value, gradient, hessian = evaluate(beta)
        if float(np.max(np.abs(gradient))) <= GRADIENT_TOLERANCE:
            return beta, value, gradient, hessian, iteration, backtracks, rejected
        try:
            direction = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Fixed Newton system failed; no fallback") from exc
        _finite(direction, "Newton direction")
        descent = float(_product(gradient, direction, "Newton descent products").sum())
        if not np.isfinite(descent) or descent <= 0:
            raise ValueError("Fixed Newton direction is not a finite descent direction")
        for trial in range(MAX_BACKTRACKS):
            step = 0.5**trial
            candidate = beta - _product(step, direction, "Newton step")
            try:
                candidate_value, _, _ = evaluate(candidate)
            except ValueError:
                rejected += 1
                continue
            threshold = value - float(_product(ARMIJO * step, descent, "Armijo decrement"))
            if candidate_value <= threshold:
                beta = candidate
                backtracks += trial
                break
        else:
            raise ValueError("Fixed Newton Armijo schedule did not converge")
    raise ValueError("Fixed Newton iteration budget did not converge")


def fit_quarter(eta0, scaled_y, z):
    """Fit the unique scalar optimum and retain an independent root bracket."""
    initial = quarter_objective(0.0, eta0, scaled_y, z)
    radius = max(1.0, abs(float(_quotient(initial[1], 2 * ALPHA, "scalar bracket"))))
    bracket = [-radius, radius]
    endpoints = [quarter_objective(endpoint, eta0, scaled_y, z)[1] for endpoint in bracket]
    if not endpoints[0] <= 0 <= endpoints[1]:
        raise ValueError("Finite quarter derivative bracket must enclose zero")

    def evaluate(beta):
        value, gradient, hessian = quarter_objective(float(beta[0]), eta0, scaled_y, z)
        return value, np.array([gradient]), np.array([[hessian]])

    beta, value, gradient, hessian, iterations, backtracks, rejected = _newton(evaluate, 1)
    return {
        "b": float(beta[0]),
        "objective_scaled": value,
        "gradient": gradient.tolist(),
        "gradient_max_abs": float(np.max(np.abs(gradient))),
        "curvature": float(hessian[0, 0]),
        "iterations": iterations,
        "backtracks": backtracks,
        "numerical_trial_rejections": rejected,
        "initial_gradient": initial[1],
        "bracket": bracket,
        "bracket_gradients": endpoints,
        "alpha": ALPHA,
        "start": [0.0],
        "baseline_frozen": True,
    }


def _feature_frame(frame):
    if (
        not isinstance(frame, pd.DataFrame)
        or frame.columns.has_duplicates
        or not set(ALL_FEATURES).issubset(frame.columns)
        or not len(frame)
    ):
        raise ValueError("Nonempty distinct complete feature schema required")
    _dates(frame.index)
    values = _finite(frame.loc[:, ALL_FEATURES].to_numpy(), "common features")
    if not np.equal(values[:, 0], 1.0).all():
        raise ValueError("Literal unit intercept required")
    if not np.isin(frame.loc[:, cf.CIVIL_FEATURES].to_numpy(), [0.0, 1.0]).all():
        raise ValueError("Raw civil indicators must be binary")


def transform(train, application):
    """Population scale old slopes; center new civil indicators at fixed scale1."""
    _feature_frame(train)
    _feature_frame(application)
    means = np.zeros(len(BASE))
    scales = np.ones(len(BASE))
    for index, column in enumerate(BASE[1:], start=1):
        values = train[column].to_numpy(dtype=float)
        # Exact constant centering is a mathematical primitive, never a
        # substitute for the separate empirical support and rank gates.
        means[index] = values[0] if np.equal(values, values[0]).all() else values.mean()
        if column in cf.OLD_FEATURES:
            scales[index] = values.std(ddof=0)
            if not np.isfinite(scales[index]) or scales[index] <= SCALE_MINIMUM:
                raise ValueError("INSUFFICIENT_DATA: old feature scale must exceed1e-12")
    _finite(means, "training feature means")
    values = train[cf.MEMORY].to_numpy(dtype=float)
    center = float(values[0] if np.equal(values, values[0]).all() else values.mean())
    transformed = []
    for frame in (train, application):
        standardized = _quotient(
            frame.loc[:, BASE].to_numpy(dtype=float) - means, scales, "transformed baseline"
        )
        transformed.append(pd.DataFrame(standardized, index=frame.index, columns=BASE))
    z = _finite(train[cf.MEMORY].to_numpy(dtype=float) - center, "centered quarter")
    appz = _finite(application[cf.MEMORY].to_numpy(dtype=float) - center, "query quarter")
    audit = {
        "columns": list(BASE),
        "means": means.tolist(),
        "scales": scales.tolist(),
        "quarter_mean": center,
        "quarter_scale": 1.0,
        "train_n": len(train),
    }
    return *transformed, z, appz, audit


def _identification(design, z):
    coefficients, _, rank, _ = np.linalg.lstsq(design, z, rcond=1e-12)
    residual = _finite(z - design @ coefficients, "quarter identification residual")
    centered_norm = float(np.linalg.norm(z))
    residual_norm = float(np.linalg.norm(residual))
    relative = float(_quotient(residual_norm, centered_norm, "relative quarter residual"))
    if not relative > 1e-8:
        raise ValueError("INSUFFICIENT_DATA: quarter increment not identified beyond baseline")
    return {
        "columns": list(BASE),
        "ols_rcond": 1e-12,
        "baseline_rank": int(rank),
        "residual_norm": residual_norm,
        "centered_norm": centered_norm,
        "relative_residual_norm": relative,
        "minimum_relative_norm": 1e-8,
    }


def _training_geometry(features, application):
    support = cf.civil_support(features, "train")
    rank = cf.require_civil_rank(features)
    train, query, z, appz, audit = transform(features, application)
    identification = _identification(train.to_numpy(), z)
    return train, query, z, appz, audit, support, rank, identification


def fit_predict(features, y, application):
    target = _positive(y, "training targets")
    if target.shape != (len(features),) or (
        isinstance(y, pd.Series) and not y.index.equals(features.index)
    ):
        raise ValueError("Training target row ordering or shape differs")
    train_mean = _mean(target, "training target mean")
    normalized = _positive(
        _quotient(target, train_mean, "normalized targets"), "normalized targets"
    )
    train, query, z, appz, transform_audit, support, rank, identification = _training_geometry(
        features, application
    )
    design, query_design = train.to_numpy(), query.to_numpy()
    beta, value, gradient, _, iterations, backtracks, rejected = _newton(
        lambda coefficient: objective(coefficient, design, normalized), len(BASE)
    )
    eta0 = _design_product(design, beta)
    app_eta0 = _design_product(query_design, beta)
    quarter = fit_quarter(eta0, normalized, z)
    log_mean = float(np.log(train_mean))
    baseline_prediction = _exp(log_mean + app_eta0, "baseline prediction")
    quarter_prediction = _exp(
        log_mean + app_eta0 + _product(quarter["b"], appz, "query quarter offset"),
        "quarter prediction",
    )
    actual_beta = beta.copy()
    actual_beta[0] += log_mean
    common = {
        "train_mean": train_mean,
        "train_n": len(features),
        "application_n": len(application),
    }
    audits = {
        "mean": {"columns": ["const"], "beta": [log_mean], **common, "gradient_max_abs": 0.0},
        "baseline": {
            "columns": list(BASE),
            "means": transform_audit["means"],
            "scales": transform_audit["scales"],
            **common,
            "scaled_beta": beta.tolist(),
            "beta": actual_beta.tolist(),
            "objective_scaled": value,
            "objective": float(value + log_mean),
            "gradient": gradient.tolist(),
            "gradient_max_abs": float(np.max(np.abs(gradient))),
            "iterations": iterations,
            "backtracks": backtracks,
            "numerical_trial_rejections": rejected,
            "alpha": ALPHA,
            "start": [0.0] * len(BASE),
        },
        "quarter": {
            "columns": [cf.MEMORY],
            "mean": transform_audit["quarter_mean"],
            "scale": 1.0,
            **common,
            **quarter,
            "objective": float(quarter["objective_scaled"] + log_mean),
            "support": support,
            "civil_rank": rank,
            "identification": identification,
        },
    }
    return {
        "mean": np.full(len(application), train_mean),
        "baseline": baseline_prediction,
        "quarter": quarter_prediction,
    }, audits


def _alignment(features, targets):
    _dates(features.index)
    if (
        not features.index.equals(targets.index)
        or features.columns.has_duplicates
        or targets.columns.has_duplicates
        or tuple(targets.columns) != ("y", "target_end", "available_date")
        or not set(ALL_FEATURES).issubset(features.columns)
    ):
        raise ValueError("Exact shared full feature/target calendar and schema required")
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
    known = targets.y.notna()
    _positive(targets.loc[known, "y"], "known next-session targets")
    if targets.loc[known, "target_end"].isna().any():
        raise ValueError("Known target lacks next-session maturity")


def training_mask(features, targets, fit_entry, min_train=1000):
    _alignment(features, targets)
    fit_entry = pd.Timestamp(fit_entry)
    if fit_entry not in features.index or not isinstance(min_train, int) or min_train < 2:
        raise ValueError("Invalid fit origin or minimum training sample")
    cutoff = features.loc[fit_entry, "feature_cutoff_date"]
    if pd.isna(cutoff):
        raise ValueError("INSUFFICIENT_DATA: no previous session")
    mask = (
        np.isfinite(features.loc[:, ALL_FEATURES]).all(axis=1)
        & features.feature_cutoff_date.notna()
        & targets.y.notna()
        & (features.index < fit_entry)
        & (targets.available_date <= cutoff)
    )
    if int(mask.sum()) < min_train:
        raise ValueError(f"INSUFFICIENT_DATA: {int(mask.sum())} common mature training labels")
    return mask


def _cohort(features, targets, config):
    _alignment(features, targets)
    if (
        tuple(config["models"]) != MODELS
        or tuple(config["baseline"]) != BASE
        or tuple(config["all_features"]) != ALL_FEATURES
    ):
        raise ValueError("Fixed civil quarter model/feature family differs")
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
        raise ValueError("Invalid historical phases or numerical fence")
    dates = features.index
    phase = ((dates >= dev_start) & (dates <= dev_end)) | (
        (dates >= ev_start) & (dates <= ev_end)
    )
    ready = (
        np.isfinite(features.loc[:, ALL_FEATURES]).all(axis=1)
        & features.feature_cutoff_date.notna()
    )
    entries = dates[ready & phase & (dates >= start) & (dates <= end)]
    if not len(entries):
        raise ValueError("INSUFFICIENT_DATA: no common complete application features")
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
    """Check every scheduled fold and scored phase before any optimization."""
    entries, labels, _ = _cohort(features, targets, config)
    scored = entries[labels.loc[entries].to_numpy()]
    fits = []
    for month in entries.to_period("M").unique():
        applications = entries[entries.to_period("M") == month]
        mask = training_mask(features, targets, applications[0], config["minimum_train"])
        _, _, _, _, _, support, rank, identification = _training_geometry(
            features.loc[mask], features.loc[applications]
        )
        fits.append(
            {
                **_metadata(features, targets, applications, mask),
                "support": support,
                "civil_rank": rank,
                "identification": identification,
            }
        )
    phases = []
    for name in ("development", "evaluation"):
        start, end = map(pd.Timestamp, config[name])
        dates = scored[(scored >= start) & (scored <= end)]
        if len(dates) < 127:
            raise ValueError(f"INSUFFICIENT_DATA: {name} needs at least127 observations")
        phase = {
            "name": name,
            "n": len(dates),
            "first_origin": str(dates[0].date()),
            "last_origin": str(dates[-1].date()),
            "support": cf.civil_support(features.loc[dates], "phase"),
            "slices": [],
        }
        if name == "evaluation":
            slices = config["evaluation_stability"]
            if len(slices) != 2:
                raise ValueError("Two fixed evaluation stability intervals required")
            for lower, upper in slices:
                subset = dates[(dates >= pd.Timestamp(lower)) & (dates <= pd.Timestamp(upper))]
                phase["slices"].append(
                    {
                        "start": lower,
                        "end": upper,
                        "n": len(subset),
                        "support": cf.civil_support(features.loc[subset], "slice"),
                    }
                )
        phases.append(phase)
    return {
        "common_application_origins": len(entries),
        "common_scored_origins": len(scored),
        "monthly_fits": len(fits),
        "fits": fits,
        "phases": phases,
    }


def validate_panel(panel):
    """Validate the complete issued three-model panel without deleting rows."""
    if (
        not isinstance(panel, pd.DataFrame)
        or tuple(panel.columns) != PANEL_COLUMNS
        or panel.empty
        or panel.duplicated(["origin", "model"]).any()
        or set(panel.model) != set(MODELS)
    ):
        raise ValueError("Complete unique fixed-schema three-model panel required")
    _positive(panel.y, "panel targets")
    _positive(panel.prediction, "panel predictions")
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
    for column in date_columns:
        dates = panel[column]
        if (
            not pd.api.types.is_datetime64_any_dtype(dates.dtype)
            or dates.isna().any()
            or dates.dt.tz is not None
            or not dates.equals(dates.dt.normalize())
        ):
            raise ValueError("Complete normalized timezone-naive panel dates required")
    if (
        not panel.horizon.eq(1).all()
        or not panel.phase.isin(["development", "evaluation"]).all()
        or not (panel.feature_cutoff_date < panel.origin).all()
        or not (panel.target_end > panel.origin).all()
        or not panel.available_date.equals(panel.target_end)
        or not (panel.fit_origin <= panel.origin).all()
        or not (panel.fit_cutoff_date < panel.fit_origin).all()
        or not (panel.fit_cutoff_date <= panel.feature_cutoff_date).all()
        or not panel.fit_origin.dt.to_period("M").equals(panel.origin.dt.to_period("M"))
        or not (panel.train_last_available <= panel.fit_cutoff_date).all()
        or not panel.train_last_available.equals(panel.train_last_target)
        or not (panel.target_end <= pd.Timestamp("2025-10-20")).all()
        or not (panel.origin <= pd.Timestamp("2025-10-17")).all()
    ):
        raise ValueError("Invalid monthly fit, maturity, phase or numerical fence metadata")
    counts = _finite(panel.train_n, "panel training counts")
    if (counts < 2).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError("Integral training counts required")
    fit_metadata = [
        "fit_origin",
        "fit_cutoff_date",
        "train_n",
        "train_last_target",
        "train_last_available",
    ]
    for _, month in panel.groupby(panel.origin.dt.to_period("M")):
        if not month[fit_metadata].nunique(dropna=False).eq(1).all():
            raise ValueError("Monthly fit metadata differs across issued applications")
    shared = [column for column in PANEL_COLUMNS if column not in ["model", "prediction"]]
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
            raise ValueError("All models require identical cohorts, labels and metadata")
    return panel


def forecast_panel(features, targets, config):
    """Monthly expanding fits chosen before any future application-label mask."""
    entries, labels, dev_end = _cohort(features, targets, config)
    rows, fits = [], []
    for month in entries.to_period("M").unique():
        applications = entries[entries.to_period("M") == month]
        fit_entry = applications[0]
        mask = training_mask(features, targets, fit_entry, config["minimum_train"])
        predictions, audit = fit_predict(
            features.loc[mask], targets.loc[mask, "y"], features.loc[applications]
        )
        metadata = _metadata(features, targets, applications, mask)
        fits.append({**metadata, "model_audit": audit})
        if set(predictions) != set(MODELS):
            raise ValueError("Every monthly fit must issue all three models")
        for name in MODELS:
            predictions[name] = _positive(
                predictions[name], "all scheduled application predictions"
            )
            if predictions[name].shape != (len(applications),):
                raise ValueError("Every scheduled application requires one prediction")
        keep = labels.loc[applications].to_numpy()
        scored = applications[keep]
        if not len(scored):
            continue
        for name in MODELS:
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
                        "y": targets.loc[scored, "y"].to_numpy(),
                        "prediction": predictions[name][keep],
                        "fit_origin": fit_entry,
                        "fit_cutoff_date": pd.Timestamp(metadata["fit_cutoff_date"]),
                        "train_n": metadata["train_n"],
                        "train_last_target": pd.Timestamp(metadata["train_last_target"]),
                        "train_last_available": pd.Timestamp(metadata["train_last_available"]),
                        "phase": np.where(scored <= dev_end, "development", "evaluation"),
                    }
                )
            )
    if not rows:
        raise ValueError("INSUFFICIENT_DATA: no scoreable common risk labels")
    panel = (
        pd.concat(rows, ignore_index=True)
        .sort_values(["origin", "model"])
        .reset_index(drop=True)
    )
    validate_panel(panel)
    return panel, fits
