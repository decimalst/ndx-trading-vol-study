"""Fixed matched monthly return models, including every unscored application."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import peak_age_models as models

RAW = (
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
)
COMMON = RAW + ("peak_age", "drawdown", "drawdown_sq", "window_return")
BASE = RAW + ("I_square", "R_square")
MODELS = ("mean", "baseline", "depth", "peak_age")
STATE_COLUMNS = (
    "origin",
    "fit_origin",
    "feature_cutoff_date",
    "train_n",
    "pred_mean",
    "pred_baseline",
    "pred_depth",
    "pred_peak_age",
)
PANEL_COLUMNS = (
    "origin",
    "fit_origin",
    "feature_cutoff_date",
    "target_end",
    "available_date",
    "horizon",
    "phase",
    "model",
    "prediction",
    "y",
    "loss",
    "train_n",
)


def _dates(index):
    if (
        not isinstance(index, pd.DatetimeIndex)
        or index.hasnans
        or index.has_duplicates
        or index.tz is not None
        or not index.is_monotonic_increasing
        or not index.equals(index.normalize())
        or np.datetime_data(index.dtype)[0] not in ("ms", "us", "ns")
    ):
        raise ValueError("Unique ordered native normalized naive reference dates required")


def _same_dates(actual, expected):
    if (
        not pd.api.types.is_datetime64_any_dtype(actual.dtype)
        or getattr(actual.dtype, "tz", None) is not None
        or np.datetime_data(actual.dtype)[0] not in ("ms", "us", "ns")
        or not np.array_equal(actual.isna(), expected.isna())
    ):
        return False
    try:
        converted = actual.dt.as_unit("ns", round_ok=False)
        back = converted.dt.as_unit(np.datetime_data(actual.dtype)[0], round_ok=False)
        return actual.equals(back) and np.array_equal(
            converted.dropna().to_numpy(),
            expected.dt.as_unit("ns", round_ok=False).dropna().to_numpy(),
        )
    except (ValueError, OverflowError):
        return False


def alignment(features, targets):
    _dates(features.index)
    if (
        not features.index.equals(targets.index)
        or not set(COMMON).issubset(features.columns)
        or list(targets.columns) != ["y", "target_end", "available_date"]
    ):
        raise ValueError("Aligned common features and sole21-session target required")
    dates = pd.Series(features.index, index=features.index)
    for frame, name, shift in (
        (features, "feature_cutoff_date", 1),
        (targets, "target_end", -21),
        (targets, "available_date", -21),
    ):
        if name not in frame or not _same_dates(frame[name], dates.shift(shift)):
            raise ValueError("Exact prior-session and21-session outcome clocks required")
    for frame, columns in ((features, COMMON), (targets, ("y",))):
        values = frame.loc[:, list(columns)].to_numpy(float)
        if np.isinf(values).any():
            raise ValueError("No infinite feature or target values")


def complete_mask(features):
    return (
        np.isfinite(features.loc[:, list(COMMON)]).all(axis=1)
        & features.feature_cutoff_date.notna()
    )


def training_mask(features, targets, fit_origin, minimum_train=1000):
    alignment(features, targets)
    if type(minimum_train) is not int or minimum_train < 2 or fit_origin not in features.index:
        raise ValueError("Literal training minimum and observed fit origin required")
    cutoff = features.loc[fit_origin, "feature_cutoff_date"]
    mask = (
        complete_mask(features)
        & np.isfinite(targets.y)
        & (features.index < fit_origin)
        & (targets.available_date <= cutoff)
    )
    if int(mask.sum()) < minimum_train:
        raise ValueError(
            f"INSUFFICIENT_DATA: {int(mask.sum())} complete mature training observations"
        )
    return mask


def validate_panel(panel):
    if list(panel.columns) != list(PANEL_COLUMNS) or panel.empty:
        raise ValueError("Complete nonempty scored panel schema required")
    if (
        panel.duplicated(["origin", "model"]).any()
        or set(panel.model) != set(MODELS)
        or set(panel.horizon) != {21}
    ):
        raise ValueError("Unique complete four-model21-session panel required")
    if (
        not panel.sort_values(["origin", "model"])
        .reset_index(drop=True)
        .equals(panel.reset_index(drop=True))
    ):
        raise ValueError("Canonical origin/model ordering required")
    if (
        not np.isfinite(panel[["prediction", "y", "loss"]].to_numpy(float)).all()
        or (panel.loss < 0).any()
        or not set(panel.phase).issubset({"development", "evaluation"})
        or not pd.api.types.is_integer_dtype(panel.train_n.dtype)
        or (panel.train_n < 2).any()
    ):
        raise ValueError("Finite squared losses and literal training counts required")
    residual = panel.y.to_numpy(float) - panel.prediction.to_numpy(float)
    with np.errstate(all="ignore"):
        expected = residual * residual
    if (
        not np.isfinite(expected).all()
        or ((residual != 0) & (expected == 0)).any()
        or not np.array_equal(panel.loss.to_numpy(float), expected)
    ):
        raise ValueError("Exact finite squared prediction error required")
    reference = panel.loc[panel.model == "mean"].set_index("origin")
    paired = (
        "fit_origin",
        "feature_cutoff_date",
        "target_end",
        "available_date",
        "horizon",
        "phase",
        "y",
        "train_n",
    )
    for name in MODELS:
        part = panel.loc[panel.model == name].set_index("origin")
        if not reference.index.equals(part.index) or any(
            not reference[c].equals(part[c]) for c in paired
        ):
            raise ValueError(
                "Exact common origins, targets, clocks and training metadata required"
            )
    for name in (
        "origin",
        "fit_origin",
        "feature_cutoff_date",
        "target_end",
        "available_date",
    ):
        if panel[name].isna().any() or not pd.api.types.is_datetime64_any_dtype(
            panel[name].dtype
        ):
            raise ValueError("Native known panel dates required")
    if (
        not (panel.feature_cutoff_date < panel.origin).all()
        or not (panel.fit_origin <= panel.origin).all()
        or not (panel.target_end > panel.origin).all()
        or not panel.target_end.equals(panel.available_date)
    ):
        raise ValueError("Causal panel clocks required")


def build_panel(features, targets, feature_states, protocol):
    config = protocol["index"]
    if (
        tuple(config["models"]) != MODELS
        or config["horizons"] != [21]
        or config["market_lag"] != 1
        or tuple(config["raw"]) != RAW
        or tuple(config["baseline"]) != BASE
        or tuple(config["common"]) != COMMON
        or not feature_states.index.equals(features.index)
    ):
        raise ValueError("Fixed matched feature/model family and feature-state index required")
    alignment(features, targets)
    dates = features.index
    start, end = pd.Timestamp(config["origin_start"]), pd.Timestamp(config["origin_end"])
    ds, de = map(pd.Timestamp, config["development"])
    es, ee = map(pd.Timestamp, config["evaluation"])
    latest = pd.Timestamp(config["latest_target"])
    if (
        not start <= ds <= de < es <= ee <= end <= latest
        or config["development_target_available_by"] != config["development"][1]
        or latest > pd.Timestamp(config["source_end"])
        or dates.max() > pd.Timestamp(config["source_end"])
        or pd.Timestamp(config["source_end"]) >= pd.Timestamp(config["sealed_start"])
    ):
        raise ValueError("Fixed phase and protected outcome boundaries required")
    complete = complete_mask(features)
    phase = ((dates >= ds) & (dates <= de)) | ((dates >= es) & (dates <= ee))
    entries = dates[complete & phase & (dates >= start) & (dates <= end)]
    if not len(entries):
        raise ValueError("INSUFFICIENT_DATA: no complete applications")
    ready = (
        np.isfinite(targets.y)
        & (targets.available_date <= latest)
        & ((dates > de) | (targets.available_date <= de))
    )
    states, rows, fits = [], [], []
    for month in entries.to_period("M").unique():
        application = entries[entries.to_period("M") == month]
        fit_origin = application[0]
        mask = training_mask(features, targets, fit_origin, config["minimum_train"])
        fitted = models.fit_predict(
            features.loc[mask], targets.loc[mask, "y"], features.loc[application]
        )
        if set(fitted["predictions"]) != set(MODELS):
            raise ValueError("All four fitted model predictions required")
        cutoff = features.loc[fit_origin, "feature_cutoff_date"]
        fits.append(
            {
                "horizon": 21,
                "fit_origin": str(fit_origin.date()),
                "feature_cutoff_date": str(cutoff.date()),
                "train_n": int(mask.sum()),
                "train_origins": [str(x.date()) for x in dates[mask]],
                "application_origins": [str(x.date()) for x in application],
                "model_audit": fitted["model_audit"],
                "transform_audit": fitted["transform_audit"],
                "scalar_audit": fitted["scalar_audit"],
            }
        )
        state = pd.DataFrame(
            {
                "origin": application,
                "fit_origin": fit_origin,
                "feature_cutoff_date": features.loc[
                    application, "feature_cutoff_date"
                ].to_numpy(),
                "train_n": int(mask.sum()),
            }
        )
        selected = ready.loc[application].to_numpy()
        scored = application[selected]
        for name in MODELS:
            prediction = np.asarray(fitted["predictions"][name], float)
            if prediction.shape != (len(application),) or not np.isfinite(prediction).all():
                raise ValueError("Finite predictions for every full application required")
            state["pred_" + name] = prediction
            one = pd.DataFrame(
                {
                    "origin": scored,
                    "fit_origin": fit_origin,
                    "feature_cutoff_date": features.loc[
                        scored, "feature_cutoff_date"
                    ].to_numpy(),
                    "horizon": 21,
                    "phase": np.where(scored <= de, "development", "evaluation"),
                    "model": name,
                    "prediction": prediction[selected],
                    "train_n": int(mask.sum()),
                }
            )
            for column in ("y", "target_end", "available_date"):
                one[column] = targets.loc[scored, column].to_numpy()
            residual = one.y.to_numpy(float) - one.prediction.to_numpy(float)
            with np.errstate(all="ignore"):
                loss = residual * residual
            if not np.isfinite(loss).all() or ((residual != 0) & (loss == 0)).any():
                raise ValueError("Nonfinite or underflowed required squared loss")
            one["loss"] = loss
            rows.append(one.loc[:, list(PANEL_COLUMNS)])
        states.append(state.loc[:, list(STATE_COLUMNS)])
    full = pd.concat(states, ignore_index=True).sort_values("origin").reset_index(drop=True)
    panel = (
        pd.concat(rows, ignore_index=True)
        .sort_values(["origin", "model"])
        .reset_index(drop=True)
    )
    validate_panel(panel)
    unscored = full.loc[~full.origin.isin(panel.origin), "origin"]
    support = {
        "calendar_rows": len(features),
        "common_feature_rows": int(complete.sum()),
        "monthly_fits": len(fits),
        "application_origins": len(full),
        "scored_origins": int(panel.origin.nunique()),
        "unscored_origins": len(unscored),
        "unscored_origin_dates": [str(x.date()) for x in unscored],
        "phases": {},
    }
    for phase in ("development", "evaluation"):
        origins = panel.loc[panel.phase == phase, "origin"].drop_duplicates()
        support["phases"][phase] = {
            "n": len(origins),
            "first_origin": str(origins.iloc[0].date()) if len(origins) else None,
            "last_origin": str(origins.iloc[-1].date()) if len(origins) else None,
        }
    return panel, fits, full, support
