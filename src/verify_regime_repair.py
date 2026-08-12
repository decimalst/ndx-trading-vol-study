"""Independent audit of the stored HMM repair forecasts and verdicts."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml
from scipy.special import expit, ndtr

from . import config
from .regime_transition import fit_gaussian_hmm


ROOT = config.ROOT
OUT = ROOT / "data" / "target_regime"


def _fit_logistic(X: pd.DataFrame, y: pd.Series, ridge: float) -> dict:
    values = X.to_numpy(dtype=float)
    target = y.to_numpy(dtype=float)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (values - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    beta = np.zeros(design.shape[1])
    penalty = np.eye(design.shape[1]) * float(ridge)
    penalty[0, 0] = 0.0
    for _ in range(100):
        probability = expit(design @ beta)
        weight = np.clip(probability * (1.0 - probability), 1e-6, None)
        hessian = design.T @ (design * weight[:, None]) + penalty
        gradient = design.T @ (target - probability) - penalty @ beta
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return {"beta": beta, "mean": mean, "scale": scale, "columns": list(X.columns)}


def _predict(model: dict, X: pd.DataFrame) -> np.ndarray:
    values = X.loc[:, model["columns"]].to_numpy(dtype=float)
    z = (values - np.asarray(model["mean"])) / np.asarray(model["scale"])
    design = np.column_stack([np.ones(len(z)), z])
    return np.clip(expit(design @ np.asarray(model["beta"])), 1e-6, 1 - 1e-6)


def _assert_model_equal(actual: dict, expected: dict, name: str) -> None:
    if list(actual["columns"]) != list(expected["columns"]):
        raise AssertionError(f"{name} columns do not reproduce")
    for key in ("beta", "mean", "scale"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-12)


def _filter(y: pd.Series, hmm: dict) -> pd.DataFrame:
    values = y.to_numpy(dtype=float)
    means = np.asarray(hmm["means"])
    variances = np.asarray(hmm["variances"])
    transition = np.asarray(hmm["transition"])
    log_density = -0.5 * (
        np.log(2 * np.pi * variances)[None, :]
        + (values[:, None] - means[None, :]) ** 2 / variances[None, :]
    )
    emissions = np.exp(log_density - log_density.max(axis=1)[:, None])
    filtered = np.empty((len(values), 2))
    filtered[0] = np.asarray(hmm["initial"]) * emissions[0]
    filtered[0] /= filtered[0].sum()
    for position in range(1, len(values)):
        filtered[position] = (filtered[position - 1] @ transition) * emissions[position]
        filtered[position] /= filtered[position].sum()
    return pd.DataFrame(filtered, index=y.index, columns=["p_low_filtered", "p_high_filtered"])


def _future_probability(filtered: np.ndarray, hmm: dict, threshold: float, horizon: int) -> float:
    survive = np.asarray(filtered, dtype=float).copy()
    means = np.asarray(hmm["means"])
    variances = np.asarray(hmm["variances"])
    transition = np.asarray(hmm["transition"])
    below = ndtr((float(threshold) - means) / np.sqrt(variances))
    for _ in range(int(horizon)):
        survive = (survive @ transition) * below
    return float(np.clip(1.0 - survive.sum(), 1e-6, 1 - 1e-6))


def _losses(frame: pd.DataFrame, probability: str) -> tuple[float, float]:
    y = frame["event"].to_numpy(dtype=float)
    p = np.clip(frame[probability].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    return float(np.mean((p - y) ** 2)), float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def _phases(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    rows = []
    for phase in range(int(horizon)):
        sample = frame.iloc[phase::horizon]
        row = {"phase": phase, "n": len(sample), "event_rate": float(sample["event"].mean())}
        for model in ("hmm_raw", "hmm_platt", "benchmark", "augmented"):
            brier, logloss = _losses(sample, f"p_{model}")
            row[f"{model}_brier"] = brier
            row[f"{model}_logloss"] = logloss
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    protocol = yaml.safe_load((ROOT / "regime_repair.yaml").read_text())
    training = protocol["training"]
    evaluation = protocol["evaluation"]
    metrics = json.loads((OUT / "regime_repair_metrics.json").read_text())
    metadata = metrics["training_and_models"]
    stored = pd.read_parquet(OUT / "regime_repair_forecasts.parquet").sort_index()

    cfg = config.load()
    y = pd.read_parquet(cfg["paths"]["processed"] / "master_daily.parquet")["log_rv"].dropna().sort_index()
    oof = pd.read_parquet(OUT / "regime_forecasts.parquet").sort_index()
    oof_start = pd.Timestamp(training["oof_start"])
    oof_end = pd.Timestamp(training["oof_end"])
    oof = oof.loc[(oof.index >= oof_start) & (oof.index <= oof_end)]
    columns = list(protocol["incremental_test"]["benchmark_features"])
    train = oof.loc[:, columns + ["p_hmm", "event"]].dropna()
    if train.index.max() > oof_end or str(train.index.max().date()) != metadata["oof_train_end"]:
        raise AssertionError("repair training crosses or misstates its cutoff")
    if len(train) != metadata["oof_train_rows"]:
        raise AssertionError("repair OOF training count does not reproduce")

    calibration = protocol["calibration"]
    clip = float(calibration["clip"])
    clipped = train["p_hmm"].clip(clip, 1 - clip)
    platt_X = pd.DataFrame({"hmm_logit": np.log(clipped / (1 - clipped))}, index=train.index)
    platt = _fit_logistic(platt_X, train["event"], float(calibration["ridge"]))
    _assert_model_equal(platt, metadata["platt"], "Platt")
    train["p_hmm_platt"] = _predict(platt, platt_X)
    benchmark = _fit_logistic(
        train[columns], train["event"], float(protocol["incremental_test"]["logistic_ridge"])
    )
    augmented_columns = columns + [protocol["incremental_test"]["augmented_feature"]]
    augmented = _fit_logistic(
        train[augmented_columns], train["event"],
        float(protocol["incremental_test"]["logistic_ridge"]),
    )
    _assert_model_equal(benchmark, metadata["benchmark"], "benchmark")
    _assert_model_equal(augmented, metadata["augmented"], "augmented")

    hmm_end = pd.Timestamp(training["hmm_fit_end"])
    train_y = y.loc[:hmm_end]
    if str(train_y.index.max().date()) != metadata["hmm_fit_end"]:
        raise AssertionError("HMM fit cutoff does not reproduce")
    refitted_hmm = fit_gaussian_hmm(train_y)
    hmm = metadata["hmm"]
    for key in ("initial", "transition", "means", "variances"):
        np.testing.assert_allclose(refitted_hmm[key], hmm[key], rtol=0, atol=1e-12)
    threshold = float(train_y.quantile(float(evaluation["stress_quantile"])))
    np.testing.assert_allclose(threshold, metadata["threshold"], rtol=0, atol=1e-15)

    start = pd.Timestamp(evaluation["start"])
    end = pd.Timestamp(evaluation["end"])
    data_end = pd.Timestamp(evaluation["required_data_end"])
    horizon = int(evaluation["horizon_sessions"])
    if stored.index.min() < start or stored.index.max() > end or stored["target_end"].max() > data_end:
        raise AssertionError("stored repair rows escape the target-specific fence")
    positions = pd.Series(np.arange(len(y)), index=y.index)
    for origin, row in stored.iterrows():
        position = int(positions.loc[origin])
        future = y.iloc[position + 1 : position + 1 + horizon]
        if len(future) != horizon or row["target_end"] != future.index[-1]:
            raise AssertionError(f"incomplete repair target at {origin.date()}")
        if bool(row["event"]) != bool((future > threshold).any()):
            raise AssertionError(f"repair event mismatch at {origin.date()}")
        if y.loc[origin] > threshold:
            raise AssertionError(f"non-calm repair origin at {origin.date()}")

    filtered = _filter(y.loc[:end], hmm)
    for column in ("p_low_filtered", "p_high_filtered"):
        np.testing.assert_allclose(stored[column], filtered.loc[stored.index, column], rtol=0, atol=1e-12)
    raw = np.array([
        _future_probability(filtered.loc[origin].to_numpy(), hmm, threshold, horizon)
        for origin in stored.index
    ])
    np.testing.assert_allclose(stored["p_hmm_raw"], raw, rtol=0, atol=1e-12)
    holdout_logit = pd.DataFrame({
        "hmm_logit": np.log(np.clip(raw, clip, 1 - clip) / (1 - np.clip(raw, clip, 1 - clip)))
    }, index=stored.index)
    np.testing.assert_allclose(stored["p_hmm_platt"], _predict(platt, holdout_logit), rtol=0, atol=1e-12)
    np.testing.assert_allclose(stored["p_benchmark"], _predict(benchmark, stored[columns]), rtol=0, atol=1e-12)
    augmented_X = stored[columns].copy()
    augmented_X["p_hmm_platt"] = stored["p_hmm_platt"]
    np.testing.assert_allclose(stored["p_augmented"], _predict(augmented, augmented_X), rtol=0, atol=1e-12)

    phases = _phases(stored, horizon)
    recorded_phases = pd.read_parquet(OUT / "regime_repair_phase_metrics.parquet")
    pd.testing.assert_frame_equal(phases, recorded_phases, check_exact=False, rtol=0, atol=1e-15)
    avg = phases.mean(numeric_only=True)
    checks = {
        "platt_brier_below_raw": bool(avg["hmm_platt_brier"] < avg["hmm_raw_brier"]),
        "platt_logloss_below_raw": bool(avg["hmm_platt_logloss"] < avg["hmm_raw_logloss"]),
        "augmented_brier_below_benchmark": bool(avg["augmented_brier"] < avg["benchmark_brier"]),
        "augmented_logloss_below_benchmark": bool(avg["augmented_logloss"] < avg["benchmark_logloss"]),
    }
    if checks != metrics["checks"]:
        raise AssertionError("repair verdict inputs do not reproduce")
    print(
        "REGIME REPAIR VERIFICATION PASSED: OOF/calibration/HMM cutoffs, fixed threshold, "
        "completed targets, forward filter, all four probabilities, phase losses, and verdicts match"
    )


if __name__ == "__main__":
    main()
