"""Post-hoc HMM calibration and incremental-state test on a fixed holdout."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import yaml

from . import config
from .jump_target import _fit_logistic, _losses, _predict_logistic
from .regime_transition import (
    filter_gaussian_hmm,
    fit_gaussian_hmm,
    future_exceedance_probability,
)

PROTOCOL_PATH = config.ROOT / "regime_repair.yaml"
PRIOR_FORECASTS = config.ROOT / "data" / "target_regime" / "regime_forecasts.parquet"
OUTPUT_DIR = config.ROOT / "data" / "target_regime"
REPORT_PATH = config.ROOT / "reports" / "regime_repair.md"


def load_protocol(path=PROTOCOL_PATH) -> dict:
    with open(path) as source:
        return yaml.safe_load(source)


def validate_protocol(protocol: dict) -> None:
    training = protocol["training"]
    evaluation = protocol["evaluation"]
    oof_start = pd.Timestamp(training["oof_start"])
    oof_end = pd.Timestamp(training["oof_end"])
    hmm_end = pd.Timestamp(training["hmm_fit_end"])
    start = pd.Timestamp(evaluation["start"])
    end = pd.Timestamp(evaluation["end"])
    data_end = pd.Timestamp(evaluation["required_data_end"])
    if not oof_start <= oof_end < start <= end < data_end:
        raise ValueError("training must end before the ordered evaluation window")
    if hmm_end > oof_end:
        raise ValueError("HMM fit must end no later than OOF training")
    if not training.get("no_holdout_refit", False):
        raise ValueError("holdout refit must be forbidden")
    if protocol["calibration"].get("method") != "platt":
        raise ValueError("the frozen repair permits only Platt calibration")
    if not evaluation.get("score_only_calm_origins", False):
        raise ValueError("repair must score only calm origins")
    if not evaluation.get("require_completed_targets", False):
        raise ValueError("repair requires completed targets")
    if not protocol["incremental_test"].get("same_training_rows_required", False):
        raise ValueError("benchmark and augmented models must use the same rows")


def _logit(probability: pd.Series, clip: float) -> pd.DataFrame:
    clipped = probability.astype(float).clip(float(clip), 1.0 - float(clip))
    return pd.DataFrame({"hmm_logit": np.log(clipped / (1.0 - clipped))}, index=probability.index)


def fit_platt(
    raw_probability: pd.Series,
    event: pd.Series,
    *,
    ridge: float,
    clip: float,
) -> dict:
    frame = _logit(raw_probability, clip).join(event.rename("event")).dropna()
    if frame.empty:
        raise ValueError("Platt calibration has no complete training rows")
    model = _fit_logistic(frame[["hmm_logit"]], frame["event"], float(ridge))
    model["clip"] = float(clip)
    return model


def predict_platt(model: dict, raw_probability: pd.Series) -> pd.Series:
    features = _logit(raw_probability, float(model["clip"]))
    values = _predict_logistic(model, features)
    return pd.Series(values, index=raw_probability.index, name="p_hmm_platt")


def build_fixed_targets(
    y: pd.Series,
    *,
    origins: pd.DatetimeIndex,
    threshold: float,
    horizon: int,
    evaluation_end: pd.Timestamp,
) -> pd.DataFrame:
    series = y.sort_index()
    positions = pd.Series(np.arange(len(series)), index=series.index)
    rows = []
    for origin in pd.DatetimeIndex(origins):
        if origin > pd.Timestamp(evaluation_end) or origin not in positions:
            continue
        position = int(positions.loc[origin])
        future = series.iloc[position + 1 : position + 1 + int(horizon)]
        if len(future) != int(horizon) or future.isna().any():
            continue
        rows.append({
            "origin": origin,
            "target_end": future.index[-1],
            "threshold": float(threshold),
            "calm": bool(series.loc[origin] <= float(threshold)),
            "event": bool((future > float(threshold)).any()),
        })
    return pd.DataFrame(rows).set_index("origin") if rows else pd.DataFrame()


def require_same_rows(benchmark: pd.DataFrame, augmented: pd.DataFrame) -> None:
    if not benchmark.index.equals(augmented.index):
        raise ValueError("benchmark and augmented models must use the same rows")


def evaluate_phases(frame: pd.DataFrame, *, step: int) -> pd.DataFrame:
    rows = []
    for phase in range(int(step)):
        sample = frame.iloc[phase::step]
        if sample.empty:
            continue
        row = {"phase": phase, "n": len(sample), "event_rate": float(sample["event"].mean())}
        for model in ("hmm_raw", "hmm_platt", "benchmark", "augmented"):
            brier, logloss = _losses(sample, f"p_{model}")
            row[f"{model}_brier"] = brier
            row[f"{model}_logloss"] = logloss
        rows.append(row)
    return pd.DataFrame(rows)


def _serializable_logistic(model: dict) -> dict:
    return {
        "beta": np.asarray(model["beta"]).tolist(),
        "mean": np.asarray(model["mean"]).tolist(),
        "scale": np.asarray(model["scale"]).tolist(),
        "columns": list(model["columns"]),
        **({"clip": float(model["clip"])} if "clip" in model else {}),
    }


def run_study(protocol: dict | None = None) -> dict:
    protocol = protocol or load_protocol()
    validate_protocol(protocol)
    main_cfg = config.load()
    y = pd.read_parquet(main_cfg["paths"]["processed"] / "master_daily.parquet")["log_rv"].dropna().sort_index()
    features = pd.DataFrame({
        "log_rv_d": y,
        "log_rv_w": y.rolling(5).mean(),
        "log_rv_m": y.rolling(22).mean(),
    })

    training = protocol["training"]
    evaluation = protocol["evaluation"]
    oof_start = pd.Timestamp(training["oof_start"])
    oof_end = pd.Timestamp(training["oof_end"])
    hmm_end = pd.Timestamp(training["hmm_fit_end"])
    start = pd.Timestamp(evaluation["start"])
    end = pd.Timestamp(evaluation["end"])
    required_data_end = pd.Timestamp(evaluation["required_data_end"])
    horizon = int(evaluation["horizon_sessions"])
    quantile = float(evaluation["stress_quantile"])
    if y.index.max() < required_data_end:
        raise RuntimeError("master data end precedes the frozen required data end")

    prior = pd.read_parquet(PRIOR_FORECASTS).sort_index()
    prior = prior.loc[(prior.index >= oof_start) & (prior.index <= oof_end)].copy()
    base_columns = list(protocol["incremental_test"]["benchmark_features"])
    needed = base_columns + ["p_hmm", "event"]
    train = prior.loc[:, needed].dropna()
    if len(train) < int(training["min_oof_observations"]):
        raise RuntimeError(f"repair has only {len(train)} prior OOF rows")
    if train.index.max() > oof_end:
        raise RuntimeError("repair training rows cross the OOF cutoff")

    calibration = protocol["calibration"]
    platt = fit_platt(
        train["p_hmm"], train["event"],
        ridge=float(calibration["ridge"]), clip=float(calibration["clip"]),
    )
    train["p_hmm_platt"] = predict_platt(platt, train["p_hmm"])
    benchmark_train = train.loc[:, base_columns + ["event"]]
    augmented_columns = base_columns + [protocol["incremental_test"]["augmented_feature"]]
    augmented_train = train.loc[:, augmented_columns + ["event"]]
    require_same_rows(benchmark_train, augmented_train)
    ridge = float(protocol["incremental_test"]["logistic_ridge"])
    benchmark = _fit_logistic(benchmark_train[base_columns], benchmark_train["event"], ridge)
    augmented = _fit_logistic(augmented_train[augmented_columns], augmented_train["event"], ridge)

    train_y = y.loc[:hmm_end]
    threshold = float(train_y.quantile(quantile))
    hmm = fit_gaussian_hmm(train_y)
    filtered_source = y.loc[:end]
    filtered = pd.DataFrame(
        filter_gaussian_hmm(filtered_source, hmm),
        index=filtered_source.index,
        columns=["p_low_filtered", "p_high_filtered"],
    )
    origins = y.index[(y.index >= start) & (y.index <= end)]
    targets = build_fixed_targets(
        y, origins=origins, threshold=threshold, horizon=horizon, evaluation_end=end,
    )
    targets = targets.loc[targets["calm"]]
    if targets.empty:
        raise RuntimeError("repair produced no calm holdout targets")
    if targets["target_end"].max() > required_data_end:
        raise RuntimeError("repair target escapes the frozen data end")
    common = targets.index.intersection(features[base_columns].dropna().index)
    scored = targets.loc[common].copy()
    for column in base_columns:
        scored[column] = features.loc[common, column]
    scored["p_low_filtered"] = filtered.loc[common, "p_low_filtered"]
    scored["p_high_filtered"] = filtered.loc[common, "p_high_filtered"]
    scored["p_hmm_raw"] = [
        future_exceedance_probability(
            filtered.loc[origin].to_numpy(), hmm, threshold=threshold, horizon=horizon
        )
        for origin in common
    ]
    scored["p_hmm_platt"] = predict_platt(platt, scored["p_hmm_raw"])
    scored["p_benchmark"] = _predict_logistic(benchmark, scored[base_columns])
    scored["p_augmented"] = _predict_logistic(augmented, scored[augmented_columns])

    phases = evaluate_phases(scored, step=horizon)
    avg = phases.mean(numeric_only=True)
    checks = {
        "platt_brier_below_raw": bool(avg["hmm_platt_brier"] < avg["hmm_raw_brier"]),
        "platt_logloss_below_raw": bool(avg["hmm_platt_logloss"] < avg["hmm_raw_logloss"]),
        "augmented_brier_below_benchmark": bool(avg["augmented_brier"] < avg["benchmark_brier"]),
        "augmented_logloss_below_benchmark": bool(avg["augmented_logloss"] < avg["benchmark_logloss"]),
    }
    calibration_pass = bool(checks["platt_brier_below_raw"] and checks["platt_logloss_below_raw"])
    incremental_pass = bool(
        checks["augmented_brier_below_benchmark"]
        and checks["augmented_logloss_below_benchmark"]
    )
    ranked = scored.sort_values("p_hmm_platt")
    quintile = max(1, len(ranked) // 5)
    metadata = {
        "oof_train_start": str(train.index.min().date()),
        "oof_train_end": str(train.index.max().date()),
        "oof_train_rows": len(train),
        "hmm_fit_end": str(train_y.index.max().date()),
        "threshold": threshold,
        "hmm": {key: np.asarray(value).tolist() for key, value in hmm.items()},
        "platt": _serializable_logistic(platt),
        "benchmark": _serializable_logistic(benchmark),
        "augmented": _serializable_logistic(augmented),
    }
    metrics = {
        "checks": checks,
        "calibration_pass": calibration_pass,
        "incremental_state_pass": incremental_pass,
        "origins": len(scored),
        "event_rate": float(scored["event"].mean()),
        "platt_top_quintile_event_rate": float(ranked.iloc[-quintile:]["event"].mean()),
        "platt_bottom_quintile_event_rate": float(ranked.iloc[:quintile]["event"].mean()),
        "phase_average": {key: float(value) for key, value in avg.items()},
        "training_and_models": metadata,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(OUTPUT_DIR / "regime_repair_forecasts.parquet")
    phases.to_parquet(OUTPUT_DIR / "regime_repair_phase_metrics.parquet", index=False)
    (OUTPUT_DIR / "regime_repair_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    _write_report(metrics)
    print(
        "regime repair: calibration "
        f"{'PASS' if calibration_pass else 'FAIL'}, incremental state "
        f"{'PASS' if incremental_pass else 'FAIL'}"
    )
    return metrics


def _write_report(metrics: dict) -> None:
    avg = metrics["phase_average"]
    lines = [
        "# HMM calibration and incremental-state repair",
        "",
        "This post-hoc repair uses a transition-target-specific holdout. The dates are not a pristine project-wide holdout.",
        "All calibration and supervised parameters were locked on annual out-of-fold rows through 2024; HMM parameters were also frozen through 2024.",
        "",
        f"- Calm holdout origins: {metrics['origins']}.",
        f"- Five-session transition event rate: {metrics['event_rate']:.1%}.",
        "",
        "| model | phase-average Brier | phase-average log loss |",
        "|---|---:|---:|",
        f"| raw HMM | {avg['hmm_raw_brier']:.6f} | {avg['hmm_raw_logloss']:.6f} |",
        f"| Platt HMM | {avg['hmm_platt_brier']:.6f} | {avg['hmm_platt_logloss']:.6f} |",
        f"| supervised benchmark | {avg['benchmark_brier']:.6f} | {avg['benchmark_logloss']:.6f} |",
        f"| benchmark + calibrated HMM | {avg['augmented_brier']:.6f} | {avg['augmented_logloss']:.6f} |",
        "",
        f"Calibration verdict: **{'PASS' if metrics['calibration_pass'] else 'FAIL'}**.",
        f"Incremental-state verdict: **{'PASS' if metrics['incremental_state_pass'] else 'FAIL'}**.",
        "",
        f"The calibrated HMM top probability quintile realized {metrics['platt_top_quintile_event_rate']:.1%} events versus {metrics['platt_bottom_quintile_event_rate']:.1%} in the bottom quintile.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",), nargs="?", default="run")
    parser.parse_args(argv)
    run_study()


if __name__ == "__main__":
    main()
