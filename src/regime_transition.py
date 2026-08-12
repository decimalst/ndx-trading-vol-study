"""Two-state, forward-filtered QQQ stress-transition diagnostic."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import yaml
from scipy.special import expit, ndtr

from . import config
from .jump_target import _fit_logistic, _predict_logistic, _losses


PROTOCOL_PATH = config.ROOT / "target_regime.yaml"
OUTPUT_DIR = config.ROOT / "data" / "target_regime"
REPORT_PATH = config.ROOT / "reports" / "regime_transition.md"


def load_protocol(path=PROTOCOL_PATH) -> dict:
    with open(path) as source:
        return yaml.safe_load(source)["regime_transition"]


def validate_protocol(protocol: dict) -> None:
    window = protocol["window"]
    start = pd.Timestamp(window["start"])
    end = pd.Timestamp(window["end"])
    clean = pd.Timestamp(window["clean_start"])
    if not start <= end < clean:
        raise ValueError("regime diagnostic window must stop before clean")
    if not window.get("forbid_clean_origins", False):
        raise ValueError("clean origins must be forbidden")
    target = protocol["target"]
    if not target.get("score_only_calm_origins", False):
        raise ValueError("transition study must score only calm origins")
    fitting = protocol["fitting"]
    if int(fitting["hmm_states"]) != 2:
        raise ValueError("only the frozen two-state HMM is allowed")


def enforce_target_fence(
    origins: pd.DatetimeIndex,
    full_index: pd.DatetimeIndex,
    *,
    horizon: int,
    clean_start: pd.Timestamp,
) -> None:
    positions = pd.Series(np.arange(len(full_index)), index=full_index)
    for origin in pd.DatetimeIndex(origins):
        if origin not in positions:
            raise ValueError(f"origin {origin} missing from target index")
        end_position = int(positions.loc[origin]) + int(horizon)
        if end_position >= len(full_index) or full_index[end_position] >= pd.Timestamp(clean_start):
            raise ValueError("regime target reaches or crosses the clean window")


def completed_training_origins(
    index: pd.DatetimeIndex, *, cutoff: pd.Timestamp, horizon: int
) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(index).sort_values()
    cutoff_position = int(index.searchsorted(pd.Timestamp(cutoff), side="right") - 1)
    last = cutoff_position - int(horizon)
    return index[: last + 1] if last >= 0 else pd.DatetimeIndex([])


def build_fold_targets(
    y: pd.Series,
    *,
    cutoff: pd.Timestamp,
    origins: pd.DatetimeIndex,
    horizon: int,
    quantile: float,
) -> pd.DataFrame:
    series = y.sort_index()
    history = series.loc[:pd.Timestamp(cutoff)].dropna()
    if history.empty:
        raise ValueError("regime fold has no threshold history")
    threshold = float(history.quantile(float(quantile)))
    positions = pd.Series(np.arange(len(series)), index=series.index)
    rows = []
    for origin in pd.DatetimeIndex(origins):
        if origin not in positions:
            continue
        position = int(positions.loc[origin])
        future = series.iloc[position + 1 : position + 1 + int(horizon)]
        if len(future) != int(horizon) or future.isna().any():
            continue
        rows.append({
            "origin": origin,
            "target_end": future.index[-1],
            "threshold": threshold,
            "calm": bool(series.loc[origin] <= threshold),
            "event": bool((future > threshold).any()),
        })
    return pd.DataFrame(rows).set_index("origin") if rows else pd.DataFrame()


def _relative_emissions(y: np.ndarray, means: np.ndarray, variances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    log_density = -0.5 * (
        np.log(2 * np.pi * variances)[None, :]
        + (y[:, None] - means[None, :]) ** 2 / variances[None, :]
    )
    offset = log_density.max(axis=1)
    return np.exp(log_density - offset[:, None]), offset


def _forward_backward(y: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray, float]:
    transition = params["transition"]
    emissions, offset = _relative_emissions(y, params["means"], params["variances"])
    n = len(y)
    alpha = np.empty((n, 2))
    scale = np.empty(n)
    alpha[0] = params["initial"] * emissions[0]
    scale[0] = alpha[0].sum()
    alpha[0] /= scale[0]
    for t in range(1, n):
        alpha[t] = (alpha[t - 1] @ transition) * emissions[t]
        scale[t] = alpha[t].sum()
        alpha[t] /= scale[t]
    beta = np.ones((n, 2))
    for t in range(n - 2, -1, -1):
        beta[t] = transition @ (emissions[t + 1] * beta[t + 1])
        beta[t] /= scale[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi_sum = np.zeros((2, 2))
    for t in range(n - 1):
        xi = alpha[t, :, None] * transition * (emissions[t + 1] * beta[t + 1])[None, :]
        xi_sum += xi / xi.sum()
    loglik = float(np.sum(np.log(scale) + offset))
    return gamma, xi_sum, loglik


def fit_gaussian_hmm(
    y: np.ndarray | pd.Series, *, max_iter: int = 200, tolerance: float = 1e-7
) -> dict:
    values = np.asarray(y, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 10:
        raise ValueError("two-state HMM needs at least 10 finite observations")
    overall_variance = max(float(np.var(values)), 1e-4)
    params = {
        "initial": np.array([0.8, 0.2], dtype=float),
        "transition": np.array([[0.97, 0.03], [0.08, 0.92]], dtype=float),
        "means": np.quantile(values, [0.30, 0.80]).astype(float),
        "variances": np.array([overall_variance * 0.5, overall_variance], dtype=float),
    }
    previous = -np.inf
    floor = max(overall_variance * 1e-4, 1e-6)
    for _ in range(int(max_iter)):
        gamma, xi_sum, loglik = _forward_backward(values, params)
        weights = gamma.sum(axis=0)
        means = (gamma * values[:, None]).sum(axis=0) / weights
        variances = (gamma * (values[:, None] - means[None, :]) ** 2).sum(axis=0) / weights
        transition = xi_sum / xi_sum.sum(axis=1, keepdims=True)
        transition = np.clip(transition, 1e-6, 1.0)
        transition /= transition.sum(axis=1, keepdims=True)
        params = {
            "initial": np.clip(gamma[0], 1e-6, 1.0),
            "transition": transition,
            "means": means,
            "variances": np.maximum(variances, floor),
        }
        params["initial"] /= params["initial"].sum()
        if np.isfinite(previous) and abs(loglik - previous) <= float(tolerance) * (1 + abs(previous)):
            break
        previous = loglik
    order = np.argsort(params["means"])
    return {
        "initial": params["initial"][order],
        "transition": params["transition"][np.ix_(order, order)],
        "means": params["means"][order],
        "variances": params["variances"][order],
    }


def filter_gaussian_hmm(y: np.ndarray | pd.Series, params: dict) -> np.ndarray:
    values = np.asarray(y, dtype=float)
    emissions, _ = _relative_emissions(values, params["means"], params["variances"])
    filtered = np.empty((len(values), 2))
    filtered[0] = params["initial"] * emissions[0]
    filtered[0] /= filtered[0].sum()
    for t in range(1, len(values)):
        filtered[t] = (filtered[t - 1] @ params["transition"]) * emissions[t]
        filtered[t] /= filtered[t].sum()
    return filtered


def future_exceedance_probability(
    filtered: np.ndarray,
    params: dict,
    *,
    threshold: float,
    horizon: int,
) -> float:
    survive = np.asarray(filtered, dtype=float).copy()
    below = ndtr((float(threshold) - params["means"]) / np.sqrt(params["variances"]))
    for _ in range(int(horizon)):
        survive = (survive @ params["transition"]) * below
    return float(np.clip(1.0 - survive.sum(), 1e-6, 1 - 1e-6))


def evaluate_phases(frame: pd.DataFrame, step: int = 5) -> pd.DataFrame:
    rows = []
    for phase in range(step):
        sample = frame.iloc[phase::step]
        if sample.empty:
            continue
        row = {"phase": phase, "n": len(sample), "event_rate": float(sample["event"].mean())}
        for model in ("logistic", "hmm"):
            brier, logloss = _losses(sample, f"p_{model}")
            row[f"{model}_brier"] = brier
            row[f"{model}_logloss"] = logloss
        rows.append(row)
    return pd.DataFrame(rows)


def run_study(protocol: dict | None = None) -> dict:
    protocol = protocol or load_protocol()
    validate_protocol(protocol)
    main_cfg = config.load()
    master = pd.read_parquet(main_cfg["paths"]["processed"] / "master_daily.parquet")
    y = master["log_rv"].dropna().sort_index()
    features = pd.DataFrame(index=y.index)
    features["log_rv_d"] = y
    features["log_rv_w"] = y.rolling(5).mean()
    features["log_rv_m"] = y.rolling(22).mean()

    window = protocol["window"]
    start = pd.Timestamp(window["start"])
    end = pd.Timestamp(window["end"])
    clean = pd.Timestamp(window["clean_start"])
    horizon = int(protocol["target"]["horizon_sessions"])
    quantile = float(protocol["target"]["stress_quantile"])
    fitting = protocol["fitting"]
    min_train = int(fitting["min_train_observations"])
    rows = []
    fold_parameters = []
    for year in range(start.year, end.year + 1):
        year_start = max(start, pd.Timestamp(f"{year}-01-01"))
        year_end = min(end, pd.Timestamp(f"{year}-12-31"))
        cutoff_candidates = y.index[y.index < year_start]
        if not len(cutoff_candidates):
            continue
        cutoff = cutoff_candidates[-1]
        train_y = y.loc[:cutoff]
        if len(train_y) < min_train:
            raise RuntimeError(f"{year} HMM has only {len(train_y)} training observations")
        threshold = float(train_y.quantile(quantile))
        params = fit_gaussian_hmm(
            train_y,
            max_iter=int(fitting["hmm_max_iter"]),
            tolerance=float(fitting["hmm_tolerance"]),
        )
        extended = y.loc[:year_end]
        filtered = pd.DataFrame(
            filter_gaussian_hmm(extended, params), index=extended.index,
            columns=["p_low_filtered", "p_high_filtered"],
        )
        train_origins = completed_training_origins(y.index, cutoff=cutoff, horizon=horizon)
        train_targets = build_fold_targets(
            y, cutoff=cutoff, origins=train_origins, horizon=horizon, quantile=quantile
        )
        train_targets = train_targets.loc[train_targets["calm"]]
        columns = ["log_rv_d", "log_rv_w", "log_rv_m"]
        train = features.loc[train_targets.index, columns].join(train_targets["event"]).dropna()
        if len(train) < min_train:
            raise RuntimeError(f"{year} logistic has only {len(train)} calm training rows")
        logistic = _fit_logistic(train[columns], train["event"], float(fitting["logistic_ridge"]))

        test_origins = y.index[(y.index >= year_start) & (y.index <= year_end)]
        targets = build_fold_targets(
            y, cutoff=cutoff, origins=test_origins, horizon=horizon, quantile=quantile
        )
        if targets.empty:
            continue
        targets = targets.loc[targets["calm"] & (targets["target_end"] < clean)]
        enforce_target_fence(targets.index, y.index, horizon=horizon, clean_start=clean)
        common = targets.index.intersection(features[columns].dropna().index)
        fold = targets.loc[common].copy()
        fold["fold_year"] = year
        fold["cutoff"] = cutoff
        for column in columns:
            fold[column] = features.loc[common, column]
        fold["p_low_filtered"] = filtered.loc[common, "p_low_filtered"]
        fold["p_high_filtered"] = filtered.loc[common, "p_high_filtered"]
        fold["p_logistic"] = _predict_logistic(logistic, features.loc[common, columns])
        fold["p_hmm"] = [
            future_exceedance_probability(
                filtered.loc[origin].to_numpy(), params,
                threshold=threshold, horizon=horizon,
            )
            for origin in common
        ]
        rows.append(fold)
        fold_parameters.append({
            "year": year,
            "cutoff": str(cutoff.date()),
            "threshold": threshold,
            "means": params["means"].tolist(),
            "variances": params["variances"].tolist(),
            "transition": params["transition"].tolist(),
        })
    if not rows:
        raise RuntimeError("regime study produced no diagnostic forecasts")
    scored = pd.concat(rows).sort_index()
    if scored.index.min() < start or scored.index.max() > end or (scored.index >= clean).any():
        raise RuntimeError("regime origins escaped the registered diagnostic fence")
    phases = evaluate_phases(scored, step=horizon)
    avg = phases.mean(numeric_only=True)
    ranked = scored.sort_values("p_hmm")
    quintile = max(1, len(ranked) // 5)
    bottom_rate = float(ranked.iloc[:quintile]["event"].mean())
    top_rate = float(ranked.iloc[-quintile:]["event"].mean())
    checks = {
        "hmm_brier_below_logistic": bool(avg["hmm_brier"] < avg["logistic_brier"]),
        "hmm_logloss_below_logistic": bool(avg["hmm_logloss"] < avg["logistic_logloss"]),
        "hmm_top_quintile_above_bottom": bool(top_rate > bottom_rate),
    }
    metrics = {
        "checks": checks,
        "pass": bool(all(checks.values())),
        "origins": len(scored),
        "event_rate": float(scored["event"].mean()),
        "top_quintile_event_rate": top_rate,
        "bottom_quintile_event_rate": bottom_rate,
        "phase_average": {key: float(value) for key, value in avg.items()},
        "fold_parameters": fold_parameters,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(OUTPUT_DIR / "regime_forecasts.parquet")
    phases.to_parquet(OUTPUT_DIR / "regime_phase_metrics.parquet", index=False)
    (OUTPUT_DIR / "regime_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    _write_report(metrics)
    print(f"regime-transition verdict: {'PASS' if metrics['pass'] else 'FAIL'}")
    return metrics


def _write_report(metrics: dict) -> None:
    avg = metrics["phase_average"]
    lines = [
        "# QQQ two-state transition diagnostic",
        "",
        "This is a modeling-frame diagnostic on an already-inspected period, not "
        "a new holdout. Both models consume only realized-variance history. HMM "
        "parameters are refit annually on prior data and scored probabilities are "
        "forward-filtered, never smoothed.",
        "",
        f"- Calm origins scored: {metrics['origins']}.",
        f"- Five-session stress-entry rate: {metrics['event_rate']:.1%}.",
        "",
        "| model | phase-average Brier | phase-average log loss |",
        "|---|---:|---:|",
        f"| supervised HAR-state logistic | {avg['logistic_brier']:.6f} | {avg['logistic_logloss']:.6f} |",
        f"| two-state Gaussian HMM | {avg['hmm_brier']:.6f} | {avg['hmm_logloss']:.6f} |",
        "",
        f"- HMM top probability quintile event rate: {metrics['top_quintile_event_rate']:.1%}",
        f"- HMM bottom probability quintile event rate: {metrics['bottom_quintile_event_rate']:.1%}",
        "",
    ]
    for name, value in metrics["checks"].items():
        lines.append(f"- {name.replace('_', ' ')}: **{value}**")
    lines.extend([
        "",
        f"Frozen verdict: **{'PASS' if metrics['pass'] else 'FAIL'}**.",
        "",
        "A failure means that discrete latent states do not add predictive value "
        "over a directly supervised nonlinear target using the same RV history. "
        "A pass still needs future or cross-asset confirmation.",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",), nargs="?", default="run")
    parser.parse_args(argv)
    run_study()


if __name__ == "__main__":
    main()
