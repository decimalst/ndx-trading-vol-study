"""History-enabled tail ranking, TiRex probing, and Eidos-style noise tools.

The frozen contract lives in ``representation_study.yaml``.  Result-producing
commands deliberately refuse to run until the hash-locked history artifact
exists.  The module keeps the scoring primitives dependency-light and exposes
the unsupported-but-pinned TiRex hidden-state hook in one auditable place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from scipy.special import expit
from scipy.stats import norm, rankdata

from . import config
from .regime_transition import (
    filter_gaussian_hmm,
    fit_gaussian_hmm,
    future_exceedance_probability,
)


ROOT = config.ROOT
PROTOCOL_PATH = ROOT / "representation_study.yaml"
HISTORY_PANEL = ROOT / "data" / "history_extension" / "qqq_price_only_daily.parquet"
HISTORY_MANIFEST = ROOT / "data" / "history_extension" / "source_manifest.json"
HISTORY_PROTOCOL = ROOT / "history_extension.yaml"
OUTPUT_DIR = ROOT / "data" / "representation_study"
REPORT_DIR = ROOT / "reports" / "representation_study"
TAIL_FORECASTS_PATH = OUTPUT_DIR / "tail_classical_forecasts.parquet"
TAIL_PHASES_PATH = OUTPUT_DIR / "tail_classical_phase_metrics.parquet"
TAIL_JACKKNIFE_PATH = OUTPUT_DIR / "tail_classical_episode_jackknife.parquet"
TAIL_DIFFERENCES_PATH = OUTPUT_DIR / "tail_classical_episode_differences.parquet"
TAIL_METRICS_PATH = OUTPUT_DIR / "tail_classical_metrics.json"


def load_protocol(path: Path = PROTOCOL_PATH) -> dict:
    return yaml.safe_load(path.read_text())


def validate_protocol(protocol: dict) -> None:
    source = protocol["source"]
    target = protocol["tail_target"]
    scores = protocol["ranking_scoreboard"]
    latent = protocol["latent_probe"]
    noise = protocol["noise_robustness"]
    reviewer = protocol.get("reviewer_controls", {}).get("trailing_rv_percentile", {})
    phase_review = protocol.get("reviewer_controls", {}).get("phase_and_episode_reporting", {})
    first = pd.Timestamp(source["first_session"])
    last = pd.Timestamp(target["final_score_date"])
    permitted = pd.Timestamp(source["last_permitted_origin"])
    clean = pd.Timestamp(source["clean_start"])
    if not first < last <= permitted < clean:
        raise ValueError("tail and noise origins must stop before the clean window")
    if not source.get("forbid_clean_origins") or not source.get("require_source_hash"):
        raise ValueError("clean fence and immutable source hash are required")
    if list(source.get("permitted_inputs", ())) != ["log_rv", "rv_total", "ret_cc"]:
        raise ValueError("the extended study permits price-only inputs")
    if int(target["horizon_sessions"]) != 5 or not np.isclose(target["stress_quantile"], .8):
        raise ValueError("the inherited target must remain the five-session 80th-percentile event")
    if not target.get("score_only_calm_origins") or not target.get("require_complete_future"):
        raise ValueError("calm-origin and completed-future target fences are required")
    if scores.get("primary") != "auc":
        raise ValueError("the frozen primary scoreboard is AUC")
    if scores.get("secondary") != "top_decile_lift" or not np.isclose(scores["top_fraction"], .1):
        raise ValueError("the frozen secondary scoreboard is top-decile lift")
    if not scores.get("common_rows_required"):
        raise ValueError("ranking comparisons require common rows")
    if (
        reviewer.get("status") != "post_result_requested_after_headline_seen"
        or reviewer.get("evidence_role") != "diagnostic_only_not_prespecified_confirmation"
        or not np.isclose(float(reviewer.get("ridge", np.nan)), 1e-6)
    ):
        raise ValueError("the post-result RV-percentile control provenance changed")
    if (
        phase_review.get("status") != "post_result_requested_after_headline_seen"
        or phase_review.get("evidence_role")
        != "diagnostic_uncertainty_not_prespecified_confirmation"
        or int(phase_review.get("ranking_phases", 0)) != 5
        or list(phase_review.get("phase_summaries", ())) != ["mean", "min", "max", "spread"]
        or phase_review.get("inference") != "leave_one_transition_episode_out_jackknife"
        or not np.isclose(float(phase_review.get("confidence_level", np.nan)), .95)
    ):
        raise ValueError("the post-result phase/episode reporting provenance changed")
    if latent.get("layer") != "output of stack_out_norm, before output_patch_embedding":
        raise ValueError("the pinned latent layer changed")
    if latent.get("pooling") != "last non-padding target token":
        raise ValueError("the pinned latent pooling changed")
    if int(latent["expected_dimension"]) != 512:
        raise ValueError("the pinned latent dimension changed")
    if not latent.get("frozen_backbone"):
        raise ValueError("the TiRex backbone must remain frozen")
    if not latent.get("no_layer_selection") or not latent.get("no_pooling_selection"):
        raise ValueError("latent layer and pooling selection are forbidden")
    if not latent.get("forbid_pca"):
        raise ValueError("PCA is forbidden because variance is not label relevance")
    ladder = latent.get("probe_ladder", {})
    if list(ladder.get("sparse", {}).get("k", ())) != [1, 5, 10]:
        raise ValueError("the sparse probe ladder must report k=1, 5, and 10")
    control = latent.get("control_task", {})
    if int(control.get("draws", 0)) != 10 or len(control.get("seeds", ())) != 10:
        raise ValueError("the matched control task requires ten frozen draws")
    if not latent.get("uncertainty", {}).get("cluster_not_rows"):
        raise ValueError("probe uncertainty must cluster on transition episodes")
    if latent.get("test_time_augmentation") or latent.get("differencing"):
        raise ValueError("the frozen probe uses raw single-pass hidden states")
    if int(noise["seed"]) != 42:
        raise ValueError("the Eidos-derived primary corruption seed is 42")
    expected_origin_seed = "first 64 bits of SHA-256('42|YYYY-MM-DD'), interpreted unsigned big-endian"
    if noise.get("origin_seed") != expected_origin_seed:
        raise ValueError("the order-invariant per-origin noise seed rule changed")
    if list(map(float, noise["gaussian"]["intensities"])) != [0, .2, .4, .6, .8]:
        raise ValueError("Gaussian intensity grid changed")
    if list(map(float, noise["impulse"]["probabilities"])) != [0, .05, .1, .15, .2]:
        raise ValueError("impulse probability grid changed")
    if not np.isclose(noise["impulse"]["magnitude_local_std"], 8):
        raise ValueError("impulse magnitude must remain eight local standard deviations")
    if not noise.get("paired_corruption_required") or not noise.get("all_intensities_reported"):
        raise ValueError("noise paths must use paired corruptions and report every intensity")
    if list(noise.get("models", ())) != [
        "chronos_2_univariate",
        "tirex_2_univariate",
        "har_univariate_expanding_clean_fit",
    ]:
        raise ValueError("the frozen noise-model ladder changed")
    preprocessing = str(noise.get("preprocessing_policy", ""))
    if "native preprocessing" not in preprocessing or "HAR does not renormalize" not in preprocessing:
        raise ValueError("the frozen noise preprocessing disclosure changed")
    inference = noise.get("comparison_inference", {})
    if (
        int(inference.get("block_sessions", 0)) != 22
        or int(inference.get("block_sampled_origins", 0)) != 2
        or int(inference.get("bootstrap_draws", 0)) != 5000
        or int(inference.get("seed", 0)) != 420042
    ):
        raise ValueError("the paired noise bootstrap contract changed")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_locked_history(protocol: dict) -> pd.DataFrame:
    if not HISTORY_PANEL.exists() or not HISTORY_MANIFEST.exists() or not HISTORY_PROTOCOL.exists():
        raise FileNotFoundError("build the frozen history extension before running this study")
    manifest = json.loads(HISTORY_MANIFEST.read_text())
    history_protocol = yaml.safe_load(HISTORY_PROTOCOL.read_text())
    expected = (
        manifest.get("output", {}).get("sha256")
        or manifest.get("panel_sha256")
        or manifest.get("artifacts", {}).get(HISTORY_PANEL.name, {}).get("sha256")
    )
    if not expected:
        raise ValueError("history manifest does not record the panel SHA-256")
    observed = _sha256(HISTORY_PANEL)
    frozen_output = history_protocol.get("output", {}).get("expected_sha256")
    protocol_hash = manifest.get("protocol", {}).get("sha256")
    if observed != expected or observed != frozen_output:
        raise ValueError("history panel hash does not match its frozen protocol and manifest")
    if protocol_hash != _sha256(HISTORY_PROTOCOL):
        raise ValueError("history protocol hash does not match its build manifest")
    frame = pd.read_parquet(HISTORY_PANEL).sort_index()
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    required = set(protocol["source"]["permitted_inputs"])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"history panel missing price-only fields: {sorted(missing)}")
    if frame.index.min() != pd.Timestamp(protocol["source"]["first_session"]):
        raise ValueError("history panel begins on an unexpected date")
    return frame.loc[:, list(protocol["source"]["permitted_inputs"])]


def build_history_features(y: pd.Series) -> pd.DataFrame:
    values = y.astype(float).sort_index()
    return pd.DataFrame({
        "log_rv_d": values,
        "log_rv_w": values.rolling(5).mean(),
        "log_rv_m": values.rolling(22).mean(),
    }, index=values.index)


def prior_reference_percentile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Empirical CDF using only the supplied fold-training reference sample."""
    sample = np.asarray(reference, dtype=float)
    sample = np.sort(sample[np.isfinite(sample)])
    query = np.asarray(values, dtype=float)
    if len(sample) == 0:
        raise ValueError("percentile control needs a non-empty training reference")
    return np.searchsorted(sample, query, side="right") / len(sample)


def completed_training_origins(index: pd.DatetimeIndex, cutoff: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    ordered = pd.DatetimeIndex(index).sort_values()
    cutoff_position = int(ordered.searchsorted(pd.Timestamp(cutoff), side="right") - 1)
    final = cutoff_position - int(horizon)
    return ordered[: final + 1] if final >= 0 else pd.DatetimeIndex([])


def build_fold_targets(
    y: pd.Series,
    *,
    cutoff: pd.Timestamp,
    origins: pd.DatetimeIndex,
    horizon: int,
    quantile: float,
) -> pd.DataFrame:
    series = y.astype(float).sort_index()
    training = series.loc[:pd.Timestamp(cutoff)].dropna()
    if training.empty:
        raise ValueError("tail fold has no threshold history")
    threshold = float(training.quantile(float(quantile)))
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
            "trigger_date": future.index[np.flatnonzero((future > threshold).to_numpy())[0]]
            if bool((future > threshold).any()) else pd.NaT,
        })
    return pd.DataFrame(rows).set_index("origin") if rows else pd.DataFrame()


def roc_auc(event: np.ndarray | pd.Series, score: np.ndarray | pd.Series) -> float:
    y = np.asarray(event, dtype=int)
    s = np.asarray(score, dtype=float)
    valid = np.isfinite(s) & np.isin(y, [0, 1])
    y, s = y[valid], s[valid]
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(s, method="average")
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def top_decile_lift(
    event: np.ndarray | pd.Series,
    score: np.ndarray | pd.Series,
    fraction: float = .1,
) -> float:
    y = np.asarray(event, dtype=float)
    s = np.asarray(score, dtype=float)
    valid = np.isfinite(y) & np.isfinite(s)
    y, s = y[valid], s[valid]
    if len(y) == 0 or y.mean() <= 0:
        return float("nan")
    count = max(1, int(math.ceil(len(y) * float(fraction))))
    # mergesort is stable; chronological order in the input resolves score ties.
    chosen = np.argsort(-s, kind="mergesort")[:count]
    return float(y[chosen].mean() / y.mean())


def evaluate_ranking_phases(
    frame: pd.DataFrame,
    models: Iterable[str],
    *,
    horizon: int,
    top_fraction: float,
) -> pd.DataFrame:
    rows = []
    for phase in range(int(horizon)):
        if "ranking_phase" in frame:
            sample = frame.loc[frame["ranking_phase"] == phase]
        else:
            sample = frame.iloc[phase::int(horizon)]
        for model in models:
            column = f"p_{model}"
            if column not in sample:
                raise ValueError(f"missing common ranking score {column}")
            event_rate = float(sample["event"].mean())
            lift = top_decile_lift(sample["event"], sample[column], top_fraction)
            rows.append({
                "phase": phase,
                "model": model,
                "n": len(sample),
                "event_rate": event_rate,
                "auc": roc_auc(sample["event"], sample[column]),
                "top_decile_lift": lift,
                "top_decile_event_rate": event_rate * lift,
            })
    return pd.DataFrame(rows)


def summarize_phase_dispersion(phases: pd.DataFrame) -> dict:
    metrics = ["auc", "top_decile_lift", "top_decile_event_rate"]
    result: dict[str, dict] = {}
    for model, sample in phases.groupby("model", sort=True):
        result[str(model)] = {}
        for metric in metrics:
            values = sample[metric].to_numpy(dtype=float)
            low = float(np.nanmin(values))
            high = float(np.nanmax(values))
            result[str(model)][metric] = {
                "mean": float(np.nanmean(values)),
                "min": low,
                "max": high,
                "spread": high - low,
            }
    return result


def episode_jackknife_ranking(
    frame: pd.DataFrame,
    models: Iterable[str],
    *,
    horizon: int,
    top_fraction: float,
    confidence: float,
) -> pd.DataFrame:
    """Delete one registered transition episode, retaining fixed phase IDs."""
    if "ranking_phase" not in frame or "event_cluster" not in frame:
        raise ValueError("episode jackknife requires fixed phases and event clusters")
    model_names = list(models)
    clusters = sorted(int(value) for value in frame["event_cluster"].dropna().unique())
    if len(clusters) < 2:
        raise ValueError("episode jackknife needs at least two transition episodes")
    metrics = ["auc", "top_decile_lift", "top_decile_event_rate"]
    full = evaluate_ranking_phases(
        frame, model_names, horizon=horizon, top_fraction=top_fraction
    ).groupby("model")[metrics].mean()
    deleted: dict[tuple[str, str], list[float]] = {
        (model, metric): [] for model in model_names for metric in metrics
    }
    for cluster in clusters:
        keep = frame["event_cluster"].isna() | frame["event_cluster"].ne(cluster).fillna(True)
        phases = evaluate_ranking_phases(
            frame.loc[keep], model_names, horizon=horizon, top_fraction=top_fraction
        ).groupby("model")[metrics].mean()
        for model in model_names:
            for metric in metrics:
                deleted[(model, metric)].append(float(phases.loc[model, metric]))
    z = float(norm.ppf(.5 + float(confidence) / 2))
    rows = []
    count = len(clusters)
    for model in model_names:
        for metric in metrics:
            values = np.asarray(deleted[(model, metric)], dtype=float)
            center = float(np.nanmean(values))
            se = float(np.sqrt((count - 1) / count * np.nansum((values - center) ** 2)))
            estimate = float(full.loc[model, metric])
            rows.append({
                "model": model,
                "metric": metric,
                "estimate": estimate,
                "episodes": count,
                "se": se,
                "ci_low": estimate - z * se,
                "ci_high": estimate + z * se,
                "min_leave_one_out": float(np.nanmin(values)),
                "max_leave_one_out": float(np.nanmax(values)),
            })
    return pd.DataFrame(rows)


def episode_jackknife_differences(
    frame: pd.DataFrame,
    comparisons: dict[str, tuple[str, str]],
    *,
    horizon: int,
    top_fraction: float,
    confidence: float,
) -> pd.DataFrame:
    """Paired delete-one-episode intervals for candidate-minus-baseline deltas."""
    if "ranking_phase" not in frame or "event_cluster" not in frame:
        raise ValueError("episode differences require fixed phases and event clusters")
    models = list(dict.fromkeys(
        model for pair in comparisons.values() for model in pair
    ))
    metrics = ["auc", "top_decile_lift", "top_decile_event_rate"]
    clusters = sorted(int(value) for value in frame["event_cluster"].dropna().unique())
    if len(clusters) < 2:
        raise ValueError("episode differences need at least two transition episodes")
    full = evaluate_ranking_phases(
        frame, models, horizon=horizon, top_fraction=top_fraction
    ).groupby("model")[metrics].mean()
    deleted: dict[tuple[str, str], list[float]] = {
        (name, metric): [] for name in comparisons for metric in metrics
    }
    for cluster in clusters:
        keep = frame["event_cluster"].isna() | frame["event_cluster"].ne(cluster).fillna(True)
        phases = evaluate_ranking_phases(
            frame.loc[keep], models, horizon=horizon, top_fraction=top_fraction
        ).groupby("model")[metrics].mean()
        for name, (candidate, baseline) in comparisons.items():
            for metric in metrics:
                deleted[(name, metric)].append(
                    float(phases.loc[candidate, metric] - phases.loc[baseline, metric])
                )
    count = len(clusters)
    z = float(norm.ppf(.5 + float(confidence) / 2))
    rows = []
    for name, (candidate, baseline) in comparisons.items():
        for metric in metrics:
            values = np.asarray(deleted[(name, metric)], dtype=float)
            center = float(np.nanmean(values))
            se = float(np.sqrt((count - 1) / count * np.nansum((values - center) ** 2)))
            estimate = float(full.loc[candidate, metric] - full.loc[baseline, metric])
            rows.append({
                "comparison": name,
                "candidate": candidate,
                "baseline": baseline,
                "metric": metric,
                "estimate": estimate,
                "episodes": count,
                "se": se,
                "ci_low": estimate - z * se,
                "ci_high": estimate + z * se,
                "min_leave_one_out": float(np.nanmin(values)),
                "max_leave_one_out": float(np.nanmax(values)),
            })
    return pd.DataFrame(rows)


def _fit_logistic(X: pd.DataFrame, y: pd.Series, ridge: float) -> dict:
    values = X.to_numpy(dtype=float)
    target = y.to_numpy(dtype=float)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-12] = 1.0
    design = np.column_stack([np.ones(len(values)), (values - mean) / scale])
    beta = np.zeros(design.shape[1])
    penalty = np.eye(design.shape[1]) * float(ridge)
    penalty[0, 0] = 0
    for _ in range(200):
        probability = expit(design @ beta)
        weight = np.clip(probability * (1 - probability), 1e-7, None)
        hessian = design.T @ (design * weight[:, None]) + penalty
        gradient = design.T @ (target - probability) - penalty @ beta
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return {"beta": beta, "mean": mean, "scale": scale, "columns": list(X.columns)}


def _predict_logistic(model: dict, X: pd.DataFrame) -> np.ndarray:
    values = X.loc[:, model["columns"]].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(values)), (values - model["mean"]) / model["scale"]])
    return np.clip(expit(design @ model["beta"]), 1e-6, 1 - 1e-6)


def gaussian_noise(values: np.ndarray, intensity: float, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if float(intensity) == 0:
        return x.copy()
    sigma = float(np.nanstd(x, ddof=0))
    return x + rng.normal(0, float(intensity) * sigma, size=x.shape)


def origin_noise_seed(base_seed: int, origin: pd.Timestamp | str) -> int:
    """Derive an order-invariant per-origin seed from the frozen base seed."""
    date = pd.Timestamp(origin).normalize().date().isoformat()
    payload = f"{int(base_seed)}|{date}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def impulse_noise(
    values: np.ndarray,
    probability: float,
    magnitude_local_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if float(probability) == 0:
        return x.copy()
    sigma = float(np.nanstd(x, ddof=0))
    mask = rng.random(x.shape) < float(probability)
    signs = rng.choice(np.array([-1.0, 1.0]), size=x.shape)
    return x + mask * signs * float(magnitude_local_std) * sigma


def paired_corruptions(values: np.ndarray, seed: int) -> dict[str, dict[float, np.ndarray]]:
    """Nested Eidos-style corruptions shared across intensity curves.

    One Gaussian z-vector and one impulse uniform/sign vector are drawn per
    origin. This makes all levels and models paired, and makes masks nested as
    the impulse probability grows.
    """
    x = np.asarray(values, dtype=float)
    sigma = float(np.nanstd(x, ddof=0))
    rng = np.random.default_rng(int(seed))
    z = rng.normal(size=x.shape)
    uniform = rng.random(x.shape)
    signs = rng.choice(np.array([-1.0, 1.0]), size=x.shape)
    gaussian = {level: x + float(level) * sigma * z for level in (0, .2, .4, .6, .8)}
    impulse = {
        level: x + (uniform < float(level)) * signs * 8.0 * sigma
        for level in (0, .05, .1, .15, .2)
    }
    return {"gaussian": gaussian, "impulse": impulse}


def select_sparse_dimensions(X: np.ndarray, event: np.ndarray, *, k: int) -> np.ndarray:
    values = np.asarray(X, dtype=float)
    y = np.asarray(event, dtype=int)
    if values.ndim != 2 or len(values) != len(y):
        raise ValueError("sparse selection needs aligned training rows")
    if not (0 < int(k) <= values.shape[1]) or not ({0, 1} <= set(np.unique(y))):
        raise ValueError("sparse selection needs both classes and a valid k")
    scale = values.std(axis=0)
    scale[scale < 1e-12] = 1.0
    effect = np.abs((values[y == 1].mean(axis=0) - values[y == 0].mean(axis=0)) / scale)
    # Stable coordinate order resolves exact ties and makes results reproducible.
    return np.argsort(-effect, kind="mergesort")[: int(k)]


def markov_control_labels(training_event: np.ndarray, *, total_length: int, seed: int) -> np.ndarray:
    """First-order surrogate preserving fitted event persistence, not feature links."""
    y = np.asarray(training_event, dtype=int)
    if len(y) < 2 or set(np.unique(y)) != {0, 1}:
        raise ValueError("Markov control needs both event classes")
    counts = np.ones((2, 2), dtype=float)  # Laplace floor prevents absorbing controls.
    for left, right in zip(y[:-1], y[1:]):
        counts[left, right] += 1
    transition = counts / counts.sum(axis=1, keepdims=True)
    rng = np.random.default_rng(int(seed))
    out = np.empty(int(total_length), dtype=int)
    out[0] = int(rng.random() < y.mean())
    for position in range(1, len(out)):
        out[position] = int(rng.random() < transition[out[position - 1], 1])
    # A degenerate finite draw makes AUC undefined. Advance the same RNG until
    # both classes are represented; the rule is frozen and label-independent.
    attempts = 0
    while len(np.unique(out)) < 2 and attempts < 100:
        out[0] = int(rng.random() < y.mean())
        for position in range(1, len(out)):
            out[position] = int(rng.random() < transition[out[position - 1], 1])
        attempts += 1
    if len(np.unique(out)) < 2:
        raise RuntimeError("could not generate a non-degenerate Markov control")
    return out


def assign_event_clusters(
    trigger_date: pd.Series,
    sessions: pd.DatetimeIndex,
    *,
    max_gap_sessions: int,
) -> pd.Series:
    """Map positive origins to stress episodes using trigger-session distance."""
    positions = pd.Series(np.arange(len(sessions)), index=pd.DatetimeIndex(sessions))
    result = pd.Series(pd.NA, index=trigger_date.index, dtype="Int64")
    valid = trigger_date.dropna().sort_values(kind="mergesort")
    cluster = -1
    previous_position = None
    trigger_to_cluster: dict[pd.Timestamp, int] = {}
    for value in pd.DatetimeIndex(valid.unique()).sort_values():
        if value not in positions:
            raise ValueError("event trigger is absent from the source session index")
        position = int(positions.loc[value])
        if previous_position is None or position - previous_position > int(max_gap_sessions):
            cluster += 1
        trigger_to_cluster[pd.Timestamp(value)] = cluster
        previous_position = position
    for origin, value in trigger_date.items():
        if pd.notna(value):
            result.loc[origin] = trigger_to_cluster[pd.Timestamp(value)]
    return result


def pool_tirex_context_token(
    hidden: np.ndarray,
    *,
    context_tokens: int,
    expected_dim: int,
) -> np.ndarray:
    values = np.asarray(hidden)
    if values.ndim != 3:
        raise ValueError("TiRex hidden state must have batch, token, dimension axes")
    if values.shape[-1] != int(expected_dim):
        raise ValueError(f"TiRex hidden dimension {values.shape[-1]} != {expected_dim}")
    if values.shape[1] < int(context_tokens):
        raise ValueError("TiRex hidden state has fewer tokens than the causal context")
    return values[:, int(context_tokens) - 1, :]


def extract_tirex_hidden(contexts: list[np.ndarray], protocol: dict) -> np.ndarray:
    """Extract pinned final pre-head states using the 0.2.1 implementation hook."""
    import torch
    from tirex2 import TimeseriesType, load_model

    latent = protocol["latent_probe"]
    model = load_model(
        latent["checkpoint"],
        device="cpu",
        hf_kwargs={"revision": latent["checkpoint_revision"], "local_files_only": True},
    )
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        captured.append(output.detach().cpu())

    handle = model.model.stack_out_norm.register_forward_hook(hook)
    try:
        items = [
            TimeseriesType(
                target=torch.as_tensor(np.asarray(values), dtype=torch.float32).unsqueeze(0),
                past_covariates=None,
                future_covariates=None,
            )
            for values in contexts
        ]
        model.forecast(
            items,
            prediction_length=1,
            output_type="torch",
            batch_size=len(items),
            tta_sign_flip=bool(latent["test_time_augmentation"]),
            tta_diff=bool(latent["differencing"]),
        )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one TiRex hook call, got {len(captured)}")
    hidden = captured[0].numpy()
    context_tokens = int(model.model.context_len // model.model.input_patch_size)
    return pool_tirex_context_token(
        hidden,
        context_tokens=context_tokens,
        expected_dim=int(latent["expected_dimension"]),
    )


def run_tail_classical(protocol: dict | None = None) -> dict:
    """Run the extended price-only benchmark and calibrated-HMM ranking study."""
    protocol = protocol or load_protocol()
    validate_protocol(protocol)
    frame = load_locked_history(protocol)
    y = frame["log_rv"].dropna().sort_index()
    features = build_history_features(y)
    target_cfg = protocol["tail_target"]
    models_cfg = protocol["classical_models"]
    start_year = int(target_cfg["first_score_year"])
    final = pd.Timestamp(target_cfg["final_score_date"])
    horizon = int(target_cfg["horizon_sessions"])
    quantile = float(target_cfg["stress_quantile"])
    min_train = int(target_cfg["minimum_training_observations"])
    columns = list(models_cfg["benchmark"]["features"])
    reviewer_cfg = protocol["reviewer_controls"]["trailing_rv_percentile"]
    rows = []
    fold_meta = []
    for year in range(start_year, final.year + 1):
        year_start = pd.Timestamp(f"{year}-01-01")
        year_end = min(pd.Timestamp(f"{year}-12-31"), final)
        candidates = y.index[y.index < year_start]
        if not len(candidates):
            continue
        cutoff = candidates[-1]
        train_y = y.loc[:cutoff]
        if len(train_y) < min_train:
            continue
        threshold = float(train_y.quantile(quantile))
        train_origins = completed_training_origins(y.index, cutoff, horizon)
        train_targets = build_fold_targets(
            y, cutoff=cutoff, origins=train_origins, horizon=horizon, quantile=quantile
        )
        train_targets = train_targets.loc[train_targets["calm"]]
        train = features.loc[train_targets.index, columns].join(train_targets["event"]).dropna()
        if len(train) < min_train:
            continue
        benchmark = _fit_logistic(train[columns], train["event"], models_cfg["benchmark"]["ridge"])
        train["rv_percentile_prior"] = prior_reference_percentile(
            train["log_rv_d"].to_numpy(), train_y.to_numpy()
        )
        rv_percentile = _fit_logistic(
            train[["rv_percentile_prior"]], train["event"], reviewer_cfg["ridge"]
        )
        hmm_cfg = models_cfg["hmm"]
        hmm = fit_gaussian_hmm(
            train_y,
            max_iter=int(hmm_cfg["max_iter"]),
            tolerance=float(hmm_cfg["tolerance"]),
        )
        filtered_train = pd.DataFrame(
            filter_gaussian_hmm(train_y, hmm), index=train_y.index, columns=["low", "high"]
        )
        raw_train = pd.Series([
            future_exceedance_probability(
                filtered_train.loc[origin].to_numpy(), hmm, threshold=threshold, horizon=horizon
            )
            for origin in train.index
        ], index=train.index)
        clip = float(hmm_cfg["calibration_clip"])
        clipped = raw_train.clip(clip, 1 - clip)
        calibration_X = pd.DataFrame({"hmm_logit": np.log(clipped / (1 - clipped))}, index=train.index)
        platt = _fit_logistic(calibration_X, train["event"], hmm_cfg["calibration_ridge"])
        train["p_hmm_platt"] = _predict_logistic(platt, calibration_X)
        augmented = _fit_logistic(
            train[columns + ["p_hmm_platt"]], train["event"], models_cfg["benchmark"]["ridge"]
        )

        origins = y.index[(y.index >= year_start) & (y.index <= year_end)]
        targets = build_fold_targets(y, cutoff=cutoff, origins=origins, horizon=horizon, quantile=quantile)
        if targets.empty:
            continue
        targets = targets.loc[targets["calm"] & (targets["target_end"] < pd.Timestamp(protocol["source"]["clean_start"]))]
        common = targets.index.intersection(features[columns].dropna().index)
        if not len(common):
            continue
        fold = targets.loc[common].copy()
        for column in columns:
            fold[column] = features.loc[common, column]
        fold["rv_percentile_prior"] = prior_reference_percentile(
            fold["log_rv_d"].to_numpy(), train_y.to_numpy()
        )
        extended = y.loc[:year_end]
        filtered = pd.DataFrame(
            filter_gaussian_hmm(extended, hmm), index=extended.index, columns=["low", "high"]
        )
        raw = pd.Series([
            future_exceedance_probability(
                filtered.loc[origin].to_numpy(), hmm, threshold=threshold, horizon=horizon
            )
            for origin in common
        ], index=common)
        raw_clip = raw.clip(clip, 1 - clip)
        holdout_cal_X = pd.DataFrame({"hmm_logit": np.log(raw_clip / (1 - raw_clip))}, index=common)
        fold["p_hmm_raw"] = raw
        fold["p_hmm_platt"] = _predict_logistic(platt, holdout_cal_X)
        fold["p_benchmark"] = _predict_logistic(benchmark, fold[columns])
        fold["p_rv_percentile"] = _predict_logistic(
            rv_percentile, fold[["rv_percentile_prior"]]
        )
        fold["p_hmm_augmented"] = _predict_logistic(augmented, fold[columns + ["p_hmm_platt"]])
        fold["fold_year"] = year
        fold["cutoff"] = cutoff
        rows.append(fold)
        fold_meta.append({"year": year, "cutoff": str(cutoff.date()), "train_n": len(train), "threshold": threshold})
    if not rows:
        raise RuntimeError("extended tail study produced no forecasts")
    scored = pd.concat(rows).sort_index()
    if (scored.index >= pd.Timestamp(protocol["source"]["clean_start"])).any():
        raise RuntimeError("extended tail origins crossed the clean fence")
    score_cfg = protocol["ranking_scoreboard"]
    model_names = ["benchmark", "rv_percentile", "hmm_platt", "hmm_augmented"]
    scored["ranking_phase"] = np.arange(len(scored)) % horizon
    scored["event_cluster"] = assign_event_clusters(
        scored["trigger_date"], frame.index, max_gap_sessions=horizon
    )
    phases = evaluate_ranking_phases(
        scored,
        model_names,
        horizon=horizon,
        top_fraction=float(score_cfg["top_fraction"]),
    )
    summary = phases.groupby("model")[[
        "auc", "top_decile_lift", "top_decile_event_rate"
    ]].mean().to_dict("index")
    dispersion = summarize_phase_dispersion(phases)
    episode_jackknife = episode_jackknife_ranking(
        scored,
        model_names,
        horizon=horizon,
        top_fraction=float(score_cfg["top_fraction"]),
        confidence=float(
            protocol["reviewer_controls"]["phase_and_episode_reporting"]["confidence_level"]
        ),
    )
    episode_differences = episode_jackknife_differences(
        scored,
        {
            "benchmark_minus_rv_percentile": ("benchmark", "rv_percentile"),
            "hmm_augmented_minus_benchmark": ("hmm_augmented", "benchmark"),
            "hmm_platt_minus_benchmark": ("hmm_platt", "benchmark"),
        },
        horizon=horizon,
        top_fraction=float(score_cfg["top_fraction"]),
        confidence=float(
            protocol["reviewer_controls"]["phase_and_episode_reporting"]["confidence_level"]
        ),
    )
    metrics = {
        "origins": len(scored),
        "first_origin": str(scored.index.min().date()),
        "last_origin": str(scored.index.max().date()),
        "event_rate": float(scored["event"].mean()),
        "positive_origins": int(scored["event"].sum()),
        "unique_trigger_sessions": int(scored["trigger_date"].nunique()),
        "transition_episodes": int(scored["event_cluster"].nunique()),
        "requested_first_score_year": start_year,
        "actual_first_score_year": int(scored.index.min().year),
        "ranking_phases": horizon,
        "reviewer_control_status": reviewer_cfg["status"],
        "phase_mean": summary,
        "phase_dispersion": dispersion,
        "episode_jackknife": episode_jackknife.to_dict("records"),
        "episode_differences": episode_differences.to_dict("records"),
        "folds": fold_meta,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(TAIL_FORECASTS_PATH)
    phases.to_parquet(TAIL_PHASES_PATH, index=False)
    episode_jackknife.to_parquet(TAIL_JACKKNIFE_PATH, index=False)
    episode_differences.to_parquet(TAIL_DIFFERENCES_PATH, index=False)
    TAIL_METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
    _write_tail_report(metrics)
    return metrics


def _render_tail_report(metrics: dict) -> str:
    rows = []
    for model, values in metrics["phase_mean"].items():
        spread = metrics["phase_dispersion"][model]
        rows.append(
            f"| {model} | {values['auc']:.4f} "
            f"[{spread['auc']['min']:.4f}, {spread['auc']['max']:.4f}] | "
            f"{values['top_decile_lift']:.3f}x "
            f"[{spread['top_decile_lift']['min']:.3f}, "
            f"{spread['top_decile_lift']['max']:.3f}] | "
            f"{values['top_decile_event_rate']:.1%} |"
        )
    jackknife = {
        (row["model"], row["metric"]): row for row in metrics["episode_jackknife"]
    }
    interval_rows = []
    for model in metrics["phase_mean"]:
        auc = jackknife[(model, "auc")]
        lift = jackknife[(model, "top_decile_lift")]
        interval_rows.append(
            f"| {model} | [{auc['ci_low']:.4f}, {auc['ci_high']:.4f}] | "
            f"[{lift['ci_low']:.3f}, {lift['ci_high']:.3f}]x |"
        )
    differences = {
        (row["comparison"], row["metric"]): row for row in metrics["episode_differences"]
    }
    difference_rows = []
    for comparison in [
        "benchmark_minus_rv_percentile",
        "hmm_augmented_minus_benchmark",
        "hmm_platt_minus_benchmark",
    ]:
        auc = differences[(comparison, "auc")]
        lift = differences[(comparison, "top_decile_lift")]
        top_rate = differences[(comparison, "top_decile_event_rate")]
        difference_rows.append(
            f"| {comparison} | {auc['estimate']:+.4f} "
            f"[{auc['ci_low']:+.4f}, {auc['ci_high']:+.4f}] | "
            f"{lift['estimate']:+.3f}x [{lift['ci_low']:+.3f}, {lift['ci_high']:+.3f}] | "
            f"{top_rate['estimate']:+.1%} "
            f"[{top_rate['ci_low']:+.1%}, {top_rate['ci_high']:+.1%}] |"
        )
    report = "\n".join([
        "# Extended-history tail ranking: classical models",
        "",
        "The supervised benchmark uses current log RV and trailing 5-/22-session means",
        "of log RV, matching the earlier transition study. It is not the standard HAR",
        "`log(mean variance)` transform.",
        "",
        f"Origins: {metrics['origins']:,} ({metrics['first_origin']} through {metrics['last_origin']}); "
        f"event rate {metrics['event_rate']:.1%}.",
        f"The {metrics['positive_origins']:,} positive origins map to "
        f"{metrics['unique_trigger_sessions']:,} trigger sessions and "
        f"{metrics['transition_episodes']:,} registered transition episodes. The positive-origin "
        "count is not treated as the independent-event denominator.",
        "At a 13.2% five-session crossing rate this target measures recurrent movement out of a "
        "calm band—threshold proximity—not rare-crisis anticipation.",
        "",
        "The phase mean averages the five fixed offsets of this five-session target—not the "
        "21 offsets used by earlier 21-session designs.",
        "",
        "| model | AUC mean [min, max] | lift mean [min, max] | top-decile event rate |",
        "|---|---:|---:|---:|",
        *rows,
        "",
        f"Post-result leave-one-episode-out jackknife intervals ({metrics['transition_episodes']} "
        "transition episodes):",
        "",
        "These are positive-episode influence intervals: each replicate removes one",
        "threshold-trigger episode while all negative origins remain fixed. They do not",
        "capture negative-origin or residual serial-score sampling variability and are",
        "not full cluster-robust standard errors.",
        "",
        "| model | AUC 95% interval | lift 95% interval |",
        "|---|---:|---:|",
        *interval_rows,
        "",
        "Paired episode-clustered differences (candidate minus baseline):",
        "",
        "| comparison | AUC delta [95% interval] | lift delta [95% interval] | top-decile rate delta [95% interval] |",
        "|---|---:|---:|---:|",
        *difference_rows,
        "",
        f"The protocol attempted scoring in {metrics['requested_first_score_year']}, but the frozen "
        f"400 completed calm-label minimum delayed the first eligible fold to "
        f"{metrics['actual_first_score_year']}. Thus 1999-2001 is training-only; the forward "
        "scoreboard captures 2002 and 2008, not the onset of the 2000 transition.",
        "",
        "Adding the calibrated HMM state changed AUC by "
        f"{metrics['phase_mean']['hmm_augmented']['auc'] - metrics['phase_mean']['benchmark']['auc']:+.4f} "
        "and top-decile lift by "
        f"{metrics['phase_mean']['hmm_augmented']['top_decile_lift'] - metrics['phase_mean']['benchmark']['top_decile_lift']:+.3f}x. "
        "That is negligible ranking gain, not evidence that the HMM beats the supervised benchmark.",
        "",
        "The one-variable `rv_percentile` row was requested only after the headline was seen. "
        "It is a reviewer diagnostic for volatility-level persistence, not pre-specified confirmation.",
        "",
        "The event and scoreboard were frozen before this run. Brier/log loss are not verdict metrics here.",
    ])
    return report + "\n"


def _write_tail_report(metrics: dict) -> None:
    (REPORT_DIR / "tail_classical.md").write_text(_render_tail_report(metrics))


def verify_tail_classical(protocol: dict | None = None) -> dict:
    """Recompute every persisted tail scoreboard from the saved forecasts."""
    protocol = protocol or load_protocol()
    validate_protocol(protocol)
    load_locked_history(protocol)
    required = [
        TAIL_FORECASTS_PATH,
        TAIL_PHASES_PATH,
        TAIL_JACKKNIFE_PATH,
        TAIL_DIFFERENCES_PATH,
        TAIL_METRICS_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"tail artifacts missing: {missing}")
    scored = pd.read_parquet(TAIL_FORECASTS_PATH).sort_index()
    clean = pd.Timestamp(protocol["source"]["clean_start"])
    if (
        scored.empty
        or scored.index.has_duplicates
        or not scored.index.is_monotonic_increasing
        or (scored.index >= clean).any()
        or (pd.to_datetime(scored["target_end"]) >= clean).any()
    ):
        raise ValueError("tail artifact violates chronology or the sealed clean fence")
    horizon = int(protocol["tail_target"]["horizon_sessions"])
    top_fraction = float(protocol["ranking_scoreboard"]["top_fraction"])
    confidence = float(
        protocol["reviewer_controls"]["phase_and_episode_reporting"]["confidence_level"]
    )
    names = ["benchmark", "rv_percentile", "hmm_platt", "hmm_augmented"]
    phases = evaluate_ranking_phases(
        scored, names, horizon=horizon, top_fraction=top_fraction
    )
    observed_phases = pd.read_parquet(TAIL_PHASES_PATH)
    pd.testing.assert_frame_equal(
        phases.reset_index(drop=True), observed_phases.reset_index(drop=True)
    )
    jackknife = episode_jackknife_ranking(
        scored,
        names,
        horizon=horizon,
        top_fraction=top_fraction,
        confidence=confidence,
    )
    pd.testing.assert_frame_equal(
        jackknife.reset_index(drop=True),
        pd.read_parquet(TAIL_JACKKNIFE_PATH).reset_index(drop=True),
    )
    comparisons = {
        "benchmark_minus_rv_percentile": ("benchmark", "rv_percentile"),
        "hmm_augmented_minus_benchmark": ("hmm_augmented", "benchmark"),
        "hmm_platt_minus_benchmark": ("hmm_platt", "benchmark"),
    }
    differences = episode_jackknife_differences(
        scored,
        comparisons,
        horizon=horizon,
        top_fraction=top_fraction,
        confidence=confidence,
    )
    pd.testing.assert_frame_equal(
        differences.reset_index(drop=True),
        pd.read_parquet(TAIL_DIFFERENCES_PATH).reset_index(drop=True),
    )
    metrics = json.loads(TAIL_METRICS_PATH.read_text())
    summaries = phases.groupby("model")[[
        "auc", "top_decile_lift", "top_decile_event_rate"
    ]].mean().to_dict("index")
    if metrics["phase_mean"] != summaries:
        raise AssertionError("tail phase means no longer reproduce the metrics JSON")
    if metrics["phase_dispersion"] != summarize_phase_dispersion(phases):
        raise AssertionError("tail phase dispersion no longer reproduces the metrics JSON")
    checks = {
        "origins": len(scored),
        "positive_origins": int(scored["event"].sum()),
        "unique_trigger_sessions": int(scored["trigger_date"].nunique()),
        "transition_episodes": int(scored["event_cluster"].nunique()),
        "ranking_phases": horizon,
    }
    for key, value in checks.items():
        if metrics[key] != value:
            raise AssertionError(f"tail metric {key} no longer reproduces")
    if (REPORT_DIR / "tail_classical.md").read_text() != _render_tail_report(metrics):
        raise AssertionError("tail report no longer reproduces from verified metrics")
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["validate", "tail-classical", "verify-tail"])
    args = parser.parse_args(argv)
    protocol = load_protocol()
    validate_protocol(protocol)
    if args.command == "validate":
        print("representation protocol valid")
    elif args.command == "tail-classical":
        print(json.dumps(run_tail_classical(protocol), indent=2))
    else:
        print(json.dumps(verify_tail_classical(protocol), indent=2))


if __name__ == "__main__":
    main()
