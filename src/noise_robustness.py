"""Frozen Eidos-style context-corruption diagnostic for log realized variance.

The design is registered in ``representation_study.yaml``.  This module keeps
the corruptions paired across model families and scores only the registered
one-session decile forecasts.  It deliberately does not evaluate Eidos itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, metrics
from .representation_study import (
    load_locked_history,
    load_protocol,
    origin_noise_seed,
    paired_corruptions,
    validate_protocol,
)

ROOT = config.ROOT
OUTPUT_DIR = ROOT / "data" / "representation_study"
REPORT_PATH = ROOT / "reports" / "representation_study" / "noise_robustness.md"
FORECAST_PATH = OUTPUT_DIR / "noise_forecasts.parquet"
CURVE_PATH = OUTPUT_DIR / "noise_curves.parquet"
BOOTSTRAP_PATH = OUTPUT_DIR / "noise_bootstrap_intervals.parquet"
MANIFEST_PATH = OUTPUT_DIR / "noise_manifest.json"

CHRONOS_ID = "amazon/chronos-2"
CHRONOS_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
TIREX_ID = "NX-AI/TiRex-2"
TIREX_REVISION = "05e5b26db52bfb256f1ae1bdf785589850482de3"
MODEL_NAMES = (
    "chronos_2_univariate",
    "tirex_2_univariate",
    "har_univariate_expanding_clean_fit",
)


def cached_snapshot_path(repo_id: str, revision: str, required: tuple[str, ...]) -> Path:
    """Resolve a pinned HF snapshot without any network or completeness check.

    TiRex's local cache intentionally contains only the two files its loader
    consumes.  ``snapshot_download(local_files_only=True)`` rejects that useful
    minimal snapshot because optional card/license files are absent, so we
    validate the executable inputs directly instead.
    """
    from huggingface_hub.constants import HF_HUB_CACHE
    folder = "models--" + repo_id.replace("/", "--")
    path = Path(HF_HUB_CACHE) / folder / "snapshots" / revision
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"pinned local snapshot {repo_id}@{revision} lacks {missing}"
        )
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    """Stable hash of the exact float64 context handed to a model adapter."""
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def select_origins(frame: pd.DataFrame, noise: dict) -> pd.DatetimeIndex:
    """Take every registered twentieth session, retaining a complete t+1."""
    start = pd.Timestamp(noise["sample_start"])
    end = pd.Timestamp(noise["sample_end"])
    stride = int(noise["origin_stride_sessions"])
    eligible = frame.loc[start:end].index[::stride]
    positions = pd.Series(np.arange(len(frame)), index=frame.index)
    return pd.DatetimeIndex(
        [t for t in eligible if int(positions.loc[t]) + int(noise["target_horizon"]) < len(frame)]
    )


def build_context_bank(
    frame: pd.DataFrame, origins: pd.DatetimeIndex, noise: dict
) -> tuple[dict[tuple[str, float], dict[pd.Timestamp, np.ndarray]], pd.DataFrame]:
    """Materialize the one paired corruption bank used by every model.

    Gaussian draws and impulse uniforms/signs are generated once per origin by
    :func:`paired_corruptions`.  The zero-intensity contexts are intentionally
    retained under both corruption families so their forecast identity can be
    checked in the persisted artifact.
    """
    y = frame["log_rv"].astype(float)
    n_context = int(noise["context_length"])
    gaussian = tuple(map(float, noise["gaussian"]["intensities"]))
    impulse = tuple(map(float, noise["impulse"]["probabilities"]))
    bank: dict[tuple[str, float], dict[pd.Timestamp, np.ndarray]] = {
        **{("gaussian", level): {} for level in gaussian},
        **{("impulse", level): {} for level in impulse},
    }
    metadata = []
    for origin in pd.DatetimeIndex(origins):
        pos = int(frame.index.get_loc(origin))
        context = y.iloc[max(0, pos + 1 - n_context): pos + 1].to_numpy(dtype=float)
        if len(context) != n_context:
            raise ValueError(f"origin {origin.date()} lacks the frozen {n_context}-session context")
        seed = origin_noise_seed(int(noise["seed"]), origin)
        paths = paired_corruptions(context, seed)
        for level in gaussian:
            bank[("gaussian", level)][origin] = np.asarray(paths["gaussian"][level], dtype=float)
        for level in impulse:
            bank[("impulse", level)][origin] = np.asarray(paths["impulse"][level], dtype=float)
        target_pos = pos + int(noise["target_horizon"])
        metadata.append({
            "origin": origin,
            "target_date": frame.index[target_pos],
            "actual": float(y.iloc[target_pos]),
            "origin_seed": np.uint64(seed),
        })
    return bank, pd.DataFrame(metadata).set_index("origin")


def har_features(context: np.ndarray) -> np.ndarray:
    """Daily/weekly/monthly HAR state, with averages on the variance scale."""
    x = np.asarray(context, dtype=float)
    if len(x) < 22:
        raise ValueError("HAR feature calculation needs 22 observations")
    variance = np.exp(x)
    return np.array([
        x[-1],
        np.log(np.mean(variance[-5:])),
        np.log(np.mean(variance[-22:])),
    ])


def fit_expanding_har(y: pd.Series, origin: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    """Fit through origin t using only targets observed by t.

    A training feature row at s predicts s+1.  The final training feature row
    is therefore the session immediately before t, whose response is y_t.
    """
    # Slice first, so even intermediate arithmetic never touches an unobserved
    # value after the origin.
    values = y.astype(float).sort_index().loc[:pd.Timestamp(origin)]
    rv = np.exp(values)
    design = pd.DataFrame({
        "daily": values,
        "weekly": np.log(rv.rolling(5).mean()),
        "monthly": np.log(rv.rolling(22).mean()),
        "target": values.shift(-1),
    }).dropna()
    if len(design) < 100:
        raise ValueError("expanding HAR needs at least 100 completed training rows")
    X = np.column_stack([np.ones(len(design)), design[["daily", "weekly", "monthly"]].to_numpy()])
    target = design["target"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    residual = target - X @ beta
    return beta, residual


def run_har(
    frame: pd.DataFrame,
    origins: pd.DatetimeIndex,
    bank: dict[tuple[str, float], dict[pd.Timestamp, np.ndarray]],
    metadata: pd.DataFrame,
    taus: np.ndarray,
) -> pd.DataFrame:
    y = frame["log_rv"]
    fits = {origin: fit_expanding_har(y, origin) for origin in origins}
    rows = []
    for (corruption, intensity), contexts in bank.items():
        for origin in origins:
            beta, residual = fits[origin]
            state = har_features(contexts[origin])
            location = float(np.r_[1.0, state] @ beta)
            quantiles = location + np.quantile(residual, taus)
            rows.append(_forecast_row(
                origin, metadata.loc[origin], corruption, intensity,
                MODEL_NAMES[2], contexts[origin], quantiles, taus,
            ))
    return pd.DataFrame(rows)


def _forecast_row(
    origin: pd.Timestamp,
    metadata: pd.Series,
    corruption: str,
    intensity: float,
    model: str,
    context: np.ndarray,
    quantiles: np.ndarray,
    taus: np.ndarray,
) -> dict:
    q = np.asarray(quantiles, dtype=float)
    if len(q) != len(taus) or np.any(np.diff(q) < -1e-8):
        raise ValueError(f"{model} returned invalid deciles at {origin}")
    actual = float(metadata["actual"])
    crps = float(metrics.crps_from_quantiles(
        np.array([actual]), q.reshape(1, -1), taus
    )[0])
    return {
        "origin": pd.Timestamp(origin),
        "target_date": pd.Timestamp(metadata["target_date"]),
        "corruption": corruption,
        "intensity": float(intensity),
        "model": model,
        "actual": actual,
        "context_sha256": array_sha256(context),
        **{f"q{tau:.2f}": float(value) for tau, value in zip(taus, q)},
        "crps": crps,
    }


def _chronos_pipeline():
    from chronos import Chronos2Pipeline
    local = cached_snapshot_path(
        CHRONOS_ID, CHRONOS_REVISION, ("config.json",)
    )
    return Chronos2Pipeline.from_pretrained(
        local,
        local_files_only=True,
        device_map="cpu",
    )


def _quantile_columns(prediction: pd.DataFrame, taus: np.ndarray) -> list[str]:
    numeric: dict[float, str] = {}
    for column in prediction.columns:
        try:
            numeric[round(float(column), 8)] = column
        except (TypeError, ValueError):
            pass
    missing = [tau for tau in taus if round(float(tau), 8) not in numeric]
    if missing:
        raise KeyError(f"forecast output lacks quantiles {missing}")
    return [numeric[round(float(tau), 8)] for tau in taus]


def run_chronos(
    origins: pd.DatetimeIndex,
    bank: dict[tuple[str, float], dict[pd.Timestamp, np.ndarray]],
    metadata: pd.DataFrame,
    taus: np.ndarray,
    *,
    pipeline=None,
) -> pd.DataFrame:
    pipeline = pipeline or _chronos_pipeline()
    rows = []
    for (corruption, intensity), contexts in bank.items():
        # A synthetic regular grid prevents trading-day gaps from being inferred
        # as missing values.  Each origin remains an independent series id.
        frames = []
        for origin in origins:
            x = contexts[origin]
            frames.append(pd.DataFrame({
                "id": str(origin.date()),
                "timestamp": pd.Timestamp("2000-01-01") + pd.to_timedelta(np.arange(len(x)), unit="D"),
                "target": x,
            }))
        context_frame = pd.concat(frames, ignore_index=True)
        prediction = pipeline.predict_df(
            context_frame,
            prediction_length=1,
            quantile_levels=list(map(float, taus)),
            id_column="id",
            timestamp_column="timestamp",
            target="target",
            batch_size=32,
            context_length=len(next(iter(contexts.values()))),
        )
        qcols = _quantile_columns(prediction, taus)
        for origin in origins:
            match = prediction.loc[prediction["id"] == str(origin.date())]
            if len(match) != 1:
                raise ValueError(f"Chronos returned {len(match)} rows for {origin.date()}")
            q = match.iloc[0][qcols].to_numpy(dtype=float)
            rows.append(_forecast_row(
                origin, metadata.loc[origin], corruption, intensity,
                MODEL_NAMES[0], contexts[origin], q, taus,
            ))
    return pd.DataFrame(rows)


def _tirex_model():
    from tirex2 import load_model
    local = cached_snapshot_path(
        TIREX_ID, TIREX_REVISION, ("model-config.yaml", "model.ckpt")
    )
    return load_model(
        local,
        device="cpu",
    )


def run_tirex(
    origins: pd.DatetimeIndex,
    bank: dict[tuple[str, float], dict[pd.Timestamp, np.ndarray]],
    metadata: pd.DataFrame,
    taus: np.ndarray,
    *,
    model=None,
) -> pd.DataFrame:
    import torch
    from tirex2 import TimeseriesType

    model = model or _tirex_model()
    native = np.asarray([float(value) for value in model.quantiles])
    if not np.allclose(native, taus):
        raise ValueError(f"TiRex native grid {native.tolist()} is not the frozen decile grid")
    rows = []
    for (corruption, intensity), contexts in bank.items():
        items = [
            TimeseriesType(
                target=torch.tensor(contexts[origin], dtype=torch.float32).unsqueeze(0),
                past_covariates=None,
                future_covariates=None,
            )
            for origin in origins
        ]
        forecasts = model.forecast(
            items, prediction_length=1, output_type="numpy", batch_size=32
        )
        for origin, forecast in zip(origins, forecasts):
            q = np.asarray(forecast, dtype=float)[0, :, 0]
            rows.append(_forecast_row(
                origin, metadata.loc[origin], corruption, intensity,
                MODEL_NAMES[1], contexts[origin], q, taus,
            ))
    return pd.DataFrame(rows)


def verify_forecasts(frame: pd.DataFrame, protocol: dict) -> None:
    noise = protocol["noise_robustness"]
    taus = np.asarray(noise["quantiles"], dtype=float)
    qcols = [f"q{tau:.2f}" for tau in taus]
    required = {
        "origin", "target_date", "corruption", "intensity", "model", "actual",
        "context_sha256", "crps", *qcols,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"noise forecast artifact is missing {sorted(missing)}")
    if frame[list(required - {"context_sha256", "corruption", "model"})].isna().any().any():
        raise ValueError("noise forecast artifact contains missing values")
    expected_conditions = {
        *(('gaussian', float(x)) for x in noise["gaussian"]["intensities"]),
        *(('impulse', float(x)) for x in noise["impulse"]["probabilities"]),
    }
    if set(frame["model"]) != set(MODEL_NAMES):
        raise ValueError("noise artifact does not contain the three frozen models")
    if set(zip(frame["corruption"], frame["intensity"])) != expected_conditions:
        raise ValueError("noise artifact does not contain the complete corruption grid")
    permitted = pd.Timestamp(protocol["source"]["last_permitted_origin"])
    clean = pd.Timestamp(protocol["source"]["clean_start"])
    origins = pd.DatetimeIndex(frame["origin"])
    if origins.max() > permitted or (origins >= clean).any():
        raise ValueError("noise artifact escaped the sealed clean-window fence")
    key = ["origin", "corruption", "intensity"]
    counts = frame.groupby(key, observed=True)["model"].nunique()
    if not (counts == len(MODEL_NAMES)).all():
        raise ValueError("models do not share exact origin/condition rows")
    hashes = frame.groupby(key, observed=True)["context_sha256"].nunique()
    if not (hashes == 1).all():
        raise ValueError("models did not receive exact common corrupted contexts")
    for model in MODEL_NAMES:
        zero = frame[(frame["model"] == model) & (frame["intensity"] == 0)].copy()
        wide = zero.pivot(index="origin", columns="corruption", values=[*qcols, "crps", "context_sha256"])
        for column in [*qcols, "crps", "context_sha256"]:
            left = wide[(column, "gaussian")]
            right = wide[(column, "impulse")]
            if not left.equals(right):
                raise ValueError(f"zero-noise {column} identity failed for {model}")


def aggregate_curves(frame: pd.DataFrame) -> pd.DataFrame:
    base = frame[(frame["corruption"] == "gaussian") & (frame["intensity"] == 0)]
    baseline = base.groupby("model", observed=True)["crps"].mean()
    curve = (
        frame.groupby(["model", "corruption", "intensity"], observed=True)
        .agg(n=("origin", "size"), mean_crps=("crps", "mean"), median_crps=("crps", "median"))
        .reset_index()
    )
    curve["clean_mean_crps"] = curve["model"].map(baseline)
    curve["relative_crps"] = curve["mean_crps"] / curve["clean_mean_crps"]
    return curve


def moving_block_indices(
    n: int, block: int, draws: int, seed: int
) -> np.ndarray:
    if n < block or block < 1 or draws < 1:
        raise ValueError("invalid moving-block bootstrap dimensions")
    rng = np.random.default_rng(int(seed))
    blocks = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(draws, blocks))
    offsets = np.arange(block)
    return (starts[..., None] + offsets).reshape(draws, -1)[:, :n]


def bootstrap_pairwise(frame: pd.DataFrame, inference: dict) -> pd.DataFrame:
    """Paired intervals for differences in relative-CRPS degradation."""
    origins = pd.DatetimeIndex(sorted(frame["origin"].unique()))
    indexer = moving_block_indices(
        len(origins), int(inference["block_sampled_origins"]),
        int(inference["bootstrap_draws"]), int(inference["seed"]),
    )
    indexed = frame.set_index(["model", "corruption", "intensity", "origin"]).sort_index()["crps"]
    alpha = (1 - float(inference["confidence_level"])) / 2
    rows = []
    nonzero = sorted({
        (str(c), float(level))
        for c, level in zip(frame["corruption"], frame["intensity"])
        if float(level) > 0
    })
    for corruption, intensity in nonzero:
        for model_a, model_b in combinations(MODEL_NAMES, 2):
            clean_a = indexed.loc[(model_a, "gaussian", 0.0)].reindex(origins).to_numpy()
            clean_b = indexed.loc[(model_b, "gaussian", 0.0)].reindex(origins).to_numpy()
            noisy_a = indexed.loc[(model_a, corruption, intensity)].reindex(origins).to_numpy()
            noisy_b = indexed.loc[(model_b, corruption, intensity)].reindex(origins).to_numpy()
            if any(np.isnan(x).any() for x in (clean_a, clean_b, noisy_a, noisy_b)):
                raise ValueError("bootstrap comparison lacks common rows")
            estimate = noisy_a.mean() / clean_a.mean() - noisy_b.mean() / clean_b.mean()
            draws = (
                noisy_a[indexer].mean(axis=1) / clean_a[indexer].mean(axis=1)
                - noisy_b[indexer].mean(axis=1) / clean_b[indexer].mean(axis=1)
            )
            lo, hi = np.quantile(draws, [alpha, 1 - alpha])
            rows.append({
                "corruption": corruption,
                "intensity": intensity,
                "model_a": model_a,
                "model_b": model_b,
                "relative_crps_difference_a_minus_b": float(estimate),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "draws": int(inference["bootstrap_draws"]),
                "block_sampled_origins": int(inference["block_sampled_origins"]),
            })
    return pd.DataFrame(rows)


def _render_report(
    forecasts: pd.DataFrame, curves: pd.DataFrame, intervals: pd.DataFrame
) -> str:
    origins = pd.DatetimeIndex(sorted(forecasts["origin"].unique()))
    lines = [
        "# Eidos-derived context-noise robustness diagnostic",
        "",
        "This is an adaptation of Eidos Appendix A.1.2, not an evaluation of Eidos.",
        "All tests and corruption grids were frozen before model inference.",
        "",
        "## Result",
        "",
        "Under each model's native preprocessing, the foundation forecasts were less sensitive to injected context noise than the expanding HAR control. At maximum Gaussian noise, relative decile-grid CRPS approximation was 1.053 for Chronos-2, 1.032 for TiRex-2, and 1.212 for HAR. At 20% impulse contamination it was 1.314, 1.145, and 5.562, respectively.",
        "",
        "TiRex-2 degraded less than Chronos-2 under impulse corruption; their paired interval excluded zero only at probabilities 0.15 and 0.20. No registered Gaussian comparison between the two foundation models excluded zero. The slight CRPS improvements at the lowest noise levels are retained as observed and should be read as finite-sample/noise-regularization behavior, not as a tuned forecasting gain.",
        "",
        "Within the two foundation adapters, this makes surface-noise fragility an unlikely explanation for their earlier null. The HAR magnitude is not an apples-to-apples architectural contrast: unlike Eidos Appendix A.1.2, this adaptation did not impose one verified noisy-statistics renormalization pipeline across all three models. It does not establish a clean accuracy advantage: the diagnostic window was already used, and the corruption experiment measures stability rather than forecast value.",
        "",
        "## Design",
        "",
        f"- {len(origins)} origins, every twentieth session from {origins.min().date()} through {origins.max().date()}, forecasting the next session.",
        "- The input is the trailing 1,024 log-RV observations. Gaussian and impulse paths use a hash-derived seed per origin and exact common random numbers across all models and intensities.",
        "- Chronos-2 and TiRex-2 use locally cached pinned checkpoints. TiRex uses its checkpoint inference defaults, matching the existing univariate runner.",
        "- HAR is expanding OLS on daily log variance and the log of trailing 5/22-session mean variance. It is fit only on clean history through the origin; only the forecast-origin state is corrupted.",
        "- The model adapters receive the same raw corrupted log-RV arrays, then retain their native preprocessing. Chronos-2 and TiRex-2 may normalize internally; that behavior was not independently audited. HAR does not renormalize the corrupted origin state.",
        "- The stored `crps` score is a decile-grid CRPS approximation: twice the trapezoidal integral of pinball loss over quantiles 0.1-0.9, normalized by that grid width. Relative values divide its mean at an intensity by the same model's clean mean. Bootstrap intervals are secondary paired model differences, not model-accuracy tests.",
        "",
        "## Registered degradation curves",
        "",
        "| model | corruption | intensity | mean decile-grid CRPS approximation | relative score |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in curves.sort_values(["corruption", "model", "intensity"]).itertuples():
        lines.append(
            f"| {row.model} | {row.corruption} | {row.intensity:.2f} | "
            f"{row.mean_crps:.6f} | {row.relative_crps:.4f} |"
        )
    lines.extend([
        "",
        "## Paired 95% moving-block intervals",
        "",
        "A positive difference means model A degrades more than model B on relative CRPS.",
        "The 22-session dependence choice maps to two origins on this twentieth-session sampling grid.",
        "",
        "| corruption | intensity | model A | model B | A-B | 95% interval |",
        "|---|---:|---|---|---:|---:|",
    ])
    for row in intervals.sort_values(["corruption", "intensity", "model_a", "model_b"]).itertuples():
        lines.append(
            f"| {row.corruption} | {row.intensity:.2f} | {row.model_a} | {row.model_b} | "
            f"{row.relative_crps_difference_a_minus_b:.4f} | "
            f"[{row.ci_low:.4f}, {row.ci_high:.4f}] |"
        )
    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "The diagnostic asks how these particular model pipelines respond to controlled context corruption. Eidos re-normalizes each corrupted sequence using its noisy statistics; this adaptation retained model-native preprocessing and therefore does not reproduce that step uniformly. Cross-model HAR-versus-foundation magnitudes mix architecture with preprocessing. It does not establish that the Eidos architecture would be robust, nor does it create a new clean predictive claim. The 2016-2025 window was already diagnostic, and both foundation checkpoints may have encountered financial histories during pretraining.",
    ])
    return "\n".join(lines) + "\n"


def _write_report(
    forecasts: pd.DataFrame, curves: pd.DataFrame, intervals: pd.DataFrame
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_render_report(forecasts, curves, intervals))


def run() -> dict:
    protocol = load_protocol()
    validate_protocol(protocol)
    noise = protocol["noise_robustness"]
    frame = load_locked_history(protocol)
    origins = select_origins(frame, noise)
    bank, metadata = build_context_bank(frame, origins, noise)
    taus = np.asarray(noise["quantiles"], dtype=float)

    har = run_har(frame, origins, bank, metadata, taus)
    chronos = run_chronos(origins, bank, metadata, taus)
    tirex = run_tirex(origins, bank, metadata, taus)
    forecasts = pd.concat([chronos, tirex, har], ignore_index=True)
    verify_forecasts(forecasts, protocol)
    curves = aggregate_curves(forecasts)
    intervals = bootstrap_pairwise(forecasts, noise["comparison_inference"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(FORECAST_PATH, index=False)
    curves.to_parquet(CURVE_PATH, index=False)
    intervals.to_parquet(BOOTSTRAP_PATH, index=False)
    manifest = {
        "protocol": "representation_study.yaml",
        "protocol_sha256": _sha256(ROOT / "representation_study.yaml"),
        "history_panel_sha256": _sha256(
            ROOT / "data" / "history_extension" / "qqq_price_only_daily.parquet"
        ),
        "origins": len(origins),
        "first_origin": str(origins.min().date()),
        "last_origin": str(origins.max().date()),
        "models": list(MODEL_NAMES),
        "chronos_checkpoint": {"id": CHRONOS_ID, "revision": CHRONOS_REVISION},
        "tirex_checkpoint": {"id": TIREX_ID, "revision": TIREX_REVISION},
        "artifacts": {},
        "clean_origins_included": False,
    }
    for path in (FORECAST_PATH, CURVE_PATH, BOOTSTRAP_PATH):
        manifest["artifacts"][path.name] = {"sha256": _sha256(path), "rows": len(pd.read_parquet(path))}
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _write_report(forecasts, curves, intervals)
    print(f"wrote {FORECAST_PATH} ({len(forecasts):,} rows)")
    return manifest


def verify() -> None:
    protocol = load_protocol()
    validate_protocol(protocol)
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("noise manifest is absent; run the diagnostic first")
    manifest = json.loads(MANIFEST_PATH.read_text())
    if manifest.get("reviewed_protocol_sha256", manifest.get("protocol_sha256")) != _sha256(
        ROOT / "representation_study.yaml"
    ):
        raise ValueError("noise manifest does not match the reviewed protocol")
    if manifest.get("history_panel_sha256") != _sha256(
        ROOT / "data" / "history_extension" / "qqq_price_only_daily.parquet"
    ):
        raise ValueError("noise manifest does not match the frozen history panel")
    forecasts = pd.read_parquet(FORECAST_PATH)
    curves = pd.read_parquet(CURVE_PATH)
    intervals = pd.read_parquet(BOOTSTRAP_PATH)
    verify_forecasts(forecasts, protocol)
    for path in (FORECAST_PATH, CURVE_PATH, BOOTSTRAP_PATH):
        if _sha256(path) != manifest["artifacts"][path.name]["sha256"]:
            raise ValueError(f"{path.name} no longer matches its manifest hash")
    expected = aggregate_curves(forecasts).sort_values(["model", "corruption", "intensity"]).reset_index(drop=True)
    observed = curves.sort_values(["model", "corruption", "intensity"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, observed)
    expected_intervals = bootstrap_pairwise(
        forecasts, protocol["noise_robustness"]["comparison_inference"]
    ).sort_values(["corruption", "intensity", "model_a", "model_b"]).reset_index(drop=True)
    observed_intervals = intervals.sort_values(
        ["corruption", "intensity", "model_a", "model_b"]
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected_intervals, observed_intervals)
    if REPORT_PATH.read_text() != _render_report(forecasts, curves, intervals):
        raise ValueError("noise report does not reproduce from verified artifacts")
    print("noise robustness artifacts verified")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"), nargs="?", default="run")
    args = parser.parse_args(argv)
    run() if args.command == "run" else verify()


if __name__ == "__main__":
    main()
