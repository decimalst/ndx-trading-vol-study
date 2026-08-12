"""Frozen TiRex-2 latent-probe ladder for the extended QQQ history.

This module is intentionally separate from ``representation_study.py``.  The
machine-readable design in ``representation_study.yaml`` was frozen before the
tests in ``tests/test_latent_probe_study.py`` and before any embedding or probe
score was produced.

The probe never uses PCA.  It reads the complete 512-coordinate state at token
63 from the final ``stack_out_norm`` output, then evaluates the fixed capacity
ladder in annual forward folds.  Ten training-estimated first-order Markov
surrogates accompany each latent-only rung so raw probe capacity is not
mistaken for information in the representation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Callable, Iterable, NamedTuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from . import representation_study as base


ROOT = config.ROOT
PROTOCOL_PATH = ROOT / "representation_study.yaml"
OUTPUT_DIR = ROOT / "data" / "representation_study"
REPORT_DIR = ROOT / "reports" / "representation_study"
EMBEDDINGS_PATH = OUTPUT_DIR / "latent_embeddings.parquet"
EMBEDDINGS_MANIFEST_PATH = OUTPUT_DIR / "latent_embeddings_manifest.json"
CHUNK_DIR = OUTPUT_DIR / "latent_embedding_chunks"
CHUNK_MANIFEST_PATH = CHUNK_DIR / "manifest.json"
CLASSICAL_PATH = OUTPUT_DIR / "tail_classical_forecasts.parquet"
FORECASTS_PATH = OUTPUT_DIR / "latent_probe_forecasts.parquet"
CONTROLS_PATH = OUTPUT_DIR / "latent_probe_controls.parquet"
SELECTED_PATH = OUTPUT_DIR / "latent_selected_dimensions.parquet"
PHASE_PATH = OUTPUT_DIR / "latent_probe_phase_metrics.parquet"
METRICS_PATH = OUTPUT_DIR / "latent_probe_metrics.json"
MLP_CONVERGENCE_PATH = OUTPUT_DIR / "latent_mlp_convergence.parquet"
MLP_CONVERGENCE_METRICS_PATH = OUTPUT_DIR / "latent_mlp_convergence.json"
VERIFICATION_PATH = OUTPUT_DIR / "latent_probe_verification.json"
REPORT_PATH = REPORT_DIR / "latent_probe.md"
AUDIT_REPORT_PATH = REPORT_DIR / "latent_probe_result_audit.md"
WORK_DIR = OUTPUT_DIR / "latent_probe_work"

TOKEN_INDEX = 63
EXPECTED_DIMENSION = 512
MODEL_PACKAGE = "tirex-2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_protocol() -> dict:
    protocol = base.load_protocol(PROTOCOL_PATH)
    base.validate_protocol(protocol)
    latent = protocol["latent_probe"]
    if latent["package"] != "tirex-2==0.2.1":
        raise ValueError("latent probe package pin changed")
    if latent["checkpoint"] != "NX-AI/TiRex-2":
        raise ValueError("latent probe checkpoint changed")
    if latent["checkpoint_revision"] != "05e5b26db52bfb256f1ae1bdf785589850482de3":
        raise ValueError("latent probe checkpoint revision changed")
    if int(latent["context_length"]) != 2048:
        raise ValueError("latent probe context length changed")
    if int(latent["expected_dimension"]) != EXPECTED_DIMENSION:
        raise ValueError("latent dimension changed")
    if not latent.get("forbid_pca"):
        raise ValueError("dimension reduction is forbidden")
    return protocol


def assert_runtime_pin(protocol: dict) -> str:
    expected = protocol["latent_probe"]["package"].split("==", 1)[1]
    observed = package_version(MODEL_PACKAGE)
    if observed != expected:
        raise RuntimeError(f"{MODEL_PACKAGE} runtime {observed} != frozen {expected}")
    return observed


def latent_columns(dimension: int = EXPECTED_DIMENSION) -> list[str]:
    width = max(3, len(str(int(dimension) - 1)))
    return [f"z{coordinate:0{width}d}" for coordinate in range(int(dimension))]


def pool_final_context_token(
    hidden: np.ndarray,
    *,
    token_index: int = TOKEN_INDEX,
    expected_dim: int = EXPECTED_DIMENSION,
) -> np.ndarray:
    values = np.asarray(hidden)
    if values.ndim != 3:
        raise ValueError("hidden state must have batch, token, and dimension axes")
    if values.shape[-1] != int(expected_dim):
        raise ValueError(
            f"hidden dimension {values.shape[-1]} does not match frozen {expected_dim}"
        )
    if not 0 <= int(token_index) < values.shape[1]:
        raise ValueError(
            f"frozen token {token_index} is absent from {values.shape[1]} hidden tokens"
        )
    return np.asarray(values[:, int(token_index), :], dtype=np.float32)


def causal_context(
    values: pd.Series,
    origin: pd.Timestamp,
    *,
    context_length: int,
) -> np.ndarray:
    series = values.astype(float).sort_index()
    timestamp = pd.Timestamp(origin)
    if timestamp not in series.index:
        raise KeyError(f"origin {timestamp} is absent from the series")
    position = int(series.index.get_loc(timestamp))
    start = max(0, position + 1 - int(context_length))
    context = series.iloc[start : position + 1].to_numpy(dtype=np.float32)
    if not len(context) or len(context) > int(context_length) or not np.isfinite(context).all():
        raise ValueError("causal context is empty, overlong, or non-finite")
    return context


def _load_tirex_cpu(protocol: dict):
    from tirex2 import load_model

    assert_runtime_pin(protocol)
    latent = protocol["latent_probe"]
    model = load_model(
        latent["checkpoint"],
        device="cpu",
        hf_kwargs={
            "revision": latent["checkpoint_revision"],
            "local_files_only": True,
        },
    )
    backbone = model.model
    if int(backbone.context_len) != int(latent["context_length"]):
        raise RuntimeError("checkpoint context length differs from frozen protocol")
    if int(backbone.embedding_dim) != int(latent["expected_dimension"]):
        raise RuntimeError("checkpoint embedding dimension differs from frozen protocol")
    if int(backbone.context_len // backbone.input_patch_size) - 1 != TOKEN_INDEX:
        raise RuntimeError("checkpoint no longer maps the final context patch to token 63")
    return model


def _extract_batch(model, contexts: list[np.ndarray], protocol: dict) -> np.ndarray:
    import torch
    from tirex2 import TimeseriesType

    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        captured.append(output.detach().cpu())

    handle = model.model.stack_out_norm.register_forward_hook(hook)
    try:
        items = [
            TimeseriesType(
                target=torch.as_tensor(context, dtype=torch.float32).unsqueeze(0),
                past_covariates=None,
                future_covariates=None,
            )
            for context in contexts
        ]
        # TTA and differencing are explicitly disabled by the frozen protocol.
        model.forecast(
            items,
            prediction_length=1,
            output_type="torch",
            batch_size=len(items),
            tta_sign_flip=False,
            tta_diff=False,
        )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one hidden-state hook call, observed {len(captured)}")
    hidden = captured[0].numpy()
    if hidden.shape[0] != len(contexts):
        raise RuntimeError("hidden-state batch was reordered or changed in size")
    return pool_final_context_token(
        hidden,
        token_index=TOKEN_INDEX,
        expected_dim=int(protocol["latent_probe"]["expected_dimension"]),
    )


def embedding_origins(history: pd.DataFrame, classical: pd.DataFrame) -> pd.DatetimeIndex:
    features = base.build_history_features(history["log_rv"])
    complete = features.dropna().index
    last = pd.DatetimeIndex(classical.index).max()
    # Canonicalize the otherwise-semantic-free axis name before both chunk
    # validation and final consolidation.  The source panel calls it ``date``;
    # every latent artifact calls it ``origin``.
    return pd.DatetimeIndex(complete[complete <= last], name="origin")


def _checkpoint_files(protocol: dict) -> dict[str, dict[str, str | int]]:
    from huggingface_hub import snapshot_download

    latent = protocol["latent_probe"]
    directory = Path(
        snapshot_download(
            repo_id=latent["checkpoint"],
            revision=latent["checkpoint_revision"],
            allow_patterns=["model-config.yaml", "model.ckpt"],
            local_files_only=True,
        )
    )
    result: dict[str, dict[str, str | int]] = {}
    for name in ("model-config.yaml", "model.ckpt"):
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"pinned checkpoint artifact missing: {path}")
        result[name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    return result


def _origin_hash(origins: pd.DatetimeIndex) -> str:
    values = pd.DatetimeIndex(origins).astype("datetime64[ns]").asi8
    return hashlib.sha256(values.tobytes()).hexdigest()


def extraction_signature(
    protocol: dict,
    origins: pd.DatetimeIndex,
    checkpoint_files: dict[str, dict[str, str | int]] | None = None,
) -> dict:
    latent = protocol["latent_probe"]
    return {
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "history_panel_sha256": _sha256(base.HISTORY_PANEL),
        "classical_forecasts_sha256": _sha256(CLASSICAL_PATH),
        "checkpoint": latent["checkpoint"],
        "checkpoint_revision": latent["checkpoint_revision"],
        "checkpoint_files": checkpoint_files or _checkpoint_files(protocol),
        "package": f"{MODEL_PACKAGE}=={assert_runtime_pin(protocol)}",
        "context_length": int(latent["context_length"]),
        "layer": latent["layer"],
        "pooling": latent["pooling"],
        "token_index_zero_based": TOKEN_INDEX,
        "dimension": EXPECTED_DIMENSION,
        "tta_sign_flip": False,
        "differencing": False,
        "origins_sha256": _origin_hash(origins),
        "origin_rows": len(origins),
        "first_origin": str(origins.min().date()),
        "last_origin": str(origins.max().date()),
    }


def validate_chunk_manifest(expected_signature: dict, manifest: dict | None) -> None:
    if manifest is None:
        raise RuntimeError(
            "embedding chunks are unbound: a run-level chunk manifest is required for reuse"
        )
    if manifest.get("signature") != expected_signature:
        raise RuntimeError("embedding chunk manifest signature differs from the requested extraction")
    if not isinstance(manifest.get("chunks"), dict):
        raise RuntimeError("embedding chunk manifest has no per-chunk hashes")


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _embedding_manifest(
    protocol: dict,
    embeddings: pd.DataFrame,
    *,
    checkpoint_files: dict[str, dict[str, str | int]],
) -> dict:
    manifest = {
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "history_panel_sha256": _sha256(base.HISTORY_PANEL),
        "classical_forecasts_sha256": _sha256(CLASSICAL_PATH),
        "checkpoint": protocol["latent_probe"]["checkpoint"],
        "checkpoint_revision": protocol["latent_probe"]["checkpoint_revision"],
        "checkpoint_files": checkpoint_files,
        "package": f"{MODEL_PACKAGE}=={assert_runtime_pin(protocol)}",
        "device": "cpu",
        "layer": protocol["latent_probe"]["layer"],
        "pooling": protocol["latent_probe"]["pooling"],
        "token_index_zero_based": TOKEN_INDEX,
        "dimension": EXPECTED_DIMENSION,
        "context_length": int(protocol["latent_probe"]["context_length"]),
        "tta_sign_flip": False,
        "differencing": False,
        "rows": len(embeddings),
        "first_origin": str(embeddings.index.min().date()),
        "last_origin": str(embeddings.index.max().date()),
        "output_sha256": _sha256(EMBEDDINGS_PATH),
        "chunk_manifest_sha256": _sha256(CHUNK_MANIFEST_PATH),
    }
    _write_json_atomic(EMBEDDINGS_MANIFEST_PATH, manifest)
    return manifest


def extract_embeddings(
    protocol: dict | None = None,
    *,
    batch_size: int = 32,
    chunk_rows: int = 256,
) -> dict:
    """Extract and persist causal states in resumable, hash-audited chunks."""
    protocol = protocol or load_protocol()
    history = base.load_locked_history(protocol)
    if not CLASSICAL_PATH.exists():
        raise FileNotFoundError("run the frozen tail-classical study before latent extraction")
    classical = pd.read_parquet(CLASSICAL_PATH).sort_index()
    origins = embedding_origins(history, classical)
    expected_columns = latent_columns()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_files = _checkpoint_files(protocol)
    signature = extraction_signature(protocol, origins, checkpoint_files)
    existing_chunks = sorted(CHUNK_DIR.glob("latent_embeddings_*.parquet"))
    if existing_chunks:
        observed_manifest = (
            json.loads(CHUNK_MANIFEST_PATH.read_text())
            if CHUNK_MANIFEST_PATH.exists()
            else None
        )
        validate_chunk_manifest(signature, observed_manifest)
        chunk_manifest = observed_manifest
    else:
        chunk_manifest = {"signature": signature, "status": "extracting", "chunks": {}}
        _write_json_atomic(CHUNK_MANIFEST_PATH, chunk_manifest)
    model = None
    parts: list[pd.DataFrame] = []
    for start in range(0, len(origins), int(chunk_rows)):
        stop = min(start + int(chunk_rows), len(origins))
        expected_index = origins[start:stop]
        part_path = CHUNK_DIR / f"latent_embeddings_{start:06d}_{stop - 1:06d}.parquet"
        if part_path.exists():
            recorded = chunk_manifest["chunks"].get(part_path.name)
            if recorded is None or recorded.get("sha256") != _sha256(part_path):
                raise RuntimeError(f"embedding chunk hash is absent or changed: {part_path}")
            part = pd.read_parquet(part_path)
            if list(part.columns) != expected_columns:
                raise RuntimeError(f"embedding chunk schema changed: {part_path}")
            pd.testing.assert_index_equal(pd.DatetimeIndex(part.index), expected_index)
            if not np.isfinite(part.to_numpy()).all():
                raise RuntimeError(f"embedding chunk contains non-finite values: {part_path}")
            if int(recorded.get("rows", -1)) != len(part):
                raise RuntimeError(f"embedding chunk row count differs from manifest: {part_path}")
            parts.append(part)
            continue
        if model is None:
            model = _load_tirex_cpu(protocol)
        chunk_arrays: list[np.ndarray] = []
        for batch_start in range(start, stop, int(batch_size)):
            batch_stop = min(batch_start + int(batch_size), stop)
            batch_origins = origins[batch_start:batch_stop]
            contexts = [
                causal_context(
                    history["log_rv"],
                    origin,
                    context_length=int(protocol["latent_probe"]["context_length"]),
                )
                for origin in batch_origins
            ]
            chunk_arrays.append(_extract_batch(model, contexts, protocol))
            print(
                f"latent embeddings {batch_stop:,}/{len(origins):,}",
                flush=True,
            )
        matrix = np.concatenate(chunk_arrays, axis=0)
        part = pd.DataFrame(matrix, index=expected_index, columns=expected_columns)
        part.index.name = "origin"
        part.to_parquet(part_path, compression="zstd")
        chunk_manifest["chunks"][part_path.name] = {
            "sha256": _sha256(part_path),
            "rows": len(part),
            "first_origin": str(part.index.min().date()),
            "last_origin": str(part.index.max().date()),
        }
        _write_json_atomic(CHUNK_MANIFEST_PATH, chunk_manifest)
        parts.append(part)
    embeddings = pd.concat(parts).sort_index()
    pd.testing.assert_index_equal(pd.DatetimeIndex(embeddings.index), origins)
    if embeddings.shape != (len(origins), EXPECTED_DIMENSION):
        raise RuntimeError("consolidated embedding matrix has the wrong shape")
    embeddings.to_parquet(EMBEDDINGS_PATH, compression="zstd")
    chunk_manifest["status"] = "complete"
    chunk_manifest["output_sha256"] = _sha256(EMBEDDINGS_PATH)
    _write_json_atomic(CHUNK_MANIFEST_PATH, chunk_manifest)
    return _embedding_manifest(protocol, embeddings, checkpoint_files=checkpoint_files)


def seal_existing_chunks(protocol: dict | None = None) -> dict:
    """Bind legacy chunks only after proving they exactly reconstruct final embeddings."""
    protocol = protocol or load_protocol()
    history = base.load_locked_history(protocol)
    classical = pd.read_parquet(CLASSICAL_PATH).sort_index()
    origins = embedding_origins(history, classical)
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError("the verified consolidated embedding matrix is missing")
    final = pd.read_parquet(EMBEDDINGS_PATH).sort_index()
    pd.testing.assert_index_equal(pd.DatetimeIndex(final.index), origins)
    checkpoint_files = _checkpoint_files(protocol)
    signature = extraction_signature(protocol, origins, checkpoint_files)
    files = sorted(CHUNK_DIR.glob("latent_embeddings_*.parquet"))
    if not files:
        raise FileNotFoundError("there are no embedding chunks to seal")
    chunks = {}
    parts = []
    expected_start = 0
    for path in files:
        fields = path.stem.rsplit("_", 2)
        start, final_position = int(fields[-2]), int(fields[-1])
        if start != expected_start or final_position < start:
            raise RuntimeError("legacy embedding chunks are not contiguous")
        expected_index = origins[start : final_position + 1]
        part = pd.read_parquet(path)
        pd.testing.assert_index_equal(pd.DatetimeIndex(part.index), expected_index)
        if list(part.columns) != latent_columns() or not np.isfinite(part.to_numpy()).all():
            raise RuntimeError(f"legacy chunk schema or values changed: {path}")
        chunks[path.name] = {
            "sha256": _sha256(path),
            "rows": len(part),
            "first_origin": str(part.index.min().date()),
            "last_origin": str(part.index.max().date()),
        }
        parts.append(part)
        expected_start = final_position + 1
    if expected_start != len(origins):
        raise RuntimeError("legacy embedding chunks do not cover every requested origin")
    reconstructed = pd.concat(parts).sort_index()
    pd.testing.assert_frame_equal(reconstructed, final, check_exact=True)
    chunk_manifest = {
        "signature": signature,
        "status": "sealed_after_exact_final_matrix_reconstruction",
        "chunks": chunks,
        "output_sha256": _sha256(EMBEDDINGS_PATH),
        "provenance_repair": (
            "Legacy chunks predated the run-level manifest. They were adopted only after "
            "exactly reconstructing the independently verified final matrix."
        ),
    }
    _write_json_atomic(CHUNK_MANIFEST_PATH, chunk_manifest)
    return _embedding_manifest(protocol, final, checkpoint_files=checkpoint_files)


def load_embeddings(protocol: dict) -> pd.DataFrame:
    if not EMBEDDINGS_PATH.exists() or not EMBEDDINGS_MANIFEST_PATH.exists():
        raise FileNotFoundError("extract the frozen TiRex embeddings before probing")
    manifest = json.loads(EMBEDDINGS_MANIFEST_PATH.read_text())
    checks = {
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "history_panel_sha256": _sha256(base.HISTORY_PANEL),
        "classical_forecasts_sha256": _sha256(CLASSICAL_PATH),
        "output_sha256": _sha256(EMBEDDINGS_PATH),
        "chunk_manifest_sha256": _sha256(CHUNK_MANIFEST_PATH),
    }
    for key, observed in checks.items():
        if manifest.get(key) != observed:
            raise RuntimeError(f"latent embedding manifest mismatch: {key}")
    if manifest.get("checkpoint_revision") != protocol["latent_probe"]["checkpoint_revision"]:
        raise RuntimeError("latent embedding checkpoint revision changed")
    if manifest.get("package") != protocol["latent_probe"]["package"]:
        raise RuntimeError("latent embedding package changed")
    if int(manifest.get("token_index_zero_based", -1)) != TOKEN_INDEX:
        raise RuntimeError("latent embedding token changed")
    if int(manifest.get("dimension", -1)) != EXPECTED_DIMENSION:
        raise RuntimeError("latent embedding dimension changed")
    frame = pd.read_parquet(EMBEDDINGS_PATH).sort_index()
    if list(frame.columns) != latent_columns():
        raise RuntimeError("latent embedding coordinate schema changed")
    if not np.isfinite(frame.to_numpy()).all():
        raise RuntimeError("latent embeddings contain non-finite values")
    return frame


def standardized_mean_difference(X: np.ndarray, event: np.ndarray) -> np.ndarray:
    values = np.asarray(X, dtype=float)
    y = np.asarray(event, dtype=int)
    if values.ndim != 2 or len(values) != len(y):
        raise ValueError("effect sizes need aligned two-dimensional training inputs")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("effect sizes need both event classes")
    scale = values.std(axis=0, ddof=0)
    scale[scale < 1e-12] = 1.0
    return (values[y == 1].mean(axis=0) - values[y == 0].mean(axis=0)) / scale


def select_sparse_dimensions(X: np.ndarray, event: np.ndarray, *, k: int) -> np.ndarray:
    values = np.asarray(X, dtype=float)
    if not 0 < int(k) <= values.shape[1]:
        raise ValueError("sparse probe k is outside the latent dimension")
    effect = np.abs(standardized_mean_difference(values, event))
    return np.argsort(-effect, kind="mergesort")[: int(k)]


@dataclass
class FittedProbe:
    estimator: Pipeline
    n_input_dimensions: int
    n_iter: int | None = None
    max_iter: int | None = None
    converged: bool | None = None
    convergence_warning: bool = False

    def predict(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_input_dimensions:
            raise ValueError("probe prediction input dimension changed")
        return np.asarray(self.estimator.predict_proba(values)[:, 1], dtype=float)


def fit_probe(
    rung: str,
    X: np.ndarray,
    event: np.ndarray,
    *,
    ridge: float,
    seed: int,
    mlp_config: dict | None = None,
) -> FittedProbe:
    values = np.asarray(X, dtype=float)
    y = np.asarray(event, dtype=int)
    if values.ndim != 2 or len(values) != len(y) or not np.isfinite(values).all():
        raise ValueError("probe training inputs are misaligned or non-finite")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("probe training needs both event classes")
    scaler = StandardScaler()
    if rung == "small_mlp":
        cfg = mlp_config or {}
        classifier = MLPClassifier(
            hidden_layer_sizes=(int(cfg.get("hidden_units", 8)),),
            activation=str(cfg.get("activation", "tanh")),
            solver=str(cfg.get("solver", "lbfgs")),
            alpha=float(cfg.get("l2_alpha", ridge)),
            max_iter=int(cfg.get("max_iter", 500)),
            random_state=int(cfg.get("seed", seed)),
        )
    else:
        if float(ridge) <= 0:
            raise ValueError("ridge penalty must be positive")
        classifier = LogisticRegression(
            penalty="l2",
            C=1.0 / float(ridge),
            solver="lbfgs",
            max_iter=1000,
            random_state=int(seed),
        )
    estimator = Pipeline([("scale", scaler), ("probe", classifier)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category=ConvergenceWarning)
        estimator.fit(values, y)
    convergence_warning = any(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )
    n_iter_value = getattr(classifier, "n_iter_", None)
    if isinstance(n_iter_value, np.ndarray):
        n_iter = int(np.max(n_iter_value))
    elif n_iter_value is None:
        n_iter = None
    else:
        n_iter = int(n_iter_value)
    max_iterations = int(getattr(classifier, "max_iter", 0)) or None
    converged = None
    if rung == "small_mlp":
        converged = bool(
            not convergence_warning
            and n_iter is not None
            and max_iterations is not None
            and n_iter < max_iterations
        )
    return FittedProbe(
        estimator=estimator,
        n_input_dimensions=values.shape[1],
        n_iter=n_iter,
        max_iter=max_iterations,
        converged=converged,
        convergence_warning=convergence_warning,
    )


class MarkovPath(NamedTuple):
    train: np.ndarray
    test: np.ndarray


def _markov_transition(training_event: np.ndarray) -> tuple[np.ndarray, float]:
    y = np.asarray(training_event, dtype=int)
    if len(y) < 2 or set(np.unique(y)) != {0, 1}:
        raise ValueError("Markov control needs both completed training classes")
    counts = np.ones((2, 2), dtype=float)
    for left, right in zip(y[:-1], y[1:]):
        counts[int(left), int(right)] += 1.0
    return counts / counts.sum(axis=1, keepdims=True), float(y.mean())


def markov_surrogate_path(
    training_event: np.ndarray,
    *,
    train_length: int,
    test_length: int,
    seed: int,
) -> MarkovPath:
    """Simulate one continuous train-plus-test path from training transitions."""
    transition, prevalence = _markov_transition(training_event)
    total = int(train_length) + int(test_length)
    if total < 4 or int(train_length) < 2 or int(test_length) < 2:
        raise ValueError("Markov control path is too short")
    rng = np.random.default_rng(int(seed))
    for _ in range(100):
        path = np.empty(total, dtype=np.int8)
        path[0] = int(rng.random() < prevalence)
        for position in range(1, total):
            path[position] = int(rng.random() < transition[path[position - 1], 1])
        train = path[: int(train_length)]
        test = path[int(train_length) :]
        if set(np.unique(train)) == {0, 1} and set(np.unique(test)) == {0, 1}:
            return MarkovPath(train=train, test=test)
    raise RuntimeError("could not simulate non-degenerate train and test control labels")


def build_annual_fold_tables(
    *,
    y: pd.Series,
    features: pd.DataFrame,
    embeddings: pd.DataFrame,
    classical_fold: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon: int,
    quantile: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    series = y.astype(float).sort_index()
    cutoff = pd.Timestamp(cutoff)
    train_origins = base.completed_training_origins(series.index, cutoff, int(horizon))
    targets = base.build_fold_targets(
        series,
        cutoff=cutoff,
        origins=train_origins,
        horizon=int(horizon),
        quantile=float(quantile),
    )
    targets = targets.loc[targets["calm"]]
    feature_columns = ["log_rv_d", "log_rv_w", "log_rv_m"]
    z_columns = latent_columns(embeddings.shape[1])
    if list(embeddings.columns) != z_columns:
        raise ValueError("embedding coordinates are not the frozen ordered schema")
    train_index = targets.index.intersection(features[feature_columns].dropna().index)
    train_index = train_index.intersection(embeddings.index)
    train = targets.loc[train_index].copy()
    train = train.join(features.loc[train_index, feature_columns])
    train = train.join(embeddings.loc[train_index, z_columns])
    if train.empty or (train["target_end"] > cutoff).any():
        raise RuntimeError("annual training labels were not completed by the cutoff")

    requested = pd.DatetimeIndex(classical_fold.index)
    if requested.has_duplicates or not requested.is_monotonic_increasing:
        raise ValueError("classical test rows must be unique and chronological")
    missing_features = requested.difference(features[feature_columns].dropna().index)
    missing_embeddings = requested.difference(embeddings.index)
    if len(missing_features) or len(missing_embeddings):
        raise RuntimeError("latent probe cannot silently shrink the classical common rows")
    test = classical_fold.copy()
    test = test.join(features.loc[requested, feature_columns], rsuffix="_feature")
    # The classical artifact already carries the same HAR columns.  When the
    # join creates duplicates, verify equality and retain the frozen columns.
    for column in feature_columns:
        duplicate = f"{column}_feature"
        if duplicate in test:
            if not np.allclose(test[column], test[duplicate], rtol=0, atol=1e-12):
                raise RuntimeError(f"classical {column} no longer matches source history")
            test = test.drop(columns=duplicate)
    test = test.join(embeddings.loc[requested, z_columns])
    pd.testing.assert_index_equal(pd.DatetimeIndex(test.index), requested)
    if len(test) != len(classical_fold) or test[z_columns].isna().any().any():
        raise RuntimeError("annual latent test rows differ from the classical common rows")
    if len(train.index.intersection(test.index)):
        raise RuntimeError("annual training and test origins overlap")
    return train, test


def phase_ranking_metrics(
    frame: pd.DataFrame,
    *,
    score_column: str,
    horizon: int,
    top_fraction: float,
    event_column: str = "event",
) -> pd.DataFrame:
    if score_column not in frame or event_column not in frame:
        raise ValueError("ranking frame is missing its event or score")
    ordered = frame.sort_index(kind="mergesort")
    rows = []
    if "phase" in ordered:
        phase_samples = [(phase, ordered.loc[ordered["phase"] == phase]) for phase in range(int(horizon))]
    else:
        phase_samples = [(phase, ordered.iloc[phase:: int(horizon)]) for phase in range(int(horizon))]
    for phase, sample in phase_samples:
        rows.append(
            {
                "phase": phase,
                "n": len(sample),
                "event_rate": float(sample[event_column].mean()),
                "auc": base.roc_auc(sample[event_column], sample[score_column]),
                "top_decile_lift": base.top_decile_lift(
                    sample[event_column], sample[score_column], float(top_fraction)
                ),
            }
        )
    return pd.DataFrame(rows)


def phase_mean(metrics: pd.DataFrame) -> dict[str, float]:
    if set(metrics["phase"]) != set(range(5)):
        raise ValueError("the frozen scoreboard requires all five phases")
    return {
        "auc": float(metrics["auc"].mean()),
        "top_decile_lift": float(metrics["top_decile_lift"].mean()),
    }


def phase_dispersion(metrics: pd.DataFrame) -> dict[str, dict[str, float]]:
    if set(metrics["phase"]) != set(range(5)):
        raise ValueError("phase dispersion requires all five frozen phases")
    result: dict[str, dict[str, float]] = {}
    for metric in ("auc", "top_decile_lift"):
        low = float(metrics[metric].min())
        high = float(metrics[metric].max())
        result[metric] = {"min": low, "max": high, "spread": high - low}
    return result


def randomization_evidence(
    actual: float,
    controls: Iterable[float],
    *,
    alpha: float,
) -> dict[str, float | int | bool | str]:
    values = np.asarray(list(controls), dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("randomization evidence requires finite controls")
    exceedances = int(np.sum(values >= float(actual)))
    exact_p = float((exceedances + 1) / (len(values) + 1))
    lower_bound = float(1 / (len(values) + 1))
    formal = bool(exact_p <= float(alpha))
    return {
        "control_draws": len(values),
        "control_exceedances": exceedances,
        "exact_randomization_p": exact_p,
        "exact_randomization_p_lower_bound": lower_bound,
        "alpha": float(alpha),
        "formal_evidence": formal,
        "reason": (
            "exact corrected Monte Carlo randomization p meets alpha"
            if formal
            else "control count cannot confirm the representation at the requested alpha"
        ),
    }


def assign_transition_episodes(
    trigger_date: pd.Series,
    sessions: pd.DatetimeIndex,
    *,
    max_gap_sessions: int,
) -> pd.Series:
    positions = pd.Series(np.arange(len(sessions)), index=pd.DatetimeIndex(sessions))
    valid_dates = pd.DatetimeIndex(trigger_date.dropna().unique()).sort_values()
    date_to_episode: dict[pd.Timestamp, int] = {}
    episode = -1
    previous: int | None = None
    for date in valid_dates:
        if date not in positions:
            raise ValueError("transition trigger is absent from source sessions")
        position = int(positions.loc[date])
        if previous is None or position - previous > int(max_gap_sessions):
            episode += 1
        date_to_episode[pd.Timestamp(date)] = episode
        previous = position
    assigned = pd.Series(pd.NA, index=trigger_date.index, dtype="Int64")
    for origin, date in trigger_date.items():
        if pd.notna(date):
            assigned.loc[origin] = date_to_episode[pd.Timestamp(date)]
    return assigned


def assign_control_episodes(
    event: np.ndarray | pd.Series,
    origins: pd.DatetimeIndex,
    folds: np.ndarray | pd.Series,
    sessions: pd.DatetimeIndex,
    *,
    max_gap_sessions: int,
) -> pd.Series:
    """Cluster surrogate positives on origin sessions, separately by fold.

    A Markov surrogate has no observed future threshold trigger.  Its positive
    origin is therefore the structural proxy trigger.  Annual fold boundaries
    remain hard boundaries because every surrogate path is re-estimated and
    restarted annually.
    """
    y = np.asarray(event, dtype=bool)
    dates = pd.DatetimeIndex(origins)
    fold_values = np.asarray(folds)
    if len(y) != len(dates) or len(y) != len(fold_values):
        raise ValueError("control episode inputs are not aligned")
    if dates.has_duplicates:
        raise ValueError("control origins must be unique within a scored path")
    positions = pd.Series(np.arange(len(sessions)), index=pd.DatetimeIndex(sessions))
    if len(dates.difference(positions.index)):
        raise ValueError("control origin is absent from source sessions")
    assigned = pd.Series(pd.NA, index=dates, dtype="Int64")
    next_episode = -1
    # Preserve chronological fold order; sorted unique years is deterministic.
    for fold in sorted(pd.unique(fold_values)):
        mask = (fold_values == fold) & y
        positive_dates = dates[mask].sort_values()
        previous: int | None = None
        for date in positive_dates:
            position = int(positions.loc[date])
            if previous is None or position - previous > int(max_gap_sessions):
                next_episode += 1
            assigned.loc[date] = next_episode
            previous = position
    return assigned


def leave_one_episode_out(
    frame: pd.DataFrame,
    *,
    episode_column: str,
    statistic: Callable[[pd.DataFrame], float],
) -> dict[str, float | int | list[float]]:
    if episode_column not in frame:
        raise ValueError("jackknife frame lacks transition episodes")
    full = float(statistic(frame))
    episodes = sorted(int(value) for value in frame[episode_column].dropna().unique())
    if len(episodes) < 2:
        return {
            "estimate": full,
            "lower": float("nan"),
            "upper": float("nan"),
            "standard_error": float("nan"),
            "episodes": len(episodes),
            "leave_one_out": [],
        }
    replicates = np.asarray(
        [
            float(statistic(frame.loc[frame[episode_column].isna() | (frame[episode_column] != episode)]))
            for episode in episodes
        ],
        dtype=float,
    )
    finite = np.isfinite(replicates)
    if finite.sum() != len(replicates):
        raise RuntimeError("episode jackknife produced an undefined replicate")
    center = float(replicates.mean())
    standard_error = float(
        np.sqrt((len(replicates) - 1) / len(replicates) * np.sum((replicates - center) ** 2))
    )
    return {
        "estimate": full,
        "lower": full - 1.96 * standard_error,
        "upper": full + 1.96 * standard_error,
        "standard_error": standard_error,
        "episodes": len(episodes),
        "leave_one_out": replicates.tolist(),
    }


def _rung_specs(protocol: dict) -> list[tuple[str, int | None]]:
    sparse = [int(k) for k in protocol["latent_probe"]["probe_ladder"]["sparse"]["k"]]
    return [("full_ridge", None), *[(f"sparse_k{k}", k) for k in sparse], ("small_mlp", None)]


def _archive_previous_outputs() -> Path | None:
    candidates = [FORECASTS_PATH, CONTROLS_PATH, SELECTED_PATH, PHASE_PATH, METRICS_PATH, REPORT_PATH, WORK_DIR]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    archive = OUTPUT_DIR / f"latent_pre_repair_{_utc_stamp()}"
    archive.mkdir(parents=True, exist_ok=False)
    for path in existing:
        destination = archive / path.name
        shutil.move(str(path), str(destination))
    return archive


def _strict_validate_classical_fold(
    y: pd.Series,
    classical_fold: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    horizon: int,
    quantile: float,
) -> None:
    rebuilt = base.build_fold_targets(
        y,
        cutoff=pd.Timestamp(cutoff),
        origins=pd.DatetimeIndex(classical_fold.index),
        horizon=int(horizon),
        quantile=float(quantile),
    )
    rebuilt = rebuilt.loc[rebuilt["calm"]]
    # Parquet round-tripping can discard the classical artifact's axis label;
    # timestamps and order are substantive, the index ``name`` is not.
    pd.testing.assert_index_equal(
        pd.DatetimeIndex(rebuilt.index),
        pd.DatetimeIndex(classical_fold.index),
        check_names=False,
    )
    for column in ("event", "target_end", "trigger_date"):
        left = rebuilt[column].reset_index(drop=True)
        right = classical_fold[column].reset_index(drop=True)
        pd.testing.assert_series_equal(left, right, check_names=False, check_dtype=False)
    if not np.allclose(rebuilt["threshold"], classical_fold["threshold"], rtol=0, atol=1e-12):
        raise RuntimeError("classical fold threshold no longer reconstructs")


def _matrix(frame: pd.DataFrame, columns: Iterable[str]) -> np.ndarray:
    values = frame.loc[:, list(columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("probe matrix contains non-finite inputs")
    return values


def _score_summary(
    frame: pd.DataFrame,
    score_column: str,
    protocol: dict,
    *,
    event_column: str = "event",
) -> dict[str, float]:
    scores = protocol["ranking_scoreboard"]
    metrics = phase_ranking_metrics(
        frame,
        score_column=score_column,
        event_column=event_column,
        horizon=int(protocol["tail_target"]["horizon_sessions"]),
        top_fraction=float(scores["top_fraction"]),
    )
    return phase_mean(metrics)


def _jackknife_variance(replicates: Iterable[float]) -> float:
    values = np.asarray(list(replicates), dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise RuntimeError("cluster jackknife needs at least two finite episode deletions")
    center = float(values.mean())
    return float((len(values) - 1) / len(values) * np.sum((values - center) ** 2))


def clustered_selectivity_interval(
    wide: pd.DataFrame,
    protocol: dict,
    *,
    candidate_score: str,
    controls: list[tuple[str, str, str, int]],
    metric: str,
) -> dict:
    """Combine actual and structurally matched control episode components.

    ``controls`` entries are ``(event, score, episode, seed)``.  Actual episode
    deletions alter only the actual metric.  Each control-seed deletion alters
    only that seed's metric before recomputing the ten-control median.  The
    independent component jackknife variances are summed.
    """
    if metric not in ("auc", "top_decile_lift") or not controls:
        raise ValueError("selectivity interval needs a frozen metric and controls")
    actual_full = _score_summary(wide, candidate_score, protocol)[metric]
    control_full = [
        _score_summary(wide, score, protocol, event_column=event)[metric]
        for event, score, _episode, _seed in controls
    ]
    estimate = float(actual_full - np.median(control_full))
    actual_episodes = sorted(wide["episode"].dropna().unique())
    actual_replicates = []
    for episode in actual_episodes:
        sample = wide.loc[wide["episode"].isna() | (wide["episode"] != episode)]
        actual_replicates.append(
            _score_summary(sample, candidate_score, protocol)[metric]
            - float(np.median(control_full))
        )
    components: dict[str, float] = {
        "actual": _jackknife_variance(actual_replicates)
    }
    control_counts: dict[str, int] = {}
    for control_index, (event, score, episode_column, seed) in enumerate(controls):
        episodes = sorted(wide[episode_column].dropna().unique())
        control_counts[str(seed)] = len(episodes)
        replicates = []
        for episode in episodes:
            sample = wide.loc[
                wide[episode_column].isna() | (wide[episode_column] != episode)
            ]
            changed = list(control_full)
            changed[control_index] = _score_summary(
                sample, score, protocol, event_column=event
            )[metric]
            replicates.append(float(actual_full - np.median(changed)))
        components[f"control_seed_{seed}"] = _jackknife_variance(replicates)
    standard_error = float(np.sqrt(sum(components.values())))
    return {
        "estimate": estimate,
        "lower": estimate - 1.96 * standard_error,
        "upper": estimate + 1.96 * standard_error,
        "standard_error": standard_error,
        "actual_episodes": len(actual_episodes),
        "control_episodes_by_seed": control_counts,
        "variance_components": components,
        "method": "sum of actual and per-seed control episode jackknife variances",
    }


def _jackknife_comparison(
    wide: pd.DataFrame,
    protocol: dict,
    *,
    candidate_score: str,
    baseline_score: str | None = None,
    control_pairs: list[tuple[str, str]] | None = None,
    metric: str,
) -> dict:
    if (baseline_score is None) == (control_pairs is None):
        raise ValueError("jackknife comparison needs exactly one baseline type")

    def statistic(sample: pd.DataFrame) -> float:
        actual = _score_summary(sample, candidate_score, protocol)[metric]
        if baseline_score is not None:
            return actual - _score_summary(sample, baseline_score, protocol)[metric]
        controls = [
            _score_summary(sample, score, protocol, event_column=event)[metric]
            for event, score in (control_pairs or [])
        ]
        return actual - float(np.median(controls))

    return leave_one_episode_out(wide, episode_column="episode", statistic=statistic)


def _dimension_stability(selected: pd.DataFrame) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for rung, group in selected.groupby("rung", sort=False):
        years = sorted(int(year) for year in group["fold_year"].unique())
        sets = {
            year: set(group.loc[group["fold_year"] == year, "dimension"].astype(int))
            for year in years
        }
        jaccards = []
        for left, right in zip(years[:-1], years[1:]):
            union = sets[left] | sets[right]
            jaccards.append(len(sets[left] & sets[right]) / len(union) if union else float("nan"))
        counts = group.groupby("dimension")["fold_year"].nunique().sort_values(ascending=False)
        top = [
            {"dimension": int(dimension), "folds": int(count), "fraction": float(count / len(years))}
            for dimension, count in counts.head(10).items()
        ]
        result[rung] = {
            "folds": len(years),
            "unique_dimensions": int(group["dimension"].nunique()),
            "mean_consecutive_jaccard": float(np.mean(jaccards)) if jaccards else float("nan"),
            "top_dimensions": top,
        }
    return result


def audit_mlp_convergence(protocol: dict | None = None) -> dict:
    """Refit the frozen MLP tasks only to expose optimizer termination status.

    This audit changes neither optimizer settings nor stored forecasts.  It
    reconstructs the same 24 annual training sets, the two actual-label fits,
    and ten fold-conditional Markov-control fits, then persists ``n_iter_`` and
    whether LBFGS stopped before the frozen iteration cap.
    """
    protocol = protocol or load_protocol()
    history = base.load_locked_history(protocol)
    embeddings = load_embeddings(protocol)
    classical = pd.read_parquet(CLASSICAL_PATH).sort_index()
    y = history["log_rv"].dropna().sort_index()
    features = base.build_history_features(y)
    target_cfg = protocol["tail_target"]
    horizon = int(target_cfg["horizon_sessions"])
    quantile = float(target_cfg["stress_quantile"])
    latent_cfg = protocol["latent_probe"]
    mlp_cfg = latent_cfg["probe_ladder"]["small_mlp"]
    ridge = float(latent_cfg["probe_ladder"]["full_ridge"]["ridge"])
    seeds = [int(seed) for seed in latent_cfg["control_task"]["seeds"]]
    z_columns = latent_columns()
    feature_columns = ["log_rv_d", "log_rv_w", "log_rv_m"]
    rows = []

    def record(year: int, task: str, seed: int | None, model: FittedProbe) -> None:
        rows.append(
            {
                "fold_year": int(year),
                "task": task,
                "seed": seed,
                "n_input_dimensions": model.n_input_dimensions,
                "n_iter": model.n_iter,
                "max_iter": model.max_iter,
                "converged": model.converged,
                "convergence_warning": model.convergence_warning,
            }
        )

    for year, classical_fold in classical.groupby("fold_year", sort=True):
        year = int(year)
        classical_fold = classical_fold.sort_index()
        cutoff = pd.DatetimeIndex(classical_fold["cutoff"].unique())[0]
        train, _test = build_annual_fold_tables(
            y=y,
            features=features,
            embeddings=embeddings,
            classical_fold=classical_fold,
            cutoff=cutoff,
            horizon=horizon,
            quantile=quantile,
        )
        actual_y = train["event"].astype(int).to_numpy()
        latent_X = _matrix(train, z_columns)
        rv_X = _matrix(train, feature_columns)
        actual_latent = fit_probe(
            "small_mlp",
            latent_X,
            actual_y,
            ridge=ridge,
            seed=int(mlp_cfg["seed"]),
            mlp_config=mlp_cfg,
        )
        record(year, "latent_actual", None, actual_latent)
        actual_augmented = fit_probe(
            "small_mlp",
            np.column_stack([rv_X, latent_X]),
            actual_y,
            ridge=ridge,
            seed=int(mlp_cfg["seed"]),
            mlp_config=mlp_cfg,
        )
        record(year, "augmented_actual", None, actual_augmented)
        for seed in seeds:
            path = markov_surrogate_path(
                actual_y,
                train_length=len(train),
                test_length=len(classical_fold),
                seed=seed,
            )
            control = fit_probe(
                "small_mlp",
                latent_X,
                path.train,
                ridge=ridge,
                seed=int(mlp_cfg["seed"]),
                mlp_config=mlp_cfg,
            )
            record(year, "control", seed, control)
        print(f"MLP convergence audit fold {year} complete", flush=True)
    audit = pd.DataFrame(rows)
    expected = classical["fold_year"].nunique() * (2 + len(seeds))
    if len(audit) != expected:
        raise RuntimeError("MLP convergence audit did not cover every frozen fit")
    summary = {}
    for task, group in audit.groupby("task", sort=False):
        summary[task] = {
            "fits": len(group),
            "converged": int(group["converged"].sum()),
            "hit_iteration_cap": int((group["n_iter"] >= group["max_iter"]).sum()),
            "convergence_warnings": int(group["convergence_warning"].sum()),
        }
    result = {
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "embeddings_sha256": _sha256(EMBEDDINGS_PATH),
        "classical_sha256": _sha256(CLASSICAL_PATH),
        "optimizer": {
            "hidden_units": int(mlp_cfg["hidden_units"]),
            "activation": mlp_cfg["activation"],
            "solver": mlp_cfg["solver"],
            "l2_alpha": float(mlp_cfg["l2_alpha"]),
            "max_iter": int(mlp_cfg["max_iter"]),
            "seed": int(mlp_cfg["seed"]),
        },
        "fits": len(audit),
        "summary": summary,
        "interpretation": (
            "Termination audit only; scores were not changed and the optimizer cap was not increased post-result."
        ),
    }
    audit.to_parquet(MLP_CONVERGENCE_PATH, index=False)
    MLP_CONVERGENCE_METRICS_PATH.write_text(json.dumps(result, indent=2) + "\n")
    return result


def refresh_result_audit(protocol: dict | None = None) -> dict:
    """Apply reporting/audit corrections without changing any stored score."""
    protocol = protocol or load_protocol()
    if not METRICS_PATH.exists() or not FORECASTS_PATH.exists() or not CONTROLS_PATH.exists():
        raise FileNotFoundError("run the latent probe before refreshing its result audit")
    metrics = json.loads(METRICS_PATH.read_text())
    actual = pd.read_parquet(FORECASTS_PATH).sort_index()
    controls = pd.read_parquet(CONTROLS_PATH)
    metrics["final_reviewed_protocol_sha256"] = _sha256(PROTOCOL_PATH)
    metrics["protocol_lineage"] = {
        "score_production_sha256": metrics["protocol_sha256"],
        "final_reviewed_sha256": _sha256(PROTOCOL_PATH),
        "post_score_change_scope": (
            "documentation and audit corrections, including noise preprocessing wording; "
            "latent labels, fits, scores, seeds, and controls were unchanged"
        ),
    }
    for rung, item in metrics["rungs"].items():
        latent_phases = phase_ranking_metrics(
            actual,
            score_column=f"p_latent_{rung}",
            horizon=5,
            top_fraction=.1,
        )
        augmented_phases = phase_ranking_metrics(
            actual,
            score_column=f"p_augmented_{rung}",
            horizon=5,
            top_fraction=.1,
        )
        item["phase_dispersion"] = {
            "latent": phase_dispersion(latent_phases),
            "augmented": phase_dispersion(augmented_phases),
        }
        control_auc = [entry["auc"] for entry in item["control"]["by_seed"]]
        frozen = item.pop(
            "representation_evidence",
            item.get("frozen_heuristic_pass", False),
        )
        item["frozen_heuristic_pass"] = bool(frozen)
        item.update(randomization_evidence(item["latent_actual"]["auc"], control_auc, alpha=.05))
    metrics["target_interpretation"] = (
        "The 13.2% outcome is recurrent five-session threshold proximity among calm origins, "
        "not a count of rare independent regime breaks."
    )
    metrics["core_interpretation"] = (
        "All latent-only rungs clear the frozen descriptive selectivity heuristic, but none "
        "adds usable ranking over the RV-history benchmark on common rows."
    )
    if MLP_CONVERGENCE_METRICS_PATH.exists():
        metrics["mlp_convergence_audit"] = json.loads(
            MLP_CONVERGENCE_METRICS_PATH.read_text()
        )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
    _write_report(metrics)
    _write_result_audit_report(metrics)
    return metrics


def verify_results(protocol: dict | None = None) -> dict:
    """Hash-check and recompute the persisted latent scorecard from artifacts."""
    protocol = protocol or load_protocol()
    metrics = json.loads(METRICS_PATH.read_text())
    actual = pd.read_parquet(FORECASTS_PATH).sort_index()
    controls = pd.read_parquet(CONTROLS_PATH)
    classical = pd.read_parquet(CLASSICAL_PATH).sort_index()
    embeddings = pd.read_parquet(EMBEDDINGS_PATH).sort_index()
    selected = pd.read_parquet(SELECTED_PATH)
    checks: dict[str, bool] = {}
    checks["final_reviewed_protocol_hash"] = (
        metrics.get("final_reviewed_protocol_sha256") == _sha256(PROTOCOL_PATH)
    )
    checks["production_protocol_hash_preserved"] = bool(
        metrics.get("protocol_lineage", {}).get("score_production_sha256")
        == metrics.get("protocol_sha256")
    )
    checks["embedding_hash"] = metrics["embeddings_sha256"] == _sha256(EMBEDDINGS_PATH)
    checks["classical_hash"] = metrics["classical_sha256"] == _sha256(CLASSICAL_PATH)
    checks["embedding_shape"] = embeddings.shape == (6668, EXPECTED_DIMENSION)
    checks["common_rows"] = len(actual) == len(classical) == int(metrics["origins"])
    checks["clean_fence"] = bool(
        len(actual) and actual.index.max() < pd.Timestamp(protocol["source"]["clean_start"])
    )
    checks["control_coverage"] = bool(
        len(controls) == len(actual) * 10 * 5
        and controls.groupby(["rung", "seed"]).size().eq(len(actual)).all()
    )
    history = base.load_locked_history(protocol)
    y = history["log_rv"].dropna().sort_index()
    features = base.build_history_features(y)
    reconstructed_selected = 0
    for year, classical_fold in classical.groupby("fold_year", sort=True):
        year = int(year)
        classical_fold = classical_fold.sort_index()
        cutoff = pd.DatetimeIndex(classical_fold["cutoff"].unique())[0]
        train, _test = build_annual_fold_tables(
            y=y,
            features=features,
            embeddings=embeddings,
            classical_fold=classical_fold,
            cutoff=cutoff,
            horizon=5,
            quantile=.8,
        )
        latent_X = _matrix(train, latent_columns())
        train_y = train["event"].astype(int).to_numpy()
        effects = standardized_mean_difference(latent_X, train_y)
        for k in (1, 5, 10):
            dimensions = select_sparse_dimensions(latent_X, train_y, k=k)
            saved = selected.loc[
                (selected["fold_year"] == year) & (selected["rung"] == f"sparse_k{k}")
            ].sort_values("rank")
            np.testing.assert_array_equal(saved["dimension"].to_numpy(dtype=int), dimensions)
            np.testing.assert_allclose(
                saved["standardized_mean_difference"].to_numpy(dtype=float),
                effects[dimensions],
                rtol=0,
                atol=1e-12,
            )
            reconstructed_selected += len(saved)
    checks["sparse_training_only_reconstruction"] = bool(
        reconstructed_selected == len(selected) == 24 * (1 + 5 + 10)
    )
    pd.testing.assert_index_equal(
        pd.DatetimeIndex(actual.index), pd.DatetimeIndex(classical.index), check_names=False
    )
    for column in (
        "event",
        "target_end",
        "trigger_date",
        "threshold",
        "p_benchmark",
        "p_hmm_platt",
        "p_hmm_augmented",
    ):
        pd.testing.assert_series_equal(
            actual[column].reset_index(drop=True),
            classical[column].reset_index(drop=True),
            check_names=False,
            check_dtype=False,
        )
    metric_checks = []
    for rung, item in metrics["rungs"].items():
        latent = _score_summary(actual, f"p_latent_{rung}", protocol)
        augmented = _score_summary(actual, f"p_augmented_{rung}", protocol)
        np.testing.assert_allclose(
            [latent["auc"], latent["top_decile_lift"]],
            [item["latent_actual"]["auc"], item["latent_actual"]["top_decile_lift"]],
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            [augmented["auc"], augmented["top_decile_lift"]],
            [item["augmented_actual"]["auc"], item["augmented_actual"]["top_decile_lift"]],
            rtol=0,
            atol=1e-12,
        )
        control_scores = []
        for seed in protocol["latent_probe"]["control_task"]["seeds"]:
            group = controls.loc[
                (controls["rung"] == rung) & (controls["seed"] == int(seed))
            ].sort_values("origin")
            pd.testing.assert_index_equal(
                pd.DatetimeIndex(group["origin"]), pd.DatetimeIndex(actual.index), check_names=False
            )
            control_scores.append(
                _score_summary(group, "p_control", protocol, event_column="event_control")
            )
        control_auc = np.asarray([value["auc"] for value in control_scores])
        control_lift = np.asarray([value["top_decile_lift"] for value in control_scores])
        np.testing.assert_allclose(
            [np.median(control_auc), np.median(control_lift)],
            [item["control"]["auc_median"], item["control"]["top_decile_lift_median"]],
            rtol=0,
            atol=1e-12,
        )
        evidence = randomization_evidence(latent["auc"], control_auc, alpha=.05)
        if evidence["exact_randomization_p"] != item["exact_randomization_p"]:
            raise AssertionError(f"{rung} exact randomization p did not recompute")
        if item["formal_evidence"]:
            raise AssertionError("ten controls cannot produce formal 5% evidence")
        metric_checks.append(rung)
    checks["metrics_recomputed"] = len(metric_checks) == 5
    if MLP_CONVERGENCE_PATH.exists() and MLP_CONVERGENCE_METRICS_PATH.exists():
        audit = pd.read_parquet(MLP_CONVERGENCE_PATH)
        convergence = json.loads(MLP_CONVERGENCE_METRICS_PATH.read_text())
        checks["mlp_convergence_coverage"] = bool(
            len(audit) == 24 * 12 and len(audit) == convergence["fits"]
        )
    else:
        checks["mlp_convergence_coverage"] = False
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "actual_rows": len(actual),
        "control_rows": len(controls),
        "embedding_shape": list(embeddings.shape),
        "recomputed_rungs": metric_checks,
    }
    VERIFICATION_PATH.write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "PASS":
        raise RuntimeError(f"latent verification failed: {checks}")
    return result


def run_probe_study(protocol: dict | None = None, *, overwrite: bool = False) -> dict:
    protocol = protocol or load_protocol()
    if METRICS_PATH.exists() or WORK_DIR.exists():
        if not overwrite:
            raise FileExistsError(
                "latent result or provisional work exists; rerun with --overwrite to archive it first"
            )
        archive = _archive_previous_outputs()
        if archive is not None:
            print(f"archived previous latent artifacts at {archive}", flush=True)
    history = base.load_locked_history(protocol)
    embeddings = load_embeddings(protocol)
    classical = pd.read_parquet(CLASSICAL_PATH).sort_index()
    if (classical.index >= pd.Timestamp(protocol["source"]["clean_start"])).any():
        raise RuntimeError("classical common rows entered the sealed clean window")
    y = history["log_rv"].dropna().sort_index()
    features = base.build_history_features(y)
    target_cfg = protocol["tail_target"]
    horizon = int(target_cfg["horizon_sessions"])
    quantile = float(target_cfg["stress_quantile"])
    latent_cfg = protocol["latent_probe"]
    ladder = latent_cfg["probe_ladder"]
    ridge = float(ladder["full_ridge"]["ridge"])
    mlp_cfg = ladder["small_mlp"]
    seeds = [int(seed) for seed in latent_cfg["control_task"]["seeds"]]
    feature_columns = ["log_rv_d", "log_rv_w", "log_rv_m"]
    z_columns = latent_columns()
    actual_parts: list[pd.DataFrame] = []
    control_parts: list[pd.DataFrame] = []
    selected_parts: list[pd.DataFrame] = []
    fold_meta: list[dict] = []
    WORK_DIR.mkdir(parents=True, exist_ok=False)
    progress_path = WORK_DIR / "progress.json"
    progress_path.write_text(json.dumps({"status": "started", "completed_years": []}, indent=2) + "\n")

    for year, classical_fold in classical.groupby("fold_year", sort=True):
        year = int(year)
        classical_fold = classical_fold.sort_index()
        cutoffs = pd.DatetimeIndex(classical_fold["cutoff"].unique())
        if len(cutoffs) != 1:
            raise RuntimeError(f"fold {year} does not have one frozen cutoff")
        cutoff = cutoffs[0]
        _strict_validate_classical_fold(
            y,
            classical_fold,
            cutoff=cutoff,
            horizon=horizon,
            quantile=quantile,
        )
        train, test = build_annual_fold_tables(
            y=y,
            features=features,
            embeddings=embeddings,
            classical_fold=classical_fold,
            cutoff=cutoff,
            horizon=horizon,
            quantile=quantile,
        )
        actual_y = train["event"].astype(int).to_numpy()
        latent_train_all = _matrix(train, z_columns)
        latent_test_all = _matrix(test, z_columns)
        rv_train = _matrix(train, feature_columns)
        rv_test = _matrix(test, feature_columns)
        controls = {
            seed: markov_surrogate_path(
                actual_y,
                train_length=len(train),
                test_length=len(test),
                seed=seed,
            )
            for seed in seeds
        }
        control_episodes = {
            seed: assign_control_episodes(
                path.test,
                pd.DatetimeIndex(test.index),
                np.full(len(test), year),
                y.index,
                max_gap_sessions=horizon,
            )
            for seed, path in controls.items()
        }
        fold_actual = test[
            [
                "target_end",
                "threshold",
                "event",
                "trigger_date",
                "fold_year",
                "cutoff",
                "p_benchmark",
                "p_hmm_platt",
                "p_hmm_augmented",
            ]
        ].copy()
        fold_controls: list[pd.DataFrame] = []
        fold_selected: list[dict] = []
        for rung, k in _rung_specs(protocol):
            if k is None:
                selected_actual = np.arange(EXPECTED_DIMENSION)
            else:
                selected_actual = select_sparse_dimensions(latent_train_all, actual_y, k=k)
                effects = standardized_mean_difference(latent_train_all, actual_y)
                for rank, dimension in enumerate(selected_actual, start=1):
                    fold_selected.append(
                        {
                            "fold_year": year,
                            "rung": rung,
                            "k": int(k),
                            "rank": rank,
                            "dimension": int(dimension),
                            "standardized_mean_difference": float(effects[dimension]),
                        }
                    )
            latent_train = latent_train_all[:, selected_actual]
            latent_test = latent_test_all[:, selected_actual]
            model_kind = "small_mlp" if rung == "small_mlp" else rung
            actual_latent = fit_probe(
                model_kind,
                latent_train,
                actual_y,
                ridge=ridge,
                seed=int(mlp_cfg.get("seed", 42)),
                mlp_config=mlp_cfg,
            )
            augmented_train = np.column_stack([rv_train, latent_train])
            augmented_test = np.column_stack([rv_test, latent_test])
            actual_augmented = fit_probe(
                model_kind,
                augmented_train,
                actual_y,
                ridge=ridge,
                seed=int(mlp_cfg.get("seed", 42)),
                mlp_config=mlp_cfg,
            )
            fold_actual[f"p_latent_{rung}"] = actual_latent.predict(latent_test)
            fold_actual[f"p_augmented_{rung}"] = actual_augmented.predict(augmented_test)

            for seed, path in controls.items():
                if k is None:
                    selected_control = np.arange(EXPECTED_DIMENSION)
                else:
                    selected_control = select_sparse_dimensions(
                        latent_train_all, path.train, k=int(k)
                    )
                control_train = latent_train_all[:, selected_control]
                control_test = latent_test_all[:, selected_control]
                control_probe = fit_probe(
                    model_kind,
                    control_train,
                    path.train,
                    ridge=ridge,
                    seed=int(mlp_cfg.get("seed", 42)),
                    mlp_config=mlp_cfg,
                )
                episode_values = [
                    pd.NA if pd.isna(value) else f"{year}:{int(value)}"
                    for value in control_episodes[seed].to_numpy()
                ]
                fold_controls.append(
                    pd.DataFrame(
                        {
                            "origin": test.index,
                            "fold_year": year,
                            "rung": rung,
                            "seed": seed,
                            "event_control": path.test.astype(bool),
                            "control_episode": pd.array(episode_values, dtype="string"),
                            "p_control": control_probe.predict(control_test),
                        }
                    )
                )
        fold_control = pd.concat(fold_controls, ignore_index=True)
        fold_selected_frame = pd.DataFrame(fold_selected)
        fold_actual.to_parquet(WORK_DIR / f"actual_{year}.parquet")
        fold_control.to_parquet(WORK_DIR / f"controls_{year}.parquet", index=False)
        fold_selected_frame.to_parquet(WORK_DIR / f"selected_{year}.parquet", index=False)
        actual_parts.append(fold_actual)
        control_parts.append(fold_control)
        selected_parts.append(fold_selected_frame)
        fold_meta.append(
            {
                "fold_year": year,
                "cutoff": str(cutoff.date()),
                "train_rows": len(train),
                "test_rows": len(test),
                "train_event_rate": float(train["event"].mean()),
                "test_event_rate": float(test["event"].mean()),
            }
        )
        progress_path.write_text(
            json.dumps(
                {"status": "running", "completed_years": [item["fold_year"] for item in fold_meta]},
                indent=2,
            )
            + "\n"
        )
        print(f"latent probe fold {year} complete ({len(test):,} common rows)", flush=True)

    actual = pd.concat(actual_parts).sort_index()
    controls_long = pd.concat(control_parts, ignore_index=True)
    selected = pd.concat(selected_parts, ignore_index=True)
    pd.testing.assert_index_equal(pd.DatetimeIndex(actual.index), pd.DatetimeIndex(classical.index))
    if len(controls_long) != len(actual) * len(seeds) * len(_rung_specs(protocol)):
        raise RuntimeError("control output does not cover every common row, seed, and rung")
    actual["phase"] = np.arange(len(actual)) % horizon
    if "ranking_phase" in classical:
        frozen_phase = classical.loc[actual.index, "ranking_phase"].to_numpy(dtype=int)
        if not np.array_equal(actual["phase"].to_numpy(dtype=int), frozen_phase):
            raise RuntimeError("latent phase assignment differs from the frozen classical scoreboard")
    actual["episode"] = assign_transition_episodes(
        actual["trigger_date"], y.index, max_gap_sessions=horizon
    )
    wide = actual.copy()
    for (rung, seed), group in controls_long.groupby(["rung", "seed"], sort=False):
        group = group.set_index("origin").loc[actual.index]
        wide[f"event_control_{rung}_{seed}"] = group["event_control"].to_numpy(dtype=bool)
        wide[f"p_control_{rung}_{seed}"] = group["p_control"].to_numpy(dtype=float)
        wide[f"episode_control_{rung}_{seed}"] = pd.array(
            group["control_episode"].to_numpy(), dtype="string"
        )

    phase_rows = []
    metrics: dict = {
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "embeddings_sha256": _sha256(EMBEDDINGS_PATH),
        "classical_sha256": _sha256(CLASSICAL_PATH),
        "origins": len(actual),
        "first_origin": str(actual.index.min().date()),
        "last_origin": str(actual.index.max().date()),
        "event_rate": float(actual["event"].mean()),
        "transition_episodes": int(actual["episode"].nunique()),
        "folds": fold_meta,
        "rungs": {},
    }
    for model, score_column in (
        ("benchmark", "p_benchmark"),
        ("hmm_platt", "p_hmm_platt"),
        ("hmm_augmented", "p_hmm_augmented"),
    ):
        phases = phase_ranking_metrics(
            actual,
            score_column=score_column,
            horizon=horizon,
            top_fraction=float(protocol["ranking_scoreboard"]["top_fraction"]),
        )
        phases["kind"] = "actual"
        phases["model"] = model
        phases["seed"] = pd.NA
        phase_rows.append(phases)
    for rung, _k in _rung_specs(protocol):
        latent_score = f"p_latent_{rung}"
        augmented_score = f"p_augmented_{rung}"
        actual_latent_summary = _score_summary(actual, latent_score, protocol)
        actual_augmented_summary = _score_summary(actual, augmented_score, protocol)
        rung_phase_dispersion = {}
        for kind, score in (("latent", latent_score), ("augmented", augmented_score)):
            phases = phase_ranking_metrics(
                actual,
                score_column=score,
                horizon=horizon,
                top_fraction=float(protocol["ranking_scoreboard"]["top_fraction"]),
            )
            phases["kind"] = "actual"
            phases["model"] = f"{kind}_{rung}"
            phases["seed"] = pd.NA
            phase_rows.append(phases)
            rung_phase_dispersion[kind] = phase_dispersion(phases)
        control_summaries = []
        control_specs = []
        for seed in seeds:
            event_column = f"event_control_{rung}_{seed}"
            score_column = f"p_control_{rung}_{seed}"
            episode_column = f"episode_control_{rung}_{seed}"
            summary = _score_summary(wide, score_column, protocol, event_column=event_column)
            summary["seed"] = seed
            control_summaries.append(summary)
            control_specs.append((event_column, score_column, episode_column, seed))
            phases = phase_ranking_metrics(
                wide,
                score_column=score_column,
                event_column=event_column,
                horizon=horizon,
                top_fraction=float(protocol["ranking_scoreboard"]["top_fraction"]),
            )
            phases["kind"] = "control"
            phases["model"] = rung
            phases["seed"] = seed
            phase_rows.append(phases)
        control_auc = np.asarray([item["auc"] for item in control_summaries])
        control_lift = np.asarray([item["top_decile_lift"] for item in control_summaries])
        selectivity = {
            "auc": actual_latent_summary["auc"] - float(np.median(control_auc)),
            "top_decile_lift": actual_latent_summary["top_decile_lift"]
            - float(np.median(control_lift)),
        }
        benchmark = _score_summary(actual, "p_benchmark", protocol)
        hmm_augmented = _score_summary(actual, "p_hmm_augmented", protocol)
        deltas = {
            "vs_benchmark": {
                metric: actual_augmented_summary[metric] - benchmark[metric]
                for metric in ("auc", "top_decile_lift")
            },
            "vs_hmm_augmented": {
                metric: actual_augmented_summary[metric] - hmm_augmented[metric]
                for metric in ("auc", "top_decile_lift")
            },
        }
        jackknife = {
            "selectivity": {
                metric: clustered_selectivity_interval(
                    wide,
                    protocol,
                    candidate_score=latent_score,
                    controls=control_specs,
                    metric=metric,
                )
                for metric in ("auc", "top_decile_lift")
            },
            "vs_benchmark": {
                metric: _jackknife_comparison(
                    wide,
                    protocol,
                    candidate_score=augmented_score,
                    baseline_score="p_benchmark",
                    metric=metric,
                )
                for metric in ("auc", "top_decile_lift")
            },
            "vs_hmm_augmented": {
                metric: _jackknife_comparison(
                    wide,
                    protocol,
                    candidate_score=augmented_score,
                    baseline_score="p_hmm_augmented",
                    metric=metric,
                )
                for metric in ("auc", "top_decile_lift")
            },
        }
        frozen_heuristic_pass = bool(
            actual_latent_summary["auc"] > float(np.quantile(control_auc, .95))
            and jackknife["selectivity"]["auc"]["lower"] > 0
        )
        formal = randomization_evidence(
            actual_latent_summary["auc"], control_auc, alpha=.05
        )
        metrics["rungs"][rung] = {
            "latent_actual": actual_latent_summary,
            "control": {
                "draws": len(seeds),
                "auc_median": float(np.median(control_auc)),
                "auc_95th_percentile": float(np.quantile(control_auc, .95)),
                "top_decile_lift_median": float(np.median(control_lift)),
                "by_seed": control_summaries,
            },
            "selectivity": selectivity,
            "augmented_actual": actual_augmented_summary,
            "deltas": deltas,
            "jackknife": jackknife,
            "phase_dispersion": rung_phase_dispersion,
            "frozen_heuristic_pass": frozen_heuristic_pass,
            **formal,
        }
    metrics["dimension_stability"] = _dimension_stability(selected)
    phases_all = pd.concat(phase_rows, ignore_index=True)
    actual.to_parquet(FORECASTS_PATH)
    controls_long.to_parquet(CONTROLS_PATH, index=False)
    selected.to_parquet(SELECTED_PATH, index=False)
    phases_all.to_parquet(PHASE_PATH, index=False)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
    _write_report(metrics)
    progress_path.write_text(
        json.dumps({"status": "complete", "completed_years": [item["fold_year"] for item in fold_meta]}, indent=2)
        + "\n"
    )
    return metrics


def _fmt_ci(item: dict, digits: int = 3) -> str:
    return f"[{item['lower']:.{digits}f}, {item['upper']:.{digits}f}]"


def _write_report(metrics: dict) -> None:
    lines = [
        "# TiRex-2 latent probe: full coordinates, controls, and episode uncertainty",
        "",
        "This diagnostic follows the frozen latent contract in `representation_study.yaml`.",
        f"Scores were produced under protocol SHA `{metrics['protocol_sha256'][:12]}…`; the",
        f"final reviewed file is `{metrics.get('final_reviewed_protocol_sha256', metrics['protocol_sha256'])[:12]}…`.",
        "Post-score changes were documentation/audit corrections and did not change latent",
        "labels, fits, scores, seeds, or controls. No PCA or",
        "post-result capacity selection is used: each representation is the complete",
        "512-coordinate `stack_out_norm` state at zero-based token 63 from pinned TiRex-2",
        "0.2.1 and checkpoint revision `05e5b26`.",
        "",
        f"Common scored origins: **{metrics['origins']:,}**, {metrics['first_origin']} through",
        f"{metrics['last_origin']}; event rate **{metrics['event_rate']:.1%}** across",
        f"**{metrics['transition_episodes']} positive-trigger episodes**. This 13.2% target is",
        "recurrent five-session threshold proximity among calm origins, not a count of rare",
        "independent regime breaks.",
        "",
        "**Result:** every latent-only rung separates the actual label from its ten",
        "capacity-matched controls on the frozen descriptive scorecard. None adds usable",
        "ranking over the RV-history benchmark. Sparse k=1 essentially ties it, k=5 has an",
        "interval including zero, k=10 is worse, and full ridge and the fixed MLP are",
        "materially worse.",
        "",
        "## Frozen probe ladder",
        "",
        "| rung | latent AUC | control median / 95th | AUC selectivity (episode CI) | latent lift / control median | augmented AUC | Δ vs RV-history (CI) | Δ vs HMM+RV-history (CI) | descriptive heuristic met | exact p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for rung, item in metrics["rungs"].items():
        latent = item["latent_actual"]
        control = item["control"]
        aug = item["augmented_actual"]
        select_jk = item["jackknife"]["selectivity"]["auc"]
        benchmark_jk = item["jackknife"]["vs_benchmark"]["auc"]
        hmm_jk = item["jackknife"]["vs_hmm_augmented"]["auc"]
        lines.append(
            f"| {rung} | {latent['auc']:.3f} | {control['auc_median']:.3f} / "
            f"{control['auc_95th_percentile']:.3f} | {item['selectivity']['auc']:+.3f} "
            f"{_fmt_ci(select_jk)} | {latent['top_decile_lift']:.2f}× / "
            f"{control['top_decile_lift_median']:.2f}× | {aug['auc']:.3f} | "
            f"{item['deltas']['vs_benchmark']['auc']:+.3f} {_fmt_ci(benchmark_jk)} | "
            f"{item['deltas']['vs_hmm_augmented']['auc']:+.3f} {_fmt_ci(hmm_jk)} | "
            f"{'yes' if item['frozen_heuristic_pass'] else 'no'} | "
            f"{item['exact_randomization_p']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The frozen heuristic is **not 5% randomization evidence**. With ten controls,",
            "the smallest exact corrected Monte Carlo p-value is `(0+1)/(10+1) = 0.0909`.",
            "`formal_evidence` is therefore false for every rung and selectivity is descriptive.",
            "No controls were added after results were seen.",
            "",
            "Actual-trigger and each control-seed episode have separate jackknife variance",
            "components. A control positive origin is its proxy trigger, clustered within its",
            "annual fold because the training-estimated Markov path resets yearly. The interval",
            "conditions on all negative origins and captures positive-episode influence; it is",
            "not a full serial-score or negative-origin uncertainty estimator. The exact method",
            "was recorded before aggregate output in `latent_probe_uncertainty_method.md`.",
            "",
            "## Five-phase range",
            "",
            "| rung | latent AUC min–max | latent lift min–max | augmented AUC min–max | augmented lift min–max |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for rung, item in metrics["rungs"].items():
        latent = item["phase_dispersion"]["latent"]
        augmented = item["phase_dispersion"]["augmented"]
        lines.append(
            f"| {rung} | {latent['auc']['min']:.3f}–{latent['auc']['max']:.3f} | "
            f"{latent['top_decile_lift']['min']:.2f}×–{latent['top_decile_lift']['max']:.2f}× | "
            f"{augmented['auc']['min']:.3f}–{augmented['auc']['max']:.3f} | "
            f"{augmented['top_decile_lift']['min']:.2f}×–{augmented['top_decile_lift']['max']:.2f}× |"
        )
    lines.extend(
        [
            "",
            "## Top-decile lift and incremental ranking",
            "",
            "| rung | lift selectivity (episode CI) | augmented lift | Δ vs RV-history (CI) | Δ vs HMM+RV-history (CI) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for rung, item in metrics["rungs"].items():
        select_jk = item["jackknife"]["selectivity"]["top_decile_lift"]
        benchmark_jk = item["jackknife"]["vs_benchmark"]["top_decile_lift"]
        hmm_jk = item["jackknife"]["vs_hmm_augmented"]["top_decile_lift"]
        lines.append(
            f"| {rung} | {item['selectivity']['top_decile_lift']:+.3f}× "
            f"{_fmt_ci(select_jk)} | {item['augmented_actual']['top_decile_lift']:.3f}× | "
            f"{item['deltas']['vs_benchmark']['top_decile_lift']:+.3f}× {_fmt_ci(benchmark_jk)} | "
            f"{item['deltas']['vs_hmm_augmented']['top_decile_lift']:+.3f}× {_fmt_ci(hmm_jk)} |"
        )
    first_rung = next(iter(metrics["rungs"].values()))
    episode_counts = first_rung["jackknife"]["selectivity"]["auc"]["control_episodes_by_seed"]
    lines.extend(
        [
            "",
            "Control episode counts by seed: "
            + ", ".join(f"{seed}: {count}" for seed, count in episode_counts.items())
            + ".",
            "",
            "## Sparse-coordinate stability",
            "",
            "| rung | distinct coordinates | mean adjacent-fold Jaccard | most recurrent coordinates (fold count) |",
            "|---|---:|---:|---|",
        ]
    )
    for rung, item in metrics["dimension_stability"].items():
        top = ", ".join(
            f"z{entry['dimension']:03d} ({entry['folds']})" for entry in item["top_dimensions"]
        )
        lines.append(
            f"| {rung} | {item['unique_dimensions']} | {item['mean_consecutive_jaccard']:.3f} | {top} |"
        )
    lines.extend(
        [
            "",
            "Coordinates are selected only from each fold's completed training labels by",
            "absolute standardized event/non-event mean difference. Test labels and embeddings",
            "never enter selection; the selected coordinates are scored on that year's disjoint",
            "held-out common rows. All yearly identities and signed effects are retained in",
            "`data/representation_study/latent_selected_dimensions.parquet`.",
        ]
    )
    convergence = metrics.get("mlp_convergence_audit")
    if convergence:
        lines.extend(
            [
                "",
                "## Fixed MLP optimizer audit",
                "",
                "| task | fits | converged before cap | hit 500-iteration cap | warnings |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for task, values in convergence["summary"].items():
            lines.append(
                f"| {task} | {values['fits']} | {values['converged']} | "
                f"{values['hit_iteration_cap']} | {values['convergence_warnings']} |"
            )
        lines.extend(
            [
                "",
                "The cap was not increased post-result. Capped fits are fixed-optimizer",
                "endpoints, not evidence that nonlinear capacity was exhausted.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- The ten-control selectivity result is descriptive, not formal 5% evidence.",
            "- Positive-episode influence is clustered; negative origins and residual serial",
            "  score uncertainty are held fixed rather than fully resampled.",
            "- Augmented probes use exactly the frozen benchmark/HMM common rows.",
            "- TiRex-2 may have encountered market histories during pretraining, so this is a",
            "  causal-origin diagnostic rather than a pristine corpus holdout.",
            "- The fixed MLP does not license wider/deeper post-result searches.",
        ]
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def _write_result_audit_report(metrics: dict) -> None:
    convergence = metrics.get("mlp_convergence_audit", {}).get("summary", {})
    lines = [
        "# Latent-probe post-result audit corrections",
        "",
        "No label, score, probe, coordinate selection, seed, or control draw changed.",
        "",
        "- Ten controls imply a minimum exact corrected randomization p-value of 1/11",
        "  (0.0909). The empirical percentile flag is retained only as",
        "  `frozen_heuristic_pass`; formal 5% evidence is false.",
        "- Episode intervals condition on negative origins and measure positive-episode",
        "  influence, including one variance component per control seed.",
        "- The 13.2% event is recurrent threshold proximity, not independent rare breaks.",
        "- Five-phase minima and maxima are reported.",
        "- Sparse dimensions reconstruct from completed annual training labels only and",
        "  score disjoint held-out rows; yearly coordinate identities are retained.",
        "- MLP termination is audited without raising its frozen 500-iteration cap.",
        "- Chunk reuse now requires a run signature and per-chunk hashes; legacy chunks",
        "  may be sealed only after exact final-matrix reconstruction.",
        "",
        "## MLP convergence",
        "",
    ]
    if convergence:
        for task, value in convergence.items():
            lines.append(
                f"- {task}: {value['converged']}/{value['fits']} converged before the cap; "
                f"{value['hit_iteration_cap']} hit the cap."
            )
    else:
        lines.append("Convergence audit not yet attached.")
    AUDIT_REPORT_PATH.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "validate",
            "extract",
            "seal-chunks",
            "run",
            "audit-mlp",
            "refresh",
            "verify",
            "all",
        ],
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    protocol = load_protocol()
    if args.command == "validate":
        assert_runtime_pin(protocol)
        print("latent probe protocol and runtime pins valid")
    if args.command in ("extract", "all"):
        print(
            json.dumps(
                extract_embeddings(
                    protocol,
                    batch_size=int(args.batch_size),
                    chunk_rows=int(args.chunk_rows),
                ),
                indent=2,
            )
        )
    if args.command == "seal-chunks":
        print(json.dumps(seal_existing_chunks(protocol), indent=2))
    if args.command in ("run", "all"):
        result = run_probe_study(protocol, overwrite=bool(args.overwrite))
        print(json.dumps(result, indent=2))
    if args.command == "audit-mlp":
        print(json.dumps(audit_mlp_convergence(protocol), indent=2))
    if args.command == "refresh":
        print(json.dumps(refresh_result_audit(protocol), indent=2))
    if args.command == "verify":
        print(json.dumps(verify_results(protocol), indent=2))


if __name__ == "__main__":
    main()
