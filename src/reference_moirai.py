"""Pinned Moirai 2.0 summary extraction; no calibration, targets, or scores.

Run with the separate data/model_memory_reference/moirai2_env interpreter.
The default command performs a synthetic checkpoint check only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import pathlib
import subprocess
import sys
import tarfile
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOCAL = ROOT / "data/model_memory_reference"
REPORT = ROOT / "reports/model_memory_study"
PLAN = REPORT / "MOIRAI_PLAN.md"
TESTS = ROOT / "tests/test_reference_moirai.py"
INPUT = ROOT / "data/orthogonal_round2/features.parquet"
CHECKPOINT = LOCAL / "moirai2_checkpoint"
CODE_REVISION = "cfd46d4510ed8896f263116f32928eede05b0a75"
MODEL_REVISION = "30f43ff08c8494f4943ae1521e9d4e94a0fbb389"
ARCHIVE = LOCAL / f"vendor/uni2ts-{CODE_REVISION}.tar.gz"
EXPECTED_HASHES = {
    "model.safetensors": "fb5652a3db8ea572606221b7cb1e77bb8962b168e4d4cc752cf31ceb04074669",
    "config.json": "6b74b03c8ec199fabc352c0203465958142ca468183da68549652734836f853d",
    "code_archive": "e1de1cbff2b6e131e1988f4f9dddce51e0801bb41c87d7e7d8609a783dd89f18",
}
CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 5
ORIGIN_START = "2010-01-04"
ORIGIN_END = "2025-10-10"
COLUMNS = ("moirai_log_h1", "moirai_log_h5")
SEED = 20260906
BATCH_SIZE = 32
CPU_THREADS = 2


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def require_hash(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Pinned artifact hash differs: {path}")


def dump(path, data):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def _dates(index, label):
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"{label} must be actual DatetimeIndex sessions")
    if (len(index) == 0 or index.has_duplicates or not index.is_monotonic_increasing
            or index.isna().any() or index.tz is not None
            or not index.equals(index.normalize())):
        raise ValueError(f"{label} must be nonempty, unique, chronological daily sessions")
    return index


def causal_windows(rv, origins):
    """Construct exact trailing contexts without inspecting values after origins."""
    if not isinstance(rv, pd.Series):
        raise ValueError("Variance input must be a single series")
    sessions = _dates(rv.index, "Variance index")
    origins = _dates(origins, "Origins")
    positions = sessions.get_indexer(origins)
    if (positions < CONTEXT_LENGTH - 1).any():
        raise ValueError("Unknown origin or insufficient exact context; never pad")
    windows = []
    records = []
    for origin, position in zip(origins, positions):
        first = position - CONTEXT_LENGTH + 1
        window = rv.iloc[first:position + 1].to_numpy(dtype=float)
        if len(window) != CONTEXT_LENGTH or not np.isfinite(window).all() or (window <= 0).any():
            raise ValueError(f"Invalid variance context at {origin}; never impute")
        windows.append(np.log(window))
        records.append({"context_start": sessions[first], "context_end": origin,
                        "context_rows": CONTEXT_LENGTH})
    return np.stack(windows), pd.DataFrame(records, index=origins.rename("origin"))


def quantile_summaries(predictions, quantile_levels):
    """First log median and log arithmetic mean of five exponentiated medians."""
    values = np.asarray(predictions, dtype=float)
    levels = np.asarray(quantile_levels, dtype=float)
    if values.ndim == 4 and values.shape[-1] == 1:
        values = values[..., 0]
    if (values.ndim != 3 or values.shape[2] != PREDICTION_LENGTH
            or levels.ndim != 1 or values.shape[1] != len(levels)
            or len(values) == 0 or not np.isfinite(values).all()
            or not np.isfinite(levels).all() or (levels <= 0).any() or (levels >= 1).any()):
        raise ValueError("Finite univariate batch/quantile/five-step predictions required")
    selected = np.flatnonzero(np.isclose(levels, .5, atol=1e-12, rtol=0))
    if len(selected) != 1 or len(np.unique(levels)) != len(levels):
        raise ValueError("One numeric median and unique quantile levels required")
    median = values[:, selected[0], :]
    maximum = median.max(axis=1)
    log_average = maximum + np.log(np.exp(median - maximum[:, None]).mean(axis=1))
    return np.column_stack([median[:, 0], log_average])


def extract_with_predictor(rv, origins, predict, quantile_levels, batch_size=BATCH_SIZE):
    """Pure extraction with injectable predictor for pre-checkpoint tests."""
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Positive integer batch size required")
    windows, audit = causal_windows(rv, origins)
    blocks = []
    for offset in range(0, len(windows), batch_size):
        block = windows[offset:offset + batch_size]
        values = quantile_summaries(predict(block), quantile_levels)
        if len(values) != len(block):
            raise ValueError("Prediction batch changed row count")
        blocks.append(values)
    return pd.DataFrame(np.concatenate(blocks), index=origins.rename("origin"), columns=COLUMNS), audit


def verify_runtime():
    """Check checkpoint bytes and installed official code against the pinned tar."""
    require_hash(ARCHIVE, EXPECTED_HASHES["code_archive"])
    for name in ("config.json", "model.safetensors"):
        require_hash(CHECKPOINT / name, EXPECTED_HASHES[name])
    if pathlib.Path(sys.prefix).resolve() != (LOCAL / "moirai2_env").resolve():
        raise ValueError("Use the isolated moirai2_env, never the shared research environment")
    package = importlib.util.find_spec("uni2ts")
    if package is None or package.origin is None:
        raise ValueError("Pinned Uni2TS package not installed")
    installed_root = pathlib.Path(package.origin).parent
    prefix = f"uni2ts-{CODE_REVISION}/src/uni2ts/"
    source_hashes = {}
    with tarfile.open(ARCHIVE) as archive:
        for member in archive.getmembers():
            if member.name.startswith(prefix) and member.name.endswith(".py"):
                relative = member.name[len(prefix):]
                if not member.isfile() or ".." in pathlib.PurePosixPath(relative).parts:
                    raise ValueError("Unexpected official source archive member")
                expected = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                require_hash(installed_root / relative, expected)
                source_hashes[relative] = expected
    if not source_hashes:
        raise ValueError("Official source archive had no checked implementation files")
    versions = {name: importlib.metadata.version(name) for name in
                ("uni2ts", "torch", "numpy", "scipy", "pandas", "lightning", "gluonts",
                 "huggingface-hub", "safetensors", "pyarrow", "einops")}
    fixed = {"torch": "2.4.1", "numpy": "1.26.4", "scipy": "1.11.4",
             "pandas": "2.2.3", "lightning": "2.4.0", "gluonts": "0.14.3"}
    if any(versions[name] != value for name, value in fixed.items()):
        raise ValueError("Pinned isolated runtime changed")
    return {"official_code_revision": CODE_REVISION, "official_model_revision": MODEL_REVISION,
            "artifact_sha256": EXPECTED_HASHES, "versions": versions,
            "checked_installed_python_files": len(source_hashes),
            "installed_source_tree_sha256": hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
            "dependency_lock_sha256": sha256(LOCAL / "moirai2_requirements.lock.txt"),
            "install_report_sha256": sha256(LOCAL / "moirai2_install_report.json"),
            "adapter_sha256": sha256(__file__), "tests_sha256": sha256(TESTS), "plan_sha256": sha256(PLAN)}


def load_model():
    provenance = verify_runtime()
    import torch
    from safetensors.torch import load_file
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

    torch.manual_seed(SEED)
    torch.set_num_threads(CPU_THREADS)
    torch.use_deterministic_algorithms(True)
    # Local safetensors loading avoids network access and pickle execution.
    config = json.loads((CHECKPOINT / "config.json").read_text())
    module = Moirai2Module(**config)
    module.load_state_dict(load_file(str(CHECKPOINT / "model.safetensors"), device="cpu"), strict=True)
    model = Moirai2Forecast(prediction_length=PREDICTION_LENGTH, target_dim=1,
                           feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
                           context_length=CONTEXT_LENGTH, module=module).to("cpu").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    provenance.update({"device": "cpu", "dtype": "float32", "context_length": CONTEXT_LENGTH,
                       "prediction_length": PREDICTION_LENGTH, "seed": SEED,
                       "cpu_threads": CPU_THREADS, "batch_size": BATCH_SIZE,
                       "quantile_levels": list(module.quantile_levels),
                       "parameters": sum(p.numel() for p in model.parameters())})

    def predict(windows):
        values = np.asarray(windows, dtype=float)
        if values.ndim != 2 or values.shape[1] != CONTEXT_LENGTH or not np.isfinite(values).all():
            raise ValueError("Only finite exact-length univariate contexts may enter checkpoint")
        with torch.inference_mode():
            return model.predict([row.astype(np.float32) for row in values])

    return predict, np.asarray(module.quantile_levels), provenance


def check_contracts():
    result = subprocess.run([sys.executable, "-m", "unittest", "tests.test_reference_moirai", "-q"],
                            cwd=ROOT, text=True, capture_output=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "moirai_precheck.txt").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    print(result.stderr.strip(), flush=True)


def synthetic_check():
    check_contracts()
    predict, levels, provenance = load_model()
    t = np.arange(CONTEXT_LENGTH)
    windows = np.stack([-9 + .5 * np.sin(t / 17), -8 + .2 * np.cos(t / 11),
                        np.full(CONTEXT_LENGTH, -10.)])
    began = time.monotonic()
    original = predict(windows)
    repeated = predict(windows)
    permutation = np.array([2, 0, 1])
    reordered = predict(windows[permutation])[np.argsort(permutation)]
    singleton = predict(windows[:1])
    summary = quantile_summaries(original, levels)
    np.testing.assert_array_equal(original, repeated)
    np.testing.assert_allclose(original, reordered, atol=2e-5, rtol=1e-6)
    np.testing.assert_allclose(original[:1], singleton, atol=2e-5, rtol=1e-6)
    result = {"status": "PASS", "evidence": "synthetic checkpoint forward only; no historical model extraction or scores",
              "created_utc": datetime.now(UTC).isoformat(), "provenance": provenance,
              "input_shape": list(windows.shape), "output_shape": list(original.shape),
              "summary_shape": list(summary.shape), "repeat_max_abs_error": float(np.max(np.abs(original - repeated))),
              "permutation_max_abs_error": float(np.max(np.abs(original - reordered))),
              "singleton_max_abs_error": float(np.max(np.abs(original[:1] - singleton))),
              "seconds": time.monotonic() - began}
    dump(REPORT / "moirai_synthetic_check.json", result)
    print(json.dumps(result, indent=2), flush=True)


def historical_extract():
    """Called only after the research coordinator authorizes fixed extraction."""
    check_contracts()
    checked = json.loads((REPORT / "moirai_synthetic_check.json").read_text())
    fixed = verify_runtime()
    if checked["status"] != "PASS" or any(checked["provenance"].get(k) != v for k, v in fixed.items()):
        raise ValueError("Synthetic checkpoint check is absent or stale")
    raw = pd.read_parquet(INPUT, columns=["rv_total"], filters=[("date", "<=", pd.Timestamp(ORIGIN_END))])
    rv = raw.rv_total
    origins = rv.index[(rv.index >= pd.Timestamp(ORIGIN_START)) & (rv.index <= pd.Timestamp(ORIGIN_END))]
    predict, levels, provenance = load_model()
    windows, contexts = causal_windows(rv, origins)
    output = LOCAL / "moirai_features.parquet"
    context_path = LOCAL / "moirai_contexts.parquet"
    quantile_path = LOCAL / "moirai_quantiles.npz"
    manifest_path = REPORT / "moirai_extraction_manifest.json"
    if any(path.exists() for path in (output, context_path, quantile_path, manifest_path)):
        raise ValueError("Existing historical Moirai artifacts must not be overwritten")
    source_digest = hashlib.sha256(rv.to_csv(float_format="%.17g", date_format="%Y-%m-%d").encode()).hexdigest()
    protocol_paths = [PLAN, ROOT / "model_memory_study.yaml"]
    protocol_paths.extend(p for p in ROOT.glob("*reference*.yaml") if p not in protocol_paths)
    manifest = {"status": "FROZEN_BEFORE_HISTORICAL_EXTRACTION",
                "created_utc": datetime.now(UTC).isoformat(), "provenance": provenance,
                "origin_start": ORIGIN_START, "origin_end": ORIGIN_END, "origins": len(origins),
                "source": str(INPUT.relative_to(ROOT)), "source_column": "rv_total",
                "bounded_rv_sha256": source_digest,
                "protocol_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in protocol_paths},
                "context_matrix_sha256": hashlib.sha256(windows.astype("<f8").tobytes()).hexdigest()}
    dump(manifest_path, manifest)
    began = time.monotonic()
    parts, raw_parts = [], []
    for offset in range(0, len(windows), BATCH_SIZE):
        raw_prediction = predict(windows[offset:offset + BATCH_SIZE])
        part = quantile_summaries(raw_prediction, levels)
        parts.append(part)
        raw_parts.append(raw_prediction)
        if offset % (BATCH_SIZE * 12) == 0:
            print(f"Moirai causal extraction: {min(offset + BATCH_SIZE, len(windows))}/{len(windows)} contexts", flush=True)
    frame = pd.DataFrame(np.concatenate(parts), index=origins.rename("origin"), columns=COLUMNS)
    for key, value in verify_runtime().items():
        if provenance.get(key) != value:
            raise ValueError("Runtime or adapter changed during extraction")
    for path, expected in manifest["protocol_sha256"].items():
        require_hash(ROOT / path, expected)
    LOCAL.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output)
    contexts.to_parquet(context_path)
    np.savez_compressed(quantile_path, predictions=np.concatenate(raw_parts), quantile_levels=levels,
                        origins=origins.to_numpy(dtype="datetime64[ns]"))
    audit = {"status": "EXTRACTED_NOT_SCORED", "provenance": provenance,
             "origin_start": str(origins[0].date()), "origin_end": str(origins[-1].date()),
             "origins": len(origins), "columns": list(COLUMNS),
             "source": str(INPUT.relative_to(ROOT)), "source_column": "rv_total",
             "bounded_rv_sha256": source_digest,
             "output_sha256": sha256(output), "contexts_sha256": sha256(context_path),
             "quantiles_sha256": sha256(quantile_path), "manifest_sha256": sha256(manifest_path),
             "seconds": time.monotonic() - began,
             "interpretation": "Frozen quantile summaries for train-only calibration; neither column is a conditional mean claim"}
    dump(REPORT / "moirai_extraction_audit.json", audit)
    print(json.dumps({k: audit[k] for k in ("status", "origins", "origin_start", "origin_end", "seconds")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["synthetic-check", "extract"], nargs="?", default="synthetic-check")
    command = parser.parse_args().command
    synthetic_check() if command == "synthetic-check" else historical_extract()


if __name__ == "__main__":
    main()
