"""Reproducible fixed neural forecast run, including pretests and saved states.

The first run was invoked with the equivalent Python orchestration before this
wrapper was persisted. Its existing manifest identifies the actual prefit
source set. This wrapper never rewrites that provenance or overwrites a run.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import torch
import yaml

from . import orthogonal_round2 as prior
from . import reference_xlstm as neural

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data/model_memory_reference"
FIRST_FIT = "2013-02-07"
WARMUP = (("2013-02-01", 496), ("2013-02-04", 497), ("2013-02-05", 498), ("2013-02-06", 499))


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def dump(path, value):
    pathlib.Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_existing(manifest):
    if manifest.get("status") != "forecasts_complete_unscored":
        raise ValueError("An incomplete neural run exists; inspect before restarting")
    for section in ("code", "protocols", "inputs"):
        for path, expected in manifest[section].items():
            if sha(ROOT / path) != expected:
                raise ValueError(f"Existing neural run differs in {path}")
    for section in ("outputs", "checkpoints"):
        for path, expected in manifest[section].items():
            if sha(OUT / path) != expected:
                raise ValueError(f"Existing neural artifact changed: {path}")
    if sha(OUT / "neural_pre_run_checks.txt") != manifest["pre_run_checks_sha256"]:
        raise ValueError("Pre-run neural test evidence changed")
    if sha(OUT / "neural_warmup_exclusions.json") != manifest["warmup_exclusions_sha256"]:
        raise ValueError("Diagnostic warm-up exclusions changed")
    if ("initial_manifest_sha256" in manifest
            and sha(OUT / "neural_manifest_initial_insufficient.json") != manifest["initial_manifest_sha256"]):
        raise ValueError("Initial insufficient-warm-up manifest changed")


def run():
    OUT.mkdir(exist_ok=True, parents=True)
    manifest_path = OUT / "neural_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        verify_existing(manifest)
        print(json.dumps({"status": "VERIFIED_EXISTING_FROZEN_NEURAL_RUN",
                          "forecast_rows": manifest["forecast_rows"],
                          "annual_fits": manifest["annual_fits"]}), flush=True)
        return

    forecasts_path = OUT / "neural_forecasts.parquet"
    fits_path = OUT / "neural_fits.json"
    state_dir = OUT / "checkpoints"
    if forecasts_path.exists() or fits_path.exists() or list(state_dir.glob("*.pt")):
        raise ValueError("Neural artifacts exist without a manifest; refusing overwrite")

    checks = subprocess.run([sys.executable, "-m", "unittest", "tests.test_reference_xlstm", "-v"],
                            cwd=ROOT, capture_output=True, text=True, check=False)
    checks_path = OUT / "neural_pre_run_checks.txt"
    checks_path.write_text(checks.stdout + checks.stderr)
    if checks.returncode:
        raise RuntimeError(checks.stdout + checks.stderr)

    reference = yaml.safe_load((ROOT / "model_memory_reference.yaml").read_text())
    core = yaml.safe_load((ROOT / "model_memory_study.yaml").read_text())
    if (reference["comparisons"]["total_with_core"] != 46
            or reference["neural"]["historical_forecast_start"] != "2013-02-01"
            or reference["sample"]["score_end"] != "2025-10-10"
            or reference["sample"]["latest_target"] != "2025-10-20"):
        raise ValueError("Fixed reference family or date fences changed")
    features = pd.read_parquet(ROOT / core["inputs"]["features"]).loc[:core["sample"]["latest_target"]]
    latents = pd.read_parquet(ROOT / core["inputs"]["latents"]).loc[:core["sample"]["score_end"]]
    common = (np.isfinite(features.loc[:, prior.ALL_FEATURES]).all(axis=1)
              & features.index.isin(latents.index))
    eligible = features.index[common]
    for date, expected_n in WARMUP:
        try:
            neural.training_windows(features, date, eligible)
        except ValueError as error:
            if str(error) != f"INSUFFICIENT_DATA: {expected_n} complete neural training windows":
                raise ValueError("Predeclared initial warm-up evidence changed") from error
        else:
            raise ValueError("Predeclared insufficient warm-up origin became sufficient")
    if len(neural.training_windows(features, FIRST_FIT, eligible)["origins"]) != 500:
        raise ValueError("Initial neural fit no longer has exactly 500 complete windows")
    warmup_path = OUT / "neural_warmup_exclusions.json"
    exclusions = [{"origin": date, "train_n": n, "status": "INSUFFICIENT_WARMUP",
                   "minimum_train": 500, "scored": False} for date, n in WARMUP]
    if warmup_path.exists():
        if json.loads(warmup_path.read_text()) != exclusions:
            raise ValueError("Existing warm-up exclusion record differs")
    else:
        dump(warmup_path, exclusions)

    source_paths = ["src/reference_xlstm.py", "src/run_reference_xlstm.py", "tests/test_reference_xlstm.py",
                    "reports/model_memory_study/XLSTM_PLAN.md", "src/orthogonal_round2.py"]
    protocol_paths = ["model_memory_study.yaml", "model_memory_reference.yaml",
                      "reports/model_memory_study/NEURAL_WARMUP_AMENDMENT.md"]
    input_paths = [core["inputs"]["features"], core["inputs"]["latents"]]
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(), "status": "prefit_fixed", "python": sys.version,
        "evidence_class": reference["evidence_class"], "models": list(neural.MODEL_NAMES),
        "code": {path: sha(ROOT / path) for path in source_paths},
        "protocols": {path: sha(ROOT / path) for path in protocol_paths},
        "inputs": {path: sha(ROOT / path) for path in input_paths},
        "pre_run_checks_sha256": sha(checks_path), "scores_read": False,
        "state_paths": "checkpoints/{fit_origin}_{model}.pt",
        "initial_fit_after_context_warmup": FIRST_FIT,
        "warmup_exclusions_sha256": sha(warmup_path),
    }
    initial_manifest = OUT / "neural_manifest_initial_insufficient.json"
    if initial_manifest.exists():
        manifest["initial_manifest_sha256"] = sha(initial_manifest)
    dump(manifest_path, manifest)

    y, endings = neural.targets(features.rv_total)
    mask = (common & np.isfinite(y).all(axis=1)
            & (endings <= pd.Timestamp(core["sample"]["latest_target"])).all(axis=1)
            & (features.index >= pd.Timestamp(FIRST_FIT))
            & (features.index <= pd.Timestamp(core["sample"]["score_end"])))
    origins = features.index[mask]
    state_dir.mkdir(exist_ok=True)
    state_hashes = {}

    def save_fit(origin, fitted):
        for name, model in fitted["models"].items():
            path = state_dir / f"{origin.date()}_{name}.pt"
            if path.exists():
                raise ValueError(f"Existing neural checkpoint: {path}")
            torch.save(model.state_dict(), path)
            state_hashes[str(path.relative_to(OUT))] = sha(path)

    def progress(info):
        print(json.dumps({"event": "year_complete", **info}), flush=True)

    started = time.monotonic()
    forecasts, fits = neural.yearly_forecasts(features, eligible, origins,
                                             progress=progress, fit_callback=save_fit)
    for section in ("code", "protocols", "inputs"):
        for path, expected in manifest[section].items():
            if sha(ROOT / path) != expected:
                raise RuntimeError(f"Changed neural run input: {path}")
    forecasts.to_parquet(forecasts_path, index=False)
    dump(fits_path, fits)
    manifest.update({
        "status": "forecasts_complete_unscored", "completed_utc": datetime.now(UTC).isoformat(),
        "wall_seconds": time.monotonic() - started, "forecast_rows": len(forecasts),
        "annual_fits": len(fits), "checkpoints": state_hashes,
        "outputs": {path.name: sha(path) for path in (forecasts_path, fits_path)},
    })
    dump(manifest_path, manifest)
    print(json.dumps({"event": "neural_forecasts_complete_unscored", "rows": len(forecasts),
                      "annual_fits": len(fits), "wall_seconds": manifest["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    run()
