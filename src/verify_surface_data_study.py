"""Independent verifier for the frozen private option-surface study."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "surface_data_study.yaml"


def _sha(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _qlike(actual: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    if (actual < 0).any() or (forecast <= 0).any():
        raise AssertionError("invalid QLIKE inputs")
    ratio = np.maximum(actual, 1e-18) / forecast
    return ratio - np.log(ratio) - 1.0


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels == 1
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return math.nan
    ranks = pd.Series(scores).rank(method="average").to_numpy(float)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) /
                 (n_pos * n_neg))


def _model_metrics(actual: np.ndarray, labels: np.ndarray, scores: np.ndarray,
                   top_fraction: float) -> dict[str, float]:
    loss = _qlike(actual, scores)
    count = max(1, int(math.ceil(len(scores) * top_fraction)))
    order = np.argsort(-scores, kind="mergesort")
    base_rate = float(labels.mean())
    top_rate = float(labels[order[:count]].mean())
    return {
        "mean_qlike": float(loss.mean()),
        "auc": _auc(labels, scores),
        "top_decile_lift": float(top_rate / base_rate) if base_rate > 0 else math.nan,
        "top_decile_event_rate": top_rate,
        "base_rate": base_rate,
    }


def _recompute(frame: pd.DataFrame, top_fraction: float) -> dict[str, Any]:
    clean = frame.dropna(subset=[
        "actual_var", "baseline_var", "augmented_var", "tail_event"
    ])
    actual = clean["actual_var"].to_numpy(float)
    labels = clean["tail_event"].to_numpy(int)
    baseline_scores = clean["baseline_var"].to_numpy(float)
    augmented_scores = clean["augmented_var"].to_numpy(float)
    baseline = _model_metrics(actual, labels, baseline_scores, top_fraction)
    augmented = _model_metrics(actual, labels, augmented_scores, top_fraction)
    base_loss = _qlike(actual, baseline_scores)
    aug_loss = _qlike(actual, augmented_scores)
    return {
        "n": int(len(clean)), "baseline": baseline, "augmented": augmented,
        "paired": {
            "mean_qlike_difference": float((aug_loss - base_loss).mean()),
            "improvement_pct": float(
                100.0 * (base_loss.mean() - aug_loss.mean()) / base_loss.mean()
            ) if base_loss.mean() != 0 else math.nan,
            "win_rate": float((aug_loss < base_loss).mean()),
            "auc_difference": float(augmented["auc"] - baseline["auc"]),
            "top_decile_lift_difference": float(
                augmented["top_decile_lift"] - baseline["top_decile_lift"]
            ),
        },
    }


def _assert_close(label: str, actual: Any, expected: Any,
                  tolerance: float = 1e-12) -> None:
    if isinstance(expected, dict):
        for key, value in expected.items():
            if key in ("moving_block_ci95",):
                continue
            if key not in actual:
                raise AssertionError(f"{label}.{key} missing")
            _assert_close(f"{label}.{key}", actual[key], value, tolerance)
        return
    if isinstance(expected, (int, np.integer)):
        if int(actual) != int(expected):
            raise AssertionError(f"{label}: {actual} != {expected}")
        return
    a, e = float(actual), float(expected)
    if math.isnan(e) and math.isnan(a):
        return
    if not math.isclose(a, e, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"{label}: {a} != {e}")


def verify_artifacts(protocol_path: str | pathlib.Path = DEFAULT_PROTOCOL,
                     root: pathlib.Path = ROOT) -> dict[str, Any]:
    protocol_path = pathlib.Path(protocol_path)
    spec = yaml.safe_load(protocol_path.read_text())
    checks = 0
    if spec["status"] != "frozen_before_first_empirical_run":
        raise AssertionError("protocol status drifted")
    checks += 1
    if spec["evidence_class"] != "private_diagnostic_only":
        raise AssertionError("evidence class drifted")
    checks += 1
    if spec["shape_construction"]["open_interest_available"]:
        raise AssertionError("source falsely claims open interest")
    if not spec["shape_construction"]["gamma_weighted_volume_is_not_dealer_gex"]:
        raise AssertionError("gamma-volume label drifted")
    checks += 1

    forecast_path = root / spec["outputs"]["forecasts"]
    metrics_path = root / spec["outputs"]["metrics"]
    forecasts = pd.read_parquet(forecast_path)
    metrics = json.loads(metrics_path.read_text())
    if metrics["protocol_sha256"] != hashlib.sha256(protocol_path.read_bytes()).hexdigest():
        raise AssertionError("metrics protocol hash mismatch")
    checks += 1
    for col in ("measurement_date", "target_date"):
        forecasts[col] = pd.to_datetime(forecasts[col])
    origins = pd.DatetimeIndex(forecasts.index)
    if not (forecasts["measurement_date"].to_numpy() < origins.to_numpy()).all():
        raise AssertionError("surface lookahead in forecasts")
    checks += 1
    if not (origins.to_numpy() < forecasts["target_date"].to_numpy()).all():
        raise AssertionError("non-forward target in forecasts")
    checks += 1
    if "surface_lag_sessions" in forecasts:
        if not (forecasts["surface_lag_sessions"] == 1).all():
            raise AssertionError("surface lag is not exactly one session")
        if not (
            (forecasts["origin_session_number"] - forecasts["measurement_session_number"] == 1)
            & (forecasts["target_session_number"] - forecasts["origin_session_number"] == 1)
        ).all():
            raise AssertionError("non-consecutive measurement/origin/target sessions")
        checks += 1
    forbidden = [c for c in forecasts.columns if "dealer_gex" in c.lower()]
    if forbidden:
        raise AssertionError(f"forbidden dealer-GEX columns: {forbidden}")
    checks += 1
    if not ((forecasts["baseline_var"] > 0) & (forecasts["augmented_var"] > 0)).all():
        raise AssertionError("nonpositive forecasts")
    checks += 1

    top_fraction = float(spec["study"]["tail"]["top_fraction"])
    stored_results = metrics.get("results", {})
    if not stored_results and "spy" in metrics:
        stored_results = {"SPY": metrics["spy"]}
    for symbol, split_results in stored_results.items():
        for split, stored in split_results.items():
            sample = forecasts.loc[
                (forecasts["symbol"] == symbol) & (forecasts["split"] == split)
            ]
            expected = _recompute(sample, top_fraction)
            _assert_close(f"{symbol}.{split}", stored, expected)
            checks += 1

    manifest_path = root / spec["outputs"]["source_manifest"]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["protocol_sha256"] != hashlib.sha256(protocol_path.read_bytes()).hexdigest():
            raise AssertionError("source manifest protocol hash mismatch")
        if manifest.get("raw_redistribution") is not False:
            raise AssertionError("manifest permits raw redistribution")
        checks += 2
        for symbol, records in manifest.get("sources", {}).items():
            raw_dir = root / spec["sources"][symbol]["raw_dir"]
            for record in records:
                candidates = list(raw_dir.rglob(record["name"]))
                if len(candidates) != 1:
                    raise AssertionError(
                        f"cannot uniquely resolve manifested {symbol}/{record['name']}"
                    )
                path = candidates[0]
                if path.stat().st_size != int(record["bytes"]) or _sha(path) != record["sha256"]:
                    raise AssertionError(f"raw source changed: {symbol}/{record['name']}")
                checks += 1
        daily_record = manifest.get("extra", {}).get("daily_surface", {})
        if daily_record:
            daily_path = root / spec["outputs"]["daily_surface"]
            if _sha(daily_path) != daily_record["sha256"]:
                raise AssertionError("daily surface artifact hash mismatch")
            checks += 1
    artifacts = metrics.get("artifacts", {})
    if "forecasts_sha256" in artifacts and artifacts["forecasts_sha256"] != _sha(forecast_path):
        raise AssertionError("forecast artifact hash mismatch")
    if "forecasts_sha256" in artifacts:
        checks += 1
    return {"status": "PASS", "checks": checks, "rows": int(len(forecasts))}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    args = parser.parse_args(argv)
    print(json.dumps(verify_artifacts(args.protocol), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
