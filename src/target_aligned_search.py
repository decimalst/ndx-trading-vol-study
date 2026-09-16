"""Directly fitted bounded cross-moment models on immutable shared forecasts."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import cross_moment_score as cs
from . import orthogonal_round2 as inference
from . import target_aligned_panel as tp
from .cross_moment_search import paired_inference
from .international_search import diagnostics
from .verify_target_aligned import validate_upstream

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "target_aligned.yaml"
REPORT = ROOT / "reports/target_aligned"
OUT = ROOT / "data/target_aligned"
CONTRASTS = (
    ("aligned_dynamic", "aligned_constant", "product_mse"),
    ("aligned_dynamic", "constant_matrix", "product_mse"),
    ("aligned_dynamic", "dynamic_correlation", "product_mse"),
)
WAVE_ALPHA = 0.05 / (12 * 13)
EFFECT = 1e-10


def validate(p):
    expected = {
        "comparisons": {
            "contrasts": [
                ["aligned_dynamic", "aligned_constant", "product_mse"],
                ["aligned_dynamic", "constant_matrix", "product_mse"],
                ["aligned_dynamic", "dynamic_correlation", "product_mse"],
            ],
            "controls": ["aligned_constant", "constant_matrix", "dynamic_correlation"],
            "cumulative_hypotheses": 117,
            "inherited_hypotheses": 114,
            "inherited_sources": [
                "reports/orthogonal_round2/metrics.json",
                "reports/model_memory_study/combined_metrics.json",
                "reports/iterative_signal_search/metrics.json",
                "reports/international_volatility/metrics.json",
                "reports/overnight_index/metrics.json",
                "reports/macro_overnight/metrics.json",
                "reports/measurement_memory/metrics.json",
                "reports/index_hinge/metrics.json",
                "reports/tail_shape/metrics.json",
                "reports/calendar_variance/metrics.json",
                "reports/relative_risk/metrics.json",
                "reports/joint_risk/metrics.json",
                "reports/cross_moment/metrics.json",
            ],
            "new_hypotheses": 3,
        },
        "fitting": {
            "cap": 0.995,
            "coefficient_bounds": {"a": [-0.995, 0.995], "b": [-1.0, 1.0]},
            "combined_forecast_rows_expected": 12310,
            "gradient_eps_multiplier": 512,
            "minimum_correlation_eigenvalue": 0.005,
            "new_forecasts_expected": 4924,
            "new_models": ["aligned_constant", "aligned_dynamic"],
            "new_monthly_fits_expected": 118,
            "new_scalar_fits_expected": 236,
            "training_staging": "current_monthly_frozen_fit_training_residuals",
        },
        "index": {
            "asset": "QQQ_ETF_and_SPX_price_index",
            "development": ["2016-01-04", "2019-12-31"],
            "development_target_available_by": "2019-12-31",
            "evaluation": ["2020-01-02", "2025-10-17"],
            "evaluation_stability": [
                ["2020-01-02", "2022-12-31"],
                ["2023-01-01", "2025-10-17"],
            ],
            "horizons": [1],
            "latest_target": "2025-10-20",
            "market_lag": 1,
            "minimum_train": 1000,
            "models": [
                "constant_matrix",
                "constant_correlation",
                "dynamic_correlation",
                "aligned_constant",
                "aligned_dynamic",
            ],
            "origin_end": "2025-10-17",
            "origin_start": "2016-01-04",
            "sealed_start": "2025-11-03",
            "source_end": "2025-10-20",
        },
        "inference": {
            "blocks": [21, 63, 126],
            "bootstrap_draws": 99999,
            "hac_lags": 126,
            "inference_scale": 1e-10,
            "minimum_phase_observations": 127,
            "seed": 20260918,
        },
        "outputs": {
            "admission": "data/target_aligned/upstream_admission.json",
            "data": "data/target_aligned",
            "fits": "data/target_aligned/fits.json",
            "forecasts": "data/target_aligned/forecasts.parquet",
            "reports": "reports/target_aligned",
        },
        "scoring": {
            "coherence_eps_multiplier": 64,
            "conditional_rho_max": 0.995,
            "constant_matrix_rho_max": 0.999999,
            "effect_threshold_absolute": 1e-10,
            "new_model_fits": 236,
        },
        "upstream": {
            "data": "data/cross_moment",
            "features": "data/joint_risk/features.parquet",
            "fits": "data/joint_risk/fits.json",
            "forecasts": "data/cross_moment/scored_forecasts.parquet",
            "forecasts_expected": 7386,
            "joint_protocol": "joint_risk.yaml",
            "manifest_sha256": "28a3fa3975bd58789d6859850579a472671443937e4de641351eb8f8217c969a",
            "monthly_fits_expected": 118,
            "protocol": "cross_moment.yaml",
            "protocol_sha256": "edeb72e2d7708c0a5227318d1ea54742050f9be0b761941ce598cdda28c26ed0",
            "reports": "reports/cross_moment",
            "required_status": "VERIFIED",
            "scored_origins_expected": 2462,
            "targets": "data/joint_risk/targets.parquet",
            "verification_sha256": "b3745d90db160e1102f2c3b987627add1d890958e38b52bb30499b77f5c13fa0",
            "verifier_sha256": "d1cb8c3fc69e44646359b839aac160acf6ed4be6cc043e2f2ca23f7c979430fa",
        },
        "verification": {
            "coefficient_absolute_tolerance": 1e-12,
            "coefficient_relative_tolerance": 1e-10,
            "coherence_eps_multiplier": 64,
            "dimensionless_relative_tolerance": 1e-09,
            "fit_gradient_eps_multiplier": 512,
            "fit_statistic_eps_multiplier": 256,
            "marginal_replay_absolute_tolerance": 1e-12,
            "marginal_replay_relative_tolerance": 1e-10,
            "probability_absolute_tolerance": 1e-14,
            "product_comparison_eps_multiplier": 256,
        },
    }
    if (
        p["wave"] != 12
        or p["wave_alpha"] != WAVE_ALPHA
        or any(
            p[group].get(key) != value
            for group, items in expected.items()
            for key, value in items.items()
        )
    ):
        raise ValueError("Fixed target-aligned protocol differs")


def inherited(p):
    rows = []
    for source in p["comparisons"]["inherited_sources"]:
        for number, row in enumerate(json.loads((ROOT / source).read_text())["rows"]):
            rows.append(
                {
                    "study": row.get("study", Path(source).parent.name or Path(source).stem),
                    "candidate": row["candidate"],
                    "control": row.get("control", "baseline"),
                    "horizon": row["horizon"],
                    "p_conservative": row["p_conservative"],
                    "source": source,
                    "source_sha256": inference.digest(ROOT / source),
                    "source_row_index": number,
                    **{name: row[name] for name in ["measure", "score"] if name in row},
                }
            )
    if len(rows) != 114:
        raise ValueError("All114 inherited comparisons required")
    return rows


def passes(row):
    phases = row["phases"]
    if len(phases) != 2 or {x["name"] for x in phases} != {"development", "evaluation"}:
        return False
    evaluation = next(x for x in phases if x["name"] == "evaluation")
    return (
        row["p_holm_wave"] < WAVE_ALPHA
        and row["p_holm_cumulative"] < 0.05
        and all(x["n"] > 0 and x["delta"] <= -1e-10 for x in phases)
        and len(evaluation["stability"]) == 2
        and all(x["delta"] < 0 for x in evaluation["stability"])
    )


def candidate_leads(rows):
    keys = [(x["candidate"], x["control"], x["score"]) for x in rows]
    if len(keys) != 3 or set(keys) != set(CONTRASTS):
        raise ValueError("Complete three-comparison family required")
    return ["aligned_dynamic"] if all(passes(row) for row in rows) else []


def evaluate(panel, calendar, p, monthly_fits):
    tp.validate_panel(panel)
    rows = []
    for candidate, control, score in CONTRASTS:
        phases = []
        for code, name in enumerate(["development", "evaluation"]):
            first, last = p["index"][name]
            frame = panel.loc[panel.origin.between(first, last)]
            models = {
                m: frame.loc[frame.model.eq(m)].set_index("origin").sort_index()
                for m in tp.MODELS
            }
            a, b = models[candidate], models[control]
            d = cs.paired_difference(
                a.forecast_product.to_numpy(),
                b.forecast_product.to_numpy(),
                a.realized_product.to_numpy(),
            )
            phase = paired_inference(
                a.product_mse.to_numpy(),
                b.product_mse.to_numpy(),
                d,
                p,
                p["inference"]["seed"] + code * 10000,
            )
            phase.update(
                name=name,
                first_origin=str(a.index[0].date()),
                last_origin=str(a.index[-1].date()),
            )
            scale = p["inference"]["inference_scale"]
            detail = diagnostics(
                a.index,
                d / scale,
                calendar,
                1,
                p["index"]["evaluation_stability"] if name == "evaluation" else [],
            )
            for records in detail.values():
                if isinstance(records, list):
                    for record in records:
                        if "delta" in record:
                            record["delta"] *= scale
            phase.update(detail)
            phases.append(phase)
        rows.append(
            {
                "study": "target_aligned",
                "candidate": candidate,
                "control": control,
                "score": score,
                "horizon": 1,
                "phases": phases,
                "p_conservative": max(x["p_conservative"] for x in phases),
            }
        )
    prior = inherited(p)
    wave = inference.holm_adjust([x["p_conservative"] for x in rows])
    cumulative = inference.holm_adjust([x["p_conservative"] for x in prior + rows])[-3:]
    for row, wp, cp in zip(rows, wave, cumulative, strict=True):
        row.update(p_holm_wave=float(wp), p_holm_cumulative=float(cp))
        row["verdict"] = "COMPARISON_GATE_PASS" if passes(row) else "DOES_NOT_QUALIFY"
    return {
        "rows": rows,
        "inherited_rows": prior,
        "leads": candidate_leads(rows),
        "hypothesis_count": 3,
        "cumulative_hypothesis_count": 117,
        "protocol_sha256": inference.digest(PROTOCOL),
        "evidence_class": p["evidence_class"],
        "new_monthly_fits": monthly_fits,
        "new_scalar_fits": 2 * monthly_fits,
        "new_forecasts": int(panel.model.isin(tp.NEW_MODELS).sum()),
        "reused_forecasts": int(panel.model.isin(cs.MODELS).sum()),
        "combined_forecast_rows": len(panel),
        "common_scored_origins": panel.origin.nunique(),
    }


def failure_metrics(error, protocol_hash):
    status = "INSUFFICIENT_DATA" if "INSUFFICIENT_DATA" in str(error) else "INVALID_RUN"
    return {
        "status": "UNEVALUABLE",
        "whole_wave_aborted": True,
        "leads": [],
        "hypothesis_count": 3,
        "cumulative_hypothesis_count": 117,
        "protocol_sha256": protocol_hash,
        "rows": [
            {
                "study": "target_aligned",
                "candidate": a,
                "control": b,
                "score": s,
                "horizon": 1,
                "p_conservative": 1.0,
                "p_holm_wave": 1.0,
                "p_holm_cumulative": 1.0,
                "phases": [],
                "status": status,
                "error": str(error),
            }
            for a, b, s in CONTRASTS
        ],
    }


def report(metrics):
    lines = [
        "# Directly fitted QQQ-SPX residual cross-moment prediction",
        "",
        "Negative absolute paired loss gap means improvement. All three comparisons retained.",
        "",
        "| Score / control | Development | Evaluation | Wave Holm p | Cumulative Holm p | Gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in metrics["rows"]:
        d, e = row["phases"]
        lines.append(
            f"| {row['score']} / {row['control']} | {d['delta']:+.3e} | {e['delta']:+.3e} | {row['p_holm_wave']:.6f} | {row['p_holm_cumulative']:.6f} | {row['verdict']} |"
        )
    lines += [
        "",
        "Passing candidates: " + json.dumps(metrics["leads"]),
        "",
        "Three new product-score comparisons after two target-aligned scalar fits per month. Frozen shared mean error, issued diagonals, archival daily products and historical reuse limit interpretation.",
        "",
    ]
    (REPORT / "results.md").write_text("\n".join(lines))


def ledger(rows):
    with (REPORT / "trial_ledger.jsonl").open("a") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def input_paths(p):
    manifest = json.loads((ROOT / p["upstream"]["reports"] / "manifest.json").read_text())
    paths = set(manifest["inputs"]) | {p["upstream"]["protocol"]}
    for directory in [p["upstream"]["reports"], p["upstream"]["data"]]:
        paths.update(
            str(f.relative_to(ROOT)) for f in (ROOT / directory).rglob("*") if f.is_file()
        )
    return paths


def read_pinned_parquet(name, manifest, **kwargs):
    """Decode exactly the admitted bytes, with no pathname reopen after hashing."""
    payload = (ROOT / name).read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["inputs"][name]:
        raise ValueError("Pinned upstream parquet changed before scoring: " + name)
    return pd.read_parquet(io.BytesIO(payload), **kwargs)


def read_pinned_json(name, manifest):
    payload = (ROOT / name).read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["inputs"][name]:
        raise ValueError("Pinned upstream JSON changed before fitting: " + name)
    return json.loads(payload)


def run():
    protocol_bytes = PROTOCOL.read_bytes()
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    p = yaml.safe_load(protocol_bytes)
    validate(p)
    REPORT.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    if (REPORT / "manifest.json").exists():
        raise ValueError("Refusing to overwrite a registered target-aligned experiment")
    modules = [
        "tests.test_target_aligned_models",
        "tests.test_target_aligned_panel",
        "tests.test_target_aligned_search",
        "tests.test_target_aligned_publication",
        "tests.test_verify_target_aligned",
        "tests.test_plot_target_aligned",
        "tests.test_round2_inference",
    ]
    checked = subprocess.run(
        [sys.executable, "-m", "unittest", *modules, "-v"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    (REPORT / "pre_run_checks.txt").write_text(checked.stdout + checked.stderr)
    if checked.returncode:
        raise RuntimeError("Prewritten target-aligned tests failed; no registration or fit")
    if inference.digest(PROTOCOL) != protocol_hash:
        raise ValueError("Frozen protocol changed during pretests; no registration")
    code = list((ROOT / "src").rglob("*.py")) + list((ROOT / "tests").rglob("*.py"))
    inputs = input_paths(p)
    preserved = [
        file
        for file in list(ROOT.glob("*.yaml")) + list((ROOT / "reports").rglob("*"))
        if file.is_file()
        and file != PROTOCOL
        and REPORT not in file.parents
        and str(file.relative_to(ROOT)) not in inputs
    ]
    backend = io.StringIO()
    with redirect_stdout(backend):
        np.show_config()
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": protocol_hash,
        "environment": {
            "python": sys.version,
            "packages": {
                name: version(name)
                for name in ["numpy", "pandas", "scipy", "pyarrow", "PyYAML"]
            },
            "numpy_backend": backend.getvalue(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "LOKY_MAX_CPU_COUNT"]
            },
        },
        "code": {str(file.relative_to(ROOT)): inference.digest(file) for file in sorted(code)},
        "inputs": {name: inference.digest(ROOT / name) for name in sorted(inputs)},
        "preserved": {
            str(file.relative_to(ROOT)): inference.digest(file) for file in sorted(preserved)
        },
    }
    inference.dump(REPORT / "manifest.json", manifest)
    metrics = None
    try:
        ledger([{"event": "inherited", **row} for row in inherited(p)])
        ledger(
            [
                {
                    "event": "registered",
                    "study": "target_aligned",
                    "candidate": a,
                    "control": b,
                    "score": s,
                    "horizon": 1,
                    "protocol_sha256": manifest["protocol_sha256"],
                }
                for a, b, s in CONTRASTS
            ]
        )
        proof = validate_upstream(ROOT)
        inference.dump(OUT / "upstream_admission.json", proof)
        original = read_pinned_parquet(p["upstream"]["forecasts"], manifest)
        features = read_pinned_parquet(p["upstream"]["features"], manifest)
        targets = read_pinned_parquet(p["upstream"]["targets"], manifest)
        frozen_fits = read_pinned_json(p["upstream"]["fits"], manifest)
        forecasts, fits = tp.forecast_panel(
            features, targets, original, frozen_fits, p["index"]
        )
        forecasts.to_parquet(OUT / "forecasts.parquet")
        inference.dump(OUT / "fits.json", fits)
        metrics = evaluate(forecasts, features.index, p, len(fits))
        if inference.digest(PROTOCOL) != manifest["protocol_sha256"]:
            raise ValueError("Frozen protocol changed during target-aligned run")
        for group in ["code", "inputs", "preserved"]:
            for name, expected in manifest[group].items():
                if inference.digest(ROOT / name) != expected:
                    raise ValueError("Frozen artifact changed: " + name)
        inference.dump(REPORT / "metrics.json", metrics)
        report(metrics)
        ledger([{"event": "evaluated", **row} for row in metrics["rows"]])
    except Exception as error:
        failure = failure_metrics(error, manifest["protocol_sha256"])
        inference.dump(REPORT / "metrics.json", failure)
        inference.dump(REPORT / "failure.json", failure)
        (REPORT / "results.md").write_text(
            "# Direct residual cross-moment experiment\n\nUNEVALUABLE: all3comparisons retained with p=1; no lead.\n"
        )
        ledger([{"event": "unevaluable", **row} for row in failure["rows"]])
        if metrics is not None:
            diagnostic = {
                "status": "UNPUBLISHED_DIAGNOSTIC_ONLY",
                "not_for_inherited_inference_or_promotion": True,
            }
            try:
                json.dumps(metrics, allow_nan=False)
                diagnostic["scored_metrics"] = metrics
            except (TypeError, ValueError):
                diagnostic["nonserializable_metrics_repr"] = repr(metrics)
            inference.dump(REPORT / "unpublished_scored_metrics.json", diagnostic)
        raise
    print(
        json.dumps(
            {
                "status": "SCORED_AWAITING_INDEPENDENT_VERIFICATION",
                "forecasts": len(forecasts),
                "new_monthly_fits": len(fits),
                "new_scalar_fits": 2 * len(fits),
                "new_forecasts": metrics["new_forecasts"],
                "leads": metrics["leads"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
