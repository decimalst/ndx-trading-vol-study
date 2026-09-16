"""Separate preregistered direct cross-moment scoring of immutable issued forecasts."""

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
from .international_search import diagnostics
from .verify_cross_moment import validate_upstream

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "cross_moment.yaml"
REPORT = ROOT / "reports/cross_moment"
OUT = ROOT / "data/cross_moment"
CONTRASTS = (
    ("dynamic_correlation", "constant_correlation", "product_mse"),
    ("dynamic_correlation", "constant_matrix", "product_mse"),
)
WAVE_ALPHA = 0.05 / (11 * 12)
EFFECT = 1e-10


def validate(p):
    expected = {
        "upstream": {
            "protocol": "joint_risk.yaml",
            "reports": "reports/joint_risk",
            "data": "data/joint_risk",
            "forecasts": "data/joint_risk/forecasts.parquet",
            "features": "data/joint_risk/features.parquet",
            "protocol_sha256": "358d863853d79c876476e67bcf91e4c7fc855f2f107c8df58ede074f19a809ef",
            "verifier_sha256": "ea4ac89c0d345506bb45ea1dd5add092497edeabdcfa83995e2e596f8596b4a4",
            "manifest_sha256": "c80e77d076cefd98c0e6d7df6c9dd213638ec486e718a9713055552575dc7848",
            "verification_sha256": "d62544e15e9575371a95fd7f1d5fdcb5e034cc49f7e78268a50ae5ffc5d3c3e3",
            "required_status": "VERIFIED",
            "forecasts_expected": 7386,
            "scored_origins_expected": 2462,
            "monthly_fits_expected": 118,
        },
        "index": {
            "asset": "QQQ_ETF_and_SPX_price_index",
            "source_end": "2025-10-20",
            "sealed_start": "2025-11-03",
            "origin_start": "2016-01-04",
            "origin_end": "2025-10-17",
            "latest_target": "2025-10-20",
            "development": ["2016-01-04", "2019-12-31"],
            "development_target_available_by": "2019-12-31",
            "evaluation": ["2020-01-02", "2025-10-17"],
            "evaluation_stability": [
                ["2020-01-02", "2022-12-31"],
                ["2023-01-01", "2025-10-17"],
            ],
            "horizons": [1],
            "models": list(cs.MODELS),
            "market_lag": 1,
        },
        "scoring": {
            "effect_threshold_absolute": EFFECT,
            "new_model_fits": 0,
            "conditional_rho_max": 0.995,
            "constant_matrix_rho_max": 1 - 1e-6,
            "coherence_eps_multiplier": 64,
        },
        "comparisons": {
            "new_hypotheses": 2,
            "inherited_hypotheses": 112,
            "cumulative_hypotheses": 114,
            "controls": ["constant_correlation", "constant_matrix"],
            "contrasts": [list(x) for x in CONTRASTS],
        },
        "inference": {
            "blocks": [21, 63, 126],
            "hac_lags": 126,
            "minimum_phase_observations": 127,
            "bootstrap_draws": 99999,
            "seed": 20260917,
            "inference_scale": 1e-10,
        },
        "verification": {
            "product_comparison_eps_multiplier": 256,
            "coherence_eps_multiplier": 64,
            "dimensionless_relative_tolerance": 1e-9,
            "probability_absolute_tolerance": 1e-14,
        },
    }
    if (
        p["wave"] != 11
        or p["wave_alpha"] != WAVE_ALPHA
        or any(
            p[group].get(k) != v for group, items in expected.items() for k, v in items.items()
        )
    ):
        raise ValueError("Fixed cross-moment protocol differs")


def paired_inference(candidate, control, difference, p, seed):
    if any(np.iscomplexobj(v) for v in [candidate, control, difference]):
        raise ValueError("Real finite product losses and paired differences required")
    candidate, control, d = (np.asarray(x, float) for x in [candidate, control, difference])
    if (
        candidate.ndim != 1
        or candidate.shape != control.shape
        or candidate.shape != d.shape
        or not np.isfinite([candidate, control, d]).all()
        or (candidate < 0).any()
        or (control < 0).any()
    ):
        raise ValueError(
            "Aligned finite nonnegative product losses and finite paired gaps required"
        )
    if len(d) <= max(p["inference"]["hac_lags"], max(p["inference"]["blocks"])):
        raise ValueError("INSUFFICIENT_DATA: phase shorter than literal inference bandwidth")
    allowance = np.zeros_like(d)
    for term in [candidate, control, d]:
        magnitude = np.abs(term)
        allowance += 64 * np.maximum(
            np.finfo(float).eps * magnitude, magnitude - np.nextafter(magnitude, 0)
        )
    if (
        not np.isfinite(allowance).all()
        or (np.abs(d - (candidate - control)) > allowance).any()
    ):
        raise ValueError("Factored difference inconsistent with individual squared losses")
    scale = float(p["inference"]["inference_scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Positive fixed inference units required")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        normalized = d / scale
        center = float(normalized.mean())
        centered = normalized - center
        squared_norm = float(centered @ centered)
    if (
        not np.isfinite(normalized).all()
        or not np.isfinite([center, squared_norm]).all()
        or (not np.equal(normalized, normalized[0]).all() and squared_norm == 0)
    ):
        raise ValueError("Invalid normalized difference variance or arithmetic underflow")
    blocks = {}
    for block in p["inference"]["blocks"]:
        samples = inference.bootstrap_means(
            normalized, block, p["inference"]["bootstrap_draws"], seed + block
        )[:, 0]
        if not np.isfinite(samples).all():
            raise ValueError("Nonfinite bootstrap means")
        probability = (1 + int((np.abs(samples - center) >= abs(center)).sum())) / (
            len(samples) + 1
        )
        blocks[str(block)] = {
            "p": probability,
            "ci95": (np.quantile(samples, [0.025, 0.975]) * scale).tolist(),
        }
    hac = inference.hac_summary(normalized, lags=p["inference"]["hac_lags"])
    hac = {
        "se": hac["se"] * scale,
        "p": hac["p"],
        "ci95": (np.asarray(hac["ci95"]) * scale).tolist(),
        "mde80_nominal": hac["mde80_nominal"] * scale,
    }
    intervals = [hac["ci95"]] + [item["ci95"] for item in blocks.values()]
    result = {
        "n": len(d),
        "delta": center * scale,
        "candidate_loss": float(candidate.mean()),
        "control_loss": float(control.mean()),
        "block_inference": blocks,
        "hac126": hac,
        "nominal_mde_effect_ratio": hac["mde80_nominal"] / EFFECT,
        "ci95_envelope": [min(x[0] for x in intervals), max(x[1] for x in intervals)],
        "p_conservative": max(hac["p"], *(x["p"] for x in blocks.values())),
    }
    json.dumps(result, allow_nan=False)
    return result


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
    if len(rows) != 112:
        raise ValueError("All112 inherited comparisons required")
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
    if len(keys) != 2 or set(keys) != set(CONTRASTS):
        raise ValueError("Complete two-comparison family required")
    return ["dynamic_correlation"] if all(passes(row) for row in rows) else []


def evaluate(panel, calendar, p):
    if tuple(panel.columns) != cs.INPUT_COLUMNS + cs.ADDED_COLUMNS:
        raise ValueError("Exact original and three derived score columns required")
    checked = cs.score_panel(panel.loc[:, cs.INPUT_COLUMNS])
    for column in cs.ADDED_COLUMNS:
        if not np.array_equal(panel[column].to_numpy(), checked[column].to_numpy()):
            raise ValueError("Cached product scores disagree with immutable issued rows")
    rows = []
    for candidate, control, score in CONTRASTS:
        phases = []
        for code, name in enumerate(["development", "evaluation"]):
            first, last = p["index"][name]
            frame = panel.loc[panel.origin.between(first, last)]
            models = {
                m: frame.loc[frame.model.eq(m)].set_index("origin").sort_index()
                for m in cs.MODELS
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
                "study": "cross_moment",
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
    cumulative = inference.holm_adjust([x["p_conservative"] for x in prior + rows])[-2:]
    for row, wp, cp in zip(rows, wave, cumulative, strict=True):
        row.update(p_holm_wave=float(wp), p_holm_cumulative=float(cp))
        row["verdict"] = "COMPARISON_GATE_PASS" if passes(row) else "DOES_NOT_QUALIFY"
    return {
        "rows": rows,
        "inherited_rows": prior,
        "leads": candidate_leads(rows),
        "hypothesis_count": 2,
        "cumulative_hypothesis_count": 114,
        "protocol_sha256": inference.digest(PROTOCOL),
        "evidence_class": p["evidence_class"],
        "new_model_fits": 0,
        "new_forecasts": 0,
        "scored_issued_forecast_rows": len(panel),
        "common_scored_origins": panel.origin.nunique(),
    }


def failure_metrics(error, protocol_hash):
    status = "INSUFFICIENT_DATA" if "INSUFFICIENT_DATA" in str(error) else "INVALID_RUN"
    return {
        "status": "UNEVALUABLE",
        "whole_wave_aborted": True,
        "leads": [],
        "hypothesis_count": 2,
        "cumulative_hypothesis_count": 114,
        "protocol_sha256": protocol_hash,
        "rows": [
            {
                "study": "cross_moment",
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
        "# Direct QQQ-SPX residual cross-moment prediction",
        "",
        "Negative absolute paired loss gap means improvement. Both comparisons retained.",
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
        "Two new product-score comparisons on unchanged issued forecasts; no new model fits or forecasts. Shared mean error, issued diagonals, archival daily products and historical reuse limit interpretation.",
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


def run():
    p = yaml.safe_load(PROTOCOL.read_text())
    validate(p)
    REPORT.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    if (REPORT / "manifest.json").exists():
        raise ValueError("Refusing to overwrite a registered cross-moment experiment")
    modules = [
        "tests.test_cross_moment_score",
        "tests.test_cross_moment_search",
        "tests.test_cross_moment_publication",
        "tests.test_verify_cross_moment",
        "tests.test_plot_cross_moment",
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
        raise RuntimeError("Prewritten cross-moment tests failed; no registration or fit")
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
        "protocol_sha256": inference.digest(PROTOCOL),
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
                    "study": "cross_moment",
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
        calendar = read_pinned_parquet(p["upstream"]["features"], manifest, columns=[]).index
        forecasts = cs.score_panel(original)
        forecasts.to_parquet(OUT / "scored_forecasts.parquet")
        metrics = evaluate(forecasts, calendar, p)
        if inference.digest(PROTOCOL) != manifest["protocol_sha256"]:
            raise ValueError("Frozen protocol changed during cross-moment run")
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
            "# Direct residual cross-moment experiment\n\nUNEVALUABLE: all2comparisons retained with p=1; no lead.\n"
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
                "new_model_fits": 0,
                "new_forecasts": 0,
                "leads": metrics["leads"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
