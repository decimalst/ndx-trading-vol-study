"""Frozen QQQ-SPX relative intraday risk study and complete trial accounting."""

from __future__ import annotations

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
import yaml

from . import orthogonal_round2 as inference
from . import relative_risk_features as cf
from . import relative_risk_models as cm
from .international_search import diagnostics

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "relative_risk.yaml"
REPORT = ROOT / "reports/relative_risk"
OUT = ROOT / "data/relative_risk"
CONTRASTS = (("correlation", "baseline", "mse"), ("correlation", "mean", "mse"))
WAVE_ALPHA = 0.05 / (9 * 10)


def validate(p):
    i, c, inf = p["index"], p["comparisons"], p["inference"]
    expected = {
        "asset": "QQQ_ETF_relative_to_SPX_price_index",
        "horizons": [1],
        "models": list(cf.MODELS),
        "baseline": list(cf.BASE),
        "all_features": list(cf.ALL_FEATURES),
        "market_lag": 1,
        "minimum_train": 1000,
        "penalty": 0.01,
        "effect_threshold_relative": 0.0025,
        "source_end": "2025-10-20",
        "latest_target": "2025-10-20",
        "sealed_start": "2025-11-03",
        "origin_start": "2016-01-04",
        "origin_end": "2025-10-17",
        "development": ["2016-01-04", "2019-12-31"],
        "development_target_available_by": "2019-12-31",
        "evaluation": ["2020-01-02", "2025-10-17"],
        "evaluation_stability": [["2020-01-02", "2022-12-31"], ["2023-01-01", "2025-10-17"]],
    }
    fixed_inf = {
        "blocks": [21, 63, 126],
        "hac_lags": 126,
        "minimum_phase_observations": 127,
        "bootstrap_draws": 99999,
        "seed": 20260915,
    }
    if (
        any(i[k] != v for k, v in expected.items())
        or any(inf[k] != v for k, v in fixed_inf.items())
        or p["wave"] != 9
        or p["wave_alpha"] != WAVE_ALPHA
        or c["new_hypotheses"] != 2
        or c["inherited_hypotheses"] != 108
        or c["cumulative_hypotheses"] != 110
        or tuple(map(tuple, c["contrasts"])) != CONTRASTS
        or p["measurement"]["gk_floor"] != 1e-10
        or p["correlation"]["window"] != 22
        or p["correlation"]["roundoff_tolerance"] != 1e-12
    ):
        raise ValueError("Fixed relative risk replication specification differs")


def paired_inference(candidate, control, p, seed):
    candidate, control = np.asarray(candidate, float), np.asarray(control, float)
    if (
        candidate.ndim != 1
        or candidate.shape != control.shape
        or not np.isfinite([candidate, control]).all()
        or (candidate < 0).any()
        or (control < 0).any()
        or len(control) == 0
        or control.mean() <= 0
    ):
        raise ValueError("Finite aligned nonnegative losses and positive control MSE required")
    d = candidate - control
    if len(d) <= max(p["inference"]["hac_lags"], max(p["inference"]["blocks"])):
        raise ValueError("INSUFFICIENT_DATA: phase shorter than literal inference bandwidth")
    delta = float(d.mean())
    blocks = {}
    for block in p["inference"]["blocks"]:
        samples = inference.bootstrap_means(
            d, block, p["inference"]["bootstrap_draws"], seed + block
        )[:, 0]
        probability = (1 + int((np.abs(samples - delta) >= abs(delta)).sum())) / (
            len(samples) + 1
        )
        blocks[str(block)] = {
            "p": probability,
            "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        }
    hac = inference.hac_summary(d, lags=p["inference"]["hac_lags"])
    intervals = [hac["ci95"]] + [item["ci95"] for item in blocks.values()]
    return {
        "n": len(d),
        "delta": delta,
        "candidate_loss": float(candidate.mean()),
        "control_loss": float(control.mean()),
        "gain_relative": float(1 - candidate.mean() / control.mean()),
        "block_inference": blocks,
        "hac126": hac,
        "ci95_envelope": [min(x[0] for x in intervals), max(x[1] for x in intervals)],
        "p_conservative": max(hac["p"], *(item["p"] for item in blocks.values())),
    }


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
    if len(rows) != 108:
        raise ValueError("All108 inherited comparisons required")
    return rows


def passes(row):
    phases = row["phases"]
    if len(phases) != 2 or {x["name"] for x in phases} != {"development", "evaluation"}:
        return False
    evaluation = next(x for x in phases if x["name"] == "evaluation")
    return (
        row["p_holm_wave"] < WAVE_ALPHA
        and row["p_holm_cumulative"] < 0.05
        and all(x["n"] > 0 and x["delta"] < 0 and x["gain_relative"] >= 0.0025 for x in phases)
        and len(evaluation["stability"]) == 2
        and all(x["delta"] < 0 for x in evaluation["stability"])
    )


def candidate_leads(rows):
    keys = [(x["candidate"], x["control"], x["score"]) for x in rows]
    if len(keys) != 2 or set(keys) != set(CONTRASTS):
        raise ValueError("Complete two-comparison family required")
    return ["correlation"] if all(passes(row) for row in rows) else []


def evaluate(panel, calendar, p):
    if panel.duplicated(["origin", "model", "horizon"]).any() or set(panel.horizon) != {1}:
        raise ValueError("Unique one-session forecasts required")
    rows = []
    for candidate, control, score in CONTRASTS:
        phases = []
        for code, name in enumerate(["development", "evaluation"]):
            first, last = p["index"][name]
            frame = panel.loc[(panel.origin >= first) & (panel.origin <= last)]
            if name == "development":
                frame = frame.loc[
                    frame.available_date <= p["index"]["development_target_available_by"]
                ]
            wide = frame.pivot(
                index="origin", columns="model", values="prediction"
            ).sort_index()
            if set(wide.columns) != set(cf.MODELS) or not np.isfinite(wide).all().all():
                raise ValueError(
                    "Complete common finite signed relative-risk forecasts required"
                )
            models = {
                model: frame.loc[frame.model == model].set_index("origin").reindex(wide.index)
                for model in cf.MODELS
            }
            actual = models[control]
            for model, other in models.items():
                for column in [
                    "y",
                    "target_end",
                    "available_date",
                    "fit_origin",
                    "fit_cutoff_date",
                    "train_n",
                    "train_last_available",
                    "train_last_target",
                    "feature_cutoff_date",
                    "phase",
                ]:
                    if not actual[column].equals(other[column]):
                        raise ValueError("Paired risk targets or fitting metadata differ")
            if not np.isfinite(actual.y).all():
                raise ValueError("Finite signed common log-risk labels required")
            loss1 = (actual.y.to_numpy() - wide[candidate].to_numpy()) ** 2
            loss0 = (actual.y.to_numpy() - wide[control].to_numpy()) ** 2
            phase = paired_inference(loss1, loss0, p, p["inference"]["seed"] + code * 10000)
            phase.update(
                {
                    "name": name,
                    "first_origin": str(wide.index[0].date()),
                    "last_origin": str(wide.index[-1].date()),
                }
            )
            phase.update(
                diagnostics(
                    wide.index,
                    loss1 - loss0,
                    calendar,
                    1,
                    p["index"]["evaluation_stability"] if name == "evaluation" else [],
                )
            )
            phases.append(phase)
        rows.append(
            {
                "study": "relative_risk",
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
        "cumulative_hypothesis_count": 110,
        "protocol_sha256": inference.digest(PROTOCOL),
        "evidence_class": p["evidence_class"],
    }


def failure_metrics(error, protocol_hash):
    status = "INSUFFICIENT_DATA" if "INSUFFICIENT_DATA" in str(error) else "INVALID_RUN"
    return {
        "status": "UNEVALUABLE",
        "whole_wave_aborted": True,
        "leads": [],
        "hypothesis_count": 2,
        "cumulative_hypothesis_count": 110,
        "protocol_sha256": protocol_hash,
        "rows": [
            {
                "study": "relative_risk",
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
        "# QQQ-SPX relative intraday risk",
        "",
        "Negative absolute paired loss gap means improvement. Both comparisons retained.",
        "",
        "| Score / control | Development | Evaluation | Wave Holm p | Cumulative Holm p | Gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in metrics["rows"]:
        d, e = row["phases"]
        lines.append(
            f"| {row['score']} / {row['control']} | {d['delta']:+.6f} | {e['delta']:+.6f} | {row['p_holm_wave']:.6f} | {row['p_holm_cumulative']:.6f} | {row['verdict']} |"
        )
    lines += [
        "",
        "Passing candidates: " + json.dumps(metrics["leads"]),
        "",
        "One intraday correlation addition; archival ETF/index OHLC proxy and historical reuse remain exploratory.",
        "",
    ]
    (REPORT / "results.md").write_text("\n".join(lines))


def ledger(rows):
    with (REPORT / "trial_ledger.jsonl").open("a") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def run():
    p = yaml.safe_load(PROTOCOL.read_text())
    validate(p)
    REPORT.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    if (REPORT / "manifest.json").exists():
        raise ValueError("Refusing to overwrite a registered relative-risk experiment")
    modules = [
        "tests.test_relative_risk_features",
        "tests.test_relative_risk_models",
        "tests.test_relative_risk_search",
        "tests.test_relative_risk_publication",
        "tests.test_verify_relative_risk",
        "tests.test_index_hinge",
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
        raise RuntimeError("Prewritten relative-risk tests failed; no registration or fit")
    code = list((ROOT / "src").rglob("*.py")) + list((ROOT / "tests").rglob("*.py"))
    inputs = set(p["sources"].values())
    inputs.update(
        p["sources"][name] + ".manifest.json" for name in ["vxn", "vix", "vix9d", "vvix"]
    )
    inputs.add("data/research_paths/source_manifest.json")
    inputs.add("data/history_extension/source_manifest.json")
    preserved = [
        file
        for file in list(ROOT.glob("*.yaml")) + list((ROOT / "reports").rglob("*"))
        if file.is_file() and file != PROTOCOL and REPORT not in file.parents
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
                    "study": "relative_risk",
                    "candidate": a,
                    "control": b,
                    "score": s,
                    "horizon": 1,
                    "protocol_sha256": manifest["protocol_sha256"],
                }
                for a, b, s in CONTRASTS
            ]
        )
        qqq, spx, iv, audit = cf.load_sources(p, ROOT)
        inference.dump(OUT / "source_audit.json", audit)
        measurement = cf.measurement_audit(qqq, spx)
        inference.dump(OUT / "measurement_audit.json", measurement)
        cf.require_measurement(measurement)
        f, t = cf.build_features(qqq, spx, iv)
        f.to_parquet(OUT / "features.parquet")
        t.to_parquet(OUT / "targets.parquet")
        forecasts, fits = cm.forecast_panel(f, t, p["index"])
        forecasts.to_parquet(OUT / "forecasts.parquet")
        inference.dump(OUT / "fits.json", fits)
        metrics = evaluate(forecasts, f.index, p)
        if inference.digest(PROTOCOL) != manifest["protocol_sha256"]:
            raise ValueError("Frozen protocol changed during relative-risk run")
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
            "# Relative intraday risk experiment\n\nUNEVALUABLE: all2comparisons retained with p=1; no lead.\n"
        )
        if metrics is not None:
            inference.dump(
                REPORT / "unpublished_scored_metrics.json",
                {
                    "status": "UNPUBLISHED_DIAGNOSTIC_ONLY",
                    "not_for_inherited_inference_or_promotion": True,
                    "scored_metrics": metrics,
                },
            )
        ledger([{"event": "unevaluable", **row} for row in failure["rows"]])
        raise
    print(
        json.dumps(
            {
                "status": "SCORED_AWAITING_INDEPENDENT_VERIFICATION",
                "forecasts": len(forecasts),
                "fits": len(fits),
                "leads": metrics["leads"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
