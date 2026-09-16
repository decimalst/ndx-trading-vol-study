"""Registered signed downside-tail shape experiment and complete trial accounting."""

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
import pandas as pd
import yaml

from . import orthogonal_round2 as inference
from . import tail_shape_features as tf
from . import tail_shape_models as tm
from .international_search import diagnostics

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "tail_shape.yaml"
REPORT = ROOT / "reports/tail_shape"
OUT = ROOT / "data/tail_shape"
CONTRASTS = (
    ("skew_shape", "constant_shape", "brier"),
    ("skew_shape", "frequency", "brier"),
    ("skew_shape", "constant_shape", "nll"),
)
WAVE_ALPHA = 0.05 / (7 * 8)


def validate(p):
    i, s, c, inf = p["index"], p["shape"], p["comparisons"], p["inference"]
    if (
        p["source_contract"]["skew_sha256"] != tf.SKEW_SHA256
        or p["source_contract"]["skew_anchor"]
        != {"date": "2018-08-13", "close": 159.03, "tolerance": 0.01}
        or p["wave"] != 7
        or p["wave_alpha"] != WAVE_ALPHA
        or c["new_hypotheses"] != 3
        or c["inherited_hypotheses"] != 103
        or c["cumulative_hypotheses"] != 106
        or tuple(map(tuple, c["contrasts"])) != CONTRASTS
        or tuple(i["raw"]) != tf.RAW
        or tuple(i["baseline"]) != tf.BASE
        or tuple(i["models"]) != tf.MODELS
        or i["horizons"] != [1]
        or i["event_threshold"] != -1.5
        or i["market_lag"] != 1
        or i["minimum_train"] != 1000
        or i["minimum_train_events"] != 50
        or i["minimum_train_nonevents"] != 50
        or i["minimum_phase_events"] != 30
        or i["minimum_phase_nonevents"] != 30
        or i["minimum_slice_events"] != 15
        or i["minimum_slice_nonevents"] != 15
        or i["mean_penalty"] != 0.01
        or i["variance_penalty"] != 0.01
        or i["effect_brier_absolute"] != 0.0005
        or i["effect_nll_absolute"] != 0.005
        or i["source_end"] != "2025-10-20"
        or i["latest_target"] != "2025-10-20"
        or i["sealed_start"] != "2025-11-03"
        or i["origin_start"] != "2016-01-04"
        or i["origin_end"] != "2025-10-17"
        or i["development"] != ["2016-01-04", "2019-12-31"]
        or i["development_target_available_by"] != "2019-12-31"
        or i["evaluation"] != ["2020-01-02", "2025-10-17"]
        or i["evaluation_stability"]
        != [["2020-01-02", "2022-12-31"], ["2023-01-01", "2025-10-17"]]
        or s["degrees_of_freedom"] != 8.0
        or s["coefficient_bounds"] != [-3.0, 3.0]
        or s["slope_penalty"] != 0.01
        or s["maximum_iterations"] != 1000
        or s["maximum_line_search"] != 50
        or s["ftol"] != 1e-14
        or s["gtol"] != 1e-9
        or s["projected_gradient_tolerance"] != 1e-7
        or inf["blocks"] != [21, 63, 126]
        or inf["hac_lags"] != 126
        or inf["minimum_phase_observations"] != 127
        or inf["bootstrap_draws"] != 99999
        or inf["seed"] != 20260913
    ):
        raise ValueError("Fixed tail-shape specification differs")


def _alignment(f, t):
    dates = f.index
    if (
        not isinstance(dates, pd.DatetimeIndex)
        or dates.hasnans
        or dates.has_duplicates
        or dates.tz is not None
        or not dates.is_monotonic_increasing
        or not dates.equals(dates.normalize())
        or not dates.equals(t.index)
    ):
        raise ValueError("Unique aligned normalized observed SPX calendar required")
    for frame, name, shift in [
        (f, "feature_cutoff_date", 1),
        (t, "target_end", -1),
        (t, "available_date", -1),
    ]:
        if not frame[name].equals(pd.Series(dates, index=dates, name=name).shift(shift)):
            raise ValueError(
                "Fixed preceding-session inputs or next-session target maturity differs"
            )
    observed = t.y.notna()
    if (
        not observed.equals(t.event.notna())
        or not np.isfinite(t.loc[observed, ["y", "raw_return", "event"]]).all().all()
    ):
        raise ValueError("Finite normalized returns and event labels must share missingness")
    if not np.array_equal(
        t.loc[observed, "event"], (t.loc[observed, "y"] < -1.5).astype(float)
    ):
        raise ValueError("Candidate-independent signed event definition differs")
    expected = (t.loc[observed, "raw_return"] - f.loc[observed, "normalization_mean"]) / f.loc[
        observed, "normalization_scale"
    ]
    if not np.allclose(t.loc[observed, "y"], expected, rtol=1e-10, atol=1e-12):
        raise ValueError("Candidate-independent ex ante return normalization differs")


def _complete(f):
    return (
        np.isfinite(f.loc[:, tf.RAW]).all(axis=1)
        & np.isfinite(f.normalization_mean)
        & np.isfinite(f.normalization_scale)
        & f.normalization_scale.gt(0)
    )


def _support(events, minimum_events, minimum_nonevents, label):
    values = np.asarray(events, float)
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or not np.isin(values, [0.0, 1.0]).all()
    ):
        raise ValueError("Finite binary outcomes required for support")
    count = int(values.sum())
    other = len(values) - count
    if count < minimum_events or other < minimum_nonevents:
        raise ValueError(
            f"INSUFFICIENT_DATA: {label} has {count} events and {other} nonevents"
        )
    return {
        "n": len(values),
        "events": count,
        "nonevents": other,
        "event_rate": float(values.mean()),
    }


def training_mask(f, t, entry, config):
    _alignment(f, t)
    entry = pd.Timestamp(entry)
    if entry not in f.index:
        raise ValueError("Fit date must belong to full reference calendar")
    mask = (
        _complete(f)
        & np.isfinite(t.y)
        & np.isfinite(t.event)
        & (f.index < entry)
        & (t.available_date <= f.loc[entry, "feature_cutoff_date"])
    )
    if int(mask.sum()) < config["minimum_train"]:
        raise ValueError(f"INSUFFICIENT_DATA: {int(mask.sum())} common matured training rows")
    _support(
        t.loc[mask, "event"],
        config["minimum_train_events"],
        config["minimum_train_nonevents"],
        "training",
    )
    return mask


def eligible_entries(f, t, config):
    _alignment(f, t)
    start, end = map(pd.Timestamp, (config["origin_start"], config["origin_end"]))
    ds, de = map(pd.Timestamp, config["development"])
    es, ee = map(pd.Timestamp, config["evaluation"])
    latest = pd.Timestamp(config["latest_target"])
    if (
        not start <= ds <= de < es <= ee <= end < latest
        or config["development_target_available_by"] != config["development"][1]
    ):
        raise ValueError("Fixed phase and maturity boundaries differ")
    dates = f.index
    in_phase = ((dates >= ds) & (dates <= de)) | ((dates >= es) & (dates <= ee))
    entries = dates[_complete(f) & in_phase & (dates >= start) & (dates <= end)]
    if not len(entries):
        raise ValueError("INSUFFICIENT_DATA: no complete feature entries")
    ready = (
        pd.Series(dates.isin(entries), index=dates)
        & np.isfinite(t.y)
        & np.isfinite(t.event)
        & (t.available_date <= latest)
        & ((dates > de) | (t.available_date <= de))
    )
    return entries, ready


def support_summary(f, t, config):
    entries, ready = eligible_entries(f, t, config)
    result = {
        "feature_complete_applications": len(entries),
        "common_scored_origins": int(ready.sum()),
        "phases": [],
    }
    for name in ["development", "evaluation"]:
        first, last = config[name]
        mask = ready & (f.index >= first) & (f.index <= last)
        phase = {
            "name": name,
            **_support(
                t.loc[mask, "event"],
                config["minimum_phase_events"],
                config["minimum_phase_nonevents"],
                name,
            ),
            "slices": [],
        }
        if name == "evaluation":
            for first, last in config["evaluation_stability"]:
                sub = mask & (f.index >= first) & (f.index <= last)
                phase["slices"].append(
                    {
                        "start": first,
                        "end": last,
                        **_support(
                            t.loc[sub, "event"],
                            config["minimum_slice_events"],
                            config["minimum_slice_nonevents"],
                            "evaluation slice",
                        ),
                    }
                )
        result["phases"].append(phase)
    return result


def forecast_panel(f, t, config):
    if (
        tuple(config["models"]) != tf.MODELS
        or tuple(config["baseline"]) != tf.BASE
        or config["horizons"] != [1]
    ):
        raise ValueError("Fixed tail model family differs")
    support_summary(f, t, config)
    entries, ready = eligible_entries(f, t, config)
    rows, fits = [], []
    for month in entries.to_period("M").unique():
        application = entries[entries.to_period("M") == month]
        entry = application[0]
        mask = training_mask(f, t, entry, config)
        fit = tm.fit_predict(f.loc[mask], t.loc[mask], f.loc[application])
        cutoff = f.loc[entry, "feature_cutoff_date"]
        last = t.loc[mask, "target_end"].max()
        events = int(t.loc[mask, "event"].sum())
        fits.append(
            {
                "fit_origin": str(entry.date()),
                "fit_cutoff_date": str(cutoff.date()),
                "train_n": int(mask.sum()),
                "train_event_count": events,
                "train_nonevent_count": int(mask.sum()) - events,
                "train_first_origin": str(f.index[mask][0].date()),
                "train_last_origin": str(f.index[mask][-1].date()),
                "train_last_target": str(last.date()),
                "train_last_available": str(last.date()),
                "application_n": len(application),
                **{
                    name: fit[name]
                    for name in [
                        "moment_audit",
                        "shape_audit",
                        "transform_audit",
                        "skew_mean",
                        "skew_scale",
                        "frequency",
                    ]
                },
            }
        )
        selected = ready.loc[application].to_numpy()
        scored = application[selected]
        for model in tf.MODELS:
            row = pd.DataFrame(
                {
                    "origin": scored,
                    "model": model,
                    "horizon": 1,
                    "prediction": fit["predictions"][model][selected],
                    "fit_origin": entry,
                    "fit_cutoff_date": cutoff,
                    "train_n": int(mask.sum()),
                    "train_event_count": events,
                    "train_last_target": last,
                    "train_last_available": last,
                    "phase": np.where(
                        scored <= config["development"][1], "development", "evaluation"
                    ),
                }
            )
            for name in ["y", "raw_return", "event", "target_end", "available_date"]:
                row[name] = t.loc[scored, name].to_numpy()
            for name in ["feature_cutoff_date", "normalization_mean", "normalization_scale"]:
                row[name] = f.loc[scored, name].to_numpy()
            if model == "frequency":
                for name in ["mu", "variance", "lambda", "log_density"]:
                    row[name] = np.nan
            else:
                for name in ["mu", "variance", "lambda"]:
                    row[name] = fit["conditional"][name][model][selected]
                row["log_density"] = tm.log_density(row.y, row.mu, row.variance, row["lambda"])
            rows.append(row)
    panel = (
        pd.concat(rows, ignore_index=True)
        .sort_values(["origin", "model"])
        .reset_index(drop=True)
    )
    if panel.empty:
        raise ValueError("INSUFFICIENT_DATA: no scored tail forecasts")
    return panel, fits


def paired_inference(candidate, control, p, seed):
    candidate, control = np.asarray(candidate, float), np.asarray(control, float)
    if (
        candidate.ndim != 1
        or candidate.shape != control.shape
        or not np.isfinite([candidate, control]).all()
    ):
        raise ValueError("Finite aligned proper losses required")
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
    if len(rows) != 103:
        raise ValueError("All103 inherited comparisons required")
    return rows


def passes(row):
    phases = row["phases"]
    if len(phases) != 2 or {x["name"] for x in phases} != {"development", "evaluation"}:
        return False
    threshold = 0.0005 if row["score"] == "brier" else 0.005
    evaluation = next(x for x in phases if x["name"] == "evaluation")
    return (
        row["p_holm_wave"] < WAVE_ALPHA
        and row["p_holm_cumulative"] < 0.05
        and all(x["n"] > 0 and x["delta"] <= -threshold for x in phases)
        and len(evaluation["stability"]) == 2
        and all(x["delta"] < 0 for x in evaluation["stability"])
    )


def candidate_leads(rows):
    keys = [(x["candidate"], x["control"], x["score"]) for x in rows]
    if len(keys) != 3 or set(keys) != set(CONTRASTS):
        raise ValueError("Complete three-comparison family required")
    return ["skew_shape"] if all(passes(row) for row in rows) else []


def _calibration(probability, event):
    p, y = np.asarray(probability, float), np.asarray(event, float)
    bins = []
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10
        keep = (p >= lo) & ((p < hi) | ((i == 9) & (p <= hi)))
        bins.append(
            {
                "lower": lo,
                "upper": hi,
                "n": int(keep.sum()),
                "mean_probability": float(p[keep].mean()) if keep.any() else None,
                "event_rate": float(y[keep].mean()) if keep.any() else None,
            }
        )
    return {
        "mean_probability": float(p.mean()),
        "event_rate": float(y.mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "bins": bins,
    }


def evaluate(panel, calendar, p):
    if panel.duplicated(["origin", "model", "horizon"]).any() or set(panel.horizon) != {1}:
        raise ValueError("Unique one-session forecasts required")
    rows = []
    calibration = {}
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
            if (
                set(wide.columns) != set(tf.MODELS)
                or not np.isfinite(wide).all().all()
                or ((wide < 0) | (wide > 1)).any().any()
            ):
                raise ValueError("Complete common finite event probabilities required")
            models = {
                model: frame.loc[frame.model == model].set_index("origin").reindex(wide.index)
                for model in tf.MODELS
            }
            actual = models[control]
            for model, other in models.items():
                for column in [
                    "y",
                    "raw_return",
                    "event",
                    "target_end",
                    "available_date",
                    "fit_origin",
                    "fit_cutoff_date",
                    "train_n",
                    "train_event_count",
                    "train_last_available",
                    "feature_cutoff_date",
                    "normalization_mean",
                    "normalization_scale",
                    "phase",
                ]:
                    if not actual[column].equals(other[column]):
                        raise ValueError("Paired tail targets or fitting metadata differ")
                if model != "frequency" and (
                    not np.isfinite(other[["mu", "variance", "lambda", "log_density"]])
                    .all()
                    .all()
                    or not other.variance.gt(0).all()
                ):
                    raise ValueError(
                        "Finite density predictions and positive variance required"
                    )
            for key in ["mu", "variance"]:
                if not models["constant_shape"][key].equals(models["skew_shape"][key]):
                    raise ValueError("Shape candidate changed shared conditional moments")
            support = _support(
                actual.event,
                p["index"]["minimum_phase_events"],
                p["index"]["minimum_phase_nonevents"],
                name,
            )
            if not np.array_equal(actual.event, (actual.y < -1.5).astype(float)):
                raise ValueError("Scored event definition differs")
            for a, b in p["index"]["evaluation_stability"] if name == "evaluation" else []:
                _support(
                    actual.loc[(actual.index >= a) & (actual.index <= b), "event"],
                    p["index"]["minimum_slice_events"],
                    p["index"]["minimum_slice_nonevents"],
                    "evaluation slice",
                )
            if score == "brier":
                loss1 = (actual.event - wide[candidate]) ** 2
                loss0 = (actual.event - wide[control]) ** 2
            else:
                loss1 = -models[candidate].log_density
                loss0 = -models[control].log_density
            phase = paired_inference(loss1, loss0, p, p["inference"]["seed"] + code * 10000)
            phase.update(
                {
                    "name": name,
                    "first_origin": str(wide.index[0].date()),
                    "last_origin": str(wide.index[-1].date()),
                    "event_support": support,
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
            calibration[name] = {
                model: _calibration(wide[model], actual.event) for model in tf.MODELS
            }
        rows.append(
            {
                "study": "tail_shape",
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
        "cumulative_hypothesis_count": 106,
        "calibration": calibration,
        "protocol_sha256": inference.digest(PROTOCOL),
        "evidence_class": p["evidence_class"],
    }


def failure_metrics(error, protocol_hash):
    status = "INSUFFICIENT_DATA" if "INSUFFICIENT_DATA" in str(error) else "INVALID_RUN"
    return {
        "status": "UNEVALUABLE",
        "whole_wave_aborted": True,
        "leads": [],
        "hypothesis_count": 3,
        "cumulative_hypothesis_count": 106,
        "protocol_sha256": protocol_hash,
        "rows": [
            {
                "study": "tail_shape",
                "candidate": a,
                "control": b,
                "score": s,
                "horizon": 1,
                "p_conservative": 1.0,
                "phases": [],
                "status": status,
                "error": str(error),
            }
            for a, b, s in CONTRASTS
        ],
    }


def report(metrics):
    lines = [
        "# Signed SPX downside-tail shape",
        "",
        "Negative absolute paired loss gap means improvement. All three comparisons retained.",
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
        "Matched predicted mean and variance; an incremental parametric shape result remains exploratory.",
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
        raise ValueError("Refusing to overwrite a registered tail-shape experiment")
    modules = [
        "tests.test_tail_shape_features",
        "tests.test_tail_shape_density",
        "tests.test_tail_shape_models",
        "tests.test_tail_shape_search",
        "tests.test_tail_shape_publication",
        "tests.test_verify_tail_shape",
        "tests.test_macro_second_moment",
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
        raise RuntimeError("Prewritten tail-shape tests failed; no registration or fit")
    code = list((ROOT / "src").rglob("*.py")) + list((ROOT / "tests").rglob("*.py"))
    inputs = set(p["sources"].values())
    inputs.update(p["sources"][name] + ".manifest.json" for name in ["vix", "vix9d", "vvix"])
    inputs.add("data/research_paths/source_manifest.json")
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
                    "study": "tail_shape",
                    "candidate": a,
                    "control": b,
                    "score": s,
                    "horizon": 1,
                    "protocol_sha256": manifest["protocol_sha256"],
                }
                for a, b, s in CONTRASTS
            ]
        )
        daily, iv, audit = tf.load_sources(p, ROOT)
        f, t = tf.build_features(daily, iv)
        f.to_parquet(OUT / "features.parquet")
        t.to_parquet(OUT / "targets.parquet")
        inference.dump(OUT / "source_audit.json", audit)
        inference.dump(OUT / "support_audit.json", support_summary(f, t, p["index"]))
        forecasts, fits = forecast_panel(f, t, p["index"])
        forecasts.to_parquet(OUT / "forecasts.parquet")
        inference.dump(OUT / "fits.json", fits)
        metrics = evaluate(forecasts, f.index, p)
        if inference.digest(PROTOCOL) != manifest["protocol_sha256"]:
            raise ValueError("Frozen protocol changed during tail-shape run")
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
            "# Tail-shape experiment\n\nUNEVALUABLE: all3comparisons retained with p=1; no lead.\n"
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
