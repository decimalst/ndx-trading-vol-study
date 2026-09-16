"""Registered uncertainty-width and persistence experiments with complete accounting."""
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

from . import measurement_memory as mm
from . import orthogonal_round2 as inference
from .international_search import diagnostics
from .macro_search import paired_inference
from .macro_second_moment import fit_second_moment, proper_score

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "measurement_memory.yaml"
REPORT = ROOT / "reports/measurement_memory"
OUT = ROOT / "data/measurement_memory"
TARGETS = ("y_qmle", "y_rv5", "y_rv15")
CONTROLS = {"width": ("baseline", "mean"), "quality": ("baseline", "mean", "width")}
CONTRASTS = tuple((candidate, control, measure) for candidate, controls in CONTROLS.items()
                  for control in controls for measure in ("qmle", "rv5", "rv15"))
WAVE_ALPHA = 1/600


def validate(p):
    index = p["index"]
    if (p["source_contract"]["risklab_sha256"] != mm.SOURCE_SHA256
            or p["source_contract"]["fields_zero_based"] != {"qmle": 2, "ci_width": 4, "rv5": 5, "rv15": 6}
            or p["wave"] != 5 or p["wave_alpha"] != WAVE_ALPHA
            or p["comparisons"]["new_hypotheses"] != 15 or p["comparisons"]["inherited_hypotheses"] != 84
            or p["comparisons"]["cumulative_hypotheses"] != 99 or index["horizons"] != [1]
            or tuple(index["models"]) != mm.MODELS or tuple(index["baseline"]) != mm.BASE
            or index["measures"] != ["qmle", "rv5", "rv15"] or index["primary_measure"] != "qmle"
            or index["minimum_train"] != 1000 or index["penalty"] != .01
            or index["effect_threshold_absolute"] != .005 or index["measurement_lag"] != 2
            or index["market_lag"] != 1 or index["label_availability_lag"] != 2
            or index["source_end"] != "2025-10-20" or index["latest_available"] != "2025-10-20"
            or index["sealed_start"] != "2025-11-03" or index["optimizer_max_iter"] != 200
            or index["gradient_tolerance"] != 1e-8 or index["armijo"] != 1e-4
            or index["maximum_backtracks"] != 60 or p["inference"]["bootstrap_draws"] != 99999
            or p["inference"]["blocks"] != [21, 63, 126] or p["inference"]["seed"] != 20260911):
        raise ValueError("Fixed measurement-memory specification differs")
    if {candidate: tuple(controls) for candidate, controls in p["comparisons"]["controls"].items()} != CONTROLS:
        raise ValueError("Fixed candidate controls differ")


def _alignment(features, targets):
    dates = features.index
    if (not isinstance(dates, pd.DatetimeIndex) or dates.has_duplicates or dates.hasnans
            or dates.tz is not None or not dates.is_monotonic_increasing or not dates.equals(dates.normalize())
            or not dates.equals(targets.index)):
        raise ValueError("Aligned unique ordered reference dates required")
    for name, shift in (("market_cutoff_date", 1), ("measurement_cutoff_date", 2)):
        expected = pd.Series(dates, index=dates, name=name).shift(shift)
        if not features[name].equals(expected):
            raise ValueError("Fixed feature availability date differs")
    for name, shift in (("target_end", -1), ("available_date", -3)):
        expected = pd.Series(dates, index=dates, name=name).shift(shift)
        if not targets[name].equals(expected):
            raise ValueError("Target observation and maturity must preserve reference-session gaps")


def training_mask(features, targets, fit_entry, min_train=1000):
    _alignment(features, targets)
    fit_entry = pd.Timestamp(fit_entry)
    if fit_entry not in features.index or min_train < 2:
        raise ValueError("Invalid fit date or minimum sample")
    cutoff = features.loc[fit_entry, "market_cutoff_date"]
    valid = (np.isfinite(features.loc[:, mm.ALL_FEATURES]).all(axis=1)
             & np.isfinite(targets.loc[:, TARGETS]).all(axis=1) & targets.loc[:, TARGETS].gt(0).all(axis=1))
    mask = valid & (features.index < fit_entry) & (targets.available_date <= cutoff)
    if int(mask.sum()) < min_train:
        raise ValueError(f"INSUFFICIENT_DATA: {int(mask.sum())} complete matured common training rows")
    return mask


def fit_predict(features, targets, application):
    if not features.index.equals(targets.index):
        raise ValueError("Primary and alternate target ordering differs from training features")
    if (not np.isfinite(features.loc[:, mm.ALL_FEATURES]).all().all()
            or not np.isfinite(application.loc[:, mm.ALL_FEATURES]).all().all()
            or not np.isfinite(targets.loc[:, TARGETS]).all().all() or not targets.loc[:, TARGETS].gt(0).all().all()
            or not features.const.eq(1).all() or not application.const.eq(1).all()):
        raise ValueError("Finite complete common designs and positive measurements required")
    if features.loc[:, mm.ALL_FEATURES[1:]].std(ddof=0).le(1e-12).any():
        raise ValueError("INSUFFICIENT_DATA: zero-scale common input; no feature fallback")
    y = targets.y_qmle.to_numpy(float)
    prediction = {"mean": np.repeat(y.mean(), len(application))}
    audit = {"mean": {"columns": ["const"], "beta": [float(np.log(y.mean()))], "train_mean": float(y.mean()),
                       "train_n": len(y), "gradient_max_abs": 0.}}
    for model in mm.MODELS[1:]:
        extra = () if model == "baseline" else (("width",) if model == "width" else ("width", "quality_memory"))
        columns = mm.BASE[1:] + extra
        fit = fit_second_moment(features.loc[:, columns], y, application.loc[:, columns])
        prediction[model] = fit.pop("prediction")
        audit[model] = {"columns": ["const", *columns], **{
            name: value.tolist() if isinstance(value, np.ndarray) else value for name, value in fit.items()}}
    return prediction, audit


def forecast_panel(features, targets, config):
    _alignment(features, targets)
    if tuple(config["models"]) != mm.MODELS or tuple(config["baseline"]) != mm.BASE:
        raise ValueError("Fixed model designs differ")
    dates = features.index
    start, end = pd.Timestamp(config["origin_start"]), pd.Timestamp(config["origin_end"])
    dev_start, dev_end = map(pd.Timestamp, config["development"])
    ev_start, ev_end = map(pd.Timestamp, config["evaluation"])
    latest = pd.Timestamp(config["latest_available"])
    if (not start <= dev_start <= dev_end < ev_start <= ev_end <= end < latest
            or config["development_target_available_by"] != config["development"][1]):
        raise ValueError("Invalid fixed phase and maturity boundaries")
    in_phase = ((dates >= dev_start) & (dates <= dev_end)) | ((dates >= ev_start) & (dates <= ev_end))
    complete = np.isfinite(features.loc[:, mm.ALL_FEATURES]).all(axis=1)
    entries = dates[complete & in_phase & (dates >= start) & (dates <= end)]
    ready = (np.isfinite(targets.loc[:, TARGETS]).all(axis=1) & targets.loc[:, TARGETS].gt(0).all(axis=1)
             & (targets.available_date <= latest) & ((dates > dev_end) | (targets.available_date <= dev_end)))
    if not len(entries):
        raise ValueError("INSUFFICIENT_DATA: no complete entry features")
    rows, fits = [], []
    for month in entries.to_period("M").unique():
        application = entries[entries.to_period("M") == month]
        fit_entry = application[0]
        mask = training_mask(features, targets, fit_entry, config["minimum_train"])
        predictions, audits = fit_predict(features.loc[mask], targets.loc[mask], features.loc[application])
        cutoff = features.loc[fit_entry, "market_cutoff_date"]
        last_target, last_available = targets.loc[mask, "target_end"].max(), targets.loc[mask, "available_date"].max()
        fits.append({"fit_origin": str(fit_entry.date()), "fit_cutoff_date": str(cutoff.date()),
                     "train_n": int(mask.sum()), "train_first_origin": str(features.index[mask][0].date()),
                     "train_last_origin": str(features.index[mask][-1].date()), "train_last_target": str(last_target.date()),
                     "train_last_available": str(last_available.date()), "application_n": len(application), "model_audit": audits})
        selected = ready.loc[application].to_numpy()
        scored = application[selected]
        for model in mm.MODELS:
            row = pd.DataFrame({"origin": scored, "model": model, "horizon": 1,
                                "prediction": predictions[model][selected], "fit_origin": fit_entry,
                                "fit_cutoff_date": cutoff, "train_n": int(mask.sum()),
                                "train_last_target": last_target, "train_last_available": last_available,
                                "phase": np.where(scored <= dev_end, "development", "evaluation")})
            for name in (*TARGETS, "target_end", "available_date"):
                row[name] = targets.loc[scored, name].to_numpy()
            for name in ("market_cutoff_date", "measurement_cutoff_date"):
                row[name] = features.loc[scored, name].to_numpy()
            rows.append(row)
    panel = pd.concat(rows, ignore_index=True).sort_values(["origin", "model"]).reset_index(drop=True)
    if panel.empty:
        raise ValueError("INSUFFICIENT_DATA: no matured common scoring observations")
    return panel, fits


def inherited(p):
    rows = []
    for path in p["comparisons"]["inherited_sources"]:
        for row in json.loads((ROOT/path).read_text())["rows"]:
            rows.append({"study": row.get("study", path.split("/")[1]), "candidate": row["candidate"],
                         "control": row.get("control", "baseline"), "horizon": row["horizon"],
                         "p_conservative": row["p_conservative"], "source": path, "source_sha256": inference.digest(ROOT/path)})
    if len(rows) != 84:
        raise ValueError("All84 previously enumerated contrasts must remain")
    return rows


def primary_passes(row):
    phases = row["phases"]
    evaluation = [phase for phase in phases if phase["name"] == "evaluation"]
    return (len(phases) == 2 and len(evaluation) == 1 and row["p_holm_wave"] < WAVE_ALPHA
            and row["p_holm_cumulative"] < .05 and all(phase["n"] > 0 and phase["delta"] <= -.005 for phase in phases)
            and len(evaluation[0]["stability"]) == 2 and all(item["delta"] < 0 for item in evaluation[0]["stability"]))


def alternate_passes(row):
    return len(row["phases"]) == 2 and all(phase["n"] > 0 and phase["delta"] < 0 for phase in row["phases"])


def candidate_leads(rows):
    keys = [(row["candidate"], row["control"], row["measure"]) for row in rows]
    if len(keys) != 15 or set(keys) != set(CONTRASTS):
        raise ValueError("The complete15-comparison family is required")
    return [candidate for candidate in CONTROLS if all(
        primary_passes(row) if row["measure"] == "qmle" else alternate_passes(row)
        for row in rows if row["candidate"] == candidate)]


def evaluate(panel, calendar, p):
    if panel.duplicated(["origin", "model", "horizon"]).any() or set(panel.horizon) != {1}:
        raise ValueError("Unique fixed-horizon forecasts required")
    rows = []
    for candidate, control, measure in CONTRASTS:
        phases = []
        for code, name in enumerate(("development", "evaluation")):
            first, last = p["index"][name]
            frame = panel.loc[(panel.origin >= first) & (panel.origin <= last)]
            if name == "development":
                frame = frame.loc[frame.available_date <= p["index"]["development_target_available_by"]]
            wide = frame.pivot(index="origin", columns="model", values="prediction").sort_index()
            if set(wide.columns) != set(mm.MODELS) or not np.isfinite(wide).all().all():
                raise ValueError("Complete paired model forecasts required")
            actual = frame.loc[frame.model == control].set_index("origin").reindex(wide.index)
            for model in mm.MODELS:
                other = frame.loc[frame.model == model].set_index("origin").reindex(wide.index)
                for column in (*TARGETS, "target_end", "available_date", "fit_origin", "market_cutoff_date", "measurement_cutoff_date"):
                    if not actual[column].equals(other[column]):
                        raise ValueError("Paired measurement or fit-date mismatch")
            candidate_loss = proper_score(actual["y_"+measure], wide[candidate])
            control_loss = proper_score(actual["y_"+measure], wide[control])
            phase = paired_inference(candidate_loss, control_loss, p, p["inference"]["seed"]+code*10000)
            phase.update({"name": name, "first_origin": str(wide.index[0].date()), "last_origin": str(wide.index[-1].date())})
            slices = p["index"]["evaluation_stability"] if name == "evaluation" else []
            phase.update(diagnostics(wide.index, candidate_loss-control_loss, calendar, 1, slices))
            phases.append(phase)
        rows.append({"study": "measurement_memory", "candidate": candidate, "control": control, "measure": measure,
                     "horizon": 1, "phases": phases, "p_conservative": max(phase["p_conservative"] for phase in phases)})
    prior = inherited(p)
    wave = inference.holm_adjust([row["p_conservative"] for row in rows])
    cumulative = inference.holm_adjust([row["p_conservative"] for row in prior+rows])[-15:]
    for row, wp, cp in zip(rows, wave, cumulative, strict=True):
        row.update({"p_holm_wave": float(wp), "p_holm_cumulative": float(cp)})
        passed = primary_passes(row) if row["measure"] == "qmle" else alternate_passes(row)
        row["verdict"] = ("PRIMARY_GATE_PASS" if row["measure"] == "qmle" else "MEASUREMENT_SIGN_PASS") if passed else "DOES_NOT_QUALIFY"
    return {"rows": rows, "inherited_rows": prior, "leads": candidate_leads(rows), "hypothesis_count": 15,
            "cumulative_hypothesis_count": 99, "protocol_sha256": inference.digest(PROTOCOL), "evidence_class": p["evidence_class"]}


def failure_metrics(error, protocol_hash):
    status = "INSUFFICIENT_DATA" if "INSUFFICIENT_DATA" in str(error) else "INVALID_RUN"
    rows = [{"study": "measurement_memory", "horizon": 1, "candidate": candidate, "control": control,
             "measure": measure, "status": status, "error": str(error), "p_conservative": 1., "phases": []}
            for candidate, control, measure in CONTRASTS]
    return {"status": "UNEVALUABLE", "whole_wave_aborted": True, "rows": rows, "leads": [],
            "hypothesis_count": 15, "cumulative_hypothesis_count": 99, "protocol_sha256": protocol_hash}


def report(metrics):
    lines = ["# Reported measurement uncertainty and persistence", "",
             "Absolute paired proper-score differences; negative means improvement. All15comparisons retained.", "",
             "| Candidate / control / measurement | Development | Evaluation | Wave Holm p | Cumulative Holm p | Comparison gate |",
             "|---|---:|---:|---:|---:|---|"]
    for row in metrics["rows"]:
        dev, ev = row["phases"]
        lines.append(f"| {row['candidate']} / {row['control']} / {row['measure']} | {dev['delta']:+.6f} | {ev['delta']:+.6f} | "
                     f"{row['p_holm_wave']:.6f} | {row['p_holm_cumulative']:.6f} | {row['verdict']} |")
    lines += ["", "Passing candidates: "+json.dumps(metrics["leads"]), "",
              "A candidate needs every primary control and both alternate-measurement signs in both periods.",
              "Squared native SPY measurements with an assumed publication delay; historical reuse and source-vintage limitations remain.", ""]
    (REPORT/"results.md").write_text("\n".join(lines))


def ledger(rows):
    with (REPORT/"trial_ledger.jsonl").open("a") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")


def run():
    p = yaml.safe_load(PROTOCOL.read_text())
    validate(p)
    REPORT.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    if (REPORT/"manifest.json").exists():
        raise ValueError("Refusing to overwrite a registered measurement experiment")
    modules = ["tests.test_measurement_memory", "tests.test_measurement_search", "tests.test_measurement_publication", "tests.test_verify_measurement_memory",
               "tests.test_macro_second_moment", "tests.test_round2_inference"]
    checked = subprocess.run([sys.executable, "-m", "unittest", *modules, "-v"], cwd=ROOT, capture_output=True, text=True, check=False)
    (REPORT/"pre_run_checks.txt").write_text(checked.stdout+checked.stderr)
    if checked.returncode:
        raise RuntimeError("Prewritten measurement tests failed; no registration or fit")
    code = list((ROOT/"src").rglob("*.py"))+list((ROOT/"tests").rglob("*.py"))
    inputs = set(p["sources"].values())
    inputs.update(p["sources"][name]+".manifest.json" for name in ("vix", "vix9d", "vvix"))
    for path in ("data/source_discovery/quarantine/risklab_spy", "data/source_discovery/risklab_documentation"):
        inputs.update(str(file.relative_to(ROOT)) for file in (ROOT/path).rglob("*") if file.is_file())
    preserved = [file for file in list(ROOT.glob("*.yaml"))+list((ROOT/"reports").rglob("*"))
                 if file.is_file() and file != PROTOCOL and REPORT not in file.parents]
    backend = io.StringIO()
    with redirect_stdout(backend):
        np.show_config()
    manifest = {"created_utc": datetime.now(UTC).isoformat(), "protocol_sha256": inference.digest(PROTOCOL),
                "environment": {"python": sys.version, "packages": {name: version(name) for name in
                                ("numpy", "pandas", "scipy", "pyarrow", "PyYAML")}, "numpy_backend": backend.getvalue(),
                                "thread_environment": {name: os.environ.get(name) for name in
                                                       ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "LOKY_MAX_CPU_COUNT")}},
                "code": {str(file.relative_to(ROOT)): inference.digest(file) for file in sorted(code)},
                "inputs": {name: inference.digest(ROOT/name) for name in sorted(inputs)},
                "preserved": {str(file.relative_to(ROOT)): inference.digest(file) for file in sorted(preserved)}}
    inference.dump(REPORT/"manifest.json", manifest)
    metrics = None
    try:
        ledger([{"event": "inherited", **row} for row in inherited(p)])
        ledger([{"event": "registered", "study": "measurement_memory", "candidate": candidate, "control": control,
                 "measure": measure, "horizon": 1, "protocol_sha256": manifest["protocol_sha256"]}
                for candidate, control, measure in CONTRASTS])
        daily, iv, table, source_audit = mm.load_sources(p, ROOT)
        features, targets = mm.build_features(daily, iv, table)
        features.to_parquet(OUT/"features.parquet")
        targets.to_parquet(OUT/"targets.parquet")
        inference.dump(OUT/"source_audit.json", source_audit)
        forecasts, fits = forecast_panel(features, targets, p["index"])
        forecasts.to_parquet(OUT/"forecasts.parquet")
        inference.dump(OUT/"fits.json", fits)
        metrics = evaluate(forecasts, features.index, p)
        if inference.digest(PROTOCOL) != manifest["protocol_sha256"]:
            raise ValueError("Frozen protocol changed during measurement run")
        for group in ("code", "inputs", "preserved"):
            for name, expected in manifest[group].items():
                if inference.digest(ROOT/name) != expected:
                    raise ValueError("Frozen artifact changed: "+name)
        inference.dump(REPORT/"metrics.json", metrics)
        report(metrics)
        ledger([{"event": "evaluated", **row} for row in metrics["rows"]])
    except Exception as error:
        failure = failure_metrics(error, manifest["protocol_sha256"])
        inference.dump(REPORT/"metrics.json", failure)
        inference.dump(REPORT/"failure.json", failure)
        (REPORT/"results.md").write_text("# Measurement-memory experiment\n\nUNEVALUABLE: all15comparisons retained with p=1; no lead.\n")
        if metrics is not None:
            inference.dump(REPORT/"unpublished_scored_metrics.json", {"status": "UNPUBLISHED_DIAGNOSTIC_ONLY",
                           "not_for_inherited_inference_or_promotion": True, "scored_metrics": metrics})
        ledger([{"event": "unevaluable", **row} for row in failure["rows"]])
        raise
    print(json.dumps({"status": "SCORED_AWAITING_INDEPENDENT_VERIFICATION", "forecasts": len(forecasts),
                      "fits": len(fits), "leads": metrics["leads"]}), flush=True)


if __name__ == "__main__":
    run()
