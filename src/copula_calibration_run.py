"""Registered wave28 execution, provenance, matched scores and failure retention."""

import argparse
import copy
import hashlib
import io
import json
import os
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.copula_calibration import ASSETS, CELLS, CONTRASTS, run_crossed
from src.orthogonal_round2 import holm_adjust
from src.treasury_dealer_inference import masked_mean_inference

ROOT = Path(__file__).resolve().parents[1]
REPORT = Path("reports/copula_calibration/predictive")
DATA = Path("data/model_memory_study/copula_calibration_wave28")
OLD = Path("reports/joint_copula/predictive")
OLD_DATA = Path("data/model_memory_study/joint_copula_wave27")
TERMINAL_SHA = "9dfbe34029b5b8095fd68ff0c3e1e624bd854de287449a1045b6700906742b2f"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())


def _bound_path(root, path):
    root, path = Path(root).absolute(), Path(path)
    if not path.is_absolute():
        path = root / path
    if ".." in path.parts:
        raise ValueError("Repository-contained artifact path required")
    return path, str(path.relative_to(root))


def read_bound_bytes(root, path, pins):
    """Return the exact byte snapshot checked against an earlier expected hash."""
    path, name = _bound_path(root, path)
    if name not in pins:
        raise ValueError(f"Missing expected artifact hash: {name}")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != pins[name]:
        raise ValueError(f"Pinned artifact changed: {name}")
    return payload


def save_bound_bytes(root, path, payload, pins):
    """Bind intended serialized bytes before an exclusive, private first write."""
    if not isinstance(payload, bytes):
        raise ValueError("Explicit serialized byte payload required")
    path, name = _bound_path(root, path)
    if name in pins:
        raise ValueError(f"Artifact already bound: {name}")
    expected = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    pins[name] = expected
    with os.fdopen(descriptor, "wb") as out:
        out.write(payload)
        out.flush()
        os.fsync(out.fileno())
    read_bound_bytes(root, path, pins)
    return expected


def _json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _save_bound_json(root, path, value, pins):
    return save_bound_bytes(root, path, _json_bytes(value), pins)


def _output_pins(bound):
    prefix = str(DATA) + "/"
    return {name: expected for name, expected in bound.items() if name.startswith(prefix)}


def _report_pins(root, report, bound):
    _, name = _bound_path(root, report)
    return {key: value for key, value in bound.items() if key.startswith(name + "/")}


def _check_commit(root, report, bound, *, frozen, freeze_sha256, registration_sha256):
    verify_pins(root, frozen["pins"])
    anchors = {
        str((Path(report) / "freeze.json").relative_to(root)): freeze_sha256,
        str((Path(report) / "registration.json").relative_to(root)): registration_sha256,
    }
    verify_pins(root, anchors)
    verify_pins(root, bound)
    actual = {
        str(path.relative_to(root)) for path in (Path(root) / DATA).iterdir() if path.is_file()
    }
    if actual != set(_output_pins(bound)):
        raise ValueError("Persisted output inventory changed after independent verification")


def commit_failure(root, report, error, bound, *, frozen, freeze_sha256, registration_sha256):
    """Invalidate this new attempt; never relabel changed data hashes as verified."""
    root, report = Path(root), Path(report)
    for name in ("metrics.json", "terminal.json"):
        path = report / name
        if path.exists():
            retained = path.with_name(path.stem + ".pre_invalidation.json")
            if retained.exists():
                raise ValueError("Refusing to replace retained provisional publication")
            path.rename(retained)
    metrics = failure_metrics(error, frozen["inherited_rows"])
    failure_bound = {}
    _save_bound_json(root, report / "failure.json", metrics, failure_bound)
    metrics_sha256 = _save_bound_json(root, report / "metrics.json", metrics, failure_bound)
    terminal = {
        "status": "UNEVALUABLE",
        "completed_utc": datetime.now(UTC).isoformat(),
        "error": str(error),
        "metrics_sha256": metrics_sha256,
        "output_hashes": {},
        "expected_unverified_output_hashes": _output_pins(bound),
        "expected_unverified_report_hashes": _report_pins(root, report, bound),
        "report_artifact_hashes": _report_pins(root, report, failure_bound),
        "freeze_sha256": freeze_sha256,
        "registration_sha256": registration_sha256,
        "leads": [],
        "cumulative_hypotheses": 158,
    }
    _save_bound_json(root, report / "terminal.json", terminal, failure_bound)
    verify_pins(root, failure_bound)
    return terminal


def commit_verified(
    root, report, metrics, bound, *, frozen, freeze_sha256, registration_sha256
):
    """Publish only the already verified bytes, or retain a complete failed family."""
    root, report = Path(root), Path(report)
    anchors = dict(
        frozen=frozen, freeze_sha256=freeze_sha256, registration_sha256=registration_sha256
    )
    try:
        _check_commit(root, report, bound, **anchors)
        metrics_sha256 = _save_bound_json(root, report / "metrics.json", metrics, bound)
        _check_commit(root, report, bound, **anchors)
        terminal = {
            "status": "COMPLETED_VERIFIED_MECHANISM_DIAGNOSTIC",
            "completed_utc": datetime.now(UTC).isoformat(),
            "metrics_sha256": metrics_sha256,
            "output_hashes": _output_pins(bound),
            "report_artifact_hashes": _report_pins(root, report, bound),
            "freeze_sha256": freeze_sha256,
            "registration_sha256": registration_sha256,
            "verification_scope": "Independent forecast and seven-contrast score reconstruction; descriptive_original_bridge is producer-computed context outside the score certificate.",
            "leads": [],
            "cumulative_hypotheses": 158,
        }
        _save_bound_json(root, report / "terminal.json", terminal, bound)
        _check_commit(root, report, bound, **anchors)
        return terminal
    except Exception as error:
        return commit_failure(root, report, error, bound, **anchors)


def verify_pins(root, pins):
    for name, expected in pins.items():
        if sha(Path(root) / name) != expected:
            raise ValueError(f"Pinned input changed: {name}")


def source_pins(root):
    root = Path(root)
    verify_pins(root, {str(OLD / "terminal.json"): TERMINAL_SHA})
    terminal = json.loads((root / OLD / "terminal.json").read_text())
    verify_pins(
        root,
        {
            str(OLD / "freeze_record.json"): terminal["freeze_record_sha256"],
            str(OLD / "metrics.json"): terminal["metrics_sha256"],
        },
    )
    frozen = json.loads((root / OLD / "freeze_record.json").read_text())
    pins = {}
    for mapping in [frozen[k] for k in ("code", "inputs", "preserved", "prefit")] + [
        terminal["output_hashes"],
        {
            str(OLD / name): digest
            for name, digest in terminal["report_artifact_hashes"].items()
        },
    ]:
        for name, digest in mapping.items():
            if name in pins and pins[name] != digest:
                raise ValueError("Conflicting inherited hash pins")
            pins[name] = digest
    verify_pins(root, pins)
    for file in (root / "reports/joint_copula").rglob("*"):
        if file.is_file() and "__pycache__" not in file.parts:
            key = str(file.relative_to(root))
            digest = sha(file)
            if key in pins and pins[key] != digest:
                raise ValueError("Inherited report changed")
            pins[key] = digest
    return pins


def inherit(previous):
    from src.joint_copula_score import _prior

    if (
        previous.get("status") != "COMPLETED"
        or previous.get("hypothesis_count") != 2
        or previous.get("cumulative_hypothesis_count") != 151
    ):
        raise ValueError("Completed original 2/151 family required")
    _prior(previous["inherited_rows"])
    if len(previous["rows"]) != 2 or [r["control"] for r in previous["rows"]] != [
        "gaussian_copula",
        "independence",
    ]:
        raise ValueError("Every original comparison required")
    prior = copy.deepcopy(previous["inherited_rows"])
    for index, row in enumerate(previous["rows"]):
        if not 0 <= row["p_conservative"] <= 1:
            raise ValueError("Invalid original comparison probability")
        prior.append(
            {
                **copy.deepcopy(row),
                "source": str(OLD / "metrics.json"),
                "source_sha256": sha(ROOT / OLD / "metrics.json"),
                "source_row_index": index,
            }
        )
    return prior


def failure_metrics(error, prior):
    return {
        "status": "UNEVALUABLE",
        "error": str(error),
        "whole_family_aborted": True,
        "inherited_rows": copy.deepcopy(prior),
        "hypothesis_count": 7,
        "cumulative_hypothesis_count": 158,
        "leads": [],
        "rows": [
            {
                "contrast": name,
                "p_conservative": 1.0,
                "p_holm_wave": 1.0,
                "p_holm_cumulative": 1.0,
                "adjusted_difference_detected": False,
            }
            for name in CONTRASTS
        ],
    }


def assemble_archive(applications, targets, coverage, calendar):
    calendar = pd.DatetimeIndex(calendar)
    if (
        calendar.hasnans
        or not calendar.is_unique
        or not calendar.is_monotonic_increasing
        or not targets.index.equals(calendar)
        or calendar[-1] > pd.Timestamp("2025-10-20")
    ):
        raise ValueError("Exact bounded full original calendar required")
    if set(applications.model) != {"t8_copula", "gaussian_copula", "independence"}:
        raise ValueError("All original arms required")
    parts = {
        model: frame.sort_values("origin").reset_index(drop=True)
        for model, frame in applications.groupby("model")
    }
    canonical = parts["t8_copula"].copy()
    origins = pd.DatetimeIndex(canonical.origin)
    if not origins.is_unique or not origins.isin(calendar).all():
        raise ValueError("Unique original issuance origins required")
    shared = [
        "origin",
        "phase",
        "offset",
        "fit_origin",
        "training_cutoff",
        "mu_qqq",
        "mu_spx",
        "h_qqq",
        "h_spx",
    ]
    for frame in parts.values():
        if not frame[shared].equals(canonical[shared]):
            raise ValueError("Original arms changed shared forecasts or clocks")
    positions = calendar.get_indexer(origins)
    fitpositions = calendar.get_indexer(pd.DatetimeIndex(canonical.fit_origin))
    if (
        (positions <= 0).any()
        or (fitpositions <= 0).any()
        or (positions >= len(calendar) - 1).any()
    ):
        raise ValueError("Original issuance requires exact predecessor and successor")
    if not pd.DatetimeIndex(canonical.training_cutoff).equals(calendar[fitpositions - 1]):
        raise ValueError("Original training cutoff is not preceding calendar session")
    if not np.array_equal(canonical.offset.to_numpy(), positions % 5):
        raise ValueError("Original global offsets changed")
    if "feature_cutoff_date" in canonical and not pd.DatetimeIndex(
        canonical.feature_cutoff_date
    ).equals(calendar[positions - 1]):
        raise ValueError("Original feature clock changed")
    actual = targets.loc[
        origins, ["target_end", "available_date", "y_qqq", "y_spx"]
    ].reset_index(drop=True)
    for field in ("target_end", "available_date"):
        if not pd.DatetimeIndex(actual[field]).equals(calendar[positions + 1]):
            raise ValueError("Targets must use the next actual original session")
    coverage = coverage.set_index("origin")
    if not coverage.index.is_unique or not origins.isin(coverage.index).all():
        raise ValueError("Complete unique original coverage required")
    out = pd.concat([canonical[shared], actual], axis=1)
    out["issued"] = True
    out["eligible_scored"] = coverage.loc[origins, "scored"].to_numpy(bool)
    out["old_rho_t8"] = canonical.rho.to_numpy(float)
    out["old_rho_gaussian"] = parts["gaussian_copula"].rho.to_numpy(float)
    return out


def evaluate(panel, calendar, prior, protocol):
    if len(prior) != 151:
        raise ValueError("Exactly151 inherited comparisons required")
    panel = panel.set_index("origin").sort_index()
    if not panel.index.is_unique:
        raise ValueError("Unique paired origins required")
    support = protocol["support"]
    for phase in ("development", "evaluation"):
        part = panel.loc[panel.phase == phase]
        if len(part) < support["phase_daily"] or any(
            part.offset.eq(k).sum() < support["offset_daily"] for k in range(5)
        ):
            raise ValueError("INSUFFICIENT_DATA: phase or offset support")
    for start, end in protocol["forecast"]["stability"]:
        if len(panel.loc[start:end]) < support["slice_daily"]:
            raise ValueError("INSUFFICIENT_DATA: stability support")
    rows = []
    inf = protocol["inference"]
    for contrast in CONTRASTS:
        phases = []
        for phase in ("development", "evaluation"):
            part = panel.loc[panel.phase == phase]
            start, end = protocol["forecast"][phase]
            full = calendar[(calendar >= start) & (calendar <= end)]
            values = part[f"d_{contrast}"].reindex(full).to_numpy(float)
            result = masked_mean_inference(
                values,
                full.isin(part.index),
                blocks=inf["blocks"],
                hac_lags=inf["hac_lags"],
                draws=inf["bootstrap_draws"],
                seed=inf["seed"] + inf["phase_codes"][phase] * 10000,
            )

            def group(mask, part=part, contrast=contrast):
                selected = part.loc[mask, f"d_{contrast}"]
                return {
                    "n": len(selected),
                    "mean": float(selected.mean()) if len(selected) else None,
                }

            intervals = [result["hac"]["ci95"]] + [
                r["ci95"] for r in result["block_inference"].values()
            ]
            result.update(
                name=phase,
                ci95_envelope=[min(v[0] for v in intervals), max(v[1] for v in intervals)],
                offsets=[{"offset": k, **group(part.offset == k)} for k in range(5)],
                stability=[
                    {"start": a, "end": b, **group((part.index >= a) & (part.index <= b))}
                    for a, b in protocol["forecast"]["stability"]
                ]
                if phase == "evaluation"
                else [],
                annual=[
                    {"year": int(year), **group(part.index.year == year)}
                    for year in sorted(set(full.year))
                ],
            )
            phases.append(result)
        rows.append(
            {
                "contrast": contrast,
                "phases": phases,
                "p_conservative": max(p["p_conservative"] for p in phases),
            }
        )
    wave = holm_adjust([r["p_conservative"] for r in rows])
    cumulative = holm_adjust([r["p_conservative"] for r in prior + rows])[-7:]
    for row, pw, pc in zip(rows, wave, cumulative, strict=True):
        row.update(
            p_holm_wave=float(pw),
            p_holm_cumulative=float(pc),
            adjusted_difference_detected=bool(
                row["phases"][0]["mean"] * row["phases"][1]["mean"] > 0
                and pw <= protocol["comparisons"]["wave_alpha"]
                and pc <= protocol["comparisons"]["cumulative_alpha"]
            ),
        )
    diagnostics, joint = {}, {}
    for phase in ("development", "evaluation"):
        part = panel.loc[panel.phase == phase]
        joint[phase] = {cell: float(part[f"loss_{cell}"].mean()) for cell in CELLS}
        diagnostics[phase] = {}
        for asset in ASSETS:
            diagnostics[phase][asset] = {}
            for margin in ("original", "calibrated"):
                normal = part[f"normal_{margin}_{asset}"].to_numpy(float)
                mean, variance = float(normal.mean()), float(normal.var())
                pit = part[f"pit_{margin}_{asset}"].to_numpy(float)
                diagnostics[phase][asset][margin] = {
                    "mean_loss": float(-part[f"marginal_{margin}_{asset}"].mean()),
                    "pit_lower": float(np.mean(pit < 0.025)),
                    "pit_upper": float(np.mean(pit > 0.975)),
                    "normal_mean": mean,
                    "normal_variance": variance,
                    "normal_skew": float(np.mean((normal - mean) ** 3) / variance**1.5)
                    if variance
                    else None,
                }
    return {
        "status": "COMPLETED",
        "rows": rows,
        "inherited_rows": copy.deepcopy(prior),
        "hypothesis_count": 7,
        "cumulative_hypothesis_count": 158,
        "leads": [],
        "common_scored_origins": len(panel),
        "calibration_diagnostics": diagnostics,
        "full_joint_losses": joint,
        "evidence_class": protocol["evidence_class"],
        "evidence_limitation": protocol["evidence_limitation"],
    }


def freeze(root=ROOT):
    root = Path(root)
    report, data = root / REPORT, root / DATA
    if (
        (report / "freeze.json").exists()
        or (report / "registration.json").exists()
        or data.exists()
    ):
        raise ValueError("Refusing to replace an existing study attempt")
    protocol = yaml.safe_load((root / "copula_calibration.yaml").read_text())
    prefit = json.loads((root / "reports/copula_calibration/PREFIT.json").read_text())
    if prefit["status"] != "PASS" or prefit["tests_run"] <= 0:
        raise ValueError("Passing nonempty prewritten selected tests required")
    verify_pins(root, prefit["code_sha256"])
    print("Authenticating inherited source and report hashes...", flush=True)
    pins = source_pins(root)
    for folder in ("src", "tests"):
        for path in (root / folder).glob("*.py"):
            name, expected = str(path.relative_to(root)), sha(path)
            if name in pins and pins[name] != expected:
                raise ValueError(
                    "Previously frozen source changed during snapshot preparation"
                )
            pins[name] = expected
    for path in [
        root / "copula_calibration.yaml",
        root / "reports/copula_calibration/DESIGN.md",
        root / "reports/copula_calibration/LITERATURE_REVIEW.md",
        root / "reports/copula_calibration/PREFIT.json",
        root / "reports/copula_calibration/PREFIT.log",
        root / ".gitignore",
        root / "Makefile",
    ]:
        pins[str(path.relative_to(root))] = sha(path)
    for path in (root / "reports/copula_calibration").rglob("*"):
        if (
            path.is_file()
            and path.suffix in {".md", ".log", ".json"}
            and "predictive" not in path.parts
        ):
            pins[str(path.relative_to(root))] = sha(path)
    for path in root.glob("*.yaml"):
        name = str(path.relative_to(root))
        if name not in pins:
            pins[name] = sha(path)
    prior = inherit(json.loads((root / OLD / "metrics.json").read_text()))
    data.mkdir(parents=True, mode=0o700)
    os.chmod(data, 0o700)
    snapshot = data / "code_and_input_snapshot.zip"
    files = {
        path
        for path in pins
        if path.startswith(
            ("src/", "tests/", "reports/joint_copula/", "reports/copula_calibration/")
        )
    }
    files.update(str(path.relative_to(root)) for path in root.glob("*.yaml"))
    files.update(json.loads((root / OLD / "terminal.json").read_text())["output_hashes"])
    files.update(
        [
            "copula_calibration.yaml",
            "reports/copula_calibration/DESIGN.md",
            "reports/copula_calibration/PREFIT.json",
            "reports/copula_calibration/PREFIT.log",
        ]
    )
    with zipfile.ZipFile(snapshot, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in sorted(files):
            bundle.writestr(name, read_bound_bytes(root, root / name, pins))
    os.chmod(snapshot, 0o600)
    verify_pins(root, pins)
    frozen = {
        "status": "FROZEN_BEFORE_NEW_MARKET_FITS",
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": sha(root / "copula_calibration.yaml"),
        "pins": pins,
        "inherited_rows": prior,
        "snapshot_sha256": sha(snapshot),
        "snapshot_files": len(files),
        "selected_tests": prefit["tests_run"],
    }
    anchor_pins = {}
    freeze_sha256 = _save_bound_json(root, report / "freeze.json", frozen, anchor_pins)
    _save_bound_json(
        root,
        report / "registration.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": frozen["protocol_sha256"],
            "freeze_sha256": freeze_sha256,
            "contrasts": list(CONTRASTS),
            "inherited_rows": prior,
            "hypothesis_count": 7,
            "cumulative_hypothesis_count": 158,
            "evidence_class": protocol["evidence_class"],
        },
        anchor_pins,
    )
    verify_pins(root, anchor_pins)
    print(
        "Frozen and registered all seven contrasts before new numerical execution.", flush=True
    )


def run(root=ROOT):
    root = Path(root)
    report, data = root / REPORT, root / DATA
    if (report / "terminal.json").exists() or (report / "started.json").exists():
        raise ValueError("No overwrite or unrecorded retry of registered attempt")
    frozen_payload = (report / "freeze.json").read_bytes()
    registration_payload = (report / "registration.json").read_bytes()
    freeze_sha256 = hashlib.sha256(frozen_payload).hexdigest()
    registration_sha256 = hashlib.sha256(registration_payload).hexdigest()
    frozen = json.loads(frozen_payload)
    registration = json.loads(registration_payload)
    if registration["freeze_sha256"] != freeze_sha256:
        raise ValueError("Changed registered executable freeze")
    prior = frozen["inherited_rows"]
    if (
        registration.get("contrasts") != list(CONTRASTS)
        or registration.get("inherited_rows") != prior
        or len(prior) != 151
        or registration.get("hypothesis_count") != 7
        or registration.get("cumulative_hypothesis_count") != 158
        or registration.get("protocol_sha256") != frozen["protocol_sha256"]
    ):
        raise ValueError("Exact durable seven-comparison registration required")
    anchors = dict(
        frozen=frozen, freeze_sha256=freeze_sha256, registration_sha256=registration_sha256
    )
    bound = {str(DATA / "code_and_input_snapshot.zip"): frozen["snapshot_sha256"]}
    _save_bound_json(
        root,
        report / "started.json",
        {
            "started_utc": datetime.now(UTC).isoformat(),
            "registration_sha256": registration_sha256,
        },
        bound,
    )
    try:
        verify_pins(root, frozen["pins"])
        if sha(data / "code_and_input_snapshot.zip") != frozen["snapshot_sha256"]:
            raise ValueError("Changed private code/input snapshot")
        protocol = yaml.safe_load(
            read_bound_bytes(root, root / "copula_calibration.yaml", frozen["pins"])
        )

        def original_frame(name, **kwargs):
            return pd.read_parquet(
                io.BytesIO(read_bound_bytes(root, root / OLD_DATA / name, frozen["pins"])),
                **kwargs,
            )

        applications = original_frame("applications.parquet")
        targets = original_frame("targets.parquet")
        coverage = original_frame("coverage.parquet")
        calendar = pd.DatetimeIndex(original_frame("features.parquet", columns=[]).index)
        archive = assemble_archive(applications, targets, coverage, calendar)
        print(
            "Issuing four monthly forecasting cells from mature historical issued errors...",
            flush=True,
        )
        produced = run_crossed(archive, minimum_train=protocol["forecast"]["minimum_archive"])
        for name in ("applications", "panel", "coverage"):
            save_bound_bytes(
                root, data / f"{name}.parquet", produced[name].to_parquet(index=False), bound
            )
        save_bound_bytes(
            root, data / "archive.parquet", archive.to_parquet(index=False), bound
        )
        _save_bound_json(root, data / "fits.json", produced["fits"], bound)
        verify_pins(root, bound)
        produced = {
            name: pd.read_parquet(
                io.BytesIO(read_bound_bytes(root, data / f"{name}.parquet", bound))
            )
            for name in ("applications", "panel", "coverage")
        }
        produced["fits"] = json.loads(read_bound_bytes(root, data / "fits.json", bound))
        archive = pd.read_parquet(
            io.BytesIO(read_bound_bytes(root, data / "archive.parquet", bound))
        )
        verify_pins(root, bound)
        from src.verify_copula_calibration import verify

        print(
            "Independently verifying chronology, normalization and every dependence certificate...",
            flush=True,
        )
        checked = verify(
            archive,
            produced,
            minimum_train=protocol["forecast"]["minimum_archive"],
            calendar=calendar,
        )
        _save_bound_json(root, report / "forecast_verification.json", checked, bound)
        print(
            "Scoring all seven registered paired contrasts with shared calendar resamples...",
            flush=True,
        )
        metrics = evaluate(produced["panel"], calendar, prior, protocol)
        from src.verify_copula_calibration_scores import verify_scores

        checked = verify_scores(produced["panel"], calendar, metrics, prior, protocol)
        _save_bound_json(root, report / "score_verification.json", checked, bound)
        # Saved original dependence is a descriptive bridge, not an eighth test.
        from src.joint_copula_density import log_copula

        matched = archive.set_index("origin").loc[produced["panel"].origin]
        z = (
            matched[["y_qqq", "y_spx"]].to_numpy() - matched[["mu_qqq", "mu_spx"]].to_numpy()
        ) / np.sqrt(0.75 * matched[["h_qqq", "h_spx"]].to_numpy())
        old = log_copula(z, matched.old_rho_gaussian.to_numpy(), "gaussian") - log_copula(
            z, matched.old_rho_t8.to_numpy(), "t8"
        )
        metrics["descriptive_original_bridge"] = {
            phase: {
                "n": int(matched.phase.eq(phase).sum()),
                "mean": float(old[matched.phase.eq(phase).to_numpy()].mean()),
            }
            for phase in ("development", "evaluation")
        }
        metrics["descriptive_original_bridge_verification"] = (
            "Producer-computed descriptive context; excluded from the independent score certificate."
        )
        terminal = commit_verified(root, report, metrics, bound, **anchors)
    except Exception as error:
        terminal = commit_failure(root, report, error, bound, **anchors)
        print(f"Registered family failed: {error}", flush=True)
    print(
        json.dumps(
            {
                "status": terminal["status"],
                "leads": terminal["leads"],
                "cumulative_hypotheses": terminal["cumulative_hypotheses"],
            }
        ),
        flush=True,
    )
    return terminal


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "run"])
    args = parser.parse_args()
    (freeze if args.action == "freeze" else run)()
