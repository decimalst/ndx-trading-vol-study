"""Independent dependency-complete replay admission and scientific verification.

The previous attempt's mathematical verifier is reused through pure interfaces.
New dependency collection, staging, failed-attempt admission and family inference
are independent of the new producer. No previous writing entry point is called.
"""

from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime
from hashlib import sha256
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml

from . import verify_civil_quarter as frozen

ROOT = Path(__file__).resolve().parents[1]
CAPTURE_MANIFEST = "data/source_discovery/bls_plan_capture/capture_manifest.json"
MODELS = frozen.MODELS
COMPARISONS = frozen.COMPARISONS
ALL_FEATURES = frozen.ALL_FEATURES
OLD = frozen.OLD
BASE = frozen.BASE
WAVE_ALPHA = 0.05 / (16 * 17)
EFFECT = 0.005
digest = frozen.digest
preflight = frozen.preflight
verify_forecasts = frozen.verify_forecasts
augment_features = frozen.augment_features
phase_statistics = frozen.phase_statistics
same_tree = frozen.same_tree
compare_tree = frozen.compare_tree
inference_equal = frozen.inference_equal
read_snapshot = frozen.read_snapshot
table_equal = frozen.table_equal


def strict_json(payload):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result

    result = json.loads(payload, object_pairs_hook=unique)
    json.dumps(result, allow_nan=False)
    return result


def read_json_snapshot(root, name, signature):
    return strict_json(read_snapshot(root, name, signature))


def safe_path(root, name):
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or "\x00" in name
        or name.startswith("/")
        or any(part in ("", ".", "..") for part in name.split("/"))
        or PurePosixPath(name).as_posix() != name
    ):
        raise ValueError("Canonical relative POSIX source path required")
    root = Path(root).resolve()
    path = root / name
    if not path.resolve().is_relative_to(root):
        raise ValueError("Source path escapes registered root")
    return path


def valid_hash(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Unambiguous lowercase SHA256 required")
    return value


def snapshot_sources(root, files):
    snapshots = {}
    for name, expected in sorted(files.items()):
        expected = valid_hash(expected)
        payload = safe_path(root, name).read_bytes()
        if sha256(payload).hexdigest() != expected:
            raise AssertionError("Source bytes changed before immutable snapshot: " + name)
        snapshots[name] = payload
    return snapshots


def positive_integer(value):
    if (
        isinstance(value, bool)
        or not re.fullmatch(r"[1-9][0-9]*", str(value))
        or not 1 <= int(value) <= 9999
    ):
        raise ValueError("Positive integer metadata required")
    return int(value)


def collect_sources(root, old_calendar_protocol, registered_pins):
    root = Path(root)
    maps = {"metadata": {}, "runtime": {}, "documentary": {}}
    files, payloads, references, excluded = {}, {}, [], []

    def bind(owner, field, name, signature, role):
        path, signature = safe_path(root, name), valid_hash(signature)
        if name in registered_pins and registered_pins[name] != signature:
            raise AssertionError("Active declaration differs from registered pin: " + name)
        if name in files and files[name] != signature:
            raise AssertionError("Conflicting active source declarations: " + name)
        if name not in payloads:
            payload = path.read_bytes()
            if sha256(payload).hexdigest() != signature:
                raise AssertionError("Active source declaration hash mismatch: " + name)
            payloads[name] = payload
        files[name] = signature
        maps[role][name] = signature
        references.append(
            {"owner": owner, "field": field, "path": name, "sha256": signature, "role": role}
        )
        return payloads[name]

    sources = old_calendar_protocol["sources"]
    metadata = {}
    for kind in ("cpi", "nfp", "fomc", "fomc_coverage"):
        name = sources[kind]
        if name not in registered_pins:
            raise AssertionError("Selected metadata root requires a direct registered pin")
        body = bind(
            "old_calendar_protocol", "sources." + kind, name, registered_pins[name], "metadata"
        )
        if kind != "fomc":
            metadata[kind] = strict_json(body)
        else:
            reader = csv.DictReader(io.StringIO(body.decode("utf-8")))
            required = {
                "annual_schedule_year",
                "source_url",
                "source_extraction_path",
                "source_extraction_sha256",
            }
            if (
                reader.fieldnames is None
                or len(set(reader.fieldnames)) != len(reader.fieldnames)
                or not required.issubset(reader.fieldnames)
            ):
                raise ValueError("Distinct complete original annual source columns required")
            metadata[kind] = list(reader)
    bounded = 0
    first, last = (
        datetime.strptime(old_calendar_protocol["calendar"][key], "%Y-%m-%d").date()
        for key in ("source_publication_start", "source_publication_end")
    )
    if first > last:
        raise ValueError("Original publication fence reversed")
    for event in ("cpi", "nfp"):
        records = metadata[event]["records"]
        if not isinstance(records, list):
            raise ValueError("Selected original BLS record list required")
        for index, record in enumerate(records):
            url = record["source_url"]
            prefix = "cpi" if event == "cpi" else "empsit"
            match = re.fullmatch(
                r"https://www\.bls\.gov/news\.release/archives/" + prefix + r"_(\d{8})\.htm",
                url,
            )
            if match is None:
                raise ValueError("Dated original BLS source identity required")
            publication = datetime.strptime(match[1], "%m%d%Y").date()
            if not first <= publication <= last:
                if record["parse_status"] != "OUTSIDE_PUBLICATION_FENCE":
                    raise AssertionError("Out-of-fence source not explicitly excluded")
                excluded.append(
                    {
                        "event": event,
                        "record_index": index,
                        "source_url": url,
                        "parse_status": record["parse_status"],
                        "reason": "OUTSIDE_PUBLICATION_FENCE",
                    }
                )
                continue
            bounded += 1
            bind(
                sources[event],
                f"records/{index}/snapshot_path",
                record["snapshot_path"],
                record["snapshot_sha256"],
                "runtime",
            )
    grouped = {}
    for index, row in enumerate(metadata["fomc"]):
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Malformed original FOMC row")
        year = positive_integer(row["annual_schedule_year"])
        grouped.setdefault(year, []).append(row)
        bind(
            sources["fomc"],
            f"rows/{index}/source_extraction_path",
            row["source_extraction_path"],
            row["source_extraction_sha256"],
            "runtime",
        )
    coverage = metadata["fomc_coverage"]
    if not isinstance(coverage, list):
        raise ValueError("Selected original annual coverage list required")
    seen = set()
    for record in coverage:
        year = positive_integer(record["annual_schedule_year"])
        if year in seen or year not in grouped:
            raise ValueError("Unique matching original annual coverage required")
        seen.add(year)
        rows = grouped[year]
        names = {
            (row["source_url"], row["source_extraction_path"], row["source_extraction_sha256"])
            for row in rows
        }
        if (
            len(names) != 1
            or record["coverage_start_date"] != f"{year:04d}-01-01"
            or record["coverage_end_date"] != f"{year:04d}-12-31"
            or type(record["plan_count"]) is not int
            or record["plan_count"] < 1
            or record["plan_count"] != len(rows)
            or not isinstance(record["source_url"], str)
            or not record["source_url"]
            or record["source_url"] != rows[0]["source_url"]
            or record["source_extraction_sha256"] != rows[0]["source_extraction_sha256"]
        ):
            raise AssertionError("Original FOMC annual row/coverage identity differs")
    if seen != set(grouped):
        raise ValueError("Every original FOMC source year requires exact coverage")
    declared = metadata["nfp"]["artifact_sha256"][CAPTURE_MANIFEST]
    documentary = bind(
        sources["nfp"],
        "artifact_sha256/" + CAPTURE_MANIFEST,
        CAPTURE_MANIFEST,
        declared,
        "documentary",
    )
    if not isinstance(strict_json(documentary), dict):
        raise ValueError("Committed documentary capture manifest must be an object")
    return {
        "schema_version": 1,
        "files": dict(sorted(files.items())),
        "metadata_files": dict(sorted(maps["metadata"].items())),
        "runtime_files": dict(sorted(maps["runtime"].items())),
        "documentary_files": dict(sorted(maps["documentary"].items())),
        "references": sorted(
            references, key=lambda row: (row["owner"], row["field"], row["path"])
        ),
        "excluded_bls": sorted(excluded, key=lambda row: (row["event"], row["record_index"])),
        "counts": {
            "metadata_files": len(maps["metadata"]),
            "runtime_files": len(maps["runtime"]),
            "documentary_files": len(maps["documentary"]),
            "unique_files": len(files),
            "bounded_bls_records": bounded,
            "fomc_reference_rows": len(metadata["fomc"]),
            "excluded_bls_records": len(excluded),
        },
    }


def invalidate_publication(root, error):
    report = Path(root) / "reports/civil_quarter_replay"
    report.mkdir(parents=True, exist_ok=True)
    path = report / "metrics.json"
    original_bytes = path.read_bytes() if path.exists() else None
    previous, invalid_bytes = None, None
    if original_bytes is not None:
        try:
            decoded = json.loads(original_bytes)
            if isinstance(decoded, dict):
                # JSON's permissive parser accepts NaN/Infinity. These cannot
                # enter either the canonical failure record or its JSON backup.
                json.dumps(decoded, allow_nan=False)
                previous = decoded
            else:
                invalid_bytes = original_bytes
        except (UnicodeDecodeError, ValueError):
            invalid_bytes = original_bytes
    protocol_hash = previous.get("protocol_sha256") if previous else None
    message = "Independent verification failed: " + type(error).__name__ + ": " + str(error)
    failure = {
        "status": "UNEVALUABLE",
        "whole_wave_aborted": True,
        "leads": [],
        "hypothesis_count": 2,
        "cumulative_hypothesis_count": 125,
        "protocol_sha256": protocol_hash,
        "rows": [
            {
                "study": "civil_quarter_replay",
                "candidate": candidate,
                "control": control,
                "score": "proper_variance",
                "horizon": 1,
                "phases": [],
                "p_conservative": 1.0,
                "p_holm_wave": 1.0,
                "p_holm_cumulative": 1.0,
                "status": "INVALID_RUN",
                "verdict": "UNEVALUABLE",
                "error": message,
            }
            for candidate, control in COMPARISONS
        ],
    }
    serialized = json.dumps(failure, indent=2, allow_nan=False) + "\n"
    path.write_text(serialized)
    (report / "failure.json").write_text(serialized)
    (report / "results.md").write_text(
        "# SPX civil quarter-end risk increment\n\nUNEVALUABLE: independent verification failed. Both comparisons retained with p=1; no lead.\n\n"
        + message
        + "\n"
    )
    (report / "verification.json").write_text(
        json.dumps(
            {"status": "FAILED", "error": message, "protocol_sha256": protocol_hash}, indent=2
        )
        + "\n"
    )
    with (report / "trial_ledger.jsonl").open("a") as stream:
        for row in failure["rows"]:
            stream.write(
                json.dumps(
                    {"event": "verification_failed", **row}, sort_keys=True, allow_nan=False
                )
                + "\n"
            )
    backup = report / "unpublished_scored_metrics.json"
    if (
        previous is not None
        and previous.get("status") != "UNEVALUABLE"
        and not backup.exists()
    ):
        backup.write_text(
            json.dumps(
                {
                    "status": "UNPUBLISHED_DIAGNOSTIC_ONLY",
                    "not_for_inherited_inference_or_promotion": True,
                    "scored_metrics": previous,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
    invalid_backup = report / "unpublished_invalid_metrics.txt"
    if invalid_bytes is not None and not invalid_backup.exists():
        invalid_backup.write_bytes(invalid_bytes)
    return failure


def verify_with_failure_guard(root=ROOT):
    try:
        return verify(root)
    except Exception as error:
        invalidate_publication(root, error)
        raise


calendar_old = frozen.calendar_old
scalar = frozen.scalar
restore_documentary_paths = frozen.restore_documentary_paths
reconstruct_previous = frozen.reconstruct_previous


def reconstruct_calendar(root, info, pins, closure):
    root = Path(root)
    oldp = yaml.safe_load(read_snapshot(root, info["protocol"], info["protocol_sha256"]))
    calendar_old.validate_protocol(oldp)
    oldm = read_json_snapshot(
        root, info["reports"] + "/manifest.json", info["manifest_sha256"]
    )

    def saved_json(name):
        return read_json_snapshot(root, name, pins[name])

    def saved_parquet(name):
        return scalar.read_issued_snapshot(root, name, pins[name])

    with TemporaryDirectory(prefix="civil-quarter-old-source-") as directory:
        staged = Path(directory)
        staged_names = (
            set(oldm["inputs"])
            | set(oldp["comparisons"]["inherited_sources"])
            | set(closure["files"])
        )
        staged_pins = {name: pins[name] for name in staged_names}
        if any(
            staged_pins.get(name) != signature for name, signature in closure["files"].items()
        ):
            raise AssertionError("Every active source must be directly pinned before staging")
        payloads = snapshot_sources(root, staged_pins)
        for name, payload in payloads.items():
            destination = staged / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        features, targets, audit = calendar_old.reconstruct(staged, oldp)
        audit = restore_documentary_paths(audit, staged, root)
        table_equal(
            saved_parquet(info["data"] + "/features.parquet"),
            features,
            OLD,
            "Entire frozen wave8 feature table",
        )
        table_equal(
            saved_parquet(info["data"] + "/targets.parquet"),
            targets,
            ("y",),
            "Entire frozen wave8 risk target",
        )
        same_tree(
            saved_json(info["data"] + "/source_audit.json"),
            audit,
            "Original source and macro-known masks",
        )
        panel = saved_parquet(info["data"] + "/forecasts.parquet")
        fits = saved_json(info["data"] + "/fits.json")
        metrics = saved_json(info["reports"] + "/metrics.json")
        if (
            metrics.get("status") == "UNEVALUABLE"
            or metrics["protocol_sha256"] != info["protocol_sha256"]
            or metrics["evidence_class"] != oldp["evidence_class"]
        ):
            raise AssertionError("Original calendar successful metric identity differs")
        result = {
            "status": "VERIFIED",
            "protocol_sha256": info["protocol_sha256"],
            "verifier_sha256": info["verifier_sha256"],
            "raw_feature_rows_verified": len(features),
            "raw_feature_columns_verified": len(OLD),
            "source_availability_rows_verified": len(audit["plan_availability"]),
            "sources": audit["plans"],
            "forecast_reconstruction": calendar_old.verify_forecasts(
                features, targets, panel, fits, oldp
            ),
            "inference": calendar_old.verify_metrics(staged, panel, oldp, metrics),
        }
    ledger_name = info["reports"] + "/trial_ledger.jsonl"
    ledger = [
        json.loads(line)
        for line in read_snapshot(root, ledger_name, pins[ledger_name]).splitlines()
        if line
    ]
    for event, expected in (
        ("inherited", metrics["inherited_rows"]),
        ("evaluated", metrics["rows"]),
    ):
        actual = [
            {key: value for key, value in row.items() if key != "event"}
            for row in ledger
            if row["event"] == event
        ]
        same_tree(actual, expected, "Original calendar " + event + " ledger")
    registered = [row for row in ledger if row["event"] == "registered"]
    if (
        len(ledger) != 110
        or len(registered) != 2
        or [(row["candidate"], row["control"]) for row in registered]
        != list(calendar_old.COMPARISONS)
        or any(
            row["protocol_sha256"] != info["protocol_sha256"]
            or row["horizon"] != 1
            or row["score"] != "qlike"
            for row in registered
        )
    ):
        raise AssertionError("Complete original wave8 ledger identity differs")
    result["ledger_events_verified"] = {"inherited": 106, "registered": 2, "evaluated": 2}
    result["limitations"] = [
        "Original-plan source and timing replication of a previously tested broad calendar mechanism.",
        "Nominal civil windows ignore holidays and early closes; target maturity alone uses future observed sessions.",
        "Historical reuse, archival revisions, incomplete original-plan coverage and back-calculated VIX9D remain exploratory limitations.",
    ]
    return result


finite = frozen.finite
validate_panel = frozen.validate_panel
civil_support = frozen.civil_support
holm = frozen.holm


def inherited_rows(root, protocol, *, pins=None):
    root = Path(root)
    if pins is None:
        pins = json.loads((root / "reports/civil_quarter_replay/manifest.json").read_bytes())[
            "inputs"
        ]
    output = []
    for name in protocol["comparisons"]["inherited_sources"]:
        signature = pins[name]
        decoded = read_json_snapshot(root, name, signature)
        for number, row in enumerate(decoded["rows"]):
            output.append(
                {
                    "study": row.get("study", Path(name).parent.name or Path(name).stem),
                    "candidate": row["candidate"],
                    "control": row.get("control", "baseline"),
                    "horizon": row["horizon"],
                    "p_conservative": row["p_conservative"],
                    "source": name,
                    "source_sha256": signature,
                    "source_row_index": number,
                    **{key: row[key] for key in ("measure", "score") if key in row},
                }
            )
    if len(output) != 123:
        raise AssertionError("All123 inherited comparisons must remain identified")
    return output


def verify_metrics(
    root,
    panel,
    features,
    protocol,
    metrics,
    monthly_fits,
    application_n,
    *,
    admitted_inputs=None,
):
    validate_panel(panel)
    reference = pd.DatetimeIndex(features.index)
    if (
        reference.has_duplicates
        or reference.hasnans
        or not reference.is_monotonic_increasing
        or not panel.origin.isin(reference).all()
    ):
        raise AssertionError("Every score origin requires its full ordered source calendar")
    values = features.loc[panel.origin, ALL_FEATURES].to_numpy()
    if np.iscomplexobj(values) or not np.isfinite(values).all():
        raise AssertionError("All scored origins require complete real registered inputs")
    dates = pd.Series(reference, index=reference)
    for column, shift in (
        ("feature_cutoff_date", 1),
        ("target_end", -1),
        ("available_date", -1),
    ):
        if not np.array_equal(panel[column], dates.shift(shift).loc[panel.origin]):
            raise AssertionError("Exact reference-calendar score timing required")
    n = int(panel.origin.nunique())
    if (
        metrics["new_monthly_fits"] != monthly_fits
        or metrics["new_forecasts"] != len(panel)
        or metrics["common_scored_origins"] != n
        or metrics["common_application_origins"] != application_n
    ):
        raise AssertionError("Complete new monthly fit and forecast accounting differs")
    section = protocol["index"]
    union = np.zeros(len(panel), dtype=bool)
    support = {}
    for name in ("development", "evaluation"):
        first, last = section[name]
        mask = panel.origin.between(first, last)
        union |= mask
        if not panel.loc[mask, "phase"].eq(name).all():
            raise AssertionError("Literal phase membership differs")
        origins = pd.DatetimeIndex(panel.loc[mask & panel.model.eq("baseline"), "origin"])
        if len(origins) < 127:
            raise AssertionError("Fixed phase bandwidth unsupported")
        support[name] = civil_support(features.loc[origins], "phase")
    if (
        not union.all()
        or (panel.train_n < section["minimum_train"]).any()
        or (panel.available_date > section["latest_target"]).any()
        or (
            panel.loc[panel.phase.eq("development"), "available_date"]
            > section["development_target_available_by"]
        ).any()
    ):
        raise AssertionError("Fixed complete inference sample or maturity fence differs")
    support["evaluation_slices"] = []
    for start, end in section["evaluation_stability"]:
        origins = pd.DatetimeIndex(
            panel.loc[panel.model.eq("baseline") & panel.origin.between(start, end), "origin"]
        )
        support["evaluation_slices"].append(
            {"start": start, "end": end, **civil_support(features.loc[origins], "slice")}
        )
    same_tree(metrics["civil_support"], support, "Every fixed phase/slice civil class count")
    rows = metrics["rows"]
    if [(row["candidate"], row["control"]) for row in rows] != list(COMPARISONS):
        raise AssertionError("Both fixed quarter contrasts required")
    probabilities, effects = [], []
    for row in rows:
        if (
            row["study"] != "civil_quarter_replay"
            or row["horizon"] != 1
            or row["score"] != "proper_variance"
        ):
            raise AssertionError("Proper-risk comparison identity differs")
        phases = [
            phase_statistics(panel, features, row["control"], phase, code, protocol)
            for code, phase in enumerate(("development", "evaluation"))
        ]
        inference_equal(row["phases"], phases, "Independent proper-risk phase inference")
        probability = max(phase["p_conservative"] for phase in phases)
        inference_equal(
            row["p_conservative"], probability, "Both-phase conjunction probability", "p"
        )
        probabilities.append(probability)
        effects.append(
            all(phase["delta"] <= -EFFECT for phase in phases)
            and len(phases[1]["stability"]) == 2
            and all(one["delta"] < 0 for one in phases[1]["stability"])
        )
    prior = inherited_rows(root, protocol, pins=admitted_inputs)
    same_tree(metrics["inherited_rows"], prior, "Entire prior123 hypothesis identity")
    wave = holm(probabilities)
    cumulative = holm([row["p_conservative"] for row in prior] + probabilities)[-2:]
    passed = []
    for number, row in enumerate(rows):
        inference_equal(row["p_holm_wave"], wave[number], "Wave16 Holm2", "p")
        inference_equal(
            row["p_holm_cumulative"], cumulative[number], "Cumulative Holm125", "p"
        )
        one = bool(effects[number] and wave[number] < WAVE_ALPHA and cumulative[number] < 0.05)
        if row["verdict"] != ("PASSES_ALL_GATES" if one else "DOES_NOT_QUALIFY"):
            raise AssertionError("Fixed effect/statistical/stability gates differ")
        passed.append(one)
    leads = ["quarter"] if all(passed) else []
    if (
        metrics["leads"] != leads
        or metrics["hypothesis_count"] != 2
        or metrics["cumulative_hypothesis_count"] != 125
    ):
        raise AssertionError("Both-control candidate and complete family required")
    return {
        "new_hypotheses_verified": 2,
        "cumulative_hypotheses_verified": 125,
        "phase_comparisons_verified": 4,
        "bootstrap_runs_verified": 12,
        "bootstrap_draws_per_run": protocol["inference"]["bootstrap_draws"],
        "civil_support": support,
        "leads": leads,
    }


def verify_ledger(root, metrics, prior, final_event, *, signature=None):
    report = Path(root) / "reports/civil_quarter_replay"
    payload = (report / "trial_ledger.jsonl").read_bytes()
    if signature is not None and sha256(payload).hexdigest() != signature:
        raise AssertionError("Trial ledger bytes changed before decoding")
    ledger = [json.loads(line) for line in payload.splitlines() if line]
    for event, expected in (("inherited", prior), (final_event, metrics["rows"])):
        actual = [
            {key: value for key, value in row.items() if key != "event"}
            for row in ledger
            if row["event"] == event
        ]
        compare_tree(actual, expected, "Independent " + event + " trial ledger")
    registered = [row for row in ledger if row["event"] == "registered"]
    if (
        len(ledger) != 127
        or len(registered) != 2
        or len(prior) != 123
        or [(row["candidate"], row["control"]) for row in registered] != list(COMPARISONS)
        or any(
            row["protocol_sha256"] != metrics["protocol_sha256"]
            or row["study"] != "civil_quarter_replay"
            or row["horizon"] != 1
            or row["score"] != "proper_variance"
            for row in registered
        )
    ):
        raise AssertionError("Complete123inherited+2registered+2terminal ledger required")
    return {"inherited": 123, "registered": 2, final_event: 2}


normalized_difference = frozen.normalized_difference
_finite = scalar._finite


CONTRACT = {
    "study_id": "civil_quarter_replay_wave16",
    "specified_on": "2026-09-07",
    "status": "specified_before_source_complete_replay_civil_features_counts_or_fits",
    "wave": 16,
    "wave_alpha": 0.0001838235294117647,
    "objective": "Repeat the unscored civil quarter-end SPX risk question with complete "
    "cryptographically bound source staging and retain the failed registration",
    "evidence_class": "exploratory_reused_history_archival_SPX_daily_risk_civil_calendar_interaction_source_complete_replay",
    "index": {
        "asset": "SPX",
        "source_end": "2025-10-20",
        "sealed_start": "2025-11-03",
        "origin_start": "2016-01-04",
        "origin_end": "2025-10-17",
        "latest_target": "2025-10-20",
        "development": ["2016-01-04", "2019-12-31"],
        "development_target_available_by": "2019-12-31",
        "evaluation": ["2020-01-02", "2025-10-17"],
        "evaluation_stability": [["2020-01-02", "2022-12-31"], ["2023-01-01", "2025-10-17"]],
        "horizons": [1],
        "market_lag": 1,
        "minimum_train": 1000,
        "models": ["mean", "baseline", "quarter"],
        "baseline": [
            "const",
            "lrv_d",
            "lrv_w",
            "lrv_m",
            "neg_d",
            "neg_w",
            "neg_m",
            "lvix",
            "term",
            "lvvix",
            "entry_dow_1",
            "entry_dow_2",
            "entry_dow_3",
            "entry_dow_4",
            "nominal_hours",
            "cpi_plan",
            "nfp_plan",
            "fomc_plan",
            "month_2",
            "month_3",
            "month_4",
            "month_5",
            "month_6",
            "month_7",
            "month_8",
            "month_9",
            "month_10",
            "month_11",
            "month_12",
            "month_end5",
            "year_end5",
        ],
        "all_features": [
            "const",
            "lrv_d",
            "lrv_w",
            "lrv_m",
            "neg_d",
            "neg_w",
            "neg_m",
            "lvix",
            "term",
            "lvvix",
            "entry_dow_1",
            "entry_dow_2",
            "entry_dow_3",
            "entry_dow_4",
            "nominal_hours",
            "cpi_plan",
            "nfp_plan",
            "fomc_plan",
            "month_2",
            "month_3",
            "month_4",
            "month_5",
            "month_6",
            "month_7",
            "month_8",
            "month_9",
            "month_10",
            "month_11",
            "month_12",
            "month_end5",
            "year_end5",
            "quarter_end5",
        ],
    },
    "upstream": {
        "protocol": "causal_pool.yaml",
        "reports": "reports/causal_pool",
        "data": "data/causal_pool",
        "protocol_sha256": "1e329ee208612ed8db1fede873b92a938820d51a3fcf2ec583c813d37ac000ad",
        "manifest_sha256": "a775d4491b3dab27f2c36c037090d37a094dfad85c5ff67256acfa262d40f4c2",
        "verification_sha256": "0f2142d97e62e4ab1dbfc02c6a27bdf23dc02676bdd43bc41fa64e180a2d1010",
        "verifier_sha256": "cb34da06786f61e50b89f494e4071abb8d250539a52145d782cc1e122aeaa900",
        "required_status": "VERIFIED",
    },
    "feature_source": {
        "protocol": "calendar_variance.yaml",
        "reports": "reports/calendar_variance",
        "data": "data/calendar_variance",
        "protocol_sha256": "b3357641d874d22eb6bf0903049e8dcfd93df4e38b0748f717923447ff367ca3",
        "manifest_sha256": "521deb0c053c3eb724f2cd84c4644688863ec263ecce62d00c8ea129580bb4c8",
        "verification_sha256": "47f2a0a333df19d258ca95dfef9fbf403ee0ae6f6a675ec85903755b84943cb7",
        "verifier_sha256": "c213b47e91c25c7293a557064dbf8c62c583fc32bf58701cbf2f5afa61f78677",
        "required_status": "VERIFIED",
        "features": "data/calendar_variance/features.parquet",
        "targets": "data/calendar_variance/targets.parquet",
    },
    "prospectus": {
        "path": "reports/civil_quarter/NEXT_SOURCE_COMPLETE_REPLAY.md",
        "sha256": "db12906c70e9f3eb41a1dff364742fa7b45fda9db898520fc958f678faa51bbf",
    },
    "source_contract": {
        "admission": "Reconstruct complete original wave14 and wave8 VERIFIED records "
        "through frozen read-only functions with full active source "
        "dependency closure; separately prove failed wave15 terminal p1 "
        "ledger and no new outputs. Exact original manifests anchor "
        "historical inventories; never invoke old writers or "
        "expanded-current-tree old coverage.",
        "immutable_read": "Check selected metadata and all source bytes against "
        "registered hashes before decoding; stage immutable checked "
        "bytes for all original wave8 inputs, inherited metrics and "
        "the complete active document closure; normalize only "
        "documentary source_path prefixes back to originalroot; "
        "rehash every registered artifact before publication.",
        "sources": {
            "daily": "data/research_paths/spx_daily.parquet",
            "vix": "data/free_sources/raw/cboe/VIX_History.csv",
            "vix9d": "data/free_sources/raw/cboe/VIX9D_History.csv",
            "vvix": "data/free_sources/raw/cboe/VVIX_History.csv",
            "cpi": "data/source_discovery/macro_plans/cpi/ledger.json",
            "nfp": "data/source_discovery/macro_plans/nfp/ledger_verified.json",
            "fomc": "data/source_discovery/macro_plans/calendar/fomc_original_annual_plans.csv",
            "fomc_coverage": "data/source_discovery/macro_plans/calendar/fomc_annual_coverage.json",
        },
        "calendar": {
            "source_publication_start": "2010-01-01",
            "source_publication_end": "2025-10-20",
            "bls_policy": "next plan printed in preceding monthly release; "
            "original planned dates retained even if later "
            "canceled or changed",
            "bls_revision_policy": "current archives with explicit "
            "this-release reissue notices remain "
            "unadmitted for both CPI and payroll; "
            "retain original source ledger and "
            "separate pre-fit admission correction",
            "source_eligibility": "publication civil date strictly before "
            "preceding observed market-session date",
            "nominal_start": "entry civil date1600America/New_York",
            "nominal_end": "next Monday-through-Friday civil "
            "date1600America/New_York; skip weekends only",
            "duration": "nominal endpoint UTC difference in hours; includes "
            "DST elapsed-time change",
            "cpi": "eligible original CPI planned timestamp falls in "
            "open-left closed-right nominal window",
            "nfp": "eligible original payroll planned timestamp falls in "
            "open-left closed-right nominal window",
            "fomc": "eligible original annual meeting final DATE equals "
            "nominal ending date; date-only timing control, no "
            "invented statement time",
            "fomc_annual": "exactly8original planned meetings per year from "
            "selected full annual announcement published "
            "before covered year",
            "coverage": "each nominal-window calendar month requires its "
            "eligible explicit original BLS plan; full eligible "
            "annual FOMC plan required",
            "missing": "pending retrieval blocks execution; intrinsically "
            "absent or ambiguous source statements remain unknown "
            "and enter all-arm complete-sample mask; no date/year "
            "inference or zero fill",
            "limitations": "nominal window ignores holidays and early "
            "closes; neither actual nor historically planned "
            "exchange holding interval; no actual next market "
            "date enters a predictor",
            "provenance": "exact currently captured official-document tool "
            "text and extraction hashes; raw provider bytes "
            "and immutable historical web vintages unverified",
            "replication": "Legacy repository calendars already forecast "
            "next-session variance; this tests original-plan "
            "admission with stronger matched SPX controls, "
            "not a first calendar mechanism",
        },
        "measurement": "Exact frozen wave8 daily SPX target/market transformations; "
        "max(GK,1e-10) is part of that preexisting risk proxy, not the "
        "later strict paired-asset raw-GK gate",
        "missing": "Keep full bounded SPX reference calendar, strict old rolling "
        "history and exact original-plan known-month masks; no zero "
        "filling, removed macro controls, holiday repair or "
        "next-common-date labels",
        "limitations": "Reused Yahoo/Cboe archival values, unverified historical "
        "publication latency and revisions, original-plan documentary "
        "coverage, nominal civil windows and early back-calculated "
        "VIX9D persist; no source acquisition or institutional "
        "mechanism claim",
    },
    "target": {
        "formula": "max(Garman-Klass[next],1e-10)+log(raw_open[next]/raw_close[entry])^2",
        "availability": "Next actual observed SPX close, target_end=available_date; strictly "
        "positive finite known risk labels; missing stays unknown",
        "interpretation": "Native squared-log-return daily OHLC full-session risk proxy, not "
        "measured high-frequency integrated variance",
        "cohort": "All32complete predictors and exact common mature labels across everymodel; "
        "monthly firstfeaturecomplete application before futurequerylabelmask; no "
        "event-only evaluation or dropped unscored month",
    },
    "civil": {
        "month_columns": [
            "month_2",
            "month_3",
            "month_4",
            "month_5",
            "month_6",
            "month_7",
            "month_8",
            "month_9",
            "month_10",
            "month_11",
            "month_12",
        ],
        "nuisance_columns": [
            "month_2",
            "month_3",
            "month_4",
            "month_5",
            "month_6",
            "month_7",
            "month_8",
            "month_9",
            "month_10",
            "month_11",
            "month_12",
            "month_end5",
            "year_end5",
        ],
        "candidate": "quarter_end5",
        "nominal_date": "First Monday-through-Friday civil date strictly after origin; skip "
        "weekends only; no future observed prices/holidays/earlycloses",
        "month_end5": "Nominal date lies within final5 civil dates of its Gregorian month, "
        "inclusive",
        "year_end5": "month_end5 times nominal December indicator",
        "quarter_end5": "month_end5 times nominal month in March,June,September; December "
        "belongs to year-end control",
        "seasonality": "Eleven nominal-month dummies omitting January",
        "training": "Original18 retain old geometry; all13 new nuisance civil columns and "
        "candidate train-centered on exact admitted rows with fixedscale1; query "
        "means unchanged",
        "constant": "Exact all-equal new civil values center exactly zero in mathematical "
        "primitives; whole-wave scientific support and rank gates still reject "
        "unsupported empirical folds",
    },
    "support": {
        "minimum_train": 1000,
        "minimum_phase_observations": 127,
        "per_class": {
            "train": {"quarter_end5": 20, "month_end5": 20, "year_end5": 5, "month_dummy": 20},
            "phase": {"quarter_end5": 30, "month_end5": 20, "year_end5": 5, "month_dummy": 20},
            "slice": {"quarter_end5": 15, "month_end5": 20, "year_end5": 5, "month_dummy": 10},
        },
        "civil_rank": "Everytraining [const,11month,E5,Y5,G5] block must have rank15 by "
        "singularvalues>1e-10*largest; no dropping or recoding",
        "civil_rank_relative_tolerance": 1e-10,
        "novelty": "Regress centeredG5 on exact transformedBASE31 using lstsq rcond1e-12; "
        "normresidual/normcenteredG5 must exceed1e-8; fullmarket rank not required "
        "because slopes regularized",
        "novelty_lstsq_rcond": 1e-12,
        "minimum_relative_residual_norm": 1e-08,
        "timing": "Audit everymonthly training group/rank/novelty and allphase/slice groups "
        "on exact commoncohorts before any optimization; any failure aborts both "
        "hypotheses, no support-dependent deletion",
    },
    "fitting": {
        "penalty": 0.01,
        "old_scale_minimum": 1e-12,
        "new_civil_scale": 1.0,
        "baseline": "31-column normalized mean eta+exp(log(q/meanq)-eta) "
        "plus.01squaredslopes; unpenalizedintercept, old17populationmean/std, "
        "new13fixedscale1center",
        "normalization": "Trainmean=arithmetic mean of positivey; qscaled=y/trainmean; "
        "nativeforecast exp(log(trainmean)+design@scaled_beta)",
        "quarter": "Freeze baselineeta; z=G5-trainmeanG5; fitonly b via "
        "mean(eta0+bz+exp(logqscaled-eta0-bz))+.01b²; no extra intercept or "
        "jointrefit",
        "mean": "Exact arithmetic training mean on sameall32featurecomplete mature rows",
        "optimizer": "Deterministic Newton from allzero baselinecoefficients and quarterb0; "
        "atmost200states and60Armijo halvings perstep, armijo1e-4; accepted "
        "fullgradientmax<=1e-8",
        "maximum_iterations": 200,
        "maximum_backtracks": 60,
        "armijo": 0.0001,
        "gradient_tolerance": 1e-08,
        "scalar_bracket": "R=max(1,abs(initial_scalar_gradient)/.02); audit finite gradients "
        "at -R,+R enclosing zero; unique optimum follows curvature>=.02",
        "arithmetic": "Strictrealfinite designs/parameters/states/predictions; "
        "positivey/mean/scaledtargets/ratio/forecast; reject unsupported "
        "nonzeroproduct,quotient,exp underflow. Tentative invalid objectives "
        "may be rejected within the fixed Armijo schedule; accepted iterates "
        "and published forecasts cannot be repaired/clipped/restarted",
    },
    "scoring": {
        "loss": "proper_variance",
        "formula": "log(h)+y/h",
        "effect_threshold_absolute": 0.005,
        "paired_difference": "d=(hA-hB)/max(hA,hB); gap=logratio-d*(y/min(hA,hB)). For "
        "abs(d)<=.5 logratio=-log1p(-d) ifd>=0 else log1p(d); otherwise "
        "log(hA)-log(hB)",
        "arithmetic": "First validate every positive real input and finite individual score, "
        "reject unsupported quotient/product underflow. Coherence allowance64 "
        "times sum of unit(x) for loghA,y/hA,loghB,y/hB,stablegap,directgap; "
        "unit=max(eps*abs(x),abs(x)-nextafter(abs(x),0)), no unit floor",
        "effect_reference": "Absolute mean natural-log proper-score decrease.005 bothphases, "
        "not percentage of potentiallynegative rawscore or profit",
    },
    "comparisons": {
        "new_hypotheses": 2,
        "inherited_hypotheses": 123,
        "cumulative_hypotheses": 125,
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
            "reports/target_aligned/metrics.json",
            "reports/sign_memory/metrics.json",
            "reports/causal_pool/metrics.json",
            "reports/civil_quarter/metrics.json",
        ],
        "controls": ["baseline", "mean"],
        "contrasts": [
            ["quarter", "baseline", "proper_variance"],
            ["quarter", "mean", "proper_variance"],
        ],
        "candidate_gate": "Both contrasts must pass "
        "everyfixedphase/effect/stability/multiplicity gate",
    },
    "inference": {
        "blocks": [21, 63, 126],
        "hac_lags": 126,
        "minimum_phase_observations": 127,
        "bootstrap_draws": 99999,
        "seed": 20260922,
        "inference_scale": 1.0,
        "seed_rule": "seed+phase_code*10000+block; development0,evaluation1",
        "p_phase": "Maximum two-sided centered-null circular-block bootstrap and "
        "BartlettHAC126 p",
        "p_hypothesis": "Maximum development/evaluation p",
        "multiplicity": "Holm2 at.05/(16*17), separately cumulativeHolm125 at.05",
        "gate": "Bothphase proper-score delta<=-.005 againstBOTHcontrols; bothfixed "
        "evaluationslice deltas<0; waveHolm<.05/272,cumulativeHolm<.05",
        "power": "Nominal HAC80percent minimumdetectableeffect dividedby.005; "
        "ordinary5percent diagnostic, noequivalence or adjustedpowerclaim",
        "failure_rule": "Any admission/support/fit/score/verification/publication failure "
        "leaves bothregistered hypotheses UNEVALUABLEp1; retain "
        "alloldresults and diagnosticfailures, no outcome-dependent repair",
    },
    "verification": {
        "baseline_method": "trust-exact",
        "baseline_initialization": "all_zero",
        "baseline_maximum_iterations": 500,
        "independent_gradient_target": 1e-10,
        "accepted_gradient_tolerance": 1.0001e-08,
        "quarter_method": "brentq",
        "scalar_absolute_tolerance": 1e-12,
        "scalar_relative_tolerance": 1e-14,
        "scalar_maximum_iterations": 200,
        "coefficient_relative_tolerance": 1e-07,
        "coefficient_absolute_tolerance": 1e-06,
        "independent_normalized_prediction_relative_tolerance": 1e-07,
        "independent_normalized_prediction_absolute_tolerance": 1e-06,
        "saved_normalized_prediction_relative_tolerance": 1e-10,
        "saved_normalized_prediction_absolute_tolerance": 1e-12,
        "feature_relative_tolerance": 1e-10,
        "feature_absolute_tolerance": 1e-12,
        "inference_relative_tolerance": 1e-09,
        "inference_absolute_tolerance": 1e-14,
        "reconstruction": "Independent active source closure, original successful proof "
        "records and failed wave15 p1 registration, civil calendar, "
        "common maturity/support/rank/novelty, training transforms, "
        "convex fits, strict saved native prediction replay normalized "
        "by exact trainmean, primitive proper losses/stable paired "
        "gaps, four-phase inference and complete125-family ledger; no "
        "new producer import",
    },
    "outputs": {
        "data": "data/civil_quarter_replay",
        "reports": "reports/civil_quarter_replay",
        "features": "data/civil_quarter_replay/features.parquet",
        "targets": "data/civil_quarter_replay/targets.parquet",
        "forecasts": "data/civil_quarter_replay/forecasts.parquet",
        "fits": "data/civil_quarter_replay/fits.json",
        "support_audit": "data/civil_quarter_replay/support_audit.json",
        "admission": "data/civil_quarter_replay/upstream_admission.json",
        "source_closure": "data/civil_quarter_replay/source_closure.json",
    },
    "failed_attempt": {
        "protocol": "civil_quarter.yaml",
        "reports": "reports/civil_quarter",
        "data": "data/civil_quarter",
        "protocol_sha256": "8b9fdabcbaf3eae80116cc1efa02e7d32372d9a628faf74996fceefe142da109",
        "manifest_sha256": "212578e5418c5f10e8e4847bbebf335bc3dcbd7b0db63aac853a4902a5605156",
        "freeze_record_sha256": "55eefd0f50d3486034e72c2cdc5ba4d4ed8b0015f39f1d92257c911ce99021ce",
        "failure_sha256": "67eb0c458001b726f1456453388283a99f22a2508424da4fdf56d535e9b0d159",
        "admission_failure_audit_sha256": "358a1655c1266ad0bb88ae0cd0f18e183693f83245e0f5afbe95484b7b9fd897",
        "publication_audit_sha256": "2553daee409f3e7ffad2c8204a20bf28a6d3aea9669e073fc0f468f30166d4a9",
        "trial_ledger_sha256": "d44aa0c1b9480cbc9dafae5cea26c0cbfaf1edfb584dcac41f856f8f417591a5",
        "required_status": "UNEVALUABLE",
        "inherited_hypotheses": 121,
        "registered_hypotheses": 2,
        "cumulative_hypotheses": 123,
        "terminal_event": "unevaluable",
        "new_forecasts": 0,
        "new_monthly_fits": 0,
        "admission": "Independently bind the entire failed protocol, freeze/manifest "
        "and current historical inventories, exact two p1 rows with no "
        "phases/leads, 121 inherited+2registered+2unevaluable ledger "
        "events and empty private outputs; reproduce no old writer and "
        "never infer null evidence from failed source packaging",
    },
    "source_closure": {
        "audit": "reports/civil_quarter/source_dependency_audit.json",
        "audit_sha256": "bf83a9bda409729a14fed45419ad081b9969d73766a91ed6a9fb2bc16a6853bb",
        "omitted_dependencies": {
            "data/source_discovery/bls_plan_capture/capture_manifest.json": "ab2a4d167d7e93e403af682dc30aca40cd05497816e626308fa7d8caa6f7c941",
            "data/source_discovery/bls_plan_capture/cpi_09112025.web-extract.txt": "5c5880834bf10c136631c68dfd023f8276ee29e795ad49d4caed95d0646e521c",
            "data/source_discovery/bls_plan_capture/cpi_12152015.web-extract.txt": "a29d4c780e626d86167de0584172b9e84c730cf2b91815372ebdba54dea40c00",
            "data/source_discovery/bls_plan_capture/empsit_09052025.web-extract.txt": "6483aaebdd8a4e320e5ca433ceaea8a2a9dcca9fb08c90cab87f71f16b1c863e",
            "data/source_discovery/bls_plan_capture/empsit_12042015.web-extract.txt": "6bacb96bc54d0242fe5a4c34a98a0b3c3b52d6216305e92a167240eddb2b5454",
        },
        "registration": "Register both successful upstream inventories, complete "
        "failed wave15 inputs/reports/protocol and all five previously "
        "omitted declared documents before any real new closure "
        "collection or fitting",
        "active_roots": "Only selected final CPI/payroll ledgers, original annual FOMC "
        "plan table/coverage and final payroll artifact_sha256 "
        "capture-manifest binding; validate same metadata byte "
        "snapshots against registered pins before decoding",
        "runtime": "Every selected BLS source publication within the original fence "
        "including excluded/reissued/ambiguous records; all annual FOMC row "
        "references checked against coverage; enumerate complete "
        "source-reader closure, no fixed observed count or dropped document",
        "documentary": "Include "
        "data/source_discovery/bls_plan_capture/capture_manifest.json "
        "via exact final payroll artifact_sha256 binding. Retained "
        "SUPERSEDED intermediate declarations remain unchanged "
        "provenance, not active alternate hashes",
        "integrity": "Strict normalized relative paths beneath root, no "
        "absolute/traversal/symlink escape; unambiguous full SHA256 "
        "bindings; reject missing files, conflicting active declarations, "
        "current-pin disagreement or hash mismatch before text/JSON "
        "decoding",
        "snapshot": "Read and verify each staged input once into immutable bytes and "
        "write those exact snapshots at same-relative paths; preserve old "
        "parser/source gates and normalize only documentary source_path "
        "prefixes for audit comparison",
        "failure": "Any postregistration source closure/admission failure preserves "
        "both new hypotheses UNEVALUABLEp1; no outcome-dependent source "
        "deletion or gate repair",
    },
    "model_reuse": {
        "producer": {
            "src/civil_quarter_features.py": "1f04b9c191ffaec234e94e15ec2cf07aa87b262ecbd1fd3a90094c3aca554984",
            "src/civil_quarter_models.py": "9b0cc7375d7ee9b5af644559a2138158c76a603152a7fb7feb11b3bf6afa70d5",
            "src/civil_quarter_score.py": "310e80614a24cdc8e010d4bfea838cf913430f3b026e9542cbf9fc8854b1421e",
        },
        "independent_math": {
            "src/verify_civil_quarter.py": "8db7b064cc18bd22449f2c8ba5c03ca6c9a976a548d945c168ff678e8443dc4b"
        },
        "contract": "Reuse exact frozen wave15 mathematical implementations through "
        "explicit pure-function interfaces. "
        "Index/civil/support/fitting/scoring/verification numerical settings "
        "are unchanged; never patch module globals or alter prior files.",
    },
}


def validate_protocol(protocol):
    if protocol != CONTRACT:
        raise AssertionError("Complete registered replay contract differs")


def validate_upstream(root=ROOT):
    root = Path(root)
    protocol_payload = (root / "civil_quarter_replay.yaml").read_bytes()
    protocol = yaml.safe_load(protocol_payload)
    manifest_payload = (root / "reports/civil_quarter_replay/manifest.json").read_bytes()
    captured = strict_json(manifest_payload)
    if captured["protocol_sha256"] != sha256(protocol_payload).hexdigest():
        raise AssertionError("New registration must precede closure collection and admission")
    pins = {}
    for group in ("code", "inputs", "preserved"):
        scalar._pins(root, captured[group])
        for name, signature in captured[group].items():
            if name in pins and pins[name] != signature:
                raise AssertionError("Conflicting registered replay artifact hashes")
            pins[name] = signature
    source = protocol["feature_source"]
    oldp = yaml.safe_load(read_snapshot(root, source["protocol"], source["protocol_sha256"]))
    closure = collect_sources(root, oldp, captured["inputs"])
    if any(
        captured["inputs"].get(name) != signature
        for name, signature in closure["files"].items()
    ):
        raise AssertionError(
            "The full active closure must be directly registered before collection"
        )
    for name, signature in protocol["source_closure"]["omitted_dependencies"].items():
        if (
            closure["files"].get(name) != signature
            or captured["inputs"].get(name) != signature
        ):
            raise AssertionError(
                "Every fixed previously omitted declaration must be recovered exactly"
            )
    audit_name = protocol["source_closure"]["audit"]
    if pins.get(audit_name) != protocol["source_closure"]["audit_sha256"]:
        raise AssertionError("Original dependency audit identity differs")
    read_json_snapshot(root, audit_name, pins[audit_name])
    closure_name = protocol["outputs"]["source_closure"]
    closure_hash = digest(root / closure_name)
    same_tree(
        read_json_snapshot(root, closure_name, closure_hash),
        closure,
        "Independent complete active source closure",
    )

    def inventory(info):
        return {info["protocol"]} | {
            str(path.relative_to(root))
            for folder in (info["reports"], info["data"])
            for path in (root / folder).rglob("*")
            if path.is_file()
        }

    inventories = {
        key: inventory(protocol[key])
        for key in ("upstream", "feature_source", "failed_attempt")
    }
    for key in inventories:
        info = protocol[key]
        if not inventories[key].issubset(captured["inputs"]):
            raise AssertionError(
                "Entire historical protocol/report/private inventory must be pinned"
            )
        oldm = read_json_snapshot(
            root, info["reports"] + "/manifest.json", info["manifest_sha256"]
        )
        if not set(oldm["code"]).issubset(captured["code"]) or not set(
            oldm["inputs"]
        ).issubset(captured["inputs"]):
            raise AssertionError("Original code and input registration must remain complete")
        for group in ("code", "inputs", "preserved"):
            for name, signature in oldm[group].items():
                if pins.get(name) != signature:
                    raise AssertionError(
                        "Original preserved artifact missing from replay registration"
                    )
                read_snapshot(root, name, signature)
    proofs = {}
    for key, module in (("upstream", "causal_pool"), ("feature_source", "calendar_variance")):
        info = protocol[key]
        oldm = read_json_snapshot(
            root, info["reports"] + "/manifest.json", info["manifest_sha256"]
        )
        record = read_json_snapshot(
            root, info["reports"] + "/verification.json", info["verification_sha256"]
        )
        if (
            (root / info["reports"] / "failure.json").exists()
            or info["required_status"] != "VERIFIED"
            or record["status"] != "VERIFIED"
            or record["protocol_sha256"] != info["protocol_sha256"]
            or record["verifier_sha256"] != info["verifier_sha256"]
            or oldm["protocol_sha256"] != info["protocol_sha256"]
            or oldm["code"]["src/verify_" + module + ".py"] != info["verifier_sha256"]
        ):
            raise AssertionError("Successful historical verification identity differs")
        read_snapshot(root, info["protocol"], info["protocol_sha256"])
        nested = None
        if key == "upstream":
            rebuilt, nested = reconstruct_previous(root, info, pins)
        else:
            rebuilt = reconstruct_calendar(root, info, pins, closure)
        compare_tree(record, rebuilt, "Complete original " + module + " VERIFIED proof")
        proofs[key] = {
            "original_verification": rebuilt,
            "previous_manifest_entries_verified": sum(
                len(oldm[g]) for g in ("code", "inputs", "preserved")
            ),
            "pinned_previous_artifacts": len(inventories[key]),
            "protocol_sha256": info["protocol_sha256"],
            "manifest_sha256": info["manifest_sha256"],
            "verification_sha256": info["verification_sha256"],
            **({"nested_admission": nested} if nested is not None else {}),
        }
    proofs["failed_attempt"] = verify_failed_attempt(root, protocol["failed_attempt"], pins)
    for key, initial in inventories.items():
        if inventory(protocol[key]) != initial:
            raise AssertionError("Historical inventory changed during read-only admission")
    for group in ("code", "inputs", "preserved"):
        scalar._pins(root, captured[group])
    if (
        digest(root / "civil_quarter_replay.yaml") != sha256(protocol_payload).hexdigest()
        or digest(root / "reports/civil_quarter_replay/manifest.json")
        != sha256(manifest_payload).hexdigest()
        or digest(root / closure_name) != closure_hash
    ):
        raise AssertionError("Registered replay/closure identity changed during admission")
    return {
        "status": "UPSTREAM_VERIFIED_READ_ONLY",
        "prior_files_written": False,
        "input_hashes": captured["inputs"],
        "source_closure": closure,
        "proofs": proofs,
    }


def verify_failed_attempt(root, info, pins):
    """Admit the unchanged failed registration without rerunning its admission."""
    root = Path(root)
    report = info["reports"]
    if (
        info["required_status"] != "UNEVALUABLE"
        or info["inherited_hypotheses"] != 121
        or info["registered_hypotheses"] != 2
        or info["cumulative_hypotheses"] != 123
        or info["terminal_event"] != "unevaluable"
        or info["new_forecasts"] != 0
        or info["new_monthly_fits"] != 0
    ):
        raise AssertionError("Exact failed-attempt accounting contract required")
    anchors = {
        info["protocol"]: info["protocol_sha256"],
        report + "/manifest.json": info["manifest_sha256"],
        report + "/freeze_record.json": info["freeze_record_sha256"],
        report + "/failure.json": info["failure_sha256"],
        report + "/admission_failure_audit.json": info["admission_failure_audit_sha256"],
        report + "/publication_audit.json": info["publication_audit_sha256"],
        report + "/trial_ledger.jsonl": info["trial_ledger_sha256"],
    }
    for name, signature in anchors.items():
        if pins.get(name) != signature:
            raise AssertionError("Failed-attempt anchor absent from current registration")
        read_snapshot(root, name, signature)
    oldp = yaml.safe_load(read_snapshot(root, info["protocol"], info["protocol_sha256"]))
    frozen.validate_protocol(oldp)

    def saved(name):
        path = report + "/" + name
        return read_json_snapshot(root, path, pins[path])

    manifest, freeze, failure = (
        saved("manifest.json"),
        saved("freeze_record.json"),
        saved("failure.json"),
    )
    if (
        manifest["protocol_sha256"] != info["protocol_sha256"]
        or freeze["protocol_sha256"] != info["protocol_sha256"]
        or freeze["code"] != manifest["code"]
    ):
        raise AssertionError("Failed attempt original protocol/code freeze identity differs")
    for group in ("code", "inputs", "preserved"):
        for name, signature in manifest[group].items():
            if pins.get(name) != signature:
                raise AssertionError("Prior failed-attempt dependency missing or changed")
            read_snapshot(root, name, signature)
    for name, signature in freeze["prefit_design"].items():
        if pins.get(name) != signature:
            raise AssertionError("Original failed-attempt prefit design must remain pinned")
        read_snapshot(root, name, signature)
    metrics = saved("metrics.json")
    if (
        metrics != failure
        or failure["status"] != "UNEVALUABLE"
        or failure["whole_wave_aborted"] is not True
        or failure["leads"]
        or failure["hypothesis_count"] != 2
        or failure["cumulative_hypothesis_count"] != 123
        or failure["protocol_sha256"] != info["protocol_sha256"]
    ):
        raise AssertionError("Original failed family must remain canonically unevaluable")
    rows = failure["rows"]
    if [(row["candidate"], row["control"]) for row in rows] != list(COMPARISONS):
        raise AssertionError("Both original failed contrasts must remain in order")
    for row in rows:
        if (
            row["study"] != "civil_quarter"
            or row["score"] != "proper_variance"
            or row["horizon"] != 1
            or row["phases"]
            or row["status"] != "INVALID_RUN"
            or not row.get("error")
            or any(
                row[key] != 1.0
                for key in ("p_conservative", "p_holm_wave", "p_holm_cumulative")
            )
        ):
            raise AssertionError(
                "Original failed comparison cannot become a null score or candidate"
            )
    prior = frozen.inherited_rows(root, oldp, pins=manifest["inputs"])
    ledger = frozen.verify_ledger(
        root, metrics, prior, "unevaluable", signature=info["trial_ledger_sha256"]
    )
    if any(path.is_file() for path in (root / info["data"]).rglob("*")):
        raise AssertionError(
            "The failed attempt must still have no feature/fit/forecast outputs"
        )
    audit = saved("admission_failure_audit.json")
    if (
        audit["status"] != "REPRODUCED_ADMISSION_FAILURE"
        or audit["protocol_sha256"] != info["protocol_sha256"]
        or audit["entrypoint_read_only"] is not True
        or audit["producer_restarted"] is not False
        or audit["verification_guard_called"] is not False
        or audit["frozen_specification_changed"] is not False
        or audit["exception"]["type"] != "FileNotFoundError"
        or audit["new_data_outputs"]
        or audit["new_forecasts"] != 0
        or audit["new_monthly_fits"] != 0
    ):
        raise AssertionError("Original independent failure audit identity differs")
    before = audit["terminal_publication"]["terminal_files_sha256_before"]
    after = audit["terminal_publication"]["terminal_files_sha256_after"]
    required = {
        "metrics.json",
        "failure.json",
        "trial_ledger.jsonl",
        "results.md",
        "manifest.json",
        "freeze_record.json",
    }
    if before != after or not required.issubset(before):
        raise AssertionError(
            "Original failure reproduction did not preserve canonical records"
        )
    for name, signature in before.items():
        if pins.get(report + "/" + name) != signature:
            raise AssertionError("Original terminal publication hash changed")
        read_snapshot(root, report + "/" + name, signature)
    publication = saved("publication_audit.json")
    if (
        publication["status"] != "UNEVALUABLE_REPORT_AUDITED"
        or publication["protocol_sha256"] != info["protocol_sha256"]
        or publication["manifest_sha256"] != info["manifest_sha256"]
        or publication["independent_admission_failure_audit_sha256"]
        != info["admission_failure_audit_sha256"]
        or publication["new_forecasts"] != 0
        or publication["new_monthly_fits"] != 0
        or publication["ledger_events"] != 125
        or publication["ledger_event_counts"] != ledger
        or publication["current_wave_new_hypotheses"] != 2
        or publication["cumulative_hypotheses"] != 123
        or publication["leads"]
        or publication["prefit_code_and_design_pins_unchanged"] is not True
        or publication["frozen_manifest_entries_checked"]
        != {g: len(manifest[g]) for g in ("code", "inputs", "preserved")}
    ):
        raise AssertionError("Original failed-attempt publication audit differs")
    audit_pins = {
        report + "/" + name: signature
        for name, signature in publication["report_artifact_hashes"].items()
    }
    audit_pins.update(publication["count_source_verification_hashes"])
    for name, signature in audit_pins.items():
        if pins.get(name) != signature:
            raise AssertionError("Original publication-audited artifact no longer registered")
        read_snapshot(root, name, signature)
    return {
        "status": "FAILED_ATTEMPT_PRESERVED",
        "prior_files_written": False,
        "protocol_sha256": info["protocol_sha256"],
        "manifest_sha256": info["manifest_sha256"],
        "failure_sha256": info["failure_sha256"],
        "admission_failure_audit_sha256": info["admission_failure_audit_sha256"],
        "publication_audit_sha256": info["publication_audit_sha256"],
        "ledger_events_verified": ledger,
        "new_forecasts": 0,
        "new_monthly_fits": 0,
        "hypotheses_retained": 2,
        "all_new_pvalues": 1.0,
        "leads": [],
    }


def verify_manifest_coverage(root, protocol, manifest):
    root = Path(root)
    code = {
        str(path.relative_to(root))
        for folder in ("src", "tests")
        for path in (root / folder).rglob("*.py")
    }
    inputs = set(protocol["comparisons"]["inherited_sources"]) | set(
        protocol["source_closure"]["omitted_dependencies"]
    )
    for key in ("upstream", "feature_source", "failed_attempt"):
        info = protocol[key]
        old = read_json_snapshot(
            root, info["reports"] + "/manifest.json", info["manifest_sha256"]
        )
        code.update(old["code"])
        inputs.update(old["inputs"])
        inputs.add(info["protocol"])
        for folder in (info["reports"], info["data"]):
            inputs.update(
                str(path.relative_to(root))
                for path in (root / folder).rglob("*")
                if path.is_file()
            )
    preserved = {
        str(path.relative_to(root))
        for path in [*root.glob("*.yaml"), *(root / "reports").rglob("*")]
        if path.is_file()
        and path != root / "civil_quarter_replay.yaml"
        and root / "reports/civil_quarter_replay" not in path.parents
        and str(path.relative_to(root)) not in inputs
    }
    if (
        set(manifest["code"]) != code
        or set(manifest["inputs"]) != inputs
        or set(manifest["preserved"]) != preserved
        or set(manifest["inputs"]) & set(manifest["preserved"])
    ):
        raise AssertionError(
            "Exact new code, dual-upstream inputs and preserved corpus required"
        )


def verify(root=ROOT):
    root = Path(root)
    payload = (root / "civil_quarter_replay.yaml").read_bytes()
    protocol = yaml.safe_load(payload)
    validate_protocol(protocol)
    protocol_hash = sha256(payload).hexdigest()
    report, out = root / "reports/civil_quarter_replay", root / "data/civil_quarter_replay"
    manifest_payload = (report / "manifest.json").read_bytes()
    manifest_hash = sha256(manifest_payload).hexdigest()
    manifest = json.loads(manifest_payload)
    if manifest["protocol_sha256"] != protocol_hash:
        raise AssertionError("Frozen protocol and manifest differ")
    verify_manifest_coverage(root, protocol, manifest)
    for group in ("code", "inputs", "preserved"):
        scalar._pins(root, manifest[group])
    names = [
        str((out / name).relative_to(root))
        for name in (
            "upstream_admission.json",
            "source_closure.json",
            "support_audit.json",
            "features.parquet",
            "targets.parquet",
            "forecasts.parquet",
            "fits.json",
        )
    ] + [
        str((report / name).relative_to(root))
        for name in ("metrics.json", "trial_ledger.jsonl")
    ]
    snapshots = {name: digest(root / name) for name in names}

    def saved_json(name):
        return read_json_snapshot(root, name, snapshots[name])

    def saved_parquet(name):
        return scalar.read_issued_snapshot(root, name, snapshots[name])

    oldinfo = protocol["feature_source"]
    oldp = yaml.safe_load(read_snapshot(root, oldinfo["protocol"], oldinfo["protocol_sha256"]))
    closure = collect_sources(root, oldp, manifest["inputs"])
    same_tree(
        saved_json("data/civil_quarter_replay/source_closure.json"),
        closure,
        "Independent complete source closure artifact",
    )
    proof = validate_upstream(root)
    same_tree(
        saved_json("data/civil_quarter_replay/upstream_admission.json"),
        proof,
        "Dual original admission proof",
    )
    source = protocol["feature_source"]
    original = scalar.read_issued_snapshot(
        root, source["features"], manifest["inputs"][source["features"]]
    )
    targets = scalar.read_issued_snapshot(
        root, source["targets"], manifest["inputs"][source["targets"]]
    )
    features = augment_features(original)
    table_equal(
        saved_parquet("data/civil_quarter_replay/features.parquet"),
        features,
        ALL_FEATURES,
        "Independent raw civil augmentation",
    )
    table_equal(
        saved_parquet("data/civil_quarter_replay/targets.parquet"),
        targets,
        ("y",),
        "Unchanged original target table",
    )
    support = preflight(features, targets, protocol)
    same_tree(
        saved_json("data/civil_quarter_replay/support_audit.json"),
        support,
        "Every preoptimization support/rank/novelty check",
    )
    panel = saved_parquet("data/civil_quarter_replay/forecasts.parquet")
    fits = saved_json("data/civil_quarter_replay/fits.json")
    metrics = saved_json("reports/civil_quarter_replay/metrics.json")
    if (
        (report / "failure.json").exists()
        or metrics.get("status") == "UNEVALUABLE"
        or metrics["protocol_sha256"] != protocol_hash
        or metrics["evidence_class"] != protocol["evidence_class"]
    ):
        raise AssertionError("New publication failed or identity differs")
    result = {
        "status": "VERIFIED",
        "upstream_admission": {"status": proof["status"], "prior_files_written": False},
        "source_closure_verified": closure["counts"],
        "failed_attempt_preserved": proof["proofs"]["failed_attempt"],
        "raw_feature_rows_verified": len(features),
        "raw_feature_columns_verified": len(ALL_FEATURES),
        "support_audit_verified": {
            "monthly_fits": support["monthly_fits"],
            "phases": len(support["phases"]),
            "evaluation_slices": len(support["phases"][1]["slices"]),
        },
        "forecast_reconstruction": verify_forecasts(features, targets, panel, fits, protocol),
        "primitive_proper_scores_verified": len(panel),
        "paired_proper_differences_verified": int(panel.origin.nunique()) * 2,
        "inference": verify_metrics(
            root,
            panel,
            features,
            protocol,
            metrics,
            len(fits),
            support["common_application_origins"],
            admitted_inputs=manifest["inputs"],
        ),
        "ledger_events_verified": verify_ledger(
            root,
            metrics,
            metrics["inherited_rows"],
            "evaluated",
            signature=snapshots["reports/civil_quarter_replay/trial_ledger.jsonl"],
        ),
        "protocol_sha256": protocol_hash,
        "verifier_sha256": digest(Path(__file__)),
        "limitations": [
            "One staged civil interaction conditional on a newly fitted market, macro, month, ordinary month-end and December year-end baseline; no joint refit in the quarter arm.",
            "Nominal next-weekday civil arithmetic ignores holidays and early closes; unchanged target maturity uses the next actual observed SPX session.",
            "The daily floored Garman-Klass plus overnight-squared-return target remains an OHLC risk proxy; original-plan missingness, archival revisions and back-calculated VIX9D persist.",
            "Adaptively selected exploratory reuse of prior historical periods; proper-score gains are absolute natural-log-score units, not percentages, causal mechanisms or profit.",
        ],
    }
    for group in ("code", "inputs", "preserved"):
        scalar._pins(root, manifest[group])
    scalar._pins(root, snapshots)
    verify_manifest_coverage(root, protocol, manifest)
    if (
        digest(root / "civil_quarter_replay.yaml") != protocol_hash
        or digest(report / "manifest.json") != manifest_hash
    ):
        raise AssertionError("Protocol or manifest changed during verification")
    (report / "verification.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(verify_with_failure_guard(), indent=2, allow_nan=False), flush=True)
