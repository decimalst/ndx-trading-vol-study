"""Read-only, byte-pinned admission of the verified wave18 issued forecasts.

This module performs no source-value reconstruction, logit reconstruction,
model fitting, calibration, event construction or scoring.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from collections import Counter
from pathlib import Path, PurePosixPath

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORT = "reports/range_alert/"
DATA = "data/range_alert/"
DEFAULT_EXPECTED = {
    "range_alert.yaml": "ff7b2a7fee497da11f1f88613a17314d3b92d683f506bc4cc51d5089d2bc16a3",
    REPORT
    + "manifest.json": "3390dfe5a8cae1d18951625fafb7cdbd5ba6693a32a3158d6df182c2856af895",
    REPORT
    + "freeze_record.json": "a0a00285aab3e3cb233ee34f62c944f693444ea0a830ca0812a65a25b295cb7f",
    REPORT
    + "verification.json": "de5391c863c1008248b8ddfbc0b2c945126a5a8b36154cc6bfcdc980848bf873",
    REPORT
    + "publication_audit.json": "d7d46dc66648c959d494c57a79feba9d8a7c4c1010ece89d4adde1fca449562e",
    REPORT
    + "publication_review.json": "38c661b97184eff4aff9792ab4891d2d9f13a814ef287b58fe363067beed16bf",
    REPORT
    + "NEXT_RESEARCH_DIRECTION.md": "8dc17f9701ebcdd05dfdaca219bb371d4c85fb432353c16d6fc777975e3d4106",
}
ANCHOR_PATHS = tuple(sorted(DEFAULT_EXPECTED))
OUTPUT_PATHS = tuple(
    sorted(
        [
            DATA + name
            for name in (
                "upstream_admission.json",
                "source_audit.json",
                "support_audit.json",
                "features.parquet",
                "targets.parquet",
                "forecasts.parquet",
                "states.parquet",
                "fits.json",
            )
        ]
        + [REPORT + "metrics.json", REPORT + "trial_ledger.jsonl"]
    )
)
GROUPS = ("code", "inputs", "preserved", "existing_artifacts_sha256")


def _name(value):
    if (
        type(value) is not str
        or not value
        or "\\" in value
        or "\x00" in value
        or PurePosixPath(value).is_absolute()
        or any(p in (".", "..") for p in value.split("/"))
        or PurePosixPath(value).as_posix() != value
    ):
        raise ValueError("Canonical relative path required")
    return value


def _signature(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Lowercase SHA-256 hash required")
    return value


def _mapping(value):
    if type(value) is not dict:
        raise ValueError("Object containing path/hash references required")
    return {_name(k): _signature(v) for k, v in value.items()}


def _integer(value, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError("Literal integer count required")
    return value


def _same(left, right, label):
    if type(left) is not type(right):
        raise ValueError("Typed identity differs: " + label)
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError("Object identity differs: " + label)
        for key in left:
            _same(left[key], right[key], label + "." + key)
    elif isinstance(left, list):
        if len(left) != len(right):
            raise ValueError("List identity differs: " + label)
        for a, b in zip(left, right, strict=True):
            _same(a, b, label)
    elif left != right:
        raise ValueError("Identity differs: " + label)


def _json(payload, *, object_required=True):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError("Duplicate JSON key")
            out[key] = value
        return out

    def invalid(value):
        raise ValueError("Nonfinite JSON value: " + value)

    def real(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Nonfinite JSON number")
        return number

    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=invalid,
        parse_float=real,
    )
    if object_required and type(value) is not dict:
        raise ValueError("JSON object required")
    return value


class _Closure:
    def __init__(self, root, expected):
        self.root = Path(root).resolve()
        self.anchors = _mapping(DEFAULT_EXPECTED if expected is None else expected)
        if set(self.anchors) != set(ANCHOR_PATHS):
            raise ValueError("Exact seven wave18 anchor paths required")
        self.files = {}
        self.payloads = {}
        self.manifests = {}
        self.add(self.anchors)

    def path(self, name):
        parts = PurePosixPath(_name(name)).parts
        path = self.root
        for part in parts:
            path = path / part
            if path.is_symlink():
                raise ValueError("Source symlink is not admitted: " + name)
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("Path escapes admitted root")
        return path

    def add(self, mapping):
        for name, signature in _mapping(mapping).items():
            if name in self.files and self.files[name] != signature:
                raise ValueError("Conflicting source hash: " + name)
            self.files[name] = signature

    def read(self, name, *, keep=True, fresh=False):
        if not fresh and name in self.payloads:
            return self.payloads[name]
        payload = self.path(name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != self.files[name]:
            raise ValueError("Pinned source hash mismatch before decoding: " + name)
        if keep and not fresh:
            self.payloads[name] = payload
        return payload

    def json(self, name):
        return _json(self.read(name))

    def no_failure(self):
        if (self.root / (REPORT + "failure.json")).exists():
            raise ValueError("Canonical upstream failure blocks admission")

    def recheck(self):
        self.no_failure()
        for name in sorted(self.files):
            self.read(name, keep=False, fresh=True)
        self.no_failure()


def _collect(root, expected):
    c = _Closure(root, expected)
    c.no_failure()
    # All anchor hashes precede parsing even their documentary JSON.
    for name in ANCHOR_PATHS:
        c.read(name)
    protocol_hash = c.anchors["range_alert.yaml"]
    manifest = c.json(REPORT + "manifest.json")
    freeze = c.json(REPORT + "freeze_record.json")
    verification = c.json(REPORT + "verification.json")
    publication = c.json(REPORT + "publication_audit.json")
    review = c.json(REPORT + "publication_review.json")
    for label, obj in (
        ("manifest", manifest),
        ("freeze", freeze),
        ("verification", verification),
        ("publication", publication),
        ("review", review),
    ):
        _same(obj["protocol_sha256"], protocol_hash, label + " protocol")
    _same(verification["status"], "VERIFIED", "verification status")
    _same(publication["status"], "VERIFIED_REPORT_AUDITED", "publication status")
    _same(review["status"], "APPROVED_VERIFIED_INTERPRETATION", "review status")
    _same(review["figure_visually_reviewed"], True, "visual review")
    for key, filename in (
        ("manifest_sha256", "manifest.json"),
        ("freeze_sha256", "freeze_record.json"),
        ("verification_sha256", "verification.json"),
    ):
        _same(publication[key], c.anchors[REPORT + filename], key)
    for group in ("code", "inputs", "preserved"):
        _mapping(manifest[group])
    _same(freeze["code"], manifest["code"], "complete frozen code inventory")
    c.add(freeze["prefit_design"])
    c.add({REPORT + "full_repository_tests.txt": freeze["checks"]["full_log_sha256"]})
    _integer(freeze["checks"]["full_repository_tests"])
    verifier_hash = manifest["code"]["src/verify_range_alert.py"]
    _same(verification["verifier_sha256"], verifier_hash, "verifier code")
    _same(publication["verifier_sha256"], verifier_hash, "publication verifier")
    outputs = _mapping(verification["verified_output_hashes"])
    if set(outputs) != set(OUTPUT_PATHS):
        raise ValueError("Exact ten verified output paths required")
    _same(publication["verified_output_hashes"], outputs, "published outputs")
    c.add(outputs)
    artifacts = {
        REPORT + _name(name): _signature(signature)
        for name, signature in publication["report_artifact_hashes"].items()
    }
    required = {
        REPORT + "SUMMARY.md",
        REPORT + "NEXT_RESEARCH_DIRECTION.md",
        REPORT + "publication_review.json",
        REPORT + "verification.json",
        REPORT + "metrics.json",
        REPORT + "trial_ledger.jsonl",
    }
    if not required.issubset(artifacts):
        raise ValueError("Complete publication identity references required")
    c.add(artifacts)
    _same(review["summary_sha256"], artifacts[REPORT + "SUMMARY.md"], "reviewed summary")
    c.add(review["document_hashes"])
    c.add(review["figure_hashes"])
    while True:
        pending = sorted(
            name
            for name in c.files
            if re.fullmatch(r"reports/[^/]+/manifest\.json", name) and name not in c.manifests
        )
        if not pending:
            break
        for name in pending:
            one = c.json(name)
            counts = {}
            for group in GROUPS:
                references = _mapping(one.get(group, {}))
                c.add(references)
                counts[group] = len(references)
            c.manifests[name] = counts
    entries = {g: len(manifest[g]) for g in ("code", "inputs", "preserved")}
    _same(publication["manifest_entries_checked"], entries, "publication inventory")
    _same(
        verification["artifact_hashes_checked"],
        {**entries, "outputs": 10},
        "verified inventory",
    )
    forecast = verification["forecast_reconstruction"]
    counts = {
        "forecasts": _integer(forecast["forecasts_verified"]),
        "monthly_fits": _integer(forecast["monthly_fits_verified"]),
        "common_scored_origins": _integer(forecast["common_scored_origins"]),
        "common_application_origins": _integer(forecast["common_application_origins"]),
        "bounded_reference_rows": _integer(
            verification["source_reconstruction"]["bounded_reference_rows"]
        ),
        "cumulative_hypotheses": 129,
    }
    if (
        counts["forecasts"] != 3 * counts["common_scored_origins"]
        or counts["common_application_origins"] < counts["common_scored_origins"]
        or counts["bounded_reference_rows"] < counts["common_application_origins"]
    ):
        raise ValueError("Original common forecast/application counts differ")
    for key, value in (
        ("current_wave_verified_forecasts", counts["forecasts"]),
        ("current_wave_verified_monthly_fits", counts["monthly_fits"]),
        ("current_wave_verified_applications", counts["common_application_origins"]),
        ("current_wave_scored_origins", counts["common_scored_origins"]),
        ("cumulative_hypotheses", 129),
    ):
        _same(publication[key], value, key)
    for key, value in (
        ("new_hypotheses_verified", 2),
        ("cumulative_hypotheses_verified", 129),
    ):
        _same(verification["inference"][key], value, key)
    # Hash the whole closure before decoding any output values. Keep only
    # metadata and the ten output payloads; opaque old source files are not parsed.
    for name in sorted(c.files):
        c.read(name, keep=name in OUTPUT_PATHS)
    c.audit = {
        "status": "PINNED_WAVE18_VERIFIED_OUTPUTS",
        "anchors": dict(sorted(c.anchors.items())),
        "files": dict(sorted(c.files.items())),
        "manifest_paths": sorted(c.manifests),
        "manifest_groups": dict(sorted(c.manifests.items())),
        "verified_output_hashes": dict(sorted(outputs.items())),
        "publication_artifact_hashes": dict(sorted(artifacts.items())),
        "counts": {
            "files": len(c.files),
            "manifests": len(c.manifests),
            "verified_outputs": len(outputs),
            "publication_artifacts": len(artifacts),
        },
        "upstream_identity": {
            "protocol_sha256": protocol_hash,
            "manifest_sha256": c.anchors[REPORT + "manifest.json"],
            "freeze_sha256": c.anchors[REPORT + "freeze_record.json"],
            "verification_sha256": c.anchors[REPORT + "verification.json"],
            "verifier_sha256": verifier_hash,
            "publication_audit_sha256": c.anchors[REPORT + "publication_audit.json"],
            "publication_review_sha256": c.anchors[REPORT + "publication_review.json"],
            "prospectus_sha256": c.anchors[REPORT + "NEXT_RESEARCH_DIRECTION.md"],
        },
        "upstream_verified_counts": counts,
        "historical_models_refitted": False,
        "source_values_reparsed": False,
        "loaded_from_checked_snapshots": True,
    }
    return c


def collect_input_pins(root=ROOT, expected=None):
    """Enumerate and verify metadata/file identity without output-value decoding."""
    closure = _collect(root, expected)
    closure.recheck()
    return closure.audit["files"]


def admit_upstream(root=ROOT, expected=None, registered_pins=None):
    """Return ``(deterministic_audit, original_loaded_tables_and_records)``.

    Extra registration pins are allowed; every admitted closure member must
    retain its exact hash. All ten original outputs are checked before decoding.
    """
    c = _collect(root, expected)
    if registered_pins is not None:
        for name, signature in c.files.items():
            if registered_pins.get(name) != signature:
                raise ValueError("Registered input hash missing or changed: " + name)
    loaded = {"protocol": yaml.safe_load(c.read("range_alert.yaml"))}
    if type(loaded["protocol"]) is not dict:
        raise ValueError("Original protocol object required")
    for key in ("features", "targets", "forecasts", "states"):
        loaded[key] = pd.read_parquet(io.BytesIO(c.read(DATA + key + ".parquet")))
    for key in ("upstream_admission", "source_audit", "support_audit"):
        loaded[key] = _json(c.read(DATA + key + ".json"))
    loaded["fits"] = _json(c.read(DATA + "fits.json"), object_required=False)
    if type(loaded["fits"]) is not list:
        raise ValueError("Original fit list required")
    loaded["metrics"] = _json(c.read(REPORT + "metrics.json"))
    loaded["ledger"] = [
        _json(line) for line in c.read(REPORT + "trial_ledger.jsonl").splitlines()
    ]
    counts = c.audit["upstream_verified_counts"]
    for key, expected_count in (
        ("features", counts["bounded_reference_rows"]),
        ("targets", counts["bounded_reference_rows"]),
        ("forecasts", counts["forecasts"]),
        ("states", counts["common_application_origins"]),
        ("fits", counts["monthly_fits"]),
    ):
        _same(len(loaded[key]), expected_count, "loaded " + key + " count")
    metrics = loaded["metrics"]
    if metrics.get("status") == "UNEVALUABLE" or metrics.get("whole_wave_aborted") is True:
        raise ValueError("Canonical unevaluable metrics cannot be admitted")
    for key, value in (
        ("protocol_sha256", c.audit["upstream_identity"]["protocol_sha256"]),
        ("hypothesis_count", 2),
        ("cumulative_hypothesis_count", 129),
        ("new_forecasts", counts["forecasts"]),
        ("new_monthly_fits", counts["monthly_fits"]),
        ("common_scored_origins", counts["common_scored_origins"]),
        ("common_application_origins", counts["common_application_origins"]),
    ):
        _same(metrics[key], value, "admitted metrics " + key)
    _same(len(metrics["rows"]), 2, "complete current family")
    _same(len(metrics["inherited_rows"]), 127, "complete inherited family")
    _same(
        dict(Counter(row["event"] for row in loaded["ledger"])),
        {"registered": 2, "inherited": 127, "evaluated": 2},
        "terminal upstream ledger",
    )
    c.recheck()
    return c.audit, loaded
