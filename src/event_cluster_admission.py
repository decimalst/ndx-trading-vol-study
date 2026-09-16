"""Admit exact wave18 records inside the immutable successful wave19 envelope.

No old optimizer, numerical source parser, target builder or scorer is called.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import yaml

from src import issued_calibration_admission as old

ROOT = Path(__file__).resolve().parents[1]
REPORT = "reports/issued_calibration/"
DATA = "data/issued_calibration/"
DOCUMENTARY = "reports/early_session_feasibility/"
DEFAULT_EXPECTED = {
    "issued_calibration.yaml": "f4c03c25718439432c176aba36ce3bc572776b67a5ff80f906ddc00cd85b2051",
    REPORT
    + "manifest.json": "5d824aad69991ea06cc35916be2e9852d9f5613c17bafe7b18e46047d93bdf6a",
    REPORT
    + "freeze_record.json": "d12459d404d4bb7cfab43a351764b1c29353d0e2dbfbec6af689ffc411ba0c71",
    REPORT
    + "verification.json": "5de5f8ed0f4a7757c475e0557b9798fe04d8fea33714a5c68437fbb181d48d14",
    REPORT
    + "publication_audit.json": "17275a47b0c8738e3177e905025869fd69523cfc4cdcca45d4fd0cbd49d45be8",
    REPORT
    + "publication_review.json": "481565b24889457d8bf87ee824aac1284e688418a02fe7b4715aeee935e96003",
    DOCUMENTARY
    + "preservation_audit.json": "feb526675c97a52bc13f372dd50024093d9fc6a7fe3580859fd20f7ab585dc41",
}
ANCHOR_PATHS = tuple(sorted(DEFAULT_EXPECTED))
OUTPUT_PATHS = tuple(
    sorted(
        [
            DATA + "upstream_admission.json",
            DATA + "forecasts.parquet",
            DATA + "states.parquet",
            DATA + "calibration_audit.json",
            REPORT + "metrics.json",
            REPORT + "trial_ledger.jsonl",
        ]
    )
)


class _Closure(old._Closure):
    """Reuse frozen checked-byte/path primitives with both absence boundaries."""

    def __init__(self, root, expected):
        self.root = Path(root).resolve()
        self.anchors = old._mapping(DEFAULT_EXPECTED if expected is None else expected)
        if set(self.anchors) != set(ANCHOR_PATHS):
            raise ValueError("Exact seven wave19/documentary anchor paths required")
        self.files = {}
        self.payloads = {}
        self.manifests = {}
        self.add(self.anchors)

    def no_failure(self):
        for report in (old.REPORT, REPORT):
            path = self.root / (report + "failure.json")
            if path.exists() or path.is_symlink():
                raise ValueError("Canonical upstream failure blocks admission: " + report)


def _relative_artifacts(prefix, mapping):
    return {prefix + name: signature for name, signature in old._mapping(mapping).items()}


def _collect(root, expected):
    c = _Closure(root, expected)
    c.no_failure()
    for name in ANCHOR_PATHS:
        c.read(name)
    protocol = yaml.safe_load(c.read("issued_calibration.yaml"))
    if type(protocol) is not dict:
        raise ValueError("Original wave19 protocol object required")
    signature = c.anchors["issued_calibration.yaml"]
    manifest = c.json(REPORT + "manifest.json")
    freeze = c.json(REPORT + "freeze_record.json")
    verified = c.json(REPORT + "verification.json")
    publication = c.json(REPORT + "publication_audit.json")
    review = c.json(REPORT + "publication_review.json")
    documentary = c.json(DOCUMENTARY + "preservation_audit.json")
    for label, obj in (
        ("manifest", manifest),
        ("freeze", freeze),
        ("verification", verified),
        ("publication", publication),
        ("review", review),
    ):
        old._same(obj["protocol_sha256"], signature, label + " protocol")
    for actual, required, label in (
        (verified["status"], "VERIFIED", "verification status"),
        (publication["status"], "VERIFIED_REPORT_AUDITED", "publication status"),
        (review["status"], "APPROVED_VERIFIED_INTERPRETATION", "review status"),
        (review["figure_visually_reviewed"], True, "visual review"),
        (
            documentary["status"],
            "DOCUMENTARY_REVIEW_COMPLETE_SOURCE_NOT_ADMITTED",
            "documentary status",
        ),
        (documentary["new_hypotheses"], 0, "documentary new hypotheses"),
        (documentary["cumulative_hypotheses_unchanged"], 131, "documentary unchanged family"),
    ):
        old._same(actual, required, label)
    for key, filename in (
        ("manifest_sha256", "manifest.json"),
        ("freeze_sha256", "freeze_record.json"),
        ("verification_sha256", "verification.json"),
    ):
        old._same(publication[key], c.anchors[REPORT + filename], key)
    for group in ("code", "inputs", "preserved"):
        old._mapping(manifest[group])
    old._same(freeze["code"], manifest["code"], "complete frozen code")
    c.add(freeze["prefit_design"])
    c.add({REPORT + "full_repository_tests.txt": freeze["checks"]["full_log_sha256"]})
    old._integer(freeze["checks"]["full_repository_tests"])
    verifier_hash = manifest["code"]["src/verify_issued_calibration.py"]
    old._same(verified["verifier_sha256"], verifier_hash, "verifier code")
    old._same(publication["verifier_sha256"], verifier_hash, "published verifier code")
    outputs = old._mapping(verified["verified_output_hashes"])
    if set(outputs) != set(OUTPUT_PATHS):
        raise ValueError("Exact six wave19 verified output paths required")
    old._same(publication["verified_output_hashes"], outputs, "published verified outputs")
    c.add(outputs)
    artifacts = _relative_artifacts(REPORT, publication["report_artifact_hashes"])
    required = {
        REPORT + n
        for n in (
            "SUMMARY.md",
            "NEXT_RESEARCH_DIRECTION.md",
            "publication_review.json",
            "verification.json",
            "metrics.json",
            "trial_ledger.jsonl",
        )
    }
    if not required.issubset(artifacts):
        raise ValueError("Complete wave19 publication identity required")
    c.add(artifacts)
    old._same(review["summary_sha256"], artifacts[REPORT + "SUMMARY.md"], "reviewed summary")
    c.add(review["document_hashes"])
    c.add(review["figure_hashes"])
    if "verified_output_hashes" in review:
        old._same(review["verified_output_hashes"], outputs, "review output identities")
    documentary_prior = _relative_artifacts(REPORT, documentary["prior_anchors_verified"])
    expected_prior = {
        REPORT + n
        for n in (
            "publication_audit.json",
            "freeze_record.json",
            "manifest.json",
            "NEXT_RESEARCH_DIRECTION.md",
        )
    }
    if set(documentary_prior) != expected_prior:
        raise ValueError("Exact documentary prior anchor references required")
    c.add(documentary_prior)
    documentary_artifacts = _relative_artifacts(
        DOCUMENTARY, documentary["new_document_sha256"]
    )
    if not documentary_artifacts:
        raise ValueError("Documentary preservation references required")
    c.add(documentary_artifacts)

    # Frozen metadata-only reconstruction establishes the exact saved inner
    # proof before any numerical output is decoded by the public loader.
    inner = old._collect(root, protocol["upstream"]["anchors"])
    inner.recheck()
    c.add(inner.audit["files"])
    saved_inner = c.json(DATA + "upstream_admission.json")
    old._same(saved_inner, inner.audit, "complete saved wave18 admission")
    while True:
        pending = sorted(
            n
            for n in c.files
            if re.fullmatch(r"reports/[^/]+/manifest\.json", n) and n not in c.manifests
        )
        if not pending:
            break
        for name in pending:
            obj = c.json(name)
            counts = {}
            for group in old.GROUPS:
                refs = old._mapping(obj.get(group, {}))
                c.add(refs)
                counts[group] = len(refs)
            c.manifests[name] = counts
    entries = {g: len(manifest[g]) for g in ("code", "inputs", "preserved")}
    old._same(publication["manifest_entries_checked"], entries, "published inventory")
    old._same(
        verified["artifact_hashes_checked"], {**entries, "outputs": 6}, "verified inventory"
    )
    f = verified["forecast_reconstruction"]
    counts = {
        "forecasts": old._integer(f["forecasts_verified"]),
        "new_forecasts": old._integer(f["new_forecasts"]),
        "reused_control_forecasts": old._integer(f["reused_control_forecasts"]),
        "monthly_fits": old._integer(f["new_monthly_fits"], 0),
        "original_monthly_fits_replayed": old._integer(f["original_monthly_fits_replayed"]),
        "common_scored_origins": old._integer(f["common_scored_origins"]),
        "common_application_origins": old._integer(f["common_application_origins"]),
        "cumulative_hypotheses": 131,
    }
    n = counts["common_scored_origins"]
    for key, value in (
        ("forecasts", 3 * n),
        ("new_forecasts", n),
        ("reused_control_forecasts", 2 * n),
        ("monthly_fits", 0),
    ):
        old._same(counts[key], value, "upstream " + key)
    if counts["common_application_origins"] < n:
        raise ValueError("Upstream application count below scored count")
    oldcounts = inner.audit["upstream_verified_counts"]
    for newkey, oldkey in (
        ("forecasts", "forecasts"),
        ("original_monthly_fits_replayed", "monthly_fits"),
        ("common_scored_origins", "common_scored_origins"),
        ("common_application_origins", "common_application_origins"),
    ):
        old._same(counts[newkey], oldcounts[oldkey], "unchanged wave18 " + oldkey)
    for key, value in (
        ("current_wave_verified_forecasts", n),
        ("current_wave_reused_forecasts", 2 * n),
        ("current_wave_combined_forecasts", 3 * n),
        ("current_wave_verified_monthly_fits", 0),
        ("current_wave_verified_applications", counts["common_application_origins"]),
        ("current_wave_scored_origins", n),
        ("cumulative_hypotheses", 131),
        ("ledger_events", 133),
        ("ledger_event_counts", {"registered": 2, "inherited": 129, "evaluated": 2}),
    ):
        old._same(publication[key], value, key)
    old._same(
        verified["ledger_events_verified"],
        {"inherited": 129, "registered": 2, "evaluated": 2},
        "verified terminal ledger role counts",
    )
    old._same(verified["inference"]["new_hypotheses_verified"], 2, "verified new family")
    old._same(
        verified["inference"]["cumulative_hypotheses_verified"],
        131,
        "verified cumulative family",
    )
    for name in sorted(c.files):
        c.read(name, keep=name in OUTPUT_PATHS)
    c.audit = {
        "status": "PINNED_WAVE19_WITH_WAVE18_RECORDS",
        "anchors": dict(sorted(c.anchors.items())),
        "files": dict(sorted(c.files.items())),
        "manifest_paths": sorted(c.manifests),
        "manifest_groups": dict(sorted(c.manifests.items())),
        "verified_output_hashes": dict(sorted(outputs.items())),
        "publication_artifact_hashes": dict(sorted(artifacts.items())),
        "documentary_artifact_hashes": dict(sorted(documentary_artifacts.items())),
        "counts": {
            "files": len(c.files),
            "manifests": len(c.manifests),
            "verified_outputs": len(outputs),
            "publication_artifacts": len(artifacts),
            "documentary_artifacts": len(documentary_artifacts),
        },
        "upstream_identity": {
            "protocol_sha256": signature,
            "manifest_sha256": c.anchors[REPORT + "manifest.json"],
            "freeze_sha256": c.anchors[REPORT + "freeze_record.json"],
            "verification_sha256": c.anchors[REPORT + "verification.json"],
            "verifier_sha256": verifier_hash,
            "publication_audit_sha256": c.anchors[REPORT + "publication_audit.json"],
            "publication_review_sha256": c.anchors[REPORT + "publication_review.json"],
            "documentary_preservation_sha256": c.anchors[
                DOCUMENTARY + "preservation_audit.json"
            ],
        },
        "upstream_verified_counts": counts,
        "wave18_admission": saved_inner,
        "historical_models_refitted": False,
        "source_values_reparsed": False,
        "loaded_from_checked_snapshots": True,
    }
    c.old_expected = protocol["upstream"]["anchors"]
    return c


def collect_input_pins(root=ROOT, expected=None):
    """Check proof metadata and file bytes; never decode a numerical table."""
    c = _collect(root, expected)
    c.recheck()
    return c.audit["files"]


def _check_terminal(c):
    metrics = c.json(REPORT + "metrics.json")
    if metrics.get("status") == "UNEVALUABLE" or metrics.get("whole_wave_aborted") is True:
        raise ValueError("Canonical failed wave19 cannot be admitted")
    counts = c.audit["upstream_verified_counts"]
    for key, value in (
        ("protocol_sha256", c.audit["upstream_identity"]["protocol_sha256"]),
        ("hypothesis_count", 2),
        ("cumulative_hypothesis_count", 131),
        ("new_forecasts", counts["new_forecasts"]),
        ("reused_control_forecasts", counts["reused_control_forecasts"]),
        ("combined_forecasts", counts["forecasts"]),
        ("new_monthly_fits", 0),
        ("common_scored_origins", counts["common_scored_origins"]),
        ("common_application_origins", counts["common_application_origins"]),
    ):
        old._same(metrics[key], value, "admitted wave19 metrics " + key)
    old._same(len(metrics["rows"]), 2, "complete wave19 current family")
    old._same(len(metrics["inherited_rows"]), 129, "complete wave19 inherited family")
    events = [
        old._json(line)["event"] for line in c.read(REPORT + "trial_ledger.jsonl").splitlines()
    ]
    old._same(
        dict(Counter(events)),
        {"registered": 2, "inherited": 129, "evaluated": 2},
        "terminal wave19 ledger",
    )


def admit_upstream(root=ROOT, expected=None, registered_pins=None):
    """Return a combined deterministic audit and unchanged original wave18 data."""
    c = _collect(root, expected)
    if registered_pins is not None:
        registered_pins = old._mapping(registered_pins)
        for name, signature in c.files.items():
            if registered_pins.get(name) != signature:
                raise ValueError("Registered source missing or changed: " + name)
    _check_terminal(c)
    inner, loaded = old.admit_upstream(root, c.old_expected, registered_pins)
    old._same(inner, c.audit["wave18_admission"], "loaded unchanged wave18 proof")
    c.recheck()
    return c.audit, loaded
