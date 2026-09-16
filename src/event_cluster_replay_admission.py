"""Checked failed-wave20 snapshots alongside unchanged verified wave18 records.

The retained wave20 outputs are unverified artifacts, never a numerical proof.
This module fits, transforms and scores nothing and decodes no unpublished loss.
"""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

import pandas as pd
import yaml

from src import event_cluster_admission as old
from src import issued_calibration_admission as checked

ROOT = Path(__file__).resolve().parents[1]
REPORT = "reports/event_cluster/"
DATA = "data/event_cluster/"
DEFAULT_EXPECTED = {
    "event_cluster.yaml": "824a9beda7cf293568dd2d16bbd3b76a8235f359a78c764147d75731c22b734c",
    REPORT
    + "publication_audit.json": "0d901a938c6a88ff4dbf1bb68d114640a50279e38826be29c97ea167ca56be15",
    REPORT
    + "failure_review.json": "ae554fd105b5a48b50739af4a6e64c3a8895c3e94bb14fbeb0230490d88d3408",
}
ANCHOR_PATHS = tuple(sorted(DEFAULT_EXPECTED))
OUTPUT_PATHS = tuple(
    sorted(
        DATA + name
        for name in (
            "forecasts.parquet",
            "states.parquet",
            "memory.parquet",
            "fits.json",
            "support_audit.json",
            "upstream_admission.json",
        )
    )
)
ERROR = (
    "Independent verification failed: AssertionError: every saved application state "
    "differs from independent reconstruction"
)
CONTROLS = ("baseline", "nuisance", "recent_frequency")
EVENT_COUNTS = {"registered": 3, "inherited": 131, "evaluated": 3, "verification_failed": 3}


class _Closure(old._Closure):
    """Frozen checked-byte primitives; successful wave18/19 failures still block."""

    def __init__(self, root, expected):
        self.root = Path(root).resolve()
        self.anchors = checked._mapping(DEFAULT_EXPECTED if expected is None else expected)
        if set(self.anchors) != set(ANCHOR_PATHS):
            raise ValueError("Exact three identified failed-wave20 anchors required")
        self.files = {}
        self.payloads = {}
        self.manifests = {}
        self.add(self.anchors)

    def no_failure(self):
        super().no_failure()
        name = REPORT + "failure.json"
        if name in self.files:
            # Unlike successful ancestors, this one exact failed marker must
            # remain present and unchanged at the final publication boundary.
            self.read(name, keep=False, fresh=True)
        super().no_failure()


def _manifest_closure(c):
    while True:
        pending = sorted(
            name
            for name in c.files
            if re.fullmatch(r"reports/[^/]+/manifest\.json", name) and name not in c.manifests
        )
        if not pending:
            return
        for name in pending:
            document = c.json(name)
            counts = {}
            for group in checked.GROUPS:
                refs = checked._mapping(document.get(group, {}))
                c.add(refs)
                counts[group] = len(refs)
            c.manifests[name] = counts


def _failure_rows():
    return [
        {
            "study": "event_cluster",
            "candidate": "cluster",
            "control": control,
            "score": "brier",
            "horizon": 1,
            "phases": [],
            "p_conservative": 1.0,
            "p_holm_wave": 1.0,
            "p_holm_cumulative": 1.0,
            "status": "INVALID_RUN",
            "verdict": "UNEVALUABLE",
            "error": ERROR,
        }
        for control in CONTROLS
    ]


def _check_failed_record(c, publication, review):
    signature = c.anchors["event_cluster.yaml"]
    rows = _failure_rows()
    expected = {
        "status": "UNEVALUABLE",
        "whole_wave_aborted": True,
        "leads": [],
        "hypothesis_count": 3,
        "cumulative_hypothesis_count": 134,
        "protocol_sha256": signature,
        "rows": rows,
    }
    checked._same(c.json(REPORT + "metrics.json"), expected, "canonical failed metrics")
    checked._same(c.json(REPORT + "failure.json"), expected, "required exact failure marker")
    checked._same(
        c.json(REPORT + "verification.json"),
        {
            "status": "FAILED",
            "error": ERROR,
            "protocol_sha256": signature,
        },
        "identified verification failure",
    )
    for key, value in {
        "canonical": ERROR,
        "column": "feature_cutoff_date",
        "saved_dtype": "datetime64[us]",
        "expected_dtype": "datetime64[ms]",
        "stage": "exact full application-state frame comparison",
        "independent_new_stage_solves_completed": 0,
        "independent_inference_reached": False,
        "all_other_values_proven_correct": False,
    }.items():
        checked._same(review["error"][key], value, "identified failure " + key)
    for key, value in {
        "metrics_equal_failure": True,
        "whole_wave_aborted": True,
        "leads": [],
        "new_hypotheses": 3,
        "inherited_hypotheses": 131,
        "cumulative_hypotheses": 134,
        "rows": [
            {
                k: row[k]
                for k in (
                    "candidate",
                    "control",
                    "p_conservative",
                    "p_holm_wave",
                    "p_holm_cumulative",
                    "phases",
                    "verdict",
                )
            }
            for row in rows
        ],
    }.items():
        checked._same(review["canonical_failure"][key], value, "review canonical " + key)
    for key, value in {
        "events": 140,
        "event_counts": EVENT_COUNTS,
        "order": [{"event": k, "count": n} for k, n in EVENT_COUNTS.items()],
        "terminal_rows_equal_canonical": True,
        "evaluated_score_values_inspected": False,
    }.items():
        checked._same(review["ledger"][key], value, "review ledger " + key)
    checked._same(publication["ledger_event_counts"], EVENT_COUNTS, "publication ledger roles")
    lines = c.read(REPORT + "trial_ledger.jsonl").splitlines()
    checked._same(len(lines), 140, "retained ledger total")
    # Hash opaque historical/evaluated lines against the pinned review. Never
    # parse their scored payloads merely to recover already reviewed role counts.
    for name, selected in (
        ("inherited_record_sha256", lines[3:134]),
        ("opaque_evaluated_record_sha256", lines[134:137]),
    ):
        expected_hashes = review["ledger"][name]
        if type(expected_hashes) is not list:
            raise ValueError("Ordered opaque ledger hashes required")
        for signature_one in expected_hashes:
            checked._signature(signature_one)
        checked._same(
            [hashlib.sha256(line).hexdigest() for line in selected],
            expected_hashes,
            "opaque retained " + name,
        )
    for position, (control, row) in enumerate(zip(CONTROLS, rows, strict=True)):
        checked._same(
            checked._json(lines[position]),
            {
                "event": "registered",
                "study": "event_cluster",
                "candidate": "cluster",
                "control": control,
                "horizon": 1,
                "score": "brier",
                "protocol_sha256": signature,
            },
            "original registration",
        )
        checked._same(
            checked._json(lines[137 + position]),
            {"event": "verification_failed", **row},
            "terminal failed row",
        )


def _collect(root, expected):
    c = _Closure(root, expected)
    c.no_failure()
    for name in ANCHOR_PATHS:
        c.read(name)
    protocol = yaml.safe_load(c.read("event_cluster.yaml"))
    if type(protocol) is not dict:
        raise ValueError("Frozen wave20 protocol object required")
    publication = c.json(REPORT + "publication_audit.json")
    review = c.json(REPORT + "failure_review.json")
    signature = c.anchors["event_cluster.yaml"]
    for label, document in (("publication", publication), ("review", review)):
        checked._same(document["protocol_sha256"], signature, label + " protocol")
    for key, value in {
        "status": "FAILED_RESEARCH_RECORD_PRESERVED_AND_AUDITED",
        "research_verification_status": "FAILED",
        "canonical_metrics_status": "UNEVALUABLE",
        "new_hypotheses": 3,
        "cumulative_hypotheses": 134,
        "ledger_events": 140,
        "canonical_all_three_p_one": True,
        "current_wave_completed_independently_verified_forecasts": 0,
        "unpublished_scores_used_for_promotion": False,
        "post_failure_code_or_protocol_changes": False,
        "post_failure_empirical_retry": False,
    }.items():
        checked._same(publication[key], value, "failure publication " + key)
    for key, value in {
        "status": "FAILURE_ACCOUNTING_REVIEWED",
        "research_status": "UNEVALUABLE",
        "numerical_verification_status": "FAILED",
        "not_a_successful_numerical_verification": True,
        "historical_producer_reruns": 0,
        "whole_verifier_reruns": 0,
        "additional_optimizer_runs": 0,
        "unpublished_candidate_score_values_inspected": False,
    }.items():
        checked._same(review[key], value, "failure review " + key)
    outputs = checked._mapping(publication["failed_output_hashes_not_numerically_verified"])
    if set(outputs) != set(OUTPUT_PATHS):
        raise ValueError("Exact six retained unverified output paths required")
    checked._same(
        review["current_private_output_sha256"], outputs, "retained output identities"
    )
    c.add(outputs)
    artifacts = old._relative_artifacts(REPORT, publication["report_artifact_hashes"])
    required = {
        REPORT + name
        for name in (
            "manifest.json",
            "freeze_record.json",
            "verification.json",
            "failure.json",
            "metrics.json",
            "trial_ledger.jsonl",
            "failure_review.json",
            "SUMMARY.md",
            "FAILURE_REVIEW.md",
            "full_repository_tests.txt",
            "pre_run_checks.txt",
            "unpublished_scored_metrics.json",
        )
    }
    if not required.issubset(artifacts):
        raise ValueError("Complete failed-wave20 publication references required")
    c.add(artifacts)
    review_artifacts = checked._mapping(review["report_artifact_sha256"])
    c.add(review_artifacts)
    for key, name in (
        ("manifest_sha256", "manifest.json"),
        ("freeze_sha256", "freeze_record.json"),
        ("verification_sha256", "verification.json"),
    ):
        checked._same(publication[key], c.files[REPORT + name], "published " + key)
        if key != "verification_sha256":
            checked._same(review[key], c.files[REPORT + name], "reviewed " + key)
    checked._same(
        review["failure_review_document_sha256"],
        c.files[REPORT + "FAILURE_REVIEW.md"],
        "failure review document",
    )
    manifest = c.json(REPORT + "manifest.json")
    freeze = c.json(REPORT + "freeze_record.json")
    for label, document in (("manifest", manifest), ("freeze", freeze)):
        checked._same(document["protocol_sha256"], signature, label + " protocol")
    counts = {
        group: len(checked._mapping(manifest[group]))
        for group in ("code", "inputs", "preserved")
    }
    checked._same(publication["manifest_entries_checked"], counts, "failed manifest coverage")
    checked._same(freeze["code"], manifest["code"], "complete failed-wave code freeze")
    c.add(freeze["prefit_design"])
    c.add({REPORT + "full_repository_tests.txt": freeze["checks"]["full_log_sha256"]})
    checked._integer(freeze["checks"]["full_repository_tests"])
    checked._same(
        publication["full_repository_tests"],
        freeze["checks"]["full_repository_tests"],
        "frozen full-suite test count",
    )
    _manifest_closure(c)
    _check_failed_record(c, publication, review)
    original = old._collect(root, protocol["upstream"]["anchors"])
    saved_original = c.json(DATA + "upstream_admission.json")
    checked._same(
        original.audit, saved_original, "retained exact original successful admission"
    )
    c.add(original.files)
    _manifest_closure(c)
    metadata = review["output_metadata"]
    mapping = {
        "combined_rows": "current_wave_combined_saved_forecasts",
        "new_generated_unverified_forecasts": "current_wave_generated_scored_forecasts_not_independently_verified",
        "reused_control_forecasts": "current_wave_reused_control_forecasts",
        "producer_reported_monthly_schedules": "current_wave_producer_reported_monthly_schedules_not_independently_verified",
    }
    for key, pubkey in mapping.items():
        checked._integer(metadata[key])
        checked._same(metadata[key], publication[pubkey], "retained metadata " + key)
    for key in ("state_rows", "memory_rows"):
        checked._integer(metadata[key])
    checked._same(
        metadata["new_fits_independently_verified"], False, "unverified retained fits"
    )
    checked._same(
        metadata["combined_rows"],
        metadata["new_generated_unverified_forecasts"] + metadata["reused_control_forecasts"],
        "retained metadata row partition",
    )
    # Capture every retained payload before any original numerical-table admission.
    # Other private diagnostics are hash checked but never decoded.
    for name in sorted(c.files):
        c.read(name, keep=name in OUTPUT_PATHS)
    c.no_failure()
    c.audit = {
        "status": "PINNED_FAILED_WAVE20_WITH_VERIFIED_WAVE19_WAVE18",
        "anchors": dict(sorted(c.anchors.items())),
        "files": dict(sorted(c.files.items())),
        "manifest_paths": sorted(c.manifests),
        "manifest_groups": dict(sorted(c.manifests.items())),
        "retained_output_hashes": dict(sorted(outputs.items())),
        "report_artifact_hashes": dict(sorted({**artifacts, **review_artifacts}.items())),
        "counts": {
            "files": len(c.files),
            "manifests": len(c.manifests),
            "retained_outputs": len(outputs),
        },
        "failed_wave_identity": {
            "protocol_sha256": signature,
            "manifest_sha256": c.files[REPORT + "manifest.json"],
            "freeze_sha256": c.files[REPORT + "freeze_record.json"],
            "verification_sha256": c.files[REPORT + "verification.json"],
            "failure_sha256": c.files[REPORT + "failure.json"],
            "publication_audit_sha256": c.anchors[REPORT + "publication_audit.json"],
            "failure_review_sha256": c.anchors[REPORT + "failure_review.json"],
        },
        "failed_wave_counts": {
            "combined_saved_forecasts": metadata["combined_rows"],
            "generated_candidate_forecasts": metadata["new_generated_unverified_forecasts"],
            "reused_control_forecasts": metadata["reused_control_forecasts"],
            "producer_reported_monthly_schedules": metadata[
                "producer_reported_monthly_schedules"
            ],
            "completed_independently_verified_forecasts": 0,
            "state_rows": metadata["state_rows"],
            "memory_rows": metadata["memory_rows"],
            "new_hypotheses": 3,
            "cumulative_hypotheses": 134,
            "ledger_events": 140,
            "ledger_event_counts": dict(EVENT_COUNTS),
        },
        "original_admission": saved_original,
        "retained_outputs_numerically_verified": False,
        "historical_models_refitted": False,
        "source_values_reparsed": False,
        "retained_outputs_transformed": False,
        "loaded_from_checked_snapshots": True,
        "unpublished_scored_payload_decoded": False,
    }
    c.old_expected = protocol["upstream"]["anchors"]
    return c


def collect_input_pins(root=ROOT, expected=None):
    """Check documentary identities/file bytes without output-table decoding."""
    c = _collect(root, expected)
    c.recheck()
    return c.audit["files"]


def admit_upstream(root=ROOT, expected=None, registered_pins=None):
    """Return audit, unchanged original wave18 tables, and unverified wave20 data."""
    c = _collect(root, expected)
    if registered_pins is not None:
        pins = checked._mapping(registered_pins)
        for name, signature in c.files.items():
            if pins.get(name) != signature:
                raise ValueError("Registered source missing or changed: " + name)
    original_audit, original_loaded = old.admit_upstream(root, c.old_expected, registered_pins)
    checked._same(
        original_audit, c.audit["original_admission"], "loaded original proof identity"
    )
    retained = {
        name: pd.read_parquet(io.BytesIO(c.read(DATA + name + ".parquet")))
        for name in ("forecasts", "states", "memory")
    }
    retained["fits"] = checked._json(c.read(DATA + "fits.json"), object_required=False)
    if type(retained["fits"]) is not list:
        raise ValueError("Retained fit records must be a JSON list")
    retained["support"] = c.json(DATA + "support_audit.json")
    retained["upstream_admission"] = c.json(DATA + "upstream_admission.json")
    c.recheck()
    return c.audit, original_loaded, retained
