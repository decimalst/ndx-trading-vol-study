"""Capture a frozen notice inventory as private bytes, without value admission."""

import hashlib
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from src.treasury_auction_documents import document_requests, validate_document_receipt

PILOT = "reports/next_signal_review/TREASURY_DOCUMENT_CAPTURE.json"
MEMBER_KEYS = {
    "announcement_date",
    "auction_date",
    "cusip",
    "security_type",
    "security_term",
    "term_bucket",
}
KINDS = (
    "announcement_pdf",
    "competitive_pdf",
    "noncompetitive_pdf",
    "special_pdf",
    "announcement_xml",
    "competitive_xml",
)
TERMS = {"2-Year", "3-Year", "5-Year", "7-Year", "10-Year", "30-Year"}
REUSED = "CAPTURED_REVIEWED_PILOT_ORIGINAL"
NEW = "NOT_YET_CAPTURED"


def _path(root, relative):
    if (
        type(relative) is not str
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ValueError("Repository-relative source path required")
    result = root / relative
    if not result.resolve().is_relative_to(root):
        raise ValueError("Source path leaves the repository")
    return result


def _hash(path):
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"Required source unavailable: {path}") from exc


def _json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"Required metadata unavailable or malformed: {path}") from exc


def _write(path, obj):
    with path.open("x", encoding="utf-8") as output:
        os.fchmod(output.fileno(), 0o600)
        json.dump(obj, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def _check_pins(root, pins):
    if type(pins) is not dict or not pins:
        raise ValueError("Nonempty checkpoint pins required")
    for name, expected in pins.items():
        if type(expected) is not str or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError("Invalid source hash")
        if _hash(_path(root, name)) != expected:
            raise ValueError(f"Checkpoint source drift: {name}")


def _selection_requests(selection):
    if (
        type(selection) is not dict
        or selection.get("status") != "FIXED_KNOWN_SPECIAL_NOTICE_METADATA_MANIFEST"
        or selection.get("source_floor") != "2010-01-01"
        or selection.get("source_ceiling") != "2025-10-20"
        or selection.get("source_admitted") is not False
        or type(selection.get("notices")) is not list
        or not selection["notices"]
    ):
        raise ValueError("Expected fixed, bounded notice metadata selection")
    events, notices = {}, {}
    counts = {REUSED: 0, NEW: 0}
    for notice in selection["notices"]:
        if type(notice) is not dict:
            raise ValueError("Invalid notice entry")
        status, url = notice.get("status"), notice.get("url")
        expected = {"url", "status", "archive_memberships"}
        if status == REUSED:
            expected |= {"body_path", "body_sha256"}
        if (
            status not in counts
            or set(notice) != expected
            or type(url) is not str
            or url in notices
        ):
            raise ValueError("Unknown, duplicate or malformed notice entry")
        members = notice["archive_memberships"]
        if type(members) is not list or not members:
            raise ValueError("Notice archive memberships required")
        seen = set()
        for member in members:
            if (
                type(member) is not dict
                or set(member) != MEMBER_KEYS
                or any(type(value) is not str or not value for value in member.values())
                or member["security_type"] not in {"Note", "Bond"}
                or member["term_bucket"] not in TERMS
            ):
                raise ValueError("Exact six-field nominal metadata membership required")
            key = (member["auction_date"], member["cusip"])
            if key in seen:
                raise ValueError("Duplicate notice association")
            seen.add(key)
            if key not in events:
                events[key] = {"metadata": member, "documents": {kind: [] for kind in KINDS}}
            elif events[key]["metadata"] != member:
                raise ValueError("Conflicting metadata for the same auction event")
            events[key]["documents"]["special_pdf"].append(url)
        counts[status] += 1
        notices[url] = notice
    declared = {
        "distinct_notice_urls": len(notices),
        "already_captured_and_reviewed": counts[REUSED],
        "not_yet_captured": counts[NEW],
        "provisional_nominal_rows_with_notices": len(events),
    }
    for field, expected in declared.items():
        if type(selection.get(field)) is not int or selection[field] != expected:
            raise ValueError(f"Declared notice count differs: {field}")
    requests = document_requests(
        {
            "records": [
                {"record": entry["metadata"] | {"documents": entry["documents"]}}
                for _, entry in sorted(events.items())
            ]
        }
    )
    if {entry["url"] for entry in requests} != set(notices):
        raise ValueError("Regrouped notice URL set changed")
    return notices, requests, declared


def _reuse(root, notices, requests, pilot):
    if type(pilot) is not dict or type(pilot.get("documents")) is not list:
        raise ValueError("Pinned pilot document manifest required")
    documents = {}
    for doc in pilot["documents"]:
        if type(doc) is not dict or type(doc.get("url")) is not str or doc["url"] in documents:
            raise ValueError("Malformed or duplicate pilot document URL")
        documents[doc["url"]] = doc
    reused = {}
    for request in requests:
        url = request["url"]
        notice = notices[url]
        if notice["status"] != REUSED:
            continue
        doc = documents.get(url)
        if (
            doc is None
            or doc.get("format") != "pdf"
            or doc.get("status") != "VERIFIED_TRANSPORT_SCHEMA_ONLY"
            or doc.get("body_path") != notice["body_path"]
            or doc.get("body_sha256") != notice["body_sha256"]
        ):
            raise ValueError("Reused original does not match pinned pilot capture")
        body = _path(root, doc["body_path"])
        receipt_path = _path(root, doc.get("receipt_path"))
        if _hash(body) != doc["body_sha256"] or _hash(receipt_path) != doc.get(
            "receipt_sha256"
        ):
            raise ValueError("Reused original body or receipt changed")
        raw, receipt = body.read_bytes(), _json(receipt_path)
        validate_document_receipt(raw, receipt, url, "pdf")
        original_members = doc.get("memberships")
        if (
            type(original_members) is not list
            or not original_members
            or any(m not in request["memberships"] for m in original_members)
            or len({json.dumps(m, sort_keys=True) for m in original_members})
            != len(original_members)
        ):
            raise ValueError("Pilot memberships do not match full inventory associations")
        reused[url] = {
            **request,
            **{
                key: doc[key]
                for key in (
                    "body_path",
                    "body_sha256",
                    "receipt_path",
                    "receipt_sha256",
                    "headers_sha256",
                )
            },
            "archive_memberships": notice["archive_memberships"],
            "source_capture_mode": "reused_original_bytes",
            "status": "VERIFIED_TRANSPORT_SCHEMA_ONLY",
        }
    return reused


def capture_notice_batch(root: Path, selection_path: Path, checkpoint_path: Path) -> dict:
    """Validate all frozen inputs, reuse originals, and capture remaining URLs once."""
    root = root.resolve()
    selection_path, checkpoint_path = selection_path.resolve(), checkpoint_path.resolve()
    private = root / "data/source_discovery/treasury_auction/full_notices_v1"
    report = root / "reports/next_signal_review/TREASURY_FULL_NOTICE_CAPTURE.json"
    if private.exists() or report.exists():
        raise FileExistsError("Existing full notice capture must be preserved")
    checkpoint = _json(checkpoint_path)
    if type(checkpoint) is not dict or checkpoint.get("status") != "READY_FULL_NOTICE_CAPTURE":
        raise ValueError("Notice capture checkpoint not ready")
    pins = checkpoint.get("pins")
    _check_pins(root, pins)
    try:
        selection_name = str(selection_path.relative_to(root))
    except ValueError as exc:
        raise ValueError("Selection must belong to the repository") from exc
    if pins.get(selection_name) != _hash(selection_path):
        raise ValueError("Exact selection is not checkpoint-pinned")
    selection = _json(selection_path)
    source_pins = selection.get("source_pins") if type(selection) is dict else None
    if type(source_pins) is not dict or not source_pins or PILOT not in source_pins:
        raise ValueError("Pinned full-inventory and pilot sources required")
    if any(pins.get(name) != signature for name, signature in source_pins.items()):
        raise ValueError("Selection source pins are not preserved in checkpoint")
    notices, requests, counts = _selection_requests(selection)
    reused = _reuse(root, notices, requests, _json(root / PILOT))
    checkpoint_hash, selection_hash = _hash(checkpoint_path), _hash(selection_path)
    private.mkdir(mode=0o700, parents=True, exist_ok=False)
    _write(
        private / "started.json",
        {
            "status": "STARTED_FIXED_NOTICE_CAPTURE",
            "started_utc": datetime.now(UTC).isoformat(),
            "checkpoint_sha256": checkpoint_hash,
            "selection_sha256": selection_hash,
            "requests": requests,
            "counts": counts,
        },
    )

    def fetch(request):
        url = request["url"]
        if url in reused:
            return reused[url]
        folder = private / hashlib.sha256(url.encode()).hexdigest()
        folder.mkdir(mode=0o700)
        body, headers = folder / "body.pdf", folder / "headers.txt"
        try:
            proc = subprocess.run(
                [
                    "curl",
                    "--silent",
                    "--show-error",
                    "--max-time",
                    "45",
                    "--proto",
                    "=https",
                    "--max-filesize",
                    str(8 * 1024 * 1024),
                    "--dump-header",
                    str(headers),
                    "--output",
                    str(body),
                    "--write-out",
                    "%{http_code}\\n%{url_effective}\\n%{content_type}",
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=50,
            )
            fields, exit_code, stderr = proc.stdout.split("\n"), proc.returncode, proc.stderr
        except (subprocess.TimeoutExpired, OSError, UnicodeError) as exc:
            fields, exit_code, stderr = (
                [],
                124,
                f"{type(exc).__name__}: transport did not complete",
            )
        raw = body.read_bytes() if body.exists() else b""
        for path in (body, headers):
            if path.exists():
                path.chmod(0o600)
        receipt = {
            "requested_url": url,
            "effective_url": fields[1] if len(fields) > 1 else "",
            "http_status": int(fields[0]) if fields and fields[0].isdigit() else 0,
            "curl_exit": exit_code,
            "content_type": fields[2] if len(fields) > 2 else "",
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "retrieved_utc": datetime.now(UTC).isoformat(),
        }
        receipt_path = folder / "receipt.json"
        _write(receipt_path, receipt)
        with (folder / "stderr.txt").open("x") as output:
            os.fchmod(output.fileno(), 0o600)
            output.write(stderr)
        entry = {
            **request,
            "archive_memberships": notices[url]["archive_memberships"],
            "source_capture_mode": "new_request",
            "body_path": str(body.relative_to(root)),
            "body_sha256": receipt["sha256"],
            "receipt_path": str(receipt_path.relative_to(root)),
            "receipt_sha256": _hash(receipt_path),
            "headers_sha256": _hash(headers) if headers.exists() else None,
        }
        try:
            validate_document_receipt(raw, receipt, url, "pdf")
            entry["status"] = "VERIFIED_TRANSPORT_SCHEMA_ONLY"
        except ValueError as exc:
            entry.update(status="REQUIRES_REVIEW", error=str(exc))
        _write(folder / "summary.json", entry)
        return entry

    with ThreadPoolExecutor(max_workers=3) as pool:
        entries = list(pool.map(fetch, requests))
    pin_errors = []
    try:
        _check_pins(root, pins)
        if (
            _hash(checkpoint_path) != checkpoint_hash
            or _hash(selection_path) != selection_hash
        ):
            raise ValueError("Capture checkpoint or selection changed during requests")
    except ValueError as exc:
        pin_errors.append(str(exc))
    result = {
        "status": "CAPTURED_SCHEMA_ONLY"
        if not pin_errors
        and all(e["status"] == "VERIFIED_TRANSPORT_SCHEMA_ONLY" for e in entries)
        else "DOCUMENTS_REQUIRE_REVIEW",
        "completed_utc": datetime.now(UTC).isoformat(),
        "checkpoint_sha256": checkpoint_hash,
        "selection_sha256": selection_hash,
        "documents": entries,
        "counts": counts,
        "archive_associations": sum(len(e["archive_memberships"]) for e in entries),
        "input_pin_errors": pin_errors,
        "numerical_source_admitted": False,
        "predictive_comparisons_added": 0,
        "inspection_scope": "PDF transport and byte signatures only; original pilot bytes reused; no monetary values or document text exported.",
    }
    _write(report, result)
    return result
