"""Pinned metadata selection and one-shot announcement/result byte capture."""

import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from src.treasury_auction_documents import document_requests, validate_document_receipt
from src.treasury_notice_acquire import _check_pins, _hash, _json, _path, _write

PILOT = "reports/next_signal_review/TREASURY_DOCUMENT_CAPTURE.json"
INVENTORY = "reports/next_signal_review/TREASURY_INVENTORY_CAPTURE.json"
NOTICE_AUDIT = "reports/next_signal_review/TREASURY_FULL_NOTICE_TERMINAL_AUDIT.json"
METADATA = tuple(
    f"data/source_discovery/treasury_auction/inventory_capture_v2/{year}/metadata.json"
    for year in range(2010, 2026)
)
ROLES = ("announcement_pdf", "announcement_xml", "competitive_pdf", "competitive_xml")
TERMS = {"2-Year", "3-Year", "5-Year", "7-Year", "10-Year", "30-Year"}
MEMBER_KEYS = {
    "announcement_date",
    "auction_date",
    "cusip",
    "security_type",
    "security_term",
    "term_bucket",
}
PRIVATE = "data/source_discovery/treasury_auction/history_capture_v1"
REPORT = "reports/next_signal_review/TREASURY_HISTORY_CAPTURE.json"


def _pilot_documents(pilot):
    if type(pilot) is not dict or pilot.get("status") != "CAPTURED_SCHEMA_ONLY":
        raise ValueError("pilot capture is not complete")
    docs = pilot.get("documents")
    if type(docs) is not list:
        raise ValueError("pilot document inventory missing")
    by_url = {}
    for doc in docs:
        if type(doc) is not dict or type(doc.get("url")) is not str or doc["url"] in by_url:
            raise ValueError("invalid or duplicate pilot document URL")
        by_url[doc["url"]] = doc
    return by_url


def build_history_selection(root: Path, source_pins: dict[str, str]) -> dict:
    """Validate all source pins before decoding projected metadata; never fetch."""
    root = Path(root).resolve()
    required = set(METADATA) | {PILOT, INVENTORY, NOTICE_AUDIT}
    if type(source_pins) is not dict or not required <= source_pins.keys():
        raise ValueError("all sixteen metadata years and source audit pins are required")
    _check_pins(root, source_pins)
    if _json(root / INVENTORY).get("status") != "VERIFIED_METADATA_INVENTORY":
        raise ValueError("metadata inventory not verified")
    audit = _json(root / NOTICE_AUDIT)
    if (
        audit.get("status") != "VERIFIED_FULL_NOTICE_REVIEW_TERMINAL_PRESERVATION"
        or audit.get("source_history_admitted") is not False
    ):
        raise ValueError("notice terminal audit does not match capture scope")
    all_entries = []
    for relative in METADATA:
        rows = _json(root / relative)
        if type(rows) is not list:
            raise ValueError("annual metadata must be a list")
        year = relative.split("/")[-2]
        for number, row in enumerate(rows, 1):
            if type(row) is not dict or set(row) != MEMBER_KEYS | {"documents"}:
                raise ValueError("metadata record schema changed")
            if any(
                type(row[k]) is not str
                or (not row[k].strip() and not (k == "term_bucket" and row[k] == ""))
                for k in MEMBER_KEYS
            ):
                raise ValueError(
                    "metadata identity fields violate the inventory string schema"
                )
            if row["auction_date"][:4] != year:
                raise ValueError("auction does not belong to pinned annual metadata")
            all_entries.append(
                {"record": row, "metadata_path": relative, "metadata_row": number}
            )
    # The frozen guard validates dates and all six original URL roles before filtering.
    document_requests({"records": all_entries})
    selected = [
        entry | {"requested_roles": list(ROLES)}
        for entry in all_entries
        if entry["record"]["security_type"] in {"Note", "Bond"}
        and entry["record"]["term_bucket"] in TERMS
    ]
    selected.sort(key=lambda e: (e["record"]["auction_date"], e["record"]["cusip"]))
    full_requests = document_requests({"records": selected})
    identities = {
        (e["record"]["auction_date"], e["record"]["cusip"]): {
            k: e["record"][k] for k in sorted(MEMBER_KEYS)
        }
        for e in selected
    }
    pilot = _pilot_documents(_json(root / PILOT))
    requests = []
    for request in full_requests:
        memberships = [m for m in request["memberships"] if m["kind"] in ROLES]
        if not memberships:
            continue
        requests.append(
            {
                "url": request["url"],
                "format": request["format"],
                "memberships": memberships,
                "archive_memberships": [
                    identities[(m["auction_date"], m["cusip"])] | {"kind": m["kind"]}
                    for m in memberships
                ],
                "source_capture_mode": "reused_original_bytes"
                if request["url"] in pilot
                else "new_request",
            }
        )
    missing = [
        {
            "auction_date": e["record"]["auction_date"],
            "cusip": e["record"]["cusip"],
            "kind": kind,
        }
        for e in selected
        for kind in ROLES
        if not e["record"]["documents"][kind]
    ]
    reused = sum(r["source_capture_mode"] == "reused_original_bytes" for r in requests)
    counts = {
        "metadata_rows": len(all_entries),
        "selected_events": len(selected),
        "original_role_associations": sum(len(r["memberships"]) for r in full_requests),
        "requested_role_associations": sum(len(r["memberships"]) for r in requests),
        "unique_requested_urls": len(requests),
        "reused_urls": reused,
        "new_requests": len(requests) - reused,
        "missing_requested_roles": len(missing),
    }
    return {
        "status": "FIXED_FULL_HISTORY_DOCUMENT_SELECTION",
        "source_floor": "2010-01-01",
        "source_ceiling": "2025-10-20",
        "source_pins": dict(sorted(source_pins.items())),
        "requested_roles": list(ROLES),
        "records": selected,
        "requests": requests,
        "counts": counts,
        "missing_requested_roles": missing,
        "numerical_source_admitted": False,
        "predictive_comparisons_added": 0,
    }


def _reuse(root, request, pilot, pins):
    doc = pilot[request["url"]]
    if (
        doc.get("status") != "VERIFIED_TRANSPORT_SCHEMA_ONLY"
        or doc.get("format") != request["format"]
    ):
        raise ValueError("pilot original is not verified transport")
    body_path = _path(root, doc.get("body_path"))
    receipt_path = _path(root, doc.get("receipt_path"))
    headers_path = receipt_path.with_name("headers.txt")
    for path, expected in (
        (body_path, doc.get("body_sha256")),
        (receipt_path, doc.get("receipt_sha256")),
        (headers_path, doc.get("headers_sha256")),
    ):
        relative = str(path.relative_to(root))
        if relative not in pins or pins[relative] != expected or _hash(path) != expected:
            raise ValueError("reused original body/receipt/header is not pinned")
    old_memberships = doc.get("memberships")
    if (
        type(old_memberships) is not list
        or not old_memberships
        or any(m not in request["memberships"] for m in old_memberships)
        or len({json.dumps(m, sort_keys=True) for m in old_memberships})
        != len(old_memberships)
    ):
        raise ValueError("pilot memberships are not a unique subset of full history")
    validate_document_receipt(
        body_path.read_bytes(), _json(receipt_path), request["url"], request["format"]
    )
    return request | {
        "body_path": doc["body_path"],
        "body_sha256": doc["body_sha256"],
        "receipt_path": doc["receipt_path"],
        "receipt_sha256": doc["receipt_sha256"],
        "headers_path": str(headers_path.relative_to(root)),
        "headers_sha256": doc["headers_sha256"],
        "status": "VERIFIED_TRANSPORT_SCHEMA_ONLY",
    }


def capture_history(root: Path, selection_path: Path, checkpoint_path: Path) -> dict:
    """One bounded attempt; every transfer failure remains in the terminal report."""
    root = Path(root).resolve()
    selection_path = _path(root, str(Path(selection_path).resolve().relative_to(root)))
    checkpoint_path = _path(root, str(Path(checkpoint_path).resolve().relative_to(root)))
    private, report = root / PRIVATE, root / REPORT
    if private.exists() or private.is_symlink() or report.exists() or report.is_symlink():
        raise FileExistsError("history capture output already exists")
    checkpoint_hash, selection_hash = _hash(checkpoint_path), _hash(selection_path)
    checkpoint = _json(checkpoint_path)
    if checkpoint.get("status") != "READY_FULL_HISTORY_CAPTURE":
        raise ValueError("history checkpoint not ready")
    pins = checkpoint.get("pins")
    _check_pins(root, pins)
    if pins.get(str(selection_path.relative_to(root))) != selection_hash:
        raise ValueError("selection is not pinned")
    selection = _json(selection_path)
    source_pins = selection.get("source_pins")
    if type(source_pins) is not dict or any(pins.get(k) != v for k, v in source_pins.items()):
        raise ValueError("selection source pins differ from checkpoint")
    reconstructed = build_history_selection(root, source_pins)
    if json.dumps(selection, sort_keys=True) != json.dumps(reconstructed, sort_keys=True):
        raise ValueError("selection differs from full pinned metadata projection")
    counts = checkpoint.get("expected_counts")
    if (
        type(counts) is not dict
        or any(type(v) is not int for v in counts.values())
        or counts != reconstructed["counts"]
    ):
        raise ValueError("checkpoint expected counts differ from complete selection")
    pilot = _pilot_documents(_json(root / PILOT))
    requests = selection["requests"]
    reused = {
        r["url"]: _reuse(root, r, pilot, pins)
        for r in requests
        if r["source_capture_mode"] == "reused_original_bytes"
    }
    private.mkdir(parents=True, mode=0o700, exist_ok=False)
    private.chmod(0o700)
    _write(
        private / "started.json",
        {
            "started_utc": datetime.now(UTC).isoformat(),
            "checkpoint_sha256": checkpoint_hash,
            "selection_sha256": selection_hash,
            "source_pins": source_pins,
            "counts": counts,
            "requests": requests,
        },
    )

    def fetch(request):
        if request["url"] in reused:
            return reused[request["url"]]
        url, fmt = request["url"], request["format"]
        folder = private / hashlib.sha256(url.encode()).hexdigest()
        folder.mkdir(mode=0o700)
        body_path, headers = folder / f"body.{fmt}", folder / "headers.txt"
        # Precreate private files so curl never writes with a permissive umask.
        for path in (body_path, headers):
            with path.open("xb") as stream:
                os.fchmod(stream.fileno(), 0o600)
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
                    str((2 if fmt == "xml" else 8) * 1024 * 1024),
                    "--dump-header",
                    str(headers),
                    "--output",
                    str(body_path),
                    "--write-out",
                    "%{http_code}\n%{url_effective}\n%{content_type}",
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=50,
            )
            fields, exitcode, stderr = proc.stdout.split("\n"), proc.returncode, proc.stderr
        except (subprocess.TimeoutExpired, OSError, UnicodeError) as exc:
            fields, exitcode = [], 124
            stderr = f"{type(exc).__name__}: transport did not complete"
        raw = body_path.read_bytes()
        receipt = {
            "requested_url": url,
            "effective_url": fields[1] if len(fields) > 1 else "",
            "http_status": int(fields[0]) if fields and fields[0].isdigit() else 0,
            "curl_exit": exitcode,
            "content_type": fields[2] if len(fields) > 2 else "",
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "retrieved_utc": datetime.now(UTC).isoformat(),
        }
        receipt_path = folder / "receipt.json"
        _write(receipt_path, receipt)
        with (folder / "stderr.txt").open("x", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(stderr)
        entry = request | {
            "body_path": str(body_path.relative_to(root)),
            "body_sha256": receipt["sha256"],
            "receipt_path": str(receipt_path.relative_to(root)),
            "receipt_sha256": _hash(receipt_path),
            "headers_path": str(headers.relative_to(root)),
            "headers_sha256": _hash(headers),
        }
        try:
            validate_document_receipt(raw, receipt, url, fmt)
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
            raise ValueError("capture checkpoint or selection changed")
    except ValueError as exc:
        pin_errors.append(str(exc))
    success = not pin_errors and all(
        e["status"] == "VERIFIED_TRANSPORT_SCHEMA_ONLY" for e in entries
    )
    result = {
        "status": "CAPTURED_SCHEMA_ONLY" if success else "DOCUMENTS_REQUIRE_REVIEW",
        "completed_utc": datetime.now(UTC).isoformat(),
        "checkpoint_sha256": checkpoint_hash,
        "selection_sha256": selection_hash,
        "counts": counts,
        "documents": entries,
        "input_pin_errors": pin_errors,
        "numerical_source_admitted": False,
        "predictive_comparisons_added": 0,
        "inspection_scope": "Transport and PDF byte signature only. No XML/PDF text extraction, numerical decoding, source identity admission or historical timestamp claim.",
    }
    _write(report, result)
    return result
