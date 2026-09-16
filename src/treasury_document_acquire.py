"""Capture the fixed documentary sample without admitting numerical values."""

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from src.treasury_auction_document_probe import inspect_xml_schema
from src.treasury_auction_documents import document_requests, validate_document_receipt


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def capture_documents(root: Path, selection_path: Path, checkpoint_path: Path):
    checkpoint = json.loads(checkpoint_path.read_text())
    if checkpoint.get("status") != "READY_DOCUMENT_SCHEMA_CAPTURE":
        raise ValueError("document checkpoint not ready")
    for rel, expected in checkpoint["pins"].items():
        if _hash(root / rel) != expected:
            raise ValueError(f"checkpoint drift: {rel}")
    if str(selection_path.relative_to(root)) not in checkpoint["pins"]:
        raise ValueError("selection is not pinned")
    requests = document_requests(json.loads(selection_path.read_text()))
    private = root / "data/source_discovery/treasury_auction/document_pilot_v1"
    report = root / "reports/next_signal_review/TREASURY_DOCUMENT_CAPTURE.json"
    if report.exists():
        raise FileExistsError("document capture report exists")
    private.mkdir(parents=True, exist_ok=False)
    _write(
        private / "started.json",
        {
            "started_utc": datetime.now(UTC).isoformat(),
            "checkpoint_sha256": _hash(checkpoint_path),
            "requests": requests,
        },
    )

    def fetch(request):
        url, fmt = request["url"], request["format"]
        folder = private / hashlib.sha256(url.encode()).hexdigest()
        folder.mkdir()
        body_path = folder / f"body.{fmt}"
        headers = folder / "headers.txt"
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
            fields = proc.stdout.split("\n")
            exitcode = proc.returncode
            stderr = proc.stderr
        except (subprocess.TimeoutExpired, OSError) as exc:
            fields = []
            exitcode = 124
            stderr = f"{type(exc).__name__}: transport did not complete"
        raw = body_path.read_bytes() if body_path.exists() else b""
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
        _write(folder / "receipt.json", receipt)
        with (folder / "stderr.txt").open("x") as f:
            f.write(stderr)
        entry = {
            **request,
            "body_path": str(body_path.relative_to(root)),
            "body_sha256": receipt["sha256"],
            "receipt_path": str((folder / "receipt.json").relative_to(root)),
            "receipt_sha256": _hash(folder / "receipt.json"),
            "headers_sha256": _hash(headers) if headers.exists() else None,
        }
        try:
            validate_document_receipt(raw, receipt, url, fmt)
            if fmt == "xml":
                schema = inspect_xml_schema(raw)
                _write(folder / "schema.json", schema)
                entry.update(
                    schema_path=str((folder / "schema.json").relative_to(root)),
                    schema_sha256=_hash(folder / "schema.json"),
                )
            entry["status"] = "VERIFIED_TRANSPORT_SCHEMA_ONLY"
        except ValueError as exc:
            entry.update(status="REQUIRES_REVIEW", error=str(exc))
        _write(folder / "summary.json", entry)
        return entry

    with ThreadPoolExecutor(max_workers=3) as pool:
        entries = list(pool.map(fetch, requests))
    result = {
        "status": "CAPTURED_SCHEMA_ONLY"
        if all(x["status"] == "VERIFIED_TRANSPORT_SCHEMA_ONLY" for x in entries)
        else "DOCUMENTS_REQUIRE_REVIEW",
        "completed_utc": datetime.now(UTC).isoformat(),
        "checkpoint_sha256": _hash(checkpoint_path),
        "selection_sha256": _hash(selection_path),
        "documents": entries,
        "numerical_source_admitted": False,
        "predictive_comparisons_added": 0,
        "inspection_scope": "XML tag paths and attribute names only; PDF byte signature only; no scalar source content exported.",
    }
    _write(report, result)
    return result
