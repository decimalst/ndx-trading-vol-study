"""Inspect the pinned known-notice capture; retain every source and header failure."""

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from pypdf import PdfReader

from src.treasury_notice_review_queue import template_fingerprint
from src.treasury_pdf_header_layout import check_pdf_identity, probe_pdf_header


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    rep = root / "reports/next_signal_review"
    cp = rep / "TREASURY_FULL_NOTICE_TEXT_CHECKPOINT.json"
    checkpoint = json.loads(cp.read_text())
    if checkpoint["status"] != "READY_FULL_NOTICE_TEXT_REVIEW":
        raise ValueError("Text inspection checkpoint is not ready")
    for name, expected in checkpoint["pins"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Notice text checkpoint drift: {name}")
    capture = json.loads((rep / "TREASURY_FULL_NOTICE_CAPTURE.json").read_text())
    previous = json.loads((rep / "TREASURY_PDF_METADATA.json").read_text())
    private = root / "data/source_discovery/treasury_auction/full_notice_text_v1"
    report = rep / "TREASURY_FULL_NOTICE_TEXT.json"
    if private.exists() or report.exists():
        raise ValueError("Preserve the existing text inspection; no implicit replay")
    if len(capture["documents"]) != 253:
        raise ValueError("Known notice inventory differs from frozen selection")
    private.mkdir()
    templates = private / "templates"
    templates.mkdir()
    started = {
        "started_utc": datetime.now(UTC).isoformat(),
        "checkpoint_sha256": digest(cp),
        "capture_sha256": digest(rep / "TREASURY_FULL_NOTICE_CAPTURE.json"),
    }
    (private / "started.json").write_text(json.dumps(started, indent=2) + "\n")
    cache = {}
    for row in previous["documents"]:
        path = root / row["text_path"]
        if digest(path) != row["text_sha256"]:
            raise ValueError("Pilot text changed before reuse")
        cache[row["body_sha256"]] = {
            "text_path": row["text_path"],
            "text_sha256": row["text_sha256"],
            "pages": row["pages"],
            "text_source": "reused_pilot_text",
        }
    rows, groups = [], {}
    for entry in capture["documents"]:
        row = {
            k: entry[k]
            for k in (
                "url",
                "body_path",
                "body_sha256",
                "archive_memberships",
                "source_capture_mode",
            )
        }
        if entry["status"] != "VERIFIED_TRANSPORT_SCHEMA_ONLY":
            row.update(status="TRANSPORT_REQUIRES_REVIEW", error=entry.get("error"))
            rows.append(row)
            continue
        source = root / entry["body_path"]
        if digest(source) != entry["body_sha256"]:
            raise ValueError("Notice bytes changed before extraction")
        try:
            if entry["body_sha256"] not in cache:
                reader = PdfReader(source)
                if not 1 <= len(reader.pages) <= 20:
                    raise ValueError("Notice page count outside the bounded review")
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                path = private / (entry["body_sha256"] + ".txt")
                with path.open("x") as out:
                    out.write(text)
                cache[entry["body_sha256"]] = {
                    "text_path": str(path.relative_to(root)),
                    "text_sha256": digest(path),
                    "pages": len(reader.pages),
                    "text_source": "extracted_in_full_batch",
                }
            row.update(cache[entry["body_sha256"]])
            text = (root / row["text_path"]).read_text()
            try:
                row["header_metadata"] = probe_pdf_header(text)
                row["release_date"] = check_pdf_identity(row["header_metadata"])
                row["status"] = "DATED_NOTICE_TEXT_READY_FOR_REVIEW"
            except ValueError as exc:
                row.update(status="HEADER_REQUIRES_REVIEW", error=str(exc))
            # Lexically masking digits does not decode financial quantities or
            # admit an uncertain clock. Every header status stays on its row.
            fingerprint = template_fingerprint(text)
            key = fingerprint["template_sha256"]
            row["template_sha256"] = key
            if key not in groups:
                path = templates / (key + ".txt")
                with path.open("x") as out:
                    out.write(fingerprint["normalized_text"])
                groups[key] = {
                    "template_path": str(path.relative_to(root)),
                    "template_sha256": digest(path),
                    "urls": [],
                }
            groups[key]["urls"].append(entry["url"])
        except Exception as exc:
            row.update(
                status="TEXT_REQUIRES_REVIEW", error=f"{type(exc).__name__}: {exc}"[:400]
            )
        rows.append(row)
    result = {
        **started,
        "status": "COMPLETED_FULL_NOTICE_TEXT_INSPECTION",
        "completed_utc": datetime.now(UTC).isoformat(),
        "documents": rows,
        "counts": dict(Counter(row["status"] for row in rows)),
        "template_groups": groups,
        "purpose_classifications_completed_by_this_script": 0,
        "numeric_financial_values_converted": False,
        "source_history_admitted": False,
        "registered_comparisons": 144,
    }
    with report.open("x") as out:
        out.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"counts": result["counts"], "templates": len(groups)}))


if __name__ == "__main__":
    main()
