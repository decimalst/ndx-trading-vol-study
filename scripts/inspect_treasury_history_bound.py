"""Read full captured documents once, keeping metadata and every failure.

Run with the pinned bundled pypdf interpreter and PYTHONPATH set to the repo.
This step extracts identity evidence; it does not reconcile events or amounts.
"""

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from pypdf import PdfReader

from src.treasury_history_bindings import bind_inventory, pinned_bytes
from src.treasury_history_inspection import inspect_history_payload


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as out:
        out.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    path.chmod(0o600)


def main():
    root = Path(__file__).resolve().parents[1]
    rep = root / "reports/next_signal_review"
    cp = rep / "TREASURY_HISTORY_INSPECTION_CHECKPOINT.json"
    initial_checkpoint_sha256 = digest(cp)
    checkpoint = json.loads(cp.read_text())

    def required(relative, expected=None):
        return pinned_bytes(root, checkpoint["pins"], relative, expected)

    if checkpoint["status"] != "READY_FULL_HISTORY_METADATA_INSPECTION":
        raise ValueError("History metadata checkpoint is not ready")
    for name, expected in checkpoint["pins"].items():
        if digest(root / name) != expected:
            raise ValueError(f"History inspection checkpoint drift: {name}")
    capture_path = rep / "TREASURY_HISTORY_CAPTURE.json"
    capture = json.loads(required(str(capture_path.relative_to(root))))
    selection_path = rep / "TREASURY_HISTORY_SELECTION.json"
    selection_raw = required(str(selection_path.relative_to(root)))
    selection = json.loads(selection_raw)
    if (
        capture["selection_sha256"] != hashlib.sha256(selection_raw).hexdigest()
        or capture["input_pin_errors"]
    ):
        raise ValueError("Capture is not bound to the selected source inventory")
    bind_inventory(selection, capture)
    if len(capture["documents"]) != checkpoint["expected_document_count"]:
        raise ValueError("Capture document count differs from declared inventory")
    urls = [d["url"] for d in capture["documents"]]
    if len(set(urls)) != len(urls):
        raise ValueError("Duplicate capture document URL")
    previous = json.loads(required("reports/next_signal_review/TREASURY_PDF_METADATA.json"))
    private = root / "data/source_discovery/treasury_auction/history_metadata_v1"
    output = rep / "TREASURY_HISTORY_METADATA.json"
    if private.exists() or output.exists():
        raise ValueError("Preserve prior history inspection; no implicit replay")
    private.mkdir(mode=0o700)
    started = {
        "started_utc": datetime.now(UTC).isoformat(),
        "checkpoint_sha256": initial_checkpoint_sha256,
        "capture_sha256": digest(capture_path),
    }
    save(private / "started.json", started)
    cache = {}
    for row in previous["documents"]:
        path = root / row["text_path"]
        required(row["text_path"], row["text_sha256"])
        cache[row["body_sha256"]] = {
            "text_path": row["text_path"],
            "text_sha256": row["text_sha256"],
            "pages": row["pages"],
            "text_source": "reused_pilot_text",
        }
    rows = []
    for index, entry in enumerate(capture["documents"], 1):
        row = {
            k: entry[k]
            for k in (
                "url",
                "format",
                "memberships",
                "body_path",
                "body_sha256",
                "source_capture_mode",
            )
        }
        if entry["status"] != "VERIFIED_TRANSPORT_SCHEMA_ONLY":
            row.update(status="TRANSPORT_REQUIRES_REVIEW", error=entry.get("error"))
            rows.append(row)
            continue
        source = root / entry["body_path"]
        raw = required(entry["body_path"], entry["body_sha256"])
        try:
            if entry["format"] == "xml":
                row.update(inspect_history_payload(raw, "xml"))
            else:
                key = entry["body_sha256"]
                if key not in cache:
                    reader = PdfReader(source)
                    if not 1 <= len(reader.pages) <= 20:
                        raise ValueError("PDF page count outside bounded inspection")
                    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
                    path = private / f"{key}.txt"
                    with path.open("x", encoding="utf-8") as out:
                        out.write(extracted)
                    path.chmod(0o600)
                    cache[key] = {
                        "text_path": str(path.relative_to(root)),
                        "text_sha256": digest(path),
                        "pages": len(reader.pages),
                        "text_source": "extracted_in_full_history",
                    }
                row.update(cache[key])
                text_path = root / row["text_path"]
                if row["text_source"] == "reused_pilot_text":
                    pdf_text = required(row["text_path"], row["text_sha256"]).decode("utf-8")
                else:
                    if digest(text_path) != row["text_sha256"]:
                        raise ValueError("Newly extracted text changed before inspection")
                    pdf_text = text_path.read_text()
                row.update(inspect_history_payload(raw, "pdf", pdf_text=pdf_text))
        except Exception as exc:
            row.update(
                status="EXTRACTION_REQUIRES_REVIEW", error=f"{type(exc).__name__}: {exc}"[:400]
            )
        rows.append(row)
        if index % 250 == 0:
            print(json.dumps({"inspected": index, "total": len(urls)}), flush=True)
    result = {
        **started,
        "status": "COMPLETED_FULL_HISTORY_METADATA_INSPECTION",
        "completed_utc": datetime.now(UTC).isoformat(),
        "documents": rows,
        "counts": dict(Counter(row["status"] for row in rows)),
        "numeric_financial_values_converted": False,
        "events_reconciled": False,
        "source_history_admitted": False,
        "registered_comparisons": 144,
        "new_predictive_comparisons": 0,
    }
    for name, expected in checkpoint["pins"].items():
        required(name, expected)
    if digest(cp) != initial_checkpoint_sha256:
        raise ValueError("Inspection checkpoint changed during execution")
    save(output, result)
    print(json.dumps(result["counts"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
