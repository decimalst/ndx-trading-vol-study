"""Inspect only PDF release dates and CUSIPs before numerical table review."""

import hashlib
import json
from pathlib import Path

from pypdf import PdfReader

from src.treasury_pdf_identity import check_pdf_identity, probe_pdf_header

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    rep = root / "reports/next_signal_review"
    checkpoint = json.loads((rep / "TREASURY_PDF_METADATA_CHECKPOINT.json").read_text())
    h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for rel, value in checkpoint["pins"].items():
        if h(root / rel) != value:
            raise ValueError("PDF metadata checkpoint drift")
    capture = json.loads((rep / "TREASURY_DOCUMENT_CAPTURE.json").read_text())
    selected = json.loads((rep / "TREASURY_DOCUMENT_PILOT_SELECTION.json").read_text())
    events = {
        (x["record"]["auction_date"], x["record"]["cusip"]): x["record"]
        for x in selected["records"]
    }
    private = root / "data/source_discovery/treasury_auction/pdf_metadata_v1"
    private.mkdir(parents=True, exist_ok=False)
    results = []
    for entry in capture["documents"]:
        if entry["format"] != "pdf":
            continue
        raw = root / entry["body_path"]
        if h(raw) != entry["body_sha256"]:
            raise ValueError("PDF byte drift")
        reader = PdfReader(raw)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        target = private / (entry["body_sha256"] + ".txt")
        if not target.exists():
            with target.open("x") as f:
                f.write(text)
        row = {
            "url": entry["url"],
            "body_path": entry["body_path"],
            "body_sha256": entry["body_sha256"],
            "pages": len(reader.pages),
            "text_path": str(target.relative_to(root)),
            "text_sha256": h(target),
            "memberships": entry["memberships"],
        }
        try:
            probe = probe_pdf_header(text)
            row["metadata"] = probe
            check_pdf_identity(probe)
            for member in entry["memberships"]:
                if member["kind"] == "special_pdf":
                    continue
                event = events[(member["auction_date"], member["cusip"])]
                day = (
                    event["announcement_date"]
                    if member["kind"] == "announcement_pdf"
                    else event["auction_date"]
                )
                check_pdf_identity(probe, expected_date=day, expected_cusip=member["cusip"])
            row["status"] = "HEADER_DATE_AND_IDENTITY_MATCH"
        except ValueError as exc:
            row.update(status="REQUIRES_HEADER_REVIEW", error=str(exc))
        results.append(row)
    result = {
        "status": "PDF_METADATA_INSPECTED",
        "documents": results,
        "numerical_values_converted": False,
        "checkpoint_sha256": h(rep / "TREASURY_PDF_METADATA_CHECKPOINT.json"),
    }
    with (rep / "TREASURY_PDF_METADATA.json").open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(
        json.dumps(
            {
                "documents": len(results),
                "matched": sum(
                    x["status"] == "HEADER_DATE_AND_IDENTITY_MATCH" for x in results
                ),
                "review": sum(x["status"] == "REQUIRES_HEADER_REVIEW" for x in results),
            }
        )
    )
