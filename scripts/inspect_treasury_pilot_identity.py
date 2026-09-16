"""Reconcile only dated PDF/XML identity metadata for the fixed source pilot."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from src.treasury_pdf_header_layout import check_pdf_identity, probe_pdf_header
from src.treasury_xml_identity import check_event_identity, read_xml_identity

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    rep = root / "reports/next_signal_review"

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    checkpoint = json.loads((rep / "TREASURY_PILOT_IDENTITY_CHECKPOINT.json").read_text())
    for path, value in checkpoint["pins"].items():
        if digest(root / path) != value:
            raise ValueError("pilot identity checkpoint drift")
    capture = json.loads((rep / "TREASURY_DOCUMENT_CAPTURE.json").read_text())
    previous = json.loads((rep / "TREASURY_PDF_METADATA.json").read_text())
    pdf_text = {x["url"]: x for x in previous["documents"]}
    selection = json.loads((rep / "TREASURY_DOCUMENT_PILOT_SELECTION.json").read_text())
    events = {
        (x["record"]["auction_date"], x["record"]["cusip"]): x["record"]
        for x in selection["records"]
    }
    rows = []
    for document in capture["documents"]:
        body = root / document["body_path"]
        if digest(body) != document["body_sha256"]:
            raise ValueError("document body drift")
        item = {
            k: document[k]
            for k in ["url", "format", "body_path", "body_sha256", "memberships"]
        }
        try:
            if document["format"] == "pdf":
                earlier = pdf_text[document["url"]]
                text = root / earlier["text_path"]
                if digest(text) != earlier["text_sha256"]:
                    raise ValueError("PDF text drift")
                metadata = probe_pdf_header(text.read_text())
                item["metadata"] = metadata
                check_pdf_identity(metadata)
                item["release_date_checked"] = True
                for member in document["memberships"]:
                    if member["kind"] == "special_pdf":
                        continue
                    event = events[(member["auction_date"], member["cusip"])]
                    expected = (
                        event["announcement_date"]
                        if member["kind"] == "announcement_pdf"
                        else event["auction_date"]
                    )
                    check_pdf_identity(
                        metadata, expected_date=expected, expected_cusip=member["cusip"]
                    )
            else:
                metadata = read_xml_identity(body.read_bytes())
                item["metadata"] = metadata
                for member in document["memberships"]:
                    event = events[(member["auction_date"], member["cusip"])]
                    check_event_identity(
                        metadata,
                        auction_date=event["auction_date"],
                        announcement_date=event["announcement_date"],
                        cusip=event["cusip"],
                    )
            item["status"] = "METADATA_MATCH"
        except ValueError as exc:
            item.update(status="METADATA_REQUIRES_REVIEW", error=str(exc))
        rows.append(item)
    result = {
        "status": "FIXED_PILOT_METADATA_REVIEW",
        "documents": rows,
        "checkpoint_sha256": digest(rep / "TREASURY_PILOT_IDENTITY_CHECKPOINT.json"),
        "numerical_values_converted": False,
        "scope": "PDF header date/security CUSIP and selected XML identity descriptors only; result ReleaseTime does not authenticate its date alone.",
    }
    with (rep / "TREASURY_PILOT_IDENTITY.json").open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(dict(Counter(x["status"] for x in rows))))
