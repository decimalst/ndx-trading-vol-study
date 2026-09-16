"""Bind full captured announcement/result identities without financial values."""

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from src.treasury_history_bindings import bind_inventory, pinned_bytes
from src.treasury_realized_identity import UnsupportedIdentity, reconcile_realized_identity


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    rep = root / "reports/next_signal_review"
    cp_path = rep / "TREASURY_HISTORY_IDENTITY_CHECKPOINT.json"
    initial_checkpoint_sha256 = digest(cp_path)
    cp = json.loads(cp_path.read_text())

    def required(relative, expected=None):
        return pinned_bytes(root, cp["pins"], relative, expected)

    def manifest(name):
        return json.loads(required("reports/next_signal_review/" + name))

    if cp["status"] != "READY_FULL_HISTORY_IDENTITY_RECONCILIATION":
        raise ValueError("Full history identity checkpoint is not ready")
    for name, expected in cp["pins"].items():
        if digest(root / name) != expected:
            raise ValueError(f"History identity source drift: {name}")
    selection = manifest("TREASURY_HISTORY_SELECTION.json")
    metadata = manifest("TREASURY_HISTORY_METADATA.json")
    capture = manifest("TREASURY_HISTORY_CAPTURE.json")
    reviewed = manifest("TREASURY_FULL_NOTICE_REVIEW.json")
    canonical = manifest("TREASURY_REALIZED_CONFIRMATIONS.json")
    audit = manifest("TREASURY_REALIZED_CONFIRMATIONS_AUDIT.json")
    if audit.get("status") != "VERIFIED_METADATA_TRANSCRIPTION":
        raise ValueError("Canonical confirmation audit is not verified")
    for name, expected in audit["input_hashes"].items():
        required(name, expected)
    if audit["discrepancies"]:
        raise ValueError("Confirmation transcription audit has discrepancies")
    bind_inventory(selection, capture, metadata)
    if metadata["capture_sha256"] != digest(rep / "TREASURY_HISTORY_CAPTURE.json") or capture[
        "selection_sha256"
    ] != digest(rep / "TREASURY_HISTORY_SELECTION.json"):
        raise ValueError("History manifests disagree on source bindings")
    if len({d["url"] for d in reviewed["documents"]}) != len(reviewed["documents"]):
        raise ValueError("Duplicate reviewed notice URL")
    docs = {d["url"]: d for d in metadata["documents"]}
    captures = {d["url"]: d for d in capture["documents"]}
    notices = {d["url"]: d for d in reviewed["documents"]}
    confirmations = {}
    for record in canonical["records"]:
        fact = record["confirmation"]
        notice = notices[fact["url"]]
        required(record["review_ledger"], record["review_ledger_sha256"])
        required(record["source_text_path"], fact["text_sha256"])
        if (fact["body_sha256"], fact["text_sha256"]) != (
            notice["body_sha256"],
            notice["text_sha256"],
        ):
            raise ValueError("Confirmation is not bound to its reviewed source")
        key = (fact["auction_date"], fact["actual_cusip"])
        if key in confirmations:
            raise ValueError("Duplicate canonical event confirmation")
        confirmations[key] = fact
    if len(selection["records"]) != 1138 or len(docs) != 4552 or len(confirmations) != 18:
        raise ValueError("Frozen history or confirmation scope changed")
    output = rep / "TREASURY_HISTORY_IDENTITY.json"
    if output.exists():
        raise ValueError("Preserve prior identity reconciliation; no implicit replay")
    events = []
    for selected in selection["records"]:
        record = selected["record"]
        key = (record["auction_date"], record["cusip"])
        row = {
            k: record[k]
            for k in (
                "auction_date",
                "announcement_date",
                "cusip",
                "security_term",
                "term_bucket",
            )
        }
        row.update(source_history_admitted=False, financial_values_converted=False)
        links = record["documents"]
        row["known_notice_dispositions"] = [
            {"url": url, "disposition": notices[url]["disposition"]}
            for url in links["special_pdf"]
        ]
        row["unavailable_notice_urls"] = [
            url for url in links["special_pdf"] if notices[url]["release_date"] is None
        ]
        if key == ("2019-06-21", "9128286T2"):
            row["status"] = "EXCLUDED_DOCUMENTED_CONTINGENCY_TEST"
            events.append(row)
            continue
        try:
            for role in (
                "announcement_pdf",
                "announcement_xml",
                "competitive_pdf",
                "competitive_xml",
            ):
                if len(links[role]) != 1:
                    raise ValueError("Required role is missing or ambiguous")
                source = docs[links[role][0]]
                required = (
                    "PDF_DATED_HEADER_EXTRACTED"
                    if role.endswith("pdf")
                    else "XML_METADATA_EXTRACTED"
                )
                if source["status"] != required:
                    raise ValueError(f"Unresolved source metadata: {role}: {source['status']}")
            apdf, axml, rpdf, rxml = (
                links[k][0]
                for k in (
                    "announcement_pdf",
                    "announcement_xml",
                    "competitive_pdf",
                    "competitive_xml",
                )
            )
            a_text = required(docs[apdf]["text_path"], docs[apdf]["text_sha256"]).decode(
                "utf-8"
            )
            r_text = required(docs[rpdf]["text_path"], docs[rpdf]["text_sha256"]).decode(
                "utf-8"
            )
            a_raw = required(captures[axml]["body_path"], captures[axml]["body_sha256"])
            r_raw = required(captures[rxml]["body_path"], captures[rxml]["body_sha256"])
            confirmation = confirmations.get(key)
            if confirmation is not None and confirmation["url"] not in links["special_pdf"]:
                raise ValueError(
                    "Final confirmation is not associated with the selected event"
                )
            identity = reconcile_realized_identity(
                a_raw,
                a_text,
                r_raw,
                r_text,
                auction_date=record["auction_date"],
                announcement_date=record["announcement_date"],
                selected_final_cusip=record["cusip"],
                confirmation=confirmation,
            )
            expected_inputs = {
                "announcement_xml_sha256": captures[axml]["body_sha256"],
                "result_xml_sha256": captures[rxml]["body_sha256"],
                "announcement_pdf_text_sha256": docs[apdf]["text_sha256"],
                "result_pdf_text_sha256": docs[rpdf]["text_sha256"],
            }
            if identity["input_hashes"] != expected_inputs:
                raise ValueError("Reconciled identity does not bind the expected source inputs")
            for fields, label, expected in (
                (
                    identity["announcement_metadata"]["announcement"],
                    "AnnouncementPDFName",
                    Path(apdf).name,
                ),
                (identity["result_metadata"]["results"], "ResultsPDFName", Path(rpdf).name),
            ):
                if fields.get(label) not in (None, "", expected):
                    raise ValueError("Source file-role identity disagreement")
            row.update(
                status="PAIRED_METADATA_RECONCILED",
                identity=identity,
                confirmation_bound_to_review=confirmation is not None,
            )
        except UnsupportedIdentity as exc:
            row.update(
                status="EXPLICIT_UNSUPPORTED_IDENTITY_CONTEXT",
                error=str(exc),
                unsupported_evidence=exc.audit,
            )
        except ValueError as exc:
            row.update(status="IDENTITY_REQUIRES_REVIEW", error=str(exc))
        events.append(row)
    result = {
        "status": "COMPLETED_FULL_HISTORY_IDENTITY_RECONCILIATION",
        "completed_utc": datetime.now(UTC).isoformat(),
        "checkpoint_sha256": initial_checkpoint_sha256,
        "events": events,
        "counts": dict(Counter(r["status"] for r in events)),
        "error_counts": dict(Counter(r["error"] for r in events if "error" in r)),
        "financial_values_converted": False,
        "source_history_admitted": False,
        "new_predictive_comparisons": 0,
        "registered_comparisons": 144,
    }
    for name, expected in cp["pins"].items():
        required(name, expected)
    if digest(cp_path) != initial_checkpoint_sha256:
        raise ValueError("Identity checkpoint changed during execution")
    with output.open("x") as out:
        out.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"counts": result["counts"], "error_counts": result["error_counts"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
