"""Reconcile saved auction quantities after bound paired-identity checks."""

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from src.treasury_auction_amounts import FIELDS
from src.treasury_history_amounts import parse_history_result_amounts
from src.treasury_history_bindings import bind_inventory, pinned_bytes
from src.treasury_history_offerings import reconcile_history_offering
from src.treasury_pdf_amounts import parse_pdf_amounts


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    rep = root / "reports/next_signal_review"
    checkpoint_path = rep / "TREASURY_HISTORY_ACCOUNTING_CHECKPOINT.json"
    checkpoint_hash = digest(checkpoint_path)
    cp = json.loads(checkpoint_path.read_text())
    if cp["status"] != "READY_FULL_HISTORY_ACCOUNTING":
        raise ValueError("Full history accounting checkpoint is not ready")

    def required(path, expected=None):
        return pinned_bytes(root, cp["pins"], path, expected)

    def manifest(name):
        return json.loads(required("reports/next_signal_review/" + name))

    for path, expected in cp["pins"].items():
        required(path, expected)
    selection = manifest("TREASURY_HISTORY_SELECTION.json")
    capture = manifest("TREASURY_HISTORY_CAPTURE.json")
    metadata = manifest("TREASURY_HISTORY_METADATA.json")
    identities = manifest("TREASURY_HISTORY_IDENTITY_V3.json")
    required("reports/next_signal_review/TREASURY_HISTORY_IDENTITY_CHECKPOINT_V4.json")
    notices = manifest("TREASURY_FULL_NOTICE_REVIEW.json")
    bind_inventory(selection, capture, metadata)
    if identities["checkpoint_sha256"] != digest(
        rep / "TREASURY_HISTORY_IDENTITY_CHECKPOINT_V4.json"
    ):
        raise ValueError("Paired identities do not bind the expected checkpoint")
    events = identities["events"]
    selected_keys = [
        (x["record"]["auction_date"], x["record"]["cusip"]) for x in selection["records"]
    ]
    if (
        len(selected_keys) != 1138
        or [(x["auction_date"], x["cusip"]) for x in events] != selected_keys
    ):
        raise ValueError("Identity event scope differs from exact selected history")
    sources = {entry["url"]: entry for entry in capture["documents"]}
    docs = {entry["url"]: entry for entry in metadata["documents"]}
    notice_map = {entry["url"]: entry for entry in notices["documents"]}
    if len(notice_map) != len(notices["documents"]):
        raise ValueError("Duplicate notice source")
    output = rep / "TREASURY_HISTORY_ACCOUNTING.json"
    private = root / "data/source_discovery/treasury_auction/history_accounting_v1"
    if output.exists() or private.exists():
        raise ValueError("Preserve prior full-history accounting; no implicit replay")
    private.mkdir(mode=0o700)
    (private / "started.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": checkpoint_hash,
                "started_utc": datetime.now(UTC).isoformat(),
            }
        )
        + "\n"
    )
    rows = []
    for selected, event in zip(selection["records"], events, strict=True):
        record = selected["record"]
        row = {key: record[key] for key in ("auction_date", "announcement_date", "cusip")}
        row.update(prior_identity_status=event["status"], source_history_admitted=False)
        row["unavailable_notice_urls"] = event["unavailable_notice_urls"]
        row["known_notice_dispositions"] = event["known_notice_dispositions"]
        if event["status"] != "PAIRED_METADATA_RECONCILED":
            row["status"] = "NOT_APPLIED_PRIOR_IDENTITY_DISPOSITION_RETAINED"
            rows.append(row)
            continue
        if event["unavailable_notice_urls"]:
            row["status"] = "NOT_APPLIED_UNAVAILABLE_NOTICE_RETAINED"
            rows.append(row)
            continue
        identity = event["identity"]
        links = record["documents"]
        for url in links["special_pdf"]:
            if notice_map[url]["release_date"] is None:
                raise ValueError("An unknown notice was concealed by the identity ledger")
        apdf, axml, rpdf, rxml = (
            links[key][0]
            for key in (
                "announcement_pdf",
                "announcement_xml",
                "competitive_pdf",
                "competitive_xml",
            )
        )
        a_text = required(docs[apdf]["text_path"], docs[apdf]["text_sha256"]).decode("utf-8")
        r_text = required(docs[rpdf]["text_path"], docs[rpdf]["text_sha256"]).decode("utf-8")
        a_raw = required(sources[axml]["body_path"], sources[axml]["body_sha256"])
        r_raw = required(sources[rxml]["body_path"], sources[rxml]["body_sha256"])
        expected = {
            "announcement_xml_sha256": sources[axml]["body_sha256"],
            "result_xml_sha256": sources[rxml]["body_sha256"],
            "announcement_pdf_text_sha256": docs[apdf]["text_sha256"],
            "result_pdf_text_sha256": docs[rpdf]["text_sha256"],
        }
        if identity["input_hashes"] != expected:
            raise ValueError("Paired identity does not authenticate these exact source bytes")
        # Each XML keeps its own announced identity; the full paired pass has
        # already authenticated substitutions and amended announcement dates.
        own_ids = {
            "announcement_pdf": {
                "auction_date": row["auction_date"],
                "announcement_date": identity["announcement_release_date"],
                "cusip": identity["announced_cusip"],
            }
        }
        for role, field in (
            ("announcement_xml", "announcement_metadata"),
            ("result_xml", "result_metadata"),
        ):
            own = identity[field]["announcement"]
            own_ids[role] = {
                "auction_date": row["auction_date"],
                "announcement_date": row["announcement_date"],
                "cusip": own["CUSIP"],
                "announced_cusip": own.get("AnnouncedCUSIP") or None,
            }
        saved = {
            "event": row.copy(),
            "identity_source_hashes": expected,
            "own_document_identities": own_ids,
            "stages": {},
            "source_history_admitted": False,
        }
        amounts = {}
        calls = {
            "result_xml": partial(
                parse_history_result_amounts,
                r_raw,
                auction_date=row["auction_date"],
                announcement_date=row["announcement_date"],
                cusip=identity["actual_cusip"],
                announced_cusip=identity["announced_cusip"],
            ),
            "result_pdf": partial(
                parse_pdf_amounts,
                r_text,
                auction_date=row["auction_date"],
                cusip=identity["actual_cusip"],
            ),
            "offering": partial(
                reconcile_history_offering,
                a_text,
                a_raw,
                r_raw,
                identities=own_ids,
                xml_unit="USD_BILLIONS",
            ),
        }
        for stage, call in calls.items():
            try:
                result = call()
                if stage != "offering":
                    if set(result) != set(FIELDS):
                        raise ValueError("Result amount field scope changed")
                    amounts[stage] = result
                    result = {key: str(value) for key, value in result.items()}
                saved["stages"][stage] = {"status": "PARSED", "result": result}
            except ValueError as exc:
                saved["stages"][stage] = {"status": "REQUIRES_REVIEW", "error": str(exc)}
        mismatches = []
        if len(amounts) == 2:
            mismatches = [
                key
                for key in FIELDS
                if amounts["result_xml"][key] != amounts["result_pdf"][key]
            ]
        success = (
            all(x["status"] == "PARSED" for x in saved["stages"].values()) and not mismatches
        )
        row.update(
            status="PAIRED_SOURCE_AMOUNTS_AND_OFFERING_MATCH"
            if success
            else "SOURCE_ACCOUNTING_REQUIRES_REVIEW",
            paired_field_comparisons=16 if len(amounts) == 2 else 0,
            mismatched_fields=mismatches,
            stage_statuses={key: value["status"] for key, value in saved["stages"].items()},
        )
        row["errors"] = {
            key: value["error"] for key, value in saved["stages"].items() if "error" in value
        }
        saved["comparison"] = {"status": row["status"], "mismatched_fields": mismatches}
        destination = private / f"{row['auction_date']}_{row['cusip']}.json"
        with destination.open("x") as stream:
            json.dump(saved, stream, indent=2, sort_keys=True)
            stream.write("\n")
        destination.chmod(0o600)
        row.update(
            private_output=str(destination.relative_to(root)),
            private_output_sha256=digest(destination),
        )
        rows.append(row)
    report = {
        "status": "COMPLETED_FULL_HISTORY_SOURCE_ACCOUNTING",
        "completed_utc": datetime.now(UTC).isoformat(),
        "checkpoint_sha256": checkpoint_hash,
        "identity_report_sha256": digest(rep / "TREASURY_HISTORY_IDENTITY_V3.json"),
        "events": rows,
        "counts": dict(Counter(row["status"] for row in rows)),
        "error_counts": dict(
            Counter(message for row in rows for message in row.get("errors", {}).values())
        ),
        "source_history_admitted": False,
        "new_predictive_comparisons": 0,
        "registered_comparisons": 144,
    }
    for path, expected_hash in cp["pins"].items():
        required(path, expected_hash)
    if digest(checkpoint_path) != checkpoint_hash:
        raise ValueError("Accounting checkpoint changed during source pass")
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {"counts": report["counts"], "error_counts": report["error_counts"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
