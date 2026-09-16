"""Add the stated offering control to the fixed pilot's reconciled source pairs."""

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from src.treasury_offering_layout import reconcile_offering
from src.treasury_pdf_header_layout import check_pdf_identity, probe_pdf_header
from src.treasury_xml_identity import check_event_identity, read_xml_identity


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as out:
        out.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    path.chmod(0o600)


def main():
    root = Path(__file__).resolve().parents[1]
    rep = root / "reports/next_signal_review"
    checkpoint_path = rep / "TREASURY_PILOT_OFFERING_CHECKPOINT_V2.json"
    cp = json.loads(checkpoint_path.read_text())
    if cp["status"] != "READY_PILOT_OFFERING_RECONCILIATION":
        raise ValueError("Offering checkpoint not ready")
    for name, expected in cp["pins"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Offering checkpoint drift: {name}")
    accounting = json.loads((rep / "TREASURY_PILOT_ACCOUNTING.json").read_text())
    selection = json.loads((rep / "TREASURY_DOCUMENT_PILOT_SELECTION.json").read_text())
    capture = json.loads((rep / "TREASURY_DOCUMENT_CAPTURE.json").read_text())
    extracted = json.loads((rep / "TREASURY_PDF_METADATA.json").read_text())
    events = {(d["auction_date"], d["cusip"]): d for d in accounting["events"]}
    bodies = {d["url"]: d for d in capture["documents"]}
    texts = {d["url"]: d for d in extracted["documents"]}
    if len(events) != 22 or len(selection["records"]) != 22:
        raise ValueError("Fixed pilot membership changed")
    private = root / "data/source_discovery/treasury_auction/offering_pilot_v2"
    report = rep / "TREASURY_PILOT_OFFERING_V2.json"
    if private.exists() or report.exists():
        raise ValueError("Preserve existing offering reconciliation; no replay")
    private.mkdir(mode=0o700)
    started = {
        "started_utc": datetime.now(UTC).isoformat(),
        "checkpoint_sha256": digest(checkpoint_path),
    }
    save(private / "started.json", started)
    rows = []
    for selected in selection["records"]:
        record = selected["record"]
        event = {k: record[k] for k in ("auction_date", "announcement_date", "cusip")}
        prior = events[(event["auction_date"], event["cusip"])]
        row = event | {"prior_accounting_status": prior["status"], "source_admitted": False}
        if prior["status"] != "PAIRED_SOURCE_AMOUNTS_MATCH":
            row["status"] = "NOT_APPLIED_PRIOR_SOURCE_DISPOSITION_RETAINED"
            rows.append(row)
            continue
        links = record["documents"]
        apdf, axml, rxml = (
            links[k][0] for k in ("announcement_pdf", "announcement_xml", "competitive_xml")
        )
        source_paths = {
            "announcement_pdf": root / texts[apdf]["text_path"],
            "announcement_xml": root / bodies[axml]["body_path"],
            "result_xml": root / bodies[rxml]["body_path"],
        }
        source_hashes = {
            "announcement_pdf": texts[apdf]["text_sha256"],
            "announcement_xml": bodies[axml]["body_sha256"],
            "result_xml": bodies[rxml]["body_sha256"],
        }
        for key, path in source_paths.items():
            if digest(path) != source_hashes[key]:
                raise ValueError("Pilot source changed before offering application")
        text = source_paths["announcement_pdf"].read_text()
        a_raw, r_raw = (
            source_paths["announcement_xml"].read_bytes(),
            source_paths["result_xml"].read_bytes(),
        )
        try:
            check_pdf_identity(
                probe_pdf_header(text),
                expected_date=event["announcement_date"],
                expected_cusip=event["cusip"],
            )
            a_meta, r_meta = read_xml_identity(a_raw), read_xml_identity(r_raw)
            check_event_identity(a_meta, **event)
            check_event_identity(r_meta, **event)
            if a_meta["announcement"] != r_meta["announcement"]:
                raise ValueError("Pilot announcement/result identity descriptors changed")
            identities = {
                "announcement_pdf": event,
                "announcement_xml": event
                | {"announced_cusip": a_meta["announcement"].get("AnnouncedCUSIP") or None},
                "result_xml": event
                | {"announced_cusip": r_meta["announcement"].get("AnnouncedCUSIP") or None},
            }
            result = reconcile_offering(
                text, a_raw, r_raw, identities=identities, xml_unit="USD_BILLIONS"
            )
            target = private / f"{event['auction_date']}_{event['cusip']}.json"
            save(
                target,
                {
                    "event": event,
                    "source_paths": {
                        k: str(v.relative_to(root)) for k, v in source_paths.items()
                    },
                    "source_hashes": source_hashes,
                    "identities": identities,
                    "offering": result,
                },
            )
            row.update(
                status=result["status"],
                private_offering_path=str(target.relative_to(root)),
                private_offering_sha256=digest(target),
            )
        except ValueError as exc:
            row.update(status="OFFERING_RECONCILIATION_REQUIRES_REVIEW", error=str(exc))
        rows.append(row)
    save(
        report,
        {
            **started,
            "completed_utc": datetime.now(UTC).isoformat(),
            "status": "COMPLETED_FIXED_PILOT_OFFERING_RECONCILIATION",
            "events": rows,
            "counts": dict(Counter(r["status"] for r in rows)),
            "source_history_admitted": False,
            "new_predictive_comparisons": 0,
            "registered_comparisons": 144,
        },
    )
    print(json.dumps(dict(Counter(r["status"] for r in rows)), sort_keys=True))


if __name__ == "__main__":
    main()
