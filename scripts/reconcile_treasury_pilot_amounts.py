"""Run the pinned fixed source pilot once; no features, targets or model imports."""

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from src.treasury_auction_amounts import parse_result_amounts
from src.treasury_pdf_amounts import parse_pdf_amounts
from src.treasury_pdf_header_layout import check_pdf_identity, probe_pdf_header
from src.treasury_security_lineage import reconcile_lineage
from src.treasury_xml_identity import check_event_identity, read_xml_identity

NOTICE_KINDS = {
    "BPD_SPL_20100625_1.pdf": "SOMA_AND_OVERALL_TOTALS_CORRECTION",
    "BPD_SPL_20160121_1.pdf": "CONDITIONAL_SECURITY_SUBSTITUTION",
    "BPD_SPL_20190620_4.pdf": "RESTRICTED_CONTINGENCY_TEST",
    "SPL_20250102_1.pdf": "AUCTION_AND_BUYBACK_SCHEDULE",
    "SPL_20250102_2.pdf": "AUCTION_AND_BUYBACK_SCHEDULE",
    "SPL_20250123_1.pdf": "AUCTION_CLOSING_TIMES",
    "SPL_20250821_2.pdf": "CONDITIONAL_SECURITY_SUBSTITUTION",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    rep = root / "reports/next_signal_review"
    checkpoint_path = rep / "TREASURY_ACCOUNTING_CHECKPOINT.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    for path, expected in checkpoint["pins"].items():
        if digest(root / path) != expected:
            raise ValueError(f"Source accounting checkpoint drift: {path}")
    target = root / "data/source_discovery/treasury_auction/accounting_pilot_v1"
    public_path = rep / "TREASURY_PILOT_ACCOUNTING.json"
    if target.exists() or public_path.exists():
        raise ValueError("Preserve existing pilot outputs; no implicit replay")
    selection = json.loads((rep / "TREASURY_DOCUMENT_PILOT_SELECTION.json").read_text())
    capture = json.loads((rep / "TREASURY_DOCUMENT_CAPTURE.json").read_text())
    identity = json.loads((rep / "TREASURY_PILOT_IDENTITY.json").read_text())
    extracted = json.loads((rep / "TREASURY_PDF_METADATA.json").read_text())
    docs = {d["url"]: d for d in capture["documents"]}
    ids = {d["url"]: d for d in identity["documents"]}
    texts = {d["url"]: d for d in extracted["documents"]}
    associations = sum(
        len(urls)
        for row in selection["records"]
        for urls in row["record"]["documents"].values()
    )
    if len(selection["records"]) != 22 or len(docs) != 116 or associations != 118:
        raise ValueError("Fixed pilot membership changed")
    target.mkdir()
    started = {
        "started_utc": datetime.now(UTC).isoformat(),
        "checkpoint_sha256": digest(checkpoint_path),
        "status": "STARTED_FIXED_SOURCE_ACCOUNTING",
    }
    (target / "started.json").write_text(json.dumps(started, indent=2) + "\n")

    def body(url):
        d = docs[url]
        p = root / d["body_path"]
        if digest(p) != d["body_sha256"]:
            raise ValueError("Source body changed")
        return p.read_bytes()

    def text(url):
        d = texts[url]
        p = root / d["text_path"]
        if digest(p) != d["text_sha256"]:
            raise ValueError("Extracted source text changed")
        return p.read_text()

    events = []
    for selected in selection["records"]:
        record = selected["record"]
        event = {k: record[k] for k in ("auction_date", "announcement_date", "cusip")}
        links = record["documents"]
        row = event | {
            "source_associations": [
                {"role": role, "url": url, "metadata_status": ids[url]["status"]}
                for role, urls in links.items()
                for url in urls
            ],
            "notices": [
                {"url": url, "disposition": NOTICE_KINDS[Path(url).name]}
                for url in links["special_pdf"]
            ],
            "numerically_processed": False,
            "first_vintage_admitted": False,
            "announcement_amounts_checked": False,
        }
        if (event["auction_date"], event["cusip"]) == ("2019-06-21", "9128286T2"):
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
                if len(links[role]) != 1 or ids[links[role][0]]["status"] != "METADATA_MATCH":
                    raise ValueError(f"Unresolved document identity or membership: {role}")
            apdf, axml = links["announcement_pdf"][0], links["announcement_xml"][0]
            rpdf, rxml = links["competitive_pdf"][0], links["competitive_xml"][0]
            check_pdf_identity(
                probe_pdf_header(text(apdf)),
                expected_date=event["announcement_date"],
                expected_cusip=event["cusip"],
            )
            announcement = read_xml_identity(body(axml))
            check_event_identity(announcement, **event)
            raw, pdf = body(rxml), text(rpdf)
            result_identity = read_xml_identity(raw)
            check_event_identity(result_identity, **event)
            if announcement["announcement"] != result_identity["announcement"]:
                raise ValueError("Announcement/result XML identity descriptor disagreement")
            for key, expected in (("AnnouncementPDFName", Path(apdf).name),):
                if announcement["announcement"].get(key) not in (None, "", expected):
                    raise ValueError("Announcement file role disagreement")
            if result_identity["results"].get("ResultsPDFName") not in (
                None,
                "",
                Path(rpdf).name,
            ):
                raise ValueError("Results file role disagreement")
            row["lineage"] = reconcile_lineage(raw, pdf, **event)
            row["supported_result_release_date"] = event["auction_date"]
            row["release_basis"] = (
                "dated competitive PDF release header with selected XML auction/CUSIP match"
            )
            row["intraday_release_authenticated"] = False
            if any(
                n["disposition"] == "CONDITIONAL_SECURITY_SUBSTITUTION" for n in row["notices"]
            ):
                if result_identity["announcement"].get("ReOpeningIndicator") != "N":
                    raise ValueError(
                        "Conditional substitution requires additional realized-event review"
                    )
                row["conditional_notice_resolution"] = (
                    "retained final selected CUSIP and explicit new-issue identity"
                )
        except ValueError as exc:
            row.update(status="METADATA_OR_LINEAGE_REQUIRES_REVIEW", error=str(exc))
            events.append(row)
            continue

        values, failures = {}, {}
        for format_name, parse in (
            ("xml", partial(parse_result_amounts, raw, **event)),
            (
                "pdf",
                partial(
                    parse_pdf_amounts,
                    pdf,
                    auction_date=event["auction_date"],
                    cusip=event["cusip"],
                ),
            ),
        ):
            try:
                values[format_name] = parse()
            except ValueError as exc:
                failures[format_name] = str(exc)
        row["numerically_processed"] = True
        differences = []
        if len(values) == 2:
            if set(values["xml"]) != set(values["pdf"]) or len(values["xml"]) != 16:
                raise ValueError("Independent parser output contracts disagree")
            differences = [k for k in values["xml"] if values["xml"][k] != values["pdf"][k]]
        row.update(
            status="SOURCE_ACCOUNTING_REQUIRES_REVIEW"
            if failures or differences
            else "PAIRED_SOURCE_AMOUNTS_MATCH",
            parse_failures=failures,
            differing_fields=differences,
            parsed_formats=sorted(values),
        )
        private_path = target / f"{event['auction_date']}_{event['cusip']}.json"
        private = {
            "event": event,
            "source_pdf_sha256": docs[rpdf]["body_sha256"],
            "source_xml_sha256": docs[rxml]["body_sha256"],
            "values": {
                fmt: {k: str(v) for k, v in data.items()} for fmt, data in values.items()
            },
            "parse_failures": failures,
            "differing_fields": differences,
            "units": "14 par-dollar amounts; published bid-to-cover ratio; HighYield in percentage points",
            "first_vintage_admitted": False,
        }
        with private_path.open("x") as out:
            out.write(json.dumps(private, indent=2, sort_keys=True) + "\n")
        row["private_accounting_path"] = str(private_path.relative_to(root))
        row["private_accounting_sha256"] = digest(private_path)
        events.append(row)
    result = {
        **started,
        "status": "COMPLETED_FIXED_PILOT_SOURCE_ACCOUNTING",
        "completed_utc": datetime.now(UTC).isoformat(),
        "events": events,
        "counts": dict(Counter(row["status"] for row in events)),
        "source_associations": associations,
        "unique_documents": len(docs),
        "registered_comparisons": 144,
        "features_or_market_arrays_or_scores_created": False,
        "full_history_admitted": False,
    }
    with public_path.open("x") as out:
        out.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
