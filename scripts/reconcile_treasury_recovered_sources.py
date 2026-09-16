"""Additive reconciliation of three archived-original Treasury recoveries."""

import base64
import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pypdf
from pypdf import PdfReader

from src.treasury_auction_amounts import FIELDS
from src.treasury_history_amounts import parse_history_result_amounts
from src.treasury_history_confirmed_identity import reconcile_confirmed_history_identity
from src.treasury_history_offerings import reconcile_history_offering
from src.treasury_pdf_amounts import parse_pdf_amounts

EVENTS = (
    ("2019-12-23", "912828YZ7", "YZ7"),
    ("2019-12-24", "912828YY0", None),
    ("2020-12-09", "91282CAV3", "AV3"),
)
ARCHIVE = "data/source_discovery/treasury_auction/clock_recovery_v1/"
REPORT = "reports/treasury_dealer/"
OLD = "reports/next_signal_review/"


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    checkpoint_path = root / (REPORT + "RECOVERED_SOURCE_CHECKPOINT.json")
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint = json.loads(checkpoint_bytes)
    if checkpoint["status"] != "READY_THREE_RECOVERED_SOURCE_APPLICATIONS":
        raise ValueError("Exact prospective application checkpoint required")
    pins = dict(checkpoint["new_pins"])
    terminal_name = OLD + "TREASURY_HISTORY_RECONCILED_TERMINAL_AUDIT.json"
    terminal_bytes = (root / terminal_name).read_bytes()
    if sha(terminal_bytes) != checkpoint["old_terminal_sha256"]:
        raise ValueError("Original terminal anchor changed")
    for name, expected in json.loads(terminal_bytes)["artifact_pins"].items():
        if name in pins and pins[name] != expected:
            raise ValueError("Inconsistent old and new source pins")
        pins[name] = expected
    # Authenticate all transitive old artifacts and new inputs before any parse.
    for name, expected in pins.items():
        if sha((root / name).read_bytes()) != expected:
            raise ValueError("Frozen source drift: " + name)

    def read(name):
        payload = (root / name).read_bytes()
        if name not in pins or sha(payload) != pins[name]:
            raise ValueError("Unpinned or changed source: " + name)
        return payload

    def document(name):
        return json.loads(read(name))

    def pdf_text(payload):
        reader = PdfReader(io.BytesIO(payload))
        if len(reader.pages) != 1:
            raise ValueError("Reviewed one-page original required")
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    selection = document(OLD + "TREASURY_HISTORY_SELECTION.json")
    originals = document(OLD + "TREASURY_HISTORY_IDENTITY_V3.json")
    metadata = document(OLD + "TREASURY_HISTORY_METADATA.json")
    capture = document(OLD + "TREASURY_HISTORY_CAPTURE.json")
    notice_review = document(OLD + "TREASURY_FULL_NOTICE_REVIEW.json")
    notices = {row["url"]: row for row in notice_review["documents"]}
    docs = {row["url"]: row for row in metadata["documents"]}
    captures = {row["url"]: row for row in capture["documents"]}
    records = {
        (row["record"]["auction_date"], row["record"]["cusip"]): row["record"]
        for row in selection["records"]
    }
    old_ids = {(row["auction_date"], row["cusip"]): row for row in originals["events"]}
    if len(records) != 1138 or set(records) != set(old_ids):
        raise ValueError("Original full event membership changed")
    notice_path = ARCHIVE + "yy0_notice_v1/archived_original.pdf"
    notice = read(notice_path)
    if (
        base64.b32encode(hashlib.sha1(notice).digest()).decode()
        != "BTDU54GMBAUJVYJRDV5XKPKCGLIG2K67"
    ):
        raise ValueError("Notice bytes disagree with archived digest")
    # The complete primary notice and final identity are reviewed separately;
    # these exact phrases bind the specific conditional warning to this event.
    notice_text = pdf_text(notice)
    for phrase in (
        "December 19, 2019",
        "Tuesday, December 24, 2019",
        "9128283P3",
        "If the auction",
        "by a special announcement",
    ):
        if phrase not in notice_text:
            raise ValueError("Recovered notice differs from reviewed conditional purpose")
    read(REPORT + "CLOCK_NOTICE_RECOVERY_REVIEW.md")
    read(REPORT + "CLOCK_RESULT_ARCHIVE_RECOVERY_REVIEW.md")
    output = root / (REPORT + "RECOVERED_SOURCE_APPLICATION.json")
    private = root / (ARCHIVE + "reconciled_events_v1")
    if output.exists() or private.exists():
        raise ValueError("Preserve every prior application; outputs already exist")
    private.mkdir(mode=0o700)
    rows = []
    for auction_date, cusip, recovered in EVENTS:
        record = records[auction_date, cusip]
        links = record["documents"]
        apdf, axml, rpdf, rxml = (
            links[role][0]
            for role in (
                "announcement_pdf",
                "announcement_xml",
                "competitive_pdf",
                "competitive_xml",
            )
        )
        if any(
            len(links[role]) != 1
            for role in (
                "announcement_pdf",
                "announcement_xml",
                "competitive_pdf",
                "competitive_xml",
            )
        ):
            raise ValueError("Unique original announcement/result documents required")
        a_text = read(docs[apdf]["text_path"]).decode()
        a_raw, r_raw = read(captures[axml]["body_path"]), read(captures[rxml]["body_path"])
        archived_evidence = None
        if recovered is not None:
            stem = ARCHIVE + "result_archive_v1/" + recovered
            receipt = document(stem + ".snapshot.receipt.json")
            snapshot = receipt["cdx_snapshot"]
            transport = receipt["transport"]
            if (
                receipt["original_url"] != rpdf
                or snapshot["original"] != rpdf
                or receipt["auction_date"] != auction_date
                or receipt["cusip"] != cusip
                or receipt["status"] != "VERIFIED_TLS_HTTP200_PDF_BYTES"
                or receipt["curl_exit"] != 0
                or transport["http_code"] != 200
                or transport["ssl_verify_result"] != 0
                or transport["num_redirects"] != 0
                or transport["url_effective"] != snapshot["snapshot_url"]
                or snapshot["statuscode"] != "200"
                or snapshot["mimetype"] != "application/pdf"
            ):
                raise ValueError("Exact archived document and successful transport required")
            raw_pdf = read(stem + ".snapshot.pdf")
            if (
                base64.b32encode(hashlib.sha1(raw_pdf).digest()).decode()
                != snapshot["archive_digest"]
            ):
                raise ValueError("Recovered PDF bytes differ from archive digest")
            r_text = pdf_text(raw_pdf)
            archived_evidence = {
                **snapshot,
                "body_path": stem + ".snapshot.pdf",
                "body_sha256": sha(raw_pdf),
                "receipt_sha256": pins[stem + ".snapshot.receipt.json"],
            }
        else:
            r_text = read(docs[rpdf]["text_path"]).decode()
            if (
                links["special_pdf"].count(
                    "https://www.treasurydirect.gov/instit/annceresult/press/preanre/2019/BPD_SPL_20191219_3.pdf"
                )
                != 1
            ):
                raise ValueError("Recovered notice is not associated with this auction")
        identity = reconcile_confirmed_history_identity(
            a_raw,
            a_text,
            r_raw,
            r_text,
            auction_date=auction_date,
            announcement_date=record["announcement_date"],
            selected_final_cusip=cusip,
            confirmation=None,
            amendment=None,
        )
        for fields, name, expected in (
            (
                identity["announcement_metadata"]["announcement"],
                "AnnouncementPDFName",
                Path(apdf).name,
            ),
            (identity["result_metadata"]["results"], "ResultsPDFName", Path(rpdf).name),
        ):
            if fields.get(name) not in (None, "", expected):
                raise ValueError("Source PDF filename differs from the original role link")
        if identity["substitution"] or identity["result_release_date"] != auction_date:
            raise ValueError(
                "Recovered event differs from reviewed unmodified final identity/clock"
            )
        if recovered is None and identity != old_ids[auction_date, cusip]["identity"]:
            raise ValueError("Existing YY0 paired identity changed")
        expected_tenor, expected_reopening = {
            "912828YZ7": (2, False),
            "912828YY0": (5, False),
            "91282CAV3": (10, True),
        }[cusip]
        lineage = identity["final_lineage"]
        if (
            lineage["original_tenor_years"] != expected_tenor
            or lineage["reopening"] is not expected_reopening
        ):
            raise ValueError("Final source tenor or reopening contradicts reviewed identity")
        suffixes = {"912828YZ7": (1, 2), "912828YY0": (3, 4), "91282CAV3": ()}[cusip]
        notice_urls = [
            "https://www.treasurydirect.gov/instit/annceresult/press/preanre/2019/"
            f"BPD_SPL_20191219_{n}.pdf"
            for n in suffixes
        ]
        if sorted(links["special_pdf"]) != sorted(notice_urls):
            raise ValueError("Exact complete known-notice membership required")
        old_dispositions = old_ids[auction_date, cusip]["known_notice_dispositions"]
        if old_dispositions != [
            {"url": u, "disposition": notices[u]["disposition"]} for u in links["special_pdf"]
        ]:
            raise ValueError("Prior notice review and identity dispositions disagree")
        closures = []
        for number, url in zip(suffixes, notice_urls, strict=True):
            prior = notices[url]
            closure = {
                "url": url,
                "prior_disposition": prior["disposition"],
                "prior_review_sha256": pins[OLD + "TREASURY_FULL_NOTICE_REVIEW.json"],
                "release_date": "2019-12-19",
                "final_cusip": identity["actual_cusip"],
                "final_original_tenor": lineage["original_tenor_years"],
                "final_reopening": lineage["reopening"],
            }
            if number == 3:
                closure.update(
                    body_sha256=sha(notice),
                    review_sha256=pins[REPORT + "CLOCK_NOTICE_RECOVERY_REVIEW.md"],
                    conditional_alternative="9128283P3",
                    status="CONDITIONAL_ALTERNATIVE_NOT_REALIZED_FINAL_NEW_ISSUE_CONFIRMED",
                )
            else:
                if prior["release_date"] != "2019-12-19":
                    raise ValueError("Known notice's release clock changed")
                closure.update(
                    body_sha256=prior["body_sha256"], text_sha256=prior["text_sha256"]
                )
                if number == 4:
                    if prior["unresolved"] or prior["purposes"] != [
                        "auction_closing_schedule"
                    ]:
                        raise ValueError("Schedule-only notice has unresolved numeric effects")
                    closure["status"] = "SCHEDULE_ONLY_NO_ALLOCATION_REVISION"
                else:
                    closure.update(
                        conditional_alternative={1: "912828U81", 2: "912828G87"}[number],
                        status="CONDITIONAL_ALTERNATIVE_NOT_REALIZED_FINAL_NEW_ISSUE_CONFIRMED",
                    )
            closures.append(closure)
        if archived_evidence is not None:
            stamp = datetime.strptime(archived_evidence["timestamp"], "%Y%m%d%H%M%S")
            if not auction_date <= stamp.date().isoformat() <= "2025-10-20":
                raise ValueError(
                    "Archived source predates its auction or exceeds source ceiling"
                )
        ids = {
            "announcement_pdf": {
                "auction_date": auction_date,
                "announcement_date": identity["announcement_release_date"],
                "cusip": identity["announced_cusip"],
            }
        }
        for role, field in (
            ("announcement_xml", "announcement_metadata"),
            ("result_xml", "result_metadata"),
        ):
            own = identity[field]["announcement"]
            ids[role] = {
                "auction_date": auction_date,
                "announcement_date": record["announcement_date"],
                "cusip": own["CUSIP"],
                "announced_cusip": own.get("AnnouncedCUSIP") or None,
            }
        xml = parse_history_result_amounts(
            r_raw,
            auction_date=auction_date,
            announcement_date=record["announcement_date"],
            cusip=cusip,
            announced_cusip=identity["announced_cusip"],
        )
        pdf = parse_pdf_amounts(r_text, auction_date=auction_date, cusip=cusip)
        offering = reconcile_history_offering(
            a_text, a_raw, r_raw, identities=ids, xml_unit="USD_BILLIONS"
        )
        if set(xml) != set(FIELDS) or set(pdf) != set(FIELDS) or xml != pdf:
            raise ValueError("Recovered sixteen-field cross-source accounting disagreement")
        saved = {
            "auction_date": auction_date,
            "cusip": cusip,
            "identity": identity,
            "original_status": old_ids[auction_date, cusip]["status"],
            "archived_result": archived_evidence,
            "recovered_notice_sha256": sha(notice) if recovered is None else None,
            "known_notice_closures": closures,
            "result_pdf_text": r_text,
            "result_pdf_text_sha256": sha(r_text.encode()),
            "result_xml": {k: str(v) for k, v in xml.items()},
            "result_pdf": {k: str(v) for k, v in pdf.items()},
            "offering": offering,
            "status": "PAIRED_SOURCE_AMOUNTS_AND_OFFERING_MATCH",
        }
        destination = private / f"{auction_date}_{cusip}.json"
        with destination.open("x") as stream:
            json.dump(saved, stream, indent=2, sort_keys=True)
            stream.write("\n")
        destination.chmod(0o600)
        rows.append(
            {
                "auction_date": auction_date,
                "cusip": cusip,
                "status": saved["status"],
                "release_date": identity["result_release_date"],
                "paired_fields": len(FIELDS),
                "offering_sources": len(offering["sources"]),
                "private_output": str(destination.relative_to(root)),
                "private_output_sha256": sha(destination.read_bytes()),
            }
        )
    for name, expected in pins.items():
        if sha((root / name).read_bytes()) != expected:
            raise ValueError("Source drift during application: " + name)
    if checkpoint_path.read_bytes() != checkpoint_bytes:
        raise ValueError("Application checkpoint changed")
    result = {
        "status": "COMPLETED_THREE_RECOVERED_SOURCE_APPLICATIONS",
        "completed_utc": datetime.now(UTC).isoformat(),
        "checkpoint_sha256": sha(checkpoint_bytes),
        "events": rows,
        "pypdf_version": pypdf.__version__,
        "preserved_pins": len(pins),
        "new_predictive_comparisons": 0,
        "registered_comparisons": 144,
        "source_history_admitted": False,
        "market_arrays_read": False,
    }
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "events"}, indent=2))


if __name__ == "__main__":
    main()
