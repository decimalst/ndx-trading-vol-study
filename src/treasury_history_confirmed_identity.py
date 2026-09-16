"""Reconcile saved history with explicit reviewed format and calendar conventions.

Derived from the preserved first identity contract; original source bytes,
initial failures and reviewed evidence remain independently bound by the runner.
"""

import copy
import hashlib
import re
import xml.etree.ElementTree as ET
from datetime import date
from urllib.parse import urlsplit

from src.treasury_history_lineage import validate_lineage
from src.treasury_history_pdf_layout import inspect_pdf
from src.treasury_history_terms import parse_offered_term
from src.treasury_pdf_identity import MONTHS
from src.treasury_xml_identity import read_xml_identity

FLOOR = date(2010, 1, 1)
CEILING = date(2025, 10, 20)
TENORS = {2, 3, 5, 7, 10, 30}
CONFIRMATION_FIELDS = {
    "kind",
    "url",
    "body_sha256",
    "text_sha256",
    "release_date",
    "auction_date",
    "announcement_date",
    "offered_term",
    "actual_cusip",
    "original_term_years",
    "original_issue_date",
    "series",
}


class UnsupportedIdentity(ValueError):
    """A known unsupported source case with preserved, nonfinancial evidence."""

    def __init__(self, status, **evidence):
        self.audit = {"status": status, **copy.deepcopy(evidence)}
        super().__init__(status)


def _day(value, *, source=False):
    if type(value) is not str:
        raise ValueError("Identity date must be an actual string")
    plain = value.removesuffix("T00:00:00") if source else value
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", plain) is None:
        raise ValueError("Exact ISO identity date required")
    return date.fromisoformat(plain)


def _bounded(day):
    if day > CEILING:
        raise ValueError("Future date exceeds source ceiling")
    if day < FLOOR:
        raise ValueError("Identity event precedes source floor")


def _cusip(value):
    if type(value) is not str or re.fullmatch(r"[A-Z0-9]{9}", value) is None:
        raise ValueError("Exact uppercase nine-character CUSIP required")
    return value


def _field(fields, label, *, optional=False):
    value = fields.get(label)
    if value is None and optional:
        return None
    if type(value) is not str or not value.strip():
        raise ValueError(f"Missing or empty PDF field: {label}")
    return value.strip()


def _pdf_day(text, label, *, optional=False):
    value = _field(text, label, optional=optional)
    if value is None:
        return None
    match = re.fullmatch(r"(" + "|".join(MONTHS) + r") ([0-9]{1,2}), ([0-9]{4})", value)
    if match is None:
        raise ValueError(f"Invalid PDF calendar date: {label}")
    month, day, year = match.groups()
    return date(int(year), MONTHS[month], int(day))


def _pair(raw, text, *, result, auction, allow_distinct_original_dates=False):
    metadata = read_xml_identity(raw)
    a = metadata["announcement"]
    source_auction = _day(a.get("AuctionDate"), source=True)
    _bounded(source_auction)
    if source_auction != auction:
        raise ValueError("Source auction date differs from selected event")
    identity = _cusip(a.get("CUSIP"))
    inspected = inspect_pdf(text, result=result, expected_cusip=identity)
    header = inspected["header"]
    fields = inspected["fields"]
    release = _day(inspected["release_date"])
    _bounded(release)
    if (not result or "Auction Date" in fields) and _pdf_day(
        fields, "Auction Date"
    ) != auction:
        raise ValueError("PDF auction date differs from selected/XML event")
    if a.get("SecurityType") not in {"NOTE", "BOND"} or a.get("InflationIndexSecurity") != "N":
        raise ValueError("Nominal note or bond required")
    if a.get("FloatingRate") != "N" and not (
        "FloatingRate" not in a and auction < date(2014, 1, 29)
    ):
        raise ValueError("Fixed-rate security identity required")
    if a.get("ReOpeningIndicator") not in {"Y", "N"}:
        raise ValueError("Explicit reopening indicator required")
    if type(a.get("TypeOfAuction")) is not str or not a["TypeOfAuction"].strip():
        raise ValueError("Explicit auction type required")

    # The frozen extractor first enforces XML safety and all existing metadata
    # rules. DatedDate is a separate, exact metadata leaf, never an alias.
    root = ET.fromstring(raw.decode("utf-8"))
    containers = [node for node in root if node.tag == "AuctionResults"]
    if len(containers) != int(result):
        raise ValueError("XML announcement/result role disagreement")
    parents = {child: node for node in root.iter() for child in node}
    dated_fields = []
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] == "DatedDate":
            parent = parents.get(node)
            if (
                node.tag != "DatedDate"
                or parent is None
                or parent.tag != "AuctionAnnouncement"
                or len(node)
                or node.attrib
            ):
                raise ValueError("DatedDate must be an exact direct announcement leaf")
            dated_fields.append(node.text if node.text is not None else "")
    if len(dated_fields) != 1:
        raise ValueError("One explicit DatedDate required")
    dated = _day(dated_fields[0], source=True)
    issue = _day(a.get("IssueDate"), source=True)
    maturity = _day(a.get("MaturityDate"), source=True)
    if not dated <= issue < maturity or issue < auction:
        raise ValueError("Inconsistent auction, dated, issue or maturity chronology")
    for label, expected in (
        ("Dated Date", dated),
        ("Issue Date", issue),
        ("Maturity Date", maturity),
    ):
        if _pdf_day(fields, label) != expected:
            raise ValueError(f"PDF/XML date disagreement: {label}")
    original_dated = (
        _day(a["OriginalDatedDate"], source=True) if a.get("OriginalDatedDate") else None
    )
    original_issue = (
        _day(a["OriginalIssueDate"], source=True) if a.get("OriginalIssueDate") else None
    )
    pdf_original = _pdf_day(
        fields, "Original Issue Date", optional=a["ReOpeningIndicator"] == "N"
    )
    lineage = validate_lineage(
        dated=dated,
        issue=issue,
        maturity=maturity,
        original_dated=original_dated,
        original_issue=original_issue,
        pdf_original_issue=pdf_original,
        reopening=a["ReOpeningIndicator"] == "Y",
        allow_distinct_original_dates=allow_distinct_original_dates,
    )
    term = _field(fields, "Term and Type of Security")
    descriptor = parse_offered_term(term, announcement=not result)
    years, months, kind = (descriptor[key] for key in ("years", "months", "kind"))
    if (
        a.get("SecurityTermWeekYear") != f"{years}-YEAR"
        or a.get("SecurityTermDayMonth") != f"{months}-MONTH"
        or a["SecurityType"] != kind
    ):
        raise ValueError("PDF/XML offered term disagreement")
    if a["ReOpeningIndicator"] == "N" and (
        years != lineage["original_tenor_years"] or months != 0
    ):
        raise ValueError("New-issue term differs from original tenor")
    return {
        "metadata": metadata,
        "dated_date_lexeme": dated_fields[0],
        "header": header,
        "release": release,
        "cusip": identity,
        "term": descriptor["canonical_term"],
        "term_lexeme": term,
        "term_descriptor": (years, months, kind),
        "amended": inspected["amended"],
        "series": _field(fields, "Series", optional=True),
        "lineage": lineage,
    }


def _confirmation(record, *, auction, announcement, final):
    if type(record) is not dict or set(record) != CONFIRMATION_FIELDS:
        raise ValueError("Canonical realized-confirmation metadata required")
    if record["kind"] != "REALIZED_REOPENING":
        raise ValueError("A conditional notice is not realized confirmation")
    for field in ("body_sha256", "text_sha256"):
        if (
            type(record[field]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", record[field]) is None
        ):
            raise ValueError("Exact confirmation evidence hashes required")
    release = _day(record["release_date"])
    _bounded(release)
    if (
        release < auction
        or _day(record["auction_date"]) != auction
        or _day(record["announcement_date"]) != announcement
    ):
        raise ValueError("Confirmation date context disagreement")
    if type(record["url"]) is not str:
        raise ValueError("Official confirmation URL required")
    url = urlsplit(record["url"])
    prefix = f"/instit/annceresult/press/preanre/{release.year}/"
    if (
        url.scheme != "https"
        or url.netloc != "www.treasurydirect.gov"
        or url.query
        or url.fragment
        or not url.path.startswith(prefix)
    ):
        raise ValueError("Unsafe or inconsistent confirmation URL")
    filename = url.path[len(prefix) :]
    if re.fullmatch(r"[A-Za-z0-9_-]+\.[Pp][Dd][Ff]", filename, re.A) is None:
        raise ValueError("Invalid confirmation filename")
    for token in re.findall(r"(?<![0-9])([0-9]{8})(?![0-9])", filename):
        _bounded(date(int(token[:4]), int(token[4:6]), int(token[6:8])))
    lineage = final["lineage"]
    if (
        _cusip(record["actual_cusip"]) != final["cusip"]
        or record["offered_term"] != final["term"]
    ):
        raise ValueError("Confirmation actual CUSIP or offered term disagreement")
    if (
        type(record["original_term_years"]) is not int
        or record["original_term_years"] != lineage["original_tenor_years"]
    ):
        raise ValueError("Confirmation original tenor disagreement")
    if _day(record["original_issue_date"]).isoformat() != lineage["original_issue_date"]:
        raise ValueError("Confirmation original issue disagreement")
    if (
        type(record["series"]) is not str
        or not record["series"].strip()
        or record["series"] != final["series"]
    ):
        raise ValueError("Confirmation original series disagreement")
    return release


AMENDMENT_FIELDS = {
    "auction_date",
    "cusip",
    "archive_announcement_date",
    "release_date",
    "announcement_pdf_text_sha256",
    "announcement_pdf_sha256",
    "review_sha256",
    "notice_url",
    "notice_body_sha256",
    "notice_text_sha256",
    "notice_release_date",
}


def _reviewed_amendment(record, *, announced, auction, archive_date, text):
    release = announced["release"]
    if not announced["amended"]:
        if record is not None or release != archive_date:
            raise ValueError(
                "Unamended announcement has inconsistent release or amendment evidence"
            )
        return []
    if type(record) is not dict or set(record) != AMENDMENT_FIELDS:
        raise ValueError("Amended announcement requires exact reviewed evidence")
    for field in (
        "announcement_pdf_text_sha256",
        "announcement_pdf_sha256",
        "review_sha256",
        "notice_body_sha256",
        "notice_text_sha256",
    ):
        if (
            type(record[field]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", record[field]) is None
        ):
            raise ValueError("Exact amendment evidence hashes required")
    if (
        _day(record["auction_date"]) != auction
        or _cusip(record["cusip"]) != announced["cusip"]
        or _day(record["archive_announcement_date"]) != archive_date
        or _day(record["release_date"]) != release
        or record["announcement_pdf_text_sha256"]
        != hashlib.sha256(text.encode("utf-8")).hexdigest()
        or not archive_date <= release <= auction
    ):
        raise ValueError("Amendment event, date or text context disagreement")
    notice_date = _day(record["notice_release_date"])
    _bounded(notice_date)
    if not archive_date <= notice_date <= auction:
        raise ValueError("Amendment notice date outside supported auction context")
    if type(record["notice_url"]) is not str:
        raise ValueError("Official amendment notice URL required")
    url = urlsplit(record["notice_url"])
    prefix = f"/instit/annceresult/press/preanre/{notice_date.year}/"
    if (
        url.scheme != "https"
        or url.netloc != "www.treasurydirect.gov"
        or url.query
        or url.fragment
        or not url.path.startswith(prefix)
        or re.fullmatch(r"[A-Za-z0-9_-]+\.[Pp][Dd][Ff]", url.path[len(prefix) :]) is None
    ):
        raise ValueError("Official dated amendment notice URL required")
    for token in re.findall(r"(?<![0-9])([0-9]{8})(?![0-9])", url.path[len(prefix) :]):
        _bounded(date(int(token[:4]), int(token[4:6]), int(token[6:8])))
    return [notice_date]


def reconcile_confirmed_history_identity(
    announcement_xml,
    announcement_pdf_text,
    result_xml,
    result_pdf_text,
    *,
    auction_date,
    announcement_date,
    selected_final_cusip,
    confirmation=None,
    conditional_alternatives=(),
    amendment=None,
):
    """Return a metadata reconciliation; the caller binds reviewed evidence pins.

    Reviewed amendments retain initial and revised dates. Uninterpreted
    OriginalCUSIP names the announced security only for confirmed substitutions.
    No financial values or conditional
    yield thresholds are interpreted.
    """
    auction = _day(auction_date)
    _bounded(auction)
    announced_date = _day(announcement_date)
    if announced_date > auction:
        raise ValueError("Announcement follows auction")
    actual = _cusip(selected_final_cusip)
    if type(conditional_alternatives) is not tuple or len(
        set(conditional_alternatives)
    ) != len(conditional_alternatives):
        raise ValueError("Unique tuple of conditional alternative CUSIPs required")
    for alternative in conditional_alternatives:
        _cusip(alternative)
    announced = _pair(announcement_xml, announcement_pdf_text, result=False, auction=auction)
    final = _pair(
        result_xml,
        result_pdf_text,
        result=True,
        auction=auction,
        allow_distinct_original_dates=confirmation is not None
        and announced["cusip"] != actual,
    )
    for role, pair in (("announcement", announced), ("result", final)):
        lexeme = pair["metadata"]["announcement"].get("AnnouncementDate")
        if _day(lexeme, source=True) != announced_date:
            raise UnsupportedIdentity(
                "UNSUPPORTED_ANNOUNCEMENT_DATE_CONTEXT",
                source_role=role,
                archive_announcement_date=announcement_date,
                source_announcement_date=lexeme,
                source_pdf_release_date=pair["release"].isoformat(),
            )
    amendment_dates = _reviewed_amendment(
        amendment,
        announced=announced,
        auction=auction,
        archive_date=announced_date,
        text=announcement_pdf_text,
    )
    if final["release"] < auction or final["cusip"] != actual:
        raise ValueError("Final result date or selected actual CUSIP disagreement")
    a = announced["metadata"]["announcement"]
    f = final["metadata"]["announcement"]
    for source in (a, f):
        if source.get("AnnouncedCUSIP", "") not in ("", announced["cusip"]):
            raise ValueError("AnnouncedCUSIP differs from the actual announcement pair")
    original = final["metadata"]["results"].get("OriginalCUSIP", "")
    if original != "" and (announced["cusip"] == actual or original != announced["cusip"]):
        raise UnsupportedIdentity(
            "UNSUPPORTED_ORIGINAL_CUSIP",
            original_cusip_lexeme=original,
            announced_cusip=announced["cusip"],
            actual_cusip=actual,
        )
    if (
        announced["term_descriptor"] != final["term_descriptor"]
        or a["SecurityType"] != f["SecurityType"]
        or a["TypeOfAuction"] != f["TypeOfAuction"]
    ):
        raise ValueError("Announced and final offered auction context disagreement")
    for field in ("issue_date", "maturity_date"):
        if announced["lineage"][field] != final["lineage"][field]:
            raise ValueError("Announced and final settlement/maturity context disagreement")
    substitution = announced["cusip"] != actual
    required_dates = [announced["release"], final["release"], *amendment_dates]
    if substitution:
        if f.get("AnnouncedCUSIP") != announced["cusip"]:
            raise ValueError(
                "Confirmed substitution requires explicit original AnnouncedCUSIP"
            )
        if announced["lineage"]["reopening"] or not final["lineage"]["reopening"]:
            raise ValueError(
                "Substitution requires announced new issue and realized reopening"
            )
        required_dates.append(
            _confirmation(
                confirmation, auction=auction, announcement=announced_date, final=final
            )
        )
    elif announced["lineage"] != final["lineage"] or confirmation is not None:
        raise ValueError("Unchanged CUSIP has contradictory lineage or realized confirmation")
    hashes = {
        "announcement_xml_sha256": hashlib.sha256(announcement_xml).hexdigest(),
        "result_xml_sha256": hashlib.sha256(result_xml).hexdigest(),
        "announcement_pdf_text_sha256": hashlib.sha256(
            announcement_pdf_text.encode("utf-8")
        ).hexdigest(),
        "result_pdf_text_sha256": hashlib.sha256(result_pdf_text.encode("utf-8")).hexdigest(),
    }
    return {
        "status": "RECONCILED_METADATA_ONLY",
        "auction_date": auction_date,
        "archive_announcement_date": announcement_date,
        "announcement_release_date": announced["release"].isoformat(),
        "result_release_date": final["release"].isoformat(),
        "identity_available_date": max(required_dates).isoformat(),
        "announced_cusip": announced["cusip"],
        "actual_cusip": actual,
        "substitution": substitution,
        "original_cusip_lexeme": original,
        "original_cusip_interpretation": "ANNOUNCED_SECURITY_FOR_CONFIRMED_SUBSTITUTION"
        if original
        else "ABSENT_OR_EMPTY",
        "offered_term": final["term"],
        "announcement_term_lexeme": announced["term_lexeme"],
        "result_term_lexeme": final["term_lexeme"],
        "amendment": copy.deepcopy(amendment),
        "amendment_provenance_verified": False,
        "announcement_metadata": copy.deepcopy(announced["metadata"]),
        "result_metadata": copy.deepcopy(final["metadata"]),
        "announcement_dated_date_lexeme": announced["dated_date_lexeme"],
        "result_dated_date_lexeme": final["dated_date_lexeme"],
        "announcement_lineage": announced["lineage"],
        "final_lineage": final["lineage"],
        "announcement_pdf_header": announced["header"],
        "result_pdf_header": final["header"],
        "confirmation": copy.deepcopy(confirmation),
        "conditional_alternatives": list(conditional_alternatives),
        "confirmation_provenance_verified": False,
        "input_hashes": hashes,
        "source_admitted": False,
    }
