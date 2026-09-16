"""Reconcile announced and realized Treasury identities using metadata only."""

import copy
import hashlib
import re
import xml.etree.ElementTree as ET
from datetime import date
from urllib.parse import urlsplit

from src.treasury_pdf_header_layout import check_pdf_identity, probe_pdf_header
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


def _field(text, label, *, optional=False):
    values = re.findall(r"^" + re.escape(label) + r"[ \t]+([^\r\n]+)$", text, re.M)
    if optional and not values:
        return None
    if len(values) != 1:
        raise ValueError(f"Missing or ambiguous PDF field: {label}")
    return values[0].strip()


def _pdf_day(text, label, *, optional=False):
    value = _field(text, label, optional=optional)
    if value is None:
        return None
    match = re.fullmatch(r"(" + "|".join(MONTHS) + r") ([0-9]{1,2}), ([0-9]{4})", value)
    if match is None:
        raise ValueError(f"Invalid PDF calendar date: {label}")
    month, day, year = match.groups()
    return date(int(year), MONTHS[month], int(day))


def _pair(raw, text, *, result, auction):
    metadata = read_xml_identity(raw)
    a = metadata["announcement"]
    source_auction = _day(a.get("AuctionDate"), source=True)
    _bounded(source_auction)
    if source_auction != auction:
        raise ValueError("Source auction date differs from selected event")
    identity = _cusip(a.get("CUSIP"))
    header = probe_pdf_header(text)
    release = _day(check_pdf_identity(header, expected_cusip=identity))
    role = "TREASURY AUCTION RESULTS" if result else "TREASURY OFFERING ANNOUNCEMENT"
    if len(re.findall(r"^" + role + r"[ \t]*$", text, re.M)) != 1:
        raise ValueError("Wrong PDF document role")
    if not result and _pdf_day(text, "Auction Date") != auction:
        raise ValueError("Announcement PDF auction date differs from selected/XML event")
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
        if _pdf_day(text, label) != expected:
            raise ValueError(f"PDF/XML date disagreement: {label}")
    original_dated = (
        _day(a["OriginalDatedDate"], source=True) if a.get("OriginalDatedDate") else None
    )
    original_issue = (
        _day(a["OriginalIssueDate"], source=True) if a.get("OriginalIssueDate") else None
    )
    pdf_original = _pdf_day(
        text, "Original Issue Date", optional=a["ReOpeningIndicator"] == "N"
    )
    if a["ReOpeningIndicator"] == "Y":
        if (
            original_dated != dated
            or original_issue is None
            or not dated <= original_issue < issue
        ):
            raise ValueError("Inconsistent original reopening lineage")
        if pdf_original != original_issue:
            raise ValueError("PDF/XML original issue disagreement")
    elif (
        original_dated not in (None, dated)
        or original_issue not in (None, issue)
        or pdf_original not in (None, issue)
    ):
        raise ValueError("New issue conflicts with original lineage")
    tenor = maturity.year - dated.year
    if (maturity.month, maturity.day) != (dated.month, dated.day) or tenor not in TENORS:
        raise ValueError("Unsupported exact original tenor from stated dates")
    term = _field(text, "Term and Type of Security")
    match = re.fullmatch(r"([0-9]+)-Year(?: ([0-9]+)-Month)? (Note|Bond)", term)
    if match is None:
        raise ValueError("Unrecognized PDF offered term")
    years, months, kind = match.groups()
    if int(months or "0") > 11 or int(years) > 30:
        raise ValueError("Invalid remaining term descriptor")
    if (
        a.get("SecurityTermWeekYear") != f"{years}-YEAR"
        or a.get("SecurityTermDayMonth") != f"{months or '0'}-MONTH"
        or a["SecurityType"] != kind.upper()
    ):
        raise ValueError("PDF/XML offered term disagreement")
    if a["ReOpeningIndicator"] == "N" and (int(years) != tenor or int(months or "0") != 0):
        raise ValueError("New-issue term differs from exact original tenor")
    return {
        "metadata": metadata,
        "dated_date_lexeme": dated_fields[0],
        "header": header,
        "release": release,
        "cusip": identity,
        "term": term,
        "series": _field(text, "Series", optional=True),
        "lineage": {
            "original_tenor_years": tenor,
            "dated_date": dated.isoformat(),
            "issue_date": issue.isoformat(),
            "maturity_date": maturity.isoformat(),
            "original_issue_date": (original_issue or issue).isoformat(),
            "reopening": a["ReOpeningIndicator"] == "Y",
        },
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


def reconcile_realized_identity(
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
):
    """Return a metadata reconciliation; the caller binds reviewed evidence pins.

    Raw inputs and frozen helper behavior remain unchanged. Replacement-date
    context and currently uninterpreted OriginalCUSIP are explicit unsupported
    cases. No financial values or conditional yield thresholds are interpreted.
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
    final = _pair(result_xml, result_pdf_text, result=True, auction=auction)
    for role, pair in (("announcement", announced), ("result", final)):
        lexeme = pair["metadata"]["announcement"].get("AnnouncementDate")
        if _day(lexeme, source=True) != announced_date or (
            role == "announcement" and pair["release"] != announced_date
        ):
            raise UnsupportedIdentity(
                "UNSUPPORTED_ANNOUNCEMENT_DATE_CONTEXT",
                source_role=role,
                archive_announcement_date=announcement_date,
                source_announcement_date=lexeme,
                source_pdf_release_date=pair["release"].isoformat(),
            )
    if final["release"] < auction or final["cusip"] != actual:
        raise ValueError("Final result date or selected actual CUSIP disagreement")
    a = announced["metadata"]["announcement"]
    f = final["metadata"]["announcement"]
    for source in (a, f):
        if source.get("AnnouncedCUSIP", "") not in ("", announced["cusip"]):
            raise ValueError("AnnouncedCUSIP differs from the actual announcement pair")
    original = final["metadata"]["results"].get("OriginalCUSIP", "")
    if original != "":
        raise UnsupportedIdentity(
            "UNSUPPORTED_ORIGINAL_CUSIP",
            original_cusip_lexeme=original,
            announced_cusip=announced["cusip"],
            actual_cusip=actual,
        )
    if (
        announced["term"] != final["term"]
        or a["SecurityType"] != f["SecurityType"]
        or a["TypeOfAuction"] != f["TypeOfAuction"]
    ):
        raise ValueError("Announced and final offered auction context disagreement")
    for field in ("issue_date", "maturity_date"):
        if announced["lineage"][field] != final["lineage"][field]:
            raise ValueError("Announced and final settlement/maturity context disagreement")
    substitution = announced["cusip"] != actual
    required_dates = [announced["release"], final["release"]]
    if substitution:
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
        "offered_term": final["term"],
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
