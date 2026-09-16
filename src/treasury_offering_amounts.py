"""Independently reconcile stated offerings with an explicit PDF-anchored scale.

Identity relationships between different announced/final securities or schedules
are caller-authenticated inputs. The frozen common-CUSIP identity check is not
changed. Agreement here neither authenticates that relationship nor admits data.
"""

import re
import xml.etree.ElementTree as ET
from datetime import date

from src.treasury_pdf_header_layout import check_pdf_identity, probe_pdf_header
from src.treasury_pdf_identity import MONTHS
from src.treasury_xml_identity import read_xml_identity

_MAX_BYTES = 2 * 1024 * 1024
_START = date(2010, 1, 1)
_CEILING = date(2025, 10, 20)
_PDF_KEYS = {"auction_date", "announcement_date", "cusip"}
_XML_KEYS = _PDF_KEYS | {"announced_cusip"}
_ROLES = ("announcement", "result")
_XML_UNIT = "USD_BILLIONS"
_DOLLARS = re.compile(r"\$(?:0|[1-9][0-9]{0,2}(?:,[0-9]{3})*)")
_PLAIN = re.compile(r"[0-9]+(?:\.[0-9]+)?")
_WRITTEN_DATE = re.compile(r"(" + "|".join(MONTHS) + r")[ \t]+([0-9]{1,2}),[ \t]+([0-9]{4})")


def _iso(value, *, source=False):
    if type(value) is not str:
        raise ValueError("Explicit string identity date required")
    plain = value.removesuffix("T00:00:00") if source else value
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", plain) is None:
        raise ValueError("Exact ISO identity date required")
    return date.fromisoformat(plain)


def _expected_identity(identity, *, xml):
    keys = _XML_KEYS if xml else _PDF_KEYS
    if type(identity) is not dict or set(identity) != keys:
        raise ValueError("Exact document-specific identity contract required")
    auction, announcement = _iso(identity["auction_date"]), _iso(identity["announcement_date"])
    if not _START <= auction <= _CEILING or announcement > auction:
        raise ValueError("Identity dates outside source scope or wrong chronology")
    for name in ("cusip", "announced_cusip") if xml else ("cusip",):
        value = identity[name]
        if name == "announced_cusip" and value is None:
            continue
        if type(value) is not str or re.fullmatch(r"[A-Z0-9]{9}", value) is None:
            raise ValueError("Exact uppercase nine-character source identity required")


def _row(lines, label):
    pattern = re.compile(re.escape(label) + r"(?:[ \t]+(.*))?")
    matches = [match[1] for line in lines if (match := pattern.fullmatch(line))]
    if len(matches) != 1 or not matches[0]:
        raise ValueError(f"Unique exact PDF field required: {label}")
    return matches[0]


def _pdf_token(text, identity):
    _expected_identity(identity, xml=False)
    if type(text) is not str or len(text) > _MAX_BYTES:
        raise ValueError("Bounded extracted announcement PDF text required")
    try:
        if len(text.encode("utf-8")) > _MAX_BYTES or "\x00" in text:
            raise ValueError("Invalid or oversized announcement text")
    except UnicodeEncodeError as error:
        raise ValueError("Invalid Unicode announcement text") from error
    check_pdf_identity(
        probe_pdf_header(text),
        expected_date=identity["announcement_date"],
        expected_cusip=identity["cusip"],
    )
    lines = [line.strip() for line in text.splitlines()]
    if lines.count("TREASURY OFFERING ANNOUNCEMENT") != 1:
        raise ValueError("Unique offering-announcement title required")
    # Do not let a set-valued header probe conceal repeated identical identity rows.
    cusips = [line for line in lines if re.match(r"CUSIP(?:[ \t]+Number)?\b", line)]
    if len(cusips) != 1 or _row(cusips, "CUSIP Number") != identity["cusip"]:
        raise ValueError("Unique labelled announcement CUSIP required")
    auction_text = _row(lines, "Auction Date")
    matched = _WRITTEN_DATE.fullmatch(auction_text)
    if matched is None:
        raise ValueError("Exact labelled PDF auction date required")
    month, day, year = matched.groups()
    if date(int(year), MONTHS[month], int(day)) != _iso(identity["auction_date"]):
        raise ValueError("PDF auction date differs from its expected identity")
    return _row(lines, "Offering Amount")


def _xml_token(raw, role, identity, unit):
    if type(unit) is not str or unit != _XML_UNIT or role not in _ROLES:
        raise ValueError("Explicit XML role and USD_BILLIONS mapping contract required")
    _expected_identity(identity, xml=True)
    metadata = read_xml_identity(raw)
    announcement = metadata["announcement"]
    for field, expected in (
        ("AuctionDate", identity["auction_date"]),
        ("AnnouncementDate", identity["announcement_date"]),
    ):
        if _iso(announcement.get(field), source=True) != _iso(expected):
            raise ValueError("XML date differs from its own expected document identity")
    if announcement.get("CUSIP") != identity["cusip"]:
        raise ValueError("XML CUSIP differs from its own expected document identity")
    announced = announcement.get("AnnouncedCUSIP", "")
    if announced != (identity["announced_cusip"] or ""):
        raise ValueError("XML AnnouncedCUSIP differs from its separately expected identity")

    # Reparse only the same bounded bytes accepted by the frozen identity decoder.
    root = ET.fromstring(raw.decode("utf-8"))
    results = [node for node in root if node.tag == "AuctionResults"]
    if len(results) != (1 if role == "result" else 0):
        raise ValueError("XML container structure disagrees with its declared document role")
    parent = {child: node for node in root.iter() for child in node}
    tokens = []
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] != "OfferingAmount":
            continue
        container = parent.get(node)
        if (
            node.tag != "OfferingAmount"
            or container is None
            or container.tag != "AuctionAnnouncement"
            or parent.get(container) is not root
            or len(node)
            or node.attrib
        ):
            raise ValueError(
                "OfferingAmount must be an exact direct unaliased announcement leaf"
            )
        tokens.append(node.text if node.text is not None else "")
    if len(tokens) != 1:
        raise ValueError("Exactly one stated XML OfferingAmount required")
    return tokens[0]


def _usd_integer(token, unit):
    """Decode exact dollars with digit arithmetic, never floating point."""
    if type(token) is not str:
        raise ValueError("Source offering token must be text")
    if unit == "USD":
        if _DOLLARS.fullmatch(token) is None:
            raise ValueError("PDF offering requires exact grouped unsigned integer dollars")
        amount = int(token[1:].replace(",", ""))
    elif unit == _XML_UNIT:
        if _PLAIN.fullmatch(token) is None:
            raise ValueError("XML offering requires unsigned plain decimal text")
        whole, _, fraction = token.partition(".")
        if any(char != "0" for char in fraction[9:]):
            raise ValueError("Declared XML scale produces fractional USD")
        amount = int(whole + fraction[:9].ljust(9, "0"))
    else:
        raise ValueError("Unsupported declared source unit")
    if amount <= 0:
        raise ValueError("Stated offering must be strictly positive integral USD")
    return amount


def _answer(token, unit):
    return {
        "offering_amount_usd": _usd_integer(token, unit),
        "source_unit": unit,
        "source_token": token,
    }


def parse_announcement_offering(text, *, identity):
    """Parse labelled PDF dollars after validating that document's own identity."""
    return _answer(_pdf_token(text, identity), "USD")


def parse_xml_offering(raw, *, role, identity, unit):
    """Apply the explicitly declared scale to one exact announcement-section leaf."""
    return _answer(_xml_token(raw, role, identity, unit), unit)


def reconcile_offering(pdf_text, announcement_xml, result_xml, *, identities, xml_unit):
    """Compare all three stated amounts, conditional on caller-authenticated linkage."""
    if type(identities) is not dict or set(identities) != {
        "announcement_pdf",
        "announcement_xml",
        "result_xml",
    }:
        raise ValueError("Three exact document-specific identity contracts required")
    if type(xml_unit) is not str or xml_unit != _XML_UNIT:
        raise ValueError("Explicit fixed USD_BILLIONS mapping contract required")
    # Every identity/container check finishes before any financial token conversion.
    tokens = {
        "announcement_pdf": _pdf_token(pdf_text, identities["announcement_pdf"]),
        "announcement_xml": _xml_token(
            announcement_xml, "announcement", identities["announcement_xml"], xml_unit
        ),
        "result_xml": _xml_token(result_xml, "result", identities["result_xml"], xml_unit),
    }
    sources = {
        name: _answer(token, "USD" if name == "announcement_pdf" else xml_unit)
        for name, token in tokens.items()
    }
    values = {parsed["offering_amount_usd"] for parsed in sources.values()}
    if len(values) != 1:
        raise ValueError("Stated offering amounts disagree across PDF and both XML documents")
    return {
        "status": "VERIFIED_STATED_OFFERING_AGREEMENT",
        "offering_amount_usd": sources["announcement_pdf"]["offering_amount_usd"],
        "unit": "USD",
        "sources": sources,
        "xml_scale_basis": (
            "Fixed 1e9 mapping hypothesis checked against the explicitly USD-labelled PDF "
            "on this record; not a schema-authenticated universal XML unit."
        ),
        "source_admitted": False,
    }
