"""Extract Treasury XML identity metadata without interpreting financial fields."""

import re
import xml.etree.ElementTree as ET
from datetime import date

ROOT = "{http://www.treasurydirect.gov/}AuctionData"
MAX_BYTES = 2 * 1024 * 1024
START = date(2010, 1, 1)
CEILING = date(2025, 10, 20)
ANNOUNCEMENT_TAGS = frozenset(
    {
        "CUSIP",
        "AnnouncedCUSIP",
        "AnnouncementDate",
        "AuctionDate",
        "IssueDate",
        "MaturityDate",
        "OriginalIssueDate",
        "OriginalDatedDate",
        "SecurityType",
        "SecurityTermDayMonth",
        "SecurityTermWeekYear",
        "ReOpeningIndicator",
        "InflationIndexSecurity",
        "FloatingRate",
        "TypeOfAuction",
        "AnnouncementPDFName",
    }
)
RESULT_TAGS = frozenset({"ReleaseTime", "ResultsPDFName", "OriginalCUSIP"})
CONTAINERS = {"AuctionAnnouncement": ANNOUNCEMENT_TAGS, "AuctionResults": RESULT_TAGS}
DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
CUSIP = re.compile(r"[A-Z0-9]{9}")


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def read_xml_identity(raw: bytes) -> dict:
    """Return only present, direct whitelisted leaves, preserving their text."""
    if type(raw) is not bytes or len(raw) > MAX_BYTES:
        raise ValueError("XML must be bytes of at most 2 MiB")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("XML must use UTF-8") from error
    if "\x00" in text:
        raise ValueError("XML must use UTF-8 without null characters")
    if re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", text, re.IGNORECASE):
        raise ValueError("DTD and entity declarations are prohibited")
    declaration = re.match(r"\ufeff?<\?xml\s+([^?]*)\?>", text)
    if declaration:
        encoding = re.search(r"\bencoding\s*=\s*(['\"])(.*?)\1", declaration[1])
        if encoding and encoding[2].lower() != "utf-8":
            raise ValueError("XML declaration must specify UTF-8")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as error:
        raise ValueError("Malformed XML") from error
    if root.tag != ROOT:
        raise ValueError("Unexpected XML root")

    parents = {child: node for node in root.iter() for child in node}
    found = {name: [] for name in CONTAINERS}
    all_fields = ANNOUNCEMENT_TAGS | RESULT_TAGS
    for node in root.iter():
        local = _local(node.tag)
        parent = parents.get(node)
        if local in CONTAINERS:
            if node.tag != local or parent is not root:
                raise ValueError("Containers must be direct and non-namespaced")
            found[local].append(node)
        elif local in all_fields:
            expected = (
                "AuctionAnnouncement" if local in ANNOUNCEMENT_TAGS else "AuctionResults"
            )
            if node.tag != local or parent is None or parent.tag != expected:
                raise ValueError("Metadata must be an exact direct container leaf")
            if len(node):
                raise ValueError("Metadata leaves cannot contain nested elements")
    if len(found["AuctionAnnouncement"]) != 1 or len(found["AuctionResults"]) > 1:
        raise ValueError("Expected one announcement and at most one results container")

    answer = {"announcement": {}, "results": {}}
    for container, output in (
        ("AuctionAnnouncement", "announcement"),
        ("AuctionResults", "results"),
    ):
        if not found[container]:
            continue
        fields = answer[output]
        for node in found[container][0]:
            if node.tag in CONTAINERS[container]:
                if node.tag in fields:
                    raise ValueError("Duplicate metadata leaf")
                fields[node.tag] = node.text if node.text is not None else ""
    return answer


def _date(value: str, *, source: bool = False) -> date:
    if type(value) is not str:
        raise ValueError("Identity dates must be actual strings")
    plain = value[:-9] if source and value.endswith("T00:00:00") else value
    if DATE.fullmatch(plain) is None:
        raise ValueError("Identity dates must use exact ISO date or source midnight ISO")
    try:
        return date.fromisoformat(plain)
    except ValueError as error:
        raise ValueError("Invalid identity calendar date") from error


def _bounded_auction(value: date) -> None:
    if value > CEILING:
        raise ValueError("Future auction exceeds source ceiling")
    if value < START:
        raise ValueError("Auction date is below the source bound")


def check_event_identity(
    metadata: dict, *, auction_date: str, announcement_date: str, cusip: str
) -> None:
    """Match selected event identity; issue/maturity and release time stay descriptive."""
    selected_auction = _date(auction_date)
    _bounded_auction(selected_auction)
    if type(metadata) is not dict or type(metadata.get("announcement")) is not dict:
        raise ValueError("Expected extracted announcement metadata")
    announcement = metadata["announcement"]
    source_auction = _date(announcement.get("AuctionDate"), source=True)
    _bounded_auction(source_auction)

    if set(metadata) != {"announcement", "results"}:
        raise ValueError("Unexpected metadata containers")
    for name, whitelist in (
        ("announcement", ANNOUNCEMENT_TAGS),
        ("results", RESULT_TAGS),
    ):
        fields = metadata[name]
        if (
            type(fields) is not dict
            or not set(fields) <= whitelist
            or any(type(value) is not str for value in fields.values())
        ):
            raise ValueError("Expected whitelisted string metadata")
    selected_announcement = _date(announcement_date)
    source_announcement = _date(announcement.get("AnnouncementDate"), source=True)
    if source_auction != selected_auction or source_announcement != selected_announcement:
        raise ValueError("Selected event dates disagree with XML identity")
    if selected_announcement > selected_auction:
        raise ValueError("Announcement cannot follow auction")
    if type(cusip) is not str or CUSIP.fullmatch(cusip) is None:
        raise ValueError("Selected CUSIP must be nine uppercase letters or digits")
    identities = [
        announcement[key] for key in ("CUSIP", "AnnouncedCUSIP") if key in announcement
    ]
    if not any(identities) or any(value != cusip for value in identities if value != ""):
        raise ValueError("Missing or conflicting source CUSIP identity")
