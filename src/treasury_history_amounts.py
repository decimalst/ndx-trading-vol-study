"""Validate explicit announced/final identities and frozen result accounting.

The caller authenticates the identity relationship. Original XML is never
rewritten, and successful accounting does not admit a source or its vintage.
"""

import xml.etree.ElementTree as ET
from decimal import Decimal

from src.treasury_auction_amounts import FIELDS, INTEGRAL, PAR_FIELDS, PLAIN, RATIO
from src.treasury_xml_identity import CUSIP, _bounded_auction, _date, read_xml_identity


def _history_identity(metadata, *, auction_date, announcement_date, cusip, announced_cusip):
    for value in (cusip, announced_cusip):
        if type(value) is not str or CUSIP.fullmatch(value) is None:
            raise ValueError(
                "Explicit final and announced CUSIPs must be nine uppercase characters"
            )
    selected_auction = _date(auction_date)
    selected_announcement = _date(announcement_date)
    _bounded_auction(selected_auction)
    announcement = metadata["announcement"]
    source_auction = _date(announcement.get("AuctionDate"), source=True)
    source_announcement = _date(announcement.get("AnnouncementDate"), source=True)
    _bounded_auction(source_auction)
    if source_auction != selected_auction or source_announcement != selected_announcement:
        raise ValueError("Selected event dates disagree with XML identity")
    if selected_announcement > selected_auction:
        raise ValueError("Announcement cannot follow auction")
    if announcement.get("CUSIP") != cusip:
        raise ValueError("Source CUSIP differs from explicitly reconciled final identity")
    original_announced = announcement.get("AnnouncedCUSIP", "")
    if original_announced and original_announced != announced_cusip:
        raise ValueError(
            "Source AnnouncedCUSIP differs from explicitly reconciled announcement"
        )
    if cusip != announced_cusip and original_announced != announced_cusip:
        raise ValueError(
            "A substituted final identity requires explicit matching AnnouncedCUSIP"
        )
    original = metadata["results"].get("OriginalCUSIP", "")
    if original and original not in {cusip, announced_cusip}:
        raise ValueError("Unknown OriginalCUSIP cannot be dropped or aliased")


def parse_history_result_amounts(
    raw: bytes, *, auction_date: str, announcement_date: str, cusip: str, announced_cusip: str
) -> dict[str, Decimal]:
    """Return the same 16 Decimal fields after explicit document identity checks."""
    metadata = read_xml_identity(raw)
    _history_identity(
        metadata,
        auction_date=auction_date,
        announcement_date=announcement_date,
        cusip=cusip,
        announced_cusip=announced_cusip,
    )
    announcement = metadata["announcement"]
    if announcement.get("SecurityType") not in {"NOTE", "BOND"}:
        raise ValueError("Amounts require a nominal note or bond")
    if announcement.get("InflationIndexSecurity") != "N":
        raise ValueError("Amounts require explicitly non-inflation-indexed security")
    floating = announcement.get("FloatingRate")
    if floating != "N" and not (floating is None and auction_date < "2014-01-29"):
        raise ValueError("Floating-rate exclusion is not authenticated")

    # Carry the frozen result-leaf and accounting contract over the same bytes.
    # Numeric leaf attributes are additionally rejected; no content is rewritten.
    root = ET.fromstring(raw.decode("utf-8"))
    containers = [node for node in root if node.tag == "AuctionResults"]
    if len(containers) != 1:
        raise ValueError("Expected exactly one results container")
    results = containers[0]
    parents = {child: parent for parent in root.iter() for child in parent}
    tokens = {}
    for node in root.iter():
        local = node.tag.rsplit("}", 1)[-1]
        if local not in FIELDS:
            continue
        if node.tag != local or parents.get(node) is not results or len(node) or node.attrib:
            raise ValueError(
                "Amounts must be exact direct unaliased unattributed results leaves"
            )
        if local in tokens:
            raise ValueError(f"Duplicate required result field: {local}")
        tokens[local] = node.text if node.text is not None else ""
    if set(tokens) != set(FIELDS):
        raise ValueError("Missing required result fields")

    for key, token in tokens.items():
        pattern = (
            INTEGRAL if key in PAR_FIELDS else RATIO if key == "BidToCoverRatio" else PLAIN
        )
        if pattern.fullmatch(token) is None:
            raise ValueError(f"Invalid unsigned source amount or precision: {key}")
    answer = {key: Decimal(tokens[key]) for key in FIELDS}

    # Integer arithmetic is exact regardless of the caller's Decimal context.
    par = {key: int(answer[key]) for key in PAR_FIELDS}
    groups = ("PrimaryDealer", "DirectBidder", "IndirectBidder")
    for suffix in ("Accepted", "Tendered"):
        if sum(par[f"{group}{suffix}"] for group in groups) != par[f"Competitive{suffix}"]:
            raise ValueError(f"Competitive {suffix.lower()} category sum does not reconcile")
    for group in (*groups, "SOMA"):
        if par[f"{group}Accepted"] > par[f"{group}Tendered"]:
            raise ValueError(f"Accepted amount exceeds tendered amount: {group}")

    noncompetitive = par["NonCompetitiveAccepted"] + par["FIMAAccepted"]
    for suffix in ("Accepted", "Tendered"):
        expected = par[f"Competitive{suffix}"] + noncompetitive + par[f"SOMA{suffix}"]
        if par[f"Total{suffix}"] != expected:
            raise ValueError(f"Total {suffix.lower()} does not reconcile")
    public_accepted = par["TotalAccepted"] - par["SOMAAccepted"]
    public_tendered = par["TotalTendered"] - par["SOMATendered"]
    if par["CompetitiveAccepted"] <= 0 or public_accepted <= 0:
        raise ValueError("Competitive and public acceptance must be positive")

    # |quoted_ratio - tendered/accepted| <= 0.005, using exact cross products.
    cents = int(tokens["BidToCoverRatio"].replace(".", ""))
    if 2 * abs(cents * public_accepted - 100 * public_tendered) > public_accepted:
        raise ValueError("Bid-to-cover ratio does not match public totals within half a cent")
    return answer
