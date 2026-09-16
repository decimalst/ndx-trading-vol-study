"""Authenticate nominal auction result accounting, without producing features.

Returned amounts are par US dollars; HighYield remains in percentage points.
Successful accounting does not establish regular-auction eligibility, original
tenor, first-publication timing, or agreement with the original PDF release.
"""

import re
import xml.etree.ElementTree as ET
from decimal import Decimal

from src.treasury_xml_identity import check_event_identity, read_xml_identity

PAR_FIELDS = (
    "PrimaryDealerAccepted",
    "DirectBidderAccepted",
    "IndirectBidderAccepted",
    "CompetitiveAccepted",
    "CompetitiveTendered",
    "PrimaryDealerTendered",
    "DirectBidderTendered",
    "IndirectBidderTendered",
    "NonCompetitiveAccepted",
    "FIMAAccepted",
    "SOMAAccepted",
    "SOMATendered",
    "TotalAccepted",
    "TotalTendered",
)
FIELDS = (*PAR_FIELDS, "BidToCoverRatio", "HighYield")
PLAIN = re.compile(r"[0-9]+(?:\.[0-9]+)?")
INTEGRAL = re.compile(r"[0-9]+(?:\.0+)?")
RATIO = re.compile(r"[0-9]+\.[0-9]{2}")


def parse_result_amounts(
    raw: bytes, *, auction_date: str, announcement_date: str, cusip: str
) -> dict[str, Decimal]:
    """Validate identity/class first, then exact source amounts and accounting."""
    metadata = read_xml_identity(raw)
    check_event_identity(
        metadata, auction_date=auction_date, announcement_date=announcement_date, cusip=cusip
    )
    announcement = metadata["announcement"]
    if announcement.get("SecurityType") not in {"NOTE", "BOND"}:
        raise ValueError("Amounts require a nominal note or bond")
    if announcement.get("InflationIndexSecurity") != "N":
        raise ValueError("Amounts require explicitly non-inflation-indexed security")
    floating = announcement.get("FloatingRate")
    if floating != "N" and not (floating is None and auction_date < "2014-01-29"):
        raise ValueError("Floating-rate exclusion is not authenticated")

    # The identity reader has already rejected malformed encodings, entities,
    # containers and metadata. Reparse only these same bounded immutable bytes.
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
        if node.tag != local or parents.get(node) is not results or len(node):
            raise ValueError("Amounts must be exact direct unaliased results leaves")
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
