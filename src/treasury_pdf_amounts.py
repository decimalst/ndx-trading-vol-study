"""Independent, narrow competitive-results PDF text accounting parser.

Input is already extracted text. This module opens no files, follows no references,
and does not reuse the XML financial parser. Unsupported layouts fail explicitly.
"""

import re
from decimal import Decimal

from src.treasury_pdf_header_layout import check_pdf_identity, probe_pdf_header


def _identity(text, auction_date, cusip):
    if (
        type(auction_date) is not str
        or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", auction_date) is None
        or type(cusip) is not str
        or re.fullmatch(r"[A-Z0-9]{9}", cusip) is None
    ):
        raise ValueError("Explicit exact auction date and CUSIP required")
    check_pdf_identity(
        probe_pdf_header(text), expected_date=auction_date, expected_cusip=cusip
    )
    lines = [line.strip() for line in text.splitlines()]
    if lines.count("TREASURY AUCTION RESULTS") != 1:
        raise ValueError("Unique exact competitive-results title required")
    for line in lines:
        if re.search(r"\b(?:TENTATIVE|NONCOMPETITIVE|NCR)\b", line, re.I) and (
            re.search(r"\b(?:AUCTION|RESULTS)\b", line, re.I) or line.upper() == "NCR"
        ):
            raise ValueError("Tentative or noncompetitive results are not admitted")


def _table(segment, labels):
    patterns = []
    for number, label in enumerate(labels):
        literal = r"\s+".join(re.escape(part) for part in label.split())
        patterns.append(
            literal
            + rf"\s+(?:[0-9]{{1,2}}\s+)?(?P<t{number}>\$\S+)"
            + rf"\s+(?P<a{number}>\$\S+)(?:\s+[0-9]{{1,2}})?\s*"
        )
    match = re.fullmatch(r"\s*" + "".join(patterns), segment)
    if match is None:
        raise ValueError("Expected unique ordered table rows and two dollar columns")
    return {
        label: (match[f"t{number}"], match[f"a{number}"])
        for number, label in enumerate(labels)
    }


def _dollars(token):
    if re.fullmatch(r"\$(?:0|[1-9][0-9]{0,2}(?:,[0-9]{3})*)", token) is None:
        raise ValueError("Dollar values require unsigned integers and exact comma grouping")
    return int(token[1:].replace(",", ""))


def _single_value(text, label, *, percent=False):
    matches = list(re.finditer(label, text))
    if len(matches) != 1:
        raise ValueError("Unique labeled yield or bid-to-cover footnote required")
    tail = text[matches[0].end() :]
    if percent:
        value = re.match(r"\s+(?:[0-9]{1,2}\s+)?(\S+)", tail)
        if value is None or re.fullmatch(r"[0-9]+(?:\.[0-9]+)?%", value[1]) is None:
            raise ValueError("High Yield must be a nonnegative plain percentage")
        return value[1][:-1]
    value = re.match(r"\s*(\$\S+)\s*/\s*(\$\S+)\s*=\s*(\S+)", tail)
    if value is None or re.fullmatch(r"[0-9]+\.[0-9]{2}", value[3]) is None:
        raise ValueError("Bid-to-cover footnote requires two dollars and a two-decimal ratio")
    return value[1], value[2], value[3]


def parse_pdf_amounts(text, *, auction_date, cusip) -> dict:
    """Return exactly sixteen Decimal source fields after identity and accounting checks."""
    _identity(text, auction_date, cusip)
    headers = list(re.finditer(r"\bTendered\s+Accepted\b", text))
    if len(headers) != 2 or re.search(r"\bAccepted\s+Tendered\b", text):
        raise ValueError("Exactly two Tendered Accepted column headers required")
    public_labels = (
        "Competitive",
        "Noncompetitive",
        "FIMA (Noncompetitive)",
        "Subtotal",
        "SOMA",
        "Total",
    )
    bidder_labels = ("Primary Dealer", "Direct Bidder", "Indirect Bidder", "Total Competitive")
    public_tokens = _table(text[headers[0].end() : headers[1].start()], public_labels)
    bidder_tokens = _table(text[headers[1].end() :], bidder_labels)
    yield_token = _single_value(text, r"\bHigh\s+Yield\b", percent=True)
    foot_t, foot_a, ratio_token = _single_value(text, r"(?<![A-Za-z])Bid-to-Cover\s+Ratio:")

    public = {
        label: tuple(_dollars(token) for token in pair)
        for label, pair in public_tokens.items()
    }
    bidders = {
        label: tuple(_dollars(token) for token in pair)
        for label, pair in bidder_tokens.items()
    }
    numerator, denominator = _dollars(foot_t), _dollars(foot_a)
    ratio_hundredths = int(ratio_token.replace(".", ""))

    for tendered, accepted in (*public.values(), *bidders.values()):
        if accepted > tendered:
            raise ValueError("Accepted dollars cannot exceed tendered dollars")
    comp_t, comp_a = public["Competitive"]
    if comp_a <= 0 or bidders["Total Competitive"] != (comp_t, comp_a):
        raise ValueError("Positive matching competitive totals required")
    for column in (0, 1):
        if (
            sum(bidders[label][column] for label in bidder_labels[:3])
            != public["Competitive"][column]
        ):
            raise ValueError("Bidder categories do not sum to competitive totals")
    for label in ("Noncompetitive", "FIMA (Noncompetitive)"):
        if public[label][0] != public[label][1]:
            raise ValueError("Noncompetitive and FIMA tenders require full awards")
    for column in (0, 1):
        subtotal = sum(public[label][column] for label in public_labels[:3])
        if public["Subtotal"][column] != subtotal:
            raise ValueError(
                "Public subtotal differs from competitive plus noncompetitive awards"
            )
        if public["Total"][column] != subtotal + public["SOMA"][column]:
            raise ValueError("Total differs from public subtotal plus SOMA")
    public_t = public["Total"][0] - public["SOMA"][0]
    public_a = public["Total"][1] - public["SOMA"][1]
    if public_a <= 0 or (numerator, denominator) != (public_t, public_a):
        raise ValueError("Bid-to-cover dollars must match positive public subtotal")
    if 2 * abs(100 * public_t - ratio_hundredths * public_a) > public_a:
        raise ValueError("Bid-to-cover ratio differs by more than half a cent")

    values = {
        "PrimaryDealerAccepted": bidders["Primary Dealer"][1],
        "DirectBidderAccepted": bidders["Direct Bidder"][1],
        "IndirectBidderAccepted": bidders["Indirect Bidder"][1],
        "CompetitiveAccepted": comp_a,
        "CompetitiveTendered": comp_t,
        "PrimaryDealerTendered": bidders["Primary Dealer"][0],
        "DirectBidderTendered": bidders["Direct Bidder"][0],
        "IndirectBidderTendered": bidders["Indirect Bidder"][0],
        "NonCompetitiveAccepted": public["Noncompetitive"][1],
        "FIMAAccepted": public["FIMA (Noncompetitive)"][1],
        "SOMAAccepted": public["SOMA"][1],
        "SOMATendered": public["SOMA"][0],
        "TotalAccepted": public["Total"][1],
        "TotalTendered": public["Total"][0],
    }
    return {key: Decimal(value) for key, value in values.items()} | {
        "BidToCoverRatio": Decimal(ratio_token),
        "HighYield": Decimal(yield_token),
    }
