"""Three-source offering agreement after explicit history-layout identity checks."""

import re
from datetime import date

from src import treasury_offering_amounts as frozen
from src.treasury_history_pdf_layout import inspect_pdf


def _pdf_token(text, identity):
    frozen._expected_identity(identity, xml=False)
    if type(text) is not str or len(text) > frozen._MAX_BYTES:
        raise ValueError("Bounded extracted announcement PDF text required")
    try:
        if len(text.encode("utf-8")) > frozen._MAX_BYTES or "\x00" in text:
            raise ValueError("Invalid or oversized announcement text")
    except UnicodeEncodeError as error:
        raise ValueError("Invalid Unicode announcement text") from error
    metadata = inspect_pdf(text, result=False, expected_cusip=identity["cusip"])
    if metadata["release_date"] != identity["announcement_date"]:
        raise ValueError("PDF actual release differs from its own expected announcement date")
    match = frozen._WRITTEN_DATE.fullmatch(metadata["fields"]["Auction Date"])
    if match is None:
        raise ValueError("Exact labelled PDF auction date required")
    month, day, year = match.groups()
    if date(int(year), frozen.MONTHS[month], int(day)) != frozen._iso(
        identity["auction_date"]
    ):
        raise ValueError("PDF auction date differs from its own expected identity")

    lines = [line.strip() for line in text.splitlines()]
    offering_lines = [line for line in lines if re.match(r"Offering[ \t]+Amount\b", line)]
    if len(offering_lines) != 1:
        raise ValueError("Exactly one unaliased PDF Offering Amount field required")
    if "Term and Type of Security" in lines:
        # inspect_pdf has authenticated the exact fourteen-value/fourteen-label
        # block. Offering Amount is cell 1, never a value selected by similarity.
        first_label = lines.index("Term and Type of Security")
        if offering_lines[0] != "Offering Amount":
            raise ValueError("Column Offering Amount label cannot contain a second value")
        token = lines[first_label - 13]
    else:
        token = frozen._row(offering_lines, "Offering Amount")
    if not token:
        raise ValueError("Empty stated PDF Offering Amount")
    return token


def reconcile_history_offering(
    pdf_text, announcement_xml, result_xml, *, identities, xml_unit
):
    """Compare amounts conditional on caller-authenticated source relationships.

    The PDF's own expected announcement date is its actual release date. The
    XML contracts preserve their own archive announcement dates. An amendment or
    realized-identity relationship is authenticated by the outer identity pass.
    """
    if type(identities) is not dict or set(identities) != {
        "announcement_pdf",
        "announcement_xml",
        "result_xml",
    }:
        raise ValueError("Three exact document-specific identity contracts required")
    if type(xml_unit) is not str or xml_unit != frozen._XML_UNIT:
        raise ValueError("Explicit fixed USD_BILLIONS mapping contract required")
    # No financial conversion occurs until all three original identities and
    # containers have passed their independent, document-specific checks.
    tokens = {
        "announcement_pdf": _pdf_token(pdf_text, identities["announcement_pdf"]),
        "announcement_xml": frozen._xml_token(
            announcement_xml, "announcement", identities["announcement_xml"], xml_unit
        ),
        "result_xml": frozen._xml_token(
            result_xml, "result", identities["result_xml"], xml_unit
        ),
    }
    sources = {
        name: frozen._answer(token, "USD" if name == "announcement_pdf" else xml_unit)
        for name, token in tokens.items()
    }
    if len({parsed["offering_amount_usd"] for parsed in sources.values()}) != 1:
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
