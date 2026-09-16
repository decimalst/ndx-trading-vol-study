"""Preserve source identity evidence without reconciling or admitting events."""

from src.treasury_pdf_header_layout import check_pdf_identity, probe_pdf_header
from src.treasury_xml_identity import read_xml_identity


def inspect_history_payload(raw, format, *, pdf_text=None):
    """Inspect whitelisted XML metadata or a complete PDF's extracted header.

    Conflicting announced/final identifiers are retained for the separate
    reconciliation step. A successful header is evidence about that document,
    not agreement with an auction or proof of first-vintage availability.
    """
    if type(raw) is not bytes or format not in {"xml", "pdf"}:
        raise ValueError("Expected XML or PDF source bytes")
    if not 0 < len(raw) <= (2 if format == "xml" else 8) * 1024 * 1024:
        raise ValueError("Source bytes exceed document bounds")
    result = {"identity_reconciled": False, "source_admitted": False}
    if format == "xml":
        try:
            result["metadata"] = read_xml_identity(raw)
            result["status"] = "XML_METADATA_EXTRACTED"
        except ValueError as exc:
            result.update(status="XML_METADATA_REQUIRES_REVIEW", error=str(exc))
        return result
    if not raw.startswith(b"%PDF-") or type(pdf_text) is not str:
        raise ValueError("PDF signature and complete extracted text required")
    try:
        result["metadata"] = probe_pdf_header(pdf_text)
        result["release_date"] = check_pdf_identity(result["metadata"])
        result["status"] = "PDF_DATED_HEADER_EXTRACTED"
    except ValueError as exc:
        result.update(status="PDF_HEADER_REQUIRES_REVIEW", error=str(exc))
    return result
