"""Independent reconstruction of quarantined Treasury archive metadata.

No producer imports, transport, auction amount conversion, or document fetching.
"""

import datetime as dt
import json
import re


class _NumericText(str):
    """A JSON numeric token, deliberately distinct from an actual JSON string."""


_CORE = {
    "a": "cusip",
    "d": "security_term",
    "z3a": "security_type",
    "t3a": "term_bucket",
}
_DOCUMENTS = (
    ("e3", "announcement_pdf", "pdf"),
    ("f3", "competitive_pdf", "pdf"),
    ("f31", "noncompetitive_pdf", "pdf"),
    ("f32", "special_pdf", "pdf"),
    ("e4", "announcement_xml", "xml"),
    ("ia1", "competitive_xml", "xml"),
)
_RECORD_KEYS = {
    *_CORE.values(),
    "announcement_date",
    "auction_date",
    "documents",
}
_DOCUMENT_KEYS = {output for _, output, _ in _DOCUMENTS}
_PDF_BASE = "https://www.treasurydirect.gov/instit/annceresult/press/preanre/"
_XML_BASE = "https://www.treasurydirect.gov/xml/"


def _reject_constant(token):
    raise ValueError(f"Nonstandard JSON constant: {token}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _date(value, *, interval=False):
    pattern = r"[0-9]{4}-[0-9]{2}-[0-9]{2}"
    if not interval:
        pattern += r"(?:T00:00:00)?"
    if type(value) is not str or re.fullmatch(pattern, value) is None:
        raise ValueError("Date must be an actual ISO string with permitted precision")
    try:
        return dt.date.fromisoformat(value[:10])
    except ValueError as exc:
        raise ValueError("Invalid calendar date") from exc


def _rows(raw):
    if type(raw) is not bytes:
        raise ValueError("Raw archive must be UTF-8 JSON bytes")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_int=_NumericText,
            parse_float=_NumericText,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError("Invalid archive JSON") from exc
    if type(value) is dict:
        if set(value) != {"securityList"}:
            raise ValueError("Invalid archive envelope")
        value = value["securityList"]
    if type(value) is not list or not value or any(type(row) is not dict for row in value):
        raise ValueError("Archive must contain a nonempty list of record objects")
    return value


def _links(row, announced, auctioned):
    documents = {}
    for source, output, extension in _DOCUMENTS:
        value = row.get(source)
        if value is None or (type(value) is str and value == ""):
            documents[output] = []
            continue
        if type(value) is not str:
            raise ValueError("Document filename must be an actual JSON string")
        filenames = value.split(", ") if source == "f32" else [value]
        for filename in filenames:
            if (
                re.fullmatch(
                    rf"[A-Za-z0-9_\-]+\.{extension}", filename, flags=re.IGNORECASE | re.ASCII
                )
                is None
            ):
                raise ValueError("Unsafe or invalid document filename")
        if extension == "xml":
            prefix = _XML_BASE
        else:
            date = announced if source == "e3" else auctioned
            prefix = _PDF_BASE + date.isoformat()[:4] + "/"
        documents[output] = [prefix + filename for filename in filenames]
    return documents


def _check_projected_shape(projected):
    if type(projected) is not list:
        raise ValueError("Projected archive must be a list")
    for record in projected:
        if type(record) is not dict or set(record) != _RECORD_KEYS:
            raise ValueError("Projected record schema mismatch")
        if any(type(record[key]) is not str for key in _RECORD_KEYS - {"documents"}):
            raise ValueError("Projected metadata must contain actual strings")
        documents = record["documents"]
        if type(documents) is not dict or set(documents) != _DOCUMENT_KEYS:
            raise ValueError("Projected document schema mismatch")
        if any(
            type(values) is not list or any(type(url) is not str for url in values)
            for values in documents.values()
        ):
            raise ValueError("Projected documents must be lists of actual strings")


def verify_archive(raw: bytes, projected: list[dict], *, start_date: str, end_date: str):
    """Verify exact metadata projection without interpreting auction numbers."""
    start, end = (_date(value, interval=True) for value in (start_date, end_date))
    if not dt.date(2010, 1, 1) <= start <= end <= dt.date(2025, 10, 20):
        raise ValueError("Requested interval is outside archive bounds")
    rows = _rows(raw)

    # Preflight every identity/date before examining any document filename.
    checked = []
    identities = set()
    for row in rows:
        for field in (*_CORE, "h", "i"):
            value = row.get(field)
            if type(value) is not str or not value.strip():
                raise ValueError("Required metadata must be a nonempty JSON string")
        if re.fullmatch(r"[A-Z0-9]{9}", row["a"]) is None:
            raise ValueError("Invalid CUSIP identity")
        announced, auctioned = _date(row["h"]), _date(row["i"])
        if not start <= auctioned <= end or announced > auctioned:
            raise ValueError("Record date outside bounds or announcement follows auction")
        identity = (auctioned, row["a"])
        if identity in identities:
            raise ValueError("Duplicate auction identity")
        identities.add(identity)
        checked.append((row, announced, auctioned))

    expected = []
    for row, announced, auctioned in checked:
        record = {output: row[source] for source, output in _CORE.items()}
        record.update(
            announcement_date=announced.isoformat(),
            auction_date=auctioned.isoformat(),
            documents=_links(row, announced, auctioned),
        )
        expected.append(record)
    expected.sort(key=lambda record: (record["auction_date"], record["cusip"]))

    _check_projected_shape(projected)
    if projected != expected:
        raise ValueError("Projected archive differs from independent metadata reconstruction")
    return {"status": "VERIFIED", "records": len(projected)}
