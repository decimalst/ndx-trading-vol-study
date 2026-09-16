"""Metadata inventory retaining exact empty Treasury display-group labels.

The initial strict archive projector remains immutable. No numerical values are read.
"""

import json
import re
from datetime import date


class _NumberToken(str):
    """Keep JSON numerical fields as text without confusing them with strings."""


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _constant(_value):
    raise ValueError("nonstandard JSON numeric constant")


def _date(value, *, allow_midnight=True):
    pattern = r"[0-9]{4}-[0-9]{2}-[0-9]{2}"
    if allow_midnight:
        pattern += r"(?:T00:00:00)?"
    if type(value) is not str or re.fullmatch(pattern, value) is None:
        raise ValueError("invalid date lexeme")
    try:
        return date.fromisoformat(value[:10]).isoformat()
    except ValueError as exc:
        raise ValueError("invalid calendar date") from exc


def project_archive(raw: bytes, *, start_date: str, end_date: str) -> list[dict]:
    """Fail the full response before URL projection when any date is invalid."""
    start = _date(start_date, allow_midnight=False)
    end = _date(end_date, allow_midnight=False)
    if not "2010-01-01" <= start <= end <= "2025-10-20":
        raise ValueError("requested date bounds outside fixed source scope")
    if type(raw) is not bytes or not raw:
        raise ValueError("nonempty UTF-8 JSON bytes required")
    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            parse_int=_NumberToken,
            parse_float=_NumberToken,
            parse_constant=_constant,
            object_pairs_hook=_object,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("invalid UTF-8 JSON") from exc
    if type(decoded) is dict and set(decoded) == {"securityList"}:
        decoded = decoded["securityList"]
    if type(decoded) is not list or not decoded:
        raise ValueError("nonempty archive row list required")

    dates = []
    for row in decoded:
        if type(row) is not dict:
            raise ValueError("archive date preflight requires object rows")
        announcement, auction = _date(row.get("h")), _date(row.get("i"))
        if not start <= auction <= end:
            raise ValueError("auction date outside requested bounds")
        if announcement > auction:
            raise ValueError("announcement date follows auction date")
        dates.append((announcement, auction))

    names = {
        "e3": "announcement_pdf",
        "f3": "competitive_pdf",
        "f31": "noncompetitive_pdf",
        "f32": "special_pdf",
        "e4": "announcement_xml",
        "ia1": "competitive_xml",
    }
    identities = set()
    projected = []
    for row, (announcement, auction) in zip(decoded, dates):
        for key in ("a", "d", "z3a", "t3a", "h", "i"):
            if type(row.get(key)) is not str or (
                not row[key].strip() and not (key == "t3a" and row[key] == "")
            ):
                raise ValueError(f"core metadata {key} must be a nonempty JSON string")
        if re.fullmatch(r"[A-Z0-9]{9}", row["a"]) is None:
            raise ValueError("CUSIP must contain nine uppercase alphanumeric characters")
        identity = (auction, row["a"])
        if identity in identities:
            raise ValueError("duplicate auction/CUSIP identity")
        identities.add(identity)
        documents = {}
        for key, name in names.items():
            value = row.get(key)
            if value is None or (type(value) is str and value == ""):
                documents[name] = []
                continue
            if type(value) is not str:
                raise ValueError("document filename must be a JSON string")
            filenames = value.split(", ") if key == "f32" else [value]
            ext = "xml" if name.endswith("_xml") else "pdf"
            documents[name] = []
            for filename in filenames:
                if (
                    re.fullmatch(
                        r"[A-Za-z0-9_\-]+\." + ext, filename, re.IGNORECASE | re.ASCII
                    )
                    is None
                ):
                    raise ValueError("unsafe or invalid document filename")
                if ext == "xml":
                    prefix = "https://www.treasurydirect.gov/xml/"
                else:
                    year = announcement[:4] if key == "e3" else auction[:4]
                    prefix = f"https://www.treasurydirect.gov/instit/annceresult/press/preanre/{year}/"
                documents[name].append(prefix + filename)
        projected.append(
            {
                "cusip": row["a"],
                "security_term": row["d"],
                "security_type": row["z3a"],
                "term_bucket": row["t3a"],
                "announcement_date": announcement,
                "auction_date": auction,
                "documents": documents,
            }
        )
    return sorted(projected, key=lambda row: (row["auction_date"], row["cusip"]))
