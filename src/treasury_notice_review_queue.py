"""Lexical grouping for human review; never classification, admission or event dedup.

Only complete month/day/year or US slash-date shapes receive date-part markers.
This does not validate calendar dates or authenticate publication. Punctuation,
ordinary word case, word order, headings and every section remain in the template.
Markers deliberately distinguish date parts, possible CUSIPs and other digit runs.
A nine-digit standalone token also meets the requested lexical CUSIP shape: this
is not a security-identifier validator. Every source still needs individual review.
"""

import hashlib
import re
import unicodedata

_MAX_BYTES = 2 * 1024 * 1024
_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
)
_DAY = r"(?:0?[1-9]|[12][0-9]|3[01])"
_WRITTEN_DATE = re.compile(
    rf"(?<!\w)(?:{_MONTHS})(?P<gap> +){_DAY}"
    r"(?P<separator> *,? +|, *)(?P<year>[0-9]{4})(?!\w)",
    re.IGNORECASE,
)
_SLASH_DATE = re.compile(
    rf"(?<!\w)(?:0?[1-9]|1[0-2])(?P<slash_one> */ *){_DAY}"
    r"(?P<slash_two> */ *)(?:[0-9]{4}|[0-9]{2})(?!\w)"
)
_CUSIP = re.compile(r"(?<!\w)[A-Z0-9]{9}(?!\w)")
_DIGITS = re.compile(r"\d+")


def template_fingerprint(text: str) -> dict:
    """Return a SHA256 of the masked UTF-8 text plus that exact text.

    Input is a strict, nonempty string of at most 2 MiB in UTF-8, without
    non-whitespace control characters, invalid Unicode or reserved marker
    delimiters. Reserved delimiters prevent raw marker-like words from colliding
    with generated markers. Amounts and yields are never numerically converted.
    The caller must preserve every original document and event association even
    when hashes match. A matching hash is only a review-queue grouping hint.
    """
    if type(text) is not str or len(text) > _MAX_BYTES:
        raise ValueError("text must be a string of at most 2 MiB in UTF-8")
    try:
        if len(text.encode("utf-8")) > _MAX_BYTES:
            raise ValueError("text exceeds 2 MiB in UTF-8")
    except UnicodeEncodeError as error:
        raise ValueError("text contains invalid Unicode") from error
    if any(unicodedata.category(char) == "Cc" and not char.isspace() for char in text):
        raise ValueError("text contains non-whitespace control characters")

    normalized = " ".join(unicodedata.normalize("NFKC", text).split())
    if not normalized or "⟦" in normalized or "⟧" in normalized:
        raise ValueError("text must be nonempty and contain no reserved marker delimiters")

    normalized = _WRITTEN_DATE.sub(
        lambda match: (
            "⟦DATE_MONTH⟧"
            + match.group("gap")
            + "⟦DATE_DAY⟧"
            + match.group("separator")
            + "⟦DATE_YEAR⟧"
        ),
        normalized,
    )
    normalized = _SLASH_DATE.sub(
        lambda match: (
            "⟦DATE_MONTH⟧"
            + match.group("slash_one")
            + "⟦DATE_DAY⟧"
            + match.group("slash_two")
            + "⟦DATE_YEAR⟧"
        ),
        normalized,
    )
    normalized = _CUSIP.sub(
        lambda match: "⟦CUSIP⟧" if any(char.isdigit() for char in match[0]) else match[0],
        normalized,
    )
    normalized = _DIGITS.sub("⟦NUMBER⟧", normalized)
    return {
        "template_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "normalized_text": normalized,
    }
