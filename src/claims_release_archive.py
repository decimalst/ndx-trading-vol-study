"""Hash-bound DOL annual-index metadata parsing, without fetching release bodies."""

from __future__ import annotations

import hashlib
import re
from datetime import date
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

SOURCE_CEILING = "2025-10-20"
ARCHIVE_URL = "https://oui.doleta.gov/unemploy/archive.asp"
_RELEASE_PATH = re.compile(r"/press/([0-9]{4})/([0-9]{6})\.(asp|pdf)\Z")


def _decode_html(payload: bytes) -> str:
    # Dated href identities are ASCII. This preserves them in legacy HTML too.
    return payload.decode("latin-1")


class _IndexHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.has_html = False
        self.hrefs: list[str] = []
        self.year_headers: list[int] = []
        self._bold_text: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "html":
            self.has_html = True
        if tag == "base":
            raise ValueError("archive base href changes release identity")
        if tag == "b":
            if self._bold_text is not None:
                raise ValueError("ambiguous archive year header")
            self._bold_text = []
        if tag == "a":
            hrefs = [value for name, value in attrs if name == "href"]
            if len(hrefs) > 1:
                raise ValueError("ambiguous duplicate archive href attributes")
            if hrefs and hrefs[0] is not None:
                self.hrefs.append(hrefs[0])

    def handle_data(self, data):
        if self._bold_text is not None:
            self._bold_text.append(data)

    def handle_endtag(self, tag):
        if tag == "b" and self._bold_text is not None:
            label = "".join(self._bold_text).strip()
            if re.fullmatch(r"[0-9]{4}", label):
                self.year_headers.append(int(label))
            self._bold_text = None


def _release_identity(href: str, year: int) -> tuple[str, str] | None:
    if any(ord(char) < 32 for char in href):
        raise ValueError("control character in archive href")
    try:
        parsed = urlsplit(urljoin(ARCHIVE_URL, href.strip()))
    except ValueError as exc:
        raise ValueError("invalid archive href") from exc
    # Navigation is not a release candidate. A year/file under /press/ is.
    if not re.match(r"/press/[^/]+/[^/]+", parsed.path):
        return None
    if (
        parsed.scheme not in {"https", "http"}
        or parsed.netloc.lower() != "oui.doleta.gov"
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("unsafe or foreign dated release URL")
    matched = _RELEASE_PATH.fullmatch(parsed.path)
    if not matched:
        raise ValueError("invalid dated release path or extension")
    path_year = int(matched[1])
    stamp = matched[2]
    if path_year != year or stamp[4:] != str(year)[2:]:
        raise ValueError("requested, path and filename release year differ")
    try:
        release_date = date(year, int(stamp[:2]), int(stamp[2:4])).isoformat()
    except ValueError as exc:
        raise ValueError("invalid calendar date in release filename") from exc
    return release_date, "https://oui.doleta.gov" + parsed.path


def parse_archive_index(
    payload: bytes,
    year: int,
    expected_sha256: str,
    ceiling: str = SOURCE_CEILING,
) -> dict:
    """Return canonical dated links only, keeping post-ceiling links separate."""
    if type(payload) is not bytes:
        raise ValueError("archive payload must be immutable bytes")
    if type(year) is not int or not 1900 <= year <= 2099:
        raise ValueError("archive year must be a four-digit integer in 1900..2099")
    if type(ceiling) is not str or ceiling != SOURCE_CEILING:
        raise ValueError("archive source ceiling must equal 2025-10-20")
    if type(expected_sha256) is not str or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("invalid expected archive hash")
    signature = hashlib.sha256(payload).hexdigest()
    if signature != expected_sha256:
        raise ValueError("archive payload hash mismatch")

    parser = _IndexHTML()
    parser.feed(_decode_html(payload))
    parser.close()
    if not parser.has_html:
        raise ValueError("archive response is not an HTML document")
    if parser.year_headers != [year]:
        raise ValueError("archive year header missing, ambiguous or mismatched")
    dated: dict[str, str] = {}
    duplicates = 0
    for href in parser.hrefs:
        identity = _release_identity(href, year)
        if identity is None:
            continue
        release_date, release_url = identity
        prior = dated.get(release_date)
        if prior is not None:
            if prior != release_url:
                raise ValueError("competing release paths for one date")
            duplicates += 1
        else:
            dated[release_date] = release_url
    if not dated:
        raise ValueError("archive has no dated release links for the requested year")
    records = []
    excluded = []
    for release_date, release_url in sorted(dated.items()):
        row = {"release_date": release_date, "release_url": release_url}
        if release_date <= ceiling:
            records.append(row)
        else:
            excluded.append({**row, "reason": "after_source_ceiling"})
    return {
        "status": "METADATA_ONLY_NOT_RELEASE_RECONCILED",
        "year": year,
        "source_ceiling": ceiling,
        "source_sha256": signature,
        "records": records,
        "excluded_postcutoff": excluded,
        "duplicate_links_removed": duplicates,
    }
