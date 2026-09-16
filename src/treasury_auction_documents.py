"""Fixed-document URL inventory and transport guards; no financial-value parser."""

import re
from datetime import date, datetime
from hashlib import sha256
from urllib.parse import urlsplit

KINDS = {
    "announcement_pdf",
    "competitive_pdf",
    "noncompetitive_pdf",
    "special_pdf",
    "announcement_xml",
    "competitive_xml",
}


def _day(value):
    if type(value) is not str or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value) is None:
        raise ValueError("invalid document selection date")
    return date.fromisoformat(value).isoformat()


def document_requests(selection):
    if (
        type(selection) is not dict
        or type(selection.get("records")) is not list
        or not selection["records"]
    ):
        raise ValueError("nonempty fixed record selection required")
    records = []
    seen = set()
    for entry in selection["records"]:
        if type(entry) is not dict or type(entry.get("record")) is not dict:
            raise ValueError("invalid selection record")
        row = entry["record"]
        auction, announced = _day(row.get("auction_date")), _day(row.get("announcement_date"))
        if not "2010-01-01" <= auction <= "2025-10-20" or announced > auction:
            raise ValueError("selection date outside scope")
        cusip = row.get("cusip")
        if type(cusip) is not str or re.fullmatch(r"[A-Z0-9]{9}", cusip) is None:
            raise ValueError("invalid CUSIP")
        if (auction, cusip) in seen:
            raise ValueError("duplicate selected event")
        seen.add((auction, cusip))
        records.append((row, auction, announced, cusip))
    urls = {}
    for row, auction, announced, cusip in records:
        docs = row.get("documents")
        if type(docs) is not dict or set(docs) != KINDS:
            raise ValueError("document schema differs from inventory")
        for kind, links in docs.items():
            if type(links) is not list:
                raise ValueError("document links must be a list")
            for url in links:
                if type(url) is not str:
                    raise ValueError("document URL must be a string")
                parts = urlsplit(url)
                fmt = kind.rsplit("_", 1)[1]
                year = announced[:4] if kind == "announcement_pdf" else auction[:4]
                prefix = (
                    "/xml/" if fmt == "xml" else f"/instit/annceresult/press/preanre/{year}/"
                )
                if (
                    parts.scheme != "https"
                    or parts.netloc != "www.treasurydirect.gov"
                    or parts.query
                    or parts.fragment
                    or not parts.path.startswith(prefix)
                ):
                    raise ValueError("document URL outside official mapped route")
                name = parts.path[len(prefix) :]
                if re.fullmatch(r"[A-Za-z0-9_\-]+\." + fmt, name, re.I | re.ASCII) is None:
                    raise ValueError("unsafe document filename")
                for embedded in re.findall(r"(?<![0-9])(20[0-9]{6})(?![0-9])", name):
                    day = _day(f"{embedded[:4]}-{embedded[4:6]}-{embedded[6:]}")
                    if day > "2025-10-20":
                        raise ValueError("filename suggests a post-ceiling release")
                membership = {"auction_date": auction, "cusip": cusip, "kind": kind}
                if url not in urls:
                    urls[url] = {"url": url, "format": fmt, "memberships": []}
                if membership in urls[url]["memberships"]:
                    raise ValueError("duplicate document membership")
                urls[url]["memberships"].append(membership)
    if not urls:
        raise ValueError("selection contains no document URLs")
    for item in urls.values():
        item["memberships"].sort(key=lambda m: (m["auction_date"], m["cusip"], m["kind"]))
    return [urls[url] for url in sorted(urls)]


def validate_document_receipt(body, receipt, expected_url, format):
    keys = {
        "requested_url",
        "effective_url",
        "http_status",
        "curl_exit",
        "content_type",
        "bytes",
        "sha256",
        "retrieved_utc",
    }
    limits = {"xml": 2 * 1024 * 1024, "pdf": 8 * 1024 * 1024}
    if format not in limits or type(body) is not bytes or not 0 < len(body) <= limits[format]:
        raise ValueError("document format/body limit")
    if type(receipt) is not dict or set(receipt) != keys:
        raise ValueError("document receipt schema")
    if receipt["requested_url"] != expected_url or receipt["effective_url"] != expected_url:
        raise ValueError("document URL changed")
    for key, value in [("http_status", 200), ("curl_exit", 0), ("bytes", len(body))]:
        if type(receipt[key]) is not int or receipt[key] != value:
            raise ValueError(f"invalid document {key}")
    if receipt["sha256"] != sha256(body).hexdigest():
        raise ValueError("document hash mismatch")
    mime = receipt["content_type"]
    accepted = {"pdf": {"application/pdf"}, "xml": {"text/xml", "application/xml"}}
    if type(mime) is not str or mime.split(";", 1)[0].strip().lower() not in accepted[format]:
        raise ValueError("unexpected document content type")
    if format == "pdf" and not body.startswith(b"%PDF-"):
        raise ValueError("PDF signature missing")
    try:
        timestamp = datetime.fromisoformat(receipt["retrieved_utc"])
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid retrieval timestamp") from exc
    if timestamp.utcoffset() is None:
        raise ValueError("retrieval timezone missing")
