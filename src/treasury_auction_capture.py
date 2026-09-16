"""Bounded request identities and transport validation for private metadata capture."""

from datetime import datetime
from hashlib import sha256


def annual_requests():
    """Use the official archive route, keeping absent announcement links visible."""
    result = []
    for year in range(2010, 2026):
        start = f"{year}-01-01"
        end = f"{year}-12-31" if year < 2025 else "2025-10-20"
        url = (
            "https://www.treasurydirect.gov/TA_WS/securities/search?"
            f"startDate={start}&endDate={end}&compact=true&dateFieldName=auctionDate&format=json"
        )
        result.append((start, end, url))
    return result


def validate_receipt(body, receipt, expected_url):
    """Require verified successful transport and exact captured-byte identity."""
    required = {
        "requested_url",
        "effective_url",
        "http_status",
        "curl_exit",
        "content_type",
        "bytes",
        "sha256",
        "retrieved_utc",
    }
    if type(body) is not bytes or not body:
        raise ValueError("empty or non-byte response")
    if type(receipt) is not dict or set(receipt) != required:
        raise ValueError("receipt schema")
    if receipt["requested_url"] != expected_url or receipt["effective_url"] != expected_url:
        raise ValueError("request/effective URL differs from fixed request")
    for key, expected in [("http_status", 200), ("curl_exit", 0), ("bytes", len(body))]:
        if type(receipt[key]) is not int or receipt[key] != expected:
            raise ValueError(f"invalid {key}")
    if receipt["sha256"] != sha256(body).hexdigest():
        raise ValueError("response byte hash mismatch")
    mime = receipt["content_type"]
    if type(mime) is not str or mime.split(";", 1)[0].strip().lower() != "application/json":
        raise ValueError("response is not application/json")
    try:
        stamp = datetime.fromisoformat(receipt["retrieved_utc"])
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid retrieval timestamp") from exc
    if stamp.utcoffset() is None:
        raise ValueError("retrieval timestamp needs timezone")
