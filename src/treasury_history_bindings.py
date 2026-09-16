"""Required byte pins and exact metadata inventories for Treasury source stages."""

import hashlib
import re
from pathlib import Path

from src.treasury_auction_documents import document_requests

ROLES = {"announcement_pdf", "announcement_xml", "competitive_pdf", "competitive_xml"}
IDENTITY = (
    "announcement_date",
    "auction_date",
    "cusip",
    "security_type",
    "security_term",
    "term_bucket",
)


def pinned_bytes(root, pins, relative_path, expected_sha256=None) -> bytes:
    """Read once, only after the required path and optional expected hash agree."""
    if type(pins) is not dict or type(relative_path) is not str:
        raise ValueError("A checkpoint map and actual relative path are required")
    relative = Path(relative_path)
    if not relative_path or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Pinned path must be repository relative")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Pinned path escapes repository")
    signature = pins.get(relative_path)
    if type(signature) is not str or re.fullmatch(r"[0-9a-f]{64}", signature) is None:
        raise ValueError("Required path lacks a valid checkpoint pin")
    if expected_sha256 is not None and (
        type(expected_sha256) is not str or expected_sha256 != signature
    ):
        raise ValueError("Source-declared hash differs from required checkpoint pin")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Pinned file could not be read: {relative_path}") from exc
    if hashlib.sha256(raw).hexdigest() != signature:
        raise ValueError(f"Pinned source bytes changed: {relative_path}")
    return raw


def _index(container, key):
    if type(container) is not dict or type(container.get(key)) is not list:
        raise ValueError(f"Required {key} inventory is missing")
    indexed = {}
    for entry in container[key]:
        if (
            type(entry) is not dict
            or type(entry.get("url")) is not str
            or entry["url"] in indexed
        ):
            raise ValueError(f"Invalid or duplicate URL in {key}")
        indexed[entry["url"]] = entry
    return indexed


def _same(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if type(left) is list:
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def _fields_match(actual, expected, fields):
    for field in fields:
        if (
            field not in actual
            or field not in expected
            or not _same(actual[field], expected[field])
        ):
            raise ValueError(f"Document inventory disagreement: {field}")


def bind_inventory(selection, capture, metadata=None) -> None:
    """Preserve every selected event and exact requested document association."""
    # This frozen helper checks duplicate event identities, dates and all six
    # literal original document roles. Only the four declared roles are requested.
    full_requests = document_requests(selection)
    identities = {}
    try:
        for entry in selection["records"]:
            row = entry["record"]
            identities[(row["auction_date"], row["cusip"])] = {k: row[k] for k in IDENTITY}
    except (KeyError, TypeError) as exc:
        raise ValueError("Selected record lacks original identity metadata") from exc
    reconstructed = {}
    for request in full_requests:
        members = [m for m in request["memberships"] if m["kind"] in ROLES]
        if members:
            reconstructed[request["url"]] = {
                "format": request["format"],
                "memberships": members,
                "archive_memberships": [
                    identities[(m["auction_date"], m["cusip"])] | {"kind": m["kind"]}
                    for m in members
                ],
            }
    requests = _index(selection, "requests")
    if requests.keys() != reconstructed.keys():
        raise ValueError("Requested URLs differ from the selected four-role inventory")
    for url, request in requests.items():
        _fields_match(
            request, reconstructed[url], ("format", "memberships", "archive_memberships")
        )
        if request.get("source_capture_mode") not in {"reused_original_bytes", "new_request"}:
            raise ValueError("Unknown source capture mode")
    captured = _index(capture, "documents")
    if captured.keys() != requests.keys():
        raise ValueError("Capture URLs differ from the complete request inventory")
    for url, doc in captured.items():
        _fields_match(
            doc,
            requests[url],
            ("format", "memberships", "archive_memberships", "source_capture_mode"),
        )
    if metadata is not None:
        inspected = _index(metadata, "documents")
        if inspected.keys() != captured.keys():
            raise ValueError("Metadata URLs differ from the complete capture inventory")
        for url, doc in inspected.items():
            _fields_match(
                doc,
                captured[url],
                ("format", "memberships", "body_path", "body_sha256", "source_capture_mode"),
            )
