"""Bounded active source-document closure for immutable calendar reconstruction.

Only source metadata and exact source bytes are read. Historical/variant ledger
declarations remain provenance; they never override the selected active record.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from datetime import date, datetime
from pathlib import Path, PurePosixPath

CAPTURE_MANIFEST = "data/source_discovery/bls_plan_capture/capture_manifest.json"
METADATA_KEYS = ("cpi", "nfp", "fomc", "fomc_coverage")


def _path(root, name):
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or "\x00" in name
        or PurePosixPath(name).is_absolute()
        or any(part in ("", ".", "..") for part in name.split("/"))
        or PurePosixPath(name).as_posix() != name
    ):
        raise ValueError("A canonical relative source path is required")
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Source path escapes the registered root")
    return path


def _hash(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("A lowercase 64-character SHA256 hash is required")
    return value


def _read(root, name, expected):
    expected = _hash(expected)
    payload = _path(root, name).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"Source hash mismatch: {name}")
    return payload


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _json(payload):
    return json.loads(
        payload.decode("utf-8"), object_pairs_hook=_object, parse_constant=_constant
    )


def _year(value):
    if isinstance(value, bool):
        raise ValueError("Invalid FOMC year")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value):
        result = int(value)
    else:
        raise ValueError("Invalid FOMC year")
    if not 1 <= result <= 9999:
        raise ValueError("Invalid FOMC year")
    return result


def collect_sources(root, old_calendar_protocol, registered_pins):
    """Return deterministic active path/hash closure, without market-data reads.

    The four metadata roots must already be registered. Newly discovered hashes
    are committed by those metadata bytes; any direct registration must agree.
    Out-of-fence BLS records are excluded before accessing snapshot fields.
    Source text is kept opaque here and decoded only by the downstream reader.
    """
    root = Path(root)
    if not isinstance(registered_pins, dict):
        raise ValueError("Registered pins must be a path-to-hash mapping")
    sources = old_calendar_protocol["sources"]
    section = old_calendar_protocol["calendar"]
    lower = date.fromisoformat(section["source_publication_start"])
    upper = date.fromisoformat(section["source_publication_end"])
    if lower > upper:
        raise ValueError("Source publication fence is reversed")
    files, metadata_files, runtime_files, documentary_files = {}, {}, {}, {}
    references, excluded = [], []
    cached = {}

    def add(name, expected, owner, field, role):
        _path(root, name)
        expected = _hash(expected)
        if name in registered_pins and _hash(registered_pins[name]) != expected:
            raise ValueError(f"Registered source hash conflict: {name}")
        if name in files and files[name] != expected:
            raise ValueError(f"Active source hash conflict: {name}")
        files[name] = expected
        mapping = {
            "metadata": metadata_files,
            "runtime": runtime_files,
            "documentary": documentary_files,
        }[role]
        mapping[name] = expected
        references.append(
            {"owner": owner, "field": field, "path": name, "sha256": expected, "role": role}
        )

    def payload(name):
        if name not in cached:
            cached[name] = _read(root, name, files[name])
        return cached[name]

    decoded = {}
    for key in METADATA_KEYS:
        name = sources.get(key)
        _path(root, name)
        if name not in registered_pins:
            raise ValueError(f"Source metadata is not registered: {name}")
        add(name, registered_pins[name], "old_calendar_protocol", f"sources.{key}", "metadata")
        raw = payload(name)
        decoded[key] = raw if key == "fomc" else _json(raw)

    bounded = 0
    for event, prefix in (("cpi", "cpi"), ("nfp", "empsit")):
        ledger = decoded[event]
        if not isinstance(ledger, dict) or not isinstance(ledger.get("records"), list):
            raise ValueError("Selected BLS ledger requires a records list")
        for index, record in enumerate(ledger["records"]):
            if not isinstance(record, dict):
                raise ValueError("Selected BLS record must be an object")
            url = record.get("source_url")
            pattern = rf"https://www\.bls\.gov/news\.release/archives/{prefix}_(\d{{8}})\.htm"
            match = re.fullmatch(pattern, url) if isinstance(url, str) else None
            if match is None:
                raise ValueError("An exact official dated BLS source URL is required")
            publication_date = datetime.strptime(match[1], "%m%d%Y").date()
            if not lower <= publication_date <= upper:
                if record.get("parse_status") != "OUTSIDE_PUBLICATION_FENCE":
                    raise ValueError("Out-of-fence BLS record is not explicitly excluded")
                excluded.append(
                    {
                        "event": event,
                        "record_index": index,
                        "source_url": url,
                        "parse_status": record["parse_status"],
                        "reason": "OUTSIDE_PUBLICATION_FENCE",
                    }
                )
                continue
            add(
                record.get("snapshot_path"),
                record.get("snapshot_sha256"),
                sources[event],
                f"records/{index}/snapshot_path",
                "runtime",
            )
            bounded += 1

    artifact = decoded["nfp"].get("artifact_sha256")
    if not isinstance(artifact, dict) or CAPTURE_MANIFEST not in artifact:
        raise ValueError("Final payroll ledger must commit the documentary capture manifest")
    add(
        CAPTURE_MANIFEST,
        artifact[CAPTURE_MANIFEST],
        sources["nfp"],
        f"artifact_sha256/{CAPTURE_MANIFEST}",
        "documentary",
    )
    if not isinstance(_json(payload(CAPTURE_MANIFEST)), dict):
        raise ValueError("Documentary capture manifest must be a JSON object")

    stream = io.StringIO(decoded["fomc"].decode("utf-8"), newline="")
    reader = csv.DictReader(stream)
    required = {
        "annual_schedule_year",
        "source_url",
        "source_extraction_path",
        "source_extraction_sha256",
    }
    if (
        reader.fieldnames is None
        or len(reader.fieldnames) != len(set(reader.fieldnames))
        or not required.issubset(reader.fieldnames)
    ):
        raise ValueError("FOMC source table requires unambiguous source columns")
    rows = list(reader)
    coverage = decoded["fomc_coverage"]
    if not isinstance(coverage, list):
        raise ValueError("FOMC coverage must be a list")
    annual = {}
    for item in coverage:
        if not isinstance(item, dict):
            raise ValueError("FOMC coverage record must be an object")
        year = _year(item.get("annual_schedule_year"))
        if (
            year in annual
            or item.get("coverage_start_date") != f"{year:04d}-01-01"
            or item.get("coverage_end_date") != f"{year:04d}-12-31"
            or type(item.get("plan_count")) is not int
            or item["plan_count"] < 1
        ):
            raise ValueError("FOMC annual coverage is ambiguous or incomplete")
        _hash(item.get("source_extraction_sha256"))
        if not isinstance(item.get("source_url"), str) or not item["source_url"]:
            raise ValueError("FOMC annual source URL is missing")
        annual[year] = item
    grouped = {}
    for index, row in enumerate(rows):
        if None in row or any(value is None for value in row.values()):
            raise ValueError("FOMC source table has malformed rows")
        year = _year(row["annual_schedule_year"])
        if year not in annual:
            raise ValueError("FOMC row has no active annual coverage")
        covered = annual[year]
        if (
            row["source_url"] != covered["source_url"]
            or row["source_extraction_sha256"] != covered["source_extraction_sha256"]
        ):
            raise ValueError("FOMC coverage source hash or URL conflict")
        add(
            row["source_extraction_path"],
            row["source_extraction_sha256"],
            sources["fomc"],
            f"rows/{index}/source_extraction_path",
            "runtime",
        )
        grouped.setdefault(year, []).append(row)
    if set(grouped) != set(annual):
        raise ValueError("FOMC active coverage years do not match source rows")
    for year, group in grouped.items():
        if (
            len(group) != annual[year]["plan_count"]
            or len({row["source_extraction_path"] for row in group}) != 1
        ):
            raise ValueError("FOMC coverage count or annual source path conflict")

    for name in sorted(files):
        payload(name)
    return {
        "schema_version": 1,
        "files": dict(sorted(files.items())),
        "metadata_files": dict(sorted(metadata_files.items())),
        "runtime_files": dict(sorted(runtime_files.items())),
        "documentary_files": dict(sorted(documentary_files.items())),
        "references": sorted(
            references, key=lambda row: (row["owner"], row["field"], row["path"])
        ),
        "excluded_bls": sorted(excluded, key=lambda row: (row["event"], row["record_index"])),
        "counts": {
            "metadata_files": len(metadata_files),
            "runtime_files": len(runtime_files),
            "documentary_files": len(documentary_files),
            "unique_files": len(files),
            "bounded_bls_records": bounded,
            "fomc_reference_rows": len(rows),
            "excluded_bls_records": len(excluded),
        },
    }


def snapshot_sources(root, files):
    """Read each file once, verify its hash, and return those exact immutable bytes.

    Stage the returned bytes directly; never reopen the original path afterward.
    No partial result is returned if any path or hash check fails.
    """
    if not isinstance(files, dict):
        raise ValueError("Source files must be a path-to-hash mapping")
    root = Path(root)
    return {name: _read(root, name, expected) for name, expected in sorted(files.items())}
