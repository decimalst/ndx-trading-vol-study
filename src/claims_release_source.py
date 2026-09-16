"""Parse a bounded ALFRED first-release export, without claiming DOL reconciliation."""

import csv
import hashlib
import io
import json
import re
import stat
import zipfile
from datetime import date, datetime, timedelta

SOURCE_CEILING = "2025-10-20"
INITIAL_SNAPSHOT = "2009-05-28"
DATA_NAME = "obs.,_initial_release_only.csv"
MAX_ARCHIVE_BYTES = 2 * 1024 * 1024
MAX_EXPANDED_BYTES = 16 * 1024 * 1024
HEADER = ["period_start_date", "ICSA", "realtime_start_date"]


def _digest(payload):
    return hashlib.sha256(payload).hexdigest()


def _date(value):
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value) is None
    ):
        raise ValueError("Canonical ISO calendar date required")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError("Valid calendar date required") from error
    return parsed


def _request(request):
    if not isinstance(request, dict):
        raise ValueError("Explicit source request required")
    required = {
        "requested_url": "https://alfred.stlouisfed.org/series/downloaddata?seid=ICSA",
        "method": "POST",
        "mode": "first_release_only",
        "series_id": "ICSA",
        "units": "lin",
        "file_type": "4",
        "file_format": "csv",
        "source_ceiling": SOURCE_CEILING,
    }
    if any(request.get(key) != value for key, value in required.items()):
        raise ValueError("Fixed ICSA first-release request identity required")
    start = _date(request.get("observation_start"))
    end = _date(request.get("observation_end"))
    ceiling = _date(SOURCE_CEILING)
    if start > end or end > ceiling:
        raise ValueError("Observation request outside fixed source ceiling")
    vintages = request.get("vintage_dates")
    if not isinstance(vintages, list) or not vintages:
        raise ValueError("Nonempty declared vintage sequence required")
    parsed = [_date(value) for value in vintages]
    if parsed != sorted(set(parsed)) or any(
        value < _date(INITIAL_SNAPSHOT) or value > ceiling for value in parsed
    ):
        raise ValueError("Ordered unique vintages within archive and source fences required")
    try:
        created = datetime.fromisoformat(request["created_utc"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Explicit UTC acquisition timestamp required") from error
    if (
        created.tzinfo is None
        or created.utcoffset() != timedelta(0)
        or created.date() < max(end, parsed[-1])
    ):
        raise ValueError("UTC acquisition must follow every requested source date")
    try:
        canonical = json.dumps(
            request, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("utf-8")
    except (ValueError, TypeError) as error:
        raise ValueError("Canonical request metadata required") from error
    return start, end, set(parsed), _digest(canonical)


def _readme(payload, vintages):
    text = payload.decode("utf-8-sig").replace("\r\n", "\n")
    identities = {
        "Series ID": "ICSA",
        "Link": "https://alfred.stlouisfed.org/series?seid=ICSA",
        "Output Format": "Observations, Initial Release Only",
    }
    for key, expected in identities.items():
        values = re.findall(r"^" + re.escape(key) + r": ([^\n]*)$", text, re.MULTILINE)
        if values != [expected]:
            raise ValueError("README source identity differs: " + key)
    labels = ["Source", "Release", "Units", "Frequency", "Seasonal Adjustment"]
    expected = [
        "Initial Claims",
        "U.S. Employment and Training Administration",
        "Unemployment Insurance Weekly Claims Report",
        "Number",
        "Weekly, Ending Saturday",
        "Seasonally Adjusted",
    ]
    lines = text.splitlines()
    title = [
        i
        for i, line in enumerate(lines)
        if re.fullmatch(r"Title\s+Real-Time Start\s+Real-Time End", line)
    ]
    if len(title) != 1:
        raise ValueError("One complete README attribute table required")
    indices = [title[0]]
    for label in labels:
        positions = [i for i, line in enumerate(lines) if line == label]
        if len(positions) != 1:
            raise ValueError("One README attribute required: " + label)
        indices += positions
    if indices != sorted(indices):
        raise ValueError("README attribute order differs")
    notes = [i for i, line in enumerate(lines) if line == "Notes"]
    if len(notes) != 1 or notes[0] <= indices[-1]:
        raise ValueError("One Notes boundary after the README attributes required")
    ends = indices[1:] + notes
    for i, end, value in zip(indices, ends, expected, strict=True):
        section = [line for line in lines[i + 1 : end] if line.strip()]
        if (
            len(section) != 1
            or re.fullmatch(re.escape(value) + r"\s+2009-05-28\s+Current", section[0])
            is None
        ):
            raise ValueError("README attribute value or real-time identity differs")
    if text.count("Vintage Dates Specified:") != 1:
        raise ValueError("One README vintage declaration required")
    section = text.split("Vintage Dates Specified:", 1)[1].strip().splitlines()
    if len(section) < 3 or section[0] != "----------" or section[-1] != "----------":
        raise ValueError("README vintage delimiters differ")
    declared = section[1:-1]
    for value in declared:
        _date(value)
    if declared != vintages:
        raise ValueError("README vintages differ from the exact declared request")


def _parse_value(token):
    """Decode only an already date-admitted count; preserve the explicit missing marker."""
    if token == ".":
        return None
    if (
        not isinstance(token, str)
        or len(token) > 64
        or re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.0+)?", token) is None
    ):
        raise ValueError("Positive integral claims count or explicit missing marker required")
    value = int(token.split(".", 1)[0])
    if value <= 0:
        raise ValueError("Claims count must be positive")
    return value


def parse_export(payload, request, expected_sha256):
    """Check the entire structural/date envelope before decoding any observation count."""
    if not isinstance(payload, bytes) or len(payload) > MAX_ARCHIVE_BYTES:
        raise ValueError("Bounded immutable ZIP bytes required")
    signature = _digest(payload)
    if not isinstance(expected_sha256, str) or signature != expected_sha256:
        raise ValueError("Source ZIP hash differs from its declared pin")
    start, end, vintages, request_signature = _request(request)
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            if len(members) != 2 or {m.filename for m in members} != {"README.txt", DATA_NAME}:
                raise ValueError("Exact two ALFRED export members required")
            if sum(m.file_size for m in members) > MAX_EXPANDED_BYTES:
                raise ValueError("Expanded ZIP exceeds declared size bound")
            for member in members:
                kind = stat.S_IFMT(member.external_attr >> 16)
                if (
                    member.is_dir()
                    or kind not in (0, stat.S_IFREG)
                    or member.flag_bits & 1
                    or member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                ):
                    raise ValueError("Plain unencrypted regular ZIP members required")
            readme = archive.read("README.txt")
            data = archive.read(DATA_NAME)
        _readme(readme, request["vintage_dates"])
        reader = csv.reader(io.StringIO(data.decode("utf-8-sig"), newline=""), strict=True)
        if next(reader, None) != HEADER:
            raise ValueError("Exact first-release CSV header required")
        pending = []
        excluded = []
        observed_dates = []
        for cells in reader:
            if len(cells) != 3 or any("\r" in value or "\n" in value for value in cells):
                raise ValueError("Three single-line source fields required")
            observation = _date(cells[0])
            release = _date(cells[2])
            if observation > _date(SOURCE_CEILING) or release > _date(SOURCE_CEILING):
                raise ValueError("Source row extends beyond fixed ceiling")
            if not start <= observation <= end or observation.weekday() != 5:
                raise ValueError("Saturday reference week within requested bounds required")
            if release <= observation or release not in vintages:
                raise ValueError("Later declared first-release vintage required")
            if observed_dates and observation <= observed_dates[-1]:
                raise ValueError("Ordered unique first-release reference weeks required")
            observed_dates.append(observation)
            common = {
                "observation_date": observation.isoformat(),
                "alfred_realtime_start_date": release.isoformat(),
                "source_line": reader.line_num,
            }
            if release.isoformat() == INITIAL_SNAPSHOT:
                excluded.append({**common, "reason": "initial_2009_snapshot"})
            else:
                if observation <= _date("2009-05-23"):
                    raise ValueError("Pre-archive history cannot masquerade as a first report")
                pending.append((common, cells[1]))
    except (zipfile.BadZipFile, UnicodeError, csv.Error, RuntimeError, OSError) as error:
        raise ValueError("Invalid first-release export encoding or archive") from error

    # No numeric conversion above this line, including for earlier eligible rows.
    records = []
    for common, token in pending:
        value = _parse_value(token)
        records.append(
            {**common, "value": value, "status": "missing" if value is None else "observed"}
        )
    expected = []
    cursor = start + timedelta(days=(5 - start.weekday()) % 7)
    while cursor <= end:
        expected.append(cursor)
        cursor += timedelta(days=7)
    observed = set(observed_dates)
    return {
        "status": "STRUCTURALLY_VERIFIED_NOT_RELEASE_RECONCILED",
        "source_sha256": signature,
        "readme_sha256": _digest(readme),
        "data_sha256": _digest(data),
        "request_sha256": request_signature,
        "records": records,
        "excluded_initial_snapshot": excluded,
        "missing_reference_weeks": [
            value.isoformat() for value in expected if value not in observed
        ],
        "source_ceiling": SOURCE_CEILING,
        "limitations": [
            "Export structure and date fences are checked; original DOL releases have not been reconciled.",
            "ALFRED real-time start may be source-dated and is not an independently certified ingestion timestamp.",
            "No market cohort, numerical predictive feature, forecasting fit or source-ready decision is produced.",
        ],
    }
