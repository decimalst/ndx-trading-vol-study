"""One-shot bounded archive capture; source numerical fields stay quarantined."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from src.treasury_auction_archive import project_archive
from src.treasury_auction_capture import annual_requests, validate_receipt
from src.verify_treasury_auction_archive import verify_archive

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data/source_discovery/treasury_auction/archive_capture_v1"
REPORTS = ROOT / "reports/next_signal_review"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def run():
    checkpoint = REPORTS / "TREASURY_ARCHIVE_CHECKPOINT.json"
    frozen = json.loads(checkpoint.read_text())
    if frozen["status"] != "READY_METADATA_CAPTURE":
        raise ValueError("checkpoint is not ready")
    for rel, expected in frozen["pins"].items():
        if digest(ROOT / rel) != expected:
            raise ValueError(f"checkpoint drift: {rel}")
    old = json.loads(
        (ROOT / "reports/commodity_implied/predictive/freeze_record.json").read_text()
    )
    for rel, expected in old["code"].items():
        if digest(ROOT / rel) != expected:
            raise ValueError(f"prior frozen code drift: {rel}")
    if (REPORTS / "TREASURY_ARCHIVE_CAPTURE.json").exists():
        raise FileExistsError("capture report exists; refusing to repeat")
    PRIVATE.mkdir(parents=True, exist_ok=False)
    write_json(
        PRIVATE / "started.json",
        {
            "started_utc": datetime.now(UTC).isoformat(),
            "checkpoint_sha256": digest(checkpoint),
            "requests": [url for _, _, url in annual_requests()],
        },
    )
    captures = []
    for start, end, url in annual_requests():
        year_dir = PRIVATE / start[:4]
        year_dir.mkdir()
        body_path = year_dir / "response.json"
        headers_path = year_dir / "headers.txt"
        proc = subprocess.run(
            [
                "curl",
                "--silent",
                "--show-error",
                "--max-time",
                "45",
                "--proto",
                "=https",
                "--dump-header",
                str(headers_path),
                "--output",
                str(body_path),
                "--write-out",
                "%{http_code}\n%{url_effective}\n%{content_type}",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=50,
        )
        # Curl never follows redirects and never disables certificate validation.
        fields = proc.stdout.split("\n")
        body = body_path.read_bytes() if body_path.exists() else b""
        receipt = {
            "requested_url": url,
            "effective_url": fields[1] if len(fields) > 1 else "",
            "http_status": int(fields[0]) if fields and fields[0].isdigit() else 0,
            "curl_exit": proc.returncode,
            "content_type": fields[2] if len(fields) > 2 else "",
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
            "retrieved_utc": datetime.now(UTC).isoformat(),
        }
        write_json(year_dir / "receipt.json", receipt)
        with (year_dir / "transport_stderr.txt").open("x") as f:
            f.write(proc.stderr)
        entry = {
            "year": start[:4],
            "start_date": start,
            "end_date": end,
            "receipt_path": str((year_dir / "receipt.json").relative_to(ROOT)),
            "receipt_sha256": digest(year_dir / "receipt.json"),
            "response_sha256": receipt["sha256"],
            "headers_sha256": digest(headers_path) if headers_path.exists() else None,
        }
        try:
            validate_receipt(body, receipt, url)
            records = project_archive(body, start_date=start, end_date=end)
            independent = verify_archive(body, records, start_date=start, end_date=end)
            if independent != {"status": "VERIFIED", "records": len(records)}:
                raise ValueError("unexpected independent verification schema")
            write_json(year_dir / "metadata.json", records)
            entry.update(
                status="VERIFIED_METADATA_ONLY",
                records=len(records),
                security_types=dict(Counter(row["security_type"] for row in records)),
                term_buckets=dict(Counter(row["term_bucket"] for row in records)),
                missing_announcement_pdf=sum(
                    not row["documents"]["announcement_pdf"] for row in records
                ),
                missing_competitive_pdf=sum(
                    not row["documents"]["competitive_pdf"] for row in records
                ),
                special_notice_rows=sum(
                    bool(row["documents"]["special_pdf"]) for row in records
                ),
                metadata_sha256=digest(year_dir / "metadata.json"),
                independent=independent,
            )
        except (ValueError, TypeError, KeyError) as exc:
            entry.update(status="REQUIRES_SOURCE_REVIEW", error=str(exc))
        captures.append(entry)
        write_json(year_dir / "summary.json", entry)
        print(
            json.dumps(
                {"year": start[:4], "status": entry["status"], "records": entry.get("records")}
            ),
            flush=True,
        )
        if entry["status"] != "VERIFIED_METADATA_ONLY":
            break
    finished = {
        "status": (
            "VERIFIED_METADATA_INVENTORY"
            if len(captures) == 16
            and all(c["status"] == "VERIFIED_METADATA_ONLY" for c in captures)
            else "REQUIRES_SOURCE_REVIEW"
        ),
        "completed_utc": datetime.now(UTC).isoformat(),
        "checkpoint_sha256": digest(checkpoint),
        "captures": captures,
        "numerical_source_admitted": False,
        "predictive_comparisons_added": 0,
        "cumulative_comparisons": 144,
    }
    write_json(REPORTS / "TREASURY_ARCHIVE_CAPTURE.json", finished)
    print(json.dumps({"terminal_status": finished["status"], "captures": len(captures)}))
    return 0 if finished["status"] == "VERIFIED_METADATA_INVENTORY" else 2


if __name__ == "__main__":
    raise SystemExit(run())
