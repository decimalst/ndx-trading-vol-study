"""Assemble checked auction source records without market feature construction."""

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from src.treasury_dealer_ledger import build_ledger

REPORT = "reports/treasury_dealer/"
OLD = "reports/next_signal_review/"


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    cp_path = root / (REPORT + "LEDGER_APPLICATION_CHECKPOINT.json")
    cp_bytes = cp_path.read_bytes()
    cp = json.loads(cp_bytes)
    if cp["status"] != "READY_QUALIFIED_SOURCE_LEDGER_ASSEMBLY":
        raise ValueError("Exact source-only assembly checkpoint required")
    terminal = (root / (OLD + "TREASURY_HISTORY_RECONCILED_TERMINAL_AUDIT.json")).read_bytes()
    if sha(terminal) != cp["old_terminal_sha256"]:
        raise ValueError("Original terminal source proof changed")
    pins = json.loads(terminal)["artifact_pins"].copy()
    for name, expected in cp["new_pins"].items():
        if name in pins and pins[name] != expected:
            raise ValueError("Conflicting source pins")
        pins[name] = expected
    for name, expected in pins.items():
        if sha((root / name).read_bytes()) != expected:
            raise ValueError("Source drift before assembly: " + name)

    def read(name, expected=None):
        if name not in pins or (expected is not None and pins[name] != expected):
            raise ValueError("Explicit matching source pin required: " + name)
        payload = (root / name).read_bytes()
        if sha(payload) != pins[name]:
            raise ValueError("Source snapshot changed: " + name)
        return json.loads(payload)

    identities = read(OLD + "TREASURY_HISTORY_IDENTITY_V3.json")["events"]
    accounting = read(OLD + "TREASURY_HISTORY_ACCOUNTING.json")["events"]
    if len(identities) != 1138 or len(accounting) != 1138:
        raise ValueError("Exact complete selected scope required")
    original_payloads = {}
    for event in accounting:
        if event["status"] == "PAIRED_SOURCE_AMOUNTS_AND_OFFERING_MATCH":
            event_id = event["auction_date"] + "_" + event["cusip"]
            original_payloads[event_id] = read(
                event["private_output"], event["private_output_sha256"]
            )
    recovery = read(REPORT + "RECOVERED_SOURCE_APPLICATION.json")
    if recovery["status"] != "COMPLETED_THREE_RECOVERED_SOURCE_APPLICATIONS":
        raise ValueError("Complete separately checked recovered-source application required")
    if {(row["auction_date"], row["cusip"]) for row in recovery["events"]} != {
        ("2019-12-23", "912828YZ7"),
        ("2019-12-24", "912828YY0"),
        ("2020-12-09", "91282CAV3"),
    }:
        raise ValueError("Exact three recovered-source events required")
    recovered = {
        event["auction_date"] + "_" + event["cusip"]: read(
            event["private_output"], event["private_output_sha256"]
        )
        for event in recovery["events"]
    }
    clocks = read(REPORT + "UNKNOWN_VALUE_CLOCK_DISPOSITIONS.json")
    if clocks["status"] != "PROSPECTIVE_UNKNOWN_VALUE_CLOCK_DISPOSITIONS":
        raise ValueError(
            "Explicit separately reviewed missing-value clock dispositions required"
        )
    result = build_ledger(
        identities, accounting, original_payloads, recovered, clocks["clock_reviews"]
    )
    if len(result["events"]) != 1138:
        raise ValueError("Source assembler changed the full selected universe")
    result.update(
        checkpoint_sha256=sha(cp_bytes),
        evidence_class="QUALIFIED_EXPLORATORY_WITH_Z52_REPORTED_CLOCK_ASSUMPTION",
        clock_assumption=clocks["assumption_sensitivity"],
        source_only=True,
        market_arrays_read=False,
        predictive_comparisons_added=0,
    )
    private = root / "data/source_discovery/treasury_auction/dealer_ledger_v1.json"
    public = root / (REPORT + "SOURCE_LEDGER_ASSEMBLY.json")
    if private.exists() or public.exists():
        raise ValueError("Preserve every original ledger attempt")
    payload = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    # Finish all source and code checks before publishing either new artifact.
    for name, expected in pins.items():
        if sha((root / name).read_bytes()) != expected:
            raise ValueError("Source drift during assembly: " + name)
    if cp_path.read_bytes() != cp_bytes:
        raise ValueError("Assembly checkpoint changed")
    with private.open("xb") as stream:
        stream.write(payload)
    private.chmod(0o600)
    summary = {
        "status": "ASSEMBLED_QUALIFIED_SOURCE_LEDGER",
        "completed_utc": datetime.now(UTC).isoformat(),
        "checkpoint_sha256": sha(cp_bytes),
        "counts": dict(Counter(row["status"] for row in result["events"])),
        "event_count": len(result["events"]),
        "preserved_pins": len(pins),
        "private_output": str(private.relative_to(root)),
        "private_output_sha256": sha(payload),
        "evidence_class": result["evidence_class"],
        "clock_assumption": result["clock_assumption"],
        "unknown_events": [
            row["event_id"] for row in result["events"] if row["status"] == "UNKNOWN"
        ],
        "new_predictive_comparisons": 0,
        "registered_comparisons": 144,
        "market_arrays_read": False,
        "features_built": False,
        "models_fitted": False,
        "independent_ledger_reconstruction_pending": True,
    }
    with public.open("x") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
