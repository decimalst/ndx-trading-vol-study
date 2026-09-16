"""Run the frozen one-shot history capture; never construct its checkpoint."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.treasury_history_acquire import capture_history  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    reports = root / "reports/next_signal_review"
    result = capture_history(
        root,
        reports / "TREASURY_HISTORY_SELECTION.json",
        reports / "TREASURY_HISTORY_CHECKPOINT.json",
    )
    print(json.dumps({"status": result["status"], "counts": result["counts"]}, sort_keys=True))
    return 0 if result["status"] == "CAPTURED_SCHEMA_ONLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
