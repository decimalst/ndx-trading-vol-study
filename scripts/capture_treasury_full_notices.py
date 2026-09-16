"""Capture the preflighted complete known special-notice inventory once."""

import json
from pathlib import Path

from src.treasury_notice_acquire import capture_notice_batch

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    rep = root / "reports/next_signal_review"
    result = capture_notice_batch(
        root,
        rep / "TREASURY_FULL_NOTICE_METADATA_SELECTION.json",
        rep / "TREASURY_FULL_NOTICE_CAPTURE_CHECKPOINT.json",
    )
    print(json.dumps({"status": result["status"], "documents": len(result["documents"])}))
