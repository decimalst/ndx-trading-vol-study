"""Run the pretested fixed Treasury documentary capture once."""

import json
from pathlib import Path

from src.treasury_document_acquire import capture_documents

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    rep = root / "reports/next_signal_review"
    result = capture_documents(
        root,
        rep / "TREASURY_DOCUMENT_PILOT_SELECTION.json",
        rep / "TREASURY_DOCUMENT_CHECKPOINT.json",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "documents": len(result["documents"]),
                "numerical_source_admitted": False,
            }
        )
    )
