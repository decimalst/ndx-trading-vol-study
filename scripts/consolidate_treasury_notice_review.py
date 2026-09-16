"""Bind individual documentary findings to the fixed sources; no source admission."""

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    rep = root / "reports/next_signal_review"

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    cp = rep / "TREASURY_FULL_NOTICE_REVIEW_CHECKPOINT.json"
    checkpoint = json.loads(cp.read_text())
    for name, expected in checkpoint["pins"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Review source changed: {name}")
    source = json.loads((rep / "TREASURY_FULL_NOTICE_TEXT.json").read_text())
    assignments = json.loads(
        (rep / "TREASURY_FULL_NOTICE_REVIEW_ASSIGNMENTS.json").read_text()
    )["assignments"]
    byurl = {row["url"]: row for row in source["documents"]}
    merged, ledger_pins = {}, {}
    for reviewer, suffix in (
        ("root", "ROOT"),
        ("new_study_tests", "A"),
        ("peak_age_admission", "B"),
        ("replay_entry_finish", "C"),
    ):
        path = rep / f"TREASURY_NOTICE_LEDGER_{suffix}.json"
        ledger = json.loads(path.read_text())
        if ledger["checkpoint_sha256"] != digest(cp):
            raise ValueError("Reviewer used a different source checkpoint")
        rows = ledger["documents"]
        if len(rows) != len(assignments[reviewer]) or {r["url"] for r in rows} != set(
            assignments[reviewer]
        ):
            raise ValueError("Reviewer did not account for the complete assigned URL set")
        ledger_pins[str(path.relative_to(root))] = digest(path)
        for row in rows:
            url = row["url"]
            if url in merged:
                raise ValueError("Duplicate reviewed URL")
            actual = byurl[url]
            if row["body_sha256"] != actual["body_sha256"] or row.get(
                "text_sha256"
            ) != actual.get("text_sha256"):
                raise ValueError("Review is not bound to its source bytes/text")
            if row["archive_memberships"] != actual["archive_memberships"]:
                raise ValueError("Review changed an archive association")
            if actual["status"] == "DATED_NOTICE_TEXT_READY_FOR_REVIEW":
                if row["release_date"] != actual["release_date"]:
                    raise ValueError("Review changed an authenticated header date")
            elif row.get("release_date") is not None and not (
                reviewer == "root"
                and row.get("manual_release_date_basis")
                and row.get("visual_evidence")
            ):
                raise ValueError("Uncertain header date lacks explicit visual resolution")
            for key in (
                "purposes",
                "body_applicability",
                "affected_fields",
                "explicit_unchanged_fields",
                "evidence_sections",
                "unresolved",
            ):
                if not isinstance(row.get(key), list):
                    raise ValueError(f"Missing documentary field: {key}")
            if (
                not row["purposes"]
                or not row["evidence_sections"]
                or not row.get("disposition")
            ):
                raise ValueError("Empty documentary classification or evidence")
            if row.get("source_admitted", False) is not False:
                raise ValueError("Documentary review cannot admit the full source")
            merged[url] = {**row, "review_ledger": str(path.relative_to(root))}
    if set(merged) != set(byurl) or len(merged) != 253:
        raise ValueError("Full notice accounting is incomplete")
    rows = [merged[row["url"]] for row in source["documents"]]
    association_count = sum(len(row["archive_memberships"]) for row in rows)
    if association_count != 298:
        raise ValueError("Full archive association count changed")
    followup = rep / "TREASURY_MISSING_NOTICE_FOLLOWUP.md"
    for row in rows:
        if row["url"].endswith("/BPD_SPL_20191219_3.pdf"):
            row["supplemental_indexed_text_followup"] = {
                "path": str(followup.relative_to(root)),
                "sha256": digest(followup),
                "original_pdf_recovered": False,
                "status": "PROVISIONAL_PURPOSE_LEAD_ORIGINAL_REMAINS_UNAVAILABLE",
            }
    result = {
        "status": "COMPLETE_KNOWN_NOTICE_DOCUMENTARY_ACCOUNTING_WITH_EXPLICIT_LIMITS",
        "completed_utc": datetime.now(UTC).isoformat(),
        "review_checkpoint_sha256": digest(cp),
        "review_ledger_pins": ledger_pins,
        "documents": rows,
        "counts": {
            "known_notice_urls": len(rows),
            "retained_archive_associations": association_count,
            "available_original_pdfs": sum("text_path" in d for d in source["documents"]),
            "unavailable_original_pdfs": sum(
                "text_path" not in d for d in source["documents"]
            ),
            "supported_release_dates": sum(
                row.get("release_date") is not None for row in rows
            ),
            "literal_purpose_labels": dict(
                Counter(p for row in rows for p in row["purposes"])
            ),
        },
        "source_history_admitted": False,
        "new_predictive_comparisons": 0,
        "cumulative_comparisons": 144,
        "limits": [
            "Bindings/counts checked here; documentary interpretations are the individual source reviews, not automatically proved by this consolidation.",
            "Actual body applicability is retained separately from API associations; retaining a link does not authenticate its applicability.",
            "Conditional and realized identities still need final result and original-lineage reconciliation during full history admission.",
            "One original PDF remains unavailable despite an indexed-text lead; no unavailable source is filled or treated as zero.",
            "Known linked notices do not establish that the index linked every relevant historical notice.",
        ],
    }
    path = rep / "TREASURY_FULL_NOTICE_REVIEW.json"
    with path.open("x") as out:
        out.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
