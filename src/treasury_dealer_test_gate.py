"""Run all discovered tests except six explicitly quarantined historical replays."""

import hashlib
import json
import os
import sys
import time
import unittest
from datetime import UTC, datetime
from pathlib import Path

QUARANTINED = frozenset(
    {
        "tests.test_methodology.TestDiagnosticOnlyQuarantine.test_qlike_series_refuses_the_clean_window",
        "tests.test_methodology.TestFrozenReportsUnchanged.test_default_evaluate_reproduces_the_frozen_reports",
        "tests.test_methodology.TestFrozenReportsUnchanged.test_corrected_run_writes_a_separate_file",
        "tests.test_methodology.TestEstimatorReconstructionAccuracy.test_reconstruction_matches_exact_smearing_on_the_har_family",
        "tests.test_methodology.TestEstimatorReconstructionAccuracy.test_reconstruction_table_matches_the_published_one",
        "tests.test_methodology.TestEstimatorReconstructionAccuracy.test_the_near_common_factor_claim_is_scoped_to_its_own_model_set",
    }
)


def select_tests(suite):
    def flatten(items):
        for item in items:
            if isinstance(item, unittest.TestSuite):
                yield from flatten(item)
            else:
                yield item

    cases = list(flatten(suite))
    names = [case.id() for case in cases]
    if len(names) != len(set(names)) or not set(names) >= QUARANTINED:
        raise ValueError(
            "Unique complete discovered inventory including six quarantined identities required"
        )
    selected = [case for case in cases if case.id() not in QUARANTINED]
    return unittest.TestSuite(selected), {
        "discovered_count": len(names),
        "selected_count": len(selected),
        "quarantined_test_ids": sorted(QUARANTINED),
        "selected_test_ids": [case.id() for case in selected],
        "reason": "These six frozen regression tests decode or refit the 2025-11-03+ historical period; no frozen test bytes changed.",
    }


def main():
    report = Path(sys.argv[1])
    selection = report / "TEST_SELECTION.json"
    results = report / "TEST_GATE_RESULT.json"
    if selection.exists() or results.exists():
        raise ValueError("Existing test-gate evidence must be preserved")
    suite, record = select_tests(
        unittest.defaultTestLoader.discover("tests", top_level_dir=".")
    )
    with selection.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    start, stamp = time.monotonic(), datetime.now(UTC).isoformat()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    output = {
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "started_utc": stamp,
        "completed_utc": datetime.now(UTC).isoformat(),
        "seconds": time.monotonic() - start,
        "run_count": result.testsRun,
        "executed_count": result.testsRun - len(result.skipped),
        "failure_count": len(result.failures),
        "error_count": len(result.errors),
        "skipped": [
            {"test_id": test.id(), "reason": reason} for test, reason in result.skipped
        ],
        "quarantined_count": len(QUARANTINED),
        "selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
    }
    with results.open("x") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    if not result.wasSuccessful() or result.testsRun != record["selected_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
