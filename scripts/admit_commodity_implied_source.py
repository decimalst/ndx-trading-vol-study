"""One-time source-only checkpoint; never builds features, cohorts or forecasts."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from src.commodity_implied_source import parse_cboe_history
from src.verify_commodity_implied_source import verify_history

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/commodity_implied"
SOURCE_HASHES = {
    "OVX": "c68b3fcf25ecc3fd13eaa2595ac179b4f753315d0b51e1366acb2c3a2adf6079",
    "GVZ": "7a2acc2e858d48fc02b7e4925f88df0716fba0a69e7d64c7d94039ed49771fd3",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value, private=False):
    with path.open("x") as handle:
        if private:
            path.chmod(0o600)
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def now():
    return datetime.now(UTC).isoformat()


def main():
    checkpoint = REPORT / "SOURCE_PRE_ADMISSION.json"
    certificate = REPORT / "SOURCE_ADMISSION.json"
    failure = REPORT / "SOURCE_ADMISSION_FAILURE.json"
    private = ROOT / "data/commodity_implied"
    outputs = [private / f"{symbol}_parsed.json" for symbol in SOURCE_HASHES]
    if any(path.exists() for path in [checkpoint, certificate, failure, *outputs]):
        raise RuntimeError("Source attempt already exists; do not overwrite or repeat")

    freeze_path = ROOT / "reports/claims_release/predictive/freeze_record.json"
    publication_path = ROOT / "reports/claims_release/predictive/publication_audit.json"
    if sha(freeze_path) != "e9571e94246a2c8714c19d936034b49cbf4a7f88c4a551425cc7103c544a772a":
        raise RuntimeError("Prior claims freeze changed")
    if (
        sha(publication_path)
        != "57bc07a84501e223178bfbb2e72998c92eedfd835e6976643b2df43ce159e13f"
    ):
        raise RuntimeError("Prior claims publication changed")
    freeze = json.loads(freeze_path.read_text())
    publication = json.loads(publication_path.read_text())
    preservation = {
        **freeze["code"],
        **freeze["inputs"],
        **freeze["preserved"],
        **freeze["prefit"],
        **{
            str((publication_path.parent / name).relative_to(ROOT)): signature
            for name, signature in publication["report_artifact_hashes"].items()
        },
    }
    for name, signature in preservation.items():
        if sha(ROOT / name) != signature:
            raise RuntimeError(f"Preserved artifact changed: {name}")

    precheck = REPORT / "SOURCE_COMPONENTS_PRE_ADMISSION.log"
    log = precheck.read_text()
    if "Ran 39 tests in " not in log or log.rstrip().splitlines()[-1] != "OK":
        raise RuntimeError("Required 39 source contracts did not pass")
    component_paths = [
        "src/commodity_implied_source.py",
        "src/verify_commodity_implied_source.py",
        "tests/test_commodity_implied_source.py",
        "tests/test_verify_commodity_implied_source.py",
        "tests/test_commodity_implied_source_integration.py",
        "scripts/admit_commodity_implied_source.py",
    ]
    evidence_paths = [
        "reports/commodity_implied/SOURCE_PARSER_DESIGN.md",
        "reports/commodity_implied/SOURCE_VERIFICATION_DESIGN.md",
        "reports/commodity_implied/SOURCE_COMPONENTS_PRE_ADMISSION.log",
        "reports/commodity_implied/SOURCE_PARSER_RED.log",
        "reports/commodity_implied/SOURCE_VERIFICATION_RED.log",
        "reports/commodity_implied/DISCOVERY_AND_ADMISSION_PLAN.md",
        "reports/commodity_implied/SOURCE_FEASIBILITY.md",
        "reports/commodity_implied/NOVELTY_AND_CONTROLS.md",
        "reports/commodity_implied/PROSPECTIVE_DESIGN_REVIEW.md",
    ]
    acquisition_path = private / "source_metadata/acquisition_ledger.json"
    ledger = json.loads(acquisition_path.read_text())
    receipts = {}
    acquisition_hashes = {str(acquisition_path.relative_to(ROOT)): sha(acquisition_path)}
    for symbol, signature in SOURCE_HASHES.items():
        receipt_path = private / f"source_metadata/{symbol}_History.receipt.json"
        receipt = json.loads(receipt_path.read_text())
        url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv"
        metadata = receipt["curl_metadata"]
        if (
            receipt not in ledger
            or receipt["symbol"] != symbol
            or receipt["request_url"] != url
            or metadata["url_effective"] != url
            or receipt["curl_returncode"] != 0
            or metadata["http_code"] != 200
            or metadata["ssl_verify_result"] != 0
            or receipt["numeric_values_decoded"] is not False
            or receipt["body_sha256"] != signature
        ):
            raise RuntimeError("Original HTTPS acquisition receipt is inconsistent")
        for key in ("body", "headers"):
            relative = receipt[f"{key}_path"]
            if sha(ROOT / relative) != receipt[f"{key}_sha256"]:
                raise RuntimeError("Original acquisition bytes changed")
            acquisition_hashes[relative] = receipt[f"{key}_sha256"]
        acquisition_hashes[str(receipt_path.relative_to(ROOT))] = sha(receipt_path)
        receipts[symbol] = receipt

    code = {name: sha(ROOT / name) for name in component_paths}
    evidence = {name: sha(ROOT / name) for name in evidence_paths}
    save(
        checkpoint,
        {
            "status": "FROZEN_SOURCE_CHECK_BEFORE_NUMERICAL_ADMISSION",
            "created_utc": now(),
            "source_start": "2009-01-02",
            "source_end": "2025-10-20",
            "source_values_admitted": False,
            "predictive_family": 142,
            "source_contract_tests": 39,
            "source_components": code,
            "evidence": evidence,
            "acquisition": acquisition_hashes,
            "prior_freeze_sha256": sha(freeze_path),
            "prior_publication_sha256": sha(publication_path),
            "preserved_artifacts_checked": len(preservation),
        },
    )
    results = {}
    try:
        for symbol, signature in SOURCE_HASHES.items():
            payload = (ROOT / receipts[symbol]["body_path"]).read_bytes()
            parsed = parse_cboe_history(payload, symbol, signature)
            verified = verify_history(payload, symbol, signature, parsed)
            # Exercise the exact persisted representation before saving any series.
            decoded = json.loads(json.dumps(parsed, allow_nan=False))
            if verify_history(payload, symbol, signature, decoded) != verified:
                raise RuntimeError("Serialized bounded history failed verification")
            results[symbol] = (parsed, verified)
        for name, signature in {
            **preservation,
            **code,
            **evidence,
            **acquisition_hashes,
        }.items():
            if sha(ROOT / name) != signature:
                raise RuntimeError(f"Pinned artifact changed during admission: {name}")
    except Exception as error:
        save(
            failure,
            {
                "status": "SOURCE_ADMISSION_FAILED_NO_FORECAST_ATTEMPT",
                "created_utc": now(),
                "source_pre_admission_sha256": sha(checkpoint),
                "error_type": type(error).__name__,
                "error": str(error),
                "predictive_family": 142,
            },
        )
        raise

    proofs = {}
    for symbol, (parsed, verified) in results.items():
        path = private / f"{symbol}_parsed.json"
        save(path, parsed, private=True)
        proofs[symbol] = {
            "verification": verified,
            "source_sha256": SOURCE_HASHES[symbol],
            "parsed_path": str(path.relative_to(ROOT)),
            "parsed_sha256": sha(path),
            "date_and_missingness_metadata": parsed["metadata"],
        }
    save(
        certificate,
        {
            "status": "BOUNDED_SOURCES_INDEPENDENTLY_VERIFIED_NO_FORECAST_ATTEMPT",
            "created_utc": now(),
            "source_pre_admission_sha256": sha(checkpoint),
            "source_start": "2009-01-02",
            "source_end": "2025-10-20",
            "sources": proofs,
            "predictive_family": 142,
            "features_cohorts_fits_or_scores_computed": False,
            "vintage_limitation": "Current Cboe captures; original immutable historical vintages unverified.",
            "outside_scope_numerical_values_converted": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "SOURCE_ADMISSION_COMPLETE",
                "certificate_sha256": sha(certificate),
                "sources": {symbol: pair[1] for symbol, pair in results.items()},
                "predictive_family": 142,
            }
        )
    )


if __name__ == "__main__":
    main()
