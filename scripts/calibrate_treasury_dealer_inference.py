"""Fixed generated-null calibration, with no historical source/data access."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from src.treasury_dealer_inference import masked_mean_inference


def main():
    root = Path(__file__).resolve().parents[1]
    rep = root / "reports/treasury_dealer"
    output = rep / "INFERENCE_NULL_CALIBRATION.json"
    if output.exists():
        raise ValueError("Preserve the first fixed null calibration")
    paths = [
        "src/treasury_dealer_inference.py",
        "tests/test_treasury_dealer_inference.py",
        "scripts/calibrate_treasury_dealer_inference.py",
        "treasury_dealer.yaml",
    ]
    pins = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths}
    config = {
        "sessions": 1500,
        "rho": 0.8,
        "burn_in": 200,
        "repetitions": 200,
        "draws": 499,
        "seed": 20261002,
        "blocks": [21, 63, 126],
        "hac_lags": 126,
        "active_residues_mod21": [0, 1, 7, 8, 14, 15],
        "missing_positions_half_open": [600, 726],
        "minimum_envelope_coverage": 0.90,
        "inference_seed_rule": "seed+10000+2*repetition+endpoint_index; internal per-block seed adds block length",
    }
    rng = np.random.default_rng(config["seed"])
    daily = np.ones(config["sessions"], dtype=bool)
    daily[600:726] = False
    active = daily & np.isin(
        np.arange(config["sessions"]) % 21, config["active_residues_mod21"]
    )
    rows = []
    for repetition in range(config["repetitions"]):
        innovation = rng.normal(
            0, np.sqrt(1 - config["rho"] ** 2), config["sessions"] + config["burn_in"]
        )
        series = np.zeros(len(innovation))
        for position in range(1, len(series)):
            series[position] = config["rho"] * series[position - 1] + innovation[position]
        differences = series[config["burn_in"] :]
        for endpoint_index, (endpoint, mask) in enumerate(
            (("daily", daily), ("active", active))
        ):
            row = {"repetition": repetition, "endpoint": endpoint, "n": int(mask.sum())}
            try:
                result = masked_mean_inference(
                    differences,
                    mask,
                    blocks=tuple(config["blocks"]),
                    hac_lags=config["hac_lags"],
                    draws=config["draws"],
                    seed=config["seed"] + 10000 + 2 * repetition + endpoint_index,
                )
                methods = {"hac": result["hac"], **result["block_inference"]}
                low = min(method["ci95"][0] for method in methods.values())
                high = max(method["ci95"][1] for method in methods.values())
                row.update(
                    status="COMPLETED",
                    envelope_covers_zero=low <= 0 <= high,
                    method_covers_zero={
                        key: value["ci95"][0] <= 0 <= value["ci95"][1]
                        for key, value in methods.items()
                    },
                    envelope=[low, high],
                    result=result,
                )
            except ValueError as error:
                # A failed replicate counts against coverage; it is never retried.
                row.update(status="UNEVALUABLE", error=str(error), envelope_covers_zero=False)
            rows.append(row)
        if (repetition + 1) % 50 == 0:
            print(f"Generated-null repetitions completed: {repetition + 1}/200", flush=True)
    summaries = {}
    for endpoint in ("daily", "active"):
        selected = [row for row in rows if row["endpoint"] == endpoint]
        summaries[endpoint] = {
            "envelope_coverage": sum(row["envelope_covers_zero"] for row in selected)
            / len(selected),
            "failures": sum(row["status"] != "COMPLETED" for row in selected),
            "individual_method_coverage": {
                key: sum(row.get("method_covers_zero", {}).get(key, False) for row in selected)
                / len(selected)
                for key in ("hac", "21", "63", "126")
            },
        }
    passed = all(
        x["failures"] == 0 and x["envelope_coverage"] >= 0.90 for x in summaries.values()
    )
    for name, expected in pins.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Calibration code/protocol changed during the fixed run")
    result = {
        "status": "SYNTHETIC_CALIBRATION_PASS" if passed else "SYNTHETIC_CALIBRATION_FAIL",
        "completed_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "pins": pins,
        "summaries": summaries,
        "replicates": rows,
        "historical_data_accessed": False,
        "predictive_comparisons_added": 0,
        "limitations": "Fixed generated null only; no claim of exact market-data coverage or multiplicity-adjusted power.",
    }
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "summaries")}, indent=2))


if __name__ == "__main__":
    main()
