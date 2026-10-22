"""Execute the preregistered synthetic TiRex activation-swap pilot locally."""

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from src.tirex_intervention_adapter import TirexAdapter
from src.tirex_intervention_pilot import (
    FAMILIES,
    make_pairs,
    rank_candidates,
    select_controls,
    summarize_swaps,
)
from src.tirex_intervention_statistics import analyze_candidate, holm_adjust


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def utc():
    return datetime.now(UTC).isoformat()


def flatten(pairs):
    return np.stack((pairs["recipient"], pairs["donor"]), axis=1).reshape(-1, pairs["recipient"].shape[1])


def batched(adapter, contexts, size, patch=None, capture=True):
    outputs = []
    for start in range(0, len(contexts), size):
        end = min(start + size, len(contexts))
        local = None if patch is None else {**patch, "values": patch["values"][start:end]}
        outputs.append(adapter.forecast(contexts[start:end], patch=local, capture=capture))
    return {
        "quantiles": np.concatenate([x["quantiles"] for x in outputs]),
        "activations": {layer: np.concatenate([x["activations"][layer] for x in outputs])
                        for layer in adapter.layers} if capture else {},
        "residual": np.concatenate([x["residual"] for x in outputs]) if capture else None,
    }


def instrumentation(adapter, contexts, protocol, out):
    """Fail closed before candidate selection; retain all measured errors."""
    rules = protocol["instrumentation"]
    base = adapter.forecast(contexts)
    records = []
    raw = {"contexts": contexts, "base_quantiles": base["quantiles"], "base_residual": base["residual"]}
    raw.update({f"base_layer_{layer}": values for layer, values in base["activations"].items()})

    def compare(name, result, expected=base, forecast_scale=1.0, forecast_offset=0.0):
        qerror = float(np.max(np.abs(result["quantiles"] - (forecast_scale * expected["quantiles"] + forecast_offset))))
        act_errors = {}
        if result["activations"]:
            for layer in adapter.layers:
                reference = expected["activations"][layer]
                act_errors[str(layer)] = float(np.max(np.abs(result["activations"][layer] - reference)) / max(1.0, float(np.max(np.abs(reference)))))
            reference = expected["residual"]
            act_errors["residual"] = float(np.max(np.abs(result["residual"] - reference)) / max(1.0, float(np.max(np.abs(reference)))))
        passed = qerror <= rules["max_forecast_abs_error"] and all(
            value <= rules["max_activation_relative_error"] for value in act_errors.values())
        records.append({"check": name, "max_forecast_abs_error": qerror,
                        "activation_relative_errors": act_errors, "passed": passed})
        raw[name + "_quantiles"] = result["quantiles"]
        raw.update({f"{name}_layer_{layer}": values for layer, values in result["activations"].items()})
        if result.get("residual") is not None:
            raw[name + "_residual"] = result["residual"]

    # Compare to the public wrapper with NO adapter hooks registered.
    import torch
    items = [adapter._timeseries_type(target=torch.tensor(row, dtype=torch.float32)[None],
             past_covariates=None, future_covariates=None) for row in contexts]
    with torch.no_grad():
        direct = adapter.model.forecast(items, prediction_length=1, output_type="torch",
            batch_size=len(items), yield_per_batch=False, tta_sign_flip=False, tta_diff=False)
    direct_q = np.stack([x.detach().cpu().numpy()[0, adapter._quantile_indices, 0] for x in direct])
    compare("unhooked", {"quantiles": direct_q, "activations": {}})
    for layer in adapter.layers:
        result = adapter.forecast(contexts, patch={"layer": layer, "unit": 0,
                                   "values": base["activations"][layer][:, 0]})
        compare(f"self_swap_layer{layer}", result)
    reversed_output = adapter.forecast(contexts[::-1].copy())
    reversed_output["quantiles"] = reversed_output["quantiles"][::-1]
    reversed_output["activations"] = {k: v[::-1] for k, v in reversed_output["activations"].items()}
    reversed_output["residual"] = reversed_output["residual"][::-1]
    compare("batch_reversal", reversed_output)
    compare("singleton_vs_batch", batched(adapter, contexts, 1))
    for index, (scale, offset) in enumerate(rules["positive_affine_transforms"]):
        result = adapter.forecast(scale * contexts + offset)
        compare(f"affine_{index}", result, forecast_scale=scale, forecast_offset=offset)
    crossings = int(np.any(np.diff(base["quantiles"], axis=1) < 0, axis=1).sum())
    report = {"checks": records, "base_quantile_crossings": crossings,
              "passed": all(x["passed"] for x in records) and crossings == 0,
              "timestamp": utc()}
    save_json(out / "instrumentation.json", report)
    np.savez_compressed(out / "instrumentation.npz", **raw)
    if not report["passed"]:
        raise RuntimeError("Instrumentation gate failed; no candidates selected. See instrumentation.json")
    return report


def save_baseline(path, pairs, output):
    values = {key: value for key, value in pairs.items() if key != "split"}
    values["quantiles"] = output["quantiles"].reshape(-1, 2, 3)
    values["residual"] = output["residual"].reshape(-1, 2, output["residual"].shape[-1])
    for layer, array in output["activations"].items():
        values[f"layer_{layer}"] = array.reshape(-1, 2, array.shape[-1])
    np.savez_compressed(path, **values)


def run(protocol_path, output_path):
    protocol = json.loads(protocol_path.read_text())
    out = output_path
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "protocol.json", protocol)
    root = Path(__file__).resolve().parent.parent
    sources = list((root / "src").glob("tirex_intervention_*.py")) + list((root / "tests").glob("test_tirex_intervention_*.py"))
    frozen = {"timestamp_before_model_load": utc(), "protocol_sha256": sha(protocol_path),
              "source_hashes": {str(p.relative_to(root)): sha(p) for p in sources}}
    save_json(out / "pre_execution.json", frozen)
    start = time.monotonic()
    print("Loading pinned cached checkpoint; protocol and source hashes frozen.", flush=True)
    adapter = TirexAdapter(checkpoint_revision=protocol["checkpoint_revision"],
                           layers=tuple(protocol["layers"]), threads=4).load()
    import tirex2
    package = Path(tirex2.__file__).parent
    adapter.metadata["installed_source_hashes"] = {str(p.relative_to(package)): sha(p) for p in sorted(package.rglob("*.py"))}
    save_json(out / "model_metadata.json", adapter.metadata)
    np.savez_compressed(out / "weight_norms.npz", **{f"layer_{k}": v for k, v in adapter.weight_norms.items()})
    discovery = make_pairs("discovery", protocol["discovery_pairs_per_family"], protocol["context_length"])
    contexts = flatten(discovery)
    n = protocol["discovery_pairs_per_family"]
    gate_rows = np.array([0, 1, 2*n, 2*n+1, 4*n, 4*n+1])
    instrumentation(adapter, contexts[gate_rows], protocol, out)
    print(f"Instrumentation passed after {time.monotonic()-start:.1f}s. Starting discovery.", flush=True)
    base_d = batched(adapter, contexts, protocol["batch_size"])
    save_baseline(out / "discovery.npz", discovery, base_d)
    acts = {k: v.reshape(-1, 2, v.shape[-1]).astype(np.float64) for k, v in base_d["activations"].items()}
    candidates = rank_candidates(discovery, acts, adapter.weight_norms)
    if len(candidates) != 3:
        raise RuntimeError("Expected three nondegenerate discovery candidates")
    for candidate in candidates:
        layer, unit = candidate["layer"], candidate["unit"]
        candidate["controls"] = select_controls(candidate, acts, adapter.weight_norms)
        mask = discovery["family"] == candidate["family"]
        pair_indices = np.flatnonzero(mask)
        rows = np.column_stack((2*pair_indices, 2*pair_indices+1)).ravel()
        donor_values = acts[layer][mask, ::-1, unit].reshape(-1)
        result = batched(adapter, contexts[rows], protocol["batch_size"],
                         patch={"layer": layer, "unit": unit, "values": donor_values}, capture=False)
        baseline = base_d["quantiles"][rows].reshape(-1, 2, 3)
        swapped = result["quantiles"].reshape(-1, 2, 3)
        delta = swapped[:, :, 1] - baseline[:, :, 1]
        mean = float(np.mean((delta[:, 0] - delta[:, 1]) / 2))
        candidate["forecast_direction"] = 1 if mean >= 0 else -1
        candidate["discovery_mean_bidirectional_shift"] = mean
        candidate["control_matching"] = [{"unit": other,
            "activation_scale": float(acts[layer][:, :, other].std()),
            "outgoing_norm": float(adapter.weight_norms[layer][other])} for other in candidate["controls"]]
        candidate["outgoing_norm"] = float(adapter.weight_norms[layer][unit])
        np.savez_compressed(out / f"discovery_swap_L{layer}_U{unit}.npz", pair_id=discovery["pair_id"][mask], quantiles=swapped)
    lock = {"timestamp_before_validation": utc(), "candidates": candidates,
            "discovery_sha256": sha(out / "discovery.npz"), "protocol_sha256": sha(out / "protocol.json")}
    save_json(out / "candidate_lock.json", lock)
    lock_sha = sha(out / "candidate_lock.json")
    print("Candidate identities, controls, and effect directions locked: " + str([(x["layer"], x["unit"]) for x in candidates]), flush=True)
    validation = make_pairs("validation", protocol["validation_pairs_per_family"], protocol["context_length"])
    contexts_v = flatten(validation)
    base_v = batched(adapter, contexts_v, protocol["batch_size"])
    save_baseline(out / "validation.npz", validation, base_v)
    interventions = sorted({(c["layer"], u) for c in candidates for u in [c["unit"], *c["controls"]]})
    swap_results = {}
    for index, (layer, unit) in enumerate(interventions):
        donor_values = base_v["activations"][layer].reshape(-1, 2, 1408)[:, ::-1, unit].reshape(-1)
        result = batched(adapter, contexts_v, protocol["batch_size"],
                         patch={"layer": layer, "unit": unit, "values": donor_values}, capture=False)
        quantiles = result["quantiles"].reshape(-1, 2, 3)
        swap_results[layer, unit] = quantiles
        np.savez_compressed(out / f"validation_swap_L{layer}_U{unit}.npz", pair_id=validation["pair_id"], quantiles=quantiles)
        progress = {"completed_interventions": index+1, "total_interventions": len(interventions),
                    "layer": layer, "unit": unit, "elapsed_seconds": time.monotonic()-start, "timestamp": utc()}
        save_json(out / "progress.json", progress)
        print(json.dumps(progress), flush=True)
    baseline = base_v["quantiles"].reshape(-1, 2, 3)
    results, raw_p = [], []
    for index, candidate in enumerate(candidates):
        layer, unit = candidate["layer"], candidate["unit"]
        swapped = swap_results[layer, unit]
        controls = np.stack([swap_results[layer, other] for other in candidate["controls"]])
        mask = validation["family"] == candidate["family"]
        primary = analyze_candidate(baseline[mask], swapped[mask], controls[:, mask],
            candidate["forecast_direction"], protocol["inference"]["floor_log_variance"],
            protocol["seeds"]["bootstrap"] + index)
        raw_p.extend([primary["influence_pvalue"], primary["advantage_pvalue"]])
        descriptions = {}
        for family in FAMILIES:
            for variant in ("all", "short", "long"):
                sub = (validation["family"] == family)
                if variant != "all":
                    sub &= validation["variant"] == variant
                if not sub.any():
                    continue
                description = summarize_swaps(baseline[sub], swapped[sub])
                description["n_pairs"] = int(sub.sum())
                change = swapped[sub, :, 1] - baseline[sub, :, 1]
                description["frozen_direction_mean_shift"] = float(candidate["forecast_direction"] * np.mean((change[:, 0]-change[:, 1])/2))
                descriptions[f"{family}/{variant}"] = description
        results.append({"candidate": candidate, "primary": primary, "descriptive_by_family_variant": descriptions})
    corrected = holm_adjust(raw_p)
    for index, result in enumerate(results):
        primary = result["primary"]
        primary["influence_holm_p"] = corrected[2*index]
        primary["advantage_holm_p"] = corrected[2*index+1]
        ordered = primary["crossing_count"] + primary["control_crossing_count"] + primary["base_crossing_count"] == 0
        primary["influence_pass"] = ordered and corrected[2*index] < 0.05
        primary["advantage_pass"] = ordered and corrected[2*index+1] < 0.05
    if sha(out / "candidate_lock.json") != lock_sha:
        raise RuntimeError("Candidate lock changed during validation")
    summary = {"finished_utc": utc(), "elapsed_seconds": time.monotonic()-start,
               "candidate_lock_sha256": lock_sha, "results": results,
               "limitations": protocol["inference"]["interpretation"]}
    save_json(out / "summary.json", summary)
    manifest = {p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != "manifest.json"}
    save_json(out / "manifest.json", manifest)
    print("Pilot complete: " + str(out / "summary.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing output is never overwritten")
    args = parser.parse_args()
    run(args.protocol, args.output)


if __name__ == "__main__":
    main()
