"""Audit saved group interventions without importing producers or loading a model."""

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

FAMILIES = ("persistence", "recency", "recovery")
CHECKS = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(name, condition):
    CHECKS.append({"check": name, "passed": bool(condition)})
    if not condition:
        raise AssertionError(name)


def close(name, actual, expected, atol=2e-10, rtol=1e-8):
    check(name, np.allclose(actual, expected, atol=atol, rtol=rtol))


def read_json(path):
    return json.loads(path.read_text())


def crossings(values):
    return int(np.any(np.diff(values, axis=-1) < 0, axis=-1).sum())


def binomial_tail(successes, n):
    return sum(math.comb(n, k) for k in range(successes, n + 1)) / 2**n


def descriptions(base, swapped, activation_contrasts, direction):
    change = swapped[:, :, 1] - base[:, :, 1]
    widths = (swapped[:, :, 2] - swapped[:, :, 0]) - (base[:, :, 2] - base[:, :, 0])
    return {
        "mean_signed_shift": float(direction * np.mean((change[:, 0] - change[:, 1]) / 2)),
        "mean_abs_median_shift": float(abs(change).mean()),
        "mean_recipient_shift": float(change[:, 0].mean()),
        "mean_donor_shift": float(change[:, 1].mean()),
        "mean_native_contrast": float(np.mean(base[:, 1, 1] - base[:, 0, 1])),
        "both_directions_expected_sign_fraction": float(np.mean((direction * change[:, 0] > 0) & (-direction * change[:, 1] > 0))),
        "mean_abs_width_shift": float(abs(widths).mean()),
        "n_pairs": len(base),
        "activation_oriented_mean": float(activation_contrasts.mean()),
        "activation_consistency_fraction": float(np.mean(activation_contrasts.mean(axis=1) > 0)),
        "all_member_signs_fraction": float(np.mean(np.all(activation_contrasts > 0, axis=1))),
    }


def verify(args):
    run, source, prior = args.run_dir.resolve(), args.source_root.resolve(), args.prior_run.resolve()
    protocol = read_json(run / "protocol.json")
    frozen = read_json(run / "pre_execution.json")
    lock = read_json(run / "group_lock.json")
    summary = read_json(run / "summary.json")
    metadata = read_json(run / "model_metadata.json")
    previous_meta = read_json(prior / "model_metadata.json")
    check("protocol pre-execution hash", sha(run / "protocol.json") == frozen["protocol_sha256"])
    for name, digest in frozen["source_hashes"].items():
        check("frozen source " + name, sha(source / name) == digest)
    previous = read_json(prior / "pre_execution.json")
    for name, digest in previous["source_hashes"].items():
        if name.startswith("src/") and (source / name).exists():
            check("unchanged dependency from scalar pilot " + name, sha(source / name) == digest)
    check("group lock discovery hash", sha(run / "discovery.npz") == lock["discovery_sha256"])
    check("group lock protocol hash", sha(run / "protocol.json") == lock["protocol_sha256"])
    check("summary group lock hash", sha(run / "group_lock.json") == summary["group_lock_sha256"])
    for name, digest in lock["prior_run_hashes"].items():
        check("prior run input receipt " + name, sha(prior / name) == digest)
    for field in ("checkpoint_revision", "config_sha256", "weights_sha256", "installed_source_hashes"):
        check("same pinned checkpoint and implementation " + field, metadata[field] == previous_meta[field])
    check("intended checkpoint", metadata["checkpoint_revision"] == protocol["checkpoint_revision"])
    check("token63 maps to first predicted step", (metadata["token_count"] * 32 - 32 - 320) // 32 == protocol["token_index"] == 63)
    check("native model version and execution settings", metadata["versions"]["tirex-2"] == "0.2.1"
          and metadata["device"] == "cpu" and not metadata["tta_diff"] and not metadata["tta_sign_flip"])

    d = np.load(run / "discovery.npz", allow_pickle=False)
    v = np.load(run / "validation.npz", allow_pickle=False)
    norms = np.load(run / "weight_norms.npz", allow_pickle=False)
    context_hash_sets = []
    variant_counts = {}
    for split, data, n in (("discovery", d, 72), ("validation", v, 96)):
        a, b = data["recipient"], data["donor"]
        check(split + " shape", a.shape == b.shape == (n, 2048))
        check(split + " finite", np.isfinite(a).all() and np.isfinite(b).all())
        check(split + " multiset", np.array_equal(np.sort(a, axis=1), np.sort(b, axis=1)))
        check(split + " terminal value", np.array_equal(a[:, -1], b[:, -1]))
        close(split + " common mean", np.concatenate([a, b]).mean(axis=1), -9.)
        close(split + " common population SD", np.concatenate([a, b]).std(axis=1), .6)
        check(split + " distinct pair IDs", len(set(data["pair_id"])) == n)
        hashes = {hashlib.sha256(row.tobytes()).hexdigest() for row in np.concatenate([a, b])}
        check(split + " unique contexts", len(hashes) == 2 * n)
        context_hash_sets.append(hashes)
        check(split + " finite ordered baseline forecasts", np.isfinite(data["quantiles"]).all() and crossings(data["quantiles"]) == 0)
        for family in FAMILIES:
            mask = data["family"] == family
            check(split + " family balance " + family, mask.sum() == n // 3)
            if split == "validation":
                variant_counts[family] = {kind: int(np.sum(mask & (data["variant"] == kind))) for kind in ("short", "long")}
    check("fresh discovery and validation contexts", not context_hash_sets[0] & context_hash_sets[1])
    check("fresh discovery and validation IDs", not set(d["pair_id"]) & set(v["pair_id"]))
    prior_hashes = set()
    for split in ("discovery", "validation"):
        old = np.load(prior / (split + ".npz"), allow_pickle=False)
        for key in ("recipient", "donor"):
            prior_hashes.update(hashlib.sha256(row.tobytes()).hexdigest() for row in old[key])
    check("no old-pilot discovery or validation context reused", not set.union(*context_hash_sets) & prior_hashes)

    # Recompute all saved scalar and group instrumentation errors from arrays.
    gate = read_json(run / "instrumentation.json")
    gate_raw = np.load(run / "instrumentation.npz", allow_pickle=False)
    group_gate = read_json(run / "group_instrumentation.json")
    group_raw = np.load(run / "group_instrumentation.npz", allow_pickle=False)
    expected_contexts = np.stack((d["recipient"], d["donor"]), axis=1).reshape(-1, 2048)[[0, 1, 48, 49, 96, 97]]
    check("base gate contexts represent each new family", np.array_equal(gate_raw["contexts"], expected_contexts))
    check("group gate contexts represent each new family", np.array_equal(group_raw["contexts"], expected_contexts))
    max_qerror = max_activation_error = 0.
    for record in gate["checks"]:
        name = record["check"]
        scale, offset = 1., 0.
        if name.startswith("affine_"):
            scale, offset = protocol["instrumentation"]["positive_affine_transforms"][int(name[-1])]
        error = float(np.max(abs(gate_raw[name + "_quantiles"] - (scale * gate_raw["base_quantiles"] + offset))))
        close("base gate reported forecast error " + name, error, record["max_forecast_abs_error"], atol=1e-14)
        check("base gate forecast tolerance " + name, error <= protocol["instrumentation"]["max_forecast_abs_error"])
        max_qerror = max(max_qerror, error)
        for layer, reported in record["activation_relative_errors"].items():
            suffix = "residual" if layer == "residual" else "layer_" + layer
            a, b = gate_raw[name + "_" + suffix], gate_raw["base_" + suffix]
            error = float(np.max(abs(a - b)) / max(1., float(np.max(abs(b)))))
            close("base gate reported activation error " + name + "/" + layer, error, reported, atol=1e-14)
            check("base gate activation tolerance " + name + "/" + layer, error <= protocol["instrumentation"]["max_activation_relative_error"])
            max_activation_error = max(max_activation_error, error)
    check("base instrumentation declared pass", gate["passed"] and crossings(gate_raw["base_quantiles"]) == 0)
    names = [f"{name}_L{layer}" for layer in protocol["layers"]
             for name in ("self_group", "one_equals_scalar", "member_permutation", "patched_singleton_batches", "cleanup")]
    check("all 15 group instrumentation checks present", [item["check"] for item in group_gate["checks"]] == names)
    for record in group_gate["checks"]:
        name = record["check"]
        error = float(np.max(abs(group_raw[name + "_actual_quantiles"] - group_raw[name + "_expected_quantiles"])))
        close("group gate reported forecast error " + name, error, record["max_forecast_abs_error"], atol=1e-14)
        check("group gate forecast tolerance " + name, error <= protocol["instrumentation"]["max_forecast_abs_error"])
        max_qerror = max(max_qerror, error)
        for layer, reported in record["activation_relative_errors"].items():
            suffix = "residual" if layer == "residual" else "layer_" + layer
            a, b = group_raw[name + "_actual_" + suffix], group_raw[name + "_expected_" + suffix]
            error = float(np.max(abs(a - b)) / max(1., float(np.max(abs(b)))))
            close("group gate reported activation error " + name + "/" + layer, error, reported, atol=1e-14)
            check("group gate activation tolerance " + name + "/" + layer, error <= protocol["instrumentation"]["max_activation_relative_error"])
            max_activation_error = max(max_activation_error, error)
        check("group gate declared pass " + name, record["passed"])
    check("group instrumentation declared pass", group_gate["passed"])

    groups = lock["groups"]
    check("complete group order", [g["group_id"] for g in groups] == [f"{f}_k{k}" for f in FAMILIES for k in (4, 8)])
    for family_index, family in enumerate(FAMILIES):
        layer = protocol["family_layers"][family]
        act = d[f"layer_{layer}"].astype(float)
        weight_norm = norms[f"layer_{layer}"].astype(float)
        sd = np.std(act, axis=(0, 1))
        paired = act[d["family"] == family, 1] - act[d["family"] == family, 0]
        difference_rms = np.sqrt(np.mean(paired * paired, axis=0))
        contrast = paired.mean(axis=0)
        valid = [j for j in range(act.shape[-1]) if sd[j] > 0 and weight_norm[j] > 0 and difference_rms[j] > 0]
        top = sorted(valid, key=lambda j: (-abs(contrast[j] / sd[j]), j))[:8]
        used = set(top)
        features = np.column_stack((sd, weight_norm, difference_rms))
        assigned, match_details = [], []
        for _ in range(4):
            remaining = [j for j in valid if j not in used]
            ratios = features[remaining][None, :, :] / features[top][:, None, :]
            costs = np.sum(np.log(ratios)**2, axis=2)
            rows, columns = linear_sum_assignment(costs)
            check(f"{family} complete Hungarian assignment", np.array_equal(rows, np.arange(8)))
            controls = [remaining[column] for column in columns]
            assigned.append(controls)
            match_details.append([(features[j] / features[u], costs[i, columns[i]]) for i, (u, j) in enumerate(zip(top, controls, strict=True))])
            used.update(controls)
        check(f"{family} disjoint candidate/control8 pools", len(used) == 40)
        for offset, size in enumerate((4, 8)):
            group = groups[family_index * 2 + offset]
            gid = group["group_id"]
            check(gid + " fixed family layer size", (group["family"], group["layer"], group["size"]) == (family, layer, size))
            check(gid + " activation-ranked members", group["units"] == top[:size])
            check(gid + " matched controls", group["control_groups"] == [row[:size] for row in assigned])
            for member, j in zip(group["members"], top[:size], strict=True):
                check(gid + f" member {j} identity/orientation", member["unit"] == j and member["orientation"] == (1 if contrast[j] >= 0 else -1))
                for field, value in (("activation_scale", sd[j]), ("outgoing_norm", weight_norm[j]),
                                     ("rms_difference", difference_rms[j]), ("standardized_effect", contrast[j] / sd[j])):
                    close(gid + f" member {j} " + field, member[field], value)
            for k, details in enumerate(group["control_matching"]):
                for slot, item in enumerate(details):
                    ratio, cost = match_details[k][slot]
                    check(gid + f" control{k}/slot{slot} IDs", (item["candidate_unit"], item["control_unit"]) == (top[slot], assigned[k][slot]))
                    close(gid + f" control{k}/slot{slot} ratios", [item["sd_ratio"], item["norm_ratio"], item["rms_difference_ratio"]], ratio)
                    close(gid + f" control{k}/slot{slot} cost", item["squared_log_cost"], cost)
            swap = np.load(run / f"discovery_group_{gid}.npz", allow_pickle=False)
            mask = d["family"] == family
            check(gid + " discovery swap IDs", np.array_equal(swap["pair_id"], d["pair_id"][mask]))
            check(gid + " discovery swap unit order", list(swap["units"]) == group["units"])
            delta = swap["quantiles"].astype(float)[:, :, 1] - d["quantiles"][mask, :, 1].astype(float)
            effect = np.mean((delta[:, 0] - delta[:, 1]) / 2)
            close(gid + " discovery direction mean", group["discovery_mean_bidirectional_shift"], effect, atol=2e-9)
            check(gid + " discovery-frozen direction", group["forecast_direction"] == (1 if effect >= 0 else -1))

    raw_p, calculated = [], []
    check("six summary groups", len(summary["results"]) == 6)
    for group_index, result in enumerate(summary["results"]):
        group = groups[group_index]
        gid, family, size = group["group_id"], group["family"], group["size"]
        check(gid + " summary equals group lock", result["group"] == group)
        primary = result["primary"]
        mask = v["family"] == family
        all_base = v["quantiles"].astype(float)
        base = all_base[mask]

        def read_forecasts(filename, expected_ids, expected_units):
            data = np.load(run / filename, allow_pickle=False)
            check(filename + " ID alignment", np.array_equal(data["pair_id"], expected_ids))
            check(filename + " unit order", list(data["units"]) == expected_units)
            values = data["quantiles"].astype(float)
            check(filename + " shape/finite", values.shape == (len(expected_ids), 2, 3) and np.isfinite(values).all())
            return values

        all_joint = read_forecasts(f"validation_group_{gid}.npz", v["pair_id"], group["units"])
        joint = all_joint[mask]
        controls = np.array([read_forecasts(f"validation_control_{gid}_{j}.npz", v["pair_id"], group["control_groups"][j])[mask] for j in range(4)])
        singles = np.array([read_forecasts(f"validation_singleton_{family}_{j}.npz", v["pair_id"][mask], [j]) for j in group["units"]])
        delta = joint[:, :, 1] - base[:, :, 1]
        control_delta = controls[:, :, :, 1] - base[None, :, :, 1]
        single_delta = singles[:, :, :, 1] - base[None, :, :, 1]
        magnitude = abs(delta).mean(axis=1)
        single_magnitude = abs(single_delta).mean(axis=2)
        envelope = single_magnitude.max(axis=0)
        direction = group["forecast_direction"]
        effects = [direction * (delta[:, 0] - delta[:, 1]) / 2,
                   magnitude - abs(control_delta).mean(axis=(0, 2)), magnitude - envelope]
        check(gid + " counts/direction/floor", (primary["n_pairs"], primary["n_controls"], primary["n_singletons"], primary["direction"], primary["floor"])
              == (32, 4, size, direction, .001))
        check(gid + " bootstrap metadata", primary["seed"] == protocol["seeds"]["bootstrap"] + group_index and primary["bootstrap_replicates"] == 5000)
        sample = np.random.default_rng(protocol["seeds"]["bootstrap"] + group_index).integers(0, 32, (5000, 32))
        for label, values in zip(("influence", "advantage", "singleton_advantage"), effects, strict=True):
            close(gid + "/" + label + " values", values, primary[label + "_effects"])
            success = int(np.count_nonzero(values > .001))
            check(gid + "/" + label + " strict threshold count", success == primary[label + "_successes"])
            probability = binomial_tail(success, 32)
            close(gid + "/" + label + " exact binomial tail", probability, primary[label + "_pvalue"], atol=1e-15)
            raw_p.append(probability)
            mean_key = "singleton_advantage_mean" if label == "singleton_advantage" else "mean_" + label + "_effect"
            ci_key = "singleton_advantage_ci95" if label == "singleton_advantage" else label + "_mean_ci95"
            close(gid + "/" + label + " mean", np.mean(values), primary[mean_key])
            close(gid + "/" + label + " bootstrap", np.quantile(values[sample].mean(axis=1), [.025, .975]), primary[ci_key])
        widths = (joint[:, :, 2] - joint[:, :, 0]) - (base[:, :, 2] - base[:, :, 0])
        control_widths = (controls[:, :, :, 2] - controls[:, :, :, 0]) - (base[None, :, :, 2] - base[None, :, :, 0])
        residual = delta - single_delta.sum(axis=0)
        expected = {
            "mean_native_donor_recipient_contrast": np.mean(base[:, 1, 1] - base[:, 0, 1]),
            "mean_recipient_median_shift": delta[:, 0].mean(), "mean_donor_median_shift": delta[:, 1].mean(),
            "mean_abs_median_shift": abs(delta).mean(), "mean_abs_width_shift": abs(widths).mean(),
            "mean_abs_control_median_shift": abs(control_delta).mean(), "mean_abs_control_width_shift": abs(control_widths).mean(),
            "both_directions_expected_sign_fraction": np.mean((direction * delta[:, 0] > 0) & (-direction * delta[:, 1] > 0)),
            "base_crossing_count": crossings(base), "crossing_count": crossings(joint), "control_crossing_count": crossings(controls),
            "singleton_crossing_count": crossings(singles), "largest_singleton_mean_abs": single_magnitude.mean(axis=1).max(),
            "mean_pairwise_singleton_envelope": envelope.mean(), "joint_minus_sum_median": residual,
            "mean_abs_joint_minus_sum_median": abs(residual).mean(), "mean_abs_sum_singleton_median": abs(single_delta.sum(axis=0)).mean(),
        }
        for key, value in expected.items():
            close(gid + "/" + key, value, primary[key])
        activations = v[f'layer_{group["layer"]}'].astype(float)
        contrast_activations = activations[:, 1, group["units"]] - activations[:, 0, group["units"]]
        contrast_activations *= np.array([member["orientation"] / member["activation_scale"] for member in group["members"]])
        expected_group_keys = set()
        for other_family in FAMILIES:
            for variant in ("all", "short", "long"):
                selected = v["family"] == other_family
                if variant != "all":
                    selected &= v["variant"] == variant
                if not selected.any():
                    continue
                key = other_family + "/" + variant
                expected_group_keys.add(key)
                expected_desc = descriptions(all_base[selected], all_joint[selected], contrast_activations[selected], direction)
                actual = result["descriptive_by_family_variant"][key]
                check(gid + "/" + key + " full descriptive fields", set(expected_desc) == set(actual))
                for field, value in expected_desc.items():
                    if value is None:
                        check(gid + "/" + key + "/" + field, actual[field] is None)
                    else:
                        close(gid + "/" + key + "/" + field, value, actual[field], atol=2e-9, rtol=2e-7)
        check(gid + " complete descriptive group inventory", set(result["descriptive_by_family_variant"]) == expected_group_keys)
        calculated.append({"group_id": gid, "mean_effects": [float(x.mean()) for x in effects],
                           "successes": [int(np.sum(x > .001)) for x in effects],
                           "mean_abs_median_shift": float(abs(delta).mean()),
                           "mean_abs_width_shift": float(abs(widths).mean())})

    order = sorted(range(18), key=raw_p.__getitem__)
    adjusted = [0.] * 18
    current = 0.
    for rank, index in enumerate(order):
        current = max(current, min(1., raw_p[index] * (18 - rank)))
        adjusted[index] = current
    for i, result in enumerate(summary["results"]):
        primary = result["primary"]
        gid = result["group"]["group_id"]
        ordered = sum(primary[k] for k in ("base_crossing_count", "crossing_count", "control_crossing_count", "singleton_crossing_count")) == 0
        passing = []
        for j, label in enumerate(("influence", "advantage", "singleton")):
            close(gid + "/" + label + " Holm18", adjusted[3*i+j], primary[label + "_holm_p"])
            passed = ordered and adjusted[3*i+j] < protocol["inference"]["alpha"]
            check(gid + "/" + label + " pass flag", primary[label + "_pass"] == passed)
            passing.append(passed)
        check(gid + " qualified conjunction", primary["qualified"] == all(passing))
        calculated[i]["passes"] = passing
        calculated[i]["qualified"] = all(passing)

    times = [datetime.fromisoformat(item) for item in (frozen["timestamp_before_model_load"], gate["timestamp"],
             group_gate["timestamp"], lock["timestamp_before_validation"], summary["finished_utc"])]
    check("recorded freeze/gates/lock/finish chronology", all(a < b for a, b in zip(times, times[1:])))
    if args.execution_log:
        records = [json.loads(line) for line in args.execution_log.read_text().splitlines() if line.startswith('{"kind"')]
        expected_files = []
        for g in groups:
            expected_files.append(f'validation_group_{g["group_id"]}.npz')
            expected_files.extend(f'validation_control_{g["group_id"]}_{j}.npz' for j in range(4))
        for g in groups:
            if g["size"] == 8:
                expected_files.extend(f'validation_singleton_{g["family"]}_{j}.npz' for j in g["units"])
        check("all 54 intervention completions recorded", [p["filename"] for p in records] == expected_files and len(records) == 54)
        check("intervention completion counters", [p["completed"] for p in records] == list(range(1, 55)) and all(p["total"] == 54 for p in records))
        progress_times = [datetime.fromisoformat(p["timestamp"]) for p in records]
        check("lock precedes first intervention completion", times[-2] < progress_times[0])
        check("all intervention completions ordered before finish", all(a < b for a, b in zip(progress_times, progress_times[1:])) and progress_times[-1] < times[-1])
    for name, digest in read_json(run / "manifest.json").items():
        check("run manifest " + name, sha(run / name) == digest)
    return {"status": "PASS", "checked_utc": datetime.now(UTC).isoformat(), "check_count": len(CHECKS),
            "group_lock_sha256": sha(run / "group_lock.json"), "validation_variant_counts": variant_counts,
            "max_instrumentation_forecast_error": max_qerror, "max_instrumentation_activation_error": max_activation_error,
            "primary": calculated, "raw_pvalues": raw_p, "holm18_pvalues": adjusted, "checks": CHECKS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--prior-run", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execution-log", type=Path)
    args = parser.parse_args()
    output = args.output or args.run_dir / "independent_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = verify(args)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    main()
