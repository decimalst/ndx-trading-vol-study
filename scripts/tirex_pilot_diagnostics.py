"""Retrospective diagnostics from frozen arrays; imports no pilot implementation."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom


def align(values, source_ids, target_ids):
    source_ids, target_ids = list(source_ids), list(target_ids)
    if len(set(source_ids)) != len(source_ids) or len(set(target_ids)) != len(
        target_ids
    ):
        raise ValueError("duplicate IDs")
    if len(values) != len(source_ids) or not set(target_ids).issubset(source_ids):
        raise ValueError("missing or misaligned IDs")
    lookup = {key: i for i, key in enumerate(source_ids)}
    return np.asarray(values)[[lookup[key] for key in target_ids]]


def summarize(base, swapped, direction, floor=0.001):
    base, swapped = np.asarray(base, dtype=float), np.asarray(swapped, dtype=float)
    if (
        base.ndim != 3
        or base.shape[1:] != (2, 3)
        or len(base) == 0
        or base.shape != swapped.shape
    ):
        raise ValueError("need matching nonempty N x 2 x 3 arrays")
    if (
        not np.isfinite(base).all()
        or not np.isfinite(swapped).all()
        or direction not in (-1, 1)
    ):
        raise ValueError("nonfinite or invalid direction")
    delta = swapped[:, :, 1] - base[:, :, 1]
    raw = (delta[:, 0] - delta[:, 1]) / 2
    oriented = direction * raw
    expected = (direction * delta[:, 0] > 0) & (-direction * delta[:, 1] > 0)
    opposite = (direction * delta[:, 0] < 0) & (-direction * delta[:, 1] < 0)
    successes = int((oriented > floor).sum())
    return {
        "n": len(base),
        "frozen_direction": direction,
        "raw_bidirectional_mean": float(raw.mean()),
        "oriented_bidirectional_mean": float(oriented.mean()),
        "recipient_mean": float(delta[:, 0].mean()),
        "donor_mean": float(delta[:, 1].mean()),
        "mean_abs_shift": float(np.abs(delta).mean()),
        "native_donor_minus_recipient_mean": float(
            (base[:, 1, 1] - base[:, 0, 1]).mean()
        ),
        "both_expected_count": int(expected.sum()),
        "both_opposite_count": int(opposite.sum()),
        "mixed_or_zero_count": int((~(expected | opposite)).sum()),
        "oriented_positive_count": int((oriented > 0).sum()),
        "oriented_negative_count": int((oriented < 0).sum()),
        "influence_successes": successes,
        "influence_raw_p": float(binom.sf(successes - 1, len(base), 0.5)),
    }


def power_audit(n, alpha=0.05, claims=18, power=0.8):
    first = alpha / claims
    cutoff = next(k for k in range(n + 1) if binom.sf(k - 1, n, 0.5) <= first)
    q = brentq(lambda value: binom.sf(cutoff - 1, n, value) - power, 0.5, 1)
    return {
        "n": n,
        "alpha": alpha,
        "claims": claims,
        "first_holm_threshold": first,
        "first_holm_min_k": cutoff,
        "first_holm_raw_p_at_cutoff": float(binom.sf(cutoff - 1, n, 0.5)),
        "first_holm_raw_p_one_fewer": float(binom.sf(cutoff - 2, n, 0.5)),
        "exceedance_probability_for_80pct_power": float(q),
        "power_at_exceedance_probability_half": float(binom.sf(cutoff - 1, n, 0.5)),
        "log_variance_minimum_detectable_mean": None,
        "mde_limit": "The endpoint is probability of strictly exceeding a fixed floor. A log-variance minimum detectable mean requires an effect distribution assumption; effect magnitude above the floor does not change this test.",
        "holm_limit": "This is the strictest first step. Later claims can use less stringent cutoffs if earlier ordered claims reject. Dependence and nested claims do not invalidate Holm.",
    }


def summarize_singletons(base, singles, units, direction):
    base, singles = np.asarray(base, dtype=float), np.asarray(singles, dtype=float)
    if (
        singles.ndim != 4
        or singles.shape[1:] != base.shape
        or len(singles) != len(units)
    ):
        raise ValueError("invalid singleton axes")
    delta = singles[:, :, :, 1] - base[None, :, :, 1]
    magnitudes = np.abs(delta).mean(axis=2)
    means = magnitudes.mean(axis=1)
    raw = (delta[:, :, 0] - delta[:, :, 1]) / 2
    return {
        "dominant_unit_by_mean_magnitude": int(units[int(means.argmax())]),
        "largest_singleton_mean_magnitude": float(means.max()),
        "mean_pairwise_envelope": float(magnitudes.max(axis=0).mean()),
        "sum_raw_bidirectional_mean": float(raw.sum(axis=0).mean()),
        "sum_oriented_bidirectional_mean": float(direction * raw.sum(axis=0).mean()),
        "members": [
            dict(unit=int(unit), **summarize(base, single, direction))
            for unit, single in zip(units, singles, strict=True)
        ],
    }


def load_prediction(path, ids):
    with np.load(path) as saved:
        return align(saved["quantiles"], saved["pair_id"].tolist(), ids)


def extract(run):
    run = Path(run)
    lock = json.loads((run / "group_lock.json").read_text())
    groups = lock["groups"]
    out = {
        "analysis_type": "Retrospective diagnostic reanalysis; no new confirmatory claims",
        "power": power_audit(32),
        "power_sensitivity": [power_audit(n) for n in (32, 160, 320)],
        "groups": [],
        "original_claims": [],
    }
    for group in groups:
        gid, family = group["group_id"], group["family"]
        direction = group["forecast_direction"]
        row = {
            "group_id": gid,
            "layer": group["layer"],
            "units": group["units"],
            "frozen_direction": direction,
        }
        for split in ("discovery", "validation"):
            with np.load(run / f"{split}.npz") as archive:
                take = archive["family"] == family
                ids = archive["pair_id"][take].tolist()
                base = archive["quantiles"][take].astype(float)
                variants = archive["variant"][take]
                acts = archive[f"layer_{group['layer']}"][take].astype(float)
            swap = load_prediction(run / f"{split}_group_{gid}.npz", ids).astype(float)
            stats = summarize(base, swap, direction)
            # Same unit-scale/orientation averaging as the report; activation
            # consistency is a group aggregate, not every unit's sign agreeing.
            effects = []
            for member in group["members"]:
                unit = member["unit"]
                effects.append(
                    member["orientation"]
                    * (acts[:, 1, unit] - acts[:, 0, unit])
                    / member["activation_scale"]
                )
            activation = np.asarray(effects).mean(axis=0)
            stats["group_activation_positive_count"] = int((activation > 0).sum())
            stats["group_activation_negative_count"] = int((activation < 0).sum())
            stats["mean_standardized_activation_contrast"] = float(activation.mean())
            if split == "validation":
                controls = np.stack(
                    [
                        load_prediction(run / f"validation_control_{gid}_{j}.npz", ids)
                        for j in range(4)
                    ]
                ).astype(float)
                singles = np.stack(
                    [
                        load_prediction(
                            run / f"validation_singleton_{family}_{unit}.npz", ids
                        )
                        for unit in group["units"]
                    ]
                ).astype(float)
                delta = swap[:, :, 1] - base[:, :, 1]
                single_delta = singles[:, :, :, 1] - base[None, :, :, 1]
                control_mag = np.abs(controls[:, :, :, 1] - base[None, :, :, 1]).mean(
                    axis=(0, 2)
                )
                group_mag = np.abs(delta).mean(axis=1)
                envelope = np.abs(single_delta).mean(axis=2).max(axis=0)
                extras = {
                    "control_advantage": group_mag - control_mag,
                    "singleton_advantage": group_mag - envelope,
                }
                row["singletons"] = summarize_singletons(
                    base, singles, group["units"], direction
                )
                stats["control_mean_abs_shift"] = float(control_mag.mean())
                stats["singleton_envelope_mean_abs_shift"] = float(envelope.mean())
                stats["mean_abs_joint_minus_sum"] = float(
                    np.abs(delta - single_delta.sum(axis=0)).mean()
                )
                for name, values in [
                    ("influence", direction * (delta[:, 0] - delta[:, 1]) / 2),
                    *extras.items(),
                ]:
                    k = int((values > 0.001).sum())
                    out["original_claims"].append(
                        {
                            "group_id": gid,
                            "claim": name,
                            "k": k,
                            "n": len(values),
                            "raw_p": float(binom.sf(k - 1, len(values), 0.5)),
                            "mean": float(values.mean()),
                        }
                    )
                strata = {}
                for variant in sorted(set(variants)):
                    mask = variants == variant
                    v = summarize(base[mask], swap[mask], direction)
                    v["group_activation_positive_count"] = int(
                        (activation[mask] > 0).sum()
                    )
                    v["group_activation_negative_count"] = int(
                        (activation[mask] < 0).sum()
                    )
                    v["mean_standardized_activation_contrast"] = float(
                        activation[mask].mean()
                    )
                    v["control_mean_abs_shift"] = float(control_mag[mask].mean())
                    v["singleton_envelope_mean_abs_shift"] = float(
                        envelope[mask].mean()
                    )
                    for name, values in extras.items():
                        k = int((values[mask] > 0.001).sum())
                        v[f"{name}_k"] = k
                        v[f"{name}_raw_p"] = float(
                            binom.sf(k - 1, int(mask.sum()), 0.5)
                        )
                    v["singletons"] = summarize_singletons(
                        base[mask], singles[:, mask], group["units"], direction
                    )
                    strata[str(variant)] = v
                row["validation_by_motif"] = strata
                if family == "recovery" and len(group["units"]) == 8:
                    row["prefix4_singleton_sum"] = summarize_singletons(
                        base, singles[:4], group["units"][:4], direction
                    )
                    row["additional4_singleton_sum"] = summarize_singletons(
                        base, singles[4:], group["units"][4:], direction
                    )
            row[split] = stats
        out["groups"].append(row)
    # Reconstruct adjusted values without production function.
    p = np.array([claim["raw_p"] for claim in out["original_claims"]])
    order = np.argsort(p, kind="stable")
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(
        1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1))
    )
    for claim, value in zip(out["original_claims"], adjusted, strict=True):
        claim["holm_p"] = float(value)
    original = json.loads((run / "summary.json").read_text())
    out["source_run_elapsed_seconds"] = original["elapsed_seconds"]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(extract(args.run_dir), indent=2) + "\n")


if __name__ == "__main__":
    main()
