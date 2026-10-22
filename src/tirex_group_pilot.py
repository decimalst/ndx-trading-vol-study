"""Fresh synthetic group design; preserves the earlier single-unit experiment.

The generator below retains its prior motif construction with new immutable
seeds and IDs. No former evaluation context is used to select these groups.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import binomtest

from src.tirex_intervention_pilot import FAMILIES, _integer, _validate_activations
from src.tirex_intervention_statistics import analyze_candidate

SEEDS = {"discovery": 641203, "validation": 908117}
FAMILY_LAYERS = {"persistence": 11, "recency": 0, "recovery": 5}


def make_group_pairs(split, pairs_per_family=24, length=2048):
    """Reorder identical values into three deliberately imperfect concepts.

    Discovery uses a fixed motif duration. Validation independently samples shorter and
    longer motifs, varying noise and amplitude, and moving burst locations.
    Every pair shares its full histogram and final observation exactly.
    """
    if split not in SEEDS:
        raise ValueError("split must be discovery or validation")
    _integer(pairs_per_family, 1, "pairs_per_family")
    _integer(length, 64, "length")
    rng = np.random.default_rng(SEEDS[split])
    recipients, donors, families, identifiers, variants = [], [], [], [], []
    for family in FAMILIES:
        for index in range(pairs_per_family):
            # Background is independent across pairs; both histories share it.
            innovations = rng.normal(0, 0.3, size=length)
            background = np.zeros(length)
            for t in range(1, length):
                background[t] = 0.92 * background[t - 1] + innovations[t]
            max_window = min(192, length // 2)
            short = bool(rng.integers(2)) if split == "validation" else False
            fraction = 0.65 if split == "discovery" else (0.40 if short else 0.95)
            window = max(12, int(max_window * fraction) // 4 * 4)
            amplitude = 1.8 if split == "discovery" else float(rng.uniform(0.8, 2.8))
            noise_sd = 0.08 if split == "discovery" else float(rng.uniform(0.04, 0.3))
            recipient, donor = background.copy(), background.copy()
            # Exclude the common terminal point from all permutations.
            end = length - 1
            start = end - window
            if family == "persistence":
                low = -amplitude + rng.normal(0, noise_sd, window // 2)
                high = amplitude + rng.normal(0, noise_sd, window // 2)
                recipient[start:end] = np.column_stack((low, high)).ravel()
                donor[start:end] = np.concatenate((low, high))
            elif family == "recency":
                burst = max(2, window // 4)
                quiet = rng.normal(0, noise_sd, window - burst)
                high = amplitude + rng.normal(0, noise_sd, burst)
                # A common quiet tail ensures both bursts end before the endpoint.
                tail = max(2, window // (8 if split == "discovery" else 6))
                recipient[start:end] = np.concatenate((high, quiet))
                donor[start:end] = np.concatenate((quiet[:-tail], high, quiet[-tail:]))
            else:
                ramp = np.linspace(-amplitude, amplitude, window) + rng.normal(0, noise_sd, window)
                recipient[start:end] = ramp
                donor[start:end] = ramp[::-1]
            recipient[-1] = donor[-1] = 0.0
            # Calculate ONE affine transform, then apply it to the permutation.
            loc, scale = recipient.mean(), recipient.std()
            recipient = -9.0 + 0.6 * (recipient - loc) / scale
            donor = -9.0 + 0.6 * (donor - loc) / scale
            recipients.append(recipient)
            donors.append(donor)
            families.append(family)
            identifiers.append(f"group-{split}-{family}-{index:03d}")
            variants.append("canonical" if split == "discovery" else ("short" if short else "long"))
    return {
        "recipient": np.asarray(recipients), "donor": np.asarray(donors),
        "family": np.asarray(families), "pair_id": np.asarray(identifiers),
        "variant": np.asarray(variants), "split": split,
    }


def select_groups(pairs, activations, weight_norms, sizes=(4, 8), n_controls=4, layers=None):
    """Nested activation-selected groups with disjoint Hungarian controls.

    Controls also match actual discovery perturbation RMS, not just typical
    activation SD. Matching is descriptive: covariance and downstream effects
    are not forced equal. Ties use ordered inputs to scipy's assignment solver.
    """
    if pairs['split'] != 'discovery':
        raise ValueError('groups must be selected on discovery only')
    _integer(n_controls, 1, 'n_controls')
    if not sizes:
        raise ValueError('at least one positive size is required')
    for size in sizes:
        _integer(size, 1, 'group size')
    if tuple(sizes) != tuple(sorted(set(sizes))):
        raise ValueError('sizes must be distinct and ascending')
    family_layers = FAMILY_LAYERS if layers is None else layers
    family_values = np.asarray(pairs['family'])
    if family_values.ndim != 1 or set(family_layers) != set(FAMILIES):
        raise ValueError('family metadata must cover all declared families')
    _validate_activations(activations, weight_norms, len(family_values))
    maximum = max(sizes)
    groups = []
    for family in FAMILIES:
        layer = family_layers[family]
        if layer not in activations:
            raise ValueError('missing fixed layer')
        values = np.asarray(activations[layer], dtype=np.float64)
        norms = np.asarray(weight_norms[layer], dtype=np.float64)
        mask = family_values == family
        if not mask.any():
            raise ValueError('missing family observations')
        scales = values.std(axis=(0, 1))
        difference = values[mask, 1] - values[mask, 0]
        means = difference.mean(axis=0)
        rms = np.sqrt(np.mean(difference ** 2, axis=0))
        eligible = np.flatnonzero((scales > 0) & (norms > 0) & (rms > 0))
        if len(eligible) < maximum * (n_controls + 1):
            raise ValueError('insufficient eligible units for disjoint controls')
        candidates = sorted(eligible.tolist(), key=lambda unit: (-abs(means[unit] / scales[unit]), unit))[:maximum]
        features = np.column_stack((scales, norms, rms))
        log_features = np.zeros_like(features)
        log_features[eligible] = np.log(features[eligible])
        used = set(candidates)
        control_sets, matching = [], []
        for _ in range(n_controls):
            available = np.array([int(unit) for unit in eligible if unit not in used])
            cost = np.sum((log_features[candidates, None, :] - log_features[available][None, :, :]) ** 2, axis=2)
            rows, columns = linear_sum_assignment(cost)
            if not np.array_equal(rows, np.arange(maximum)):
                raise RuntimeError('incomplete control assignment')
            matched = available[columns].tolist()
            control_sets.append(matched)
            slot_details = []
            for slot, (candidate, control) in enumerate(zip(candidates, matched, strict=True)):
                ratios = features[control] / features[candidate]
                slot_details.append({'candidate_unit': candidate, 'control_unit': control,
                    'sd_ratio': float(ratios[0]), 'norm_ratio': float(ratios[1]),
                    'rms_difference_ratio': float(ratios[2]), 'squared_log_cost': float(cost[slot, columns[slot]])})
            matching.append(slot_details)
            used.update(matched)
        members = [{'unit': unit, 'activation_scale': float(scales[unit]),
            'outgoing_norm': float(norms[unit]), 'rms_difference': float(rms[unit]),
            'standardized_effect': float(means[unit] / scales[unit]),
            'orientation': 1 if means[unit] >= 0 else -1} for unit in candidates]
        for size in sizes:
            groups.append({'group_id': f'{family}_k{size}', 'family': family, 'layer': int(layer),
                'size': int(size), 'units': candidates[:size],
                'control_groups': [control[:size] for control in control_sets],
                'members': members[:size], 'control_matching': [detail[:size] for detail in matching]})
    return groups


def analyze_group(base, swapped, controls, singles, direction, floor=0.001, seed=0):
    """Compare joint influence with controls and a conservative member envelope."""
    result = analyze_candidate(base, swapped, controls, direction, floor=floor, seed=seed)
    base = np.asarray(base, dtype=np.float64)
    swapped = np.asarray(swapped, dtype=np.float64)
    singles = np.asarray(singles, dtype=np.float64)
    if singles.ndim != 4 or singles.shape[0] == 0 or singles.shape[1:] != base.shape or not np.isfinite(singles).all():
        raise ValueError('singletons must have finite shape (K>0, N, 2, 3)')
    singleton_delta = singles[:, :, :, 1] - base[None, :, :, 1]
    singleton_magnitude = np.abs(singleton_delta).mean(axis=2)
    envelope = singleton_magnitude.max(axis=0)
    joint_delta = swapped[:, :, 1] - base[:, :, 1]
    advantage = np.abs(joint_delta).mean(axis=1) - envelope
    residual = joint_delta - singleton_delta.sum(axis=0)
    indices = np.random.default_rng(seed).integers(0, len(base), size=(5000, len(base)))
    interval = np.quantile(advantage[indices].mean(axis=1), [.025, .975])
    successes = int((advantage > floor).sum())
    result.update({
        'singleton_advantage_effects': advantage.tolist(),
        'singleton_advantage_successes': successes,
        'singleton_advantage_pvalue': float(binomtest(successes, len(base), p=.5, alternative='greater').pvalue),
        'singleton_advantage_mean': float(advantage.mean()),
        'singleton_advantage_ci95': interval.tolist(),
        'largest_singleton_mean_abs': float(singleton_magnitude.mean(axis=1).max()),
        'mean_pairwise_singleton_envelope': float(envelope.mean()),
        'singleton_crossing_count': int(np.any(np.diff(singles, axis=-1) < 0, axis=-1).sum()),
        'joint_minus_sum_median': residual.tolist(),
        'mean_abs_joint_minus_sum_median': float(np.abs(residual).mean()),
        'mean_abs_sum_singleton_median': float(np.abs(singleton_delta.sum(axis=0)).mean()),
        'n_singletons': int(len(singles)),
    })
    return result
