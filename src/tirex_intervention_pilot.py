"""Synthetic discovery design and descriptive activation-swap summaries.

All inputs are synthetic log-variance histories. Pairing removes the marginal
histogram and terminal level as explanations; it does not match every temporal
summary, nor establish prediction skill on real financial data.
"""

import numpy as np

FAMILIES = ("persistence", "recency", "recovery")
SEEDS = {"discovery": 73129, "validation": 298613}


def _integer(value, minimum, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def make_pairs(split, pairs_per_family=12, length=2048):
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
            identifiers.append(f"{split}-{family}-{index:03d}")
            variants.append("canonical" if split == "discovery" else ("short" if short else "long"))
    return {
        "recipient": np.asarray(recipients), "donor": np.asarray(donors),
        "family": np.asarray(families), "pair_id": np.asarray(identifiers),
        "variant": np.asarray(variants), "split": split,
    }


def _validate_activations(activations, weight_norms, n_pairs=None):
    if set(activations) != set(weight_norms):
        raise ValueError("activation and weight layers must match")
    for layer, values in activations.items():
        values = np.asarray(values)
        norms = np.asarray(weight_norms[layer])
        if values.ndim != 3 or values.shape[1] != 2 or values.shape[0] == 0:
            raise ValueError("activations must be nonempty pairs x 2 x units")
        if n_pairs is not None and values.shape[0] != n_pairs:
            raise ValueError("activation rows must match pair count")
        if norms.shape != (values.shape[2],):
            raise ValueError("weight norms must match units")
        if not np.isfinite(values).all() or not np.isfinite(norms).all() or np.any(norms < 0):
            raise ValueError("activations/norms must be finite; norms nonnegative")


def rank_candidates(pairs, activations, weight_norms, max_units=3):
    """One unique activation-contrast winner per family, discovery only."""
    if pairs["split"] != "discovery":
        raise ValueError("selection must use discovery data")
    _integer(max_units, 0, "max_units")
    family_values = np.asarray(pairs["family"])
    _validate_activations(activations, weight_norms, len(family_values))
    selected, used = [], set()
    for family in FAMILIES:
        if len(selected) >= max_units:
            break
        mask = family_values == family
        if not mask.any():
            continue
        candidates = []
        for layer in sorted(activations):
            values = activations[layer]
            scales = values.std(axis=(0, 1))
            deltas = (values[mask, 1] - values[mask, 0]).mean(axis=0)
            for unit in np.flatnonzero((scales > 0) & (weight_norms[layer] > 0)):
                if (layer, int(unit)) in used:
                    continue
                effect = float(deltas[unit] / scales[unit])
                candidates.append({
                    "layer": int(layer), "unit": int(unit), "family": family,
                    "orientation": 1 if effect >= 0 else -1,
                    "activation_scale": float(scales[unit]),
                    "standardized_effect": effect,
                })
        if candidates:
            winner = min(candidates, key=lambda x: (-abs(x["standardized_effect"]), x["layer"], x["unit"]))
            selected.append(winner)
            used.add((winner["layer"], winner["unit"]))
    return selected


def select_controls(candidate, activations, weight_norms, n_controls=4):
    """Nearest same-layer log-SD/log-output-norm neighbors; unit breaks ties."""
    _integer(n_controls, 0, "n_controls")
    _validate_activations(activations, weight_norms)
    layer, unit = candidate["layer"], candidate["unit"]
    values, norms = activations[layer], weight_norms[layer]
    scales = values.std(axis=(0, 1))
    if not 0 <= unit < len(scales) or scales[unit] <= 0 or norms[unit] <= 0:
        raise ValueError("candidate must have positive activation SD and output norm")
    choices = []
    for other in np.flatnonzero((scales > 0) & (norms > 0)):
        if other == unit:
            continue
        distance = np.log(scales[other] / scales[unit]) ** 2 + np.log(norms[other] / norms[unit]) ** 2
        choices.append((float(distance), int(other)))
    return [unit for _, unit in sorted(choices)[:n_controls]]


def summarize_swaps(base, swapped, unit_scale=1.0):
    """Descriptive baseline-contrast orientation, NOT a frozen hypothesis test."""
    base, swapped = np.asarray(base), np.asarray(swapped)
    if base.ndim != 3 or base.shape[1:] != (2, 3) or base.shape[0] == 0 or swapped.shape != base.shape:
        raise ValueError("forecasts must be matching nonempty pairs x 2 x 3")
    if not np.isfinite(base).all() or not np.isfinite(swapped).all():
        raise ValueError("forecasts must be finite")
    if not np.isfinite(unit_scale) or unit_scale <= 0:
        raise ValueError("unit_scale must be finite and positive")
    delta = swapped[:, :, 1] - base[:, :, 1]
    contrast = base[:, 1, 1] - base[:, 0, 1]
    eligible = np.abs(contrast) > 1e-8
    direction = np.sign(contrast[eligible])
    oriented = np.column_stack((delta[eligible, 0] * direction, -delta[eligible, 1] * direction))
    width_delta = (swapped[:, :, 2] - swapped[:, :, 0]) - (base[:, :, 2] - base[:, :, 0])
    return {
        "mean_signed_median_shift": float(oriented.mean() / unit_scale) if eligible.any() else None,
        "mean_abs_median_shift": float(np.abs(delta).mean() / unit_scale),
        "mean_abs_width_shift": float(np.abs(width_delta).mean() / unit_scale),
        "directional_agreement": float((oriented > 0).mean()) if eligible.any() else None,
        "both_direction_effect": float((oriented > 0).all(axis=1).mean()) if eligible.any() else None,
        "near_zero_contrast_count": int((~eligible).sum()),
        "directional_pair_count": int(eligible.sum()),
    }
