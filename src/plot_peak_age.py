"""Render six complete phase/control effects from exact verified return metrics."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = ("baseline", "depth", "mean")
PHASES = ("development", "evaluation")
LABELS = ("Original market controls", "Plus depth and window return", "Training mean")
BLOCKING = ("peak_age", "index_hinge", "range_alert", "issued_calibration", "event_cluster_replay")
OLD_FAILED = tuple("reports/event_cluster/" + name for name in ("failure.json", "metrics.json", "verification.json"))


def _markers(root):
    for name in BLOCKING:
        path = root / f"reports/{name}/failure.json"
        if path.exists() or path.is_symlink():
            raise ValueError("Current or required successful ancestor failure blocks figure")


def _read(root, relative):
    path = root / relative
    cursor = path
    while cursor != root:
        if cursor.is_symlink():
            raise ValueError("Regular verified paths required; symlink blocks figure")
        cursor = cursor.parent
    if not path.is_file():
        raise ValueError("Required verified figure input absent: " + relative)
    return path.read_bytes()


def _json(payload):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate documentary JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Nonfinite documentary JSON constant: " + value)

    result = json.loads(payload, object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(result, dict):
        raise ValueError("Documentary JSON object required")
    return result


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Finite numeric " + label + " required")
    return float(value)


def _values(metrics):
    if (not isinstance(metrics, dict) or metrics.get("status") not in (None, "SCORED")
            or metrics.get("whole_wave_aborted") is True):
        raise ValueError("Failed or unpublished metrics cannot supply a figure")
    if metrics.get("hypothesis_count") != 3 or metrics.get("cumulative_hypothesis_count") != 140:
        raise ValueError("Registered three/cumulative140 hypothesis accounting required")
    rows = metrics.get("rows", [])
    keys = [(row.get("candidate"), row.get("control"), row.get("horizon"), row.get("score")) for row in rows]
    if len(keys) != 3 or set(keys) != {("peak_age", control, 21, "mse") for control in CONTROLS}:
        raise ValueError("Exact complete three-control21-session MSE family required")
    values = {}
    for row in rows:
        phases = row.get("phases", [])
        if len(phases) != 2 or {phase.get("name") for phase in phases} != set(PHASES):
            raise ValueError("Both fixed phases must remain visible")
        for phase in phases:
            control = _number(phase["control_loss"], "control MSE")
            candidate = _number(phase["candidate_loss"], "candidate MSE")
            delta = _number(phase["delta"], "paired MSE difference")
            gain = _number(phase["gain_relative"], "relative MSE improvement")
            if control <= 0 or candidate < 0:
                raise ValueError("Positive control MSE and nonnegative candidate MSE required")
            if (not math.isclose(delta, candidate - control, rel_tol=1e-10, abs_tol=1e-12)
                    or not math.isclose(gain, -delta / control, rel_tol=1e-10, abs_tol=1e-12)):
                raise ValueError("Consistent paired loss and relative improvement units required")
            interval = phase["ci95_envelope"]
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError("Two-sided nominal interval required")
            lo, hi = (_number(value, "MSE interval endpoint") for value in interval)
            if lo > hi:
                raise ValueError("Ordered nominal interval required")
            coordinates = np.asarray([100 * gain, -100 * hi / control, -100 * lo / control])
            if not np.isfinite(coordinates).all():
                raise ValueError("Finite relative-percent figure coordinates required")
            values[row["control"], phase["name"]] = coordinates
    return values


def load_verified(root=ROOT):
    """Bind metric and manifest bytes, preserve old failure, and recheck markers."""
    root = Path(root)
    _markers(root)
    paths = ["reports/peak_age/verification.json", "reports/peak_age/metrics.json", "reports/peak_age/manifest.json"]
    snapshots = {name: _read(root, name) for name in paths}
    proof, metrics, manifest = (_json(snapshots[name]) for name in paths)
    if (proof.get("status") != "VERIFIED"
            or proof.get("verified_output_hashes", {}).get(paths[1]) != hashlib.sha256(snapshots[paths[1]]).hexdigest()
            or proof.get("manifest_sha256") != hashlib.sha256(snapshots[paths[2]]).hexdigest()):
        raise ValueError("Exact independently verified metric and manifest bytes required")
    pins = manifest.get("inputs")
    if not isinstance(pins, dict):
        raise ValueError("Verified manifest input pins required")
    for name in OLD_FAILED:
        payload = _read(root, name)
        if pins.get(name) != hashlib.sha256(payload).hexdigest():
            raise ValueError("Required old failed-family record changed: " + name)
        snapshots[name] = payload
    _values(metrics)
    for name, payload in snapshots.items():
        if _read(root, name) != payload:
            raise ValueError("Figure input changed during verified load: " + name)
    _markers(root)
    return metrics


def draw(metrics):
    """Draw all six paired effects with each phase's own control-MSE denominator."""
    values = _values(metrics)
    figure, ax = plt.subplots(figsize=(10.4, 5.6))
    figure.subplots_adjust(left=.30, right=.96, bottom=.28, top=.79)
    colors = {"development": "#416B91", "evaluation": "#B06A40"}
    phase_labels = {"development": "2016–2019", "evaluation": "2020–2025"}
    for phase, offset in [("development", -.12), ("evaluation", .12)]:
        for i, control in enumerate(CONTROLS):
            center, lo, hi = values[control, phase]
            ax.plot([lo, hi], [i + offset, i + offset], color=colors[phase], linewidth=2)
            ax.scatter(center, i + offset, color=colors[phase], s=40, zorder=3,
                       label=phase_labels[phase] if i == 0 else None)
    ax.axvline(0, color="#42505A", linewidth=.9)
    ax.axvline(.25, color="#7D858B", linewidth=1, linestyle="--", label="0.25% effect gate")
    ax.set_yticks(range(3), LABELS)
    ax.set_ylim(2.45, -.45)
    ax.set_xlabel("Relative MSE improvement (%)  •  positive means improvement", labelpad=10)
    ax.grid(axis="x", alpha=.15)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(.40, -.27), ncol=3, frameon=False, fontsize=9)
    figure.suptitle("Peak age and SPX 21-session returns", x=.035, ha="left", fontsize=17)
    figure.text(.035, .835, "Sequential age correction compared with all three registered controls",
                fontsize=10, color="#52616A")
    figure.text(.035, .045,
                "Lines show nominal block-bootstrap/HAC interval envelopes. All three controls and every registered gate apply.\n"
                "Sequential correction may retune ridge shrinkage. Reused archival price-index history; dividends are excluded.",
                fontsize=8.5, color="#52616A")
    return figure


def main(root=None):
    root = ROOT if root is None else Path(root)
    metrics = load_verified(root)
    figure = draw(metrics)
    report = root / "reports/peak_age"
    staged = []
    try:
        for extension in ["png", "pdf"]:
            with tempfile.NamedTemporaryFile(dir=report, prefix=".comparison-", suffix="." + extension, delete=False) as output:
                path = Path(output.name)
            staged.append((path, report / ("comparison_intervals." + extension)))
            figure.savefig(path, dpi=180, facecolor="white")
        if load_verified(root) != metrics:
            raise ValueError("Verified metrics changed while rendering figure")
        for path, destination in staged:
            path.replace(destination)
    finally:
        plt.close(figure)
        for path, _destination in staged:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
