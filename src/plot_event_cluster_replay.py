"""Render only verified replay metrics while preserving the prior failed record."""

import hashlib
import json
from pathlib import Path

from . import plot_event_cluster as original

plt = original.plt
ROOT = Path(__file__).resolve().parents[1]


def load_verified(root=ROOT):
    root = Path(root)
    for name in ("event_cluster_replay", "range_alert", "issued_calibration"):
        path = root / f"reports/{name}/failure.json"
        if path.exists() or path.is_symlink():
            raise ValueError("Current or required verified upstream failure blocks figure")
    report = root / "reports/event_cluster_replay"
    proof = json.loads((report / "verification.json").read_bytes())
    payload = (report / "metrics.json").read_bytes()
    if (
        proof.get("status") != "VERIFIED"
        or proof.get("verified_output_hashes", {}).get(
            "reports/event_cluster_replay/metrics.json"
        )
        != hashlib.sha256(payload).hexdigest()
    ):
        raise ValueError("Exact verified replay metric bytes required")
    manifest = json.loads((report / "manifest.json").read_bytes())
    old = "reports/event_cluster/failure.json"
    if manifest["inputs"].get(old) != hashlib.sha256((root / old).read_bytes()).hexdigest():
        raise ValueError("Pinned original failure record must remain unchanged")
    result = json.loads(payload)
    if result.get("status") == "UNEVALUABLE" or result.get("whole_wave_aborted") is True:
        raise ValueError("Unevaluable replay cannot supply a figure")
    return result


def draw(metrics):
    figure = original.draw(metrics)
    figure.suptitle("Event clustering: verification replay", x=0.03, ha="left", fontsize=16)
    return figure


def main():
    figure = draw(load_verified())
    report = ROOT / "reports/event_cluster_replay"
    figure.savefig(report / "comparison_intervals.png", dpi=180)
    figure.savefig(report / "comparison_intervals.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
