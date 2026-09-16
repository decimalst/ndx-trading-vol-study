"""Prewritten verified-artifact gates for the research figure."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src import plot_event_cluster as plot


def metrics():
    return {
        "rows": [
            {
                "candidate": "cluster",
                "control": c,
                "score": "brier",
                "phases": [
                    {"name": name, "delta": -0.0007, "ci95_envelope": [-0.002, 0.0004]}
                    for name in ("development", "evaluation")
                ],
            }
            for c in ("baseline", "nuisance", "recent_frequency")
        ],
        "leads": [],
    }


class PlotContracts(unittest.TestCase):
    def setup_files(self, folder):
        root = Path(folder)
        report = root / "reports/event_cluster"
        report.mkdir(parents=True)
        payload = json.dumps(metrics()).encode()
        (report / "metrics.json").write_bytes(payload)
        proof = {
            "status": "VERIFIED",
            "verified_output_hashes": {
                "reports/event_cluster/metrics.json": hashlib.sha256(payload).hexdigest()
            },
        }
        (report / "verification.json").write_text(json.dumps(proof))
        return root, report, proof

    def test_metric_bytes_must_match_verified_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            root, report, _ = self.setup_files(folder)
            self.assertEqual(plot.load_verified(root), metrics())
            (report / "metrics.json").write_text("{}")
            with self.assertRaises(ValueError):
                plot.load_verified(root)

    def test_unverified_or_failure_artifact_cannot_be_plotted(self):
        with tempfile.TemporaryDirectory() as folder:
            root, report, proof = self.setup_files(folder)
            proof["status"] = "SCORED"
            (report / "verification.json").write_text(json.dumps(proof))
            with self.assertRaises(ValueError):
                plot.load_verified(root)
        for name in ("event_cluster", "range_alert", "issued_calibration"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                root, _, _ = self.setup_files(folder)
                failure = root / "reports" / name / "failure.json"
                failure.parent.mkdir(exist_ok=True)
                failure.write_text("{}")
                with self.assertRaises(ValueError):
                    plot.load_verified(root)

    def test_complete_three_control_two_phase_figure(self):
        with tempfile.TemporaryDirectory() as folder:
            figure = plot.draw(metrics())
            self.assertTrue(any("Brier" in ax.get_xlabel() for ax in figure.axes))
            self.assertEqual(len(figure.axes[0].get_yticklabels()), 3)
            figure.savefig(Path(folder) / "figure.png")
            figure.savefig(Path(folder) / "figure.pdf")
            self.assertGreater((Path(folder) / "figure.png").stat().st_size, 1000)
            self.assertGreater((Path(folder) / "figure.pdf").stat().st_size, 1000)
            plot.plt.close(figure)

    def test_missing_or_invalid_comparison_cannot_be_hidden(self):
        missing, duplicate, missing_phase = metrics(), metrics(), metrics()
        missing["rows"].pop()
        duplicate["rows"][1] = duplicate["rows"][0]
        missing_phase["rows"][0]["phases"].pop()
        for damaged in (missing, duplicate, missing_phase):
            with self.assertRaises(ValueError):
                plot.draw(damaged)
        bad = metrics()
        bad["rows"][0]["phases"][0]["ci95_envelope"] = [float("nan"), 0.01]
        with self.assertRaises(ValueError):
            plot.draw(bad)


if __name__ == "__main__":
    unittest.main()
