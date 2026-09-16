"""Prewritten synthetic Bernoulli-family guards and literal interval rendering."""

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

from src import plot_sign_memory as plot


def fixture():
    rows = []
    for number, control in enumerate(("baseline", "frequency")):
        rows.append({
            "candidate": "memory", "control": control, "score": "brier", "horizon": 1,
            "p_holm_wave": .0123, "p_holm_cumulative": .876,
            "phases": [
                {"name": name, "n": n, "delta": delta+number*.0002,
                 "candidate_loss": .2+delta+number*.0002, "control_loss": .2,
                 "ci95_envelope": [delta+number*.0002-.0008, delta+number*.0002+.0007]}
                for name, n, delta in (("evaluation", 360, -.0007), ("development", 240, -.0006))
            ],
        })
    metrics = {
        "protocol_sha256": "a"*64, "rows": rows[::-1], "hypothesis_count": 2,
        "cumulative_hypothesis_count": 119, "new_forecasts": 1800,
        "common_scored_origins": 600, "new_monthly_fits": 7,
    }
    verification = {
        "status": "VERIFIED", "protocol_sha256": "a"*64,
        "forecast_reconstruction": {"forecasts_verified": 1800, "monthly_fits_verified": 7},
    }
    return metrics, verification


class PlotContracts(unittest.TestCase):
    def test_exact_order_literal_values_and_no_input_mutation(self):
        metrics, verification = fixture()
        before = deepcopy(metrics)
        rows = plot.validated_rows(metrics, verification, "a"*64)
        self.assertEqual(metrics, before)
        self.assertEqual([(r["candidate"], r["control"], r["score"]) for r in rows], list(plot.CONTRASTS))
        self.assertEqual([p["name"] for p in rows[0]["phases"]], ["development", "evaluation"])
        self.assertEqual(rows[0]["phases"][0], before["rows"][1]["phases"][1])

    def test_verified_current_identity_success_family_and_reconstruction_counts_required(self):
        for mode in ("status", "verification_hash", "metrics_hash", "failure", "aborted",
                     "family", "cumulative", "forecast", "fit", "origins", "proof_forecast", "proof_fit", "proof_missing"):
            with self.subTest(mode=mode):
                metrics, verification = fixture()
                if mode == "status":
                    verification["status"] = "FAILED"
                elif mode == "verification_hash":
                    verification["protocol_sha256"] = "b"*64
                elif mode == "metrics_hash":
                    metrics["protocol_sha256"] = "b"*64
                elif mode == "failure":
                    metrics["status"] = "UNEVALUABLE"
                elif mode == "aborted":
                    metrics["whole_wave_aborted"] = True
                elif mode in ("family", "cumulative", "forecast", "fit", "origins"):
                    key = {"family": "hypothesis_count", "cumulative": "cumulative_hypothesis_count",
                           "forecast": "new_forecasts", "fit": "new_monthly_fits", "origins": "common_scored_origins"}[mode]
                    metrics[key] += 1
                elif mode == "proof_missing":
                    verification.pop("forecast_reconstruction")
                else:
                    key = "forecasts_verified" if mode == "proof_forecast" else "monthly_fits_verified"
                    verification["forecast_reconstruction"][key] += 1
                with self.assertRaises(ValueError):
                    plot.validated_rows(metrics, verification, "a"*64)

    def test_counts_are_validated_without_fixed_empirical_expectations(self):
        metrics, verification = fixture()
        metrics.update(new_monthly_fits=8, new_forecasts=1803, common_scored_origins=601)
        verification["forecast_reconstruction"].update(forecasts_verified=1803, monthly_fits_verified=8)
        for row in metrics["rows"]:
            row["phases"][0]["n"] += 1
        plot.validated_rows(metrics, verification, "a"*64)
        for key in ("new_monthly_fits", "new_forecasts", "common_scored_origins"):
            for value in (0, True, 1.5, np.nan):
                bad = deepcopy(metrics)
                bad[key] = value
                with self.assertRaises(ValueError):
                    plot.validated_rows(bad, verification, "a"*64)

    def test_complete_two_contrast_four_phase_same_row_family_required(self):
        for mode in ("missing", "duplicate", "score", "candidate", "horizon", "phase", "duplicate_phase", "sample"):
            metrics, verification = fixture()
            row = metrics["rows"][0]
            if mode == "missing":
                metrics["rows"].pop()
            elif mode == "duplicate":
                metrics["rows"][1] = deepcopy(row)
            elif mode == "score":
                row["score"] = "product_mse"
            elif mode == "candidate":
                row["candidate"] = "baseline"
            elif mode == "horizon":
                row["horizon"] = 5
            elif mode == "phase":
                row["phases"].pop()
            elif mode == "duplicate_phase":
                row["phases"][1]["name"] = "evaluation"
            else:
                row["phases"][0]["n"] += 1
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                plot.validated_rows(metrics, verification, "a"*64)

    def test_finite_bounded_brier_losses_ordered_intervals_and_probabilities(self):
        for field, value in (("delta", np.nan), ("delta", True), ("candidate_loss", -.1),
                             ("control_loss", 1.1), ("control_loss", np.inf),
                             ("ci95_envelope", [.01, -.01]), ("ci95_envelope", [0., np.nan]),
                             ("n", 0), ("n", True)):
            metrics, verification = fixture()
            metrics["rows"][0]["phases"][0][field] = value
            with self.assertRaises(ValueError):
                plot.validated_rows(metrics, verification, "a"*64)
        for field in ("p_holm_wave", "p_holm_cumulative"):
            for value in (-.1, 1.1, True, np.inf):
                metrics, verification = fixture()
                metrics["rows"][0][field] = value
                with self.assertRaises(ValueError):
                    plot.validated_rows(metrics, verification, "a"*64)
        metrics, verification = fixture()
        metrics["rows"][0]["phases"][0].update(delta=0., candidate_loss=0., control_loss=0., ci95_envelope=[0., 0.])
        plot.validated_rows(metrics, verification, "a"*64)

    def test_invalid_input_creates_no_artifacts_and_main_failure_blocks_rendering(self):
        metrics, verification = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            verification["status"] = "FAILED"
            with self.assertRaises(ValueError):
                plot.render_verified(metrics, verification, "a"*64, root/"figures")
            self.assertFalse((root/"figures").exists())
            folder = root/"reports/sign_memory"
            folder.mkdir(parents=True)
            payload = b"synthetic protocol\n"
            (root/"sign_memory.yaml").write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            metrics["protocol_sha256"] = verification["protocol_sha256"] = digest
            verification["status"] = "VERIFIED"
            (folder/"metrics.json").write_text(json.dumps(metrics))
            (folder/"verification.json").write_text(json.dumps(verification))
            with patch.object(plot, "render_verified", return_value={}) as render:
                plot.main(root)
                self.assertEqual(render.call_args.args[2:], (digest, folder))
                (folder/"failure.json").write_text("{}")
                render.reset_mock()
                with self.assertRaises(ValueError):
                    plot.main(root)
                render.assert_not_called()

    def test_literal_intervals_zero_threshold_and_readable_render(self):
        metrics, verification = fixture()
        expected = plot.validated_rows(metrics, verification, "a"*64)
        with tempfile.TemporaryDirectory() as directory, patch.object(plot.plt, "close", wraps=plot.plt.close) as closed:
            paths = plot.render_verified(metrics, verification, "a"*64, directory)
            self.assertTrue(Path(paths["png"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertTrue(Path(paths["pdf"]).read_bytes().startswith(b"%PDF"))
            fig = closed.call_args.args[0]
            self.assertEqual(len(fig.axes), 2)
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            renderer = canvas.get_renderer()
            titles, notes, labels = [], [], []
            for ax, row in zip(fig.axes, expected, strict=True):
                for index, phase in enumerate(row["phases"]):
                    np.testing.assert_array_equal(ax.lines[index].get_xdata(), phase["ci95_envelope"])
                    np.testing.assert_array_equal(ax.collections[index].get_offsets()[0], [phase["delta"], 1-index])
                self.assertTrue(any(np.array_equal(line.get_xdata(), [-.0005, -.0005]) for line in ax.lines))
                self.assertTrue(any(np.array_equal(line.get_xdata(), [0., 0.]) for line in ax.lines))
                self.assertIn("Brier", ax.get_xlabel())
                self.assertNotIn("%", ax.get_xlabel())
                note = next(x for x in ax.texts if x.get_text().startswith("Wave Holm"))
                titles.append(ax._left_title.get_window_extent(renderer))
                notes.append(note.get_window_extent(renderer))
                labels.append(ax.xaxis.label.get_window_extent(renderer))
            for boxes in (titles, notes, labels):
                self.assertFalse(boxes[0].overlaps(boxes[1]))
            for title, note in zip(titles, notes, strict=True):
                self.assertFalse(title.overlaps(note))
            footer = next(x for x in fig.texts if x.get_text().startswith("Lines:"))
            footer_box = footer.get_window_extent(renderer)
            for label in labels:
                self.assertFalse(footer_box.overlaps(label))
            for box in titles+notes+labels+[footer_box]:
                self.assertGreaterEqual(box.x0, fig.bbox.x0)
                self.assertLessEqual(box.x1, fig.bbox.x1)
                self.assertGreaterEqual(box.y0, fig.bbox.y0)
                self.assertLessEqual(box.y1, fig.bbox.y1)


if __name__ == "__main__":
    unittest.main()
