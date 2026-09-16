"""Prewritten synthetic guards and native-unit three-panel rendering contracts."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

from src import plot_target_aligned as plot


def fixture():
    rows = []
    for index, control in enumerate(
        ("aligned_constant", "constant_matrix", "dynamic_correlation")
    ):
        rows.append(
            {
                "candidate": "aligned_dynamic",
                "control": control,
                "score": "product_mse",
                "horizon": 1,
                "p_holm_wave": 0.0123,
                "p_holm_cumulative": 0.876,
                "phases": [
                    {
                        "name": name,
                        "n": 1005 if name == "development" else 1457,
                        "delta": delta + index * 1e-10,
                        "control_loss": 3e-9,
                        "candidate_loss": 3e-9 + delta + index * 1e-10,
                        "ci95_envelope": [
                            delta + index * 1e-10 - 3e-10,
                            delta + index * 1e-10 + 4e-10,
                        ],
                    }
                    for name, delta in (("evaluation", -2e-10), ("development", -1.2e-10))
                ],
            }
        )
    metrics = {
        "protocol_sha256": "a" * 64,
        "rows": rows[::-1],
        "hypothesis_count": 3,
        "cumulative_hypothesis_count": 117,
        "new_forecasts": 4924,
        "new_monthly_fits": 118,
        "new_scalar_fits": 236,
    }
    verification = {"status": "VERIFIED", "protocol_sha256": "a" * 64}
    return metrics, verification


class PlotContracts(unittest.TestCase):
    def test_complete_family_is_ordered_without_mutating_literal_values(self):
        metrics, verification = fixture()
        before = deepcopy(metrics)
        rows = plot.validated_rows(metrics, verification, "a" * 64)
        self.assertEqual(metrics, before)
        self.assertEqual(
            [(x["candidate"], x["control"], x["score"]) for x in rows],
            list(plot.CONTRASTS),
        )
        for index, row in enumerate(rows):
            self.assertEqual([x["name"] for x in row["phases"]], list(plot.PHASES))
            original = next(x for x in before["rows"] if x["control"] == row["control"])
            self.assertEqual(row["phases"], original["phases"][::-1])
            self.assertEqual(row["phases"][0]["delta"], -1.2e-10 + index * 1e-10)

    def test_exact_current_verified_hash_identity_and_success_required(self):
        for mode in ("unverified", "verification_hash", "metric_hash", "failed", "aborted"):
            with self.subTest(mode=mode):
                metrics, verification = fixture()
                if mode == "unverified":
                    verification["status"] = "FAILED"
                elif mode == "verification_hash":
                    verification["protocol_sha256"] = "b" * 64
                elif mode == "metric_hash":
                    metrics["protocol_sha256"] = "b" * 64
                elif mode == "failed":
                    metrics["status"] = "UNEVALUABLE"
                else:
                    metrics["whole_wave_aborted"] = True
                with self.assertRaises(ValueError):
                    plot.validated_rows(metrics, verification, "a" * 64)

    def test_all_registered_family_and_new_fit_counts_required(self):
        for key in (
            "hypothesis_count",
            "cumulative_hypothesis_count",
            "new_forecasts",
            "new_monthly_fits",
            "new_scalar_fits",
        ):
            for value in (None, 0, True, 999):
                with self.subTest(key=key, value=value):
                    metrics, verification = fixture()
                    metrics[key] = value
                    with self.assertRaises(ValueError):
                        plot.validated_rows(metrics, verification, "a" * 64)

    def test_missing_duplicate_or_wrong_comparison_and_phase_rejected(self):
        for mode in (
            "missing", "duplicate", "candidate", "control", "score", "horizon",
            "phase", "duplicate_phase",
        ):
            with self.subTest(mode=mode):
                metrics, verification = fixture()
                row = metrics["rows"][0]
                if mode == "missing":
                    metrics["rows"].pop()
                elif mode == "duplicate":
                    metrics["rows"][1] = deepcopy(row)
                elif mode == "candidate":
                    row["candidate"] = "dynamic_correlation"
                elif mode == "control":
                    row["control"] = "constant_correlation"
                elif mode == "score":
                    row["score"] = "matrix_qlike"
                elif mode == "horizon":
                    row["horizon"] = 5
                elif mode == "phase":
                    row["phases"].pop()
                else:
                    row["phases"][1]["name"] = "evaluation"
                with self.assertRaises(ValueError):
                    plot.validated_rows(metrics, verification, "a" * 64)

    def test_finite_native_losses_intervals_probabilities_and_n_required(self):
        for field, value in (
            ("delta", np.nan), ("delta", True), ("candidate_loss", np.inf),
            ("candidate_loss", -1e-12), ("control_loss", -1e-12),
            ("ci95_envelope", [1e-10, -1e-10]),
            ("ci95_envelope", [-1e-10, np.nan]), ("ci95_envelope", []),
            ("n", 0), ("n", True), ("n", 1.5),
        ):
            with self.subTest(field=field, value=value):
                metrics, verification = fixture()
                metrics["rows"][0]["phases"][0][field] = value
                with self.assertRaises(ValueError):
                    plot.validated_rows(metrics, verification, "a" * 64)
        for field in ("p_holm_wave", "p_holm_cumulative"):
            for value in (np.nan, np.inf, -0.1, 1.1, True):
                metrics, verification = fixture()
                metrics["rows"][0][field] = value
                with self.assertRaises(ValueError):
                    plot.validated_rows(metrics, verification, "a" * 64)

    def test_exact_zero_loss_and_effect_are_valid(self):
        metrics, verification = fixture()
        phase = metrics["rows"][0]["phases"][0]
        phase.update(delta=0.0, candidate_loss=0.0, control_loss=0.0, ci95_envelope=[0.0, 0.0])
        plot.validated_rows(metrics, verification, "a" * 64)

    def test_invalid_input_produces_no_plot_artifacts(self):
        metrics, verification = fixture()
        verification["status"] = "FAILED"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "figures"
            with self.assertRaises(ValueError):
                plot.render_verified(metrics, verification, "a" * 64, output)
            self.assertFalse(output.exists())

    def test_main_binds_current_protocol_and_blocks_canonical_failure(self):
        metrics, verification = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "reports/target_aligned"
            folder.mkdir(parents=True)
            protocol = b"synthetic target-aligned protocol\n"
            (root / "target_aligned.yaml").write_bytes(protocol)
            digest = hashlib.sha256(protocol).hexdigest()
            metrics["protocol_sha256"] = verification["protocol_sha256"] = digest
            (folder / "metrics.json").write_text(json.dumps(metrics))
            (folder / "verification.json").write_text(json.dumps(verification))
            with patch.object(plot, "render_verified", return_value={}) as render:
                plot.main(root)
                self.assertEqual(render.call_args.args[2:], (digest, folder))
                (folder / "failure.json").write_text("{}")
                render.reset_mock()
                with self.assertRaises(ValueError):
                    plot.main(root)
                render.assert_not_called()

    def test_three_panels_preserve_literal_intervals_and_render_without_label_overlap(self):
        metrics, verification = fixture()
        expected = plot.validated_rows(metrics, verification, "a" * 64)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(plot.plt, "close", wraps=plot.plt.close) as closed,
        ):
            paths = plot.render_verified(metrics, verification, "a" * 64, Path(directory))
            self.assertEqual(set(paths), {"png", "pdf"})
            self.assertTrue(Path(paths["png"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertTrue(Path(paths["pdf"]).read_bytes().startswith(b"%PDF"))
            fig = closed.call_args.args[0]
            self.assertEqual(len(fig.axes), 3)
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            renderer = canvas.get_renderer()
            title_boxes, note_boxes, label_boxes = [], [], []
            for ax, row in zip(fig.axes, expected, strict=True):
                for index, phase in enumerate(row["phases"]):
                    np.testing.assert_array_equal(
                        ax.lines[index].get_xdata(), phase["ci95_envelope"]
                    )
                    np.testing.assert_array_equal(
                        ax.collections[index].get_offsets()[0], [phase["delta"], 1 - index]
                    )
                    self.assertTrue(
                        any(f"{phase['delta']:+.2e}" in text.get_text() for text in ax.texts)
                    )
                self.assertTrue(any(
                    np.array_equal(line.get_xdata(), [-1e-10, -1e-10]) for line in ax.lines
                ))
                self.assertTrue(any(
                    np.array_equal(line.get_xdata(), [0.0, 0.0]) for line in ax.lines
                ))
                self.assertIn("decimal log return⁴", ax.get_xlabel())
                self.assertNotIn("%", ax.get_xlabel())
                note = next(x for x in ax.texts if x.get_text().startswith("Wave Holm"))
                title_boxes.append(ax._left_title.get_window_extent(renderer))
                note_boxes.append(note.get_window_extent(renderer))
                label_boxes.append(ax.xaxis.label.get_window_extent(renderer))
            for boxes in (title_boxes, note_boxes, label_boxes):
                for left, right in zip(boxes, boxes[1:]):
                    self.assertFalse(left.overlaps(right))
            for title, note in zip(title_boxes, note_boxes, strict=True):
                self.assertFalse(title.overlaps(note))
            footer = next(x for x in fig.texts if x.get_text().startswith("Lines:"))
            footer_box = footer.get_window_extent(renderer)
            for box in label_boxes:
                self.assertFalse(footer_box.overlaps(box))
            figure_box = fig.bbox
            for box in title_boxes + note_boxes + label_boxes + [footer_box]:
                self.assertGreaterEqual(box.x0, figure_box.x0)
                self.assertLessEqual(box.x1, figure_box.x1)
                self.assertGreaterEqual(box.y0, figure_box.y0)
                self.assertLessEqual(box.y1, figure_box.y1)


if __name__ == "__main__":
    unittest.main()
