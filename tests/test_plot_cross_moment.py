"""Prewritten synthetic publication guard and literal quartic-unit plot tests."""

from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

from src import plot_cross_moment as plot


def fixture():
    rows = []
    for control in ("constant_correlation", "constant_matrix"):
        rows.append(
            {
                "candidate": "dynamic_correlation",
                "control": control,
                "score": "product_mse",
                "horizon": 1,
                "p_holm_wave": 0.02,
                "p_holm_cumulative": 0.8,
                "phases": [
                    {
                        "name": name,
                        "n": 1005 if name == "development" else 1457,
                        "delta": delta,
                        "control_loss": 3e-9,
                        "candidate_loss": 3e-9 + delta,
                        "ci95_envelope": [delta - 3e-10, delta + 4e-10],
                    }
                    for name, delta in (("evaluation", -2e-10), ("development", -1.2e-10))
                ],
            }
        )
    metrics = {
        "protocol_sha256": "a" * 64,
        "rows": rows[::-1],
        "hypothesis_count": 2,
        "cumulative_hypothesis_count": 114,
        "new_model_fits": 0,
        "new_forecasts": 0,
    }
    verification = {"status": "VERIFIED", "protocol_sha256": "a" * 64}
    return metrics, verification


class PlotContracts(unittest.TestCase):
    def test_guard_orders_both_literal_contrasts_and_phases_without_mutation(self):
        metrics, verification = fixture()
        before = deepcopy(metrics)
        rows = plot.validated_rows(metrics, verification, "a" * 64)
        self.assertEqual(metrics, before)
        self.assertEqual(
            [row["control"] for row in rows], ["constant_correlation", "constant_matrix"]
        )
        self.assertEqual([x["name"] for x in rows[0]["phases"]], ["development", "evaluation"])
        self.assertEqual(rows[0]["phases"][0]["delta"], -1.2e-10)
        self.assertEqual(rows[0]["phases"][0]["ci95_envelope"], [-4.2e-10, 2.8e-10])

    def test_current_verified_identity_and_no_failure_or_new_fit_required(self):
        for mode in (
            "unverified",
            "wrong_verification_hash",
            "wrong_metric_hash",
            "failed",
            "aborted",
            "new_fit",
            "new_forecast",
            "family",
        ):
            metrics, verification = fixture()
            if mode == "unverified":
                verification["status"] = "FAILED"
            elif mode == "wrong_verification_hash":
                verification["protocol_sha256"] = "b" * 64
            elif mode == "wrong_metric_hash":
                metrics["protocol_sha256"] = "b" * 64
            elif mode == "failed":
                metrics["status"] = "UNEVALUABLE"
            elif mode == "aborted":
                metrics["whole_wave_aborted"] = True
            elif mode == "new_fit":
                metrics["new_model_fits"] = 1
            elif mode == "new_forecast":
                metrics["new_forecasts"] = 1
            else:
                metrics["cumulative_hypothesis_count"] = 112
            with self.assertRaises(ValueError):
                plot.validated_rows(metrics, verification, "a" * 64)

    def test_missing_duplicate_or_wrong_functional_and_phase_rejected(self):
        for mode in ("missing", "duplicate", "score", "horizon", "phase", "duplicate_phase"):
            metrics, verification = fixture()
            if mode == "missing":
                metrics["rows"].pop()
            elif mode == "duplicate":
                metrics["rows"][1] = deepcopy(metrics["rows"][0])
            elif mode == "score":
                metrics["rows"][0]["score"] = "matrix_qlike"
            elif mode == "horizon":
                metrics["rows"][0]["horizon"] = 5
            elif mode == "phase":
                metrics["rows"][0]["phases"].pop()
            else:
                metrics["rows"][0]["phases"][1]["name"] = "evaluation"
            with self.assertRaises(ValueError):
                plot.validated_rows(metrics, verification, "a" * 64)

    def test_nonfinite_or_negative_losses_reversed_intervals_and_bad_probabilities_reject(
        self,
    ):
        for field, value in (
            ("delta", np.nan),
            ("candidate_loss", np.inf),
            ("control_loss", -1e-12),
            ("candidate_loss", -1e-12),
            ("ci95_envelope", [1e-10, -1e-10]),
            ("ci95_envelope", [-1e-10, np.nan]),
            ("n", 0),
        ):
            metrics, verification = fixture()
            metrics["rows"][0]["phases"][0][field] = value
            with self.assertRaises(ValueError):
                plot.validated_rows(metrics, verification, "a" * 64)
        for value in (np.nan, np.inf, -0.1, 1.1):
            metrics, verification = fixture()
            metrics["rows"][0]["p_holm_wave"] = value
            with self.assertRaises(ValueError):
                plot.validated_rows(metrics, verification, "a" * 64)

    def test_invalid_input_produces_no_plot_artifacts(self):
        metrics, verification = fixture()
        verification["status"] = "FAILED"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "figures"
            with self.assertRaises(ValueError):
                plot.render_verified(metrics, verification, "a" * 64, output)
            self.assertFalse(output.exists())

    def test_render_preserves_unscaled_intervals_threshold_and_writes_png_and_pdf(self):
        metrics, verification = fixture()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(plot.plt, "close", wraps=plot.plt.close) as closed,
        ):
            paths = plot.render_verified(metrics, verification, "a" * 64, Path(directory))
            self.assertEqual(set(paths), {"png", "pdf"})
            self.assertTrue(Path(paths["png"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertTrue(Path(paths["pdf"]).read_bytes().startswith(b"%PDF"))
            fig = closed.call_args.args[0]
            self.assertEqual(len(fig.axes), 2)
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            renderer = canvas.get_renderer()
            for ax in fig.axes:
                np.testing.assert_array_equal(ax.lines[0].get_xdata(), [-4.2e-10, 2.8e-10])
                self.assertTrue(
                    any(
                        np.array_equal(line.get_xdata(), [-1e-10, -1e-10]) for line in ax.lines
                    )
                )
                self.assertTrue(
                    any(np.array_equal(line.get_xdata(), [0.0, 0.0]) for line in ax.lines)
                )
                self.assertIn("return", ax.get_xlabel())
                self.assertTrue(any("-1.20e-10" in text.get_text() for text in ax.texts))
                note = next(
                    text for text in ax.texts if text.get_text().startswith("Wave Holm")
                )
                self.assertFalse(
                    ax._left_title.get_window_extent(renderer).overlaps(
                        note.get_window_extent(renderer)
                    )
                )


if __name__ == "__main__":
    unittest.main()
