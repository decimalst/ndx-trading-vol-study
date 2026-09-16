"""Generated-only preimplementation figure and verified-artifact contracts."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src import plot_peak_age as plot


def metrics():
    rows = []
    for i, control in enumerate(("baseline", "depth", "mean")):
        phases = []
        for j, name in enumerate(("development", "evaluation")):
            loss, gain = .01 * (j + 1), (i + 1) * .003 - j * .01
            delta = -loss * gain
            phases.append({"name": name, "n": 600, "control_loss": loss,
                           "candidate_loss": loss * (1 - gain), "delta": delta,
                           "gain_relative": gain, "ci95_envelope": [delta - .0001, delta + .0001]})
        rows.append({"candidate": "peak_age", "control": control, "horizon": 21, "score": "mse",
                     "phases": phases, "verdict": "DOES_NOT_QUALIFY"})
    return {"rows": rows, "leads": [], "hypothesis_count": 3, "cumulative_hypothesis_count": 140}


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


class PlotPeakAgeContracts(unittest.TestCase):
    def setup_files(self, folder, values=None):
        root = Path(folder)
        report = root / "reports/peak_age"
        report.mkdir(parents=True)
        payload = json.dumps(metrics() if values is None else values, allow_nan=False).encode()
        (report / "metrics.json").write_bytes(payload)
        old = root / "reports/event_cluster"
        old.mkdir()
        pins = {}
        for name in ["failure.json", "metrics.json", "verification.json"]:
            value = json.dumps({"synthetic_failed_record": name}).encode()
            (old / name).write_bytes(value)
            pins["reports/event_cluster/" + name] = sha(value)
        manifest = json.dumps({"inputs": pins}).encode()
        (report / "manifest.json").write_bytes(manifest)
        proof = {"status": "VERIFIED", "manifest_sha256": sha(manifest),
                 "verified_output_hashes": {"reports/peak_age/metrics.json": sha(payload)}}
        (report / "verification.json").write_text(json.dumps(proof))
        return root, report, proof

    def test_load_requires_exact_verified_metrics_and_manifest_byte_identities(self):
        for name in ["metrics.json", "manifest.json"]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                root, report, _ = self.setup_files(folder)
                self.assertEqual(plot.load_verified(root), metrics())
                with (report / name).open("a") as stream:
                    stream.write(" ")
                with self.assertRaises(ValueError):
                    plot.load_verified(root)

    def test_unverified_proof_and_missing_manifest_binding_are_rejected(self):
        for change in ["status", "manifest_hash"]:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                root, report, proof = self.setup_files(folder)
                if change == "status":
                    proof["status"] = "SCORED"
                else:
                    proof.pop("manifest_sha256")
                (report / "verification.json").write_text(json.dumps(proof))
                with self.assertRaises(ValueError):
                    plot.load_verified(root)

    def test_all_current_and_successful_ancestor_failure_markers_block_rendering(self):
        for name in ["peak_age", "index_hinge", "range_alert", "issued_calibration", "event_cluster_replay"]:
            for dangling in [False, True]:
                with self.subTest(name=name, dangling=dangling), tempfile.TemporaryDirectory() as folder:
                    root, _, _ = self.setup_files(folder)
                    marker = root / f"reports/{name}/failure.json"
                    marker.parent.mkdir(exist_ok=True)
                    if dangling:
                        marker.symlink_to(root / "absent_failure_target")
                    else:
                        marker.write_text("{}")
                    with self.assertRaises(ValueError):
                        plot.load_verified(root)

    def test_old_failed_trio_must_remain_present_bound_regular_and_byte_exact(self):
        for name in ["failure.json", "metrics.json", "verification.json"]:
            for mode in ["mutation", "remove", "symlink"]:
                with self.subTest(name=name, mode=mode), tempfile.TemporaryDirectory() as folder:
                    root, _, _ = self.setup_files(folder)
                    old = root / "reports/event_cluster" / name
                    self.assertEqual(plot.load_verified(root), metrics())
                    if mode == "mutation":
                        old.write_bytes(b"mutated failed-family record")
                    elif mode == "remove":
                        old.unlink()
                    else:
                        sentinel = root / "same_bytes.json"
                        sentinel.write_bytes(old.read_bytes())
                        old.unlink()
                        old.symlink_to(sentinel)
                    with self.assertRaises(ValueError):
                        plot.load_verified(root)

    def test_failed_metrics_cannot_be_rendered_even_if_hashes_match(self):
        for flag, value in [("status", "UNEVALUABLE"), ("whole_wave_aborted", True)]:
            bad = metrics()
            bad[flag] = value
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as folder:
                root, _, _ = self.setup_files(folder, bad)
                with self.assertRaises(ValueError):
                    plot.load_verified(root)
            with self.assertRaises(ValueError):
                plot.draw(bad)

    def test_complete_three_control_two_phase_family_cannot_be_reduced_or_renamed(self):
        for mode in ["missing", "duplicate", "candidate", "control", "horizon", "score", "phase"]:
            bad = metrics()
            if mode == "missing":
                bad["rows"].pop()
            elif mode == "duplicate":
                bad["rows"][1] = copy.deepcopy(bad["rows"][0])
            elif mode == "phase":
                bad["rows"][0]["phases"].pop()
            else:
                bad["rows"][0][mode] = 63 if mode == "horizon" else "foreign"
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                plot.draw(bad)

    def test_loss_units_effects_and_intervals_are_finite_consistent_and_ordered(self):
        for field, value in [("control_loss", 0.), ("control_loss", float("inf")),
                             ("candidate_loss", -.01), ("delta", float("nan")),
                             ("gain_relative", .9), ("ci95_envelope", [.1, -.1])]:
            bad = metrics()
            bad["rows"][0]["phases"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                plot.draw(bad)

    def test_six_effects_use_relative_mse_percent_and_positive_quarter_percent_gate(self):
        values = metrics()
        figure = plot.draw(values)
        try:
            self.assertEqual(len(figure.axes), 1)
            ax = figure.axes[0]
            self.assertIn("Relative MSE improvement (%)", ax.get_xlabel())
            self.assertEqual(len(ax.get_yticklabels()), 3)
            points = np.concatenate([item.get_offsets()[:, 0] for item in ax.collections])
            expected = [phase["gain_relative"] * 100 for row in values["rows"] for phase in row["phases"]]
            np.testing.assert_allclose(np.sort(points), np.sort(expected), rtol=0, atol=1e-12)
            self.assertEqual(len(points), 6)
            verticals = [line for line in ax.lines if len(line.get_xdata()) == 2
                         and np.all(np.asarray(line.get_xdata()) == .25)]
            self.assertEqual(len(verticals), 1)
            self.assertEqual(verticals[0].get_linestyle(), "--")
            intervals = [tuple(line.get_xdata()) for line in ax.lines if len(line.get_xdata()) == 2
                         and line.get_xdata()[0] != line.get_xdata()[1]]
            for row in values["rows"]:
                for phase in row["phases"]:
                    lo, hi = phase["ci95_envelope"]
                    expected_interval = (-hi / phase["control_loss"] * 100, -lo / phase["control_loss"] * 100)
                    self.assertTrue(any(np.allclose(one, expected_interval, rtol=0, atol=1e-12) for one in intervals))
        finally:
            plot.plt.close(figure)

    def test_row_and_phase_order_do_not_select_or_hide_comparisons(self):
        values = metrics()
        values["rows"].reverse()
        for row in values["rows"]:
            row["phases"].reverse()
        expected, actual = plot.draw(metrics()), plot.draw(values)
        try:
            first = np.concatenate([c.get_offsets() for c in expected.axes[0].collections])
            second = np.concatenate([c.get_offsets() for c in actual.axes[0].collections])
            np.testing.assert_array_equal(first, second)
        finally:
            plot.plt.close(expected)
            plot.plt.close(actual)

    def test_main_writes_usable_png_pdf_and_preserves_all_verified_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root, report, _ = self.setup_files(folder)
            inputs = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*.json")}
            plot.main(root)
            self.assertGreater((report / "comparison_intervals.png").stat().st_size, 1000)
            self.assertGreater((report / "comparison_intervals.pdf").stat().st_size, 1000)
            for path, content in inputs.items():
                self.assertEqual((root / path).read_bytes(), content)

    def test_late_failure_marker_during_bound_record_reads_is_rechecked(self):
        with tempfile.TemporaryDirectory() as folder:
            root, _, _ = self.setup_files(folder)
            original = Path.read_bytes
            state = {"mutated": False}

            def read(path):
                value = original(path)
                if path == root / "reports/event_cluster/verification.json" and not state["mutated"]:
                    marker = root / "reports/issued_calibration/failure.json"
                    marker.parent.mkdir(exist_ok=True)
                    marker.write_text("{}")
                    state["mutated"] = True
                return value

            with patch.object(Path, "read_bytes", read), self.assertRaises(ValueError):
                plot.load_verified(root)
            self.assertTrue(state["mutated"])


if __name__ == "__main__":
    unittest.main()
