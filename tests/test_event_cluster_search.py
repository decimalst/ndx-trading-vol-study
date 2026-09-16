"""Prospective registration, inference and publication gates for range alerts."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from src import event_cluster_search as run


class EventClusterContractTests(unittest.TestCase):
    def setUp(self):
        self.p = copy.deepcopy(run.CONTRACT)

    def test_literal_contract(self):
        run.validate(self.p)
        self.assertEqual(self.p["comparisons"]["cumulative_hypotheses"], 134)
        self.assertEqual(self.p["comparisons"]["new_hypotheses"], 3)

    def test_every_leaf_change_rejected(self):
        def leaves(x, prefix=()):
            if isinstance(x, dict):
                for k, v in x.items():
                    yield from leaves(v, prefix + (k,))
            elif isinstance(x, list):
                for k, v in enumerate(x):
                    yield from leaves(v, prefix + (k,))
            else:
                yield prefix, x

        for path, value in leaves(self.p):
            p = copy.deepcopy(self.p)
            node = p
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = "altered" if not isinstance(value, str) else value + " altered"
            with self.subTest(path=path), self.assertRaises(ValueError):
                run.validate(p)

    def test_bool_cannot_replace_numeric(self):
        self.p["original_index"]["market_lag"] = True
        with self.assertRaises(ValueError):
            run.validate(self.p)

    def test_extra_contract_key_rejected(self):
        self.p["alternate_target"] = True
        with self.assertRaises(ValueError):
            run.validate(self.p)

    @staticmethod
    def row(control="baseline"):
        return {
            "candidate": "cluster",
            "control": control,
            "score": "brier",
            "p_holm_wave": 1e-6,
            "p_holm_cumulative": 0.001,
            "phases": [
                {"name": "development", "n": 200, "delta": -0.001},
                {
                    "name": "evaluation",
                    "n": 200,
                    "delta": -0.001,
                    "stability": [{"delta": -0.001}, {"delta": -0.001}],
                },
            ],
        }

    def test_all_three_controls_required(self):
        rows = [self.row(), self.row("nuisance"), self.row("recent_frequency")]
        self.assertEqual(run.candidate_leads(rows), ["cluster"])
        rows[1]["phases"][0]["delta"] = 0
        self.assertEqual(run.candidate_leads(rows), [])
        with self.assertRaises(ValueError):
            run.candidate_leads(rows[:1])

    def test_all_gates_mandatory(self):
        for key, value in [("p_holm_wave", run.WAVE_ALPHA), ("p_holm_cumulative", 0.05)]:
            r = self.row()
            r[key] = value
            self.assertFalse(run.passes(r))
        r = self.row()
        r["phases"][1]["stability"][0]["delta"] = 0
        self.assertFalse(run.passes(r))
        r = self.row()
        r["phases"][0]["delta"] = -0.000499
        self.assertFalse(run.passes(r))
        r = self.row()
        r["phases"][0]["delta"] = -0.0005
        self.assertTrue(run.passes(r))

    def test_failure_retains_three_p1(self):
        for error in [ValueError("INSUFFICIENT_DATA: support"), ValueError("optimizer")]:
            m = run.failure_metrics(error, "fixed")
            self.assertEqual(m["status"], "UNEVALUABLE")
            self.assertEqual(m["leads"], [])
            self.assertEqual(m["cumulative_hypothesis_count"], 134)
            self.assertEqual(
                [(x["candidate"], x["control"], x["p_conservative"]) for x in m["rows"]],
                [
                    ("cluster", "baseline", 1.0),
                    ("cluster", "nuisance", 1.0),
                    ("cluster", "recent_frequency", 1.0),
                ],
            )
            self.assertTrue(all(x["phases"] == [] for x in m["rows"]))

    def test_probability_domain(self):
        for a, b, y in [([1.01], [0.5], [1]), ([0.5], [0.5], [0.5]), ([np.inf], [0.5], [0])]:
            with self.assertRaises(ValueError):
                run.paired_difference(a, b, y)
        np.testing.assert_array_equal(run.paired_difference([0, 1], [1, 0], [0, 1]), [-1, -1])

    def test_positive_literal_monthly_fit_count_required(self):
        for value in [0, True, -1]:
            with self.assertRaisesRegex(ValueError, "positive literal"):
                run.evaluate(None, None, self.p, value)

    def test_direct_brier_difference_and_identical_forecasts(self):
        a = np.array([0.1, 0.7, 0.99])
        b = np.array([0.2, 0.65, 0.91])
        y = np.array([0, 1, 1])
        np.testing.assert_allclose(
            run.paired_difference(a, b, y), (a - y) ** 2 - (b - y) ** 2, rtol=1e-14, atol=0
        )
        np.testing.assert_array_equal(run.paired_difference(a, a, y), np.zeros(3))

    def test_zero_difference_inference_is_p1(self):
        p = copy.deepcopy(self.p)
        p["inference"]["bootstrap_draws"] = 99
        a = np.full(200, 0.2)
        result = run.paired_inference(a, a, np.zeros(200), p, 12)
        self.assertEqual(result["delta"], 0)
        self.assertEqual(result["p_conservative"], 1)
        self.assertEqual(result["nominal_mde_effect_ratio"], 0)

    def test_phase_bandwidth_not_shortened(self):
        with self.assertRaises(ValueError):
            run.paired_inference(np.zeros(126), np.ones(126), -np.ones(126), self.p, 0)

    def test_inherited_files_pinned_before_decode(self):
        with tempfile.TemporaryDirectory() as d, patch.object(run, "ROOT", Path(d)):
            p = copy.deepcopy(self.p)
            p["comparisons"]["inherited_sources"] = ["prior.json"]
            Path(d, "prior.json").write_text("{invalid")
            with self.assertRaisesRegex(ValueError, "Pinned inherited"):
                run.inherited(p, {"prior.json": "wrong"})

    def test_bootstrap_resolution_respects_three_comparison_family(self):
        threshold = run.WAVE_ALPHA / 3 / 10
        self.assertLess(1 / (self.p["inference"]["bootstrap_draws"] + 1), threshold)

    def test_complete_inherited_family(self):
        prior = run.inherited(self.p)
        self.assertEqual(len(prior), 131)
        self.assertEqual(
            sum(x["source"] == "reports/civil_quarter/metrics.json" for x in prior), 2
        )
        self.assertEqual(
            sum(x["source"] == "reports/civil_quarter_replay/metrics.json" for x in prior), 2
        )
        self.assertTrue(
            all(x["p_conservative"] == 1 for x in prior if "civil_quarter" in x["source"])
        )


if __name__ == "__main__":
    unittest.main()
