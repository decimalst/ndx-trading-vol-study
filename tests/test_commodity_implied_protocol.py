"""Prospective design invariants; no source arrays or empirical results."""

import copy
import unittest
from pathlib import Path

import yaml

from src.commodity_implied_protocol import CONTRACT, validate


class CommodityImpliedProtocolTests(unittest.TestCase):
    def test_serialized_contract_and_fixed_family(self):
        protocol = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "commodity_implied.yaml").read_text()
        )
        validate(protocol)
        self.assertEqual(protocol["wave"], 24)
        self.assertEqual(protocol["comparisons"]["new_hypotheses"], 2)
        self.assertEqual(protocol["comparisons"]["inherited_hypotheses"], 142)
        self.assertEqual(protocol["comparisons"]["cumulative_hypotheses"], 144)
        self.assertEqual(protocol["inference"]["wave_alpha"], 0.05 / (24 * 25))
        self.assertEqual(protocol["measurement"]["candidate_additions"], ["lovx", "lgvz"])
        self.assertEqual(len(protocol["measurement"]["matched_additions"]), 8)
        self.assertEqual(protocol["forecast"]["source_end"], "2025-10-20")
        self.assertEqual(protocol["availability"]["commodity_source_floor"], "2009-01-02")
        self.assertNotIn("minimum_train_releases", protocol["forecast"])
        self.assertNotIn("minimum_phase_releases", protocol["inference"])

    def test_no_relaxed_effect_support_or_future_boundary(self):
        changes = [
            ("forecast", "source_end", "2026-01-01"),
            ("forecast", "minimum_train", 999),
            ("inference", "effect_threshold_absolute", 0.001),
            ("inference", "minimum_phase_observations", 100),
            ("inference", "minimum_slice_observations", 1),
            ("inference", "minimum_offset_observations", 1),
            ("inference", "bootstrap_draws", 999),
            ("inference", "seed", 20260929),
            ("estimator", "minimum_scale", 0),
            ("measurement", "candidate_additions", ["lovx"]),
            ("comparisons", "cumulative_hypotheses", 2),
            ("availability", "commodity_source_floor", "2008-01-01"),
        ]
        for section, key, value in changes:
            with self.subTest(section=section, key=key):
                protocol = copy.deepcopy(CONTRACT)
                protocol[section][key] = value
                with self.assertRaises(ValueError):
                    validate(protocol)

    def test_no_extra_variants_missing_fields_or_type_coercion(self):
        for change in (
            lambda p: p.update(extra_model="selected"),
            lambda p: p.pop("limitations"),
            lambda p: p["measurement"].update(horizon=True),
            lambda p: p["inference"].update(wave_alpha=float("nan")),
        ):
            protocol = copy.deepcopy(CONTRACT)
            change(protocol)
            with self.assertRaises(ValueError):
                validate(protocol)


if __name__ == "__main__":
    unittest.main()
