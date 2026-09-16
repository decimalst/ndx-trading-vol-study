"""Prewritten fixed-contract and execution-envelope checks."""

import copy
import unittest

from src.treasury_dealer_protocol import EXECUTION, load_protocol, pipeline_config, validate


class TreasuryDealerProtocolTests(unittest.TestCase):
    def test_exact_original_protocol_and_pipeline_projection(self):
        p = load_protocol()
        validate(p)
        self.assertEqual(p["comparisons"]["cumulative"], 146)
        self.assertEqual(p["comparisons"]["inherited"], 144)
        self.assertEqual(
            pipeline_config(p),
            {
                k: p["forecast"][k]
                for k in (
                    "origin_start",
                    "origin_end",
                    "development",
                    "evaluation",
                    "source_end",
                    "minimum_train",
                )
            },
        )

    def test_every_numerical_and_identity_change_is_rejected(self):
        changes = [
            ("support", "training_activation_dates", 199),
            ("inference", "bootstrap_draws", 399998),
            ("comparisons", "cumulative", 145),
            ("source", "ceiling", "2026-01-01"),
            ("forecast", "minimum_train", True),
            ("feature", "history", "use older available donors"),
        ]
        for group, key, value in changes:
            p = load_protocol()
            p[group][key] = value
            with self.subTest(group=group, key=key), self.assertRaises(ValueError):
                validate(p)

    def test_extra_or_missing_configuration_is_rejected(self):
        p = load_protocol()
        for broken in ({**p, "tuning": []}, {k: v for k, v in p.items() if k != "source"}):
            with self.assertRaises(ValueError):
                validate(broken)

    def test_loaded_protocol_is_detached_and_outputs_are_new(self):
        p = load_protocol()
        p["source"]["ceiling"] = "2030-01-01"
        self.assertEqual(load_protocol()["source"]["ceiling"], "2025-10-20")
        self.assertEqual(EXECUTION["reports"], "reports/treasury_dealer/predictive")
        self.assertEqual(EXECUTION["data"], "data/treasury_dealer/forecasting")
        self.assertEqual(len(EXECUTION["market_pins"]), 4)

    def test_pipeline_projection_cannot_relax_protocol(self):
        p = copy.deepcopy(load_protocol())
        p["forecast"]["minimum_train"] = 10
        with self.assertRaises(ValueError):
            pipeline_config(p)


if __name__ == "__main__":
    unittest.main()
