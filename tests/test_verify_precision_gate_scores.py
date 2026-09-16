"""Prewritten independent precision-gate score reconstruction contracts."""

import copy
import unittest
from unittest.mock import patch

from src.precision_gate_score import evaluate
from src.treasury_dealer_inference import masked_mean_inference
from src.verify_precision_gate_scores import verify_scores
from src.verify_treasury_dealer_scores import _indexed_inference
from tests.test_precision_gate_score import generated_inputs


class VerifyPrecisionGateScoresTests(unittest.TestCase):
    def make_verified(self):
        args = generated_inputs()

        def small(values, mask, **kw):
            return masked_mean_inference(values, mask, **(kw | {"draws": 31}))

        with patch("src.precision_gate_score.masked_mean_inference", side_effect=small):
            metrics = evaluate(*args)
        return args, metrics

    def check(self, args, metrics):
        def small(values, mask, **kw):
            return _indexed_inference(values, mask, **(kw | {"draws": 31}))

        with patch("src.verify_precision_gate_scores._indexed_inference", side_effect=small):
            return verify_scores(args[0], args[1], metrics, args[2], args[3])

    def test_all_three_paired_scores_and_149_family_reconstruct(self):
        args, metrics = self.make_verified()
        result = self.check(args, metrics)
        self.assertEqual(result["status"], "VERIFIED")
        self.assertEqual(result["comparisons_verified"], 3)
        self.assertEqual(result["endpoints_verified"], 6)
        self.assertEqual(result["bootstrap_runs_verified"], 18)
        self.assertEqual(result["inherited_comparisons_verified"], 146)
        self.assertEqual(
            result["fitted_gate_scored_origins"], metrics["fitted_gate_scored_origins"]
        )

    def test_all_inference_and_gate_diagnostic_corruptions_reject(self):
        args, metrics = self.make_verified()
        for defect in (
            "mean",
            "full_calendar",
            "hac",
            "block_p",
            "block_ci",
            "conservative",
            "wave",
            "cumulative",
            "offset",
            "slice",
            "annual",
            "fitted",
            "cold",
            "lead",
            "total",
        ):
            bad = copy.deepcopy(metrics)
            row = bad["rows"][0]
            phase = row["phases"][0]
            if defect == "mean":
                phase["mean"] += 0.01
            elif defect == "full_calendar":
                phase["full_calendar_n"] -= 1
            elif defect == "hac":
                phase["hac"]["se"] += 0.01
            elif defect == "block_p":
                phase["block_inference"]["21"]["p"] += 1e-12
            elif defect == "block_ci":
                phase["block_inference"]["126"]["ci95"][0] += 0.01
            elif defect == "conservative":
                row["p_conservative"] = 0.999
            elif defect in ("wave", "cumulative"):
                row["p_holm_" + defect] = 0.999
            elif defect == "offset":
                phase["offsets"][0]["mean"] += 0.01
            elif defect == "slice":
                row["phases"][1]["stability"][0]["mean"] += 0.01
            elif defect == "annual":
                phase["annual"][0]["missing_origins"] += 1
            elif defect == "fitted":
                phase["fitted_gate_n"] += 1
            elif defect == "cold":
                phase["cold_start_n"] -= 1
            elif defect == "lead":
                bad["leads"] = ["precision_gate"]
            else:
                bad["fitted_gate_scored_origins"] += 1
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                self.check(args, bad)

    def test_exact_inherited_rows_and_market_only_qualification(self):
        args, metrics = self.make_verified()
        for defect in (
            "prior",
            "class_missing",
            "limitation_missing",
            "class_altered",
            "limitation_altered",
            "treasury_qualifier",
        ):
            bad = copy.deepcopy(metrics)
            if defect == "prior":
                bad["inherited_rows"][0]["status"] = "COMPLETED"
            elif defect.endswith("missing"):
                del bad["evidence_" + defect.split("_")[0]]
            elif defect.endswith("altered"):
                bad["evidence_" + defect.split("_")[0]] += " "
            else:
                bad["source_clock_class"] = "QUALIFIED_REPORTED_CLOCK_ONLY_FOR_Z52"
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                self.check(args, bad)

    def test_consistent_cold_start_relabel_cannot_reuse_prior_counts(self):
        args, metrics = self.make_verified()
        p = args[0]
        first = p[p.gate_status == "fitted"].origin.iloc[0]
        p.loc[p.origin == first, ["gate_status", "gate_n"]] = ["cold_start", 251]
        with self.assertRaises(ValueError):
            self.check(args, metrics)

    def test_protocol_scope_and_paired_metadata_corruption_reject(self):
        args, metrics = self.make_verified()
        for defect in (
            "seed",
            "draws",
            "phase",
            "support",
            "pair",
            "endpoint",
            "offset",
            "gate_status",
        ):
            bad = list(copy.deepcopy(args))
            if defect == "seed":
                bad[3]["inference"]["seed"] += 1
            elif defect == "draws":
                bad[3]["inference"]["bootstrap_draws"] = 31
            elif defect == "phase":
                bad[3]["forecast"]["development"][1] = "2020-01-01"
            elif defect == "support":
                bad[3]["support"]["phase_fitted_gate"] = 1
            elif defect == "pair":
                bad[0] = bad[0].iloc[1:].copy()
            elif defect == "endpoint":
                bad[0].loc[0, "target_end"] = bad[0].loc[0, "origin"]
            elif defect == "offset":
                bad[0].loc[0, "offset"] = (bad[0].loc[0, "offset"] + 1) % 5
            else:
                bad[0].loc[0, "gate_status"] = "fitted"
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                self.check(bad, metrics)


if __name__ == "__main__":
    unittest.main()
