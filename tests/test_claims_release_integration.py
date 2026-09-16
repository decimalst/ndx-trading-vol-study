"""Prewritten end-to-end integration using generated inputs and reduced budgets."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.claims_release_features import build_claim_features
from src.claims_release_market import build_market_features, make_five_session_targets
from src.claims_release_pipeline import build_panel
from src.claims_release_score import evaluate
from src.verify_claims_release_forecasts import verify_forecasts
from src.verify_claims_release_scores import verify_scores
from tests.test_verify_claims_release_forecasts import synthetic_inputs


def small_protocol(forecast):
    """Test settings only; no production protocol validation or registration."""
    return {
        "forecast": forecast,
        "evaluation_stability": [
            ["2011-07-05", "2011-07-22"],
            ["2011-07-25", "2011-08-10"],
        ],
        "inference": {
            "blocks": [3, 7, 11],
            "bootstrap_draws": 39,
            "seed": 20260929,
            "hac_lags": 7,
            "minimum_phase_observations": 8,
            "minimum_phase_releases": 2,
            "minimum_slice_observations": 4,
            "minimum_slice_releases": 2,
            "minimum_offset_observations": 1,
            "effect_threshold_absolute": 0.005,
            "wave_alpha": 0.05 / (23 * 24),
            "cumulative_alpha": 0.05,
        },
    }


class ClaimsReleaseIntegrationTests(unittest.TestCase):
    def run_chain(self, inputs):
        daily, cross, iv, ledger, config = inputs
        sources = {"daily": daily, "cross": cross, "iv": iv}
        market = build_market_features(**sources)
        features = market.join(build_claim_features(market.index, ledger))
        targets = make_five_session_targets(market.rv_total)
        produced = build_panel(features, targets, config)
        old = copy.deepcopy(produced)
        proof = verify_forecasts(
            **sources, ledger_rows=ledger, config=config, produced=produced
        )
        self.assertEqual(proof["status"], "VERIFIED")
        self.assertEqual(proof["application_origins"], len(produced["applications"]))
        self.assertEqual(proof["scored_origins"], produced["panel"].origin.nunique())
        self.assertGreater(proof["application_origins"], proof["scored_origins"])
        self.assertEqual(len(produced["panel"]), 3 * proof["scored_origins"])
        prior = [
            {
                "study": f"generated_prior_{i}",
                "p_conservative": 1.0,
                "status": "UNEVALUABLE",
                "source": f"generated_source_{i}",
            }
            for i in range(140)
        ]
        protocol = small_protocol(config)
        metrics = evaluate(produced["panel"], market.index, prior, protocol)
        verified = verify_scores(produced["panel"], market.index, metrics, prior, protocol)
        self.assertEqual(verified["status"], "VERIFIED")
        self.assertEqual(metrics["status"], "COMPLETED")
        self.assertEqual(metrics["inherited_rows"], prior)
        self.assertEqual(metrics["hypothesis_count"], 2)
        self.assertEqual(metrics["cumulative_hypothesis_count"], 142)
        self.assertEqual([row["control"] for row in metrics["rows"]], ["matched", "market"])
        # Exercise the same table/JSON types used by the root runner, in a temp dir.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            restored = {}
            for key in ("applications", "panel", "coverage", "schedules"):
                produced[key].to_parquet(root / f"{key}.parquet")
                restored[key] = pd.read_parquet(root / f"{key}.parquet")
            (root / "fits.json").write_text(json.dumps(produced["fits"], allow_nan=False))
            restored["fits"] = json.loads((root / "fits.json").read_text())
            roundtrip_proof = verify_forecasts(
                **sources, ledger_rows=ledger, config=config, produced=restored
            )
            self.assertEqual(roundtrip_proof, proof)
            metrics = json.loads(json.dumps(metrics, allow_nan=False))
            self.assertEqual(
                verify_scores(restored["panel"], market.index, metrics, prior, protocol),
                verified,
            )
        for key in ("applications", "panel", "coverage", "schedules"):
            pd.testing.assert_frame_equal(produced[key], old[key])
        self.assertEqual(produced["fits"], old["fits"])
        return produced

    def test_generated_sources_to_independently_verified_forecasts_and_scores(self):
        produced = self.run_chain(synthetic_inputs())
        self.assertEqual(
            produced["schedules"].month.tolist(), ["2011-06", "2011-07", "2011-08"]
        )
        self.assertTrue((produced["schedules"].status == "fitted").all())
        self.assertTrue((produced["coverage"].status == "target_not_mature").any())
        self.assertTrue((produced["coverage"].status == "target_after_phase_cutoff").any())

    def test_generated_source_gaps_and_opaque_audit_counts_survive_full_chain(self):
        daily, cross, iv, ledger, config = synthetic_inputs()
        iv = iv.drop(pd.Timestamp("2011-06-03"))
        cross = cross.drop(pd.Timestamp("2010-10-01"))
        for row in ledger:
            if row["reference_week"] == "2010-11-06":
                row["status"] = "unresolved_correction"
                row["first_report_value"] = None
            row["source_comparison"] = {"dol_value": 999999999, "alfred_value": -123}
        produced = self.run_chain((daily, cross, iv, ledger, config))
        coverage = produced["coverage"].set_index("origin")
        self.assertEqual(coverage.loc["2011-06-06", "status"], "incomplete_features")
        self.assertEqual(coverage.loc["2011-06-06", "missing_features"], "liv|lvix|term")
        self.assertNotIn(pd.Timestamp("2011-06-06"), produced["applications"].origin.tolist())
        self.assertTrue(
            all("2010-11-12" not in item["train_origins"] for item in produced["fits"])
        )


if __name__ == "__main__":
    unittest.main()
