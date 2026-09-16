"""Generated scoring, multiplicity and fixed-decision contracts, prewritten."""

import copy
import unittest

import numpy as np
import pandas as pd

from src.claims_release_score import evaluate, failure_metrics, inherit_family


def fixture():
    calendar = pd.bdate_range("2018-01-01", periods=66)
    origins = calendar[:25].delete([3]).append(calendar[30:60].delete([4]))
    dev_end = calendar[29]
    config = {
        "forecast": {
            "source_end": str(calendar[-1].date()),
            "development": [str(calendar[0].date()), str(dev_end.date())],
            "evaluation": [str(calendar[30].date()), str(calendar[59].date())],
        },
        "evaluation_stability": [
            [str(calendar[30].date()), str(calendar[44].date())],
            [str(calendar[45].date()), str(calendar[59].date())],
        ],
        "inference": {
            "blocks": [3, 5, 10],
            "bootstrap_draws": 3999,
            "seed": 20260929,
            "hac_lags": 10,
            "minimum_phase_observations": 20,
            "minimum_phase_releases": 4,
            "minimum_slice_observations": 10,
            "minimum_slice_releases": 2,
            "minimum_offset_observations": 4,
            "effect_threshold_absolute": 0.005,
            "wave_alpha": 0.05,
            "cumulative_alpha": 0.05,
        },
    }
    rows = []
    for origin in origins:
        pos = calendar.get_loc(origin)
        for model in ("candidate", "market", "matched"):
            prediction = 1.0 if model == "candidate" else 2.0
            ratio = 1.0 / prediction
            rows.append(
                {
                    "origin": origin,
                    "model": model,
                    "prediction": prediction,
                    "y": 1.0,
                    "loss": ratio - np.log(ratio) - 1,
                    "target_end": calendar[pos + 5],
                    "phase": "development" if origin <= dev_end else "evaluation",
                    "claim_reference_week": calendar[(pos // 5) * 5] - pd.Timedelta(days=2),
                    "offset": pos % 5,
                }
            )
    prior = [{"study": "generated", "id": k, "p_conservative": 1.0} for k in range(140)]
    return pd.DataFrame(rows), calendar, prior, config


class ClaimsReleaseScoreTests(unittest.TestCase):
    def test_empty_valid_panel_is_insufficient_data(self):
        panel, calendar, prior, config = fixture()
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"):
            evaluate(panel.iloc[:0], calendar, prior, config)

    def test_complete_family_effect_support_and_shared_inference(self):
        panel, calendar, prior, config = fixture()
        result = evaluate(panel, calendar, prior, config)
        self.assertEqual(result["leads"], ["claims_first_report"])
        self.assertEqual(result["inherited_rows"], prior)
        self.assertEqual(result["cumulative_hypothesis_count"], 142)
        self.assertEqual(result["common_scored_origins"], 53)
        a, b = result["rows"]
        self.assertEqual([a["control"], b["control"]], ["matched", "market"])
        self.assertEqual(a["phases"], b["phases"])
        phase = a["phases"][0]
        self.assertAlmostEqual(phase["delta"], -(np.log(2) - 0.5))
        self.assertEqual(phase["annual"][0]["calendar_origins"], 30)
        self.assertEqual(phase["annual"][0]["missing_origins"], 6)
        self.assertEqual([x["n"] for x in phase["offsets"]], [5, 5, 5, 4, 5])
        self.assertEqual(a["p_holm_wave"], 0.0005)
        self.assertAlmostEqual(a["p_holm_cumulative"], 0.0355)

    def test_one_improving_contrast_is_not_a_lead(self):
        panel, calendar, prior, config = fixture()
        panel.loc[panel.model == "market", ["prediction", "loss"]] = [1.0, 0.0]
        result = evaluate(panel, calendar, prior, config)
        self.assertEqual(result["leads"], [])
        self.assertEqual(result["rows"][1]["p_conservative"], 1.0)
        self.assertEqual(result["rows"][1]["verdict"], "DOES_NOT_QUALIFY")

    def test_phase_slice_release_and_offset_support_fail_whole_wave(self):
        panel, calendar, prior, config = fixture()
        for field, value in (
            ("minimum_phase_observations", 30),
            ("minimum_phase_releases", 7),
            ("minimum_slice_observations", 16),
            ("minimum_slice_releases", 4),
            ("minimum_offset_observations", 7),
        ):
            changed = copy.deepcopy(config)
            changed["inference"][field] = value
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"),
            ):
                evaluate(panel, calendar, prior, changed)

    def test_release_weighted_mean_is_mean_of_group_means(self):
        panel, calendar, prior, config = fixture()
        # Alter a single candidate forecast: the report has fewer retained origins.
        at = (panel.origin == calendar[0]) & (panel.model == "candidate")
        panel.loc[at, "prediction"] = 0.5
        panel.loc[at, "loss"] = 2 - np.log(2) - 1
        result = evaluate(panel, calendar, prior, config)
        phase = result["rows"][0]["phases"][0]
        a = panel[(panel.model == "candidate") & (panel.phase == "development")].set_index(
            "origin"
        )
        b = panel[(panel.model == "matched") & (panel.phase == "development")].set_index(
            "origin"
        )
        expected = (a.loss - b.loss).groupby(a.claim_reference_week).mean().mean()
        self.assertAlmostEqual(phase["release_weighted_delta"], expected)

    def test_panel_corruptions_rejected_before_inference(self):
        panel, calendar, prior, config = fixture()
        cases = []
        broken = panel.copy()
        broken.loc[0, "loss"] += 0.1
        cases.append(broken)
        broken = panel.copy()
        broken.loc[0, "y"] = 0.0
        cases.append(broken)
        broken = panel.copy()
        broken.loc[0, "offset"] = 4
        cases.append(broken)
        broken = panel.copy()
        broken.loc[0, "phase"] = "evaluation"
        cases.append(broken)
        cases.extend([panel.iloc[1:], pd.concat([panel, panel.iloc[:1]])])
        for broken in cases:
            with self.subTest(n=len(broken)), self.assertRaises(ValueError):
                evaluate(broken, calendar, prior, config)

    def test_prior_comparisons_cannot_be_dropped_or_invalid(self):
        panel, calendar, prior, config = fixture()
        bad = copy.deepcopy(prior)
        bad[0]["p_conservative"] = True
        for rows in (prior[:-1], bad):
            with self.assertRaises(ValueError):
                evaluate(panel, calendar, rows, config)

    def test_inheritance_preserves_every_existing_row_and_new_three(self):
        previous = {
            "inherited_rows": [{"p_conservative": 1.0, "id": k} for k in range(137)],
            "rows": [{"study": "peak_age", "p_conservative": 0.8, "id": k} for k in range(3)],
            "hypothesis_count": 3,
            "cumulative_hypothesis_count": 140,
        }
        snapshot = copy.deepcopy(previous)
        inherited = inherit_family(previous, "a" * 64)
        self.assertEqual(len(inherited), 140)
        self.assertEqual(inherited[:137], previous["inherited_rows"])
        self.assertEqual(inherited[-1]["source_row_index"], 2)
        self.assertEqual(previous, snapshot)
        previous["rows"].pop()
        with self.assertRaises(ValueError):
            inherit_family(previous, "a" * 64)

    def test_registered_failures_keep_two_p_one_comparisons(self):
        for message, status in (
            ("INSUFFICIENT_DATA: too short", "INSUFFICIENT_DATA"),
            ("bad fit", "INVALID_RUN"),
        ):
            result = failure_metrics(ValueError(message), "a" * 64)
            self.assertEqual(result["leads"], [])
            self.assertEqual(len(result["rows"]), 2)
            for row in result["rows"]:
                self.assertEqual(row["status"], status)
                self.assertEqual(row["p_conservative"], 1.0)
                self.assertEqual(row["p_holm_cumulative"], 1.0)


if __name__ == "__main__":
    unittest.main()
