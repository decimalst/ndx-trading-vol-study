"""Prewritten generated precision-gate pairing, support and family contracts."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.precision_gate_score import _passes, evaluate, failure_metrics, inherit_family


def protocol():
    return {
        "forecast": {
            "issuance_start": "2010-01-04",
            "origin_start": "2016-01-01",
            "origin_end": "2025-10-20",
            "source_end": "2025-10-20",
            "development": ["2016-01-01", "2019-12-31"],
            "evaluation": ["2020-01-01", "2025-10-20"],
            "stability": [["2020-01-01", "2022-12-31"], ["2023-01-01", "2025-10-20"]],
            "horizon": 5,
            "minimum_train": 1000,
            "adaptive_half_life": 252,
            "gate_window": 1260,
            "gate_minimum_train": 252,
        },
        "support": {
            "phase_daily": 505,
            "slice_daily": 252,
            "offset_daily": 63,
            "phase_fitted_gate": 252,
            "slice_fitted_gate": 126,
        },
        "inference": {
            "blocks": [21, 63, 126],
            "hac_lags": 126,
            "bootstrap_draws": 399999,
            "seed": 20261003,
        },
        "comparisons": {
            "wave": 26,
            "inherited": 146,
            "new": 3,
            "cumulative": 149,
            "contrasts": [
                ["contextual", "base"],
                ["contextual", "adaptive"],
                ["contextual", "constant"],
            ],
            "wave_alpha": 0.05 / (26 * 27),
            "cumulative_alpha": 0.05,
        },
    }


def prior_rows(n=146):
    return [
        {
            "study": "invented",
            "candidate": "old",
            "control": "reference",
            "horizon": 5,
            "p_conservative": 1.0,
            "p_holm_wave": 1.0,
            "p_holm_cumulative": 1.0,
            "source": "invented/source.json",
            "source_sha256": "a" * 64,
            "source_row_index": i,
            "status": "UNEVALUABLE" if i == 0 else "COMPLETED",
        }
        for i in range(n)
    ]


def generated_inputs():
    calendar = pd.bdate_range("2010-01-04", "2025-10-20").as_unit("ns")
    positions = np.arange(len(calendar))
    eligible = (calendar >= "2016-01-01") & (positions + 5 < len(calendar))
    endpoint = pd.Series(calendar, index=calendar).shift(-5)
    eligible &= (calendar > "2019-12-31") | (
        endpoint.to_numpy() <= np.datetime64("2019-12-31")
    )
    eligible &= ~calendar.isin(pd.DatetimeIndex(["2018-02-07", "2021-06-09"]))
    origins, pos = calendar[eligible], positions[eligible]
    months = calendar.to_period("M")
    first_positions = {month: np.flatnonzero(months == month)[0] for month in set(months)}
    fit_pos = np.array([first_positions[x] for x in origins.to_period("M")])
    fitted = np.arange(len(origins)) % 7 != 0
    target = 1 + 0.08 * np.sin(pos / 11)
    parts = []
    for model, factor in (
        ("base", 1.8),
        ("adaptive", 1.6),
        ("constant", 1.4),
        ("contextual", 1.0),
    ):
        prediction = target * factor
        if model in ("constant", "contextual"):
            prediction = np.where(fitted, prediction, target * 1.8)
        ratio = target / prediction
        parts.append(
            pd.DataFrame(
                {
                    "origin": origins,
                    "model": model,
                    "prediction": prediction,
                    "y": target,
                    "loss": ratio - np.log(ratio) - 1,
                    "target_end": calendar[pos + 5],
                    "phase": np.where(origins <= "2019-12-31", "development", "evaluation"),
                    "offset": pos % 5,
                    "fit_origin": calendar[fit_pos],
                    "training_cutoff": calendar[fit_pos - 1],
                    "train_n": 1000,
                    "gate_n": np.where(fitted, 300, 100),
                    "gate_status": np.where(fitted, "fitted", "cold_start"),
                    "state": np.sin(pos / 17),
                }
            )
        )
    panel = (
        pd.concat(parts, ignore_index=True)
        .sort_values(["origin", "model"])
        .reset_index(drop=True)
    )
    return panel, calendar, prior_rows(), protocol()


def fake_inference(values, mask, *, blocks, hac_lags, draws, seed):
    mean = float(np.asarray(values)[mask].mean())
    return {
        "mean": mean,
        "n": int(mask.sum()),
        "full_calendar_n": len(mask),
        "hac": {
            "se": 0.0005,
            "p": 1e-10,
            "ci95": [mean - 0.001, mean + 0.001],
            "mde80_nominal": 0.0014,
        },
        "block_inference": {
            str(b): {"p": 1e-10, "ci95": [mean - 0.001, mean + 0.001]} for b in blocks
        },
        "p_conservative": 1e-10,
    }


class PrecisionGateScoreTests(unittest.TestCase):
    def run_fake(self, args):
        with patch(
            "src.precision_gate_score.masked_mean_inference", side_effect=fake_inference
        ) as sampled:
            result = evaluate(*args)
        return result, sampled

    def test_four_arms_three_contrasts_and_preserved_149_family(self):
        args = generated_inputs()
        before = copy.deepcopy(args)
        result, sampled = self.run_fake(args)
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(
            [r["control"] for r in result["rows"]], ["base", "adaptive", "constant"]
        )
        self.assertEqual(result["hypothesis_count"], 3)
        self.assertEqual(result["cumulative_hypothesis_count"], 149)
        self.assertEqual(result["inherited_rows"], args[2])
        self.assertEqual(result["leads"], ["precision_gate"])
        self.assertEqual(sampled.call_count, 6)
        self.assertEqual(
            result["evidence_class"], "EXPLORATORY_REUSED_MARKET_HISTORY_CURRENT_VINTAGES"
        )
        self.assertIn("not untouched confirmation", result["evidence_limitation"])
        self.assertNotIn("source_clock_class", result)
        for row in result["rows"]:
            self.assertAlmostEqual(row["p_holm_wave"], 3e-10)
            self.assertAlmostEqual(row["p_holm_cumulative"], 149e-10)
        pd.testing.assert_frame_equal(args[0], before[0])
        self.assertEqual(args[2:], before[2:])

    def test_full_phase_calendar_holes_tail_and_shared_seed(self):
        args = generated_inputs()
        result, sampled = self.run_fake(args)
        signatures = {}
        for call in sampled.call_args_list:
            values, mask = call.args
            self.assertEqual(call.kwargs["draws"], 399999)
            self.assertEqual(call.kwargs["blocks"], [21, 63, 126])
            self.assertEqual(call.kwargs["hac_lags"], 126)
            self.assertGreater(len(mask), int(mask.sum()))
            self.assertTrue(np.isnan(np.asarray(values)[~mask]).all())
            signatures.setdefault(call.kwargs["seed"], []).append(mask)
        self.assertEqual(set(signatures), {20261003, 20271003})
        for values in signatures.values():
            self.assertEqual(len(values), 3)
            for v in values[1:]:
                np.testing.assert_array_equal(values[0], v)
        for phase in result["rows"][0]["phases"]:
            a, b = args[3]["forecast"][phase["name"]]
            self.assertEqual(
                phase["full_calendar_n"], int(((args[1] >= a) & (args[1] <= b)).sum())
            )
            self.assertEqual([r["offset"] for r in phase["offsets"]], list(range(5)))

    def test_cold_start_rows_count_and_are_not_dropped(self):
        args = generated_inputs()
        result, _ = self.run_fake(args)
        common = args[0][args[0].model == "contextual"]
        self.assertEqual(result["common_scored_origins"], len(common))
        self.assertEqual(
            result["fitted_gate_scored_origins"], int((common.gate_status == "fitted").sum())
        )
        for phase in result["rows"][0]["phases"]:
            self.assertEqual(phase["fitted_gate_n"] + phase["cold_start_n"], phase["n"])
            self.assertGreater(phase["cold_start_n"], 0)

    def test_fitted_mechanism_floors_at_boundary_and_before_inference(self):
        args = list(generated_inputs())
        p = args[0]
        p["gate_status"], p["gate_n"] = "cold_start", 100
        origins = p.drop_duplicates("origin")
        scopes = [
            ("2016-01-01", "2019-12-31", 252),
            ("2020-01-01", "2022-12-31", 126),
            ("2023-01-01", "2025-10-20", 126),
        ]
        for a, b, n in scopes:
            dates = origins.loc[origins.origin.between(a, b), "origin"].iloc[:n]
            p.loc[p.origin.isin(dates), ["gate_status", "gate_n"]] = ["fitted", 252]
        result, _ = self.run_fake(args)
        self.assertEqual(result["rows"][0]["phases"][0]["fitted_gate_n"], 252)
        for a, b, _ in scopes:
            bad = copy.deepcopy(args)
            row = (
                bad[0]
                .loc[bad[0].origin.between(a, b) & (bad[0].gate_status == "fitted")]
                .iloc[0]
            )
            bad[0].loc[bad[0].origin == row.origin, ["gate_status", "gate_n"]] = [
                "cold_start",
                100,
            ]
            with (
                patch("src.precision_gate_score.masked_mean_inference") as sampled,
                self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"),
            ):
                evaluate(*bad)
            sampled.assert_not_called()

    def test_daily_slice_and_offset_floors_fail_before_sampling(self):
        for defect in ("phase", "slice", "offset"):
            args = list(generated_inputs())
            p = args[0]
            common = p[p.model == "contextual"]
            if defect == "phase":
                dates = common[common.phase == "development"].origin.iloc[504:]
            elif defect == "slice":
                dates = common[common.origin.between("2020-01-01", "2022-12-31")].origin.iloc[
                    251:
                ]
            else:
                dates = common[
                    (common.phase == "evaluation") & (common.offset == 0)
                ].origin.iloc[62:]
            args[0] = p[~p.origin.isin(dates)].copy()
            with (
                patch("src.precision_gate_score.masked_mean_inference") as sampled,
                self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"),
            ):
                evaluate(*args)
            sampled.assert_not_called()

    def test_exact_pairing_dates_status_loss_and_state_guards(self):
        for defect in (
            "missing_arm",
            "duplicate",
            "target",
            "offset",
            "loss",
            "gate_status",
            "gate_n",
            "state",
            "future_fit",
            "cutoff",
            "train_n",
        ):
            args = list(generated_inputs())
            p = args[0]
            if defect == "missing_arm":
                args[0] = p.iloc[1:].copy()
            elif defect == "duplicate":
                args[0] = pd.concat([p, p.iloc[:1]], ignore_index=True)
            elif defect == "target":
                p.loc[0, "target_end"] = p.loc[0, "origin"]
            elif defect == "offset":
                p.loc[0, "offset"] = (p.loc[0, "offset"] + 1) % 5
            elif defect == "loss":
                p.loc[0, "loss"] += 0.01
            elif defect == "gate_status":
                p.loc[0, "gate_status"] = "unknown"
            elif defect == "gate_n":
                p.loc[0, "gate_n"] = 1261
            elif defect == "state":
                p.loc[0, "state"] = np.inf
            elif defect == "future_fit":
                p.loc[0, "fit_origin"] = p.loc[0, "target_end"]
            elif defect == "cutoff":
                p.loc[0, "training_cutoff"] = p.loc[0, "fit_origin"]
            else:
                p.loc[0, "train_n"] = 999
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                self.run_fake(args)

    def test_fixed_scientific_fields_and_prior_provenance(self):
        for branch, key, value in (
            ("inference", "bootstrap_draws", 999),
            ("inference", "seed", 1),
            ("support", "phase_fitted_gate", 1),
            ("forecast", "source_end", "2025-11-03"),
            ("comparisons", "inherited", 145),
        ):
            args = list(generated_inputs())
            args[3][branch][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.run_fake(args)
        args = list(generated_inputs())
        args[2][1]["source_row_index"] = 0
        with self.assertRaises(ValueError):
            self.run_fake(args)

    def test_inclusive_probability_effect_boundaries_and_stability_veto(self):
        row = {
            "p_holm_wave": 0.05 / (26 * 27),
            "p_holm_cumulative": 0.05,
            "phases": [
                {
                    "mean": -0.005,
                    "offsets": [{"mean": -0.001}],
                    "stability": [{"mean": -0.001}],
                }
            ]
            * 2,
        }
        self.assertTrue(_passes(row, protocol()))
        for field in ("mean", "offset", "slice", "wave", "cumulative"):
            bad = copy.deepcopy(row)
            if field == "mean":
                bad["phases"][0]["mean"] = -0.004999
            elif field in ("offset", "slice"):
                bad["phases"][0]["offsets" if field == "offset" else "stability"][0][
                    "mean"
                ] = 0
            else:
                bad["p_holm_" + field] += 1e-10
            self.assertFalse(_passes(bad, protocol()))

    def test_worst_phase_method_and_single_control_veto(self):
        args = generated_inputs()
        count = 0

        def bad_last(*a, **k):
            nonlocal count
            count += 1
            result = fake_inference(*a, **k)
            if count == 6:
                result["hac"]["p"] = result["p_conservative"] = 0.9
            return result

        with patch("src.precision_gate_score.masked_mean_inference", side_effect=bad_last):
            result = evaluate(*args)
        self.assertEqual(result["leads"], [])
        self.assertEqual(result["rows"][2]["p_conservative"], 0.9)
        self.assertEqual(result["rows"][2]["verdict"], "DOES_NOT_QUALIFY")

    def test_failure_and_unevaluable_treasury_inheritance_preserved(self):
        old = prior_rows(144)
        treasury = {
            "status": "UNEVALUABLE",
            "hypothesis_count": 2,
            "cumulative_hypothesis_count": 146,
            "inherited_rows": old,
            "rows": [
                {
                    "study": "treasury_dealer",
                    "candidate": "candidate",
                    "control": c,
                    "horizon": 5,
                    "score": "qlike",
                    "status": "INSUFFICIENT_DATA",
                    "verdict": "UNEVALUABLE",
                    "p_conservative": 1.0,
                    "p_holm_wave": 1.0,
                    "p_holm_cumulative": 1.0,
                }
                for c in ("matched", "market")
            ],
        }
        inherited = inherit_family(treasury, "b" * 64)
        self.assertEqual(inherited[:144], old)
        self.assertEqual(len(inherited), 146)
        self.assertEqual(
            inherited[-1]["source"], "reports/treasury_dealer/predictive/metrics.json"
        )
        result = failure_metrics(
            ValueError("INSUFFICIENT_DATA: invented"), "c" * 64, inherited
        )
        self.assertEqual(result["status"], "UNEVALUABLE")
        self.assertEqual(result["inherited_rows"], inherited)
        self.assertEqual(
            [r["control"] for r in result["rows"]], ["base", "adaptive", "constant"]
        )
        for row in result["rows"]:
            self.assertEqual(row["status"], "INSUFFICIENT_DATA")
            for field in ("p_conservative", "p_holm_wave", "p_holm_cumulative"):
                self.assertEqual(row[field], 1.0)
        self.assertIn("evidence_limitation", result)
        bad = copy.deepcopy(treasury)
        bad["rows"][0]["p_conservative"] = 0.1
        with self.assertRaises(ValueError):
            inherit_family(bad, "b" * 64)


if __name__ == "__main__":
    unittest.main()
