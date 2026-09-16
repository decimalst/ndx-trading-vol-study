"""Prewritten generated paired-inference, accounting and decision contracts."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.commodity_implied_score import (
    _passes,
    calibrate,
    evaluate,
    failure_metrics,
    inherit_family,
)
from src.orthogonal_round2 import bootstrap_means


def prior_rows(n=142):
    return [
        {
            "study": "generated_prior",
            "candidate": f"c{k}",
            "control": "baseline",
            "horizon": 5,
            "p_conservative": 1.0,
            "source": "reports/generated_prior/metrics.json",
            "source_sha256": "a" * 64,
            "source_row_index": k,
        }
        for k in range(n)
    ]


def fixture():
    calendar = pd.bdate_range("2018-01-01", periods=66)
    origins = calendar[:25].delete([3]).append(calendar[30:60].delete([4]))
    end = calendar[29]
    protocol = {
        "forecast": {
            "origin_start": str(calendar[0].date()),
            "origin_end": str(calendar[59].date()),
            "development": [str(calendar[0].date()), str(end.date())],
            "evaluation": [str(calendar[30].date()), str(calendar[59].date())],
            "source_end": str(calendar[-1].date()),
            "minimum_train": 20,
        },
        "evaluation_stability": [
            [str(calendar[30].date()), str(calendar[44].date())],
            [str(calendar[45].date()), str(calendar[59].date())],
        ],
        "inference": {
            "blocks": [3, 5, 10],
            "bootstrap_draws": 3999,
            "seed": 20260930,
            "hac_lags": 10,
            "minimum_phase_observations": 20,
            "minimum_slice_observations": 10,
            "minimum_offset_observations": 4,
            "effect_threshold_absolute": 0.005,
            "wave_alpha": 0.05,
            "cumulative_alpha": 0.05,
        },
        "calibration": {
            "rho": 0.8,
            "n": 40,
            "burn_in": 10,
            "replications": 4,
            "bootstrap_draws": 39,
            "minimum_envelope_coverage": 0.9,
        },
    }
    rows = []
    for origin in origins:
        position = calendar.get_loc(origin)
        for model in ("market", "matched", "candidate"):
            prediction = 1.0 if model == "candidate" else 2.0
            ratio = 1 / prediction
            rows.append(
                {
                    "origin": origin,
                    "model": model,
                    "prediction": prediction,
                    "y": 1.0,
                    "loss": ratio - np.log(ratio) - 1,
                    "target_end": calendar[position + 5],
                    "phase": "development" if origin <= end else "evaluation",
                    "offset": position % 5,
                }
            )
    return pd.DataFrame(rows), calendar, prior_rows(), protocol


def explicit_bootstrap(values, block, draws, seed):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    n = len(values)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(draws, n // block))
    tails = rng.integers(0, n, size=draws) if n % block else None
    samples = []
    for draw in range(draws):
        positions = [(int(s) + k) % n for s in starts[draw] for k in range(block)]
        if tails is not None:
            positions += [(int(tails[draw]) + k) % n for k in range(n % block)]
        samples.append(values[positions].mean(axis=0))
    return np.asarray(samples)


def explicit_hac(values, lags):
    values = np.asarray(values, dtype=float)
    n, mean = len(values), float(values.mean())
    centered = values - mean
    lag = min(lags, n - 1)
    covariance = [
        sum(centered[t] * centered[t - k] for t in range(k, n)) / n for k in range(lag + 1)
    ]
    longvar = covariance[0] + sum(
        2 * (1 - k / (lag + 1)) * covariance[k] for k in range(1, lag + 1)
    )
    se = float(np.sqrt(max(longvar, 0) / n))
    return {
        "se": se,
        "p": float(2 * norm.sf(abs(mean) / se)) if se else float(mean == 0),
        "ci95": [mean - 1.96 * se, mean + 1.96 * se],
        "mde80_nominal": float((norm.ppf(0.975) + norm.ppf(0.8)) * se),
    }


class CommodityScoreTests(unittest.TestCase):
    def test_exact_family_daily_diagnostics_and_both_multiplicity_counts(self):
        panel, calendar, prior, protocol = fixture()
        result = evaluate(panel, calendar, prior, protocol)
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["leads"], ["commodity_implied"])
        self.assertEqual(result["hypothesis_count"], 2)
        self.assertEqual(result["cumulative_hypothesis_count"], 144)
        self.assertEqual(result["common_scored_origins"], 53)
        self.assertEqual(result["inherited_rows"], prior)
        first, second = result["rows"]
        self.assertEqual([first["control"], second["control"]], ["matched", "market"])
        self.assertEqual(first["study"], "commodity_implied")
        self.assertEqual(first["phases"], second["phases"])
        phase = first["phases"][0]
        self.assertAlmostEqual(phase["delta"], 0.5 - np.log(2))
        self.assertEqual(
            {k: v for k, v in phase["annual"][0].items() if k != "delta"},
            {"year": 2018, "n": 24, "calendar_origins": 30, "missing_origins": 6},
        )
        self.assertAlmostEqual(phase["annual"][0]["delta"], phase["delta"])
        self.assertEqual([row["n"] for row in phase["offsets"]], [5, 5, 5, 4, 5])
        self.assertEqual(first["p_holm_wave"], 0.0005)
        self.assertAlmostEqual(first["p_holm_cumulative"], 0.036)
        for row in result["rows"]:
            for phase in row["phases"]:
                self.assertNotIn("distinct_releases", phase)
                self.assertNotIn("release_weighted_delta", phase)
                self.assertTrue(
                    all(
                        "distinct_releases" not in item
                        for item in phase["stability"] + phase["annual"]
                    )
                )

    def test_bootstrap_design_is_shared_across_controls_and_has_exact_seeds(self):
        panel, calendar, prior, protocol = fixture()
        with patch(
            "src.commodity_implied_score.bootstrap_means", wraps=bootstrap_means
        ) as sampled:
            evaluate(panel, calendar, prior, protocol)
        self.assertEqual(sampled.call_count, 6)
        expected = [(phase, block) for phase in range(2) for block in (3, 5, 10)]
        for call, (phase, block) in zip(sampled.call_args_list, expected):
            values, used_block, draws, seed = call.args
            self.assertEqual(values.shape, (24 if phase == 0 else 29, 2))
            np.testing.assert_array_equal(values[:, 0], values[:, 1])
            self.assertEqual(
                (used_block, draws, seed),
                (block, 3999, 20260930 + 5_000_000 + phase * 10000 + block),
            )

    def test_variable_loss_inference_matches_explicit_circular_and_hac_oracles(self):
        panel, calendar, prior, protocol = fixture()
        protocol["inference"]["bootstrap_draws"] = 79
        candidate = panel.model == "candidate"
        panel.loc[candidate, "prediction"] = 1.0 + 0.25 * np.sin(np.arange(candidate.sum()))
        ratio = panel.y / panel.prediction
        panel["loss"] = ratio - np.log(ratio) - 1
        result = evaluate(panel, calendar, prior, protocol)
        for phase_code, phase in enumerate(result["rows"][0]["phases"]):
            a = panel[candidate & (panel.phase == phase["name"])].sort_values("origin")
            b = panel[(panel.model == "matched") & (panel.phase == phase["name"])].sort_values(
                "origin"
            )
            differences = a.loss.to_numpy() - b.loss.to_numpy()
            expected_hac = explicit_hac(differences, 10)
            for name in expected_hac:
                np.testing.assert_allclose(
                    phase["hac"][name], expected_hac[name], rtol=1e-11, atol=1e-14
                )
            for block in (3, 5, 10):
                samples = explicit_bootstrap(
                    differences, block, 79, 20260930 + 5_000_000 + phase_code * 10000 + block
                )[:, 0]
                expected_p = (
                    1
                    + np.count_nonzero(
                        np.abs(samples - differences.mean()) >= abs(differences.mean())
                    )
                ) / 80
                self.assertEqual(phase["block_inference"][str(block)]["p"], expected_p)
                np.testing.assert_allclose(
                    phase["block_inference"][str(block)]["ci95"],
                    np.quantile(samples, [0.025, 0.975]),
                    rtol=1e-12,
                )
            self.assertEqual(
                phase["p_conservative"],
                max(phase["hac"]["p"], *(x["p"] for x in phase["block_inference"].values())),
            )
        self.assertEqual(
            result["rows"][0]["p_conservative"],
            max(x["p_conservative"] for x in result["rows"][0]["phases"]),
        )

    def test_one_improving_comparison_cannot_be_a_lead(self):
        panel, calendar, prior, protocol = fixture()
        panel.loc[panel.model == "market", ["prediction", "loss"]] = [1.0, 0.0]
        result = evaluate(panel, calendar, prior, protocol)
        self.assertEqual(result["leads"], [])
        self.assertEqual(result["rows"][1]["p_conservative"], 1.0)
        self.assertEqual(result["rows"][1]["verdict"], "DOES_NOT_QUALIFY")

    def test_each_effect_offset_slice_and_multiplicity_gate_is_required(self):
        config = fixture()[3]["inference"]
        row = {
            "p_holm_wave": 0.01,
            "p_holm_cumulative": 0.01,
            "phases": [
                {
                    "delta": -0.01,
                    "offsets": [{"delta": -0.01}],
                    "stability": [{"delta": -0.01}],
                }
                for _ in range(2)
            ],
        }
        self.assertTrue(_passes(row, config))
        for changed in ("wave", "cumulative", "effect", "offset", "slice"):
            bad = copy.deepcopy(row)
            if changed == "wave":
                bad["p_holm_wave"] = 0.05
            elif changed == "cumulative":
                bad["p_holm_cumulative"] = 0.05
            elif changed == "effect":
                bad["phases"][0]["delta"] = -0.0049
            elif changed == "offset":
                bad["phases"][1]["offsets"][0]["delta"] = 0.0
            else:
                bad["phases"][1]["stability"][0]["delta"] = 0.0
            with self.subTest(changed=changed):
                self.assertFalse(_passes(bad, config))

    def test_empty_or_unsupported_daily_panel_aborts_whole_wave(self):
        panel, calendar, prior, protocol = fixture()
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"):
            evaluate(panel.iloc[:0], calendar, prior, protocol)
        for name in (
            "minimum_phase_observations",
            "minimum_slice_observations",
            "minimum_offset_observations",
        ):
            bad = copy.deepcopy(protocol)
            bad["inference"][name] = 10000
            with (
                self.subTest(name=name),
                patch("src.commodity_implied_score.bootstrap_means") as sampled,
            ):
                with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"):
                    evaluate(panel, calendar, prior, bad)
                sampled.assert_not_called()

    def test_corrupted_cohorts_dates_loss_and_metadata_fail_before_inference(self):
        for defect in (
            "duplicate",
            "missing_arm",
            "loss",
            "offset",
            "endpoint",
            "phase",
            "string_date",
            "bool_prediction",
            "metadata",
        ):
            panel, calendar, prior, protocol = fixture()
            if defect == "duplicate":
                panel = pd.concat([panel, panel.iloc[:1]])
            elif defect == "missing_arm":
                panel = panel.iloc[1:]
            elif defect == "loss":
                panel.loc[0, "loss"] += 0.1
            elif defect == "offset":
                panel.loc[0, "offset"] = 4
            elif defect == "endpoint":
                panel.loc[0, "target_end"] = calendar[6]
            elif defect == "phase":
                panel.loc[0, "phase"] = "evaluation"
            elif defect == "string_date":
                panel["origin"] = panel.origin.astype(str)
            elif defect == "bool_prediction":
                panel["prediction"] = True
            else:
                panel["train_n"] = 1000
                panel.loc[0, "train_n"] = 999
            with (
                self.subTest(defect=defect),
                patch("src.commodity_implied_score.bootstrap_means") as sampled,
            ):
                with self.assertRaises(ValueError):
                    evaluate(panel, calendar, prior, protocol)
                sampled.assert_not_called()

    def test_full_calendar_and_source_ceiling_cannot_be_changed(self):
        panel, calendar, prior, protocol = fixture()
        for changed in (
            calendar[1:],
            calendar.tz_localize("UTC"),
            calendar + pd.Timedelta(hours=1),
            calendar.append(pd.DatetimeIndex(["2025-10-21"])),
        ):
            with self.assertRaises(ValueError):
                evaluate(panel, changed, prior, protocol)

    def test_inherited_identity_probability_and_hash_closure_are_required(self):
        panel, calendar, prior, protocol = fixture()
        for defect in (
            "missing",
            "duplicate",
            "bool",
            "nan",
            "probability",
            "sha",
            "source_index",
            "identity",
        ):
            bad = copy.deepcopy(prior)
            if defect == "missing":
                bad.pop()
            elif defect == "duplicate":
                bad[-1] = copy.deepcopy(bad[0])
            elif defect == "bool":
                bad[0]["p_conservative"] = True
            elif defect == "nan":
                bad[0]["p_conservative"] = float("nan")
            elif defect == "probability":
                bad[0]["p_conservative"] = 1.01
            elif defect == "sha":
                bad[0]["source_sha256"] = "b" * 64
            elif defect == "source_index":
                bad[0]["source_row_index"] = True
            else:
                del bad[0]["candidate"]
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                evaluate(panel, calendar, bad, protocol)

    def test_inherit_exact_completed_claims_family_preserves_every_row(self):
        previous = {
            "status": "COMPLETED",
            "hypothesis_count": 2,
            "cumulative_hypothesis_count": 142,
            "inherited_rows": prior_rows(140),
            "rows": [
                {
                    "study": "claims_release",
                    "candidate": "candidate",
                    "control": control,
                    "horizon": 5,
                    "score": "qlike",
                    "p_conservative": 0.8,
                    "verdict": "DOES_NOT_QUALIFY",
                    "phases": [{"retained": "verbatim"}],
                }
                for control in ("matched", "market")
            ],
        }
        old = copy.deepcopy(previous)
        actual = inherit_family(previous, "b" * 64)
        self.assertEqual(len(actual), 142)
        self.assertEqual(actual[:140], previous["inherited_rows"])
        for position, row in enumerate(actual[-2:]):
            self.assertEqual(
                row,
                {
                    **previous["rows"][position],
                    "source": "reports/claims_release/predictive/metrics.json",
                    "source_sha256": "b" * 64,
                    "source_row_index": position,
                },
            )
        self.assertEqual(previous, old)
        for defect in ("status", "count", "order", "probability", "signature"):
            changed = copy.deepcopy(previous)
            signature = "b" * 64
            if defect == "status":
                changed["status"] = "UNEVALUABLE"
            elif defect == "count":
                changed["cumulative_hypothesis_count"] = 140
            elif defect == "order":
                changed["rows"].reverse()
            elif defect == "probability":
                changed["rows"][0]["p_conservative"] = True
            else:
                signature = "not-a-hash"
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                inherit_family(changed, signature)

    def test_registered_failures_preserve_both_p_one_entries(self):
        for message, status in (
            ("INSUFFICIENT_DATA: too short", "INSUFFICIENT_DATA"),
            ("unsafe loss", "INVALID_RUN"),
        ):
            result = failure_metrics(ValueError(message), "c" * 64)
            self.assertEqual(result["status"], "UNEVALUABLE")
            self.assertTrue(result["whole_wave_aborted"])
            self.assertEqual(result["leads"], [])
            self.assertEqual(result["cumulative_hypothesis_count"], 144)
            self.assertEqual([r["control"] for r in result["rows"]], ["matched", "market"])
            for row in result["rows"]:
                self.assertEqual(row["status"], status)
                self.assertEqual(row["p_conservative"], 1.0)
                self.assertEqual(row["p_holm_wave"], 1.0)
                self.assertEqual(row["p_holm_cumulative"], 1.0)

    def test_generated_serial_null_calibration_matches_separate_observation_oracle(self):
        protocol = fixture()[3]
        config, inf = protocol["calibration"], protocol["inference"]
        result = calibrate(protocol)
        rng = np.random.default_rng(inf["seed"])
        hits, by_block = 0, {str(b): 0 for b in inf["blocks"]}
        for trial in range(config["replications"]):
            shocks = rng.normal(
                scale=np.sqrt(1 - config["rho"] ** 2), size=config["n"] + config["burn_in"]
            )
            state, history = rng.normal(), []
            for shock in shocks:
                state = config["rho"] * state + shock
                history.append(state)
            values = np.asarray(history[config["burn_in"] :])
            low, high = explicit_hac(values, inf["hac_lags"])["ci95"]
            for block in inf["blocks"]:
                samples = explicit_bootstrap(
                    values,
                    block,
                    config["bootstrap_draws"],
                    inf["seed"] + trial * 1000 + block,
                )
                a, b = np.quantile(samples[:, 0], [0.025, 0.975])
                by_block[str(block)] += int(a <= 0 <= b)
                low, high = min(low, a), max(high, b)
            hits += int(low <= 0 <= high)
        self.assertEqual(result["envelope_coverage"], hits / config["replications"])
        self.assertEqual(
            result["individual_block_coverage"],
            {k: value / config["replications"] for k, value in by_block.items()},
        )
        self.assertEqual(
            result["status"], "PASS" if hits / config["replications"] >= 0.9 else "FAIL"
        )
        self.assertEqual(result["seed"], 20260930)
        for key, value in config.items():
            self.assertEqual(result[key], value)


if __name__ == "__main__":
    unittest.main()
