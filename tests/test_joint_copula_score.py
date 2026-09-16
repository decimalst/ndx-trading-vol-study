"""Prewritten generated full-density, cancellation and wave27 score contracts."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, multivariate_t, norm, t

from src.joint_copula_score import (
    _density_components,
    _passes,
    evaluate,
    failure_metrics,
    inherit_family,
)


def protocol():
    return {
        "forecast": {
            "origin_start": "2016-01-04",
            "origin_end": "2025-10-17",
            "source_end": "2025-10-20",
            "development": ["2016-01-04", "2019-12-31"],
            "evaluation": ["2020-01-01", "2025-10-20"],
            "stability": [["2020-01-01", "2022-12-31"], ["2023-01-01", "2025-10-20"]],
            "horizon": 1,
            "minimum_train": 1000,
        },
        "support": {"phase_daily": 505, "slice_daily": 252, "offset_daily": 63},
        "inference": {
            "blocks": [21, 63, 126],
            "hac_lags": 126,
            "bootstrap_draws": 399999,
            "seed": 20260909,
        },
        "comparisons": {
            "wave": 27,
            "inherited": 149,
            "new": 2,
            "cumulative": 151,
            "contrasts": [["t8_copula", "gaussian_copula"], ["t8_copula", "independence"]],
            "wave_alpha": 0.05 / (27 * 28),
            "cumulative_alpha": 0.05,
        },
    }


def prior_rows(n=149):
    return [
        {
            "study": "invented",
            "candidate": "old",
            "control": "old_reference",
            "horizon": 5,
            "p_conservative": 1.0,
            "p_holm_wave": 1.0,
            "p_holm_cumulative": 1.0,
            "source": "invented/old.json",
            "source_sha256": "a" * 64,
            "source_row_index": i,
            "status": "UNEVALUABLE" if i == 0 else "COMPLETED",
        }
        for i in range(n)
    ]


def generated_inputs():
    calendar = pd.bdate_range("2010-01-04", "2025-10-20").as_unit("ns")
    pos = np.arange(len(calendar))
    eligible = (
        (calendar >= "2016-01-04") & (calendar <= "2025-10-17") & (pos + 1 < len(calendar))
    )
    endpoints = pd.Series(calendar, index=calendar).shift(-1)
    eligible &= (calendar > "2019-12-31") | (
        endpoints.to_numpy() <= np.datetime64("2019-12-31")
    )
    eligible &= ~calendar.isin(pd.DatetimeIndex(["2018-02-07", "2021-06-09"]))
    origins = calendar[eligible]
    p = pos[eligible]
    months = calendar.to_period("M")
    first = {m: np.flatnonzero(months == m)[0] for m in set(months)}
    fit = np.array(
        [max(first[m], calendar.get_loc("2016-01-04")) for m in origins.to_period("M")]
    )
    z = np.column_stack([4 + 0.1 * np.sin(p / 13), 4 + 0.1 * np.cos(p / 17)])
    h = np.column_stack([0.002 + 0.0001 * np.sin(p / 19), 0.003 + 0.0001 * np.cos(p / 23)])
    mu = np.column_stack([0.001 * np.sin(p / 7), 0.002 * np.cos(p / 11)])
    y = mu + np.sqrt(h * 0.75) * z
    parts = []
    for model, rho in [("t8_copula", 0.6), ("gaussian_copula", 0.6), ("independence", 0.0)]:
        parts.append(
            pd.DataFrame(
                {
                    "origin": origins,
                    "model": model,
                    "horizon": 1,
                    "feature_cutoff_date": calendar[p - 1],
                    "target_end": calendar[p + 1],
                    "available_date": calendar[p + 1],
                    "y_qqq": y[:, 0],
                    "y_spx": y[:, 1],
                    "mu_qqq": mu[:, 0],
                    "mu_spx": mu[:, 1],
                    "h_qqq": h[:, 0],
                    "h_spx": h[:, 1],
                    "rho": rho,
                    "fit_origin": calendar[fit],
                    "training_cutoff": calendar[fit - 1],
                    "train_n": 1000,
                    "phase": np.where(origins <= "2019-12-31", "development", "evaluation"),
                    "offset": p % 5,
                }
            )
        )
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["origin", "model"])
        .reset_index(drop=True),
        calendar,
        prior_rows(),
        protocol(),
    )


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


class JointCopulaScoreTests(unittest.TestCase):
    def run_fake(self, args):
        with patch(
            "src.joint_copula_score.masked_mean_inference", side_effect=fake_inference
        ) as sampled:
            result = evaluate(*args)
        return result, sampled

    def test_full_joint_density_matches_distribution_oracles(self):
        p = generated_inputs()[0].iloc[:12]
        result = _density_components(p)
        z = (p[["y_qqq", "y_spx"]].to_numpy() - p[["mu_qqq", "mu_spx"]].to_numpy()) / np.sqrt(
            p[["h_qqq", "h_spx"]].to_numpy() * 0.75
        )
        marginal = t.logpdf(z, 8) - np.log(np.sqrt(p[["h_qqq", "h_spx"]].to_numpy() * 0.75))
        expected = []
        for row, model, rho in zip(z, p.model, p.rho, strict=True):
            if model == "independence":
                value = 0.0
            elif model == "t8_copula":
                value = (
                    multivariate_t.logpdf(row, shape=[[1, rho], [rho, 1]], df=8)
                    - t.logpdf(row, 8).sum()
                )
            else:
                normal = norm.ppf(t.cdf(row, 8))
                value = (
                    multivariate_normal.logpdf(normal, cov=[[1, rho], [rho, 1]])
                    - norm.logpdf(normal).sum()
                )
            expected.append(value)
        np.testing.assert_allclose(result["log_copula"], expected, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(
            result["marginal_log_density"], marginal, atol=1e-12, rtol=1e-12
        )
        np.testing.assert_allclose(
            result["loss"], -marginal.sum(axis=1) - expected, atol=1e-10, rtol=1e-10
        )

    def test_common_marginal_cancellation_and_unit_transport(self):
        args = list(generated_inputs())
        before, _ = self.run_fake(args)
        scale = 10000.0
        for name in ["y_qqq", "y_spx", "mu_qqq", "mu_spx"]:
            args[0][name] *= scale
        for name in ["h_qqq", "h_spx"]:
            args[0][name] *= scale**2
        after, _ = self.run_fake(args)
        for a, b in zip(before["rows"], after["rows"], strict=True):
            for x, y in zip(a["phases"], b["phases"], strict=True):
                self.assertAlmostEqual(x["mean"], y["mean"], places=11)
                self.assertAlmostEqual(
                    y["candidate_loss"] - x["candidate_loss"], 2 * np.log(scale), places=10
                )
                self.assertAlmostEqual(
                    x["candidate_loss"] - x["control_loss"], x["mean"], places=11
                )

    def test_negative_full_density_loss_is_valid(self):
        p = generated_inputs()[0].iloc[:6].copy()
        p[["mu_qqq", "mu_spx", "y_qqq", "y_spx"]] = 0.0
        p[["h_qqq", "h_spx"]] = 1e-8
        self.assertTrue((_density_components(p)["loss"] < 0).all())

    def test_three_models_two_comparisons_151_family_and_qualification(self):
        args = generated_inputs()
        before = copy.deepcopy(args)
        result, sampled = self.run_fake(args)
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["leads"], ["joint_copula"])
        self.assertEqual(result["hypothesis_count"], 2)
        self.assertEqual(result["cumulative_hypothesis_count"], 151)
        self.assertEqual(
            [r["control"] for r in result["rows"]], ["gaussian_copula", "independence"]
        )
        self.assertEqual(result["inherited_rows"], args[2])
        self.assertEqual(sampled.call_count, 4)
        self.assertIn("QQQ", result["evidence_limitation"])
        self.assertIn("SPX", result["evidence_limitation"])
        for row in result["rows"]:
            self.assertEqual(row["score"], "negative_joint_log_density")
            self.assertEqual(row["horizon"], 1)
            self.assertAlmostEqual(row["p_holm_wave"], 2e-10)
            self.assertAlmostEqual(row["p_holm_cumulative"], 151e-10)
        pd.testing.assert_frame_equal(args[0], before[0])
        self.assertEqual(args[2:], before[2:])

    def test_phase_calendar_gaps_final_origin_and_shared_rng(self):
        args = generated_inputs()
        result, sampled = self.run_fake(args)
        seen = {}
        for call in sampled.call_args_list:
            values, mask = call.args
            self.assertTrue(np.isnan(np.asarray(values)[~mask]).all())
            self.assertGreater(len(mask), int(mask.sum()))
            self.assertEqual(call.kwargs["blocks"], [21, 63, 126])
            self.assertEqual(call.kwargs["hac_lags"], 126)
            self.assertEqual(call.kwargs["draws"], 399999)
            seen.setdefault(call.kwargs["seed"], []).append(mask)
        self.assertEqual(set(seen), {20260909, 20270909})
        for masks in seen.values():
            np.testing.assert_array_equal(masks[0], masks[1])
        for phase in result["rows"][0]["phases"]:
            a, b = args[3]["forecast"][phase["name"]]
            self.assertEqual(
                phase["full_calendar_n"], int(((args[1] >= a) & (args[1] <= b)).sum())
            )
            self.assertEqual([g["offset"] for g in phase["offsets"]], list(range(5)))

    def test_all_support_checks_precede_sampling(self):
        for scope in ["phase", "slice", "offset"]:
            args = list(generated_inputs())
            p = args[0]
            common = p[p.model == "t8_copula"]
            if scope == "phase":
                bad = common[common.phase == "development"].origin.iloc[504:]
            elif scope == "slice":
                bad = common[common.origin.between("2020-01-01", "2022-12-31")].origin.iloc[
                    251:
                ]
            else:
                bad = common[
                    (common.phase == "evaluation") & (common.offset == 0)
                ].origin.iloc[62:]
            args[0] = p[~p.origin.isin(bad)].copy()
            with (
                patch("src.joint_copula_score.masked_mean_inference") as sampled,
                self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"),
            ):
                evaluate(*args)
            sampled.assert_not_called()

    def test_pairing_identity_and_conservative_lag_guards(self):
        for defect in [
            "missing",
            "duplicate",
            "mean",
            "variance",
            "rho",
            "independence_rho",
            "target",
            "available",
            "cutoff",
            "offset",
            "train",
            "horizon",
            "fit",
        ]:
            args = list(generated_inputs())
            p = args[0]
            if defect == "missing":
                args[0] = p.iloc[1:].copy()
            elif defect == "duplicate":
                args[0] = pd.concat([p, p.iloc[:1]], ignore_index=True)
            elif defect == "mean":
                p.loc[0, "mu_qqq"] += 0.01
            elif defect == "variance":
                p.loc[0, "h_spx"] = 0.0
            elif defect == "rho":
                p.loc[0, "rho"] = 0.996
            elif defect == "independence_rho":
                p.loc[p.model == "independence", "rho"] = 0.1
            elif defect == "target":
                p.loc[0, "target_end"] = p.loc[0, "origin"]
            elif defect == "available":
                p.loc[0, "available_date"] = p.loc[0, "origin"]
            elif defect == "cutoff":
                p.loc[0, "feature_cutoff_date"] = p.loc[0, "origin"]
            elif defect == "offset":
                p.loc[0, "offset"] = (p.loc[0, "offset"] + 1) % 5
            elif defect == "train":
                p.loc[0, "train_n"] = 999
            elif defect == "horizon":
                p.loc[0, "horizon"] = 5
            else:
                p.loc[0, "training_cutoff"] = p.loc[0, "fit_origin"]
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                self.run_fake(args)

    def test_finite_extreme_standardized_tails_and_invalid_values(self):
        p = generated_inputs()[0].iloc[:9].copy()
        p[["mu_qqq", "mu_spx"]] = 0.0
        p[["h_qqq", "h_spx"]] = 1.0
        p["y_qqq"] = 1e200
        p["y_spx"] = -1e200
        out = _density_components(p)
        self.assertTrue(np.isfinite(out["loss"]).all())
        self.assertTrue(np.isfinite(out["log_copula"]).all())
        for value in [np.inf, np.nan, True]:
            bad = p.copy()
            bad["h_spx"] = value
            with self.assertRaises(ValueError):
                _density_components(bad)

    def test_fixed_protocol_and_exact_probability_effect_boundaries(self):
        good = {
            "p_holm_wave": 0.05 / (27 * 28),
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
        self.assertTrue(_passes(good, protocol()))
        bad = copy.deepcopy(good)
        bad["phases"][0]["mean"] = -0.00499
        self.assertFalse(_passes(bad, protocol()))
        bad = copy.deepcopy(good)
        bad["phases"][0]["offsets"][0]["mean"] = 0.0
        self.assertFalse(_passes(bad, protocol()))
        for branch, key, value in [
            ("inference", "seed", 1),
            ("inference", "bootstrap_draws", 99),
            ("forecast", "origin_end", "2025-10-20"),
            ("support", "phase_daily", 1),
            ("comparisons", "inherited", 148),
        ]:
            args = list(generated_inputs())
            args[3][branch][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.run_fake(args)

    def test_last_control_worst_method_veto(self):
        count = 0

        def bad(*a, **k):
            nonlocal count
            count += 1
            result = fake_inference(*a, **k)
            if count == 4:
                result["hac"]["p"] = result["p_conservative"] = 0.9
            return result

        with patch("src.joint_copula_score.masked_mean_inference", side_effect=bad):
            result = evaluate(*generated_inputs())
        self.assertEqual(result["leads"], [])
        self.assertEqual(result["rows"][1]["p_conservative"], 0.9)

    def test_inherited_precision_rows_and_failure_preservation(self):
        old = {
            "status": "COMPLETED",
            "hypothesis_count": 3,
            "cumulative_hypothesis_count": 149,
            "inherited_rows": prior_rows(146),
            "rows": [
                {
                    "study": "precision_gate",
                    "candidate": "contextual",
                    "control": c,
                    "horizon": 5,
                    "score": "qlike",
                    "verdict": "DOES_NOT_QUALIFY",
                    "p_conservative": 1.0,
                    "p_holm_wave": 1.0,
                    "p_holm_cumulative": 1.0,
                }
                for c in ["base", "adaptive", "constant"]
            ],
        }
        inherited = inherit_family(old, "b" * 64)
        self.assertEqual(len(inherited), 149)
        self.assertEqual(inherited[:146], old["inherited_rows"])
        self.assertEqual(
            inherited[-1]["source"], "reports/precision_gate/predictive/metrics.json"
        )
        result = failure_metrics(
            ValueError("INSUFFICIENT_DATA: generated"), "c" * 64, inherited
        )
        self.assertEqual(result["status"], "UNEVALUABLE")
        self.assertEqual(result["inherited_rows"], inherited)
        self.assertEqual(result["cumulative_hypothesis_count"], 151)
        self.assertEqual(len(result["rows"]), 2)
        for row in result["rows"]:
            self.assertEqual(row["status"], "INSUFFICIENT_DATA")
            for key in ["p_conservative", "p_holm_wave", "p_holm_cumulative"]:
                self.assertEqual(row[key], 1.0)
        old["rows"].reverse()
        with self.assertRaises(ValueError):
            inherit_family(old, "b" * 64)


if __name__ == "__main__":
    unittest.main()
