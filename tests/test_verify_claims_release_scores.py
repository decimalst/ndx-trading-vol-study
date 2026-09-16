"""Generated independent score contracts; no empirical files or producer imports."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import norm

from src import verify_claims_release_scores as verify


def explicit_bootstrap(values, block, draws, seed):
    """Individual circular observations, independent of prefix-sum production."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(draws, n // block))
    tails = rng.integers(0, n, size=draws) if n % block else None
    result = []
    for draw in range(draws):
        positions = [(int(start) + j) % n for start in starts[draw] for j in range(block)]
        if tails is not None:
            positions += [(int(tails[draw]) + j) % n for j in range(n % block)]
        result.append(np.mean(values[positions], axis=0))
    return np.asarray(result)


def hac_oracle(values, lags):
    values = np.asarray(values, dtype=float)
    n, mean = len(values), float(np.mean(values))
    centered = values - mean
    lag = min(lags, n - 1)
    # Serialized p-values retain the declared NumPy reduction rounding. A
    # separate scalar-loop kernel test below checks the underlying formula.
    longvar = float(centered @ centered / n)
    for k in range(1, lag + 1):
        longvar += float(2 * (1 - k / (lag + 1)) * (centered[k:] @ centered[:-k]) / n)
    se = float(np.sqrt(max(longvar, 0) / n))
    return {
        "se": se,
        "p": float(2 * norm.sf(abs(mean) / se)) if se else float(mean == 0),
        "ci95": [mean - 1.96 * se, mean + 1.96 * se],
        "mde80_nominal": float((norm.ppf(0.975) + norm.ppf(0.8)) * se),
    }


def holm_oracle(values):
    result = [None] * len(values)
    previous = 0.0
    for i, position in enumerate(sorted(range(len(values)), key=lambda j: values[j])):
        previous = max(previous, min(1.0, (len(values) - i) * values[position]))
        result[position] = previous
    return result


def fixture():
    calendar = pd.bdate_range("2019-11-01", periods=150, name="date")
    final = calendar[-1].strftime("%Y-%m-%d")
    protocol = {
        "forecast": {
            "development": ["2019-11-01", "2019-12-31"],
            "evaluation": ["2020-01-01", final],
            "source_end": final,
        },
        "evaluation_stability": [["2020-01-01", "2020-02-29"], ["2020-03-01", final]],
        "inference": {
            "blocks": [3, 7, 11],
            "bootstrap_draws": 79,
            "seed": 20260929,
            "hac_lags": 7,
            "minimum_phase_observations": 10,
            "minimum_phase_releases": 3,
            "minimum_slice_observations": 10,
            "minimum_slice_releases": 3,
            "minimum_offset_observations": 3,
            "effect_threshold_absolute": 0.005,
            "wave_alpha": 0.05,
            "cumulative_alpha": 0.05,
        },
    }
    prior = [
        {
            "study": f"prior_{i}",
            "p_conservative": 0.0,
            "status": "COMPLETED",
            "source": f"synthetic_provenance_{i}",
        }
        for i in range(140)
    ]
    records = []
    for position, origin in enumerate(calendar[:-5]):
        if position in (7, 29, 53, 78, 111):
            continue
        phase = "development" if origin <= pd.Timestamp("2019-12-31") else "evaluation"
        target_end = calendar[position + 5]
        if phase == "development" and target_end > pd.Timestamp("2019-12-31"):
            continue
        reference = origin - pd.Timedelta(days=(origin.dayofweek + 2) % 7)
        for model, prediction in (("market", 3.0), ("matched", 2.0), ("candidate", 1.0)):
            ratio = 1 / prediction
            records.append(
                {
                    "origin": origin,
                    "model": model,
                    "prediction": prediction,
                    "y": 1.0,
                    "loss": ratio - np.log(ratio) - 1,
                    "target_end": target_end,
                    "phase": phase,
                    "claim_reference_week": reference,
                    "offset": position % 5,
                    "fit_origin": origin,
                    "training_cutoff": origin - pd.Timedelta(days=1),
                    "train_n": 1000,
                    "train_releases": 200,
                }
            )
    return pd.DataFrame(records), calendar, prior, protocol


def expected_metrics(panel, calendar, prior, protocol):
    """Generated expected artifact using direct observation resampling and grouping."""
    inf = protocol["inference"]
    candidate = panel[panel.model == "candidate"].sort_values("origin").set_index("origin")
    rows = [
        {
            "study": "claims_release",
            "candidate": "candidate",
            "control": control,
            "horizon": 5,
            "score": "qlike",
            "phases": [],
        }
        for control in ("matched", "market")
    ]
    for phase_code, phase_name in enumerate(("development", "evaluation")):
        part = candidate[candidate.phase == phase_name]
        origins = part.index
        candidate_loss = (
            part.y / part.prediction - np.log(part.y / part.prediction) - 1
        ).to_numpy()
        controls, delta = [], []
        for control in ("matched", "market"):
            other = panel[panel.model == control].set_index("origin").loc[origins]
            loss = (
                other.y / other.prediction - np.log(other.y / other.prediction) - 1
            ).to_numpy()
            controls.append(loss)
            delta.append(candidate_loss - loss)
        d = np.column_stack(delta)
        draws = {
            str(b): explicit_bootstrap(
                d, b, inf["bootstrap_draws"], inf["seed"] + 5_000_000 + phase_code * 10_000 + b
            )
            for b in inf["blocks"]
        }
        for j, row in enumerate(rows):
            mean = float(np.mean(d[:, j]))
            blocks = {
                str(b): {
                    "p": float(
                        (1 + np.count_nonzero(np.abs(draws[str(b)][:, j] - mean) >= abs(mean)))
                        / (inf["bootstrap_draws"] + 1)
                    ),
                    "ci95": list(np.quantile(draws[str(b)][:, j], [0.025, 0.975])),
                }
                for b in inf["blocks"]
            }
            hac = hac_oracle(d[:, j], inf["hac_lags"])
            annual = []
            start, end = map(pd.Timestamp, protocol["forecast"][phase_name])
            for year in range(start.year, end.year + 1):
                chosen = origins.year == year
                n = int(chosen.sum())
                calendar_n = int(
                    ((calendar >= start) & (calendar <= end) & (calendar.year == year)).sum()
                )
                annual.append(
                    {
                        "year": year,
                        "n": n,
                        "distinct_releases": int(
                            part.loc[chosen, "claim_reference_week"].nunique()
                        ),
                        "delta": float(np.mean(d[chosen, j])) if n else None,
                        "calendar_origins": calendar_n,
                        "missing_origins": calendar_n - n,
                    }
                )
            offsets = []
            positions = calendar.get_indexer(origins) % 5
            for offset in range(5):
                chosen = positions == offset
                offsets.append(
                    {
                        "offset": offset,
                        "n": int(chosen.sum()),
                        "delta": float(np.mean(d[chosen, j])),
                    }
                )
            stability = []
            if phase_name == "evaluation":
                for a, z in protocol["evaluation_stability"]:
                    chosen = (origins >= pd.Timestamp(a)) & (origins <= pd.Timestamp(z))
                    stability.append(
                        {
                            "start": a,
                            "end": z,
                            "n": int(chosen.sum()),
                            "distinct_releases": int(
                                part.loc[chosen, "claim_reference_week"].nunique()
                            ),
                            "delta": float(np.mean(d[chosen, j])),
                        }
                    )
            within_report = []
            for release in part.claim_reference_week.unique():
                chosen = part.claim_reference_week.to_numpy() == release
                within_report.append(float(np.mean(d[chosen, j])))
            row["phases"].append(
                {
                    "name": phase_name,
                    "n": len(part),
                    "distinct_releases": int(part.claim_reference_week.nunique()),
                    "delta": mean,
                    "candidate_loss": float(np.mean(candidate_loss)),
                    "control_loss": float(np.mean(controls[j])),
                    "first_origin": origins[0].strftime("%Y-%m-%d"),
                    "last_origin": origins[-1].strftime("%Y-%m-%d"),
                    "block_inference": blocks,
                    "hac": hac,
                    "p_conservative": max([hac["p"], *[x["p"] for x in blocks.values()]]),
                    "ci95_envelope": [
                        min([hac["ci95"][0], *[x["ci95"][0] for x in blocks.values()]]),
                        max([hac["ci95"][1], *[x["ci95"][1] for x in blocks.values()]]),
                    ],
                    "release_weighted_delta": float(np.mean(within_report)),
                    "annual": annual,
                    "offsets": offsets,
                    "stability": stability,
                }
            )
    ps = [max(p["p_conservative"] for p in row["phases"]) for row in rows]
    waves, totals = (
        holm_oracle(ps),
        holm_oracle([r["p_conservative"] for r in prior] + ps)[-2:],
    )
    for row, p, wave, total in zip(rows, ps, waves, totals):
        ok = wave < inf["wave_alpha"] and total < inf["cumulative_alpha"]
        for phase in row["phases"]:
            ok &= (
                phase["delta"] <= -inf["effect_threshold_absolute"]
                and phase["release_weighted_delta"] < 0
            )
            ok &= all(x["delta"] < 0 for x in phase["offsets"] + phase["stability"])
        row.update(
            p_conservative=p,
            p_holm_wave=wave,
            p_holm_cumulative=total,
            verdict="COMPARISON_GATE_PASS" if ok else "DOES_NOT_QUALIFY",
        )
    return {
        "status": "COMPLETED",
        "rows": rows,
        "inherited_rows": copy.deepcopy(prior),
        "leads": ["claims_first_report"]
        if all(r["verdict"] == "COMPARISON_GATE_PASS" for r in rows)
        else [],
        "hypothesis_count": 2,
        "cumulative_hypothesis_count": 142,
        "common_scored_origins": len(candidate),
    }


def refresh_losses(panel):
    ratio = panel.y / panel.prediction
    panel["loss"] = ratio - np.log(ratio) - 1


class CompleteScoreVerificationTests(unittest.TestCase):
    def setUp(self):
        self.panel, self.calendar, self.prior, self.protocol = fixture()
        self.metrics = expected_metrics(self.panel, self.calendar, self.prior, self.protocol)

    def check(self, panel=None, metrics=None, prior=None, protocol=None):
        return verify.verify_scores(
            self.panel if panel is None else panel,
            self.calendar,
            self.metrics if metrics is None else metrics,
            self.prior if prior is None else prior,
            self.protocol if protocol is None else protocol,
        )

    def test_complete_generated_artifact_and_joint_lead_verify(self):
        self.assertEqual(self.metrics["leads"], ["claims_first_report"])
        self.assertEqual(
            self.check(),
            {
                "status": "VERIFIED",
                "hypothesis_count": 2,
                "cumulative_hypothesis_count": 142,
                "common_scored_origins": self.panel.origin.nunique(),
            },
        )

    def test_valid_zero_effect_is_negative_completed_result_not_support_failure(self):
        panel = self.panel.copy()
        panel["prediction"] = 1.0
        refresh_losses(panel)
        metrics = expected_metrics(panel, self.calendar, self.prior, self.protocol)
        self.assertEqual(metrics["leads"], [])
        self.assertTrue(all(r["verdict"] == "DOES_NOT_QUALIFY" for r in metrics["rows"]))
        self.assertEqual(self.check(panel=panel, metrics=metrics)["status"], "VERIFIED")

    def test_one_failed_comparison_never_produces_joint_headline(self):
        panel = self.panel.copy()
        panel.loc[panel.model == "matched", "prediction"] = 1.01
        refresh_losses(panel)
        metrics = expected_metrics(panel, self.calendar, self.prior, self.protocol)
        self.assertEqual(
            [r["verdict"] for r in metrics["rows"]],
            ["DOES_NOT_QUALIFY", "COMPARISON_GATE_PASS"],
        )
        self.assertEqual(self.check(panel=panel, metrics=metrics)["status"], "VERIFIED")
        metrics["leads"] = ["claims_first_report"]
        with self.assertRaises(ValueError):
            self.check(panel=panel, metrics=metrics)

    def test_significance_thresholds_are_strict_and_cumulative_142_is_recomputed(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["inference"]["wave_alpha"] = self.metrics["rows"][0]["p_holm_wave"]
        metrics = expected_metrics(self.panel, self.calendar, self.prior, protocol)
        self.assertEqual(metrics["leads"], [])
        self.assertEqual(self.check(metrics=metrics, protocol=protocol)["status"], "VERIFIED")
        prior = copy.deepcopy(self.prior)
        for r in prior:
            r["p_conservative"] = 1.0
        metrics = expected_metrics(self.panel, self.calendar, prior, self.protocol)
        self.assertEqual(metrics["leads"], [])
        self.assertEqual(self.check(metrics=metrics, prior=prior)["status"], "VERIFIED")

    def test_offset_sign_gate_uses_unfiltered_calendar_not_paired_row_counter(self):
        panel = self.panel.copy()
        panel.loc[(panel.model == "candidate") & (panel.offset == 0), "prediction"] = 2.0
        refresh_losses(panel)
        metrics = expected_metrics(panel, self.calendar, self.prior, self.protocol)
        self.assertEqual(metrics["rows"][0]["phases"][0]["offsets"][0]["delta"], 0)
        self.assertEqual(metrics["rows"][0]["verdict"], "DOES_NOT_QUALIFY")
        self.assertEqual(self.check(panel=panel, metrics=metrics)["status"], "VERIFIED")
        compressed = {origin: i % 5 for i, origin in enumerate(sorted(panel.origin.unique()))}
        panel["offset"] = panel.origin.map(compressed)
        with self.assertRaises(ValueError):
            self.check(panel=panel, metrics=metrics)

    def test_equal_release_weighting_can_fail_despite_negative_daily_mean(self):
        panel = self.panel.copy()
        protocol = copy.deepcopy(self.protocol)
        protocol["inference"]["minimum_phase_releases"] = 1
        protocol["inference"]["minimum_slice_releases"] = 1
        panel["claim_reference_week"] = pd.Timestamp("2019-10-26")
        for phase in ("development", "evaluation"):
            origins = sorted(panel.loc[panel.phase == phase, "origin"].unique())
            for i, origin in enumerate(origins[:3]):
                panel.loc[panel.origin == origin, "claim_reference_week"] = pd.Timestamp(
                    "2019-10-05"
                ) + pd.Timedelta(days=7 * i)
                panel.loc[
                    (panel.origin == origin) & (panel.model == "candidate"), "prediction"
                ] = 4.0
        refresh_losses(panel)
        metrics = expected_metrics(panel, self.calendar, self.prior, protocol)
        self.assertLess(metrics["rows"][0]["phases"][0]["delta"], -0.005)
        self.assertGreater(metrics["rows"][0]["phases"][0]["release_weighted_delta"], 0)
        self.assertEqual(metrics["rows"][0]["verdict"], "DOES_NOT_QUALIFY")
        self.assertEqual(
            self.check(panel=panel, metrics=metrics, protocol=protocol)["status"], "VERIFIED"
        )

    def test_each_support_floor_fails_entire_wave(self):
        for key in (
            "minimum_phase_observations",
            "minimum_phase_releases",
            "minimum_slice_observations",
            "minimum_slice_releases",
            "minimum_offset_observations",
        ):
            protocol = copy.deepcopy(self.protocol)
            protocol["inference"][key] = 10000
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"),
            ):
                self.check(protocol=protocol)

    def test_duplicate_missing_unknown_model_or_one_arm_metadata_disagreement_rejected(self):
        variants = [
            pd.concat([self.panel, self.panel.iloc[:1]], ignore_index=True),
            self.panel.iloc[1:].copy(),
        ]
        for field, value in (
            ("model", "unknown"),
            ("y", 1.1),
            ("phase", "evaluation"),
            ("claim_reference_week", pd.Timestamp("2019-10-05")),
            ("train_n", 1001),
        ):
            bad = self.panel.copy()
            bad.loc[0, field] = value
            variants.append(bad)
        for bad in variants:
            with self.subTest(n=len(bad)), self.assertRaises(ValueError):
                self.check(panel=bad)

    def test_losses_are_reconstructed_and_nonpositive_or_nonfinite_inputs_fail(self):
        for field, value in (
            ("loss", 123.0),
            ("prediction", 0.0),
            ("prediction", np.inf),
            ("prediction", np.nan),
            ("y", -1.0),
        ):
            bad = self.panel.copy()
            bad.loc[0, field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.check(panel=bad)

    def test_target_endpoint_phase_and_maturity_are_not_trusted(self):
        for mode in ("wrong_end", "dev_boundary", "wrong_phase"):
            bad = self.panel.copy()
            first = bad.origin.min()
            if mode == "wrong_end":
                bad.loc[bad.origin == first, "target_end"] = self.calendar[6]
            elif mode == "wrong_phase":
                bad.loc[bad.origin == first, "phase"] = "evaluation"
            else:
                origin = pd.Timestamp("2019-12-30")
                position = self.calendar.get_loc(origin)
                extra = bad[bad.origin == first].copy()
                extra["origin"], extra["target_end"], extra["offset"] = (
                    origin,
                    self.calendar[position + 5],
                    position % 5,
                )
                bad = pd.concat([bad, extra], ignore_index=True)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.check(panel=bad)

    def test_metrics_fields_counts_intervals_stability_accounting_and_leads_are_verified(self):
        paths = [
            ("common_scored_origins",),
            ("hypothesis_count",),
            ("cumulative_hypothesis_count",),
            ("rows", 0, "phases", 0, "delta"),
            ("rows", 0, "phases", 0, "candidate_loss"),
            ("rows", 0, "phases", 0, "control_loss"),
            ("rows", 0, "phases", 0, "n"),
            ("rows", 0, "phases", 0, "distinct_releases"),
            ("rows", 0, "phases", 0, "hac", "se"),
            ("rows", 0, "phases", 0, "hac", "mde80_nominal"),
            ("rows", 0, "phases", 0, "ci95_envelope", 0),
            ("rows", 0, "phases", 0, "block_inference", "3", "ci95", 1),
            ("rows", 0, "phases", 0, "release_weighted_delta"),
            ("rows", 0, "phases", 0, "annual", 0, "missing_origins"),
            ("rows", 0, "phases", 0, "annual", 0, "calendar_origins"),
            ("rows", 0, "phases", 0, "offsets", 0, "n"),
            ("rows", 0, "phases", 1, "stability", 0, "delta"),
        ]
        for path in paths:
            bad = copy.deepcopy(self.metrics)
            node = bad
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] += 1
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.check(metrics=bad)

    def test_pvalues_compare_exactly_and_not_under_general_float_tolerance(self):
        for path in (
            ("p_holm_wave",),
            ("p_holm_cumulative",),
            ("p_conservative",),
            ("phases", 0, "p_conservative"),
            ("phases", 0, "hac", "p"),
            ("phases", 0, "block_inference", "3", "p"),
        ):
            bad = copy.deepcopy(self.metrics)
            node = bad["rows"][0]
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = float(np.nextafter(node[path[-1]], 1.0))
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.check(metrics=bad)

    def test_exact_schemas_identities_prior_preservation_and_boolean_count_rejected(self):
        for mode in (
            "extra",
            "missing",
            "row_identity",
            "prior_changed",
            "prior_dropped",
            "bool_count",
            "lead",
            "row_order",
        ):
            bad = copy.deepcopy(self.metrics)
            if mode == "extra":
                bad["rows"][0]["unknown"] = 1
            elif mode == "missing":
                del bad["rows"][0]["phases"][0]["annual"]
            elif mode == "row_identity":
                bad["rows"][0]["candidate"] = "different"
            elif mode == "prior_changed":
                bad["inherited_rows"][0]["source"] = "altered"
            elif mode == "prior_dropped":
                bad["inherited_rows"].pop()
            elif mode == "bool_count":
                bad["rows"][0]["phases"][0]["offsets"][0]["n"] = True
            elif mode == "lead":
                bad["leads"] = []
            else:
                bad["rows"].reverse()
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.check(metrics=bad)

    def test_prior_family_must_have_exactly_140_finite_pvalues(self):
        for bad in (self.prior[:-1], self.prior + [self.prior[0]]):
            with self.assertRaises(ValueError):
                self.check(prior=bad)
        for p in (-0.1, 1.1, np.nan, np.inf, True):
            prior = copy.deepcopy(self.prior)
            prior[0]["p_conservative"] = p
            with self.subTest(p=p), self.assertRaises(ValueError):
                self.check(prior=prior)

    def test_reordering_input_and_verification_do_not_mutate_supplied_artifacts(self):
        before = copy.deepcopy((self.panel, self.metrics, self.prior, self.protocol))
        self.assertEqual(
            self.check(panel=self.panel.sample(frac=1, random_state=42))["status"], "VERIFIED"
        )
        pd.testing.assert_frame_equal(self.panel, before[0])
        self.assertEqual((self.metrics, self.prior, self.protocol), before[1:])


class IndependentInferenceTests(unittest.TestCase):
    def test_shared_circular_draws_exact_remainder_and_rng_order(self):
        x = np.sin(np.arange(17)) + np.arange(17) / 20
        values = np.column_stack([x, 2 * x + 7])
        expected = explicit_bootstrap(values, 5, 97, 73019)
        actual = verify._bootstrap_means(values, 5, 97, 73019)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(actual[:, 1], 2 * actual[:, 0] + 7, atol=1e-13)
        np.testing.assert_array_equal(actual, verify._bootstrap_means(values, 5, 97, 73019))

    def test_exact_block_division_uses_no_extra_tail_sample(self):
        x = np.arange(24, dtype=float).reshape(12, 2) / 7
        np.testing.assert_allclose(
            verify._bootstrap_means(x, 3, 71, 92),
            explicit_bootstrap(x, 3, 71, 92),
            rtol=1e-13,
            atol=1e-13,
        )

    def test_constant_bootstrap_means_and_full_length_block(self):
        values = np.full((17, 2), [3.5, -2.25])
        np.testing.assert_allclose(
            verify._bootstrap_means(values, 5, 31, 5),
            np.tile(values[0], (31, 1)),
            rtol=0,
            atol=1e-14,
        )
        x = np.column_stack([np.arange(17), np.arange(17) ** 2])
        np.testing.assert_allclose(
            verify._bootstrap_means(x, 17, 31, 5), np.tile(x.mean(axis=0), (31, 1)), atol=1e-12
        )

    def test_bootstrap_p_uses_centered_null_two_sides_ties_and_plus_one(self):
        means = np.array([[-2.0, 0.1], [-1.0, 0.2], [0.0, 0.3], [1.0, 0.4]])
        center = np.array([-1.0, 0.25])
        actual = verify._bootstrap_p(means, center)
        np.testing.assert_array_equal(actual, np.array([4 / 5, 1 / 5]))
        np.testing.assert_array_equal(
            verify._bootstrap_p(np.zeros((4, 2)), np.zeros(2)), np.ones(2)
        )

    def test_hac_matches_direct_bartlett_autocovariance_formula(self):
        x = np.sin(np.arange(37) / 3) + np.cos(np.arange(37) / 7) - 0.05
        center = x - np.mean(x)
        for lags in (0, 7, 126):
            lag = min(lags, len(x) - 1)
            longvar = sum(float(v * v) for v in center) / len(x)
            for k in range(1, lag + 1):
                autocov = sum(
                    float(center[t] * center[t - k]) for t in range(k, len(x))
                ) / len(x)
                longvar += 2 * (1 - k / (lag + 1)) * autocov
            se = np.sqrt(max(longvar, 0) / len(x))
            actual = verify._hac(x, lags)
            self.assertEqual(set(actual), {"se", "p", "ci95", "mde80_nominal"})
            self.assertAlmostEqual(actual["se"], se, places=14)
            self.assertAlmostEqual(actual["p"], 2 * norm.sf(abs(x.mean()) / se), places=14)
            np.testing.assert_allclose(
                actual["ci95"], x.mean() + np.array([-1, 1]) * 1.96 * se, atol=1e-14
            )
            self.assertAlmostEqual(
                actual["mde80_nominal"], (norm.ppf(0.975) + norm.ppf(0.8)) * se, places=14
            )

    def test_hac_zero_scale_retains_zero_and_nonzero_mean_p_conventions(self):
        for value, p in ((0.0, 1.0), (-0.25, 0.0), (0.25, 0.0)):
            actual = verify._hac(np.full(30, value), 126)
            self.assertEqual(
                actual, {"se": 0.0, "p": p, "ci95": [value, value], "mde80_nominal": 0.0}
            )

    def test_holm_order_monotonicity_and_two_versus_142_families(self):
        np.testing.assert_array_equal(
            verify._holm([0.02, 0.004, 0.008, 0.7]), [0.04, 0.016, 0.024, 0.7]
        )
        new = np.array([1e-8, 1e-7])
        np.testing.assert_array_equal(verify._holm(new), [2e-8, 1e-7])
        total = verify._holm([1.0] * 140 + list(new))
        np.testing.assert_array_equal(total[-2:], [142 * 1e-8, 141 * 1e-7])
        np.testing.assert_array_equal(total[:140], np.ones(140))

    def test_offsets_are_full_calendar_positions_before_filtering(self):
        calendar = pd.bdate_range("2019-01-02", periods=12)
        origins = calendar[[0, 2, 5, 6, 10]]
        np.testing.assert_array_equal(verify._offsets(origins, calendar), [0, 2, 0, 1, 0])
        self.assertNotEqual(
            list(verify._offsets(origins, calendar)), list(np.arange(len(origins)) % 5)
        )
        for unit in ("ms", "us", "ns"):
            np.testing.assert_array_equal(
                verify._offsets(origins.as_unit(unit), calendar.as_unit(unit)), [0, 2, 0, 1, 0]
            )

    def test_invalid_numerical_kernel_inputs_fail(self):
        for bad in ([-0.1, 0.2], [0.1, 1.1], [0.1, np.nan], [0.1, np.inf]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                verify._holm(bad)
        for block, draws in ((0, 2), (5, 2), (2, 0), (True, 2)):
            with self.subTest(block=block, draws=draws), self.assertRaises(ValueError):
                verify._bootstrap_means(np.ones((4, 2)), block, draws, 1)


def calibration_oracle(protocol):
    settings = protocol["calibration"]
    seed = protocol["inference"]["seed"]
    rng = np.random.default_rng(seed)
    hit = 0
    block_hits = {str(block): 0 for block in protocol["inference"]["blocks"]}
    paths = []
    for trial in range(settings["replications"]):
        shocks = rng.normal(
            0, np.sqrt(1 - settings["rho"] ** 2), settings["n"] + settings["burn_in"]
        )
        state = rng.normal()
        path = []
        for shock in shocks:
            state = settings["rho"] * state + shock
            path.append(state)
        values = np.asarray(path[settings["burn_in"] :])
        paths.append(values)
        lo, hi = hac_oracle(values, protocol["inference"]["hac_lags"])["ci95"]
        for block in protocol["inference"]["blocks"]:
            means = explicit_bootstrap(
                values[:, None],
                block,
                settings["bootstrap_draws"],
                seed + trial * 1000 + block,
            )[:, 0]
            a, b = np.quantile(means, [0.025, 0.975])
            block_hits[str(block)] += int(a <= 0 <= b)
            lo, hi = min(lo, a), max(hi, b)
        hit += int(lo <= 0 <= hi)
    coverage = hit / settings["replications"]
    return (
        {
            "status": "PASS" if coverage >= settings["minimum_envelope_coverage"] else "FAIL",
            **settings,
            "seed": seed,
            "envelope_coverage": coverage,
            "individual_block_coverage": {
                k: n / settings["replications"] for k, n in block_hits.items()
            },
            "limitation": "One stationary synthetic null; not market-data coverage certification",
        },
        paths,
    )


class CalibrationVerificationTests(unittest.TestCase):
    def setUp(self):
        self.protocol = fixture()[3]
        self.protocol["calibration"] = {
            "rho": 0.8,
            "n": 60,
            "burn_in": 10,
            "replications": 7,
            "bootstrap_draws": 59,
            "minimum_envelope_coverage": 0.90,
        }
        self.result, self.paths = calibration_oracle(self.protocol)

    def test_reduced_generated_calibration_reconstructs_coverage_and_status(self):
        self.assertEqual(
            verify.verify_calibration(self.protocol, self.result), {"status": "VERIFIED"}
        )

    def test_shocks_precede_stationary_state_and_block_seeds_have_no_phase_offset(self):
        with (
            patch.object(verify, "_hac", wraps=verify._hac) as hac,
            patch.object(
                verify, "_bootstrap_means", wraps=verify._bootstrap_means
            ) as bootstrap,
        ):
            verify.verify_calibration(self.protocol, self.result)
        self.assertEqual(hac.call_count, 7)
        for call, path in zip(hac.call_args_list, self.paths):
            np.testing.assert_array_equal(call.args[0], path)
        expected_seeds = [
            self.protocol["inference"]["seed"] + trial * 1000 + block
            for trial in range(7)
            for block in self.protocol["inference"]["blocks"]
        ]
        self.assertEqual([call.args[3] for call in bootstrap.call_args_list], expected_seeds)
        self.assertTrue(all(call.args[2] == 59 for call in bootstrap.call_args_list))

    def test_tampered_coverage_status_seed_design_and_schema_rejected(self):
        for field in (
            "envelope_coverage",
            "status",
            "seed",
            "rho",
            "n",
            "burn_in",
            "replications",
            "bootstrap_draws",
            "minimum_envelope_coverage",
            "limitation",
            "block",
            "extra",
        ):
            result = copy.deepcopy(self.result)
            if field == "status":
                result[field] = "FAIL" if result[field] == "PASS" else "PASS"
            elif field == "block":
                key = str(self.protocol["inference"]["blocks"][0])
                result["individual_block_coverage"][key] += 1 / 7
            elif field == "extra":
                result[field] = True
            elif field == "limitation":
                result[field] = "Overstated certification"
            else:
                result[field] += 0.1 if type(result[field]) is float else 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify.verify_calibration(self.protocol, result)


if __name__ == "__main__":
    unittest.main()
