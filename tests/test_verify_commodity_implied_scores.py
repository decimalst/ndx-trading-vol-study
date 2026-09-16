"""Prewritten independent statistical contracts on invented daily panels."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import verify_commodity_implied_scores as verify
from tests.test_verify_claims_release_scores import (
    calibration_oracle,
    explicit_bootstrap,
    hac_oracle,
    holm_oracle,
)


def fixture():
    calendar = pd.bdate_range("2019-10-01", "2020-05-29").as_unit("ns")
    protocol = {
        "forecast": {
            "origin_start": "2019-11-01",
            "origin_end": "2020-05-29",
            "development": ["2019-11-01", "2019-12-31"],
            "evaluation": ["2020-01-01", "2020-05-29"],
            "source_end": "2020-05-29",
            "minimum_train": 1000,
        },
        "evaluation_stability": [["2020-01-01", "2020-02-29"], ["2020-03-01", "2020-05-29"]],
        "inference": {
            "blocks": [3, 7, 11],
            "bootstrap_draws": 79,
            "seed": 20260930,
            "hac_lags": 7,
            "minimum_phase_observations": 10,
            "minimum_slice_observations": 10,
            "minimum_offset_observations": 3,
            "effect_threshold_absolute": 0.005,
            "wave_alpha": 0.05,
            "cumulative_alpha": 0.05,
        },
        "comparisons": {
            "controls": ["matched", "market"],
            "new_hypotheses": 2,
            "inherited_hypotheses": 142,
            "cumulative_hypotheses": 144,
        },
        "calibration": {
            "rho": 0.8,
            "n": 60,
            "burn_in": 10,
            "replications": 7,
            "bootstrap_draws": 59,
            "minimum_envelope_coverage": 0.9,
        },
    }
    prior = [
        {
            "study": f"invented_{i}",
            "candidate": "old_candidate",
            "control": "old_control",
            "horizon": 5,
            "p_conservative": 0.0,
            "provenance": f"pinned_{i}",
            "source": "invented_prior.json",
            "source_sha256": "a" * 64,
            "source_row_index": i,
        }
        for i in range(142)
    ]
    records = []
    for pos, origin in enumerate(calendar[:-5]):
        if origin < pd.Timestamp("2019-11-01") or pos in (29, 53, 78, 111):
            continue
        phase = "development" if origin <= pd.Timestamp("2019-12-31") else "evaluation"
        end = calendar[pos + 5]
        if phase == "development" and end > pd.Timestamp("2019-12-31"):
            continue
        fit = calendar[
            (calendar.to_period("M") == origin.to_period("M")) & (calendar >= "2019-11-01")
        ][0]
        cutoff = calendar[calendar.get_loc(fit) - 1]
        for model, prediction in (("market", 3.0), ("matched", 2.0), ("candidate", 1.0)):
            ratio = 1.0 / prediction
            records.append(
                {
                    "origin": origin,
                    "model": model,
                    "prediction": prediction,
                    "y": 1.0,
                    "loss": ratio - np.log(ratio) - 1,
                    "target_end": end,
                    "phase": phase,
                    "commodity_cutoff_date": calendar[pos - 1],
                    "offset": pos % 5,
                    "fit_origin": fit,
                    "training_cutoff": cutoff,
                    "train_n": 1000,
                }
            )
    return pd.DataFrame(records), calendar, prior, protocol


def refresh_losses(panel):
    ratio = panel.y / panel.prediction
    panel["loss"] = ratio - np.log(ratio) - 1


def expected_metrics(panel, calendar, prior, protocol):
    """Literal panel grouping plus direct observation resampling, no score code."""
    inf = protocol["inference"]
    candidate = panel[panel.model == "candidate"].set_index("origin").sort_index()
    rows = [
        {
            "study": "commodity_implied",
            "candidate": "candidate",
            "control": name,
            "horizon": 5,
            "score": "qlike",
            "phases": [],
        }
        for name in ("matched", "market")
    ]
    for phase_code, phase_name in enumerate(("development", "evaluation")):
        part = candidate[candidate.phase == phase_name]
        origins = part.index
        candidate_loss = (
            part.y / part.prediction - np.log(part.y / part.prediction) - 1
        ).to_numpy()
        control_losses = []
        for name in ("matched", "market"):
            other = panel[panel.model == name].set_index("origin").loc[origins]
            control_losses.append(
                (
                    other.y / other.prediction - np.log(other.y / other.prediction) - 1
                ).to_numpy()
            )
        delta = np.column_stack([candidate_loss - values for values in control_losses])
        samples = {
            str(b): explicit_bootstrap(
                delta,
                b,
                inf["bootstrap_draws"],
                inf["seed"] + 5_000_000 + phase_code * 10_000 + b,
            )
            for b in inf["blocks"]
        }
        positions = calendar.get_indexer(origins) % 5
        start, end = map(pd.Timestamp, protocol["forecast"][phase_name])
        bounded = calendar[(calendar >= start) & (calendar <= end)]
        for j, row in enumerate(rows):
            values, mean = delta[:, j], float(np.mean(delta[:, j]))
            blocks = {
                str(b): {
                    "p": float(
                        (
                            1
                            + np.count_nonzero(
                                np.abs(samples[str(b)][:, j] - mean) >= abs(mean)
                            )
                        )
                        / (inf["bootstrap_draws"] + 1)
                    ),
                    "ci95": list(np.quantile(samples[str(b)][:, j], [0.025, 0.975])),
                }
                for b in inf["blocks"]
            }
            hac = hac_oracle(values, inf["hac_lags"])
            intervals = [hac["ci95"], *[x["ci95"] for x in blocks.values()]]
            annual = []
            for year in sorted(set(bounded.year)):
                choose = origins.year == year
                count = int(choose.sum())
                calendar_n = int((bounded.year == year).sum())
                annual.append(
                    {
                        "year": int(year),
                        "n": count,
                        "delta": float(values[choose].mean()) if count else None,
                        "calendar_origins": calendar_n,
                        "missing_origins": calendar_n - count,
                    }
                )
            stability = []
            if phase_name == "evaluation":
                for a, z in protocol["evaluation_stability"]:
                    choose = (origins >= pd.Timestamp(a)) & (origins <= pd.Timestamp(z))
                    stability.append(
                        {
                            "start": a,
                            "end": z,
                            "n": int(choose.sum()),
                            "delta": float(values[choose].mean()),
                        }
                    )
            row["phases"].append(
                {
                    "name": phase_name,
                    "n": len(part),
                    "delta": mean,
                    "candidate_loss": float(candidate_loss.mean()),
                    "control_loss": float(control_losses[j].mean()),
                    "first_origin": origins[0].strftime("%Y-%m-%d"),
                    "last_origin": origins[-1].strftime("%Y-%m-%d"),
                    "block_inference": blocks,
                    "hac": hac,
                    "p_conservative": max(hac["p"], *[x["p"] for x in blocks.values()]),
                    "ci95_envelope": [
                        min(x[0] for x in intervals),
                        max(x[1] for x in intervals),
                    ],
                    "annual": annual,
                    "offsets": [
                        {
                            "offset": k,
                            "n": int((positions == k).sum()),
                            "delta": float(values[positions == k].mean()),
                        }
                        for k in range(5)
                    ],
                    "stability": stability,
                }
            )
    probabilities = [max(p["p_conservative"] for p in row["phases"]) for row in rows]
    wave = holm_oracle(probabilities)
    cumulative = holm_oracle([r["p_conservative"] for r in prior] + probabilities)[-2:]
    for j, row in enumerate(rows):
        passes = wave[j] < inf["wave_alpha"] and cumulative[j] < inf["cumulative_alpha"]
        for phase in row["phases"]:
            passes &= phase["delta"] <= -inf["effect_threshold_absolute"]
            passes &= all(x["delta"] < 0 for x in phase["offsets"] + phase["stability"])
        row.update(
            p_conservative=probabilities[j],
            p_holm_wave=wave[j],
            p_holm_cumulative=cumulative[j],
            verdict="COMPARISON_GATE_PASS" if passes else "DOES_NOT_QUALIFY",
        )
    return {
        "status": "COMPLETED",
        "rows": rows,
        "inherited_rows": copy.deepcopy(prior),
        "leads": ["commodity_implied"]
        if all(r["verdict"] == "COMPARISON_GATE_PASS" for r in rows)
        else [],
        "hypothesis_count": 2,
        "cumulative_hypothesis_count": 144,
        "common_scored_origins": len(candidate),
    }


class ScoreVerificationTests(unittest.TestCase):
    def setUp(self):
        self.panel, self.calendar, self.prior, self.protocol = fixture()
        self.metrics = expected_metrics(self.panel, self.calendar, self.prior, self.protocol)

    def check(self, **changes):
        args = dict(
            panel=self.panel,
            calendar=self.calendar,
            metrics=self.metrics,
            prior=self.prior,
            protocol=self.protocol,
        )
        args.update(changes)
        return verify.verify_scores(**args)

    def test_known_signal_verifies_both_comparisons_without_weekly_fields(self):
        self.assertEqual(self.metrics["leads"], ["commodity_implied"])
        self.assertEqual(
            self.check(),
            {
                "status": "VERIFIED",
                "hypothesis_count": 2,
                "cumulative_hypothesis_count": 144,
                "common_scored_origins": self.panel.origin.nunique(),
            },
        )
        self.assertNotIn("release", str(self.metrics))

    def test_serial_null_has_no_lead_and_is_independently_reconstructed(self):
        panel = self.panel.copy(deep=True)
        for phase in ("development", "evaluation"):
            origins = sorted(panel.loc[panel.phase == phase, "origin"].unique())
            for i, origin in enumerate(origins):
                state = 0.05 * np.sin(i * np.pi / 10)
                panel.loc[
                    (panel.origin == origin) & (panel.model == "candidate"), "prediction"
                ] = np.exp(state)
                panel.loc[
                    (panel.origin == origin) & (panel.model != "candidate"), "prediction"
                ] = np.exp(-state)
        refresh_losses(panel)
        metrics = expected_metrics(panel, self.calendar, self.prior, self.protocol)
        self.assertEqual(metrics["leads"], [])
        self.assertEqual(self.check(panel=panel, metrics=metrics)["status"], "VERIFIED")

    def test_shared_draws_use_phase_block_seed_and_full_two_column_deltas(self):
        with patch.object(
            verify, "_bootstrap_means", wraps=verify._bootstrap_means
        ) as bootstrap:
            self.check()
        self.assertEqual(
            [call.args[3] for call in bootstrap.call_args_list],
            [
                20260930 + 5_000_000 + phase * 10_000 + block
                for phase in range(2)
                for block in (3, 7, 11)
            ],
        )
        self.assertTrue(
            all(
                call.args[0].shape[1] == 2 and call.args[2] == 79
                for call in bootstrap.call_args_list
            )
        )

    def test_each_support_floor_and_block_length_fails_whole_wave(self):
        for field in (
            "minimum_phase_observations",
            "minimum_slice_observations",
            "minimum_offset_observations",
        ):
            protocol = copy.deepcopy(self.protocol)
            protocol["inference"][field] = 10000
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"),
            ):
                self.check(protocol=protocol)
        protocol = copy.deepcopy(self.protocol)
        protocol["inference"]["blocks"] = [10000]
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"):
            self.check(protocol=protocol)

    def test_effect_offset_slice_bothphase_and_bothcontrol_gates(self):
        for mode in ("effect", "offset", "slice", "phase", "control"):
            panel, protocol = self.panel.copy(deep=True), copy.deepcopy(self.protocol)
            if mode == "effect":
                protocol["inference"]["effect_threshold_absolute"] = 1.0
            elif mode == "offset":
                panel.loc[(panel.model == "candidate") & (panel.offset == 0), "prediction"] = (
                    2.0
                )
            elif mode == "slice":
                panel.loc[
                    (panel.model == "candidate") & (panel.origin >= "2020-03-01"), "prediction"
                ] = 2.0
            elif mode == "phase":
                panel.loc[
                    (panel.model == "candidate") & (panel.phase == "evaluation"), "prediction"
                ] = 2.0
            else:
                panel.loc[panel.model == "matched", "prediction"] = 1.0
            refresh_losses(panel)
            metrics = expected_metrics(panel, self.calendar, self.prior, protocol)
            self.assertEqual(metrics["leads"], [])
            with self.subTest(mode=mode):
                self.assertEqual(
                    self.check(panel=panel, protocol=protocol, metrics=metrics)["status"],
                    "VERIFIED",
                )

    def test_holm_two_and_144_corrections_each_can_block_a_lead(self):
        for mode in ("wave", "cumulative"):
            protocol, prior = copy.deepcopy(self.protocol), copy.deepcopy(self.prior)
            if mode == "wave":
                protocol["inference"]["wave_alpha"] = 0.05 / (24 * 25)
            else:
                for row in prior:
                    row["p_conservative"] = 1.0
            metrics = expected_metrics(self.panel, self.calendar, prior, protocol)
            self.assertEqual(metrics["leads"], [])
            self.assertEqual(
                self.check(metrics=metrics, prior=prior, protocol=protocol)["status"],
                "VERIFIED",
            )

    def test_every_reported_statistical_and_accounting_field_is_checked(self):
        paths = [
            ("hypothesis_count",),
            ("cumulative_hypothesis_count",),
            ("common_scored_origins",),
        ]
        paths += [
            ("rows", 0, "phases", 0, field)
            for field in ("n", "delta", "candidate_loss", "control_loss")
        ]
        paths += [
            ("rows", 0, "phases", 0, "hac", "se"),
            ("rows", 0, "phases", 0, "hac", "mde80_nominal"),
            ("rows", 0, "phases", 0, "hac", "ci95", 0),
            ("rows", 0, "phases", 0, "ci95_envelope", 1),
            ("rows", 0, "phases", 0, "block_inference", "3", "ci95", 0),
            ("rows", 0, "phases", 0, "annual", 0, "calendar_origins"),
            ("rows", 0, "phases", 0, "annual", 0, "missing_origins"),
            ("rows", 0, "phases", 0, "annual", 0, "n"),
            ("rows", 0, "phases", 0, "annual", 0, "delta"),
            ("rows", 0, "phases", 0, "offsets", 0, "delta"),
            ("rows", 0, "phases", 0, "offsets", 0, "n"),
            ("rows", 0, "phases", 1, "stability", 0, "delta"),
            ("rows", 0, "phases", 1, "stability", 0, "n"),
        ]
        for path in paths:
            bad = copy.deepcopy(self.metrics)
            node = bad
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] += 1
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.check(metrics=bad)

    def test_all_pvalues_require_exact_equality(self):
        paths = [
            ("p_conservative",),
            ("p_holm_wave",),
            ("p_holm_cumulative",),
            ("phases", 0, "p_conservative"),
            ("phases", 0, "hac", "p"),
        ]
        paths += [
            ("phases", phase, "block_inference", str(block), "p")
            for phase in range(2)
            for block in (3, 7, 11)
        ]
        for path in paths:
            bad = copy.deepcopy(self.metrics)
            node = bad["rows"][0]
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = float(np.nextafter(node[path[-1]], 1.0))
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.check(metrics=bad)

    def test_paired_rows_metadata_and_positive_qlike_are_not_trusted(self):
        variants = [
            pd.concat([self.panel, self.panel.iloc[:1]], ignore_index=True),
            self.panel.iloc[1:].copy(),
        ]
        for field, value in (
            ("model", "unknown"),
            ("y", 2.0),
            ("loss", 123.0),
            ("prediction", 0.0),
            ("prediction", np.inf),
            ("prediction", np.nan),
            ("train_n", 1001),
            ("phase", "evaluation"),
            ("commodity_cutoff_date", pd.Timestamp("2019-10-25")),
        ):
            bad = self.panel.copy()
            bad.loc[0, field] = value
            variants.append(bad)
        for i, bad in enumerate(variants):
            with self.subTest(i=i), self.assertRaises(ValueError):
                self.check(panel=bad)

    def test_fullcalendar_endpoints_offsets_phasefences_and_fit_clocks(self):
        origin = self.panel.origin.min()
        for field, value in (
            ("target_end", pd.Timestamp("2019-11-11")),
            ("commodity_cutoff_date", pd.Timestamp("2019-10-30")),
            ("fit_origin", pd.Timestamp("2019-11-04")),
            ("training_cutoff", pd.Timestamp("2019-10-30")),
            ("phase", "evaluation"),
            ("train_n", 999),
        ):
            bad = self.panel.copy()
            bad.loc[bad.origin == origin, field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(panel=bad)
        bad = self.panel.copy()
        compressed = {date: i % 5 for i, date in enumerate(sorted(bad.origin.unique()))}
        bad["offset"] = bad.origin.map(compressed)
        with self.assertRaises(ValueError):
            self.check(panel=bad)
        extra = self.panel[self.panel.origin == origin].copy()
        late = pd.Timestamp("2019-12-30")
        pos = self.calendar.get_loc(late)
        extra["origin"], extra["target_end"], extra["offset"] = (
            late,
            self.calendar[pos + 5],
            pos % 5,
        )
        extra["commodity_cutoff_date"] = self.calendar[pos - 1]
        extra["fit_origin"], extra["training_cutoff"] = (
            pd.Timestamp("2019-12-02"),
            pd.Timestamp("2019-11-29"),
        )
        with self.assertRaises(ValueError):
            self.check(panel=pd.concat([self.panel, extra], ignore_index=True))

    def test_exact_output_schema_identities_verdicts_and_inherited_preservation(self):
        for mode in (
            "extra",
            "missing",
            "order",
            "study",
            "lead",
            "verdict",
            "bool_count",
            "prior_changed",
            "prior_missing",
        ):
            bad = copy.deepcopy(self.metrics)
            if mode == "extra":
                bad["rows"][0]["phases"][0]["release_weighted_delta"] = -1.0
            elif mode == "missing":
                del bad["rows"][0]["phases"][0]["annual"]
            elif mode == "order":
                bad["rows"].reverse()
            elif mode == "study":
                bad["rows"][0]["study"] = "claims_release"
            elif mode == "lead":
                bad["leads"] = []
            elif mode == "verdict":
                bad["rows"][0]["verdict"] = "DOES_NOT_QUALIFY"
            elif mode == "bool_count":
                bad["rows"][0]["phases"][0]["n"] = True
            elif mode == "prior_changed":
                bad["inherited_rows"][0]["provenance"] = "different"
            else:
                bad["inherited_rows"].pop()
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.check(metrics=bad)

    def test_exact_prior_family_and_declared_comparison_counts(self):
        for prior in (self.prior[:-1], self.prior + [self.prior[0]]):
            with self.assertRaises(ValueError):
                self.check(prior=prior)
        for probability in (-0.1, 1.1, np.nan, np.inf, True):
            prior = copy.deepcopy(self.prior)
            prior[0]["p_conservative"] = probability
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                self.check(prior=prior)
        for field, value in (
            ("new_hypotheses", 3),
            ("inherited_hypotheses", 140),
            ("cumulative_hypotheses", 142),
            ("controls", ["market", "matched"]),
        ):
            protocol = copy.deepcopy(self.protocol)
            protocol["comparisons"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(protocol=protocol)

    def test_inherited_provenance_identity_and_optional_adjustments_are_validated(self):
        for field, value in (
            ("source", ""),
            ("source_sha256", "invalid"),
            ("source_row_index", -1),
            ("source_row_index", True),
            ("horizon", 0),
            ("candidate", ""),
            ("p_holm_wave", np.nan),
            ("p_holm_cumulative", 1.1),
        ):
            prior = copy.deepcopy(self.prior)
            prior[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(prior=prior)
        for mode in ("duplicate", "conflicting_hash"):
            prior = copy.deepcopy(self.prior)
            if mode == "duplicate":
                prior[1]["source_row_index"] = 0
            else:
                prior[1]["source_sha256"] = "b" * 64
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.check(prior=prior)

    def test_input_order_units_and_immutable_artifacts(self):
        before = copy.deepcopy((self.panel, self.prior, self.protocol, self.metrics))
        for unit in ("ms", "us", "ns"):
            self.assertEqual(
                self.check(
                    panel=self.panel.sample(frac=1, random_state=7),
                    calendar=self.calendar.as_unit(unit),
                )["status"],
                "VERIFIED",
            )
        pd.testing.assert_frame_equal(self.panel, before[0])
        self.assertEqual((self.prior, self.protocol, self.metrics), before[1:])


class CalibrationVerificationTests(unittest.TestCase):
    def setUp(self):
        self.protocol = fixture()[3]
        self.expected, self.paths = calibration_oracle(self.protocol)

    def test_generated_stationary_null_coverage_and_status(self):
        self.assertEqual(
            verify.verify_calibration(self.protocol, self.expected), {"status": "VERIFIED"}
        )

    def test_innovation_order_retained_paths_and_calibration_seed_schedule(self):
        with (
            patch.object(verify, "_hac", wraps=verify._hac) as hac,
            patch.object(
                verify, "_bootstrap_means", wraps=verify._bootstrap_means
            ) as bootstrap,
        ):
            verify.verify_calibration(self.protocol, self.expected)
        for call, path in zip(hac.call_args_list, self.paths):
            np.testing.assert_array_equal(call.args[0], path)
        self.assertEqual(hac.call_count, 7)
        self.assertEqual(
            [c.args[3] for c in bootstrap.call_args_list],
            [20260930 + trial * 1000 + block for trial in range(7) for block in (3, 7, 11)],
        )
        self.assertTrue(all(c.args[2] == 59 for c in bootstrap.call_args_list))

    def test_calibration_fields_coverage_status_and_schema_corruption(self):
        for field in (*self.expected.keys(), "extra"):
            bad = copy.deepcopy(self.expected)
            if field == "individual_block_coverage":
                bad[field]["3"] = float(np.nextafter(bad[field]["3"], -1.0))
            elif field in ("status", "limitation", "extra"):
                bad[field] = "corrupted"
            elif type(bad[field]) is float:
                bad[field] = float(np.nextafter(bad[field], -1.0))
            else:
                bad[field] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify.verify_calibration(self.protocol, bad)


if __name__ == "__main__":
    unittest.main()
