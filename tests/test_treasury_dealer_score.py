"""Prewritten generated Treasury score/support/family contracts."""

import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from src.treasury_dealer_score import evaluate, failure_metrics, inherit_family

ROOT = Path(__file__).resolve().parents[1]
TENORS = ("2", "3", "5", "7", "10", "30")


def protocol():
    return yaml.safe_load((ROOT / "treasury_dealer.yaml").read_text())


def prior_rows(n=144):
    return [
        {
            "study": "old",
            "candidate": "old_candidate",
            "control": "old_control",
            "horizon": 5,
            "p_conservative": 1.0,
            "p_holm_wave": 1.0,
            "p_holm_cumulative": 1.0,
            "source": "invented/prior.json",
            "source_sha256": "a" * 64,
            "source_row_index": i,
            "status": "UNEVALUABLE" if i == 0 else "COMPLETED",
        }
        for i in range(n)
    ]


def generated_inputs():
    p = protocol()
    calendar = pd.bdate_range("2011-01-03", "2025-10-20").as_unit("ns")
    active = pd.Series(np.arange(len(calendar)) % 7 == 0, index=calendar)
    rows = []
    for position, day in enumerate(calendar):
        if day < pd.Timestamp("2016-01-01") or position + 5 >= len(calendar):
            continue
        phase = "development" if day <= pd.Timestamp("2019-12-31") else "evaluation"
        if phase == "development" and calendar[position + 5] > pd.Timestamp("2019-12-31"):
            continue
        if day in pd.DatetimeIndex(["2018-02-07", "2021-06-09"]):
            continue
        month_days = calendar[calendar.to_period("M") == day.to_period("M")]
        fit = month_days[0]
        cutoff = calendar[calendar.get_loc(fit) - 1]
        y = 1.0 + 0.08 * np.sin(position / 11)
        for model, factor in (("market", 1.6), ("matched", 1.4), ("candidate", 1.0)):
            pred = y * factor
            ratio = y / pred
            rows.append(
                dict(
                    origin=day,
                    model=model,
                    prediction=pred,
                    y=y,
                    loss=ratio - np.log(ratio) - 1,
                    target_end=calendar[position + 5],
                    phase=phase,
                    treasury_cutoff_date=calendar[position - 1],
                    offset=position % 5,
                    fit_origin=fit,
                    training_cutoff=cutoff,
                    train_n=1000,
                )
            )
    panel = pd.DataFrame(rows)
    common = panel[panel.model == "candidate"].set_index("origin")

    def support_row(start, end, offset=None):
        selected = common.loc[start:end]
        if offset is not None:
            selected = selected[selected.offset == offset]
        n = int(active.reindex(selected.index).sum())
        events = 2 * n
        counts = {t: events // 6 + (i < events % 6) for i, t in enumerate(TENORS)}
        result = dict(
            origin_start=start,
            origin_end=end,
            daily_n=len(selected),
            activation_dates=n,
            event_n=events,
            events_per_tenor=counts,
            passed=True,
            failures=[],
        )
        if offset is not None:
            result["offset"] = offset
        return result

    support = dict(
        status="SUPPORT_PASS",
        passed=True,
        monthly=[],
        phases={},
        slices=[],
        offsets={},
        discrepancies=[],
    )
    for phase in ("development", "evaluation"):
        start, end = p["forecast"][phase]
        support["phases"][phase] = support_row(start, end)
        support["offsets"][phase] = [support_row(start, end, k) for k in range(5)]
    support["slices"] = [
        support_row(start, end) | {"slice_index": i}
        for i, (start, end) in enumerate(p["forecast"]["stability"])
    ]
    for month in pd.period_range("2016-01", "2025-10", freq="M"):
        dates = calendar[calendar.to_period("M") == month]
        fit = dates[0]
        cutoff = calendar[calendar.get_loc(fit) - 1]
        support["monthly"].append(
            dict(
                month=str(month),
                status="SUPPORTED",
                fit_origin=str(fit.date()),
                training_cutoff=str(cutoff.date()),
                requested_n=len(dates),
                application_n=len(dates),
                train_n=1000,
                train_activation_dates=200,
                train_event_n=300,
                train_events_per_tenor=dict.fromkeys(TENORS, 50),
                passed=True,
                failures=[],
            )
        )
    return panel, calendar, active, prior_rows(), p, support


def fake_inference(differences, mask, *, blocks, hac_lags, draws, seed):
    values = np.asarray(differences)[mask]
    mean = float(values.mean())
    interval = [mean - 0.001, mean + 0.001]
    return dict(
        mean=mean,
        n=len(values),
        full_calendar_n=len(mask),
        hac=dict(se=0.0005, p=1e-10, ci95=interval, mde80_nominal=0.0014),
        block_inference={str(b): dict(p=1e-10, ci95=interval) for b in blocks},
        p_conservative=1e-10,
    )


class TreasuryDealerScoreTests(unittest.TestCase):
    def run_fake(self, args):
        with patch(
            "src.treasury_dealer_score.masked_mean_inference", side_effect=fake_inference
        ) as inference:
            out = evaluate(*args)
        return out, inference

    def test_two_controls_both_endpoints_and_exact_family(self):
        args = generated_inputs()
        out, inf = self.run_fake(args)
        self.assertEqual(out["status"], "COMPLETED")
        self.assertEqual(out["hypothesis_count"], 2)
        self.assertEqual(out["cumulative_hypothesis_count"], 146)
        self.assertEqual(out["leads"], ["treasury_dealer"])
        self.assertEqual(out["inherited_rows"], args[3])
        self.assertEqual(inf.call_count, 8)
        self.assertEqual([r["control"] for r in out["rows"]], ["matched", "market"])
        for row in out["rows"]:
            self.assertAlmostEqual(row["p_holm_wave"], 2e-10)
            self.assertAlmostEqual(row["p_holm_cumulative"], 146e-10)
            self.assertEqual(row["verdict"], "COMPARISON_GATE_PASS")
            self.assertEqual([x["name"] for x in row["phases"]], ["development", "evaluation"])

    def test_full_phase_calendar_gaps_tail_positions_offsets_and_shared_rng(self):
        args = generated_inputs()
        out, inf = self.run_fake(args)
        seen = {}
        for call in inf.call_args_list:
            d, mask = call.args[:2]
            seed = call.kwargs["seed"]
            phase = "evaluation" if seed % 1000000 == 20261001 % 1000000 + 10000 else None
            del phase
            self.assertEqual(call.kwargs["draws"], 399999)
            self.assertEqual(call.kwargs["blocks"], [21, 63, 126])
            self.assertEqual(call.kwargs["hac_lags"], 126)
            self.assertGreater(len(mask), int(mask.sum()))
            self.assertTrue(
                np.isnan(np.asarray(d)[~mask]).any() or np.isfinite(np.asarray(d)[~mask]).any()
            )
            seen.setdefault(seed, []).append(mask.copy())
        self.assertEqual(set(seen), {20261001, 20271001, 21261001, 21271001})
        for masks in seen.values():
            np.testing.assert_array_equal(masks[0], masks[1])
        for phase in out["rows"][0]["phases"]:
            start, end = args[4]["forecast"][phase["name"]]
            expected = int(((args[1] >= start) & (args[1] <= end)).sum())
            self.assertEqual(phase["daily"]["full_calendar_n"], expected)
            self.assertEqual(phase["active"]["full_calendar_n"], expected)
            self.assertEqual([x["offset"] for x in phase["offsets"]], list(range(5)))

    def test_activity_comes_from_explicit_mask_even_zero_difference(self):
        args = list(generated_inputs())
        p = args[0]
        candidate = p.model == "candidate"
        active_origins = args[2].index[args[2]]
        # Candidate equals matched on active dates only: activity remains counted.
        target = candidate & p.origin.isin(active_origins)
        p.loc[target, "prediction"] = p.loc[target, "y"] * 1.4
        ratio = p.loc[target, "y"] / p.loc[target, "prediction"]
        p.loc[target, "loss"] = ratio - np.log(ratio) - 1
        out, _ = self.run_fake(args)
        self.assertEqual(out["leads"], [])
        for phase in out["rows"][0]["phases"]:
            self.assertEqual(phase["active"]["mean"], 0.0)
            self.assertGreater(phase["active"]["n"], 0)

    def test_support_corruptions_fail_before_resampling(self):
        for defect in (
            "status",
            "phase_count",
            "slice_count",
            "offset_count",
            "train_count",
            "train_events",
            "phase_tenor",
            "slice_tenor",
            "extra_month",
        ):
            args = list(generated_inputs())
            s = args[-1]
            if defect == "status":
                s["passed"] = False
            elif defect == "phase_count":
                s["phases"]["evaluation"]["daily_n"] -= 1
            elif defect == "slice_count":
                s["slices"][0]["activation_dates"] -= 1
            elif defect == "offset_count":
                s["offsets"]["development"][0]["daily_n"] -= 1
            elif defect == "train_count":
                s["monthly"][0]["train_activation_dates"] = 199
            elif defect == "train_events":
                s["monthly"][0]["train_events_per_tenor"]["2"] = 23
            elif defect == "phase_tenor":
                s["phases"]["development"]["events_per_tenor"]["2"] = 11
            elif defect == "slice_tenor":
                s["slices"][0]["events_per_tenor"]["2"] = 5
            else:
                s["monthly"].append(copy.deepcopy(s["monthly"][0]))
            with (
                self.subTest(defect=defect),
                patch("src.treasury_dealer_score.masked_mean_inference") as inf,
            ):
                with self.assertRaises(ValueError):
                    evaluate(*args)
                inf.assert_not_called()

    def test_all_fixed_protocol_numeric_dates_and_masks_are_enforced(self):
        mutations = [
            ("inference", "bootstrap_draws", 49),
            ("inference", "seed", 1),
            ("inference", "blocks", [21]),
            ("inference", "hac_lags", 21),
            ("forecast", "horizon", 1),
            ("forecast", "source_end", "2025-11-03"),
            ("support", "phase_daily", 1),
            ("support", "phase_activation_dates", 1),
            ("comparisons", "inherited", 143),
            ("comparisons", "wave_alpha", 0.05),
            ("comparisons", "cumulative", 145),
        ]
        for section, key, value in mutations:
            args = list(generated_inputs())
            args[4][section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                evaluate(*args)
        for kind in ("unaligned", "numeric"):
            args = list(generated_inputs())
            args[2] = args[2].iloc[::-1] if kind == "unaligned" else args[2].astype(int)
            with self.assertRaises(ValueError):
                evaluate(*args)

    def test_pairing_qlike_and_global_target_offset_corruption(self):
        for defect in (
            "loss",
            "y",
            "target_end",
            "offset",
            "missing_arm",
            "prior_probability",
            "prior_duplicate",
        ):
            args = list(generated_inputs())
            if defect == "loss":
                args[0].loc[0, "loss"] += 0.01
            elif defect == "y":
                args[0].loc[0, "y"] *= 2
            elif defect == "target_end":
                args[0].loc[0, "target_end"] += pd.Timedelta(days=1)
            elif defect == "offset":
                args[0].loc[0, "offset"] = (int(args[0].loc[0, "offset"]) + 1) % 5
            elif defect == "missing_arm":
                args[0] = args[0].iloc[1:]
            elif defect == "prior_probability":
                args[3][0]["p_conservative"] = True
            else:
                args[3][1]["source_row_index"] = 0
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                evaluate(*args)

    def test_worst_endpoint_phase_method_and_single_control_veto(self):
        args = generated_inputs()

        def bad(d, m, **kwargs):
            out = fake_inference(d, m, **kwargs)
            if kwargs["seed"] == 21271001:
                out["hac"]["p"] = 1.0
                out["p_conservative"] = 1.0
            return out

        with patch("src.treasury_dealer_score.masked_mean_inference", side_effect=bad):
            out = evaluate(*args)
        self.assertEqual(out["leads"], [])
        self.assertTrue(all(r["p_conservative"] == 1 for r in out["rows"]))

    def test_failure_metrics_preserve_all_prior_and_two_p_one_rows(self):
        old = prior_rows()
        for error, status in (
            (ValueError("INSUFFICIENT_DATA: generated"), "UNSUPPORTED"),
            (ValueError("zero-support bootstrap replicate"), "FAILED"),
        ):
            out = failure_metrics(error, "b" * 64, old)
            self.assertEqual(out["status"], "UNEVALUABLE")
            self.assertEqual(out["error_kind"], status)
            self.assertEqual(out["inherited_rows"], old)
            self.assertEqual(out["leads"], [])
            self.assertEqual(len(out["rows"]), 2)
            for row in out["rows"]:
                self.assertEqual(
                    [row[k] for k in ("p_conservative", "p_holm_wave", "p_holm_cumulative")],
                    [1.0, 1.0, 1.0],
                )

    def test_inheritance_and_input_preservation(self):
        args = generated_inputs()
        before = copy.deepcopy(args)
        out, _ = self.run_fake(args)
        pd.testing.assert_frame_equal(args[0], before[0])
        pd.testing.assert_series_equal(args[2], before[2])
        self.assertEqual(args[3:], before[3:])
        previous = dict(
            status="COMPLETED",
            hypothesis_count=2,
            cumulative_hypothesis_count=144,
            inherited_rows=prior_rows(142),
            rows=[],
        )
        for control in ("matched", "market"):
            previous["rows"].append(
                dict(
                    study="commodity_implied",
                    candidate="candidate",
                    control=control,
                    horizon=5,
                    score="qlike",
                    p_conservative=1.0,
                    verdict="DOES_NOT_QUALIFY",
                )
            )
        inherited = inherit_family(previous, "c" * 64)
        self.assertEqual(len(inherited), 144)
        self.assertEqual(inherited[:142], previous["inherited_rows"])
        self.assertEqual(
            inherited[-1]["source"], "reports/commodity_implied/predictive/metrics.json"
        )
        out["inherited_rows"][0]["p_conservative"] = 0
        self.assertEqual(args[3][0]["p_conservative"], 1)


if __name__ == "__main__":
    unittest.main()
