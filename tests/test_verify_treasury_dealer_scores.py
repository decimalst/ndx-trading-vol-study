"""Prewritten independent Treasury masked-score reconstruction tests."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import norm

from src.treasury_dealer_inference import masked_mean_inference
from src.treasury_dealer_score import evaluate
from src.verify_treasury_dealer_scores import _indexed_inference, verify_scores
from tests.test_treasury_dealer_score import generated_inputs


def literal_inference(values, mask, block, draws, seed, lag):
    n = len(values)
    full, tail = divmod(n, block)
    rng = np.random.default_rng(seed + block)
    starts = rng.integers(0, n, size=(draws, full))
    tails = rng.integers(0, n, size=draws) if tail else None
    samples = []
    for i, row in enumerate(starts):
        positions = [(int(s) + k) % n for s in row for k in range(block)]
        if tail:
            positions += [(int(tails[i]) + k) % n for k in range(tail)]
        selected = [values[j] for j in positions if mask[j]]
        if not selected:
            raise ValueError("zero support")
        samples.append(sum(selected) / len(selected))
    mean = float(np.mean(values[mask]))
    influence = np.where(mask, (np.nan_to_num(values) - mean) / mask.mean(), 0)
    variance = sum(v * v for v in influence) / n
    for k in range(1, lag + 1):
        variance += (
            2
            * (1 - k / (lag + 1))
            * sum(influence[i] * influence[i - k] for i in range(k, n))
            / n
        )
    se = np.sqrt(variance / n)
    return dict(
        mean=mean,
        p=(1 + sum(abs(v - mean) >= abs(mean) for v in samples)) / (draws + 1),
        interval=np.quantile(samples, [0.025, 0.975]),
        se=se,
        hac_p=2 * norm.sf(abs(mean) / se),
    )


class VerifyTreasuryDealerScoresTests(unittest.TestCase):
    def test_indexed_bootstrap_and_hac_match_literal_calendar_oracle(self):
        values = np.sin(np.arange(137) * 0.17) - 0.08
        mask = np.arange(137) % 4 != 1
        mask[41:59] = False
        values[~mask] = np.nan
        result = _indexed_inference(
            values, mask, blocks=[7, 21], hac_lags=12, draws=99, seed=170
        )
        for block in (7, 21):
            expected = literal_inference(values, mask, block, 99, 170, 12)
            self.assertEqual(result["block_inference"][str(block)]["p"], expected["p"])
            np.testing.assert_allclose(
                result["block_inference"][str(block)]["ci95"], expected["interval"], atol=1e-12
            )
            self.assertAlmostEqual(result["hac"]["se"], expected["se"], places=13)
            self.assertAlmostEqual(result["hac"]["p"], expected["hac_p"], places=13)
        self.assertEqual(result["full_calendar_n"], 137)
        self.assertEqual(result["n"], int(mask.sum()))

    def test_separate_indexed_implementation_matches_frozen_prefix_inference(self):
        rng = np.random.default_rng(54)
        v = rng.normal(-0.1, 0.3, 217)
        m = np.arange(217) % 3 != 0
        m[80:103] = False
        v[~m] = np.nan
        kw = dict(blocks=[7, 21, 63], hac_lags=31, draws=199, seed=71)
        actual = _indexed_inference(v, m, **kw)
        expected = masked_mean_inference(v, m, **kw)
        self.assertEqual(actual["p_conservative"], expected["p_conservative"])
        for b in kw["blocks"]:
            self.assertEqual(
                actual["block_inference"][str(b)]["p"],
                expected["block_inference"][str(b)]["p"],
            )
            np.testing.assert_allclose(
                actual["block_inference"][str(b)]["ci95"],
                expected["block_inference"][str(b)]["ci95"],
                atol=1e-12,
            )
        np.testing.assert_allclose(actual["hac"]["ci95"], expected["hac"]["ci95"], atol=1e-12)

    def test_zero_support_replicate_is_fatal_without_redraw(self):
        v = np.zeros(137)
        m = np.zeros(137, dtype=bool)
        m[:2] = True
        for function in (_indexed_inference, masked_mean_inference):
            with self.assertRaisesRegex(ValueError, "zero.support"):
                function(v, m, blocks=[7], hac_lags=12, draws=99, seed=15)

    def make_verified(self):
        args = generated_inputs()

        def small(d, m, **kw):
            return masked_mean_inference(d, m, **(kw | {"draws": 31}))

        with patch("src.treasury_dealer_score.masked_mean_inference", side_effect=small):
            metrics = evaluate(*args)
        return args, metrics

    def check(self, args, metrics):
        def small(d, m, **kw):
            return _indexed_inference(d, m, **(kw | {"draws": 31}))

        with patch("src.verify_treasury_dealer_scores._indexed_inference", side_effect=small):
            return verify_scores(args[0], args[1], args[2], metrics, args[3], args[4], args[5])

    def test_generated_end_to_end_verifies_both_masks_phases_controls(self):
        args, metrics = self.make_verified()
        out = self.check(args, metrics)
        self.assertEqual(out["status"], "VERIFIED")
        self.assertEqual(out["comparisons_verified"], 2)
        self.assertEqual(out["endpoints_verified"], 8)
        self.assertEqual(out["bootstrap_runs_verified"], 24)
        self.assertEqual(out["inherited_comparisons_verified"], 144)

    def test_every_inference_family_gate_and_diagnostic_group_corruption_rejected(self):
        args, original = self.make_verified()
        for defect in (
            "mean",
            "active_n",
            "hac",
            "block_p",
            "block_ci",
            "conservative",
            "wave",
            "cumulative",
            "inherited",
            "offset",
            "slice",
            "leads",
            "support",
            "count",
        ):
            m = copy.deepcopy(original)
            row = m["rows"][0]
            phase = row["phases"][0]
            if defect == "mean":
                phase["daily"]["mean"] += 0.01
            elif defect == "active_n":
                phase["active"]["n"] += 1
            elif defect == "hac":
                phase["daily"]["hac"]["se"] += 0.01
            elif defect == "block_p":
                phase["active"]["block_inference"]["21"]["p"] += 1e-12
            elif defect == "block_ci":
                phase["daily"]["block_inference"]["63"]["ci95"][0] += 0.01
            elif defect == "conservative":
                row["p_conservative"] = 0.99
            elif defect == "wave":
                row["p_holm_wave"] = 0.99
            elif defect == "cumulative":
                row["p_holm_cumulative"] = 0.99
            elif defect == "inherited":
                m["inherited_rows"][0]["p_conservative"] = 0
            elif defect == "offset":
                phase["offsets"][0]["active_delta"] += 0.01
            elif defect == "slice":
                row["phases"][1]["stability"][0]["daily_delta"] += 0.01
            elif defect == "leads":
                m["leads"] = ["treasury_dealer"]
            elif defect == "support":
                m["support_report"]["monthly"][0]["train_activation_dates"] -= 1
            else:
                m["common_scored_origins"] += 1
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                self.check(args, m)

    def test_changed_panel_activity_and_protocol_cannot_reuse_old_proof(self):
        args, metrics = self.make_verified()
        for defect in ("activity", "target", "protocol", "support"):
            bad = copy.deepcopy(args)
            if defect == "activity":
                bad[2].iloc[bad[1].get_loc("2020-01-07")] = ~bad[2].iloc[
                    bad[1].get_loc("2020-01-07")
                ]
            elif defect == "target":
                bad[0].loc[0, "y"] *= 2
            elif defect == "protocol":
                bad[4]["inference"]["bootstrap_draws"] = 31
            else:
                bad[5]["offsets"]["evaluation"][0]["activation_dates"] -= 1
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                self.check(bad, metrics)

    @staticmethod
    def qualification():
        return {
            "source_clock_class": "QUALIFIED_REPORTED_CLOCK_ONLY_FOR_Z52",
            "source_clock_limitation": (
                "January27,2020 Z52 timing assumes faithful cached Treasury-attributed release text; "
                "original PDF and historical public-delivery timestamp remain unavailable and quantities unknown. "
                "Rejecting this assumption leaves unbounded strict-clock uncertainty after that date. "
                "Prior regression tests accessed the later historical period; it is not untouched confirmation."
            ),
        }

    def test_published_exact_clock_qualification_is_verified(self):
        args, metrics = self.make_verified()
        qualified = metrics | self.qualification()
        before = copy.deepcopy(qualified)
        self.assertEqual(self.check(args, qualified)["status"], "VERIFIED")
        self.assertEqual(qualified, before)

    def test_partial_or_changed_clock_qualification_is_rejected(self):
        args, metrics = self.make_verified()
        for field in self.qualification():
            for defect in ("missing", "altered", "wrong_type"):
                qualified = metrics | self.qualification()
                if defect == "missing":
                    del qualified[field]
                elif defect == "altered":
                    qualified[field] += " "
                else:
                    qualified[field] = None
                with self.subTest(field=field, defect=defect), self.assertRaises(ValueError):
                    self.check(args, qualified)


if __name__ == "__main__":
    unittest.main()
