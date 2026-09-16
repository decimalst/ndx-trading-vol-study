"""Prewritten independent copula-density, tail and score reconstruction tests."""

import copy
import unittest
from unittest.mock import patch

import mpmath as mp
import numpy as np
from scipy.stats import multivariate_normal, multivariate_t, norm, t

from src.joint_copula_score import evaluate
from src.treasury_dealer_inference import masked_mean_inference
from src.verify_joint_copula_scores import (
    _independent_components,
    _independent_logcopula,
    verify_scores,
)
from src.verify_treasury_dealer_scores import _indexed_inference
from tests.test_joint_copula_score import generated_inputs


def high_precision_copula(z, rho, family):
    with mp.workdps(100):
        a, b = (mp.mpf(str(x)) for x in z)
        r = mp.mpf(str(rho))
        det = 1 - r * r

        def marginal(x):
            return (
                mp.loggamma(mp.mpf("4.5"))
                - mp.loggamma(4)
                - mp.log(8 * mp.pi) / 2
                - mp.mpf("4.5") * mp.log1p(x * x / 8)
            )

        if family == "t8":
            q = (a * a - 2 * r * a * b + b * b) / det
            return float(
                mp.loggamma(5)
                - mp.loggamma(4)
                - mp.log(8 * mp.pi)
                - mp.log(det) / 2
                - 5 * mp.log1p(q / 8)
                - marginal(a)
                - marginal(b)
            )

        def normal(x):
            if x == 0:
                return mp.mpf(0)
            tail = mp.betainc(4, mp.mpf(".5"), 0, 8 / (8 + x * x), regularized=True) / 2
            if abs(x) < 8:
                answer = mp.sqrt(2) * mp.erfinv(1 - 2 * tail)
            else:
                target = mp.log(tail)
                start = mp.sqrt(-2 * target)
                answer = mp.findroot(
                    lambda w: mp.log(mp.erfc(w / mp.sqrt(2)) / 2) - target,
                    (start * 0.8, start),
                )
            return mp.sign(x) * answer

        a, b = normal(a), normal(b)
        return float(-mp.log(det) / 2 + (2 * r * a * b - r * r * (a * a + b * b)) / (2 * det))


class VerifyJointCopulaScoresTests(unittest.TestCase):
    def make_verified(self):
        args = generated_inputs()

        def small(values, mask, **kw):
            return masked_mean_inference(values, mask, **(kw | {"draws": 31}))

        with patch("src.joint_copula_score.masked_mean_inference", side_effect=small):
            metrics = evaluate(*args)
        return args, metrics

    def check(self, args, metrics):
        def small(values, mask, **kw):
            return _indexed_inference(values, mask, **(kw | {"draws": 31}))

        with patch("src.verify_joint_copula_scores._indexed_inference", side_effect=small):
            return verify_scores(args[0], args[1], metrics, args[2], args[3])

    def test_independent_formulas_match_distribution_ratios(self):
        z = np.array([[-3.0, 0.2], [0.0, 0.0], [1e-10, -2e-10], [0.8, 2.0], [-5.0, 4.0]])
        r = np.array([-0.995, 0.0, 0.3, 0.8, 0.995])
        for family in ("t8", "gaussian"):
            actual = _independent_logcopula(z, r, family)
            expected = []
            for pair, rho in zip(z, r, strict=True):
                if family == "t8":
                    value = (
                        multivariate_t.logpdf(pair, shape=[[1, rho], [rho, 1]], df=8)
                        - t.logpdf(pair, 8).sum()
                    )
                else:
                    w = norm.ppf(t.cdf(pair, 8))
                    value = (
                        multivariate_normal.logpdf(w, cov=[[1, rho], [rho, 1]])
                        - norm.logpdf(w).sum()
                    )
                expected.append(value)
            np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
        np.testing.assert_array_equal(
            _independent_logcopula(z, np.zeros(5), "independence"), np.zeros(5)
        )
        self.assertGreater(np.abs(_independent_logcopula(z, np.zeros(5), "t8")).max(), 0.01)

    def test_extreme_and_near_zero_tails_match_high_precision(self):
        z = np.array([[1e80, -1e60], [1e200, 1e200], [0.0, 0.0], [1e-10, -2e-10]])
        rho = np.array([-0.9, 0.995, -0.995, 0.6])
        for family in ("t8", "gaussian"):
            expected = [
                high_precision_copula(pair, r, family) for pair, r in zip(z, rho, strict=True)
            ]
            actual = _independent_logcopula(z, rho, family)
            self.assertTrue(np.isfinite(actual).all())
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-9)

    def test_full_components_and_generated_scores_independently_verify(self):
        args, metrics = self.make_verified()
        components = _independent_components(args[0])
        np.testing.assert_allclose(
            components["loss"],
            -components["marginal_log_density"].sum(axis=1) - components["log_copula"],
            rtol=0,
            atol=0,
        )
        proof = self.check(args, metrics)
        self.assertEqual(proof["status"], "VERIFIED")
        self.assertEqual(proof["comparisons_verified"], 2)
        self.assertEqual(proof["endpoints_verified"], 4)
        self.assertEqual(proof["bootstrap_runs_verified"], 12)
        self.assertEqual(proof["inherited_comparisons_verified"], 149)
        self.assertEqual(proof["density_rows_verified"], len(args[0]))

    def test_every_metric_and_exact_discrete_probability_corruption_rejected(self):
        args, metrics = self.make_verified()
        for defect in [
            "mean",
            "full_loss",
            "block_p",
            "hac_p",
            "block_ci",
            "envelope",
            "conservative",
            "wave",
            "cumulative",
            "offset",
            "slice",
            "annual",
            "count",
            "prior",
            "lead",
        ]:
            bad = copy.deepcopy(metrics)
            row = bad["rows"][0]
            phase = row["phases"][0]
            if defect == "mean":
                phase["mean"] += 0.01
            elif defect == "full_loss":
                phase["candidate_loss"] += 0.01
            elif defect == "block_p":
                phase["block_inference"]["21"]["p"] += 1e-12
            elif defect == "hac_p":
                phase["hac"]["p"] += 0.01
            elif defect == "block_ci":
                phase["block_inference"]["63"]["ci95"][0] += 0.01
            elif defect == "envelope":
                phase["ci95_envelope"][0] += 0.01
            elif defect == "conservative":
                row["p_conservative"] = 0.99
            elif defect in ["wave", "cumulative"]:
                row["p_holm_" + defect] = 0.99
            elif defect == "offset":
                phase["offsets"][0]["mean"] += 0.01
            elif defect == "slice":
                row["phases"][1]["stability"][0]["mean"] += 0.01
            elif defect == "annual":
                phase["annual"][0]["missing_origins"] += 1
            elif defect == "count":
                bad["common_scored_origins"] += 1
            elif defect == "prior":
                bad["inherited_rows"][0]["status"] = "COMPLETED"
            else:
                bad["leads"] = ["joint_copula"]
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                self.check(args, bad)

    def test_source_qualification_and_scientific_contract_cannot_change(self):
        args, metrics = self.make_verified()
        for field in ["evidence_class", "evidence_limitation"]:
            bad = copy.deepcopy(metrics)
            del bad[field]
            with self.assertRaises(ValueError):
                self.check(args, bad)
            bad = copy.deepcopy(metrics)
            bad[field] += " "
            with self.assertRaises(ValueError):
                self.check(args, bad)
        for branch, key, value in [
            ("forecast", "origin_end", "2025-10-20"),
            ("inference", "seed", 1),
            ("inference", "bootstrap_draws", 31),
            ("support", "phase_daily", 1),
        ]:
            bad = list(copy.deepcopy(args))
            bad[3][branch][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check(bad, metrics)

    def test_nonshared_marginals_wrong_dates_and_rho_fail(self):
        args, metrics = self.make_verified()
        for field, value in [
            ("mu_qqq", 0.9),
            ("h_spx", 0.8),
            ("rho", 0.999),
            ("offset", 99),
            ("train_n", 999),
        ]:
            bad = list(copy.deepcopy(args))
            bad[0].loc[0, field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(bad, metrics)
        bad = list(copy.deepcopy(args))
        bad[0].loc[0, "target_end"] = bad[0].loc[0, "origin"]
        with self.assertRaises(ValueError):
            self.check(bad, metrics)


if __name__ == "__main__":
    unittest.main()
