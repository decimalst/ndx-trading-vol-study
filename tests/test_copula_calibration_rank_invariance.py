"""Generated rank identities only; no market trial or OOS density claim.

Fixed seed 20260911, n=80. Two deliberately tied asset samples use average
ranks divided by n+1. There is exactly one generated fit per copula family;
bit-identical transformed inputs establish the same deterministic fit problem.
"""

import unittest

import numpy as np
from numpy.testing import assert_array_equal
from scipy.stats import rankdata, t

from src.copula_calibration import calibrate
from src.joint_copula_density import fit_dependence, log_copula
from src.verify_joint_copula_forecasts import verify_dependence


def generated_variants():
    rng = np.random.default_rng(20260911)
    original = rng.standard_t(8, size=(80, 2))
    original[:, 1] = 0.45 * original[:, 0] + original[:, 1]
    original[:3, 0] = -0.25
    original[2:5, 1] = 0.5
    constant_scale = original * [3.25, 0.125]
    calibration = {"location": [0.3, -0.4], "scale": [0.8, 1.2], "train_n": 80}
    corrected = calibrate(original, np.ones_like(original), calibration)["z"]
    return original, constant_scale, corrected


def pseudo_observations(values):
    return rankdata(values, method="average", axis=0) / (len(values) + 1)


class CopulaCalibrationRankInvarianceTests(unittest.TestCase):
    def test_constant_scale_and_fixed_calibration_preserve_average_ranks(self):
        original, scaled, corrected = generated_variants()
        expected = pseudo_observations(original)
        self.assertTrue(((expected > 0) & (expected < 1)).all())
        for changed in (scaled, corrected):
            self.assertFalse(np.array_equal(changed, original))
            assert_array_equal(pseudo_observations(changed), expected)
            for asset in range(2):
                order = np.argsort(original[:, asset], kind="stable")
                assert_array_equal(
                    np.sign(np.diff(original[order, asset])),
                    np.sign(np.diff(changed[order, asset])),
                )
        for asset, value, rows in ((0, -0.25, [0, 1, 2]), (1, 0.5, [2, 3, 4])):
            less = int((original[:, asset] < value).sum())
            tied = int((original[:, asset] == value).sum())
            wanted = (less + (tied + 1) / 2) / 81
            assert_array_equal(expected[rows, asset], np.full(3, wanted))

    def test_both_pseudo_objectives_are_identical_for_each_fixed_parameter(self):
        coordinates = [t.ppf(pseudo_observations(x), 8) for x in generated_variants()]
        for family in ("gaussian", "t8"):
            for rho in (-0.8, -0.1, 0.0, 0.5, 0.9):
                with self.subTest(family=family, rho=rho):
                    expected = log_copula(coordinates[0], rho, family)
                    for changed in coordinates[1:]:
                        assert_array_equal(changed, coordinates[0])
                        actual = log_copula(changed, rho, family)
                        assert_array_equal(actual, expected)
                        self.assertEqual(float(actual.mean()), float(expected.mean()))

    def test_one_certified_fit_per_family_has_identical_inputs_and_selected_objective(self):
        coordinates = [t.ppf(pseudo_observations(x), 8) for x in generated_variants()]
        for family in ("gaussian", "t8"):
            with self.subTest(family=family):
                # Exactly one actual fit, not separate fits of the identical arrays.
                rho, audit = fit_dependence(coordinates[0], family)
                proof = verify_dependence(coordinates[0], family, rho, audit)
                self.assertEqual(proof["status"], "VERIFIED")
                self.assertEqual(audit["train_n"], 80)
                selected = -float(log_copula(coordinates[0], rho, family).mean())
                self.assertAlmostEqual(audit["objective"], selected, places=12)
                for changed in coordinates[1:]:
                    self.assertEqual(changed.tobytes(), coordinates[0].tobytes())
                    self.assertEqual(-float(log_copula(changed, rho, family).mean()), selected)
                # The entire objective, domain and tie rule receive identical data.
                # No assertion is made about constancy under time-varying transforms.

    def test_time_varying_positive_scale_can_reorder_ranks(self):
        original = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        varying = original.copy()
        varying[:, 0] *= [8.0, 1.0, 1.0, 1.0]
        before, after = pseudo_observations(original), pseudo_observations(varying)
        assert_array_equal(before[:, 1], after[:, 1])
        self.assertFalse(np.array_equal(before[:, 0], after[:, 0]))
        assert_array_equal(after[:, 0], np.array([4, 1, 2, 3]) / 5)
        for family in ("gaussian", "t8"):
            difference = (
                log_copula(t.ppf(after, 8), 0.5, family).mean()
                - log_copula(t.ppf(before, 8), 0.5, family).mean()
            )
            self.assertGreater(abs(float(difference)), 0.1)


if __name__ == "__main__":
    unittest.main()
