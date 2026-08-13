"""Contract for the corrected-methodology fork.

Two jobs. First, pin the corrections themselves: power, equivalence, overlap-
aware standard errors, the specification gate, and the point-forecast
estimator. Second — and more important — pin the guarantee that makes the fork
safe: the DEFAULT path must reproduce the frozen pre-registered reports
byte-for-byte, so a correction can never quietly rewrite a result it was
supposed to be compared against.

Run:  python -m unittest tests.test_methodology
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, methodology, models  # noqa: E402


def _lognormal_quantiles(mu: float, sigma: float, taus: np.ndarray) -> np.ndarray:
    from scipy.stats import norm
    return mu + sigma * norm.ppf(taus)


class TestPointForecastEstimator(unittest.TestCase):
    """The frozen estimator, what is wrong with it, and what replaces it."""

    PRE = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    DEC = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    def test_trunc_understates_the_true_mean(self):
        # Exact lognormal: E[e^X] = exp(mu + s^2/2). The frozen estimator
        # discards both tails and normalises by the grid width.
        for s in (0.4, 0.8, 1.2):
            q = _lognormal_quantiles(0.0, s, self.PRE)
            est = models.trunc_mean_var(self.PRE, q)
            self.assertLess(est, np.exp(s * s / 2),
                            f"trunc should understate at sigma={s}")

    def test_trunc_understatement_grows_with_dispersion(self):
        ratios = [models.trunc_mean_var(self.PRE, _lognormal_quantiles(0, s, self.PRE))
                  / np.exp(s * s / 2) for s in (0.4, 0.8, 1.2)]
        self.assertTrue(ratios[0] > ratios[1] > ratios[2],
                        "the discarded tail share must grow with the spread")

    def test_trunc_is_grid_dependent(self):
        # Same distribution, different grid, different answer. This is why
        # results_clean.md and results_clean_dec.md are not comparable.
        s = 0.8
        a = models.trunc_mean_var(self.PRE, _lognormal_quantiles(0, s, self.PRE))
        b = models.trunc_mean_var(self.DEC, _lognormal_quantiles(0, s, self.DEC))
        self.assertGreater(abs(a - b) / a, 0.05)

    def test_tail_ext_is_near_exact_on_a_true_lognormal(self):
        for s in (0.4, 0.8, 1.2):
            for grid in (self.PRE, self.DEC):
                q = _lognormal_quantiles(0.0, s, grid)
                est = models.mean_var_from_quantiles(grid, q, method="tail_ext")
                self.assertAlmostEqual(est / np.exp(s * s / 2), 1.0, delta=0.02,
                                       msg=f"tail_ext off at sigma={s}")

    def test_tail_ext_is_far_less_grid_dependent_than_trunc(self):
        s = 0.8
        a = models.mean_var_from_quantiles(self.PRE, _lognormal_quantiles(0, s, self.PRE),
                                           method="tail_ext")
        b = models.mean_var_from_quantiles(self.DEC, _lognormal_quantiles(0, s, self.DEC),
                                           method="tail_ext")
        self.assertLess(abs(a - b) / a, 0.02)

    def test_smearing_is_exact_on_its_own_residuals(self):
        rng = np.random.default_rng(0)
        resid = rng.standard_normal(50_000) * 0.8
        got = models.smearing_mean_var(1.5, resid)
        self.assertAlmostEqual(got / (np.exp(1.5) * np.exp(0.8 ** 2 / 2)), 1.0,
                               delta=0.02)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            models.mean_var_from_quantiles(self.PRE, np.zeros(7), method="nope")


class TestPowerAndEquivalence(unittest.TestCase):

    def test_mde_shrinks_with_sample_size(self):
        rng = np.random.default_rng(1)
        big = rng.standard_normal(4000) * 0.1
        small = big[:200]
        self.assertLess(methodology.dm_mde(big, np.zeros_like(big))["mde"],
                        methodology.dm_mde(small, np.zeros_like(small))["mde"])

    def test_identical_forecasts_are_declared_equivalent(self):
        rng = np.random.default_rng(2)
        a = rng.random(500) + 1.0
        t = methodology.dm_tost(a, a.copy(), margin=0.05)
        self.assertTrue(t["equivalent"])

    def test_a_real_difference_is_not_declared_equivalent(self):
        rng = np.random.default_rng(3)
        a = rng.standard_normal(500) * 0.1
        b = a + 0.5                       # far outside any sane margin
        self.assertFalse(methodology.dm_tost(a, b, margin=0.05)["equivalent"])

    def test_underpowered_null_is_inconclusive_not_equivalent(self):
        """The core fix. A tiny sample with a wide spread rejects neither the
        difference nor the equivalence, and must say so."""
        rng = np.random.default_rng(4)
        a = rng.standard_normal(30) * 1.0
        b = a + 0.05
        tost = methodology.dm_tost(a, b, margin=0.01)
        verdict = methodology.dm_verdict(dm_p=0.7, gap=-0.05, tost=tost)
        self.assertEqual(verdict, "inconclusive")

    def test_margin_is_a_fraction_of_the_benchmark(self):
        self.assertAlmostEqual(
            methodology.equivalence_margin(np.full(100, 0.30), frac=0.03), 0.009)


class TestOverlappingWindowInference(unittest.TestCase):
    """Standard errors under a 21-day overlapping target."""

    @staticmethod
    def _overlapping(n=1200, overlap=21, seed=5):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2016-01-04", periods=n + overlap)
        x = pd.Series(rng.standard_normal(n + overlap), index=idx).cumsum() / 10
        e = pd.Series(rng.standard_normal(n + overlap), index=idx)
        # y_t aggregates the next `overlap` innovations -> neighbouring rows
        # share almost all of their target, exactly like fwd_var_30c.
        y = e[::-1].rolling(overlap).sum()[::-1].shift(-1)
        d = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
        return d["y"] + 0.5 * d["x"], d[["x"]]

    def test_effective_n_divides_by_the_overlap(self):
        self.assertAlmostEqual(methodology.effective_n(171, 21), 171 / 21)

    def test_block_bootstrap_se_exceeds_iid_ols_se(self):
        y, X = self._overlapping()
        Xc = X.assign(const=1.0)[["const", "x"]]
        boot = methodology.block_bootstrap_ols(y, Xc, block=21, n_boot=400, seed=9)
        resid = y.values - Xc.values @ np.array([boot["beta"]["const"],
                                                 boot["beta"]["x"]])
        dof = len(y) - 2
        iid_se = np.sqrt(np.diag(np.linalg.pinv(Xc.values.T @ Xc.values))
                         * (resid @ resid) / dof)[1]
        self.assertGreater(boot["se_boot"]["x"], iid_se,
                           "overlap must inflate the standard error")

    def test_phase_split_returns_one_subsample_per_offset(self):
        y, X = self._overlapping()
        ph = methodology.nonoverlap_phases(y, X.assign(const=1.0)[["const", "x"]],
                                            step=21)
        self.assertEqual(ph["n_phases"], 21)
        self.assertLess(ph["n_per_phase"], len(y) / 20)

    def test_mz_corrected_reports_n_eff_not_n(self):
        y, X = self._overlapping()
        r = methodology.mz_corrected(y, X["x"], overlap=21, n_boot=200)
        self.assertAlmostEqual(r["n_eff"], r["n"] / 21, places=6)
        self.assertLess(r["n_eff"], r["n"])

    def test_bootstrap_is_seed_deterministic(self):
        y, X = self._overlapping()
        kw = dict(overlap=21, n_boot=200, seed=123)
        a = methodology.mz_corrected(y, X["x"], **kw)
        b = methodology.mz_corrected(y, X["x"], **kw)
        self.assertEqual(a["boot"]["se_boot"]["f"], b["boot"]["se_boot"]["f"])


class TestSpecificationRegistry(unittest.TestCase):

    def setUp(self):
        self.reg = config.load_spec_registry()

    def test_registry_loads(self):
        self.assertIn("models", self.reg)

    def test_preregistered_model_is_confirmatory_in_the_clean_window(self):
        s = config.spec_status(self.reg, "har_iv", "2025-11-03")
        self.assertTrue(s["confirmatory"])

    def test_post_hoc_model_is_not_confirmatory_in_the_window_that_made_it(self):
        s = config.spec_status(self.reg, "har_iv_x", "2025-11-03")
        self.assertFalse(s["confirmatory"])
        self.assertEqual(s["confirmatory_from"], pd.Timestamp("2026-08-11"))

    def test_model_published_after_the_window_opens_starts_late(self):
        s = config.spec_status(self.reg, "tirex_uni", "2025-11-03")
        self.assertFalse(s["confirmatory"])
        self.assertEqual(s["confirmatory_from"], pd.Timestamp("2026-07-01"))

    def test_undetermined_date_is_treated_as_exploratory(self):
        s = config.spec_status(self.reg, "har_sv", "2025-11-03")
        self.assertFalse(s["confirmatory"])
        self.assertIsNone(s["confirmatory_from"])

    def test_unknown_model_is_exploratory_rather_than_silently_confirmatory(self):
        s = config.spec_status(self.reg, "not_a_model", "2025-11-03")
        self.assertFalse(s["confirmatory"])


class TestPairedSummary(unittest.TestCase):

    def test_flip_k_counts_days_needed_to_reverse_the_sign(self):
        # One huge win funding many small losses: dropping it flips the mean.
        idx = pd.bdate_range("2026-01-01", periods=50)
        b = pd.Series(np.full(50, 1.0), index=idx)
        a = pd.Series(np.full(50, 1.01), index=idx)
        a.iloc[0] = 1.0 - 5.0
        r = methodology.paired_summary(a, b)
        self.assertGreater(r["gap"], 0)
        self.assertEqual(r["flip_k"], 1)

    def test_robust_gap_has_no_flip_point(self):
        idx = pd.bdate_range("2026-01-01", periods=50)
        b = pd.Series(np.full(50, 1.0), index=idx)
        a = b - 0.2
        self.assertIsNone(methodology.paired_summary(a, b)["flip_k"])


class TestFrozenReportsUnchanged(unittest.TestCase):
    """The guarantee that makes the fork safe.

    Skips rather than fails when the real data or forecasts are absent, so the
    suite still runs on a fresh checkout.
    """

    PHASES = [("clean", []), ("diagnostic", []),
              ("clean", ["--quantile-grid", "deciles"])]

    def test_default_evaluate_reproduces_the_frozen_reports(self):
        if not (ROOT / "data" / "processed" / "master_daily.parquet").exists():
            self.skipTest("no processed master; run `make features` first")
        for phase, extra in self.PHASES:
            sfx = "_dec" if "deciles" in extra else ""
            path = ROOT / "reports" / f"results_{phase}{sfx}.md"
            if not path.exists():
                self.skipTest(f"{path.name} absent")
            before = path.read_text()
            r = subprocess.run(
                [sys.executable, "-m", "src.experiment", "evaluate",
                 "--phase", phase, *extra],
                cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            self.assertEqual(before, path.read_text(),
                             f"default evaluate changed {path.name} — the "
                             f"corrected fork must never touch a frozen report")

    def test_corrected_run_writes_a_separate_file(self):
        if not (ROOT / "data" / "processed" / "master_daily.parquet").exists():
            self.skipTest("no processed master")
        frozen = ROOT / "reports" / "results_clean.md"
        v2 = ROOT / "reports" / "results_clean_v2.md"
        before = frozen.read_text() if frozen.exists() else None
        r = subprocess.run(
            [sys.executable, "-m", "src.experiment", "evaluate", "--phase", "clean",
             "--estimator", "smearing", "--inference", "corrected"],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertTrue(v2.exists())
        if before is not None:
            self.assertEqual(before, frozen.read_text())


class TestEstimatorReconstructionAccuracy(unittest.TestCase):
    """Pins the measured reconstruction error so it cannot drift silently.

    These are the numbers the corrected report quotes when it says Chronos-2
    and TiRex-2 rows carry a reconstruction band. If this test starts failing,
    that sentence in the report is stale.
    """

    def test_reconstruction_matches_exact_smearing_on_the_har_family(self):
        proc = ROOT / "data" / "processed" / "master_daily.parquet"
        if not proc.exists():
            self.skipTest("no processed master")
        from src import experiment as X
        cfg = config.load()
        cfg["_estimator"], cfg["_grid_suffix"] = "trunc", ""
        df = pd.read_parquet(proc)
        taus = np.asarray(cfg["quantiles"])
        origins = X._phase_origins(df, cfg, "clean")
        specs = {"har": {}, "har_x": {"with_x": True}, "har_iv": {"with_iv": True},
                 "har_iv_x": {"with_x": True, "with_iv": True}}
        ratios = {"trunc": [], "tail_ext": []}
        for m, kw in specs.items():
            exact = models.run_har(df, origins, cfg, estimator="smearing", **kw)
            saved = X._load_forecast(cfg, m, "clean")
            if saved is None:
                self.skipTest(f"no saved forecasts for {m}")
            for meth in ratios:
                approx = models.rescore_mean_var(saved, taus, method=meth)
                j = pd.concat([exact["mean_var"].rename("e"), approx.rename("a")],
                              axis=1).dropna()
                ratios[meth].append(float((j["a"] / j["e"]).mean()))
        t, x = np.array(ratios["trunc"]), np.array(ratios["tail_ext"])
        # Level: tail_ext is much closer to the truth.
        self.assertLess(abs(x.mean() - 1), abs(t.mean() - 1))
        self.assertLess(abs(x.mean() - 1), 0.05)
        # Spread: trunc is a near-common factor, which is why correcting it
        # barely moves the DM statistics. Documented, not assumed.
        self.assertLess(t.max() - t.min(), 0.02)
        self.assertLess(x.max() - x.min(), 0.05)


if __name__ == "__main__":
    unittest.main()
