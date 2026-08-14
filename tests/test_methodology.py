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


class TestBootstrapCalibration(unittest.TestCase):
    """Measure the block bootstrap's actual coverage instead of assuming it.

    `block_bootstrap_ols` replaced HAC(32) because HAC at lag/n = 0.19 is badly
    biased. It is better, but it is not calibrated: resampling blocks of rows
    conditions on the one realized near-unit-root regressor path. The reports
    quote its intervals and p-values, so the size distortion is a published
    quantity and belongs in the test suite rather than in a footnote.

    This is a Monte Carlo, so it is slow-ish and its numbers are stochastic;
    the bounds are deliberately wide enough to be stable at this seed count and
    narrow enough to fail if the estimator is silently replaced.
    """

    N, OVERLAP, REPS, NBOOT = 171, 21, 300, 250
    AR = 0.895                       # measured persistence of log(VXN exp var)

    @classmethod
    def _coverage(cls, seed0: int) -> float:
        rng = np.random.default_rng(seed0)
        idx = pd.bdate_range("2016-01-04", periods=cls.N)
        hit = 0
        for r in range(cls.REPS):
            e = rng.standard_normal(cls.N + cls.OVERLAP)
            # MA(overlap-1): neighbouring targets share ~21 trading days, which
            # is exactly the dependence the block bootstrap claims to handle.
            u = np.convolve(e, np.ones(cls.OVERLAP), "valid")[:cls.N]
            x = np.zeros(cls.N + 200)
            for t in range(1, len(x)):
                x[t] = cls.AR * x[t - 1] + rng.standard_normal()
            x = x[200:]
            y = 1.0 * x + u
            X = pd.DataFrame({"const": 1.0, "f": x}, index=idx)
            b = methodology.block_bootstrap_ols(pd.Series(y, index=idx), X,
                                                block=cls.OVERLAP,
                                                n_boot=cls.NBOOT, seed=9000 + r)
            if b["ci_lo"]["f"] <= 1.0 <= b["ci_hi"]["f"]:
                hit += 1
        return hit / cls.REPS

    def test_nominal_95_interval_undercovers_and_the_reports_say_so(self):
        cov = self._coverage(seed0=7)
        # The published claim is "~82-85%". Fail if it drifts outside a band
        # that would make that sentence wrong in either direction.
        self.assertGreater(cov, 0.74, f"coverage {cov:.3f} even worse than published")
        self.assertLess(cov, 0.91,
                        f"coverage {cov:.3f} — if the bootstrap has been fixed, "
                        f"update the anti-conservatism note in "
                        f"src/experiment.py and src/methodology.py")
        # And the reports must actually carry the warning.
        diag = ROOT / "reports" / "results_diagnostic_v2.md"
        if diag.exists():
            body = diag.read_text()
            self.assertNotIn("is a real rejection", body)
            self.assertIn("anti-conservative", body)

    def test_se_within_subsample_is_not_labelled_a_standard_error_of_beta(self):
        """It is ~sqrt(step) x the full-sample OLS se, so it measures a
        different quantity than the coefficient it used to be printed beside."""
        rng = np.random.default_rng(3)
        n, step = 420, 21
        idx = pd.bdate_range("2016-01-04", periods=n)
        x = pd.Series(rng.standard_normal(n), index=idx).cumsum() / 10
        y = 1.0 * x + pd.Series(rng.standard_normal(n), index=idx)
        X = pd.DataFrame({"const": 1.0, "f": x}, index=idx)
        ph = methodology.nonoverlap_phases(y, X, step=step)
        self.assertIn("se_within_subsample", ph)
        self.assertNotIn("se_honest", ph, "the misleading key must be gone")
        Xv = X.values
        beta, *_ = np.linalg.lstsq(Xv, y.values, rcond=None)
        resid = y.values - Xv @ beta
        full_se = np.sqrt(np.diag(np.linalg.pinv(Xv.T @ Xv))
                          * (resid @ resid) / (n - 2))[1]
        ratio = ph["se_within_subsample"]["f"] / (full_se * np.sqrt(step))
        self.assertAlmostEqual(ratio, 1.0, delta=0.25,
                               msg="se_within_subsample should track "
                                   "sqrt(step) x the full-sample OLS se")


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


class TestDiagnosticOnlyQuarantine(unittest.TestCase):
    """The clean window must not be SCORED for quarantined models, anywhere.

    The original guard filtered only the h=1 table, so the corrected fork's
    replication panel reached into the clean window through its OTHER-window
    column and published clean win rates and mean gaps for `har_lev` and
    `har_iv_lev`. Suppressing a row from a report is not a quarantine; not
    looking is.
    """

    def setUp(self):
        from src import experiment as X
        self.X = X
        self.reg = config.load_spec_registry()

    def test_registry_and_module_tuple_agree(self):
        from_registry = {m for m, e in (self.reg["models"] or {}).items()
                         if e and e.get("diagnostic_only")}
        self.assertEqual(from_registry, set(self.X.DIAGNOSTIC_ONLY))

    def test_quarantined_in_clean_but_not_diagnostic(self):
        for m in self.X.DIAGNOSTIC_ONLY:
            self.assertTrue(self.X._quarantined(m, "clean"), m)
            self.assertTrue(self.X._quarantined(m, "all"), m)
            self.assertFalse(self.X._quarantined(m, "diagnostic"), m)

    def test_ordinary_models_are_never_quarantined(self):
        for m in ("har", "har_iv", "har_iv_x", "chronos_cov_iv"):
            self.assertFalse(self.X._quarantined(m, "clean"), m)

    def test_qlike_series_refuses_the_clean_window(self):
        if not (ROOT / "data" / "processed" / "master_daily.parquet").exists():
            self.skipTest("no processed master")
        cfg = config.load()
        cfg["_estimator"], cfg["_inference"], cfg["_grid_suffix"] = "smearing", "corrected", ""
        df = pd.read_parquet(ROOT / "data" / "processed" / "master_daily.parquet")
        taus = np.asarray(cfg["quantiles"])
        for m in self.X.DIAGNOSTIC_ONLY:
            self.assertIsNone(self.X._qlike_series(cfg, df, m, "clean", taus),
                              f"{m} was scored on the quarantined clean window")
        # Control: the guard must not be silently blocking everything.
        self.assertIsNotNone(self.X._qlike_series(cfg, df, "har", "clean", taus))

    def test_no_report_publishes_a_clean_number_for_a_quarantined_model(self):
        import re
        for path in sorted((ROOT / "reports").glob("results_diagnostic*.md")):
            body = path.read_text()
            block = re.search(r"^## Replication.*?(?=^## |\Z)", body,
                              re.S | re.M)
            if not block:
                continue
            for line in block.group(0).splitlines():
                if not line.startswith("| "):
                    continue
                name = line.split("|")[1].strip().split(" vs ")[0]
                if name in self.X.DIAGNOSTIC_ONLY:
                    self.assertNotIn("192", line,
                                     f"{path.name} publishes clean-window "
                                     f"origins for quarantined {name}: {line}")


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

    def test_frozen_reports_match_their_pinned_digests(self):
        """The external anchor, and the only check here that is not self-healing.

        `test_default_evaluate_reproduces_the_frozen_reports` below reads the
        current bytes, overwrites the file, and compares -- so it detects drift
        introduced by THAT subprocess and is structurally incapable of noticing
        that a frozen report had already diverged. Worse, it is self-healing:
        with a corrupted dependency, consecutive invocations launder the
        corruption into the "frozen" record and then go green. These digests
        were pinned once and are compared without writing anything, so a report
        that drifted at any point in the past still fails.

        Changing a digest is a pre-registration amendment, not a test fix.
        """
        import hashlib
        import json
        pin = ROOT / "reports" / "FROZEN_REPORT_HASHES.json"
        self.assertTrue(pin.exists(), "frozen-report digests are missing")
        want = json.loads(pin.read_text())["sha256"]
        self.assertTrue(want, "no digests pinned")
        for name, digest in want.items():
            path = ROOT / "reports" / name
            self.assertTrue(path.exists(), f"{name} is missing")
            got = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(
                got, digest,
                f"{name} no longer matches its pinned digest. If this change is "
                f"intended, it is an amendment to a frozen pre-registered "
                f"artifact: log it in reports/AMENDMENTS.md and re-pin "
                f"deliberately. Do not silently update the hash.")

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

    # The exact table printed in reports/METHODOLOGY_FORK.md, on the exact
    # sample it was computed from. The inequalities above pass for a wide
    # family of reconstruction schemes, which is how the published `~` rows
    # once drifted away from any method the repository still contained. This
    # pins the numbers instead, so a change to the estimator either updates
    # the report or fails here.
    DOC_MODELS = {"har": {}, "har_x": {"with_x": True}, "har_iv": {"with_iv": True},
                  "har_iv_x": {"with_x": True, "with_iv": True},
                  "har_ic": {"with_iv": True, "with_ic": True}}
    DOC_TABLE = {                      # method: (lo, hi, level_pct, spread_pp)
        "trunc":     (0.866, 0.873, -13.0, 0.70),
        "lognormal": (0.952, 0.962, -4.3, 1.00),
        "tail_ext":  (0.952, 0.976, -3.5, 2.49),
    }
    # The published 0.70pp "near-common factor" quantifies over DOC_MODELS only.
    # Every other quantile model scored in the same DM column is outside that
    # band -- persistence by 5.8pp -- so a test that pins only DOC_MODELS can
    # pass while the prose built on it is false. These pin the wider scopes the
    # reports actually claim.
    WIDER_MODELS = {"har_sv": {"with_sv": True}, "har_lev": {"with_lev": True},
                    "har_iv_lev": {"with_iv": True, "with_lev": True}}
    TRUNC_BAND_HAR_FAMILY = (0.866, 0.883, 1.64)     # lo, hi, spread_pp
    TRUNC_BAND_ALL_QUANTILE = (0.812, 0.883, 7.05)

    def test_reconstruction_table_matches_the_published_one(self):
        proc = ROOT / "data" / "processed" / "master_daily.parquet"
        if not proc.exists():
            self.skipTest("no processed master")
        from src import experiment as X
        cfg = config.load()
        cfg["_estimator"], cfg["_grid_suffix"] = "trunc", ""
        df = pd.read_parquet(proc)
        taus = np.asarray(cfg["quantiles"])
        origins = X._phase_origins(df, cfg, "clean")
        got = {m: [] for m in self.DOC_TABLE}
        for name, kw in self.DOC_MODELS.items():
            exact = models.run_har(df, origins, cfg, estimator="smearing", **kw)
            saved = X._load_forecast(cfg, name, "clean")
            if saved is None:
                self.skipTest(f"no saved forecasts for {name}")
            for meth in got:
                approx = models.rescore_mean_var(saved, taus, method=meth)
                j = pd.concat([exact["mean_var"].rename("e"), approx.rename("a")],
                              axis=1).dropna()
                got[meth].append(float((j["a"] / j["e"]).mean()))
        for meth, (lo, hi, level, spread) in self.DOC_TABLE.items():
            v = np.array(got[meth])
            self.assertAlmostEqual(v.min(), lo, places=3, msg=f"{meth} low end")
            self.assertAlmostEqual(v.max(), hi, places=3, msg=f"{meth} high end")
            self.assertAlmostEqual(100 * (v.mean() - 1), level, places=1,
                                   msg=f"{meth} level")
            self.assertAlmostEqual(100 * (v.max() - v.min()), spread, places=2,
                                   msg=f"{meth} cross-model spread")

    def test_the_near_common_factor_claim_is_scoped_to_its_own_model_set(self):
        """The 0.70pp band does not hold outside DOC_MODELS. Pin what does."""
        proc = ROOT / "data" / "processed" / "master_daily.parquet"
        if not proc.exists():
            self.skipTest("no processed master")
        from src import experiment as X
        cfg = config.load()
        cfg["_estimator"], cfg["_grid_suffix"] = "trunc", ""
        df = pd.read_parquet(proc)
        taus = np.asarray(cfg["quantiles"])
        origins = X._phase_origins(df, cfg, "clean")

        def _ratio(exact):
            saved = X._load_forecast(cfg, name, "clean")
            if saved is None:
                return None
            approx = models.rescore_mean_var(saved, taus, method="trunc")
            j = pd.concat([exact["mean_var"].rename("e"), approx.rename("a")],
                          axis=1).dropna()
            return float((j["a"] / j["e"]).mean())

        har_family = {}
        for name, kw in {**self.DOC_MODELS, **self.WIDER_MODELS}.items():
            r = _ratio(models.run_har(df, origins, cfg, estimator="smearing", **kw))
            if r is None:
                self.skipTest(f"no saved forecasts for {name}")
            har_family[name] = r
        name = "persistence"
        pers = _ratio(models.run_persistence(df, origins, cfg, estimator="smearing"))
        if pers is None:
            self.skipTest("no saved forecasts for persistence")

        fam = np.array(list(har_family.values()))
        lo, hi, spread = self.TRUNC_BAND_HAR_FAMILY
        self.assertAlmostEqual(fam.min(), lo, places=3)
        self.assertAlmostEqual(fam.max(), hi, places=3)
        self.assertAlmostEqual(100 * (fam.max() - fam.min()), spread, places=2)

        allq = np.append(fam, pers)
        lo, hi, spread = self.TRUNC_BAND_ALL_QUANTILE
        self.assertAlmostEqual(allq.min(), lo, places=3)
        self.assertAlmostEqual(allq.max(), hi, places=3)
        self.assertAlmostEqual(100 * (allq.max() - allq.min()), spread, places=2)
        # The point of the whole test: persistence is nowhere near the band the
        # "ranking is fair" argument is built on.
        self.assertGreater(min(har_family.values()) - pers, 0.05)


if __name__ == "__main__":
    unittest.main()
