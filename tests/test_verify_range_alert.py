"""Synthetic independent range-alert contracts, written before implementation."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.special import expit

from src import verify_range_alert as v


def bars(n=180):
    rng = np.random.default_rng(814)
    dates = pd.bdate_range("2010-01-04", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.008, n)))
    opening = close * np.exp(rng.normal(0, 0.006, n))
    high = np.maximum(close, opening) * np.exp(rng.uniform(0.003, 0.015, n))
    low = np.minimum(close, opening) * np.exp(-rng.uniform(0.003, 0.015, n))
    daily = pd.DataFrame(dict(open=opening, high=high, low=low, close=close), index=dates)
    iv = pd.DataFrame(
        dict(
            vix=rng.uniform(15, 30, n),
            vix9d=rng.uniform(14, 29, n),
            vvix=rng.uniform(80, 120, n),
            skew=rng.uniform(100, 150, n),
        ),
        index=dates,
    )
    return daily, iv


class MeasurementContracts(unittest.TestCase):
    def test_range_endpoints_center_and_zero_range(self):
        d, _ = bars(4)
        d.loc[:, ["open", "low", "high", "close"]] = [
            [2, 1, 4, 1],
            [2, 1, 4, 4],
            [2, 1, 4, 2],
            [2, 2, 2, 2],
        ]
        m = v.measurements(d)
        np.testing.assert_array_equal(m.range_extremity.iloc[:3], [1.0, 1.0, 0.0])
        self.assertTrue(np.isnan(m.range_extremity.iloc[3]))
        self.assertTrue(np.isfinite(m.risk.iloc[3]))

    def test_missing_open_unknown_E_but_independent_partial_validation(self):
        d, _ = bars(30)
        d.loc[d.index[10], "open"] = np.nan
        self.assertTrue(np.isnan(v.measurements(d).range_extremity.iloc[10]))
        d.loc[d.index[10], "close"] = d.high.iloc[10] * 2
        with self.assertRaises(ValueError):
            v.measurements(d)

    def test_units_and_no_future_mutation(self):
        d, iv = bars()
        f, t = v.feature_target_tables(d, iv)
        ff, tt = v.feature_target_tables(d * 8, iv)
        np.testing.assert_allclose(
            f.range_extremity, ff.range_extremity, equal_nan=True, rtol=1e-12
        )
        np.testing.assert_array_equal(t.y, tt.y)
        changed = d.copy()
        changed.iloc[120:] *= 1.01
        cf, ct = v.feature_target_tables(changed, iv)
        pd.testing.assert_frame_equal(f.iloc[:120], cf.iloc[:120])
        pd.testing.assert_frame_equal(t.iloc[:119], ct.iloc[:119])

    def test_strict_event_window_ties_and_missing(self):
        dates = pd.bdate_range("2010-01-01", periods=30)
        risk = pd.Series(np.ones(30), index=dates)
        risk.iloc[24] = 2
        t = v.event_targets(risk)
        self.assertEqual(t.y.iloc[23], 0)
        risk.iloc[24] = np.nextafter(2.0, np.inf)
        self.assertEqual(v.event_targets(risk).y.iloc[23], 1)
        risk.iloc[1] = np.nan
        self.assertTrue(np.isnan(v.event_targets(risk).y.iloc[23]))
        risk.iloc[1] = 1
        risk.iloc[23] = 999
        self.assertEqual(v.event_targets(risk).y.iloc[23], 1)  # current t excluded
        self.assertTrue(t.y.iloc[:22].isna().all())

    def test_labels_do_not_depend_on_E_or_IV_completeness(self):
        d, iv = bars()
        _, t = v.feature_target_tables(d, iv)
        iv.loc[:, :] = np.nan
        f, tt = v.feature_target_tables(d, iv)
        pd.testing.assert_frame_equal(t, tt)
        self.assertTrue(f.I.isna().all())

    def test_invalid_arithmetic_rejects_not_missing(self):
        d, _ = bars(30)
        d.loc[d.index[4], ["high", "low"]] = [1e308, 1e-308]
        with self.assertRaises(ValueError):
            v.measurements(d)
        for bad in (0, -1, np.inf):
            d, _ = bars(30)
            d.loc[d.index[4], "open"] = bad
            with self.assertRaises(ValueError):
                v.measurements(d)

    def test_exact_compensated_mean_and_threshold_overflow(self):
        dates = pd.bdate_range("2010-01-01", periods=26)
        x = pd.Series(np.full(26, np.finfo(float).max), index=dates)
        with self.assertRaises(ValueError):
            v.event_targets(x)
        x[:] = 1
        x.iloc[2] = np.nextafter(1.0, 2.0)
        import math

        self.assertEqual(v.reference_mean(x).iloc[23], math.fsum(x.iloc[1:23]) / 22)


class NumericalContracts(unittest.TestCase):
    def test_transform_training_only_and_constant_E(self):
        d, iv = bars(180)
        f, _ = v.feature_target_tables(d, iv)
        f = f.dropna()
        a = f.iloc[:70].copy()
        b = f.iloc[70:].copy()
        a["range_extremity"] = 0.1
        x, _q, e, _qe, audit = v.transform(a, b)
        self.assertEqual(x.shape[1], 26)
        self.assertTrue((e == 0).all())
        self.assertTrue(audit["range_extremity_constant"])
        other = b.copy()
        other.loc[:, v.ALL_FEATURES[1:-1]] = 2 * other.loc[:, v.ALL_FEATURES[1:-1]]
        xx, _, ee, _, aa = v.transform(a, other)
        np.testing.assert_array_equal(x, xx)
        np.testing.assert_array_equal(e, ee)
        self.assertEqual(audit, aa)
        a["I"] = 0.3
        with self.assertRaises(ValueError):
            v.transform(a, b)

    def test_logistic_gradient_and_independent_root(self):
        rng = np.random.default_rng(13)
        x = np.c_[np.ones(800), rng.normal(size=(800, 25))]
        y = (rng.random(800) < expit(0.2 + 0.4 * x[:, 1])).astype(float)
        b = rng.normal(scale=0.1, size=26)
        _, g, h = v.logistic_objective(b, x, y)
        eps = 1e-5
        for j in (0, 1, 25):
            step = np.eye(26)[j] * eps
            fd = (
                v.logistic_objective(b + step, x, y)[0]
                - v.logistic_objective(b - step, x, y)[0]
            ) / (2 * eps)
            self.assertAlmostEqual(g[j], fd, places=8)
        fit = v.independent_baseline_fit(x, y)
        self.assertTrue(fit["success"])
        self.assertLessEqual(fit["gradient_max_abs"], 1.0001e-8)
        self.assertEqual(fit["start"], [0.0] * 26)
        np.testing.assert_allclose(h, h.T, atol=1e-14)

    def test_solver_status_and_gradient_are_both_mandatory(self):
        rng = np.random.default_rng(1)
        x = np.c_[np.ones(100), rng.normal(size=(100, 25))]
        y = np.tile([0.0, 1.0], 50)
        for success, beta in ((False, np.zeros(26)), (True, np.ones(26))):
            fake = SimpleNamespace(
                x=beta, success=success, status=4, message="synthetic", nfev=1, njev=1
            )
            with patch.object(v, "root", return_value=fake), self.assertRaises(ValueError):
                v.independent_baseline_fit(x, y)

    def test_scalar_flat_and_monotone_root(self):
        y = np.tile([0.0, 1.0], 100)
        eta = np.full(200, 0.3)
        e = np.linspace(-0.4, 0.4, 200)
        result = v.independent_location_fit(eta, y, e)
        self.assertLessEqual(result["gradient_max_abs"], 1.0001e-8)
        flat = v.independent_location_fit(eta, y, np.zeros(200))
        self.assertEqual(flat["b"], 0.0)

    def test_explicit_filter_no_seed_refeed_and_missing_decay(self):
        dates = pd.bdate_range("2016-01-01", periods=8)
        t = pd.DataFrame(
            {
                "y": [1.0, 0.0, 1.0, np.nan, 1.0, 0.0, 1.0, np.nan],
                "target_end": pd.Series(dates, index=dates).shift(-1),
                "available_date": pd.Series(dates, index=dates).shift(-1),
            },
            index=dates,
        )
        f = v.explicit_states(t, dates, dates[2], 0.4, dates[1], dates[6])
        self.assertEqual(f.iloc[0].S, 0.4)
        self.assertEqual(f.iloc[0].cumulative_updates, 0)
        self.assertEqual(f.iloc[2].cumulative_updates, 1)
        self.assertAlmostEqual(f.iloc[2].recent_frequency, f.iloc[1].recent_frequency)
        self.assertEqual(f.iloc[-1].cumulative_updates, 3)


class PublicationContracts(unittest.TestCase):
    def test_guard_nonfinite_metrics_preserves_two_terminal_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            rootdir = Path(directory)
            report = rootdir / "reports/range_alert"
            report.mkdir(parents=True)
            (report / "metrics.json").write_text('{"rows": [], "delta": NaN}')
            with (
                patch.object(v, "verify", side_effect=ValueError("synthetic")),
                self.assertRaises(ValueError),
            ):
                v.verify_with_failure_guard(rootdir)
            m = json.loads((report / "metrics.json").read_text())
            self.assertEqual(m["cumulative_hypothesis_count"], 129)
            self.assertEqual(m["leads"], [])
            self.assertEqual(len(m["rows"]), 2)
            self.assertTrue(
                all(
                    r["p_conservative"] == r["p_holm_wave"] == r["p_holm_cumulative"] == 1
                    for r in m["rows"]
                )
            )
            self.assertEqual(len((report / "trial_ledger.jsonl").read_text().splitlines()), 2)
            self.assertTrue((report / "unpublished_invalid_metrics.txt").exists())

    def test_protocol_extra_and_numeric_mutation_rejected(self):
        for mutation in (
            lambda p: p.update(extra=True),
            lambda p: p["verification"]["baseline"].update(factor=10),
        ):
            p = copy.deepcopy(v.CONTRACT)
            mutation(p)
            with self.assertRaises(ValueError):
                v.validate_protocol(p)


if __name__ == "__main__":
    unittest.main()


class SourceClosureContracts(unittest.TestCase):
    def fixture(self, rootdir):
        def put(name, value):
            path = rootdir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = value if isinstance(value, bytes) else json.dumps(value).encode()
            path.write_bytes(payload)
            import hashlib

            return hashlib.sha256(payload).hexdigest()

        ph = put("old.yaml", b"old: true\n")
        vh = put("src/verify_old.py", b"# frozen independent verifier")
        pin = put("raw.dat", b"bounded synthetic original source")
        manifest = {
            "protocol_sha256": ph,
            "code": {"src/verify_old.py": vh},
            "inputs": {"raw.dat": pin},
            "preserved": {},
        }
        mh = put("reports/old/manifest.json", manifest)
        verified = {"status": "VERIFIED", "protocol_sha256": ph, "verifier_sha256": vh}
        vhash = put("reports/old/verification.json", verified)
        return {
            "protocol": "old.yaml",
            "reports": "reports/old",
            "data": "data/old",
            "protocol_sha256": ph,
            "manifest_sha256": mh,
            "verification_sha256": vhash,
            "verifier_sha256": vh,
            "required_status": "VERIFIED",
        }

    def test_anchor_closure_identity_and_no_prior_writes(self):
        with tempfile.TemporaryDirectory() as d:
            rootdir = Path(d)
            info = self.fixture(rootdir)
            before = {str(p): p.read_bytes() for p in rootdir.rglob("*") if p.is_file()}
            proof = v.admit_anchor(rootdir, info)
            self.assertEqual(
                proof["entries_checked"], {"code": 1, "inputs": 1, "preserved": 0}
            )
            self.assertEqual(
                before, {str(p): p.read_bytes() for p in rootdir.rglob("*") if p.is_file()}
            )
            (rootdir / "raw.dat").write_bytes(b"tampered")
            with self.assertRaises(AssertionError):
                v.admit_anchor(rootdir, info)

    def test_source_same_bytes_before_decode_and_no_decoder_on_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            rootdir = Path(d)
            (rootdir / "raw").write_bytes(b"changed")
            p = {"source_files": {"raw": "0" * 64}}
            with (
                patch.object(v.source, "load_source_tables") as decoder,
                self.assertRaises(AssertionError),
            ):
                v.load_source_tables(rootdir, p, {"inputs": {"raw": "0" * 64}})
            decoder.assert_not_called()

    def test_inherited_json_snapshot_rejects_tamper_and_nan(self):
        with tempfile.TemporaryDirectory() as d:
            rootdir = Path(d)
            p = {"comparisons": {"inherited_sources": ["metrics.json"]}}
            (rootdir / "metrics.json").write_text('{"rows":[]}')
            with self.assertRaises(AssertionError):
                v.inherited_rows(rootdir, p, pins={"metrics.json": "0" * 64})

    def test_exact_endpoint_domain_before_tolerance(self):
        # A coherent tiny negative p and its squared loss may fit replay tolerances;
        # literal probability domains must nevertheless reject it.
        dates = pd.bdate_range("2016-01-01", periods=5)
        rows = []
        for model in v.MODELS:
            rows.append(
                dict(
                    origin=dates[2],
                    model=model,
                    horizon=1,
                    feature_cutoff_date=dates[1],
                    target_end=dates[3],
                    available_date=dates[3],
                    y=0.0,
                    probability=-1e-15,
                    loss=1e-30,
                    fit_origin=dates[2],
                    fit_cutoff_date=dates[1],
                    train_n=1000,
                    train_last_target=dates[1],
                    train_last_available=dates[1],
                    phase="development",
                )
            )
        with self.assertRaises(ValueError):
            v.validate_panel(pd.DataFrame(rows, columns=v.PANEL_COLUMNS))


class WholeEntryContracts(unittest.TestCase):
    def test_whole_entry_exact_output_paths_and_snapshots(self):
        import yaml

        with tempfile.TemporaryDirectory() as d:
            r = Path(d)
            out = r / "data/range_alert"
            report = r / "reports/range_alert"
            out.mkdir(parents=True)
            report.mkdir(parents=True)
            p = copy.deepcopy(v.CONTRACT)
            payload = yaml.safe_dump(p).encode()
            (r / "range_alert.yaml").write_bytes(payload)
            import hashlib

            ph = hashlib.sha256(payload).hexdigest()
            manifest = {"protocol_sha256": ph, "code": {}, "inputs": {}, "preserved": {}}
            (report / "manifest.json").write_text(json.dumps(manifest))
            daily, iv = bars()
            f, t = v.feature_target_tables(daily, iv)
            f.to_parquet(out / "features.parquet")
            t.to_parquet(out / "targets.parquet")
            pd.DataFrame({"placeholder": [1]}).to_parquet(out / "forecasts.parquet")
            pd.DataFrame({"origin": [f.index[-2]]}).to_parquet(out / "states.parquet")
            for name, value in [
                ("upstream_admission", {"status": "synthetic"}),
                ("source_audit", {"synthetic": True}),
                ("support_audit", {"fits": []}),
                ("fits", []),
            ]:
                (out / (name + ".json")).write_text(json.dumps(value))
            metrics = {
                "protocol_sha256": ph,
                "evidence_class": p["evidence_class"],
                "inherited_rows": [],
            }
            (report / "metrics.json").write_text(json.dumps(metrics))
            (report / "trial_ledger.jsonl").write_text("")
            with (
                patch.object(v, "verify_manifest_coverage"),
                patch.object(v, "validate_upstream", return_value={"status": "synthetic"}),
                patch.object(
                    v, "load_source_tables", return_value=(daily, iv, {"synthetic": True})
                ),
                patch.object(v, "preflight", return_value={"fits": []}),
                patch.object(
                    v,
                    "verify_forecasts",
                    return_value={"forecasts_verified": 1, "monthly_fits_verified": 0},
                ) as forecasts,
                patch.object(v, "verify_metrics", return_value={"new_hypotheses_verified": 2}),
                patch.object(v, "verify_ledger", return_value={}) as ledger,
            ):
                result = v.verify(r)
            self.assertEqual(result["status"], "VERIFIED")
            self.assertTrue((report / "verification.json").exists())
            self.assertEqual(forecasts.call_args.args[2].columns.tolist(), ["placeholder"])
            self.assertEqual(forecasts.call_args.args[4].columns.tolist(), ["origin"])
            self.assertIn("signature", ledger.call_args.kwargs)
            self.assertEqual(
                result["artifact_hashes_checked"],
                {"code": 0, "inputs": 0, "preserved": 0, "outputs": 10},
            )

    def test_target_binary_comparison_is_exact(self):
        daily, iv = bars()
        _, t = v.feature_target_tables(daily, iv)
        changed = t.copy()
        date = changed.y.dropna().index[0]
        changed.loc[date, "y"] += 1e-15
        with self.assertRaises(AssertionError):
            v.compare_targets(changed, t)


class IndependentRawComparison(unittest.TestCase):
    def test_all24_predictors_and_exact_binary_targets_match_generated_source(self):
        from src import range_alert_features as producer

        daily, iv = bars(240)
        daily.loc[daily.index[80], ["high", "low", "open", "close"]] = [
            100.0,
            100.0,
            100.0,
            100.0,
        ]
        iv.loc[iv.index[100:110], "skew"] = np.nan
        actual_f, actual_t = producer.build_features(daily, iv)
        expected_f, expected_t = v.feature_target_tables(daily, iv)
        pd.testing.assert_frame_equal(actual_f, expected_f, rtol=1e-12, atol=1e-14)
        pd.testing.assert_frame_equal(actual_t, expected_t, check_exact=True)

    def test_invalid_IV_outside_reference_dates_not_silently_ignored(self):
        daily, iv = bars(40)
        iv.loc[daily.index[-1] + pd.Timedelta(days=3)] = [-1.0, 20.0, 90.0, 110.0]
        with self.assertRaises(ValueError):
            v.feature_target_tables(daily, iv)

    def test_complete_risk_window_computed_nan_is_not_missing_source(self):
        daily, iv = bars(40)
        original = pd.Series.rolling

        class BrokenRolling:
            def __init__(self, series):
                self.series = series

            def mean(self):
                return pd.Series(np.nan, index=self.series.index)

        def replaced(series, *args, **kwargs):
            if series.name == "risk":
                return BrokenRolling(series)
            return original(series, *args, **kwargs)

        with patch.object(pd.Series, "rolling", replaced), self.assertRaises(ValueError):
            v.feature_target_tables(daily, iv)


class TypedProtocolContracts(unittest.TestCase):
    def test_boolean_cannot_substitute_for_integer_or_float(self):
        for path in [("index", "market_lag"), ("verification", "baseline", "factor")]:
            p = copy.deepcopy(v.CONTRACT)
            node = p
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = True
            with self.subTest(path=path), self.assertRaises(ValueError):
                v.validate_protocol(p)
