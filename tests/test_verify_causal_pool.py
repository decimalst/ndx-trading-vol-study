"""Prewritten independent weighted-sum, timing and preservation contracts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import verify_causal_pool as verify


def target_fixture(n=80):
    dates = pd.bdate_range("2015-12-01", periods=n)
    maturity = pd.Series(dates, index=dates).shift(-1)
    y = pd.Series((np.arange(n) % 3 != 0).astype(float), index=dates)
    y.iloc[-1] = np.nan
    return dates, pd.DataFrame({"y": y, "target_end": maturity, "available_date": maturity})


class ExplicitStateContracts(unittest.TestCase):
    def states(self, targets, dates):
        return verify.explicit_states(targets, dates, dates[20], 0.6, dates[18], dates[65])

    def test_first_cutoff_seed_is_not_refed_and_included_date_is_separate(self):
        dates, t = target_fixture()
        state = self.states(t, dates)
        first = state.loc[dates[20]]
        self.assertEqual(first.S, 0.6)
        self.assertEqual(first.W, 1.0)
        self.assertEqual(first.recent_frequency, 0.6)
        self.assertEqual(first.latest_consumed_available, dates[18])
        self.assertEqual(first.elapsed_sessions, 0)
        self.assertEqual(first.cumulative_updates, 0)
        changed = t.copy()
        changed.loc[changed.available_date <= dates[20], "y"] = (
            1 - changed.loc[changed.available_date <= dates[20], "y"]
        )
        pd.testing.assert_frame_equal(state, self.states(changed, dates), check_exact=True)

    def test_each_label_enters_only_when_available_and_future_mutations_do_not_change_prefix(
        self,
    ):
        dates, t = target_fixture()
        state = self.states(t, dates)
        changed = t.copy()
        changed.loc[dates[35], "y"] = 1 - changed.loc[dates[35], "y"]
        actual = self.states(changed, dates)
        pd.testing.assert_frame_equal(
            state.loc[: dates[35]], actual.loc[: dates[35]], check_exact=True
        )
        self.assertNotEqual(
            state.loc[dates[36], "recent_frequency"], actual.loc[dates[36], "recent_frequency"]
        )
        self.assertEqual(state.loc[dates[36], "latest_consumed_available"], dates[36])

    def test_missing_labels_decay_both_masses_on_every_reference_session(self):
        dates, t = target_fixture()
        t.loc[dates[30:33], "y"] = np.nan
        state = self.states(t, dates)
        delta = 2 ** (-1 / 63)
        before = state.loc[dates[30]]
        for count, date in enumerate(dates[31:34], 1):
            self.assertAlmostEqual(state.loc[date, "S"], before.S * delta**count)
            self.assertAlmostEqual(state.loc[date, "W"], before.W * delta**count)
            self.assertAlmostEqual(
                state.loc[date, "recent_frequency"], before.recent_frequency
            )
            self.assertEqual(state.loc[date, "cumulative_updates"], before.cumulative_updates)
            self.assertEqual(
                state.loc[date, "latest_consumed_available"], before.latest_consumed_available
            )

    def test_explicit_states_match_separate_synthetic_recurrence_without_selecting_origins(
        self,
    ):
        dates, t = target_fixture()
        t.loc[dates[25], "y"] = 0.0
        state = self.states(t, dates)
        delta = 2 ** (-1 / 63)
        s = 0.6
        w = 1.0
        updates = 0
        for k, date in enumerate(dates[20:66]):
            if k:
                s *= delta
                w *= delta
                label = t.loc[t.available_date == date, "y"].iloc[0]
                if np.isfinite(label):
                    s += (1 - delta) * label
                    w += 1 - delta
                    updates += 1
            verify.state_equal(s, state.loc[date, "S"], k, "numerator")
            verify.state_equal(w, state.loc[date, "W"], k, "denominator")
            self.assertEqual(state.loc[date, "cumulative_updates"], updates)

    def test_invalid_calendar_availability_binary_seed_and_nonfinite_values_reject(self):
        dates, t = target_fixture()
        for fault in ("available", "target_end", "binary", "infinite"):
            bad = t.copy()
            if fault == "available":
                bad.loc[dates[25], "available_date"] = dates[25]
            elif fault == "target_end":
                bad.loc[dates[25], "target_end"] = dates[27]
            elif fault == "binary":
                bad.loc[dates[25], "y"] = 0.5
            else:
                bad.loc[dates[25], "y"] = np.inf
            with self.assertRaises((AssertionError, ValueError)):
                self.states(bad, dates)
        with self.assertRaises((AssertionError, ValueError)):
            verify.explicit_states(t, dates, dates[20], 0.0, dates[18], dates[65])

    def test_state_comparison_preserves_exact_zero_sign_and_declared_operation_budget(self):
        verify.state_equal(0.6, 0.6, 10, "same")
        with self.assertRaises(AssertionError):
            verify.state_equal(0.0, np.nextafter(0.0, 1.0), 10, "zero mask")
        with self.assertRaises(AssertionError):
            verify.state_equal(-1e-15, 1e-15, 10, "sign mask")
        with self.assertRaises(AssertionError):
            verify.state_equal(0.6 + 1e-8, 0.6, 100, "large discrepancy")
        with self.assertRaises(ValueError):
            verify.multiply(np.nextafter(0.0, 1.0), 0.5)

    def test_nonreal_state_values_cannot_be_cast_to_an_accepted_real_state(self):
        with self.assertRaises((AssertionError, ValueError, TypeError)):
            verify.state_equal(np.complex128(0.6 + 1j), 0.6, 5, "nonreal numerator")


class AdmissionAndFailureContracts(unittest.TestCase):
    def fixture(self, root):
        import yaml

        oldreport = root / "reports/sign_memory"
        olddata = root / "data/sign_memory"
        report = root / "reports/causal_pool"
        for folder in (oldreport, olddata, report, root / "src"):
            folder.mkdir(parents=True, exist_ok=True)
        (root / "src/verify_sign_memory.py").write_text("frozen previous verifier")
        (root / "raw.bin").write_bytes(b"original raw bytes")
        (olddata / "forecasts.parquet").write_bytes(b"original issued bytes")
        (root / "sign_memory.yaml").write_text("evidence_class: synthetic\n")
        oldp = verify.digest(root / "sign_memory.yaml")
        oldv = verify.digest(root / "src/verify_sign_memory.py")
        oldm = {
            "protocol_sha256": oldp,
            "code": {"src/verify_sign_memory.py": oldv},
            "inputs": {"raw.bin": verify.digest(root / "raw.bin")},
            "preserved": {},
        }
        (oldreport / "manifest.json").write_text(json.dumps(oldm))
        record = {"status": "VERIFIED", "protocol_sha256": oldp, "verifier_sha256": oldv}
        (oldreport / "verification.json").write_text(json.dumps(record))
        p = {
            "upstream": {
                "protocol": "sign_memory.yaml",
                "reports": "reports/sign_memory",
                "data": "data/sign_memory",
                "protocol_sha256": oldp,
                "verifier_sha256": oldv,
                "required_status": "VERIFIED",
                "manifest_sha256": verify.digest(oldreport / "manifest.json"),
                "verification_sha256": verify.digest(oldreport / "verification.json"),
            }
        }
        (root / "causal_pool.yaml").write_text(yaml.safe_dump(p))
        inputs = {**oldm["inputs"], "sign_memory.yaml": oldp}
        inputs.update(
            {
                str(p.relative_to(root)): verify.digest(p)
                for folder in (oldreport, olddata)
                for p in folder.rglob("*")
                if p.is_file()
            }
        )
        manifest = {
            "protocol_sha256": verify.digest(root / "causal_pool.yaml"),
            "code": oldm["code"],
            "inputs": inputs,
            "preserved": {},
        }
        (report / "manifest.json").write_text(json.dumps(manifest))
        return record

    def test_upstream_admission_writes_nothing_and_rejects_changed_raw_issued_or_verified_bytes(
        self,
    ):
        for fault in (
            None,
            "raw.bin",
            "data/sign_memory/forecasts.parquet",
            "reports/sign_memory/verification.json",
        ):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                record = self.fixture(root)
                if fault:
                    with (root / fault).open("ab") as stream:
                        stream.write(b"changed")
                before = {
                    str(p.relative_to(root)): p.read_bytes()
                    for p in root.rglob("*")
                    if p.is_file()
                }
                with patch.object(verify, "reconstruct_previous", return_value=(record, {})):
                    if fault:
                        with self.assertRaises(AssertionError):
                            verify.validate_upstream(root)
                    else:
                        self.assertEqual(
                            verify.validate_upstream(root)["status"],
                            "UPSTREAM_VERIFIED_READ_ONLY",
                        )
                self.assertEqual(
                    before,
                    {
                        str(p.relative_to(root)): p.read_bytes()
                        for p in root.rglob("*")
                        if p.is_file()
                    },
                )

    def test_missing_original_code_pin_or_added_old_failure_rejected(self):
        for fault in ("missing", "added"):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                record = self.fixture(root)

                def reconstruct(*args, root=root, record=record, fault=fault):
                    if fault == "added":
                        (root / "reports/sign_memory/failure.json").write_text("{}")
                    return record, {}

                if fault == "missing":
                    path = root / "reports/causal_pool/manifest.json"
                    m = json.loads(path.read_text())
                    m["code"] = {}
                    path.write_text(json.dumps(m))
                with (
                    patch.object(verify, "reconstruct_previous", side_effect=reconstruct),
                    self.assertRaises(AssertionError),
                ):
                    verify.validate_upstream(root)

    def test_new_failure_retains_both_hypotheses_with_malformed_or_nonfinite_backup(self):
        for payload in (
            '{"leads":["pooled"]}',
            "{broken",
            '{"value":NaN}',
            '{"value":Infinity}',
        ):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                report = root / "reports/causal_pool"
                report.mkdir(parents=True)
                old = root / "reports/sign_memory"
                old.mkdir()
                (old / "proof").write_bytes(b"frozen")
                (report / "metrics.json").write_text(payload)
                with (
                    patch.object(verify, "verify", side_effect=AssertionError("failure")),
                    self.assertRaises(AssertionError),
                ):
                    verify.verify_with_failure_guard(root)
                failure = json.loads((report / "metrics.json").read_text())
                self.assertEqual(failure["leads"], [])
                self.assertEqual(failure["status"], "UNEVALUABLE")
                self.assertEqual(failure["cumulative_hypothesis_count"], 121)
                self.assertEqual(len(failure["rows"]), 2)
                self.assertTrue(
                    all(
                        row[key] == 1
                        for row in failure["rows"]
                        for key in ("p_conservative", "p_holm_wave", "p_holm_cumulative")
                    )
                )
                self.assertEqual(
                    len((report / "trial_ledger.jsonl").read_text().splitlines()), 2
                )
                self.assertEqual((old / "proof").read_bytes(), b"frozen")

    def test_frozen_protocol_rejects_seed_timing_memory_weight_arithmetic_and_family_mutations(
        self,
    ):
        from copy import deepcopy

        import yaml

        p = yaml.safe_load((verify.ROOT / "causal_pool.yaml").read_text())
        verify.validate_protocol(p)
        for group, key, value in (
            ("pooling", "half_life_sessions", 62),
            ("pooling", "baseline_weight", 0.6),
            ("pooling", "seed", "Full phase event frequency"),
            ("index", "market_lag", 0),
            ("verification", "state_relative_roundoff_multiplier", 65),
            ("comparisons", "cumulative_hypotheses", 120),
        ):
            bad = deepcopy(p)
            bad[group][key] = value
            with self.assertRaises(ValueError):
                verify.validate_protocol(bad)
        extra = deepcopy(p)
        extra["unregistered_choice"] = True
        with self.assertRaises(ValueError):
            verify.validate_protocol(extra)

    def test_metadata_parser_uses_only_the_hash_checked_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "metadata.json"
            path.write_text('{"frozen":1}')
            digest = verify.digest(path)
            with patch.object(
                Path, "read_text", side_effect=AssertionError("must not reopen text")
            ):
                self.assertEqual(
                    verify.read_json_snapshot(root, "metadata.json", digest), {"frozen": 1}
                )
            path.write_text('{"frozen":2}')
            with self.assertRaises(AssertionError):
                verify.read_json_snapshot(root, "metadata.json", digest)


class FullPoolContracts(unittest.TestCase):
    def fixture(self, **options):
        import yaml

        from tests.test_causal_pool_models import fixture, run

        args = fixture(**options)
        issued, t, dates, fits, section, applications = args
        panel, states = run(args)
        f = pd.DataFrame(1.0, index=dates, columns=verify.previous.ALL_FEATURES)
        inside = (dates >= section["origin_start"]) & (dates <= section["origin_end"])
        f.loc[inside & ~dates.isin(applications), "corr22"] = np.nan
        f["feature_cutoff_date"] = pd.Series(dates, index=dates).shift()
        p = yaml.safe_load((verify.ROOT / "causal_pool.yaml").read_text())
        p["index"] = section
        return f, t, issued, fits, panel, states, p

    def test_explicit_full_clock_reproduces_every_saved_state_and_three_forecast_cohorts(self):
        args = self.fixture(missing_labels=(1380, 1400), omit_applications=(1382, 1383))
        result = verify.verify_forecasts(*args)
        self.assertEqual(result["new_monthly_fits"], 0)
        self.assertEqual(result["new_forecasts_verified"], 2 * args[4].origin.nunique())
        self.assertEqual(result["preserved_forecasts_verified"], args[4].origin.nunique())
        self.assertEqual(result["application_states_verified"], len(args[5]))
        self.assertGreater(result["full_calendar_states_reconstructed"], len(args[5]))

    def test_seed_gap_unscored_first_and_whole_unscored_month_are_retained(self):
        args = self.fixture(seed_gap=2, unscored_first=True, unscored_month=True)
        result = verify.verify_forecasts(*args)
        self.assertGreater(
            result["common_application_origins"], result["common_scored_origins"]
        )
        first = args[5].iloc[0]
        self.assertLess(first.seed_last_available, first.seed_cutoff_date)

    def test_state_clock_seed_mass_probability_pool_and_original_row_tampering_reject(self):
        from copy import deepcopy

        fixture = self.fixture(omit_applications=(1382,))
        for fault in (
            "S",
            "W",
            "q",
            "updates",
            "elapsed",
            "last",
            "seed",
            "original",
            "pool",
            "missing",
            "unscored",
        ):
            f, t, old, fits, panel, states, p = deepcopy(fixture)
            if fault in ("S", "W"):
                states.loc[5, fault] += 0.001
            elif fault == "q":
                states.loc[5, "recent_frequency"] += 1e-13
            elif fault == "updates":
                states.loc[5, "cumulative_updates"] += 1
            elif fault == "elapsed":
                states.loc[5, "elapsed_sessions"] += 1
            elif fault == "last":
                states.loc[5, "latest_consumed_available"] = states.loc[5, "origin"]
            elif fault == "seed":
                states.loc[0, "seed_last_available"] = states.loc[0, "origin"]
            elif fault == "unscored":
                states = states.iloc[1:]
            elif fault == "missing":
                panel = panel.iloc[1:]
            else:
                model = "frozen_baseline" if fault == "original" else "pooled"
                idx = panel.index[panel.model.eq(model)][0]
                panel.loc[idx, "probability"] += 1e-13
                panel.loc[idx, "loss"] = (
                    panel.loc[idx, "y"] - panel.loc[idx, "probability"]
                ) ** 2
            with self.subTest(fault=fault), self.assertRaises((AssertionError, ValueError)):
                verify.verify_forecasts(f, t, old, fits, panel, states, p)


class InferenceAndPublicationContracts(unittest.TestCase):
    def test_four_phase_inference_calibration_and_123_events_preserve_121_hypotheses(self):
        from copy import deepcopy

        from src import causal_pool_search as producer
        from tests.test_causal_pool_search import panel, protocol

        frame, calendar = panel()
        p = protocol()
        p["inference"]["bootstrap_draws"] = 39
        prior = [
            {
                "study": "synthetic",
                "candidate": str(i),
                "control": "baseline",
                "horizon": 1,
                "p_conservative": 1.0,
            }
            for i in range(119)
        ]
        m = producer.evaluate(frame, calendar, p, prior=prior)
        m["common_application_origins"] = frame.origin.nunique() + 1
        with patch.object(verify, "inherited_rows", return_value=prior):
            result = verify.verify_metrics(
                verify.ROOT, frame, p, m, m["common_application_origins"]
            )
        self.assertEqual(result["bootstrap_runs_verified"], 12)
        self.assertEqual(result["calibration_rows_verified"], 6)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = root / "reports/causal_pool"
            report.mkdir(parents=True)
            ledger = [{"event": "inherited", **row} for row in prior]
            ledger += [
                {
                    "event": "registered",
                    "study": "causal_pool",
                    "candidate": a,
                    "control": b,
                    "horizon": 1,
                    "score": "brier",
                    "protocol_sha256": m["protocol_sha256"],
                }
                for a, b in verify.COMPARISONS
            ]
            ledger += [{"event": "evaluated", **row} for row in m["rows"]]
            (report / "trial_ledger.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in ledger)
            )
            signature = verify.digest(report / "trial_ledger.jsonl")
            with patch.object(
                Path,
                "read_text",
                side_effect=AssertionError("decode only checked ledger bytes"),
            ):
                self.assertEqual(
                    verify.verify_ledger(root, m, prior, "evaluated", signature=signature),
                    {"inherited": 119, "registered": 2, "evaluated": 2},
                )
            with (report / "trial_ledger.jsonl").open("ab") as stream:
                stream.write(b"\n")
            with self.assertRaises(AssertionError):
                verify.verify_ledger(root, m, prior, "evaluated", signature=signature)
        for fault in ("counts", "p", "delta", "calibration", "support", "mde"):
            bad = deepcopy(m)
            if fault == "counts":
                bad["new_forecasts"] += 1
            elif fault == "p":
                bad["rows"][0]["p_conservative"] += 0.01
            elif fault == "delta":
                bad["rows"][0]["phases"][0]["delta"] += 0.01
            elif fault == "calibration":
                bad["calibration"]["development"]["pooled"]["mean_probability"] += 0.01
            elif fault == "support":
                bad["class_support"]["development"]["events"] += 1
            else:
                bad["rows"][0]["phases"][0]["nominal_mde_effect_ratio"] *= 100
            with (
                patch.object(verify, "inherited_rows", return_value=prior),
                self.assertRaises(AssertionError),
            ):
                verify.verify_metrics(
                    verify.ROOT, frame, p, bad, m["common_application_origins"]
                )

    def test_inherited_metric_snapshot_identity_cannot_change_between_decode_and_digest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = root / "reports/causal_pool"
            report.mkdir(parents=True)
            path = root / "old.json"
            rows = [
                {"candidate": str(i), "control": "b", "horizon": 1, "p_conservative": 1.0}
                for i in range(119)
            ]
            path.write_text(json.dumps({"rows": rows}))
            pin = verify.digest(path)
            (report / "manifest.json").write_text(json.dumps({"inputs": {"old.json": pin}}))
            p = {"comparisons": {"inherited_sources": ["old.json"]}}
            with patch.object(
                Path, "read_text", side_effect=AssertionError("snapshot-only decoding")
            ):
                checked = verify.inherited_rows(root, p)
            self.assertEqual(len(checked), 119)
            self.assertTrue(all(row["source_sha256"] == pin for row in checked))
            (report / "manifest.json").write_text("{unreadable redundant manifest")
            self.assertEqual(checked, verify.inherited_rows(root, p, pins={"old.json": pin}))
            path.write_text(json.dumps({"rows": rows[:-1]}))
            with self.assertRaises(AssertionError):
                verify.inherited_rows(root, p, pins={"old.json": pin})

    def test_entire_temporary_tree_entrypoint_reads_all_declared_paths_and_writes_only_new_record(
        self,
    ):
        from contextlib import ExitStack

        import yaml

        args = FullPoolContracts().fixture()
        f, t, issued, fits, panel, states, p = args
        for corrupt in (False, True):
            with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
                root = Path(folder)
                out = root / "data/causal_pool"
                report = root / "reports/causal_pool"
                out.mkdir(parents=True)
                report.mkdir(parents=True)
                inputs = {}
                for key, value in (("features", f), ("targets", t), ("forecasts", issued)):
                    name = p["upstream"][key]
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    value.to_parquet(path)
                    inputs[name] = verify.digest(path)
                name = p["upstream"]["fits"]
                (root / name).write_text(json.dumps(fits))
                inputs[name] = verify.digest(root / name)
                (root / "causal_pool.yaml").write_text(yaml.safe_dump(p))
                ph = verify.digest(root / "causal_pool.yaml")
                (report / "manifest.json").write_text(
                    json.dumps(
                        {"protocol_sha256": ph, "code": {}, "inputs": inputs, "preserved": {}}
                    )
                )
                (report / "trial_ledger.jsonl").write_text("")
                proof = {"status": "UPSTREAM_VERIFIED_READ_ONLY", "prior_files_written": False}
                (out / "upstream_admission.json").write_text(json.dumps(proof))
                panel.to_parquet(out / "forecasts.parquet")
                altered = states.copy()
                if corrupt:
                    altered.loc[4, "cumulative_updates"] += 1
                altered.to_parquet(out / "states.parquet")
                (report / "metrics.json").write_text(
                    json.dumps(
                        {
                            "protocol_sha256": ph,
                            "evidence_class": p["evidence_class"],
                            "inherited_rows": [],
                        }
                    )
                )
                stack.enter_context(patch.object(verify, "validate_protocol"))
                stack.enter_context(patch.object(verify, "verify_manifest_coverage"))
                stack.enter_context(
                    patch.object(verify, "validate_upstream", return_value=proof)
                )
                stack.enter_context(
                    patch.object(verify, "verify_metrics", return_value={"synthetic": True})
                )
                stack.enter_context(
                    patch.object(
                        verify,
                        "verify_ledger",
                        return_value={"inherited": 119, "registered": 2, "evaluated": 2},
                    )
                )
                before = {name: (root / name).read_bytes() for name in inputs}
                if corrupt:
                    with self.assertRaises(AssertionError):
                        verify.verify_with_failure_guard(root)
                    self.assertEqual(
                        json.loads((report / "verification.json").read_text())["status"],
                        "FAILED",
                    )
                else:
                    result = verify.verify_with_failure_guard(root)
                    self.assertEqual(result["status"], "VERIFIED")
                    self.assertEqual(
                        result["forecast_reconstruction"]["application_states_verified"],
                        len(states),
                    )
                self.assertEqual(before, {name: (root / name).read_bytes() for name in inputs})


if __name__ == "__main__":
    unittest.main()
