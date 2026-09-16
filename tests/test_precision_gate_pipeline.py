"""Generated temporal membership contracts written before implementation."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.claims_release_models import MARKET
from src.precision_gate_pipeline import InsufficientDataError, build_panel


def synthetic_inputs():
    rng = np.random.default_rng(893)
    dates = pd.bdate_range("2009-09-01", "2011-03-18", name="date")
    x = np.column_stack((np.ones(len(dates)), rng.normal(size=(len(dates), 11))))
    features = pd.DataFrame(x, columns=MARKET, index=dates)
    y = np.exp(-5 + 0.1 * x[:, 1] + rng.normal(0, 0.25, len(dates)))
    y[-5:] = np.nan
    targets = pd.DataFrame(
        {"y": y, "target_end": pd.Series(dates).shift(-5).to_numpy()}, index=dates
    )
    config = {
        "issuance_start": "2010-01-04",
        "origin_start": "2010-07-01",
        "origin_end": "2011-03-18",
        "source_end": "2011-03-18",
        "development": ["2010-07-01", "2010-12-31"],
        "evaluation": ["2011-01-03", "2011-03-18"],
        "minimum_train": 120,
        "adaptive_half_life": 30,
        "gate_window": 70,
        "gate_minimum_train": 30,
    }
    return features, targets, config


class PrecisionPipelineTests(unittest.TestCase):
    def test_full_issued_membership_maturity_and_global_offsets(self):
        f, t, c = synthetic_inputs()
        got = build_panel(f, t, c)
        self.assertEqual(set(got), {"applications", "panel", "coverage", "schedules", "fits"})
        for fit in got["fits"]:
            position = f.index.get_loc(pd.Timestamp(fit["fit_origin"]))
            self.assertEqual(fit["training_cutoff"], f.index[position - 1].date().isoformat())
            expected = np.flatnonzero(
                np.isfinite(t.y) & (t.target_end <= f.index[position - 1])
            )
            self.assertEqual(
                fit["train_origins"], [d.date().isoformat() for d in f.index[expected]]
            )
            if fit["status"] == "expert_warmup":
                self.assertEqual(fit["application_origins"], [])
                continue
            previous = got["applications"]
            previous = previous[
                (previous.origin < pd.Timestamp(fit["fit_origin"]))
                & (previous.origin >= f.index[max(0, position - c["gate_window"])])
            ]
            expected_gate = [
                d.date().isoformat()
                for d in previous.origin
                if pd.notna(t.loc[d, "y"]) and t.loc[d, "target_end"] <= f.index[position - 1]
            ]
            self.assertEqual(fit["gate_train_origins"], expected_gate)
        apps = got["applications"]
        np.testing.assert_array_equal(apps.offset, f.index.get_indexer(apps.origin) % 5)
        self.assertTrue(
            set(got["panel"].model) == {"base", "adaptive", "constant", "contextual"}
        )
        self.assertTrue(got["panel"].groupby("origin").size().eq(4).all())
        self.assertEqual(len(got["coverage"]), int((f.index >= c["issuance_start"]).sum()))
        self.assertEqual(set(apps.tail(5).origin), set(f.index[-5:]))
        self.assertFalse(got["coverage"].tail(5).scored.any())

    def test_warmup_does_not_issue_or_refill_and_cold_start_is_baseline(self):
        f, t, c = synthetic_inputs()
        c["gate_minimum_train"] = 1000
        got = build_panel(f, t, c)
        self.assertTrue((got["schedules"].status == "expert_warmup").any())
        self.assertTrue((got["coverage"].status == "expert_warmup").any())
        a = got["applications"]
        self.assertTrue(a.gate_status.eq("cold_start").all())
        np.testing.assert_array_equal(a.pred_base, a.pred_constant)
        np.testing.assert_array_equal(a.pred_base, a.pred_contextual)
        self.assertTrue(len(got["panel"]) > 0)
        self.assertTrue(got["panel"].gate_status.eq("cold_start").all())

    def test_target_blind_month_start_and_previous_full_session(self):
        f, t, c = synthetic_inputs()
        f.loc["2010-08-02", "liv"] = np.nan
        t.loc["2010-08-03", "y"] = np.nan
        got = build_panel(f, t, c)
        schedule = got["schedules"].set_index("month").loc["2010-08"]
        self.assertEqual(schedule.fit_origin, pd.Timestamp("2010-08-03"))
        self.assertEqual(schedule.training_cutoff, pd.Timestamp("2010-08-02"))
        self.assertIn(pd.Timestamp("2010-08-03"), set(got["applications"].origin))

    def test_future_target_permutation_cannot_change_earlier_forecasts(self):
        f, t, c = synthetic_inputs()
        before = build_panel(f, t, c)
        changed = t.copy()
        boundary = pd.Timestamp("2010-10-01")
        mask = changed.index >= boundary
        finite = mask & changed.y.notna()
        changed.loc[finite, "y"] = changed.loc[finite, "y"].to_numpy()[::-1] * 3
        after = build_panel(f, changed, c)
        left = before["applications"].query("origin < @boundary").reset_index(drop=True)
        right = after["applications"].query("origin < @boundary").reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right)
        pd.testing.assert_series_equal(
            before["schedules"].fit_origin, after["schedules"].fit_origin
        )

    def test_issued_state_is_saved_and_frozen_gate_gets_only_issued_mature_pairs(self):
        from src.precision_gate import fit_gate

        f, t, c = synthetic_inputs()
        with patch("src.precision_gate_pipeline.fit_gate", wraps=fit_gate) as wrapped:
            got = build_panel(f, t, c)
        fitted = [r for r in got["fits"] if r["gate_status"] == "fitted"]
        self.assertEqual(wrapped.call_count, 2 * len(fitted))
        a = got["applications"].set_index("origin")
        for i, fit in enumerate(fitted):
            dates = pd.DatetimeIndex(fit["gate_train_origins"])
            for call in wrapped.call_args_list[2 * i : 2 * i + 2]:
                np.testing.assert_array_equal(call.args[0], a.loc[dates, "pred_base"])
                np.testing.assert_array_equal(call.args[1], a.loc[dates, "pred_adaptive"])
                np.testing.assert_array_equal(call.args[2], t.loc[dates, "y"])
                np.testing.assert_array_equal(call.args[3], a.loc[dates, "state"])
        np.testing.assert_array_equal(
            a.state, f.loc[a.index, "lrv_d"] - f.loc[a.index, "lrv_m"]
        )

    def test_scoring_phase_endpoint_and_common_cohort(self):
        f, t, c = synthetic_inputs()
        f.loc["2010-11-03", "lrv_d"] = np.nan
        got = build_panel(f, t, c)
        p = got["panel"]
        self.assertNotIn(pd.Timestamp("2010-11-03"), set(p.origin))
        self.assertTrue((p.loc[p.phase == "development", "target_end"] <= "2010-12-31").all())
        self.assertNotIn(pd.Timestamp("2010-12-31"), set(p.origin))
        self.assertIn(pd.Timestamp("2010-12-31"), set(got["applications"].origin))

    def test_insufficient_scoring_month_aborts_and_retains_diagnostics(self):
        f, t, c = synthetic_inputs()
        c["minimum_train"] = 1000
        with self.assertRaises(InsufficientDataError) as raised:
            build_panel(f, t, c)
        error = raised.exception
        self.assertIn("INSUFFICIENT_DATA", str(error))
        self.assertTrue(error.produced["panel"].empty)
        self.assertIn("insufficient_training", set(error.schedules.status))
        self.assertEqual(len(error.coverage), int((f.index >= c["issuance_start"]).sum()))

    def test_rank_failure_does_not_become_warmup_or_fallback(self):
        f, t, c = synthetic_inputs()
        f["liv"] = f["lvix"]
        with self.assertRaisesRegex(ValueError, "rank"):
            build_panel(f, t, c)

    def test_later_gate_failure_retains_earlier_issued_records_and_stage(self):
        from src import precision_gate_pipeline as pipeline
        from src.precision_gate import fit_gate

        f, t, c = synthetic_inputs()
        expected = build_panel(f, t, c)
        calls = []

        def failing_gate(*args, **kwargs):
            calls.append(1)
            if len(calls) == 3:
                raise ValueError("invented gate optimizer failure")
            return fit_gate(*args, **kwargs)

        with (
            patch.object(pipeline, "fit_gate", side_effect=failing_gate),
            self.assertRaises(pipeline.PipelineExecutionError) as caught,
        ):
            build_panel(f, t, c)
        error = caught.exception
        self.assertIn("invented gate optimizer failure", str(error))
        self.assertEqual(error.stage, "constant_gate_fit")
        self.assertEqual(error.month, error.produced["fits"][-1]["month"])
        partial = error.produced["applications"]
        self.assertGreater(len(partial), 0)
        pd.testing.assert_frame_equal(partial, expected["applications"].iloc[: len(partial)])
        self.assertTrue(error.produced["panel"].empty)
        self.assertEqual(error.produced["fits"][-1]["status"], "execution_failed")
        self.assertIsNotNone(error.produced["fits"][-1]["expert_audits"])
        self.assertEqual(error.produced["fits"][-1]["application_origins"], [])
        self.assertEqual(calls, [1, 1, 1])

    def test_full_calendar_dates_and_target_schema_validated(self):
        f, t, c = synthetic_inputs()
        for kind in ("endpoint", "unaligned", "infinity", "boolean", "ceiling"):
            x, y, config = f.copy(), t.copy(), copy.deepcopy(c)
            if kind == "endpoint":
                y.iloc[3, 1] = f.index[7]
            if kind == "unaligned":
                y = y.iloc[1:]
            if kind == "infinity":
                x.iloc[3, 2] = np.inf
            if kind == "boolean":
                x["liv"] = True
            if kind == "ceiling":
                config["source_end"] = "2025-10-21"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                build_panel(x, y, config)

    def test_no_complete_month_and_input_immutability(self):
        f, t, c = synthetic_inputs()
        f.loc["2010-09", "liv"] = np.nan
        before, targets = f.copy(deep=True), t.copy(deep=True)
        got = build_panel(f, t, c)
        self.assertEqual(
            got["schedules"].set_index("month").loc["2010-09", "status"], "no_complete_origin"
        )
        pd.testing.assert_frame_equal(f, before)
        pd.testing.assert_frame_equal(t, targets)


if __name__ == "__main__":
    unittest.main()
