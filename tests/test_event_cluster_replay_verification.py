"""Prewritten lossless state-date transport tests; all observations generated."""

import copy
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import event_cluster_features as feature
from src import event_cluster_models as models
from src import event_cluster_pipeline as producer
from src import event_cluster_replay_verification as verification
from src import event_cluster_verification as frozen
from src import range_alert_models as original
from tests.test_event_cluster_pipeline import fixture


def roundtrip(frame):
    payload = io.BytesIO()
    frame.to_parquet(payload)
    return pd.read_parquet(io.BytesIO(payload.getvalue()))


def date_units(frame, unit):
    result = frame.copy(deep=True)
    if isinstance(result.index, pd.DatetimeIndex):
        result.index = result.index.as_unit(unit, round_ok=False)
    for column in result.columns:
        if isinstance(result[column].dtype, np.dtype) and result[column].dtype.kind == "M":
            result[column] = result[column].dt.as_unit(unit, round_ok=False)
    return result


def tiny_states(unit="ms"):
    dates = pd.date_range("2001-02-01", periods=3).as_unit(unit)
    out = pd.DataFrame(dict.fromkeys(verification.STATE_DATE_COLUMNS, dates))
    out["probability"] = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    out["scored"] = [True, False, True]
    out["count"] = np.array([1, 2, 3], dtype=np.int64)
    return out


class DeclaredDateContracts(unittest.TestCase):
    def test_exact_seven_columns_and_no_index_or_other_frame_waiver(self):
        self.assertEqual(
            verification.STATE_DATE_COLUMNS,
            (
                "origin",
                "feature_cutoff_date",
                "source_fit_origin",
                "window_first_available",
                "window_last_available",
                "window_first_origin",
                "window_last_origin",
            ),
        )
        a, b = tiny_states("us"), tiny_states("ms")
        before = copy.deepcopy((a, b))
        proof = verification.compare_state_dates(a, b, "synthetic states")
        self.assertEqual(proof["comparison_unit"], "ns")
        self.assertEqual(proof["columns"], list(verification.STATE_DATE_COLUMNS))
        self.assertEqual(proof["rows"], 3)
        self.assertTrue(
            all(
                x["actual_unit"] == "us" and x["expected_unit"] == "ms"
                for x in proof["fields"]
            )
        )
        pd.testing.assert_frame_equal(a, before[0], check_exact=True)
        pd.testing.assert_frame_equal(b, before[1], check_exact=True)
        with self.assertRaises(AssertionError):
            verification.exact_frame(a, b, "non-state frame still exact")

    def test_all_ms_us_ns_pairs_and_exact_unknown_masks(self):
        for first in ("ms", "us", "ns"):
            for second in ("ms", "us", "ns"):
                a, b = tiny_states(first), tiny_states(second)
                a.loc[0, "window_first_origin"] = pd.NaT
                b.loc[0, "window_first_origin"] = pd.NaT
                proof = verification.compare_state_dates(a, b, "equal dates")
                row = next(x for x in proof["fields"] if x["column"] == "window_first_origin")
                self.assertEqual(row["nat_rows"], 1)
                self.assertTrue(row["lossless"])

    def test_changed_date_subday_timezone_object_unit_and_unknown_masks_reject(self):
        for kind in ("day", "subday", "timezone", "object", "string", "seconds", "nat"):
            a, b = tiny_states("ms"), tiny_states("us")
            column = "feature_cutoff_date"
            if kind == "day":
                a.loc[1, column] += pd.Timedelta(days=1)
            elif kind == "subday":
                a[column] = a[column].dt.as_unit("ns")
                a.loc[1, column] += pd.Timedelta(nanoseconds=1)
            elif kind == "timezone":
                a[column] = a[column].dt.tz_localize("UTC")
            elif kind == "object":
                a[column] = a[column].astype(object)
            elif kind == "string":
                a[column] = a[column].astype(str)
            elif kind == "seconds":
                a[column] = a[column].dt.as_unit("s")
            else:
                a.loc[1, column] = pd.NaT
            with self.subTest(kind=kind), self.assertRaises((ValueError, AssertionError)):
                verification.compare_state_dates(a, b, "invalid dates")

    def test_overflow_is_not_wrapped_even_for_equal_midnight_instants(self):
        a = tiny_states("ms")
        for column in verification.STATE_DATE_COLUMNS:
            a[column] = pd.Series(np.array(["2500-01-01"] * 3, dtype="datetime64[ms]"))
        with self.assertRaises((ValueError, AssertionError, OverflowError)):
            verification.compare_state_dates(a, a.copy(), "out-of-range conversion")

    def test_index_row_column_names_and_nondate_dtype_or_bits_stay_exact(self):
        for kind in (
            "index_name",
            "index_values",
            "row_order",
            "column_order",
            "column_name",
            "column_axis_name",
            "float32",
            "intfloat",
            "boolint",
            "numeric_value",
            "extra_date",
        ):
            a, b = tiny_states("us"), tiny_states("ms")
            if kind == "index_name":
                a.index.name = "changed"
            elif kind == "index_values":
                a.index = [1, 2, 3]
            elif kind == "row_order":
                a = a.iloc[::-1]
            elif kind == "column_order":
                a = a.loc[:, a.columns[::-1]]
            elif kind == "column_name":
                a = a.rename(columns={"probability": "new_probability"})
            elif kind == "column_axis_name":
                a.columns.name = "changed"
            elif kind == "float32":
                a["probability"] = a.probability.astype("float32")
            elif kind == "intfloat":
                a["count"] = a["count"].astype(float)
            elif kind == "boolint":
                a["scored"] = a.scored.astype(int)
            elif kind == "numeric_value":
                a.loc[0, "probability"] = np.nextafter(a.loc[0, "probability"], 1)
            else:
                a["undeclared_date"] = a.origin
                b["undeclared_date"] = b.origin
            with self.subTest(kind=kind), self.assertRaises((ValueError, AssertionError)):
                verification.compare_state_dates(a, b, "strict remainder")


class FullTransportContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {}
        for unit in ("ms", "us", "ns"):
            f, t, config = fixture()
            # Units are specified before either synthetic original or retained fit.
            f, t = roundtrip(date_units(f, unit)), roundtrip(date_units(t, unit))
            old_panel, old_fits, old_states = original.forecast_panel(f, t, config)
            old_panel, old_states = roundtrip(old_panel), roundtrip(old_states)
            old_fits = json.loads(json.dumps(old_fits, allow_nan=False))
            panel, fits, states, memory, support = producer.forecast_panel(
                f, t, old_panel, old_fits, old_states, config
            )
            outputs = (
                roundtrip(panel),
                json.loads(json.dumps(fits, allow_nan=False)),
                roundtrip(states),
                roundtrip(memory),
                json.loads(json.dumps(support, allow_nan=False)),
            )
            cls.cases[unit] = (f, t, old_panel, old_fits, old_states, config, outputs)
        cls.proofs = {unit: cls.call(case) for unit, case in cls.cases.items()}

    @staticmethod
    def call(case, outputs=None):
        f, t, old_panel, old_fits, old_states, config, current = case
        return verification.verify_pipeline(
            f,
            t,
            old_panel,
            old_fits,
            old_states,
            config,
            *(current if outputs is None else outputs),
        )

    def test_unmocked_full_transport_in_all_units_and_retained_counts(self):
        for unit, case in self.cases.items():
            f, _, _, old_fits, _, config, outputs = case
            panel, fits, states, memory, _ = outputs
            proof = self.proofs[unit]
            with self.subTest(unit=unit):
                self.assertEqual(str(f.index.dtype), f"datetime64[{unit}]")
                self.assertEqual(proof["forecasts_verified"], len(panel))
                self.assertEqual(proof["retained_candidate_scored_forecasts"], len(panel) // 2)
                self.assertEqual(proof["original_control_forecasts"], len(panel) // 2)
                self.assertEqual(proof["newly_generated_forecasts"], 0)
                self.assertEqual(proof["new_producer_fits"], 0)
                self.assertEqual(proof["retained_monthly_schedules"], len(fits))
                self.assertEqual(proof["independent_stage_fits_verified"], 2 * len(fits))
                self.assertEqual(proof["original_monthly_fits_replayed"], len(old_fits))
                self.assertEqual(proof["application_states_verified"], len(states))
                self.assertEqual(proof["memory_rows_verified"], len(memory))
                self.assertEqual(len(proof["monthly_stage_audits"]), len(fits))
                self.assertFalse(states.iloc[-1].scored)
                self.assertEqual(states.iloc[-1].origin, pd.Timestamp(config["origin_end"]))
                self.assertNotIn(states.iloc[-1].origin, set(panel.origin))
                self.assertGreater(
                    proof["common_application_origins"], proof["common_scored_origins"]
                )
                json.dumps(proof, allow_nan=False)

    def test_real_source_unit_mismatch_falsifies_frozen_helper_before_any_solver(self):
        for unit in ("ms", "ns"):
            case = self.cases[unit]
            with self.subTest(unit=unit), patch.object(frozen.stage, "verify_stages") as fit:
                with self.assertRaisesRegex(AssertionError, "every saved application state"):
                    frozen.verify_pipeline(*case[:6], *case[6])
                fit.assert_not_called()

    def test_replay_calls_no_original_or_retained_producer_calculators_and_preserves_inputs(
        self,
    ):
        case = self.cases["ms"]
        before = copy.deepcopy(case)
        with (
            patch.object(
                original,
                "fit_predict",
                side_effect=AssertionError("original optimizer forbidden"),
            ),
            patch.object(
                producer,
                "forecast_panel",
                side_effect=AssertionError("retained producer forbidden"),
            ),
            patch.object(
                models, "fit_stages", side_effect=AssertionError("producer stages forbidden")
            ),
            patch.object(
                feature,
                "build_memory",
                side_effect=AssertionError("producer memory forbidden"),
            ),
        ):
            self.call(case)
        for a, b in zip(before[:5], case[:5], strict=True):
            if isinstance(a, pd.DataFrame):
                pd.testing.assert_frame_equal(a, b, check_exact=True)
            else:
                self.assertEqual(a, b)
        for a, b in zip(before[6], case[6], strict=True):
            if isinstance(a, pd.DataFrame):
                pd.testing.assert_frame_equal(a, b, check_exact=True)
            else:
                self.assertEqual(a, b)

    def test_actual_serialized_state_date_and_nondatetime_tampering_rejects(self):
        case = self.cases["ms"]
        for kind in (
            "day",
            "subday",
            "timezone",
            "object",
            "nat",
            "nondate_dtype",
            "numeric",
            "row_order",
            "names",
        ):
            outputs = list(copy.deepcopy(case[6]))
            states = outputs[2]
            if kind == "day":
                states.loc[states.index[-1], "feature_cutoff_date"] += pd.Timedelta(days=1)
            elif kind == "subday":
                states["feature_cutoff_date"] = states.feature_cutoff_date.dt.as_unit("ns")
                states.loc[states.index[-1], "feature_cutoff_date"] += pd.Timedelta(
                    nanoseconds=1
                )
            elif kind == "timezone":
                states["feature_cutoff_date"] = states.feature_cutoff_date.dt.tz_localize(
                    "UTC"
                )
            elif kind == "object":
                states["feature_cutoff_date"] = states.feature_cutoff_date.astype(object)
            elif kind == "nat":
                states.loc[states.index[-1], "feature_cutoff_date"] = pd.NaT
            elif kind == "nondate_dtype":
                states["event_count22"] = states.event_count22.astype("int64")
            elif kind == "numeric":
                states.loc[states.index[-1], "cluster_probability"] += 0.01
            elif kind == "row_order":
                outputs[2] = states.iloc[::-1].reset_index(drop=True)
            else:
                states.index.name = "changed"
            with self.subTest(kind=kind), patch.object(frozen.stage, "verify_stages") as fit:
                with self.assertRaises((AssertionError, ValueError)):
                    self.call(case, tuple(outputs))
                fit.assert_not_called()

    def test_memory_and_index_do_not_get_the_new_state_exception(self):
        case = self.cases["ms"]
        for kind in ("memory_unit", "index_unit", "unknown_numeric"):
            outputs = list(copy.deepcopy(case[6]))
            if kind == "memory_unit":
                outputs[3]["feature_cutoff_date"] = outputs[3].feature_cutoff_date.dt.as_unit(
                    "ns"
                )
            elif kind == "index_unit":
                outputs[3].index = outputs[3].index.as_unit("ns")
            else:
                outputs[3].iloc[0, 0] = 0.0
            with self.subTest(kind=kind), patch.object(frozen.stage, "verify_stages") as fit:
                with self.assertRaises((AssertionError, ValueError)):
                    self.call(case, tuple(outputs))
                fit.assert_not_called()

    def test_saved_coefficients_and_coherent_unscored_forecast_still_need_independent_solves(
        self,
    ):
        case = self.cases["ms"]
        for kind in ("coefficient", "coherent_unscored"):
            outputs = list(copy.deepcopy(case[6]))
            if kind == "coefficient":
                outputs[1][0]["model_audit"]["cluster"]["coefficient"] += 0.1
            else:
                probability = outputs[1][-1]["application_probabilities"]["cluster"][-1] + 0.01
                outputs[1][-1]["application_probabilities"]["cluster"][-1] = probability
                outputs[2].loc[outputs[2].index[-1], "cluster_probability"] = probability
            with self.subTest(kind=kind), self.assertRaises((AssertionError, ValueError)):
                self.call(case, tuple(outputs))


if __name__ == "__main__":
    unittest.main()
