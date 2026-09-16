"""Prewritten, generated-only contracts for mature event-arrival adjacency."""

import itertools
import json
import unittest
from collections import defaultdict
from fractions import Fraction

import numpy as np
import pandas as pd

from src import event_cluster_features as cf


def target_fixture(n=85):
    """A generated irregular full reference calendar, never a source-data read."""
    dates = pd.bdate_range("2014-01-02", periods=n + 3).delete([5, 18, 31])
    dates = dates.rename("date")
    labels = np.array([float((i * 7 + i // 3) % 11 < 4) for i in range(n)])
    labels[-1] = np.nan
    next_dates = pd.Series(dates, index=dates).shift(-1)
    return pd.DataFrame(
        {"y": labels, "target_end": next_dates, "available_date": next_dates},
        index=dates,
    )


def rational_reference(events):
    """Independent exact rational oracle, including integer audit values."""
    e = tuple(int(value) for value in events)
    n = sum(e)
    last = e[-1]
    pairs = sum(a == b == 1 for a, b in zip(e, e[1:]))
    expected = (n - last) * (n - 1)
    recent = sum((2 * i - 23) * value for i, value in enumerate(e, start=1))
    excess = 21 * pairs - expected
    return {
        "event_fraction22": float(Fraction(n, 22)),
        "event_fraction22_sq": float(Fraction(n * n, 484)),
        "last_event": float(last),
        "expected_adjacency22": float(Fraction(expected, 441)),
        "linear_recency22": float(Fraction(recent, 462)),
        "adjacency_fraction22": float(Fraction(pairs, 21)),
        "excess_adjacency22": float(Fraction(excess, 441)),
        "event_count22": n,
        "adjacent_pairs22": pairs,
        "expected_adjacency_numerator": expected,
        "linear_recency_numerator": recent,
        "excess_adjacency_numerator": excess,
    }


class ClusterSummary(unittest.TestCase):
    def test_fitted_adjacency_is_constant_when_only_reference_changes(self):
        left, right = np.zeros(22), np.zeros(22)
        left[np.array([2, 5]) - 1] = 1
        right[np.array([1, 4, 22]) - 1] = 1
        a, b = cf.cluster_summary(left), cf.cluster_summary(right)
        self.assertEqual(a["adjacent_pairs22"], 0)
        self.assertEqual(b["adjacent_pairs22"], 0)
        self.assertEqual(a[cf.MEMORY], 0.0)
        self.assertEqual(b[cf.MEMORY], 0.0)
        self.assertNotEqual(a["event_count22"], b["event_count22"])
        self.assertNotEqual(a["last_event"], b["last_event"])
        self.assertNotEqual(a["expected_adjacency22"], b["expected_adjacency22"])
        self.assertNotEqual(a["linear_recency22"], b["linear_recency22"])
        self.assertNotEqual(a["excess_adjacency22"], b["excess_adjacency22"])

    def test_fixed_exports_and_order(self):
        self.assertEqual(cf.L, 22)
        self.assertEqual(
            cf.NUISANCE,
            (
                "event_fraction22",
                "event_fraction22_sq",
                "last_event",
                "expected_adjacency22",
                "linear_recency22",
            ),
        )
        self.assertEqual(cf.MEMORY, "adjacency_fraction22")
        self.assertEqual(cf.DIAGNOSTIC, "excess_adjacency22")
        self.assertEqual(cf.FEATURES, cf.NUISANCE + (cf.MEMORY,))
        self.assertNotIn(cf.DIAGNOSTIC, cf.FEATURES)
        self.assertEqual(
            cf.AUDIT_FIELDS,
            (
                "event_count22",
                "adjacent_pairs22",
                "expected_adjacency_numerator",
                "linear_recency_numerator",
                "excess_adjacency_numerator",
            ),
        )
        self.assertEqual(cf.SUMMARY_COLUMNS, cf.FEATURES + (cf.DIAGNOSTIC,) + cf.AUDIT_FIELDS)

    def test_generated_vectors_equal_exact_fraction_oracle(self):
        rng = np.random.default_rng(20260927)
        for events in rng.integers(0, 2, size=(200, 22)):
            result = cf.cluster_summary(events)
            self.assertEqual(tuple(result), cf.SUMMARY_COLUMNS)
            self.assertEqual(result, rational_reference(events))
            for key in cf.AUDIT_FIELDS:
                self.assertIs(type(result[key]), int)
            json.dumps(result, allow_nan=False)

    def test_all_zero_all_one_and_signed_zero_have_constant_raw_adjacency(self):
        for value in (0.0, -0.0, 1.0):
            result = cf.cluster_summary(np.full(22, value))
            self.assertEqual(result[cf.MEMORY], abs(value))
            self.assertEqual(result[cf.DIAGNOSTIC], 0.0)
            self.assertFalse(np.signbit(result[cf.DIAGNOSTIC]))
            self.assertEqual(result["linear_recency22"], 0.0)
            self.assertEqual(result["event_fraction22"], abs(value))

    def test_same_count_endpoint_and_linear_recency_different_adjacency(self):
        a, b = np.zeros(22), np.zeros(22)
        a[np.array([2, 3, 8]) - 1] = 1
        b[np.array([1, 5, 7]) - 1] = 1
        left, right = cf.cluster_summary(a), cf.cluster_summary(b)
        for key in cf.NUISANCE:
            self.assertEqual(left[key], right[key], key)
        self.assertEqual(left["adjacent_pairs22"], 1)
        self.assertEqual(right["adjacent_pairs22"], 0)
        self.assertEqual(left[cf.MEMORY], float(Fraction(1, 21)))
        self.assertEqual(right[cf.MEMORY], 0.0)
        self.assertEqual(left[cf.DIAGNOSTIC], float(Fraction(5, 147)))
        self.assertEqual(right[cf.DIAGNOSTIC], float(Fraction(-2, 147)))

    def test_exhaustive_short_sequence_conditional_expectation_identity(self):
        # Independent design proof: enumerate every ordering, not a simulation.
        for length in range(2, 9):
            groups = defaultdict(list)
            for sequence in itertools.product((0, 1), repeat=length):
                pairs = sum(a and b for a, b in zip(sequence, sequence[1:]))
                groups[(sum(sequence), sequence[-1])].append(pairs)
            for (count, last), values in groups.items():
                enumerated = Fraction(sum(values), len(values))
                self.assertEqual(
                    enumerated, Fraction((count - last) * (count - 1), length - 1)
                )

    def test_fixed22_exhaustive_sparse_permutations_center_diagnostic(self):
        # Checks the actual fixed22 implementation against exact summed audits.
        for last in (0, 1):
            for preceding_count in range(4):
                total = Fraction(0)
                for positions in itertools.combinations(range(21), preceding_count):
                    sequence = np.zeros(22, dtype=int)
                    sequence[list(positions)] = 1
                    sequence[-1] = last
                    result = cf.cluster_summary(sequence)
                    total += Fraction(result["excess_adjacency_numerator"], 441)
                self.assertEqual(total, 0)

    def test_last_event_reference_and_recency_direction(self):
        early, late = np.zeros(22), np.zeros(22)
        early[0], late[-1] = 1, 1
        a, b = cf.cluster_summary(early), cf.cluster_summary(late)
        self.assertEqual(a["linear_recency22"], -1 / 22)
        self.assertEqual(b["linear_recency22"], 1 / 22)
        self.assertEqual(a["last_event"], 0)
        self.assertEqual(b["last_event"], 1)
        self.assertEqual(a[cf.MEMORY], 0)
        self.assertEqual(b[cf.MEMORY], 0)

    def test_vector_shape_and_strict_real_numeric_domains(self):
        invalid = [
            [],
            np.zeros(21),
            np.zeros(23),
            np.zeros((22, 1)),
            np.zeros(22, dtype=bool),
            np.zeros(22, dtype=complex),
            np.zeros(22, dtype=object),
            ["0"] * 22,
            [True] + [0.0] * 21,
        ]
        for events in invalid:
            with (
                self.subTest(dtype=str(np.asarray(events).dtype)),
                self.assertRaises(ValueError),
            ):
                cf.cluster_summary(events)

    def test_nonbinary_missing_and_infinite_observations_reject(self):
        for value in (0.5, -1.0, 2.0, np.nan, np.inf, -np.inf):
            events = np.zeros(22)
            events[4] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                cf.cluster_summary(events)

    def test_input_is_not_mutated(self):
        events = np.arange(22) % 2
        old = events.copy()
        cf.cluster_summary(events)
        np.testing.assert_array_equal(events, old)


class MatureArrivalMemory(unittest.TestCase):
    def test_full_ordered_schema_index_and_targets_preserved(self):
        targets = target_fixture()
        before = targets.copy(deep=True)
        result = cf.build_memory(targets)
        self.assertEqual(
            cf.DATE_COLUMNS,
            (
                "feature_cutoff_date",
                "window_first_available",
                "window_last_available",
                "window_first_origin",
                "window_last_origin",
            ),
        )
        self.assertEqual(tuple(result), cf.SUMMARY_COLUMNS + cf.DATE_COLUMNS)
        pd.testing.assert_index_equal(result.index, targets.index)
        pd.testing.assert_frame_equal(targets, before)

    def test_exact23row_warmup_and_availability_join_on_irregular_calendar(self):
        targets = target_fixture()
        result = cf.build_memory(targets)
        self.assertTrue(result.loc[:, cf.SUMMARY_COLUMNS].iloc[:23].isna().all().all())
        self.assertTrue(result.loc[:, cf.DATE_COLUMNS[1:]].iloc[:23].isna().all().all())
        self.assertTrue(pd.isna(result.iloc[0].feature_cutoff_date))
        for i in (23, 24, 40, len(targets) - 1):
            row = result.iloc[i]
            cutoff = targets.index[i - 1]
            arrivals = targets.index[i - 22 : i]
            history = targets.loc[targets.available_date.isin(arrivals), "y"]
            self.assertEqual(len(history), 22)
            self.assertEqual(
                row.loc[list(cf.SUMMARY_COLUMNS)].to_dict(), rational_reference(history)
            )
            self.assertEqual(row.feature_cutoff_date, cutoff)
            self.assertEqual(row.window_first_available, targets.index[i - 22])
            self.assertEqual(row.window_last_available, cutoff)
            self.assertEqual(row.window_first_origin, targets.index[i - 23])
            self.assertEqual(row.window_last_origin, targets.index[i - 2])

    def test_arrival_maturity_is_two_origin_positions_not_one(self):
        targets = target_fixture()
        before = cf.build_memory(targets)
        position = 45
        targets.iloc[position, targets.columns.get_loc("y")] = 1 - targets.iloc[position].y
        after = cf.build_memory(targets)
        pd.testing.assert_frame_equal(before.iloc[: position + 2], after.iloc[: position + 2])
        self.assertNotEqual(
            before.iloc[position + 2].event_count22, after.iloc[position + 2].event_count22
        )
        pd.testing.assert_frame_equal(
            before.iloc[position + 24 :], after.iloc[position + 24 :]
        )

    def test_strict_missing_label_window_is_not_compressed_or_zero_filled(self):
        targets = target_fixture()
        before = cf.build_memory(targets)
        position = 32
        targets.iloc[position, targets.columns.get_loc("y")] = np.nan
        after = cf.build_memory(targets)
        self.assertTrue(
            after.loc[:, cf.SUMMARY_COLUMNS]
            .iloc[position + 2 : position + 24]
            .isna()
            .all()
            .all()
        )
        pd.testing.assert_frame_equal(before.iloc[: position + 2], after.iloc[: position + 2])
        pd.testing.assert_frame_equal(
            before.iloc[position + 24 :], after.iloc[position + 24 :]
        )
        pd.testing.assert_frame_equal(
            before.loc[:, cf.DATE_COLUMNS], after.loc[:, cf.DATE_COLUMNS]
        )

    def test_query_and_future_label_mutation_cannot_change_earlier_features(self):
        targets = target_fixture()
        before = cf.build_memory(targets)
        changed = targets.copy()
        changed.loc[changed.index[43:-1], "y"] = 1 - changed.loc[changed.index[43:-1], "y"]
        after = cf.build_memory(changed)
        pd.testing.assert_frame_equal(before.iloc[:45], after.iloc[:45])

    def test_prefix_invariance_including_its_unavailable_terminal_label(self):
        targets = target_fixture()
        before = cf.build_memory(targets)
        prefix = targets.iloc[:55].copy()
        prefix.iloc[-1, prefix.columns.get_loc("y")] = np.nan
        prefix.iloc[-1, prefix.columns.get_loc("target_end")] = pd.NaT
        prefix.iloc[-1, prefix.columns.get_loc("available_date")] = pd.NaT
        pd.testing.assert_frame_equal(before.iloc[:55], cf.build_memory(prefix))

    def test_unscored_and_feature_incomplete_origins_still_supply_labels(self):
        # No score, phase or feature mask is an input to this full-target builder.
        targets = target_fixture()
        targets.loc[targets.index[:35], "y"] = 0
        targets.loc[targets.index[12], "y"] = 1
        result = cf.build_memory(targets)
        self.assertEqual(result.iloc[23].event_count22, 1)
        self.assertEqual(result.iloc[35].event_count22, 1)
        self.assertEqual(result.iloc[36].event_count22, 0)

    def test_short_calendar_returns_unknown_without_inventing_history(self):
        targets = target_fixture().iloc[:12].copy()
        targets.iloc[-1] = [np.nan, pd.NaT, pd.NaT]
        result = cf.build_memory(targets)
        self.assertTrue(result.loc[:, cf.SUMMARY_COLUMNS].isna().all().all())
        self.assertEqual(result.iloc[-1].feature_cutoff_date, targets.index[-2])


class TargetAdmission(unittest.TestCase):
    def test_exact_target_schema_required(self):
        targets = target_fixture()
        cases = [
            targets.drop(columns="target_end"),
            targets.assign(extra=0),
            targets.loc[:, ["target_end", "y", "available_date"]],
            pd.concat([targets, targets[["y"]]], axis=1),
            targets.to_numpy(),
        ]
        for frame in cases:
            with self.assertRaises(ValueError):
                cf.build_memory(frame)

    def test_index_must_be_nonempty_unique_ascending_normalized_naive_dates(self):
        targets = target_fixture()
        cases = [
            targets.iloc[:0],
            targets.iloc[::-1],
            pd.concat([targets, targets.iloc[[-1]]]),
        ]
        for index in (
            pd.RangeIndex(len(targets)),
            targets.index.tz_localize("UTC"),
            targets.index + pd.Timedelta(hours=1),
            pd.DatetimeIndex([pd.NaT, *targets.index[1:]]),
        ):
            frame = targets.copy()
            frame.index = index
            cases.append(frame)
        for frame in cases:
            with self.assertRaises(ValueError):
                cf.build_memory(frame)

    def test_literal_source_fence_rejects_even_unknown_later_dates(self):
        targets = target_fixture()
        dates = pd.bdate_range("2025-10-01", periods=len(targets), name="date")
        targets.index = dates
        following = pd.Series(dates, index=dates).shift(-1)
        targets["target_end"] = following
        targets["available_date"] = following
        targets["y"] = np.nan
        with self.assertRaises(ValueError):
            cf.build_memory(targets)

    def test_wrong_or_missing_maturity_rejects_even_for_unknown_labels(self):
        for column in ("target_end", "available_date"):
            for value in (pd.NaT, pd.Timestamp("2014-02-04")):
                targets = target_fixture()
                targets.iloc[10, targets.columns.get_loc("y")] = np.nan
                targets.iloc[10, targets.columns.get_loc(column)] = value
                with self.assertRaises(ValueError):
                    cf.build_memory(targets)

    def test_date_strings_timezones_and_intraday_timestamps_reject(self):
        for column in ("target_end", "available_date"):
            for mode in ("strings", "timezone", "intraday"):
                targets = target_fixture()
                if mode == "strings":
                    targets[column] = targets[column].astype(str)
                elif mode == "timezone":
                    targets[column] = targets[column].dt.tz_localize("UTC")
                else:
                    targets[column] += pd.Timedelta(minutes=1)
                with self.assertRaises(ValueError):
                    cf.build_memory(targets)

    def test_known_terminal_label_or_terminal_date_rejects(self):
        for column, value in (
            ("y", 0.0),
            ("target_end", pd.Timestamp("2015-01-02")),
            ("available_date", pd.Timestamp("2015-01-02")),
        ):
            targets = target_fixture()
            targets.iloc[-1, targets.columns.get_loc(column)] = value
            with self.assertRaises(ValueError):
                cf.build_memory(targets)

    def test_binary_label_domain_rejects_before_any_window_mask(self):
        for value in (0.5, -1.0, np.inf, -np.inf):
            targets = target_fixture()
            targets.iloc[0, targets.columns.get_loc("y")] = value
            with self.assertRaises(ValueError):
                cf.build_memory(targets)
        for dtype in (str, object, complex, bool):
            targets = target_fixture()
            targets["y"] = targets.y.astype(dtype)
            with self.assertRaises(ValueError):
                cf.build_memory(targets)


if __name__ == "__main__":
    unittest.main()
