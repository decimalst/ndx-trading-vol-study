"""Prewritten source authentication and saved-selection adapter contracts."""

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.spread_geometry_hindsight_run import authenticate, extract_inputs
from tests.test_spread_geometry_hindsight import EXPECTED_POLICIES, synthetic_inputs


def source_fixtures():
    returns, selections = synthetic_inputs()
    panel = returns.assign(unused_source_score=np.arange(len(returns), dtype=float))
    anchor = selections.copy()
    anchor["model"] = anchor.policy.map(lambda p: p if p == "always_sell" else p.removeprefix("fixed_2pct_"))
    anchor = anchor.drop(columns="policy").merge(returns[["origin", "target_end", "phase"]], on="origin", how="left", validate="many_to_one")
    anchor["distance"] = .02
    anchor["width"] = .01
    anchor["unused_probability"] = .07
    frames = []
    for distance in (.01, .02, .03):
        frame = anchor.copy()
        frame["distance"] = distance
        if distance != .02:
            # Distinct prior selection decisions cannot bleed into the anchor.
            frame.loc[frame.model != "always_sell", "sell"] = ~frame.loc[frame.model != "always_sell", "sell"]
        frames.append(frame)
    return returns, selections, panel, pd.concat(frames, ignore_index=True)


class SpreadGeometryHindsightRunTests(unittest.TestCase):
    def test_authentication_checks_exact_bytes_without_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "nested").mkdir()
            payloads = {"nested/forecast.bin": b"\x00\xff\x01\n", "source with spaces.json": b'{"value": 1}\n'}
            for name, data in payloads.items():
                (root / name).write_bytes(data)
            pins = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
            before = pins.copy()
            self.assertIsNone(authenticate(root, pins))
            self.assertEqual(pins, before)
            for name, data in payloads.items():
                self.assertEqual((root / name).read_bytes(), data)
            # Semantically identical JSON with distinct bytes must fail.
            (root / "source with spaces.json").write_bytes(b'{"value":1}\n')
            with self.assertRaises(ValueError):
                authenticate(root, pins)

    def test_authentication_missing_file_and_mismatched_digest_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = b"saved source"
            (root / "source").write_bytes(data)
            with self.assertRaises(ValueError):
                authenticate(root, {"missing": hashlib.sha256(data).hexdigest()})
            with self.assertRaises(ValueError):
                authenticate(root, {"source": "0" * 64})
            (root / "directory").mkdir()
            with self.assertRaises(ValueError):
                authenticate(root, {"directory": hashlib.sha256(data).hexdigest()})

    def test_exact_anchor_extraction_preserves_row_order_models_and_masks(self):
        returns, selections, panel, positions = source_fixtures()
        panel_before = panel.copy(deep=True)
        positions_before = positions.copy(deep=True)
        actual_returns, actual_selections = extract_inputs(panel, positions)
        self.assertEqual(list(actual_returns.columns), ["origin", "target_end", "phase", "y_qqq", "y_spx"])
        self.assertEqual(list(actual_selections.columns), ["origin", "structure", "policy", "sell"])
        assert_frame_equal(actual_returns.reset_index(drop=True), returns.reset_index(drop=True))
        keys = ["origin", "structure", "policy"]
        assert_frame_equal(actual_selections.sort_values(keys).reset_index(drop=True), selections.sort_values(keys).reset_index(drop=True))
        self.assertEqual(set(actual_selections.policy), set(EXPECTED_POLICIES))
        self.assertEqual(len(actual_selections), len(returns) * 21)
        self.assertTrue(actual_selections.loc[actual_selections.policy == "always_sell", "sell"].all())
        assert_frame_equal(panel, panel_before)
        assert_frame_equal(positions, positions_before)
        # Returned frames must be independent copies rather than mutable views.
        actual_returns.loc[actual_returns.index[0], "y_qqq"] = 99.0
        actual_selections.loc[actual_selections.index[0], "sell"] = False
        assert_frame_equal(panel, panel_before)
        assert_frame_equal(positions, positions_before)

    def test_nonanchor_rows_are_ignored_and_near_anchor_is_not_silently_rounded(self):
        _, _, panel, positions = source_fixtures()
        base_returns, base_selections = extract_inputs(panel, positions)
        other = positions.copy()
        other.loc[other.distance != .02, "sell"] = False
        r, s = extract_inputs(panel, other)
        assert_frame_equal(r, base_returns)
        assert_frame_equal(s, base_selections)
        anchor = positions.loc[positions.distance == .02].copy()
        anchor.loc[anchor.index[0], "distance"] = np.nextafter(.02, .03)
        with self.assertRaises(ValueError):
            extract_inputs(panel, anchor)
        anchor = positions.loc[positions.distance == .02].copy()
        anchor.loc[anchor.index[0], "width"] = np.nextafter(.01, .02)
        with self.assertRaises(ValueError):
            extract_inputs(panel, anchor)

    def test_complete_anchor_grid_rejects_duplicates_aliases_and_missing_cases(self):
        _, _, panel, positions = source_fixtures()
        anchor = positions.loc[positions.distance == .02].copy().reset_index(drop=True)
        cases = [anchor.iloc[1:], pd.concat([anchor, anchor.iloc[[0]]]), anchor.drop(columns="model"), anchor.drop(columns="width"), anchor.drop(columns="distance"), anchor.iloc[:0]]
        for column, value in [("model", "fixed_2pct_orig_t8"), ("model", "shape_student"), ("structure", "straddle"), ("sell", 1), ("sell", "True"), ("sell", None)]:
            bad = anchor.copy()
            bad[column] = bad[column].astype(object)
            bad.loc[0, column] = value
            cases.append(bad)
        bad = anchor.copy()
        bad.loc[bad.model == "always_sell", "sell"] = False
        cases.append(bad)
        for index, bad in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                extract_inputs(panel, bad)

    def test_every_anchor_clock_and_phase_must_match_panel_without_join_expansion(self):
        _, _, panel, positions = source_fixtures()
        anchor = positions.loc[positions.distance == .02].copy().reset_index(drop=True)
        cases = []
        for column, value in [("target_end", pd.Timestamp("2019-12-31")), ("origin", pd.Timestamp("2019-12-20")), ("phase", "evaluation")]:
            bad = anchor.copy()
            bad.loc[0, column] = value
            cases.append(bad)
        for column in ("origin", "target_end"):
            bad = anchor.copy()
            bad[column] = bad[column].dt.strftime("%Y-%m-%d")
            cases.append(bad)
            bad = anchor.copy()
            bad.loc[0, column] += pd.Timedelta(hours=1)
            cases.append(bad)
        for index, bad in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                extract_inputs(panel, bad)
        for bad_panel in [panel.drop(columns="y_spx"), pd.concat([panel, panel.iloc[[0]]]), panel.iloc[:-1]]:
            with self.assertRaises(ValueError):
                extract_inputs(bad_panel, anchor)


if __name__ == "__main__":
    unittest.main()
