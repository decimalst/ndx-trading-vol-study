"""Prewritten generated join, authentication and future-poison contracts."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.treasury_dealer_inputs import build_inputs, read_inputs
from src.treasury_dealer_models import ALL
from tests.test_claims_release_market import fixture
from tests.test_treasury_dealer_features import known


def generated():
    daily, cross, iv = fixture(340)
    dates = daily.index
    events = [
        known(f"event{i}", dates[150 + i * 8].date().isoformat(), dealer=10 + i)
        for i in range(18)
    ]
    return {"daily": daily, "cross": cross, "iv": iv}, events


class TreasuryDealerInputTests(unittest.TestCase):
    def test_join_retains_full_calendar_metadata_and_unscored_targets(self):
        sources, events = generated()
        result = build_inputs(sources, events)
        self.assertEqual(set(result), {"features", "targets", "event_audit"})
        f, y = result["features"], result["targets"]
        pd.testing.assert_index_equal(f.index, sources["daily"].index)
        pd.testing.assert_index_equal(y.index, f.index)
        self.assertEqual(set(f), set(ALL) | {"rv_total", "treasury_cutoff_date"})
        self.assertTrue(y.y.iloc[-5:].isna().all())
        self.assertTrue(y.target_end.iloc[-5:].isna().all())
        self.assertEqual(len(result["event_audit"]), len(events))
        for position in (200, 290, 334):
            expected = sum(f.rv_total.iloc[position + j] for j in range(1, 6)) / 5
            self.assertAlmostEqual(y.y.iloc[position], expected, places=14)
            self.assertEqual(y.target_end.iloc[position], f.index[position + 5])
        self.assertEqual(f.treasury_cutoff_date.iloc[-1], f.index[-2])

    def test_join_matches_hand_computed_dealer_pulse(self):
        sources, events = generated()
        f = build_inputs(sources, events)["features"]
        day = pd.Timestamp(events[12]["auction_date"])
        activation = f.index[f.index > day][0]
        expected_prior = sum((10 + i) / 100 for i in range(12)) / 12
        self.assertAlmostEqual(f.at[activation, "prior_share_mean_sum"], expected_prior)
        self.assertAlmostEqual(f.at[activation, "dealer_surprise"], 22 / 100 - expected_prior)
        self.assertEqual(f.at[activation, "auction_count"], 1)
        self.assertEqual(f.at[activation, "tenor_5"], 1)
        self.assertEqual(f.at[f.index[f.index > activation][0], "dealer_surprise"], 0)

    def test_future_source_and_event_changes_do_not_change_earlier_features(self):
        sources, events = generated()
        before = build_inputs(sources, events)
        changed_sources, changed_events = copy.deepcopy(sources), copy.deepcopy(events)
        cutoff = sources["daily"].index[280]
        for name in changed_sources:
            changed_sources[name].loc[changed_sources[name].index > cutoff] *= 1.1
        for event in changed_events:
            if pd.Timestamp(event["auction_date"]) > cutoff:
                event["primary_dealer_accepted"] = 75
        after = build_inputs(changed_sources, changed_events)
        pd.testing.assert_frame_equal(
            before["features"].loc[:cutoff], after["features"].loc[:cutoff]
        )
        # Forward targets near the poison date may change; mature labels may not.
        mature = before["targets"].target_end <= cutoff
        pd.testing.assert_frame_equal(
            before["targets"].loc[mature], after["targets"].loc[mature]
        )

    def test_sources_and_events_are_not_mutated(self):
        sources, events = generated()
        saved_sources, saved_events = copy.deepcopy(sources), copy.deepcopy(events)
        build_inputs(sources, events)
        for name in sources:
            pd.testing.assert_frame_equal(sources[name], saved_sources[name])
        self.assertEqual(events, saved_events)

    def test_missing_cross_session_is_never_filled_or_compressed(self):
        sources, events = generated()
        missing = sources["daily"].index[300]
        sources["cross"] = sources["cross"].drop(index=missing)
        f = build_inputs(sources, events)["features"]
        self.assertIn(missing, f.index)
        self.assertTrue(np.isnan(f.tlt_ret.iloc[301]))
        self.assertTrue(np.isnan(f.tlt_ret.iloc[302]))
        self.assertTrue(np.isfinite(f.tlt_ret.iloc[303]))
        self.assertTrue(np.isnan(f.xasset_stress.iloc[300]))

    def test_extra_source_key_is_rejected(self):
        sources, events = generated()
        with self.assertRaises(ValueError):
            build_inputs({**sources, "alternate": sources["iv"]}, events)

    def test_all_byte_hashes_precede_any_market_decode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = json.dumps({"events": [], "evidence": []}).encode()
            (root / "ledger.json").write_bytes(payload)
            (root / "market").write_bytes(b"generated-market-snapshot")
            market_pins = {"market": hashlib.sha256(b"generated-market-snapshot").hexdigest()}
            ledger_hash = hashlib.sha256(payload).hexdigest()
            sentinel = {"daily": "generated"}
            with patch(
                "src.treasury_dealer_inputs.read_market_sources", return_value=sentinel
            ) as decoder:
                sources, ledger = read_inputs(root, market_pins, "ledger.json", ledger_hash)
                self.assertEqual(sources, sentinel)
                self.assertEqual(ledger["events"], [])
                decoder.assert_called_once_with(root, market_pins)
            for broken in ("ledger.json", "market"):
                original = (root / broken).read_bytes()
                (root / broken).write_bytes(b"changed")
                with patch("src.treasury_dealer_inputs.read_market_sources") as decoder:
                    with self.assertRaises(ValueError):
                        read_inputs(root, market_pins, "ledger.json", ledger_hash)
                    decoder.assert_not_called()
                (root / broken).write_bytes(original)

    def test_malformed_ledger_fails_before_market_decode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for value in ({"events": {}}, {"events": []}, [1, 2]):
                payload = json.dumps(value).encode()
                (root / "ledger.json").write_bytes(payload)
                with patch("src.treasury_dealer_inputs.read_market_sources") as decoder:
                    with self.assertRaises(ValueError):
                        read_inputs(
                            root, {}, "ledger.json", hashlib.sha256(payload).hexdigest()
                        )
                    decoder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
