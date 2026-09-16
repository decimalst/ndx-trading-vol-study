"""Synthetic producer/checker agreement before any numerical source admission."""

import copy
import hashlib
import json
import unittest

from src.commodity_implied_source import parse_cboe_history
from src.verify_commodity_implied_source import verify_history


class CommoditySourceIntegrationTests(unittest.TestCase):
    def test_complete_generated_json_roundtrip_is_independently_verified(self):
        for symbol in ("OVX", "GVZ"):
            for start, end in (
                ("2009-01-02", "2025-10-20"),
                ("2010-01-04", "2010-01-04"),
                ("2011-01-04", "2011-01-04"),
            ):
                with self.subTest(symbol=symbol, start=start):
                    payload = (
                        f"DATE,{symbol}\n2008-01-01,UNREAD_EARLY\n"
                        "2009-01-02,20\n2010-01-04,.\n2010-01-05,\n"
                        "2025-10-20,+2.5e1\n2026-01-01,UNREAD_LATE\n"
                    ).encode()
                    signature = hashlib.sha256(payload).hexdigest()
                    parsed = parse_cboe_history(payload, symbol, signature, start, end)
                    decoded = json.loads(json.dumps(parsed, allow_nan=False))
                    proof = verify_history(payload, symbol, signature, decoded, start, end)
                    self.assertEqual(proof["status"], "VERIFIED")
                    self.assertEqual(proof["retained_rows"], len(parsed["records"]))

    def test_checker_rejects_silent_corruption_of_producer_output(self):
        payload = b"DATE,OVX\n2009-01-02,20\n2009-01-05,.\n"
        signature = hashlib.sha256(payload).hexdigest()
        parsed = parse_cboe_history(payload, "OVX", signature)
        corruptions = [
            ("record_value", lambda obj: obj["records"][0].update(value=21.0)),
            (
                "filled_missing",
                lambda obj: obj["records"][1].update(value=20.0, status="observed"),
            ),
            ("source_line", lambda obj: obj["records"][0].update(source_line=1)),
            ("count", lambda obj: obj["metadata"].update(after_end_rows=1)),
        ]
        for name, corrupt in corruptions:
            with self.subTest(name=name):
                result = copy.deepcopy(parsed)
                corrupt(result)
                with self.assertRaises(ValueError):
                    verify_history(payload, "OVX", signature, result)


if __name__ == "__main__":
    unittest.main()
