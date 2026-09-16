"""Generated six-source checked-snapshot admission contracts."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import joint_copula_inputs as inputs


class JointCopulaInputsTests(unittest.TestCase):
    def fixture(self, root):
        dates = pd.DatetimeIndex(
            ["2017-04-12", "2017-04-13", "2025-10-20", "2025-11-03"], name="date"
        )
        pins = {}
        for name, path in inputs.SOURCES.items():
            full = root / path
            full.parent.mkdir(parents=True, exist_ok=True)
            if name in ("qqq", "spx"):
                pd.DataFrame(
                    {
                        "open": [100.0, 101.0, 102.0, float("inf")],
                        "high": [102.0, 103.0, 104.0, float("inf")],
                        "low": [99.0, 100.0, 101.0, float("inf")],
                        "close": [101.0, 102.0, 103.0, float("inf")],
                    },
                    index=dates,
                ).to_parquet(full)
            else:
                field = "VVIX" if name == "vvix" else "CLOSE"
                full.write_text(
                    f"DATE,{field}\n04/12/2017,20\n04/13/2017,21\n10/20/2025,22\n11/03/2025,FORBIDDEN\n"
                )
            pins[path] = hashlib.sha256(full.read_bytes()).hexdigest()
        return pins

    def test_bounded_numerical_decode_keeps_sources_and_dates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = self.fixture(root)
            q, s, iv, audit = inputs.read_sources(root, pins)
        self.assertEqual(len(q), 3)
        self.assertEqual(len(s), 3)
        self.assertEqual(len(iv), 3)
        self.assertEqual(set(iv), {"vxn", "vix", "vix9d", "vvix"})
        self.assertEqual(audit["source_sha256"], pins)
        self.assertEqual(audit["status"], "VERIFIED_BOUNDED_SOURCE_BUFFERS")

    def test_every_hash_precedes_any_metadata_or_numeric_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = self.fixture(root)
            pins[inputs.SOURCES["vvix"]] = "0" * 64
            with (
                patch.object(inputs, "_metadata") as decode,
                self.assertRaises(ValueError),
            ):
                inputs.read_sources(root, pins)
            decode.assert_not_called()

    def test_all_date_metadata_precedes_any_numeric_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = self.fixture(root)
            p = root / inputs.SOURCES["vvix"]
            p.write_text(p.read_text().replace("11/03/2025", "NOT_A_DATE"))
            pins[inputs.SOURCES["vvix"]] = hashlib.sha256(p.read_bytes()).hexdigest()
            with (
                patch.object(inputs, "_decode") as decode,
                self.assertRaises(ValueError),
            ):
                inputs.read_sources(root, pins)
            decode.assert_not_called()

    def test_numeric_decode_uses_original_checked_buffers_after_file_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = self.fixture(root)
            metadata = inputs._metadata
            changed = []

            def mutate(*args):
                result = metadata(*args)
                if not changed:
                    (root / inputs.SOURCES["vix"]).write_bytes(b"not the admitted source")
                    changed.append(True)
                return result

            with patch.object(inputs, "_metadata", side_effect=mutate):
                _, _, iv, audit = inputs.read_sources(root, pins)
            self.assertEqual(iv.vix.tolist(), [20.0, 21.0, 22.0])
            self.assertEqual(audit["source_sha256"], pins)

    def test_wrong_inventory_and_unbounded_source_end_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = self.fixture(root)
            with self.assertRaises(ValueError):
                inputs.read_sources(root, dict(list(pins.items())[:-1]))
            with self.assertRaises(ValueError):
                inputs.read_sources(root, pins, source_end="2025-10-21")


if __name__ == "__main__":
    unittest.main()
