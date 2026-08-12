"""Unit contracts for the independent research-path verifier."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.verify_research_paths import (
    assert_close,
    file_sha256,
    independent_qlike,
    verify_origin_fence,
)


class IndependentPrimitiveContract(unittest.TestCase):
    def test_qlike_matches_hand_calculation(self):
        actual = np.array([1.0, 2.0])
        forecast = np.array([2.0, 1.0])
        expected = actual / forecast - np.log(actual / forecast) - 1
        np.testing.assert_allclose(independent_qlike(actual, forecast), expected)

    def test_hash_changes_when_source_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            path.write_text("a\n1\n")
            before = file_sha256(path)
            path.write_text("a\n2\n")
            self.assertNotEqual(before, file_sha256(path))

    def test_numeric_mismatch_fails_loudly(self):
        assert_close("same", 1.0, 1.0 + 1e-10)
        with self.assertRaises(AssertionError):
            assert_close("different", 1.0, 1.1)

    def test_origin_fence_rejects_clean_overlap(self):
        verify_origin_fence(pd.to_datetime(["2025-10-17"]), "2025-11-03")
        with self.assertRaises(AssertionError):
            verify_origin_fence(pd.to_datetime(["2025-11-03"]), "2025-11-03")

    def test_persisted_forecasts_name_the_origin_column(self):
        path = Path(__file__).resolve().parents[1] / (
            "data/research_paths/single_name_earnings_forecasts.parquet"
        )
        if path.exists():
            self.assertIn("origin", pd.read_parquet(path).columns)


if __name__ == "__main__":
    unittest.main()
