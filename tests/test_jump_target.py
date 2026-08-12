"""Pre-run contracts for the frozen SPX jump-target study."""

from __future__ import annotations

import io
import unittest
import zipfile

import numpy as np
import pandas as pd

from src import jump_target


def _protocol() -> dict:
    return {
        "source": {
            "asset": "SPX",
            "accepted_symbols": [".SPX", "SPX", "SPX2"],
            "forbidden_symbols": [".IXIC", "IXIC"],
            "required_columns": ["rv5", "bv"],
            "immutable_hash_required": True,
        },
        "windows": {
            "training_start": "2004-01-01",
            "confirmation_start": "2014-01-02",
            "confirmation_end": "2017-12-29",
            "forbid_2022_and_later": True,
        },
        "target": {"horizon_sessions": 2, "material_jump_quantile": 0.90},
        "inputs": {"no_forward_fill": True},
        "fitting": {"min_train_observations": 3, "logistic_ridge": 1e-6},
    }


def _archive(csv_text: str) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("OxfordManRealizedVolatilityIndices.csv", csv_text)
    return output.getvalue()


class SourceContract(unittest.TestCase):
    def test_long_archive_selects_spx_and_never_ixic(self):
        content = _archive(
            "date,Symbol,rv5,bv\n"
            "2014-01-02,.IXIC,9.0,8.0\n"
            "2014-01-02,.SPX,4.0,3.0\n"
            "2014-01-03,.SPX,5.0,4.5\n"
        )
        got = jump_target.parse_oxford_archive(content, _protocol())
        self.assertEqual(got.index.tolist(), pd.to_datetime(["2014-01-02", "2014-01-03"]).tolist())
        self.assertEqual(got["rv5"].tolist(), [4.0, 5.0])

    def test_missing_spx_fails_instead_of_substituting_nasdaq(self):
        content = _archive("date,Symbol,rv5,bv\n2014-01-02,.IXIC,9.0,8.0\n")
        with self.assertRaisesRegex(ValueError, "SPX"):
            jump_target.parse_oxford_archive(content, _protocol())

    def test_mixed_dst_offsets_preserve_the_stated_trading_date(self):
        content = _archive(
            "date,Symbol,rv5,bv\n"
            "2014-01-02 00:00:00+00:00,.SPX,4.0,3.0\n"
            "2014-06-02 00:00:00+01:00,.SPX,5.0,4.0\n"
        )
        got = jump_target.parse_oxford_archive(content, _protocol())
        self.assertEqual(
            got.index.tolist(),
            pd.to_datetime(["2014-01-02", "2014-06-02"]).tolist(),
        )


class TargetContract(unittest.TestCase):
    def test_components_reconcile_and_bv_above_rv_is_truncated(self):
        raw = pd.DataFrame(
            {"rv5": [4.0, 3.0], "bv": [3.0, 4.0]},
            index=pd.bdate_range("2014-01-02", periods=2),
        )
        got = jump_target.decompose_jump(raw)
        np.testing.assert_allclose(got["continuous"] + got["jump"], got["rv5"])
        self.assertEqual(got["jump"].tolist(), [1.0, 0.0])
        self.assertEqual(got["jump_share"].tolist(), [0.25, 0.0])

    def test_fold_threshold_and_training_rows_use_only_completed_history(self):
        idx = pd.bdate_range("2013-12-23", periods=8)
        share = pd.Series([0.0, 0.1, 0.2, 0.3, 0.0, 0.9, 0.0, 0.0], index=idx)
        frame = jump_target.build_fold_targets(
            share, cutoff=idx[3], origins=idx[4:6], horizon=2, quantile=0.90
        )
        expected = share.loc[:idx[3]].quantile(0.90)
        self.assertTrue((frame["threshold"] == expected).all())
        self.assertTrue(frame.loc[idx[4], "event"])
        eligible = jump_target.completed_training_origins(idx, cutoff=idx[5], horizon=2)
        self.assertEqual(eligible.max(), idx[3])

    def test_cboe_close_is_delayed_without_forward_fill(self):
        sessions = pd.bdate_range("2014-01-02", periods=4)
        raw = pd.Series([10.0, 20.0, 40.0], index=sessions[[0, 1, 3]])
        got = jump_target.align_cboe_close(raw, sessions, delay_sessions=1)
        self.assertTrue(np.isnan(got.iloc[0]))
        self.assertEqual(got.iloc[1], 10.0)
        self.assertEqual(got.iloc[2], 20.0)
        self.assertTrue(np.isnan(got.iloc[3]))


if __name__ == "__main__":
    unittest.main()
