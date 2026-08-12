"""Independent recomputation of the frozen SKEW-carry diagnostic outputs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml

from . import config
from .carry import build_trades


def _stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    ordered = np.sort(values)
    tail = max(1, int(np.ceil(0.05 * len(values))))
    cumulative = np.cumsum(values)
    return {
        "n": len(values),
        "mean": values.mean(),
        "cvar5": ordered[:tail].mean(),
        "maxdd": (np.maximum.accumulate(cumulative) - cumulative).max(),
    }


def main() -> None:
    with open(config.ROOT / "skew_carry.yaml") as source:
        protocol = yaml.safe_load(source)
    main_cfg = config.load()
    stored = pd.read_parquet(config.ROOT / "data/skew_carry/diagnostic_trades.parquet")
    stored_phases = pd.read_parquet(config.ROOT / "data/skew_carry/phase_metrics.parquet")
    stored_metrics = json.loads(
        (config.ROOT / "data/skew_carry/metrics.json").read_text()
    )
    clean = pd.Timestamp(protocol["window"]["clean_start"])
    if len(stored) == 0 or (stored.index >= clean).any():
        raise AssertionError("stored diagnostic is empty or crosses clean fence")

    trades = build_trades(main_cfg).reindex(stored.index)
    threshold = float(trades["richness"].median())
    if not np.isclose(threshold, stored_metrics["richness_threshold"]):
        raise AssertionError("richness threshold mismatch")

    skew = pd.read_parquet(main_cfg["paths"]["raw"] / "skew_daily.parquet")["close"]
    master = pd.read_parquet(main_cfg["paths"]["processed"] / "master_daily.parquet")
    aligned = skew.reindex(master.index).shift(1)
    q = (
        aligned.rolling(252, min_periods=126).quantile(0.80).shift(1)
    ).reindex(stored.index)
    aligned = aligned.reindex(stored.index)
    allowed = aligned.notna() & q.notna() & (aligned <= q)
    original = trades["richness"] > threshold
    repaired = original & allowed

    pd.testing.assert_series_equal(
        aligned.rename("skew_lagged"), stored["skew_lagged"], check_freq=False
    )
    pd.testing.assert_series_equal(
        q.rename("skew_threshold"), stored["skew_threshold"], check_freq=False
    )
    pd.testing.assert_series_equal(
        original.rename("richness_taken"), stored["richness_taken"], check_freq=False
    )
    pd.testing.assert_series_equal(
        repaired.rename("repaired_taken"), stored["repaired_taken"], check_freq=False
    )

    rows = []
    for phase in range(21):
        sample = stored.iloc[phase::21]
        row = {"phase": phase, "n_all": len(sample)}
        for name, mask in (
            ("all", pd.Series(True, index=sample.index)),
            ("richness", sample["richness_taken"]),
            ("repaired", sample["repaired_taken"]),
        ):
            stats = _stats(sample.loc[mask, "pnl_vol"].to_numpy())
            row.update({f"{name}_{key}": value for key, value in stats.items()})
        rows.append(row)
    recalculated = pd.DataFrame(rows)
    for column in (
        "n_all",
        "all_n",
        "all_mean",
        "all_cvar5",
        "all_maxdd",
        "richness_n",
        "richness_mean",
        "richness_cvar5",
        "richness_maxdd",
        "repaired_n",
        "repaired_mean",
        "repaired_cvar5",
        "repaired_maxdd",
    ):
        np.testing.assert_allclose(
            recalculated[column], stored_phases[column], rtol=0, atol=1e-12
        )

    adverse = pd.to_datetime(protocol["evaluation"]["known_adverse_origins"])
    if not original.reindex(adverse).all() or repaired.reindex(adverse).any():
        raise AssertionError("known-adverse trade masks do not match the protocol")
    participation = stored_phases["repaired_n"].mean() / stored_phases["richness_n"].mean()
    if not np.isclose(participation, stored_metrics["participation"]):
        raise AssertionError("participation mismatch")
    print(
        "SKEW CARRY VERIFICATION PASSED: timing, masks, 21 phases, tail metrics, "
        "clean fence, and frozen verdict inputs independently match"
    )


if __name__ == "__main__":
    main()
