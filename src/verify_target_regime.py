"""Independent audit of stored jump-target and regime-transition artifacts."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import yaml

from . import config

ROOT = config.ROOT
OUT = ROOT / "data" / "target_regime"


def _losses(frame: pd.DataFrame, probability: str) -> tuple[float, float]:
    y = frame["event"].to_numpy(dtype=float)
    p = np.clip(frame[probability].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    return float(np.mean((p - y) ** 2)), float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def _phase_metrics(frame: pd.DataFrame, models: tuple[str, ...], step: int) -> pd.DataFrame:
    rows = []
    for phase in range(step):
        sample = frame.iloc[phase::step]
        row = {"phase": phase, "n": len(sample), "event_rate": float(sample["event"].mean())}
        for model in models:
            brier, logloss = _losses(sample, f"p_{model}")
            row[f"{model}_brier"] = brier
            row[f"{model}_logloss"] = logloss
        rows.append(row)
    return pd.DataFrame(rows)


def _verify_jump(protocol: dict) -> None:
    metadata = json.loads((ROOT / "data/raw/oxford_man_spx_source.json").read_text())
    archive_hash = hashlib.sha256((ROOT / "data/raw/oxford_man_realized.zip").read_bytes()).hexdigest()
    if archive_hash != metadata["sha256"]:
        raise AssertionError("Oxford-Man archive hash does not match source metadata")
    raw = pd.read_parquet(ROOT / "data/raw/oxford_man_spx.parquet")
    continuous = np.minimum(raw["rv5"], raw["bv"])
    jump = np.maximum(raw["rv5"] - raw["bv"], 0.0)
    share = jump / raw["rv5"]
    np.testing.assert_allclose(continuous + jump, raw["rv5"], rtol=1e-12, atol=1e-15)

    stored = pd.read_parquet(OUT / "jump_forecasts.parquet")
    np.testing.assert_allclose(stored["jump_share_d"], share.reindex(stored.index))
    np.testing.assert_allclose(stored["jump_share_w"], share.rolling(5).mean().reindex(stored.index))
    np.testing.assert_allclose(stored["jump_share_m"], share.rolling(22).mean().reindex(stored.index))
    cfg = config.load()
    implied = pd.read_parquet(cfg["paths"]["raw"] / "short_dated_iv.parquet")["vix"]
    skew = pd.read_parquet(cfg["paths"]["raw"] / "skew_daily.parquet")["close"]
    expected_vix = implied.reindex(raw.index).shift(1).reindex(stored.index)
    expected_skew = skew.reindex(raw.index).shift(1).reindex(stored.index)
    pd.testing.assert_series_equal(expected_vix.rename("vix_lagged"), stored["vix_lagged"], check_freq=False)
    pd.testing.assert_series_equal(expected_skew.rename("skew_lagged"), stored["skew_lagged"], check_freq=False)

    horizon = int(protocol["target"]["horizon_sessions"])
    quantile = float(protocol["target"]["material_jump_quantile"])
    positions = pd.Series(np.arange(len(raw)), index=raw.index)
    for year, fold in stored.groupby("fold_year"):
        cutoff_values = pd.to_datetime(fold["cutoff"].unique())
        if len(cutoff_values) != 1 or cutoff_values[0] >= fold.index.min():
            raise AssertionError(f"jump fold {year} has an invalid annual cutoff")
        threshold = float(share.loc[:cutoff_values[0]].quantile(quantile))
        np.testing.assert_allclose(fold["threshold"], threshold)
        for origin, row in fold.iterrows():
            position = int(positions.loc[origin])
            future = share.iloc[position + 1 : position + 1 + horizon]
            if row["target_end"] != future.index[-1] or bool(row["event"]) != bool((future > threshold).any()):
                raise AssertionError(f"jump target mismatch at {origin.date()}")
    lo = pd.Timestamp(protocol["windows"]["confirmation_start"])
    hi = pd.Timestamp(protocol["windows"]["confirmation_end"])
    if stored.index.min() < lo or stored.index.max() > hi or (stored["target_end"] > hi).any():
        raise AssertionError("jump origins or targets escape confirmation")
    recomputed = _phase_metrics(stored, ("history", "atm", "surface"), horizon)
    recorded = pd.read_parquet(OUT / "jump_phase_metrics.parquet")
    pd.testing.assert_frame_equal(recomputed, recorded, check_exact=False, rtol=0, atol=1e-15)
    metrics = json.loads((OUT / "jump_metrics.json").read_text())
    avg = recomputed.mean(numeric_only=True)
    checks = {
        "surface_brier_below_atm": bool(avg["surface_brier"] < avg["atm_brier"]),
        "surface_logloss_below_atm": bool(avg["surface_logloss"] < avg["atm_logloss"]),
    }
    if checks != metrics["checks"] or bool(all(checks.values())) != metrics["pass"]:
        raise AssertionError("jump verdict inputs do not reproduce")


def _verify_regime(protocol: dict) -> None:
    cfg = config.load()
    y = pd.read_parquet(cfg["paths"]["processed"] / "master_daily.parquet")["log_rv"].dropna().sort_index()
    stored = pd.read_parquet(OUT / "regime_forecasts.parquet")
    expected = pd.DataFrame({
        "log_rv_d": y,
        "log_rv_w": y.rolling(5).mean(),
        "log_rv_m": y.rolling(22).mean(),
    }).reindex(stored.index)
    for column in expected:
        pd.testing.assert_series_equal(expected[column], stored[column], check_freq=False)
    if not np.allclose(stored["p_low_filtered"] + stored["p_high_filtered"], 1.0):
        raise AssertionError("stored filtered state probabilities do not sum to one")
    horizon = int(protocol["target"]["horizon_sessions"])
    quantile = float(protocol["target"]["stress_quantile"])
    positions = pd.Series(np.arange(len(y)), index=y.index)
    for year, fold in stored.groupby("fold_year"):
        cutoff_values = pd.to_datetime(fold["cutoff"].unique())
        if len(cutoff_values) != 1 or cutoff_values[0] >= fold.index.min():
            raise AssertionError(f"regime fold {year} has an invalid annual cutoff")
        threshold = float(y.loc[:cutoff_values[0]].quantile(quantile))
        np.testing.assert_allclose(fold["threshold"], threshold)
        if not (y.reindex(fold.index) <= threshold).all():
            raise AssertionError(f"regime fold {year} includes a non-calm origin")
        for origin, row in fold.iterrows():
            position = int(positions.loc[origin])
            future = y.iloc[position + 1 : position + 1 + horizon]
            if row["target_end"] != future.index[-1] or bool(row["event"]) != bool((future > threshold).any()):
                raise AssertionError(f"regime target mismatch at {origin.date()}")
    clean = pd.Timestamp(protocol["window"]["clean_start"])
    if (stored.index >= clean).any() or (stored["target_end"] >= clean).any():
        raise AssertionError("regime origin or target crosses the clean fence")
    recomputed = _phase_metrics(stored, ("logistic", "hmm"), horizon)
    recorded = pd.read_parquet(OUT / "regime_phase_metrics.parquet")
    pd.testing.assert_frame_equal(recomputed, recorded, check_exact=False, rtol=0, atol=1e-15)
    metrics = json.loads((OUT / "regime_metrics.json").read_text())
    avg = recomputed.mean(numeric_only=True)
    ranked = stored.sort_values("p_hmm")
    quintile = max(1, len(ranked) // 5)
    bottom = float(ranked.iloc[:quintile]["event"].mean())
    top = float(ranked.iloc[-quintile:]["event"].mean())
    checks = {
        "hmm_brier_below_logistic": bool(avg["hmm_brier"] < avg["logistic_brier"]),
        "hmm_logloss_below_logistic": bool(avg["hmm_logloss"] < avg["logistic_logloss"]),
        "hmm_top_quintile_above_bottom": bool(top > bottom),
    }
    if checks != metrics["checks"] or bool(all(checks.values())) != metrics["pass"]:
        raise AssertionError("regime verdict inputs do not reproduce")
    for fold in metrics["fold_parameters"]:
        means = np.asarray(fold["means"])
        transition = np.asarray(fold["transition"])
        if not means[0] < means[1] or not np.allclose(transition.sum(axis=1), 1.0):
            raise AssertionError(f"invalid stored HMM parameters for {fold['year']}")


def main() -> None:
    with open(ROOT / "target_regime.yaml") as source:
        protocol = yaml.safe_load(source)
    _verify_jump(protocol["jump_target"])
    _verify_regime(protocol["regime_transition"])
    # SCOPE, stated precisely because the earlier banner overstated it. This
    # module re-derives features, thresholds, targets, lags and fences from the
    # raw source, and recomputes every loss and verdict from the stored
    # probability columns. It does NOT re-fit the HMM or the logistic models,
    # and does NOT re-derive p_history / p_atm / p_surface / p_logistic /
    # p_hmm. A look-ahead injected into the forecast models themselves is
    # therefore invisible to it -- demonstrated by an audit that swapped in a
    # full-sample HMM fit and smoothed gammas and still saw this print PASSED.
    # Independent re-derivation is implemented for the repair study in
    # src/verify_regime_repair.py and is the standard this module should meet.
    print(
        "TARGET/REGIME VERIFICATION PASSED: source hash, component reconciliation, "
        "Cboe lags, annual thresholds, completed targets, filtered-state sums, "
        "five-phase losses and clean fence independently re-derived from source; "
        "verdict arithmetic recomputed from the stored forecast probabilities. "
        "NOTE: the forecasts themselves are NOT independently reproduced — the "
        "HMM and logistic fits are taken from the pipeline, so this check cannot "
        "detect leakage inside those models (see src/verify_regime_repair.py for "
        "the stronger standard)."
    )


if __name__ == "__main__":
    main()
