"""Independent recomputation of every five-path result.

This file intentionally does not import :mod:`src.research_paths`. It reads raw
or persisted artifacts and reconstructs source hashes, fences, samples, and
headline metrics through a second code path.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "research_paths.yaml"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def independent_qlike(actual, forecast) -> np.ndarray:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    ratio = actual / forecast
    return ratio - np.log(ratio) - 1.0


def assert_close(label: str, observed, expected, *, rtol=1e-8, atol=1e-10) -> None:
    if np.isnan(observed) and np.isnan(expected):
        return
    if not np.isclose(observed, expected, rtol=rtol, atol=atol):
        raise AssertionError(f"{label}: observed {observed!r}, expected {expected!r}")


def verify_origin_fence(origins, forbidden_start) -> None:
    index = pd.DatetimeIndex(pd.to_datetime(origins))
    if len(index) and index.max().tz_localize(None).normalize() >= pd.Timestamp(forbidden_start):
        raise AssertionError(
            f"origin fence violated: {index.max()} >= {pd.Timestamp(forbidden_start)}"
        )


def _load() -> tuple[dict, Path, Path]:
    spec = yaml.safe_load(SPEC_PATH.read_text())
    data = ROOT / spec["outputs"]["data_dir"]
    reports = ROOT / spec["outputs"]["report_dir"]
    return spec, data, reports


def verify_source_hashes(data: Path) -> dict:
    manifest = json.loads((data / "source_manifest.json").read_text())
    checked = []
    for key, metadata in manifest["sources"].items():
        if metadata["provider"] == "Cboe":
            path = data / "raw" / f"{key.upper()}_History.csv"
        elif key == "spx":
            path = data / "spx_daily.parquet"
        else:
            path = data / f"{key}_daily.parquet"
        actual = file_sha256(path)
        if actual != metadata["sha256"]:
            raise AssertionError(f"source hash mismatch: {key}")
        checked.append(key)
    return {"sources": len(checked), "keys": checked}


def verify_top25(spec: dict, data: Path) -> dict:
    raw = pd.read_parquet(ROOT / spec["quarterly_top25"]["source"])
    raw["pct_value"] = pd.to_numeric(raw["pct_value"])
    raw["cusip"] = raw["cusip"].fillna("").astype(str).str.upper()
    raw = raw.loc[
        (raw["asset_category"] == "EC") & (raw["pct_value"] > 0)
        & (raw["cusip"].str.len() >= 6)
    ].copy()
    raw["issuer_id"] = raw["cusip"].str[:6]
    expected = (raw.groupby(["accession", "report_date", "accepted_at", "issuer_id"],
                            as_index=False)["pct_value"].sum())
    expected = expected.sort_values(
        ["accession", "pct_value", "issuer_id"],
        ascending=[True, False, True], kind="stable",
    )
    expected["rank"] = expected.groupby("accession").cumcount() + 1
    expected = expected.loc[expected["rank"] <= 25]
    saved = pd.read_parquet(data / "qqq_top25_quarterly.parquet")
    if not (saved.groupby("accession").size() == 25).all():
        raise AssertionError("saved top-25 snapshot does not contain 25 issuers")
    merged = expected.merge(
        saved[["accession", "issuer_id", "pct_value", "rank"]],
        on=["accession", "issuer_id"], suffixes=("_expected", "_saved"),
        validate="one_to_one",
    )
    if len(merged) != len(saved):
        raise AssertionError("saved top-25 keys differ from independent ranking")
    np.testing.assert_allclose(merged["pct_value_expected"], merged["pct_value_saved"])
    np.testing.assert_array_equal(merged["rank_expected"], merged["rank_saved"])

    events = pd.read_parquet(data / "top25_earnings_events.parquet")
    origin_at = (pd.to_datetime(events["origin"]).dt.tz_localize("America/New_York")
                 + pd.Timedelta(hours=16)).dt.tz_convert("UTC")
    if (pd.to_datetime(events["accepted_at"], utc=True) > origin_at).any():
        raise AssertionError("future top-25 filing entered earnings event panel")
    if (events["rank"] > 25).any() or events.duplicated(["target_date", "issuer_id"]).any():
        raise AssertionError("invalid top-25 earnings membership")
    return {
        "snapshots": int(saved["accession"].nunique()),
        "rows": int(len(saved)), "issuers": int(saved["issuer_id"].nunique()),
        "eligible_events": int(len(events)),
    }


def _absorption_design(master: pd.DataFrame, earnings: pd.Series) -> pd.DataFrame:
    design = pd.DataFrame(index=master.index)
    design["const"] = 1.0
    design["lrv_d"] = master["log_rv"]
    design["lrv_w"] = np.log(master["rv_total"].rolling(5).mean())
    design["lrv_m"] = np.log(master["rv_total"].rolling(22).mean())
    ret = master["ret_cc"]
    for name, values in (
        ("lev_d", ret), ("lev_w", ret.rolling(5).mean()),
        ("lev_m", ret.rolling(22).mean()),
    ):
        design[name] = values.where(values < 0, 0.0)
    design["liv"] = np.log(master["vxn"])
    target_day = pd.Series(master.index.dayofweek, index=master.index).shift(-1)
    for number, name in zip((1, 2, 3, 4),
                            ("tuesday", "wednesday", "thursday", "friday")):
        design[f"target_{name}"] = (target_day == number).astype(float)
    design["target_fomc"] = master["is_fomc"].shift(-1)
    design["target_cpi"] = master["is_cpi"].shift(-1)
    design["target_nfp"] = master["is_nfp"].shift(-1)
    design["target_post_fomc"] = master["is_fomc"]
    design["target_top25_earnings_weight"] = earnings.reindex(master.index).shift(-1)
    design["y_next"] = master["log_rv"].shift(-1)
    return design


def _wald(result, names: list[str]) -> float:
    restriction = np.zeros((len(names), len(result.params)))
    parameters = list(result.params.index)
    for row, name in enumerate(names):
        restriction[row, parameters.index(name)] = 1
    return float(result.wald_test(restriction, scalar=True).statistic)


def verify_absorption(spec: dict, data: Path) -> dict:
    study = spec["absorption_map"]
    master_all = pd.read_parquet(ROOT / study["source"])
    end_pos = int(master_all.index.get_loc(pd.Timestamp(study["end"])))
    master = master_all.iloc[:end_pos + 2]
    earnings = pd.read_parquet(data / "top25_earnings_daily.parquet").iloc[:, 0]
    design = _absorption_design(master, earnings)
    saved = json.loads((data / "absorption_map_metrics.json").read_text())["groups"]
    base = ["const", "lrv_d", "lrv_w", "lrv_m"]
    for group, names in study["groups"].items():
        names = list(names)
        columns = [*base, "liv", *names, "y_next"]
        sample = design.loc[study["start"]:study["end"], columns].dropna()
        without = sm.OLS(sample["y_next"], sample[[*base, *names]]).fit(
            cov_type="HAC", cov_kwds={"maxlags": int(study["hac_maxlags"])}
        )
        with_market = sm.OLS(sample["y_next"], sample[[*base, "liv", *names]]).fit(
            cov_type="HAC", cov_kwds={"maxlags": int(study["hac_maxlags"])}
        )
        first, second = _wald(without, names), _wald(with_market, names)
        assert_close(f"{group} Wald without", first, saved[group]["wald_without"])
        assert_close(f"{group} Wald with", second, saved[group]["wald_with"])
        assert_close(f"{group} attenuation", 1 - second / first,
                     saved[group]["attenuation"])
        if int(without.nobs) != saved[group]["n_without"]:
            raise AssertionError(f"{group}: absorption n mismatch")
    assert_close("leverage ledger without", saved["leverage"]["wald_without"],
                 103.18752571911071)
    assert_close("leverage ledger with", saved["leverage"]["wald_with"],
                 62.63046666979283)
    return {group: {"n": row["n_with"], "attenuation": row["attenuation"]}
            for group, row in saved.items()}


def verify_horizon(spec: dict, data: Path) -> dict:
    metrics = json.loads((data / "horizon_curve_metrics.json").read_text())["rows"]
    forecasts = pd.read_parquet(data / "horizon_curve_forecasts.parquet")
    verify_origin_fence(forecasts["origin"], spec["fences"]["sealed_ndx_clean_start"])
    verified = []
    for row in metrics:
        horizon = int(row["horizon"])
        part = forecasts.loc[forecasts["horizon"] == horizon]
        base = part.loc[part["model"] == "har"].set_index("origin")
        market = part.loc[part["model"] == "har_iv"].set_index("origin")
        common = base.index.intersection(market.index)
        actual = base.loc[common, "actual_var"].to_numpy()
        loss_base = independent_qlike(actual, base.loc[common, "mean_var"])
        loss_market = independent_qlike(actual, market.loc[common, "mean_var"])
        assert_close(f"h{horizon} HAR QLIKE", loss_base.mean(), row["qlike_har"])
        assert_close(f"h{horizon} IV QLIKE", loss_market.mean(), row["qlike_har_iv"])
        improvement = 100 * (loss_base.mean() - loss_market.mean()) / loss_base.mean()
        assert_close(f"h{horizon} improvement", improvement,
                     row["qlike_improvement_pct"])
        if len(common) != row["n"]:
            raise AssertionError(f"h{horizon}: forecast n mismatch")
        verified.append({"horizon": horizon, "n": len(common),
                         "improvement_pct": improvement})
    return {"rows": verified}


def _calendar_realized(returns: pd.Series, horizon: int) -> pd.Series:
    output = pd.Series(np.nan, index=returns.index)
    for date in returns.index:
        end = date + pd.Timedelta(days=horizon)
        if returns.index[-1] < end:
            continue
        window = returns.loc[(returns.index > date) & (returns.index <= end)].dropna()
        if len(window):
            output.loc[date] = 100 * np.sqrt(365 / horizon * float((window ** 2).sum()))
    return output


def verify_vrp(spec: dict, data: Path) -> dict:
    saved = json.loads((data / "vrp_term_structure_metrics.json").read_text())
    observations = pd.read_parquet(data / "vrp_term_structure_observations.parquet")
    spx = pd.read_parquet(data / "spx_daily.parquet")
    cboe = pd.read_parquet(data / "cboe_indices.parquet")
    returns = np.log(spx["close"] / spx["close"].shift(1))
    for row in saved["rows"]:
        symbol, horizon = row["symbol"], int(row["horizon_calendar_days"])
        realized = _calendar_realized(returns, horizon).reindex(observations.index)
        implied = cboe[symbol].reindex(observations.index)
        np.testing.assert_allclose(
            observations[f"{symbol}_realized"], realized, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(observations[f"{symbol}_implied"], implied)
        premium = observations[f"{symbol}_premium"]
        state = observations["negative_leverage_state"].astype(bool)
        assert_close(f"{symbol} premium", premium.mean(), row["premium_mean"])
        assert_close(f"{symbol} negative premium", premium[state].mean(),
                     row["negative_state_premium"])
        assert_close(f"{symbol} other premium", premium[~state].mean(),
                     row["nonnegative_state_premium"])
        if not row["premium_mean_ci95"][0] <= row["premium_mean"] <= row["premium_mean_ci95"][1]:
            raise AssertionError(f"{symbol}: point outside bootstrap interval")
    return {"common_n": len(observations),
            "premiums": {row["symbol"]: row["premium_mean"] for row in saved["rows"]}}


def _latest_accession(top: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.Series:
    snapshots = top[["accession", "accepted_at"]].drop_duplicates().sort_values("accepted_at")
    snapshots["accepted_at"] = pd.to_datetime(snapshots["accepted_at"], utc=True)
    left = pd.DataFrame({"origin": origins})
    left["origin_at"] = (left["origin"].dt.tz_localize("America/New_York")
                         + pd.Timedelta(hours=16)).dt.tz_convert("UTC")
    return pd.merge_asof(left.sort_values("origin_at"), snapshots,
                         left_on="origin_at", right_on="accepted_at",
                         direction="backward").set_index("origin")["accession"]


def verify_single_name(spec: dict, data: Path) -> dict:
    saved = json.loads((data / "single_name_earnings_metrics.json").read_text())
    forecasts = pd.read_parquet(data / "single_name_earnings_forecasts.parquet")
    top = pd.read_parquet(data / "qqq_top25_quarterly.parquet")
    events = pd.read_parquet(data / "top25_earnings_events.parquet")
    effects = []
    for row in saved["rows"]:
        asset, issuer = row["asset"], row["issuer_id"]
        part = forecasts.loc[forecasts["asset"] == asset].copy()
        if row["eligible_n"] == 0:
            if len(part):
                raise AssertionError(f"{asset}: expected zero eligible rows")
            continue
        part["origin"] = pd.to_datetime(part["origin"])
        if len(part) != row["eligible_n"]:
            raise AssertionError(f"{asset}: eligible n mismatch")
        residual = np.log(part["actual_var"] / part["mean_var"])
        np.testing.assert_allclose(residual, part["log_forecast_residual"])
        event = part["is_earnings"].astype(bool)
        effect = float(residual[event].mean() - residual[~event].mean())
        assert_close(f"{asset} event effect", effect, row["event_effect_log_variance"])
        if int(event.sum()) != row["event_n"]:
            raise AssertionError(f"{asset}: event n mismatch")
        qlike = independent_qlike(part["actual_var"], part["mean_var"])
        assert_close(f"{asset} QLIKE", qlike.mean(), row["mean_qlike"])

        latest = _latest_accession(top, pd.DatetimeIndex(part["origin"]))
        allowed = set(zip(top["accession"], top["issuer_id"]))
        if any((accession, issuer) not in allowed for accession in latest):
            raise AssertionError(f"{asset}: current-name or future-membership row escaped")
        expected_events = set(pd.to_datetime(events.loc[events["issuer_id"] == issuer, "origin"]))
        if not np.array_equal(event.to_numpy(), part["origin"].isin(expected_events).to_numpy()):
            raise AssertionError(f"{asset}: event label mismatch")
        effects.append(effect)
    pooled = float(np.mean(effects))
    assert_close("equal asset pooled earnings effect", pooled,
                 saved["equal_asset_pool"]["equal_asset_effect_log_variance"])
    return {"assets": len(effects), "pooled_effect": pooled,
            "forecast_rows": len(forecasts)}


def verify_spx_term(spec: dict, data: Path) -> dict:
    saved = json.loads((data / "spx_term_slope_metrics.json").read_text())
    forecasts = pd.read_parquet(data / "spx_term_slope_forecasts.parquet")
    if pd.Timestamp(forecasts["origin"].max()) > pd.Timestamp(spec["spx_term_slope_replication"]["score_end"]):
        raise AssertionError("SPX term score escaped frozen window")
    pivot = {name: frame.set_index("origin") for name, frame in forecasts.groupby("model")}
    common = pivot["baseline"].index
    actual = pivot["baseline"].loc[common, "actual_var"]
    losses = {name: independent_qlike(actual, frame.loc[common, "mean_var"])
              for name, frame in pivot.items()}
    baseline_mean = float(losses["baseline"].mean())
    assert_close("SPX baseline QLIKE", baseline_mean, saved["baseline_qlike"])
    for key in ("unconditional_slope", "dislocation_only"):
        mean = float(losses[key].mean())
        assert_close(f"SPX {key} QLIKE", mean, saved[key]["qlike"])
        improvement = 100 * (baseline_mean - mean) / baseline_mean
        assert_close(f"SPX {key} improvement", improvement,
                     saved[key]["improvement_pct"])
    rule = (
        losses["dislocation_only"].mean() < losses["baseline"].mean()
        and saved["dislocation_only"]["dm_p"] < 0.05
        and saved["dislocation_only"]["paired_win_rate"] > 0.5
        and losses["dislocation_only"].mean() < losses["unconditional_slope"].mean()
    )
    if ("PASS" if rule else "FAIL") != saved["verdict"]:
        raise AssertionError("SPX registered verdict mismatch")
    return {"n": len(common), "verdict": saved["verdict"],
            "dislocation_improvement_pct": saved["dislocation_only"]["improvement_pct"]}


def verify_all(write: bool = True) -> dict:
    spec, data, reports = _load()
    checks = {
        "source_hashes": verify_source_hashes(data),
        "quarterly_top25": verify_top25(spec, data),
        "absorption_map": verify_absorption(spec, data),
        "horizon_curve": verify_horizon(spec, data),
        "vrp_term_structure": verify_vrp(spec, data),
        "single_name_earnings": verify_single_name(spec, data),
        "spx_term_slope": verify_spx_term(spec, data),
    }
    payload = {"verdict": "PASS", "independent_checks": checks}
    if write:
        (data / "verification.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        reports.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Independent verification of the five-path extension", "",
            "**PASS.** This verifier does not import `src.research_paths`. It reconstructs",
            "source hashes, quarterly issuer rankings, acceptance-time fences, the full",
            "absorption regressions, all forecast losses, VRP targets, single-name event",
            "effects, and the SPX registered verdict from persisted source artifacts.", "",
            f"- Source files matched all {checks['source_hashes']['sources']} recorded hashes.",
            f"- Quarterly universe: {checks['quarterly_top25']['snapshots']} snapshots, "
            f"{checks['quarterly_top25']['rows']} rows, "
            f"{checks['quarterly_top25']['eligible_events']} eligible realized events.",
            "- Leverage anchor independently reconstructs Wald 103.1875 → 62.6305.",
            f"- Horizon curve: {len(checks['horizon_curve']['rows'])} frozen horizons recomputed.",
            f"- VRP common sample: {checks['vrp_term_structure']['common_n']} origins.",
            f"- Single-name pool: {checks['single_name_earnings']['assets']} eligible assets; "
            f"effect {checks['single_name_earnings']['pooled_effect']:+.4f} log variance.",
            f"- SPX term-slope verdict: {checks['spx_term_slope']['verdict']} on "
            f"{checks['spx_term_slope']['n']} origins.",
        ]
        (reports / "verification.md").write_text("\n".join(lines) + "\n")
    return payload


if __name__ == "__main__":
    print(json.dumps(verify_all(write=True), indent=2, sort_keys=True))
