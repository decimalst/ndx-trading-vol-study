"""Independent source-to-forecast verifier for the additive round-two study.

No producer, shared feature, model, or metric implementation is imported. Raw
inputs are date-filtered before transformation; direct augmented least squares
checks the producer's training-only residualization by a different calculation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
BASELINE = (
    "const", "lrv_d", "lrv_w", "lrv_m", "lev_d", "lev_w", "lev_m",
    "liv", "lvix", "term", "xasset_stress", "market_stress",
)
CANDIDATES = ("vvix", "rv_dispersion", "stock_bond_corr", "close_pressure")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dated(frame: pd.DataFrame, cutoff: pd.Timestamp, label: str) -> pd.DataFrame:
    frame = frame.copy()
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise AssertionError(f"{label}: dates must be unique and increasing")
    if frame.empty or frame.index.max() > cutoff:
        raise AssertionError(f"{label}: source date fence violated")
    return frame


def reconstruct_features(root: Path, protocol: dict) -> pd.DataFrame:
    """Derive the fixed feature set directly from bounded source observations."""
    cutoff = pd.Timestamp(protocol["latest_target"])

    def parquet(relative: str) -> pd.DataFrame:
        raw = pd.read_parquet(root / relative, filters=[("date", "<=", cutoff)])
        return _dated(raw, cutoff, relative)

    qqq = parquet("data/raw/daily_ohlc.parquet")
    assets = parquet("data/raw/cross_asset_daily.parquet")
    vxn = parquet("data/raw/vxn_daily.parquet")
    short = parquet("data/raw/short_dated_iv.parquet")
    vvix_raw = pd.read_csv(root / "data/free_sources/raw/cboe/VVIX_History.csv")
    if list(vvix_raw.columns) != ["DATE", "VVIX"]:
        raise AssertionError("VVIX schema differs from the declared official source")
    vvix_raw["DATE"] = pd.to_datetime(vvix_raw["DATE"], format="%m/%d/%Y")
    vvix = _dated(
        vvix_raw.loc[vvix_raw["DATE"] <= cutoff].set_index("DATE"), cutoff, "VVIX"
    )
    index = qqq.index
    out = pd.DataFrame(index=index)
    log_range = np.log(qqq["high"] / qqq["low"])
    intraday_return = np.log(qqq["close"] / qqq["open"])
    intraday = (
        0.5 * log_range**2 - (2.0 * np.log(2.0) - 1.0) * intraday_return**2
    ).clip(lower=1e-10)
    overnight = np.log(qqq["open"] / qqq["close"].shift(1)) ** 2
    variance = intraday + overnight
    logged = np.log(variance.clip(lower=1e-10))
    qqq_return = np.log(qqq["adj close"] / qqq["adj close"].shift(1))
    out["const"] = 1.0
    out["rv_total"] = variance
    out["log_rv"] = logged
    out["lrv_d"] = logged
    out["lrv_w"] = np.log(variance.rolling(5).mean())
    out["lrv_m"] = np.log(variance.rolling(22).mean())
    for name, span in (("lev_d", 1), ("lev_w", 5), ("lev_m", 22)):
        aggregated = qqq_return if span == 1 else qqq_return.rolling(span).mean()
        out[name] = aggregated.clip(upper=0.0)
    out["liv"] = np.log(vxn["close"].where(vxn["close"] > 0).reindex(index)).shift(1)
    out["lvix"] = np.log(short["vix"].where(short["vix"] > 0).reindex(index)).shift(1)
    out["term"] = np.log(
        short["vix9d"].where(short["vix9d"] > 0)
        / short["vix"].where(short["vix"] > 0)
    ).reindex(index).shift(1)
    out["vvix"] = np.log(vvix["VVIX"].where(vvix["VVIX"] > 0).reindex(index)).shift(1)

    def standardized(series):
        history = series.rolling(252, min_periods=126)
        mean = history.mean().shift(1)
        scale = history.std(ddof=1).shift(1).replace(0.0, np.nan)
        return (series - mean) / scale

    prices = assets.reindex(index).loc[:, ["hyg", "tlt", "gld", "uso", "uup"]]
    returns = np.log(prices.where(prices > 0.0)).diff()
    z = standardized(returns).reindex(index)
    out["xasset_stress"] = np.sqrt(z.pow(2).mean(axis=1, skipna=False))
    volume = np.log(qqq["volume"].where(qqq["volume"] > 0.0))
    overnight_share = (overnight / variance.replace(0.0, np.nan)).clip(0.0, 1.0)
    out["market_stress"] = np.sqrt(
        (standardized(volume).clip(lower=0.0) ** 2
         + standardized(overnight_share).clip(lower=0.0) ** 2) / 2.0
    )
    out["rv_dispersion"] = logged.rolling(22).std(ddof=1)
    out["stock_bond_corr"] = qqq_return.rolling(22).corr(returns["tlt"].reindex(index))
    pressure = 2.0 * np.log(qqq["close"] / qqq["low"]) / log_range.replace(0.0, np.nan) - 1.0
    out["close_pressure"] = pressure.rolling(5).mean()
    return out.replace([np.inf, -np.inf], np.nan)


def reconstruct_target(features: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Require every future session in the arithmetic-mean variance target."""
    future = pd.concat(
        [features["rv_total"].shift(-step) for step in range(1, horizon + 1)], axis=1
    )
    end = pd.Series(features.index, index=features.index).shift(-horizon)
    return pd.DataFrame({"y": future.mean(axis=1, skipna=False), "target_end": end})


def _same_number(actual, expected, label: str, *, rtol: float = 2e-8, atol: float = 1e-12) -> None:
    if not np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True):
        raise AssertionError(f"{label}: independent calculation disagrees")


def verify_forecasts(
    features: pd.DataFrame, forecasts: pd.DataFrame, protocol: dict
) -> dict:
    """Independently refit every monthly model and check every saved forecast."""
    required = {
        "origin", "horizon", "model", "target_end", "y", "prediction", "fit_origin",
        "train_n", "train_last_target",
    }
    missing = required - set(forecasts.columns)
    if missing:
        raise AssertionError(f"forecast columns missing: {sorted(missing)}")
    saved = forecasts.copy()
    for column in ("origin", "target_end", "fit_origin", "train_last_target"):
        saved[column] = pd.to_datetime(saved[column])
    if saved.empty or saved.duplicated(["origin", "horizon", "model"]).any():
        raise AssertionError("forecasts must be nonempty and uniquely keyed")
    if set(saved["model"]) != {"baseline", *CANDIDATES}:
        raise AssertionError("forecast model set differs from protocol")
    if set(saved["horizon"]) != set(protocol["horizons"]):
        raise AssertionError("forecast horizon set differs from protocol")
    if not np.isfinite(saved[["y", "prediction"]]).all().all() or (
        saved[["y", "prediction"]] <= 0.0
    ).any().any():
        raise AssertionError("targets and forecasts must be finite and positive")
    start, stop = pd.Timestamp(protocol["origin_start"]), pd.Timestamp(protocol["origin_end"])
    latest, sealed = pd.Timestamp(protocol["latest_target"]), pd.Timestamp(protocol["sealed_start"])
    if not (saved["origin"].between(start, stop).all()
            and (saved["target_end"] <= latest).all()
            and (saved["target_end"] < sealed).all()):
        raise AssertionError("forecast origin or target fence violated")
    all_inputs = features[list(BASELINE) + list(CANDIDATES)]
    complete = np.isfinite(all_inputs).all(axis=1)
    errors, relative_errors, groups, checked, summaries = [], [], 0, 0, []
    for horizon in protocol["horizons"]:
        target = reconstruct_target(features, int(horizon))
        eligible = features.index[
            complete & target["y"].notna() & (target["y"] > 0.0)
            & target["target_end"].notna() & (target["target_end"] <= latest)
            & (features.index >= start) & (features.index <= stop)
        ]
        expected_origins = []
        independent = {name: [] for name in ("baseline", *CANDIDATES)}
        observed_y = []
        for _, month in pd.Series(eligible, index=eligible).groupby(eligible.to_period("M")):
            origins = pd.DatetimeIndex(month.to_numpy())
            fit = origins[0]
            train_mask = (
                complete & (features.index < fit) & (target["target_end"] <= fit)
                & target["y"].notna() & (target["y"] > 0.0)
            )
            training = features.index[train_mask]
            if len(training) < int(protocol["minimum_training_rows"]):
                continue
            if (features.loc[training, list(CANDIDATES)].std(ddof=0) <= 1e-12).any():
                raise AssertionError(f"{horizon}/{fit}: a candidate has zero training scale")
            expected_origins.extend(origins)
            observed_y.extend(target.loc[origins, "y"].to_numpy())
            y_train = np.log(target.loc[training, "y"].to_numpy())
            for model in ("baseline", *CANDIDATES):
                block = saved.loc[(saved["horizon"] == horizon) & (saved["model"] == model)]
                block = block.set_index("origin").sort_index().reindex(origins)
                if len(block) != len(origins) or block["prediction"].isna().any():
                    raise AssertionError(f"{horizon}/{model}/{fit}: expected origins absent")
                if not ((block["fit_origin"] == fit).all()
                        and (block["train_n"] == len(training)).all()
                        and (block["train_last_target"] == target.loc[training, "target_end"].max()).all()
                        and (block["target_end"] == target.loc[origins, "target_end"]).all()):
                    raise AssertionError(f"{horizon}/{model}/{fit}: training/target timestamps disagree")
                columns = list(BASELINE) + ([] if model == "baseline" else [model])
                design = features.loc[training, columns].to_numpy(dtype=float)
                beta = np.linalg.lstsq(design, y_train, rcond=None)[0]
                errors_train = y_train - design @ beta
                estimate = np.exp(features.loc[origins, columns].to_numpy(dtype=float) @ beta)
                estimate *= np.exp(errors_train).mean()
                actual = block["prediction"].to_numpy()
                _same_number(actual, estimate, f"{horizon}/{model}/{fit} predictions")
                _same_number(block["y"].to_numpy(), target.loc[origins, "y"].to_numpy(), "target values")
                errors.append(float(np.max(np.abs(actual - estimate))))
                relative_errors.append(float(np.max(np.abs(actual - estimate) / estimate)))
                independent[model].extend(estimate)
                checked += len(origins)
                groups += 1
        expected = pd.DatetimeIndex(expected_origins).sort_values()
        for model in ("baseline", *CANDIDATES):
            actual = pd.DatetimeIndex(saved.loc[
                (saved["horizon"] == horizon) & (saved["model"] == model), "origin"
            ]).sort_values()
            if not actual.equals(expected):
                raise AssertionError(f"{horizon}/{model}: full common-origin set differs")
        if expected.empty:
            raise AssertionError(f"horizon {horizon}: no independently evaluable origins")
        losses = {}
        for model, prediction in independent.items():
            ratio = np.asarray(observed_y) / np.asarray(prediction)
            losses[model] = ratio - np.log(ratio) - 1.0
        for model in CANDIDATES:
            base, challenger = losses["baseline"], losses[model]
            gap = challenger - base
            periods = []
            for lower, upper in protocol["inference"]["stability_periods"]:
                mask = (expected >= pd.Timestamp(lower)) & (expected <= pd.Timestamp(upper))
                periods.append(float(gap[mask].mean()) if mask.any() else None)
            summaries.append({
                "horizon": int(horizon), "candidate": model, "n": len(expected),
                "baseline_qlike": float(base.mean()), "candidate_qlike": float(challenger.mean()),
                "gap": float(gap.mean()), "improvement_pct": float(-100.0 * gap.mean() / base.mean()),
                "win_rate": float((gap < 0.0).mean()), "period_gaps": periods,
            })
    if checked != len(saved):
        raise AssertionError("not every stored prediction was independently checked")
    return {
        "independent_forecast_verification": "PASS", "n": checked,
        "monthly_model_fits": groups, "max_absolute_error": max(errors),
        "max_relative_error": max(relative_errors), "comparisons": summaries,
    }


def explicit_bootstrap_means(values: np.ndarray, block: int, draws: int, seed: int) -> np.ndarray:
    """Resample actual row indices, independently of cumulative block sums."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    n = len(array)
    if not 1 <= block <= n or draws < 1 or not np.isfinite(array).all():
        raise AssertionError("invalid independent bootstrap inputs")
    blocks, remainder = divmod(n, block)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(draws, blocks))
    tails = rng.integers(0, n, size=draws) if remainder else None
    means = np.empty((draws, array.shape[1]))
    for lower in range(0, draws, 128):
        upper = min(lower + 128, draws)
        indices = (
            starts[lower:upper, :, None] + np.arange(block)[None, None, :]
        ).reshape(upper - lower, blocks * block) % n
        if remainder:
            tail_indices = (tails[lower:upper, None] + np.arange(remainder)) % n
            indices = np.concatenate([indices, tail_indices], axis=1)
        if indices.shape[1] != n:
            raise AssertionError("bootstrap resample does not have exactly n observations")
        means[lower:upper] = array[indices].mean(axis=1)
    return means


def independent_hac(values: np.ndarray) -> dict:
    """Use statsmodels' covariance implementation rather than producer sums."""
    values = np.asarray(values, dtype=float)
    fitted = sm.OLS(values, np.ones((len(values), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": min(126, len(values) - 1), "use_correction": False},
        use_t=False,
    )
    mean, se = float(values.mean()), float(fitted.bse[0])
    p_value = float(fitted.pvalues[0]) if se > 0.0 else float(mean == 0.0)
    return {
        "se": se, "p": p_value, "ci95": [mean - 1.96 * se, mean + 1.96 * se],
        "mde80_nominal": float((stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * se),
    }


def verify_metrics(
    forecasts: pd.DataFrame, features: pd.DataFrame, point_audit: dict,
    recorded: dict, protocol: dict,
) -> dict:
    rows = recorded["rows"]
    expected_keys = {(int(h), c) for h in protocol["horizons"] for c in CANDIDATES}
    keyed = {(int(row["horizon"]), row["candidate"]): row for row in rows}
    if set(keyed) != expected_keys or len(rows) != len(expected_keys):
        raise AssertionError("metrics do not contain exactly all eight fixed comparisons")
    points = {(row["horizon"], row["candidate"]): row for row in point_audit["comparisons"]}
    inferred = {}
    for horizon in protocol["horizons"]:
        panel = forecasts.loc[forecasts["horizon"] == horizon].copy()
        panel["origin"] = pd.to_datetime(panel["origin"])
        wide = panel.pivot(index="origin", columns="model", values="prediction").sort_index()
        actual = panel.drop_duplicates("origin").set_index("origin")["y"].reindex(wide.index).to_numpy()

        def loss(prediction, realized=actual):
            ratio = realized / np.asarray(prediction)
            return ratio - np.log(ratio) - 1.0

        baseline = loss(wide["baseline"])
        differences = np.column_stack([loss(wide[c]) - baseline for c in CANDIDATES])
        means = differences.mean(axis=0)
        independent_blocks = {}
        for block in protocol["inference"]["blocks"]:
            draws = explicit_bootstrap_means(
                differences, int(block), int(protocol["inference"]["bootstrap_draws"]),
                int(protocol["inference"]["seed"]) + int(horizon) * 1000 + int(block),
            )
            independent_blocks[str(block)] = {
                "ci95": np.quantile(draws, [0.025, 0.975], axis=0),
                "p": (1.0 + (np.abs(draws - means) >= np.abs(means)).sum(axis=0)) / (len(draws) + 1),
            }
        for column, candidate in enumerate(CANDIDATES):
            key = (int(horizon), candidate)
            row, point = keyed[key], points[key]
            difference = differences[:, column]
            for name in ("n", "baseline_qlike", "candidate_qlike", "improvement_pct", "win_rate"):
                _same_number(row[name], point[name], f"{key}/{name}")
            _same_number(row["delta"], point["gap"], f"{key}/delta")
            if (row["first_origin"] != str(wide.index.min().date())
                    or row["last_origin"] != str(wide.index.max().date())):
                raise AssertionError(f"{key}: reported origin bounds differ")
            hac = independent_hac(difference)
            for name, value in hac.items():
                _same_number(row["hac126"][name], value, f"{key}/HAC/{name}", rtol=1e-10)
            if set(row["block_inference"]) != set(independent_blocks):
                raise AssertionError(f"{key}: bootstrap block set differs")
            p_values, lower_limits, upper_limits = [hac["p"]], [hac["ci95"][0]], [hac["ci95"][1]]
            for block, values in independent_blocks.items():
                p_value, interval = float(values["p"][column]), values["ci95"][:, column]
                _same_number(row["block_inference"][block]["p"], p_value, f"{key}/bootstrap/{block}/p",
                             rtol=0.0, atol=1e-14)
                _same_number(row["block_inference"][block]["ci95"], interval,
                             f"{key}/bootstrap/{block}/interval", rtol=1e-10)
                p_values.append(p_value)
                lower_limits.append(float(interval[0]))
                upper_limits.append(float(interval[1]))
            conservative = max(p_values)
            _same_number(row["p_conservative"], conservative, f"{key}/conservative p", rtol=1e-10)
            _same_number(row["ci95_envelope"], [min(lower_limits), max(upper_limits)], f"{key}/envelope")
            if len(row["periods"]) != len(protocol["inference"]["stability_periods"]):
                raise AssertionError(f"{key}: stability period count differs")
            for index, ((lower, upper), period) in enumerate(zip(
                protocol["inference"]["stability_periods"], row["periods"], strict=True
            )):
                mask = (wide.index >= pd.Timestamp(lower)) & (wide.index <= pd.Timestamp(upper))
                if period["start"] != lower or period["end"] != upper or period["n"] != int(mask.sum()):
                    raise AssertionError(f"{key}: fixed stability period changed")
                _same_number(period["delta"], point["period_gaps"][index], f"{key}/period {index}")
            annual = {int(item["year"]): item for item in row["annual"]}
            if set(annual) != set(wide.index.year) or len(annual) != len(row["annual"]):
                raise AssertionError(f"{key}: annual slice set differs")
            for year, item in annual.items():
                mask = wide.index.year == year
                if item["n"] != int(mask.sum()):
                    raise AssertionError(f"{key}: annual count differs")
                _same_number(item["delta"], difference[mask].mean(), f"{key}/year {year}")
            positions = features.index.get_indexer(wide.index) % int(horizon)
            if len(row["nonoverlap_phases"]) != int(horizon):
                raise AssertionError(f"{key}: phase count differs")
            for phase, item in enumerate(row["nonoverlap_phases"]):
                mask = positions == phase
                if item["phase"] != phase or item["n"] != int(mask.sum()):
                    raise AssertionError(f"{key}: phase membership differs")
                _same_number(item["delta"], difference[mask].mean(), f"{key}/phase {phase}")
            inferred[key] = conservative
    keys = list(keyed)
    corrected = multipletests([inferred[key] for key in keys], alpha=0.05, method="holm")[1]
    verdicts = []
    for key, adjusted in zip(keys, corrected, strict=True):
        row = keyed[key]
        _same_number(row["p_holm"], adjusted, f"{key}/Holm", rtol=1e-10)
        verdict = "EXPLORATORY_SHORTLIST" if (
            adjusted < 0.05 and points[key]["improvement_pct"] >= 1.0
            and all(delta is not None and delta < 0.0 for delta in points[key]["period_gaps"])
        ) else "INCONCLUSIVE"
        if row["verdict"] != verdict:
            raise AssertionError(f"{key}: verdict differs from the fixed joint gate")
        verdicts.append({"horizon": key[0], "candidate": key[1], "p_holm": float(adjusted), "verdict": verdict})
    return {"independent_inference_verification": "PASS", "verified_verdicts": verdicts,
            "bootstrap_method": "explicit circular row indices, all 4999 draws for all block lengths and hypotheses",
            "hac_method": "statsmodels Bartlett HAC(126), no small-sample correction",
            "holm_method": "statsmodels multipletests across all eight comparisons"}


def verify_manifest(root: Path, protocol: dict, manifest: dict) -> dict:
    cutoff = pd.Timestamp(protocol["latest_target"])
    identities = {}
    for key, path in zip(("daily", "cross", "vxn", "short", "vvix"), protocol["sources"], strict=True):
        if path.endswith(".parquet"):
            frame = pd.read_parquet(root / path, filters=[("date", "<=", cutoff)])
        else:
            frame = pd.read_csv(root / path, usecols=["DATE", "VVIX"])
            frame["DATE"] = pd.to_datetime(frame["DATE"], format="%m/%d/%Y")
            frame = frame.set_index("DATE").loc[:cutoff]
        identities[key] = {
            "path": path, "rows": len(frame), "start": str(frame.index.min().date()),
            "end": str(frame.index.max().date()),
            "bounded_content_sha256": hashlib.sha256(frame.to_csv(float_format="%.17g").encode()).hexdigest(),
        }
    if identities != manifest["sources"]:
        raise AssertionError("bounded source identity differs from producer manifest")
    for key, path in (
        ("protocol_sha256", root / "orthogonal_round2.yaml"),
        ("producer_sha256", root / "src/orthogonal_round2.py"),
        ("tests_sha256", root / "tests/test_orthogonal_round2.py"),
    ):
        if manifest[key] != file_digest(path):
            raise AssertionError(f"manifest {key} differs from current file")
    for path, expected in manifest["existing_artifacts_sha256"].items():
        if file_digest(root / path) != expected:
            raise AssertionError(f"previously frozen artifact changed: {path}")
    calibration = manifest["calibration"]
    if (calibration["status"] != "PASS" or calibration["envelope_coverage"] < 0.90
            or calibration["protocol_sha256"] != manifest["protocol_sha256"]
            or calibration["producer_sha256"] != manifest["producer_sha256"]):
        raise AssertionError("pre-run calibration identity or gate differs")
    return {"independent_manifest_verification": "PASS", "bounded_sources": identities,
            "preserved_artifact_count": len(manifest["existing_artifacts_sha256"])}


def verify(root: Path = ROOT) -> dict:
    protocol_path = root / "orthogonal_round2.yaml"
    protocol = yaml.safe_load(protocol_path.read_text())
    if tuple(protocol["baseline"]) != BASELINE or tuple(protocol["candidates"]) != CANDIDATES:
        raise AssertionError("protocol feature/model specification differs from verifier")
    data_dir = root / protocol["outputs"]["local_data"]
    report_dir = (root / protocol["outputs"]["report"]).parent
    manifest = json.loads((report_dir / "manifest.json").read_text())
    identities = verify_manifest(root, protocol, manifest)
    features = reconstruct_features(root, protocol)
    cached = pd.read_parquet(data_dir / "features.parquet")
    if not cached.index.equals(features.index):
        raise AssertionError("cached feature dates differ from bounded raw QQQ dates")
    for column in features.columns:
        if column not in cached:
            raise AssertionError(f"feature cache lacks {column}")
        _same_number(cached[column].to_numpy(), features[column].to_numpy(), f"feature {column}",
                     rtol=1e-10, atol=1e-12)
    forecasts = pd.read_parquet(data_dir / "forecasts.parquet")
    result = verify_forecasts(features, forecasts, protocol)
    recorded = json.loads((report_dir / "metrics.json").read_text())
    for key in ("protocol_sha256", "producer_sha256"):
        if recorded[key] != manifest[key]:
            raise AssertionError(f"metrics {key} differs from manifest")
    result.update(verify_metrics(forecasts, features, result, recorded, protocol))
    result.update(identities)
    result.update({
        "independent_feature_verification": "PASS", "feature_rows": len(features),
        "protocol_sha256": file_digest(protocol_path),
        "source_sha256": {path: file_digest(root / path) for path in protocol["sources"]},
        "verifier_sha256": file_digest(Path(__file__)),
        "limitation": "The finite AR(1) calibration is hash-bound but not independently rerun; market-data inference remains exploratory.",
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = verify(args.root)
    protocol = yaml.safe_load((args.root / "orthogonal_round2.yaml").read_text())
    destination = args.root / protocol["outputs"]["audit"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
