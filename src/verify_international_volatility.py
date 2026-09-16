"""Independent raw-source, timing, geometry and inference audit for wave two."""
from __future__ import annotations

import json
import pathlib
import zipfile

import numpy as np
import pandas as pd
import yaml

from .verify_iterative_signal_search import (
    HF_BASE,
    digest,
    expected_origins,
    hf_features,
    hf_target,
    holm,
    local_archive_dates,
    ols_prediction,
    same,
    training_eligibility,
)
from .verify_model_memory_study import explicit_bootstrap_means, independent_hac

ROOT = pathlib.Path(__file__).resolve().parents[1]
SYMBOLS = (".N225", ".HSI", ".KS11", ".FTSE", ".GDAXI", ".FCHI")
WINDOWS = (1, 5, 22)
FOREIGN_COLUMNS = tuple(f"{symbol[1:].lower()}_{suffix}" for symbol in SYMBOLS for suffix in ("d", "w", "m"))
REGION_COLUMNS = ("asia_d", "asia_w", "asia_m", "europe_d", "europe_w", "europe_m")


def foreign_design(foreign, sessions, max_age=3):
    sessions = pd.DatetimeIndex(sessions)
    if sessions.has_duplicates or not sessions.is_monotonic_increasing or sessions.tz is not None:
        raise ValueError("Unique local US session dates required")
    values = pd.DataFrame(index=sessions)
    audit = {}
    for symbol in SYMBOLS:
        raw = foreign[symbol]
        if raw.index.has_duplicates or not raw.index.is_monotonic_increasing or raw.index.tz is not None:
            raise ValueError("Unique ordered foreign local session labels required")
        usable = raw.where(np.isfinite(raw) & (raw > 0))
        local = np.column_stack([np.log(usable.rolling(window, min_periods=window).mean()) for window in WINDOWS])
        selected = np.full((len(sessions), 3), np.nan)
        source_dates = np.full(len(sessions), np.datetime64("NaT"), dtype="datetime64[ns]")
        cutoff_dates = np.full(len(sessions), np.datetime64("NaT"), dtype="datetime64[ns]")
        ages = np.full(len(sessions), np.nan)
        complete = np.zeros(len(sessions), dtype=bool)
        for index in range(1, len(sessions)):
            cutoff = sessions[index - 1]
            cutoff_dates[index] = cutoff.to_datetime64()
            source_index = raw.index.searchsorted(cutoff, side="right") - 1
            if source_index < 0:
                continue
            source = raw.index[source_index]
            age = sessions.searchsorted(cutoff, side="right") - sessions.searchsorted(source, side="right")
            source_dates[index] = source.to_datetime64()
            ages[index] = age
            complete[index] = np.isfinite(local[source_index]).all()
            if age <= max_age:
                selected[index] = local[source_index]
        for column, window in enumerate(WINDOWS):
            values[f"{symbol}_{window}"] = selected[:, column]
        audit[symbol] = pd.DataFrame({"source_date": source_dates, "cutoff_date": cutoff_dates,
                                      "extra_age": ages, "complete_window": complete}, index=sessions)
    return values, audit


def regional_design(foreign):
    result = pd.DataFrame(index=foreign.index)
    for region, symbols in (("asia", SYMBOLS[:3]), ("europe", SYMBOLS[3:])):
        for window in WINDOWS:
            result[f"{region}_{window}"] = foreign[[f"{symbol}_{window}" for symbol in symbols]].sum(axis=1, min_count=3) / 3
    return result


def pca_design(train, query):
    train, query = np.asarray(train, float), np.asarray(query, float)
    if train.ndim != 2 or train.shape[1] != 18 or query.ndim != 2 or query.shape[1] != 18 or not np.isfinite(train).all() or not np.isfinite(query).all():
        raise ValueError("Finite eighteen-dimensional PCA inputs required")
    means, scales = train.mean(axis=0), train.std(axis=0)
    if (scales <= 1e-12).any():
        raise ValueError("Zero-scale international feature")
    standardized = (train - means) / scales
    # The producer uses SVD. A separate symmetric eigendecomposition reconstructs
    # the same top-three subspace without importing its transform implementation.
    eigenvalues, vectors = np.linalg.eigh(standardized.T @ standardized)
    eigenvalues, vectors = eigenvalues[::-1], vectors[:, ::-1]
    if eigenvalues[2] - eigenvalues[3] <= 1e-8 * eigenvalues[0]:
        raise ValueError("PCA boundary eigengap is too small")
    components = vectors[:, :3].T.copy()
    for component in components:
        if component[np.argmax(np.abs(component))] < 0:
            component *= -1
    audit = {"means": means, "scales": scales, "components": components,
             "squared_singular_values": eigenvalues, "eigen_gap": eigenvalues[2] - eigenvalues[3],
             "gap_threshold": 1e-8 * eigenvalues[0]}
    return standardized @ components.T, ((query - means) / scales) @ components.T, audit


def reconstruct(root, protocol):
    source = protocol["sources"]
    end = pd.Timestamp(source["source_end"])
    archive_path = root / source["archive"]
    if digest(archive_path) != source["archive_sha256"]:
        raise AssertionError("Oxford archive hash differs")
    with zipfile.ZipFile(archive_path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(names) != 1:
            raise AssertionError("Ambiguous archived CSV")
        with archive.open(names[0]) as stream:
            raw = pd.read_csv(stream)
    markets = {}
    for symbol in (".SPX", *SYMBOLS):
        frame = raw.loc[raw.Symbol.eq(symbol)].copy()
        frame.index = local_archive_dates(frame.iloc[:, 0])
        frame.index.name = "date"
        if frame.index.has_duplicates:
            raise AssertionError("Duplicated local archive date")
        markets[symbol] = frame.sort_index().loc[:end]
    prior = pd.read_parquet(root / source["spx_extension"], filters=[("date", "<=", end)])
    existing = pd.read_parquet(root / source["spx_existing"], filters=[("date", "<=", end)])
    overlap = prior.index.intersection(existing.index)
    if len(overlap) < 250:
        raise AssertionError("Insufficient independent SPX source overlap")
    if not prior.loc[overlap.min():overlap.max()].index.equals(existing.loc[overlap.min():overlap.max()].index):
        raise AssertionError("SPX source overlap calendar differs")
    same(prior.loc[overlap, ["open", "high", "low", "close"]], existing.loc[overlap, ["open", "high", "low", "close"]],
         "Independent SPX overlap", rtol=1e-6, atol=1e-4)
    daily = pd.concat([prior.loc[prior.index < "2009-01-01", existing.columns], existing])
    if daily.index.has_duplicates or not daily.index.is_monotonic_increasing:
        raise AssertionError("Invalid merged SPX calendar")
    vix = pd.read_parquet(root / source["cboe"], filters=[("date", "<=", end)]).vix
    domestic = hf_features(daily, markets[".SPX"], vix).loc[:, HF_BASE]
    foreign, trace = foreign_design({symbol: markets[symbol].rv5.astype(float) for symbol in SYMBOLS}, daily.index,
                                     protocol["international"]["max_extra_us_sessions"])
    regional = regional_design(foreign)
    aliases = {f"{symbol}_{window}": f"{symbol[1:].lower()}_{suffix}" for symbol in SYMBOLS for window, suffix in zip(WINDOWS, ("d", "w", "m"), strict=True)}
    region_aliases = {f"{region}_{window}": f"{region}_{suffix}" for region in ("asia", "europe") for window, suffix in zip(WINDOWS, ("d", "w", "m"), strict=True)}
    foreign = foreign.rename(columns=aliases)
    regional = regional.rename(columns=region_aliases)
    features = pd.concat([domestic, foreign], axis=1)
    target_frames = {h: hf_target(markets[".SPX"].rv5.astype(float), daily.index, h) for h in (1, 5, 21)}
    return features, regional, target_frames, trace, markets


def verify_forecasts(features, regional, target_frames, forecasts, fits, protocol):
    section = protocol["hf"]
    if not {"origin", "horizon", "model", "y", "prediction", "available_date"}.issubset(forecasts):
        raise AssertionError("Explicit international forecast schema required")
    if set(forecasts.model) != {"baseline", "regional", "latent"} or forecasts.duplicated(["origin", "horizon", "model"]).any():
        raise AssertionError("Forecast model set or uniqueness differs")
    origins, complete = expected_origins("hf", features, target_frames, section)
    mapped = {(row["horizon"], pd.Timestamp(row["fit_origin"])): row for row in fits}
    used, count = set(), 0
    for horizon, targets in target_frames.items():
        for model in ("baseline", "regional", "latent"):
            rows = forecasts.loc[(forecasts.model == model) & (forecasts.horizon == horizon)].sort_values("origin")
            if not pd.DatetimeIndex(rows.origin).equals(origins):
                raise AssertionError("All arms and horizons must share complete eligible origins")
            same(rows.y, targets.loc[origins, "y"], "International shared target", rtol=1e-12, atol=0)
            for date_column in ("target_end", "available_date"):
                if not np.array_equal(rows[date_column].to_numpy(), targets.loc[origins, date_column].to_numpy()):
                    raise AssertionError("International target publication dates differ")
        for month in origins.to_period("M").unique():
            query = origins[origins.to_period("M") == month]
            origin = query[0]
            key = (horizon, origin)
            used.add(key)
            if key not in mapped:
                raise AssertionError("Missing international monthly fit")
            record = mapped[key]
            mask = training_eligibility(complete, targets, origin, availability=True)
            n = int(mask.sum())
            if n < section["minimum_train"] or record["train_n"] != n:
                raise AssertionError("International training sample differs")
            last_target = targets.loc[mask, "target_end"].max()
            last_available = targets.loc[mask, "available_date"].max()
            if pd.Timestamp(record["train_last_target"]) != last_target or pd.Timestamp(record["train_last_available"]) != last_available or last_available > origin:
                raise AssertionError("International training label was not available")
            raw_train, raw_apply = features.loc[mask, FOREIGN_COLUMNS].to_numpy(float), features.loc[query, FOREIGN_COLUMNS].to_numpy(float)
            pc_train, pc_apply, independent_pca = pca_design(raw_train, raw_apply)
            recorded_pca = record["pca_audit"]
            if tuple(recorded_pca["columns"]) != FOREIGN_COLUMNS:
                raise AssertionError("PCA feature order differs")
            same(recorded_pca["means"], independent_pca["means"], "PCA train mean")
            same(recorded_pca["scales"], independent_pca["scales"], "PCA train scale")
            components = np.asarray(recorded_pca["components"], float)
            if components.shape != (3, 18):
                raise AssertionError("PCA width differs")
            same(components @ components.T, np.eye(3), "PCA orthonormality", rtol=1e-10, atol=1e-10)
            same(components.T @ components, independent_pca["components"].T @ independent_pca["components"],
                 "Independent top-three PCA projector", rtol=1e-7, atol=1e-9)
            for component in components:
                if component[np.argmax(np.abs(component))] <= 0:
                    raise AssertionError("PCA sign convention differs")
            squared = np.asarray(recorded_pca["singular_values"]) ** 2
            same(squared, independent_pca["squared_singular_values"], "PCA singular spectrum", rtol=1e-7, atol=1e-8)
            same(recorded_pca["eigen_gap"], independent_pca["eigen_gap"], "PCA boundary eigengap")
            same(recorded_pca["gap_threshold"], independent_pca["gap_threshold"], "PCA gap threshold")
            if recorded_pca["eigen_gap"] <= recorded_pca["gap_threshold"]:
                raise AssertionError("PCA boundary is not identified")
            # Refit in the stored, independently verified eigenspace basis to
            # check coordinate-dependent coefficients as well as predictions.
            saved_pc_train = ((raw_train - independent_pca["means"]) / independent_pca["scales"]) @ components.T
            saved_pc_apply = ((raw_apply - independent_pca["means"]) / independent_pca["scales"]) @ components.T
            base_train, base_apply = features.loc[mask, HF_BASE].to_numpy(float), features.loc[query, HF_BASE].to_numpy(float)
            y = targets.loc[mask, "y"].to_numpy(float)
            for model in ("baseline", "regional", "latent"):
                columns = HF_BASE
                train, apply = base_train, base_apply
                if model == "regional":
                    columns += REGION_COLUMNS
                    train = np.column_stack([base_train, regional.loc[mask, REGION_COLUMNS]])
                    apply = np.column_stack([base_apply, regional.loc[query, REGION_COLUMNS]])
                if model == "latent":
                    columns += ("pc1", "pc2", "pc3")
                    train = np.column_stack([base_train, saved_pc_train])
                    apply = np.column_stack([base_apply, saved_pc_apply])
                prediction, audit = ols_prediction(train, y, apply)
                recorded = record["model_audit"][model]
                if tuple(recorded["columns"]) != columns or recorded["rank"] != audit["rank"]:
                    raise AssertionError("International OLS model design differs")
                same(recorded["means"], np.r_[0., audit["means"]], "International OLS centers")
                same(recorded["scales"], np.r_[1., audit["scales"]], "International OLS scales")
                same(recorded["beta"], audit["beta"], "International OLS coefficients")
                same(recorded["smear"], audit["smear"], "International exact Duan smear")
                block = forecasts.loc[(forecasts.model == model) & (forecasts.horizon == horizon) & forecasts.origin.isin(query)].sort_values("origin")
                if not block.fit_origin.eq(origin).all() or not block.train_n.eq(n).all() or not block.train_last_target.eq(last_target).all() or not block.train_last_available.eq(last_available).all():
                    raise AssertionError("International row-level fit provenance differs")
                same(block.prediction, prediction, "Every independently reconstructed international forecast", rtol=1e-8, atol=1e-12)
                if model == "latent":
                    alternate, _ = ols_prediction(np.column_stack([base_train, pc_train]), y, np.column_stack([base_apply, pc_apply]))
                    same(prediction, alternate, "Predictions invariant to independently reconstructed PCA basis", rtol=1e-8, atol=1e-12)
                count += len(block)
    if set(mapped) != used or len(mapped) != len(fits):
        raise AssertionError("Unexpected international fit records")
    return {"forecasts_verified": count, "monthly_fits_verified": len(used),
            "common_origins": len(origins), "pca_projectors_verified": len(used)}


def verify_availability(recorded, trace):
    if recorded.duplicated(["origin", "symbol"]).any() or set(recorded.symbol) != set(SYMBOLS):
        raise AssertionError("Missing or duplicated foreign source provenance")
    count = 0
    for symbol in SYMBOLS:
        actual = recorded.loc[recorded.symbol == symbol].sort_values("origin").set_index("origin")
        expected = trace[symbol]
        if not actual.index.equals(expected.index):
            raise AssertionError("Foreign source audit omitted a US origin")
        for date_column in ("source_date", "cutoff_date"):
            if not np.array_equal(actual[date_column].to_numpy(), expected[date_column].to_numpy(), equal_nan=True):
                raise AssertionError("Foreign backward-asof source date differs")
        same(actual.extra_us_sessions, expected.extra_age, "Foreign source age", rtol=0, atol=0)
        fresh = expected.extra_age.notna() & (expected.extra_age <= 3)
        if not np.array_equal(actual.fresh.to_numpy(), fresh.to_numpy()) or not np.array_equal(actual.complete_window.to_numpy(), expected.complete_window.to_numpy()):
            raise AssertionError("Foreign source freshness or complete-window status differs")
        count += len(actual)
    return count


def same_tree(actual, expected, label):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError(f"{label}: schema differs")
        for key in expected:
            same_tree(actual[key], expected[key], f"{label}/{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f"{label}: list count differs")
        for index, value in enumerate(expected):
            same_tree(actual[index], value, f"{label}/{index}")
    elif expected is None or isinstance(expected, (str, bool, int)):
        if actual != expected:
            raise AssertionError(f"{label}: value differs")
    else:
        same(actual, expected, label, rtol=2e-8, atol=1e-14)


def phase_statistics(panel, horizon, candidate, phase, phase_code, calendar, protocol):
    section = protocol["hf"]
    dates_start, dates_end = map(pd.Timestamp, section[phase])
    block = panel.loc[(panel.horizon == horizon) & (panel.origin >= dates_start) & (panel.origin <= dates_end)].copy()
    if phase == "development":
        block = block.loc[block.available_date <= pd.Timestamp(section["development_target_available_by"])]
    wide = block.pivot(index="origin", columns="model", values="prediction").sort_index()
    base = block.loc[block.model == "baseline"].set_index("origin").reindex(wide.index)
    y = base.y.to_numpy(float)
    ratio_b, ratio_c = y / wide.baseline.to_numpy(), y / wide[candidate].to_numpy()
    loss_b = ratio_b - np.log(ratio_b) - 1
    loss_c = ratio_c - np.log(ratio_c) - 1
    difference = loss_c - loss_b
    mean = float(difference.mean())
    hac = independent_hac(difference)
    settings = protocol["inference"]
    blocks = {}
    for width in settings["blocks"]:
        seed = settings["seed"] + horizon * 1000 + phase_code * 10000 + width
        samples = explicit_bootstrap_means(difference, width, settings["bootstrap_draws"], seed)[:, 0]
        p_value = float((1 + np.count_nonzero(np.abs(samples - mean) >= abs(mean))) / (len(samples) + 1))
        blocks[str(width)] = {"p": p_value, "ci95": np.quantile(samples, [.025, .975]).tolist()}
    lows = [hac["ci95"][0], *(item["ci95"][0] for item in blocks.values())]
    highs = [hac["ci95"][1], *(item["ci95"][1] for item in blocks.values())]
    result = {"name": phase, "first_origin": str(wide.index[0].date()), "last_origin": str(wide.index[-1].date()),
              "n": len(wide), "delta": mean, "control_loss": float(loss_b.mean()),
              "candidate_loss": float(loss_c.mean()), "improvement_pct": float(-100 * mean / loss_b.mean()),
              "block_inference": blocks, "hac126": hac, "ci95_envelope": [min(lows), max(highs)],
              "p_conservative": max(hac["p"], *(item["p"] for item in blocks.values())),
              "stability": [], "annual": [], "nonoverlap_phases": []}
    for year in sorted(set(wide.index.year)):
        mask = wide.index.year == year
        result["annual"].append({"year": int(year), "n": int(mask.sum()), "delta": float(difference[mask].mean())})
    positions = pd.DatetimeIndex(calendar).get_indexer(wide.index)
    if (positions < 0).any():
        raise AssertionError("International origin missing from actual session calendar")
    for index in range(horizon):
        mask = positions % horizon == index
        result["nonoverlap_phases"].append({"phase": index, "n": int(mask.sum()),
                                            "delta": float(difference[mask].mean()) if mask.any() else None})
    if phase == "evaluation":
        for start, finish in section["evaluation_stability"]:
            mask = (wide.index >= start) & (wide.index <= finish)
            result["stability"].append({"start": start, "end": finish, "n": int(mask.sum()), "delta": float(difference[mask].mean())})
    return result


def verify_metrics(root, forecasts, calendar, protocol, recorded):
    rows = recorded["rows"]
    expected_keys = {(horizon, candidate) for horizon in (1, 5, 21) for candidate in ("regional", "latent")}
    keys = [(row["horizon"], row["candidate"]) for row in rows]
    if len(keys) != 6 or len(set(keys)) != 6 or set(keys) != expected_keys:
        raise AssertionError("All six international contrasts are required")
    p_new, qualifying = [], []
    for row in rows:
        if row["study"] != "international" or row["control"] != "baseline":
            raise AssertionError("International hypothesis identity differs")
        expected = [phase_statistics(forecasts, row["horizon"], row["candidate"], name, code, calendar, protocol)
                    for code, name in enumerate(("development", "evaluation"))]
        same_tree(row["phases"], expected, "Independent phase inference and diagnostics")
        p_value = max(phase["p_conservative"] for phase in expected)
        same(row["p_conservative"], p_value, "International two-phase conjunction p")
        p_new.append(p_value)
        qualifying.append(all(phase["improvement_pct"] >= 1. for phase in expected)
                          and all(slice_["delta"] < 0 for slice_ in expected[1]["stability"]))
    inherited = []
    for path in protocol["comparisons"]["inherited_sources"]:
        source_rows = json.loads((root / path).read_text())["rows"]
        for row in source_rows:
            inherited.append({"study": row.get("study", path.split("/")[1]), "horizon": row["horizon"],
                              "candidate": row["candidate"], "control": row.get("control", "baseline"),
                              "p_conservative": row["p_conservative"], "source": path,
                              "source_sha256": digest(root / path)})
    if len(inherited) != 66:
        raise AssertionError("Cumulative denominator must retain all 66 prior contrasts")
    same_tree(recorded["inherited_rows"], inherited, "All inherited international-search trials")
    p_wave = holm(p_new)
    p_cumulative = holm([row["p_conservative"] for row in inherited] + p_new)[-6:]
    leads = []
    for index, row in enumerate(rows):
        same(row["p_holm_wave"], p_wave[index], "Six-way wave Holm")
        same(row["p_holm_cumulative"], p_cumulative[index], "72-way cumulative Holm")
        passed = bool(qualifying[index] and p_wave[index] < .05 / 6 and p_cumulative[index] < .05)
        if row["verdict"] != ("EXPLORATORY_LEAD" if passed else "DOES_NOT_QUALIFY"):
            raise AssertionError("International lead gates differ")
        if passed:
            leads.append({"candidate": row["candidate"], "horizon": row["horizon"]})
    if recorded["leads"] != leads:
        raise AssertionError("International lead list differs")
    return {"phase_comparisons_verified": 12, "bootstrap_runs_verified": 36,
            "new_hypotheses_verified": 6, "cumulative_hypotheses_verified": 72, "leads": leads}


def verify(root=ROOT):
    root = pathlib.Path(root)
    report, output = root / "reports/international_volatility", root / "data/international_volatility"
    protocol_path = root / "international_volatility.yaml"
    protocol = yaml.safe_load(protocol_path.read_text())
    manifest = json.loads((report / "manifest.json").read_text())
    if manifest["protocol_sha256"] != digest(protocol_path):
        raise AssertionError("International protocol changed after registration")
    for group in ("code", "inputs", "preserved"):
        for path, expected in manifest[group].items():
            if digest(root / path) != expected:
                raise AssertionError(f"International frozen input or prior artifact changed: {path}")
    features, regional, targets, trace, _ = reconstruct(root, protocol)
    produced = pd.read_parquet(output / "features.parquet")
    if not produced.index.equals(features.index) or tuple(produced.columns) != tuple(features.columns):
        raise AssertionError("International feature rows or identities differ")
    same(produced, features, "Every international input feature", rtol=1e-11, atol=1e-13)
    availability_count = verify_availability(pd.read_parquet(output / "foreign_availability.parquet"), trace)
    forecast = pd.read_parquet(output / "forecasts.parquet")
    fits = json.loads((output / "fits.json").read_text())
    result = {"status": "VERIFIED", "protocol_sha256": digest(protocol_path), "verifier_sha256": digest(pathlib.Path(__file__)),
              "foreign_source_rows_verified": availability_count,
              "forecast_reconstruction": verify_forecasts(features, regional, targets, forecast, fits, protocol)}
    metrics = json.loads((report / "metrics.json").read_text())
    if metrics["protocol_sha256"] != digest(protocol_path):
        raise AssertionError("International metrics protocol differs")
    result["inference"] = verify_metrics(root, forecast, features.index, protocol, metrics)
    ledger = [json.loads(line) for line in (report / "trial_ledger.jsonl").read_text().splitlines() if line]
    for event, expected in (("inherited", metrics["inherited_rows"]), ("evaluated", metrics["rows"])):
        actual = [{key: value for key, value in row.items() if key != "event"} for row in ledger if row["event"] == event]
        same_tree(actual, expected, f"International {event} trial ledger")
    registered = [row for row in ledger if row["event"] == "registered"]
    if len(ledger) != 78 or len(registered) != 6:
        raise AssertionError("International trial ledger count differs")
    identities = {(row["candidate"], row["horizon"]) for row in registered}
    if identities != {(row["candidate"], row["horizon"]) for row in metrics["rows"]}:
        raise AssertionError("International registered trial identities differ")
    if any(row["protocol_sha256"] != digest(protocol_path) for row in registered):
        raise AssertionError("International ledger protocol differs")
    result["ledger_events_verified"] = {"inherited": 66, "registered": 6, "evaluated": 6}
    result["limitation"] = "Historical reuse and unknown publication vintages remain exploratory despite verified construction and cumulative correction."
    (report / "verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, allow_nan=False))
