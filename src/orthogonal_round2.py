"""Additive exploratory orthogonality tests; see orthogonal_round2.yaml.

No frozen producer or estimator is imported. Every empirical command gates on
pre-written synthetic tests and a code-hash-matched null calibration.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "orthogonal_round2.yaml"
OUT = ROOT / "data/orthogonal_round2"
REPORT = ROOT / "reports/orthogonal_round2"
BASE = ("const", "lrv_d", "lrv_w", "lrv_m", "lev_d", "lev_w", "lev_m",
        "liv", "lvix", "term", "xasset_stress", "market_stress")
CANDIDATES = ("vvix", "rv_dispersion", "stock_bond_corr", "close_pressure")
ALL_FEATURES = BASE + CANDIDATES


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def dump(path, value):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_protocol():
    p = yaml.safe_load(PROTOCOL.read_text())
    if (tuple(p["baseline"]) != BASE or tuple(p["candidates"]) != CANDIDATES
            or p["horizons"] != [1, 5] or p["inference"]["hypothesis_count"] != 8
            or p["origin_end"] != "2025-10-17"
            or p["latest_target"] != "2025-10-20"
            or p["sealed_start"] != "2025-11-03"):
        raise ValueError("Protocol differs from implemented contract")
    return p


def build_features(daily, cross, iv):
    if (not daily.index.is_monotonic_increasing or daily.index.has_duplicates
            or cross.index.has_duplicates or iv.index.has_duplicates):
        raise ValueError("Unique sorted session index required")
    d = daily.copy()
    prices = d[["open", "high", "low", "close", "adj close"]]
    if ((prices <= 0).any().any()
            or (d.high < d[["open", "close", "low"]].max(axis=1)).any()
            or (d.low > d[["open", "close", "high"]].min(axis=1)).any()):
        raise ValueError("Invalid OHLC prices")
    f = pd.DataFrame(index=d.index)
    intraday = (0.5 * np.log(d.high / d.low) ** 2
                - (2 * np.log(2) - 1) * np.log(d.close / d.open) ** 2).clip(lower=1e-10)
    overnight = np.log(d.open / d.close.shift(1)) ** 2
    f["rv_total"] = intraday + overnight
    f["log_rv"] = np.log(f.rv_total)
    f["const"] = 1.0
    f["lrv_d"] = f.log_rv
    for label, window in [("w", 5), ("m", 22)]:
        f[f"lrv_{label}"] = np.log(f.rv_total.rolling(window).mean())
    ret = np.log(d["adj close"]).diff()
    f["lev_d"] = ret.clip(upper=0)
    f["lev_w"] = ret.rolling(5).mean().clip(upper=0)
    f["lev_m"] = ret.rolling(22).mean().clip(upper=0)
    safe_iv = iv.reindex(d.index).where(lambda x: x > 0).shift(1)
    f["liv"] = np.log(safe_iv.vxn)
    f["lvix"] = np.log(safe_iv.vix)
    f["term"] = np.log(safe_iv.vix9d / safe_iv.vix)

    def prior_z(s):
        r = s.rolling(252, min_periods=126)
        return (s - r.mean().shift(1)) / r.std(ddof=1).shift(1).replace(0, np.nan)

    asset_ret = np.log(cross.reindex(d.index).where(lambda x: x > 0)).diff()
    z = prior_z(asset_ret.loc[:, ["hyg", "tlt", "gld", "uso", "uup"]])
    f["xasset_stress"] = np.sqrt(z.pow(2).mean(axis=1, skipna=False))
    zvol = prior_z(np.log(d.volume.where(d.volume > 0))).clip(lower=0)
    zovernight = prior_z((overnight / f.rv_total).clip(0, 1)).clip(lower=0)
    f["market_stress"] = np.sqrt((zvol ** 2 + zovernight ** 2) / 2)
    f["vvix"] = np.log(safe_iv.vvix)
    f["rv_dispersion"] = f.log_rv.rolling(22).std(ddof=1)
    f["stock_bond_corr"] = ret.rolling(22).corr(asset_ret.tlt)
    location = 2 * np.log(d.close / d.low) / np.log(d.high / d.low).replace(0, np.nan) - 1
    f["close_pressure"] = location.rolling(5).mean()
    return f.replace([np.inf, -np.inf], np.nan)


def make_targets(rv, h):
    if not isinstance(h, (int, np.integer)) or h < 1:
        raise ValueError("Positive integer horizon required")
    forward = pd.concat([rv.shift(-i) for i in range(1, h + 1)], axis=1)
    y = forward.mean(axis=1, skipna=False)
    y = y.where((forward > 0).all(axis=1))
    end = pd.Series(rv.index, index=rv.index).shift(-h)
    return pd.DataFrame({"y": y, "target_end": end})


def training_mask(features, targets, origin, min_train=500):
    mask = (np.isfinite(features.loc[:, ALL_FEATURES]).all(axis=1)
            & np.isfinite(targets.y) & (targets.y > 0)
            & (targets.target_end <= pd.Timestamp(origin))
            & (features.index < pd.Timestamp(origin)))
    if int(mask.sum()) < min_train:
        raise ValueError(f"INSUFFICIENT_DATA: {int(mask.sum())} training rows")
    if (features.loc[mask, CANDIDATES].std(ddof=1) <= 1e-12).any():
        raise ValueError("INSUFFICIENT_DATA: zero-scale candidate")
    return mask


def fit_predict(train_features, train_y, origin_features):
    x = train_features.loc[:, BASE].to_numpy(float)
    yraw = np.asarray(train_y, dtype=float)
    if not np.isfinite(yraw).all() or (yraw <= 0).any():
        raise ValueError("Positive finite training target required")
    y = np.log(yraw)
    if isinstance(origin_features, pd.Series):
        origin_features = origin_features.to_frame().T
        scalar = True
    else:
        scalar = False
    o = origin_features.loc[:, BASE].to_numpy(float)
    if not np.isfinite(x).all() or not np.isfinite(o).all():
        raise ValueError("Finite design required")

    def forecast(a, b):
        beta = np.linalg.lstsq(a, y, rcond=None)[0]
        pred = np.exp(b @ beta) * np.exp(y - a @ beta).mean()
        if not np.isfinite(pred).all() or (pred <= 0).any():
            raise ValueError("Nonfinite forecast; no clipping fallback")
        return float(pred[0]) if scalar else pred

    forecasts = {"baseline": forecast(x, o)}
    r2 = {}
    for col in CANDIDATES:
        z = train_features[col].to_numpy(float)
        zo = origin_features[col].to_numpy(float)
        if not np.isfinite(z).all() or not np.isfinite(zo).all():
            raise ValueError("Finite candidate required")
        gamma = np.linalg.lstsq(x, z, rcond=None)[0]
        residual = z - x @ gamma
        if np.std(z, ddof=1) <= 1e-12 or np.std(residual, ddof=1) <= 1e-12:
            raise ValueError("INSUFFICIENT_DATA: zero-scale or redundant candidate")
        r2[col] = float(1 - np.sum(residual ** 2) / np.sum((z - z.mean()) ** 2))
        forecasts[col] = forecast(np.column_stack([x, residual]),
                                  np.column_stack([o, zo - o @ gamma]))
    return {"forecasts": forecasts, "orthogonal_r2": r2}


def qlike(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    if (not np.isfinite(y).all() or not np.isfinite(pred).all()
            or (y <= 0).any() or (pred <= 0).any()):
        raise ValueError("QLIKE requires positive finite values")
    ratio = y / pred
    return ratio - np.log(ratio) - 1


def holm_adjust(p):
    p = np.asarray(p, float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid p-value")
    order = np.argsort(p)
    out = np.empty_like(p)
    out[order] = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    return out


def hac_summary(d, lags=126):
    d = np.asarray(d, float)
    n = len(d)
    centered = d - d.mean()
    lag = min(lags, n - 1)
    longvar = centered @ centered / n
    for k in range(1, lag + 1):
        longvar += 2 * (1 - k / (lag + 1)) * (centered[k:] @ centered[:-k]) / n
    se = float(np.sqrt(max(longvar, 0) / n))
    p = float(2 * norm.sf(abs(d.mean()) / se)) if se > 0 else float(d.mean() == 0)
    return {"se": se, "p": p, "ci95": [float(d.mean() - 1.96 * se),
                                             float(d.mean() + 1.96 * se)],
            "mde80_nominal": float((norm.ppf(.975) + norm.ppf(.8)) * se)}


def bootstrap_means(d, block, draws, seed):
    """Exactly n circularly resampled observations, shared draws across columns."""
    d = np.asarray(d, float)
    if d.ndim == 1:
        d = d[:, None]
    n = len(d)
    if not 1 <= block <= n:
        raise ValueError("Block outside sample")
    extended = np.concatenate([d, d[:block]], axis=0)
    sums = np.concatenate([np.zeros((1, d.shape[1])), np.cumsum(extended, axis=0)], axis=0)
    full = sums[np.arange(n) + block] - sums[np.arange(n)]
    count, remainder = divmod(n, block)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(draws, count))
    totals = full[starts].sum(axis=1)
    if remainder:
        starts_tail = rng.integers(0, n, size=draws)
        totals += sums[starts_tail + remainder] - sums[starts_tail]
    return totals / n


def calibrate(p):
    rng = np.random.default_rng(p["inference"]["seed"])
    hits = 0
    per_block = {str(b): 0 for b in p["inference"]["blocks"]}
    for trial in range(200):
        shocks = rng.normal(scale=np.sqrt(1 - .8 ** 2), size=1700)
        state = rng.normal()
        vals = []
        for shock in shocks:
            state = .8 * state + shock
            vals.append(state)
        d = np.asarray(vals[200:])
        lo, hi = hac_summary(d)["ci95"]
        for b in p["inference"]["blocks"]:
            draws = bootstrap_means(d, b, 499, p["inference"]["seed"] + trial * 1000 + b)
            a, z = np.quantile(draws[:, 0], [.025, .975])
            per_block[str(b)] += int(a <= 0 <= z)
            lo, hi = min(lo, a), max(hi, z)
        hits += int(lo <= 0 <= hi)
    result = {"status": "PASS" if hits / 200 >= .90 else "FAIL",
              "rho": .8, "n": 1500, "replications": 200,
              "draws_per_replication": 499, "envelope_coverage": hits / 200,
              "individual_block_coverage": {k: v / 200 for k, v in per_block.items()},
              "limitation": "One stationary synthetic null; not market-data coverage certification",
              "protocol_sha256": digest(PROTOCOL), "producer_sha256": digest(__file__)}
    dump(REPORT / "calibration.json", result)
    if result["status"] != "PASS":
        raise ValueError("Null calibration failed; do not score")
    print(json.dumps(result), flush=True)


def load_inputs(p):
    cutoff = pd.Timestamp(p["latest_target"])
    frames = {}
    for name, path in zip(("daily", "cross", "vxn", "short", "vvix"), p["sources"]):
        if path.endswith("parquet"):
            frame = pd.read_parquet(ROOT / path, filters=[("date", "<=", cutoff)])
        else:
            raw = pd.read_csv(ROOT / path, usecols=["DATE", "VVIX"])
            raw["DATE"] = pd.to_datetime(raw.DATE, format="%m/%d/%Y")
            frame = raw.set_index("DATE").loc[:cutoff]
        frames[name] = frame
    iv = frames["short"][["vix", "vix9d"]].reindex(frames["daily"].index)
    iv["vxn"] = frames["vxn"].close.reindex(iv.index)
    iv["vvix"] = frames["vvix"].VVIX.reindex(iv.index)
    source_audit = {}
    for (name, frame), path in zip(frames.items(), p["sources"]):
        source_audit[name] = {"path": path, "rows": len(frame),
                              "start": str(frame.index.min().date()),
                              "end": str(frame.index.max().date()),
                              "bounded_content_sha256": hashlib.sha256(
                                  frame.to_csv(float_format="%.17g").encode()).hexdigest()}
    return frames["daily"], frames["cross"], iv, source_audit


def walk_forward(features, p):
    forecasts, audits = [], []
    complete = np.isfinite(features.loc[:, ALL_FEATURES]).all(axis=1)
    for h in p["horizons"]:
        targets = make_targets(features.rv_total, h)
        valid = (complete & np.isfinite(targets.y)
                 & (targets.target_end <= pd.Timestamp(p["latest_target"]))
                 & (features.index >= pd.Timestamp(p["origin_start"]))
                 & (features.index <= pd.Timestamp(p["origin_end"])))
        dates = features.index[valid]
        for month in dates.to_period("M").unique():
            origins = dates[dates.to_period("M") == month]
            fit_origin = origins[0]
            mask = training_mask(features, targets, fit_origin, p["minimum_training_rows"])
            result = fit_predict(features.loc[mask], targets.loc[mask, "y"], features.loc[origins])
            last_target = targets.loc[mask, "target_end"].max()
            for candidate, r2 in result["orthogonal_r2"].items():
                audits.append({"horizon": h, "fit_origin": fit_origin, "candidate": candidate,
                               "projection_r2": r2, "train_n": int(mask.sum())})
            for model, values in result["forecasts"].items():
                forecasts.append(pd.DataFrame({"origin": origins, "horizon": h,
                    "model": model, "target_end": targets.loc[origins, "target_end"].to_numpy(),
                    "y": targets.loc[origins, "y"].to_numpy(), "prediction": values,
                    "fit_origin": fit_origin, "train_n": int(mask.sum()),
                    "train_last_target": last_target}))
        print(f"h={h}: {len(dates)} common origins, {len(dates.to_period('M').unique())} monthly fits", flush=True)
    if not forecasts:
        raise ValueError("INSUFFICIENT_DATA: no evaluable origins")
    return pd.concat(forecasts, ignore_index=True), pd.DataFrame(audits)


def evaluate(forecasts, fits, features, p):
    results = []
    for h in p["horizons"]:
        fh = forecasts[forecasts.horizon == h]
        wide = fh.pivot(index="origin", columns="model", values="prediction").sort_index()
        y = fh.drop_duplicates("origin").set_index("origin").y.reindex(wide.index)
        baseline = qlike(y, wide.baseline)
        differences = np.column_stack([qlike(y, wide[c]) - baseline for c in CANDIDATES])
        boot = {b: bootstrap_means(differences, b, p["inference"]["bootstrap_draws"],
                                  p["inference"]["seed"] + h * 1000 + b)
                for b in p["inference"]["blocks"]}
        for j, candidate in enumerate(CANDIDATES):
            d = differences[:, j]
            delta = float(d.mean())
            block_results = {}
            for b, draws in boot.items():
                lo, hi = np.quantile(draws[:, j], [.025, .975])
                bp = (1 + int((np.abs(draws[:, j] - delta) >= abs(delta)).sum())) / (len(draws) + 1)
                block_results[str(b)] = {"p": bp, "ci95": [float(lo), float(hi)]}
            hac = hac_summary(d)
            intervals = [v["ci95"] for v in block_results.values()] + [hac["ci95"]]
            periods = []
            for start, end in p["inference"]["stability_periods"]:
                selected = (wide.index >= start) & (wide.index <= end)
                periods.append({"start": start, "end": end, "n": int(selected.sum()),
                                "delta": float(d[selected].mean())})
            years = [{"year": int(year), "n": int((wide.index.year == year).sum()),
                      "delta": float(d[wide.index.year == year].mean())}
                     for year in np.unique(wide.index.year)]
            phase = features.index.get_indexer(wide.index) % h
            phases = [{"phase": i, "n": int((phase == i).sum()),
                       "delta": float(d[phase == i].mean())} for i in range(h)]
            r2 = fits.loc[(fits.horizon == h) & (fits.candidate == candidate), "projection_r2"]
            results.append({"horizon": h, "candidate": candidate, "n": len(d),
                "first_origin": str(wide.index.min().date()), "last_origin": str(wide.index.max().date()),
                "baseline_qlike": float(baseline.mean()), "candidate_qlike": float(baseline.mean() + delta),
                "delta": delta, "improvement_pct": -100 * delta / float(baseline.mean()),
                "win_rate": float((d < 0).mean()), "block_inference": block_results,
                "hac126": hac, "ci95_envelope": [min(v[0] for v in intervals), max(v[1] for v in intervals)],
                "p_conservative": max(hac["p"], *(v["p"] for v in block_results.values())),
                "projection_r2_median": float(r2.median()), "periods": periods,
                "annual": years, "nonoverlap_phases": phases})
    corrected = holm_adjust([r["p_conservative"] for r in results])
    for row, hp in zip(results, corrected):
        row["p_holm"] = float(hp)
        row["verdict"] = "EXPLORATORY_SHORTLIST" if (
            hp < .05 and row["improvement_pct"] >= 1
            and all(x["delta"] < 0 for x in row["periods"])) else "INCONCLUSIVE"
    return {"evidence_class": p["evidence_class"], "rows": results,
            "protocol_sha256": digest(PROTOCOL), "producer_sha256": digest(__file__)}


def write_report(result, calibration):
    lines = ["# Orthogonal-signal round 2 — exploratory results", "",
             "Eight fixed tests on previously inspected history; no new confirmation claim.", "",
             "The baseline contains HAR, lagged VXN/VIX, leverage, term slope, and the previously tested stress composites.",
             "Positive improvement means lower QLIKE. All arms use identical rows and monthly refits.", "",
             "| candidate | horizon | n | baseline QLIKE | candidate QLIKE | improvement | corrected p | verdict |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in result["rows"]:
        lines.append(f"| {r['candidate']} | {r['horizon']} | {r['n']} | {r['baseline_qlike']:.6f} | {r['candidate_qlike']:.6f} | {r['improvement_pct']:+.2f}% | {r['p_holm']:.4f} | {r['verdict']} |")
    lines += ["", "The corrected p uses the largest result across block lengths 21/63/126 and HAC(126),",
              "then Holm across all eight hypotheses. No equivalence claim follows from an inconclusive result.",
              "Shortlisting additionally requires >=1% improvement and a favorable gap in every fixed period.", "",
              "## Dependence and resolution", "",
              "The interval below is the envelope of the three nominal block intervals and HAC(126),",
              "not a simultaneous family confidence interval. MDE is nominal 80% power from HAC(126),",
              "before multiplicity and stability gates; it is indicative under stationarity, not achieved power.", "",
              "| candidate | h | paired gap | 95% envelope | nominal MDE (% baseline loss) | projection R² median |",
              "|---|---:|---:|---|---:|---:|"]
    for r in result["rows"]:
        lo, hi = r["ci95_envelope"]
        mde = 100 * r["hac126"]["mde80_nominal"] / r["baseline_qlike"]
        lines.append(f"| {r['candidate']} | {r['horizon']} | {r['delta']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {mde:.2f}% | {r['projection_r2_median']:.3f} |")
    lines += ["", "## Fixed-period stability", "",
              "Candidate-minus-baseline QLIKE: negative favors the candidate.", "",
              "| candidate | h | 2016–2019 | 2020–2022 | 2023–2025 |", "|---|---:|---:|---:|---:|"]
    for r in result["rows"]:
        values = " | ".join(f"{v['delta']:+.6f}" for v in r["periods"])
        lines.append(f"| {r['candidate']} | {r['horizon']} | {values} |")
    lines += ["", "## Annual and nonoverlapping phase checks", "",
              "All years and all five offsets are reported; none is selected for inference.", "",
              "| year | " + " | ".join(f"{r['candidate']} h{r['horizon']}" for r in result["rows"]) + " |",
              "|---|" + "---:|" * len(result["rows"])]
    for year in range(2016, 2026):
        values = [next(v["delta"] for v in r["annual"] if v["year"] == year) for r in result["rows"]]
        lines.append(f"| {year} | " + " | ".join(f"{v:+.5f}" for v in values) + " |")
    lines += ["", "| candidate, h=5 | phase 0 | phase 1 | phase 2 | phase 3 | phase 4 |",
              "|---|---:|---:|---:|---:|---:|"]
    for r in result["rows"]:
        if r["horizon"] == 5:
            lines.append(f"| {r['candidate']} | " + " | ".join(f"{v['delta']:+.6f} (n={v['n']})" for v in r["nonoverlap_phases"]) + " |")
    lines += ["", "## Scope and reproducibility", "",
              f"- Pre-run AR(1) calibration: {calibration['envelope_coverage']:.1%} coverage over 200 stationary null simulations; this does not certify market-data coverage.",
              "- Target: mean future daily GK-plus-raw-overnight variance, not five-minute RV; latest-vintage vendor history is not a point-in-time archive.",
              "- Origins stop 2025-10-17 (h=1) / 2025-10-13 (h=5); targets stop 2025-10-20. The sealed phase is unused.",
              "- Price-derived features are new summaries of old inputs. Projection removes training-sample linear dependence only.",
              "- Any lead needs genuinely new observations and an alternative variance proxy. No thresholds are adjusted after this run.",
              "- Full numerical results: `metrics.json`; provenance and hashes: `manifest.json`; independent reconstruction: `verification.json`.",
              "- Reproduce: `make orthogonal-round2`; independently check: `make verify-orthogonal-round2`.",
              f"- Protocol SHA-256: `{result['protocol_sha256']}`.",
              f"- Producer SHA-256: `{result['producer_sha256']}`.", ""]
    (REPORT / "results.md").write_text("\n".join(lines))


def check_contracts():
    command = [sys.executable, "-m", "unittest", "tests.test_orthogonal_round2",
               "tests.test_verify_orthogonal_round2", "tests.test_round2_inference",
               "tests.test_methodology", "tests.test_signal_safety", "-q"]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "pre_run_checks.txt").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    print(result.stderr.strip(), flush=True)


def run(p):
    calibration = json.loads((REPORT / "calibration.json").read_text())
    if (calibration["status"] != "PASS" or calibration["producer_sha256"] != digest(__file__)
            or calibration["protocol_sha256"] != digest(PROTOCOL)):
        raise ValueError("Calibration missing or stale for this exact code/protocol")
    daily, cross, iv, sources = load_inputs(p)
    manifest = {"created_utc": datetime.now(UTC).isoformat(), "sources": sources,
                "protocol_sha256": digest(PROTOCOL), "producer_sha256": digest(__file__),
                "tests_sha256": digest(ROOT / "tests/test_orthogonal_round2.py"),
                "python": sys.version, "calibration": calibration,
                "existing_artifacts_sha256": {str(f.relative_to(ROOT)): digest(f)
                    for f in list(ROOT.glob("*.yaml")) + list((ROOT / "reports").rglob("*"))
                    if f.is_file() and "orthogonal_round2" not in str(f)}}
    old_path = REPORT / "manifest.json"
    if old_path.exists():
        old = json.loads(old_path.read_text())
        for key in ("sources", "producer_sha256", "protocol_sha256", "tests_sha256"):
            if old[key] != manifest[key]:
                raise ValueError(f"Previous run differs in {key}; require a separately labeled amendment")
    dump(old_path, manifest)
    features = build_features(daily, cross, iv)
    OUT.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUT / "features.parquet")
    forecasts, fits = walk_forward(features, p)
    forecasts.to_parquet(OUT / "forecasts.parquet", index=False)
    fits.to_parquet(OUT / "fits.parquet", index=False)
    result = evaluate(forecasts, fits, features, p)
    for path, expected in manifest["existing_artifacts_sha256"].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"Existing artifact changed: {path}")
    dump(REPORT / "metrics.json", result)
    write_report(result, calibration)
    print(json.dumps([{k: r[k] for k in ("candidate", "horizon", "n", "improvement_pct", "p_holm", "verdict")}
                      for r in result["rows"]], indent=2), flush=True)


def main():
    p = load_protocol()
    check_contracts()
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "calibrate":
        calibrate(p)
    elif command == "run":
        run(p)
    else:
        raise SystemExit("usage: python -m src.orthogonal_round2 [calibrate|run]")


if __name__ == "__main__":
    main()
