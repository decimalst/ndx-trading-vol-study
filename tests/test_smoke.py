"""End-to-end smoke test on synthetic data. Exercises everything except the
Chronos model itself (context-frame construction is tested; inference is not).

Run:  python -m tests.test_smoke
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import config, metrics, models
from src.features import build_covariates, forward_realized_var


def make_synthetic(n: int = 1200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n)
    # log-vol AR(1) with event bumps
    log_rv = np.empty(n)
    log_rv[0] = -9.3
    phi, mu, s = 0.97, -9.3, 0.35
    fomc = np.zeros(n)
    fomc[np.arange(20, n, 42)] = 1.0
    for i in range(1, n):
        log_rv[i] = mu + phi * (log_rv[i - 1] - mu) + s * rng.standard_normal()
        log_rv[i] += 0.6 * fomc[i]
    rv = np.exp(log_rv)
    df = pd.DataFrame(index=dates)
    df["rv_total"] = rv
    df["log_rv"] = log_rv
    df["ret_cc"] = np.sqrt(rv) * rng.standard_normal(n)
    df["is_fomc"], df["is_cpi"], df["is_nfp"] = fomc, 0.0, 0.0
    df["days_to_fomc"] = 15.0
    df["days_to_cpi"] = 15.0
    df["days_to_nfp"] = 15.0
    df["earnings_wt"] = 0.0
    df["dow"] = df.index.dayofweek.astype(float)
    df["fwd_var_30c"] = forward_realized_var(df["rv_total"], 30)
    df["vxn"] = np.sqrt(rv * 252).clip(0.05, 1.5) * 100 * 1.1  # crude "implied"
    df["vxn_exp_var_30c"] = (df["vxn"] / 100.0) ** 2 * (30.0 / 365.0)
    return df


def main() -> None:
    cfg = config.load()
    cfg["min_train_days"] = 300
    df = make_synthetic()
    origins = df.index[600:-40]
    taus = np.asarray(cfg["quantiles"])

    pers = models.run_persistence(df, origins, cfg)
    ewma = models.run_ewma(df, origins, cfg)
    har = models.run_har(df, origins, cfg, with_x=False)
    harx = models.run_har(df, origins, cfg, with_x=True)
    for name, fc in [("pers", pers), ("ewma", ewma), ("har", har), ("harx", harx)]:
        assert len(fc) > 400, f"{name}: too few forecasts ({len(fc)})"
        assert np.isfinite(fc["mean_var"]).all(), f"{name}: non-finite mean_var"

    y_log = df["log_rv"].shift(-1)
    y_var = df["rv_total"].shift(-1)

    def score(fc):
        j = fc.join(y_log.rename("y")).join(y_var.rename("yv")).dropna()
        ql = metrics.qlike(j["yv"].values, j["mean_var"].values).mean()
        qcols = [f"q{t:.2f}" for t in taus]
        crps = metrics.crps_from_quantiles(j["y"].values, j[qcols].values, taus).mean() \
            if all(c in j for c in qcols) else np.nan
        return ql, crps, j

    ql_p, _, jp = score(pers)
    ql_h, crps_h, jh = score(har)
    ql_hx, _, _ = score(harx)
    ql_e = metrics.qlike(ewma.join(y_var.rename("yv")).dropna()["yv"].values,
                         ewma.join(y_var.rename("yv")).dropna()["mean_var"].values).mean()
    print(f"QLIKE  persistence={ql_p:.4f} ewma={ql_e:.4f} har={ql_h:.4f} har_x={ql_hx:.4f}")
    print(f"CRPS   har={crps_h:.4f}")
    assert ql_h < ql_p * 1.05, "HAR should be near or beat persistence on AR(1) log-vol"
    assert ql_hx < ql_h, "HAR-X should beat HAR when a genuine event effect exists"

    cov = metrics.coverage_tests(jh["y"].values, jh["q0.05"].values,
                                 jh["q0.95"].values, nominal=0.90)
    print(f"90% interval coverage (har) = {cov['coverage']:.3f}  p_uc={cov['p_uc']:.3f}")
    assert 0.80 < cov["coverage"] < 0.98, "coverage badly off on synthetic"

    dm = metrics.dm_test(
        jh["mean_var"].pipe(lambda s: metrics.qlike(jh["yv"].values, s.values)),
        jp.reindex(jh.index).dropna().pipe(
            lambda d: metrics.qlike(d["yv"].values, d["mean_var"].values)),
    )
    print(f"DM har vs persistence: dm={dm['dm']:.2f} p={dm['p']:.4f} n={dm['n']}")

    cum = models.predict_cum(df, origins, cfg, model="har")
    yc = np.log(df["fwd_var_30c"].clip(lower=1e-12))
    mz = metrics.mz_regression(yc, cum["log_cum_var_hat"])
    enc = metrics.encompassing(yc, np.log(df["vxn_exp_var_30c"]), cum["log_cum_var_hat"])
    print(f"30c MZ beta={mz['beta']:.3f} R2={mz['r2']:.3f} | "
          f"encompassing c_model={enc['c_model']:.3f} p={enc['p_model']:.4f}")
    assert 0.3 < mz["beta"] < 2.0

    # covariate builder + Chronos frame construction (no inference)
    covdf = build_covariates(cfg, df.index)
    assert {"is_fomc", "days_to_fomc", "earnings_wt"}.issubset(covdf.columns)
    from src.chronos_runner import _frames_for_batch
    ctx, fut = _frames_for_batch(df, [origins[100], origins[101]], cfg, "cov", 21)
    assert ctx["id"].nunique() == 2 and len(fut) == 42
    assert (fut.groupby("id").size() == 21).all()
    print("chronos frame construction ok")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
