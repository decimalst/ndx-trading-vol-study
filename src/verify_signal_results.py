"""Independent recomputation of the locked signal-study confirmation.

Deliberately does not import src.signal_study or src.metrics. It reconstructs
losses, DM, coverage, fences, and the protocol digest directly from artifacts.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _sha(protocol: dict) -> str:
    raw = json.dumps(protocol, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _qlike(realized: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    ratio = realized / forecast
    return ratio - np.log(ratio) - 1.0


def _dm_h1(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = a - b
    n = len(d)
    stat = float(d.mean() / np.sqrt(np.var(d, ddof=0) / n))
    stat *= float(np.sqrt((n - 1) / n))
    return stat, float(2 * stats.t.sf(abs(stat), df=n - 1))


def _coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray,
              nominal: float = 0.90) -> dict:
    inside = ((y >= lo) & (y <= hi)).astype(int)
    v = 1 - inside
    n, x = len(v), int(v.sum())
    p, phat = 1 - nominal, x / n
    lr_uc = -2 * ((n - x) * np.log((1 - p) / (1 - phat)) +
                  x * np.log(p / phat))
    n00 = int(np.sum((v[:-1] == 0) & (v[1:] == 0)))
    n01 = int(np.sum((v[:-1] == 0) & (v[1:] == 1)))
    n10 = int(np.sum((v[:-1] == 1) & (v[1:] == 0)))
    n11 = int(np.sum((v[:-1] == 1) & (v[1:] == 1)))
    p01 = n01 / (n00 + n01)
    p11 = n11 / (n10 + n11)
    pu = (n01 + n11) / (n00 + n01 + n10 + n11)

    def ll(k1: int, k0: int, prob: float) -> float:
        if prob in (0.0, 1.0):
            return 0.0
        return k1 * np.log(prob) + k0 * np.log(1 - prob)

    lr_ind = -2 * (ll(n01 + n11, n00 + n10, pu) -
                   ll(n01, n00, p01) - ll(n11, n10, p11))
    return {"coverage": float(inside.mean()),
            "p_uc": float(stats.chi2.sf(lr_uc, 1)),
            "p_ind": float(stats.chi2.sf(lr_ind, 1))}


def main() -> None:
    protocol = yaml.safe_load((ROOT / "signal_study.yaml").read_text())
    out = ROOT / protocol["fences"]["outputs_dir"]
    lock = json.loads((out / "discovery_lock.json").read_text())
    if lock["protocol_sha256"] != _sha(protocol):
        raise AssertionError("protocol digest does not match discovery lock")
    if lock["winner"] != "term_slope":
        raise AssertionError(f"unexpected locked winner: {lock['winner']}")

    master = pd.read_parquet(ROOT / "data/processed/master_daily.parquet")
    base = pd.read_parquet(out / "confirmation_safe_har_iv_lev.parquet")
    cand = pd.read_parquet(out / "confirmation_term_slope.parquet")
    pd.testing.assert_index_equal(base.index, cand.index)
    lo = pd.Timestamp(protocol["windows"]["confirmation_start"])
    hi = pd.Timestamp(protocol["windows"]["confirmation_end"])
    clean = pd.Timestamp(protocol["fences"]["clean_start"])
    if base.index.min() != lo or base.index.max() != hi or (base.index >= clean).any():
        raise AssertionError("confirmation origins violate their registered bounds")

    target_var = master["rv_total"].shift(-1).rename("realized_var")
    target_log = master["log_rv"].shift(-1).rename("realized_log")
    j = pd.concat([target_var, target_log, base.add_prefix("b_"),
                   cand.add_prefix("c_")], axis=1).reindex(base.index).dropna()
    if len(j) != len(base):
        raise AssertionError("confirmation forecasts do not all have realized targets")
    next_dates = pd.Series(master.index, index=master.index).shift(-1).reindex(j.index)
    if next_dates.max() >= clean:
        raise AssertionError("a confirmation target reaches the clean window")

    lb = _qlike(j["realized_var"].to_numpy(), j["b_mean_var"].to_numpy())
    lc = _qlike(j["realized_var"].to_numpy(), j["c_mean_var"].to_numpy())
    dm, dm_p = _dm_h1(lc, lb)
    wins = int(np.sum(lc < lb))
    cov = _coverage(j["realized_log"].to_numpy(), j["c_q0.05"].to_numpy(),
                    j["c_q0.95"].to_numpy())
    result = {
        "n": int(len(j)),
        "candidate_mean": float(lc.mean()),
        "baseline_mean": float(lb.mean()),
        "improvement_pct": float(100 * (lb.mean() - lc.mean()) / lb.mean()),
        "wins": wins,
        "win_rate": float(wins / len(j)),
        "dm": dm,
        "dm_p": dm_p,
        **cov,
        "last_target_date": str(next_dates.max().date()),
    }
    recorded = json.loads((out / "confirmation_metrics.json").read_text())
    for key in ("n", "candidate_mean", "baseline_mean", "improvement_pct",
                "wins", "win_rate", "dm", "dm_p", "coverage", "p_uc", "p_ind"):
        if not np.isclose(result[key], recorded[key], rtol=1e-12, atol=1e-12):
            raise AssertionError(f"recorded {key} differs: {recorded[key]} vs {result[key]}")

    diff = pd.Series(lc - lb, index=j.index, name="loss_diff")
    hac_p = {}
    for lags in (5, 22):
        fit = sm.OLS(diff.to_numpy(), np.ones((len(diff), 1))).fit(
            cov_type="HAC", cov_kwds={"maxlags": lags}
        )
        hac_p[lags] = float(fit.pvalues[0])

    yearly = []
    for year, grp in pd.DataFrame({"candidate": lc, "baseline": lb},
                                  index=j.index).groupby(j.index.year):
        yearly.append({
            "year": int(year), "n": int(len(grp)),
            "candidate": float(grp["candidate"].mean()),
            "baseline": float(grp["baseline"].mean()),
            "improvement_pct": float(100 * (grp["baseline"].mean() -
                                              grp["candidate"].mean()) /
                                     grp["baseline"].mean()),
            "win_rate": float((grp["candidate"] < grp["baseline"]).mean()),
        })

    lines = ["# Independent signal-study verification", "",
             "This recomputation imports neither `src.signal_study` nor "
             "`src.metrics`. All recorded confirmation metrics matched at "
             "1e-12 absolute/relative tolerance.", "",
             "- Protocol SHA-256 matched the discovery lock.",
             f"- Origins: {j.index.min().date()} through {j.index.max().date()} "
             f"(n={len(j)}); last realized target: {next_dates.max().date()}.",
             f"- QLIKE: term slope {result['candidate_mean']:.6f}; safe baseline "
             f"{result['baseline_mean']:.6f}; improvement {result['improvement_pct']:+.3f}%.",
             f"- Paired wins: {wins}/{len(j)} ({result['win_rate']:.3f}).",
             f"- DM: {dm:.3f}, p={dm_p:.4f}.",
             f"- 90% interval: coverage {cov['coverage']:.3f}, p_uc={cov['p_uc']:.4f}, "
             f"p_ind={cov['p_ind']:.4f}.",
             f"- Post-hoc conservative loss-differential HAC checks: p={hac_p[5]:.4f} "
             f"at 5 lags; p={hac_p[22]:.4f} at 22 lags. These do not alter the "
             "pre-registered verdict.", "", "## Year-by-year description", "",
             "| year | n | term QLIKE | baseline QLIKE | improvement | win rate |",
             "|---:|---:|---:|---:|---:|---:|"]
    for row in yearly:
        lines.append(f"| {row['year']} | {row['n']} | {row['candidate']:.6f} | "
                     f"{row['baseline']:.6f} | {row['improvement_pct']:+.3f}% | "
                     f"{row['win_rate']:.3f} |")
    lines += ["", "Year rows are post-hoc stability descriptions, not additional tests."]
    report = ROOT / protocol["fences"]["reports_dir"] / "verification.md"
    report.write_text("\n".join(lines) + "\n")
    print(json.dumps({**result, "hac_p_5": hac_p[5], "hac_p_22": hac_p[22]},
                     indent=2, sort_keys=True))
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
