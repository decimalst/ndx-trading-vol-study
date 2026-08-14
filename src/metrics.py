"""Scoring: QLIKE, pinball/CRPS, interval coverage tests, DM, MZ/encompassing."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def qlike(rv: np.ndarray, var_hat: np.ndarray) -> np.ndarray:
    z = rv / var_hat
    return z - np.log(z) - 1.0


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    d = y - q
    return np.where(d >= 0, tau * d, (tau - 1) * d)


def crps_from_quantiles(y: np.ndarray, qmat: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """Approx CRPS = 2 * integral of pinball over tau (trapezoid on the grid)."""
    losses = np.stack([pinball(y, qmat[:, j], t) for j, t in enumerate(taus)], axis=1)
    return 2.0 * _trapz(losses, taus, axis=1) / (taus[-1] - taus[0])


def coverage_tests(y: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                   nominal: float) -> dict:
    """Empirical coverage + Kupiec LR_uc + Christoffersen LR_ind + LR_cc."""
    inside = ((y >= lo) & (y <= hi)).astype(int)
    n = len(inside)
    viol = 1 - inside
    x = viol.sum()
    p = 1 - nominal
    pi_hat = x / n if n else np.nan
    # Kupiec unconditional coverage
    with np.errstate(divide="ignore", invalid="ignore"):
        lr_uc = -2 * (
            (n - x) * np.log((1 - p) / (1 - pi_hat)) + x * np.log(p / pi_hat)
        ) if 0 < x < n else np.nan
    # Christoffersen independence (2-state Markov chain on violations)
    v = viol
    n00 = int(np.sum((v[:-1] == 0) & (v[1:] == 0)))
    n01 = int(np.sum((v[:-1] == 0) & (v[1:] == 1)))
    n10 = int(np.sum((v[:-1] == 1) & (v[1:] == 0)))
    n11 = int(np.sum((v[:-1] == 1) & (v[1:] == 1)))
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi_ = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    def _ll(k1, k0, pr):
        if pr in (0.0, 1.0):
            return 0.0
        return k1 * np.log(pr) + k0 * np.log(1 - pr)
    ll_null = _ll(n01 + n11, n00 + n10, pi_)
    ll_alt = _ll(n01, n00, pi01) + _ll(n11, n10, pi11)
    lr_ind = -2 * (ll_null - ll_alt)
    lr_cc = (lr_uc + lr_ind) if np.isfinite(lr_uc) else np.nan
    return {
        "coverage": float(inside.mean()) if n else np.nan,
        "p_uc": float(stats.chi2.sf(lr_uc, 1)) if np.isfinite(lr_uc) else np.nan,
        "p_ind": float(stats.chi2.sf(lr_ind, 1)),
        "p_cc": float(stats.chi2.sf(lr_cc, 2)) if np.isfinite(lr_cc) else np.nan,
    }


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> dict:
    """Diebold-Mariano with Newey-West (h-1 lags) and Harvey small-sample
    adjustment. Negative stat => model A has lower loss."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return {"dm": np.nan, "p": np.nan, "n": n}
    dbar = d.mean()
    gamma0 = np.var(d, ddof=0)
    s = gamma0
    for k in range(1, h):
        cov = np.cov(d[k:], d[:-k], ddof=0)[0, 1]
        s += 2 * (1 - k / h) * cov
    dm = dbar / np.sqrt(s / n)
    harvey = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm *= harvey
    p = 2 * stats.t.sf(abs(dm), df=n - 1)
    return {"dm": float(dm), "p": float(p), "n": n}


# Overlapping 30-calendar-day forecasts share ~21 trading days of target; HAC
# lags must be >= the overlap, and standard practice is ~1.5h. maxlags=21 (the
# old value) sat exactly at the floor and understated the standard errors.
HAC_LAGS_30D = 32  # ceil(1.5 * 21)


def mz_regression(realized_log: pd.Series, forecast_log: pd.Series,
                  maxlags: int = HAC_LAGS_30D) -> dict:
    """Mincer-Zarnowitz in logs with HAC errors: realized = a + b * forecast."""
    import statsmodels.api as sm

    df = pd.concat([realized_log, forecast_log], axis=1, keys=["y", "f"]).dropna()
    X = sm.add_constant(df["f"])
    res = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    # se_beta is reported by the corrected fork so the claim "HAC understates
    # the standard error under overlap" is a measured ratio in each report
    # rather than a number memorised from one window.
    return {"alpha": float(res.params["const"]), "beta": float(res.params["f"]),
            "r2": float(res.rsquared), "p_beta1": float(res.t_test("f = 1").pvalue),
            "se_beta": float(res.bse["f"]), "maxlags": int(maxlags),
            "n": int(res.nobs)}


def encompassing(realized_log: pd.Series, implied_log: pd.Series,
                 model_log: pd.Series, maxlags: int = HAC_LAGS_30D) -> dict:
    """realized = a + b*implied + c*model. Significant c => model adds
    information beyond the market's implied forecast."""
    import statsmodels.api as sm

    df = pd.concat([realized_log, implied_log, model_log], axis=1,
                   keys=["y", "iv", "m"]).dropna()
    X = sm.add_constant(df[["iv", "m"]])
    res = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return {"b_implied": float(res.params["iv"]), "c_model": float(res.params["m"]),
            "p_model": float(res.pvalues["m"]), "r2": float(res.rsquared),
            "n": int(res.nobs)}
