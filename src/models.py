"""Baselines: persistence, EWMA (RiskMetrics), log-HAR-RV, and HAR-X.

Every model emits, per origin date t (forecast for t+1):
  quantiles of log_rv(t+1) on the config grid, plus a point variance forecast
  `mean_var` for QLIKE. For quantile models, mean_var is the truncated-mean
  estimator: trapezoid of exp(q(tau)) over the available grid, divided by its
  mass. It understates the true mean identically across models, so QLIKE
  rankings among quantile models are apples-to-apples; EWMA's mean_var is its
  native variance forecast (flagged in reports).

That "identically" is the frozen pre-registration's claim, and it is only
approximately true -- see `mean_var_from_quantiles` below and
reports/METHODOLOGY_FORK.md. The frozen estimator stays the default so the
pre-registered reports keep reproducing byte-for-byte; `estimator="smearing"`
selects the corrected one and writes to a parallel `*_sm.parquet`.

For the 30-calendar-day horizon, HAR and persistence produce direct point
forecasts of cumulative variance via `predict_cum` (log-target direct
projection). Distributional eval at that horizon is out of scope for v1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

ESTIMATORS = ("trunc", "smearing")
QUANTILE_METHODS = ("trunc", "lognormal", "tail_ext")


def trunc_mean_var(qgrid: np.ndarray, log_q: np.ndarray) -> float:
    """Truncated E[exp(X)] from quantiles of X on grid qgrid (ascending).

    The frozen pre-registered estimator. It integrates exp(q) over the quantile
    grid and divides by the grid's mass, so everything outside [tau_lo, tau_hi]
    is discarded -- and for a right-skewed variable most of E[exp(X)] lives in
    exactly that discarded upper tail. Measured against exact smearing on the
    HAR family it returns 0.866-0.873 of the true conditional mean, and the
    discarded share depends on the grid, which is why every model's QLIKE
    differs between results_clean.md and results_clean_dec.md while EWMA's --
    which never touches this function -- is 0.4036 in both.

    Kept unchanged and kept as the default: the frozen reports are the thing
    corrections get compared against, so they must not move.
    """
    return float(_trapz(np.exp(log_q), qgrid) / (qgrid[-1] - qgrid[0]))


def smearing_mean_var(mu: float, resid: np.ndarray) -> float:
    """Duan smearing: E[exp(X)] = exp(mu) * mean(exp(resid)).

    Exact whenever the residuals are available and exchangeable with the
    forecast error, which is the case for every model fitted here by OLS on
    log RV. No distributional assumption and no grid dependence -- the whole
    residual distribution enters, tails included.
    """
    r = np.asarray(resid, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    return float(np.exp(mu) * np.mean(np.exp(r)))


def _fit_lognormal(qgrid: np.ndarray, log_q: np.ndarray) -> tuple[float, float]:
    """OLS of the quantiles on z = Phi^-1(tau): the implied (mu, sigma)."""
    z = norm.ppf(np.asarray(qgrid, dtype=float))
    A = np.column_stack([np.ones_like(z), z])
    beta, *_ = np.linalg.lstsq(A, np.asarray(log_q, dtype=float), rcond=None)
    return float(beta[0]), float(abs(beta[1]))


def _tail_ext_mean_var(qgrid: np.ndarray, log_q: np.ndarray) -> float:
    """E[exp(X)] with q piecewise-linear in z = Phi^-1(tau), tails extrapolated.

    Chronos-2 and TiRex-2 emit quantiles and nothing else -- no residuals, so
    no smearing. This reconstructs the mean from the quantiles instead of
    discarding the tails.

    The quantile function of a lognormal is exactly linear in z, so linear
    interpolation between knots plus linear extrapolation past the outermost
    ones (at the end-segment slope) makes this EXACT on a true lognormal, on
    any grid -- verified in tests/test_methodology.py. On each segment where
    q(z) = a + b*z the contribution integrates in closed form:

        int exp(a + b*z) phi(z) dz = exp(a + b^2/2) * [Phi(z2 - b) - Phi(z1 - b)]

    Nothing is assumed about the interior shape; only the tail beyond the last
    knot is extrapolated. On real log-RV residuals that extrapolation is
    thin-tailed relative to the truth, so this lands ~3.5% low -- a smaller
    but more dispersion-dependent bias than the ~13% it replaces. Used ONLY
    where residuals do not exist, and those rows are flagged `~` in reports.
    """
    z = norm.ppf(np.asarray(qgrid, dtype=float))
    q = np.asarray(log_q, dtype=float)
    if len(z) < 2 or not np.all(np.isfinite(q)):
        return float("nan")

    def _seg(a: float, b: float, z1: float, z2: float) -> float:
        return float(np.exp(a + b * b / 2) * (norm.cdf(z2 - b) - norm.cdf(z1 - b)))

    total = 0.0
    for j in range(len(z) - 1):
        b = (q[j + 1] - q[j]) / (z[j + 1] - z[j])
        total += _seg(q[j] - b * z[j], b, z[j], z[j + 1])
    b_lo = (q[1] - q[0]) / (z[1] - z[0])
    total += _seg(q[0] - b_lo * z[0], b_lo, -np.inf, z[0])
    b_hi = (q[-1] - q[-2]) / (z[-1] - z[-2])
    total += _seg(q[-1] - b_hi * z[-1], b_hi, z[-1], np.inf)
    return float(total)


def mean_var_from_quantiles(qgrid: np.ndarray, log_q: np.ndarray,
                            method: str = "trunc") -> float:
    """Point variance forecast from quantiles of log variance.

    `trunc` is the frozen estimator, `tail_ext` the correction, `lognormal` a
    reference point kept because the original review used it and thereby
    overstated the defect: a lognormal fit is itself dispersion-biased, which
    is where the retracted "4.7pp spread" figure came from.
    """
    if method == "trunc":
        return trunc_mean_var(qgrid, log_q)
    if method == "tail_ext":
        return _tail_ext_mean_var(qgrid, log_q)
    if method == "lognormal":
        mu, sigma = _fit_lognormal(qgrid, log_q)
        return float(np.exp(mu + sigma * sigma / 2))
    raise ValueError(f"unknown mean-variance method {method!r}; "
                     f"expected one of {QUANTILE_METHODS}")


def rescore_mean_var(fc: pd.DataFrame, taus: np.ndarray,
                     method: str = "tail_ext") -> pd.Series:
    """Recompute `mean_var` for a saved forecast frame from its own quantiles.

    The recovery path for models with no residuals. Returns an all-NaN series
    when the frame carries no quantile columns (EWMA), so the caller keeps the
    native variance forecast rather than inventing one.
    """
    cols = [f"q{t:.2f}" for t in np.asarray(taus, dtype=float)]
    if not all(c in fc.columns for c in cols):
        return pd.Series(np.nan, index=fc.index, name="mean_var")
    grid = np.asarray(taus, dtype=float)
    vals = [mean_var_from_quantiles(grid, row, method=method)
            for row in fc[cols].values]
    return pd.Series(vals, index=fc.index, name="mean_var")


def _point_var(mu: float, resid: np.ndarray, taus: np.ndarray,
               q: np.ndarray, estimator: str) -> float:
    """Dispatch the point forecast for a model that has its own residuals."""
    if estimator == "smearing":
        return smearing_mean_var(mu, resid)
    if estimator == "trunc":
        return trunc_mean_var(taus, q)
    raise ValueError(f"unknown estimator {estimator!r}; expected one of {ESTIMATORS}")


def _emp_quantiles(resid: np.ndarray, taus: np.ndarray) -> np.ndarray:
    return np.quantile(resid, taus)


# ------------------------------------------------------------------ persistence
def run_persistence(df: pd.DataFrame, origins: pd.DatetimeIndex, cfg: dict,
                    estimator: str = "trunc") -> pd.DataFrame:
    taus = np.asarray(cfg["quantiles"])
    y = df["log_rv"]
    dlog = y.diff().dropna()
    rows = []
    for t in origins:
        hist = dlog.loc[:t].tail(250).values
        if len(hist) < 100:
            continue
        base = y.loc[t]
        q = base + _emp_quantiles(hist, taus)
        # The log-change history IS this model's residual distribution, so
        # smearing is exact here too, not just for the fitted regressions.
        rows.append({"origin": t, **{f"q{tau:.2f}": v for tau, v in zip(taus, q)},
                     "mean_var": _point_var(base, hist, taus, q, estimator)})
    return pd.DataFrame(rows).set_index("origin")


# ------------------------------------------------------------------------ EWMA
def run_ewma(df: pd.DataFrame, origins: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    lam = cfg["ewma_lambda"]
    r2 = (df["ret_cc"] ** 2).dropna()
    var = r2.iloc[:50].mean()
    fore = {}
    for t, x in r2.items():
        fore[t] = var          # forecast made at t-1 close for day t ... shift below
        var = lam * var + (1 - lam) * x
    # sigma^2_{t+1} known at close of t:
    sig = pd.Series(fore).shift(-1).rename("mean_var")  # value at index t is forecast for t+1
    sig = sig.reindex(origins).dropna()
    out = sig.to_frame()
    out.index.name = "origin"
    return out


# ------------------------------------------------------------------- HAR / HAR-X
HARX_COLS = ("is_fomc", "is_cpi", "is_nfp", "earnings_wt")


def _har_design(df: pd.DataFrame, cfg: dict, with_x: bool,
                with_iv: bool = False, with_sv: bool = False,
                with_ic: bool = False, with_lev: bool = False) -> pd.DataFrame:
    d = pd.DataFrame(index=df.index)
    d["const"] = 1.0
    d["lrv_d"] = df["log_rv"]
    d["lrv_w"] = np.log(df["rv_total"].rolling(5).mean())
    d["lrv_m"] = np.log(df["rv_total"].rolling(22).mean())
    if with_sv:
        # Patton-Sheppard: replace the daily term with its signed halves, which
        # sum back to rv_total, so this swaps a term rather than adding one.
        # Floored at 1e-3 of the day's total so a one-sided session cannot send
        # a log to -inf; that bounds each term within ~6.9 of log(rv_total).
        floor = 1e-3 * df["rv_total"]
        d["lrv_neg"] = np.log(df["rv_neg"].clip(lower=floor))
        d["lrv_pos"] = np.log(df["rv_pos"].clip(lower=floor))
        d = d.drop(columns=["lrv_d"])
    if with_lev:
        # LHAR (Corsi-Reno): return asymmetry from DAILY data, so it reaches
        # back to 1999 instead of being capped at yfinance's intraday history.
        # This is the economics underneath signed semivariance — downside moves
        # forecast variance more strongly than upside ones — tested on a sample
        # large enough to settle it. r^- = r * 1{r < 0} at each aggregation.
        r = df["ret_cc"]
        rw = r.rolling(5).mean()
        rm = r.rolling(22).mean()
        d["lev_d"] = r.where(r < 0, 0.0)
        d["lev_w"] = rw.where(rw < 0, 0.0)
        d["lev_m"] = rm.where(rm < 0, 0.0)
    if with_ic:
        # Index variance ~ average constituent variance x average correlation.
        # VXN already prices the product; these test whether the SPLIT carries
        # anything beyond the level. SPX-built, so a regime proxy for NDX.
        d["lcor"] = np.log(df["cor1m"])
        d["lvixeq"] = np.log(df["vixeq"])
    if with_iv:
        # log VXN at the close of t, forecasting t+1 — no lookahead. This is the
        # control for any model handed VXN (chronos_cov_iv, tirex_cov_iv): if a
        # foundation model cannot beat four OLS terms on the same information
        # set, the win belonged to the input, not the model. Level vs implied
        # daily variance differ only by an affine transform, which the intercept
        # and slope absorb, so plain log(VXN) is used.
        d["liv"] = np.log(df["vxn"])
    if with_x:
        tau = cfg["event_decay_tau"]
        # features dated t+1 are known at t (calendars are forward-published)
        for c in HARX_COLS:
            d[f"x_{c}"] = df[c].shift(-1)
        d["x_fomc_prox"] = np.exp(-df["days_to_fomc"].shift(-1) / tau)
    d["y_next"] = df["log_rv"].shift(-1)
    return d


def run_har(df: pd.DataFrame, origins: pd.DatetimeIndex, cfg: dict,
            with_x: bool = False, with_iv: bool = False,
            with_sv: bool = False, with_ic: bool = False,
            with_lev: bool = False, estimator: str = "trunc") -> pd.DataFrame:
    taus = np.asarray(cfg["quantiles"])
    design = _har_design(df, cfg, with_x, with_iv, with_sv, with_ic, with_lev)
    feat_cols = [c for c in design.columns if c != "y_next"]
    rows = []
    for t in origins:
        train = design.loc[:t].dropna()
        # y_next at origin t is the unknown target -> ensure it's excluded
        train = train[train.index < t]
        if len(train) < cfg["min_train_days"]:
            continue
        X, y = train[feat_cols].values, train["y_next"].values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        x_t = design.loc[t, feat_cols].values.astype(float)
        if np.any(~np.isfinite(x_t)):
            continue
        mu = float(x_t @ beta)
        q = mu + _emp_quantiles(resid, taus)
        rows.append({"origin": t, **{f"q{tau:.2f}": v for tau, v in zip(taus, q)},
                     "mean_var": _point_var(mu, resid, taus, q, estimator)})
    return pd.DataFrame(rows).set_index("origin")


def predict_cum(df: pd.DataFrame, origins: pd.DatetimeIndex, cfg: dict,
                model: str = "har") -> pd.DataFrame:
    """Direct point forecast of log cumulative 30-calendar-day forward variance."""
    y = np.log(df["fwd_var_30c"].clip(lower=1e-12))
    d = pd.DataFrame(index=df.index)
    d["const"] = 1.0
    d["lrv_d"] = df["log_rv"]
    d["lrv_w"] = np.log(df["rv_total"].rolling(5).mean())
    d["lrv_m"] = np.log(df["rv_total"].rolling(22).mean())
    d["y"] = y
    rows = []
    H = cfg["cum_horizon_cal_days"]
    for t in origins:
        # training rows must have fully realized targets: origin + H cal days <= t
        train = d.dropna()
        train = train[train.index <= t - pd.Timedelta(days=H)]
        if len(train) < cfg["min_train_days"]:
            continue
        if model == "persistence":
            n_ahead = 21  # ~trading days in 30 calendar days
            pred = np.log(df.loc[t, "rv_total"] * n_ahead)
        else:
            X = train[["const", "lrv_d", "lrv_w", "lrv_m"]].values
            beta, *_ = np.linalg.lstsq(X, train["y"].values, rcond=None)
            x_t = d.loc[t, ["const", "lrv_d", "lrv_w", "lrv_m"]].values.astype(float)
            if np.any(~np.isfinite(x_t)):
                continue
            pred = float(x_t @ beta)
        rows.append({"origin": t, "log_cum_var_hat": pred})
    return pd.DataFrame(rows).set_index("origin")
