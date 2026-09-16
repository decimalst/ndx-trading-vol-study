"""Pre-outcome bounded spread-risk forecasts from all six frozen model cells."""

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import chi2, norm, qmc

from src.copula_shape import from_normal, inverse_shape
from src.joint_copula_density import normal_scores

MODELS = ("orig_gaussian", "orig_t8", "cal_gaussian", "cal_t8", "shape_gaussian", "shape_t8")
STRUCTURES = ("put", "call", "condor")
DISTANCES = (0.01, 0.02, 0.03)
QUAD_NODES, QUAD_WEIGHTS = leggauss(48)
CHECK_NODES, CHECK_WEIGHTS = leggauss(96)


def parameters(row, margin):
    if margin == "orig":
        return {
            "location": [0.0, 0.0],
            "scale": [1.0, 1.0],
            "epsilon": [0.0, 0.0],
            "delta": [1.0, 1.0],
        }
    return {
        "location": [row.a_qqq, row.a_spx],
        "scale": [row.b_qqq, row.b_spx],
        "epsilon": [row.epsilon_qqq, row.epsilon_spx] if margin == "shape" else [0.0, 0.0],
        "delta": [row.delta_qqq, row.delta_spx] if margin == "shape" else [1.0, 1.0],
    }


def marginal_probability(log_strike, mu, h, pars, upper=False):
    z = (np.asarray(log_strike) - mu) / np.sqrt(0.75 * h)
    w = normal_scores(z)
    a, b, epsilon, delta = (
        np.asarray(pars[k]) for k in ("location", "scale", "epsilon", "delta")
    )
    v = np.sinh(delta * np.arcsinh((w - a) / b) - epsilon)
    return ndtr(-v if upper else v)


def exact_marginal_risk(mu, h, pars, structure, distance, width=0.01):
    """Marginal probabilities and checked quadrature of the bounded payoff."""
    mu, h = np.asarray(mu, float), np.asarray(h, float)
    if (
        mu.shape != (2,)
        or h.shape != (2,)
        or not np.isfinite(mu).all()
        or not np.isfinite(h).all()
        or (h <= 0).any()
        or not (distance > 0 and width > 0 and distance + width < 1)
    ):
        raise ValueError("Finite marginal inputs and valid spread geometry required")

    def integrate(low, high, upper):
        def fixed(nodes, weights):
            strikes = 0.5 * (high + low) + 0.5 * width * nodes
            return (
                0.5
                * weights
                @ marginal_probability(np.log(strikes)[:, None], mu, h, pars, upper=upper)
            )

        first = fixed(QUAD_NODES, QUAD_WEIGHTS)
        second = fixed(CHECK_NODES, CHECK_WEIGHTS)
        for j in np.flatnonzero(np.abs(first - second) > 1e-8):

            def scalar(strike, j=j):
                return (
                    float(marginal_probability(np.log(strike), mu, h, pars, upper=upper)[j])
                    / width
                )

            midpoint = np.exp(np.clip(mu[j], np.log(low), np.log(high)))
            points = [midpoint] if low < midpoint < high else None
            value, error = quad(
                scalar, low, high, epsabs=1e-11, epsrel=1e-10, points=points, limit=150
            )
            if not np.isfinite(value) or error > 1e-8:
                raise ValueError("Bounded marginal payoff quadrature failed tolerance")
            second[j] = value
        return second

    breach, full, debit = np.zeros(2), np.zeros(2), np.zeros(2)
    if structure in ("put", "condor"):
        short, long = 1.0 - distance, 1.0 - distance - width
        breach += marginal_probability(np.log(short), mu, h, pars)
        full += marginal_probability(np.log(long), mu, h, pars)
        debit += integrate(long, short, False)
    if structure in ("call", "condor"):
        short, long = 1.0 + distance, 1.0 + distance + width
        breach += marginal_probability(np.log(short), mu, h, pars, upper=True)
        full += marginal_probability(np.log(long), mu, h, pars, upper=True)
        debit += integrate(short, long, True)
    if structure not in STRUCTURES:
        raise ValueError("Known bounded spread structure required")
    return {"breach": breach, "full": full, "debit": debit}


def base_draws(power=14, seed=20260913):
    points = qmc.Sobol(3, scramble=True, seed=seed).random_base2(power)
    if ((points <= 0) | (points >= 1)).any():
        raise ValueError("Sobol inverse probability endpoint")
    return norm.ppf(points[:, :2]), np.sqrt(chi2.ppf(points[:, 2], 8) / 8)


def copula_draws(rho, family, normals, scales):
    if not np.isfinite(rho) or abs(rho) > 0.995:
        raise ValueError("Correlation outside declared bounds")
    correlated = np.column_stack(
        (normals[:, 0], rho * normals[:, 0] + np.sqrt(1 - rho * rho) * normals[:, 1])
    )
    if family == "gaussian":
        return correlated
    if family == "t8":
        return normal_scores(correlated / scales[:, None])
    raise ValueError("Unknown dependence family")


def sample_risks(log_returns, structure, distance, width=0.01):
    from src.copula_spread_backtest import terminal_debit

    y = np.asarray(log_returns, float)
    debit = terminal_debit(y, structure, distance, width)
    breach = np.zeros_like(y, dtype=bool)
    if structure in ("put", "condor"):
        breach |= y < np.log(1 - distance)
    if structure in ("call", "condor"):
        breach |= y > np.log(1 + distance)
    portfolio = debit.mean(axis=-1)
    # Fractional empirical tail, avoiding a changing number of observations at ties.
    ordered = np.sort(portfolio, axis=-1)
    tail_n = 0.025 * ordered.shape[-1]
    whole = int(np.floor(tail_n))
    remainder = tail_n - whole
    total = ordered[..., -whole:].sum(axis=-1) if whole else np.zeros(ordered.shape[:-1])
    es = (total + remainder * ordered[..., -whole - 1]) / tail_n
    return {
        "p_any_breach": breach.any(axis=-1).mean(axis=-1),
        "p_both_breach": breach.all(axis=-1).mean(axis=-1),
        "p_both_full": (debit >= 1).all(axis=-1).mean(axis=-1),
        "sample_mean_debit": portfolio.mean(axis=-1),
        "var97_5": np.quantile(portfolio, 0.975, axis=-1),
        "es97_5": es,
    }


def build_risks(
    applications, scored_origins, archive, calendar, *, power=14, seeds=(20260913, 20260914)
):
    """No outcome columns are read to produce a risk forecast."""
    needed = ["origin", "target_end", "phase"]
    clock = archive[needed].set_index("origin")
    query = applications.loc[applications.origin.isin(scored_origins)].copy()
    records, diagnostics = [], []
    draws = [base_draws(power, seed) for seed in seeds]
    for fit_origin, group in query.groupby("fit_origin", sort=True):
        first = group.iloc[0]
        for name in (
            "a_qqq",
            "a_spx",
            "b_qqq",
            "b_spx",
            "epsilon_qqq",
            "epsilon_spx",
            "delta_qqq",
            "delta_spx",
            "training_cutoff",
        ):
            if not group[name].eq(first[name]).all():
                raise ValueError("Monthly marginal or training parameters changed within fit")
        mu = group[["mu_qqq", "mu_spx"]].to_numpy(float)
        h = group[["h_qqq", "h_spx"]].to_numpy(float)
        for model in MODELS:
            margin, family = model.split("_", 1)
            pars = parameters(first, margin)
            rho = float(first[f"rho_{model}"])
            if not group[f"rho_{model}"].eq(rho).all():
                raise ValueError("Monthly dependence parameter changed within fit")
            samples = []
            for normal, scale in draws:
                v = copula_draws(rho, family, normal, scale)
                z = from_normal(v) if margin == "orig" else inverse_shape(v, pars)
                samples.append(mu[:, None, :] + np.sqrt(0.75 * h[:, None, :]) * z[None, :, :])
            for structure in STRUCTURES:
                for distance in DISTANCES:
                    estimates = [sample_risks(s, structure, distance) for s in samples]
                    discrepancy = {
                        k: float(np.max(np.abs(estimates[0][k] - estimates[1][k])))
                        for k in estimates[0]
                    }
                    diagnostics.append(
                        {
                            "fit_origin": str(pd.Timestamp(fit_origin).date()),
                            "model": model,
                            "structure": structure,
                            "distance": distance,
                            "differences": discrepancy,
                        }
                    )
                    for i, row in enumerate(group.itertuples(index=False)):
                        item = {
                            "origin": row.origin,
                            "target_end": clock.loc[row.origin, "target_end"],
                            "feature_cutoff_date": row.training_cutoff,
                            "phase": row.phase,
                            "model": model,
                            "structure": structure,
                            "distance": distance,
                            "width": 0.01,
                        }
                        # Daily forecast feature cutoff is previous full session, not monthly fit cutoff.
                        position = calendar.get_loc(row.origin)
                        if position < 1 or calendar[position + 1] != item["target_end"]:
                            raise ValueError("Exact previous/next full-session clock required")
                        if clock.loc[row.origin, "phase"] != row.phase:
                            raise ValueError("Archive and forecast phase disagree")
                        item["feature_cutoff_date"] = calendar[position - 1]
                        exact = exact_marginal_risk(mu[i], h[i], pars, structure, distance)
                        item.update(
                            {
                                k: float(0.5 * (estimates[0][k][i] + estimates[1][k][i]))
                                for k in estimates[0]
                            }
                        )
                        item["mean_debit"] = float(exact["debit"].mean())
                        for j, asset in enumerate(("qqq", "spx")):
                            for key in ("breach", "full", "debit"):
                                item[f"{key}_{asset}"] = float(exact[key][j])
                        p0, p1 = (float(e["p_any_breach"][i]) for e in estimates)
                        item.update(
                            p_any_scramble0=p0,
                            p_any_scramble1=p1,
                            integration_gate_ambiguous=bool((p0 <= 0.10) != (p1 <= 0.10)),
                        )
                        records.append(item)
    return pd.DataFrame(records).sort_values(
        ["origin", "model", "structure", "distance"]
    ).reset_index(drop=True), diagnostics
