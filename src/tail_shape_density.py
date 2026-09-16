"""Hansen's skewed Student density with zero mean and unit variance.

The degrees of freedom and skewness broadcast with observations. This module
has no source reader, estimator, parameter search, or empirical entrypoint.
Equations: Hansen (1994), International Economic Review 35(3), pp. 705-730;
https://www.ssc.wisc.edu/~bhansen/papers/ier_94.pdf. The official arch 7.2.0
SkewStudent density and CDF provide a separate implementation reference.
"""
from __future__ import annotations

import numpy as np
from scipy.special import betaln, expit, stdtr


def _parts(z, nu, lam):
    """Broadcast real inputs and compute each skewness's internal constants."""
    if any(np.iscomplexobj(value) for value in (z, nu, lam)):
        raise ValueError("Real observations and distribution parameters required")
    z, nu, lam = np.broadcast_arrays(np.asarray(z, float), np.asarray(nu, float),
                                     np.asarray(lam, float))
    if (np.isnan(z).any() or not np.isfinite(nu).all() or (nu <= 2).any()
            or not np.isfinite(lam).all() or (np.abs(lam) >= 1).any()):
        raise ValueError("Require nu>2, |lambda|<1, finite parameters, and non-NaN observations")
    # c = Gamma((nu+1)/2)/(sqrt(pi*(nu-2))*Gamma(nu/2)).
    # The beta identity avoids subtracting two large log-Gamma values.
    logc = -betaln(nu/2, .5)-.5*np.log(nu-2)
    a_scale = 4*np.exp(logc)*((nu-2)/(nu-1))
    curvature = 3-a_scale**2
    b2 = 1+curvature*lam**2
    if not np.isfinite(logc).all() or not np.isfinite(b2).all() or (b2 <= 0).any():
        raise ValueError("Unrepresentable standardized skew-t constants")
    b = np.sqrt(b2)
    a = a_scale*lam
    shifted = z+a/b
    side = np.where(shifted < 0, -1., 1.)
    width = 1+side*lam
    # Log magnitude keeps valid extreme observations from overflowing when
    # the standardized residual is squared. No density/probability clipping.
    with np.errstate(divide="ignore"):
        logr = np.log(b)+np.log(np.abs(shifted))-np.log(width)
    return nu, lam, logc, a_scale, curvature, b2, b, shifted, side, width, logr


def logpdf(z, nu, lam):
    """Log density of standardized z; nu>2 and |lam|<1, broadcast elementwise.

    Infinite observations have log density -inf. NaNs and invalid parameters
    are rejected. For external location mu and scale sigma>0, use
    logpdf((u-mu)/sigma, nu, lam)-log(sigma).
    """
    nu, _, logc, _, _, _, b, _, _, _, logr = _parts(z, nu, lam)
    return np.log(b)+logc-.5*(nu+1)*np.logaddexp(0., 2*logr-np.log(nu-2))


def cdf(z, nu, lam):
    """Standardized skew-t CDF; infinities map to exact zero or one."""
    nu, lam, _, _, _, _, _, shifted, side, _, logr = _parts(z, nu, lam)
    with np.errstate(over="ignore", under="ignore"):
        ordinary_t = np.sign(shifted)*np.exp(logr+.5*(np.log(nu)-np.log(nu-2)))
    # Using the reflected t CDF on the right avoids subtracting two nearby
    # t probabilities before adding the left-side probability mass.
    left = (1-lam)*stdtr(nu, ordinary_t)
    right = 1-(1+lam)*stdtr(nu, -ordinary_t)
    return np.where(side < 0, left, right)


def dlogpdf_dlambda(z, nu, lam):
    """Analytic skewness score at fixed observation and degrees of freedom.

    Internal centering and scaling derivatives are included. At the moving
    piecewise join both one-sided first derivatives equal b'/b. Infinite
    observations return the corresponding finite tail-score limit.
    """
    nu, lam, _, a_scale, curvature, b2, _, shifted, side, width, logr = _parts(z, nu, lam)
    logk = np.log(nu-2)
    b_relative = curvature*lam/b2
    squared_fraction = expit(2*logr-logk)
    # r/(nu-2+r²), evaluated without constructing r². Its tail limit is zero.
    with np.errstate(invalid="ignore", under="ignore"):
        residual_fraction = np.sign(shifted)*np.exp(logr-np.logaddexp(logk, 2*logr))
    residual_fraction = np.where(np.isposinf(logr), 0., residual_fraction)
    score_quadratic = squared_fraction*(b_relative-side/width)
    score_quadratic += residual_fraction*a_scale/(b2*width)
    return b_relative-(nu+1)*score_quadratic
