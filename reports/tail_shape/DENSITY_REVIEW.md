# Density mathematics review — pre-fit

The new density implementation is ready for the registered synthetic and independent checks. No observed returns, fitted residuals, event counts, or predictive scores were used in this review.

Hansen's equation (10), with constants (11)–(13) on printed page 710, defines a skewed Student density with mean zero and variance one. The original scanned page was visually checked. Its internal centering and scaling depend on the skew parameter; changing only the two piecewise denominators would not preserve those moments. [Hansen (1994), original paper](https://www.ssc.wisc.edu/~bhansen/papers/ier_94.pdf)

Let `c = Γ((ν+1)/2) / [√(π(ν−2)) Γ(ν/2)]`, `A = 4c(ν−2)/(ν−1)`, `a = Aλ`, and `b² = 1 + (3−A²)λ²`. Put `s = −1` below `−a/b` and `+1` otherwise, and `r = (bz+a)/(1+sλ)`. Then `log f = log b + log c − (ν+1)/2 log(1+r²/(ν−2))`. Each supplied skew parameter gets its own constants.

For `t = r√(ν/(ν−2))`, the CDF is `(1−λ)Tν(t)` on the left and `1−(1+λ)Tν(−t)` on the right. This matches the official `arch` standardized skewed Student implementation algebraically; the right branch uses the survival form to reduce cancellation. Its source was checked directly, without installing or importing `arch`. [Official arch source documentation, version 7.2.0](https://arch.readthedocs.io/en/latest/_modules/arch/univariate/distribution.html#SkewStudent)

`src/tail_shape_density.py` exposes `logpdf(z, nu, lam)`, `cdf(z, nu, lam)`, and `dlogpdf_dlambda(z, nu, lam)`. Inputs broadcast through NumPy. The mathematical domain is finite `ν > 2` and `|λ| < 1`; NaN observations and invalid parameters raise errors. Infinite observations have explicit distributional limits. The wave protocol fixes `ν = 8`; supporting other valid values in the math API does not authorize a search.

The analytic score differentiates both internal constants. With `B=3−A²`, `δ=Bλ/b²`, `d=1+sλ`, `W=r²/(ν−2+r²)`, and `V=r/(ν−2+r²)`, it is `δ − (ν+1)[W(δ−s/d) + VA/(b²d)]`. The join limit is `δ`. Log-space expressions avoid squaring very large observations; normalization uses a beta-function expression instead of a ratio of gamma functions.

All 16 tests were written before implementation: the initial run failed because the module did not yet exist, then the first implementation passed. Final tests and scoped lint both pass. The tests independently check quadrature mass, mean, and variance at `ν=8`, `λ ∈ {−0.95,−0.6,0,0.6,0.95}`; additional `ν=4,30`; the symmetric Student case; CDF integration and differentiation; reflection and continuity; vector broadcasting; finite-difference gradients, including the moving join; zero integrated score; large observations; and fixed external moments.

The external density is `f((u−μ)/σ)/σ`. Both experiment arms must reuse exactly the same fitted `μ` and positive `σ²`; the skew parameter cannot change those modeled moments. The registered bounded shape optimizer belongs to the separate model module. A successful fixed-start optimization with projected-gradient tolerance establishes numerical stationarity, not a global optimum. No retries, alternative starts, or post-result tuning are justified by this math review. Floating arithmetic may saturate `0.95*tanh(...)` at exactly `±0.95`; these remain inside the density's valid `|λ|<1` domain.
