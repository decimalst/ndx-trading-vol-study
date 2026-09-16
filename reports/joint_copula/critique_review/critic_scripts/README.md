# Critique scripts (2026-09-09)

Scripts behind the supplied critique, copied verbatim so its numbers can be reproduced.
Run from the repository root with `.venv/bin/python`. They read only the saved
wave27 outputs and the pinned raw sources; they write nothing.

- `recompute.py` — SciPy multivariate_t / multivariate_normal reconstruction of all
  7,386 densities, phase/offset/slice means, trimmed means, per-year table.
- `clock.py` — training-clock check over all 118 fits, certificate gaps, out-of-sample
  residual moments.
- `infer.py` — independent Bartlett HAC126, 20,000-draw circular block bootstrap
  (numpy default_rng seed 1), gain by max|z| bin (quantile edges, so the top bin is
  max|z| > 4.043, 25 days, not > 4.0), and a first simulation using the saved rho for
  both arms (not refit).
- `sim.py` — the simulation quoted in the critique: 30 replicates (default_rng seed 7),
  n = 2,462 pairs per replicate, generated from each arm's saved per-origin rho with
  exact t8 marginals, then BOTH arms refit by maximum likelihood on the same simulated
  sample and scored in-sample on that sample. The reported sd is across the 30 replicate
  means. It does not reproduce the monthly training/evaluation schedule. As the
  assessment notes, the result is E[loss_T - loss_G] under P = q_T (resp. P = q_G),
  a benchmark under that assumed law, not a bound on the contrast under the true law.
- `leak.py` — truncation/perturbation look-ahead test on feature construction.
- `run_verifiers.py scores|forecasts` — re-runs the repository verifiers on saved outputs.
