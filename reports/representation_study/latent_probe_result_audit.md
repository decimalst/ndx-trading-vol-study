# Latent-probe post-result audit corrections

No label, score, probe, coordinate selection, seed, or control draw changed.

- Ten controls imply a minimum exact corrected randomization p-value of 1/11
  (0.0909). The empirical percentile flag is retained only as
  `frozen_heuristic_pass`; formal 5% evidence is false.
- Episode intervals condition on negative origins and measure positive-episode
  influence, including one variance component per control seed.
- The 13.2% event is recurrent threshold proximity, not independent rare breaks.
- Five-phase minima and maxima are reported.
- Sparse dimensions reconstruct from completed annual training labels only and
  score disjoint held-out rows; yearly coordinate identities are retained.
- MLP termination is audited without raising its frozen 500-iteration cap.
- Chunk reuse now requires a run signature and per-chunk hashes; legacy chunks
  may be sealed only after exact final-matrix reconstruction.

## MLP convergence

- latent_actual: 3/24 converged before the cap; 21 hit the cap.
- augmented_actual: 3/24 converged before the cap; 21 hit the cap.
- control: 9/240 converged before the cap; 231 hit the cap.
