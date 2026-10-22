# Post-result sensitivity supplement

This supplement was specified after inspecting the primary results. It adds descriptive checks only. All preregistered means, tests and confidence intervals retain all 320 pairs in each cell. No selection, exclusion or criterion was changed.

An effect e below zero does not imply that both swap directions have opposite signs. The paired effect is e=(recipient median shift−donor median shift)/2. We therefore report the two-direction conjunction separately.

| U666 motif | Negative e | Recipient<0 and donor>0 | Recipient<−tau and donor>tau |
|---|---:|---:|---:|
| short | 320/320 | 320/320 | 320/320 |
| long | 320/320 | 319/320 | 318/320 |

The long-motif U666 result has one pair failing the strict-zero conjunction and two failing the numerical-band conjunction, even though all 320 paired effects are negative.

## Recovery effect heterogeneity

All numbers below are milli log-variance units. Removing the cases with the largest absolute paired effects is an outcome-dependent descriptive sensitivity check. These reduced-sample means are not replacement estimates or new confirmatory tests.

| Group / motif | Mean | Median | q05 | q95 | Mean excluding largest absolute 1 / 5 / 16 |
|---|---:|---:|---:|---:|---:|
| recovery_k4 / short | -1.484129 | -1.044989 | -4.438972 | 0.200963 | -1.444126 / -1.377487 / -1.248416 |
| recovery_k4 / long | -1.039264 | -0.613689 | -7.547283 | 3.434968 | -0.918619 / -0.608709 / -0.303490 |
| recovery_k8 / short | -1.662025 | -1.209736 | -5.000806 | 0.207043 | -1.619488 / -1.544834 / -1.402387 |
| recovery_k8 / long | -1.178241 | -0.700951 | -7.205606 | 3.646326 | -1.046668 / -0.701830 / -0.386017 |
| recovery_tail4 / short | -0.177121 | -0.130892 | -0.888062 | 0.229692 | -0.189449 / -0.189462 / -0.161573 |
| recovery_tail4 / long | -0.136435 | -0.103712 | -1.037169 | 0.771022 | -0.121057 / -0.094670 / -0.087645 |

Recovery’s long-motif distributions contain both positive and negative effects and larger tails than the short-motif distributions. Means and medians remain negative, but excluding the 16 largest absolute effects materially reduces the long-motif mean magnitude. This supports reporting heterogeneity and tail sensitivity rather than claiming every pair has the same effect.

Reproduce with `python scripts/tirex_stratified_postdiagnostics.py --run-dir <new run> --output POSTRESULT_SENSITIVITY.json`. Three dedicated tests were written and failed before implementation, including a case where every paired effect is negative but only one pair has the anticipated signs in both swap directions.
