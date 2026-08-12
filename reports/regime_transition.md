# QQQ two-state transition diagnostic

This is a modeling-frame diagnostic on an already-inspected period, not a new holdout. Both models consume only realized-variance history. HMM parameters are refit annually on prior data and scored probabilities are forward-filtered, never smoothed.

- Calm origins scored: 2205.
- Five-session stress-entry rate: 22.0%.

| model | phase-average Brier | phase-average log loss |
|---|---:|---:|
| supervised HAR-state logistic | 0.126650 | 0.420496 |
| two-state Gaussian HMM | 0.137616 | 0.457171 |

- HMM top probability quintile event rate: 61.7%
- HMM bottom probability quintile event rate: 2.9%

- hmm brier below logistic: **False**
- hmm logloss below logistic: **False**
- hmm top quintile above bottom: **True**

Frozen verdict: **FAIL**.

A failure means that discrete latent states do not add predictive value over a directly supervised nonlinear target using the same RV history. A pass still needs future or cross-asset confirmation.
