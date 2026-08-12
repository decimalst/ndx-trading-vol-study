# HMM calibration and incremental-state repair

This post-hoc repair uses a transition-target-specific holdout. The dates are not a pristine project-wide holdout.
All calibration and supervised parameters were locked on annual out-of-fold rows through 2024; HMM parameters were also frozen through 2024.

- Calm holdout origins: 166.
- Five-session transition event rate: 30.1%.

| model | phase-average Brier | phase-average log loss |
|---|---:|---:|
| raw HMM | 0.218189 | 0.673834 |
| Platt HMM | 0.205309 | 0.604679 |
| supervised benchmark | 0.195725 | 0.578219 |
| benchmark + calibrated HMM | 0.196617 | 0.579151 |

Calibration verdict: **PASS**.
Incremental-state verdict: **FAIL**.

The calibrated HMM top probability quintile realized 54.5% events versus 18.2% in the bottom quintile.
