# NQ intraday free-data diagnostic

**Status: `VERIFIED_NO_EVALUABLE_FOLDS`.** This is a data-adequacy outcome,
not a model verdict.

The frozen exploratory protocol tested whether the free Kaggle NQ minute file
could support a five-minute RV/BPV/tripower-quarticity/BNS panel and a later
price-history-versus-lagged-surface comparison. The source does not identify
contracts or document its continuous-futures stitch, so every output remains
diagnostic even if more history is eventually obtained.

## Source and transform audit

| Quantity | Value |
|---|---:|
| Raw rows | 1,048,575 |
| Retained RTH minute rows | 278,041 |
| Constructed RTH sessions | 726 |
| Quality-eligible sessions | 678 |
| Stitch-neighborhood exclusions | 16 |
| BNS-significant jump sessions | 48 |
| Forecast origins | **0** |

The transform keeps 09:30-16:00 ET interval-start bars, requires a complete
390-minute session, uses no cross-session return in realized variance, and
removes a one-session neighborhood around large stitch-or-bad-print flags. The
48 significant jump sessions are descriptive rows in the daily panel; they are
not 48 independent prediction events and were never scored by a model.

## Why no folds were evaluated

The model protocol froze a minimum of 180 common eligible training rows before
the raw file was inspected. Neither annual forward fold reached it:

| Test year | Training rows | Test rows | Frozen minimum | Evaluable? |
|---|---:|---:|---:|---|
| 2024 | 63 | 69 | 180 | No |
| 2025 | 135 | 26 | 180 | No |

Consequently the price-history and augmented models were never fitted or
compared, AUC and lift remain undefined, and no claim about predictive power is
available. Lowering the training minimum after seeing these counts would be a
new protocol rather than a repair.

## Independent verification

The independent verifier imports none of the study implementation. It checked
the frozen protocol, raw and implementation hashes, strict ET localization,
the pre-transform date fence, RTH/session completeness, every daily
RV/BPV/tripower/BNS and intraday-shape calculation, stitch flags, lagged Cboe
features, next-five-session target construction, annual-fold eligibility, and
the clean-window fence. It reproduced:

- 726 sessions, 678 quality-eligible, and 16 stitch-excluded;
- the 63/69 and 135/26 train/test row counts;
- both failed `min_train_rows` gates; and
- zero forecast origins.

Raw source SHA-256:
`1577e60a7feab411e49da7a56c7052a64738cd1757cfd60aa11fd783ff43b60b`.
See [`nq_intraday_study.yaml`](../nq_intraday_study.yaml),
[`src/nq_intraday_study.py`](../src/nq_intraday_study.py), and
[`src/verify_nq_intraday_study.py`](../src/verify_nq_intraday_study.py).
