# Orthogonal-signal discovery

Selection window: 2016-01-04 through 2021-12-31. All models use the same origins shown below.

| model | n | mean QLIKE | improvement vs safe baseline | DM p vs baseline | win rate |
|---|---:|---:|---:|---:|---:|
| safe_har_iv_lev | 1511 | 0.342753 | — | — | — |
| term_slope | 1511 | 0.331136 | +3.389% | 0.0377 | 0.578 |
| cross_asset | 1511 | 0.344417 | -0.486% | 0.0773 | 0.489 |
| market_state | 1511 | 0.349875 | -2.078% | 0.0061 | 0.517 |
| term_cross | 1511 | 0.332451 | +3.006% | 0.0675 | 0.573 |
| term_market | 1511 | 0.337112 | +1.646% | 0.3353 | 0.576 |
| cross_market | 1511 | 0.350995 | -2.405% | 0.0041 | 0.506 |
| full | 1511 | 0.338156 | +1.341% | 0.4397 | 0.574 |

Locked winner: **term_slope**.

The DM and win-rate columns are descriptive in discovery; only mean QLIKE selects the winner under the frozen rule.
