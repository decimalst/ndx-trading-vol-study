# Peak age and one-month SPX returns

**No new qualifying predictive signal.** Independent verification passed, and none of the three registered comparisons qualified.

The experiment asks whether time since the most recent closing-price maximum in a fixed 252-session window improves 21-session SPX return forecasts. All market inputs stop at the preceding observed session. Four models use identical training rows and application dates: the training mean, the existing market baseline, a baseline augmented with drawdown depth/depth squared/window return, and that augmented model with a single peak-age correction.

## Results

Positive relative MSE gain means lower forecast error. Each comparison had to improve by at least 0.25% in both periods and satisfy the fixed stability, dependence and multiple-comparison gates.

| Control | Development gain | Evaluation gain | Wave Holm p | Cumulative Holm p | Result |
|---|---:|---:|---:|---:|---|
| Existing market baseline | +3.9623% | -10.9208% | 1.00000000 | 1.00000000 | DOES_NOT_QUALIFY |
| Matched drawdown/window-return model | -0.1796% | -0.6044% | 1.00000000 | 1.00000000 | DOES_NOT_QUALIFY |
| Training mean | -3.7601% | -11.5104% | 1.00000000 | 1.00000000 | DOES_NOT_QUALIFY |

The matched drawdown model isolates the added peak-age correction most directly: adding age slightly worsened measured error in both periods. Its reported interval envelopes include zero, so these estimates do not establish a reliable deterioration. They do provide no basis for promoting the correction. The development improvement over the original market baseline reverses in evaluation and cannot be attributed to age alone because the candidate also includes the drawdown/window-return controls.

There are 985 common development origins, 2016-01-04 through 2019-11-29, and 1,437 evaluation origins, 2020-01-02 through 2025-09-19. The three new hypotheses bring the cumulative family to 140. No configuration, threshold or tolerance was changed after registration or historical numerical access.

## Verification and preservation

All 2,378 repository tests passed before the prospective freeze. The runner then passed all 154 focused tests before registering the three comparisons and admitting source values. The freeze covers 336 Python files, preserving all 319 earlier files unchanged, and 75 prefit artifacts. The run manifest binds 2,199 input artifacts.

The independent verifier reconstructed 9,688 scored forecasts and squared losses across 2,422 origins. It also checked all 9,852 application forecasts on 2,463 origins, including 41 unscored applications; 118 monthly schedules; 236 nuisance fits; 118 scalar corrections; and 118 training means. It independently recomputed all six phase comparisons, 18 block-bootstrap runs with 399,999 draws each, HAC504 inference, 126 nonoverlap-offset checks, and Holm adjustment over the new and cumulative families. The exact ledger contains three registrations, 137 inherited records and three evaluated records.

The prior failed clustering study remains failed and counted at p=1. The completed clustering replay and all other preceding records remain unchanged. The new forecast outputs are local research artifacts; no trades or repository publication were performed.

## Limits and evidence

This is another test on reused archival price-index history, with overlapping targets and existing source-vintage/backcalculation limitations. Sequential age correction can also adjust ridge shrinkage inside the control-feature span; even a favorable result would establish model-relative utility rather than an independent information channel. SPX price returns omit dividends and risk-free subtraction. No new numerical observations from the protected period beginning 2025-11-03 were admitted, and this is not a trading-profit test.

- [Complete effects, uncertainty, slices and offset diagnostics](metrics.json).
- [Independent reconstruction and inference proof](verification.json).
- [Prospective design](DESIGN.md) and [freeze record](freeze_record.json).
- [Comparison figure](comparison_intervals.png) and [exportable PDF](comparison_intervals.pdf).
