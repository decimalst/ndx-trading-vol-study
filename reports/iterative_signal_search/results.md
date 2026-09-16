# Iterative signal search: wave 1

Exploratory comparisons on reused history, with separate development and algorithm-evaluation periods.
All 12 new hypotheses remain in the search; cumulative adjustment includes 54 audited earlier comparisons.
Positive improvement means lower forecast loss. A lead must pass both phases, both evaluation subperiods, effect-size and multiplicity gates.

| Study | Candidate vs control | Horizon | Development gain | Evaluation gain | Wave adjusted p | Cumulative adjusted p | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| hf | semivariance vs baseline | 1 | +0.560% | +0.859% | 1.00000 | 1.00000 | DOES_NOT_QUALIFY |
| hf | kernel vs baseline | 1 | +0.224% | +0.666% | 1.00000 | 1.00000 | DOES_NOT_QUALIFY |
| hf | semivariance vs baseline | 5 | -0.836% | +1.417% | 1.00000 | 1.00000 | DOES_NOT_QUALIFY |
| hf | kernel vs baseline | 5 | -0.188% | +0.289% | 1.00000 | 1.00000 | DOES_NOT_QUALIFY |
| hf | semivariance vs baseline | 21 | -1.582% | +2.647% | 1.00000 | 1.00000 | DOES_NOT_QUALIFY |
| hf | kernel vs baseline | 21 | -0.669% | +0.625% | 1.00000 | 1.00000 | DOES_NOT_QUALIFY |
| index | session_split vs baseline | 1 | +0.800% | -0.433% | 1.00000 | 1.00000 | DOES_NOT_QUALIFY |
| index | session_split vs mean | 1 | +1.147% | -0.208% | 1.00000 | 1.00000 | DOES_NOT_QUALIFY |
| index | cross_signed vs baseline | 1 | +0.158% | -0.035% | 1.00000 | 1.00000 | DOES_NOT_QUALIFY |
| index | cross_signed vs mean | 1 | +0.507% | +0.189% | 1.00000 | 1.00000 | DOES_NOT_QUALIFY |
| index | calendar vs baseline | 1 | -0.289% | +0.053% | 1.00000 | 1.00000 | DOES_NOT_QUALIFY |
| index | calendar vs mean | 1 | +0.062% | +0.276% | 1.00000 | 1.00000 | DOES_NOT_QUALIFY |

Leads satisfying all required controls: **none**.

## Interpretation boundaries

- HF candidates use previously unused five-minute archive measures, with current daily price controls, lagged HF history and lagged VIX. Archived rsv direction is unresolved; it is not labeled downside variance.
- Index models predict the next open-to-close return using prior-close market data and calendar information finalized before the opening. Auction prices and costs are assumptions, not demonstrated historical fills.
- All index candidates must beat both the market model and historical mean. Cost screens below are descriptive and do not establish net trading alpha.
- Reused history and older unenumerated repository exploration prevent a pristine confirmation claim. Multiplicity accounting cannot reverse prior inspection.
- Complete uncertainty, MDE, years and nonoverlapping phases are retained in metrics.json.

## Descriptive index execution at two basis points per side

| Phase | Model | Held days | Exposure | Net total return | Sharpe | Max drawdown |
|---|---|---:|---:|---:|---:|---:|
| development | mean | 0 | 0.0% | +0.00% | n/a | 0.00% |
| development | baseline | 458 | 45.6% | +12.77% | 0.34 | -11.81% |
| development | session_split | 491 | 48.9% | +3.19% | 0.13 | -13.56% |
| development | cross_signed | 452 | 45.0% | +10.44% | 0.29 | -12.75% |
| development | calendar | 449 | 44.7% | +1.55% | 0.09 | -18.74% |
| development | always_daylong | 1005 | 100.0% | -17.37% | -0.27 | -32.22% |
| evaluation | mean | 0 | 0.0% | +0.00% | n/a | 0.00% |
| evaluation | baseline | 634 | 43.5% | +16.71% | 0.25 | -30.63% |
| evaluation | session_split | 632 | 43.4% | +18.12% | 0.27 | -18.81% |
| evaluation | cross_signed | 644 | 44.2% | +28.74% | 0.37 | -24.08% |
| evaluation | calendar | 650 | 44.6% | +38.01% | 0.44 | -27.46% |
| evaluation | always_daylong | 1457 | 100.0% | -12.32% | -0.02 | -37.08% |

No signals are promoted automatically. No strategy trades, orders, or capital allocations were executed.
