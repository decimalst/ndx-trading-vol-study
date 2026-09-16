# Fixed joint-return dependence: complete saved result

Status: **COMPLETED; both comparisons passed; exploratory joint-density lead.**

Paired loss differences are candidate minus control negative joint log density, in nats per pair. Negative is better. The marginals are identical across all arms.

| Control | Phase | Pairs / full calendar | Mean difference | 95% diagnostic envelope | Conservative p |
| --- | --- | ---: | ---: | --- | ---: |
| gaussian_copula | development | 1005 / 1006 | -0.030568716 | [-0.041251715, -0.019885717] | 0.00000250 |
| gaussian_copula | evaluation | 1457 / 1458 | -0.031017082 | [-0.041177590, -0.020856573] | 0.00000250 |
| independence | development | 1005 / 1006 | -0.606796812 | [-0.704343504, -0.509250120] | 0.00000250 |
| independence | evaluation | 1457 / 1458 | -0.746935325 | [-0.842245648, -0.649844225] | 0.00000250 |

The envelopes combine the saved HAC and block21/63/126 intervals. They are unadjusted diagnostics; selection used both multiplicity gates separately.

| Control | Whole-comparison p | Holm2 | Holm151 | Verdict |
| --- | ---: | ---: | ---: | --- |
| gaussian_copula | 0.00000250 | 0.00000500 | 0.00037750 | COMPARISON_GATE_PASS |
| independence | 0.00000250 | 0.00000500 | 0.00037750 | COMPARISON_GATE_PASS |

Wave alpha: 0.05/(27×28) = 0.00006613756613756614. Cumulative alpha: 0.05. Both controls required. Bootstrap p values are at the finite-draw resolution floor 1/400000; they are not exact probabilities. The151 entries are the tracked family, not a lifetime search census.

## Fixed stability slices

| Control | Slice | Pairs | Mean difference |
| --- | --- | ---: | ---: |
| gaussian_copula | 2020-01-01 to 2022-12-31 | 756 | -0.024662743 |
| gaussian_copula | 2023-01-01 to 2025-10-20 | 701 | -0.037869978 |
| independence | 2020-01-01 to 2022-12-31 | 756 | -0.692271407 |
| independence | 2023-01-01 to 2025-10-20 | 701 | -0.805888138 |

## Full-calendar modulo-five checks

| Control | Phase | Offset | Pairs | Mean difference |
| --- | --- | ---: | ---: | ---: |
| gaussian_copula | development | 0 | 201 | -0.027531329 |
| gaussian_copula | development | 1 | 201 | -0.035124879 |
| gaussian_copula | development | 2 | 201 | -0.031680814 |
| gaussian_copula | development | 3 | 201 | -0.027355361 |
| gaussian_copula | development | 4 | 201 | -0.031151196 |
| gaussian_copula | evaluation | 0 | 291 | -0.030490975 |
| gaussian_copula | evaluation | 1 | 291 | -0.027338665 |
| gaussian_copula | evaluation | 2 | 291 | -0.030459024 |
| gaussian_copula | evaluation | 3 | 292 | -0.038723316 |
| gaussian_copula | evaluation | 4 | 292 | -0.028057117 |
| independence | development | 0 | 201 | -0.652394428 |
| independence | development | 1 | 201 | -0.615572268 |
| independence | development | 2 | 201 | -0.563591647 |
| independence | development | 3 | 201 | -0.550731553 |
| independence | development | 4 | 201 | -0.651694164 |
| independence | evaluation | 0 | 291 | -0.715869366 |
| independence | evaluation | 1 | 291 | -0.743382149 |
| independence | evaluation | 2 | 291 | -0.690188924 |
| independence | evaluation | 3 | 292 | -0.829806238 |
| independence | evaluation | 4 | 292 | -0.755117053 |

## Full joint-density loss

Continuous log-density losses can be negative; percentages of these losses are not a meaningful accuracy or return measure.

| Model | 2016–2019 | 2020–Oct 2025 |
| --- | ---: | ---: |
| t8_copula | -7.870561058 | -7.275407752 |
| gaussian_copula | -7.839992342 | -7.244390670 |
| independence | -7.263764246 | -6.528472427 |

## Reproduction and scope

The fixed execution is src.joint_copula_search with joint_copula.yaml. It refuses to overwrite or silently restart this registered attempt. Saved outputs are private under data/model_memory_study/joint_copula_wave27. Existing study files and protocols remain unchanged.

Selected tests: 3,400 passed, including87 new generated checks; six exact historical replays quarantined. Runtime forecast verification:4,226 calendar positions,2,463 requested coverage origins,7,389 issued forecasts,7,386 scored forecasts,118 monthly fits,472 marginal models and236 dependence fits. Largest independently reconstructed global optimum gap:9.974728820694168e-9, within the fixed1e-8 limit. Score verification: two comparisons, four phase endpoints,12 bootstrap runs,four HAC runs and149 inherited identities.

Historical development and evaluation windows have been reused. Prior regression tests accessed the period beginning 2025-11-03; it is not untouched confirmation. Current vendor snapshots do not certify complete point-in-time price or implied-volatility vintages. Numerical inputs in this experiment end 2025-10-20. QQQ ETF and SPX price index are distinct instruments; exact synchronized auctions and historical publication latency are not certified.

This is a joint-density research finding conditional on fixed shared marginals. It does not identify pure tail dependence, a new external signal, standalone marginal-volatility improvement, expected-return alpha, calibrated portfolio tail risk, or trading profit. Future untouched outcomes and a separate utility protocol are the next validation stage.

Authoritative metrics SHA256: `d1de728c8fb6909a4ce8ecd6edb055d8b3652693550de140dfca3ff688593442`.
