# Orthogonal-signal round 2 — exploratory results

Eight fixed tests on previously inspected history; no new confirmation claim.

The baseline contains HAR, lagged VXN/VIX, leverage, term slope, and the previously tested stress composites.
Positive improvement means lower QLIKE. All arms use identical rows and monthly refits.

| candidate | horizon | n | baseline QLIKE | candidate QLIKE | improvement | corrected p | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| vvix | 1 | 2463 | 0.322215 | 0.322474 | -0.08% | 1.0000 | INCONCLUSIVE |
| rv_dispersion | 1 | 2463 | 0.322215 | 0.322413 | -0.06% | 1.0000 | INCONCLUSIVE |
| stock_bond_corr | 1 | 2463 | 0.322215 | 0.322361 | -0.05% | 1.0000 | INCONCLUSIVE |
| close_pressure | 1 | 2463 | 0.322215 | 0.323324 | -0.34% | 1.0000 | INCONCLUSIVE |
| vvix | 5 | 2459 | 0.186639 | 0.186944 | -0.16% | 1.0000 | INCONCLUSIVE |
| rv_dispersion | 5 | 2459 | 0.186639 | 0.186215 | +0.23% | 1.0000 | INCONCLUSIVE |
| stock_bond_corr | 5 | 2459 | 0.186639 | 0.187139 | -0.27% | 1.0000 | INCONCLUSIVE |
| close_pressure | 5 | 2459 | 0.186639 | 0.187412 | -0.41% | 1.0000 | INCONCLUSIVE |

The corrected p uses the largest result across block lengths 21/63/126 and HAC(126),
then Holm across all eight hypotheses. No equivalence claim follows from an inconclusive result.
Shortlisting additionally requires >=1% improvement and a favorable gap in every fixed period.

## Dependence and resolution

The interval below is the envelope of the three nominal block intervals and HAC(126),
not a simultaneous family confidence interval. MDE is nominal 80% power from HAC(126),
before multiplicity and stability gates; it is indicative under stationarity, not achieved power.

| candidate | h | paired gap | 95% envelope | nominal MDE (% baseline loss) | projection R² median |
|---|---:|---:|---|---:|---:|
| vvix | 1 | +0.000259 | [-0.000434, +0.000979] | 0.29% | 0.508 |
| rv_dispersion | 1 | +0.000198 | [-0.000542, +0.001028] | 0.33% | 0.242 |
| stock_bond_corr | 1 | +0.000146 | [-0.000518, +0.000834] | 0.28% | 0.194 |
| close_pressure | 1 | +0.001109 | [-0.001382, +0.003664] | 1.03% | 0.348 |
| vvix | 5 | +0.000305 | [-0.001359, +0.001942] | 1.25% | 0.508 |
| rv_dispersion | 5 | -0.000424 | [-0.001917, +0.000951] | 1.05% | 0.241 |
| stock_bond_corr | 5 | +0.000500 | [-0.000582, +0.001757] | 0.83% | 0.194 |
| close_pressure | 5 | +0.000773 | [-0.000489, +0.002083] | 0.97% | 0.348 |

## Fixed-period stability

Candidate-minus-baseline QLIKE: negative favors the candidate.

| candidate | h | 2016–2019 | 2020–2022 | 2023–2025 |
|---|---:|---:|---:|---:|
| vvix | 1 | +0.000164 | -0.000164 | +0.000851 |
| rv_dispersion | 1 | +0.001019 | -0.000399 | -0.000337 |
| stock_bond_corr | 1 | -0.000302 | +0.000748 | +0.000141 |
| close_pressure | 1 | +0.001748 | +0.000491 | +0.000858 |
| vvix | 5 | -0.000120 | -0.000190 | +0.001456 |
| rv_dispersion | 5 | +0.000664 | -0.001709 | -0.000602 |
| stock_bond_corr | 5 | +0.000354 | +0.000970 | +0.000201 |
| close_pressure | 5 | +0.000263 | +0.001802 | +0.000393 |

## Annual and nonoverlapping phase checks

All years and all five offsets are reported; none is selected for inference.

| year | vvix h1 | rv_dispersion h1 | stock_bond_corr h1 | close_pressure h1 | vvix h5 | rv_dispersion h5 | stock_bond_corr h5 | close_pressure h5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2016 | -0.00047 | +0.00138 | -0.00173 | -0.00005 | +0.00111 | +0.00155 | +0.00029 | +0.00113 |
| 2017 | +0.00006 | +0.00253 | +0.00187 | +0.00495 | -0.00509 | +0.00114 | +0.00221 | +0.00043 |
| 2018 | +0.00064 | -0.00002 | +0.00012 | -0.00325 | +0.00185 | +0.00001 | +0.00068 | -0.00225 |
| 2019 | +0.00043 | +0.00020 | -0.00146 | +0.00534 | +0.00163 | -0.00004 | -0.00175 | +0.00174 |
| 2020 | -0.00057 | -0.00128 | +0.00271 | +0.00395 | -0.00122 | -0.00409 | +0.00536 | +0.00540 |
| 2021 | -0.00134 | +0.00031 | -0.00051 | +0.00075 | -0.00543 | +0.00123 | -0.00136 | +0.00079 |
| 2022 | +0.00143 | -0.00023 | +0.00003 | -0.00326 | +0.00611 | -0.00226 | -0.00112 | -0.00081 |
| 2023 | +0.00051 | -0.00027 | -0.00063 | +0.00266 | +0.00274 | -0.00238 | -0.00163 | +0.00103 |
| 2024 | -0.00017 | -0.00104 | +0.00047 | -0.00222 | -0.00106 | -0.00258 | +0.00051 | -0.00156 |
| 2025 | +0.00258 | +0.00048 | +0.00070 | +0.00249 | +0.00306 | +0.00423 | +0.00215 | +0.00209 |

| candidate, h=5 | phase 0 | phase 1 | phase 2 | phase 3 | phase 4 |
|---|---:|---:|---:|---:|---:|
| vvix | +0.000165 (n=492) | +0.000414 (n=491) | +0.000208 (n=492) | +0.000242 (n=492) | +0.000497 (n=492) |
| rv_dispersion | -0.000411 (n=492) | -0.000451 (n=491) | -0.000607 (n=492) | +0.000086 (n=492) | -0.000739 (n=492) |
| stock_bond_corr | +0.000401 (n=492) | +0.000520 (n=491) | +0.000542 (n=492) | +0.000645 (n=492) | +0.000393 (n=492) |
| close_pressure | +0.000512 (n=492) | +0.000980 (n=491) | +0.000970 (n=492) | +0.000491 (n=492) | +0.000912 (n=492) |

## Scope and reproducibility

- Pre-run AR(1) calibration: 93.5% coverage over 200 stationary null simulations; this does not certify market-data coverage.
- Target: mean future daily GK-plus-raw-overnight variance, not five-minute RV; latest-vintage vendor history is not a point-in-time archive.
- Origins stop 2025-10-17 (h=1) / 2025-10-13 (h=5); targets stop 2025-10-20. The sealed phase is unused.
- Price-derived features are new summaries of old inputs. Projection removes training-sample linear dependence only.
- Any lead needs genuinely new observations and an alternative variance proxy. No thresholds are adjusted after this run.
- Full numerical results: `metrics.json`; provenance and hashes: `manifest.json`; independent reconstruction: `verification.json`.
- Reproduce: `make orthogonal-round2`; independently check: `make verify-orthogonal-round2`.
- Protocol SHA-256: `2c081c8e9dcc9a407cd19d2d49c5e8b76462befeabcebc9a3539dbf69d516c93`.
- Producer SHA-256: `9e8b9f47c2c7f9ef1c8fceb9cda6f910e5f95cfb9dbf1e110161e65440800779`.
