# Prior research exposure and the new search ledger

Prepared before the first wave's empirical fits. This is an inventory of study
families, not a claim that every historical comparison has been counted.

The cumulative numerical correction contains exactly 54 previously audited
contrasts: eight from `reports/orthogonal_round2/metrics.json` and 46 from
`reports/model_memory_study/combined_metrics.json`. The next wave adds all 12
registered contrasts, including unsuccessful or unevaluable entries. These
66 contrasts do not represent the repository's entire prior search.

| Earlier family | Local specification or findings | Relevance to this wave |
|---|---|---|
| Original forecasting, implied-volatility controls, return asymmetry and carry | `config.yaml`, `reports/FINDINGS.md` | Establishes the value of strong volatility controls; daily history has been inspected repeatedly. |
| Orthogonal signals, event timing and directional term structure | `signal_study.yaml`, `reports/ORTHOGONAL_SIGNALS_PROTOCOL.md` | New calendar or term-structure claims require comparison with this prior work. |
| SKEW and conditional carry | `skew_carry.yaml`, `reports/SKEW_CARRY_PROTOCOL.md` | A mechanism finding did not establish an executable strategy. |
| Jump targets and HMM transitions, including calibration repair | `target_regime.yaml`, `regime_repair.yaml`, `reports/TARGET_REGIME_FINDINGS.md` | The SPX Oxford-Man archive and 2014–2017 history have prior target-specific exposure. New archive fields do not create untouched dates. |
| Horizon curves, single-name earnings, implied premium and SPX term-slope replication | `research_paths.yaml`, `reports/RESEARCH_PATHS_FINDINGS.md` | Longer volatility horizons and conditional SPX slope have already been tried; earnings labels have availability limits. |
| Extended history, nonlinear models and later diagnostic follow-ups | `history_extension.yaml`, `gbm_study.yaml`, `gbm_post_result.yaml` | More model flexibility is not a new source of information by itself. |
| Latent representations, residual and tail probes | `representation_study.yaml`, `residual_probe.yaml`, `latent_k1_confirmation.yaml` | Historical latent-space searches inform the current negative assessment of generic memory additions. |
| Surface, futures, positioning and high-frequency option-flow availability | `surface_data_study.yaml`, `nq_intraday_study.yaml`, `free_signal_study.yaml` | Insufficient data and no evaluable folds remain distinct from negative forecast results. |
| Recent orthogonal inputs | `orthogonal_round2.yaml` | All eight contrasts enter the new cumulative ledger unchanged. |
| Modeling, memory and supplied reference papers | `model_memory_study.yaml`, `model_memory_reference.yaml`, `reports/model_memory_study/SUMMARY.md` | All 46 final contrasts enter the cumulative ledger unchanged. No alternative qualified as improving the established benchmark. |

The historical development/evaluation split isolates the new algorithm's
specification from its new scores. It does not erase earlier knowledge of market
history, enforce prospective p-value validity under adaptive reuse, or make
these outcomes confirmatory. A passing result is an exploratory lead for a
separately frozen follow-up. Previously protected dates and report bytes remain
unchanged.
