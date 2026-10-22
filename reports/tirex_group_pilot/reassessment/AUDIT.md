# Retrospective audit of the TiRex group-pilot critique

This is a post-result diagnostic analysis of the saved October 22 group pilot, not a new preregistered test. The extractor imports no producer implementation. Eight tests were written first, observed to fail on the absent implementation, and then passed. This is internal verification by a second implementation in the same AI-assisted project, not independent replication or peer review. Frozen data and original criteria are unchanged.

## Correct the framing without replacing it with new overclaims

- Replace the headline “0/6” with three separate observations: a motif-dependent layer-0 recency effect concentrated in U666; selected layer-11 persistence coordinates with small absolute swap influence relative to these reference sets; and unstable discovery orientation for the layer-5 recovery group. Keep 0/6 only as the result of the historical composite criterion, alongside its endpoint, raw p-values, counts, floor and power limitations.
- The recency pooled probability endpoint conflates two visibly different motif strata. All 16 short pairs exceed the original floor; all 16 long pairs do not. This explains the observed 16/32 result, but the generator does not mathematically force a null effect. The long native donor–recipient forecast contrast is +5.437 milli versus +2.633 milli for short motifs, so “near-zero native contrast” does not explain the difference. Conditional on an exactly half responsive mixture, simply increasing n cannot rescue a majority-exceedance endpoint whose population probability remains 0.5.
- Holm correction remains valid under arbitrary dependence. Nested claims can make it conservative; dependence does not invalidate it. The restrictive probability endpoint and motif pooling are more fundamental here than multiplicity alone.
- L0/U666, not U103, dominates the group result. U103 was selected by the earlier single-unit pilot; its validation mean absolute shift was 0.00940 milli on different contexts. U666 was selected by the new group discovery sample and measured through the constituent-singleton ablation. It is a different unit and finding, not replication of the previous U103 result.
- Recovery k4 and k8 have separately frozen forecast directions (-1 and +1). Their validation RAW effects are both negative. Adding members 5–8 did not flip the raw validation sign. The visual +1.263 to -1.422 comparison used different orientations. The full group does reverse its own small, unstable discovery mean, but its discovery sign was weak: the raw bidirectional effect was positive on 13/24 discovery pairs and negative on 11/24. The four-unit discovery effect split 12/12.
- “0% both expected” does not imply “100% both opposite”: recovery k8 has 21/32 both opposite and 11/32 mixed or zero. Its oriented bidirectional average is negative on 28/32, positive on 4/32.
- Matching activation SD, outgoing-column norm and paired input displacement did not match downstream influence. The selected persistence set is less influential than those four references under these swaps. This does not show L11 is globally sensitive, establish that all effects are off-manifold artifacts, or establish that these units are forecast-irrelevant under other inputs/interventions. The frozen protocol did not set a hard matching tolerance; it minimized a proxy distance and reported ratios. A random-set reference distribution avoids implying a match it does not enforce.
- Small joint-minus-sum differences are compatible with local linearization. Additivity neither establishes nor refutes distributed representations or superposition. A dose-response experiment can measure departures from linear scaling; merely increasing dose does not guarantee departure from the local linear regime.

## Original endpoint and power

The original endpoint is P(effect > 0.001) > 0.5, not a nonzero mean effect. No distribution-free minimum detectable mean in log-variance units exists for this binary endpoint: arbitrarily large effects in fewer than half the pairs can fail. The table gives the strictest first Holm step at alpha 0.05/18 and the true exceedance probability needed for 80% rejection probability. Later Holm steps can be less stringent if preceding claims reject.

| Pairs | First-step minimum successes | Raw p at cutoff | Raw p one fewer | True exceedance probability for 80% power |
|---:|---:|---:|---:|---:|
| 32 | 25 | 0.00105120079 | 0.00350018335 | 0.821449 |
| 160 | 98 | 0.00274527328 | 0.00443679519 | 0.641205 |
| 320 | 186 | 0.00214286939 | 0.00303606719 | 0.602680 |

## All 18 historical claims

One-sided exact binomial tests; success means strict exceedance of +1 milli. All Holm-adjusted p-values are 1.

| Group | Claim | k/32 | Raw p | Mean (milli) |
|---|---|---:|---:|---:|
| persistence_k4 | influence | 0/32 | 1 | 0.167355 |
| persistence_k4 | control_advantage | 0/32 | 1 | -2.377136 |
| persistence_k4 | singleton_advantage | 0/32 | 1 | -0.067897 |
| persistence_k8 | influence | 12/32 | 0.944907917408 | 1.206063 |
| persistence_k8 | control_advantage | 0/32 | 1 | -23.103079 |
| persistence_k8 | singleton_advantage | 0/32 | 1 | 0.106364 |
| recency_k4 | influence | 16/32 | 0.569974967046 | 2.359375 |
| recency_k4 | control_advantage | 14/32 | 0.811457206262 | 1.221303 |
| recency_k4 | singleton_advantage | 1/32 | 0.999999999767 | 0.112399 |
| recency_k8 | influence | 16/32 | 0.569974967046 | 2.431825 |
| recency_k8 | control_advantage | 12/32 | 0.944907917408 | 1.018740 |
| recency_k8 | singleton_advantage | 1/32 | 0.999999999767 | 0.191644 |
| recovery_k4 | influence | 15/32 | 0.701692552073 | 1.262948 |
| recovery_k4 | control_advantage | 8/32 | 0.998948799213 | 0.559770 |
| recovery_k4 | singleton_advantage | 4/32 | 0.999998721993 | 0.528812 |
| recovery_k8 | influence | 3/32 | 0.999999876833 | -1.422316 |
| recovery_k8 | control_advantage | 7/32 | 0.999732473632 | 0.541084 |
| recovery_k8 | singleton_advantage | 8/32 | 0.998948799213 | 0.662163 |

## Discovery versus validation

Raw bidirectional effect = (recipient median change − donor median change)/2. Oriented effect multiplies that by each group’s own discovery-frozen direction. All numeric effects below are milli log-variance units. Discovery p-values are descriptive because selection and direction fixing used discovery.

| Group | Frozen direction | Discovery raw / oriented | Validation raw / oriented | Discovery both expected/opposite/mixed | Validation both expected/opposite/mixed |
|---|---:|---:|---:|---:|---:|
| persistence_k4 | -1 | -0.217567 / 0.217567 | -0.167355 / 0.167355 | 7/0/17 | 11/0/21 |
| persistence_k8 | +1 | 1.431396 / 1.431396 | 1.206063 / 1.206063 | 24/0/0 | 30/0/2 |
| recency_k4 | -1 | -5.371054 / 5.371054 | -2.359375 / 2.359375 | 24/0/0 | 31/0/1 |
| recency_k8 | -1 | -6.799420 / 6.799420 | -2.431825 / 2.431825 | 24/0/0 | 27/1/4 |
| recovery_k4 | -1 | -0.031193 / 0.031193 | -1.262948 / 1.262948 | 5/4/15 | 21/0/11 |
| recovery_k8 | +1 | 0.099679 / 0.099679 | -1.422316 / -1.422316 | 3/4/17 | 0/21/11 |

## Motif-stratified diagnostics

These strata were present in the generator, but the original confirmatory test pooled them. The following are retrospective descriptive diagnostics, not a replacement pass/fail rule. Activation consistency is the sign of the mean across individually standardized/oriented member activation differences, not unanimity of every member.

| Group / motif | n | Raw effect | Oriented effect | Recipient / donor shift | Native contrast | Exceeds1 milli | Both expected/opposite/mixed | Positive activation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| persistence_k4 / long | 18 | -0.136720 | 0.136720 | -0.302898 / -0.029458 | -1342.144860 | 0/18 | 6/0/12 | 18/18 |
| persistence_k4 / short | 14 | -0.206743 | 0.206743 | -0.447409 / -0.033924 | -1405.250583 | 0/14 | 5/0/9 | 14/14 |
| persistence_k8 / long | 18 | 0.662963 | 0.662963 | 0.771470 / -0.554456 | -1342.144860 | 2/18 | 16/0/2 | 18/18 |
| persistence_k8 / short | 14 | 1.904334 | 1.904334 | 2.691303 / -1.117366 | -1405.250583 | 10/14 | 14/0/0 | 14/14 |
| recency_k4 / long | 16 | -0.161260 | 0.161260 | -0.121593 / 0.200927 | 5.436897 | 0/16 | 15/0/1 | 0/16 |
| recency_k4 / short | 16 | -4.557490 | 4.557490 | -4.520357 / 4.594624 | 2.633154 | 16/16 | 16/0/0 | 16/16 |
| recency_k8 / long | 16 | -0.117958 | 0.117958 | -0.082314 / 0.153601 | 5.436897 | 0/16 | 11/1/4 | 0/16 |
| recency_k8 / short | 16 | -4.745692 | 4.745692 | -4.756868 / 4.734516 | 2.633154 | 16/16 | 16/0/0 | 16/16 |
| recovery_k4 / long | 14 | -1.226016 | 1.226016 | -3.048624 / -0.596591 | -104.221889 | 4/14 | 6/0/8 | 14/14 |
| recovery_k4 / short | 18 | -1.291672 | 1.291672 | -1.481745 / 1.101600 | -75.351874 | 11/18 | 15/0/3 | 18/18 |
| recovery_k8 / long | 14 | -1.490218 | -1.490218 | -2.553531 / 0.426905 | -104.221889 | 3/14 | 0/6/8 | 14/14 |
| recovery_k8 / short | 18 | -1.369503 | -1.369503 | -1.177364 / 1.561642 | -75.351874 | 0/18 | 0/15/3 | 18/18 |

## Dominant recency member and recovery decomposition

U666 has a mean absolute effect 2.247781 milli versus 2.360180 milli for the four-unit group and 2.439424 milli for the eight-unit group. Those are magnitudes, not a causal percentage attribution. U666 moves both directions with the group’s frozen sign on 32/32 pairs. Its short-motif mean is 4.299700 milli (16/16 above 1 milli), long-motif mean 0.195861 milli (0/16 above 1 milli). Other group members have small but nonzero effects; “contribute nothing” is too strong.

For recovery validation, the sum of prefix-four singleton RAW bidirectional effects is−1.234710 milli; the added-four singleton sum is−0.154912 milli. Their sum is−1.389623 milli, close to the full group RAW effect−1.422316 milli. This is primarily evidence against interpreting the differently oriented chart values as a validation sign flip caused by antagonistic additions. Discovery has no saved singleton interventions, so discovery member-level antagonism cannot be reconstructed from these artifacts.

## Mechanics and limits

The patched “unit” is one of 1408 coordinates of the native gated FFN output immediately before its 512-dimensional down projection: `stack[layer].time_mixer.ffn.proj_down` input, at token 63. It is neither a 512-dimensional residual-stream coordinate nor an sLSTM/mLSTM cell state. Token63 is the final token of 2048 history observations with 32-observation input patches and is the position producing the first future patch; only its first observation is evaluated here. The forecast horizon is one observation of synthetic log variance. The primary scalar is the restored q50 forecast; q10/q90 and width are secondary saved diagnostics. No realized targets are present to make this a forecast-accuracy test.

The layer-0 coordinate is early in depth, but follows nonlinear patch embedding, normalization, recurrent mixing and a SiLU product gate. A near-linear temporal-summary explanation is plausible but untested in the original pilot. Native elementwise gating gives these coordinates an architectural meaning, but these experiments do not compare native against rotated bases and therefore cannot establish Goodfire’s native-basis thesis.

The original 1 milli floor was a numerical-influence choice, not calibrated to measured no-op noise or economic materiality. A new protocol should freeze a reproducibility-based numerical threshold using calibration runs, separately report effect sizes and uncertainty, and avoid implying that statistical detectability means useful prediction.

## Revision schema

1. Preserve original protocol, raw arrays and numerical summary as historical evidence. Add a dated correction/readme pointer and a new retrospective diagnostic report rather than silently rewriting the original decision rule.
2. Replace public README/post headline with motif dependence, U666 concentration and discovery-direction instability. Retain the historical 0/6 result only with its exact criterion and limitations.
3. Add raw and oriented discovery/validation effects, both-direction sign categories, all 18 k/n and raw p-values, and motif-specific diagnostics. Explicitly distinguish U666 from earlier U103.
4. Run fresh, balanced short/long calibration and validation draws under the same law; freeze identities and thresholds before validation. Use a mean-influence estimand with stated distributional assumptions and separately report raw signs. Use random-set references or genuinely sensitivity-matched controls without overstating exchangeability.
5. Use “internal numerical check by a second implementation” throughout. AI assistance accelerates coding and checking; it is not independent scientific replication.

Reproduce: `python scripts/tirex_pilot_diagnostics.py --run-dir <repo>/reports/tirex_group_pilot --output diagnostics.json`. See `RED.log`, `TESTS_FIRST.json`, `GREEN.log`, and full numerical `diagnostics.json`.
