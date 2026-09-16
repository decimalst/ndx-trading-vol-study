# Treasury inference review before empirical access

**No blocking mathematical or fixed-configuration discrepancy found in the inspected versions.** This bounded review read the unregistered protocol, complete inference/calibration implementations, twelve generated tests and saved RED/GREEN description. It did not run calibration, tests, historical arrays, support counts, models or scores. Source recovery review and all frozen outputs remain unchanged.

## Full-calendar endpoints and uncertainty

For a full calendar of T positions and mask count n, the implementation calculates the selected paired mean μ and influence m[t](d[t]−μ)/(n/T). False-mask entries receive internal zero contributions without multiplying missing losses by zero. HAC covariance products use original calendar distances and denominator T; the final standard error divides the Bartlett long-run variance by T before taking its square root. Therefore missing dates do not become adjacent observations and the ratio-mean normalization is preserved. The core now rejects every negative computed HAC variance instead of clamping it. Its saved twelfth generated regression demonstrates that failure path.

The circular bootstrap samples the mask and masked loss difference with the same full-calendar block starts. Each draw has exactly T sampled calendar positions, using complete blocks and one remainder; the estimate is the sampled numerator divided by the sampled mask count. It neither fixes the denominator to observed n nor samples compressed auction rows. The zero-support rule rejects the entire endpoint, with no replacement draw or deletion. Chunking preserves the literal full-starts-then-tail RNG sequence tested by the explicit-index oracle. The centered, two-sided plus-one p-value and percentile mean intervals match the written core contract. The maximum p across HAC and all three blocks is conservative relative to each included method; it is not a substitute for the separate endpoint/phase, effect, support, stability and family gates.

The API has no dates. The forthcoming study wrapper must still supply the complete intended calendar and correct paired daily/activation masks. A boolean activation mask weights simultaneous reports once as an information date, while the event statistics can contain their aggregate. Neither this primitive nor its two-observation domain minimum establishes the protocol's phase, offset, training, activation or per-tenor support.

## Fixed null diagnostic

The calibration script agrees with the prospective values: 1,500 retained sessions after 200 burn-in positions, Gaussian AR(1) rho 0.8, 200 repetitions, 499 bootstrap draws, blocks 21/63/126, HAC126, base seed 20261002, active residues 0/1/7/8/14/15 modulo21, and the 126-position missing interval [600,726). The masks preserve 1,374 daily observations and 394 activation observations on the same 1,500-position calendar. Each repetition draws a new innovation sequence; the two endpoints use the same generated series.

For each endpoint/repetition, the interval envelope uses the smallest lower and largest upper endpoint among HAC and all three bootstrap intervals. Both endpoint coverages must be at least 0.90. Failed calls count against coverage and are retained; the script additionally requires zero failures for a passing calibration. Individual-method coverage and failure counts remain visible. Input code/tests/protocol hashes are recorded and checked again before exclusive output creation. A failed diagnostic is not retried or tuned in this script.

The declared seed schedule is reproducible, but bootstrap seeds are not unique across all repetitions/methods: base+10000+2*repetition+endpoint plus block makes, for example, repetition0/block63 equal repetition21/block21 within an endpoint. This is shared resampling randomness across independently generated series, not independent Monte Carlo streams for every call. It does not change marginal endpoint coverage or violate a stated uniqueness condition, and this bounded diagnostic makes no binomial Monte Carlo-error claim. The parent was informed during this pre-empirical review; the first fixed calibration is preserved and no seed change is required by the inspected contract.

The diagnostic only evaluates its specified stationary Gaussian null and deterministic missing/pulse pattern. A passing envelope does not guarantee nominal market coverage, handle every missingness process, certify the very small family-adjusted tail probabilities, or establish power. The 499 diagnostic draws assess nominal intervals; the separately fixed empirical399999 draws remain necessary for the registered inference resolution. The wave alpha equals .05/(25*26), and the 144 inherited plus two prospective contrasts total146 only upon registration.

## Inspected pins

| File | SHA256 |
| --- | --- |
| `treasury_dealer.yaml` | `194fd2ef49cfa8db5542f5e672383b5e2d4ec8f8c724b2af8559d51319aef6a8` |
| `scripts/calibrate_treasury_dealer_inference.py` | `be2a270d5062a430d8515f20b0b70b1585700ac8872f75dfee7f1e8a60d46bf8` |
| `src/treasury_dealer_inference.py` | `e96f398594e1bc59d3e7c9b4fd4f76fcc68cc8e3a0e2338923a0c8498174992e` |
| `tests/test_treasury_dealer_inference.py` | `82eb7ae026c5bef2a473257cb52e9b4838e0417023126a2300afdc441c4ecec4` |
| `reports/treasury_dealer/INFERENCE_CORE_CONTRACT.md` | `4893d3051ddaad68af19a01e66c85c57485b589aaa6af37f429ac66af1b0a385` |
| `reports/treasury_dealer/INFERENCE_CORE_GREEN.log` | `9c49ef7be2c165f639dd084d44c6f339d82529a8c8e0e1fd3087bec71f0512f3` |

## Saved first calibration: independent summary check

The parent completed the first fixed generated-null run. This review subsequently verified the saved result SHA256, all four recorded code/test/script/protocol pins, exact fixed configuration, and all400 unique endpoint/repetition records (200 daily and200 active). It reconstructed every individual coverage flag and the combined interval envelope from the saved interval endpoints, checked their counts and summaries, and confirmed the recorded pass decision. No generated series, resampling draws or inference calculations were rerun.

| Endpoint | Envelope | HAC126 | Block21 | Block63 | Block126 | Failed repetitions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Daily | 186/200 = 93.0% | 89.5% | 89.5% | 89.0% | 88.5% | 0 |
| Activation date | 188/200 = 94.0% | 92.0% | 92.0% | 91.5% | 90.5% | 0 |

The result is **SYNTHETIC_CALIBRATION_PASS** because the preregistration diagnostic criterion was both envelope coverages at least90%, with no failed repetitions. The criterion was not95% empirical coverage, nor90% for every individual method. The table does **not** show exact nominal95% coverage: both envelopes are below95%, and daily individual-method coverage is below90%. The repeated resampling streams noted above remain part of this fixed result. Nothing here guarantees market-data coverage, adjusted-tail calibration or model benefit. Preserve this first result without changing seeds, settings, masks or acceptance criteria in response to its coverage.

Saved calibration: `reports/treasury_dealer/INFERENCE_NULL_CALIBRATION.json`, SHA256 `7992298ec031139d2d249fc74fad5115bd816ac8e0fa78f86e866220f0840acf`. Historical access is recorded false and new predictive comparisons zero. The full study's prospective source, support, maturity, model, multiplicity and independent verification gates remain separate.
