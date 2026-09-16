# Measurement-memory source and feature compatibility

**PASS before model fitting.** After all 19 prewritten producer contracts passed, the producer's bounded source loader, features, targets, and complete source audit were compared with the independently written reconstruction. That reconstruction does not import the measurement-memory producer or its frozen Risk Lab parser. All 13 compatibility check groups passed.

The comparison covers 6,696 reference-calendar rows and 16 numerical features: 107,136 feature cells. It also covers all three next-session measurement targets: 20,088 target cells. Numerical values and missingness match exactly, as do feature/target column order, index labels, both feature cutoff dates, target end dates, and delayed label availability dates. The complete nested source audit also matches exactly.

The source has 7,680 dated records, of which 7,473 are inside the numerical cutoff. The source audit confirms that no numerical source fields after 2025-10-20 were parsed. Full raw-source bytes were hashed for identity; date metadata outside the numerical fence was used only for source count and extent checks.

The fingerprints independently supplied before this producer comparison were reproduced exactly:

| Object | SHA256 |
|---|---|
| Sixteen numerical feature columns | `45b83eaed5ed3a796bac544e76b719fe4231ec40f0e44c4a7ef313010ed5dc1e` |
| Three numerical target columns | `084cbd2f2aeee15d65399a09ceb626388826f35bbe3302e0a8ba4f2cb88e3b8a` |
| Complete canonical source audit | `bdd2ff976bb8587d1891e6f67b98aceb657797185ad7adb119479086da83760f` |

Numerical fingerprints use C-contiguous little-endian Float64 array bytes in the declared column order. The source audit fingerprint uses UTF-8 bytes from `json.dumps(sort_keys=True, allow_nan=False)`. Exact agreement includes NaN positions; separate equality checks verify all date-valued fields.

The ignored `data/measurement_memory/source_compatibility.json` records all checks, dimensions, schemas, fingerprint methods, protocol identity at this check, and hashes of the producer, its 19-test file, frozen Risk Lab parser, independent verifier, and independent test file. The producer and its tests were not changed by this comparison.

This is implementation and source compatibility, not predictive validation. There were zero empirical fits, forecast scores, or predictor–target association statistics. No source corpus or earlier code/report was modified. The source's historical availability, revisions, calendar limitations, and research-only use remain unresolved as stated in the registered design.
