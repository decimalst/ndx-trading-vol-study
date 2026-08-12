# Independent verification of the five-path extension

**PASS.** This verifier does not import `src.research_paths`. It reconstructs
source hashes, quarterly issuer rankings, acceptance-time fences, the full
absorption regressions, all forecast losses, VRP targets, single-name event
effects, and the SPX registered verdict from persisted source artifacts.

- Source files matched all 12 recorded hashes.
- Quarterly universe: 27 snapshots, 675 rows, 578 eligible realized events.
- Leverage anchor independently reconstructs Wald 103.1875 → 62.6305.
- Horizon curve: 6 frozen horizons recomputed.
- VRP common sample: 3656 origins.
- Single-name pool: 3 eligible assets; effect +2.0495 log variance.
- SPX term-slope verdict: FAIL on 503 origins.
