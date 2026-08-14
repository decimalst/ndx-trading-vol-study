# Runbook — how to reproduce every result

*Split out of `README.md` on 2026-08-13 to keep the top-level document short.
Nothing here changed in the move; it is the same text.*

## Run order

Needs **Python 3.11+** (pandas ≥2.1 and torch). `make setup` creates `.venv`
with `python3.11` and every other target uses it automatically; override with
`make PY=/path/to/python <target>`.

```bash
make setup                      # creates .venv, installs requirements.txt
make smoke                      # synthetic end-to-end check, no data needed
make fetch-free                 # yfinance daily OHLC + CBOE VXN
# calendars/{fomc,cpi,nfp}.csv are checked in and verified through 2026
# (FOMC through 2027) — extend them from the source pages when they run out.
# For earnings, first save the Invesco QQQ "Complete Holdings" CSV to
# data/raw/qqq_holdings_YYYY-MM-DD.csv (the site blocks scripted downloads):
make fetch-earnings             # yfinance; weights read from that holdings file
#    review calendars/earnings_fetched.csv (esp. the session column), then:
make merge-earnings             # -> calendars/earnings_top.csv
make features                   # SOURCE=bars after a polygon fetch, else GK daily
make baselines PHASE=all
make chronos-clean              # ~190 origins x 3 variants; ~2 min on CPU
make evaluate PHASE=clean       # -> reports/results_clean.md
make evaluate PHASE=diagnostic  # baseline sanity on the long window
```

`baselines PHASE=all` writes `*_all.parquet`, and `evaluate` slices those to the
requested phase. Every model refits per origin from history up to that origin, so
slicing an `all` run is identical to having run the phase directly.

Forward accrual: `make daily-update` (cron it after the close if you want, e.g.
`30 18 * * 1-5`). Infrastructure is deliberately one box + parquet + Make — no
cloud, no scheduler, nothing to babysit. State lives in `data/`, results in
`reports/`, and `config.yaml` is the pre-registration artifact.

The five-path extension is separate from the frozen harness and runs its safety
tests before every empirical command:

```bash
.venv/bin/python -m unittest tests.test_research_paths -v
.venv/bin/python -m src.research_paths top25
.venv/bin/python -m src.research_paths fetch-external
.venv/bin/python -m src.research_paths fetch-top25-earnings
.venv/bin/python -m src.research_paths absorption-map
.venv/bin/python -m src.research_paths horizon-curve
.venv/bin/python -m src.research_paths vrp-term-structure
.venv/bin/python -m src.research_paths single-name-earnings
.venv/bin/python -m src.research_paths spx-term-slope
.venv/bin/python -m src.verify_research_paths
```

The later functional-form and representation studies are isolated from the
original harness and each has its own machine-readable protocol:

```bash
.venv/bin/python -m unittest tests.test_history_extension -v
.venv/bin/python -m src.history_extension build
.venv/bin/python -m unittest tests.test_gbm_study tests.test_gbm_post_result -v
.venv/bin/python -m src.gbm_study verify
.venv/bin/python -m src.gbm_post_result verify
.venv/bin/python -m unittest tests.test_representation_study tests.test_latent_probe_study tests.test_noise_robustness -v
.venv/bin/python -m src.representation_study verify-tail
.venv/bin/python -m src.latent_probe_study verify
.venv/bin/python -m unittest tests.test_latent_k1_confirmation -v
.venv/bin/python -m src.latent_k1_confirmation verify
.venv/bin/python -m src.noise_robustness verify
```

The 1999 panel is price-only: the current free Cboe VXN file begins in September
2009, so no proxy splice was used. The tail benchmark intentionally inherits the
earlier transition study's `mean(log RV)` 5/22-session convention rather than
standard HAR's `log(mean variance)`; that distinction is explicit in its report.


## The corrected-methodology fork (added 2026-08-13)

The commands above produce the **frozen pre-registered** reports. The
corrections in [`reports/METHODOLOGY_FORK.md`](../reports/METHODOLOGY_FORK.md)
write parallel reports and never touch those:

```bash
make test-methodology     # the fork's own contract; gates everything below
make baselines-smearing PHASE=all   # exact Duan-smearing forecasts -> *_sm.parquet
make scenarios PHASE=clean          # frozen + _est + _inf + _v2 for one window
make scenarios-all                  # all 12 reports: clean, diagnostic, decile grid
make pooling-diagnostic             # what the annual-fold ranking scoreboard measures
```

| file | estimator | inference | isolates |
|---|---|---|---|
| `results_{phase}.md` | `trunc` | `naive` | the frozen pre-registration |
| `results_{phase}_est.md` | `smearing` | `naive` | the point-forecast fix alone |
| `results_{phase}_inf.md` | `trunc` | `corrected` | the inference fixes alone |
| `results_{phase}_v2.md` | `smearing` | `corrected` | both |

`make scenarios` deliberately regenerates the frozen report first, so every run
re-proves that the default path still reproduces it byte-for-byte.

## Running the tests

```bash
make test          # every module (~350 tests). Use this.
make test-fast     # methodology contract + smoke, for pre-commit
make smoke         # synthetic end-to-end, needs no data
```

Before 2026-08-13 there was no `make test` and no `tests/__init__.py`, so
`python -m unittest discover` collected **0 tests and printed OK**. CI now
asserts a minimum collected-test count for exactly that reason — a suite that
runs nothing must fail, not pass.

Some assertions skip when `data/processed/master_daily.parquet` is absent. A
skip is not a pass; CI prints the skip list to the job summary.

## Known technical limitations of the harness

- `predict_df` gets synthetic contiguous timestamps per origin (trading days are
  irregular); mapping back to real dates is positional and exact.
- 30-day distributional evaluation for Chronos is out of scope — quantiles do not
  sum across steps. The cumulative point forecast sums per-step means, which is
  exact for the mean regardless of dependence.
- Chronos's 30-calendar-day cumulative forecast needs 21 future covariate rows,
  which only exist for dates already in the master frame. The newest ~20 origins
  therefore get `log_cum_var_hat = NaN` (h=1 is unaffected). They fill in as data
  accrues; their 30-day targets are not realized yet anyway.
- GK-based RV is noisier than 5-minute RV. QLIKE is chosen partly because it is
  robust to proxy noise, but buy the intraday history if results get close.

## Forward accrual, and what goes stale

`make daily-update` refreshes the **frozen clean-window path only** —
`fetch-free features baselines chronos-clean evaluate`, and `evaluate` defaults
to `PHASE=clean`. Everything else beside it goes stale silently:

- `results_diagnostic.md` until `make evaluate PHASE=diagnostic`.
- The whole `_est`/`_inf`/`_v2` fork until `make baselines-smearing` **and then**
  `make scenarios-all`. The `_est`/`_v2` reports are scored off `*_sm.parquet`,
  which the loader prefers whenever it exists (`src/experiment.py:_load_forecast`),
  so skipping the smearing rebuild leaves them scored on **fewer origins** than
  `results_clean.md` — with no warning.

Two sharp edges in that recovery:

1. **The digest pin blocks the fork after every accrual round.**
   `baselines-smearing`, `scenarios` and `scenarios-all` are gated on
   `test-methodology`, which pins the SHA-256 of the three frozen reports. An
   accrual round legitimately rewrites `results_clean.md`, so those targets
   refuse to run until you re-pin:

   ```bash
   make repin-frozen-reports    # then log the amendment in reports/AMENDMENTS.md
   ```

   That is deliberate friction, not a bug — extending a frozen report is an
   amendment. But it means accrual is a **three-step** operation:
   `daily-update` → `repin-frozen-reports` → `baselines-smearing && scenarios-all`.

2. **The decile grid has no smearing target.** `baselines-smearing` passes no
   `--quantile-grid`, so `make deciles-all` restores the frozen decile reports
   but not `*_dec_sm.parquet`. The decile `_est` and `_v2` reports need:

   ```bash
   .venv/bin/python -m src.experiment baselines --phase all --estimator smearing --quantile-grid deciles
   ```

## Test layout, and what CI does not collect

```bash
make test        # code contracts. This is what CI runs.
make test-env    # environment prerequisites. Run where the artifacts live.
make check-env   # is this box inside requirements.txt?
make lint        # ruff check (no formatter -- see pyproject.toml)
```

`tests/env/` holds checks that assert **large git-ignored inputs exist on this
machine**: the TiRex embedding chunk store, the raw HuggingFace option-flow
shards, the cached Chronos-2 / TiRex-2 model snapshots. They cannot pass on a
clean clone or a CI runner and they are not claims about whether the code is
correct, so `make test` and CI do not collect them.

They used to sit in the ordinary suite behind guards that checked the **wrong
precondition** — `METRICS_PATH.exists()` is true on a clean clone because the
metrics JSON is committed, while the verify path it gated then read
`latent_embedding_chunks/manifest.json`, which `.gitignore` excludes. Guard
passed, code errored, fresh checkout saw a hard failure unrelated to the
science. Splitting them out is cleaner than adding more guards.

## If artifacts stop reproducing, check the environment first

`reports/FROZEN_REPORT_HASHES.json` pins report bytes, but a digest over an
output does not constrain the code that produced it. A reviewer on
scikit-learn 1.8 — outside the `>=1.7,<1.8` pin — hit three GBM artifact
failures reading `timing-safe metrics do not recompute from saved forecasts`,
which looks like a corrupted artifact and is not.

```bash
make check-env      # names any load-bearing package outside its pin
```

`src/envcheck.py` now asserts the pins, `tests/test_environment.py` fails
loudly when they drift, and the artifact verifiers append the diagnosis to
their own error messages. **A recompute mismatch on an unpinned environment is
an environment problem until proven otherwise.**
