# Use the project venv if it exists (the system python3 may be too old for
# pandas>=2.1 / torch). Override with: make PY=python3.11 <target>
PY=$(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)

.venv/bin/python:
	python3.11 -m venv .venv

setup: .venv/bin/python
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements.txt

smoke:
	$(PY) -m tests.test_smoke

# Run EVERY test module. Before 2026-08-13 there was no such target and no
# tests/__init__.py, so `python -m unittest discover` collected 0 tests and
# printed OK -- most of the suite was unreachable from any make target. A guard
# nobody runs is the same failure as a guard that was never implemented.
# `-p 'test_*.py'` on tests/ only: tests/env/ holds ENVIRONMENT-PREREQUISITE
# checks (large git-ignored caches, model snapshots) which cannot pass on a
# clean clone or a CI runner and are not contracts about the code.
test:
	$(PY) -m unittest discover -s tests -t . -p 'test_*.py' -v

# Environment prerequisites. Run where the git-ignored artifacts actually live.
test-env:
	$(PY) -m unittest discover -s tests/env -t . -p 'env_test_*.py' -v

# Is this box inside requirements.txt for the numerically load-bearing packages?
# A frozen digest over a report does not constrain the code that regenerates it.
check-env:
	$(PY) -m src.envcheck

# Linter only, deliberately no `ruff format`. The formatter would reflow 69
# files that regenerate byte-for-byte-fenced artifacts, which trades a real risk
# of a silent numerical edit for tidier whitespace. The rules that catch actual
# rot -- unused imports, unsorted imports, bugbear, pyflakes -- are enforced;
# the cosmetic bulk is listed with reasons in pyproject.toml and tracked as O9.
lint:
	$(PY) -m ruff check src tests

# Fast subset for pre-commit: the methodology contract (which includes the
# frozen-report digest pin) plus the smoke test.
test-fast: test-methodology smoke

test-signal-safety:
	$(PY) -m unittest tests.test_signal_safety

test-nport-weights:
	$(PY) -m unittest tests.test_nport_weights

test-historical-weights:
	$(PY) -m unittest tests.test_historical_weights

test-skew-carry:
	$(PY) -m unittest tests.test_skew_carry

test-jump-target:
	$(PY) -m unittest tests.test_jump_target

test-regime-transition:
	$(PY) -m unittest tests.test_regime_transition

test-regime-repair:
	$(PY) -m unittest tests.test_regime_repair

test-target-regime: test-jump-target test-regime-transition test-regime-repair

fetch-nport-weights: test-nport-weights
	$(PY) -m src.nport_weights fetch

# Public QQQ annual reports (2004-2018) + existing N-PORT + official Nasdaq
# membership snapshots. Both parser/timing suites must pass before networking.
fetch-historical-weights: test-nport-weights test-historical-weights
	$(PY) -m src.historical_weights fetch

# Frozen post-hoc mechanism diagnostic; never reads the clean window.
fetch-skew-data: test-skew-carry
	$(PY) -m src.skew_carry fetch

skew-carry: test-skew-carry
	$(PY) -m src.skew_carry run

verify-skew-carry:
	$(PY) -m src.verify_skew_carry

# Target-changing studies. Data acquisition/scoring cannot bypass their
# pre-written source, timing, target, HMM-filtering, and clean-fence contracts.
fetch-jump-target: test-jump-target
	$(PY) -m src.jump_target fetch

jump-target: test-jump-target
	$(PY) -m src.jump_target run

regime-transition: test-regime-transition
	$(PY) -m src.regime_transition run

regime-repair: test-regime-repair
	$(PY) -m src.regime_repair run

verify-target-regime:
	$(PY) -m src.verify_target_regime
	$(PY) -m src.verify_regime_repair

verify-regime-repair:
	$(PY) -m src.verify_regime_repair

fetch-free:
	$(PY) -m src.fetch free

# usage: make fetch-polygon START=2016-01-04 END=2026-08-11  (needs POLYGON_API_KEY)
fetch-polygon:
	$(PY) -m src.fetch polygon $(START) $(END)

# usage: make fetch-earnings TICKERS=NVDA:12.7,AAPL:10.7,MSFT:9.0
# TICKER:WEIGHT pairs — the weight is the approximate NDX index weight in
# percent. A bare ticker defaults to weight 0 and contributes nothing.
fetch-earnings:
	$(PY) -m src.fetch earnings $(TICKERS)

# alternative earnings source; needs FMP_API_KEY
fetch-earnings-fmp:
	$(PY) -m src.fetch earnings-fmp $(TICKERS)

# review calendars/earnings_fetched.csv, then promote it to earnings_top.csv
merge-earnings:
	$(PY) -m src.fetch merge-earnings

features:
	$(PY) -m src.experiment features --source $(or $(SOURCE),daily)

baselines:
	$(PY) -m src.experiment baselines --phase $(or $(PHASE),all) --source $(or $(SOURCE),daily)

chronos-clean:
	$(PY) -m src.experiment chronos --phase clean --variant uni --source $(or $(SOURCE),daily)
	$(PY) -m src.experiment chronos --phase clean --variant cov --source $(or $(SOURCE),daily)
	$(PY) -m src.experiment chronos --phase clean --variant cov_iv --source $(or $(SOURCE),daily)

evaluate:
	$(PY) -m src.experiment evaluate --phase $(or $(PHASE),clean) --source $(or $(SOURCE),daily)

# --- Corrected-methodology fork ------------------------------------------
# reports/METHODOLOGY_FORK.md. Additive: the four corrections land in parallel
# `_est`/`_inf`/`_v2` reports and the frozen `results_{phase}.md` are never
# touched. test-methodology asserts exactly that, so it gates the run rather
# than merely accompanying it.
test-methodology:
	$(PY) -m unittest tests.test_methodology

# Re-pin the frozen-report digests. REQUIRED after a forward-accrual round:
# `daily-update` legitimately extends results_clean.md with new origins, which
# breaks the SHA-256 pin, which fails test-methodology, which blocks
# baselines-smearing / scenarios / scenarios-all. Without this target the whole
# corrected fork becomes unrunnable after every accrual -- a deadlock introduced
# with the pin itself on 2026-08-13.
#
# This is deliberately a separate, explicit command and NOT a dependency of
# anything. Re-pinning is an amendment to a frozen pre-registered artifact: run
# it only when you intend the frozen reports to have changed, and log why in
# reports/AMENDMENTS.md. Never run it to make a red test go green.
repin-frozen-reports:
	@echo "Re-pinning frozen report digests. This is an AMENDMENT."
	@echo "Log the reason in reports/AMENDMENTS.md before committing."
	$(PY) -c "import hashlib, json, pathlib; \
files = ['results_clean.md', 'results_diagnostic.md', 'results_clean_dec.md']; \
d = {f: hashlib.sha256((pathlib.Path('reports')/f).read_bytes()).hexdigest() for f in files}; \
p = pathlib.Path('reports/FROZEN_REPORT_HASHES.json'); \
m = json.loads(p.read_text()); m['sha256'] = d; \
p.write_text(json.dumps(m, indent=2) + chr(10)); \
[print(f'  {k}  {v}') for k, v in d.items()]"

# Post-result, additive: measures how much of the registered ranking statistic
# is between-fold rather than within-fold. Refits nothing, rewrites no verdict.
pooling-diagnostic:
	$(PY) -m src.pooling_diagnostic

test-residual-probe:
	$(PY) -m unittest tests.test_residual_probe

# Post-result, additive: does anything in TiRex's latent state survive
# projecting out HAR realized-volatility history and still rank transitions
# within fold? Gated on its pre-written contracts, which fence the three
# separate per-fold fits against leakage.
residual-probe: test-residual-probe
	$(PY) -m src.residual_probe

# Exact-smearing forecasts for every model with recoverable residuals.
# Chronos-2/TiRex-2 have none and are reconstructed at evaluate time.
baselines-smearing: test-methodology
	$(PY) -m src.experiment baselines --phase $(or $(PHASE),all) --estimator smearing --source $(or $(SOURCE),daily)

# The full 2x2: frozen, estimator-only, inference-only, both. Each correction
# is readable in isolation instead of as one undifferentiated "corrected".
# The first line rewrites the FROZEN report; it is in the list deliberately, so
# every scenario run re-proves that the default path still reproduces it.
EV=$(PY) -m src.experiment evaluate --phase $(or $(PHASE),clean) --source $(or $(SOURCE),daily) $(EXTRA)

scenarios: test-methodology
	$(EV)
	$(EV) --estimator smearing
	$(EV) --inference corrected
	$(EV) --estimator smearing --inference corrected

# Every scenario on every window, including the decile grid: 12 reports.
scenarios-all: test-methodology
	$(MAKE) scenarios PHASE=clean
	$(MAKE) scenarios PHASE=diagnostic
	$(MAKE) scenarios PHASE=clean EXTRA="$(GRID)"

# --- TiRex-2 comparison -------------------------------------------------
# TiRex-2 emits only deciles (0.1..0.9), and mean_var is not comparable across
# quantile grids, so the whole model set is recomputed on that grid into a
# parallel `_dec` report. config.yaml is never modified. See
# reports/LEAKAGE_TIREX2.md before reading any TiRex result.
GRID=--quantile-grid deciles

tirex-clean:
	$(PY) -m src.experiment tirex --phase clean --variant uni    $(GRID) --source $(or $(SOURCE),daily)
	$(PY) -m src.experiment tirex --phase clean --variant cov    $(GRID) --source $(or $(SOURCE),daily)
	$(PY) -m src.experiment tirex --phase clean --variant cov_iv $(GRID) --source $(or $(SOURCE),daily)

# full decile-grid comparison set: baselines + chronos + tirex, then evaluate
deciles-all:
	$(PY) -m src.experiment baselines --phase all $(GRID) --source $(or $(SOURCE),daily)
	$(PY) -m src.experiment chronos --phase clean --variant uni    $(GRID) --source $(or $(SOURCE),daily)
	$(PY) -m src.experiment chronos --phase clean --variant cov    $(GRID) --source $(or $(SOURCE),daily)
	$(PY) -m src.experiment chronos --phase clean --variant cov_iv $(GRID) --source $(or $(SOURCE),daily)
	$(MAKE) tirex-clean
	$(PY) -m src.experiment evaluate --phase clean $(GRID) --source $(or $(SOURCE),daily)

# forward accrual: refresh data, rebuild features, extend forecasts, re-score
daily-update: fetch-free features baselines chronos-clean evaluate

# point-in-time index weights from market cap (removes the snapshot look-ahead)
pit-weights:
	$(PY) -m src.fetch pit-weights

# pre-registered conditional-vs-unconditional carry study (diagnostic window only)
carry:
	$(PY) -m src.carry

# Diagnostic-only orthogonal-signal study. Both empirical stages are gated on
# the pre-written safety suite; confirmation also requires the discovery lock.
fetch-signal-inputs:
	$(PY) -m src.fetch signal-inputs

signals-discover: test-signal-safety
	$(PY) -m src.signal_study discover

signals-confirm: test-signal-safety
	$(PY) -m src.signal_study confirm

verify-signals:
	$(PY) -m src.verify_signal_results

# Additive exploratory search on already-open history; never uses the sealed phase.
test-orthogonal-round2:
	$(PY) -m unittest tests.test_orthogonal_round2 tests.test_verify_orthogonal_round2 tests.test_round2_inference -v

orthogonal-round2: test-orthogonal-round2
	$(PY) -m src.orthogonal_round2 calibrate
	$(PY) -m src.orthogonal_round2 run
	$(PY) -m src.verify_orthogonal_round2

verify-orthogonal-round2: test-orthogonal-round2
	$(PY) -m src.verify_orthogonal_round2

test-model-memory-study:
	$(PY) -m unittest tests.test_model_memory_estimators tests.test_model_memory_study tests.test_verify_model_memory_study -v

model-memory-study: test-model-memory-study
	$(PY) -m src.model_memory_study
	$(PY) -m src.verify_model_memory_study

verify-model-memory-study: test-model-memory-study
	$(PY) -m src.verify_model_memory_study

test-model-memory-reference:
	$(PY) -m unittest tests.test_model_memory_reference tests.test_reference_xlstm tests.test_reference_moirai tests.test_verify_model_memory_reference -v

# These extraction targets require the separately pinned local model environment.
# Extraction refuses to overwrite existing hash-audited artifacts.
model-memory-moirai: test-model-memory-reference
	data/model_memory_reference/moirai2_env/bin/python -m src.reference_moirai synthetic-check
	data/model_memory_reference/moirai2_env/bin/python -m src.reference_moirai extract

model-memory-neural: test-model-memory-reference
	$(PY) -m src.run_reference_xlstm

# Uses completed core, neural and Moirai local caches.
model-memory-reference: test-model-memory-reference
	$(PY) -m src.model_memory_reference
	$(PY) -m src.verify_model_memory_reference

verify-model-memory-reference: test-model-memory-reference
	$(PY) -m src.verify_model_memory_reference

test-iterative-signal-search:
	$(PY) -m unittest tests.test_iterative_signal_search tests.test_iterative_hf tests.test_iterative_index tests.test_verify_iterative_signal_search tests.test_round2_inference -v

iterative-signal-search: test-iterative-signal-search
	$(PY) -m src.iterative_signal_search
	$(PY) -m src.verify_iterative_signal_search

verify-iterative-signal-search: test-iterative-signal-search
	$(PY) -m src.verify_iterative_signal_search

test-international-volatility:
	$(PY) -m unittest tests.test_international_search tests.test_international_volatility tests.test_verify_international_volatility tests.test_round2_inference -v

international-volatility: test-international-volatility
	$(PY) -m src.international_search
	$(PY) -m src.verify_international_volatility

verify-international-volatility: test-international-volatility
	$(PY) -m src.verify_international_volatility

test-overnight-index:
	$(PY) -m unittest tests.test_overnight_search tests.test_overnight_index tests.test_verify_overnight_index tests.test_round2_inference -v

overnight-index: test-overnight-index
	$(PY) -m src.overnight_search
	$(PY) -m src.verify_overnight_index

verify-overnight-index: test-overnight-index
	$(PY) -m src.verify_overnight_index

.PHONY: test-volatility-source-audits
test-volatility-source-audits:
	$(PY) -m unittest tests.test_audit_harnet_extension tests.test_audit_risklab_source -v

.PHONY: test-macro-overnight macro-overnight verify-macro-overnight
test-macro-overnight:
	$(PY) -m unittest tests.test_macro_second_moment tests.test_macro_plan_features tests.test_macro_overnight tests.test_macro_search tests.test_verify_macro_overnight tests.test_round2_inference -v

macro-overnight: test-macro-overnight
	$(PY) -m src.macro_search
	$(PY) -m src.verify_macro_overnight

verify-macro-overnight: test-macro-overnight
	$(PY) -m src.verify_macro_overnight

.PHONY: test-measurement-memory measurement-memory verify-measurement-memory
test-measurement-memory:
	$(PY) -m unittest tests.test_measurement_memory tests.test_measurement_search tests.test_measurement_publication tests.test_verify_measurement_memory tests.test_macro_second_moment tests.test_round2_inference -v

measurement-memory: test-measurement-memory
	$(PY) -m src.measurement_search
	$(PY) -m src.verify_measurement_memory

verify-measurement-memory: test-measurement-memory
	$(PY) -m src.verify_measurement_memory

.PHONY: index-hinge verify-index-hinge plot-index-hinge
index-hinge:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 $(PY) -m src.index_hinge_search

verify-index-hinge:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 $(PY) -m src.verify_index_hinge

plot-index-hinge:
	$(PY) -m src.plot_index_hinge

.PHONY: tail-shape verify-tail-shape plot-tail-shape
tail-shape:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 $(PY) -m src.tail_shape_search

verify-tail-shape:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 $(PY) -m src.verify_tail_shape

plot-tail-shape:
	$(PY) -m src.plot_tail_shape

.PHONY: calendar-variance verify-calendar-variance
calendar-variance:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 $(PY) -m src.calendar_variance_search

verify-calendar-variance:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 $(PY) -m src.verify_calendar_variance

.PHONY: relative-risk verify-relative-risk
relative-risk:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 $(PY) -m src.relative_risk_search

verify-relative-risk:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 $(PY) -m src.verify_relative_risk

.PHONY: joint-risk verify-joint-risk
joint-risk:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 $(PY) -m src.joint_risk_search

verify-joint-risk:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 $(PY) -m src.verify_joint_risk

.PHONY: cross-moment verify-cross-moment
cross-moment:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 $(PY) -m src.cross_moment_search

verify-cross-moment:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 $(PY) -m src.verify_cross_moment

.PHONY: target-aligned verify-target-aligned
target-aligned:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 $(PY) -m src.target_aligned_search

verify-target-aligned:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 $(PY) -m src.verify_target_aligned

.PHONY: sign-memory verify-sign-memory
sign-memory:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 $(PY) -m src.sign_memory_search

verify-sign-memory:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 $(PY) -m src.verify_sign_memory

.PHONY: causal-pool verify-causal-pool
causal-pool:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 $(PY) -m src.causal_pool_search

verify-causal-pool:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 $(PY) -m src.verify_causal_pool

.PHONY: civil-quarter verify-civil-quarter
civil-quarter:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 $(PY) -m src.civil_quarter_search

verify-civil-quarter:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 $(PY) -m src.verify_civil_quarter

.PHONY: civil-quarter-replay verify-civil-quarter-replay
civil-quarter-replay:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 $(PY) -m src.civil_quarter_replay_search

verify-civil-quarter-replay:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 $(PY) -m src.verify_civil_quarter_replay

.PHONY: profiled-quarter verify-profiled-quarter
profiled-quarter:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 $(PY) -m src.profiled_quarter_search

verify-profiled-quarter:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 $(PY) -m src.verify_profiled_quarter

.PHONY: range-alert verify-range-alert
range-alert:
	MPLCONFIGDIR=/tmp/ndx-vol-mplcache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 $(PY) -m src.range_alert_search

verify-range-alert:
	MPLCONFIGDIR=/tmp/ndx-vol-mplcache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 $(PY) -m src.verify_range_alert

.PHONY: issued-calibration verify-issued-calibration
issued-calibration:
	MPLCONFIGDIR=/tmp/ndx-vol-mplcache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 $(PY) -m src.issued_calibration_search

verify-issued-calibration:
	MPLCONFIGDIR=/tmp/ndx-vol-mplcache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 $(PY) -m src.verify_issued_calibration
