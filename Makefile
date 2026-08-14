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
test:
	$(PY) -m unittest discover -s tests -t . -v

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
