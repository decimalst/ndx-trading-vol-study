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

test-signal-safety:
	$(PY) -m unittest tests.test_signal_safety

test-nport-weights:
	$(PY) -m unittest tests.test_nport_weights

fetch-nport-weights: test-nport-weights
	$(PY) -m src.nport_weights fetch

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
