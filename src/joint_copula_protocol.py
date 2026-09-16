"""Exact prospective wave27 specification and admitted execution inventory."""

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA256 = "cf43700c6c37ba860e9ef8ae4adb4c70158030efeedd3bca9b073f0691cd5d50"
EVIDENCE_CLASS = "EXPLORATORY_REUSED_MARKET_HISTORY_CURRENT_VINTAGES"
EVIDENCE_LIMITATION = (
    "Historical development and evaluation windows have been reused. "
    "Prior regression tests accessed the period beginning 2025-11-03; it is not untouched confirmation. "
    "Current vendor snapshots do not certify complete point-in-time price or implied-volatility vintages. "
    "Numerical inputs in this experiment end 2025-10-20. "
    "QQQ ETF and SPX price index are distinct instruments; exact synchronized auctions "
    "and historical publication latency are not certified."
)
EXECUTION = {
    "reports": "reports/joint_copula/predictive",
    "data": "data/model_memory_study/joint_copula_wave27",
    "prior": "reports/precision_gate/predictive/metrics.json",
    "prior_sha256": "3ad635e31bf81f224ccf690bb0efd109b8f0b23dea6fc7bfb30d6073b7ffde58",
    "prior_freeze": "reports/precision_gate/predictive/freeze_record.json",
    "prior_freeze_sha256": "3ff970b3e4fa0ed77e862cd1f1fe2f8e872378f643bd1366f4fa81e7e4c95b43",
    "prior_terminal": "reports/precision_gate/predictive/terminal.json",
    "prior_terminal_sha256": "85e66cca0c8750fb994fbb5e11256ada8a274206487564c7a7e8b186b00030bf",
    "prior_publication": "reports/precision_gate/predictive/PUBLICATION_AUDIT.json",
    "prior_publication_sha256": "f950acc71bf77fe995fa08bef9ef2d0066a92ad79cac22ff7bbc370038d329d5",
    "prior_audit": "reports/precision_gate/predictive/PUBLICATION_INDEPENDENT_AUDIT.json",
    "prior_audit_sha256": "b3322c02f6906345c1b2998d37d583b15fc477cfb9a59deff081c4efb9474aa5",
    "calibration": "reports/treasury_dealer/INFERENCE_NULL_CALIBRATION.json",
    "calibration_sha256": "7992298ec031139d2d249fc74fad5115bd816ac8e0fa78f86e866220f0840acf",
    "market_pins": {
        "data/research_paths/spx_daily.parquet": "3958fbb1eb36689df1596c26b9f0e02e3f1d4fa3b0032381e5287a2a03697bf0",
        "data/raw/daily_ohlc.parquet": "710290d8ad8569172559334b23e423b50ecf7cd10f0fd8ec24a9fc29608bca4d",
        "data/free_sources/raw/cboe/VIX_History.csv": "a34aabce269632f30904cf482986dd50b6d4cf51f2203dc51a0a9f460f3c90b2",
        "data/free_sources/raw/cboe/VIX9D_History.csv": "0d6f600ee71bf6ffb5069d0c583cbe0b1ec97df6da4e4e439d5f0f22f5616abe",
        "data/free_sources/raw/cboe/VVIX_History.csv": "f6bc726455fa3859c662875a005e97ba656b9536ae58fa6977ad24c6228e4f6a",
        "data/free_sources/raw/cboe/VXN_History.csv": "753b5a406a7a888a9e4ca1a3e37a71fbb415a3883bc4b02ff20dbf0c5bdc6c98",
    },
    "proposal_pins": {
        "reports/precision_gate/predictive/NEXT_STUDY_HANDOFF.json": "f04ad7ec728b7cb85ae6816ab5801a9f51f102619151e9e4db344dcdac67fb20",
        "reports/next_signal_review/AFTER_PRECISION_GATE.md": "acdd2023b0846a68d12afbfd8a09ebf23b1b19829c39ed60af4e5b7ec741e2ab",
    },
    "forecast_atol": 1e-12,
    "forecast_rtol": 1e-07,
    "numerical_atol": 1e-12,
    "numerical_rtol": 1e-09,
    "score_atol": 1e-10,
    "score_rtol": 1e-08,
    "probability_atol": 1e-12,
    "probability_rtol": 1e-08,
    "dependence_objective_atol": 1e-10,
    "dependence_objective_rtol": 1e-09,
    "dependence_gradient_limit": 1e-07,
    "global_gap_limit": 1e-08,
    "marginal_oracle_tolerances": {
        "mean_coefficients_and_predictions": {"atol": 1e-10, "rtol": 1e-07},
        "variance_coefficients": {"atol": 1e-06, "rtol": 1e-07},
        "variance_predictions": {"atol": 1e-12, "rtol": 1e-06},
        "variance_geometry_and_objective": {"atol": 1e-12, "rtol": 1e-09},
        "kkt_diagnostics": {"atol": 1e-12, "rtol": 0.0001},
    },
}
PIPELINE_FIELDS = (
    "origin_start",
    "origin_end",
    "source_end",
    "development",
    "evaluation",
    "minimum_train",
)


def load_protocol():
    payload = (ROOT / "joint_copula.yaml").read_bytes()
    if hashlib.sha256(payload).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Original joint-copula protocol changed")
    return yaml.safe_load(payload)


def validate(protocol):
    try:
        actual = json.dumps(protocol, sort_keys=True, allow_nan=False)
        expected = json.dumps(load_protocol(), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Exact original joint-copula protocol required") from error
    if actual != expected:
        raise ValueError("Exact original joint-copula protocol required")


def pipeline_config(protocol):
    validate(protocol)
    return {key: protocol["forecast"][key] for key in PIPELINE_FIELDS}
