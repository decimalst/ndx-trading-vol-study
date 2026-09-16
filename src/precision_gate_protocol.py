"""Exact prospective wave26 specification and admitted execution inventory."""

import hashlib
import json
from pathlib import Path

import yaml

from src.treasury_dealer_protocol import EXECUTION as PREVIOUS_EXECUTION

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA256 = "c9ac4c6f238cc898e219b8e1b1e9c8f1bf84371e40bf363a7572ee63182f11a2"
EVIDENCE_CLASS = "EXPLORATORY_REUSED_MARKET_HISTORY_CURRENT_VINTAGES"
EVIDENCE_LIMITATION = (
    "Historical development and evaluation windows have been reused. "
    "Prior regression tests accessed the period beginning 2025-11-03; it is not untouched confirmation. "
    "Current vendor snapshots do not certify complete point-in-time price or implied-volatility vintages. "
    "Numerical inputs in this experiment end 2025-10-20."
)
EXECUTION = {
    "reports": "reports/precision_gate/predictive",
    "data": "data/model_memory_study/precision_gate_wave26",
    "prior": "reports/treasury_dealer/predictive/metrics.json",
    "prior_sha256": "05e363aeaa1fbe2d7a67099d8a12471590099f72aa99615bd35cc1222288daa4",
    "prior_freeze": "reports/treasury_dealer/predictive/freeze_record.json",
    "prior_freeze_sha256": "99cca9121cfd6a691d99243845518315871071ecd3c9566fcdb965a11597c195",
    "prior_terminal": "reports/treasury_dealer/predictive/terminal.json",
    "prior_terminal_sha256": "5db462b7bef69003b0b08aab7b970a4293f6ca4db4233489e4d364956f1f3cda",
    "prior_publication": "reports/treasury_dealer/predictive/PUBLICATION_AUDIT.json",
    "prior_publication_sha256": "0de5398f05f6be73dfea1ed77613618747162b9ef8f59ff42c3021ccb1e80207",
    "prior_audit": "reports/treasury_dealer/predictive/PUBLICATION_INDEPENDENT_AUDIT.json",
    "prior_audit_sha256": "d7857ace5f3d5b73c32cb9678223cfe87b390f12fde883bf9d6d80427c114f32",
    "calibration": PREVIOUS_EXECUTION["calibration"],
    "calibration_sha256": PREVIOUS_EXECUTION["calibration_sha256"],
    "market_pins": dict(PREVIOUS_EXECUTION["market_pins"]),
    "expert_coefficient_atol": 1e-7,
    "expert_coefficient_rtol": 1e-7,
    "gate_coefficient_atol": 5e-6,
    "gate_coefficient_rtol": 1e-7,
    "gate_objective_atol": 1e-10,
    "gate_objective_rtol": 1e-9,
    "forecast_atol": 1e-12,
    "forecast_rtol": 1e-7,
    "numerical_atol": 1e-12,
    "numerical_rtol": 1e-9,
    "score_atol": 1e-10,
    "score_rtol": 1e-8,
    "independent_gate_solver": {
        "method": "SLSQP",
        "start": 0.5,
        "ftol": 1e-14,
        "maxiter": 2000,
        "projected_gradient_limit": 1e-7,
    },
}
PIPELINE_FIELDS = (
    "issuance_start",
    "origin_start",
    "origin_end",
    "source_end",
    "development",
    "evaluation",
    "minimum_train",
    "adaptive_half_life",
    "gate_window",
    "gate_minimum_train",
)


def load_protocol():
    payload = (ROOT / "precision_gate.yaml").read_bytes()
    if hashlib.sha256(payload).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Original precision-gate protocol changed")
    return yaml.safe_load(payload)


def validate(protocol):
    try:
        actual = json.dumps(protocol, sort_keys=True, allow_nan=False)
        expected = json.dumps(load_protocol(), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Exact original precision-gate protocol required") from error
    if actual != expected:
        raise ValueError("Exact original precision-gate protocol required")


def pipeline_config(protocol):
    validate(protocol)
    return {key: protocol["forecast"][key] for key in PIPELINE_FIELDS}
