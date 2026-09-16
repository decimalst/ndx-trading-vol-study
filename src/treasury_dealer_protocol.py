"""Bind the immutable prospective design and its exact execution inventory."""

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA256 = "194fd2ef49cfa8db5542f5e672383b5e2d4ec8f8c724b2af8559d51319aef6a8"
EXECUTION = {
    "reports": "reports/treasury_dealer/predictive",
    "data": "data/treasury_dealer/forecasting",
    "ledger": "data/source_discovery/treasury_auction/dealer_ledger_v1.json",
    "ledger_sha256": "a22e04995429115780d2a33cd5ec7c18b7d56a1f41f9570b9160dc375156b423",
    "source_terminal": "reports/treasury_dealer/SOURCE_PREPARATION_TERMINAL.json",
    "source_terminal_sha256": "19b8fcb116bfc3014c8c219b8e9d9027c5fe6839246ca7a6d6a170f2f59bfa04",
    "source_audit": "reports/treasury_dealer/SOURCE_LEDGER_INDEPENDENT_AUDIT.json",
    "source_audit_sha256": "12aebe87db5a3fe2d6bed64ce6c5d9735805653fa1e61a200b412b68cdee432c",
    "calibration": "reports/treasury_dealer/INFERENCE_NULL_CALIBRATION.json",
    "calibration_sha256": "7992298ec031139d2d249fc74fad5115bd816ac8e0fa78f86e866220f0840acf",
    "prior": "reports/commodity_implied/predictive/metrics.json",
    "prior_sha256": "02762e0403d5f09e8601ac66c324ddb78e2bacb401958cef6f40531476a5543a",
    "prior_freeze": "reports/commodity_implied/predictive/freeze_record.json",
    "prior_freeze_sha256": "a126d35cf061614631b7be2ab62b4f7f24d39a9c7c49059d2bb18ad54f5016b1",
    "market_pins": {
        "data/raw/daily_ohlc.parquet": "710290d8ad8569172559334b23e423b50ecf7cd10f0fd8ec24a9fc29608bca4d",
        "data/raw/cross_asset_daily.parquet": "53f11ec90fc4f68d8d5592af7d893f7b5707bfbd7799b0735681acc18e72ac53",
        "data/raw/vxn_daily.parquet": "61c4dec804140831dadb05d0cc34ee38a3d662d31c5b6fc258eae6918b455731",
        "data/raw/short_dated_iv.parquet": "b238e404c661140a47f8eb7cd2bbdf5931fe7f2aa62151a08f71b52a20bfd39c",
    },
    "coefficient_atol": 1e-7,
    "coefficient_rtol": 1e-7,
    "forecast_atol": 1e-12,
    "forecast_rtol": 1e-7,
    "numerical_atol": 1e-12,
    "numerical_rtol": 1e-9,
    "score_atol": 1e-10,
    "score_rtol": 1e-8,
    "seed_rule": "base+phase_code*10000+endpoint_code*1000000; development0/evaluation1, daily0/active1; core adds block; shared comparisons",
    "inference_domain": "Every original calendar position within the literal phase, retaining all missing/nonauction/immature positions; offsets use global original positions modulo5",
}


CLOCK_CLASS = "QUALIFIED_REPORTED_CLOCK_ONLY_FOR_Z52"
CLOCK_LIMITATION = (
    "January27,2020 Z52 timing assumes faithful cached Treasury-attributed release text; "
    "original PDF and historical public-delivery timestamp remain unavailable and quantities unknown. "
    "Rejecting this assumption leaves unbounded strict-clock uncertainty after that date. "
    "Prior regression tests accessed the later historical period; it is not untouched confirmation."
)


def load_protocol():
    payload = (ROOT / "treasury_dealer.yaml").read_bytes()
    if hashlib.sha256(payload).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Original prospective Treasury protocol changed")
    return yaml.safe_load(payload)


def validate(protocol):
    try:
        actual = json.dumps(protocol, sort_keys=True, allow_nan=False)
        expected = json.dumps(load_protocol(), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Exact original Treasury protocol required") from error
    if actual != expected:
        raise ValueError("Exact original Treasury protocol required")


def pipeline_config(protocol):
    validate(protocol)
    return {
        key: protocol["forecast"][key]
        for key in (
            "origin_start",
            "origin_end",
            "development",
            "evaluation",
            "source_end",
            "minimum_train",
        )
    }
