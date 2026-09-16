"""Authenticate and join the declared inputs without changing their calendars."""

import hashlib
import json
import re
from pathlib import Path

from src.claims_release_inputs import read_market_sources
from src.claims_release_market import build_market_features, make_five_session_targets
from src.treasury_dealer_features import build_auction_features
from src.treasury_market_context import build_market_context


def read_inputs(root, market_pins, ledger_path, ledger_sha256):
    """Call only after registration; authenticate every input before decoding.

    The runner binds the exact inventory and independent source admission. The
    existing market reader additionally validates all full date columns before
    decoding numerical values through the fixed October 20, 2025 ceiling.
    """
    if type(market_pins) is not dict or ledger_path in market_pins:
        raise ValueError("Separate market inventory and ledger required")
    pins = {**market_pins, ledger_path: ledger_sha256}
    if any(
        type(v) is not str or re.fullmatch(r"[0-9a-f]{64}", v) is None for v in pins.values()
    ):
        raise ValueError("Literal SHA256 input pins required")
    root = Path(root)
    snapshots = {name: (root / name).read_bytes() for name in pins}
    for name, payload in snapshots.items():
        if hashlib.sha256(payload).hexdigest() != pins[name]:
            raise ValueError("Input hash mismatch: " + name)
    ledger = json.loads(snapshots[ledger_path])
    if (
        type(ledger) is not dict
        or type(ledger.get("events")) is not list
        or type(ledger.get("evidence")) is not list
        or len(ledger["events"]) != len(ledger["evidence"])
    ):
        raise ValueError("Admitted ledger with aligned event/evidence lists required")
    sources = read_market_sources(root, market_pins)
    return sources, ledger


def build_inputs(sources, ledger_events):
    """Join the three fixed feature blocks; retain every original QQQ position."""
    if type(sources) is not dict or set(sources) != {"daily", "cross", "iv"}:
        raise ValueError("Exact daily, cross and IV sources required")
    market = build_market_features(**sources)
    context = build_market_context(market.index, sources["cross"])
    auction = build_auction_features(market.index, ledger_events)
    features = market.join(context).join(auction["features"])
    return {
        "features": features,
        "targets": make_five_session_targets(market.rv_total),
        "event_audit": auction["events"],
    }
