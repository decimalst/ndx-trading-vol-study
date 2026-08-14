"""Load config.yaml and resolve paths relative to the repo root."""
from __future__ import annotations

import pathlib

import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(path: str | pathlib.Path | None = None) -> dict:
    cfg_path = pathlib.Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    for key, rel in cfg["paths"].items():
        p = ROOT / rel
        p.mkdir(parents=True, exist_ok=True)
        cfg["paths"][key] = p
    return cfg


# --------------------------------------------------------------------------
# Specification registry
# --------------------------------------------------------------------------
# config.yaml is frozen but records no date for when each model spec was
# written, so a model designed after seeing clean-window results could be
# scored on that window beside genuinely pre-registered models with unadjusted
# p-values. spec_registry.yaml records the dates; these two functions apply
# them by rule rather than by hand.
def load_spec_registry(path: str | pathlib.Path | None = None) -> dict:
    reg_path = pathlib.Path(path) if path else ROOT / "spec_registry.yaml"
    with open(reg_path) as f:
        return yaml.safe_load(f)


def _as_date(value) -> pd.Timestamp | None:
    """Registry date -> Timestamp, or None when it does not pin a date.

    `pre-registration` predates every phase by definition, so it imposes no
    lower bound and returns None here; `undetermined` also returns None but is
    handled differently by the caller -- that asymmetry is the whole point, so
    the two cases are separated at the call site, not here.
    """
    if value is None:
        return None
    if isinstance(value, str) and value in ("pre-registration", "undetermined"):
        return None
    return pd.Timestamp(value)


def spec_status(registry: dict, model: str, phase_start) -> dict:
    """When `model` becomes a confirmatory specification in a phase.

    confirmatory_from = max(phase_start, specified_on, available_from)

    A model specified after the window opened was tested on the data that
    generated it, so it is EXPLORATORY there: still reported, but never in the
    headline table and never with an inferential p-value. An UNDETERMINED
    specification date is also exploratory -- the burden of proof sits on the
    specification, not on the reader, so a missing date can never resolve in
    the specification's favour. An unknown model falls through to the same
    defaults for the same reason.
    """
    start = pd.Timestamp(phase_start)
    entry = (registry.get("models") or {}).get(model)
    defaults = registry.get("defaults") or {}
    if entry is None:
        entry = dict(defaults)
        known = False
    else:
        known = True

    spec_raw = entry.get("specified_on", defaults.get("specified_on"))
    avail_raw = entry.get("available_from", defaults.get("available_from"))
    undetermined = (spec_raw is None or spec_raw == "undetermined")

    if undetermined:
        reason = ("model not in the registry" if not known
                  else "specification date not recorded")
        return {"model": model, "confirmatory": False, "confirmatory_from": None,
                "reason": reason, "diagnostic_only": bool(entry.get("diagnostic_only")),
                "specified_on": spec_raw, "available_from": avail_raw, "known": known}

    bounds = [start]
    spec_on, avail_from = _as_date(spec_raw), _as_date(avail_raw)
    if spec_on is not None:
        bounds.append(spec_on)
    if avail_from is not None:
        bounds.append(avail_from)
    frm = max(bounds)

    # Both clauses are reported when both bind. They are different objections:
    # a late specification means the window generated the hypothesis, while a
    # late availability date means the model could not have been run earlier
    # and no peeking is implied. Collapsing them would read as an accusation
    # where none is warranted -- TiRex-2 is late for the innocent reason.
    why = []
    if spec_on is not None and spec_on >= frm:
        why.append(f"specified {spec_on.date()}, after the phase opened")
    if avail_from is not None and avail_from >= frm:
        why.append(f"model available {avail_from.date()}")
    reason = "; ".join(why) if frm > start else ""
    return {"model": model, "confirmatory": bool(frm <= start),
            "confirmatory_from": frm, "reason": reason,
            "diagnostic_only": bool(entry.get("diagnostic_only")),
            "specified_on": spec_raw, "available_from": avail_raw, "known": known}
