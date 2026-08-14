"""Assert the installed environment matches what the frozen artifacts were built with.

WHY THIS EXISTS, and it is the same lesson as everything else in this repository.

`reports/FROZEN_REPORT_HASHES.json` pins the frozen reports byte-for-byte. That
fence is worthless if the thing that REGENERATES them floats: a digest over an
output does not constrain the code path that produced it.

`requirements.txt` states `scikit-learn>=1.7,<1.8`, and nothing enforced it. A
reviewer on scikit-learn 1.8 ran the suite and got three GBM artifact tests
failing with

    timing-safe metrics do not recompute from saved forecasts

which reads as a CORRUPTED ARTIFACT, not as "your scikit-learn is outside the
pin". That is a fence that looks stronger than it is, misdirecting the reader
toward the science when the cause was the environment -- exactly the failure
mode the methodology fork was built to catch.

So: check the pins explicitly, fail with a message that names the real cause,
and let the artifact verifiers append that diagnosis when a recompute mismatch
coincides with a pin violation.

    make check-env
"""
from __future__ import annotations

import re
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"

# Distributions whose numerical output the frozen artifacts depend on. A drift
# here can change a fitted coefficient in the last decimal and cascade into a
# recompute mismatch. Packages that only affect I/O are deliberately omitted.
NUMERICALLY_LOAD_BEARING = ("scikit-learn", "numpy", "scipy", "pandas",
                            "statsmodels")

_SPEC = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*(.*)$")


def _parse_requirements() -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw in REQUIREMENTS.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = _SPEC.match(line)
        if m:
            pins[m.group(1).lower()] = m.group(2).strip()
    return pins


def _version_tuple(v: str) -> tuple[int, ...]:
    parts = []
    for chunk in v.split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _satisfies(installed: str, spec: str) -> bool:
    if not spec:
        return True
    for clause in spec.split(","):
        clause = clause.strip()
        m = re.match(r"^(>=|<=|==|<|>|!=)\s*(.+)$", clause)
        if not m:
            continue
        op, want = m.group(1), m.group(2).strip()
        a, b = _version_tuple(installed), _version_tuple(want)
        ok = {">=": a >= b, "<=": a <= b, "==": a == b,
              "<": a < b, ">": a > b, "!=": a != b}[op]
        if not ok:
            return False
    return True


def violations(only_load_bearing: bool = True) -> list[dict]:
    """Installed distributions that fall outside requirements.txt."""
    pins = _parse_requirements()
    out = []
    for name, spec in pins.items():
        if only_load_bearing and name not in NUMERICALLY_LOAD_BEARING:
            continue
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            out.append({"package": name, "installed": None, "required": spec,
                        "problem": "not installed"})
            continue
        if not _satisfies(installed, spec):
            out.append({"package": name, "installed": installed,
                        "required": spec, "problem": "outside pin"})
    return out


def pin_advice() -> str:
    """One line naming the environment as a likely cause, or empty if clean.

    Artifact verifiers append this to a recompute-mismatch error so the reader
    is not sent hunting for a corrupted artifact when the real cause is that
    their scikit-learn is a minor version ahead.
    """
    bad = violations()
    if not bad:
        return ""
    parts = [f"{v['package']} {v['installed'] or 'missing'} "
             f"(requires {v['required']})" for v in bad]
    return ("NOTE: the installed environment is outside requirements.txt — "
            + "; ".join(parts)
            + ". Frozen artifacts were produced inside those pins, so a "
              "recompute mismatch here is more likely environment drift than a "
              "corrupted artifact. Reproduce inside the pins before "
              "investigating the science.")


def assert_pins() -> None:
    bad = violations()
    if bad:
        raise RuntimeError(pin_advice())


if __name__ == "__main__":
    bad = violations()
    if not bad:
        print("environment matches requirements.txt for all "
              f"numerically load-bearing packages: {', '.join(NUMERICALLY_LOAD_BEARING)}")
    else:
        print(pin_advice())
        raise SystemExit(1)
