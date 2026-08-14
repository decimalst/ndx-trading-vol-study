"""Construct the frozen 1999 QQQ price-only history extension.

This module deliberately stops at data construction. It does not define or
score tail labels, HMMs, foundation-model embeddings, or boosted trees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import config

PROTOCOL_PATH = config.ROOT / "history_extension.yaml"
IMPLEMENTATION_PATH = Path(__file__).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(path: Path) -> str:
    """Prefer portable repository-relative paths in persisted provenance."""
    try:
        return str(path.resolve().relative_to(config.ROOT.resolve()))
    except ValueError:
        return str(path)


def load_protocol(path: str | Path = PROTOCOL_PATH) -> dict:
    with Path(path).open() as source:
        return yaml.safe_load(source)


def validate_protocol(protocol: dict) -> None:
    if protocol.get("status") != "enabling_data_only":
        raise ValueError("history extension must remain enabling_data_only")
    window = protocol["window"]
    start = pd.Timestamp(window["source_start"])
    end = pd.Timestamp(window["origin_end"])
    clean = pd.Timestamp(window["clean_start"])
    if not start < end < clean or not window.get("forbid_clean_origins", False):
        raise ValueError("history window must end before the sealed clean window")
    for name in ("qqq", "vxn"):
        value = str(protocol["source"][name].get("expected_sha256", ""))
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"{name} expected_sha256 must be a lowercase sha256")
    output_hash = str(protocol.get("output", {}).get("expected_sha256", ""))
    if len(output_hash) != 64 or any(c not in "0123456789abcdef" for c in output_hash):
        raise ValueError("output expected_sha256 must be a lowercase sha256")
    implementation_hash = str(
        protocol.get("implementation", {}).get("expected_sha256", "")
    )
    if len(implementation_hash) != 64 or any(
        c not in "0123456789abcdef" for c in implementation_hash
    ):
        raise ValueError("implementation expected_sha256 must be a lowercase sha256")
    transform = protocol["transform"]
    if float(transform["variance_floor"]) <= 0:
        raise ValueError("variance floor must be positive")
    if int(transform["har_week_sessions"]) >= int(transform["har_month_sessions"]):
        raise ValueError("HAR week must be shorter than HAR month")


def _normal_index(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    out = frame.copy()
    index = pd.DatetimeIndex(pd.to_datetime(out.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    out.index = index.normalize()
    out.index.name = "date"
    out = out.sort_index()
    if out.index.duplicated().any():
        raise ValueError(f"{name} source has duplicate dates")
    if not out.index.is_monotonic_increasing:
        raise ValueError(f"{name} source dates are not sorted")
    return out


def validate_qqq_source(frame: pd.DataFrame, source_protocol: dict) -> pd.DataFrame:
    daily = _normal_index(frame, "QQQ")
    required = list(source_protocol["required_columns"])
    missing = sorted(set(required) - set(daily.columns))
    if missing:
        raise ValueError(f"QQQ source missing columns {missing}")
    if daily.empty or daily.index.min() != pd.Timestamp(source_protocol["expected_first_date"]):
        raise ValueError("QQQ source does not begin at the frozen inception date")
    if daily[required].isna().any().any():
        raise ValueError("QQQ source contains missing required values")
    prices = ["open", "high", "low", "close", "adj close"]
    if (daily[prices] <= 0).any().any() or (daily["volume"] < 0).any():
        raise ValueError("QQQ source contains invalid nonpositive prices or volume")
    upper = daily[["open", "close", "low"]].max(axis=1)
    lower = daily[["open", "close", "high"]].min(axis=1)
    if (daily["high"] < upper).any() or (daily["low"] > lower).any():
        raise ValueError("QQQ source has impossible OHLC geometry")
    return daily


def validate_vxn_source(frame: pd.DataFrame, source_protocol: dict) -> pd.DataFrame:
    vxn = _normal_index(frame, "VXN")
    if "close" not in vxn or vxn["close"].isna().any() or (vxn["close"] <= 0).any():
        raise ValueError("VXN source needs finite positive closes")
    if vxn.empty or vxn.index.min() != pd.Timestamp(source_protocol["free_file_first_date"]):
        raise ValueError("VXN free-file boundary changed; re-audit before use")
    return vxn


def build_price_only_panel(daily: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    """Derive features available by the QQQ close, with no implied inputs."""
    validate_protocol(protocol)
    source = validate_qqq_source(daily, protocol["source"]["qqq"])
    source = source.loc[
        pd.Timestamp(protocol["window"]["source_start"]):
        pd.Timestamp(protocol["window"]["origin_end"])
    ].copy()
    floor = float(protocol["transform"]["variance_floor"])
    hl = np.log(source["high"] / source["low"])
    co = np.log(source["close"] / source["open"])
    intraday = (0.5 * hl.pow(2) - (2 * np.log(2.0) - 1) * co.pow(2)).clip(lower=floor)
    overnight_return = np.log(source["open"] / source["close"].shift(1))
    total = intraday + overnight_return.pow(2)
    adjusted_return = np.log(source["adj close"] / source["adj close"].shift(1))
    out = pd.DataFrame(
        {
            "rv_intraday": intraday,
            "var_overnight": overnight_return.pow(2),
            "rv_total": total,
            "log_rv": np.log(total.clip(lower=floor)),
            "ret_cc": adjusted_return,
        },
        index=source.index,
    ).dropna()
    out["log_rv_d"] = out["log_rv"]
    out["log_rv_w"] = out["log_rv"].rolling(
        int(protocol["transform"]["har_week_sessions"])
    ).mean()
    out["log_rv_m"] = out["log_rv"].rolling(
        int(protocol["transform"]["har_month_sessions"])
    ).mean()
    if out.index.max() >= pd.Timestamp(protocol["window"]["clean_start"]):
        raise RuntimeError("price-only panel escaped the clean-window fence")
    return out


def build_manifest(
    *,
    source_path: Path,
    output_path: Path,
    source_rows: int,
    output_rows: int,
    source_first: str,
    source_last: str,
    output_first: str,
    output_last: str,
) -> dict:
    return {
        "source": {
            "path": _manifest_path(source_path),
            "sha256": _sha256(source_path),
            "rows": int(source_rows),
            "first_date": source_first,
            "last_date": source_last,
        },
        "output": {
            "path": _manifest_path(output_path),
            "sha256": _sha256(output_path),
            "rows": int(output_rows),
            "first_date": output_first,
            "last_date": output_last,
        },
    }


def _write_report(protocol: dict, panel: pd.DataFrame, vxn: pd.DataFrame, manifest: dict) -> None:
    report_path = config.ROOT / protocol["output"]["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    qqq = manifest["qqq"]
    vx = manifest["vxn"]
    before_2016 = int((panel.index < pd.Timestamp("2016-01-01")).sum())
    dot_com = int(((panel.index >= "1999-03-11") & (panel.index <= "2002-12-31")).sum())
    gfc = int(((panel.index >= "2007-01-01") & (panel.index <= "2009-12-31")).sum())
    har_complete = int(panel[["log_rv_d", "log_rv_w", "log_rv_m"]].dropna().shape[0])
    lines = [
        "# QQQ 1999 price-only history extension audit",
        "",
        "This is an enabling-data artifact, not a model result. No tail labels,",
        "thresholds, HMMs, embeddings, boosted trees, or scorecards were fit.",
        "",
        "## Usable history",
        "",
        f"- Price-only panel: **{len(panel):,} sessions**, {panel.index.min().date()} through {panel.index.max().date()}.",
        f"- Rows before 2016: **{before_2016:,}**; dot-com window: **{dot_com:,}**; 2007-09 GFC window: **{gfc:,}**.",
        f"- Complete daily/weekly/monthly HAR state: **{har_complete:,} sessions** after the 22-session warmup.",
        "- Inputs are QQQ OHLC known at the 16:00 close. All rolling features are trailing-only.",
        "- The panel stops on 2025-10-17 and does not enter the sealed NDX clean window beginning 2025-11-03.",
        "- QQQ is a hash-frozen Yahoo Finance snapshot, not an exchange point-in-time archive; vendor revisions before this snapshot cannot be ruled out.",
        "",
        "This panel can enable price-only HMM, tail-ranking, or latent-probe studies",
        "from 1999. It cannot extend HAR-IV or any VXN-fed model to 1999.",
        "",
        "## VXN boundary",
        "",
        "Cboe's current methodology lists January 1995 as VXN's first value month,",
        f"but the frozen free Cboe file available here starts **{vxn.index.min().date()}**.",
        "The earlier values were not silently sourced from a vendor, proxy-spliced,",
        "or reconstructed. A pre-2009 VXN-fed study remains blocked on an official",
        "complete daily series and a separate methodology-change audit.",
        "",
        f"- Official methodology: {protocol['source']['vxn']['methodology_url']}",
        f"- Free history endpoint: {protocol['source']['vxn']['url']}",
        "",
        "## Provenance",
        "",
        f"- QQQ inception reference: {protocol['source']['qqq']['inception_url']}",
        f"- QQQ raw SHA-256: `{qqq['sha256']}` ({qqq['rows']:,} rows).",
        f"- VXN raw SHA-256: `{vx['sha256']}` ({vx['rows']:,} rows).",
        f"- Derived panel SHA-256: `{manifest['output']['sha256']}`.",
        f"- Frozen build protocol SHA-256: `{manifest['protocol']['sha256']}`.",
        f"- Frozen transform implementation SHA-256: `{manifest['implementation']['sha256']}`.",
        "- The exact hashes are frozen in `history_extension.yaml`; a refreshed raw file fails closed.",
        "",
        "## Downstream fence",
        "",
        "Freeze the tail definition, ranking metrics, time splits, and latent-probe",
        "incremental comparison before reading any downstream scores. A downstream",
        "origin may consume only panel rows at or before that origin. This audit",
        "does not authorize re-opening or peeking at the existing clean window.",
    ]
    report_path.write_text("\n".join(lines) + "\n")


def build_history(protocol: dict | None = None) -> dict:
    protocol = protocol or load_protocol()
    validate_protocol(protocol)
    implementation_hash = _sha256(IMPLEMENTATION_PATH)
    if implementation_hash != protocol["implementation"]["expected_sha256"]:
        raise RuntimeError("history transform implementation changed; re-audit before rebuilding")
    qqq_path = config.ROOT / protocol["source"]["qqq"]["path"]
    vxn_path = config.ROOT / protocol["source"]["vxn"]["path"]
    for name, path in (("QQQ", qqq_path), ("VXN", vxn_path)):
        if not path.exists():
            raise FileNotFoundError(f"{name} frozen source missing: {path}")
    qqq_hash = _sha256(qqq_path)
    vxn_hash = _sha256(vxn_path)
    if qqq_hash != protocol["source"]["qqq"]["expected_sha256"]:
        raise RuntimeError("QQQ raw source hash changed; re-audit before rebuilding")
    if vxn_hash != protocol["source"]["vxn"]["expected_sha256"]:
        raise RuntimeError("VXN raw source hash changed; re-audit before rebuilding")
    qqq = validate_qqq_source(pd.read_parquet(qqq_path), protocol["source"]["qqq"])
    vxn = validate_vxn_source(pd.read_parquet(vxn_path), protocol["source"]["vxn"])
    panel = build_price_only_panel(qqq, protocol)
    output_path = config.ROOT / protocol["output"]["panel"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_path)
    output_hash = _sha256(output_path)
    if output_hash != protocol["output"]["expected_sha256"]:
        raise RuntimeError("derived panel hash changed; re-audit transform before use")
    output_manifest = build_manifest(
        source_path=qqq_path,
        output_path=output_path,
        source_rows=len(qqq),
        output_rows=len(panel),
        source_first=str(qqq.index.min().date()),
        source_last=str(qqq.index.max().date()),
        output_first=str(panel.index.min().date()),
        output_last=str(panel.index.max().date()),
    )
    manifest = {
        "protocol_version": int(protocol["protocol_version"]),
        "frozen_on": protocol["frozen_on"],
        "protocol": {
            "path": _manifest_path(PROTOCOL_PATH),
            "sha256": _sha256(PROTOCOL_PATH),
        },
        "implementation": {
            "path": _manifest_path(IMPLEMENTATION_PATH),
            "sha256": implementation_hash,
        },
        "qqq": output_manifest["source"],
        "vxn": {
            "path": _manifest_path(vxn_path),
            "sha256": vxn_hash,
            "rows": len(vxn),
            "first_date": str(vxn.index.min().date()),
            "last_date": str(vxn.index.max().date()),
            "official_first_value_month": protocol["source"]["vxn"]["official_first_value_month"],
            "joined_to_panel": False,
        },
        "output": output_manifest["output"],
        "fences": {
            "origin_end": protocol["window"]["origin_end"],
            "clean_start": protocol["window"]["clean_start"],
            "clean_origins_included": False,
        },
    }
    manifest_path = config.ROOT / protocol["output"]["manifest"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _write_report(protocol, panel, vxn, manifest)
    print(
        f"wrote {output_path} ({len(panel)} rows, "
        f"{panel.index.min().date()} .. {panel.index.max().date()})"
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build",), nargs="?", default="build")
    parser.parse_args(argv)
    build_history()


if __name__ == "__main__":
    main()
