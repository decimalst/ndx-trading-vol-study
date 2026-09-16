"""Admit bounded raw SPX/Cboe bytes through the completed wave21 proof envelope.

Collection is metadata/hash-only. Admission decodes only the four declared raw
sources from previously checked snapshots, and never refits an old model or
loads a prior forecast table. Frozen failed attempts retain their exact bytes.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import event_cluster_replay_admission as previous
from src import issued_calibration_admission as checked

ROOT = Path(__file__).resolve().parents[1]
REPORT = "reports/event_cluster_replay/"
DATA = "data/event_cluster_replay/"
SOURCE_END = "2025-10-20"
SEALED_START = "2025-11-03"
DEFAULT_EXPECTED = {
    "event_cluster_replay.yaml": "b3eeea877c536a25cde0edca69cd7cc1474a425a2458ac36978adfeacee7e76f",
    REPORT
    + "publication_audit.json": "1e962011d0ccf66776a05a09ba09e8420f5d48f4c646e6bc8616ec07c81bdac5",
}
ANCHOR_PATHS = tuple(sorted(DEFAULT_EXPECTED))
SOURCES = {
    "daily": "data/research_paths/spx_daily.parquet",
    "vix": "data/free_sources/raw/cboe/VIX_History.csv",
    "vix9d": "data/free_sources/raw/cboe/VIX9D_History.csv",
    "vvix": "data/free_sources/raw/cboe/VVIX_History.csv",
}
IV_FIELDS = {"vix": "CLOSE", "vix9d": "CLOSE", "vvix": "VVIX"}
OHLC = ("open", "high", "low", "close")
OUTPUT_PATHS = tuple(
    sorted(
        (
            DATA + "upstream_admission.json",
            DATA + "reconstruction_audit.json",
            REPORT + "metrics.json",
            REPORT + "trial_ledger.jsonl",
        )
    )
)
EVENT_COUNTS = {"registered": 3, "inherited": 134, "evaluated": 3}
CONTROLS = ("baseline", "nuisance", "recent_frequency")


class _Closure(checked._Closure):
    def __init__(self, root, expected):
        self.root = Path(root).resolve()
        self.anchors = checked._mapping(DEFAULT_EXPECTED if expected is None else expected)
        if set(self.anchors) != set(ANCHOR_PATHS):
            raise ValueError("Exact two completed wave21 anchor paths required")
        self.files, self.payloads, self.manifests = {}, {}, {}
        self.add(self.anchors)

    def _successful_markers_absent(self):
        for report in (
            "range_alert",
            "issued_calibration",
            "event_cluster_replay",
            "index_hinge",
        ):
            path = self.root / "reports" / report / "failure.json"
            if path.exists() or path.is_symlink():
                raise ValueError("Successful upstream now has failure marker: " + report)

    def no_failure(self):
        self._successful_markers_absent()
        # Check the required failed trio again after the final general file hash.
        # An earlier visit to these alphabetically ordered files is insufficient.
        for name in ("failure.json", "metrics.json", "verification.json"):
            relative = "reports/event_cluster/" + name
            if relative in self.files:
                self.read(relative, keep=False, fresh=True)
        self._successful_markers_absent()


def _source_metadata(c):
    p = yaml.safe_load(c.read("index_hinge.yaml"))
    checked._same(p["sources"], SOURCES, "original four raw source paths")
    checked._same(p["index"]["source_end"], SOURCE_END, "original source ceiling")
    checked._same(p["index"]["sealed_start"], SEALED_START, "protected start")
    m, v = (c.json("reports/index_hinge/" + n) for n in ("manifest.json", "verification.json"))
    for name, doc in (("manifest", m), ("verification", v)):
        checked._same(
            doc["protocol_sha256"], c.files["index_hinge.yaml"], "hinge " + name + " protocol"
        )
    checked._same(v["status"], "VERIFIED", "hinge source verification")
    checked._same(
        v["verifier_sha256"], m["code"]["src/verify_index_hinge.py"], "hinge verifier code"
    )
    spx = c.json("data/research_paths/source_manifest.json")
    checked._same(spx["protocol_version"], 1, "SPX source manifest version")
    record = spx["sources"]["spx"]
    for key, value in {
        "sha256": c.files[SOURCES["daily"]],
        "auto_adjust": False,
        "ticker": "^GSPC",
        "provider": "Yahoo Finance via yfinance",
    }.items():
        checked._same(record[key], value, "SPX provenance " + key)
    for name in IV_FIELDS:
        doc = c.json(SOURCES[name] + ".manifest.json")
        for key, value in {
            "manifest_version": 1,
            "sha256": c.files[SOURCES[name]],
            "source_id": "cboe:daily-index:" + name.upper(),
            "source_version": "daily-live",
        }.items():
            checked._same(doc[key], value, "Cboe provenance " + name + " " + key)
        checked._integer(doc["bytes"])
        checked._same(doc["bytes"], len(c.read(SOURCES[name])), "Cboe raw byte count")


def _record(c, verified, publication):
    metrics = c.json(REPORT + "metrics.json")
    checked._same(
        metrics["protocol_sha256"],
        c.anchors["event_cluster_replay.yaml"],
        "canonical prior protocol",
    )
    checked._same(metrics["hypothesis_count"], 3, "prior new hypotheses")
    checked._same(metrics["cumulative_hypothesis_count"], 137, "prior complete family")
    checked._same(len(metrics["inherited_rows"]), 134, "prior inherited row count")
    checked._same(len(metrics["rows"]), 3, "prior terminal rows")
    checked._same(publication["leads"], metrics["leads"], "published prior leads")
    lines = c.read(REPORT + "trial_ledger.jsonl").splitlines()
    checked._same(len(lines), 140, "preceding ledger length")
    rows = [checked._json(line) for line in lines]
    checked._same(
        [r["event"] for r in rows],
        ["registered"] * 3 + ["inherited"] * 134 + ["evaluated"] * 3,
        "ordered preceding ledger roles",
    )
    for i, control in enumerate(CONTROLS):
        checked._same(
            rows[i],
            {
                "event": "registered",
                "study": "event_cluster_replay",
                "candidate": "cluster",
                "control": control,
                "horizon": 1,
                "score": "brier",
                "protocol_sha256": c.anchors["event_cluster_replay.yaml"],
            },
            "prior registration",
        )
        row = metrics["rows"][i]
        for key, value in {
            "study": "event_cluster_replay",
            "candidate": "cluster",
            "control": control,
            "horizon": 1,
            "score": "brier",
        }.items():
            checked._same(row[key], value, "prior terminal identity " + key)
        checked._same(
            rows[137 + i], {"event": "evaluated", **row}, "canonical prior terminal event"
        )
    for i, row in enumerate(metrics["inherited_rows"]):
        checked._same(rows[3 + i], {"event": "inherited", **row}, "preceding inherited event")
    checked._same(
        verified["ledger_events_verified"], EVENT_COUNTS, "independently verified prior ledger"
    )


def _collect(root, expected):
    c = _Closure(root, expected)
    c.no_failure()
    for name in ANCHOR_PATHS:
        c.read(name)
    protocol = yaml.safe_load(c.read("event_cluster_replay.yaml"))
    if type(protocol) is not dict:
        raise ValueError("Completed prior protocol object required")
    publication = c.json(REPORT + "publication_audit.json")
    signature = c.anchors["event_cluster_replay.yaml"]
    for key, value in {
        "status": "VERIFIED_REPLAY_PUBLICATION_AUDITED",
        "protocol_sha256": signature,
        "prior_failed_wave20_preserved": True,
        "new_hypotheses": 3,
        "cumulative_hypotheses": 137,
        "ledger_event_counts": EVENT_COUNTS,
        "new_producer_fits": 0,
        "newly_generated_forecasts": 0,
    }.items():
        checked._same(publication[key], value, "wave21 publication " + key)
    artifacts = previous.old._relative_artifacts(REPORT, publication["report_artifact_hashes"])
    required = {
        REPORT + n
        for n in (
            "manifest.json",
            "freeze_record.json",
            "verification.json",
            "metrics.json",
            "trial_ledger.jsonl",
            "SUMMARY.md",
            "NEXT_PEAK_AGE_DESIGN.md",
            "PUBLICATION_REVIEW.md",
            "full_repository_tests.txt",
            "pre_run_checks.txt",
        )
    }
    if not required.issubset(artifacts):
        raise ValueError("Complete prior publication report references required")
    c.add(artifacts)
    for key, name in [
        ("manifest_sha256", "manifest.json"),
        ("freeze_sha256", "freeze_record.json"),
        ("verification_sha256", "verification.json"),
    ]:
        checked._same(publication[key], c.files[REPORT + name], "published " + key)
    manifest, freeze, verified = (
        c.json(REPORT + n)
        for n in ("manifest.json", "freeze_record.json", "verification.json")
    )
    for label, doc in [("manifest", manifest), ("freeze", freeze), ("verification", verified)]:
        checked._same(doc["protocol_sha256"], signature, label + " prior protocol")
    checked._same(verified["status"], "VERIFIED", "completed prior verification")
    checked._same(
        verified["verifier_sha256"],
        manifest["code"]["src/verify_event_cluster_replay.py"],
        "completed prior verifier code",
    )
    checked._same(
        verified["inference"]["new_hypotheses_verified"], 3, "prior verified new hypotheses"
    )
    checked._same(
        verified["inference"]["cumulative_hypotheses_verified"], 137, "prior verified family"
    )
    checked._same(freeze["code"], manifest["code"], "complete prior code freeze")
    counts = {k: len(checked._mapping(manifest[k])) for k in ("code", "inputs", "preserved")}
    checked._same(
        publication["manifest_entries_checked"], counts, "published complete manifest counts"
    )
    checked._same(
        verified["artifact_hashes_checked"],
        {**counts, "outputs": 4},
        "verified closure counts",
    )
    c.add(freeze["prefit_design"])
    c.add({REPORT + "full_repository_tests.txt": freeze["checks"]["full_log_sha256"]})
    checked._integer(freeze["checks"]["full_repository_tests"])
    checked._same(
        publication["full_repository_tests"],
        freeze["checks"]["full_repository_tests"],
        "frozen prior full-suite count",
    )
    outputs = checked._mapping(verified["verified_output_hashes"])
    if set(outputs) != set(OUTPUT_PATHS):
        raise ValueError("Exact four independently verified wave21 output paths required")
    checked._same(
        publication["verified_output_hashes"], outputs, "published verified output identities"
    )
    c.add(outputs)
    retained = checked._mapping(verified["verified_retained_input_hashes"])
    if set(retained) != set(previous.OUTPUT_PATHS):
        raise ValueError("Exact six retained input identities required")
    checked._same(
        publication["verified_retained_input_hashes"],
        retained,
        "verified retained input identities",
    )
    c.add(retained)
    previous._manifest_closure(c)
    # This frozen call performs checked metadata traversal only. Its numerical
    # admission loader is deliberately never invoked.
    inner = previous._collect(root, protocol["upstream"]["anchors"])
    checked._same(
        c.json(DATA + "upstream_admission.json"), inner.audit, "exact saved prior admission"
    )
    checked._same(
        publication["prior_wave20_publication_sha256"],
        inner.anchors["reports/event_cluster/publication_audit.json"],
        "required failed wave20 publication identity",
    )
    checked._same(
        retained,
        inner.audit["retained_output_hashes"],
        "exact unchanged wave20 output identities",
    )
    c.add(inner.files)
    previous._manifest_closure(c)
    _source_metadata(c)
    _record(c, verified, publication)
    for name in sorted(c.files):
        c.read(name, keep=name in SOURCES.values())
    c.no_failure()
    c.audit = {
        "status": "VERIFIED_WAVE21_BOUND_RAW_SPX_CBOE_SNAPSHOTS",
        "anchors": dict(sorted(c.anchors.items())),
        "files": dict(sorted(c.files.items())),
        "manifest_paths": sorted(c.manifests),
        "manifest_groups": dict(sorted(c.manifests.items())),
        "counts": {
            "files": len(c.files),
            "manifests": len(c.manifests),
            "raw_sources": 4,
            "preceding_cumulative_hypotheses": 137,
            "preceding_ledger_events": 140,
        },
        "preceding_verified_output_hashes": outputs,
        "preserved_wave20_output_hashes": retained,
        "source_paths": dict(SOURCES),
        "source_end": SOURCE_END,
        "sealed_start": SEALED_START,
        "numeric_post_cutoff_values_parsed": False,
        "prior_forecast_tables_decoded": False,
        "historical_models_refitted": False,
        "loaded_from_checked_snapshots": True,
        "historical_vintage_certified": False,
        "timing_assumption": "all market predictors through previous observed SPX close; current entry weekday only",
        "vix9d_caveat": "prelaunch January2011-October2013 values are back-calculated archival training inputs",
        "target_interpretation": "SPX raw-close price-index returns; no dividend or risk-free adjustment or execution claim",
    }
    return c


def collect_input_pins(root=ROOT, expected=None):
    """Check all documentary identity and hashes; decode no raw numerical table."""
    c = _collect(root, expected)
    c.recheck()
    return c.audit["files"]


def _dates(index):
    if (
        not isinstance(index, pd.DatetimeIndex)
        or index.hasnans
        or index.has_duplicates
        or not index.is_monotonic_increasing
        or index.tz is not None
        or index.unit not in ("ms", "us", "ns")
        or not index.equals(index.normalize())
    ):
        raise ValueError("Unique ordered naive midnight ms/us/ns source sessions required")


def _values(frame):
    if any(dtype != np.dtype("float64") for dtype in frame.dtypes):
        raise ValueError("Stored float64 raw market values required")
    values = frame.to_numpy()
    if np.isinf(values).any() or ((values <= 0) & np.isfinite(values)).any():
        raise ValueError(
            "Observed raw prices must be positive finite; missing values remain missing"
        )


def _audit_source(c, name, frame, reference):
    return {
        "source_path": SOURCES[name],
        "source_sha256": c.files[SOURCES[name]],
        "bounded_rows": len(frame),
        "first_date": str(frame.index.min().date()),
        "last_date": str(frame.index.max().date()),
        "missing_values": {n: int(frame[n].isna().sum()) for n in frame},
        "missing_reference_dates": [str(d.date()) for d in reference.difference(frame.index)],
        "outside_reference_dates": [str(d.date()) for d in frame.index.difference(reference)],
    }


def admit_upstream(root=ROOT, expected=None, registered_pins=None):
    """Return audit, bounded raw daily OHLC and joined IV from checked byte buffers."""
    c = _collect(root, expected)
    if registered_pins is not None:
        pins = checked._mapping(registered_pins)
        for name, signature in c.files.items():
            if pins.get(name) != signature:
                raise ValueError("Registered source missing or changed: " + name)
    c.recheck()
    end = pd.Timestamp(SOURCE_END)
    daily = pd.read_parquet(
        io.BytesIO(c.read(SOURCES["daily"])), columns=list(OHLC), filters=[("date", "<=", end)]
    )
    _dates(daily.index)
    if daily.empty or (daily.index > end).any():
        raise ValueError("Nonempty bounded raw SPX calendar required")
    _values(daily)
    complete = daily.dropna(subset=list(OHLC))
    if (complete.high < complete[["open", "low", "close"]].max(axis=1)).any() or (
        complete.low > complete[["open", "high", "close"]].min(axis=1)
    ).any():
        raise ValueError("Invalid raw OHLC range")
    audits = {"daily": _audit_source(c, "daily", daily, daily.index)}
    series = []
    for name, column in IV_FIELDS.items():
        raw = pd.read_csv(
            io.BytesIO(c.read(SOURCES[name])), usecols=["DATE", column], dtype=str
        )
        dates = pd.to_datetime(raw.DATE, format="%m/%d/%Y", errors="raise")
        keep = dates <= end
        index = pd.DatetimeIndex(dates.loc[keep], name="date")
        _dates(index)
        if not len(index):
            raise ValueError("Nonempty bounded Cboe source required")
        values = pd.to_numeric(raw.loc[keep, column], errors="raise").to_numpy(
            dtype=np.float64
        )
        one = pd.Series(values, index=index, name=name)
        _values(one.to_frame())
        audits[name] = {
            **_audit_source(c, name, one.to_frame(), daily.index),
            "provider_date_field": "DATE",
            "provider_value_field": column,
        }
        series.append(one)
    iv = pd.concat(series, axis=1).sort_index()
    _dates(iv.index)
    c.audit["sources"] = audits
    dates_payload = ("\n".join(str(d.date()) for d in daily.index) + "\n").encode("ascii")
    c.audit["reference_calendar"] = {
        "definition": "full bounded observed SPX raw-OHLC sessions before missing-value filtering",
        "sessions": len(daily),
        "first_date": str(daily.index.min().date()),
        "last_date": str(daily.index.max().date()),
        "date_sequence_sha256": hashlib.sha256(dates_payload).hexdigest(),
        "hash_encoding": "ASCII YYYY-MM-DD per observed session, newline terminated",
        "input_date_unit": daily.index.unit,
    }
    c.recheck()
    return c.audit, daily, iv
