"""Local prospective receipts and descriptive benchmark maturation.

No network, orders, existing market files, or model fitting. Local hash chains
detect damage; an independently retained head is needed to detect suffix removal.
An injected clock is for synthetic tests, never an external timestamp attestation.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from scipy import stats

VERSION = 1
ZERO = "0" * 64
SPEC_FIELDS = ("kind", "instrument", "unit", "session", "definition_id", "provider",
               "dataset", "feed", "adjustment")
SOURCE_FIELDS = ("provider", "dataset", "instrument", "feed", "adjustment",
                 "endpoint", "event_end", "released_at")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _time(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps require an explicit UTC offset")
    return value.astimezone(UTC)


def _stamp(value):
    return _time(value).isoformat().replace("+00:00", "Z")


def _strings(value, fields):
    if not isinstance(value, dict):
        raise ValueError("object required")
    if any(not isinstance(value.get(k), str) or not value[k].strip() for k in fields):
        raise ValueError(f"nonempty strings required: {fields}")


def _finite(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("finite numeric value required")
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError("invalid numeric domain")
    return float(value)


def _target(value):
    _strings(value, (*SPEC_FIELDS, "start", "end", "session_id"))
    if set(value) != {*SPEC_FIELDS, "start", "end", "session_id"}:
        raise ValueError("unknown target fields; use a new schema for extensions")
    if value["kind"] not in ("log_return", "variance"):
        raise ValueError("unsupported target kind")
    value = dict(value, start=_stamp(value["start"]), end=_stamp(value["end"]))
    if _time(value["start"]) >= _time(value["end"]):
        raise ValueError("target interval must have positive duration")
    return value


def _prediction(value, kind):
    if not isinstance(value, dict):
        raise ValueError("prediction object required")
    family = value.get("family")
    if kind == "variance":
        if family != "variance_mean" or set(value) != {"family", "mean"}:
            raise ValueError("variance requires a positive mean prediction")
        _finite(value["mean"], positive=True)
    else:
        required = {"family", "location", "scale"}
        if family == "student_t":
            required.add("df")
            _finite(value.get("df"), positive=True)
        elif family != "normal":
            raise ValueError("unsupported return distribution")
        if set(value) != required:
            raise ValueError("unknown or missing prediction fields")
        _finite(value["location"])
        _finite(value["scale"], positive=True)
    return json.loads(_json(value))


class ProspectiveLedger:
    """Small POSIX append-only ledger; all mutation methods take a process lock."""

    def __init__(self, directory, *, clock=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.blobs = self.directory / "blobs"
        self.blobs.mkdir(exist_ok=True)
        self.path = self.directory / "events.jsonl"
        self.clock = clock or (lambda: datetime.now(UTC))

    @contextlib.contextmanager
    def _lock(self):
        with (self.directory / ".lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _read(self):
        data = self.path.read_bytes() if self.path.exists() else b""
        if data and not data.endswith(b"\n"):
            raise ValueError("incomplete ledger line; preserve and investigate")
        rows, head, previous_time = [], ZERO, None
        for number, line in enumerate(data.splitlines(), 1):
            try:
                event = json.loads(line)
                claimed = event.pop("event_hash")
                if event["schema_version"] != VERSION or event["sequence"] != number:
                    raise ValueError("schema or sequence mismatch")
                if event["previous_hash"] != head or _digest(event) != claimed:
                    raise ValueError("ledger hash mismatch")
                receipt = _time(event["received_at"])
                if previous_time is not None and receipt < previous_time:
                    raise ValueError("receipt clock regression")
                for key in ("blob_sha256", "artifact_sha256"):
                    if key in event["payload"]:
                        digest = event["payload"][key]
                        raw = (self.blobs / digest).read_bytes()
                        if hashlib.sha256(raw).hexdigest() != digest:
                            raise ValueError("blob hash mismatch")
                event["event_hash"] = claimed
                rows.append(event)
                head, previous_time = claimed, receipt
            except (KeyError, TypeError, OSError, json.JSONDecodeError) as error:
                raise ValueError("invalid ledger or missing content blob") from error
        return rows

    def verify(self, *, expected_head=None):
        with self._lock():
            rows = self._read()
        head = rows[-1]["event_hash"] if rows else ZERO
        if expected_head is not None and head != expected_head:
            raise ValueError("ledger differs from retained checkpoint")
        return {"status": "VERIFIED_LOCAL_CHAIN", "head": head, "events": rows}

    def _now(self, rows):
        now = _time(self.clock())
        if rows and now < _time(rows[-1]["received_at"]):
            raise ValueError("local clock moved backward")
        return now

    def _blob(self, raw):
        if not isinstance(raw, bytes):
            raise ValueError("raw source/model artifact must be bytes")
        digest = hashlib.sha256(raw).hexdigest()
        destination = self.blobs / digest
        try:
            with destination.open("xb") as output:
                output.write(raw)
                output.flush()
                os.fsync(output.fileno())
        except FileExistsError:
            if destination.read_bytes() != raw:
                raise ValueError("existing blob is corrupt") from None
        return digest

    def _append(self, rows, kind, key, payload, now):
        event = {"schema_version": VERSION, "sequence": len(rows) + 1, "kind": kind,
                 "key": key, "received_at": _stamp(now),
                 "previous_hash": rows[-1]["event_hash"] if rows else ZERO,
                 "payload": payload}
        event["event_hash"] = _digest(event)
        with self.path.open("ab") as output:
            output.write(_json(event) + b"\n")
            output.flush()
            os.fsync(output.fileno())
        rows.append(event)
        return event["event_hash"]

    @staticmethod
    def _get(rows, event_id, kind):
        for row in rows:
            if row["event_hash"] == event_id and row["kind"] == kind:
                return row
        raise ValueError(f"unknown {kind} receipt")

    def ingest_source(self, source_key, raw, metadata):
        _strings({"key": source_key}, ("key",))
        _strings(metadata, SOURCE_FIELDS)
        metadata = json.loads(_json(metadata))
        metadata["event_end"] = _stamp(metadata["event_end"])
        metadata["released_at"] = _stamp(metadata["released_at"])
        with self._lock():
            rows = self._read()
            now = self._now(rows)
            if _time(metadata["released_at"]) > now:
                raise ValueError("claimed source release is in the future")
            if _time(metadata["event_end"]) > _time(metadata["released_at"]):
                raise ValueError("source market interval ends after its claimed release")
            prior = [r for r in rows if r["kind"] == "source" and r["key"] == source_key]
            if prior and any(metadata[k] != prior[0]["payload"]["metadata"][k]
                             for k in ("provider", "dataset", "instrument", "feed", "adjustment")):
                raise ValueError("source identity changed; use a distinct source key")
            digest = self._blob(raw)
            for row in prior:
                if (row["payload"]["blob_sha256"] == digest
                        and row["payload"]["metadata"] == metadata):
                    return row["event_hash"]
            payload = {"blob_sha256": digest, "metadata": metadata,
                       "supersedes": prior[-1]["event_hash"] if prior else None}
            return self._append(rows, "source", source_key, payload, now)

    def freeze_model(self, model_id, artifact, protocol):
        _strings({"model_id": model_id}, ("model_id",))
        _strings(protocol, ("benchmark_group", "training_cutoff", "input_schema",
                            "code_sha256", "evaluation_plan_sha256"))
        _strings(protocol.get("target_spec"), SPEC_FIELDS)
        if set(protocol["target_spec"]) != set(SPEC_FIELDS):
            raise ValueError("target specification fields differ from schema")
        for key in ("code_sha256", "evaluation_plan_sha256"):
            text = protocol[key]
            if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
                raise ValueError("SHA256 digest required")
        required = protocol.get("required_input_datasets")
        if (not isinstance(required, list) or not required or len(set(required)) != len(required)
                or any(not isinstance(x, str) or not x for x in required)):
            raise ValueError("declare required input datasets")
        protocol = json.loads(_json(protocol))
        protocol["training_cutoff"] = _stamp(protocol["training_cutoff"])
        with self._lock():
            rows = self._read()
            now = self._now(rows)
            if _time(protocol["training_cutoff"]) > now:
                raise ValueError("model training cutoff is in the future")
            payload = {"model_id": model_id, "artifact_sha256": self._blob(artifact),
                       "protocol": protocol}
            key = _digest(payload)
            for row in rows:
                if row["kind"] == "freeze" and row["key"] == key:
                    return row["event_hash"]
            return self._append(rows, "freeze", key, payload, now)

    def _scope(self, rows, freeze_id, target):
        freeze = self._get(rows, freeze_id, "freeze")["payload"]
        spec = {k: target[k] for k in SPEC_FIELDS}
        if freeze["protocol"]["target_spec"] != spec:
            raise ValueError("target and frozen model scope differ")
        return freeze

    def schedule_target(self, target, freeze_ids):
        target = _target(target)
        if not freeze_ids or len(set(freeze_ids)) != len(freeze_ids):
            raise ValueError("nonempty unique model freeze IDs required")
        with self._lock():
            rows = self._read()
            now = self._now(rows)
            if now >= _time(target["start"]):
                raise ValueError("target cohort must be declared before its start")
            protocols = [self._scope(rows, f, target)["protocol"] for f in freeze_ids]
            for field in ("benchmark_group", "evaluation_plan_sha256"):
                if len({p[field] for p in protocols}) != 1:
                    raise ValueError("cohort evaluation plans differ")
            key = _digest(target)
            payload = {"target": target, "freeze_ids": sorted(freeze_ids)}
            for row in rows:
                if row["kind"] == "schedule" and row["key"] == key:
                    if row["payload"] != payload:
                        raise ValueError("scheduled cohort is immutable")
                    return row["event_hash"]
            return self._append(rows, "schedule", key, payload, now)

    def issue_forecast(self, freeze_id, target, prediction, input_receipts):
        target = _target(target)
        prediction = _prediction(prediction, target["kind"])
        with self._lock():
            rows = self._read()
            now = self._now(rows)
            freeze = self._scope(rows, freeze_id, target)
            target_id = _digest(target)
            if not any(r["kind"] == "schedule" and r["key"] == target_id
                       and freeze_id in r["payload"]["freeze_ids"] for r in rows):
                raise ValueError("forecast not in predeclared cohort")
            if len(set(input_receipts)) != len(input_receipts):
                raise ValueError("duplicate input receipts")
            inputs = [self._get(rows, key, "source") for key in input_receipts]
            datasets = {row["payload"]["metadata"]["dataset"] for row in inputs}
            if not set(freeze["protocol"]["required_input_datasets"]) <= datasets:
                raise ValueError("missing required input dataset")
            payload = {"freeze_id": freeze_id, "target_id": target_id, "target": target,
                       "prediction": prediction, "input_receipts": sorted(input_receipts)}
            key = _digest([freeze_id, target_id])
            for row in rows:
                if row["kind"] == "forecast" and row["key"] == key:
                    prior = {k: row["payload"][k] for k in payload}
                    if prior != payload:
                        raise ValueError("issued forecast is immutable; declare a new model")
                    return row["event_hash"]
            payload["timing_eligible"] = now < _time(target["start"])
            payload["timing_reason"] = ("issued_before_target" if payload["timing_eligible"]
                                         else "received_at_or_after_target_start")
            return self._append(rows, "forecast", key, payload, now)

    def record_observation(self, target, value, source_receipts):
        target = _target(target)
        value = _finite(value, positive=target["kind"] == "variance")
        with self._lock():
            rows = self._read()
            now = self._now(rows)
            if now < _time(target["end"]):
                raise ValueError("target has not matured")
            if not source_receipts or len(set(source_receipts)) != len(source_receipts):
                raise ValueError("unique outcome source receipts required")
            sources = [self._get(rows, r, "source") for r in source_receipts]
            for source in sources:
                metadata = source["payload"]["metadata"]
                if any(metadata[k] != target[k] for k in
                       ("provider", "dataset", "instrument", "feed", "adjustment")):
                    raise ValueError("outcome provenance differs from frozen target")
            if max(_time(s["payload"]["metadata"]["event_end"]) for s in sources
                   ) < _time(target["end"]):
                raise ValueError("source does not reach target end")
            key = _digest(target)
            if not any(r["kind"] == "schedule" and r["key"] == key for r in rows):
                raise ValueError("unknown target cohort")
            payload = {"target_id": key, "target": target, "value": value,
                       "source_receipts": sorted(source_receipts)}
            prior = [r for r in rows if r["kind"] == "observation" and r["key"] == key]
            for row in prior:
                if {k: row["payload"][k] for k in payload} == payload:
                    return row["event_hash"]
            payload["supersedes"] = prior[-1]["event_hash"] if prior else None
            return self._append(rows, "observation", key, payload, now)

    def _asof(self, rows, as_of):
        now = self._now(rows)
        cutoff = _time(as_of) if as_of is not None else now
        if cutoff > now:
            raise ValueError("as_of cannot be in the future")
        return cutoff, [r for r in rows if _time(r["received_at"]) <= cutoff]

    def mature_scores(self, *, as_of=None):
        with self._lock():
            rows = self._read()
            now = self._now(rows)
            cutoff, visible = self._asof(rows, as_of)
            first_outcomes = {}
            for row in visible:
                if row["kind"] == "observation":
                    first_outcomes.setdefault(row["key"], row)
            for forecast in list(visible):
                p = forecast["payload"]
                if forecast["kind"] != "forecast" or not p["timing_eligible"]:
                    continue
                observation = first_outcomes.get(p["target_id"])
                if observation is None or _time(p["target"]["end"]) > cutoff:
                    continue
                key = forecast["event_hash"]
                if any(r["kind"] == "score" and r["key"] == key for r in rows):
                    continue
                prediction, y = p["prediction"], observation["payload"]["value"]
                if prediction["family"] == "variance_mean":
                    ratio = y / prediction["mean"]
                    loss, pit, metric = ratio - math.log(ratio) - 1, None, "qlike"
                else:
                    distribution = (stats.norm if prediction["family"] == "normal"
                                    else stats.t(prediction["df"]))
                    z = (y - prediction["location"]) / prediction["scale"]
                    loss = float(-distribution.logpdf(z) + math.log(prediction["scale"]))
                    pit, metric = float(distribution.cdf(z)), "negative_log_density"
                _finite(loss)
                payload = {"forecast_id": key, "observation_id": observation["event_hash"],
                           "target_id": p["target_id"], "freeze_id": p["freeze_id"],
                           "metric": metric, "loss": loss, "pit": pit,
                           "outcome_policy": "first_received", "scorer_version": VERSION}
                self._append(rows, "score", key, payload, now)
            return [r for r in rows if r["kind"] == "score"
                    and _time(r["received_at"]) <= cutoff]

    def coverage(self, *, as_of=None):
        with self._lock():
            rows = self._read()
            cutoff, rows = self._asof(rows, as_of)
        forecasts = {(r["payload"]["freeze_id"], r["payload"]["target_id"]): r
                     for r in rows if r["kind"] == "forecast"}
        scores = {r["payload"]["forecast_id"]: r for r in rows if r["kind"] == "score"}
        result = []
        for schedule in (r for r in rows if r["kind"] == "schedule"):
            target = schedule["payload"]["target"]
            for freeze in schedule["payload"]["freeze_ids"]:
                forecast = forecasts.get((freeze, schedule["key"]))
                score = None
                if forecast is None:
                    status = "missing_forecast" if cutoff >= _time(target["start"]) else "awaiting_forecast"
                elif not forecast["payload"]["timing_eligible"]:
                    status = "late_forecast"
                elif cutoff < _time(target["end"]):
                    status = "unmatured_target"
                else:
                    score = scores.get(forecast["event_hash"])
                    status = "scored" if score else "awaiting_observation_or_score"
                result.append({"target_id": schedule["key"], "target": target,
                               "freeze_id": freeze, "status": status,
                               "score": score["payload"] if score else None})
        return result

    def paired_benchmark(self, baseline_freeze, candidate_freeze, *, as_of=None):
        rows = self.verify()["events"]
        protocols = [self._get(rows, f, "freeze")["payload"]["protocol"]
                     for f in (baseline_freeze, candidate_freeze)]
        for field in ("target_spec", "benchmark_group", "evaluation_plan_sha256"):
            if protocols[0][field] != protocols[1][field]:
                raise ValueError("benchmark scope differs")
        if baseline_freeze == candidate_freeze:
            raise ValueError("benchmark requires two distinct freezes")
        coverage = self.coverage(as_of=as_of)
        indexed = [{r["target_id"]: r for r in coverage if r["freeze_id"] == freeze}
                   for freeze in (baseline_freeze, candidate_freeze)]
        targets = sorted(set(indexed[0]) | set(indexed[1]))
        if set(indexed[0]) != set(indexed[1]):
            raise ValueError("benchmark models must share the declared target cohort")
        pairs = []
        for target in targets:
            a, b = indexed[0][target], indexed[1][target]
            if a["status"] == b["status"] == "scored":
                if a["score"]["observation_id"] != b["score"]["observation_id"]:
                    raise ValueError("paired outcome vintages differ")
                pairs.append({"target_id": target, "baseline_loss": a["score"]["loss"],
                              "candidate_loss": b["score"]["loss"],
                              "difference": b["score"]["loss"] - a["score"]["loss"]})
        return {"status": "DESCRIPTIVE_ONLY", "n": len(pairs), "expected_targets": len(targets),
                "mean_loss_difference": (sum(p["difference"] for p in pairs) / len(pairs)
                                         if pairs else None), "pairs": pairs,
                "coverage": {name: dict(Counter(r["status"] for r in group.values()))
                             for name, group in zip(("baseline", "candidate"), indexed)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("command", choices=("ingest", "freeze", "schedule", "issue",
                                             "observe", "score", "coverage", "pair", "verify"))
    parser.add_argument("--request", type=Path, help="JSON object with API arguments")
    parser.add_argument("--payload", type=Path, help="source or model artifact bytes")
    args = parser.parse_args()
    ledger = ProspectiveLedger(args.ledger)
    request = json.loads(args.request.read_text()) if args.request else {}
    if args.command in ("ingest", "freeze"):
        if args.payload is None:
            parser.error("--payload is required for source/model receipts")
        request["raw" if args.command == "ingest" else "artifact"] = args.payload.read_bytes()
    method = {"ingest": ledger.ingest_source, "freeze": ledger.freeze_model,
              "schedule": ledger.schedule_target, "issue": ledger.issue_forecast,
              "observe": ledger.record_observation, "score": ledger.mature_scores,
              "coverage": ledger.coverage, "pair": ledger.paired_benchmark,
              "verify": ledger.verify}[args.command]
    result = method(**request)
    if args.command == "verify":
        result = {**result, "event_count": len(result["events"])}
        del result["events"]
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
