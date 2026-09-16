"""Independent macro-plan, second-moment and inference reconstruction.

No imports of macro producer, plan-feature or estimator functions. Numerical
tolerances are fixed before empirical forecasts: producer KKT1e-8, independent
BFGS KKT2e-8, coefficient absolute1e-6, forecast relative1e-6/absolute1e-12.
"""
from __future__ import annotations

import calendar
import json
import re
from datetime import UTC, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize

from .verify_international_volatility import same_tree
from .verify_iterative_signal_search import digest, holm, same
from .verify_model_memory_study import explicit_bootstrap_means, independent_hac

ROOT = Path(__file__).resolve().parents[1]
ZONE = ZoneInfo("America/New_York")
BASE = ("const", "on_rms_d", "on_rms_w", "on_rms_m", "day_rms_d", "day_rms_w", "day_rms_m",
        "lrv_d", "lrv_w", "lrv_m", "liv", "lvix", "entry_dow_1", "entry_dow_2", "entry_dow_3",
        "entry_dow_4", "nominal_hours")
ADDITIONS = {"cpi": "cpi_plan", "nfp": "nfp_plan", "fomc": "fomc_plan"}
ALL_FEATURES = BASE + tuple(ADDITIONS.values())
MODELS = ("mean", "baseline", *ADDITIONS)
COMPARISONS = tuple((candidate, control) for candidate in ADDITIONS for control in ("baseline", "mean"))
ALPHA = .01
WAVE_ALPHA = .0025
INDEPENDENT_GRADIENT_TOLERANCE = 2e-8
PRODUCER_GRADIENT_TOLERANCE = 1e-8
COEFFICIENT_ATOL = 1e-6
FORECAST_RTOL = 1e-6
MONTHS = {name: number for number, name in enumerate(calendar.month_name) if name}
MONTH_PATTERN = "(?:" + "|".join(MONTHS) + ")"


def document_section(text, url):
    headings = list(re.finditer(r"(?m)^[^\n]+\((https?://[^\s)]+)\)\s*$", text))
    selected = [index for index, heading in enumerate(headings) if heading[1] == url]
    if not selected:
        raise ValueError("A captured section for the requested official source is required")
    sections = []
    for index in selected:
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        sections.append(text[headings[index].start():end])
    return "\n".join(sections)


def _plain(text):
    return " ".join(re.sub(r"\bL\d+(?:@P\d+(?:-\d+)?)?:\s*", "", text).split())


def document_timestamp(evidence):
    text = _plain(evidence)
    # A following sentence can describe a superseded plan; only the explicit
    # preceding next-release clause supplies this timestamp.
    text = re.split(r"\.\s+(?=[A-Z])", text)[0]
    dates = list(re.finditer(rf"(?:(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+)?"
                             rf"({MONTH_PATTERN})\s+(\d{{1,2}}),?\s+(\d{{4}})", text))
    times = list(re.finditer(r"(\d{1,2}):(\d{2})\s*([ap])\.?m\.?", text, re.I))
    zones = set(re.findall(r"\b(?:EST|EDT|ET)\b", text))
    if len(dates) != 1 or len(times) != 1 or len(zones) != 1:
        raise ValueError("One explicitly printed full date, clock and Eastern zone are required")
    date, clock = dates[0], times[0]
    hour = int(clock[1]) % 12 + (12 if clock[3].lower() == "p" else 0)
    zone = next(iter(zones))
    tz = ZONE if zone == "ET" else timezone(timedelta(hours=-5 if zone == "EST" else -4))
    result = datetime(int(date[4]), MONTHS[date[2]], int(date[3]), hour, int(clock[2]), tzinfo=tz)
    if date[1] and date[1] != result.strftime("%A"):
        raise ValueError("Printed weekday contradicts printed date")
    return result


def same_release_reissue(text):
    passive = r"\b(?:this|the)\s+(?:news\s+)?release\s+(?:was|has\s+been|is)\s+re[\s-]?issued\b"
    active = r"\bre[\s-]?issued\s+(?:this|the)\s+(?:news\s+)?release\b"
    return re.search(passive + "|" + active, _plain(text), re.I) is not None


def scan_source_admissions(root, protocol):
    """Scan every selected bounded document and report all admission mismatches."""
    rows, mismatches = [], []
    for event in ("cpi", "nfp"):
        path = root / protocol["sources"][event]
        records = json.loads(path.read_text())["records"]
        for record in records:
            row = {"event": event, "source_url": record["source_url"], "status": record["parse_status"],
                   "ledger_sha256": digest(path), "same_release_reissue": None, "admission_mismatch": False}
            date = datetime.strptime(re.search(r"_(\d{8})\.htm$", record["source_url"])[1], "%m%d%Y").date().isoformat()
            bounded = protocol["calendar"]["source_publication_start"] <= date <= protocol["calendar"]["source_publication_end"]
            if not bounded:
                if record["parse_status"] != "OUTSIDE_PUBLICATION_FENCE":
                    row["admission_mismatch"] = True
                    row["reason"] = "Out-of-fence document is not excluded"
            else:
                snapshot = root / record["snapshot_path"]
                if digest(snapshot) != record["snapshot_sha256"]:
                    raise AssertionError("Admission-scan source extraction hash differs")
                text = document_section(snapshot.read_text(), record.get("snapshot_source_url", record["source_url"]))
                row["same_release_reissue"] = same_release_reissue(text)
                row["snapshot_path"], row["snapshot_sha256"] = record["snapshot_path"], record["snapshot_sha256"]
                admitted = record["parse_status"] in {"EXPLICIT_ORIGINAL_PLAN", "VERIFIED_EXPLICIT_PLAN"}
                if row["same_release_reissue"] and admitted:
                    row["admission_mismatch"] = True
                    row["reason"] = "Admitted document explicitly describes its own reissue"
                if record["parse_status"] == "ORIGINAL_VINTAGE_UNCERTAIN" and not row["same_release_reissue"]:
                    row["admission_mismatch"] = True
                    row["reason"] = "Same-release reissue exclusion lacks a recognized notice"
            rows.append(row)
            if row["admission_mismatch"]:
                mismatches.append(row)
    return {"records": rows, "mismatches": mismatches, "records_scanned": len(rows),
            "bounded_source_sections_scanned": sum(row["same_release_reissue"] is not None for row in rows),
            "same_release_reissues": {event: sum(row["same_release_reissue"] is True for row in rows if row["event"] == event)
                                      for event in ("cpi", "nfp")}}


def verify_bls_sources(root, path, event, section):
    ledger = json.loads((root / path).read_text())
    output, counts = [], {}
    for record in ledger["records"]:
        status = record["parse_status"]
        counts[status] = counts.get(status, 0) + 1
        if "PENDING" in status or "ERROR" in status or "FAIL" in status:
            raise AssertionError("Pending source capture cannot become missing feature coverage")
        url = record["source_url"]
        match = re.search(r"/(?:cpi|empsit)_(\d{8})\.htm$", url)
        if match is None:
            raise AssertionError("Official dated BLS archive identity required")
        url_date = datetime.strptime(match[1], "%m%d%Y").date()
        in_fence = section["source_publication_start"] <= url_date.isoformat() <= section["source_publication_end"]
        if not in_fence:
            if status != "OUTSIDE_PUBLICATION_FENCE":
                raise AssertionError("Out-of-fence source was treated as model input")
            continue
        if record.get("raw_provider_bytes_sha256") is not None:
            raise AssertionError("Tool text cannot be relabeled as raw provider bytes")
        snapshot = root / record["snapshot_path"]
        if digest(snapshot) != record["snapshot_sha256"]:
            raise AssertionError("BLS captured extraction hash differs")
        body = document_section(snapshot.read_text(), record.get("snapshot_source_url", url))
        if status == "ORIGINAL_VINTAGE_UNCERTAIN":
            evidence = record.get("source_revision_evidence")
            if record.get("historical_plan_admitted") is not False or not evidence or _plain(evidence) not in _plain(body):
                raise AssertionError("Reissued-source exclusion requires captured revision evidence")
            continue
        if status not in {"EXPLICIT_ORIGINAL_PLAN", "VERIFIED_EXPLICIT_PLAN"}:
            if status not in {"AMBIGUOUS_PRINTED_TIMESTAMP_OR_ID", "INTRINSICALLY_AMBIGUOUS"}:
                raise AssertionError(f"Unregistered source omission status: {status}")
            if record.get("original_plan_timestamp") is not None:
                raise AssertionError("Ambiguous source has an invented plan timestamp")
            continue
        if same_release_reissue(body):
            raise AssertionError("Explicit reissue notice prevents original-vintage plan admission")
        if record.get("historical_plan_admitted") is False:
            raise AssertionError("An explicitly excluded historical source cannot be admitted")
        publication_text, plan_text = record["source_publication_evidence"], record["original_plan_evidence"]
        if any(_plain(evidence) not in _plain(body) for evidence in (publication_text, plan_text)):
            raise AssertionError("Declared source evidence does not occur in its captured document section")
        publication, planned = document_timestamp(publication_text), document_timestamp(plan_text)
        if publication.date() != url_date:
            raise AssertionError("Parsed header date differs from archive document identity")
        if (publication.isoformat() != record["source_publication_timestamp"]
                or planned.isoformat() != record["original_plan_timestamp"]
                or planned.strftime("%Y-%m") != record["planned_calendar_month"]):
            raise AssertionError("Independent BLS timestamp parsing differs from source ledger")
        if not record.get("source_document_id") or record["source_document_id"] not in body:
            raise AssertionError("BLS document identifier lacks captured evidence")
        output.append({"event_type": event, "announced_at": publication.isoformat(), "planned_at": planned.isoformat(),
                       "source_id": url, "source_sha256": record["snapshot_sha256"]})
    return output, {"record_count": len(ledger["records"]), "explicit_plans_verified": len(output), "statuses": counts}


def verify_fomc_sources(root, sources):
    frame = pd.read_csv(root / sources["fomc"], dtype=str, keep_default_na=False)
    coverage = json.loads((root / sources["fomc_coverage"]).read_text())
    annual = []
    if len(frame) != 128 or set(frame.annual_schedule_year.astype(int)) != set(range(2010, 2026)):
        raise AssertionError("All128 original FOMC plans and16years are required")
    for year, rows in frame.groupby("annual_schedule_year", sort=True):
        first = rows.iloc[0]
        year = int(year)
        if (len(rows) != 8 or rows.planned_final_date.nunique() != 8 or rows.source_url.nunique() != 1
                or rows.source_extraction_sha256.nunique() != 1 or rows.planned_statement_time_et.ne("").any()):
            raise AssertionError("Original annual FOMC plan is incomplete or reinterpreted")
        source = root / first.source_extraction_path
        if digest(source) != first.source_extraction_sha256:
            raise AssertionError("FOMC source extraction hash differs")
        text = document_section(source.read_text(), first.source_url)
        plain = _plain(text)
        pub = re.search(rf"({MONTH_PATTERN}) (\d{{1,2}}), (\d{{4}})", plain)
        publication = datetime(int(pub[3]), MONTHS[pub[1]], int(pub[2])).date().isoformat()
        if publication != first.source_publication_date or not publication < f"{year}-01-01":
            raise AssertionError("Original FOMC source was not available before its year")
        if year == 2013:
            plain = plain.split("Tentative meeting schedule for 2013:", 1)[1]
        elif year == 2025:
            plain = plain.split("For 2025:", 1)[1].split("For 2026:", 1)[0]
        for _, row in rows.iterrows():
            evidence = row.source_meeting_text
            if _plain(evidence) not in plain:
                raise AssertionError("FOMC date evidence is not in its selected original annual block")
            parts = re.findall(rf"({MONTH_PATTERN}) (\d{{1,2}})", evidence)
            if len(parts) == 2:
                start_parts, end_parts = parts
            elif len(parts) == 1:
                start_parts = parts[0]
                ending = re.search(r"\d-(\d{1,2})", evidence)
                end_parts = (start_parts[0], ending[1]) if ending else start_parts
            else:
                raise AssertionError("Ambiguous FOMC date evidence")
            start_date = datetime(year, MONTHS[start_parts[0]], int(start_parts[1])).date()
            end_date = datetime(year, MONTHS[end_parts[0]], int(end_parts[1])).date()
            if start_date.isoformat() != row.planned_start_date or end_date.isoformat() != row.planned_final_date:
                raise AssertionError("Independent FOMC original date reconstruction differs")
        covered = [item for item in coverage if item["annual_schedule_year"] == year]
        if (len(covered) != 1 or covered[0]["coverage_start_date"] != f"{year}-01-01"
                or covered[0]["coverage_end_date"] != f"{year}-12-31" or covered[0]["plan_count"] != 8
                or covered[0]["source_extraction_sha256"] != first.source_extraction_sha256):
            raise AssertionError("FOMC explicit annual coverage differs")
        annual.append({"year": year, "announced_date": publication, "final_dates": rows.planned_final_date.tolist(),
                       "source_id": first.source_url, "source_sha256": first.source_extraction_sha256})
    return annual


def reconstruct(root, protocol):
    sources, section = protocol["sources"], protocol["index"]
    limit = pd.Timestamp(section["source_end"])
    if limit >= pd.Timestamp(section["sealed_start"]):
        raise AssertionError("Protected source fence crossed")
    admission_scan = scan_source_admissions(root, protocol)
    if admission_scan["mismatches"]:
        raise AssertionError("All source admission mismatches: " + json.dumps(admission_scan["mismatches"]))
    bls, source_audit = [], {}
    for event in ("cpi", "nfp"):
        records, audit = verify_bls_sources(root, sources[event], event, protocol["calendar"])
        bls.extend(records)
        source_audit[event] = audit
    annual = verify_fomc_sources(root, sources)
    source_audit["fomc"] = {"annual_documents": len(annual), "planned_meetings": sum(len(record["final_dates"]) for record in annual)}
    daily = pd.read_parquet(root / sources["daily"], filters=[("date", "<=", limit)])
    if daily.index.max() > limit:
        raise AssertionError("Out-of-fence daily observation decoded")
    implied = {}
    for name in ("vxn", "vix"):
        raw = pd.read_csv(root / sources[name], dtype=str, usecols=["DATE", "CLOSE"])
        dates = pd.to_datetime(raw.DATE, format="%m/%d/%Y")
        selected = dates <= limit
        index = pd.DatetimeIndex(dates.loc[selected])
        if index.has_duplicates or not index.is_monotonic_increasing:
            raise AssertionError("Ordered unique implied-volatility dates required")
        implied[name] = pd.Series(pd.to_numeric(raw.loc[selected, "CLOSE"]).to_numpy(), index=index)
    cutoffs = pd.Series(daily.index, index=daily.index).shift()
    plans = calendar_features(daily.index, cutoffs, bls, annual)
    output = market_features(daily, pd.DataFrame(implied)).join(plans).loc[:, ALL_FEATURES]
    return output, second_moment_targets(daily), source_audit, bls, annual


def second_moment_targets(daily):
    factor = daily["adj close"] / daily.close
    adjustment = np.log(factor.shift(-1) / factor)
    overnight_log = np.log(daily.open.shift(-1) / daily.close) + adjustment
    dates = pd.Series(daily.index, index=daily.index).shift(-1)
    return pd.DataFrame({"y": overnight_log.pow(2).where(np.isfinite(overnight_log)),
                         "target_end": dates, "available_date": dates, "adjustment_jump": adjustment}, index=daily.index)


def market_features(daily, iv):
    if daily.index.has_duplicates or not daily.index.is_monotonic_increasing:
        raise ValueError("Ordered unique sessions required")
    prices = daily[["open", "high", "low", "close", "adj close"]]
    if (prices.where(prices.notna(), 1.) <= 0).any().any() or np.isinf(prices.to_numpy()).any():
        raise ValueError("Positive finite or missing prices required")
    daytime = np.log(daily.close / daily.open)
    overnight = np.log(daily["adj close"]).diff() - daytime
    gk = (.5 * np.log(daily.high / daily.low).pow(2)
          - (2 * np.log(2) - 1) * daytime.pow(2)).clip(lower=1e-10)
    total_variance = gk + np.log(daily.open / daily.close.shift()).pow(2)
    output = pd.DataFrame({"const": 1.}, index=daily.index)
    for suffix, width in (("d", 1), ("w", 5), ("m", 22)):
        output[f"on_rms_{suffix}"] = np.sqrt(overnight.pow(2).rolling(width, min_periods=width).mean()).shift()
        output[f"day_rms_{suffix}"] = np.sqrt(daytime.pow(2).rolling(width, min_periods=width).mean()).shift()
        output[f"lrv_{suffix}"] = np.log(total_variance.rolling(width, min_periods=width).mean()).shift()
    aligned = iv.reindex(daily.index)
    output["liv"] = np.log(aligned.vxn.where(aligned.vxn > 0)).shift()
    output["lvix"] = np.log(aligned.vix.where(aligned.vix > 0)).shift()
    for weekday in range(1, 5):
        output[f"entry_dow_{weekday}"] = (daily.index.dayofweek == weekday).astype(float)
    return output.loc[:, BASE[:-1]].replace([np.inf, -np.inf], np.nan)


def nominal_window(origin):
    stamp = pd.Timestamp(origin)
    if pd.isna(stamp) or stamp.tz is not None or stamp != stamp.normalize() or stamp.weekday() > 4:
        raise ValueError("Normalized weekday civil entry required")
    day = stamp.date()
    tomorrow = day + timedelta(days=1)
    while tomorrow.weekday() > 4:
        tomorrow += timedelta(days=1)
    start = datetime.combine(day, time(16), ZONE)
    end = datetime.combine(tomorrow, time(9, 30), ZONE)
    hours = (end.astimezone(UTC) - start.astimezone(UTC)).total_seconds() / 3600
    return start, end, hours


def _source_identity(record):
    if not record.get("source_id") or re.fullmatch("[0-9a-f]{64}", record.get("source_sha256", "")) is None:
        raise ValueError("Source identity and saved extraction SHA256 required")


def calendar_features(origins, cutoffs, bls_records, fomc_records):
    by_event_month, schedules, identities = {}, {}, set()
    for record in bls_records:
        _source_identity(record)
        event = record["event_type"]
        announced, planned = pd.Timestamp(record["announced_at"]), pd.Timestamp(record["planned_at"])
        if (event not in ("cpi", "nfp") or announced.tz is None or planned.tz is None
                or pd.isna(announced) or pd.isna(planned) or announced >= planned):
            raise ValueError("Valid dated original BLS plan required")
        announced, planned = announced.tz_convert(ZONE), planned.tz_convert(ZONE)
        key = event, planned.strftime("%Y-%m")
        if key in by_event_month or record["source_id"] in identities:
            raise ValueError("Ambiguous original monthly source")
        identities.add(record["source_id"])
        by_event_month[key] = (announced.date(), planned.to_pydatetime())
    for record in fomc_records:
        _source_identity(record)
        year = int(record["year"])
        announced = pd.Timestamp(record["announced_date"])
        dates = tuple(pd.Timestamp(value) for value in record["final_dates"])
        if (year in schedules or len(dates) != 8 or len(set(dates)) != 8
                or announced.tz is not None or announced >= pd.Timestamp(f"{year}-01-01")
                or any(value.tz is not None or value != value.normalize() or value.year != year for value in dates)):
            raise ValueError("A complete original annual schedule with eight dates is required")
        schedules[year] = announced.date(), {value.date() for value in dates}
    if not cutoffs.index.equals(origins):
        raise ValueError("Aligned calendar cutoffs required")
    rows = []
    for origin, cutoff in zip(origins, cutoffs, strict=True):
        start, end, hours = nominal_window(origin)
        known_cutoff = None if pd.isna(cutoff) else pd.Timestamp(cutoff).date()
        if known_cutoff is not None and known_cutoff >= origin.date():
            raise ValueError("Source cutoff must precede entry")
        covered_months, day = set(), start.date()
        while day <= end.date():
            covered_months.add(day.strftime("%Y-%m"))
            day += timedelta(days=1)
        row = {"nominal_hours": hours}
        for event in ("cpi", "nfp"):
            selected = [by_event_month.get((event, month)) for month in sorted(covered_months)]
            known = known_cutoff is not None and all(value is not None and value[0] < known_cutoff for value in selected)
            row[event + "_plan"] = float(any(start < value[1] <= end for value in selected)) if known else np.nan
        source = schedules.get(end.year)
        known = source is not None and known_cutoff is not None and source[0] < known_cutoff
        row["fomc_plan"] = float(end.date() in source[1]) if known else np.nan
        rows.append(row)
    return pd.DataFrame(rows, index=origins, columns=["nominal_hours", *ADDITIONS.values()])


def training_mask(complete, targets, fit_entry, sessions):
    sessions = pd.DatetimeIndex(sessions)
    position = sessions.get_indexer([pd.Timestamp(fit_entry)])[0]
    if position < 1:
        raise ValueError("Previous observed session required")
    cutoff = sessions[position - 1]
    return (complete & np.isfinite(targets.y) & (targets.y >= 0)
            & (targets.index < fit_entry) & targets.available_date.notna() & (targets.available_date <= cutoff))


def monthly_fit_origins(complete, start, end):
    selected = complete.index[complete & (complete.index >= pd.Timestamp(start)) & (complete.index <= pd.Timestamp(end))]
    return selected[~selected.to_period("M").duplicated()]


def measurement_mask(jumps):
    values = np.asarray(jumps, float)
    return np.isfinite(values) & (np.abs(values) <= 1e-5)


def proper_score(y, prediction):
    target, fitted = np.asarray(y, float), np.asarray(prediction, float)
    if (target.shape != fitted.shape or target.ndim != 1 or not np.isfinite(target).all()
            or not np.isfinite(fitted).all() or (target < 0).any() or (fitted <= 0).any()):
        raise ValueError("Nonnegative finite target and positive finite prediction required")
    with np.errstate(over="ignore", invalid="ignore"):
        answer = np.log(fitted) + target / fitted
    if not np.isfinite(answer).all():
        raise ValueError("Finite score required")
    return answer


def penalized_objective(beta, design, y, alpha=ALPHA):
    beta, x, target = np.asarray(beta, float), np.asarray(design, float), np.asarray(y, float)
    eta = x @ beta
    with np.errstate(over="ignore", invalid="ignore"):
        ratio = target * np.exp(-eta)
    value = float(np.mean(eta + ratio) + alpha * np.dot(beta[1:], beta[1:]))
    gradient = x.T @ (1 - ratio) / len(target)
    gradient[1:] += 2 * alpha * beta[1:]
    return value, gradient


def second_moment_prediction(training, y, application):
    x, q, a = np.asarray(training, float), np.asarray(y, float), np.asarray(application, float)
    if (x.ndim != 2 or a.ndim != 2 or x.shape[1] != a.shape[1] or x.shape[1] < 1
            or len(x) < 2 or q.shape != (len(x),) or not np.isfinite(x).all() or not np.isfinite(a).all()
            or not np.isfinite(q).all() or (q < 0).any() or not (x[:, 0] == 1).all() or not (a[:, 0] == 1).all()):
        raise ValueError("Finite aligned common data with intercept required")
    target_mean = float(q.mean())
    if not 0 < target_mean < np.inf:
        raise ValueError("Positive finite training mean required")
    means, scales = x[:, 1:].mean(axis=0), x[:, 1:].std(axis=0, ddof=0)
    if not np.isfinite(scales).all() or (scales <= 1e-12).any():
        raise ValueError("Zero-scale input cannot be replaced")
    z, az = np.c_[np.ones(len(x)), (x[:, 1:] - means) / scales], np.c_[np.ones(len(a)), (a[:, 1:] - means) / scales]
    fit = minimize(penalized_objective, np.zeros(x.shape[1]), args=(z, q / target_mean, ALPHA),
                   method="BFGS", jac=True, options={"gtol": 1e-9, "maxiter": 1000})
    value, gradient = penalized_objective(fit.x, z, q / target_mean)
    if not np.isfinite(value) or np.max(np.abs(gradient)) > INDEPENDENT_GRADIENT_TOLERANCE:
        raise AssertionError("Independent BFGS did not meet its fixed KKT tolerance")
    actual_beta = fit.x.copy()
    actual_beta[0] += np.log(target_mean)
    prediction = np.exp(az @ actual_beta)
    if not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise AssertionError("Independent positive forecast failed")
    return prediction, {"means": means, "scales": scales, "beta": actual_beta, "scaled_beta": fit.x,
                        "train_mean": target_mean, "objective": value + np.log(target_mean),
                        "gradient_max_abs": float(np.max(np.abs(gradient))), "train_n": len(q)}


def qualifies_effect(phases):
    return (len(phases) == 2 and all(phase["delta"] <= -.005 and phase["action_sensitivity"]["delta"] < 0 for phase in phases)
            and len(phases[1]["stability"]) == 2 and all(item["delta"] < 0 for item in phases[1]["stability"]))


def eligible_entries(features, targets, section):
    complete = pd.Series(np.isfinite(features.loc[:, ALL_FEATURES].to_numpy()).all(axis=1), index=features.index)
    phases = pd.Series(False, index=features.index)
    for name in ("development", "evaluation"):
        first, last = map(pd.Timestamp, section[name])
        phases |= (features.index >= first) & (features.index <= last)
    in_bounds = ((features.index >= pd.Timestamp(section["origin_start"]))
                 & (features.index <= pd.Timestamp(section["origin_end"])))
    applications = features.index[complete & phases & in_bounds]
    valid = (np.isfinite(targets.y) & (targets.y >= 0) & np.isfinite(targets.adjustment_jump)
             & targets.available_date.notna() & (targets.available_date <= pd.Timestamp(section["latest_target"])))
    dev_end = pd.Timestamp(section["development_target_available_by"])
    valid &= (features.index > pd.Timestamp(section["development"][1])) | (targets.available_date <= dev_end)
    scored = applications[valid.loc[applications].to_numpy()]
    return applications, scored, complete


def verify_forecasts(features, targets, forecasts, fits, protocol):
    section = protocol["index"]
    required = {"origin", "model", "horizon", "feature_cutoff_date", "target_end", "available_date", "y", "prediction",
                "adjustment_event", "adjustment_log_change", "fit_origin", "fit_cutoff_date", "train_n",
                "train_last_target", "train_last_available", "phase"}
    if not required.issubset(forecasts) or set(forecasts.model) != set(MODELS) or set(forecasts.horizon) != {1}:
        raise AssertionError("All five models and explicit macro forecast fields required")
    if forecasts.duplicated(["origin", "model", "horizon"]).any():
        raise AssertionError("Duplicated macro forecasts")
    entries, scored, complete = eligible_entries(features, targets, section)
    cutoff_dates = pd.Series(features.index, index=features.index).shift()
    for model in MODELS:
        rows = forecasts.loc[forecasts.model == model].sort_values("origin")
        if not pd.DatetimeIndex(rows.origin).equals(scored):
            raise AssertionError("Every macro arm must use exactly the independently completed common rows")
        expected = targets.loc[scored]
        same(rows.y, expected.y, "Squared adjusted overnight targets", rtol=1e-9, atol=1e-14)
        same(rows.adjustment_log_change, expected.adjustment_jump, "Adjusted-factor measurement changes", rtol=1e-10, atol=1e-13)
        if not np.array_equal(rows.adjustment_event.to_numpy(), ~measurement_mask(expected.adjustment_jump)):
            raise AssertionError("Macro measurement-event indicator differs")
        for name in ("target_end", "available_date"):
            if not np.array_equal(rows[name].to_numpy(), expected[name].to_numpy()):
                raise AssertionError("Macro label dates differ")
        if not np.array_equal(rows.feature_cutoff_date.to_numpy(), cutoff_dates.loc[scored].to_numpy()):
            raise AssertionError("Macro features must stop at the prior observed session")
        phases = np.where(scored <= pd.Timestamp(section["development"][1]), "development", "evaluation")
        if not np.array_equal(rows.phase.to_numpy(), phases):
            raise AssertionError("Macro development/evaluation labels differ")
    lookup = {pd.Timestamp(record["fit_origin"]): record for record in fits}
    expected_fits = entries[~entries.to_period("M").duplicated()]
    if len(lookup) != len(fits) or set(lookup) != set(expected_fits):
        raise AssertionError("Monthly fits must start at the first feature-complete entry, independent of its label")
    count, producer_kkt_max, independent_kkt_max = 0, 0., 0.
    for fit_entry in expected_fits:
        application = entries[entries.to_period("M") == fit_entry.to_period("M")]
        query = scored[scored.to_period("M") == fit_entry.to_period("M")]
        record = lookup[fit_entry]
        selected = training_mask(complete, targets, fit_entry, features.index)
        training_origins = features.index[selected]
        n, cutoff = int(selected.sum()), cutoff_dates.loc[fit_entry]
        if n < section["minimum_train"] or record["train_n"] != n or record["application_n"] != len(application):
            raise AssertionError("Macro training/application counts differ")
        dates = {"fit_cutoff_date": cutoff, "train_first_origin": training_origins[0],
                 "train_last_origin": training_origins[-1], "train_last_target": targets.loc[selected, "target_end"].max(),
                 "train_last_available": targets.loc[selected, "available_date"].max()}
        if any(pd.Timestamp(record[key]) != value for key, value in dates.items()):
            raise AssertionError("Macro fit information dates differ")
        if dates["train_last_available"] > cutoff:
            raise AssertionError("Unavailable adjusted target entered a macro fit")
        if set(record["model_audit"]) != set(MODELS):
            raise AssertionError("Macro monthly fit omitted a registered model")
        train_all = features.loc[selected, ALL_FEATURES].to_numpy(float)
        if (train_all[:, 1:].std(axis=0, ddof=0) <= 1e-12).any():
            raise AssertionError("Degenerate common input cannot be dropped after inspection")
        y = targets.loc[selected, "y"].to_numpy(float)
        for model in MODELS:
            columns = ("const",) if model == "mean" else BASE + (() if model == "baseline" else (ADDITIONS[model],))
            audit = record["model_audit"][model]
            if tuple(audit["columns"]) != columns or audit["train_n"] != n:
                raise AssertionError("Macro model design or sample differs")
            training = features.loc[selected, columns].to_numpy(float)
            apply = features.loc[query, columns].to_numpy(float)
            prediction, rebuilt = second_moment_prediction(training, y, apply)
            same(audit["train_mean"], rebuilt["train_mean"], "Macro target scaling from training only", rtol=1e-9, atol=1e-14)
            same(audit["beta"], rebuilt["beta"], "Independent macro BFGS coefficients", rtol=0, atol=COEFFICIENT_ATOL)
            if model == "mean":
                same(audit["gradient_max_abs"], 0., "Unpenalized historical-mean optimum")
                replay = np.repeat(y.mean(), len(query))
            else:
                if audit["alpha"] != ALPHA or not 0 <= audit["iterations"] < 200 or audit["backtracks"] < 0:
                    raise AssertionError("Fixed macro objective or optimizer budget differs")
                same(audit["means"], rebuilt["means"], "Macro train-only centers")
                same(audit["scales"], rebuilt["scales"], "Macro population train-only scales")
                z = np.c_[np.ones(n), (training[:, 1:] - rebuilt["means"]) / rebuilt["scales"]]
                beta = np.asarray(audit["beta"], float).copy()
                beta[0] -= np.log(y.mean())
                same(audit["scaled_beta"], beta, "Recorded scaled/unscaled intercept identity", rtol=1e-9, atol=1e-12)
                value, gradient = penalized_objective(beta, z, y / y.mean())
                maximum = float(np.max(np.abs(gradient)))
                if maximum > PRODUCER_GRADIENT_TOLERANCE + 1e-12:
                    raise AssertionError("Saved macro coefficients fail independently evaluated KKT equations")
                same(audit["gradient_max_abs"], maximum, "Saved macro first-order residual", rtol=1e-4, atol=1e-12)
                same(audit["objective"], value + np.log(y.mean()), "Saved mean quasi-likelihood plus standardized slope penalty", rtol=1e-9, atol=1e-12)
                az = np.c_[np.ones(len(query)), (apply[:, 1:] - rebuilt["means"]) / rebuilt["scales"]]
                replay = np.exp(np.log(y.mean()) + az @ np.asarray(audit["scaled_beta"]))
                producer_kkt_max = max(producer_kkt_max, maximum)
            independent_kkt_max = max(independent_kkt_max, rebuilt["gradient_max_abs"])
            rows = forecasts.loc[(forecasts.model == model) & forecasts.origin.isin(query)].sort_values("origin")
            if not rows.fit_origin.eq(fit_entry).all() or not rows.fit_cutoff_date.eq(cutoff).all() or not rows.train_n.eq(n).all():
                raise AssertionError("Macro forecast fit provenance differs")
            for name in ("train_last_target", "train_last_available"):
                if not rows[name].eq(dates[name]).all():
                    raise AssertionError("Macro forecast label-maturity audit differs")
            same(rows.prediction, replay, "Exact saved-parameter forecast replay", rtol=1e-9, atol=1e-12)
            same(rows.prediction, prediction, "Independent BFGS macro forecasts", rtol=FORECAST_RTOL, atol=1e-12)
            count += len(rows)
    return {"forecasts_verified": count, "common_scored_origins": len(scored), "feature_complete_applications": len(entries),
            "monthly_fits_verified": len(expected_fits), "models_verified": len(MODELS),
            "producer_kkt_max": producer_kkt_max, "independent_bfgs_kkt_max": independent_kkt_max,
            "adjustment_flagged_origins": int((~measurement_mask(targets.loc[scored, "adjustment_jump"])).sum())}


def phase_statistics(panel, candidate, control, phase, phase_code, protocol):
    section = protocol["index"]
    first, last = section[phase]
    selected = panel.loc[(panel.origin >= first) & (panel.origin <= last)]
    if phase == "development":
        selected = selected.loc[selected.available_date <= pd.Timestamp(section["development_target_available_by"])]
    wide = selected.pivot(index="origin", columns="model", values="prediction").sort_index()
    actual = selected.loc[selected.model == control].set_index("origin").reindex(wide.index)
    candidate_loss = proper_score(actual.y, wide[candidate])
    control_loss = proper_score(actual.y, wide[control])
    difference = candidate_loss - control_loss
    if len(difference) < max(protocol["inference"]["blocks"]):
        raise AssertionError("Too few phase observations for fixed inference blocks")
    delta = float(difference.mean())
    hac = independent_hac(difference)
    blocks = {}
    for width in protocol["inference"]["blocks"]:
        seed = protocol["inference"]["seed"] + phase_code * 10000 + width
        samples = explicit_bootstrap_means(difference, width, protocol["inference"]["bootstrap_draws"], seed)[:, 0]
        probability = float((1 + np.count_nonzero(np.abs(samples - delta) >= abs(delta))) / (len(samples) + 1))
        blocks[str(width)] = {"p": probability, "ci95": np.quantile(samples, [.025, .975]).tolist()}
    intervals = [hac["ci95"], *(item["ci95"] for item in blocks.values())]
    unflagged = measurement_mask(actual.adjustment_log_change.to_numpy(float))
    if not unflagged.any():
        raise AssertionError("No unflagged phase measurements")
    sensitivity = {"n": int(unflagged.sum()), "delta": float(difference[unflagged].mean()),
                   "candidate_loss": float(candidate_loss[unflagged].mean()), "control_loss": float(control_loss[unflagged].mean())}
    output = {"name": phase, "first_origin": str(wide.index[0].date()), "last_origin": str(wide.index[-1].date()),
              "n": len(wide), "delta": delta, "candidate_loss": float(candidate_loss.mean()), "control_loss": float(control_loss.mean()),
              "block_inference": blocks, "hac126": hac,
              "ci95_envelope": [min(interval[0] for interval in intervals), max(interval[1] for interval in intervals)],
              "p_conservative": max(hac["p"], *(item["p"] for item in blocks.values())), "action_sensitivity": sensitivity,
              "annual": [], "stability": [], "nonoverlap_phases": [{"phase": 0, "n": len(wide), "delta": delta}]}
    for year in sorted(set(wide.index.year)):
        mask = wide.index.year == year
        output["annual"].append({"year": int(year), "n": int(mask.sum()), "delta": float(difference[mask].mean())})
    if phase == "evaluation":
        for start, end in section["evaluation_stability"]:
            mask = (wide.index >= start) & (wide.index <= end)
            if not mask.any():
                raise AssertionError("Fixed evaluation stability period is empty")
            output["stability"].append({"start": start, "end": end, "n": int(mask.sum()), "delta": float(difference[mask].mean())})
    return output


def inherited_rows(root, protocol):
    output = []
    for source in protocol["comparisons"]["inherited_sources"]:
        for row in json.loads((root / source).read_text())["rows"]:
            output.append({"study": row.get("study", source.split("/")[1]), "candidate": row["candidate"], "horizon": row["horizon"],
                           "control": row.get("control", "baseline"), "p_conservative": row["p_conservative"],
                           "source": source, "source_sha256": digest(root / source)})
    if len(output) != 78:
        raise AssertionError("All78 inherited comparisons must remain")
    return output


def verify_metrics(root, panel, protocol, metrics):
    rows = metrics["rows"]
    identities = [(row["candidate"], row["control"]) for row in rows]
    if len(rows) != 6 or len(set(identities)) != 6 or set(identities) != set(COMPARISONS):
        raise AssertionError("Every registered macro comparison must remain")
    probabilities, effects = [], []
    for row in rows:
        if row["study"] != "macro_overnight" or row["horizon"] != 1:
            raise AssertionError("Macro hypothesis identity differs")
        phases = [phase_statistics(panel, row["candidate"], row["control"], name, code, protocol)
                  for code, name in enumerate(("development", "evaluation"))]
        same_tree(row["phases"], phases, "Independently reconstructed macro phase inference")
        probability = max(phase["p_conservative"] for phase in phases)
        same(row["p_conservative"], probability, "Macro two-phase conjunction probability")
        probabilities.append(probability)
        effects.append(qualifies_effect(phases))
    prior = inherited_rows(root, protocol)
    same_tree(metrics["inherited_rows"], prior, "Complete inherited macro research family")
    wave = holm(probabilities)
    cumulative = holm([row["p_conservative"] for row in prior] + probabilities)[-6:]
    passed = {}
    for number, row in enumerate(rows):
        same(row["p_holm_wave"], wave[number], "Macro six-way Holm")
        same(row["p_holm_cumulative"], cumulative[number], "Macro84-way cumulative Holm")
        eligible = bool(effects[number] and wave[number] < WAVE_ALPHA and cumulative[number] < .05)
        passed[row["candidate"], row["control"]] = eligible
        if row["verdict"] != ("EXPLORATORY_LEAD" if eligible else "DOES_NOT_QUALIFY"):
            raise AssertionError("Macro comparison verdict differs")
    leads = [{"candidate": candidate, "horizon": 1} for candidate in ("cpi", "nfp")
             if all(passed[candidate, control] for control in ("baseline", "mean"))]
    timing = all(passed["fomc", control] for control in ("baseline", "mean"))
    if (metrics["leads"] != leads or metrics["fomc_timing_control_passes"] != timing
            or metrics["hypothesis_count"] != 6 or metrics["cumulative_hypothesis_count"] != 84):
        raise AssertionError("Macro candidates must pass both controls; FOMC remains a timing-control arm")
    return {"new_hypotheses_verified": 6, "cumulative_hypotheses_verified": 84, "phase_comparisons_verified": 12,
            "bootstrap_runs_verified": 36, "bootstrap_draws_per_run": protocol["inference"]["bootstrap_draws"],
            "measurement_sensitivities_verified": 12, "leads": leads, "fomc_timing_control_passes": timing}


def verify_source_audit(features, bls, annual, checked, saved):
    expected = {event: {"records": checked[event]["record_count"], "admitted": checked[event]["explicit_plans_verified"],
                        "states": checked[event]["statuses"]} for event in ("cpi", "nfp")}
    expected["fomc"] = checked["fomc"]
    source_rows = []
    cutoffs = pd.Series(features.index, index=features.index).shift()
    for origin, cutoff in zip(features.index, cutoffs, strict=True):
        start, end, _ = nominal_window(origin)
        evidence = {"origin": str(origin.date()), "nominal_start": start.isoformat(), "nominal_end": end.isoformat(),
                    "source_rule": "publication_date strictly before preceding observed market-session date"}
        months = {period.strftime("%Y-%m") for period in pd.period_range(start.date(), end.date(), freq="M")}
        for event in ("cpi", "nfp"):
            available = [record for record in bls if record["event_type"] == event and not pd.isna(cutoff)
                         and pd.Timestamp(record["announced_at"]).tz_convert(ZONE).date() < cutoff.date()
                         and pd.Timestamp(record["planned_at"]).tz_convert(ZONE).strftime("%Y-%m") in months]
            covered = {pd.Timestamp(record["planned_at"]).tz_convert(ZONE).strftime("%Y-%m") for record in available}
            evidence[event + "_missing_months"] = sorted(months - covered)
            evidence[event + "_source_ids"] = sorted(record["source_id"] for record in available)
            evidence[event + "_source_hashes"] = sorted(record["source_sha256"] for record in available)
        schedule = next((record for record in annual if record["year"] == end.year), None)
        known = schedule is not None and not pd.isna(cutoff) and pd.Timestamp(schedule["announced_date"]) < cutoff
        evidence["fomc_source_id"] = schedule["source_id"] if known else None
        evidence["fomc_source_sha256"] = schedule["source_sha256"] if known else None
        source_rows.append(evidence)
    expected["plan_availability"] = source_rows
    expected["plan_counts_full_bounded_calendar"] = {name: int(features[name].sum()) for name in ADDITIONS.values()}
    expected["unknown_rows"] = int(features.loc[:, ["nominal_hours", *ADDITIONS.values()]].isna().any(axis=1).sum())
    same_tree(saved, expected, "Independently reconstructed source eligibility and original-plan joins")
    return {"source_availability_rows_verified": len(source_rows), **checked}


def verify(root=ROOT):
    root = Path(root)
    report, output = root / "reports/macro_overnight", root / "data/macro_overnight"
    protocol_file = root / "macro_overnight.yaml"
    protocol = yaml.safe_load(protocol_file.read_text())
    manifest = json.loads((report / "manifest.json").read_text())
    if manifest["protocol_sha256"] != digest(protocol_file):
        raise AssertionError("Registered macro protocol changed")
    for group in ("code", "inputs", "preserved"):
        for path, expected in manifest[group].items():
            if digest(root / path) != expected:
                raise AssertionError(f"Frozen macro provenance or earlier artifact changed: {path}")
    features, targets, source_audit, bls, annual = reconstruct(root, protocol)
    recorded_features = pd.read_parquet(output / "features.parquet")
    if not recorded_features.index.equals(features.index) or tuple(recorded_features.columns) != ALL_FEATURES + ("feature_cutoff_date",):
        raise AssertionError("Macro feature dates or schema differ")
    same(recorded_features.loc[:, ALL_FEATURES], features, "Every independently reconstructed macro input", rtol=1e-10, atol=1e-12)
    expected_cutoffs = pd.Series(features.index, index=features.index).shift()
    if not np.array_equal(recorded_features.feature_cutoff_date.to_numpy(), expected_cutoffs.to_numpy(), equal_nan=True):
        raise AssertionError("Macro recorded feature cutoff dates differ")
    result = {"status": "VERIFIED", "protocol_sha256": digest(protocol_file), "verifier_sha256": digest(Path(__file__)),
              "feature_rows_verified": len(features), "feature_columns_verified": len(ALL_FEATURES)}
    saved_sources = json.loads((output / "source_audit.json").read_text())
    result["source_reconstruction"] = verify_source_audit(features, bls, annual, source_audit, saved_sources)
    forecasts = pd.read_parquet(output / "forecasts.parquet")
    fits = json.loads((output / "fits.json").read_text())
    result["forecast_reconstruction"] = verify_forecasts(features, targets, forecasts, fits, protocol)
    metrics = json.loads((report / "metrics.json").read_text())
    if metrics["protocol_sha256"] != digest(protocol_file) or metrics["evidence_class"] != protocol["evidence_class"]:
        raise AssertionError("Macro metric provenance differs")
    result["inference"] = verify_metrics(root, forecasts, protocol, metrics)
    ledger = [json.loads(line) for line in (report / "trial_ledger.jsonl").read_text().splitlines() if line]
    for event, expected in (("inherited", metrics["inherited_rows"]), ("evaluated", metrics["rows"])):
        actual = [{key: value for key, value in row.items() if key != "event"} for row in ledger if row["event"] == event]
        same_tree(actual, expected, f"Macro {event} trial ledger")
    registered = [row for row in ledger if row["event"] == "registered"]
    if (len(ledger) != 90 or len(registered) != 6
            or {(row["candidate"], row["control"]) for row in registered} != set(COMPARISONS)
            or any(row["protocol_sha256"] != digest(protocol_file) or row["horizon"] != 1 for row in registered)):
        raise AssertionError("Complete78inherited+6registered+6evaluated macro ledger required")
    result["ledger_events_verified"] = {"inherited": 78, "registered": 6, "evaluated": 6}
    result["limitations"] = ["Adaptive historical reuse remains exploratory after multiplicity corrections.",
                            "Original documentary plans and nominal civil windows do not certify exchange holding intervals.",
                            "Adjusted overnight second moments are not integrated variance or executable cash profit.",
                            "Current official-document extracts are not immutable historical web vintages."]
    (report / "verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, allow_nan=False))
