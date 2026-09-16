"""Compare pinned Oxford source vintages; no modelling or predictive evaluation.

Run only after tests.test_audit_harnet_extension. Outputs are additive and refuse
overwrite. Tolerances and boundary diagnostics are fixed before raw comparison.
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOLS = (".SPX", ".N225", ".HSI", ".KS11", ".FTSE", ".GDAXI", ".FCHI")
FIELDS = ("open_to_close", "rsv", "medrv", "rsv_ss", "nobs", "rv5",
          "close_price", "rv10", "open_price")
RTOL = 1e-10
ATOL = 1e-12
MISSING = frozenset(("", "na", "nan", "n/a", "null", "none", "<na>", "#n/a"))
OLD_SHA256 = "e0dd80edc0c2cedac5ed3f72250ee4460e963b4efd458d68525a61bcc5c27ea2"
NEW_GZIP_SHA256 = "30ef85340891ea9b8fdd69b8f15e459037c2586505096a0fa27c9ea546685143"
NEW_CSV_SHA256 = "c6be6c68280e9100ce734721a4c05bd557ec9009053ad7526fda20b6bd62851e"
CONTRACT = {
    "evidence_class": "measurement_audit_not_predictive_validation",
    "symbols": SYMBOLS, "fields": FIELDS,
    "date_key": "exact Symbol plus validated first ten local date characters",
    "floating_tolerance": {"atol": ATOL, "rtol_against_old": RTOL},
    "nobs_tolerance": "exact decimal equality",
    "equality_levels": ["literal", "decimal", "float64", "frozen_default_pandas_parser"],
    "field_equivalence": "every pair finite and within tolerance, or both missing; no invalid/nonfinite pairs",
    "exact_reproduction": "identical old-period keys and all rv5/rsv finite and identical under frozen parser",
    "discrepancy_quantiles": [0, .01, .05, .5, .95, .99, 1],
    "scale_diagnostic": {"minimum_positive_pairs": 20, "minimum_median_departure": .001,
                         "maximum_relative_dispersion": 1e-6},
    "boundary": {"archived_observations_per_side": 22, "first_ratio_range": [.01, 100],
                 "median_ratio_range": [.1, 10], "missing_rows": "retain"},
    "rsv_upper_tolerance": "rv5 * (1 + 1e-12)",
    "prohibited": ["fits", "predictive_scores", "imputation", "rescaling", "raw_edits",
                   "country_selection", "directional_relabelling"],
}


@dataclass
class SourceFrame:
    tokens: pd.DataFrame
    parsed: pd.DataFrame


def verify_hash(raw, expected):
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Source hash differs from pinned value")


def parse_csv(raw):
    """Keep decimal tokens plus the existing studies' default pandas interpretation."""
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    columns = reader.fieldnames
    if columns is None or not {"Symbol", *FIELDS}.issubset(columns):
        raise ValueError("Source schema missing required numeric fields")
    date_column = columns[0]
    records, keys = [], []
    for row in reader:
        if row["Symbol"] not in SYMBOLS:
            continue
        text = row[date_column]
        if text is None or not re.match(r"^\d{4}-\d{2}-\d{2}(?:$|[ T])", text):
            raise ValueError("Malformed local date label")
        date = text[:10]
        datetime.strptime(date, "%Y-%m-%d")
        if any(row[field] is None for field in FIELDS):
            raise ValueError("Malformed CSV row")
        keys.append((row["Symbol"], date))
        records.append({field: row[field] for field in FIELDS})
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate Symbol/local-date key")
    if {key[0] for key in keys} != set(SYMBOLS):
        raise ValueError("A registered market is absent")
    index = pd.MultiIndex.from_tuples(keys, names=["Symbol", "date"])
    tokens = pd.DataFrame(records, index=index, columns=FIELDS)
    # Deliberately match the frozen parser's default CSV float conversion, then
    # coerce malformed tokens for diagnostic counting rather than dropping rows.
    parsed = pd.read_csv(io.BytesIO(raw))
    parsed = parsed.loc[parsed.Symbol.isin(SYMBOLS)].copy()
    parsed.index = index
    parsed = parsed.loc[:, FIELDS].apply(pd.to_numeric, errors="coerce").astype(float)
    return SourceFrame(tokens, parsed)


def _number(token):
    token = str(token).strip()
    if token.lower() in MISSING:
        return "missing", None, np.nan
    try:
        decimal = Decimal(token)
        value = float(decimal)
        if not decimal.is_finite() or not np.isfinite(value):
            return "nonfinite", None, value
        return "finite", decimal, value
    except (InvalidOperation, ValueError, OverflowError):
        return "invalid", None, np.nan


def _distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    result = {"count": int(len(values))}
    names = ("min", "p01", "p05", "p50", "p95", "p99", "max")
    if not len(values):
        return result | dict.fromkeys(names)
    quantiles = np.quantile(values, CONTRACT["discrepancy_quantiles"])
    return result | {name: float(value) for name, value in zip(names, quantiles)}


def compare_field(old, new, field, old_parsed=None, new_parsed=None):
    """Describe paired raw values; never repair them or infer predictive direction."""
    old, new = list(old), list(new)
    if len(old) != len(new):
        raise ValueError("Paired field lengths differ")
    a, b = [_number(x) for x in old], [_number(x) for x in new]
    av = np.array([x[2] for x in a])
    bv = np.array([x[2] for x in b])
    finite = np.array([x[0] == y[0] == "finite" for x, y in zip(a, b)], dtype=bool)
    old_missing = np.array([x[0] == "missing" for x in a], dtype=bool)
    new_missing = np.array([x[0] == "missing" for x in b], dtype=bool)
    decimals_equal = np.array([f and x[1] == y[1] for f, x, y in zip(finite, a, b)],
                              dtype=bool)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        delta = bv - av
        equivalent = finite & (np.abs(delta) <= ATOL + RTOL * np.abs(av))
        if field == "nobs":
            equivalent = decimals_equal
        ratio_mask = finite & (av != 0)
        ratios = bv[ratio_mask] / av[ratio_mask]
        relative = np.abs(delta[ratio_mask] / av[ratio_mask])
    old_parser = av if old_parsed is None else np.asarray(old_parsed, dtype=float)
    new_parser = bv if new_parsed is None else np.asarray(new_parsed, dtype=float)
    if len(old_parser) != len(old) or len(new_parser) != len(new):
        raise ValueError("Parsed and raw lengths differ")
    both_missing = old_missing & new_missing
    if len(old) and np.all(decimals_equal | both_missing):
        status = "EXACT_NUMERIC_WITH_MATCHED_MISSINGNESS"
    elif len(old) and np.all(equivalent | both_missing):
        status = "TOLERANCE_EQUIVALENT"
    else:
        status = "MATERIAL_REVISION_OR_INVALID"
    positive = finite & (av > 0) & (bv > 0)
    constant_scale = None
    if positive.sum() >= 20:
        positive_ratios = bv[positive] / av[positive]
        median = float(np.median(positive_ratios))
        if abs(median - 1) > .001 and np.max(np.abs(positive_ratios / median - 1)) <= 1e-6:
            constant_scale = median
    return {
        "paired_rows": len(old), "paired_finite": int(finite.sum()),
        "literal_equal": int(sum(x == y for x, y in zip(old, new))),
        "decimal_equal": int(decimals_equal.sum()),
        "float64_equal": int((finite & (av == bv)).sum()),
        "frozen_parser_equal": int((finite & (old_parser == new_parser)).sum()),
        "both_missing": int(both_missing.sum()),
        "old_missing_only": int((old_missing & ~new_missing).sum()),
        "new_missing_only": int((new_missing & ~old_missing).sum()),
        "old_invalid_or_nonfinite": int(sum(x[0] in {"invalid", "nonfinite"} for x in a)),
        "new_invalid_or_nonfinite": int(sum(x[0] in {"invalid", "nonfinite"} for x in b)),
        "within_tolerance": int(equivalent.sum()),
        "material_difference": int((finite & ~equivalent).sum()),
        "signed_difference": _distribution(delta[finite]),
        "absolute_difference": _distribution(np.abs(delta[finite])),
        "absolute_relative_difference": _distribution(relative),
        "ratio_new_over_old": _distribution(ratios),
        "constant_scale_candidate": constant_scale, "status": status,
    }


def quality_summary(frame):
    result = {"rows": len(frame), "by_field": {}}
    for field in FIELDS:
        x = frame[field].to_numpy(float)
        result["by_field"][field] = {
            "finite": int(np.isfinite(x).sum()), "missing": int(np.isnan(x).sum()),
            "infinite": int(np.isinf(x).sum()),
            "negative": int((np.isfinite(x) & (x < 0)).sum()),
            "zero": int((x == 0).sum()),
        }
    result["rv5_nonpositive"] = int((frame.rv5 <= 0).sum())
    result["rsv_negative"] = int((frame.rsv < 0).sum())
    result["rsv_above_rv5"] = int((frame.rsv > frame.rv5 * (1 + 1e-12)).sum())
    result["variance_negative"] = {
        f: int((frame[f] < 0).sum()) for f in ("rv5", "rv10", "rsv", "rsv_ss", "medrv")}
    result["price_nonpositive"] = {
        f: int((frame[f] <= 0).sum()) for f in ("open_price", "close_price")}
    result["nobs_not_positive_integer"] = int((~np.isfinite(frame.nobs)
                                               | (frame.nobs <= 0)
                                               | (frame.nobs != np.floor(frame.nobs))).sum())
    return result


def _token_quality(frame):
    result = quality_summary(frame.parsed)
    for field in FIELDS:
        states = [_number(x)[0] for x in frame.tokens[field]]
        result["by_field"][field]["raw_states"] = {
            state: states.count(state) for state in ("finite", "missing", "nonfinite", "invalid")}
        result["by_field"][field]["missing_tokens"] = {
            token: int((frame.tokens[field] == token).sum())
            for token in sorted(set(frame.tokens[field])) if _number(token)[0] == "missing"}
    return result


def boundary_summary(old, extension, window=22):
    before = old.sort_index().iloc[-window:]
    after = extension.sort_index().iloc[:window]

    def describe(series):
        finite = series[np.isfinite(series)]
        return {"rows": len(series), "finite": len(finite),
                "missing_or_nonfinite": len(series) - len(finite),
                "first_date": str(series.index[0]) if len(series) else None,
                "last_date": str(series.index[-1]) if len(series) else None,
                "median": float(finite.median()) if len(finite) else None}

    a, b = describe(before), describe(after)
    first_ratio = median_ratio = None
    if (len(before) and len(after) and np.isfinite(before.iloc[-1])
            and before.iloc[-1] != 0 and np.isfinite(after.iloc[0])):
        first_ratio = float(after.iloc[0] / before.iloc[-1])
    if a["median"] is not None and a["median"] != 0 and b["median"] is not None:
        median_ratio = b["median"] / a["median"]
    return {"before": a, "after": b, "first_over_last": first_ratio,
            "median_ratio": median_ratio,
            "first_ratio_review": first_ratio is not None and not .01 <= first_ratio <= 100,
            "median_ratio_review": median_ratio is not None and not .1 <= median_ratio <= 10}


def audit_frames(old, new):
    overlap = old.tokens.index.intersection(new.tokens.index).sort_values()
    missing = old.tokens.index.difference(new.tokens.index)
    cutoffs = {s: old.tokens.loc[s].index.max() for s in SYMBOLS}
    extension_keys = [key for key in new.tokens.index if key[1] > cutoffs[key[0]]]
    extension = pd.MultiIndex.from_tuples(extension_keys, names=old.tokens.index.names)
    old_period_new = new.tokens.index.difference(extension)
    added_old = old_period_new.difference(old.tokens.index)

    def fields(keys):
        return {f: compare_field(old.tokens.loc[keys, f], new.tokens.loc[keys, f], f,
                                old.parsed.loc[keys, f], new.parsed.loc[keys, f]) for f in FIELDS}

    aggregate = fields(overlap)
    geometry_ok = not len(missing) and not len(added_old)
    exact = geometry_ok and all(aggregate[f]["frozen_parser_equal"] == len(old.tokens)
                               for f in ("rv5", "rsv"))
    equivalent = geometry_ok and all(aggregate[f]["within_tolerance"] == len(old.tokens)
                                    for f in ("rv5", "rsv"))
    if exact:
        reproduction = "EXACT_FROZEN_PARSER_INPUTS"
    elif equivalent:
        reproduction = "TOLERANCE_ONLY_NOT_EXACT"
    else:
        reproduction = "DOES_NOT_REPRODUCE"
    result = {
        "old_rows": len(old.tokens), "new_rows": len(new.tokens),
        "overlap_rows": len(overlap), "extension_rows": len(extension),
        "missing_old_keys": len(missing), "added_old_period_keys": len(added_old),
        "missing_old_key_list": list(missing), "added_old_key_list": list(added_old),
        "fields": aggregate, "rv5_rsv_reproduction": reproduction, "symbols": {},
    }
    for symbol in SYMBOLS:
        keys = overlap[overlap.get_level_values("Symbol") == symbol]
        old_part = SourceFrame(old.tokens.loc[[symbol]], old.parsed.loc[[symbol]])
        new_part = SourceFrame(new.tokens.loc[[symbol]], new.parsed.loc[[symbol]])
        ext_keys = extension[extension.get_level_values("Symbol") == symbol]
        ext_part = SourceFrame(new.tokens.loc[ext_keys], new.parsed.loc[ext_keys])
        result["symbols"][symbol] = {
            "old_rows": len(old_part.tokens), "new_rows": len(new_part.tokens),
            "overlap_rows": len(keys), "extension_rows": len(ext_keys),
            "old_last_date": cutoffs[symbol],
            "first_extension_date": min(ext_keys.get_level_values("date")) if len(ext_keys) else None,
            "last_extension_date": max(ext_keys.get_level_values("date")) if len(ext_keys) else None,
            "fields": fields(keys), "quality_old": _token_quality(old_part),
            "quality_new_all": _token_quality(new_part), "quality_extension": _token_quality(ext_part),
            "boundary": {f: boundary_summary(old.parsed.loc[symbol, f],
                                              new.parsed.loc[ext_keys, f].droplevel("Symbol"))
                         for f in FIELDS},
        }
    return result


def run_audit(root):
    root = Path(root)
    old_path = root / "data/raw/oxford_man_realized.zip"
    source_dir = root / "data/source_discovery/quarantine/harnet_2020"
    new_path = source_dir / "MAN_data.csv.gz"
    manifest_path = source_dir / "source_manifest.json"
    old_bytes, new_bytes = old_path.read_bytes(), new_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    verify_hash(old_bytes, OLD_SHA256)
    verify_hash(new_bytes, NEW_GZIP_SHA256)
    new_csv = gzip.decompress(new_bytes)
    verify_hash(new_csv, NEW_CSV_SHA256)
    with zipfile.ZipFile(io.BytesIO(old_bytes)) as archive:
        names = [x for x in archive.namelist() if x.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError("Exactly one old source CSV required")
        old_csv = archive.read(names[0])
    output = root / "data/source_discovery/harnet_numeric_audit"
    output.mkdir(exist_ok=True)
    contract_path, result_path = output / "contract.json", output / "audit.json"
    if contract_path.exists() or result_path.exists():
        raise FileExistsError("Audit outputs already exist; refuse overwrite")
    report_path = root / "reports/overnight_index/HARNET_EXTENSION_NUMERIC_AUDIT.md"
    contract_text = report_path.read_text().split("## Results")[0]
    identities = {
        "old_zip_sha256": OLD_SHA256, "old_csv_sha256": hashlib.sha256(old_csv).hexdigest(),
        "new_gzip_sha256": NEW_GZIP_SHA256, "new_csv_sha256": NEW_CSV_SHA256,
        "unchanged_source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tests_sha256": hashlib.sha256((root / "tests/test_audit_harnet_extension.py").read_bytes()).hexdigest(),
        "precomparison_report_contract_sha256": hashlib.sha256(contract_text.encode()).hexdigest(),
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
    }
    preflight = {"registered_before_comparison_utc": datetime.now(UTC).isoformat(),
                 "contract": CONTRACT, "source_and_code_identity": identities}
    with contract_path.open("x") as stream:
        json.dump(preflight, stream, indent=2, allow_nan=False)
    result = audit_frames(parse_csv(old_csv), parse_csv(new_csv))
    verify_hash(old_path.read_bytes(), OLD_SHA256)
    verify_hash(new_path.read_bytes(), NEW_GZIP_SHA256)
    verify_hash(manifest_path.read_bytes(), identities["unchanged_source_manifest_sha256"])
    result = preflight | {"completed_utc": datetime.now(UTC).isoformat(),
                          "source_files_unchanged": True, "audit": result}
    with result_path.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(run_audit(args.root))
