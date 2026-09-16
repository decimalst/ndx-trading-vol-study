"""Registered announcement-plan experiment and complete cumulative inference."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import macro_overnight
from . import orthogonal_round2 as inference
from .international_search import diagnostics
from .macro_plan_features import build_plan_features, validate_bls_plans
from .macro_second_moment import proper_score

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'macro_overnight.yaml'
REPORT = ROOT/'reports/macro_overnight'
OUT = ROOT/'data/macro_overnight'
CONTRASTS = [(candidate, control) for candidate in ['cpi', 'nfp', 'fomc'] for control in ['baseline', 'mean']]


def validate(p):
    i = p['index']
    if (p['wave'] != 4 or p['wave_alpha'] != .0025 or p['comparisons']['new_hypotheses'] != 6
            or p['comparisons']['cumulative_hypotheses'] != 84 or i['horizons'] != [1]
            or i['effect_threshold_absolute'] != .005 or i['penalty'] != .01 or i['minimum_train'] != 1000
            or i['latest_target'] != '2025-10-20' or i['sealed_start'] != '2025-11-03'
            or p['inference']['blocks'] != [21, 63, 126] or p['inference']['bootstrap_draws'] != 49999
            or p['inference']['seed'] != 20260910 or i['optimizer_max_iter'] != 200
            or i['gradient_tolerance'] != 1e-8 or i['armijo'] != 1e-4 or i['maximum_backtracks'] != 60
            or tuple(i['baseline']) != macro_overnight.BASE or tuple(i['models']) != macro_overnight.MODELS):
        raise ValueError('Fixed macro specification differs')


def paired_inference(candidate, control, protocol, seed):
    candidate, control = np.asarray(candidate, float), np.asarray(control, float)
    if candidate.shape != control.shape or candidate.ndim != 1 or not np.isfinite([candidate, control]).all():
        raise ValueError('Finite aligned proper losses required')
    d = candidate-control
    if len(d) < max(protocol['inference']['blocks']):
        raise ValueError('INSUFFICIENT_DATA: phase shorter than inference block')
    delta = float(d.mean())
    blocks = {}
    for block in protocol['inference']['blocks']:
        samples = inference.bootstrap_means(d, block, protocol['inference']['bootstrap_draws'], seed+block)[:, 0]
        p = (1+int((np.abs(samples-delta) >= abs(delta)).sum()))/(len(samples)+1)
        blocks[str(block)] = {'p': p, 'ci95': np.quantile(samples, [.025, .975]).tolist()}
    hac = inference.hac_summary(d)
    intervals = [hac['ci95']]+[item['ci95'] for item in blocks.values()]
    return {'n': len(d), 'delta': delta, 'candidate_loss': float(candidate.mean()), 'control_loss': float(control.mean()),
            'block_inference': blocks, 'hac126': hac,
            'ci95_envelope': [min(x[0] for x in intervals), max(x[1] for x in intervals)],
            'p_conservative': max(hac['p'], *(item['p'] for item in blocks.values()))}


def sensitivity(candidate, control, flagged):
    candidate, control, flagged = np.asarray(candidate), np.asarray(control), np.asarray(flagged)
    if (candidate.shape != control.shape or candidate.ndim != 1 or flagged.shape != candidate.shape
            or flagged.dtype != bool or not np.isfinite([candidate, control]).all()):
        raise ValueError('Aligned finite losses and boolean measurement flags required')
    keep = ~flagged
    if not keep.any():
        raise ValueError('INSUFFICIENT_DATA: no unflagged labels')
    return {'n': int(keep.sum()), 'delta': float((candidate[keep]-control[keep]).mean()),
            'candidate_loss': float(candidate[keep].mean()), 'control_loss': float(control[keep].mean())}


def inherited(p):
    rows = []
    for path in p['comparisons']['inherited_sources']:
        for row in json.loads((ROOT/path).read_text())['rows']:
            rows.append({'study': row.get('study', path.split('/')[1]), 'candidate': row['candidate'],
                         'horizon': row['horizon'], 'control': row.get('control', 'baseline'),
                         'p_conservative': row['p_conservative'], 'source': path, 'source_sha256': inference.digest(ROOT/path)})
    if len(rows) != 78:
        raise ValueError('All 78 previously enumerated contrasts must remain')
    return rows


def adjust(prior, rows):
    if len(prior) != 78 or len(rows) != 6:
        raise ValueError('Incomplete cumulative comparison family')
    return (inference.holm_adjust([r['p_conservative'] for r in rows]),
            inference.holm_adjust([r['p_conservative'] for r in prior+rows])[-6:])


def passes(row):
    phases = row['phases']
    stability = next((phase.get('stability', []) for phase in phases if phase['name'] == 'evaluation'), [])
    return (len(phases) == 2 and row['p_holm_wave'] < .0025 and row['p_holm_cumulative'] < .05
            and all(phase['delta'] <= -.005 and phase['action_sensitivity']['n'] > 0
                    and phase['action_sensitivity']['delta'] < 0 for phase in phases)
            and len(stability) == 2 and all(item['delta'] < 0 for item in stability))


def failure_rows(error):
    status = 'INSUFFICIENT_DATA' if 'INSUFFICIENT_DATA' in str(error) else 'INVALID_RUN'
    return [{'study': 'macro_overnight', 'horizon': 1, 'candidate': candidate, 'control': control,
             'status': status, 'error': str(error), 'p_conservative': 1., 'phases': [],
             'p_holm_wave': 1., 'p_holm_cumulative': 1., 'verdict': 'UNEVALUABLE'}
            for candidate, control in CONTRASTS]


def _snapshot(path, expected):
    file = ROOT/path
    if not file.resolve().is_relative_to(ROOT) or inference.digest(file) != expected:
        raise ValueError('Source extraction identity differs')


def load_plan_records(p):
    bls, source_audit = [], {}
    for event in ['cpi', 'nfp']:
        document = json.loads((ROOT/p['sources'][event]).read_text())
        records = document['records']
        statuses, admitted = {}, 0
        for record in records:
            state = record.get('parse_status', record.get('status', 'UNKNOWN'))
            statuses[state] = statuses.get(state, 0)+1
            if 'PENDING' in state or 'PENDING' in record.get('status', ''):
                raise ValueError('INSUFFICIENT_DATA: calendar retrieval remains pending')
            if state not in {'VERIFIED_EXPLICIT_PLAN', 'EXPLICIT_ORIGINAL_PLAN'}:
                continue
            publication = pd.Timestamp(record['source_publication_timestamp'])
            planned = pd.Timestamp(record['original_plan_timestamp'])
            if publication.tz is None or planned.tz is None:
                raise ValueError('Calendar source time requires explicit zone')
            local_date = publication.tz_convert('America/New_York').strftime('%Y-%m-%d')
            if not p['calendar']['source_publication_start'] <= local_date <= p['calendar']['source_publication_end']:
                raise ValueError('Admitted calendar source is outside declared publication fence')
            _snapshot(record['snapshot_path'], record['snapshot_sha256'])
            if record['planned_calendar_month'] != planned.tz_convert('America/New_York').strftime('%Y-%m'):
                raise ValueError('Original plan month differs from its timestamp')
            bls.append({'event_type': event, 'announced_at': publication.isoformat(), 'planned_at': planned.isoformat(),
                        'source_id': record['source_url'], 'source_sha256': record['snapshot_sha256']})
            admitted += 1
        source_audit[event] = {'records': len(records), 'admitted': admitted, 'states': statuses}
    validate_bls_plans(bls)
    frame = pd.read_csv(ROOT/p['sources']['fomc'], dtype=str, keep_default_na=False)
    coverage = json.loads((ROOT/p['sources']['fomc_coverage']).read_text())
    annual = []
    for year, rows in frame.groupby('annual_schedule_year', sort=True):
        for column in ['source_publication_date', 'source_url', 'source_extraction_sha256', 'source_extraction_path']:
            if rows[column].nunique() != 1:
                raise ValueError('FOMC year must use one complete original annual document')
        first = rows.iloc[0]
        _snapshot(first.source_extraction_path, first.source_extraction_sha256)
        cover = [item for item in coverage if item['annual_schedule_year'] == int(year)]
        if (len(cover) != 1 or cover[0]['plan_count'] != 8 or len(rows) != 8
                or cover[0]['source_extraction_sha256'] != first.source_extraction_sha256
                or rows.planned_statement_time_et.ne('').any()):
            raise ValueError('Incomplete or relabelled annual FOMC source')
        annual.append({'year': int(year), 'announced_date': first.source_publication_date,
                       'final_dates': rows.planned_final_date.tolist(), 'source_id': first.source_url,
                       'source_sha256': first.source_extraction_sha256})
    if len(annual) != 16 or {record['year'] for record in annual} != set(range(2010, 2026)):
        raise ValueError('All 16 original FOMC annual plans are required')
    source_audit['fomc'] = {'annual_documents': len(annual), 'planned_meetings': len(frame)}
    return bls, annual, source_audit


def load_inputs(p):
    end, sealed = pd.Timestamp(p['index']['source_end']), pd.Timestamp(p['index']['sealed_start'])
    if end >= sealed:
        raise ValueError('Protected source fence crossed')
    daily = pd.read_parquet(ROOT/p['sources']['daily'], filters=[('date', '<=', end)]).loc[:end]
    iv = {}
    for event in ['vxn', 'vix']:
        frame = pd.read_csv(ROOT/p['sources'][event])
        dates = pd.to_datetime(frame.DATE, format='%m/%d/%Y')
        selected = dates <= end
        iv[event] = pd.Series(frame.loc[selected, 'CLOSE'].to_numpy(float), index=pd.DatetimeIndex(dates[selected]), name=event)
    bls, annual, audit = load_plan_records(p)
    cutoff = pd.Series(daily.index, index=daily.index).shift(1)
    plans, evidence = build_plan_features(daily.index, cutoff, bls, annual)
    audit['plan_availability'] = evidence
    audit['plan_counts_full_bounded_calendar'] = {key: int(plans[key].sum()) for key in ['cpi_plan', 'nfp_plan', 'fomc_plan']}
    audit['unknown_rows'] = int(plans.isna().any(axis=1).sum())
    return daily, pd.concat(iv.values(), axis=1).sort_index(), plans, audit


def evaluate(panel, calendar, p):
    if set(panel.horizon) != {1} or panel.duplicated(['origin', 'model', 'horizon']).any():
        raise ValueError('Fixed horizon and unique forecast keys required')
    rows = []
    for candidate, control in CONTRASTS:
        phases = []
        for phase_code, name in enumerate(['development', 'evaluation']):
            start, end = p['index'][name]
            frame = panel.loc[(panel.origin >= start) & (panel.origin <= end)]
            if name == 'development':
                frame = frame.loc[frame.available_date <= p['index']['development_target_available_by']]
            wide = frame.pivot(index='origin', columns='model', values='prediction').sort_index()
            if set(wide.columns) != set(p['index']['models']) or not np.isfinite(wide).all().all():
                raise ValueError('Complete common model sample required')
            baseline = frame.loc[frame.model == control].set_index('origin').reindex(wide.index)
            for model in p['index']['models']:
                other = frame.loc[frame.model == model].set_index('origin').reindex(wide.index)
                for column in ['y', 'target_end', 'available_date', 'adjustment_event', 'feature_cutoff_date', 'fit_origin']:
                    if not baseline[column].equals(other[column]):
                        raise ValueError(f'Paired label or timing mismatch: {column}')
            candidate_loss = proper_score(baseline.y, wide[candidate])
            control_loss = proper_score(baseline.y, wide[control])
            phase = paired_inference(candidate_loss, control_loss, p, p['inference']['seed']+phase_code*10000)
            phase.update({'name': name, 'first_origin': str(wide.index[0].date()), 'last_origin': str(wide.index[-1].date())})
            slices = p['index']['evaluation_stability'] if name == 'evaluation' else []
            phase.update(diagnostics(wide.index, candidate_loss-control_loss, calendar, 1, slices))
            phase['action_sensitivity'] = sensitivity(candidate_loss, control_loss, baseline.adjustment_event.to_numpy())
            phases.append(phase)
        rows.append({'study': 'macro_overnight', 'horizon': 1, 'candidate': candidate, 'control': control,
                     'phases': phases, 'p_conservative': max(item['p_conservative'] for item in phases)})
    prior = inherited(p)
    wave, total = adjust(prior, rows)
    for row, wp, tp in zip(rows, wave, total, strict=True):
        row.update({'p_holm_wave': float(wp), 'p_holm_cumulative': float(tp)})
        row['verdict'] = 'EXPLORATORY_LEAD' if passes(row) else 'DOES_NOT_QUALIFY'
    leads = [{'candidate': candidate, 'horizon': 1} for candidate in ['cpi', 'nfp']
             if all(passes(row) for row in rows if row['candidate'] == candidate)]
    timing = all(passes(row) for row in rows if row['candidate'] == 'fomc')
    return {'rows': rows, 'inherited_rows': prior, 'hypothesis_count': 6, 'cumulative_hypothesis_count': 84,
            'leads': leads, 'fomc_timing_control_passes': timing, 'protocol_sha256': inference.digest(PROTOCOL),
            'evidence_class': p['evidence_class']}


def report(metrics):
    lines = ['# Original announcement plans and overnight second moments', '',
             'Absolute paired proper-score differences; negative means improvement. No percentage-QLIKE gains.', '',
             '| Addition vs control | Development gap | Evaluation gap | Wave Holm p | Cumulative Holm p | Gate |',
             '|---|---:|---:|---:|---:|---|']
    for row in metrics['rows']:
        dev, ev = row['phases']
        lines.append(f"| {row['candidate']} vs {row['control']} | {dev['delta']:+.6f} | {ev['delta']:+.6f} | {row['p_holm_wave']:.6f} | {row['p_holm_cumulative']:.6f} | {row['verdict']} |")
    lines += ['', 'Substantive passing exploratory leads: '+json.dumps(metrics['leads']),
              'FOMC date-only timing control passes both comparisons: '+str(metrics['fomc_timing_control_passes']), '',
              'All uncertainty, fixed-period and yearly checks, and adjustment sensitivities are in metrics.json.',
              'The calendar uses original plans and nominal civil weekday windows, not realized announcement dates or a certified exchange holding interval.',
              'The target is a conditional second moment of an adjusted log-return proxy. Historical reuse and source-vintage limitations remain exploratory.', '']
    (REPORT/'results.md').write_text('\n'.join(lines))


def ledger(rows):
    with (REPORT/'trial_ledger.jsonl').open('a') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')


def run():
    p = yaml.safe_load(PROTOCOL.read_text())
    validate(p)
    REPORT.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    if (REPORT/'manifest.json').exists():
        raise ValueError('Refusing to overwrite a registered macro experiment')
    load_plan_records(p)
    test_names = ['tests.test_macro_second_moment', 'tests.test_macro_plan_features', 'tests.test_macro_overnight',
                  'tests.test_macro_search', 'tests.test_verify_macro_overnight', 'tests.test_round2_inference']
    checked = subprocess.run([sys.executable, '-m', 'unittest', *test_names, '-v'], cwd=ROOT, capture_output=True, text=True, check=False)
    (REPORT/'pre_run_checks.txt').write_text(checked.stdout+checked.stderr)
    if checked.returncode:
        raise RuntimeError(checked.stdout+checked.stderr)
    names = ['src/macro_second_moment.py', 'src/macro_plan_features.py', 'src/macro_overnight.py',
             'src/macro_search.py', 'src/verify_macro_overnight.py',
             *[name.replace('.', '/')+'.py' for name in test_names],
             'src/overnight_index.py', 'src/orthogonal_round2.py', 'src/international_search.py',
             'src/iterative_signal_search.py', 'src/verify_model_memory_study.py',
             'src/verify_iterative_signal_search.py', 'src/verify_international_volatility.py']
    prior_files = [file for file in list(ROOT.glob('*.yaml'))+list((ROOT/'reports').rglob('*'))
                   if file.is_file() and file != PROTOCOL and REPORT not in file.parents]
    inputs = set(p['sources'].values())
    inputs.update(str(file.relative_to(ROOT)) for file in (ROOT/'data/source_discovery/macro_plans').rglob('*') if file.is_file())
    inputs.update(p['sources'][name]+'.manifest.json' for name in ['vxn', 'vix'])
    manifest = {'created_utc': datetime.now(UTC).isoformat(), 'protocol_sha256': inference.digest(PROTOCOL),
                'environment': {'python': sys.version, 'packages': {name: version(name) for name in ['numpy', 'pandas', 'scipy', 'pyarrow', 'PyYAML']}},
                'code': {name: inference.digest(ROOT/name) for name in names},
                'inputs': {name: inference.digest(ROOT/name) for name in sorted(inputs)},
                'preserved': {str(file.relative_to(ROOT)): inference.digest(file) for file in prior_files}}
    inference.dump(REPORT/'manifest.json', manifest)
    ledger([{'event': 'inherited', **row} for row in inherited(p)])
    ledger([{'event': 'registered', 'study': 'macro_overnight', 'horizon': 1, 'candidate': candidate, 'control': control,
             'protocol_sha256': manifest['protocol_sha256']} for candidate, control in CONTRASTS])
    metrics = None
    try:
        daily, iv, plans, source_audit = load_inputs(p)
        features, targets = macro_overnight.build_features(daily, iv, plans)
        features.to_parquet(OUT/'features.parquet')
        inference.dump(OUT/'source_audit.json', source_audit)
        panel, fits = macro_overnight.forecast_panel(features, targets, p['index'])
        panel.to_parquet(OUT/'forecasts.parquet')
        inference.dump(OUT/'fits.json', fits)
        metrics = evaluate(panel, features.index, p)
        for section in ['code', 'inputs', 'preserved']:
            for name, expected in manifest[section].items():
                if inference.digest(ROOT/name) != expected:
                    raise ValueError(f'Frozen artifact changed during run: {name}')
        inference.dump(REPORT/'metrics.json', metrics)
        report(metrics)
        ledger([{'event': 'evaluated', **row} for row in metrics['rows']])
    except Exception as error:
        failed = failure_rows(error)
        failure = {'status': 'UNEVALUABLE', 'rows': failed, 'whole_wave_aborted': True,
                   'hypothesis_count': 6, 'cumulative_hypothesis_count': 84,
                   'leads': [], 'fomc_timing_control_passes': False,
                   'protocol_sha256': manifest['protocol_sha256'], 'evidence_class': p.get('evidence_class')}
        # Inherited-family readers consume metrics.json. Invalidate it before
        # attempting any secondary output that could itself fail.
        inference.dump(REPORT/'metrics.json', failure)
        inference.dump(REPORT/'failure.json', failure)
        (REPORT/'results.md').write_text(
            '# Original announcement plans and overnight second moments\n\n'
            'UNEVALUABLE: the complete wave was aborted. All six comparisons retain p-values of one; '
            'no candidate qualifies. See failure.json for the recorded error.\n'
        )
        if metrics is not None:
            inference.dump(REPORT/'unpublished_scored_metrics.json', {
                'status': 'UNPUBLISHED_DIAGNOSTIC_ONLY', 'whole_wave_aborted': True,
                'not_for_inherited_inference_or_promotion': True, 'scored_metrics': metrics,
            })
        ledger([{'event': 'unevaluable', **row} for row in failed])
        raise
    print(json.dumps({'status': 'SCORED_AWAITING_INDEPENDENT_VERIFICATION', 'forecasts': len(panel),
                      'fits': len(fits), 'leads': metrics['leads']}), flush=True)


if __name__ == '__main__':
    run()
