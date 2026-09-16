"""Fixed overnight-index wave and complete cumulative inference ledger."""
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

from . import orthogonal_round2 as inference
from .international_search import diagnostics
from .iterative_signal_search import loss, paired_inference, phase_frame

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'overnight_index.yaml'
REPORT = ROOT / 'reports/overnight_index'
OUT = ROOT / 'data/overnight_index'
CONTRASTS = [(candidate, control) for candidate in ['cross_signed', 'volume_pressure', 'iv_shape'] for control in ['baseline', 'mean']]


def validate(p):
    if (p['wave'] != 3 or not np.isclose(p['wave_alpha'], .05 / 12, rtol=1e-14, atol=0)
            or p['comparisons']['new_hypotheses'] != 6 or p['comparisons']['cumulative_hypotheses'] != 78
            or p['index']['horizons'] != [1] or p['index']['effect_threshold_pct'] != .25
            or p['index']['ridge_alpha'] != .01 or p['index']['minimum_train'] != 1000
            or p['index']['latest_target'] != '2025-10-20' or p['index']['sealed_start'] != '2025-11-03'
            or p['measurement']['event_threshold'] != 1e-5
            or p['inference']['blocks'] != [21, 63, 126] or p['inference']['bootstrap_draws'] != 19999
            or p['inference']['seed'] != 20260909):
        raise ValueError('Frozen overnight specification changed')


def inherited(p):
    rows = []
    for path in p['comparisons']['inherited_sources']:
        for row in json.loads((ROOT / path).read_text())['rows']:
            rows.append({'study': row.get('study', path.split('/')[1]), 'candidate': row['candidate'],
                         'horizon': row['horizon'], 'control': row.get('control', 'baseline'),
                         'p_conservative': row['p_conservative'], 'source': path,
                         'source_sha256': inference.digest(ROOT / path)})
    if len(rows) != 72:
        raise ValueError('All 72 earlier contrasts must remain')
    return rows


def adjust(prior, rows):
    if len(prior) != 72 or len(rows) != 6:
        raise ValueError('Incomplete 78-comparison family')
    return (inference.holm_adjust([r['p_conservative'] for r in rows]),
            inference.holm_adjust([r['p_conservative'] for r in prior + rows])[-6:])


def passes(row):
    phases = row['phases']
    stability = next((r.get('stability', []) for r in phases if r['name'] == 'evaluation'), [])
    return (len(phases) == 2 and row['p_holm_wave'] < .05 / 12 and row['p_holm_cumulative'] < .05
            and all(r['improvement_pct'] >= .25 and r['action_sensitivity']['n'] > 0
                    and r['action_sensitivity']['delta'] < 0 for r in phases)
            and len(stability) == 2 and all(r['delta'] < 0 for r in stability))


def action_sensitivity(candidate_loss, control_loss, flagged):
    candidate_loss, control_loss, flagged = np.asarray(candidate_loss), np.asarray(control_loss), np.asarray(flagged)
    if (candidate_loss.shape != control_loss.shape or flagged.shape != candidate_loss.shape
            or candidate_loss.ndim != 1 or flagged.dtype != bool
            or not np.isfinite([candidate_loss, control_loss]).all()):
        raise ValueError('Finite aligned losses and boolean adjustment flag required')
    keep = ~flagged
    if not keep.any():
        raise ValueError('INSUFFICIENT_DATA: no unflagged measurement observations')
    candidate, control = candidate_loss[keep].mean(), control_loss[keep].mean()
    if control <= 0:
        raise ValueError('INSUFFICIENT_DATA: zero sensitivity control loss')
    delta = float(candidate - control)
    return {'n': int(keep.sum()), 'delta': delta, 'control_loss': float(control),
            'candidate_loss': float(candidate), 'improvement_pct': float(-100 * delta / control)}


def failure_rows(error):
    status = 'INSUFFICIENT_DATA' if 'INSUFFICIENT_DATA' in str(error) else 'INVALID_RUN'
    return [{'study': 'overnight', 'horizon': 1, 'candidate': candidate, 'control': control,
             'status': status, 'error': str(error), 'p_conservative': 1., 'phases': []} for candidate, control in CONTRASTS]


def evaluate(panel, calendar, p):
    if set(panel.horizon) != {1} or panel.duplicated(['origin', 'model', 'horizon']).any():
        raise ValueError('Fixed horizon and unique forecast keys required')
    rows = []
    for candidate, control in CONTRASTS:
        phases = []
        for phase_code, name in enumerate(['development', 'evaluation']):
            frame = phase_frame(panel, p['index'], name)
            wide = frame.pivot(index='origin', columns='model', values='prediction').sort_index()
            if len(wide) < max(p['inference']['blocks']):
                raise ValueError('INSUFFICIENT_DATA: phase shorter than inference block')
            if set(wide.columns) != set(p['index']['models']) or not np.isfinite(wide).all().all():
                raise ValueError('Common complete model sample required')
            base = frame.loc[frame.model == control].set_index('origin').reindex(wide.index)
            for model in p['index']['models']:
                other = frame.loc[frame.model == model].set_index('origin').reindex(wide.index)
                for column in ['y', 'target_end', 'available_date', 'adjustment_event', 'feature_cutoff_date']:
                    if not base[column].equals(other[column]):
                        raise ValueError(f'Paired label or information-date mismatch: {column}')
            candidate_loss = loss(base.y, wide[candidate], 'index')
            control_loss = loss(base.y, wide[control], 'index')
            seed = p['inference']['seed'] + phase_code * 10000
            result = paired_inference(candidate_loss, control_loss, p, seed)
            result.update({'name': name, 'first_origin': str(wide.index[0].date()), 'last_origin': str(wide.index[-1].date())})
            slices = p['index']['evaluation_stability'] if name == 'evaluation' else []
            result.update(diagnostics(wide.index, candidate_loss - control_loss, calendar, 1, slices))
            result['action_sensitivity'] = action_sensitivity(candidate_loss, control_loss, base.adjustment_event.to_numpy())
            phases.append(result)
        rows.append({'study': 'overnight', 'horizon': 1, 'candidate': candidate, 'control': control,
                     'phases': phases, 'p_conservative': max(r['p_conservative'] for r in phases)})
    prior = inherited(p)
    wave, total = adjust(prior, rows)
    for row, wp, tp in zip(rows, wave, total, strict=True):
        row.update({'p_holm_wave': float(wp), 'p_holm_cumulative': float(tp)})
        row['verdict'] = 'EXPLORATORY_LEAD' if passes(row) else 'DOES_NOT_QUALIFY'
    leads = [{'candidate': candidate, 'horizon': 1} for candidate in p['index']['models'][2:]
             if all(passes(r) for r in rows if r['candidate'] == candidate)]
    return {'rows': rows, 'inherited_rows': prior, 'hypothesis_count': 6, 'cumulative_hypothesis_count': 78,
            'leads': leads, 'protocol_sha256': inference.digest(PROTOCOL), 'evidence_class': p['evidence_class']}


def report(metrics):
    lines = ['# Overnight index returns: third fixed wave', '',
             'The target is a vendor-adjusted overnight-return proxy. All market inputs precede the entry session.',
             'Six new comparisons remain in the cumulative family of 78; each candidate must pass both controls.', '',
             '| Candidate vs control | Development gain | Evaluation gain | Wave Holm p | Cumulative Holm p | Gate |',
             '|---|---:|---:|---:|---:|---|']
    for row in metrics['rows']:
        dev, ev = row['phases']
        lines.append(f"| {row['candidate']} vs {row['control']} | {dev['improvement_pct']:+.3f}% | {ev['improvement_pct']:+.3f}% | {row['p_holm_wave']:.5f} | {row['p_holm_cumulative']:.5f} | {row['verdict']} |")
    lines += ['', 'Passing exploratory leads: ' + json.dumps(metrics['leads']), '',
              '## Measurement sensitivity', '',
              'The following excludes flagged adjustment-factor changes after forecasts are fixed. It is a retrospective label audit, not a tradable filter.', '',
              '| Candidate vs control | Development unflagged gain | Evaluation unflagged gain |',
              '|---|---:|---:|']
    for row in metrics['rows']:
        dev, ev = [phase['action_sensitivity'] for phase in row['phases']]
        lines.append(f"| {row['candidate']} vs {row['control']} | {dev['improvement_pct']:+.3f}% (n={dev['n']}) | {ev['improvement_pct']:+.3f}% (n={ev['n']}) |")
    lines += ['', 'All numerical intervals, MDE, years, stability slices and adjustment sensitivities are in metrics.json.',
              'The adjusted target does not establish cash profit, corporate-action receivable accounting, or execution at the auction prices.',
              'Historical reuse and retrospectively acquired sources remain exploratory; no frozen criterion is changed after scores.', '']
    (REPORT / 'results.md').write_text('\n'.join(lines))


def ledger(records):
    with (REPORT / 'trial_ledger.jsonl').open('a') as stream:
        for row in records:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')


def run():
    p = yaml.safe_load(PROTOCOL.read_text())
    validate(p)
    REPORT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = REPORT / 'manifest.json'
    if path.exists():
        raise ValueError('A registered overnight wave already exists; preserve its outputs')
    modules = ['tests.test_overnight_search', 'tests.test_overnight_index', 'tests.test_verify_overnight_index', 'tests.test_round2_inference']
    checked = subprocess.run([sys.executable, '-m', 'unittest', *modules, '-q'], cwd=ROOT,
                             capture_output=True, text=True, check=False)
    (REPORT / 'pre_run_checks.txt').write_text(checked.stdout + checked.stderr)
    if checked.returncode:
        raise RuntimeError(checked.stdout + checked.stderr)
    names = ['src/overnight_search.py', 'src/overnight_index.py', 'src/verify_overnight_index.py',
             'tests/test_overnight_search.py', 'tests/test_overnight_index.py', 'tests/test_verify_overnight_index.py',
             'src/international_search.py', 'src/iterative_signal_search.py', 'src/orthogonal_round2.py',
             'src/verify_iterative_signal_search.py', 'src/verify_model_memory_study.py',
             'src/verify_international_volatility.py', 'src/fetch.py']
    old = [f for f in list(ROOT.glob('*.yaml')) + list((ROOT / 'reports').rglob('*'))
           if f.is_file() and 'overnight_index' not in str(f)]
    inputs = list(p['sources'].values()) + [f + '.manifest.json' for f in p['sources'].values() if f.endswith('.csv')]
    manifest = {'created_utc': datetime.now(UTC).isoformat(), 'protocol_sha256': inference.digest(PROTOCOL),
                'environment': {'python': sys.version, 'packages': {name: version(name) for name in ['numpy', 'pandas', 'scipy', 'pyarrow', 'PyYAML']}},
                'code': {f: inference.digest(ROOT / f) for f in names},
                'inputs': {f: inference.digest(ROOT / f) for f in inputs},
                'preserved': {str(f.relative_to(ROOT)): inference.digest(f) for f in old}}
    inference.dump(path, manifest)
    ledger([{'event': 'inherited', **r} for r in inherited(p)])
    ledger([{'event': 'registered', 'study': 'overnight', 'horizon': 1, 'candidate': c, 'control': b,
             'protocol_sha256': manifest['protocol_sha256']} for c, b in CONTRASTS])
    try:
        from . import overnight_index
        overnight_index.run_overnight(p, root=ROOT)
        panel = pd.read_parquet(OUT / 'forecasts.parquet')
        calendar = pd.read_parquet(OUT / 'features.parquet').index
        metrics = evaluate(panel, calendar, p)
        for name, expected in manifest['preserved'].items():
            if inference.digest(ROOT / name) != expected:
                raise ValueError(f'Earlier frozen artifact changed: {name}')
        inference.dump(REPORT / 'metrics.json', metrics)
        report(metrics)
        ledger([{'event': 'evaluated', **r} for r in metrics['rows']])
    except Exception as error:
        failed = failure_rows(error)
        inference.dump(REPORT / 'failure.json', {'rows': failed, 'whole_wave_aborted': True})
        ledger([{'event': 'unevaluable', **r} for r in failed])
        raise
    print(json.dumps({'status': 'SCORED_AWAITING_INDEPENDENT_VERIFICATION', 'leads': metrics['leads']}), flush=True)


if __name__ == '__main__':
    run()
