"""Frozen second-wave inference for international volatility information."""
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
from .iterative_signal_search import loss, paired_inference, phase_frame

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'international_volatility.yaml'
REPORT = ROOT / 'reports/international_volatility'
OUT = ROOT / 'data/international_volatility'
CONTRASTS = [(h, c) for h in [1, 5, 21] for c in ['regional', 'latent']]


def validate(p):
    if (p['wave'] != 2 or not np.isclose(p['wave_alpha'], .05 / 6, atol=0, rtol=1e-14)
            or p['comparisons']['new_hypotheses'] != 6 or p['comparisons']['cumulative_hypotheses'] != 72
            or p['hf']['pca_components'] != 3 or p['international']['max_extra_us_sessions'] != 3
            or p['international']['asia'] != ['.N225', '.HSI', '.KS11']
            or p['international']['europe'] != ['.FTSE', '.GDAXI', '.FCHI']
            or p['hf']['horizons'] != [1, 5, 21] or p['hf']['models'] != ['baseline', 'regional', 'latent']
            or p['hf']['latest_target'] != '2017-12-29' or p['hf']['effect_threshold_pct'] != 1.
            or p['inference']['blocks'] != [21, 63, 126] or p['inference']['bootstrap_draws'] != 19999
            or p['inference']['seed'] != 20260908):
        raise ValueError('Frozen second-wave specification changed')


def inherited(p):
    records = []
    for path in p['comparisons']['inherited_sources']:
        for row in json.loads((ROOT / path).read_text())['rows']:
            records.append({'study': row.get('study', path.split('/')[1]),
                            'horizon': row['horizon'], 'candidate': row['candidate'],
                            'control': row.get('control', 'baseline'), 'p_conservative': row['p_conservative'],
                            'source': path, 'source_sha256': inference.digest(ROOT / path)})
    if len(records) != 66:
        raise ValueError('All 66 earlier contrasts must be retained')
    return records


def adjust(prior, rows):
    if len(prior) != 66 or len(rows) != 6:
        raise ValueError('Incomplete cumulative comparison family')
    wave = inference.holm_adjust([r['p_conservative'] for r in rows])
    cumulative = inference.holm_adjust([r['p_conservative'] for r in prior + rows])[-6:]
    return wave, cumulative


def passes(row):
    phases = row['phases']
    stability = next((r.get('stability', []) for r in phases if r['name'] == 'evaluation'), [])
    return (len(phases) == 2 and row['p_holm_wave'] < .05 / 6 and row['p_holm_cumulative'] < .05
            and all(r['improvement_pct'] >= 1. for r in phases)
            and len(stability) == 2 and all(r['delta'] < 0 for r in stability))


def failure_rows(error):
    status = 'INSUFFICIENT_DATA' if 'INSUFFICIENT_DATA' in str(error) else 'INVALID_RUN'
    return [{'study': 'international', 'horizon': h, 'candidate': c, 'control': 'baseline',
             'status': status, 'error': str(error), 'p_conservative': 1., 'phases': []} for h, c in CONTRASTS]


def diagnostics(origins, difference, calendar, horizon, slices):
    origins = pd.DatetimeIndex(origins)
    position = pd.DatetimeIndex(calendar).get_indexer(origins)
    if (position < 0).any():
        raise ValueError('Scored origins must belong to the original market calendar')
    result = {'annual': [], 'stability': [], 'nonoverlap_phases': []}
    for year in sorted(set(origins.year)):
        mask = origins.year == year
        result['annual'].append({'year': int(year), 'n': int(mask.sum()), 'delta': float(difference[mask].mean())})
    for start, end in slices:
        mask = (origins >= start) & (origins <= end)
        if not mask.any():
            raise ValueError('INSUFFICIENT_DATA: missing stability slice')
        result['stability'].append({'start': start, 'end': end, 'n': int(mask.sum()), 'delta': float(difference[mask].mean())})
    for phase in range(horizon):
        mask = position % horizon == phase
        result['nonoverlap_phases'].append({'phase': phase, 'n': int(mask.sum()),
                                            'delta': float(difference[mask].mean()) if mask.any() else None})
    return result


def validate_panel(panel, p):
    if (set(panel.horizon) != set(p['hf']['horizons'])
            or panel.duplicated(['origin', 'horizon', 'model']).any()):
        raise ValueError('Missing horizon or duplicate forecast key')
    for phase in ['development', 'evaluation']:
        selected = phase_frame(panel, p['hf'], phase)
        expected = None
        for horizon in p['hf']['horizons']:
            for model in p['hf']['models']:
                origins = pd.DatetimeIndex(selected.loc[(selected.horizon == horizon) & (selected.model == model), 'origin']).sort_values()
                if expected is None:
                    expected = origins
                elif not expected.equals(origins):
                    raise ValueError('All models and horizons require identical phase origins')


def evaluate(panel, calendar, p):
    validate_panel(panel, p)
    rows = []
    for horizon, candidate in CONTRASTS:
        phases = []
        section = p['hf']
        for phase_code, name in enumerate(['development', 'evaluation']):
            frame = phase_frame(panel.loc[panel.horizon == horizon], section, name)
            wide = frame.pivot(index='origin', columns='model', values='prediction').sort_index()
            if len(wide) < max(p['inference']['blocks']):
                raise ValueError('INSUFFICIENT_DATA: phase shorter than maximum inference block')
            if set(wide.columns) != set(section['models']) or not np.isfinite(wide).all().all():
                raise ValueError('Invalid common forecast sample')
            base = frame.loc[frame.model == 'baseline'].set_index('origin').reindex(wide.index)
            for model in section['models']:
                other = frame.loc[frame.model == model].set_index('origin').reindex(wide.index)
                if (not np.array_equal(base.y, other.y) or not base.target_end.equals(other.target_end)
                        or not base.available_date.equals(other.available_date)):
                    raise ValueError('Paired actual labels or dates differ')
            candidate_loss = loss(base.y, wide[candidate], 'hf')
            control_loss = loss(base.y, wide.baseline, 'hf')
            seed = p['inference']['seed'] + horizon * 1000 + phase_code * 10000
            result = paired_inference(candidate_loss, control_loss, p, seed)
            result.update({'name': name, 'first_origin': str(wide.index[0].date()), 'last_origin': str(wide.index[-1].date())})
            slices = section['evaluation_stability'] if name == 'evaluation' else []
            result.update(diagnostics(wide.index, candidate_loss - control_loss, calendar, horizon, slices))
            phases.append(result)
        rows.append({'study': 'international', 'horizon': horizon, 'candidate': candidate, 'control': 'baseline',
                     'phases': phases, 'p_conservative': max(r['p_conservative'] for r in phases)})
    prior = inherited(p)
    wave, cumulative = adjust(prior, rows)
    for row, wp, cp in zip(rows, wave, cumulative, strict=True):
        row.update({'p_holm_wave': float(wp), 'p_holm_cumulative': float(cp)})
        row['verdict'] = 'EXPLORATORY_LEAD' if passes(row) else 'DOES_NOT_QUALIFY'
    return {'rows': rows, 'inherited_rows': prior, 'hypothesis_count': 6, 'cumulative_hypothesis_count': 72,
            'protocol_sha256': inference.digest(PROTOCOL), 'evidence_class': p['evidence_class'],
            'leads': [{'candidate': r['candidate'], 'horizon': r['horizon']} for r in rows if passes(r)]}


def report(metrics):
    lines = ['# International volatility: second fixed wave', '',
             'Six new comparisons of regional summaries and training-only latent factors against the strong SPX baseline.',
             'Positive gain means lower QLIKE loss. All 72 enumerated past and current contrasts remain in the cumulative family.', '',
             '| Candidate | Horizon | Development gain | Evaluation gain | Wave Holm p | Cumulative Holm p | Gate |',
             '|---|---:|---:|---:|---:|---:|---|']
    for row in metrics['rows']:
        dev, ev = row['phases']
        lines.append(f"| {row['candidate']} | {row['horizon']} | {dev['improvement_pct']:+.3f}% | {ev['improvement_pct']:+.3f}% | {row['p_holm_wave']:.5f} | {row['p_holm_cumulative']:.5f} | {row['verdict']} |")
    lines += ['', 'Passing exploratory leads: ' + json.dumps(metrics['leads']), '',
              'Foreign features use each market\'s archived observation calendar, then a backward date join with an explicit freshness bound.',
              'The prior-US-session cutoff limits timing leakage; it does not establish historical publication vintages or remove revisions.',
              'The six comparisons test added international information. They do not test whether latent factors beat regional summaries.',
              'All estimates, nominal uncertainty, MDE, annual slices and nonoverlapping phases are in metrics.json.',
              'Historical reuse remains exploratory. No revised threshold or post-score model selection is permitted.', '']
    (REPORT / 'results.md').write_text('\n'.join(lines))


def ledger(records):
    with (REPORT / 'trial_ledger.jsonl').open('a') as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')


def run():
    p = yaml.safe_load(PROTOCOL.read_text())
    validate(p)
    REPORT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = REPORT / 'manifest.json'
    if path.exists():
        raise ValueError('Registered wave already exists; preserve its artifacts')
    modules = ['tests.test_international_search', 'tests.test_international_volatility',
               'tests.test_verify_international_volatility', 'tests.test_round2_inference']
    result = subprocess.run([sys.executable, '-m', 'unittest', *modules, '-q'], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    (REPORT / 'pre_run_checks.txt').write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    names = ['src/international_search.py', 'src/international_volatility.py', 'src/verify_international_volatility.py',
             'tests/test_international_search.py', 'tests/test_international_volatility.py', 'tests/test_verify_international_volatility.py',
             'src/iterative_hf.py', 'src/iterative_signal_search.py', 'src/verify_iterative_signal_search.py',
             'src/orthogonal_round2.py', 'src/verify_model_memory_study.py']
    old = [f for f in list(ROOT.glob('*.yaml')) + list((ROOT / 'reports').rglob('*'))
           if f.is_file() and 'international_volatility' not in str(f)]
    inputs = [p['sources'][key] for key in ['archive', 'spx_existing', 'spx_extension', 'spx_extension_metadata', 'cboe']]
    manifest = {'created_utc': datetime.now(UTC).isoformat(), 'protocol_sha256': inference.digest(PROTOCOL),
                'environment': {'python': sys.version, 'packages': {name: version(name) for name in ['numpy', 'pandas', 'scipy', 'pyarrow', 'PyYAML']}},
                'code': {f: inference.digest(ROOT / f) for f in names},
                'inputs': {f: inference.digest(ROOT / f) for f in inputs},
                'preserved': {str(f.relative_to(ROOT)): inference.digest(f) for f in old}}
    inference.dump(path, manifest)
    ledger([{'event': 'inherited', **r} for r in inherited(p)])
    ledger([{'event': 'registered', 'study': 'international', 'horizon': h, 'candidate': c,
             'control': 'baseline', 'protocol_sha256': manifest['protocol_sha256']} for h, c in CONTRASTS])
    try:
        from . import international_volatility
        international_volatility.run_international(p, root=ROOT)
        panel = pd.read_parquet(OUT / 'forecasts.parquet')
        calendar = pd.read_parquet(OUT / 'features.parquet').index
        metrics = evaluate(panel, calendar, p)
        for name, expected in manifest['preserved'].items():
            if inference.digest(ROOT / name) != expected:
                raise ValueError(f'Prior frozen artifact changed: {name}')
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
