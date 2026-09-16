"""Registered medium-horizon SPX return experiment with complete trial accounting."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import index_hinge as ih
from . import orthogonal_round2 as inference
from .international_search import diagnostics

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'index_hinge.yaml'
REPORT = ROOT/'reports/index_hinge'
OUT = ROOT/'data/index_hinge'
CONTRASTS = tuple(('hinge', control, h) for h in (21, 63) for control in ('baseline', 'mean'))
WAVE_ALPHA = 1/840


def validate(p):
    i, c, s = p['index'], p['comparisons'], p['inference']
    if (p['wave'] != 6 or p['wave_alpha'] != WAVE_ALPHA or c['new_hypotheses'] != 4
            or c['inherited_hypotheses'] != 99 or c['cumulative_hypotheses'] != 103
            or c['candidates'] != ['hinge'] or c['controls'] != ['baseline', 'mean']
            or i['horizons'] != [21, 63] or tuple(i['models']) != ih.MODELS
            or tuple(i['raw']) != ih.RAW or tuple(i['baseline']) != ih.BASE
            or i['minimum_train'] != 1000 or i['penalty'] != .01 or i['market_lag'] != 1
            or i['effect_threshold_relative'] != .0025 or i['source_end'] != '2025-10-20'
            or i['latest_target'] != '2025-10-20' or i['sealed_start'] != '2025-11-03'
            or i['origin_start'] != '2016-01-04' or i['origin_end'] != '2025-10-17'
            or i['development'] != ['2016-01-04', '2019-12-31']
            or i['evaluation'] != ['2020-01-02', '2025-10-17']
            or i['development_target_available_by'] != '2019-12-31'
            or i['evaluation_stability'] != [['2020-01-02', '2022-12-31'], ['2023-01-01', '2025-10-17']]
            or s['blocks'] != [126, 252, 504] or s['hac_lags'] != 504
            or s['minimum_phase_observations'] != 505
            or s['bootstrap_draws'] != 99999 or s['seed'] != 20260912):
        raise ValueError('Fixed index-hinge specification differs')


def _alignment(features, targets, horizon):
    dates = features.index
    if (horizon not in (21, 63) or not isinstance(dates, pd.DatetimeIndex) or dates.has_duplicates
            or dates.hasnans or dates.tz is not None or not dates.is_monotonic_increasing
            or not dates.equals(dates.normalize()) or not dates.equals(targets.index)):
        raise ValueError('Unique aligned normalized SPX calendar and fixed horizon required')
    for frame, name, shift in ((features, 'feature_cutoff_date', 1), (targets, 'target_end', -horizon),
                               (targets, 'available_date', -horizon)):
        expected = pd.Series(dates, index=dates, name=name).shift(shift)
        if not frame[name].equals(expected):
            raise ValueError('Fixed observed-session feature or target availability differs')


def training_mask(features, targets, fit_entry, horizon, min_train=1000):
    _alignment(features, targets, horizon)
    fit_entry = pd.Timestamp(fit_entry)
    if fit_entry not in features.index or min_train < 2:
        raise ValueError('Invalid fit entry or training minimum')
    cutoff = features.loc[fit_entry, 'feature_cutoff_date']
    valid = np.isfinite(features.loc[:, ih.RAW]).all(axis=1) & np.isfinite(targets.y)
    mask = valid & (features.index < fit_entry) & (targets.available_date <= cutoff)
    if int(mask.sum()) < min_train:
        raise ValueError(f'INSUFFICIENT_DATA: {int(mask.sum())} complete matured common training rows')
    return mask


def forecast_panel(features, targets_by_horizon, config):
    if (tuple(config['models']) != ih.MODELS or tuple(config['baseline']) != ih.BASE
            or config['horizons'] != [21, 63] or set(targets_by_horizon) != {21, 63}
            or config['penalty'] != .01):
        raise ValueError('Fixed model family differs')
    dates = features.index
    start, end = map(pd.Timestamp, (config['origin_start'], config['origin_end']))
    dev_start, dev_end = map(pd.Timestamp, config['development'])
    ev_start, ev_end = map(pd.Timestamp, config['evaluation'])
    latest = pd.Timestamp(config['latest_target'])
    if (not start <= dev_start <= dev_end < ev_start <= ev_end <= end < latest
            or config['development_target_available_by'] != config['development'][1]):
        raise ValueError('Invalid phase and outcome boundaries')
    complete = np.isfinite(features.loc[:, ih.RAW]).all(axis=1)
    in_phase = ((dates >= dev_start) & (dates <= dev_end)) | ((dates >= ev_start) & (dates <= ev_end))
    entries = dates[complete & in_phase & (dates >= start) & (dates <= end)]
    if not len(entries):
        raise ValueError('INSUFFICIENT_DATA: no complete application entries')
    rows, fits = [], []
    for horizon in (21, 63):
        targets = targets_by_horizon[horizon]
        _alignment(features, targets, horizon)
        ready = (np.isfinite(targets.y) & (targets.available_date <= latest)
                 & ((dates > dev_end) | (targets.available_date <= dev_end)))
        for month in entries.to_period('M').unique():
            application = entries[entries.to_period('M') == month]
            fit_entry = application[0]
            mask = training_mask(features, targets, fit_entry, horizon, config['minimum_train'])
            fit = ih.fit_predict(features.loc[mask], targets.loc[mask, 'y'], features.loc[application])
            cutoff = features.loc[fit_entry, 'feature_cutoff_date']
            last_target = targets.loc[mask, 'target_end'].max()
            fits.append({'horizon': horizon, 'fit_origin': str(fit_entry.date()),
                         'fit_cutoff_date': str(cutoff.date()), 'train_n': int(mask.sum()),
                         'train_first_origin': str(features.index[mask][0].date()),
                         'train_last_origin': str(features.index[mask][-1].date()),
                         'train_last_target': str(last_target.date()), 'train_last_available': str(last_target.date()),
                         'application_n': len(application), 'model_audit': fit['model_audit'],
                         'transform_audit': fit['transform_audit']})
            selected = ready.loc[application].to_numpy()
            scored = application[selected]
            for model in ih.MODELS:
                row = pd.DataFrame({'origin': scored, 'model': model, 'horizon': horizon,
                                    'prediction': fit['predictions'][model][selected], 'fit_origin': fit_entry,
                                    'fit_cutoff_date': cutoff, 'train_n': int(mask.sum()),
                                    'train_last_target': last_target, 'train_last_available': last_target,
                                    'phase': np.where(scored <= dev_end, 'development', 'evaluation')})
                for name in ('y', 'target_end', 'available_date'):
                    row[name] = targets.loc[scored, name].to_numpy()
                row['feature_cutoff_date'] = features.loc[scored, 'feature_cutoff_date'].to_numpy()
                rows.append(row)
    panel = pd.concat(rows, ignore_index=True).sort_values(['horizon', 'origin', 'model']).reset_index(drop=True)
    if panel.empty:
        raise ValueError('INSUFFICIENT_DATA: no matured common scoring observations')
    return panel, fits


def paired_inference(candidate, control, protocol, seed):
    candidate, control = np.asarray(candidate, float), np.asarray(control, float)
    if (candidate.ndim != 1 or candidate.shape != control.shape or not np.isfinite([candidate, control]).all()
            or (candidate < 0).any() or (control < 0).any() or not control.mean() > 0):
        raise ValueError('Finite aligned nonnegative losses and positive reference MSE required')
    difference = candidate-control
    if len(difference) <= max(protocol['inference']['hac_lags'], max(protocol['inference']['blocks'])):
        raise ValueError('INSUFFICIENT_DATA: phase shorter than fixed dependence block')
    delta = float(difference.mean())
    blocks = {}
    for block in protocol['inference']['blocks']:
        samples = inference.bootstrap_means(difference, block, protocol['inference']['bootstrap_draws'], seed+block)[:, 0]
        probability = (1+int((np.abs(samples-delta) >= abs(delta)).sum()))/(len(samples)+1)
        blocks[str(block)] = {'p': probability, 'ci95': np.quantile(samples, [.025, .975]).tolist()}
    hac = inference.hac_summary(difference, lags=protocol['inference']['hac_lags'])
    intervals = [hac['ci95']]+[block['ci95'] for block in blocks.values()]
    return {'n': len(difference), 'delta': delta, 'candidate_loss': float(candidate.mean()),
            'control_loss': float(control.mean()), 'gain_relative': float(1-candidate.mean()/control.mean()),
            'block_inference': blocks, 'hac504': hac,
            'ci95_envelope': [min(item[0] for item in intervals), max(item[1] for item in intervals)],
            'p_conservative': max(hac['p'], *(item['p'] for item in blocks.values()))}


def inherited(p):
    rows = []
    for path in p['comparisons']['inherited_sources']:
        for source_row_index, row in enumerate(json.loads((ROOT/path).read_text())['rows']):
            rows.append({'study': row.get('study', Path(path).parent.name or Path(path).stem), 'candidate': row['candidate'],
                         'control': row.get('control', 'baseline'), 'horizon': row['horizon'],
                         'p_conservative': row['p_conservative'], 'source': path, 'source_sha256': inference.digest(ROOT/path),
                         'source_row_index': source_row_index,
                         **({'measure': row['measure']} if 'measure' in row else {})})
    if len(rows) != 99:
        raise ValueError('All99 previously enumerated comparisons must remain')
    return rows


def passes(row):
    phases = row['phases']
    if len(phases) != 2 or {phase['name'] for phase in phases} != {'development', 'evaluation'}:
        return False
    evaluation = next(phase for phase in phases if phase['name'] == 'evaluation')
    return (row['p_holm_wave'] < WAVE_ALPHA and row['p_holm_cumulative'] < .05
            and all(phase['n'] > 0 and phase['gain_relative'] >= .0025 and phase['delta'] < 0 for phase in phases)
            and len(evaluation['stability']) == 2 and all(item['delta'] < 0 for item in evaluation['stability'])
            and all(len(phase['nonoverlap_phases']) == row['horizon']
                    and {item['phase'] for item in phase['nonoverlap_phases']} == set(range(row['horizon']))
                    and all(item['n'] > 0 and item['delta'] is not None and item['delta'] < 0
                            for item in phase['nonoverlap_phases']) for phase in phases))


def candidate_leads(rows):
    keys = [(r['candidate'], r['control'], r['horizon']) for r in rows]
    if len(keys) != 4 or set(keys) != set(CONTRASTS):
        raise ValueError('Complete four-comparison family required')
    return [h for h in (21, 63) if all(passes(row) for row in rows if row['horizon'] == h)]


def evaluate(panel, calendar, p):
    if panel.duplicated(['origin', 'model', 'horizon']).any() or set(panel.horizon) != {21, 63}:
        raise ValueError('Unique fixed-horizon forecasts required')
    rows = []
    for candidate, control, horizon in CONTRASTS:
        phases = []
        for code, name in enumerate(('development', 'evaluation')):
            first, last = p['index'][name]
            frame = panel.loc[(panel.horizon == horizon) & (panel.origin >= first) & (panel.origin <= last)]
            if name == 'development':
                frame = frame.loc[frame.available_date <= p['index']['development_target_available_by']]
            wide = frame.pivot(index='origin', columns='model', values='prediction').sort_index()
            if set(wide.columns) != set(ih.MODELS) or not np.isfinite(wide).all().all():
                raise ValueError('Complete paired forecasts required')
            actual = frame.loc[frame.model == control].set_index('origin').reindex(wide.index)
            for model in ih.MODELS:
                other = frame.loc[frame.model == model].set_index('origin').reindex(wide.index)
                for column in ('y', 'target_end', 'available_date', 'fit_origin', 'feature_cutoff_date',
                               'fit_cutoff_date', 'train_n', 'train_last_available', 'phase'):
                    if not actual[column].equals(other[column]):
                        raise ValueError('Paired labels, features or training metadata differ')
            candidate_loss = (actual.y-wide[candidate])**2
            control_loss = (actual.y-wide[control])**2
            phase = paired_inference(candidate_loss, control_loss, p, p['inference']['seed']+horizon*1000000+code*10000)
            phase.update({'name': name, 'first_origin': str(wide.index[0].date()), 'last_origin': str(wide.index[-1].date())})
            phase.update(diagnostics(wide.index, candidate_loss-control_loss, calendar, horizon,
                                     p['index']['evaluation_stability'] if name == 'evaluation' else []))
            phases.append(phase)
        rows.append({'study': 'index_hinge', 'candidate': candidate, 'control': control, 'horizon': horizon,
                     'phases': phases, 'p_conservative': max(phase['p_conservative'] for phase in phases)})
    prior = inherited(p)
    wave = inference.holm_adjust([row['p_conservative'] for row in rows])
    cumulative = inference.holm_adjust([row['p_conservative'] for row in prior+rows])[-4:]
    for row, wp, cp in zip(rows, wave, cumulative, strict=True):
        row.update(p_holm_wave=float(wp), p_holm_cumulative=float(cp))
        row['verdict'] = 'COMPARISON_GATE_PASS' if passes(row) else 'DOES_NOT_QUALIFY'
    return {'rows': rows, 'inherited_rows': prior, 'leads': candidate_leads(rows), 'hypothesis_count': 4,
            'cumulative_hypothesis_count': 103, 'protocol_sha256': inference.digest(PROTOCOL),
            'evidence_class': p['evidence_class']}


def failure_metrics(error, protocol_hash):
    status = 'INSUFFICIENT_DATA' if 'INSUFFICIENT_DATA' in str(error) else 'INVALID_RUN'
    rows = [{'study': 'index_hinge', 'candidate': candidate, 'control': control, 'horizon': horizon,
             'status': status, 'error': str(error), 'p_conservative': 1., 'phases': []}
            for candidate, control, horizon in CONTRASTS]
    return {'status': 'UNEVALUABLE', 'whole_wave_aborted': True, 'rows': rows, 'leads': [], 'hypothesis_count': 4,
            'cumulative_hypothesis_count': 103, 'protocol_sha256': protocol_hash}


def report(metrics):
    lines = ['# Medium-horizon SPX gap hinge', '', 'Positive relative MSE gain indicates improvement.', '',
             '| Horizon / control | Development gain | Evaluation gain | Wave Holm p | Cumulative Holm p | Gate |',
             '|---|---:|---:|---:|---:|---|']
    for row in metrics['rows']:
        dev, ev = row['phases']
        lines.append(f"| {row['horizon']} / {row['control']} | {dev['gain_relative']:.3%} | {ev['gain_relative']:.3%} | "
                     f"{row['p_holm_wave']:.6f} | {row['p_holm_cumulative']:.6f} | {row['verdict']} |")
    lines += ['', 'Passing horizons: '+json.dumps(metrics['leads']), '',
              'Each horizon requires both controls, both periods, every nonoverlapping offset, and fixed evaluation slices.',
              'Price-index returns, overlapping labels and archival data; exploratory evidence without a trading claim.', '']
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
        raise ValueError('Refusing to overwrite a registered index-hinge experiment')
    modules = ['tests.test_index_hinge', 'tests.test_index_hinge_search', 'tests.test_index_hinge_publication',
               'tests.test_verify_index_hinge', 'tests.test_round2_inference']
    checked = subprocess.run([sys.executable, '-m', 'unittest', *modules, '-v'], cwd=ROOT, capture_output=True, text=True, check=False)
    (REPORT/'pre_run_checks.txt').write_text(checked.stdout+checked.stderr)
    if checked.returncode:
        raise RuntimeError('Prewritten index-hinge tests failed; no registration or fit')
    code = list((ROOT/'src').rglob('*.py'))+list((ROOT/'tests').rglob('*.py'))
    inputs = set(p['sources'].values())
    inputs.update(p['sources'][name]+'.manifest.json' for name in ('vix', 'vix9d', 'vvix'))
    inputs.add('data/research_paths/source_manifest.json')
    preserved = [file for file in list(ROOT.glob('*.yaml'))+list((ROOT/'reports').rglob('*'))
                 if file.is_file() and file != PROTOCOL and REPORT not in file.parents]
    backend = io.StringIO()
    with redirect_stdout(backend):
        np.show_config()
    manifest = {'created_utc': datetime.now(UTC).isoformat(), 'protocol_sha256': inference.digest(PROTOCOL),
                'environment': {'python': sys.version, 'packages': {name: version(name) for name in
                                ('numpy', 'pandas', 'scipy', 'pyarrow', 'PyYAML')}, 'numpy_backend': backend.getvalue(),
                                'thread_environment': {name: os.environ.get(name) for name in
                                                       ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'LOKY_MAX_CPU_COUNT')}},
                'code': {str(file.relative_to(ROOT)): inference.digest(file) for file in sorted(code)},
                'inputs': {name: inference.digest(ROOT/name) for name in sorted(inputs)},
                'preserved': {str(file.relative_to(ROOT)): inference.digest(file) for file in sorted(preserved)}}
    inference.dump(REPORT/'manifest.json', manifest)
    metrics = None
    try:
        ledger([{'event': 'inherited', **row} for row in inherited(p)])
        ledger([{'event': 'registered', 'study': 'index_hinge', 'candidate': candidate, 'control': control,
                 'horizon': horizon, 'protocol_sha256': manifest['protocol_sha256']}
                for candidate, control, horizon in CONTRASTS])
        daily, iv, source_audit = ih.load_sources(p, ROOT)
        features, targets = ih.build_features(daily, iv)
        features.to_parquet(OUT/'features.parquet')
        pd.concat([target.assign(horizon=h) for h, target in targets.items()]).to_parquet(OUT/'targets.parquet')
        inference.dump(OUT/'source_audit.json', source_audit)
        forecasts, fits = forecast_panel(features, targets, p['index'])
        forecasts.to_parquet(OUT/'forecasts.parquet')
        inference.dump(OUT/'fits.json', fits)
        metrics = evaluate(forecasts, features.index, p)
        if inference.digest(PROTOCOL) != manifest['protocol_sha256']:
            raise ValueError('Frozen protocol changed during index-hinge run')
        for group in ('code', 'inputs', 'preserved'):
            for name, expected in manifest[group].items():
                if inference.digest(ROOT/name) != expected:
                    raise ValueError('Frozen artifact changed: '+name)
        inference.dump(REPORT/'metrics.json', metrics)
        report(metrics)
        ledger([{'event': 'evaluated', **row} for row in metrics['rows']])
    except Exception as error:
        failure = failure_metrics(error, manifest['protocol_sha256'])
        inference.dump(REPORT/'metrics.json', failure)
        inference.dump(REPORT/'failure.json', failure)
        (REPORT/'results.md').write_text('# Index-hinge experiment\n\nUNEVALUABLE: all4comparisons retained with p=1; no lead.\n')
        if metrics is not None:
            inference.dump(REPORT/'unpublished_scored_metrics.json', {'status': 'UNPUBLISHED_DIAGNOSTIC_ONLY',
                           'not_for_inherited_inference_or_promotion': True, 'scored_metrics': metrics})
        ledger([{'event': 'unevaluable', **row} for row in failure['rows']])
        raise
    print(json.dumps({'status': 'SCORED_AWAITING_INDEPENDENT_VERIFICATION', 'forecasts': len(forecasts),
                      'fits': len(fits), 'leads': metrics['leads']}), flush=True)


if __name__ == '__main__':
    run()
