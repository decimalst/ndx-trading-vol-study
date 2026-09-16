"""Fixed first wave of the continuing signal search and cumulative trial ledger."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import yaml

from . import orthogonal_round2 as inference

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'iterative_signal_search.yaml'
REPORT = ROOT / 'reports/iterative_signal_search'
OUT = ROOT / 'data/iterative_signal_search'
CONTRASTS = [('hf', h, c, 'baseline') for h in (1, 5, 21) for c in ('semivariance', 'kernel')]
CONTRASTS += [('index', 1, c, b) for c in ('session_split', 'cross_signed', 'calendar') for b in ('baseline', 'mean')]


def validate_protocol(p):
    if (p['comparisons']['new_hypotheses'] != 12 or p['comparisons']['cumulative_hypotheses'] != 66
            or p['wave'] != 1 or p['wave_alpha'] != .025
            or p['hf']['horizons'] != [1, 5, 21] or p['index']['horizons'] != [1]
            or p['hf']['effect_threshold_pct'] != 1 or p['index']['effect_threshold_pct'] != .25
            or p['index']['latest_target'] != '2025-10-20'
            or p['index']['sealed_start'] != '2025-11-03'
            or p['hf']['latest_target'] != '2017-12-29'
            or p['inference']['blocks'] != [21, 63, 126]
            or p['inference']['bootstrap_draws'] != 19999
            or p['inference']['seed'] != 20260907):
        raise ValueError('Frozen first-wave specification changed')


def loss(actual, prediction, study):
    actual, prediction = np.asarray(actual, float), np.asarray(prediction, float)
    if actual.shape != prediction.shape or actual.ndim != 1 or not np.isfinite([actual, prediction]).all():
        raise ValueError('Finite aligned loss inputs required')
    if study == 'hf':
        return inference.qlike(actual, prediction)
    if study == 'index':
        return (actual - prediction) ** 2
    raise ValueError('Unknown study loss')


def paired_inference(candidate_loss, control_loss, p, base_seed):
    candidate_loss, control_loss = np.asarray(candidate_loss, float), np.asarray(control_loss, float)
    d = candidate_loss - control_loss
    if (d.ndim != 1 or len(d) < max(p['inference']['blocks'])
            or not np.isfinite(d).all() or control_loss.mean() <= 0):
        raise ValueError('Insufficient or invalid paired losses')
    delta = float(d.mean())
    blocks = {}
    for block in p['inference']['blocks']:
        sampled = inference.bootstrap_means(d, block, p['inference']['bootstrap_draws'], base_seed + block)[:, 0]
        pv = (1 + int((np.abs(sampled - delta) >= abs(delta)).sum())) / (len(sampled) + 1)
        blocks[str(block)] = {'p': pv, 'ci95': np.quantile(sampled, [.025, .975]).tolist()}
    hac = inference.hac_summary(d)
    intervals = [hac['ci95']] + [r['ci95'] for r in blocks.values()]
    return {'n': len(d), 'delta': delta, 'control_loss': float(control_loss.mean()),
            'candidate_loss': float(candidate_loss.mean()), 'improvement_pct': float(-100 * delta / control_loss.mean()),
            'block_inference': blocks, 'hac126': hac,
            'ci95_envelope': [min(x[0] for x in intervals), max(x[1] for x in intervals)],
            'p_conservative': max(hac['p'], *(r['p'] for r in blocks.values()))}


def inherited_rows(p):
    result = []
    for source in p['comparisons']['inherited_sources']:
        old = json.loads((ROOT / source).read_text())['rows']
        for r in old:
            result.append({'study': source.split('/')[1], 'horizon': r['horizon'],
                           'candidate': r['candidate'], 'control': r.get('control', 'baseline'),
                           'p_conservative': r['p_conservative'], 'source': source,
                           'source_sha256': inference.digest(ROOT / source)})
    if len(result) != 54:
        raise ValueError('Inherited audited family must include all 54 comparisons')
    return result


def adjust(prior, new):
    if len(prior) != 54 or len(new) != 12:
        raise ValueError('All prior, new and failed hypotheses must remain in family')
    wave = inference.holm_adjust([r['p_conservative'] for r in new])
    total = inference.holm_adjust([r['p_conservative'] for r in prior + new])[-12:]
    return wave, total


def passes(row):
    threshold = 1. if row['study'] == 'hf' else .25
    phases = row.get('phases', [])
    evaluation = next((r for r in phases if r['name'] == 'evaluation'), {})
    stability = evaluation.get('stability', [])
    return (len(phases) == 2 and row['p_holm_wave'] < .025 and row['p_holm_cumulative'] < .05
            and all(r['improvement_pct'] >= threshold for r in phases)
            and len(stability) == 2 and all(r['delta'] < 0 for r in stability))


def execution_returns(actual, positions, cost_per_side):
    actual, positions = np.asarray(actual, float), np.asarray(positions, float)
    if (actual.ndim != 1 or actual.shape != positions.shape
            or not np.isfinite([actual, positions]).all() or not np.isin(positions, [0., 1.]).all()
            or not np.isfinite(cost_per_side) or cost_per_side < 0):
        raise ValueError('Valid long/flat daily positions and two-sided costs required')
    return positions * actual - 2 * cost_per_side * positions


def phase_frame(forecasts, section, phase):
    start, end = section[phase]
    selected = forecasts.loc[(forecasts.origin >= start) & (forecasts.origin <= end)].copy()
    if phase == 'development':
        available = selected.get('available_date', selected.target_end)
        selected = selected.loc[available <= pd.Timestamp(section['development_target_available_by'])]
    return selected


def evaluate(panels, p, calendars):
    rows = []
    for study, h, candidate, control in CONTRASTS:
        phases = []
        section = p[study]
        panel = panels[study].loc[panels[study].horizon == h]
        for phase_code, name in enumerate(('development', 'evaluation')):
            frame = phase_frame(panel, section, name)
            wide = frame.pivot(index='origin', columns='model', values='prediction').sort_index()
            if set(wide) != set(section['models']) or not np.isfinite(wide).all().all():
                raise ValueError('Common model/phase sample failed')
            base = frame.loc[frame.model == control].set_index('origin').reindex(wide.index)
            actual = base.y.to_numpy()
            # Every model must refer to exactly the same observed target.
            for model in section['models']:
                block = frame.loc[frame.model == model].set_index('origin').reindex(wide.index)
                if not np.array_equal(block.y, actual) or not block.target_end.equals(base.target_end):
                    raise ValueError('Mismatched paired actual target')
            losses_c, losses_b = loss(actual, wide[candidate], study), loss(actual, wide[control], study)
            seed = p['inference']['seed'] + (1 if study == 'hf' else 2) * 100000 + h * 1000 + phase_code * 10000
            result = paired_inference(losses_c, losses_b, p, seed)
            d = losses_c - losses_b
            result.update({'name': name, 'first_origin': str(wide.index[0].date()), 'last_origin': str(wide.index[-1].date())})
            result['stability'] = []
            if name == 'evaluation':
                for first, last in section['evaluation_stability']:
                    mask = (wide.index >= first) & (wide.index <= last)
                    if not mask.any():
                        raise ValueError('Missing fixed evaluation stability slice')
                    result['stability'].append({'start': first, 'end': last, 'n': int(mask.sum()), 'delta': float(d[mask].mean())})
            result['annual'] = [{'year': int(year), 'n': int((wide.index.year == year).sum()),
                                 'delta': float(d[wide.index.year == year].mean())} for year in sorted(set(wide.index.year))]
            phase_membership = pd.DatetimeIndex(calendars[study]).get_indexer(wide.index)
            if (phase_membership < 0).any():
                raise ValueError('Scored origin absent from actual session calendar')
            result['nonoverlap_phases'] = [{'phase': i, 'n': int((phase_membership % h == i).sum()),
                                           'delta': float(d[phase_membership % h == i].mean())} for i in range(h)]
            phases.append(result)
        rows.append({'study': study, 'horizon': h, 'candidate': candidate, 'control': control,
                     'phases': phases, 'p_conservative': max(r['p_conservative'] for r in phases)})
    prior = inherited_rows(p)
    wave, total = adjust(prior, rows)
    for row, wp, tp in zip(rows, wave, total):
        row.update({'p_holm_wave': float(wp), 'p_holm_cumulative': float(tp)})
        row['verdict'] = 'EXPLORATORY_LEAD' if passes(row) else 'DOES_NOT_QUALIFY'
    leads = []
    for row in rows:
        if row['verdict'] != 'EXPLORATORY_LEAD':
            continue
        if row['study'] == 'index' and not all(r['verdict'] == 'EXPLORATORY_LEAD' for r in rows if r['study'] == 'index' and r['candidate'] == row['candidate']):
            continue
        key = {'study': row['study'], 'candidate': row['candidate'], 'horizon': row['horizon']}
        if key not in leads:
            leads.append(key)
    return {'rows': rows, 'inherited_rows': prior, 'hypothesis_count': 12,
            'cumulative_hypothesis_count': 66, 'leads': leads,
            'protocol_sha256': inference.digest(PROTOCOL), 'evidence_class': p['evidence_class']}


def execution_screen(forecasts, p):
    records = []
    for name in ('development', 'evaluation'):
        frame = phase_frame(forecasts, p['index'], name)
        for model in p['index']['models'] + ['always_daylong']:
            block = frame.loc[frame.model == ('baseline' if model == 'always_daylong' else model)].sort_values('origin')
            position = np.ones(len(block)) if model == 'always_daylong' else (block.prediction.to_numpy() > .0004).astype(float)
            for cost in p['index']['descriptive_execution']['sensitivity_costs_per_side']:
                net = execution_returns(block.y, position, cost)
                curve = np.cumprod(1 + net)
                peaks = np.maximum.accumulate(np.r_[1., curve])[1:]
                records.append({'phase': name, 'model': model, 'cost_per_side': cost, 'n': len(net),
                    'held_days': int(position.sum()), 'exposure': float(position.mean()),
                    'mean_daily_net': float(net.mean()), 'total_net_return': float(curve[-1] - 1),
                    'max_drawdown': float(np.min(curve / peaks - 1)),
                    'annualized_sharpe': float(np.sqrt(252) * net.mean() / net.std(ddof=1)) if net.std(ddof=1) > 0 else None})
    return {'status': 'DESCRIPTIVE_EXECUTION_ASSUMPTIONS_ONLY', 'rows': records}


def write_report(metrics, execution):
    lines = ['# Iterative signal search: wave 1', '',
        'Exploratory comparisons on reused history, with separate development and algorithm-evaluation periods.',
        'All 12 new hypotheses remain in the search; cumulative adjustment includes 54 audited earlier comparisons.',
        'Positive improvement means lower forecast loss. A lead must pass both phases, both evaluation subperiods, effect-size and multiplicity gates.', '',
        '| Study | Candidate vs control | Horizon | Development gain | Evaluation gain | Wave adjusted p | Cumulative adjusted p | Gate |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for row in metrics['rows']:
        dev, ev = row['phases']
        lines.append(f"| {row['study']} | {row['candidate']} vs {row['control']} | {row['horizon']} | {dev['improvement_pct']:+.3f}% | {ev['improvement_pct']:+.3f}% | {row['p_holm_wave']:.5f} | {row['p_holm_cumulative']:.5f} | {row['verdict']} |")
    lines += ['', 'Leads satisfying all required controls: ' + (json.dumps(metrics['leads']) if metrics['leads'] else '**none**.'), '',
        '## Interpretation boundaries', '',
        '- HF candidates use previously unused five-minute archive measures, with current daily price controls, lagged HF history and lagged VIX. Archived rsv direction is unresolved; it is not labeled downside variance.',
        '- Index models predict the next open-to-close return using prior-close market data and calendar information finalized before the opening. Auction prices and costs are assumptions, not demonstrated historical fills.',
        '- All index candidates must beat both the market model and historical mean. Cost screens below are descriptive and do not establish net trading alpha.',
        '- Reused history and older unenumerated repository exploration prevent a pristine confirmation claim. Multiplicity accounting cannot reverse prior inspection.',
        '- Complete uncertainty, MDE, years and nonoverlapping phases are retained in metrics.json.', '',
        '## Descriptive index execution at two basis points per side', '',
        '| Phase | Model | Held days | Exposure | Net total return | Sharpe | Max drawdown |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in execution['rows']:
        if row['cost_per_side'] == .0002:
            sharpe = 'n/a' if row['annualized_sharpe'] is None else f"{row['annualized_sharpe']:.2f}"
            lines.append(f"| {row['phase']} | {row['model']} | {row['held_days']} | {100*row['exposure']:.1f}% | {100*row['total_net_return']:+.2f}% | {sharpe} | {100*row['max_drawdown']:.2f}% |")
    lines += ['', 'No signals are promoted automatically. No strategy trades, orders, or capital allocations were executed.', '']
    (REPORT / 'results.md').write_text('\n'.join(lines))


def append_ledger(records):
    path = REPORT / 'trial_ledger.jsonl'
    with path.open('a') as stream:
        for row in records:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')


def failure_rows(error):
    status = 'INSUFFICIENT_DATA' if 'INSUFFICIENT_DATA' in str(error) else 'INVALID_RUN'
    return [{'study': s, 'horizon': h, 'candidate': c, 'control': b,
             'status': status, 'error': str(error), 'phases': [], 'p_conservative': 1.}
            for s, h, c, b in CONTRASTS]


def fit_and_score(p):
    from . import iterative_hf, iterative_index
    iterative_hf.run_hf(p['hf'], root=ROOT)
    iterative_index.run()
    panels = {s: pd.read_parquet(OUT / f'{s}_forecasts.parquet') for s in ('hf', 'index')}
    # Map each origin to the original market calendar, never the retained sample.
    calendars = {'hf': pd.read_parquet(OUT / 'hf_features.parquet').index,
                 'index': pd.read_parquet(ROOT / p['index']['source_price'], filters=[('date', '<=', pd.Timestamp(p['index']['source_end']))]).index}
    return evaluate(panels, p, calendars), execution_screen(panels['index'], p)


def run():
    p = yaml.safe_load(PROTOCOL.read_text())
    validate_protocol(p)
    REPORT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tests = ['tests.test_iterative_signal_search', 'tests.test_iterative_hf',
             'tests.test_iterative_index', 'tests.test_verify_iterative_signal_search', 'tests.test_round2_inference']
    checked = subprocess.run([sys.executable, '-m', 'unittest', *tests, '-q'], cwd=ROOT,
                             capture_output=True, text=True, check=False)
    (REPORT / 'pre_run_checks.txt').write_text(checked.stdout + checked.stderr)
    if checked.returncode:
        raise RuntimeError(checked.stdout + checked.stderr)
    names = ['src/iterative_signal_search.py', 'src/iterative_hf.py', 'src/iterative_index.py',
             'tests/test_iterative_signal_search.py', 'tests/test_iterative_hf.py', 'tests/test_iterative_index.py',
             'src/verify_iterative_signal_search.py', 'tests/test_verify_iterative_signal_search.py',
             'src/orthogonal_round2.py', 'src/verify_model_memory_study.py']
    inputs = [p['hf']['source_archive'], p['hf']['source_price_existing'], p['hf']['source_cboe'],
              'data/iterative_signal_search/spx_daily_pre2009_download.parquet',
              'data/iterative_signal_search/spx_daily_pre2009_source.json',
              p['index']['source_price'], p['index']['source_cross'], *p['index']['source_cboe'].values()]
    old = [f for f in list(ROOT.glob('*.yaml')) + list((ROOT / 'reports').rglob('*'))
           if f.is_file() and 'iterative_signal_search' not in str(f)]
    manifest = {'created_utc': datetime.now(UTC).isoformat(), 'protocol_sha256': inference.digest(PROTOCOL),
                'code': {f: inference.digest(ROOT / f) for f in names},
                'inputs': {f: inference.digest(ROOT / f) for f in inputs},
                'preserved': {str(f.relative_to(ROOT)): inference.digest(f) for f in old}}
    path = REPORT / 'manifest.json'
    if path.exists():
        raise ValueError('A registered wave already exists; do not overwrite or repeat selectively')
    inference.dump(path, manifest)
    prior = inherited_rows(p)
    append_ledger([{'event': 'inherited', **r} for r in prior])
    append_ledger([{'event': 'registered', 'study': s, 'horizon': h, 'candidate': c, 'control': b,
                   'protocol_sha256': manifest['protocol_sha256']} for s, h, c, b in CONTRASTS])
    try:
        metrics, execution = fit_and_score(p)
    except Exception as error:
        failed = failure_rows(error)
        inference.dump(REPORT / 'failure.json', {'rows': failed, 'whole_wave_aborted': True})
        append_ledger([{'event': 'unevaluable', **row} for row in failed])
        raise
    for name, expected in manifest['preserved'].items():
        if inference.digest(ROOT / name) != expected:
            raise ValueError(f'Prior frozen artifact changed: {name}')
    inference.dump(REPORT / 'metrics.json', metrics)
    inference.dump(REPORT / 'execution.json', execution)
    append_ledger([{'event': 'evaluated', **r} for r in metrics['rows']])
    write_report(metrics, execution)
    print(json.dumps({'status': 'SCORED_AWAITING_INDEPENDENT_VERIFICATION', 'leads': metrics['leads'],
                      'comparisons': len(metrics['rows'])}), flush=True)


if __name__ == '__main__':
    run()
