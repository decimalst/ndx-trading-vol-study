"""Predeclared reference-model calibration and complete-family comparison."""
from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import yaml

from . import model_memory_study as core
from . import orthogonal_round2 as parent
from .model_memory_estimators import _gamma

ROOT = core.ROOT
PROTOCOL = ROOT / 'model_memory_reference.yaml'
OUT = ROOT / 'data/model_memory_reference'
REPORT = ROOT / 'reports/model_memory_study'
MODELS = ('nlinear', 'xlstm', 'moirai_univariate_gamma', 'moirai_augmented_gamma')
CONTRASTS = [(m, 'baseline', 'primary') for m in MODELS] + [
    ('xlstm', 'nlinear', 'mechanism'),
    ('moirai_augmented_gamma', 'gamma', 'mechanism'),
    ('moirai_augmented_gamma', 'moirai_univariate_gamma', 'mechanism'),
]


def validate_protocol(p):
    if (tuple(p['models']) != MODELS or p['comparisons']['total_with_core'] != 46
            or p['comparisons']['total_reference_hypotheses'] != 14
            or p['sample']['score_start'] != '2016-01-04'
            or p['sample']['score_end'] != '2025-10-10'
            or p['sample']['latest_target'] != '2025-10-20'
            or p['sample']['horizons'] != [1, 5]
            or p['moirai']['context_sessions'] != 512
            or p['neural']['epochs'] != 12 or p['neural']['seed'] != 20260906):
        raise ValueError('Fixed reference family, settings or fences changed')


def training_mask(complete, target, fit_origin):
    return (complete & np.isfinite(target.y) & (target.y > 0)
            & (complete.index < fit_origin) & (target.target_end <= fit_origin))


def fit_gamma(train, y, apply):
    train, y, apply = np.asarray(train, float), np.asarray(y, float), np.asarray(apply, float)
    if (train.ndim != 2 or apply.ndim != 2 or train.shape[1] != apply.shape[1]
            or len(train) < 2 or y.shape != (len(train),) or (y <= 0).any()
            or not all(np.isfinite(a).all() for a in (train, y, apply))):
        raise ValueError('Invalid reference Gamma input')
    mu, sd = train.mean(axis=0), train.std(axis=0)
    if (sd <= 1e-12).any():
        raise ValueError('Zero-scale reference feature')
    return _gamma((train - mu) / sd, y, (apply - mu) / sd, np.ones(len(y)), 0.,
                  {'input_center': mu.tolist(), 'input_scale': sd.tolist()}, False)


def calibrated_forecasts(features, latents, summaries, p, core_predictions):
    if summaries.index.has_duplicates or not summaries.index.is_monotonic_increasing:
        raise ValueError('Invalid Moirai origin calendar')
    complete = (np.isfinite(features.loc[:, parent.ALL_FEATURES]).all(axis=1)
                & features.index.isin(latents.index))
    if not complete.loc[complete].index.isin(summaries.index).all():
        # Only rows through the reference scoring fence are needed.
        missing = complete.loc[:p['sample']['score_end']].loc[lambda a: a].index.difference(summaries.index)
        if len(missing):
            raise ValueError(f'Moirai summaries omit eligible Gamma training rows: {missing[0]}')
    outputs, audits = [], []
    for horizon in (1, 5):
        target = parent.make_targets(features.rv_total, horizon)
        ref = core_predictions.loc[(core_predictions.model == 'baseline') & (core_predictions.horizon == horizon)].set_index('origin').sort_index()
        summary = summaries[f'moirai_log_h{horizon}'].reindex(features.index)
        # Both models train on exactly the original Gamma eligibility set.
        for month in ref.index.to_period('M').unique():
            query = ref.index[ref.index.to_period('M') == month]
            fit_origin = query[0]
            mask = training_mask(complete, target, fit_origin)
            if mask.sum() < p['sample']['minimum_train'] or not np.isfinite(summary.loc[mask]).all():
                raise ValueError('Insufficient or missing reference calibration history')
            for model in MODELS[2:]:
                tr = summary.loc[mask].to_numpy()[:, None]
                ap = summary.loc[query].to_numpy()[:, None]
                if model == 'moirai_augmented_gamma':
                    tr = np.column_stack([features.loc[mask, core.BASE[1:]], tr])
                    ap = np.column_stack([features.loc[query, core.BASE[1:]], ap])
                prediction, audit = fit_gamma(tr, target.loc[mask, 'y'], ap)
                last = target.loc[mask, 'target_end'].max()
                outputs.append(pd.DataFrame({'origin': query, 'horizon': horizon, 'model': model,
                    'prediction': prediction, 'target_end': target.loc[query, 'target_end'].to_numpy(),
                    'y': target.loc[query, 'y'].to_numpy(), 'fit_origin': fit_origin,
                    'train_n': int(mask.sum()), 'train_last_target': last}))
                audits.append({'model': model, 'horizon': horizon, 'fit_origin': str(fit_origin.date()),
                    'train_n': int(mask.sum()), 'train_last_target': str(last.date()), 'audit': audit})
    return pd.concat(outputs, ignore_index=True), audits


def joint_holm(core_rows, reference_rows):
    if len(core_rows) != 32 or len(reference_rows) != 14:
        raise ValueError('All 46 registered hypotheses required, including failures as p=1')
    return parent.holm_adjust([r['p_conservative'] for r in core_rows + reference_rows])


def write_joint_report(core_metrics, reference_metrics):
    rows = deepcopy(core_metrics['rows'] + reference_metrics['rows'])
    adjusted = joint_holm(core_metrics['rows'], reference_metrics['rows'])
    for row, hp in zip(rows, adjusted):
        row['p_holm_all46'] = float(hp)
        passes = hp < .05 and row['improvement_pct'] >= 1 and all(p['delta'] < 0 for p in row['periods'])
        row['verdict_all46'] = 'EXPLORATORY_SHORTLIST' if passes else 'INCONCLUSIVE'
    result = {'hypothesis_count': 46, 'evidence_class': 'exploratory_previously_inspected_history',
              'rows': rows, 'reference_protocol_sha256': parent.digest(PROTOCOL)}
    parent.dump(REPORT / 'combined_metrics.json', result)
    lines = ['# Complete modeling and reference comparison', '',
        'All 46 prespecified comparisons enter the final multiplicity correction. Positive improvement means lower QLIKE.',
        'These historical experiments reuse inspected data. A statistical shortlist is not prospective confirmation.', '',
        '| Model | h1 improvement | h5 improvement | h1 adjusted p | h5 adjusted p |',
        '|---|---:|---:|---:|---:|']
    for model in core.MODELS[1:] + MODELS:
        r = [next(x for x in rows if x['candidate'] == model and x['control'] == 'baseline' and x['horizon'] == h) for h in (1, 5)]
        lines.append(f"| {model} | {r[0]['improvement_pct']:+.2f}% | {r[1]['improvement_pct']:+.2f}% | {r[0]['p_holm_all46']:.4f} | {r[1]['p_holm_all46']:.4f} |")
    lines += ['', '## Mechanism comparisons', '', '| Candidate vs control | h | improvement | adjusted p | verdict |', '|---|---:|---:|---:|---|']
    for r in rows:
        if r['role'] == 'mechanism':
            lines.append(f"| {r['candidate']} vs {r['control']} | {r['horizon']} | {r['improvement_pct']:+.2f}% | {r['p_holm_all46']:.4f} | {r['verdict_all46']} |")
    lines += ['', 'The neural pair uses a fixed small CPU adaptation and a matched linear control. Moirai summaries receive causal Gamma calibration; exponentiated quantiles are not presumed to be mean forecasts.',
        'Moirai and TiRex pretraining overlap with market history is unknown. Timer-S1 and the TimeCopilot orchestration service are reviewed separately, not counted as fitted models.',
        'All periods, annual slices, nonoverlapping phases, uncertainty intervals and nominal minimum detectable effects are retained in combined_metrics.json.', '']
    (REPORT / 'combined_results.md').write_text('\n'.join(lines))
    return result


def run():
    p = yaml.safe_load(PROTOCOL.read_text())
    validate_protocol(p)
    cp = yaml.safe_load(core.PROTOCOL.read_text())
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'tests.test_model_memory_reference',
                            'tests.test_reference_xlstm', 'tests.test_reference_moirai',
                            'tests.test_verify_model_memory_reference', '-q'],
                           capture_output=True, text=True, cwd=ROOT, check=False)
    (REPORT / 'reference_pre_run_checks.txt').write_text(tests.stdout + tests.stderr)
    if tests.returncode:
        raise RuntimeError(tests.stdout + tests.stderr)
    features, latents = core.load_inputs(cp)
    core_forecasts = pd.read_parquet(core.OUT / 'forecasts.parquet')
    summaries = pd.read_parquet(OUT / 'moirai_features.parquet')
    neural = pd.read_parquet(OUT / 'neural_forecasts.parquet')
    source_paths = ['model_memory_reference.yaml', 'src/model_memory_reference.py',
                    'src/reference_xlstm.py', 'src/reference_moirai.py',
                    'tests/test_model_memory_reference.py', 'tests/test_reference_xlstm.py',
                    'tests/test_reference_moirai.py',
                    'reports/model_memory_study/moirai_extraction_manifest.json',
                    'reports/model_memory_study/moirai_extraction_audit.json',
                    'data/model_memory_reference/neural_manifest.json',
                    'data/model_memory_reference/moirai_features.parquet',
                    'data/model_memory_reference/neural_forecasts.parquet',
                    'data/model_memory_study/forecasts.parquet']
    manifest = {'created_utc': datetime.now(UTC).isoformat(),
                'hashes': {f: parent.digest(ROOT / f) for f in source_paths},
                'protocol_sha256': parent.digest(PROTOCOL)}
    path = REPORT / 'reference_manifest.json'
    if path.exists() and json.loads(path.read_text())['hashes'] != manifest['hashes']:
        raise ValueError('Previous reference run hashes changed; do not overwrite')
    parent.dump(path, manifest)
    calibrated, audits = calibrated_forecasts(features, latents, summaries, cp, core_forecasts)
    neural = neural.loc[(neural.origin >= cp['sample']['score_start']) & (neural.origin <= cp['sample']['score_end'])]
    forecasts = pd.concat([core_forecasts, neural, calibrated], ignore_index=True)
    for horizon in (1, 5):
        expected = core_forecasts.loc[(core_forecasts.model == 'baseline') & (core_forecasts.horizon == horizon)].set_index('origin').sort_index()
        for model in MODELS:
            block = forecasts.loc[(forecasts.model == model) & (forecasts.horizon == horizon)].set_index('origin').sort_index()
            if not block.index.equals(expected.index) or not block.target_end.equals(expected.target_end) or not np.allclose(block.y, expected.y, rtol=1e-12, atol=0):
                raise ValueError(f'Reference sample/target mismatch: {model} h{horizon}')
    forecasts.to_parquet(OUT / 'forecasts.parquet', index=False)
    parent.dump(OUT / 'calibration_fits.json', audits)
    ep = deepcopy(cp)
    ep['comparisons']['total_hypotheses'] = 14
    metrics = core.evaluate(forecasts, features, ep, contrasts=CONTRASTS, models=core.MODELS + MODELS)
    metrics['protocol_sha256'] = parent.digest(PROTOCOL)
    parent.dump(REPORT / 'reference_metrics.json', metrics)
    result = write_joint_report(json.loads((core.REPORT / 'metrics.json').read_text()), metrics)
    print(json.dumps([{k: r[k] for k in ('candidate', 'control', 'horizon', 'improvement_pct', 'p_holm_all46', 'verdict_all46')}
                      for r in result['rows'] if r['role'] == 'primary'], indent=2))


if __name__ == '__main__':
    run()
