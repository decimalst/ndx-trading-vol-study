"""Execute frozen TiRex interventions on fresh, balanced motif strata.

Original adapters and earlier pilot archives remain unchanged. All interventions
are simultaneous replacements from untouched baseline activations. Calibrated
numerical resolvability is frozen before any validation generation or forecast.
"""

import argparse
import hashlib
import importlib.metadata
import json
import time
from numbers import Real
from pathlib import Path

import numpy as np

from src.tirex_group_adapter import GroupTirexAdapter
from src.tirex_group_run import donor_values, group_instrumentation
from src.tirex_intervention_run import (
    batched,
    flatten,
    save_baseline,
    save_json,
    sha,
    utc,
)
from src.tirex_intervention_statistics import holm_adjust


def dose_values(activations, units, alpha, mask=None):
    """Recipient + alpha*(untouched original donor - original recipient)."""
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real) or not np.isfinite(alpha) or alpha < 0:
        raise ValueError('alpha must be a finite nonnegative real number')
    donor = donor_values(activations, units, mask)
    selected = np.asarray(activations) if mask is None else np.asarray(activations)[mask]
    original = selected[:, :, units].reshape(-1, len(units))
    return (original.astype(float) + float(alpha) * (donor.astype(float) - original)).astype(np.float32)


def first_per_cell(pairs, count):
    """Deterministic ordered subsample with no forecast-dependent filtering."""
    if type(count) is not int or count < 1:
        raise ValueError('count must be a positive integer')
    families, variants = np.asarray(pairs['family']), np.asarray(pairs['variant'])
    if families.ndim != 1 or families.shape != variants.shape or not len(families):
        raise ValueError('family and variant must be matching nonempty vectors')
    mask = np.zeros(len(families), dtype=bool)
    for family, variant in dict.fromkeys(zip(families.tolist(), variants.tolist(), strict=True)):
        indices = np.flatnonzero((families == family) & (variants == variant))
        if len(indices) < count:
            raise ValueError('every represented cell must contain count pairs')
        mask[indices[:count]] = True
    return mask


def make_random_reference(groups, count, seed):
    """Uniform within-draw samples; independent draws may overlap each other."""
    if type(count) is not int or count < 1 or type(seed) is not int or seed < 0:
        raise ValueError('count and seed must be valid integers')
    top = [g for g in groups if g['group_id'].endswith('_k8')]
    if len(top) != 3 or len({g['family'] for g in top}) != 3 or any(len(g['units']) != 8 for g in top):
        raise ValueError('exactly three original eight-unit family groups required')
    rng, result = np.random.default_rng(seed), []
    for group in top:
        forbidden = set(group['units']) | ({103} if group['layer'] == 0 else set())
        allowed = np.asarray([unit for unit in range(1408) if unit not in forbidden])
        for draw in range(count):
            units = rng.choice(allowed, 8, replace=False).tolist()
            for size in (4, 8):
                result.append({'group_id': f"random_{group['family']}_{draw:02d}_k{size}",
                    'family': group['family'], 'layer': group['layer'], 'units': units[:size],
                    'draw': draw, 'size': size})
    return result


def forecast_swap(adapter, contexts, activations, layer, units, mask, batch_size, alpha=1.):
    """Forecast selected complete pairs with aligned original-donor patches."""
    mask = np.asarray(mask)
    patch = {'layer': layer, 'units': units, 'values': dose_values(activations, units, alpha, mask)}
    output = batched(adapter, np.asarray(contexts)[np.repeat(mask, 2)], batch_size, patch=patch, capture=False)
    return output['quantiles'].reshape(-1, 2, 3)


def save_cell_baselines(out, pairs, baseline):
    """Keep each raw archive bounded while preserving all baseline fields."""
    paths = []
    for family, variant in dict.fromkeys(zip(pairs['family'].tolist(), pairs['variant'].tolist(), strict=True)):
        mask = (pairs['family'] == family) & (pairs['variant'] == variant)
        rows = np.repeat(mask, 2)
        local = {key: value if key == 'split' else value[mask] for key, value in pairs.items()}
        output = {'quantiles': baseline['quantiles'][rows],
            'activations': {layer: values[rows] for layer, values in baseline['activations'].items()},
            'residual': baseline['residual'][rows]}
        path = out / f"{pairs['split']}_{family}_{variant}.npz"
        save_baseline(path, local, output)
        paths.append(path)
    return paths


def _context_hashes(pairs):
    return {hashlib.sha256(np.asarray(row, dtype=np.float32).tobytes()).hexdigest()
            for side in ('recipient', 'donor') for row in pairs[side]}


def assert_disjoint(pairs, prior_runs, calibration=None):
    """Compare exact adapter-input float32 bytes and pair identities."""
    hashes = _context_hashes(pairs)
    expected = 2 * len(pairs['pair_id'])
    if len(hashes) != expected or len(set(pairs['pair_id'])) != expected // 2:
        raise RuntimeError('Duplicate new input context or pair identifier')
    receipts = {}
    for prior in prior_runs:
        for split in ('discovery', 'validation'):
            path = prior / f'{split}.npz'
            with np.load(path, allow_pickle=False) as old:
                if hashes & _context_hashes(old) or set(pairs['pair_id']) & set(old['pair_id']):
                    raise RuntimeError('Fresh data overlap a prior pilot')
            receipts[str(path)] = sha(path)
    if calibration is not None and (hashes & _context_hashes(calibration) or set(pairs['pair_id']) & set(calibration['pair_id'])):
        raise RuntimeError('Calibration and validation overlap')
    return receipts


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _effect(base, swapped):
    change = swapped[:, :, 1].astype(float) - base[:, :, 1].astype(float)
    return (change[:, 0] - change[:, 1]) / 2


def _description(base, swapped, activations, units):
    change = swapped.astype(float) - base.astype(float)
    displacement = activations[:, 1, units].astype(float) - activations[:, 0, units].astype(float)
    return {'n_pairs': len(base), 'mean_e': float(_effect(base, swapped).mean()),
        'mean_abs_median_shift': float(np.abs(change[:, :, 1]).mean()),
        'mean_recipient_quantile_change': change[:, 0].mean(axis=0).tolist(),
        'mean_donor_quantile_change': change[:, 1].mean(axis=0).tolist(),
        'mean_abs_width_change': float(np.abs(change[:, :, 2] - change[:, :, 0]).mean()),
        'input_displacement_rms': float(np.sqrt(np.mean(np.sum(displacement**2, axis=1)))),
        'mean_native_median_contrast': float((base[:, 1, 1].astype(float) - base[:, 0, 1]).mean()),
        'forecast_crossing_count': int(np.any(np.diff(swapped, axis=2) < 0, axis=2).sum()),
        'base_crossing_count': int(np.any(np.diff(base, axis=2) < 0, axis=2).sum())}


def numerical_band_from_flat(baseline, nulls):
    """Adapt wrapper rows to the statistics API's paired forecast convention."""
    from src.tirex_stratified_core import calibrate_deadband

    baseline = np.asarray(baseline)
    if baseline.ndim != 2 or baseline.shape[1] != 3 or len(baseline) % 2:
        raise ValueError('baseline must have interleaved 2N x 3 forecast rows')
    if any(np.asarray(value).shape != baseline.shape for value in nulls.values()):
        raise ValueError('every null must match the baseline row shape')
    numeric = calibrate_deadband(baseline.reshape(-1, 2, 3),
        *(np.asarray(value).reshape(-1, 2, 3) for value in nulls.values()))
    numeric['null_max_abs_errors'] = {name: float(np.abs(np.asarray(value).astype(float) - baseline).max())
        for name, value in nulls.items()}
    return numeric


def calibrate_nulls(adapter, contexts, baseline, pairs, groups, batch_size, out):
    """Measure deterministic jitter, no-op and alternate-batch discrepancies."""
    original = baseline['quantiles']
    nulls = {'repeat': batched(adapter, contexts, batch_size, capture=False)['quantiles'],
        'reversed_batch': batched(adapter, contexts[::-1], batch_size, capture=False)['quantiles'][::-1]}
    fixed_rows = np.repeat(first_per_cell(pairs, 1), 2)
    singleton = original.copy()
    singleton[fixed_rows] = batched(adapter, contexts[fixed_rows], 1, capture=False)['quantiles']
    nulls['singleton_fixed_first_pair_per_cell'] = singleton
    for group in groups:
        if not group['group_id'].endswith('_k8'):
            continue
        layer, units = group['layer'], group['units']
        patch = {'layer': layer, 'units': units, 'values': baseline['activations'][layer][:, units]}
        nulls[f'self_value_L{layer}'] = batched(adapter, contexts, batch_size, patch=patch, capture=False)['quantiles']
    np.savez_compressed(out / 'calibration_nulls.npz', baseline_quantiles=original, pair_id=pairs['pair_id'],
                        singleton_tested_rows=np.flatnonzero(fixed_rows), **nulls)
    return nulls


def run(protocol_path, out, prior_runs):
    from src.tirex_recent_mean import evaluate_readout, fit_readout
    from src.tirex_stratified_core import (
        SEEDS,
        make_stratified_pairs,
        minimum_detectable_mean,
        summarize_swaps,
    )

    protocol = json.loads(protocol_path.read_text())
    if any(protocol['seeds'][key] != value for key, value in SEEDS.items()):
        raise ValueError('Protocol seeds must match the tested generator')
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / 'protocol.json', protocol)
    root = Path(__file__).resolve().parent.parent
    source_paths = [root / 'src' / f'{stem}.py' for stem in (
        'tirex_stratified_run', 'tirex_stratified_core', 'tirex_recent_mean', 'tirex_group_adapter', 'tirex_group_run',
        'tirex_group_pilot', 'tirex_intervention_adapter', 'tirex_intervention_run',
        'tirex_intervention_pilot', 'tirex_intervention_statistics')]
    source_paths += sorted((root / 'tests').glob('test_tirex_stratified_*.py'))
    source_paths += [root / 'tests' / 'test_tirex_recent_mean.py']
    save_json(out / 'pre_execution.json', {'timestamp_before_model_load': utc(),
        'protocol_sha256': sha(protocol_path),
        'source_hashes': {str(path.relative_to(root)): sha(path) for path in source_paths}})
    groups = protocol['interventions']
    save_json(out / 'intervention_lock.json', {'timestamp_before_calibration': utc(),
        'interventions': groups, 'protocol_sha256': sha(out / 'protocol.json')})
    initial_lock = sha(out / 'intervention_lock.json')
    start = time.monotonic()
    print('Protocol, original units, and source bytes frozen; loading pinned cached model.', flush=True)
    adapter = GroupTirexAdapter(checkpoint_revision=protocol['checkpoint_revision'],
        layers=tuple(protocol['layers']), threads=protocol['threads']).load()
    import tirex2
    package = Path(tirex2.__file__).parent
    adapter.metadata['installed_source_hashes'] = {str(path.relative_to(package)): sha(path)
        for path in sorted(package.rglob('*.py'))}
    adapter.metadata['versions']['scipy'] = importlib.metadata.version('scipy')
    save_json(out / 'model_metadata.json', adapter.metadata)
    np.savez_compressed(out / 'weight_norms.npz', **{f'layer_{layer}': values for layer, values in adapter.weight_norms.items()})
    calibration = make_stratified_pairs('calibration', protocol['calibration_pairs_per_cell'], protocol['context_length'])
    prior_hashes = assert_disjoint(calibration, prior_runs)
    contexts_c = flatten(calibration)
    first = first_per_cell(calibration, 1)
    group_instrumentation(adapter, contexts_c[np.repeat(first, 2)], protocol, out)
    print(f'Frozen scalar/group normalization gates passed at {time.monotonic()-start:.1f}s.', flush=True)
    batch_size = protocol['batch_size']
    baseline_c = batched(adapter, contexts_c, batch_size)
    calibration_paths = save_cell_baselines(out, calibration, baseline_c)
    base_c = baseline_c['quantiles'].reshape(-1, 2, 3)
    acts_c = {layer: values.reshape(-1, 2, 1408) for layer, values in baseline_c['activations'].items()}
    nulls = calibrate_nulls(adapter, contexts_c, baseline_c, calibration, groups, batch_size, out)
    numeric = numerical_band_from_flat(baseline_c['quantiles'], nulls)
    tau = numeric['deadband']
    save_json(out / 'numerical_calibration.json', _plain(numeric))
    calibration_results = {}
    for index, group in enumerate(groups):
        mask = calibration['family'] == group['family']
        q = forecast_swap(adapter, contexts_c, acts_c[group['layer']], group['layer'], group['units'], mask, batch_size)
        np.savez_compressed(out / f"calibration_swap_{group['group_id']}.npz", quantiles=q,
            pair_id=calibration['pair_id'][mask], variant=calibration['variant'][mask], units=np.asarray(group['units']))
        cells = {}
        for variant in ('short', 'long'):
            local = calibration['variant'][mask] == variant
            stats = summarize_swaps(base_c[mask][local], q[local], tau, seed=protocol['seeds']['bootstrap'] + index)
            sd = float(np.std(_effect(base_c[mask][local], q[local]), ddof=1))
            cells[variant] = {'summary': _plain(stats), 'mde80': _plain(minimum_detectable_mean(sd,
                n=protocol['validation_pairs_per_cell'], tau=tau, claims=protocol['primary']['claims']))}
        calibration_results[group['group_id']] = cells
    readout_mask = calibration['family'] == 'recency'
    readout_model = fit_readout(np.stack((calibration['recipient'][readout_mask], calibration['donor'][readout_mask]), axis=1),
        acts_c[0][readout_mask, :, 666])
    references = make_random_reference(groups, protocol['random_reference']['draws'], protocol['seeds']['random_groups'])
    freeze = {'timestamp_before_validation': utc(), 'tau': tau, 'numerical_calibration_sha256': sha(out / 'numerical_calibration.json'),
        'intervention_lock_sha256': initial_lock, 'calibration': calibration_results, 'random_reference': references, 'recent_mean_readout': readout_model,
        'calibration_file_hashes': {path.name: sha(path) for path in calibration_paths}, 'prior_run_hashes': prior_hashes}
    save_json(out / 'validation_lock.json', freeze)
    validation_lock_sha = sha(out / 'validation_lock.json')
    print(f'Calibration complete; tau={tau:.9g}; MDE and random references frozen before validation at {time.monotonic()-start:.1f}s.', flush=True)
    validation = make_stratified_pairs('validation', protocol['validation_pairs_per_cell'], protocol['context_length'])
    assert_disjoint(validation, prior_runs, calibration)
    contexts_v = flatten(validation)
    baseline_v = batched(adapter, contexts_v, batch_size)
    save_cell_baselines(out, validation, baseline_v)
    base_v = baseline_v['quantiles'].reshape(-1, 2, 3)
    acts_v = {layer: values.reshape(-1, 2, 1408) for layer, values in baseline_v['activations'].items()}
    readout_results = {}
    for variant in ('all', 'short', 'long'):
        mask = validation['family'] == 'recency'
        if variant != 'all':
            mask &= validation['variant'] == variant
        readout_results[variant] = evaluate_readout(readout_model,
            np.stack((validation['recipient'][mask], validation['donor'][mask]), axis=1), acts_v[0][mask, :, 666])
    save_json(out / 'recent_mean_diagnostic.json', {'frozen_model': readout_model, 'validation': readout_results,
        'interpretation': 'One-feature activation surrogate; not an identified forecast mechanism or a proof of linearity.'})
    completed, records = 0, []

    def execute(kind, filename, group, mask, alpha=1.):
        nonlocal completed
        q = forecast_swap(adapter, contexts_v, acts_v[group['layer']], group['layer'], group['units'], mask, batch_size, alpha)
        np.savez_compressed(out / filename, quantiles=q, pair_id=validation['pair_id'][mask],
            variant=validation['variant'][mask], units=np.asarray(group['units']), alpha=np.asarray(alpha))
        completed += 1
        record = {'kind': kind, 'filename': filename, 'completed': completed,
                  'n_pairs': int(mask.sum()), 'elapsed_seconds': time.monotonic()-start, 'timestamp': utc()}
        records.append(record)
        save_json(out / 'progress.json', record)
        print(json.dumps(record), flush=True)
        return q

    primary, forecasts, pvalues = [], {}, []
    for index, group in enumerate(groups):
        gid, family = group['group_id'], group['family']
        mask = validation['family'] == family
        q = execute('primary', f'validation_swap_{gid}.npz', group, mask)
        forecasts[gid] = q
        for cell, variant in enumerate(('short', 'long')):
            local = validation['variant'][mask] == variant
            stats = _plain(summarize_swaps(base_v[mask][local], q[local], tau,
                seed=protocol['seeds']['bootstrap'] + 100 + 2*index + cell))
            pvalues.append(stats['primary_pvalue'])
            primary.append({'group_id': gid, 'family': family, 'variant': variant, 'units': group['units'],
                'layer': group['layer'], 'summary': stats, 'calibration': calibration_results[gid][variant],
                'description': _description(base_v[mask][local], q[local], acts_v[group['layer']][mask][local], group['units'])})
    adjusted = holm_adjust(pvalues)
    for item, pvalue in zip(primary, adjusted, strict=True):
        item['holm_pvalue'] = float(pvalue)
        desc = item['description']
        item['numerical_influence_detected'] = bool(pvalue < protocol['primary']['alpha'] and desc['forecast_crossing_count'] == 0 and desc['base_crossing_count'] == 0)
    save_json(out / 'primary_summary.json', {'tau': tau, 'claims': primary})
    print(f'All18 primary strata summarized at {time.monotonic()-start:.1f}s; remaining reference/dose analyses descriptive.', flush=True)
    reference_mask = first_per_cell(validation, protocol['random_reference']['pairs_per_cell'])
    if protocol['dose_response']['pairs_per_cell'] != protocol['random_reference']['pairs_per_cell']:
        raise ValueError('Frozen run requires the same fixed reference and dose subset size')
    reference_results = []
    for reference in references:
        mask = reference_mask & (validation['family'] == reference['family'])
        q = execute('random_reference', f"validation_{reference['group_id']}.npz", reference, mask)
        desc = {variant: _description(base_v[mask][validation['variant'][mask] == variant],
            q[validation['variant'][mask] == variant],
            acts_v[reference['layer']][mask][validation['variant'][mask] == variant], reference['units'])
            for variant in ('short', 'long')}
        reference_results.append({'reference': reference, 'by_variant': desc})
    candidate_reference = []
    for group in groups:
        mask = validation['family'] == group['family']
        local = reference_mask[mask]
        base, q, acts = base_v[mask][local], forecasts[group['group_id']][local], acts_v[group['layer']][mask][local]
        variants = validation['variant'][mask][local]
        candidate_reference.append({'group_id': group['group_id'], 'by_variant': {
            variant: _description(base[variants == variant], q[variants == variant], acts[variants == variant], group['units'])
            for variant in ('short', 'long')}})
    save_json(out / 'random_reference_summary.json', {'interpretation': protocol['random_reference']['interpretation'],
        'candidates_on_identical_subset': candidate_reference, 'references': reference_results})
    recovery = next(group for group in groups if group['group_id'] == 'recovery_k8')
    dose_groups = [dict(group) for group in groups if group['family'] == 'recovery']
    dose_groups += [{'group_id': f'recovery_U{unit}', 'family': 'recovery', 'layer': 5, 'units': [unit]} for unit in recovery['units']]
    mask = reference_mask & (validation['family'] == 'recovery')
    dose_q, dose_records = {}, []
    for group in dose_groups:
        for alpha in protocol['dose_response']['alphas']:
            gid = group['group_id']
            reused = alpha == 1 and gid in forecasts
            filename = f"validation_dose_{gid}_a{str(alpha).replace('.', 'p')}.npz"
            if reused:
                family_mask = validation['family'] == group['family']
                q = forecasts[gid][reference_mask[family_mask]].copy()
                np.savez_compressed(out / filename, quantiles=q, pair_id=validation['pair_id'][mask],
                    variant=validation['variant'][mask], units=np.asarray(group['units']), alpha=np.asarray(alpha))
            else:
                q = execute('dose', filename, group, mask, alpha)
            dose_q[gid, alpha] = q
            dose_records.append({'group_id': gid, 'alpha': alpha, 'filename': filename, 'reused_primary': reused})
    dose_descriptions = []
    for record in dose_records:
        gid, alpha = record['group_id'], record['alpha']
        q = dose_q[gid, alpha]
        delta = q.astype(float) - base_v[mask].astype(float)
        unit_dose = dose_q[gid, 1.].astype(float) - base_v[mask].astype(float)
        row = {**record, 'mean_abs_median_curvature': float(np.abs(delta[:, :, 1] - alpha * unit_dose[:, :, 1]).mean()),
            'mean_recipient_median_curvature': float((delta[:, 0, 1] - alpha * unit_dose[:, 0, 1]).mean()),
            'mean_donor_median_curvature': float((delta[:, 1, 1] - alpha * unit_dose[:, 1, 1]).mean())}
        group = next(g for g in dose_groups if g['group_id'] == gid)
        if len(group['units']) > 1:
            singleton_sum = sum(dose_q[f'recovery_U{unit}', alpha].astype(float) - base_v[mask].astype(float) for unit in group['units'])
            row['mean_abs_joint_minus_sum_median'] = float(np.abs(delta[:, :, 1] - singleton_sum[:, :, 1]).mean())
            row['mean_recipient_joint_minus_sum_median'] = float((delta[:, 0, 1] - singleton_sum[:, 0, 1]).mean())
            row['mean_donor_joint_minus_sum_median'] = float((delta[:, 1, 1] - singleton_sum[:, 1, 1]).mean())
        row['by_variant'] = {variant: _description(base_v[mask][validation['variant'][mask] == variant],
            q[validation['variant'][mask] == variant], acts_v[group['layer']][mask][validation['variant'][mask] == variant], group['units'])
            for variant in ('short', 'long')}
        dose_descriptions.append(row)
    save_json(out / 'dose_summary.json', {'interpretation': protocol['dose_response']['analysis'], 'results': dose_descriptions})
    if sha(out / 'validation_lock.json') != validation_lock_sha or sha(out / 'intervention_lock.json') != initial_lock:
        raise RuntimeError('Frozen selection or calibration changed during validation')
    save_json(out / 'execution_records.json', records)
    save_json(out / 'summary.json', {'finished_utc': utc(), 'elapsed_seconds': time.monotonic()-start,
        'validation_lock_sha256': validation_lock_sha, 'tau': tau, 'primary': primary,
        'n_random_reference_groups': len(references), 'n_dose_combinations': len(dose_records),
        'limitations': protocol['primary']['meaning']})
    save_json(out / 'manifest.json', {path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file() and path.name != 'manifest.json'})
    print(f'Stratified pilot finished after {time.monotonic()-start:.1f}s: {out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prior-run', type=Path, action='append', required=True)
    args = parser.parse_args()
    run(args.protocol, args.output, args.prior_run)


if __name__ == '__main__':
    main()
