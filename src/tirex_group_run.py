"""Run fixed groups and matched controls on fresh synthetic TiRex histories."""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np

from src.tirex_group_adapter import GroupTirexAdapter
from src.tirex_group_pilot import FAMILIES, analyze_group, make_group_pairs, select_groups
from src.tirex_intervention_adapter import TirexAdapter
from src.tirex_intervention_run import (
    batched,
    flatten,
    instrumentation,
    save_baseline,
    save_json,
    sha,
    utc,
)
from src.tirex_intervention_statistics import holm_adjust


def donor_values(activations, units, mask=None):
    """Original opposite-side scalars in recipient/donor row and member order."""
    values = np.asarray(activations)
    if values.ndim != 3 or values.shape[0] == 0 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError('activations must be finite nonempty pairs x 2 x units')
    if not isinstance(units, list) or not units or any(type(u) is not int or not 0 <= u < values.shape[2] for u in units) or len(set(units)) != len(units):
        raise ValueError('units must be a distinct nonempty list of valid integers')
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != (len(values),) or mask.dtype != np.bool_ or not mask.any():
            raise ValueError('mask must be a nonempty boolean pair selection')
        values = values[mask]
    return values[:, ::-1, :][:, :, units].reshape(-1, len(units)).copy()


def group_instrumentation(adapter, contexts, protocol, out):
    """Supplement the frozen scalar gates with simultaneous-group checks."""
    rules = protocol['instrumentation']
    scalar = TirexAdapter.from_model(adapter.model, adapter._timeseries_type, layers=adapter.layers)
    instrumentation(scalar, contexts, protocol, out)
    baseline = adapter.forecast(contexts)
    raw, records = {'contexts': contexts}, []

    def compare(name, actual, expected):
        raw[name + '_actual_quantiles'] = actual['quantiles']
        raw[name + '_expected_quantiles'] = expected['quantiles']
        qerr = float(np.max(np.abs(actual['quantiles'] - expected['quantiles'])))
        errors = {}
        for layer in adapter.layers:
            a, b = actual['activations'][layer], expected['activations'][layer]
            errors[str(layer)] = float(np.max(np.abs(a-b)) / max(1., float(np.max(np.abs(b)))))
            raw[f'{name}_actual_layer_{layer}'] = a
            raw[f'{name}_expected_layer_{layer}'] = b
        a, b = actual['residual'], expected['residual']
        errors['residual'] = float(np.max(np.abs(a-b)) / max(1., float(np.max(np.abs(b)))))
        raw[name + '_actual_residual'], raw[name + '_expected_residual'] = a, b
        records.append({'check': name, 'max_forecast_abs_error': qerr,
            'activation_relative_errors': errors,
            'passed': qerr <= rules['max_forecast_abs_error'] and all(x <= rules['max_activation_relative_error'] for x in errors.values())})

    units = rules['fixed_gate_units']
    for layer in adapter.layers:
        acts = baseline['activations'][layer].reshape(-1, 2, 1408)
        natural = donor_values(acts, units)
        self_values = baseline['activations'][layer][:, units]
        compare(f'self_group_L{layer}', adapter.forecast(contexts, patch={'layer':layer, 'units':units, 'values':self_values}), baseline)
        scalar_result = scalar.forecast(contexts, patch={'layer':layer, 'unit':units[0], 'values':natural[:, 0]})
        one = adapter.forecast(contexts, patch={'layer':layer, 'units':[units[0]], 'values':natural[:, :1]})
        compare(f'one_equals_scalar_L{layer}', one, scalar_result)
        patch = {'layer':layer, 'units':units, 'values':natural}
        group = adapter.forecast(contexts, patch=patch)
        permuted = adapter.forecast(contexts, patch={'layer':layer, 'units':units[::-1], 'values':natural[:, ::-1]})
        compare(f'member_permutation_L{layer}', permuted, group)
        compare(f'patched_singleton_batches_L{layer}', batched(adapter, contexts, 1, patch=patch), group)
        compare(f'cleanup_L{layer}', adapter.forecast(contexts), baseline)
    result = {'checks': records, 'passed': all(x['passed'] for x in records), 'timestamp': utc()}
    save_json(out / 'group_instrumentation.json', result)
    np.savez_compressed(out / 'group_instrumentation.npz', **raw)
    if not result['passed']:
        raise RuntimeError('Group instrumentation failed before selection')


def describe_group(pairs, base, swapped, activations, group):
    result = {}
    delta = swapped[:, :, 1].astype(float) - base[:, :, 1]
    width = (swapped[:, :, 2].astype(float)-swapped[:, :, 0]) - (base[:, :, 2].astype(float)-base[:, :, 0])
    a = activations[group['layer']].astype(float)
    members = group['members']
    oriented_activation = (a[:, 1, group['units']] - a[:, 0, group['units']])
    oriented_activation *= np.array([m['orientation']/m['activation_scale'] for m in members])
    direction = group['forecast_direction']
    for family in FAMILIES:
        for variant in ('all', 'short', 'long'):
            mask = pairs['family'] == family
            if variant != 'all':
                mask &= pairs['variant'] == variant
            if not mask.any():
                continue
            change = delta[mask]
            result[f'{family}/{variant}'] = {
                'n_pairs': int(mask.sum()),
                'mean_signed_shift': float(direction * ((change[:, 0]-change[:, 1])/2).mean()),
                'mean_abs_median_shift': float(np.abs(change).mean()),
                'mean_recipient_shift': float(change[:, 0].mean()),
                'mean_donor_shift': float(change[:, 1].mean()),
                'mean_native_contrast': float((base[mask, 1, 1].astype(float)-base[mask, 0, 1]).mean()),
                'both_directions_expected_sign_fraction': float(((direction*change[:, 0]>0)&(-direction*change[:, 1]>0)).mean()),
                'mean_abs_width_shift': float(np.abs(width[mask]).mean()),
                'activation_oriented_mean': float(oriented_activation[mask].mean()),
                'activation_consistency_fraction': float((oriented_activation[mask].mean(axis=1)>0).mean()),
                'all_member_signs_fraction': float((oriented_activation[mask]>0).all(axis=1).mean()),
            }
    return result


def check_disjoint(pairs, prior_run, other=None):
    values = {row.tobytes() for row in np.concatenate((pairs['recipient'], pairs['donor']))}
    prior_hashes = {}
    for split in ('discovery', 'validation'):
        path = prior_run / f'{split}.npz'
        previous = np.load(path, allow_pickle=False)
        prior_values = {row.tobytes() for row in np.concatenate((previous['recipient'], previous['donor']))}
        if values & prior_values or set(pairs['pair_id']) & set(previous['pair_id']):
            raise RuntimeError('New experiment overlaps old pilot contexts or IDs')
        prior_hashes[path.name] = sha(path)
    if other is not None and (
        values & {row.tobytes() for row in np.concatenate((other['recipient'], other['donor']))}
        or set(pairs['pair_id']) & set(other['pair_id'])
    ):
        raise RuntimeError('Discovery and validation overlap')
    return prior_hashes


def run(protocol_path, out, prior_run):
    protocol = json.loads(protocol_path.read_text())
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / 'protocol.json', protocol)
    root = Path(__file__).resolve().parent.parent
    dependencies = ['src/tirex_group_adapter.py','src/tirex_group_pilot.py','src/tirex_group_run.py',
        'src/tirex_intervention_adapter.py','src/tirex_intervention_pilot.py',
        'src/tirex_intervention_statistics.py','src/tirex_intervention_run.py']
    tests = list((root / 'tests').glob('test_tirex_group_*.py')) + [root/'tests/test_tirex_intervention_adapter.py']
    sources = [root/p for p in dependencies] + tests
    save_json(out / 'pre_execution.json', {'timestamp_before_model_load':utc(),
        'protocol_sha256':sha(protocol_path), 'source_hashes':{str(p.relative_to(root)):sha(p) for p in sources}})
    start = time.monotonic()
    print('Group protocol and sources frozen; loading cached checkpoint.', flush=True)
    adapter = GroupTirexAdapter(checkpoint_revision=protocol['checkpoint_revision'], layers=tuple(protocol['layers']), threads=4).load()
    import tirex2
    package = Path(tirex2.__file__).parent
    adapter.metadata['installed_source_hashes'] = {str(p.relative_to(package)):sha(p) for p in sorted(package.rglob('*.py'))}
    adapter.metadata['versions']['scipy'] = importlib.metadata.version('scipy')
    adapter.metadata['group_patch'] = 'Simultaneous original-donor coordinates, inherited cloned pre-hook assignment'
    save_json(out/'model_metadata.json', adapter.metadata)
    np.savez_compressed(out/'weight_norms.npz', **{f'layer_{k}':v for k,v in adapter.weight_norms.items()})
    discovery = make_group_pairs('discovery', protocol['discovery_pairs_per_family'], protocol['context_length'])
    prior_hashes = check_disjoint(discovery, prior_run)
    contexts = flatten(discovery)
    n = protocol['discovery_pairs_per_family']
    group_instrumentation(adapter, contexts[[0,1,2*n,2*n+1,4*n,4*n+1]], protocol, out)
    print(f'All scalar and group instrumentation passed after {time.monotonic()-start:.1f}s.', flush=True)
    baseline_d = batched(adapter, contexts, protocol['batch_size'])
    save_baseline(out/'discovery.npz', discovery, baseline_d)
    acts_d = {k:v.reshape(-1,2,1408).astype(float) for k,v in baseline_d['activations'].items()}
    groups = select_groups(discovery, acts_d, adapter.weight_norms, sizes=tuple(protocol['sizes']), n_controls=protocol['n_controls'], layers=protocol['family_layers'])
    for group in groups:
        mask = discovery['family'] == group['family']
        rows = np.repeat(mask, 2)
        patch = {'layer':group['layer'], 'units':group['units'], 'values':donor_values(acts_d[group['layer']], group['units'], mask)}
        swapped = batched(adapter, contexts[rows], protocol['batch_size'], patch=patch, capture=False)['quantiles'].reshape(-1,2,3)
        delta = swapped[:,:,1].astype(float) - baseline_d['quantiles'][rows].reshape(-1,2,3)[:,:,1]
        signed = float(((delta[:,0]-delta[:,1])/2).mean())
        group['forecast_direction'] = 1 if signed >= 0 else -1
        group['discovery_mean_bidirectional_shift'] = signed
        np.savez_compressed(out/f"discovery_group_{group['group_id']}.npz", quantiles=swapped, pair_id=discovery['pair_id'][mask], units=np.array(group['units']))
    lock = {'timestamp_before_validation':utc(), 'groups':groups, 'discovery_sha256':sha(out/'discovery.npz'), 'protocol_sha256':sha(out/'protocol.json'), 'prior_run_hashes':prior_hashes}
    save_json(out/'group_lock.json', lock)
    lock_sha = sha(out/'group_lock.json')
    print('Six groups, their controls, and directions locked before fresh validation.', flush=True)
    validation = make_group_pairs('validation', protocol['validation_pairs_per_family'], protocol['context_length'])
    check_disjoint(validation, prior_run, other=discovery)
    contexts_v = flatten(validation)
    baseline_v = batched(adapter, contexts_v, protocol['batch_size'])
    save_baseline(out/'validation.npz', validation, baseline_v)
    acts_v = {k:v.reshape(-1,2,1408) for k,v in baseline_v['activations'].items()}
    total = len(groups)*(1+protocol['n_controls']) + sum(g['size'] for g in groups if g['size']==max(protocol['sizes']))
    completed = 0

    def execute(kind, filename, layer, units, family=None):
        nonlocal completed
        mask = np.ones(len(validation['pair_id']),dtype=bool) if family is None else validation['family']==family
        patch = {'layer':layer,'units':units,'values':donor_values(acts_v[layer], units, mask)}
        q = batched(adapter, contexts_v[np.repeat(mask,2)], protocol['batch_size'], patch=patch, capture=False)['quantiles'].reshape(-1,2,3)
        np.savez_compressed(out/filename, quantiles=q, pair_id=validation['pair_id'][mask], units=np.array(units))
        completed += 1
        record = {'kind':kind,'filename':filename,'completed':completed,'total':total,'elapsed_seconds':time.monotonic()-start,'timestamp':utc()}
        save_json(out/'progress.json',record)
        print(json.dumps(record),flush=True)
        return q

    joint, comparisons, singles = {}, {}, {}
    for group in groups:
        gid, layer = group['group_id'], group['layer']
        joint[gid] = execute('group',f'validation_group_{gid}.npz',layer,group['units'])
        comparisons[gid] = np.stack([execute('control',f'validation_control_{gid}_{i}.npz',layer,units) for i,units in enumerate(group['control_groups'])])
    for group in groups:
        if group['size'] != max(protocol['sizes']):
            continue
        for unit in group['units']:
            singles[group['family'],unit] = execute('singleton',f"validation_singleton_{group['family']}_{unit}.npz",group['layer'],[unit],family=group['family'])
    results, pvalues = [], []
    base = baseline_v['quantiles'].reshape(-1,2,3)
    for index, group in enumerate(groups):
        gid = group['group_id']
        mask = validation['family']==group['family']
        members = np.stack([singles[group['family'],u] for u in group['units']])
        primary = analyze_group(base[mask],joint[gid][mask],comparisons[gid][:,mask],members,
            group['forecast_direction'],floor=protocol['inference']['floor_log_variance'],seed=protocol['seeds']['bootstrap']+index)
        pvalues.extend([primary['influence_pvalue'],primary['advantage_pvalue'],primary['singleton_advantage_pvalue']])
        results.append({'group':group,'primary':primary,
            'descriptive_by_family_variant':describe_group(validation,base,joint[gid],acts_v,group)})
    corrected = holm_adjust(pvalues)
    for index,result in enumerate(results):
        p = result['primary']
        ordered = sum(p[key] for key in ('crossing_count','base_crossing_count','control_crossing_count','singleton_crossing_count'))==0
        for offset,name in enumerate(('influence','advantage','singleton')):
            p[name+'_holm_p'] = corrected[3*index+offset]
            p[name+'_pass'] = ordered and corrected[3*index+offset] < protocol['inference']['alpha']
        p['qualified'] = all(p[name+'_pass'] for name in ('influence','advantage','singleton'))
    if sha(out/'group_lock.json') != lock_sha:
        raise RuntimeError('Group lock changed after selection')
    save_json(out/'summary.json',{'finished_utc':utc(),'elapsed_seconds':time.monotonic()-start,
        'group_lock_sha256':lock_sha,'results':results,'limitations':protocol['inference']['limitations']})
    save_json(out/'manifest.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'})
    print('Group pilot complete: '+str(out/'summary.json'),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prior-run',type=Path,required=True)
    args = parser.parse_args()
    run(args.protocol,args.output,args.prior_run)


if __name__=='__main__':
    main()
