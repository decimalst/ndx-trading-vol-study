"""Independent saved-output audit; never imports producer modules or loads a model."""
import argparse
import hashlib
import importlib.util
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = RUN = OUTPUT = None
VERIFY_INSTALLED = False
EXECUTION_LOG = None
FAMILIES = ('persistence', 'recency', 'recovery')
checks = []


def load(name):
    return json.loads((RUN / name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(name, condition):
    checks.append({'check': name, 'passed': bool(condition)})
    if not condition:
        raise AssertionError(name)


def close(name, actual, expected, atol=1e-10, rtol=1e-8):
    check(name, np.allclose(actual, expected, atol=atol, rtol=rtol))


def cross(x):
    return int(np.any(np.diff(x, axis=-1) < 0, axis=-1).sum())


def tail(k, n):
    return sum(math.comb(n, j) for j in range(k, n + 1)) / 2**n


def main():
    protocol = load('protocol.json')
    pre = load('pre_execution.json')
    lock = load('candidate_lock.json')
    gate = load('instrumentation.json')
    meta = load('model_metadata.json')
    check('protocol equals pre-execution hash', sha(RUN / 'protocol.json') == pre['protocol_sha256'])
    for name, digest in pre['source_hashes'].items():
        check('frozen source: ' + name, sha(ROOT / name) == digest)
    check('discovery lock hash', sha(RUN / 'discovery.npz') == lock['discovery_sha256'])
    check('protocol lock hash', sha(RUN / 'protocol.json') == lock['protocol_sha256'])
    initial_path = OUTPUT.parent / 'initial_lock_receipt.json'
    if not initial_path.exists():
        initial_path.write_text(json.dumps({'observed_utc': datetime.now(UTC).isoformat(),
            'candidate_lock_sha256': sha(RUN / 'candidate_lock.json')}, indent=2) + '\n')
    check('candidate lock unchanged since independent first observation',
        json.loads(initial_path.read_text())['candidate_lock_sha256'] == sha(RUN / 'candidate_lock.json'))
    check('pinned revision', meta['checkpoint_revision'] == protocol['checkpoint_revision'])
    check('native model 0.2.1', meta['versions']['tirex-2'] == '0.2.1')
    check('zero based token endpoint mapping',
          (meta['token_count'] * 32 - 32 - 320) // 32 == protocol['token_index'] == 63)
    check('both augmentation paths disabled', not meta['tta_sign_flip'] and not meta['tta_diff'])
    if VERIFY_INSTALLED:
        spec = importlib.util.find_spec('tirex2')
        check('installed package available for optional source validation', spec is not None)
        package = Path(spec.origin).parent
        for name, digest in meta['installed_source_hashes'].items():
            check('installed source: ' + name, sha(package / name) == digest)

    discovery = np.load(RUN / 'discovery.npz', allow_pickle=False)
    validation = np.load(RUN / 'validation.npz', allow_pickle=False)
    norm = np.load(RUN / 'weight_norms.npz', allow_pickle=False)
    variant_counts = {}
    for split, data, expected_n in [('discovery', discovery, 36), ('validation', validation, 72)]:
        a, b = data['recipient'], data['donor']
        check(split + ' shape', a.shape == b.shape == (expected_n, 2048))
        check(split + ' finite histories', np.isfinite(a).all() and np.isfinite(b).all())
        check(split + ' exact paired multiset', np.array_equal(np.sort(a, axis=1), np.sort(b, axis=1)))
        check(split + ' exact terminal value', np.array_equal(a[:, -1], b[:, -1]))
        close(split + ' recipient mean', a.mean(axis=1), -9, atol=1e-12)
        close(split + ' donor mean', b.mean(axis=1), -9, atol=1e-12)
        close(split + ' recipient SD', a.std(axis=1), .6, atol=1e-12)
        close(split + ' donor SD', b.std(axis=1), .6, atol=1e-12)
        check(split + ' exact IDs', len(set(data['pair_id'])) == expected_n)
        check(split + ' unique contexts', len({row.tobytes() for row in np.concatenate([a, b])}) == 2 * expected_n)
        check(split + ' zero baseline crossings', cross(data['quantiles']) == 0)
        for family in FAMILIES:
            mask = data['family'] == family
            check(split + ' family count ' + family, int(mask.sum()) == expected_n // 3)
            if split == 'validation':
                variant_counts[family] = {v: int(np.count_nonzero(mask & (data['variant'] == v)))
                                         for v in ('short', 'long')}
    check('disjoint IDs', not set(discovery['pair_id']) & set(validation['pair_id']))
    dr = {row.tobytes() for row in np.concatenate([discovery['recipient'], discovery['donor']])}
    check('disjoint contexts', all(row.tobytes() not in dr for row in np.concatenate([validation['recipient'], validation['donor']])))

    raw = np.load(RUN / 'instrumentation.npz', allow_pickle=False)
    expected_rows = np.stack((discovery['recipient'], discovery['donor']), axis=1).reshape(-1, 2048)[[0, 1, 24, 25, 48, 49]]
    check('gate includes first pair of every family', np.array_equal(raw['contexts'], expected_rows))
    max_qerror, max_aerror = 0., 0.
    for entry in gate['checks']:
        name = entry['check']
        scale, offset = 1., 0.
        if name.startswith('affine_'):
            scale, offset = protocol['instrumentation']['positive_affine_transforms'][int(name[-1])]
        error = float(np.max(np.abs(raw[name + '_quantiles'] - (scale * raw['base_quantiles'] + offset))))
        close('gate qerror ' + name, error, entry['max_forecast_abs_error'], atol=1e-14)
        check('gate forecast threshold ' + name, error <= protocol['instrumentation']['max_forecast_abs_error'])
        max_qerror = max(max_qerror, error)
        for layer, reported in entry['activation_relative_errors'].items():
            suffix = 'residual' if layer == 'residual' else 'layer_' + layer
            old, new = raw['base_' + suffix], raw[name + '_' + suffix]
            err = float(np.max(np.abs(new - old)) / max(1., float(np.max(np.abs(old)))))
            close('gate activation error ' + name + '/' + layer, err, reported, atol=1e-14)
            check('gate activation threshold ' + name + '/' + layer, err <= protocol['instrumentation']['max_activation_relative_error'])
            max_aerror = max(max_aerror, err)
    check('instrumentation passed declaration', gate['passed'] and cross(raw['base_quantiles']) == 0)

    used = set()
    check('three candidates', len(lock['candidates']) == 3)
    for family, candidate in zip(FAMILIES, lock['candidates']):
        options = []
        family_mask = discovery['family'] == family
        for layer in protocol['layers']:
            data = discovery[f'layer_{layer}'].astype(float)
            sd = data.std(axis=(0, 1))
            delta = np.mean(data[family_mask, 1] - data[family_mask, 0], axis=0)
            weights = norm[f'layer_{layer}']
            for unit in range(data.shape[-1]):
                if sd[unit] > 0 and weights[unit] > 0 and (layer, unit) not in used:
                    options.append((-abs(delta[unit] / sd[unit]), layer, unit, float(delta[unit] / sd[unit]), sd[unit]))
        _, layer, unit, effect, scale = min(options)
        check('winner ' + family, (candidate['layer'], candidate['unit'], candidate['family']) == (layer, unit, family))
        check('zero based candidate limits ' + family, 0 <= layer < 12 and 0 <= unit < 1408)
        close('winner effect ' + family, candidate['standardized_effect'], effect)
        close('winner scale ' + family, candidate['activation_scale'], scale)
        check('activation direction ' + family, candidate['orientation'] == (1 if effect >= 0 else -1))
        used.add((layer, unit))
        acts = discovery[f'layer_{layer}'].astype(float)
        scales, weights = acts.std(axis=(0, 1)), norm[f'layer_{layer}']
        distances = [(math.log(scales[j] / scales[unit])**2 + math.log(weights[j] / weights[unit])**2, j)
                     for j in range(1408) if j != unit and scales[j] > 0 and weights[j] > 0]
        check('control identities ' + family, candidate['controls'] == [j for _, j in sorted(distances)[:4]])
        swap = np.load(RUN / f'discovery_swap_L{layer}_U{unit}.npz', allow_pickle=False)
        check('discovery swap ID alignment ' + family, np.array_equal(swap['pair_id'], discovery['pair_id'][family_mask]))
        delta = swap['quantiles'].astype(float)[:, :, 1] - discovery['quantiles'][family_mask, :, 1].astype(float)
        effect = np.mean((delta[:, 0] - delta[:, 1]) / 2)
        close('discovery swap direction mean ' + family, effect, candidate['discovery_mean_bidirectional_shift'], atol=1e-10)
        check('forecast direction frozen from discovery ' + family, candidate['forecast_direction'] == (1 if effect >= 0 else -1))

    report = {'status': 'PARTIAL_PASS', 'checked_utc': datetime.now(UTC).isoformat(),
              'max_gate_forecast_abs_error': max_qerror, 'max_gate_activation_relative_error': max_aerror,
              'validation_variant_counts': variant_counts, 'candidate_lock_sha256': sha(RUN / 'candidate_lock.json'),
              'check_count': len(checks), 'checks': checks}
    if (RUN / 'summary.json').exists():
        summary = load('summary.json')
        timestamps = [datetime.fromisoformat(pre['timestamp_before_model_load']),
                      datetime.fromisoformat(gate['timestamp']),
                      datetime.fromisoformat(lock['timestamp_before_validation']),
                      datetime.fromisoformat(load('progress.json')['timestamp']),
                      datetime.fromisoformat(summary['finished_utc'])]
        check('recorded execution chronology', timestamps[0] < timestamps[1] <= timestamps[2]
              < timestamps[3] <= timestamps[4])
        if EXECUTION_LOG is not None:
            progress_rows = [json.loads(line) for line in EXECUTION_LOG.read_text().splitlines()
                             if line.startswith('{"completed_interventions"')]
            expected_units = sorted({(c['layer'], u) for c in lock['candidates']
                                     for u in [c['unit'], *c['controls']]})
            check('execution log covers every intervention exactly once',
                  [(p['layer'], p['unit']) for p in progress_rows] == expected_units)
            check('execution log counters', [p['completed_interventions'] for p in progress_rows]
                  == list(range(1, len(expected_units) + 1)))
            progress_times = [datetime.fromisoformat(p['timestamp']) for p in progress_rows]
            check('first validation intervention completed after candidate freeze',
                  timestamps[2] < progress_times[0])
            check('all intervention completions ordered before finish',
                  all(a < b for a, b in zip(progress_times, progress_times[1:]))
                  and progress_times[-1] <= timestamps[-1])
            report['execution_log_sha256'] = sha(EXECUTION_LOG)
        # Per-intervention arrays have no individual creation receipt. The final
        # progress timestamp is independently ordered; the producer source shows
        # validation generation/forecasting after the lock, but timestamps alone
        # do not certify the exact time of the first validation forecast.
        check('summary candidate lock hash', summary['candidate_lock_sha256'] == sha(RUN / 'candidate_lock.json'))
        pvalues, independently_calculated = [], []
        for i, result in enumerate(summary['results']):
            c = lock['candidates'][i]
            check('summary candidate equals lock ' + c['family'], result['candidate'] == c)
            mask = validation['family'] == c['family']
            baseline = validation['quantiles'][mask].astype(float)
            def read_swap(unit, layer=c['layer'], selection=mask):
                a = np.load(RUN / f'validation_swap_L{layer}_U{unit}.npz', allow_pickle=False)
                check(f'validation IDs L{layer} U{unit}', np.array_equal(a['pair_id'], validation['pair_id']))
                return a['quantiles'].astype(float)[selection]
            swap = read_swap(c['unit'])
            controls = np.asarray([read_swap(j) for j in c['controls']])
            delta = swap[:, :, 1] - baseline[:, :, 1]
            ctrl = controls[:, :, :, 1] - baseline[None, :, :, 1]
            signed = c['forecast_direction'] * (delta[:, 0] - delta[:, 1]) / 2
            advantage = np.mean(abs(delta), axis=1) - np.mean(abs(ctrl), axis=(0, 2))
            primary = result['primary']
            check(c['family'] + ' bootstrap count', primary['bootstrap_replicates'] == 5000)
            check(c['family'] + ' bootstrap seed', primary['seed'] == protocol['seeds']['bootstrap'] + i)
            sampled = np.random.default_rng(protocol['seeds']['bootstrap'] + i).integers(0, 24, (5000, 24))
            for label, effect in [('influence', signed), ('advantage', advantage)]:
                successes = int(np.count_nonzero(effect > protocol['inference']['floor_log_variance']))
                p = tail(successes, len(effect))
                pvalues.append(p)
                close(c['family'] + '/' + label + ' effects', effect, primary[label + '_effects'])
                close(c['family'] + '/' + label + ' mean', np.mean(effect), primary['mean_' + label + '_effect'])
                check(c['family'] + '/' + label + ' successes', successes == primary[label + '_successes'])
                close(c['family'] + '/' + label + ' binomial p', p, primary[label + '_pvalue'], atol=1e-15)
                ci = np.quantile(effect[sampled].mean(axis=1), [.025, .975])
                close(c['family'] + '/' + label + ' bootstrap interval', ci, primary[label + '_mean_ci95'])
            check(c['family'] + ' raw pair count', primary['n_pairs'] == len(signed) == 24)
            check(c['family'] + ' crossing counts', (cross(baseline), cross(swap), cross(controls)) ==
                  (primary['base_crossing_count'], primary['crossing_count'], primary['control_crossing_count']))
            close(c['family'] + ' recipient shift', delta[:, 0].mean(), primary['mean_recipient_median_shift'])
            close(c['family'] + ' donor shift', delta[:, 1].mean(), primary['mean_donor_median_shift'])
            close(c['family'] + ' absolute median shift', abs(delta).mean(), primary['mean_abs_median_shift'])
            close(c['family'] + ' native forecast contrast', np.mean(baseline[:, 1, 1] - baseline[:, 0, 1]),
                  primary['mean_native_donor_recipient_contrast'])
            close(c['family'] + ' both swap directions',
                  np.mean((c['forecast_direction'] * delta[:, 0] > 0) & (-c['forecast_direction'] * delta[:, 1] > 0)),
                  primary['both_directions_expected_sign_fraction'])
            close(c['family'] + ' control absolute median shift', abs(ctrl).mean(), primary['mean_abs_control_median_shift'])
            widths = (swap[:, :, 2] - swap[:, :, 0]) - (baseline[:, :, 2] - baseline[:, :, 0])
            close(c['family'] + ' width shifts', abs(widths).mean(), primary['mean_abs_width_shift'])
            ctrl_widths = (controls[:, :, :, 2] - controls[:, :, :, 0]) - (baseline[None, :, :, 2] - baseline[None, :, :, 0])
            close(c['family'] + ' control width shifts', abs(ctrl_widths).mean(), primary['mean_abs_control_width_shift'])
            check(c['family'] + ' count/direction/floor', primary['n_controls'] == 4
                  and primary['direction'] == c['forecast_direction']
                  and primary['floor'] == protocol['inference']['floor_log_variance'])

            # All 3 families and both duration variants, independently recomputed
            # in float64. Producer's descriptive-only means originate in float32.
            all_swap = np.load(RUN / f'validation_swap_L{c["layer"]}_U{c["unit"]}.npz', allow_pickle=False)['quantiles'].astype(float)
            all_base = validation['quantiles'].astype(float)
            wanted_groups = set()
            for family in FAMILIES:
                for variant in ('all', 'short', 'long'):
                    select = validation['family'] == family
                    if variant != 'all':
                        select &= validation['variant'] == variant
                    if not select.any():
                        continue
                    key = family + '/' + variant
                    wanted_groups.add(key)
                    desc = result['descriptive_by_family_variant'][key]
                    b, s = all_base[select], all_swap[select]
                    change = s[:, :, 1] - b[:, :, 1]
                    contrast = b[:, 1, 1] - b[:, 0, 1]
                    eligible = abs(contrast) > 1e-8
                    signs = np.sign(contrast[eligible])
                    oriented = np.column_stack((change[eligible, 0] * signs, -change[eligible, 1] * signs))
                    width_change = (s[:, :, 2] - s[:, :, 0]) - (b[:, :, 2] - b[:, :, 0])
                    expected = {
                        'mean_signed_median_shift': float(oriented.mean()) if eligible.any() else None,
                        'mean_abs_median_shift': float(abs(change).mean()),
                        'mean_abs_width_shift': float(abs(width_change).mean()),
                        'directional_agreement': float((oriented > 0).mean()) if eligible.any() else None,
                        'both_direction_effect': float(np.all(oriented > 0, axis=1).mean()) if eligible.any() else None,
                        'near_zero_contrast_count': int((~eligible).sum()),
                        'directional_pair_count': int(eligible.sum()),
                        'n_pairs': int(select.sum()),
                        'frozen_direction_mean_shift': float(c['forecast_direction'] * np.mean((change[:, 0] - change[:, 1]) / 2)),
                    }
                    check(c['family'] + ' descriptive field inventory ' + key, set(desc) == set(expected))
                    for field, value in expected.items():
                        name = c['family'] + ' descriptive ' + key + '/' + field
                        if value is None:
                            check(name, desc[field] is None)
                        else:
                            close(name, value, desc[field], atol=2e-9, rtol=2e-7)
            check(c['family'] + ' descriptive group inventory', set(result['descriptive_by_family_variant']) == wanted_groups)
            independent = {'family': c['family'], 'layer': c['layer'], 'unit': c['unit'],
                'influence_successes': int(np.sum(signed > .001)), 'advantage_successes': int(np.sum(advantage > .001)),
                'mean_influence': float(signed.mean()), 'mean_advantage': float(advantage.mean()),
                'mean_abs_median_shift': float(abs(delta).mean()), 'mean_abs_width_shift': float(abs(widths).mean())}
            independently_calculated.append(independent)
        order = sorted(range(6), key=pvalues.__getitem__)
        adjusted = [0.] * 6
        running = 0.
        for rank, index in enumerate(order):
            running = max(running, min(1., pvalues[index] * (6 - rank)))
            adjusted[index] = running
        for i, result in enumerate(summary['results']):
            p = result['primary']
            ordered = p['base_crossing_count'] + p['crossing_count'] + p['control_crossing_count'] == 0
            for j, label in enumerate(('influence', 'advantage')):
                close(result['candidate']['family'] + '/' + label + ' Holm6', adjusted[2*i+j], p[label + '_holm_p'])
                check(result['candidate']['family'] + '/' + label + ' final pass', p[label + '_pass'] == (ordered and adjusted[2*i+j] < .05))
        manifest = load('manifest.json')
        for name, digest in manifest.items():
            check('run manifest: ' + name, sha(RUN / name) == digest)
        report.update(status='PASS', primary=independently_calculated, raw_pvalues=pvalues, holm6_pvalues=adjusted)
    report['check_count'] = len(checks)
    report['checks'] = checks
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'checks'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-installed-package', action='store_true')
    parser.add_argument('--execution-log', type=Path)
    args = parser.parse_args()
    ROOT, RUN = args.source_root.resolve(), args.run_dir.resolve()
    OUTPUT = args.output.resolve() if args.output else RUN / 'independent_verification.json'
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    VERIFY_INSTALLED = args.verify_installed_package
    EXECUTION_LOG = args.execution_log
    main()
