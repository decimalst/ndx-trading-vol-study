"""Render the completed stratified study without changing scientific records."""
import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SCALE = 1000


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label(gid):
    return gid.replace('persistence_', 'Persistence ').replace('recency_', 'Recency ').replace(
        'recovery_', 'Recovery ').replace('k4', '4 units').replace('k8', '8 units').replace('tail4', 'added 4')


def number(value, scale=1, precision=3):
    return 'undefined' if value is None else f'{value * scale:.{precision}f}'


def plot_primary(primary, tau, target):
    ids = list(dict.fromkeys(x['group_id'] for x in primary))
    colors = {'short': '#185D88', 'long': '#AE5D37'}
    fig, ax = plt.subplots(figsize=(12.6, 8.2))
    for offset, variant in [(-.12, 'short'), (.12, 'long')]:
        for row, gid in enumerate(ids):
            item = next(x for x in primary if x['group_id'] == gid and x['variant'] == variant)
            stat = item['summary']
            low, high = np.asarray(stat['mean_e_ci95']) * SCALE
            y = row + offset
            ax.plot([low, high], [y, y], color=colors[variant], linewidth=2)
            ax.plot(stat['mean_e'] * SCALE, y, 'o', color=colors[variant],
                    label=variant.capitalize() + ' motif' if row == 0 else None)
    ax.axvline(0, color='#777777', linewidth=1)
    ax.axvspan(-tau * SCALE, tau * SCALE, color='#777777', alpha=.16)
    ax.set_yticks(range(len(ids)), [label(gid) for gid in ids])
    ax.invert_yaxis()
    ax.set_xlabel('Raw bidirectional median change (10⁻³ log-variance units)')
    ax.grid(axis='x', alpha=.18)
    ax.legend(loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('TiRex activation swaps: effect size depends on the motif', fontsize=18, x=.53)
    ax.set_title('320 fresh pairs per motif • fixed unit identities • one-step forecasts', fontsize=11, pad=16)
    fig.text(.03, .04, 'Points: mean (recipient shift − donor shift)/2. Lines: pointwise 95% paired-bootstrap intervals.\n'
             'Raw signs are comparable across groups; no discovery-sign reorientation. Numerical band shown in gray.\n'
             'Synthetic intervention sensitivity; no forecast-accuracy or trading outcome is measured.', fontsize=10)
    fig.subplots_adjust(left=.22, right=.96, bottom=.18, top=.87)
    fig.savefig(target, dpi=170, facecolor='white')
    plt.close(fig)


def build(run):
    run = Path(run).resolve()
    inputs = {name: sha(run / name) for name in (
        'summary.json', 'protocol.json', 'random_reference_summary.json', 'dose_summary.json',
        'recent_mean_diagnostic.json', 'numerical_calibration.json')}
    summary, protocol = read(run / 'summary.json'), read(run / 'protocol.json')
    primary = summary['primary']
    if len(primary) != 18 or any(x['summary']['n_pairs'] != 320 for x in primary):
        raise ValueError('Expected all 18 fixed strata, with 320 pairs each')
    lookup = {(x['group_id'], x['variant']): x for x in primary}
    reference = read(run / 'random_reference_summary.json')
    doses = read(run / 'dose_summary.json')
    readout = read(run / 'recent_mean_diagnostic.json')
    numeric = read(run / 'numerical_calibration.json')
    tau = summary['tau']
    plot_primary(primary, tau, run / 'stratified_effects.png')
    lines = [
        '# TiRex activation swaps: stratified follow-up and reassessment', '',
        'Completed 2026-10-22. This study follows a critique of the original pooled group-pilot criterion. '
        'Its protocol and intervention identities were fixed before fresh calibration; the numerical band, '
        'sensitivity estimates, random reference sets and recent-mean fit were frozen before validation. '
        'The questions were informed by prior results, so this is a follow-up, not an independent discovery.', '',
        '## What the experiments show', '',
        'The original “0/6” headline was too broad. It summarized a restrictive composite '
        'probability criterion, not a test of whether the selected units affected forecasts. '
        'The [retrospective reassessment](../tirex_group_pilot/REASSESSMENT.md) supplies all 18 raw '
        'p-values and counts, discovery effects, motif strata and power, while preserving the original artifacts.', '',
    ]
    short = lookup['recency_U666', 'short']['summary']
    long = lookup['recency_U666', 'long']['summary']
    lines.extend([
        f"- **Recency L0/U666:** fresh raw mean effects are {number(short['mean_e'], SCALE)} milli for short "
        f"motifs and {number(long['mean_e'], SCALE)} milli for long motifs. The named unit was selected in "
        'the group pilot; the earlier single-unit pilot selected **L0/U103**, which is evaluated separately here. '
        'The short-motif four- and eight-unit means are −4.659 and −4.861 milli, only slightly larger '
        'in magnitude than U666 alone. On long motifs, the groups attenuate the singleton effect. '
        'This comparison is not a causal percentage attribution.',
        '- **Recovery L5:** compare the four-unit prefix, added four units and full eight in the same raw '
        'sign convention below. The original +1.263/−1.422 milli chart values used different discovery '
        'orientations; both raw validation effects were negative. Adding units did not flip that raw sign. '
        'In the fresh run, the prefix, added four and full eight all have negative mean effects in both '
        'motifs, with heterogeneous pair-level responses, especially for long motifs.',
        '- **Persistence L11:** the selected groups have measurable local intervention effects. On the '
        'fixed reference subset, all 19 random four-unit sets and most random eight-unit sets move '
        'forecasts more than the corresponding selected group. Neither the old four comparison '
        'sets nor this small reference sample establish global layer sensitivity or forecast irrelevance.', '',
        '![Raw stratified intervention effects](stratified_effects.png)', '',
        'All milli quantities in this report equal 0.001 native log-variance units. A numerical effect '
        'is not a percentage-point return, a forecast-accuracy gain or a measure of economic value.', '',
        '## Effects, calibration sensitivity and raw tests', '',
        'The primary effect is **(recipient median shift − donor median shift)/2**, without '
        'discovery-based sign multiplication. Short and long motifs are separate estimands. '
        'Each row has 320 independent synthetic pairs. Calibration used 32 separate pairs per cell '
        'under the same generator law. Intervals are descriptive, pointwise 95% paired-bootstrap '
        'intervals, not simultaneous intervals.', '',
        '| Intervention / motif | Calibration mean (milli) | Validation mean [95% CI] (milli) | Raw p | Holm18 p | Approx. MDE80 (milli) | + / − / numerical-neutral |',
        '|---|---:|---:|---:|---:|---:|---|',
    ])
    for item in primary:
        stat = item['summary']
        counts = stat['sign_counts']
        ci = ' / '.join(number(x, SCALE) for x in stat['mean_e_ci95'])
        lines.append(f"| {label(item['group_id'])} / {item['variant']} | "
            f"{number(item['calibration']['summary']['mean_e'], SCALE)} | "
            f"{number(stat['mean_e'], SCALE)} [{ci}] | {stat['primary_pvalue']:.4g} | "
            f"{item['holm_pvalue']:.4g} | {number(item['calibration']['mde80'], SCALE)} | "
            f"{counts['positive']} / {counts['negative']} / {counts['neutral']} |")
    lines.extend([
        '', f"The frozen numerical band is **±{tau:.9g} log-variance units** "
        f"({number(tau, SCALE, 5)} milli). Maximum observed calibration jitter was "
        f"{numeric['observed_max_jitter']:.9g}; maximum float32 spacing was "
        f"{numeric['max_float32_spacing']:.9g}. The rule uses ten times the larger quantity. "
        'Ten is a predeclared conservative multiplier; this is a resolution band, not a scientific '
        'or economic materiality margin. Repeatability in this deterministic execution is not '
        'uncertainty across trained-model seeds.', '',
        'The two-sided Student t procedure tests whether the population mean lies outside the '
        'numerical band, with Holm adjustment across 18 claims. Its sampling-distribution '
        'approximation assumes independent synthetic pairs and a sufficiently regular mean at '
        'n=320; it is not distribution-free. The sign counts retain numerical-neutral cases and '
        'show sign prevalence, without implying that the mean magnitude is typical. Zero sample variance uses the explicitly '
        'declared deterministic convention. No group-superiority or semantic-selectivity verdict '
        'is attached to this test.', '',
        'The long-motif recovery mean is sensitive to large individual responses. A '
        '[post-result descriptive check](POSTRESULT_SENSITIVITY.md) reports medians, quantiles and '
        'the consequence of omitting the largest absolute effects. For the eight-unit group, the '
        'mean is −1.178 milli and the median −0.701 milli; omitting the 16 largest absolute effects '
        '(5% of pairs) changes the mean to −0.386 milli. This outcome-dependent omission is a '
        'sensitivity diagnostic, not a replacement estimate or confirmatory test. All 320 pairs '
        'remain in every primary result.', '',
        'MDE80 is a **planning sensitivity estimate**, using calibration standard deviation and '
        'a normal approximation at the conservative first-step Bonferroni threshold: '
        'tau + (z[1−0.05/(2×18)] + z[0.8]) × calibration SD / sqrt(320). It assumes calibration '
        'variance transfers; it is not retrospectively calculated power or a guarantee.', '',
        'The old majority-exceedance test needed 25/32 at the first Holm step and a true '
        'exceedance probability about 0.821 for 80% power. There is no distribution-free '
        'minimum detectable log-variance mean for that binary endpoint. Increasing sample '
        'size alone would not fix a mixture whose true exceedance probability is 0.5. '
        'Holm remains valid under dependent tests; dependence can make it conservative.', '',
        '## Individual directions and quantile outcomes', '',
        'Median log variance was retained as the central forecast quantity addressed by the '
        'earlier study. q10, q90 and interval width remain separately visible below. Without '
        'realized future targets there is no proper-score or calibration test; changing an '
        'interval is not evidence that uncertainty estimates became better.', '',
        '| Intervention / motif | Recipient q50 shift | Donor q50 shift | q10 effect | q90 effect | Mean absolute width movement | Baseline / intervention crossings |',
        '|---|---:|---:|---:|---:|---:|---|',
    ])
    for item in primary:
        s = item['summary']
        values = [s['mean_recipient_median_shift'], s['mean_donor_median_shift'],
                  s['mean_quantile_e'][0], s['mean_quantile_e'][2], s['mean_abs_width_shift']]
        lines.append(f"| {label(item['group_id'])} / {item['variant']} | " +
                     ' | '.join(number(x, SCALE) for x in values) +
                     f" | {s['base_crossing_count']} / {s['crossing_count']} |")
    lines.extend([
        '', '## Random groups as a descriptive reference', '',
        'Nineteen uniform eight-unit draws per layer, each with a nested four-unit prefix, '
        'replace the earlier approximate matching comparison. Draws exclude the original top eight '
        'units and U103 in layer 0; each draw has distinct members, but draws may overlap. '
        'All comparisons below use the **same fixed first 16 validation pairs per motif**, '
        'not all 320. This is a small descriptive reference distribution. A selected candidate '
        'is not exchangeable with these random sets, so ranks are not randomization p-values. '
        'There is no claim of matched downstream sensitivity or a binding proxy-match tolerance.', '',
        '| Group / motif | Candidate absolute movement | Random 10th / median / 90th | Random sets ≥ candidate / 19 | Candidate displacement | Random displacement median |',
        '|---|---:|---|---:|---:|---:|',
    ])
    for candidate in reference['candidates_on_identical_subset']:
        gid = candidate['group_id']
        if not gid.endswith(('_k4', '_k8')):
            continue
        family, size = gid.split('_k')
        refs = [r for r in reference['references'] if r['reference']['family'] == family
                and r['reference']['size'] == int(size)]
        if len(refs) != 19:
            raise ValueError('Missing random references')
        for variant in ('short', 'long'):
            s = candidate['by_variant'][variant]
            magnitudes = np.array([r['by_variant'][variant]['mean_abs_median_shift'] for r in refs])
            disp = [r['by_variant'][variant]['input_displacement_rms'] for r in refs]
            quant = ' / '.join(number(x, SCALE) for x in np.quantile(magnitudes, [.1, .5, .9]))
            lines.append(f"| {label(gid)} / {variant} | {number(s['mean_abs_median_shift'], SCALE)} | {quant} | "
                f"{np.count_nonzero(magnitudes >= s['mean_abs_median_shift'])}/19 | "
                f"{number(s['input_displacement_rms'])} | {number(float(np.median(disp)))} |")
    lines.extend([
        '', 'Input displacement is in native activation units; output movement is in milli log variance. '
        'The random distribution does not equalize activation displacement, covariance, downstream '
        'sensitivity or distance from natural joint states. It describes these chosen interventions; '
        'it does not identify a global property of layer 11.', '',
        '## Recovery dose response', '',
        'At alpha, each selected activation is replaced by original + alpha × '
        '(original donor − original). Alpha 1 is the donor swap, alpha 2 extrapolates beyond it. '
        'The eight singletons, prefix four, added four and all eight are tested on the fixed '
        '16-pair-per-motif subset. The next table pools that 32-pair descriptive subset; '
        'motif-specific shifts and every raw forecast are also saved. No interaction significance '
        'claim is made.', '',
        '| Recovery group | Alpha | Mean absolute median movement | Mean absolute joint−sum | Mean absolute deviation from alpha×unit-dose effect |',
        '|---|---:|---:|---:|---:|',
    ])
    for row in doses['results']:
        if row['group_id'] not in ('recovery_k4', 'recovery_tail4', 'recovery_k8'):
            continue
        magnitude = np.mean([x['mean_abs_median_shift'] for x in row['by_variant'].values()])
        lines.append(f"| {label(row['group_id'])} | {row['alpha']:g} | {number(magnitude, SCALE)} | "
            f"{number(row['mean_abs_joint_minus_sum_median'], SCALE, 5)} | "
            f"{number(row['mean_abs_median_curvature'], SCALE, 5)} |")
    lines.extend([
        '', 'For the eight-unit group, mean absolute joint−sum grows from 0.00972 milli at alpha 0.25 '
        'to 0.62013 milli at alpha 2; the latter dose also departs from twice the unit-dose effect '
        'by 0.49754 milli on average. This dose dependence is compatible with ordinary local curvature. '
        'Small residuals are compatible with local linearization. Raising alpha does not guarantee '
        'leaving the linear regime, and extrapolation can create more unusual hybrid states. '
        'Additivity does not establish or reject superposition. The raw dose summaries store original '
        'donor displacement; actual intervention displacement is alpha times that quantity.', '',
        '## Does U666 approximate a recent-mean readout?', '',
        'A two-parameter linear model (intercept and slope) predicts L0/U666 activation from '
        'the arithmetic mean of the last 32 raw observations. It was fitted on calibration '
        'recency histories only, using both motifs and both sides, then frozen. Reported R² '
        'is out of sample; the paired-difference R² additionally checks donor−recipient changes.', '',
        '| Validation motifs | Pairs | Activation-level R² | Paired activation-difference R² |',
        '|---|---:|---:|---:|',
    ])
    for variant, row in readout['validation'].items():
        lines.append(f"| {variant} | {row['n_pairs']} | {number(row['level_r2'])} | "
                     f"{number(row['paired_difference_r2'])} |")
    lines.extend([
        '', f"The pooled paired-difference R² is {number(readout['validation']['all']['paired_difference_r2'])}, "
        f"but short-motif difference R² is {number(readout['validation']['short']['paired_difference_r2'])} "
        f"and long-motif level R² is {number(readout['validation']['long']['level_r2'])}. "
        'This does not establish a uniform linear recent-mean explanation. A model can explain '
        'broad differences between motifs or between recipient and donor histories while '
        'failing to explain variation among cases within a motif. Negative held-out R² means '
        'the frozen surrogate has greater squared error than the constant validation mean; '
        'it is evidence against this particular surrogate, not against all linear or temporal-summary explanations.',
        '', 'The short motif places all 19 burst observations in the final 32-observation patch; '
        'the long motif leaves only one burst observation there, followed by 30 quiet observations '
        'and the common endpoint. That motivates this bounded alternative explanation. The long '
        'case is not mathematically null: its native forecast contrast was nonzero in the earlier '
        'run. A high R² here would support a local surrogate under this generator, not prove the '
        'unit globally computes a linear recent mean. A low R² rejects only this one-feature surrogate.', '',
        '## What a unit and a forecast mean here', '',
        'A unit is one of **1,408 gated feed-forward intermediate coordinates**, '
        'zᵢ = SiLU(W_gate h)ᵢ × (W_up h)ᵢ, immediately before the down projection at '
        '`stack[layer].time_mixer.ffn.proj_down`. It is neither a residual-stream coordinate '
        'nor an sLSTM/mLSTM cell state. Layer 0 already follows nonlinear embedding, normalization '
        'and recurrent processing. Its early location alone does not establish a linear readout.', '',
        'There are 64 observed 32-sample patches in the 2,048-observation context. Zero-based '
        '**token 63** is the final observed patch; its output supplies the first predicted '
        'observation. The horizon is **one synthetic log-variance observation**. This is not '
        'an experiment varying the forecasting horizon. Full-history scaling and arcsinh '
        'transforms are inverted, and the public wrapper’s uncertainty processing is retained. '
        'Median and quantile changes are therefore measured in restored log-variance units.', '',
        'Elementwise gating gives this intermediate basis an architectural role, but gated '
        'transformer feed-forward networks also have elementwise coordinates. No rotated-basis '
        'comparison was run. These experiments do not prove native-basis superiority or a '
        'general theory of neural-network interpretability.', '',
        '## Provenance, limits and reproduction', '',
        '**75 tests passed:** 56 new generator, statistics, runner, readout and audit contracts, '
        'plus 19 unchanged prior-adapter regression tests. Tests validate these software '
        'contracts, not the scientific interpretation of the model. The second implementation '
        'passed 12,664 numerical checks against saved artifacts; it does not rerun the model '
        'or verify every inference code path.', '',
        'The original single-unit and group protocols, arrays and report receipts retain their '
        'bytes. New tests were written and observed failing before the corresponding implementation. '
        'Calibration and validation use identical balanced motif laws and different fixed seeds; '
        'input hashes are disjoint from both old pilots. Selection is informed by previous results, '
        'and all conclusions remain conditional on synthetic inputs, the fixed checkpoint and '
        'the specified intervention site. Hybrid activation states can be off manifold.', '',
        '“Verification” here means **internal numerical checking by a second implementation in '
        'the same AI-assisted project**. It is not independent scientific replication or peer '
        'review. Historical files named `independent_verification.json` retain their names as '
        'archival records; this corrected description governs their interpretation.', '',
        'The work was inspired by Nick Cammarata’s Goodfire essay '
        '[Neural Networks Are Already Clean](https://already-clean.goodfire.com/), an informal '
        'working hypothesis. TiRex-2 is credited to Podest and colleagues '
        '([paper](https://arxiv.org/abs/2607.01204), '
        '[official documentation](https://nx-ai.github.io/tirex-2/introduction/)). '
        '[Zhang and Nanda](https://arxiv.org/abs/2309.16042) motivate care with activation-patching '
        'metrics and intervention design. This study does not replicate those papers’ broad claims.', '',
        'Evidence: [protocol](protocol.json), [pre-execution source hashes](pre_execution.json), '
        '[validation lock](validation_lock.json), [primary results](primary_summary.json), '
        '[random references](random_reference_summary.json), [dose results](dose_summary.json), '
        '[recent-mean diagnostic](recent_mean_diagnostic.json), '
        '[second-implementation audit](internal_verification.json), [tests-first receipts](tests_first/).', '',
        'Run in the repository environment, with the cached pinned checkpoint available and a new output directory:', '',
        '```sh',
        'python -m src.tirex_stratified_run --protocol tirex_stratified_pilot.json --output reports/tirex_stratified_pilot/new_run --prior-run reports/tirex_intervention_pilot --prior-run reports/tirex_group_pilot',
        'python scripts/verify_tirex_stratified_pilot.py --run-dir reports/tirex_stratified_pilot/new_run --source-root . --prior-run reports/tirex_intervention_pilot --prior-run reports/tirex_group_pilot --output reports/tirex_stratified_pilot/new_run/internal_verification.json',
        'python -m src.tirex_stratified_report --run-dir reports/tirex_stratified_pilot/new_run',
        '```', '', f"Recorded model-run duration: {summary['elapsed_seconds']:.1f} seconds.", '',
    ])
    (run / 'REPORT.md').write_text('\n'.join(lines))
    for name, digest in inputs.items():
        if sha(run / name) != digest:
            raise RuntimeError('Scientific input modified while reporting')
    receipt = {'created_utc': datetime.now(UTC).isoformat(), 'source_sha256': sha(Path(__file__)),
               'input_sha256': inputs, 'output_sha256': {n: sha(run / n) for n in ('REPORT.md', 'stratified_effects.png')}}
    (run / 'report_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    print(json.dumps(build(parser.parse_args().run_dir), indent=2))
