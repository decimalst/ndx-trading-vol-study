"""Post-result descriptive checks of the supplied critique, not a new trial.

No refitting, model selection, calibration adjustment or inferential resampling.
Original saved outputs and frozen reports are read only.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, multivariate_t, norm, t


def coordinates(y, mu, h):
    y, mu, h = [np.asarray(v, dtype=float) for v in (y, mu, h)]
    if y.shape != mu.shape or y.shape != h.shape or not all(np.isfinite(v).all() for v in (y,mu,h)) or (h <= 0).any():
        raise ValueError('Aligned finite observations, locations and positive variances required')
    return (y-mu)/np.sqrt(.75*h)


def describe(z, gain):
    z, gain = np.asarray(z, dtype=float), np.asarray(gain, dtype=float)
    if z.ndim != 2 or z.shape[1] != 2 or len(z) < 2 or gain.shape != (len(z),) or not np.isfinite(z).all() or not np.isfinite(gain).all():
        raise ValueError('Finite paired coordinates and aligned gains required')
    lo, hi = t.ppf([.025, .975], 8)
    assets = []
    for j in range(2):
        v = z[:, j]
        assets.append({'mean':float(v.mean()), 'variance_ddof0':float(v.var()),
            'second_moment':float(np.mean(v*v)), 'lower_tail_count':int(np.sum(v<lo)),
            'upper_tail_count':int(np.sum(v>hi)), 'lower_tail_rate':float(np.mean(v<lo)),
            'upper_tail_rate':float(np.mean(v>hi))})
    total = float(gain.sum())
    def subset(mask):
        value = float(gain[mask].sum())
        return {'n':int(mask.sum()),'fraction_of_days':float(mask.mean()),'gain_sum':value,
                'share_of_total_gain':value/total if total else None}
    trim = int(np.floor(.01*len(gain)))
    middle = np.sort(gain)[trim:len(gain)-trim]
    return {'n':len(z),'nominal_variance':8/6,'assets':assets,
        'squared_coordinate_correlation':float(np.corrcoef(z*z,rowvar=False)[0,1]),
        'mean_gain':float(gain.mean()),'central':subset((np.abs(z)<.53).all(axis=1)),
        'extreme':subset((np.abs(z)>4).any(axis=1)),
        'gain_trimmed_one_percent_each_end':float(middle.mean()),'trimmed_count_each_end':trim,
        'interpretation':'Post-result descriptions; no new significance test or mechanism identification.'}


def main():
    root = Path(__file__).resolve().parents[3]
    report = root/'reports/joint_copula/predictive'
    destination = Path(__file__).with_name('DIAGNOSTICS.json')
    if destination.exists(): raise ValueError('Preserve earlier descriptive audit')
    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    terminal = json.loads((report/'terminal.json').read_bytes())
    path = root/'data/model_memory_study/joint_copula_wave27/panel.parquet'
    expected = terminal['output_hashes'][str(path.relative_to(root))]
    if sha(path)!=expected: raise ValueError('Original panel changed')
    if sha(report/'metrics.json')!=terminal['metrics_sha256']: raise ValueError('Original metrics changed')
    data = pd.read_parquet(path)
    a = data.loc[data.model=='t8_copula'].sort_values('origin').reset_index(drop=True)
    b = data.loc[data.model=='gaussian_copula'].sort_values('origin').reset_index(drop=True)
    if not a.origin.equals(b.origin):raise ValueError('Identical paired origins required')
    for names in [['mu_qqq','mu_spx'],['h_qqq','h_spx'],['y_qqq','y_spx']]:
        if not np.array_equal(a[names].to_numpy(),b[names].to_numpy()):raise ValueError('Shared marginals and labels required')
    z = coordinates(a[['y_qqq','y_spx']],a[['mu_qqq','mu_spx']],a[['h_qqq','h_spx']])
    w = norm.ppf(t.cdf(z,8))
    if not np.isfinite(w).all():raise ValueError('This direct scipy audit cannot represent a saturated normal transform')
    log_t, log_g = [], []
    for pair, normal_pair, rt, rg in zip(z,w,a.rho,b.rho,strict=True):
        log_t.append(multivariate_t.logpdf(pair,shape=[[1.,rt],[rt,1.]],df=8)-t.logpdf(pair,8).sum())
        log_g.append(multivariate_normal.logpdf(normal_pair,cov=[[1.,rg],[rg,1.]])-norm.logpdf(normal_pair).sum())
    gain = np.asarray(log_t)-np.asarray(log_g)
    metrics = json.loads((report/'metrics.json').read_bytes())
    phases = {}
    for phase in metrics['rows'][0]['phases']:
        mask = a.phase.eq(phase['name']).to_numpy()
        if not np.isclose(-gain[mask].mean(),phase['mean'],atol=1e-10,rtol=1e-9):raise ValueError('Reconstructed contrast differs from saved result')
        phases[phase['name']] = describe(z[mask],gain[mask])
    out = {'status':'VERIFIED_DESCRIPTIVE_RECONSTRUCTION','scope':'Post-result critique audit; original trial and verdict unchanged; no additional registered hypothesis.',
        'panel_sha256':expected,'metrics_sha256':sha(report/'metrics.json'),
        'pooled':describe(z,gain),'phases':phases,
        'critique_thresholds':{'central_both_abs_less_than':.53,'extreme_any_abs_greater_than':4.,'tail_probability':.025},
        'simulation_reproduced':False,'simulation_limitation':'The supplied critique contains no simulation program, seeds, sample sizes or replication definition for its reported sd.',
        'source_array_or_fit_or_bootstrap_rerun':False}
    if sha(path)!=expected:raise ValueError('Panel changed during audit')
    with destination.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':out['status'],'pooled':out['pooled']},indent=2))


if __name__=='__main__':main()
