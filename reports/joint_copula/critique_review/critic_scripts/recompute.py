import json, numpy as np, pandas as pd
from scipy.stats import t as student, norm, multivariate_t, multivariate_normal
D='data/model_memory_study/joint_copula_wave27/'
panel=pd.read_parquet(D+'panel.parquet'); cov=pd.read_parquet(D+'coverage.parquet'); apps=pd.read_parquet(D+'applications.parquet')
m=json.load(open('reports/joint_copula/predictive/metrics.json'))
print(panel.shape, panel.model.value_counts().to_dict())
NU=8
def loss(df):
    y=df[['y_qqq','y_spx']].to_numpy(); mu=df[['mu_qqq','mu_spx']].to_numpy(); h=df[['h_qqq','h_spx']].to_numpy(); rho=df.rho.to_numpy()
    s=np.sqrt(0.75*h); z=(y-mu)/s
    marg=student.logpdf(y,NU,loc=mu,scale=s).sum(1)
    out=np.empty(len(df))
    for i in range(len(df)):
        fam=df.model.iloc[i]
        if fam=='independence': out[i]=marg[i]
        elif fam=='t8_copula':
            R=np.array([[1,rho[i]],[rho[i],1]]); S=np.diag(s[i])@R@np.diag(s[i])
            out[i]=multivariate_t.logpdf(y[i],loc=mu[i],shape=S,df=NU)
        else:
            u=student.cdf(z[i],NU); w=norm.ppf(u)
            R=np.array([[1,rho[i]],[rho[i],1]])
            out[i]=multivariate_normal.logpdf(w,cov=R)-norm.logpdf(w).sum()+marg[i]
    return -out
panel['loss_ind']=loss(panel)
P={k:panel[panel.model==k].set_index('origin').sort_index() for k in ['t8_copula','gaussian_copula','independence']}
assert P['t8_copula'].index.equals(P['gaussian_copula'].index) and P['t8_copula'].index.equals(P['independence'].index)
cand=P['t8_copula']
print("phases:", cand.phase.value_counts().to_dict())
print("rho_t8 range",cand.rho.min(),cand.rho.max()," rho_G range",P['gaussian_copula'].rho.min(),P['gaussian_copula'].rho.max())
print("rho_t8 - rho_G mean",(cand.rho-P['gaussian_copula'].rho).mean())
rows={r['control']:r for r in m['rows']}
for ctrl in ['gaussian_copula','independence']:
    for ph in ['development','evaluation']:
        sel=cand.phase==ph
        d=(cand.loss_ind[sel]-P[ctrl].loss_ind[sel])
        saved=[p for p in rows[ctrl]['phases'] if p['name']==ph][0]
        print(f"{ctrl:16s} {ph:12s} n={sel.sum()} mine mean={d.mean():.9f} saved={saved['mean']:.9f} diff={d.mean()-saved['mean']:.2e} | cand loss mine={cand.loss_ind[sel].mean():.9f} saved={saved['candidate_loss']:.9f}  ctrl loss mine={P[ctrl].loss_ind[sel].mean():.9f} saved={saved['control_loss']:.9f}")
        # offsets
        for k in range(5):
            s2=sel&(cand.offset==k); sv=[o for o in saved['offsets'] if o['offset']==k][0]
            assert abs(d[s2].mean()-sv['mean'])<1e-8 and s2.sum()==sv['n'], (ctrl,ph,k,d[s2].mean(),sv)
        for sv in saved['stability']:
            s2=sel&(cand.index>=sv['start'])&(cand.index<=sv['end'])
            assert abs(d[s2].mean()-sv['mean'])<1e-8 and s2.sum()==sv['n'], (ctrl,ph,sv,d[s2].mean())
        # robustness diagnostics
        q=np.quantile(d,[0.01,0.05,0.25,0.5,0.75,0.95,0.99])
        frac_neg=(d<0).mean()
        srt=np.sort(d.to_numpy())
        top1=srt[:max(1,int(0.01*len(srt)))].sum()/len(srt); 
        trimmed=d[(d>q[0])&(d<q[-1])].mean()
        print(f"   share pairs where t8 better={frac_neg:.3f} median={q[3]:.5f} q01={q[0]:.4f} q99={q[-1]:.4f} contrib of worst1%(most negative) to mean={top1:.5f} 1%-trimmed mean={trimmed:.5f} max={d.max():.4f}")
print("offset and stability subgroup means all match saved to 1e-8")
# per-year means for gaussian
d=(cand.loss_ind-P['gaussian_copula'].loss_ind)
print(d.groupby(cand.index.year).agg(['count','mean','median']).round(5))
# saved p-values
for ctrl in rows:
    for p in rows[ctrl]['phases']:
        print(ctrl,p['name'],'hac p',p['hac']['p_two_sided'] if 'p_two_sided' in p['hac'] else p['hac'], 'block p',{k:v.get('p_two_sided',v) for k,v in p['block_inference'].items()} if isinstance(p['block_inference'],dict) else '')
