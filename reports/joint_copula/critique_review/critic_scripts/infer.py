import json, numpy as np, pandas as pd
from scipy.stats import norm, t as student, multivariate_t, multivariate_normal
D='data/model_memory_study/joint_copula_wave27/'
panel=pd.read_parquet(D+'panel.parquet'); feats=pd.read_parquet(D+'features.parquet'); cal=pd.DatetimeIndex(feats.index)
m=json.load(open('reports/joint_copula/predictive/metrics.json'))
P={k:panel[panel.model==k].set_index('origin').sort_index() for k in ['t8_copula','gaussian_copula','independence']}
c=P['t8_copula']; y=c[['y_qqq','y_spx']].to_numpy(); mu=c[['mu_qqq','mu_spx']].to_numpy(); h=c[['h_qqq','h_spx']].to_numpy(); s=np.sqrt(0.75*h); z=(y-mu)/s
def logc_t8(z,rho):
    out=np.empty(len(z))
    for i in range(len(z)):
        R=np.array([[1,rho[i]],[rho[i],1]]); out[i]=multivariate_t.logpdf(z[i],shape=R,df=8)-student.logpdf(z[i],8).sum()
    return out
def logc_g(z,rho):
    w=norm.ppf(student.cdf(z,8)); out=np.empty(len(z))
    for i in range(len(z)):
        R=np.array([[1,rho[i]],[rho[i],1]]); out[i]=multivariate_normal.logpdf(w[i],cov=R)-norm.logpdf(w[i]).sum()
    return out
rt=c.rho.to_numpy(); rg=P['gaussian_copula'].rho.to_numpy()
d_obs=logc_g(z,rg)-logc_t8(z,rt)   # candidate loss minus control loss
rng=np.random.default_rng(1)
for ph,(a,b) in [('development',('2016-01-04','2019-12-31')),('evaluation',('2020-01-01','2025-10-20'))]:
    sel=(c.index>=a)&(c.index<=b); full=cal[(cal>=a)&(cal<=b)]
    vals=pd.Series(d_obs[sel],index=c.index[sel]).reindex(full).to_numpy(); mask=np.isfinite(vals); v=np.where(mask,vals,0.0)
    T=len(v); mean=v[mask].mean()
    infl=np.where(mask,(v-mean)/(mask.mean()),0.0)
    lv=infl@infl/T
    for L in range(1,127): lv+=2*(1-L/127)*(infl[L:]@infl[:-L])/T
    se=np.sqrt(lv/T); saved=[p for p in m['rows'][0]['phases'] if p['name']==ph][0]
    print(f"{ph}: mean={mean:.9f} my HAC126 se={se:.9f} saved se={saved['hac']['se']:.9f} z={mean/se:.2f} my p={2*norm.sf(abs(mean/se)):.3e} saved p={saved['hac']['p']:.3e}")
    # my own circular block bootstrap, ratio of sums, 20000 draws, block 63
    for B in (21,63,126):
        nb=int(np.ceil(T/B)); draws=20000
        starts=rng.integers(0,T,size=(draws,nb))
        idx=(starts[:,:,None]+np.arange(B)[None,None,:]).reshape(draws,-1)[:,:T]%T
        num=v[idx].sum(1); den=mask[idx].sum(1)
        boot=num/den
        lo,hi=np.quantile(boot,[0.025,0.975]); centered=boot-mean
        p=(np.sum(np.abs(centered)>=abs(mean))+1)/(draws+1)
        print(f"   block{B}: my percentile CI=[{lo:.5f},{hi:.5f}] saved=[{saved['block_inference'][str(B)]['ci95'][0]:.5f},{saved['block_inference'][str(B)]['ci95'][1]:.5f}]  my p={p:.2e} (floor {1/(draws+1):.2e}) boot se={boot.std():.5f}")
# where does gain come from: bins of max|z|
mz=np.abs(z).max(1); q=np.quantile(mz,[0,.25,.5,.75,.9,.99,1])
lab=pd.cut(mz,q,include_lowest=True)
print(pd.DataFrame({'d':d_obs,'bin':lab}).groupby('bin',observed=True).d.agg(['count','mean','sum']).assign(share=lambda x:x['sum']/x['sum'].sum()).round(4))
# simulation: expected diff if data truly came from the Gaussian-copula arm vs t8 arm with saved parameters
K=20; acc={'gauss_true':[], 't8_true':[]}
for k in range(K):
    # gaussian copula truth
    w=np.column_stack([rng.standard_normal(len(z)),rng.standard_normal(len(z))]); w[:,1]=rg*w[:,0]+np.sqrt(1-rg**2)*w[:,1]
    zg=student.ppf(norm.cdf(w),8)
    acc['gauss_true'].append((logc_g(zg,rg)-logc_t8(zg,rt)).mean())
    # t8 truth: bivariate t = normal/sqrt(chi2/8)
    w=np.column_stack([rng.standard_normal(len(z)),rng.standard_normal(len(z))]); w[:,1]=rt*w[:,0]+np.sqrt(1-rt**2)*w[:,1]
    zt=w/np.sqrt(rng.chisquare(8,len(z))/8)[:,None]
    acc['t8_true'].append((logc_g(zt,rt)-logc_t8(zt,rt)).mean())
print("observed mean diff (all scored):",d_obs.mean().round(5))
for k,v in acc.items(): print(f"expected diff if {k}: {np.mean(v):.5f} +- {np.std(v):.5f}")
