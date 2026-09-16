import sys, numpy as np, pandas as pd
sys.path.insert(0,'.')
from scipy.stats import norm, t as student
from src.joint_copula_density import fit_dependence, log_copula
D='data/model_memory_study/joint_copula_wave27/'
panel=pd.read_parquet(D+'panel.parquet')
P={k:panel[panel.model==k].set_index('origin').sort_index() for k in ['t8_copula','gaussian_copula']}
c=P['t8_copula']; y=c[['y_qqq','y_spx']].to_numpy(); mu=c[['mu_qqq','mu_spx']].to_numpy(); h=c[['h_qqq','h_spx']].to_numpy(); z=(y-mu)/np.sqrt(0.75*h)
rt=c.rho.to_numpy(); rg=P['gaussian_copula'].rho.to_numpy(); ph=c.phase.to_numpy()
d_obs=log_copula(z,rg,'gaussian')-log_copula(z,rt,'t8')
print("observed diff: dev",d_obs[ph=='development'].mean().round(5),"eval",d_obs[ph=='evaluation'].mean().round(5))
# Fully fair simulation: generate from each arm's fitted copula, then refit BOTH arms by ML on the simulated pairs, score.
rng=np.random.default_rng(7); n=len(z); res={'gaussian_true':[], 't8_true':[]}
for k in range(30):
    for truth in ('gaussian_true','t8_true'):
        r=rg if truth=='gaussian_true' else rt
        w=np.column_stack([rng.standard_normal(n),rng.standard_normal(n)]); w[:,1]=r*w[:,0]+np.sqrt(1-r**2)*w[:,1]
        zs = student.ppf(norm.cdf(w),8) if truth=='gaussian_true' else w/np.sqrt(rng.chisquare(8,n)/8)[:,None]
        rho_t,_=fit_dependence(zs,'t8'); rho_g,_=fit_dependence(zs,'gaussian')
        res[truth].append((log_copula(zs,rho_g,'gaussian')-log_copula(zs,rho_t,'t8')).mean())
for k,v in res.items(): print(f"expected (candidate-control) diff if {k}, both arms refit in-sample on {n} pairs: {np.mean(v):.5f} sd {np.std(v):.5f}")
# Co-movement of squared residuals: observed vs t8 / gaussian implied
def sqcorr(zz): return np.corrcoef(zz[:,0]**2,zz[:,1]**2)[0,1]
sims={'gaussian':[], 't8':[]}
for k in range(30):
    w=np.column_stack([rng.standard_normal(n),rng.standard_normal(n)]); w[:,1]=rg*w[:,0]+np.sqrt(1-rg**2)*w[:,1]
    sims['gaussian'].append(sqcorr(student.ppf(norm.cdf(w),8)))
    w=np.column_stack([rng.standard_normal(n),rng.standard_normal(n)]); w[:,1]=rt*w[:,0]+np.sqrt(1-rt**2)*w[:,1]
    sims['t8'].append(sqcorr(w/np.sqrt(rng.chisquare(8,n)/8)[:,None]))
print(f"corr(z1^2,z2^2): observed={sqcorr(z):.3f}  gaussian-copula implied={np.mean(sims['gaussian']):.3f}  t8-copula implied={np.mean(sims['t8']):.3f}")
# Marginal PIT calibration out of sample
u=student.cdf(z,8)
for j,a in enumerate(['qqq','spx']):
    print(f"{a}: PIT share below 0.025={np.mean(u[:,j]<0.025):.4f} above 0.975={np.mean(u[:,j]>0.975):.4f} (nominal 0.025 each); |z|>3 share={np.mean(np.abs(z[:,j])>3):.4f} (t8 nominal {2*student.sf(3,8):.4f}); var(z)={z[:,j].var():.3f} (nominal 1.333)")
print("both |z|>2.5 same day: observed",np.mean((np.abs(z)>2.5).all(1)).round(4), " t8-implied", np.mean([np.mean((np.abs(w/np.sqrt(rng.chisquare(8,n)/8)[:,None])>2.5).all(1)) for w in [np.column_stack([rng.standard_normal(n),rng.standard_normal(n)]) for _ in range(1)] for w in [np.column_stack([w[:,0],rt*w[:,0]+np.sqrt(1-rt**2)*w[:,1]])]]).round(4))
