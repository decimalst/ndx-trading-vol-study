import json, numpy as np, pandas as pd
D='data/model_memory_study/joint_copula_wave27/'
fits=json.load(open(D+'fits.json')); panel=pd.read_parquet(D+'panel.parquet'); cov=pd.read_parquet(D+'coverage.parquet'); apps=pd.read_parquet(D+'applications.parquet'); sched=pd.read_parquet(D+'schedules.parquet'); feats=pd.read_parquet(D+'features.parquet'); targ=pd.read_parquet(D+'targets.parquet')
cal=pd.DatetimeIndex(feats.index)
print("calendar",len(cal),cal.min().date(),cal.max().date(), "features cols",len(feats.columns))
print("targets cols",list(targ.columns))
print("coverage status",cov.status.value_counts().to_dict())
print("schedules status",sched.status.value_counts().to_dict(), "fits",len(fits))
bad=0; rhos=[]
for f in fits:
    fo=pd.Timestamp(f['fit_origin']); tc=pd.Timestamp(f['training_cutoff'])
    tr=pd.DatetimeIndex(pd.to_datetime(f['train_origins']))
    pos=cal.get_indexer(tr); nxt=cal[pos+1]
    # training origin < fit origin, and its label (next session) <= training cutoff = session before fit origin
    if not ((tr<fo).all() and (nxt<=tc).all() and cal[cal.get_loc(fo)-1]==tc): bad+=1; print("CLOCK VIOLATION",f['month'])
    # applications after fit origin within month all use that fit
    ap=pd.DatetimeIndex(pd.to_datetime(f['application_origins']))
    if not (ap>=fo).all() or ap.to_period('M').nunique()!=1: bad+=1; print("APP VIOLATION",f['month'])
    if f['status']!='fitted': print("status",f['month'],f['status'])
    dep=f['model_audit']['dependence']
    rhos.append((f['month'],dep['t8_copula']['rho'],dep['gaussian_copula']['rho'],dep['t8_copula']['audit']['certificate']['gap'],dep['t8_copula']['audit']['projected_gradient'],dep['gaussian_copula']['audit']['certificate']['gap'],f['train_n'],dep['t8_copula']['audit']['certificate'].get('splits')))
print("clock violations:",bad)
r=pd.DataFrame(rhos,columns=['month','rho_t8','rho_g','gap_t8','pg_t8','gap_g','train_n','splits'])
print(r.describe().round(6))
print(r.iloc[[0,1,50,116,117]])
# check panel rho equals fit rho per month
pm=panel[panel.model=='t8_copula'].assign(month=lambda x: x.fit_origin.dt.to_period('M').astype(str)).groupby('month').rho.agg(['min','max','count'])
mism=(pm.join(r.set_index('month')).pipe(lambda x: (abs(x['min']-x.rho_t8)>1e-15)|(abs(x['max']-x.rho_t8)>1e-15))).sum()
print("panel/fit rho mismatches:",int(mism))
# feature cutoff check vs panel
pos=cal.get_indexer(panel.origin)
print("feature_cutoff == prev session:", (panel.feature_cutoff_date.to_numpy()==cal[pos-1].to_numpy()).all(), " target_end == next session:", (panel.target_end.to_numpy()==cal[pos+1].to_numpy()).all())
print("train_n min",panel.train_n.min(), "fit_origin<=origin",(panel.fit_origin<=panel.origin).all())
# growing training window?
print("train_n first/last:", r.train_n.iloc[0], r.train_n.iloc[-1])
# sanity: what's the empirical corr of standardized residuals in eval
t8=panel[panel.model=='t8_copula']
z=(t8[['y_qqq','y_spx']].to_numpy()-t8[['mu_qqq','mu_spx']].to_numpy())/np.sqrt(0.75*t8[['h_qqq','h_spx']].to_numpy())
print("out-of-sample z stats: mean",z.mean(0).round(4),"var",z.var(0).round(4),"(t8 std var=1.333) corr",np.corrcoef(z.T)[0,1].round(4), "kurt", (((z-z.mean(0))**4).mean(0)/z.var(0)**2).round(2), "(t8 kurt=4.5)")
print("|z|>5 rows:",(np.abs(z)>5).any(1).sum(), " max|z|",np.abs(z).max().round(2))
