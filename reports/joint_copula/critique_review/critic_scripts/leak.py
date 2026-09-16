import sys, numpy as np, pandas as pd
sys.path.insert(0,'.')
from src.joint_copula_protocol import EXECUTION
from src.joint_copula_inputs import read_sources
from src.joint_risk_features import build_features, ALL_FEATURES
D='data/model_memory_study/joint_copula_wave27/'
saved_f=pd.read_parquet(D+'features.parquet'); saved_t=pd.read_parquet(D+'targets.parquet')
qqq,spx,iv,_=read_sources('.',EXECUTION['market_pins'])
f_full,t_full=build_features(qqq,spx,iv)
print("full rebuild equals saved features:", np.allclose(f_full[list(ALL_FEATURES)].to_numpy(float),saved_f[list(ALL_FEATURES)].to_numpy(float),equal_nan=True), "targets:", np.allclose(t_full[['y_qqq','y_spx']].to_numpy(float),saved_t[['y_qqq','y_spx']].to_numpy(float),equal_nan=True))
for T in ['2018-03-15','2022-06-30','2025-10-01']:
    T=pd.Timestamp(T)
    f_cut,t_cut=build_features(qqq[qqq.index<=T],spx[spx.index<=T],iv[iv.index<=T])
    cal=spx.index; pos=cal.get_loc(T)
    # origins with feature_cutoff <= T i.e. origin <= cal[pos+1]
    ok_origins=cal[:pos+1]; ok2=cal[:pos+2]
    a=f_cut.loc[ok_origins,list(ALL_FEATURES)].to_numpy(float); b=f_full.loc[ok_origins,list(ALL_FEATURES)].to_numpy(float)
    same=np.allclose(a,b,equal_nan=True)
    # also: does perturbing data strictly after T change anything for origins<=T+1?
    q2=qqq.copy(); q2.loc[q2.index>T,['open','high','low','close']]*=1.37
    s2=spx.copy(); s2.loc[s2.index>T,['open','high','low','close']]*=0.81
    f_pert,_=build_features(q2,s2,iv)
    same2=np.allclose(f_pert.loc[ok2,list(ALL_FEATURES)].to_numpy(float),f_full.loc[ok2,list(ALL_FEATURES)].to_numpy(float),equal_nan=True)
    print(f"cut at {T.date()}: features for origins <= {ok_origins[-1].date()} unchanged by truncation: {same}; unchanged by future perturbation (origins <= {ok2[-1].date()}): {same2}")
    # a feature at origin cal[pos+2] should depend on T+1 data -> perturbation should change it
    nxt=cal[pos+2]
    print(f"   sanity: origin {nxt.date()} features changed by perturbation after T: {not np.allclose(f_pert.loc[[nxt],list(ALL_FEATURES)].to_numpy(float),f_full.loc[[nxt],list(ALL_FEATURES)].to_numpy(float),equal_nan=True)}")
