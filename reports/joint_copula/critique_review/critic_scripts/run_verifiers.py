import json, sys, time, pandas as pd
sys.path.insert(0,'.')
from src.joint_copula_protocol import load_protocol, EXECUTION, pipeline_config
from src.joint_copula_inputs import read_sources
from src.verify_joint_copula_forecasts import verify_forecasts
from src.verify_joint_copula_scores import verify_scores
D='data/model_memory_study/joint_copula_wave27/'
protocol=load_protocol()
metrics=json.load(open('reports/joint_copula/predictive/metrics.json'))
produced={n:pd.read_parquet(D+n+'.parquet') for n in ("features","targets","applications","panel","coverage","schedules")}
produced['fits']=json.load(open(D+'fits.json'))
calendar=pd.DatetimeIndex(produced['features'].index)
which=sys.argv[1]
t=time.time()
if which=='scores':
    r=verify_scores(produced['panel'],calendar,metrics,metrics['inherited_rows'],protocol)
else:
    qqq,spx,iv,adm=read_sources('.',EXECUTION['market_pins'])
    r=verify_forecasts(qqq,spx,iv,produced,pipeline_config(protocol))
print(which,'seconds',round(time.time()-t,1)); print(json.dumps(r,indent=1,default=str)[:3000])
