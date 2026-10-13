import json,math
from pathlib import Path
p=Path('analysis_outputs/v2x_radar_i_three_model_selection');ds=json.loads((p/'metrics.json').read_text(encoding='utf-8-sig'))
records=[]
for d in ds:
 for s in d['sequences']:
  vals={}
  try:
   for v in ['relative','optimized_original']:
    a=s[v];m={}
    for cat in ['depth','point']:
     for k in ['rel','delta1','mae','rmse']:m[cat+'.'+k]=a[cat+'_frame_mean'][k]
    for k in ['acc','comp','chamfer','pairwise_mae','pairwise_rmse']:m['reconstruction.'+k]=a['reconstruction_frame_mean'][k]
    for cat,k in [('ate','rmse'),('ate','mean'),('rpe_trans','rmse'),('rpe_rot','rmse')]:m['pose.'+cat+'.'+k]=a['pose'][cat][k]
    assert all(isinstance(x,(float,int)) and math.isfinite(x) for x in m.values())
    vals[v]=m
   assert s['relative']['frames_evaluated']==s['optimized_original']['frames_evaluated']
  except (KeyError,TypeError,AssertionError):print('INVALID',d['model'],s['sequence_name']);continue
  b,a=vals.values();g={};bad=[];eq=[]
  for k in b:
   x,y=(1-b[k],1-a[k]) if k.endswith('delta1') else (b[k],a[k])
   diff=x-y;g[k]=diff/max(abs(x),1e-15)
   if diff < -1e-12:bad.append(k)
   elif abs(diff)<=1e-12:eq.append(k)
  scores=[sum(v for k,v in g.items() if k.startswith(prefix))/sum(k.startswith(prefix) for k in g) for prefix in ['depth.','point.','reconstruction.']]
  pg=[v for k,v in g.items() if k.startswith('pose.') and 'rpe_rot' not in k]
  score=(scores[0]+(scores[1]*4+scores[2]*5)/9+sum(pg)/len(pg))/3
  records.append(dict(model=d['model'],sequence=s['sequence_name'],before=b,after=a,bad=bad,equal=eq,improved=17-len(bad)-len(eq),score=score,source=d['source']))
models=['pi3_long','vggt_omega_long','4rc_long']
def rank(ms):
 common=set.intersection(*[{r['sequence'] for r in records if r['model']==m} for m in ms]);out=[]
 for s in common:
  rs=[r for r in records if r['sequence']==s and r['model'] in ms]
  out.append(dict(sequence=s,bad=sum(len(r['bad']) for r in rs),improved=sum(r['improved'] for r in rs),equal=sum(len(r['equal']) for r in rs),score=sum(r['score'] for r in rs)/len(rs)))
 return sorted(out,key=lambda x:(x['bad'],-x['score']))
three=rank(models);nine=rank([d['model'] for d in ds])
print('COUNTS',[(d['model'],len(d['sequences'])) for d in ds]);print('THREE',json.dumps(three[:8]));print('NINE',json.dumps(nine[:3]))
for r in records:
 if r['sequence']==three[0]['sequence'] and r['model'] in models:print('WINNER',json.dumps(r))
(p/'ranking.json').write_text(json.dumps(dict(three_models=three,nine_models=nine,records=records),indent=2),encoding='utf-8')
