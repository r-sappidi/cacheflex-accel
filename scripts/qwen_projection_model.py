"""Exact projection work accounting; bounded heuristic candidate selection."""
import itertools,json,math
from pathlib import Path
ROOT=Path('/data2/rsappidi/cacheflex_mc');BASE=ROOT/'results/ai_workloads/qwen_projection_20260928'
BASELINE=dict(mp=2,kc=128)
def describe(t,op,c):
 n=1152 if op=='qkv' else 896;k=896;mp=c['mp'];kc=c['kc'];np_=8//mp;blocks=(t+7)//8
 assert op in ('qkv','out') and mp in (1,2,4,8) and kc in (128,256,512,1024)
 cores=[]
 for i in range(8):
  mi,ni=divmod(i,np_);rows=8*(blocks*(mi+1)//mp-blocks*mi//mp);cols=128*((n//128)*(ni+1)//np_-(n//128)*ni//np_)
  cores.append(dict(rows=rows,cols=cols,flops=2*rows*k*cols,copies=4*k*(cols//128) if rows else 0,
   a_pack_bytes=2*rows*k,accumulator_bytes=4*rows*cols*(2*math.ceil(k/kc)-1)))
 keys=('flops','copies','a_pack_bytes','accumulator_bytes');totals={key:sum(x[key] for x in cores) for key in keys};critical={key:max(x[key] for x in cores) for key in keys}
 assert totals['flops']==2*(blocks*8)*k*n
 pressure=max((2*x['rows']*min(k,kc)+256*min(k,kc)+512*x['rows'])/(512*1024) for x in cores)
 return dict(case=c,t=t,op=op,cores=cores,totals=totals,critical=critical,pressure=pressure,
  spm_claim_bytes=256*min(k,kc),imbalance=critical['flops']/(totals['flops']/8),
  layout_bytes=4*t*(n if op=='qkv' else k),note='Logical traffic; not measured misses or additive runtime.')
def analyze():
 result=dict(ledger=['AI-01','AI-03','AI-04','AI-06','AI-10'],selected={},rankings={},method='27 heuristic sensitivity scenarios, no predicted latency; equal four-case budget per backend/operator.',objective='First full invocation including packing/bias/layout; second call separately reported.')
 for op in ('qkv','out'):
  rows=[describe(1024,op,dict(mp=m,kc=k)) for m,k in itertools.product((1,2,4,8),(128,256,512,1024))];base=describe(1024,op,BASELINE);result['selected'][op]={};result['rankings'][op]={}
  for arm in ('cache','spm'):
   orders=[]
   for move,acc,pressure in itertools.product((.05,.15,.30),(.05,.15,.30),(.02,.10,.25)):
    scores=[]
    for r in rows:
     c,b=r['critical'],base['critical'];traffic=lambda x:x['copies']*64+x['a_pack_bytes']
     delivery=max(0,r['spm_claim_bytes']/65536-1) if arm=='cache' else 0
     scores.append((1-move-acc)*c['flops']/b['flops']+move*traffic(c)/traffic(b)+acc*c['accumulator_bytes']/b['accumulator_bytes']+pressure*(max(0,r['pressure']-1)/max(1,base['pressure'])+.25*delivery))
    orders.append(sorted(range(len(rows)),key=lambda i:(scores[i],rows[i]['case']['kc'])))
   ranks=sorted([dict(case=r['case'],mean_rank=sum(o.index(i) for o in orders)/len(orders)) for i,r in enumerate(rows)],key=lambda x:x['mean_rank'])
   picks=[dict(BASELINE)]
   for x in ranks:
    if x['case'] not in picks:picks.append(x['case'])
    if len(picks)==3:break
   used={x['mp'] for x in picks};picks.append(next(x['case'] for x in ranks if x['case']['mp'] not in used))
   result['selected'][op][arm]=picks;result['rankings'][op][arm]=ranks
 BASE.mkdir(parents=True,exist_ok=True);(BASE/'selection.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':print(json.dumps(analyze()['selected'],indent=2))
