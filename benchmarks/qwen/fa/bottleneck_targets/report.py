"""Accepted-only comparison of backend-specific response to identical interventions."""
import json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R/'scripts'))
from run_access_timing import stats
OUT=R/'results/cacheflex_flash_attention/bottleneck_20261004/targets_continuation_v1'
def main():
 state=json.loads((OUT/'status.json').read_text());rows=[];data={}
 for item in state['completed']:
  p=Path(item['path']);a=json.loads((p/'accepted.json').read_text());assert a==item
  c=a['case']
  if c['kind']!='fa':continue
  s=stats(p)[0];m={}
  for suffix in ['spmLsq0.spmRawWaitCycles','spmLsq0.spmPortBlockedEvents','spmLsq0.spmLoadResponses','spmLsq0.spmLoadServiceCycles','iew.spmLsqFullEvents','iew.lsqFullEvents','fetchStats0.icacheStallCycles']:
   m[suffix]=sum(s[f'system.cpu{i}.{suffix}'] for i in range(8))
  for level in ('l0','l1'):
   key='dataMissCredits' if level=='l0' else 'missCredits'
   m[level+'_credit_full_checks']=sum(s[f'system.ruby.{level}_cntrl{i}.{key}.fullChecks'] for i in range(8))
  m['instructions']=s['simInsts'];m['link_flits']=s['system.ruby.network.link_flits::total']
  data[(c['t'],c['spm'],c['variant'])]=a
  rows.append(dict(case=c,roi_us=a['roi_us'],counters=m))
 lines=['FA-39 SPM-specific bottleneck attribution (accepted cases only)',
 'T4096 non-fusion continuation. Packed-producer gate failed; excluded, not accepted.',
 'Compare fractional ROI gains within each backend, then subtract cache gain from SPM gain.',
 'A shared gain is not evidence of an SPM-specific limit. Counter reductions alone are insufficient.',
 'No gain is guaranteed; pairs and full-ROI validation must finish before conclusions.','']
 comparisons=[]
 for t in (4096,):
  for v in ('stage','l1','l2','both','packed'):
   keys=[(t,b,w) for b in (False,True) for w in ('control',v)]
   if not all(k in data for k in keys):continue
   gains={b:1-data[(t,b,v)]['roi_us']/data[(t,b,'control')]['roi_us'] for b in (False,True)}
   ratio=data[(t,False,v)]['roi_us']/data[(t,True,v)]['roi_us']
   row=dict(t=t,variant=v,cache_latency_reduction=gains[False],spm_latency_reduction=gains[True],spm_excess_gain=gains[True]-gains[False],cache_over_spm=ratio)
   comparisons.append(row);lines.append(f'T{t} {v}: cache gain {100*gains[False]:.4f}%, SPM gain {100*gains[True]:.4f}%, SPM excess {100*(gains[True]-gains[False]):.4f} percentage points, cache/SPM {ratio:.6f}x')
 lines+=['',f'Accepted {len(state["completed"])}/10; stage={state["stage"]}']
 (OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n');(OUT/'analysis.json').write_text(json.dumps(dict(cases=rows,comparisons=comparisons),indent=2)+'\n')
if __name__=='__main__':main()
