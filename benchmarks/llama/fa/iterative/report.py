"""Accepted-only full ROI and actual counter review; counters are not stall fractions."""
import json,re
from pathlib import Path
R=Path('/data2/rsappidi/cacheflex_mc')
OUT=R/'results/llama_fa_iterative_20261004/experiment_v1'
def counters(s):
 def total(fmt):return sum(s[fmt.format(i)] for i in range(8))
 m=dict(instructions=s['simInsts'],microops=s['simOps'],
  l1_credit_full=total('system.ruby.l0_cntrl{}.dataMissCredits.fullChecks'),
  l2_credit_full=total('system.ruby.l1_cntrl{}.missCredits.fullChecks'),
  spm_sq_full=total('system.cpu{}.iew.spmLsqFullEvents'),lsq_full=total('system.cpu{}.iew.lsqFullEvents'),
  iq_full=total('system.cpu{}.iew.iqFullEvents'),icache_stall=total('system.cpu{}.fetchStats0.icacheStallCycles'),
  spm_read_responses=total('system.cpu{}.spmLsq0.spmLoadResponses'),spm_read_cycles=total('system.cpu{}.spmLsq0.spmLoadServiceCycles'),
  spm_raw_wait=total('system.cpu{}.spmLsq0.spmRawWaitCycles'),spm_port_blocks=total('system.cpu{}.spmLsq0.spmPortBlockedEvents'))
 for label,stem in [('l1','l0_cntrl{}.Dcache'),('l2','l1_cntrl{}.cache'),('l3','l2_cntrl{}.L2cache')]:
  for kind in ('misses','accesses'):m[label+'_'+kind]=total('system.ruby.'+stem+'.m_demand_'+kind)
 for key in ('flits_injected::total','link_flits::total','average_packet_network_latency','average_packet_queueing_latency'):
  m[key]=s['system.ruby.network.'+key]
 pcs={}
 for k,v in s.items():
  match=re.fullmatch(r'system.cpu\d+.commit.pcNotExecutedCycles::(\d+)',k)
  if match:pcs[match[1]]=pcs.get(match[1],0)+v
 m['top_pending_pcs']=[dict(pc=hex(int(pc)),samples=v) for pc,v in sorted(pcs.items(),key=lambda x:-x[1])[:15]]
 return m
def write(state):
 rows=[]
 for a in state['completed']:
  assert json.loads((Path(a['path'])/'accepted.json').read_text())==a
  rows.append(a)
 targets=[a for a in rows if a['case']['t']==4096];best={}
 for backend in (False,True):
  subset=[a for a in targets if a['case']['spm']==backend]
  if subset:best['spm' if backend else 'cache']=min(subset,key=lambda a:a['roi_us'])
 result=dict(stage=state['stage'],winners={k:dict(name=v['name'],roi_us=v['roi_us'],case=v['case']) for k,v in best.items()},
  comparisons=[],caveat='Synthetic Llama3.2-1B-shaped FA only; FP16 GEMM rounding and FP32 state, not native model accuracy or end-to-end latency. Winners provisional until final review.')
 if len(best)==2:result['cache_over_spm']=best['cache']['roi_us']/best['spm']['roi_us']
 for a in targets:
  history=[b for b in targets if b['case']['spm']==a['case']['spm'] and b['name']!=a['name']]
  parent=a.get('hypothesis',{}).get('parent');p=next((b for b in history if b['name']==parent),None)
  if p:result['comparisons'].append(dict(name=a['name'],parent=parent,latency_reduction=1-a['roi_us']/p['roi_us'],
   counters={k:[p['counters'][k],a['counters'][k]] for k in ('instructions','l1_misses','l2_misses','spm_sq_full','l1_credit_full')},
   interpretation='Full ROI decides; pressure counters and PC samples overlap and are not removable-time fractions.'))
 (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['Llama3.2-1B-shaped FA iterative tuning',result['caveat'],f'Stage: {state["stage"]}',f'Accepted: {len(rows)}','']
 for a in rows:lines.append(f'{a["name"]}: {a["roi_us"]:.4f} us; '+('TARGET' if a['case']['t']==4096 else 'correctness gate'))
 lines+=['',json.dumps(result['winners'],indent=2)]
 (OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
