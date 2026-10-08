import json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R/'scripts'))
from run_access_timing import stats
OUT=R/'results/cacheflex_flash_attention/spm_isolation_20261004/experiment_v1'
def main():
 state=json.loads((OUT/'status.json').read_text());rows=[];baseline={}
 for a in state['completed']:
  if a['case']['kind']!='fa':continue
  c=a['case'];s=stats(Path(a['path']))[0]
  m={key:sum(v for k,v in s.items() if k.endswith('.'+key)) for key in ['m_idealStagingCopies','m_spmcpRequestsIssued','spmIsolation.baselineFull','spmIsolation.relieved','spmLoadResponses','spmLoadServiceCycles','spmRawWaitCycles','spmPortBlockedEvents','iew.spmLsqFullEvents']}
  m['maxCopyShadows']=max(s[f'system.cpu{i}.spmIsolation.maxCopyShadows'] for i in range(8))
  if c['variant']=='control':baseline[(c['t'],c['spm'])]=a['roi_us']
  rows.append(dict(case=c,roi_us=a['roi_us'],counters=m))
 lines=['FA-40 SPM-only isolated constraints; accepted results only.',
 'Gate timings are correctness evidence. Instruction costs remain; oracle is not realizable hardware.',
 'Bridge96/4096 are separate diagnostic metadata budgets, not free production storage.',
 'Cache combined arm must reproduce default cache stats exactly. No full cache-equivalence claim.','']
 for row in rows:
  c=row['case'];b=baseline.get((c['t'],c['spm']));row['latency_reduction']=None if b is None else 1-row['roi_us']/b
  lines.append(f'T{c["t"]} {"SPM" if c["spm"] else "cache"} {c["variant"]}: {row["roi_us"]:.4f}us gain={row["latency_reduction"]}; {row["counters"]}')
 (OUT/'analysis.json').write_text(json.dumps(rows,indent=2)+'\n');(OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
