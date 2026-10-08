"""Read accepted FA43 cases and compare with frozen same-guest FA39 controls."""
import json,sys,os
from pathlib import Path
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R/'scripts'))
from run_access_timing import stats
from plan import control_path
ROOT=R/'results/cacheflex_flash_attention/network_relief_20261004'
def main():
 rows=[]
 for arm in ('cache_buf8','spm_buf8','cache_buf16','spm_buf16'):
  p=ROOT/arm/'status.json'
  if not p.exists():continue
  d=json.loads(p.read_text())
  for a in d['completed']:
   c=a['case'];base=control_path(c);b=json.loads((base/'accepted.json').read_text());s=stats(Path(a['path']))[0];bs=stats(base)[0]
   assert json.loads((Path(a['path'])/'accepted.json').read_text())==a
   metrics=['average_flit_network_latency','average_flit_queueing_latency','average_flit_latency','average_packet_network_latency','flits_injected::total','link_flits::total']
   rows.append(dict(name=a['name'],t=c['t'],backend='spm' if c['spm'] else 'cache',depth=int(c['variant'][3:]),roi_us=a['roi_us'],baseline_us=b['roi_us'],latency_reduction=1-a['roi_us']/b['roi_us'],metrics={k:dict(baseline=bs.get('system.ruby.network.'+k),modified=s.get('system.ruby.network.'+k)) for k in metrics}))
 # Each reporter has its own output to avoid simultaneous rewrite races.
 out=ROOT/os.environ['FA43_ARM'];(out/'analysis.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(json.dumps([dict(name=x['name'],roi_us=x['roi_us'],gain_pct=100*x['latency_reduction']) for x in rows]))
if __name__=='__main__':main()
