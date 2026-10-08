"""Compare accepted same-ELF FA44 scheduling variants; historical timings not pooled."""
import json,os,sys
from pathlib import Path
R=Path(__file__).resolve().parents[4];ROOT=R/'results/cacheflex_flash_attention/paced_staging_20261004'
sys.path.insert(0,str(R/'scripts'));from run_access_timing import stats
ARMS=('cache','burst','paced','overlap','dbburst')
def main():
 rows=[];data={}
 for arm in ARMS:
  p=ROOT/arm/'status.json'
  if not p.exists():continue
  d=json.loads(p.read_text())
  for a in d['completed']:
   c=a['case'];path=Path(a['path']);assert json.loads((path/'accepted.json').read_text())==a
   s=stats(path)[0];row=dict(arm=arm,t=c['t'],roi_us=a['roi_us'],output_sha256=a['output_sha256'],copies=a['copies'],network_ns=s['system.ruby.network.average_flit_network_latency']/1000,queue_ns=s['system.ruby.network.average_flit_queueing_latency']/1000,total_flit_ns=s['system.ruby.network.average_flit_latency']/1000,flits=s['system.ruby.network.flits_injected::total'],spm_slot_bytes=(65536 if c['schedule']>=2 and c['t']>=512 else 32768) if c['spm'] else 0)
   rows.append(row);data[(c['t'],arm)]=row
 comparisons=[]
 for t in (256,512,1024):
  hashes={x['output_sha256'] for x in rows if x['t']==t};assert len(hashes)<=1,'cross-backend output bits differ'
  for arm,base in [('paced','burst'),('overlap','burst'),('overlap','dbburst'),('dbburst','burst'),('burst','cache')]:
   if (t,arm) in data and (t,base) in data:comparisons.append(dict(t=t,arm=arm,base=base,latency_reduction=1-data[t,arm]['roi_us']/data[t,base]['roi_us']))
 out=ROOT/os.environ['FA44_ARM'];(out/'analysis.json').write_text(json.dumps(dict(rows=rows,comparisons=comparisons),indent=2)+'\n');print(json.dumps(comparisons),flush=True)
if __name__=='__main__':main()
