"""Bounded overnight screening; failures remain evidence, never accepted timings."""
import os,sys,json,time,subprocess,fcntl
from pathlib import Path
R=Path(__file__).resolve().parents[4];H=Path(__file__).resolve().parent;ROOT=R/'results/cacheflex_flash_attention/pv_diag_20261005'
sys.path.insert(0,str(H));from plan import ARMS,service
state=dict(pid=os.getpid(),stage='screening',active={},pending=list(ARMS),completed=[],failed=[])
def read(p):return json.loads(p.read_text())
def save(p,x):tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def status():state['updated']=time.time();save(ROOT/'status.json',state)
def report():
 rows=[]
 for arm in ARMS:
  for suffix in ('','_target'):
   p=ROOT/(arm+suffix)/'status.json'
   if not p.exists():continue
   s=read(p)
   for a in s['completed']:
    assert read(Path(a['path'])/'accepted.json')==a;rows.append(dict(arm=arm,**a))
 for t in (256,1024,4096):assert len({a['output_sha256'] for a in rows if a['case']['t']==t})<=1,'peer output bits differ'
 comparisons=[]
 for t in (1024,4096):
  base=next((a for a in rows if a['arm']=='spm' and a['case']['t']==t),None)
  if base:
   for a in rows:
    if a['case']['t']==t:comparisons.append(dict(arm=a['arm'],t=t,roi_us=a['roi_us'],latency_reduction=1-a['roi_us']/base['roi_us'],service_cycles=service(a['case']['mode'],a['case']['flags'])))
 save(ROOT/'analysis.json',dict(rows=rows,comparisons=comparisons,failed=state['failed'],scope='Passive phase observations and matched intervention factorials; functional engine service model, not hardware claims.'))
 lines=['FA-48 asynchronous PV validation','', 'Full-oracle accepted timings only. Ordinary-SPM denominator; same ELF.', 'Coherent payload uses timed CPU loads/stores through four SVE registers per1024B. PrivateV read remains functional. Engine service overlaps useful CPU QK/softmax in async arms; transport/probe costs retained. Not hardware speedups.','']
 lines += [f"T{a['t']} {a['arm']}: {a['roi_us']:.4f} us; reduction {100*a['latency_reduction']:.2f}%" for a in comparisons]
 lines+=['','Failed/unaccepted: '+repr(state['failed']), '', 'Observer ON/OFF must preserve every non-host statistic. Flags1 remove extra CPU P/V probe;2 zero modeled service;4 remove two outer fences only. Register chunk barriers/ISBs and correctness remain. Full ROI differences are intervention effects, not additive bottleneck percentages.']
 (ROOT/'REPORT.txt').write_text('\n'.join(lines)+'\n');return comparisons
lock=(ROOT/'overnight.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
procs={};free=[10,12,14,16,18]
def batch(arms,target=False):
 queue=list(arms);state['pending']=queue.copy();status()
 while queue or procs:
  while queue and free:
   arm=queue.pop(0);offset=free.pop(0);label=arm+('_target' if target else '')
   env=dict(os.environ,FA48_ARM=arm,FA48_CPU_OFFSET=str(offset),FA48_TARGET='1' if target else '0')
   log=(ROOT/(label+'.log')).open('w');p=subprocess.Popen([sys.executable,str(H/'run.py')],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT);log.close();procs[label]=(p,offset);state['active'][label]=dict(pid=p.pid,cpu_offset=offset);state['pending']=queue.copy();status()
  for label,(p,offset) in list(procs.items()):
   if p.poll() is None:continue
   sfile=ROOT/label/'status.json';s=read(sfile) if sfile.exists() else {}
   if p.returncode!=0 or s.get('stage')!='complete':state['failed'].append(dict(arm=label,code=p.returncode,details=s.get('failed')))
   else:state['completed'].append(label)
   free.append(offset);del procs[label];del state['active'][label];status();report()
  status();time.sleep(5)
try:
 assert (ROOT/'build_v1/ready.json').exists() and (ROOT/'guest_v1/frozen.json').exists()
 batch(['spm_off','async_off','spm','async'])
 assert not state['failed'],'Observer controls failed; do not launch interventions'
 batch(['no_probe','zero_service','no_probe_zero','less_fence','no_probe_less_fence']);comparison=report()
 state['stage']='complete_requires_scientific_review';report();status()
 with (R/'docs/experiments/LEDGER.md').open('a') as f:f.write('\n**FA-48 overnight screening finished:** accepted arms '+repr(state['completed'])+'; failed/unaccepted '+repr([x.get('arm',str(x)) for x in state['failed']])+'. Read pv_diag_20261005/REPORT.txt and analysis.json. Asynchronous register-transport/service-time proxies, not hardware performance claims.\n')
except BaseException as e:
 state['stage']='failed_requires_review';state['failed'].append(repr(e));status()
 for p,off in procs.values():
  if p.poll() is None:p.terminate()
 raise
