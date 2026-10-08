"""Bounded overnight screening; failures remain evidence, never accepted timings."""
import os,sys,json,time,subprocess,fcntl
from pathlib import Path
R=Path(__file__).resolve().parents[4];H=Path(__file__).resolve().parent;ROOT=R/'results/cacheflex_flash_attention/ncc_probe_20261005_v3'
sys.path.insert(0,str(H));from plan import ARMS,service
state=dict(pid=os.getpid(),stage='screening',active={},pending=['spm','drained','pv_oracle','ppack','kpack','pv_traffic','pv128','pv64','pv256','pv_l3_budget','cache'],completed=[],failed=[])
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
    if a['case']['t']==t:comparisons.append(dict(arm=a['arm'],t=t,roi_us=a['roi_us'],latency_reduction=1-a['roi_us']/base['roi_us'],service_cycles=service(a['case']['mode'])))
 save(ROOT/'analysis.json',dict(rows=rows,comparisons=comparisons,failed=state['failed'],scope='Blocking functional operation proxies; finite-service sensitivity, NOT a timed decoupled engine or NoC model.'))
 lines=['Overnight NCC screening','', 'Full-oracle accepted timings only. Ordinary-SPM denominator; same ELF.', 'Functional transfers are untimed. Traffic variants retain CPU operand-load probes; finite variants ADD serialized service/bandwidth budgets. Not hardware speedups.','']
 lines += [f"T{a['t']} {a['arm']}: {a['roi_us']:.4f} us; reduction {100*a['latency_reduction']:.2f}%" for a in comparisons]
 lines+=['','Failed/unaccepted: '+repr(state['failed']), '', 'Interpretation: oracle alone measures optimistic opportunity. >=5% traffic-preserving/finite gain warrants T4096 confirmation. <5% is a weak signal, not proof against a truly asynchronous engine. Compare drained control to diagnose synchronization overhead. K/P pack cases retain timed read/write probes and added DSB boundaries. L3 budget is transfer sensitivity only; no actual L3 relocation/network contention modeled.']
 (ROOT/'REPORT.txt').write_text('\n'.join(lines)+'\n');return comparisons
lock=(ROOT/'overnight.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
procs={};free=[12,14,16,18]
def batch(arms,target=False):
 queue=list(arms);state['pending']=queue.copy();status()
 while queue or procs:
  while queue and free:
   arm=queue.pop(0);offset=free.pop(0);label=arm+('_target' if target else '')
   env=dict(os.environ,FA46_ARM=arm,FA46_CPU_OFFSET=str(offset),FA46_TARGET='1' if target else '0')
   log=(ROOT/(label+'.log')).open('w');p=subprocess.Popen([sys.executable,str(H/'run.py')],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT);log.close();procs[label]=(p,offset);state['active'][label]=dict(pid=p.pid,cpu_offset=offset);state['pending']=queue.copy();status()
  for label,(p,offset) in list(procs.items()):
   if p.poll() is None:continue
   sfile=ROOT/label/'status.json';s=read(sfile) if sfile.exists() else {}
   if p.returncode!=0 or s.get('stage')!='complete':state['failed'].append(dict(arm=label,code=p.returncode,details=s.get('failed')))
   else:state['completed'].append(label)
   free.append(offset);del procs[label];del state['active'][label];status();report()
  status();time.sleep(5)
try:
 assert (R/'results/cacheflex_flash_attention/ncc_probe_20261005/build_v1/ready.json').exists()
 batch(state['pending']);comparison=report()
 # Screen all, then select at most two realistic-budget candidates; do not promote
 # zero-transfer oracle alone. Always include matched cache and SPM target controls.
 pool=[a for a in comparison if a['t']==1024 and a['arm'] in ('kpack','ppack','pv_traffic','pv64','pv128','pv256','pv_l3_budget') and a['latency_reduction']>=.05]
 selected=sorted(pool,key=lambda a:-a['latency_reduction'])[:2]
 save(ROOT/'target_decision.json',dict(threshold=.05,selected=selected,reason='Bounded full-ROI confirmation; no global optimum, no automatic hardware feasibility claim.'))
 if selected and 'spm' in state['completed'] and 'cache' in state['completed']:
  state['stage']='target_confirmation';batch(['spm','cache']+[a['arm'] for a in selected],True)
 state['stage']='complete_requires_scientific_review';report();status()
 with (R/'docs/experiments/LEDGER.md').open('a') as f:f.write('\n**FA-46 overnight screening finished:** accepted arms '+repr(state['completed'])+'; failed/unaccepted '+repr(state['failed'])+'. Read ncc_probe_20261005_v3/REPORT.txt and target_decision.json. Functional proxy/serialized service sensitivities, not hardware performance claims.\n')
except BaseException as e:
 state['stage']='failed_requires_review';state['failed'].append(repr(e));status()
 for p,off in procs.values():
  if p.poll() is None:p.terminate()
 raise
