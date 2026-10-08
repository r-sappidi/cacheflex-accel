"""FA-33 full-oracle residency factorial; shared OPS-23 resource admission."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import concurrent.futures as cf,fcntl,hashlib,importlib,json,re,resource,shutil,signal,subprocess,sys,threading,time,traceback
from pathlib import Path
ROOT=Path('/data2/rsappidi/cacheflex_mc');sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'benchmarks/cacheflex_flash_attention'),str(ROOT/'results/parallel_overlap_20260930')]
import numpy as np,broker
from run_access_timing import stats
from fa_headroom_profiles import leaves
from test_micro_timing import check_global_guard
from export_safe_elf import export_hash
from qwen_l3_reference import reference,validate
from parallel_budget_scheduler import owned
BASE=ROOT/'results/cacheflex_flash_attention/qwen_qo_ring_20260930';OUT=BASE/os.environ.get('QO_EXPERIMENT','experiment_v3');GUEST=BASE/os.environ.get('QO_GUEST','guest_v2')
OLD=ROOT/'results/cacheflex_l3_spm/lifecycle_20260928/experiment_v4';GIB=2**30
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',PYTHONDONTWRITEBYTECODE='1')
def acquire(ticket):
 # Same atomic accounting, permitting at most two additional FA leases when
 # full reservations fit. Existing broker/source remains byte-identical.
 with broker.locked() as b:
  if (time.time()-b['heartbeat']>20 or b['hold_new'] or
      sum(k.startswith('fa33_') for k in b['leases'])>=2 or
      len(b['leases'])>=6 or broker.total(b)+8*GIB>b['cap']):return False
  assert ticket not in b['leases'];b['leases'][ticket]=dict(bytes=8*GIB,owner=os.getpid(),acquired=time.time(),family='FA-33')
  return True
NAMES=['legacy_cache','legacy_v','head_cache','head_v','head_qo','head_qov']
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def name(c):return f"t{c['t']}_{NAMES[c['mode']]}_bc{c['bc']}_causal{int(c['causal'])}"
def work(c):
 t,bc,mode=c['t'],c['bc'],c['mode'];rows=t//8;half=rows//2;tasks=[];updates=[];copies=[]
 if mode<2:
  from run_qwen_l3_attention import work as prior
  x=prior(dict(t=t,r=c['r'],bc=bc,hq=14,causal=c['causal']));return dict(tasks=x['tasks'],updates=x['updates'],copies=x['copies'] if mode==1 else [0]*8)
 for core in range(8):
  qs=list(range(core*half,(core+1)*half))+list(range((15-core)*half,(16-core)*half))
  blocks=sum(((qs[b]+16+bc-1)//bc if c['causal'] else t//bc)*16 for b in range(0,rows,16))
  end=((qs[-1]+1+bc-1)//bc)*bc if c['causal'] else t
  tasks.append(14);updates.append(blocks*14);copies.append(14*((6*rows if mode>=4 else 0)+(2*end if mode in (3,5) else 0)))
 return dict(tasks=tasks,updates=updates,copies=copies)
def main():
 OUT.mkdir(parents=True);lockfile=(OUT/'coordinator.lock').open('w');fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'status.json').exists()
 (OUT/'runs').mkdir();(OUT/'references').mkdir();budget=read(ROOT/'results/parallel_budget_20260929/budget.json');os.sched_setaffinity(0,budget['cpu_set']);resource.setrlimit(resource.RLIMIT_AS,(2*GIB,resource.getrlimit(resource.RLIMIT_AS)[1]))
 lock=threading.RLock();stop=threading.Event();state=dict(pid=os.getpid(),stage='preparing',active={},completed=[],downstream='FA-34 authorized; awaits FA-33 acceptance and headroom model')
 frozen=read(GUEST/'frozen.json');frozen.update(read(OLD/'inputs.json'))
 for p in [Path(__file__),ROOT/'scripts/build_qwen_qo.py',ROOT/'scripts/test_qwen_qo.py',ROOT/'docs/experiments/QWEN_QO_RING_PLAN.md']:
  frozen[str(p)]=sha(p)
 for mod in ('qwen_l3_reference','fa_independent_reference','run_access_timing','fa_headroom_profiles','test_micro_timing','export_safe_elf','parallel_budget_scheduler','broker'):
  p=Path(importlib.import_module(mod).__file__);frozen[str(p)]=sha(p)
 for arm in ('cache','private_spm'):
  for n in ('command.json','config.json'):p=OLD/'runs'/f't4096_{arm}'/n;frozen[str(p)]=sha(p)
 def status():
  with lock:
   state['updated']=time.time();save(OUT/'status.json',state)
   lines=['# FA-33 accepted Qwen residency results','',f"Stage: {state['stage']}",'','All CPU arithmetic; no L3 offload. Small cases are validation, not performance conclusions.','', '| Case | Full FA us | SPM copies |','|---|---:|---:|']
   lines += [f"| {r['name']} | {r['roi_us']:.4f} | {r['copies']} |" for r in state['completed']]
   (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
 def unchanged():
  for p,h in list(frozen.items()):assert sha(p)==h,('frozen changed',p)
 def pressure():
  procs=owned();rss=sum(x['rss'] for x in procs.values());avail=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
  mon=read(ROOT/'results/parallel_memory_recovery_20260929/resources.json');cpu_high=time.time()-mon['updated']<30 and mon.get('cpu_pct_of_host',0)>20
  return cpu_high or rss>budget['ram_limit']-2*GIB or avail<64*GIB+max(0,budget['ram_limit']-rss)
 refs={}
 def getref(c):
  key=(c['t'],c['bc'],c['causal'])
  if key in refs:return
  filename=f't{c["t"]}_bc{c["bc"]}_q14_kv2_causal{int(c["causal"])}.npy';p=ROOT/'results/cacheflex_l3_spm/qwen_gqa_20260928/experiment_v2/references'/filename
  if not p.exists():
   p=OUT/'references'/filename;np.save(p,reference(c['t'],c['bc'],14,2,c['causal']))
  refs[key]=np.load(p);frozen[str(p)]=sha(p);save(OUT/'inputs.json',frozen)
 def run(c,ticket):
  path=OUT/'runs'/name(c);path.mkdir();proc=None;start=time.monotonic();simcap=6*GIB
  try:
   unchanged();mode=c['mode'];spm=mode in (1,3,4,5);source=OLD/'runs'/('t4096_private_spm' if spm else 't4096_cache');cmd=read(source/'command.json');cmd[1]='--outdir='+str(path)
   args=list(map(str,[c['t'],8,16,c['bc'],c['r']//16,1024,0,int(spm),1,1,14,2,int(c['causal']),mode]))
   cmd[cmd.index('--cmd')+1]=str(GUEST/'guest.export-safe');cmd[cmd.index('--options')+1]=' '.join(args)
   for i,x in enumerate(cmd):
    if x.startswith('system.cpu[0].workload[0].cmd='):cmd[i]='system.cpu[0].workload[0].cmd='+repr(['kv-fa']+args)
   save(path/'command.json',cmd);save(path/'case.json',c)
   with (path/'run.log').open('w') as log:
    proc=subprocess.Popen(['prlimit',f'--as={simcap}:','--','nice','-n','10',*cmd],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
    with lock:state['active'][name(c)]=dict(pid=proc.pid,path=str(path),lease=ticket,simulator_as=simcap);status()
    paused=False
    while proc.poll() is None:
     assert not stop.is_set(),'peer gate failed';assert time.monotonic()-start<96*3600,'96h timeout';assert shutil.disk_usage(OUT).free>75*GIB,'disk guard'
     ps=owned().get(proc.pid);hold=pressure()
     if ps and ps['vms']>=simcap-2*GIB:
      with broker.locked() as b:
       if not hold and broker.total(b)+4*GIB<=b['cap']:
        b['leases'][ticket]['bytes']+=4*GIB;simcap+=4*GIB;resource.prlimit(proc.pid,resource.RLIMIT_AS,(simcap,resource.getrlimit(resource.RLIMIT_AS)[1]));state['active'][name(c)]['simulator_as']=simcap
       else:hold=True
     # Respect the shared supervisor's all-user pressure decision too.
     br=read(ROOT/'results/parallel_overlap_20260930/broker.json');hold|=br.get('hold_new',False) and pressure()
     if hold!=paused:proc.send_signal(signal.SIGSTOP if hold else signal.SIGCONT);paused=hold
     with lock:state['active'][name(c)]['paused']=paused;status()
     try:proc.wait(timeout=5)
     except subprocess.TimeoutExpired:pass
   logtext=(path/'run.log').read_text();assert proc.returncode==0 and 'KV_ROI_END' in logtext and 'because exiting with last active thread context' in logtext,logtext[-2500:]
   assert not re.search(r'panic:|fatal:|Simulated exit code not 0|\bFAIL\b',logtext)
   def fields(p):return dict(leaves(read(p/'config.json')))
   old,new=fields(source),fields(path);delta={k:[old.get(k),new.get(k)] for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
   # Output paths are present in some config strings; normalize only those paths.
   old=dict(leaves(json.loads((source/'config.json').read_text().replace(str(source),'<OUT>'))));new=dict(leaves(json.loads((path/'config.json').read_text().replace(str(path),'<OUT>'))));delta={k:[old.get(k),new.get(k)] for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
   assert not {k:v for k,v in delta.items() if not re.fullmatch(r'system.cpu\[\d+\].workload\[0\].(cmd\[\d+\]|executable)',k)},delta
   expected=work(c);actual={int(i):(int(n),int(u)) for i,n,u in re.findall(r'GQA_WORK core=(\d+) tasks=(\d+) row_updates=(\d+)',logtext)}
   assert actual=={i:(expected['tasks'][i],expected['updates'][i]) for i in range(8)},(actual,expected)
   ep=stats(path);s=ep[0];total=lambda suffix:sum(v for k,v in s.items() if k.endswith(suffix));copies=sum(expected['copies'])
   assert total('.m_spmcpRequestsIssued')==total('.m_spmcpRequestsCompleted')==copies,(total('.m_spmcpRequestsIssued'),copies)
   assert total('.reductionAccepted')==total('.reductionCompleted')==total('.l3ReductionStarts')==total('.l3ReductionFinishes')==0
   for e in ep:
    check_global_guard(e)
    for i in range(8):
     assert not new[f'system.ruby.l2_cntrl{i}.l3_reduction_enabled'];assert new[f'system.ruby.l1_cntrl{i}.cache.size']==524288
     for domain in ('regular','spm'):
      for k,cap in [('Load',32),('Store',48)]:assert e[f'system.cpu{i}.{domain}Lsq0.boundedMax{k}Occupancy']<=cap
   export_hash(path/'output.fp16',logtext);arithmetic=validate(path/'output.fp16',refs[(c['t'],c['bc'],c['causal'])],c['t'],14,2,c['causal'],dense=c['t']<=1024)
   digest=sha(path/'output.fp16')
   with lock:
    for r in state['completed']:
     if (r['case']['t'],r['case']['bc'],r['case']['causal'])==(c['t'],c['bc'],c['causal']):assert r['output_sha256']==digest,'same-Bc output bits changed'
   unchanged();row=dict(name=name(c),case=c,path=str(path),roi_us=s['simTicks']/1e6,copies=copies,commands=0,arithmetic=arithmetic,work=expected,output_sha256=digest,config_delta=delta,host_seconds=time.monotonic()-start)
   save(path/'accepted.json',row)
   with lock:state['completed'].append(row);status()
   return row
  except BaseException as e:
   stop.set();save(path/'failed.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
  finally:
   if proc and proc.poll() is None:
    proc.terminate()
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()
   with lock:state['active'].pop(name(c),None);status()
 def phase(label,cases):
  for c in cases:getref(c)
  state['stage']=label;status();pending=list(cases);futures={}
  with cf.ThreadPoolExecutor(max_workers=2) as pool:
   try:
    while pending or futures:
     for f in list(futures):
      if f.done():
       ticket=futures.pop(f)
       try:f.result()
       finally:broker.release(ticket)
     if pending and len(futures)<2 and not pressure():
      c=pending[0];ticket='fa33_'+name(c)
      if acquire(ticket):pending.pop(0);futures[pool.submit(run,c,ticket)]=ticket
     status();time.sleep(3)
   finally:
    if futures:stop.set()
    for f,ticket in list(futures.items()):
     try:f.result()
     finally:broker.release(ticket)
 try:
  unchanged();save(OUT/'inputs.json',frozen);save(OUT/'plan.json',dict(ledger='FA-33',max_parallel=2,resource='shared OPS-23 broker,6GiB initial simulator+2GiB margin per8GiB lease; proactive growth;24CPUs',modes=NAMES,ring_next='FA-34 dependent, not yet implemented'))
  for t,bc in [(256,128),(1024,256),(4096,256)]:
   cases=[dict(t=t,bc=bc,mode=m,r=(128 if t<=1024 else 256) if m<2 else t//8,causal=True) for m in (0,4,2,5,1,3)]
   phase(f't{t}_'+('gates' if t==256 else 'factorial'),cases)
   if t==256:save(OUT/'small_gates_passed.json',dict(cases=6,full_oracle=True,peer_bits=True,config=True,copy_counts=True))
  state['stage']='completed_validated';status();save(OUT/'complete.json',dict(cases=len(state['completed'])))
  effects=[]
  for t in (1024,4096):
   arms={r['case']['mode']:r for r in state['completed'] if r['case']['t']==t}
   for m in range(2,6):effects.append(dict(t=t,mode=NAMES[m],roi_us=arms[m]['roi_us'],vs_legacy_private=arms[1]['roi_us']/arms[m]['roi_us'],vs_same_layout_cache=arms[2]['roi_us']/arms[m]['roi_us']))
  save(OUT/'effects.json',effects)
  save(BASE/'next_stage.json',dict(stage='FA-34 headroom evaluation ready for implementation on accepted FA-33 layout',authorized=True,prerequisites_passed=True,results=effects))
 except BaseException as e:
  state['error']=repr(e);state['traceback']=traceback.format_exc();state['stage']='failed';status();raise
if __name__=='__main__':main()
