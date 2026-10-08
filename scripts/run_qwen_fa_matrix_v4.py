"""FA-35 bounded placement/geometry matrix; all checks and shared admission retained."""
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
BASE=ROOT/'results/cacheflex_flash_attention/qwen_matrix_20260930';OUT=BASE/os.environ.get('MATRIX_EXPERIMENT','experiment_v4');GUEST=BASE/os.environ.get('MATRIX_GUEST','guest_v3')
OLD=ROOT/'results/cacheflex_l3_spm/direct_link_20260928/experiment_v2';GIB=2**30
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',PYTHONDONTWRITEBYTECODE='1')
def acquire(ticket):
 # Same atomic accounting, permitting at most five additional FA leases when
 # full reservations fit. Existing broker/source remains byte-identical.
 with broker.locked() as b:
  if time.time()-b['heartbeat']>20:
   # Continue the SAME ledger only after the old owner has exited cleanly.
   # Retain its conservative reservation; never erase another family's leases.
   old=read(ROOT/'results/parallel_memory_recovery_20260929/status.json')
   proc=owned().get(old['pid'])
   if ((ROOT/'results/parallel_overlap_20260930/complete.json').exists() and
       (proc is None or proc['state']=='Z') and
       all(k.startswith('fa35_') for k in b['leases'])):
    b['heartbeat']=time.time();b['hold_new']=False;b['continuation_owner']=os.getpid()
  if (time.time()-b['heartbeat']>20 or b['hold_new'] or
      sum(k.startswith('fa35_') for k in b['leases'])>=8 or
      len(b['leases'])>=8 or broker.total(b)+8*GIB>b['cap']):return False
  assert ticket not in b['leases'];b['leases'][ticket]=dict(bytes=8*GIB,owner=os.getpid(),acquired=time.time(),family='FA-35')
  return True
from qwen_fa_matrix_plan import case,NAMES,name,describe,refkey,arithmetic_class,anchors,axial,gate,interactions,neighbors
from qwen_kv_reference import reference as kv_reference
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def work(c):
 x=describe(c);return {k:x[k] for k in ('tasks','updates','copies')}
def main():
 OUT.mkdir(parents=True);lockfile=(OUT/'coordinator.lock').open('w');fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'status.json').exists()
 (OUT/'runs').mkdir();(OUT/'references').mkdir();budget=read(ROOT/'results/parallel_budget_20260929/budget.json');os.sched_setaffinity(0,budget['cpu_set']);resource.setrlimit(resource.RLIMIT_AS,(2*GIB,resource.getrlimit(resource.RLIMIT_AS)[1]))
 lock=threading.RLock();stop=threading.Event();state=dict(pid=os.getpid(),stage='preparing',active={},completed=[],scope='bounded T4096 matrix; no global-optimum claim',pending=[])
 frozen=read(GUEST/'frozen.json');frozen.update(read(OLD/'inputs.json'))
 for p in [Path(__file__),ROOT/'scripts/build_qwen_fa_matrix_v3.py',ROOT/'scripts/test_qwen_fa_matrix_v3.py',ROOT/'scripts/qwen_fa_matrix_plan.py',ROOT/'scripts/qwen_kv_reference.py',ROOT/'docs/experiments/QWEN_FA_MATRIX.txt']:
  frozen[str(p)]=sha(p)
 for mod in ('qwen_l3_reference','fa_independent_reference','run_access_timing','fa_headroom_profiles','test_micro_timing','export_safe_elf','parallel_budget_scheduler','broker'):
  p=Path(importlib.import_module(mod).__file__);frozen[str(p)]=sha(p)
 for arm in ('cache','private','noc'):
  for n in ('command.json','config.json'):p=OLD/'runs'/f't4096_{arm}'/n;frozen[str(p)]=sha(p)
 def status():
  with lock:
   state['updated']=time.time();save(OUT/'status.json',state)
   lines=['# FA-35 accepted Qwen placement/geometry results','',f"Stage: {state['stage']}",'','Modes 0-7 and9-11 CPU arithmetic; mode8 matched existing L3 reduction. Gates are not target conclusions.','', '| Case | Full FA us | SPM copies |','|---|---:|---:|']
   lines += [f"| {r['name']} | {r['roi_us']:.4f} | {r['copies']} |" for r in state['completed']]
   (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
 def unchanged():
  for p,h in list(frozen.items()):assert sha(p)==h,('frozen changed',p)
 sample_lock=threading.Lock();previous=[];last_cpu=[0.]
 def pressure():
  procs=owned();rss=sum(x['rss'] for x in procs.values());avail=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
  with sample_lock:
   now=time.monotonic();ticks={(pid,p['start_ticks']):p['cpu_ticks'] for pid,p in procs.items()}
   if not previous:previous.extend([now,ticks])
   elif now-previous[0]>=1:
    delta=sum(max(0,v-previous[1].get(k,v)) for k,v in ticks.items())
    last_cpu[0]=100*delta/os.sysconf('SC_CLK_TCK')/(now-previous[0])/os.cpu_count();previous[:]=[now,ticks]
   hold=last_cpu[0]>20 or rss>budget['ram_limit']-2*GIB or avail<64*GIB+max(0,budget['ram_limit']-rss)
   save(OUT/'resources.json',dict(updated=time.time(),cpu_pct_of_host=last_cpu[0],user_rss_bytes=rss,ram_limit_bytes=budget['ram_limit'],available_bytes=avail,pressure=hold,max_new_simulators=8))
  return hold
 refs={}
 def getref(c):
  key=refkey(c)
  if key in refs:return
  t,bc,hq,hkv,causal,kind=key
  filename=f't{t}_bc{bc}_q{hq}_kv{hkv}_causal{int(causal)}.npy'
  p=ROOT/'results/cacheflex_l3_spm/qwen_gqa_20260928/experiment_v2/references'/filename
  if kind=='kv' or not p.exists():
   p=OUT/'references'/(kind+'_'+filename)
   if not p.exists():np.save(p,(kv_reference if kind=='kv' else reference)(t,bc,hq,hkv,causal))
  refs[key]=np.load(p);frozen[str(p)]=sha(p);save(OUT/'inputs.json',frozen)
 def run(c,ticket):
  path=OUT/'runs'/name(c);path.mkdir();proc=None;start=time.monotonic();simcap=6*GIB
  try:
   unchanged();mode=c['mode'];spm=mode not in (0,2,9);source=OLD/'runs'/('t4096_noc' if mode==8 else ('t4096_private' if spm else 't4096_cache'));cmd=read(source/'command.json');cmd[1]='--outdir='+str(path)
   args=list(map(str,[c['t'],8,c['br'],c['bc'],c['r']//c['br'],1024,0,int(spm),5 if mode==8 else 1,1,c['hq'],c['hkv'],int(c['causal']),mode]))
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
   commands=describe(c)['commands']
   assert total('.reductionAccepted')==total('.reductionCompleted')==total('.l3ReductionStarts')==total('.l3ReductionFinishes')==commands
   assert total('.l3ReductionHalfCommands')==commands
   assert total('.l3ReductionInitializations')==(c['hq']*c['t'] if mode==8 else 0)
   assert total('.l3Finalizations')==0
   assert total('.reductionTransportBytes')==144*commands
   for e in ep:
    check_global_guard(e)
    for i in range(8):
     assert bool(new[f'system.ruby.l2_cntrl{i}.l3_reduction_enabled'])==(mode==8);assert new[f'system.ruby.l1_cntrl{i}.cache.size']==524288
     assert new[f'system.ruby.l2_cntrl{i}.L2cache.size']==524288
     for q in ('Source','Sink'):assert e[f'system.ruby.l0_cntrl{i}.sequencer.reductionMax{q}']<=4
     assert e[f'system.ruby.l2_cntrl{i}.L2cache.l3ReductionMaxActive']<=(4 if new[f'system.ruby.l2_cntrl{i}.l3_reduction_pipeline'] else 1)
     for domain in ('regular','spm'):
      for k,cap in [('Load',32),('Store',48)]:assert e[f'system.cpu{i}.{domain}Lsq0.boundedMax{k}Occupancy']<=cap
   export_hash(path/'output.fp16',logtext);arithmetic=validate(path/'output.fp16',refs[refkey(c)],c['t'],c['hq'],c['hkv'],c['causal'],dense=c['t']<=1024)
   digest=sha(path/'output.fp16')
   with lock:
    for r in state['completed']:
     if refkey(r['case'])==refkey(c):assert r['output_sha256']==digest,'same-Bc output bits changed'
   unchanged();row=dict(name=name(c),case=c,path=str(path),roi_us=s['simTicks']/1e6,copies=copies,commands=commands,arithmetic_class=arithmetic_class(c),arithmetic=arithmetic,work=expected,output_sha256=digest,config_delta=delta,host_seconds=time.monotonic()-start)
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
  state['stage']=label;state['pending']=[name(c) for c in cases];status();pending=list(cases);futures={}
  with cf.ThreadPoolExecutor(max_workers=8) as pool:
   try:
    while pending or futures:
     for f in list(futures):
      if f.done():
       ticket=futures.pop(f)
       try:f.result()
       finally:broker.release(ticket)
     if pending and len(futures)<8 and not pressure():
      c=pending[0];ticket='fa35_'+name(c)
      getref(c)
      if acquire(ticket):pending.pop(0);state['pending'].remove(name(c));futures[pool.submit(run,c,ticket)]=ticket
     status();time.sleep(3)
   finally:
    if futures:stop.set()
    errors=[]
    for f,ticket in list(futures.items()):
     try:f.result()
     except BaseException as error:errors.append(error)
     finally:broker.release(ticket)
    if errors:raise errors[0]
 def unique_pending(cases):
  seen={r['name'] for r in state['completed']};result=[]
  for c in cases:
   describe(c)
   if name(c) not in seen:result.append(c);seen.add(name(c))
  return result
 def gated_phase(label,cases):
  cases=unique_pending(cases)
  gates=unique_pending([gate(c) for c in cases])
  save(OUT/(label+'_plan.json'),dict(cases=cases,gates=gates,reason='exact geometry and equal family budgets; no MOD-12 timing ranking'))
  phase(label+'_gates',gates)
  phase(label+'_targets',unique_pending(cases))
  summarize()
 def summarize():
  target=[r for r in state['completed'] if r['case']['t']==4096 and r['case']['hq']==14]
  winners=[]
  for m in range(12):
   rr=[r for r in target if r['case']['mode']==m]
   if rr:winners.append(min(rr,key=lambda x:x['roi_us']))
  save(OUT/'winners.json',dict(stage=state['stage'],winners=winners,scope='best measured within bounded matrix',prior_private_us=9473.3184,prior_cache_us=10509.4604))
 try:
  unchanged();save(OUT/'inputs.json',frozen);screen,rejected=axial()
  save(OUT/'plan.json',dict(ledger='FA-35',max_parallel=8,resource='shared OPS-23 broker; 20%CPU/RAM; finite AS reservations and growth',
       anchors=anchors(),axial=screen,rejected=rejected,interaction_budget_per_family=4,neighbor_budget_per_family=2,
       maximum_target_cases=12+len(screen)+48+24,all_target_T=4096,numerical_policy='same arithmetic class/Bc exact bits; KV separate full oracle',
       model_policy='MOD-12 timing ranking failed: exact accounting only',existing_results='FA-33/L3-11/12 configurations revalidated with same new ELF; no prior timing overwritten'))
  # Missing placements and their matched controls first, then other organizations.
  phase('full_capacity_q_boundary_gates',[case(m,t=4096,hq=2,r=512) for m in (4,6,5)])
  ordered=sorted(anchors(),key=lambda c:0 if c['mode']==4 else 1)
  gated_phase('placement_anchors',ordered)
  gated_phase('axial_geometry',screen)
  selected,audit=interactions(state['completed']);save(OUT/'interaction_selection.json',audit)
  gated_phase('measured_interactions',selected)
  selected,audit=neighbors(state['completed']);save(OUT/'neighbor_selection.json',audit)
  gated_phase('winner_neighbors',selected)
  state['stage']='completed_validated';state['pending']=[];status();summarize()
  save(OUT/'complete.json',dict(cases=len(state['completed']),scope='best within documented tested space; not global optimum'))
  with (ROOT/'docs/experiments/LEDGER.md').open('a') as f:
   f.write('\n**FA-35 completed:** '+str(len(state['completed']))+' accepted gates/targets; best measured per-family results: '+str(OUT/'winners.json')+'. No global-optimum claim.\n')
 except BaseException as e:
  state['error']=repr(e);state['traceback']=traceback.format_exc();state['stage']='failed';status();raise
if __name__=='__main__':main()
