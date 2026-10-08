"""Llama FA adaptive local tuning, exact retained guest and simulator."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import ast,json,time,hashlib,subprocess,resource,signal,fcntl,traceback,re,sys,shutil
from pathlib import Path
R=Path('/data2/rsappidi/cacheflex_mc');H=Path(__file__).resolve().parent
sys.path[:0]=[str(R/'scripts'),str(R/'benchmarks/cacheflex_flash_attention'),str(R/'results/parallel_overlap_20260930')]
import broker,numpy as np
from plan import case,name,parameters,describe,argv,gate,choose,MAX_TARGETS
from report import counters,write as report
from run_access_timing import stats
from fa_headroom_profiles import leaves
from test_micro_timing import check_global_guard
from export_safe_elf import export_hash
from qwen_l3_reference import reference,validate
from parallel_budget_scheduler import owned
BASE=R/'results/llama_fa_iterative_20261004';G=R/'results/cacheflex_flash_attention/qwen_matrix_20260930/guest_v3';OUT=BASE/'experiment_v1';GIB=2**30
PARENT=R/'results/retained_protocol_restart_20261002/fa_matrix_spm_only/runs/t256_q14_legacy_v_br16_bc256_r256'
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib')
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def pressure():
 monitor=read(R/'results/retained_protocol_restart_20261002/spm_only_resources.json')
 mem={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()}
 # Host coordinator accounts all host processes; sandbox RSS is added conservatively.
 rss=monitor['user_rss_bytes']+sum(v['rss'] for v in owned().values());cap=monitor['ram_limit_bytes']
 return time.time()-monitor['updated']>30 or rss>cap-2*GIB or mem['MemAvailable']<64*GIB+max(0,cap-rss)
def config_check(path,c,cmd):
 old=dict(leaves(read(PARENT/'config.json')));new=dict(leaves(read(path/'config.json')))
 expected=dict(old)
 # Normalize only output-root redirects, then explicitly set expected guest/argv.
 for k,v in list(expected.items()):
  if isinstance(v,str):expected[k]=v.replace(str(PARENT),str(path))
 if c['kind']=='fa':
  expected.update(parameters(c))
  expected['system.cpu[0].workload[0].executable']=str(G/'guest.export-safe')
  args=['kv-fa',*map(str,argv(c))]
  for i,x in enumerate(args):expected[f'system.cpu[0].workload[0].cmd[{i}]']=x
 assert new==expected,{k:[expected.get(k),new.get(k)] for k in expected.keys()|new.keys() if expected.get(k)!=new.get(k)}
 assert cmd[cmd.index('--cmd')+1]==new['system.cpu[0].workload[0].executable']
 return {k:[old.get(k),new.get(k)] for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
def command(c,path):
 cmd=read(PARENT/'command.json');cmd[1]='--outdir='+str(path)
 if c['kind']=='fa':
  cmd[cmd.index('--cmd')+1]=str(G/'guest.export-safe');cmd[cmd.index('--options')+1]=' '.join(map(str,argv(c)))
  cmd[cmd.index('--l1d_num_spm_ways=6')]='--l1d_num_spm_ways='+str(6 if c['spm'] else 0)
  for i,x in enumerate(cmd):
   if x.startswith('system.cpu[0].workload[0].cmd='):cmd[i]='system.cpu[0].workload[0].cmd='+repr(['kv-fa',*map(str,argv(c))])
  for k,v in parameters(c).items():
   found=[i for i,x in enumerate(cmd) if x.startswith(k+'=')]
   if found:
    for i in found:cmd[i]=k+'='+repr(v)
   else:cmd+=['-P',k+'='+repr(v)]
 return cmd
def main():
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'runs').mkdir();(OUT/'references').mkdir()
 lock=(OUT/'coordinator.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 budget=read(R/'results/parallel_budget_20260929/budget.json');os.sched_setaffinity(0,budget['cpu_set'][:2]);os.nice(10)
 resource.setrlimit(resource.RLIMIT_AS,(GIB//2,resource.RLIM_INFINITY))
 state=dict(pid=os.getpid(),stage='preparing',active={},pending=[],completed=[],failed=[],decisions=[],backend_status={})
 def status():state['updated']=time.time();save(OUT/'status.json',state)
 status()
 idle='fa41_coordinator_idle'
 while True:
  with broker.locked() as ledger:
   if time.time()-ledger['heartbeat']<20 and not ledger['hold_new'] and broker.total(ledger)+GIB//2<=ledger['cap']:
    assert idle not in ledger['leases'];ledger['leases'][idle]=dict(bytes=GIB//2,owner=os.getpid(),family='FA-41 idle coordinator');break
  state['stage']='waiting_for_coordinator_reservation';status();time.sleep(5)
 save(OUT/'plan.json',dict(model='Llama-3.2-1B',target=case(False),method='measure, explain, one-variable test, retain/recenter; no Cartesian placement sweep',
  max_target_cases_per_backend=MAX_TARGETS,hardware='retained finite Qwen profile, unchanged',source_ledger=['FA-35','FA-39','MOD-12','FA-30'],
  acceptance='full oracle, dense small, same-Bc exact bits, work/copy/config/resource checks; confidence requires final review'))
 frozen=read(G/'frozen.json');frozen.update(read(R/'results/retained_protocol_restart_20261002/build/frozen.json'))
 for f in [*H.glob('*.py'),*PARENT.glob('*.json'),Path(read(PARENT/'command.json')[0]),Path(read(PARENT/'command.json')[2]),BASE/'MODEL_AND_METHOD.txt']:
  if f.is_file():frozen[str(f)]=sha(f)
 for mod in (broker,sys.modules['run_access_timing'],sys.modules['fa_headroom_profiles'],sys.modules['qwen_l3_reference'],sys.modules['fa_independent_reference'],sys.modules['export_safe_elf'],sys.modules['test_micro_timing'],sys.modules['parallel_budget_scheduler']):
  f=Path(mod.__file__);frozen[str(f)]=sha(f)
 frozen[str(G/'guest.export-safe')]=sha(G/'guest.export-safe')
 save(OUT/'inputs.json',frozen)
 def unchanged():
  for p,h in frozen.items():assert sha(p)==h,('frozen input drift',p)
 def acquire(ticket):
  while True:
   state['stage']='waiting_for_reservation';status()
   with broker.locked() as s:
    if time.time()-s['heartbeat']<20 and not s['hold_new'] and not pressure() and broker.total(s)+10*GIB-s['leases'][idle]['bytes']<=s['cap']:
     assert ticket not in s['leases'];del s['leases'][idle];s['leases'][ticket]=dict(bytes=10*GIB,owner=os.getpid(),family='FA-41 Llama')
     resource.setrlimit(resource.RLIMIT_AS,(2*GIB,resource.RLIM_INFINITY));return
   time.sleep(5)
 def execute(c,hypothesis):
  n=name(c)
  previous=next((a for a in state['completed'] if a['name']==n),None)
  if previous:return previous
  describe(c);state['pending']=[n];ticket='fa41_'+n;acquire(ticket)
  path=OUT/'runs'/n;path.mkdir();proc=None
  try:
   unchanged();cmd=command(c,path);save(path/'command.json',cmd);limit=8*GIB
   def child():resource.setrlimit(resource.RLIMIT_AS,(limit,resource.RLIM_INFINITY))
   with (path/'run.log').open('w') as log:proc=subprocess.Popen(cmd,cwd=R,env=ENV,stdout=log,stderr=subprocess.STDOUT,preexec_fn=child)
   state['stage']='running_'+n;state['active']={n:dict(pid=proc.pid,path=str(path),paused=False,simulator_as=limit)};status();start=time.monotonic();paused=False
   while proc.poll() is None:
    assert time.monotonic()-start<96*3600,'case timeout'
    hold=pressure();info=owned().get(proc.pid)
    if info and info['vms']>=limit-GIB:
     with broker.locked() as s:
      if not hold and broker.total(s)+4*GIB<=s['cap']:
       s['leases'][ticket]['bytes']+=4*GIB;limit+=4*GIB;resource.prlimit(proc.pid,resource.RLIMIT_AS,(limit,resource.RLIM_INFINITY))
      else:hold=True
    if hold!=paused:proc.send_signal(signal.SIGSTOP if hold else signal.SIGCONT);paused=hold
    state['active'][n].update(paused=paused,simulator_as=limit);status()
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:pass
   log=(path/'run.log').read_text();assert proc.returncode==0 and 'because exiting with last active thread context' in log,log[-3000:]
   assert not re.search(r'panic:|fatal:|\bFAIL\b',log),log[-3000:]
   delta=config_check(path,c,cmd);blocks=stats(path);s=blocks[0]
   assert len(blocks)>=1 and s['simTicks']>0
   for block in blocks:
    check_global_guard(block)
    for i in range(8):
     for domain in ('regular','spm'):
      for kind,cap in [('Load',32),('Store',48)]:assert block[f'system.cpu{i}.{domain}Lsq0.boundedMax{kind}Occupancy']<=cap
   state['stage']='validating_'+n;state['active']={};status()
   ref=OUT/'references'/f't{c["t"]}_bc{c["bc"]}_q32_kv8.npy'
   if not ref.exists():
    def progress(head,total):state['reference_progress']=dict(head=head,total=total);status()
    np.save(ref,reference(c['t'],c['bc'],32,8,True,progress=progress))
   frozen[str(ref)]=sha(ref);save(OUT/'inputs.json',frozen)
   export_hash(path/'output.fp16',log);arithmetic=validate(path/'output.fp16',np.load(ref),c['t'],32,8,True,dense=c['t']<=512)
   w=describe(c);actual={int(i):(int(t),int(u)) for i,t,u in re.findall(r'GQA_WORK core=(\d+) tasks=(\d+) row_updates=(\d+)',log)}
   assert actual=={i:(w['tasks'][i],w['updates'][i]) for i in range(8)},actual
   copies=sum(v for k,v in s.items() if k.endswith('.m_spmcpRequestsIssued'))
   assert copies==sum(w['copies'])
   assert copies==sum(v for k,v in s.items() if k.endswith('.m_spmcpRequestsCompleted'))
   assert sum(v for k,v in s.items() if k.endswith(('.reductionAccepted','.l3ReductionStarts')))==0
   assert f'GQA_CONFIG hq=32 hkv=8 d=64 batch=1 causal=1 tasks={sum(w["tasks"])}' in log
   assert f'QO_CONFIG mode={int(c["spm"])} rows_per_core={c["t"]//8}' in log
   digest=sha(path/'output.fp16')
   for prev in state['completed']:
    if prev['case']['t']==c['t'] and prev['case']['bc']==c['bc']:assert prev['output_sha256']==digest,'same-Bc exact peer bits changed'
   row=dict(name=n,case=c,path=str(path),outcome='full_fa_pass',roi_us=s['simTicks']/1e6,arithmetic=arithmetic,copies=copies,
    output_sha256=digest,config_delta=delta,work=w,counters=counters(s),hypothesis=hypothesis,
    evidence_sha256={f:sha(path/f) for f in ('stats.txt','config.json','command.json','run.log')})
   unchanged();save(path/'accepted.json',row);state['completed'].append(row);state['pending']=[];state.pop('reference_progress',None);status();report(state)
   return row
  finally:
   if proc is not None and proc.poll() is None:
    proc.terminate();proc.send_signal(signal.SIGCONT)
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()
   with broker.locked() as ledger:
    del ledger['leases'][ticket];ledger['leases'][idle]=dict(bytes=2*GIB,owner=os.getpid(),family='FA-41 idle coordinator')
 try:
  # Both correctness controls precede targets, so backend bugs fail early.
  for spm in (False,True):execute(gate(case(spm)),dict(reason='Full-head Llama correctness before target timing; no new guest.',status='gate'))
  while True:
   any_case=False
   for spm in (False,True):
    c,decision=choose(state['completed'],spm);state['backend_status']['spm' if spm else 'cache']=decision
    if c is None:continue
    any_case=True;decision=dict(decision,case=c,decision_index=len(state['decisions']))
    state['decisions'].append(decision);save(OUT/'decisions.json',state['decisions']);status()
    execute(gate(c),dict(reason='Validate chosen geometry, all32heads and exact percore work before target.',parent_decision=decision))
    execute(c,decision)
   if not any_case:break
  state['stage']='awaiting_final_review';status();report(state)
 except BaseException as e:
  state['stage']='failed';state['active']={};state['failed'].append(dict(error=repr(e),traceback=traceback.format_exc()));status();raise
 finally:
  broker.release(idle)
if __name__=='__main__':main()
