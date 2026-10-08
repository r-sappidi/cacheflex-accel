"""FA-34 serial fail-closed queue: isolated link, exact controls, directed gates, FA pairs."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',CCACHE_DISABLE='1')
import json,sys,time,hashlib,subprocess,resource,signal,re,fcntl,traceback,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'benchmarks/cacheflex_flash_attention'),str(ROOT/'results/parallel_overlap_20260930')]
import broker
import numpy as np
from run_access_timing import stats
from fa_headroom_profiles import leaves
from test_micro_timing import check_global_guard
from export_safe_elf import export_hash
from qwen_l3_reference import reference as ordinary_reference,validate
from reference import reference as ring_reference,panels
from parallel_budget_scheduler import owned
BASE=ROOT/'results/cacheflex_flash_attention/v_only_ring_20261003';B=ROOT/'results/cacheflex_flash_attention/optimistic_ring_20261001/build_retained_v1';G=BASE/'guest_v1';OUT=BASE/'experiment_v1';GIB=2**30
OLD=ROOT/'results/retained_protocol_restart_20261002/fa_matrix_spm_only/runs'
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def parent():return OLD/'t256_q14_legacy_v_br16_bc256_r256'
def pressure():
 mem={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()}
 rss=sum(x['rss'] for x in owned().values());snapshot=ROOT/'results/retained_protocol_restart_20261002/spm_only_resources.json';rss+=read(snapshot).get('user_rss_bytes',0);cap=read(ROOT/'results/parallel_budget_20260929/budget.json')['ram_limit']
 return rss>cap-2*GIB or mem['MemAvailable']<64*GIB+max(0,cap-rss)
def main():
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'runs').mkdir();(OUT/'references').mkdir()
 lock=(OUT/'coordinator.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 os.nice(10);budget=read(ROOT/'results/parallel_budget_20260929/budget.json');os.sched_setaffinity(0,budget['cpu_set'][:2])
 # Coordinator + reference arithmetic bounded separately inside each reservation.
 resource.setrlimit(resource.RLIMIT_AS,(2*GIB,resource.getrlimit(resource.RLIMIT_AS)[1]))
 state=dict(pid=os.getpid(),stage='preparing',active={},completed=[],pending=[],failed=[])
 def status():state['updated']=time.time();save(OUT/'status.json',state)
 frozen={}
 for f in [*HERE.glob('*.py'),*HERE.glob('*.hpp'),*HERE.glob('*.hh'),*HERE.glob('*.cpp'),*HERE.glob('*.sh'),*G.iterdir(),B/'pseudo_inst.cc',B/'pseudo_inst.o',B/'optimistic_hook.hh',B/'compile_command.json']:
  if f.is_file():frozen[str(f)]=sha(f)
 frozen.update(read(G/'frozen.json'))
 frozen.update(read(ROOT/'results/retained_protocol_restart_20261002/build/frozen.json'))
 for p in parent().glob('*'):
  if p.is_file():frozen[str(p)]=sha(p)
 assert read(parent()/'accepted.json')['case']['mode']==1
 for m in ('run_access_timing','fa_headroom_profiles','test_micro_timing','export_safe_elf','qwen_l3_reference','fa_independent_reference','parallel_budget_scheduler','broker'):
  f=Path(sys.modules[m].__file__);frozen[str(f)]=sha(f)
 def unchanged():
  for p,h in frozen.items():assert sha(p)==h,('frozen input drift',p)
 def acquire(ticket,size,build=False):
  while True:
   state['stage']='waiting_for_build_reservation' if build else 'waiting_for_simulation_reservation';status()
   with broker.locked() as ledger:
    # Keep the existing ledger alive after its original supervisor finishes.
    if time.time()-ledger['heartbeat']>20 and (ROOT/'results/parallel_overlap_20260930/complete.json').exists():
     old=read(ROOT/'results/parallel_memory_recovery_20260929/status.json');p=owned().get(old['pid'])
     if p is None or p['state']=='Z':ledger['heartbeat']=time.time();ledger['hold_new']=pressure();ledger['continuation_owner']=os.getpid()
    if time.time()-ledger['heartbeat']<20 and not pressure() and (build or not ledger['hold_new']) and broker.total(ledger)+size<=ledger['cap']:
     assert ticket not in ledger['leases'];ledger['leases'][ticket]=dict(bytes=size,owner=os.getpid(),family='FA-36',acquired=time.time());return
   time.sleep(5)
 def launch(cmd,path,ticket,limit,env):
  path.mkdir(exist_ok=True);save(path/'command.json',cmd)
  def child():resource.setrlimit(resource.RLIMIT_AS,(limit,resource.RLIM_INFINITY))
  with (path/'run.log').open('w') as log:
   proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,preexec_fn=child)
   state['active']={path.name:dict(pid=proc.pid,path=str(path),paused=False,simulator_as=limit)};status();start=time.monotonic();paused=False
   try:
    while proc.poll() is None:
     assert time.monotonic()-start<96*3600,'96h timeout'
     hold=pressure();info=owned().get(proc.pid)
     if info and info['vms']>=limit-GIB:
      with broker.locked() as l:
       if not hold and broker.total(l)+4*GIB<=l['cap']:
        l['leases'][ticket]['bytes']+=4*GIB;limit+=4*GIB
        # The inherited hard ceiling is deliberately unlimited; child soft cap grows.
        resource.prlimit(proc.pid,resource.RLIMIT_AS,(limit,resource.RLIM_INFINITY))
       else:hold=True
     if hold!=paused:proc.send_signal(signal.SIGSTOP if hold else signal.SIGCONT);paused=hold
     state['active'][path.name].update(paused=paused,simulator_as=limit);status()
     try:proc.wait(timeout=5)
     except subprocess.TimeoutExpired:pass
    return proc.returncode,(path/'run.log').read_text()
   finally:
    if proc.poll() is None:
     proc.terminate();proc.send_signal(signal.SIGCONT)
     try:proc.wait(timeout=10)
     except subprocess.TimeoutExpired:proc.kill();proc.wait()
    state['active']={};status()
 def command(name,args=None,exe=None,cache=False):
  source=OLD/('t256_q14_legacy_cache_br16_bc256_r256' if cache else 't256_q14_legacy_v_br16_bc256_r256')
  cmd=read(source/'command.json');cmd[0]=str(B/'gem5.opt');cmd[1]='--outdir='+str(OUT/'runs'/name)
  if exe:
   cmd[cmd.index('--cmd')+1]=str(exe);cmd[cmd.index('--options')+1]=' '.join(map(str,args))
   for i,x in enumerate(cmd):
    if x.startswith('system.cpu[0].workload[0].cmd='):cmd[i]='system.cpu[0].workload[0].cmd='+repr(['kv-fa',*map(str,args)])
  return cmd,source
 def config_check(path,source):
  old=dict(leaves(json.loads((source/'config.json').read_text().replace(str(source),'<OUT>'))));new=dict(leaves(json.loads((path/'config.json').read_text().replace(str(path),'<OUT>'))))
  delta={k:[old.get(k),new.get(k)] for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
  assert not {k:v for k,v in delta.items() if not re.fullmatch(r'system.cpu\[\d+\].workload\[0\].(cmd\[\d+\]|executable)',k)},delta
  for i in range(8):
   assert new[f'system.ruby.l1_cntrl{i}.cache.size']==524288
   assert not new[f'system.ruby.l2_cntrl{i}.l3_reduction_enabled']
  return delta
 def guards(path):
  for s in stats(path):
   check_global_guard(s)
   for i in range(8):
    for domain in ('regular','spm'):
     for typ,cap in [('Load',32),('Store',48)]:assert s[f'system.cpu{i}.{domain}Lsq0.boundedMax{typ}Occupancy']<=cap
  return stats(path)[0]
 plans=[]
 for t,bc,hq,label in [(1024,128,2,'gate'),(1024,128,14,'pilot'),(4096,256,14,'target')]:
  plans += [dict(name=f'{label}_t{t}_m{mode}',kind='fa',t=t,bc=bc,hq=hq,mode=mode) for mode in (1,16,17)]
 save(OUT/'plan.json',dict(cases=plans,max_parallel=1,scope='optimistic zero-time V-only neighbor transfer; coherent K; legacy task ownership; finite storage, timed barriers/CPU/copies',threshold='lower full ROI than same-schedule staging and strongest V-only'))
 state['pending']=[p['name'] for p in plans];status()
 try:
  # Existing accepted optimistic hook reused byte-for-byte; no new simulator build.
  frozen[str(B/'gem5.opt')]=sha(B/'gem5.opt')
  assert frozen[str(B/'gem5.opt')]==read(B/'status.json')['sha256']
  save(OUT/'inputs.json',frozen)
  for c in plans:
   ticket='fa36_'+c['name'];acquire(ticket,8*GIB)
   path=OUT/'runs'/c['name']
   try:
    unchanged();state['stage']=c['name'];status();env=dict(ENV)
    if c['kind']=='exact':cmd,source=command(c['name'],cache=c['cache'])
    elif c['kind']=='smoke':
     cmd,source=command(c['name'],[c['mode']],G/'smoke');env['CACHEFLEX_OPTIMISTIC_RING']='1'
    else:
     t,bc,hq,mode=c['t'],c['bc'],c['hq'],c['mode'];rows=128 if t==1024 else 256
     args=[t,8,16,bc,rows//16,1024,0,int(mode!=0),1,1,hq,2,1,mode]
     cmd,source=command(c['name'],args,G/'guest.export-safe',cache=mode==0)
     if mode>=12:env['CACHEFLEX_OPTIMISTIC_RING']='1'
    code,log=launch(cmd,path,ticket,6*GIB,env)
    actual_config=dict(leaves(read(path/'config.json')))
    assert actual_config['system.cpu[0].workload[0].executable']==str(G/'guest.export-safe')
    assert [actual_config[f'system.cpu[0].workload[0].cmd[{i}]'] for i in range(len(args)+1)]==['kv-fa',*map(str,args)]
    if c['kind']=='smoke' and c['mode']==2:
     assert code!=0 and 'Ring requires preclaimed source AND destination slots' in log,log[-2000:]
     row=dict(name=c['name'],outcome='expected_counterexample',case=c)
    else:
     assert code==0 and 'because exiting with last active thread context' in log,log[-3000:]
     assert not re.search(r'panic:|fatal:|\bFAIL\b',log),log[-3000:]
     delta=config_check(path,source);s=guards(path)
     row=dict(name=c['name'],case=c,path=str(path),config_delta=delta)
     if c['kind']=='exact':
      a=stats(source);b=stats(path);assert a==b,'default simulator stats changed'
      assert sha(source/'output.fp16')==sha(path/'output.fp16'),'default output changed'
      row.update(outcome='exact_default_control',roi_us=s['simTicks']/1e6)
     elif c['kind']=='smoke':
      assert 'RING_SMOKE PASS errors=0' in log
      assert 'OPTIMISTIC_RING rotations=32 bytes=4194304' in log
      row.update(outcome='directed_pass')
     else:
      refpath=OUT/'references'/f't{t}_bc{bc}_q{hq}_{"ring" if mode>=12 else "ordinary"}.npy'
      if not refpath.exists():np.save(refpath,(ring_reference if mode>=12 else ordinary_reference)(t,bc,hq,2,True))
      frozen[str(refpath)]=sha(refpath);save(OUT/'inputs.json',frozen)
      export_hash(path/'output.fp16',log);arithmetic=validate(path/'output.fp16',np.load(refpath),t,hq,2,True,dense=t<=1024)
      digest=sha(path/'output.fp16');copies=sum(v for k,v in s.items() if k.endswith('.m_spmcpRequestsIssued'))
      assert copies==sum(v for k,v in s.items() if k.endswith('.m_spmcpRequestsCompleted'))
      assert sum(v for k,v in s.items() if k.endswith(('.reductionAccepted','.l3ReductionStarts')))==0
      if mode>=12:
       rows=128 if t==1024 else 256;groups=t//rows
       expected=hq*groups*(t//4 if mode%2 else 2*t)
       assert copies==expected,(copies,expected)
       count=hq*(groups//8)*(t//(8*bc))*7 if mode%2 else 0
       moved=count*8*bc*128
       assert f'OPTIMISTIC_RING rotations={count} bytes={moved}' in log
      # Both ordinary and ring preserve the exact legacy query-group ownership/work.
      rows=128 if t==1024 else 256;groups=t//rows
      actual={int(i):(int(n),int(u)) for i,n,u in re.findall(r'GQA_WORK core=(\d+) tasks=(\d+) row_updates=(\d+)',log)}
      expected_tasks=[0]*8;expected_updates=[0]*8;legacy_copies=0
      for head in range(hq):
       for group in range(groups):
        core=(group-head)%groups%8;end=math.ceil(((group+1)*rows)/bc)*bc
        expected_tasks[core]+=1;expected_updates[core]+=rows*(end//bc);legacy_copies+=2*end
      assert actual=={i:(expected_tasks[i],expected_updates[i]) for i in range(8)},actual
      if mode==1:assert copies==legacy_copies,(copies,legacy_copies)
      for prior in state['completed']:
       pc=prior['case']
       if pc.get('kind')=='fa' and pc['t']==t and pc['bc']==bc and pc['hq']==hq and (pc['mode']>=12)==(mode>=12):assert prior['output_sha256']==digest,'matched arithmetic output bits changed'
      row.update(outcome='full_fa_pass',roi_us=s['simTicks']/1e6,copies=copies,arithmetic=arithmetic,output_sha256=digest)
    unchanged();save(path/'accepted.json',row);state['completed'].append(row);state['pending'].remove(c['name']);status()
    lines=['FA-36 V-only optimistic transfer results (accepted only)','Not realistic link/port timing.','']+[f"{x['name']}: {x['outcome']} ROI_us={x.get('roi_us','n/a')}" for x in state['completed']]
    (OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
   finally:broker.release(ticket)
  state['stage']='complete';status();save(OUT/'complete.json',dict(cases=len(state['completed'])))
 except BaseException as e:
  state['stage']='failed';state['failed'].append(dict(error=repr(e),traceback=traceback.format_exc()));status();raise
if __name__=='__main__':main()
